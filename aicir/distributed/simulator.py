"""Distributed circuit simulator."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import math

import numpy as np
import torch

from ..core.state import State
from ..ir import (
    circuit_instructions,
    instruction_name,
    instruction_to_gate_dict,
)
from .backend import DistNPUBackend
from .density import _MatrixKernel
from .gates import _GatePlanner, _VectorKernel
from .layout import _Layout, _ShardSpec
from .reducers import _Reducer
from .result import DistResult
from .state import DistState


class DistSimulator:
    """Coordinate one sharded simulation across a process group."""

    def __init__(self, backend: DistNPUBackend):
        if not isinstance(backend, DistNPUBackend):
            raise TypeError("backend 必须是 DistNPUBackend")
        self._backend = backend

    @classmethod
    def from_env(cls, **backend_options) -> "DistSimulator":
        """Build an explicit distributed simulator from launcher variables."""

        return cls(DistNPUBackend.from_env(**backend_options))

    @property
    def backend(self) -> DistNPUBackend:
        return self._backend

    def _resolve_layout(self, circuit, n_qubits: int, layout) -> _Layout:
        distributed_axes = int(math.log2(self._backend.world_size))
        if layout is None:
            return _Layout.auto(
                circuit,
                n_qubits=n_qubits,
                distributed_axes=distributed_axes,
            )
        if isinstance(layout, _Layout):
            if (
                layout.n_qubits != n_qubits
                or layout.distributed_axes != distributed_axes
            ):
                raise ValueError("layout 与 n_qubits/world_size 不一致")
            return layout
        return _Layout.explicit(
            layout,
            n_qubits=n_qubits,
            distributed_axes=distributed_axes,
        )

    def _preflight(self, circuit, *, shots, collapse, observables, layout):
        n_qubits = int(getattr(circuit, "n_qubits", 0))
        if n_qubits <= 0:
            raise ValueError("分布式模拟要求 circuit.n_qubits 是正整数")
        if n_qubits < int(math.log2(self._backend.world_size)):
            raise ValueError("n_qubits 不能小于 log2(world_size)")
        if shots is not None:
            shots = int(shots)
            if shots <= 0:
                raise ValueError("shots 必须是正整数或 None")
        if collapse and shots != 1:
            raise ValueError("collapse=True 仅支持 shots == 1")
        if observables is not None and not isinstance(observables, Mapping):
            raise TypeError("observables 必须是名称到 observable 的映射")

        instructions = circuit_instructions(circuit)
        for instruction in instructions:
            if instruction_name(instruction) in {
                "measure",
                "measurement",
                "reset",
                "if",
                "while",
            }:
                raise ValueError(
                    "分布式首期不支持中途测量、reset 或经典控制流"
                )

        resolved_layout = self._resolve_layout(circuit, n_qubits, layout)
        planner = _GatePlanner(
            self._backend,
            resolved_layout,
            n_qubits,
        )
        plans = tuple(
            planner.plan(instruction, index)
            for index, instruction in enumerate(instructions)
        )
        return n_qubits, instructions, plans, resolved_layout, shots

    def _assert_process_agreement(
        self,
        *,
        circuit,
        layout,
        shots,
        measure_qubits,
        collapse,
        return_state,
        return_probabilities,
    ) -> None:
        payload = (
            int(circuit.n_qubits),
            tuple(
                instruction_to_gate_dict(instruction)
                for instruction in circuit_instructions(circuit)
            ),
            layout.digest(),
            shots,
            tuple(int(qubit) for qubit in measure_qubits),
            bool(collapse),
            bool(return_state),
            bool(return_probabilities),
        )
        digest = hashlib.sha256(repr(payload).encode("utf-8")).digest()
        local = torch.tensor(
            list(digest),
            dtype=torch.uint8,
            device=self._backend._device,
        )
        gathered = self._backend.communicator.all_gather(local)
        values = {bytes(item.detach().cpu().tolist()) for item in gathered}
        if len(values) != 1:
            raise ValueError("各 rank 的线路、布局或运行选项不一致")

    def _initial_modes(self, initial_state, initial_density_matrix):
        if initial_state is not None and initial_density_matrix is not None:
            raise ValueError(
                "initial_state 与 initial_density_matrix 不能同时提供"
            )
        value = (
            initial_state
            if initial_state is not None
            else initial_density_matrix
        )
        if isinstance(value, DistState):
            local_mode = 3
        elif initial_state is not None:
            local_mode = 1
        elif initial_density_matrix is not None:
            local_mode = 2
        else:
            local_mode = 0
        mode_tensor = torch.tensor(
            [local_mode],
            dtype=torch.long,
            device=self._backend._device,
        )
        return tuple(
            int(item.detach().cpu().item())
            for item in self._backend.communicator.all_gather(mode_tensor)
        )

    def _validate_dist_state(
        self,
        state: DistState,
        *,
        n_qubits: int,
        layout: _Layout,
        expected_kind: str | None,
    ) -> DistState:
        if state.backend is not self._backend:
            raise ValueError("DistState 必须属于当前 DistSimulator.backend")
        if state.n_qubits != n_qubits or state.layout != layout:
            raise ValueError("DistState 的 n_qubits/layout 与线路不一致")
        if expected_kind is not None and state.kind != expected_kind:
            raise ValueError(
                f"初态类型应为 {expected_kind!r}，实际为 {state.kind!r}"
            )
        return state

    def _as_numpy(self, value):
        if isinstance(value, State):
            return np.asarray(value.to_numpy())
        if isinstance(value, torch.Tensor):
            return np.asarray(self._backend.to_numpy(value))
        return np.asarray(value)

    def _storage_order(self, array, layout: _Layout, *, kind: str):
        n_qubits = layout.n_qubits
        if kind == "vector":
            return (
                array.reshape([2] * n_qubits)
                .transpose(layout.storage_to_logical)
                .reshape(-1, 1)
            )
        permutation = layout.storage_to_logical + tuple(
            n_qubits + logical
            for logical in layout.storage_to_logical
        )
        return (
            array.reshape([2] * (2 * n_qubits))
            .transpose(permutation)
            .reshape(1 << n_qubits, 1 << n_qubits)
        )

    def _scatter_root_state(
        self,
        value,
        *,
        n_qubits: int,
        layout: _Layout,
        kind: str,
    ) -> DistState:
        spec = _ShardSpec.build(
            n_qubits,
            self._backend.world_size,
            self._backend.rank,
            kind,
            layout,
        )
        tensors = None
        if self._backend.rank == 0:
            array = self._as_numpy(value).astype(np.complex64, copy=False)
            expected = spec.global_shape
            if kind == "vector":
                if array.size != expected[0]:
                    raise ValueError(
                        f"initial_state 必须包含 {expected[0]} 个振幅"
                    )
                array = array.reshape(expected)
            elif tuple(array.shape) != expected:
                raise ValueError(
                    f"initial_density_matrix 形状必须是 {expected}"
                )
            storage = self._storage_order(array, layout, kind=kind)
            full = self._backend.cast(storage)
            tensors = [
                part.contiguous()
                for part in torch.split(full, spec.local_shape[0], dim=0)
            ]
        local = self._backend.communicator.scatter_from_root(
            tensors,
            root=0,
            shape=spec.local_shape,
            dtype=torch.complex64,
        )
        return DistState.from_local(
            local,
            spec=spec,
            backend=self._backend,
        )

    def _prepare_initial_state(
        self,
        *,
        n_qubits: int,
        layout: _Layout,
        initial_state,
        initial_density_matrix,
    ) -> DistState:
        modes = self._initial_modes(initial_state, initial_density_matrix)
        if all(mode == 0 for mode in modes):
            return DistState.zero(
                n_qubits,
                backend=self._backend,
                layout=layout,
            )
        if all(mode == 3 for mode in modes):
            value = (
                initial_state
                if initial_state is not None
                else initial_density_matrix
            )
            expected_kind = (
                "vector"
                if initial_state is not None
                else "matrix"
            )
            return self._validate_dist_state(
                value,
                n_qubits=n_qubits,
                layout=layout,
                expected_kind=expected_kind,
            )
        if modes[0] in {1, 2} and all(mode == 0 for mode in modes[1:]):
            kind = "vector" if modes[0] == 1 else "matrix"
            value = (
                initial_state
                if kind == "vector"
                else initial_density_matrix
            )
            return self._scatter_root_state(
                value,
                n_qubits=n_qubits,
                layout=layout,
                kind=kind,
            )
        raise ValueError(
            "初态必须由所有 rank 提供匹配的 DistState，或仅由 rank 0 "
            "提供完整 statevector/density matrix"
        )

    def run(
        self,
        circuit,
        *,
        initial_state=None,
        initial_density_matrix=None,
        observables=None,
        shots=None,
        measure_qubits=(),
        collapse: bool = False,
        seed=None,
        layout=None,
        return_state: bool = True,
        return_probabilities: bool = True,
    ) -> DistResult:
        """Run one circuit cooperatively on all ranks."""

        (
            n_qubits,
            instructions,
            plans,
            resolved_layout,
            shots,
        ) = self._preflight(
            circuit,
            shots=shots,
            collapse=collapse,
            observables=observables,
            layout=layout,
        )
        measure_qubits = tuple(int(qubit) for qubit in measure_qubits)
        self._assert_process_agreement(
            circuit=circuit,
            layout=resolved_layout,
            shots=shots,
            measure_qubits=measure_qubits,
            collapse=collapse,
            return_state=return_state,
            return_probabilities=return_probabilities,
        )

        with torch.no_grad():
            state = self._prepare_initial_state(
                n_qubits=n_qubits,
                layout=resolved_layout,
                initial_state=initial_state,
                initial_density_matrix=initial_density_matrix,
            )
            vector_kernel = _VectorKernel(self._backend)
            matrix_kernel = _MatrixKernel(self._backend)
            noise_model = getattr(circuit, "noise_model", None)

            for index, (instruction, plan) in enumerate(
                zip(instructions, plans)
            ):
                state = (
                    vector_kernel.apply(state, plan)
                    if state.kind == "vector"
                    else matrix_kernel.apply_unitary(state, plan)
                )
                if noise_model is None:
                    continue
                gate_type = instruction_name(instruction)
                for rule_index, rule in enumerate(noise_model.rules):
                    if not noise_model._match_rule(rule, gate_type):
                        continue
                    if not noise_model._should_apply_to_gate(
                        rule,
                        instruction,
                    ):
                        continue
                    state = matrix_kernel.apply_channel(
                        state,
                        rule.channel,
                        instruction_index=(index + 1) * 1000 + rule_index,
                    )

            reducer = _Reducer(self._backend)
            expectations = {
                str(name): reducer.expectation(state, observable)
                for name, observable in (observables or {}).items()
            }
            local_probabilities = (
                reducer.probabilities(state)
                if return_probabilities
                else None
            )
            counts = None
            if shots is not None:
                counts, collapsed = reducer.sample_z(
                    state,
                    shots=shots,
                    measure_qubits=measure_qubits,
                    seed=seed,
                    collapse=collapse,
                )
                if collapsed is not None:
                    state = collapsed

        return DistResult(
            state=state if return_state else None,
            local_probabilities=local_probabilities,
            expectations=expectations,
            counts=counts,
            rank=self._backend.rank,
            world_size=self._backend.world_size,
            _probability_state=(
                state
                if return_probabilities and not return_state
                else None
            ),
        )
