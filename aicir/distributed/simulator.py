"""Distributed circuit simulator."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import math
from typing import Literal

import numpy as np
import torch

from ..core.state import State
from ..ir import (
    circuit_instructions,
    instruction_name,
    instruction_params,
    instruction_to_gate_dict,
)
from ._contracts import (
    AUTOGRAD_ERROR,
    AUTOGRAD_COLLAPSE_ERROR,
    AUTOGRAD_DIRECT_COMPLEX_STATE_ERROR,
    AUTOGRAD_ROUTE_MISMATCH_ERROR,
    AUTOGRAD_SAMPLING_ERROR,
    contains_paired_real,
    contains_requires_grad,
)
from .autograd._collectives import _scatter_root_pair
from .autograd._channels import _selected_noise_rules
from .autograd._checkpoint import (
    _CheckpointMetrics,
    _CheckpointPlanner,
    _CheckpointPolicy,
    _agree_checkpoint_selection,
    _available_memory_bytes,
    _recompute_segment,
)
from .autograd._density import _PairMatrixKernel
from .autograd._pair import _Pair
from .autograd._parameters import (
    PureStateParam,
    _bind_replicated_gradient_bucket,
    _preflight_parameter_structure,
    _replicated_parameter_entries,
)
from .autograd._reducers import _PairReducer
from .autograd._vector import _PairVectorKernel
from .backend import DistNPUBackend
from .density import _MatrixKernel
from .gates import _AutogradExecutionContext, _GatePlanner, _VectorKernel
from .layout import _Layout, _ShardSpec
from .reducers import _Reducer
from .result import DistResult
from .state import DistState


_ROOT_STATE_ERROR_MAX_BYTES = 4096
_ROOT_STATE_ERROR_TRUNCATION_SUFFIX = b"... <truncated>"
_ROOT_STATE_ERROR_PROTOCOL_MESSAGE = "rank 0 初态准备失败同步协议无效"
_DIST_STATE_ERROR_PROTOCOL_MESSAGE = "DistState 本地校验同步协议无效"


def _peak_allocation_bytes(device) -> int | None:
    """Read an already-maintained allocator peak when the runtime exposes it."""

    device = torch.device(device)
    try:
        if device.type == "cuda" and torch.cuda.is_available():
            return int(torch.cuda.max_memory_allocated(device))
        if device.type == "npu":
            peak = getattr(getattr(torch, "npu", None), "max_memory_allocated", None)
            if callable(peak):
                return int(peak(device))
    except Exception:
        pass
    return None


def _synchronize_device(device) -> None:
    """Synchronize only an already-selected accelerator allocator."""

    device = torch.device(device)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device.type == "npu":
        synchronize = getattr(getattr(torch, "npu", None), "synchronize", None)
        if callable(synchronize):
            synchronize(device)


def _reset_peak_memory_stats(device) -> str | None:
    """Reset allocator peak only at an explicit measurement boundary."""

    device = torch.device(device)
    try:
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
            return "cuda"
        if device.type == "npu":
            reset = getattr(getattr(torch, "npu", None), "reset_peak_memory_stats", None)
            if callable(reset):
                reset(device)
                return "npu"
    except Exception:
        return None
    return None


class _PairedReplayEngine:
    """Apply existing plans without rebuilding their communication metadata."""

    def __init__(self, backend, instructions, noise_model):
        self._backend = backend
        self._instructions = tuple(instructions)
        self._noise_model = noise_model
        self._vector = _PairVectorKernel(backend)
        self._matrix = _PairMatrixKernel(backend)

    def _channels(self, operation_index: int):
        if self._noise_model is None:
            return ()
        return _selected_noise_rules(
            self._noise_model,
            self._instructions[operation_index],
        )

    def spec_after(self, spec, start: int, stop: int):
        if spec.kind == "matrix" or not any(self._channels(index) for index in range(start, stop)):
            return spec
        return _ShardSpec.build(
            spec.n_qubits,
            spec.world_size,
            spec.rank,
            "matrix",
            spec.layout,
        )

    def apply(self, state, plan, *, operation_index: int):
        if state.kind == "vector":
            pair = self._vector.apply(
                state._pair,
                plan,
                operation_index=operation_index,
            )
            state = DistState.from_pair(
                pair,
                spec=state.spec,
                backend=self._backend,
                bit_order=state.bit_order,
            )
        else:
            state = self._matrix.apply_unitary(
                state,
                plan,
                operation_index=operation_index,
            )
        for rule_index, channel in enumerate(self._channels(operation_index)):
            state = self._matrix.apply_channel(
                state,
                channel,
                instruction_index=(operation_index + 1) * 1000 + rule_index,
            )
        return state


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

    def _preflight(
        self,
        circuit,
        *,
        shots,
        collapse,
        observables,
        layout,
        autograd: bool = False,
    ):
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
            execution_context=_AutogradExecutionContext() if autograd else None,
        )
        plans = tuple(
            planner.plan(instruction, index)
            for index, instruction in enumerate(instructions)
        )
        return n_qubits, instructions, plans, resolved_layout, shots

    def _collective_autograd_route(
        self,
        circuit,
        initial_state,
        initial_density_matrix,
    ) -> bool:
        """Choose the native path before any state-data collective.

        Root-owned initial parameters intentionally exist only on rank zero;
        sharded states and replicated circuit/noise parameters must agree on
        every rank.  The compact float control payload keeps the decision in
        the paired-real collective subset.
        """

        gate_trainable = any(
            contains_requires_grad(instruction_params(instruction))
            for instruction in circuit_instructions(circuit)
        )
        gate_pair = any(
            contains_paired_real(instruction_params(instruction))
            for instruction in circuit_instructions(circuit)
        )
        local_initial = initial_state if initial_state is not None else initial_density_matrix
        local_initial_trainable = contains_requires_grad(local_initial)
        local_initial_pair = contains_paired_real(local_initial)
        root_owned = not isinstance(local_initial, DistState)
        # The rank-zero value is authoritative only for the existing root
        # ownership mode.  A DistState has one local shard on every rank.
        local = torch.tensor(
            [
                float(gate_trainable or gate_pair),
                float(local_initial_trainable or local_initial_pair),
                float(root_owned and local_initial is not None),
            ],
            dtype=torch.float32,
            device=self._backend._device,
        )
        values = self._backend.communicator.all_gather(local)
        decoded = [
            tuple(int(item.detach().cpu().reshape(-1)[index].item()) for index in range(3))
            for item in values
        ]
        gate_values = {item[0] for item in decoded}
        if len(gate_values) != 1:
            raise ValueError(AUTOGRAD_ROUTE_MISMATCH_ERROR)
        root_modes = {item[2] for item in decoded}
        if root_modes == {0}:
            initial_values = {item[1] for item in decoded}
            if len(initial_values) != 1:
                raise ValueError(AUTOGRAD_ROUTE_MISMATCH_ERROR)
            initial_route = bool(next(iter(initial_values)))
        elif decoded[0][2] == 1 and all(item[2] == 0 for item in decoded[1:]):
            initial_route = bool(decoded[0][1])
        else:
            # Ownership itself will receive the established exact error from
            # _prepare_initial_state; it must not become a rank-divergent
            # routing decision first.
            initial_route = any(item[1] for item in decoded)
        return bool(next(iter(gate_values)) or initial_route)

    def _paired_zero_state(self, *, n_qubits: int, layout: _Layout) -> DistState:
        """Create |0...0> directly as paired float32 shards."""

        spec = _ShardSpec.build(
            n_qubits,
            self._backend.world_size,
            self._backend.rank,
            "vector",
            layout,
        )
        real = torch.zeros(spec.local_shape, dtype=torch.float32, device=self._backend._device)
        if self._backend.rank == 0:
            real[0, 0] = 1.0
        return DistState.from_pair(
            _Pair(real, torch.zeros_like(real)), spec=spec, backend=self._backend
        )

    def _prepare_paired_initial_state(
        self,
        *,
        n_qubits: int,
        layout: _Layout,
        initial_state,
        initial_density_matrix,
    ) -> DistState:
        """Prepare an autograd-safe initial state without complex transport."""

        if initial_state is None and initial_density_matrix is None:
            return self._paired_zero_state(n_qubits=n_qubits, layout=layout)
        state = self._prepare_initial_state(
            n_qubits=n_qubits,
            layout=layout,
            initial_state=initial_state,
            initial_density_matrix=initial_density_matrix,
        )
        if state._pair is not None:
            return state
        raise TypeError(
            "自动微分模式的初态必须是 PureStateParam、DensityParam 或 paired-real DistState"
        )

    def _preflight_autograd_capabilities(
        self,
        *,
        circuit,
        initial_state,
        initial_density_matrix,
        observables,
        shots,
        collapse,
    ) -> None:
        """Reject unsupported gradient requests before state transport."""

        if collapse:
            raise ValueError(AUTOGRAD_COLLAPSE_ERROR)
        if shots is not None:
            raise ValueError(AUTOGRAD_SAMPLING_ERROR)
        root_value = initial_state if initial_state is not None else initial_density_matrix
        if (
            self._backend.rank == 0
            and isinstance(root_value, torch.Tensor)
            and torch.is_complex(root_value)
            and root_value.requires_grad
        ):
            self._raise_root_initial_error(AUTOGRAD_DIRECT_COMPLEX_STATE_ERROR)
        # Keep the root protocol collective even when the root has no error.
        if not (
            self._backend.rank == 0
            and isinstance(root_value, torch.Tensor)
            and torch.is_complex(root_value)
            and root_value.requires_grad
        ):
            self._raise_root_initial_error(None)
        if observables is not None:
            from ..core.operators import Hamiltonian, PauliString
            from ..ir import Observable

            unsupported = next(
                (
                    name
                    for name, observable in observables.items()
                    if not isinstance(observable, (Hamiltonian, PauliString, Observable))
                ),
                None,
            )
            if unsupported is not None:
                raise TypeError(
                    f"自动微分模式不支持 observable {unsupported!r}"
                )
        noise_model = getattr(circuit, "noise_model", None)
        if noise_model is not None:
            from ..noise import (
                AmplitudeDampingChannel,
                BitFlipChannel,
                DepolarizingChannel,
                PhaseFlipChannel,
            )
            from .autograd._parameters import StinespringParam

            supported = (
                AmplitudeDampingChannel,
                BitFlipChannel,
                DepolarizingChannel,
                PhaseFlipChannel,
                StinespringParam,
            )
            for rule in noise_model.rules:
                if not isinstance(rule.channel, supported):
                    raise TypeError(
                        "自动微分模式不支持噪声通道 "
                        f"{type(rule.channel).__name__}"
                    )

    def _assert_forward_only(
        self,
        circuit,
        initial_state,
        initial_density_matrix,
    ) -> None:
        local_rejected = (
            contains_paired_real(initial_state)
            or contains_paired_real(initial_density_matrix)
            or contains_requires_grad(initial_state)
            or contains_requires_grad(initial_density_matrix)
            or any(
                contains_requires_grad(instruction_params(instruction))
                for instruction in circuit_instructions(circuit)
            )
        )
        flag = torch.tensor(
            [int(local_rejected)],
            dtype=torch.long,
            device=self._backend._device,
        )
        rejected = self._backend.communicator.all_reduce_sum(flag)
        if int(rejected.detach().cpu().item()) > 0:
            raise ValueError(AUTOGRAD_ERROR)

    def _run_paired_real(
        self,
        circuit,
        *,
        initial_state: DistState,
        layout=None,
        grad_checkpoint="auto",
        available_memory_bytes: int | None = None,
    ):
        """Private paired-real execution hook used before the public release gate.

        This intentionally accepts an already-prepared :class:`DistState` and
        is not reached from :meth:`run`.  It is the narrow integration point
        for native-kernel tests and the NPU probe while public trainable routing
        remains disabled through Task 11/release qualification.
        """

        policy = _CheckpointPolicy.parse(grad_checkpoint)
        if not isinstance(initial_state, DistState) or initial_state._pair is None:
            raise TypeError("_run_paired_real 需要 paired-real DistState 初态")
        n_qubits = int(circuit.n_qubits)
        resolved_layout = self._resolve_layout(circuit, n_qubits, layout)
        if initial_state.backend is not self._backend:
            raise ValueError("DistState 必须属于当前 DistSimulator.backend")
        if initial_state.n_qubits != n_qubits or initial_state.layout != resolved_layout:
            raise ValueError("DistState 的 n_qubits/layout 与线路不一致")

        # Public ``run`` remains forward-only until Task 11.  The private
        # paired-real hook is nevertheless the native integration point: it
        # preflights the replicated circuit/noise/Stinespring schema and binds
        # one differentiable float32 gradient bucket without touching caller
        # instructions or initial-state ownership.
        circuit = _bind_replicated_gradient_bucket(
            circuit, communicator=self._backend.communicator
        )
        instructions = tuple(circuit_instructions(circuit))
        context = _AutogradExecutionContext()
        planner = _GatePlanner(
            self._backend,
            resolved_layout,
            n_qubits,
            execution_context=context,
        )
        plans = tuple(
            planner.plan(instruction, index)
            for index, instruction in enumerate(instructions)
        )
        if available_memory_bytes is None:
            available, memory_source = _available_memory_bytes(self._backend._device)
        else:
            available, memory_source = int(available_memory_bytes), "provided"
        if policy.value == "none":
            interval = 0
        elif policy.value == "auto":
            interval = (
                1
                if available is None
                else _CheckpointPlanner(initial_state.spec, len(plans), available).interval()
            )
        else:
            interval = int(policy.value)
        if interval:
            interval, memory_source = _agree_checkpoint_selection(
                interval, memory_source, self._backend.communicator
            )

        engine = _PairedReplayEngine(self._backend, instructions, getattr(circuit, "noise_model", None))
        metrics = _CheckpointMetrics(
            policy=policy.value,
            interval=interval,
            saved_state_count=(len(plans) + interval - 1) // interval + 1 if interval else len(plans) + 1,
        )
        metrics.memory_source = memory_source
        state = initial_state
        if not interval:
            state = _recompute_segment(state, plans, 0, len(plans), engine)
            return state, metrics

        from torch.utils.checkpoint import checkpoint

        for start in range(0, len(plans), interval):
            stop = min(len(plans), start + interval)
            start_spec = state.spec
            end_spec = engine.spec_after(state.spec, start, stop)
            calls = [0]

            def replay(real, imag, *, _start=start, _stop=stop, _spec=start_spec):
                calls[0] += 1
                if calls[0] > 1:
                    metrics.recomputed_gate_count += _stop - _start
                replayed = _recompute_segment(
                    DistState.from_pair(_Pair(real, imag), spec=_spec, backend=self._backend),
                    plans,
                    _start,
                    _stop,
                    engine,
                )
                return replayed._pair.real, replayed._pair.imag

            real, imag = checkpoint(replay, state._pair.real, state._pair.imag, use_reentrant=False)
            state = DistState.from_pair(_Pair(real, imag), spec=end_spec, backend=self._backend, bit_order=state.bit_order)
        return state, metrics

    def _measure_paired_real(self, *args, **kwargs):
        """Run one private policy behind an explicit per-run peak boundary."""

        # Finish the preceding policy's asynchronous accelerator work before
        # resetting this policy's allocator peak counter.
        _synchronize_device(self._backend._device)
        source = _reset_peak_memory_stats(self._backend._device)
        if source is None:
            state, metrics = self._run_paired_real(*args, **kwargs)
            metrics.peak_allocation_bytes = None
            metrics.peak_allocation_status = "UNAVAILABLE"
            return state, metrics
        state, metrics = self._run_paired_real(*args, **kwargs)
        _synchronize_device(self._backend._device)
        metrics.peak_allocation_bytes = _peak_allocation_bytes(self._backend._device)
        metrics.peak_allocation_status = "MEASURED" if metrics.peak_allocation_bytes is not None else "UNAVAILABLE"
        return state, metrics

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
            local_mode = 5
        elif isinstance(
            (
                initial_state
                if initial_state is not None
                else initial_density_matrix
            ),
            DistState,
        ):
            local_mode = 3 if initial_state is not None else 4
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
        modes = tuple(
            int(item.detach().cpu().item())
            for item in self._backend.communicator.all_gather(mode_tensor)
        )
        if any(mode == 5 for mode in modes):
            raise ValueError(
                "initial_state 与 initial_density_matrix 不能同时提供"
            )
        return modes

    def _validate_dist_state(
        self,
        state: DistState,
        *,
        n_qubits: int,
        layout: _Layout,
        expected_kind: str | None,
    ) -> DistState:
        local_message = None
        try:
            if state.backend is not self._backend:
                local_message = (
                    "DistState 必须属于当前 DistSimulator.backend"
                )
            elif state.n_qubits != n_qubits or state.layout != layout:
                local_message = (
                    "DistState 的 n_qubits/layout 与线路不一致"
                )
            elif (
                state.spec.rank != self._backend.rank
                or state.spec.world_size != self._backend.world_size
            ):
                local_message = (
                    "spec 的 rank/world_size 与 backend 不一致"
                )
            elif expected_kind is not None and state.kind != expected_kind:
                local_message = (
                    f"初态类型应为 {expected_kind!r}，"
                    f"实际为 {state.kind!r}"
                )
            else:
                expected_spec = _ShardSpec.build(
                    n_qubits,
                    self._backend.world_size,
                    self._backend.rank,
                    state.kind,
                    layout,
                )
                if state.spec != expected_spec:
                    local_message = (
                        "DistState.spec 与线路/backend 不一致"
                    )
                elif state._pair is not None:
                    pair = state._pair
                    if tuple(int(axis) for axis in pair.real.shape) != state.local_shape:
                        local_message = (
                            "pair shape="
                            f"{tuple(pair.real.shape)} 与 "
                            f"local_shape={state.local_shape} 不一致"
                        )
                    elif (
                        pair.real.dtype != torch.float32
                        or pair.imag.dtype != torch.float32
                        or pair.real.device != self._backend._device
                        or pair.imag.device != self._backend._device
                    ):
                        local_message = "DistState paired-real 初态必须是当前 backend 的实数 torch.float32"
                elif tuple(
                    int(axis) for axis in state.local_data.shape
                ) != state.local_shape:
                    local_message = (
                        "local_data shape="
                        f"{tuple(state.local_data.shape)} 与 "
                        f"local_shape={state.local_shape} 不一致"
                    )
                elif state.local_data.dtype != torch.complex64:
                    local_message = (
                        "DistState 首期仅支持 torch.complex64"
                    )
        except Exception as error:  # noqa: BLE001
            try:
                error_text = str(error)
            except Exception:  # noqa: BLE001
                error_text = "<unprintable exception>"
            local_message = (
                "DistState 本地校验失败: "
                f"{type(error).__name__}: {error_text}"
            )

        failure = torch.tensor(
            [int(local_message is not None)],
            dtype=torch.long,
            device=self._backend._device,
        )
        failures = self._backend.communicator.all_gather(failure)
        source = next(
            (
                rank
                for rank, item in enumerate(failures)
                if int(item.detach().cpu().item()) != 0
            ),
            None,
        )
        if source is None:
            pair_flags = self._backend.communicator.all_gather(
                torch.tensor(
                    [
                        int(state._pair is not None),
                        int(
                            state._pair is not None
                            and (
                                state._pair.real.requires_grad
                                or state._pair.imag.requires_grad
                            )
                        ),
                    ],
                    dtype=torch.long,
                    device=self._backend._device,
                )
            )
            pair_modes = [int(item[0].detach().cpu().item()) for item in pair_flags]
            pair_grad_modes = [int(item[1].detach().cpu().item()) for item in pair_flags]
            if any(mode != pair_modes[0] for mode in pair_modes[1:]):
                raise ValueError("DistState paired-real 表示在各 rank 间不一致")
            if pair_modes[0] and any(
                mode != pair_grad_modes[0] for mode in pair_grad_modes[1:]
            ):
                raise ValueError("DistState paired-real requires_grad 在各 rank 间不一致")
            return state

        encoded = (
            local_message.encode("utf-8", errors="replace")
            if self._backend.rank == source
            else b""
        )
        if len(encoded) > _ROOT_STATE_ERROR_MAX_BYTES:
            keep = (
                _ROOT_STATE_ERROR_MAX_BYTES
                - len(_ROOT_STATE_ERROR_TRUNCATION_SUFFIX)
            )
            encoded = (
                encoded[:keep]
                .decode("utf-8", errors="ignore")
                .encode("utf-8")
                + _ROOT_STATE_ERROR_TRUNCATION_SUFFIX
            )
        size = torch.tensor(
            [len(encoded)],
            dtype=torch.long,
            device=self._backend._device,
        )
        size = self._backend.communicator.broadcast(size, root=source)
        size_values = size.detach().cpu().reshape(-1).tolist()
        if (
            len(size_values) != 1
            or not 0 < int(size_values[0]) <= _ROOT_STATE_ERROR_MAX_BYTES
        ):
            raise RuntimeError(_DIST_STATE_ERROR_PROTOCOL_MESSAGE)
        message_size = int(size_values[0])
        message_tensor = (
            torch.tensor(
                list(encoded),
                dtype=torch.uint8,
                device=self._backend._device,
            )
            if self._backend.rank == source
            else torch.empty(
                message_size,
                dtype=torch.uint8,
                device=self._backend._device,
            )
        )
        message_tensor = self._backend.communicator.broadcast(
            message_tensor,
            root=source,
        )
        message = bytes(
            message_tensor.detach().cpu().tolist()
        ).decode("utf-8", errors="replace")
        raise ValueError(message)

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
        failure_code = 0
        failure_message = b""
        if self._backend.rank == 0:
            try:
                array = self._as_numpy(value).astype(
                    np.complex64,
                    copy=False,
                )
                expected = spec.global_shape
                if kind == "vector" and array.size != expected[0]:
                    failure_code = 1
                elif kind == "matrix" and tuple(array.shape) != expected:
                    failure_code = 2
                else:
                    array = array.reshape(expected)
                    storage = self._storage_order(
                        array,
                        layout,
                        kind=kind,
                    )
                    full = self._backend.cast(storage)
                    tensors = [
                        part.contiguous()
                        for part in torch.split(
                            full,
                            spec.local_shape[0],
                            dim=0,
                        )
                    ]
            except Exception as error:  # noqa: BLE001
                failure_code = 3
                try:
                    error_text = str(error)
                except Exception:  # noqa: BLE001
                    error_text = "<unprintable exception>"
                failure_message = (
                    f"{type(error).__name__}: {error_text}".encode(
                        "utf-8",
                        errors="replace",
                    )
                )
                if len(failure_message) > _ROOT_STATE_ERROR_MAX_BYTES:
                    keep = (
                        _ROOT_STATE_ERROR_MAX_BYTES
                        - len(_ROOT_STATE_ERROR_TRUNCATION_SUFFIX)
                    )
                    failure_message = (
                        failure_message[:keep]
                        .decode("utf-8", errors="ignore")
                        .encode("utf-8")
                        + _ROOT_STATE_ERROR_TRUNCATION_SUFFIX
                    )

        status = torch.tensor(
            [failure_code, len(failure_message)],
            dtype=torch.long,
            device=self._backend._device,
        )
        status = self._backend.communicator.broadcast(status, root=0)
        status_values = status.detach().cpu().reshape(-1).tolist()
        if len(status_values) != 2:
            raise RuntimeError(_ROOT_STATE_ERROR_PROTOCOL_MESSAGE)
        failure_code, message_size = map(int, status_values)
        if (
            failure_code not in {0, 1, 2, 3}
            or not 0 <= message_size <= _ROOT_STATE_ERROR_MAX_BYTES
            or (failure_code == 3) != (message_size > 0)
        ):
            raise RuntimeError(_ROOT_STATE_ERROR_PROTOCOL_MESSAGE)
        if failure_code == 1:
            raise ValueError(
                f"initial_state 必须包含 {spec.global_shape[0]} 个振幅"
            )
        if failure_code == 2:
            raise ValueError(
                "initial_density_matrix 形状必须是 "
                f"{spec.global_shape}"
            )
        if failure_code == 3:
            message_tensor = (
                torch.tensor(
                    list(failure_message),
                    dtype=torch.uint8,
                    device=self._backend._device,
                )
                if self._backend.rank == 0
                else torch.empty(
                    message_size,
                    dtype=torch.uint8,
                    device=self._backend._device,
                )
            )
            message_tensor = self._backend.communicator.broadcast(
                message_tensor,
                root=0,
            )
            message = bytes(
                message_tensor.detach().cpu().tolist()
            ).decode("utf-8", errors="replace")
            raise RuntimeError(f"rank 0 初态准备失败: {message}")
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

    def _raise_root_initial_error(self, message: str | None) -> None:
        """Broadcast one bounded root-preparation error before raising it."""

        encoded = (
            message.encode("utf-8", errors="replace")
            if self._backend.rank == 0 and message is not None
            else b""
        )
        if len(encoded) > _ROOT_STATE_ERROR_MAX_BYTES:
            encoded = encoded[:_ROOT_STATE_ERROR_MAX_BYTES]
        size = self._backend.communicator.broadcast(
            torch.tensor(
                [len(encoded)], dtype=torch.long, device=self._backend._device
            ),
            root=0,
        )
        message_size = int(size.detach().cpu().reshape(-1).item())
        if not 0 <= message_size <= _ROOT_STATE_ERROR_MAX_BYTES:
            raise RuntimeError(_ROOT_STATE_ERROR_PROTOCOL_MESSAGE)
        if message_size == 0:
            return
        payload = (
            torch.tensor(
                list(encoded), dtype=torch.uint8, device=self._backend._device
            )
            if self._backend.rank == 0
            else torch.empty(
                message_size, dtype=torch.uint8, device=self._backend._device
            )
        )
        payload = self._backend.communicator.broadcast(payload, root=0)
        raise ValueError(
            bytes(payload.detach().cpu().tolist()).decode(
                "utf-8", errors="replace"
            )
        )

    def _scatter_root_pure_state(
        self,
        value: PureStateParam | None,
        *,
        n_qubits: int,
        layout: _Layout,
    ) -> DistState:
        """Normalize one root-owned real pair and scatter differentiable shards."""

        spec = _ShardSpec.build(
            n_qubits,
            self._backend.world_size,
            self._backend.rank,
            "vector",
            layout,
        )
        pair = None
        error = None
        if self._backend.rank == 0:
            try:
                if not isinstance(value, PureStateParam):
                    raise TypeError("initial_state 必须是 PureStateParam")
                pair = value._raw_pair()
                if pair.real.device != self._backend._device:
                    raise ValueError("PureStateParam 必须位于当前 backend device")
                if pair.real.numel() != spec.global_shape[0]:
                    raise ValueError(
                        f"initial_state 必须包含 {spec.global_shape[0]} 个振幅"
                    )
                storage_pair = _Pair(
                    pair.real.reshape([2] * n_qubits)
                    .permute(layout.storage_to_logical)
                    .reshape(spec.global_shape),
                    pair.imag.reshape([2] * n_qubits)
                    .permute(layout.storage_to_logical)
                    .reshape(spec.global_shape),
                )
                norm = torch.sqrt(storage_pair.abs_sq().sum())
                if float(norm.detach().cpu()) == 0.0:
                    raise ValueError("纯态参数的范数必须大于 0")
                pair = storage_pair.div_real(norm).real.reshape(
                    (self._backend.world_size,) + spec.local_shape
                )
                imag = storage_pair.div_real(norm).imag.reshape(
                    (self._backend.world_size,) + spec.local_shape
                )
                pair = _Pair(pair, imag)
            except Exception as caught:  # noqa: BLE001 - synchronize root failure
                error = str(caught)
        self._raise_root_initial_error(error)
        return DistState.from_pair(
            _scatter_root_pair(
                pair,
                communicator=self._backend.communicator,
                root=0,
                local_shape=spec.local_shape,
            ),
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
        if all(mode == 3 for mode in modes) or all(
            mode == 4 for mode in modes
        ):
            expected_kind = "vector" if modes[0] == 3 else "matrix"
            value = (
                initial_state
                if expected_kind == "vector"
                else initial_density_matrix
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
            if kind == "vector":
                root_is_pure = self._backend.communicator.broadcast(
                    torch.tensor(
                        [
                            int(
                                self._backend.rank == 0
                                and isinstance(value, PureStateParam)
                            )
                        ],
                        dtype=torch.long,
                        device=self._backend._device,
                    ),
                    root=0,
                )
                direct_complex_error = (
                    "原生 distributed autograd 不接受 requires_grad complex initial_state；"
                    "请使用 PureStateParam(real, imag)"
                    if self._backend.rank == 0
                    and isinstance(value, torch.Tensor)
                    and torch.is_complex(value)
                    and value.requires_grad
                    else None
                )
                self._raise_root_initial_error(direct_complex_error)
                if int(root_is_pure.detach().cpu().item()):
                    return self._scatter_root_pure_state(
                        value,
                        n_qubits=n_qubits,
                        layout=layout,
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
        grad_checkpoint: Literal["none", "auto"] | int = "auto",
    ) -> DistResult:
        """Run one circuit cooperatively on all ranks."""

        _CheckpointPolicy.parse(grad_checkpoint)
        autograd = self._collective_autograd_route(
            circuit,
            initial_state,
            initial_density_matrix,
        )
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
            autograd=autograd,
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

        if autograd:
            self._preflight_autograd_capabilities(
                circuit=circuit,
                initial_state=initial_state,
                initial_density_matrix=initial_density_matrix,
                observables=observables,
                shots=shots,
                collapse=collapse,
            )
            # The deterministic parameter schema has to agree before a
            # root-owned paired state could enter scatter transport.  The
            # execution hook binds the aliases later, after this control-only
            # preflight has made an all-rank failure safe.
            _preflight_parameter_structure(
                _replicated_parameter_entries(circuit),
                communicator=self._backend.communicator,
            )
            state = self._prepare_paired_initial_state(
                n_qubits=n_qubits,
                layout=resolved_layout,
                initial_state=initial_state,
                initial_density_matrix=initial_density_matrix,
            )
            state, _metrics = self._run_paired_real(
                circuit,
                initial_state=state,
                layout=resolved_layout,
                grad_checkpoint=grad_checkpoint,
            )
            reducer = _PairReducer(self._backend)
            expectations = {
                str(name): reducer.expectation(state._pair, state.spec, observable)
                for name, observable in (observables or {}).items()
            }
            local_probabilities = (
                reducer.probabilities(state._pair, state.spec)
                if return_probabilities
                else None
            )
            return DistResult(
                state=state if return_state else None,
                local_probabilities=local_probabilities,
                expectations=expectations,
                counts=None,
                rank=self._backend.rank,
                world_size=self._backend.world_size,
                _probability_state=(
                    state if return_probabilities and not return_state else None
                ),
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
