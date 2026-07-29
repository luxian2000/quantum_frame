"""Distributed probabilities, structured expectations, and sampling."""

from __future__ import annotations

import math

import numpy as np
import torch

from ..core.operators import Hamiltonian, PauliString
from ..ir import Observable
from .density import _MatrixKernel
from .gates import _GatePlanner, _VectorKernel
from .state import DistState


_PAULI = {
    "I": np.array([[1, 0], [0, 1]], dtype=np.complex64),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex64),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex64),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex64),
}


def _pauli_local_matrix(pauli: PauliString):
    labels = pauli.qubit_labels
    axes = tuple(index for index, label in enumerate(labels) if label != "I")
    matrix = np.array([[1]], dtype=np.complex64)
    for axis in axes:
        matrix = np.kron(matrix, _PAULI[labels[axis]]).astype(
            np.complex64
        )
    return matrix, axes


def _as_pauli(observable) -> PauliString:
    if isinstance(observable, PauliString):
        return observable
    if isinstance(observable, Observable) and observable.kind == "pauli":
        if isinstance(observable.value, PauliString):
            return observable.value
    raise TypeError(
        "分布式首期仅支持 PauliString 或 Observable.pauli 结构化期望值"
    )


def _scalar_value(tensor):
    value = complex(tensor.detach().cpu().item())
    return value.real if abs(value.imag) < 1e-6 else value


class _Reducer:
    """Reduce rank-local state information without implicit full gathers."""

    def __init__(self, backend):
        self._backend = backend
        self._observable_index = 0

    def probabilities(self, state: DistState):
        return state.local_probabilities()

    def expectation(self, state: DistState, observable):
        if isinstance(observable, Hamiltonian):
            if observable.n_qubits != state.n_qubits:
                raise ValueError(
                    "Hamiltonian 的 n_qubits 与状态不一致"
                )
            return sum(
                self.expectation(state, term)
                for term in observable.terms
            )
        if (
            isinstance(observable, Observable)
            and observable.kind == "hamiltonian"
        ):
            return self.expectation(state, observable.value)
        if (
            isinstance(observable, Observable)
            and observable.kind == "matrix"
        ):
            logical_axes = tuple(
                int(qubit)
                for qubit in observable.metadata.get("qubits", ())
            )
            if not logical_axes:
                raise TypeError(
                    "分布式稠密 observable 必须在 metadata['qubits'] "
                    "中显式给出逻辑目标比特"
                )
            matrix = self._backend.cast_local_matrix(observable.value)
            expected_dimension = 1 << len(logical_axes)
            if tuple(int(axis) for axis in matrix.shape) != (
                expected_dimension,
                expected_dimension,
            ):
                raise ValueError(
                    "稠密 observable 的矩阵维度与 metadata['qubits'] "
                    "不一致"
                )
            return self._matrix_expectation(
                state,
                matrix,
                logical_axes,
            )

        pauli = _as_pauli(observable)
        if pauli.n_qubits != state.n_qubits:
            raise ValueError("Pauli observable 的 n_qubits 与状态不一致")
        matrix, logical_axes = _pauli_local_matrix(pauli)
        return self._matrix_expectation(
            state,
            matrix,
            logical_axes,
            coefficient=complex(pauli.coefficient),
        )

    def _matrix_expectation(
        self,
        state,
        matrix,
        logical_axes,
        *,
        coefficient=1.0,
    ):
        planner = _GatePlanner(
            self._backend,
            state.layout,
            state.n_qubits,
        )
        plan = planner.plan_matrix(
            matrix,
            logical_axes,
            instruction_index=10000 + self._observable_index,
        )
        self._observable_index += 1

        if state.kind == "vector":
            transformed = _VectorKernel(self._backend).apply(state, plan)
            local = self._backend.inner_product(
                state.local_data,
                transformed.local_data,
            )
        else:
            product = _MatrixKernel(self._backend).apply_left(state, plan)
            rows = torch.arange(
                state.local_shape[0],
                dtype=torch.long,
                device=product.device,
            )
            columns = rows + state.spec.global_start
            local = product[rows, columns].sum()
        total = self._backend.communicator.all_reduce_sum(
            local.reshape(())
        )
        value = _scalar_value(total) * complex(coefficient)
        if isinstance(value, complex) and abs(value.imag) < 1e-6:
            return value.real
        return value

    def _storage_bits(self, index: int, state: DistState):
        return tuple(
            (int(index) >> (state.n_qubits - 1 - storage)) & 1
            for storage in range(state.n_qubits)
        )

    def _logical_measurement_key(
        self,
        storage_index: int,
        state: DistState,
        measure_qubits,
    ) -> str:
        storage_bits = self._storage_bits(storage_index, state)
        return "".join(
            str(storage_bits[state.layout.logical_to_storage[logical]])
            for logical in measure_qubits
        )

    def _collapse(
        self,
        state: DistState,
        storage_index: int,
        measure_qubits,
    ) -> DistState:
        selected = {
            state.layout.logical_to_storage[logical]:
            self._storage_bits(storage_index, state)[
                state.layout.logical_to_storage[logical]
            ]
            for logical in measure_qubits
        }
        device = state.local_data.device
        local_indices = (
            torch.arange(
                state.local_shape[0],
                dtype=torch.long,
                device=device,
            )
            + state.spec.global_start
        )
        row_mask = torch.ones(
            state.local_shape[0],
            dtype=torch.float32,
            device=device,
        )
        for storage_axis, bit in selected.items():
            shift = state.n_qubits - 1 - storage_axis
            row_mask = row_mask * (
                torch.remainder(
                    torch.floor(
                        local_indices.to(torch.float32) / float(1 << shift)
                    ),
                    2.0,
                )
                == float(bit)
            ).to(torch.float32)

        if state.kind == "vector":
            data = self._backend.mul(
                state.local_data,
                row_mask.reshape(-1, 1),
            )
            norm2 = self._backend.communicator.all_reduce_sum(
                self._backend.abs_sq(data).sum().reshape(())
            )
            data = self._backend.div(data, torch.sqrt(norm2))
        else:
            columns = torch.arange(
                state.global_shape[1],
                dtype=torch.long,
                device=device,
            )
            column_mask = torch.ones(
                state.global_shape[1],
                dtype=torch.float32,
                device=device,
            )
            for storage_axis, bit in selected.items():
                shift = state.n_qubits - 1 - storage_axis
                column_mask = column_mask * (
                    torch.remainder(
                        torch.floor(
                            columns.to(torch.float32) / float(1 << shift)
                        ),
                        2.0,
                    )
                    == float(bit)
                ).to(torch.float32)
            data = self._backend.mul(
                state.local_data,
                row_mask.reshape(-1, 1),
            )
            data = self._backend.mul(data, column_mask.reshape(1, -1))
            rows = torch.arange(
                state.local_shape[0],
                dtype=torch.long,
                device=device,
            )
            diagonal = data[rows, rows + state.spec.global_start].sum()
            trace = self._backend.communicator.all_reduce_sum(
                diagonal.reshape(())
            )
            data = self._backend.div(data, trace)
        return DistState.from_local(
            data,
            spec=state.spec,
            backend=self._backend,
            bit_order=state.bit_order,
        )

    def sample_z(
        self,
        state: DistState,
        *,
        shots: int,
        measure_qubits=(),
        seed=None,
        collapse: bool = False,
    ):
        shots = int(shots)
        if shots <= 0:
            raise ValueError("shots 必须是正整数")
        if collapse and shots != 1:
            raise ValueError("collapse=True 仅支持 shots == 1")
        measure_qubits = (
            tuple(range(state.n_qubits))
            if len(tuple(measure_qubits)) == 0
            else tuple(int(qubit) for qubit in measure_qubits)
        )
        if len(set(measure_qubits)) != len(measure_qubits):
            raise ValueError("measure_qubits 不能重复")
        if any(
            qubit < 0 or qubit >= state.n_qubits
            for qubit in measure_qubits
        ):
            raise ValueError("measure_qubits 超出范围")

        local_probabilities = state.local_probabilities()
        local_mass = local_probabilities.sum().reshape(1)
        masses = self._backend.communicator.gather_to_root(
            local_mass,
            root=0,
        )
        if self._backend.rank == 0:
            mass_array = np.array(
                [float(value.detach().cpu()) for value in masses],
                dtype=np.float64,
            )
            mass_array = np.clip(mass_array, 0.0, None)
            mass_array /= mass_array.sum()
            base_seed = (
                int(seed)
                if seed is not None
                else int(np.random.SeedSequence().generate_state(1)[0])
            )
            generator = np.random.default_rng(base_seed)
            owners = generator.choice(
                state.world_size,
                size=shots,
                p=mass_array,
            )
            rank_counts_np = np.bincount(
                owners,
                minlength=state.world_size,
            )
            rank_counts = torch.tensor(
                rank_counts_np,
                dtype=torch.long,
                device=state.local_data.device,
            )
            seed_tensor = torch.tensor(
                [base_seed],
                dtype=torch.long,
                device=state.local_data.device,
            )
        else:
            rank_counts = torch.zeros(
                state.world_size,
                dtype=torch.long,
                device=state.local_data.device,
            )
            seed_tensor = torch.zeros(
                1,
                dtype=torch.long,
                device=state.local_data.device,
            )
        rank_counts = self._backend.communicator.broadcast(
            rank_counts,
            root=0,
        )
        seed_tensor = self._backend.communicator.broadcast(
            seed_tensor,
            root=0,
        )
        local_shots = int(
            rank_counts[self._backend.rank].detach().cpu()
        )
        padded = torch.full(
            (shots,),
            -1,
            dtype=torch.long,
            device=state.local_data.device,
        )
        if local_shots:
            conditional = local_probabilities / local_probabilities.sum()
            generator = torch.Generator(
                device=state.local_data.device
            )
            generator.manual_seed(
                int(seed_tensor[0].detach().cpu()) + self._backend.rank
            )
            local_indices = torch.multinomial(
                conditional,
                num_samples=local_shots,
                replacement=True,
                generator=generator,
            )
            padded[:local_shots] = (
                local_indices + state.spec.global_start
            )
        sampled_by_rank = self._backend.communicator.gather_to_root(
            padded,
            root=0,
        )

        selected_storage_index = torch.zeros(
            1,
            dtype=torch.long,
            device=state.local_data.device,
        )
        counts = None
        if self._backend.rank == 0:
            sampled = []
            for values in sampled_by_rank:
                sampled.extend(
                    int(value)
                    for value in values.detach().cpu().tolist()
                    if int(value) >= 0
                )
            counts = {}
            for storage_index in sampled:
                key = self._logical_measurement_key(
                    storage_index,
                    state,
                    measure_qubits,
                )
                counts[key] = counts.get(key, 0) + 1
            if collapse:
                selected_storage_index[0] = sampled[0]

        collapsed = None
        if collapse:
            selected_storage_index = self._backend.communicator.broadcast(
                selected_storage_index,
                root=0,
            )
            collapsed = self._collapse(
                state,
                int(selected_storage_index[0].detach().cpu()),
                measure_qubits,
            )
        return counts, collapsed
