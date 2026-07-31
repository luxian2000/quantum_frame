"""Differentiable paired-real statevector reductions."""

from __future__ import annotations

import torch

from ...core.operators import Hamiltonian, PauliString
from ...ir import Observable
from ..gates import _GatePlanner
from ..reducers import _as_pauli, _pauli_local_matrix
from ._collectives import _replicated_all_reduce
from ._pair import _Pair
from ._vector import _PairVectorKernel


class _PairReducer:
    """Return replicated real reductions without gathering a full state."""

    def __init__(self, backend):
        self._backend = backend
        self._observable_index = 0

    def probabilities(self, state_pair: _Pair, spec) -> torch.Tensor:
        del spec
        probabilities = state_pair.abs_sq().reshape(-1)
        mean_total = _replicated_all_reduce(
            probabilities.sum().reshape(()),
            communicator=self._backend.communicator,
        )
        # The collective primitive returns a replicated mean in both forward
        # and backward.  Probability normalization needs the physical sum;
        # retaining this factor in the graph scales each backward seed before
        # its mean reduction and therefore restores the global denominator
        # derivative for sharded probability VJPs.
        total = float(self._backend.world_size) * mean_total
        if float(total.detach().cpu()) <= 0.0:
            raise ValueError("分布式状态的全局概率和必须大于 0")
        return probabilities / total

    def expectation(self, state_pair: _Pair, spec, observable) -> torch.Tensor:
        if isinstance(observable, Hamiltonian):
            if observable.n_qubits != spec.n_qubits:
                raise ValueError("Hamiltonian 的 n_qubits 与状态不一致")
            result = torch.zeros((), dtype=torch.float32, device=state_pair.real.device)
            for term in observable.terms:
                result = result + self.expectation(state_pair, spec, term)
            return result
        if isinstance(observable, Observable) and observable.kind == "hamiltonian":
            return self.expectation(state_pair, spec, observable.value)
        if isinstance(observable, Observable) and observable.kind == "matrix":
            axes = tuple(int(qubit) for qubit in observable.metadata.get("qubits", ()))
            if not axes:
                raise TypeError("分布式稠密 observable 必须在 metadata['qubits'] 中显式给出逻辑目标比特")
            return self._matrix_expectation(state_pair, spec, observable.value, axes)

        pauli = _as_pauli(observable)
        if pauli.n_qubits != spec.n_qubits:
            raise ValueError("Pauli observable 的 n_qubits 与状态不一致")
        matrix, axes = _pauli_local_matrix(pauli)
        coefficient = complex(pauli.coefficient)
        if abs(coefficient.imag) > 1e-6:
            raise ValueError("paired-real observable 的系数必须为实数")
        return float(coefficient.real) * self._matrix_expectation(state_pair, spec, matrix, axes)

    def _matrix_expectation(self, state_pair: _Pair, spec, matrix, axes) -> torch.Tensor:
        planner = _GatePlanner(self._backend, spec.layout, spec.n_qubits)
        plan = planner.plan_matrix(
            matrix,
            axes,
            instruction_index=10000 + self._observable_index,
        )
        operation_index = 10000 + self._observable_index
        self._observable_index += 1
        transformed = _PairVectorKernel(self._backend).apply(
            state_pair,
            plan,
            operation_index=operation_index,
        )
        local = (
            state_pair.real.reshape(-1) * transformed.real.reshape(-1)
            + state_pair.imag.reshape(-1) * transformed.imag.reshape(-1)
        ).sum()
        # The collective returns a replicated mean.  Its forward value must be
        # promoted to the physical global sum, while its backward seed must
        # stay one: every rank calls ``backward`` on the same replicated loss.
        # A plain ``* world_size`` would multiply that seed on every rank and
        # therefore over-count the adjoint by world size.
        mean = _replicated_all_reduce(
            local.reshape(()), communicator=self._backend.communicator
        )
        world_size = float(self._backend.world_size)
        return world_size * mean.detach() + mean - mean.detach()
