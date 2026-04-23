"""
quantum_sim/execution/engine.py

执行引擎：统一调度 "电路 -> 末态 -> 概率/采样/期望值" 的流程。
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..core.states import DensityMatrix, StateVector
from .result import ExecutionResult
from .sampler import Sampler


class ExecutionEngine:
    """
    量子电路执行引擎。

    使用方式:
        engine = ExecutionEngine(backend)
        result = engine.run(circuit, shots=1024)
        print(result.counts)
    """

    def __init__(self, backend):
        self.backend = backend
        self.sampler = Sampler(backend)
        self._variance_eps = 1e-7

    def _build_initial_state(self, n_qubits: int, initial_state=None) -> StateVector:
        if initial_state is None:
            return StateVector.zero_state(n_qubits, self.backend)

        if isinstance(initial_state, StateVector):
            if initial_state.n_qubits != n_qubits:
                raise ValueError("initial_state.n_qubits 与电路 n_qubits 不一致")
            return initial_state

        # 兼容输入后端张量或 numpy/list
        return StateVector(initial_state, n_qubits, self.backend)

    def _build_initial_density_matrix(
        self,
        n_qubits: int,
        initial_density_matrix=None,
    ) -> DensityMatrix:
        if initial_density_matrix is None:
            return DensityMatrix.zero_state(n_qubits, self.backend)

        if isinstance(initial_density_matrix, DensityMatrix):
            if initial_density_matrix.n_qubits != n_qubits:
                raise ValueError("initial_density_matrix.n_qubits 与电路 n_qubits 不一致")
            return initial_density_matrix

        # 兼容输入后端张量或 numpy/list
        return DensityMatrix(initial_density_matrix, n_qubits, self.backend)

    def run(
        self,
        circuit,
        shots: Optional[int] = None,
        initial_state=None,
        observables: Optional[Dict[str, object]] = None,
        return_state: bool = True,
    ) -> ExecutionResult:
        """
        运行一个电路，返回统一结果对象。

        参数:
            circuit: 必须具有 `n_qubits` 属性和 `unitary()` 方法
            shots: 采样次数；为 None 或 0 时不采样，仅返回概率
            initial_state: 初始态（None 表示 |0...0>）
            observables: 可观测量字典 {name: operator_matrix}
            return_state: 是否在结果中附带最终态向量
        """
        if not hasattr(circuit, "n_qubits") or not hasattr(circuit, "unitary"):
            raise TypeError("circuit 需要具备 n_qubits 属性和 unitary() 方法")

        n_qubits = int(circuit.n_qubits)
        if n_qubits <= 0:
            raise ValueError("n_qubits 必须为正整数")

        # 构造初始态
        sv0 = self._build_initial_state(n_qubits, initial_state=initial_state)

        # 计算电路酉矩阵并转换到当前后端
        unitary_raw = circuit.unitary()
        unitary = self.backend.cast(self.backend.to_numpy(unitary_raw))

        # 演化
        sv = sv0.evolve(unitary)
        probs_backend = sv.probabilities()
        probs = self.backend.to_numpy(probs_backend).real

        # 采样
        counts = None
        use_shots = shots if shots is not None else 0
        if use_shots > 0:
            counts = self.sampler.sample_counts(probs_backend, n_qubits=n_qubits, shots=use_shots)

        # 期望值与方差
        exp_vals: Dict[str, float] = {}
        exp_vars: Dict[str, float] = {}
        if observables:
            for name, op in observables.items():
                op_backend = self.backend.cast(self.backend.to_numpy(op))
                exp_val = float(self.backend.to_numpy(sv.expectation(op_backend)))
                op2 = self.backend.matmul(op_backend, op_backend)
                exp2 = float(self.backend.to_numpy(sv.expectation(op2)))
                var = exp2 - exp_val * exp_val
                if abs(var) < self._variance_eps:
                    var = 0.0
                exp_vals[name] = exp_val
                exp_vars[name] = max(var, 0.0)

        final_state_np = sv.to_numpy() if return_state else None

        return ExecutionResult(
            n_qubits=n_qubits,
            backend_name=self.backend.name,
            probabilities=probs,
            counts=counts,
            shots=use_shots if use_shots > 0 else None,
            expectation_values=exp_vals,
            expectation_variances=exp_vars,
            final_state=final_state_np,
            metadata={
                "engine": "ExecutionEngine",
                "circuit_type": type(circuit).__name__,
                "state_mode": "state_vector",
            },
        )

    def run_density_matrix(
        self,
        circuit,
        shots: Optional[int] = None,
        initial_density_matrix=None,
        observables: Optional[Dict[str, object]] = None,
        return_state: bool = True,
    ) -> ExecutionResult:
        """
        以密度矩阵路径运行一个电路，返回统一结果对象。

        参数:
            circuit: 必须具有 `n_qubits` 属性和 `unitary()` 方法
            shots: 采样次数；为 None 或 0 时不采样，仅返回概率
            initial_density_matrix: 初始密度矩阵（None 表示 |0...0><0...0|）
            observables: 可观测量字典 {name: operator_matrix}
            return_state: 是否在结果中附带最终密度矩阵（flatten 一维）
        """
        if not hasattr(circuit, "n_qubits") or not hasattr(circuit, "unitary"):
            raise TypeError("circuit 需要具备 n_qubits 属性和 unitary() 方法")

        n_qubits = int(circuit.n_qubits)
        if n_qubits <= 0:
            raise ValueError("n_qubits 必须为正整数")

        rho0 = self._build_initial_density_matrix(
            n_qubits,
            initial_density_matrix=initial_density_matrix,
        )

        unitary_raw = circuit.unitary()
        unitary = self.backend.cast(self.backend.to_numpy(unitary_raw))

        rho = rho0.evolve(unitary)
        probs = rho.probabilities()
        probs_backend = self.backend.cast(probs)

        counts = None
        use_shots = shots if shots is not None else 0
        if use_shots > 0:
            counts = self.sampler.sample_counts(probs_backend, n_qubits=n_qubits, shots=use_shots)

        exp_vals: Dict[str, float] = {}
        exp_vars: Dict[str, float] = {}
        if observables:
            for name, op in observables.items():
                op_backend = self.backend.cast(self.backend.to_numpy(op))
                exp_val = float(self.backend.to_numpy(rho.expectation(op_backend)))
                op2 = self.backend.matmul(op_backend, op_backend)
                exp2 = float(self.backend.to_numpy(rho.expectation(op2)))
                var = exp2 - exp_val * exp_val
                if abs(var) < self._variance_eps:
                    var = 0.0
                exp_vals[name] = exp_val
                exp_vars[name] = max(var, 0.0)

        final_state_np = rho.to_numpy().reshape(-1) if return_state else None

        return ExecutionResult(
            n_qubits=n_qubits,
            backend_name=self.backend.name,
            probabilities=np.asarray(probs, dtype=np.float64),
            counts=counts,
            shots=use_shots if use_shots > 0 else None,
            expectation_values=exp_vals,
            expectation_variances=exp_vars,
            final_state=final_state_np,
            metadata={
                "engine": "ExecutionEngine",
                "circuit_type": type(circuit).__name__,
                "state_mode": "density_matrix",
            },
        )
