"""
quantum_sim/measure/measure.py

测量入口：统一调度 "电路 -> 末态 -> 概率/采样/期望值" 的流程。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

from ..circuit.model import Circuit
from ..core.states import DensityMatrix, StateVector
from .result import Result
from .sampler import Sampler


class Measure:
    """
    量子测量与结果生成入口。

    使用方式:
        measure = Measure(backend)
        result = measure.run(circuit, shots=1024)
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

        return DensityMatrix(initial_density_matrix, n_qubits, self.backend)

    def run(
        self,
        circuit,
        shots: Optional[int] = None,
        initial_state=None,
        observables: Optional[Dict[str, object]] = None,
        return_state: bool = True,
    ) -> Result:
        """
        测量一个电路，返回统一结果对象。

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

        sv0 = self._build_initial_state(n_qubits, initial_state=initial_state)

        unitary_raw = circuit.unitary()
        unitary = self.backend.cast(self.backend.to_numpy(unitary_raw))

        sv = sv0.evolve(unitary)
        probs_backend = sv.probabilities()
        probs = self.backend.to_numpy(probs_backend).real

        counts = None
        use_shots = shots if shots is not None else 0
        if use_shots > 0:
            counts = self.sampler.sample_counts(probs_backend, n_qubits=n_qubits, shots=use_shots)

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

        return Result(
            n_qubits=n_qubits,
            backend_name=self.backend.name,
            probabilities=probs,
            counts=counts,
            shots=use_shots if use_shots > 0 else None,
            expectation_values=exp_vals,
            expectation_variances=exp_vars,
            final_state=final_state_np,
            metadata={
                "measure": "Measure",
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
        noise_model=None,
        return_state: bool = True,
    ) -> Result:
        """
        以密度矩阵路径测量一个电路，返回统一结果对象。

        参数:
            circuit: 必须具有 `n_qubits` 属性和 `unitary()` 方法
            shots: 采样次数；为 None 或 0 时不采样，仅返回概率
            initial_density_matrix: 初始密度矩阵（None 表示 |0...0><0...0|）
            observables: 可观测量字典 {name: operator_matrix}
            noise_model: NoiseModel，可选。若提供且电路具有 gates，则在每个门后施加噪声。
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

        rho = rho0
        if noise_model is not None and hasattr(circuit, "gates"):
            for gate in circuit.gates:
                gate_unitary_raw = Circuit(gate, n_qubits=n_qubits).unitary()
                gate_unitary = self.backend.cast(self.backend.to_numpy(gate_unitary_raw))
                rho = rho.evolve(gate_unitary)
                rho_noisy = noise_model.apply(
                    rho.data,
                    n_qubits=n_qubits,
                    backend=self.backend,
                    gate_type=gate.get("type"),
                )
                rho = DensityMatrix(rho_noisy, n_qubits, self.backend)
        else:
            unitary_raw = circuit.unitary()
            unitary = self.backend.cast(self.backend.to_numpy(unitary_raw))
            rho = rho.evolve(unitary)

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

        return Result(
            n_qubits=n_qubits,
            backend_name=self.backend.name,
            probabilities=np.asarray(probs, dtype=np.float64),
            counts=counts,
            shots=use_shots if use_shots > 0 else None,
            expectation_values=exp_vals,
            expectation_variances=exp_vars,
            final_state=final_state_np,
            metadata={
                "measure": "Measure",
                "circuit_type": type(circuit).__name__,
                "state_mode": "density_matrix",
                "noise_model": type(noise_model).__name__ if noise_model is not None else None,
            },
        )

    def run_batch(
        self,
        circuits: Sequence[object],
        shots: Optional[int] = None,
        observables: Optional[Dict[str, object]] = None,
        mode: str = "state_vector",
        per_circuit_options: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[Result]:
        """
        批量运行多个电路。

        参数:
            circuits: 电路序列，每个元素需具备 n_qubits 和 unitary()
            shots: 全局采样次数（可被 per_circuit_options 覆盖）
            observables: 全局可观测量字典（可被 per_circuit_options 覆盖）
            mode: "state_vector" 或 "density_matrix"
            per_circuit_options: 每个电路的附加参数字典序列，长度需与 circuits 相同
        返回:
            Result 列表，顺序与输入 circuits 一致
        """
        if not isinstance(circuits, Sequence) or len(circuits) == 0:
            raise ValueError("circuits 必须是非空序列")

        if per_circuit_options is not None and len(per_circuit_options) != len(circuits):
            raise ValueError("per_circuit_options 长度必须与 circuits 一致")

        mode_norm = mode.strip().lower()
        if mode_norm not in {"state_vector", "density_matrix"}:
            raise ValueError("mode 仅支持 'state_vector' 或 'density_matrix'")

        results: List[Result] = []
        for idx, circ in enumerate(circuits):
            opts = per_circuit_options[idx] if per_circuit_options is not None else {}
            run_shots = opts.get("shots", shots)
            run_observables = opts.get("observables", observables)
            run_return_state = opts.get("return_state", True)
            label = opts.get("label")

            if mode_norm == "state_vector":
                result = self.run(
                    circ,
                    shots=run_shots,
                    initial_state=opts.get("initial_state"),
                    observables=run_observables,
                    return_state=run_return_state,
                )
            else:
                result = self.run_density_matrix(
                    circ,
                    shots=run_shots,
                    initial_density_matrix=opts.get("initial_density_matrix"),
                    observables=run_observables,
                    noise_model=opts.get("noise_model"),
                    return_state=run_return_state,
                )

            result.metadata["batch_index"] = idx
            if label is not None:
                result.metadata["label"] = str(label)
            results.append(result)

        return results

    def scan_parameters(
        self,
        circuit_builder: Callable[[Any], object],
        param_values: Iterable[Any],
        shots: Optional[int] = None,
        observables: Optional[Dict[str, object]] = None,
        mode: str = "state_vector",
        return_state: bool = False,
    ) -> List[Result]:
        """
        参数扫描：针对一组参数值构造电路并批量执行。

        返回:
            Result 列表，每个结果 metadata 包含 `scan_param`
        """
        params = list(param_values)
        if len(params) == 0:
            raise ValueError("param_values 不能为空")

        circuits = [circuit_builder(p) for p in params]
        options = [{"return_state": return_state, "label": f"scan_{i}"} for i in range(len(params))]
        results = self.run_batch(
            circuits,
            shots=shots,
            observables=observables,
            mode=mode,
            per_circuit_options=options,
        )

        for i, p in enumerate(params):
            results[i].metadata["scan_param"] = p
            results[i].metadata["scan_index"] = i
        return results