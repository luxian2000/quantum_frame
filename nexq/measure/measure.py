"""
nexq/measure/measure.py

测量入口：统一调度 "电路 -> 末态 -> 概率/采样/期望值" 的流程。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

from ..circuit.gates import gate_to_matrix
from ..circuit.density_matrix import DensityMatrix
from ..circuit.state_vector import StateVector
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

    def _resolve_backend(self, circuit):
        circuit_backend = getattr(circuit, "backend", None)
        return circuit_backend if circuit_backend is not None else self.backend

    @staticmethod
    def _has_gate_sequence(circuit) -> bool:
        gates = getattr(circuit, "gates", None)
        return isinstance(gates, Sequence)

    def _build_initial_state(self, n_qubits: int, backend, initial_state=None) -> StateVector:
        if initial_state is None:
            return StateVector.zero_state(n_qubits, backend)

        if isinstance(initial_state, StateVector):
            if initial_state.n_qubits != n_qubits:
                raise ValueError("initial_state.n_qubits 与电路 n_qubits 不一致")
            return initial_state

        return StateVector(initial_state, n_qubits, backend)

    def _build_initial_density_matrix(
        self,
        n_qubits: int,
        backend,
        initial_density_matrix=None,
    ) -> DensityMatrix:
        if initial_density_matrix is None:
            return DensityMatrix.zero_state(n_qubits, backend)

        if isinstance(initial_density_matrix, DensityMatrix):
            if initial_density_matrix.n_qubits != n_qubits:
                raise ValueError("initial_density_matrix.n_qubits 与电路 n_qubits 不一致")
            return initial_density_matrix

        return DensityMatrix(initial_density_matrix, n_qubits, backend)

    def _circuit_unitary_on_backend(self, circuit, backend):
        """
        优先走 `unitary(backend=...)` 路径以在设备端组装/累乘矩阵。

        兼容旧式或外部电路实现：若不支持 backend 参数，则回退到 unitary()。
        """
        try:
            unitary_raw = circuit.unitary(backend=backend)
        except TypeError:
            unitary_raw = circuit.unitary()
        # Fast path: when unitary_raw is already a backend-native tensor on the
        # target device/dtype, backend.cast can return without host round-trip.
        return backend.cast(unitary_raw)

    def _evolve_state_vector_gatewise(self, circuit, sv0: StateVector, backend) -> StateVector:
        sv = sv0
        for gate in circuit.gates:
            gm = gate_to_matrix(gate, cir_qubits=sv.n_qubits, backend=backend)
            sv = sv.evolve(gm)
        return sv

    def _evolve_density_matrix_gatewise(
        self,
        circuit,
        rho0: DensityMatrix,
        backend,
        noise_model=None,
    ) -> DensityMatrix:
        rho = rho0
        for gate in circuit.gates:
            gate_unitary = gate_to_matrix(gate, cir_qubits=rho.n_qubits, backend=backend)
            rho = rho.evolve(gate_unitary)
            if noise_model is not None:
                rho_noisy = noise_model.apply(
                    rho.data,
                    n_qubits=rho.n_qubits,
                    backend=backend,
                    gate_type=gate.get("type"),
                )
                rho = DensityMatrix(rho_noisy, rho.n_qubits, backend)
        return rho

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
        if not hasattr(circuit, "n_qubits"):
            raise TypeError("circuit 需要具备 n_qubits 属性")
        if not self._has_gate_sequence(circuit) and not hasattr(circuit, "unitary"):
            raise TypeError("circuit 需要具备 gates 序列或 unitary() 方法")

        n_qubits = int(circuit.n_qubits)
        if n_qubits <= 0:
            raise ValueError("n_qubits 必须为正整数")

        backend = self._resolve_backend(circuit)
        sampler = Sampler(backend)

        sv0 = self._build_initial_state(n_qubits, backend, initial_state=initial_state)

        if self._has_gate_sequence(circuit):
            # Preferred execution path: apply each gate directly on the state.
            sv = self._evolve_state_vector_gatewise(circuit, sv0, backend)
        else:
            unitary = self._circuit_unitary_on_backend(circuit, backend)
            sv = sv0.evolve(unitary)
        probs_backend = sv.probabilities()
        probs = backend.to_numpy(probs_backend).real

        counts = None
        use_shots = shots if shots is not None else 0
        if use_shots > 0:
            counts = sampler.sample_counts(probs_backend, n_qubits=n_qubits, shots=use_shots)

        exp_vals: Dict[str, float] = {}
        exp_vars: Dict[str, float] = {}
        if observables:
            for name, op in observables.items():
                op_backend = backend.cast(backend.to_numpy(op))
                exp_val = float(backend.to_numpy(sv.expectation(op_backend)))
                op2 = backend.matmul(op_backend, op_backend)
                exp2 = float(backend.to_numpy(sv.expectation(op2)))
                var = exp2 - exp_val * exp_val
                if abs(var) < self._variance_eps:
                    var = 0.0
                exp_vals[name] = exp_val
                exp_vars[name] = max(var, 0.0)

        final_state_np = sv.to_numpy() if return_state else None

        return Result(
            n_qubits=n_qubits,
            backend_name=backend.name,
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
        if not hasattr(circuit, "n_qubits"):
            raise TypeError("circuit 需要具备 n_qubits 属性")
        if not self._has_gate_sequence(circuit) and not hasattr(circuit, "unitary"):
            raise TypeError("circuit 需要具备 gates 序列或 unitary() 方法")

        n_qubits = int(circuit.n_qubits)
        if n_qubits <= 0:
            raise ValueError("n_qubits 必须为正整数")

        backend = self._resolve_backend(circuit)
        sampler = Sampler(backend)

        rho0 = self._build_initial_density_matrix(
            n_qubits,
            backend,
            initial_density_matrix=initial_density_matrix,
        )

        if self._has_gate_sequence(circuit):
            rho = self._evolve_density_matrix_gatewise(
                circuit,
                rho0,
                backend,
                noise_model=noise_model,
            )
        else:
            rho = rho0
            unitary = self._circuit_unitary_on_backend(circuit, backend)
            rho = rho.evolve(unitary)

        probs = rho.probabilities()
        probs_backend = backend.cast(probs)

        counts = None
        use_shots = shots if shots is not None else 0
        if use_shots > 0:
            counts = sampler.sample_counts(probs_backend, n_qubits=n_qubits, shots=use_shots)

        exp_vals: Dict[str, float] = {}
        exp_vars: Dict[str, float] = {}
        if observables:
            for name, op in observables.items():
                op_backend = backend.cast(backend.to_numpy(op))
                exp_val = float(backend.to_numpy(rho.expectation(op_backend)))
                op2 = backend.matmul(op_backend, op_backend)
                exp2 = float(backend.to_numpy(rho.expectation(op2)))
                var = exp2 - exp_val * exp_val
                if abs(var) < self._variance_eps:
                    var = 0.0
                exp_vals[name] = exp_val
                exp_vars[name] = max(var, 0.0)

        final_state_np = rho.to_numpy().reshape(-1) if return_state else None

        return Result(
            n_qubits=n_qubits,
            backend_name=backend.name,
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