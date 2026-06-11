# aicir.primitives

Sampler / Estimator primitives：算法层的统一执行入口（NEXT.md 第 4 节第一片）。

## 概述

```python
from aicir import Circuit, Hamiltonian, hadamard, cx, pauli_x
from aicir.primitives import ShotSampler, StatevectorEstimator, ShotEstimator

bell = Circuit(hadamard(0), cx(1, [0]), n_qubits=2)
ham = Hamiltonian(n_qubits=1, terms=[("Z", 1.0)])

sample = ShotSampler(shots=1024).run(bell)          # SampleResult
exact = StatevectorEstimator().run(Circuit(pauli_x(0), n_qubits=1), ham)   # EstimateResult, shots=None
noisy = ShotEstimator(shots=4096).run(Circuit(pauli_x(0), n_qubits=1), ham)
```

## 约定（第一片）

- 接收**已绑定参数**的电路；`parameter_values=` 延迟绑定留待后续。
- 单个电路入参返回单个结果；序列入参返回结果列表。
- Estimator 支持单个可观测量广播到多个电路；两者皆为序列时按位置配对
  （长度必须一致）。
- `StatevectorEstimator` 是精确路径，拒绝 `shots=`；可观测量接受
  `Hamiltonian` 或稠密矩阵。
- `ShotEstimator` 包装 `aicir.measure.estimator.PauliEstimator`
  （qubit-wise commuting 分组、基变换测量、shot 分配），并暴露
  `estimate(circuit, hamiltonian)` 直通方法——可直接作为
  `BasicVQE(energy_estimator=...)` 注入，无需改动 VQE。

## 结果对象（NEXT.md 第 9 节最小切片）

- `SampleResult`：`counts` / `probs` / `shots` / `measured_qubits` / `metadata`。
- `EstimateResult`：`value` / `variance` / `shots`（`None` 表示精确路径）/
  `term_results`（逐 Pauli 项明细）/ `metadata`。

## 后续方向（尚未实现）

`NoisySampler`/`NoisyEstimator`（density matrix / noise model 路径——目前可
经 `ShotEstimator(use_density_matrix=True, noise_model=...)` 透传）、
`BackendSampler`/`BackendEstimator`（真实硬件/远端扩展点）、
`parameter_values=` 延迟绑定、`vqc`/`qas`/`metrics` 切换到 primitives。
