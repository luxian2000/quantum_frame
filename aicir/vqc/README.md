# aicir.vqc — 变分量子算法编排

本模块提供 VQE、QAOA、VQD、SSVQE 等基础变分算法编排实现。参数化线路模板由 `aicir.ansatze` 提供；`aicir.vqc` 接收 ansatz 产出的 `Circuit` 或 callable builder，并负责 Hamiltonian、backend、Measure、optimizer 等运行编排。

---

## 1. 公共接口一览

| 函数                           | 文件         | 返回            | 用途                                                                                 |
| ------------------------------ | ------------ | --------------- | ------------------------------------------------------------------------------------ |
| `BasicVQE` / `run_vqe`     | `VQE.py`   | `VQEResult`   | VQE 编排：Hamiltonian、ansatz、backend、Measure、noise、optimizer                    |
| `BasicQAOA` / `run_qaoa`   | `QAOA.py`  | `QAOAResult`  | QAOA 编排：aicir`Hamiltonian`、gate-level `Circuit`、exact/shots 能量、optimizer |
| `BasicVQD` / `run_vqd`     | `VQD.py`   | `VQDResult`   | VQD 编排：带 deflation penalty 的激发态近似                                          |
| `BasicSSVQE` / `run_ssvqe` | `SSVQE.py` | `SSVQEResult` | SSVQE 编排：多参考态加权目标的低能谱近似                                             |

---

## 2. VQE 编排

`BasicVQE` 支持两条路径：

- 不传 `ansatz`：使用内置 dense-matrix RY/CNOT ansatz。
- 传入 `Circuit` 或 `ansatz(params) -> Circuit`：使用线路编排路径，接入 `Hamiltonian`、`Parameter`、`Measure`、backend、shots、density-matrix noise 和 `aicir.optimizer` 参数优化器。

### 参数说明

| 参数                                           | 说明                                                                                                                                                                                                 |
| ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `hamiltonian`                                | dense_matrix 或`aicir.operators.Hamiltonian`                                                                                                                                                       |
| `ansatz`                                     | 可选`Circuit` 模板或 callable builder                                                                                                                                                              |
| `backend`                                    | `NumpyBackend` / `GPUBackend` / `NPUBackend` 等                                                                                                                                                |
| `optimizer`                                  | 可选`GD`、`Adam`、`SPSA`、`COBYLA`、`LBFGSB`、`ScipyMinimize` 等                                                                                                                         |
| `shots`                                      | 透传给`Measure` 或 `PauliEstimator` 生成采样 counts                                                                                                                                              |
| `noise_model`                                | 传入后自动走 density-matrix 测量路径                                                                                                                                                                 |
| `energy_estimator`                           | 默认`"exact"`；也可传入 `PauliEstimator` 做有限 shots Pauli-term 能量估计                                                                                                                        |
| `target`                                     | 可选`aicir.devices.Target`；未显式注入 `energy_estimator` 时经 `estimator_for_target` 按设备能力（statevector/shots/noisy）注入估计器，使能量求值走 primitives。显式 `energy_estimator` 优先 |
| `n_params` / `parameter_shape`             | callable ansatz 无法从`Circuit.parameters` 推断参数量时使用                                                                                                                                        |
| `initial_state` / `initial_density_matrix` | 自定义初态                                                                                                                                                                                           |

### 示例：Circuit + Hamiltonian + optimizer

```python
import numpy as np
from aicir import Circuit, Hamiltonian, NumpyBackend, Parameter, ry
from aicir.optimizer import GD
from aicir.vqc import BasicVQE

theta = Parameter("theta")
ansatz = Circuit(ry(theta, 0), n_qubits=1)
hamiltonian = Hamiltonian([("Z", 1.0)])

solver = BasicVQE(
    hamiltonian,
    ansatz=ansatz,
    backend=NumpyBackend(),
    optimizer=GD(max_iters=80, learning_rate=0.15, gradient_method="psr"),
)
result = solver.run(init_params=np.array([0.1]))
print(result.energy, result.parameters)
```

### 示例：callable ansatz

```python
from aicir import Circuit, ry
from aicir.vqc import BasicVQE

def build(params):
    return Circuit(ry(params[0], 0), n_qubits=1)

solver = BasicVQE(hamiltonian, ansatz=build, n_params=1)
energy = solver.energy(np.array([np.pi]))
```

### 示例：shots 与 noise

```python
from aicir import BitFlipChannel, NoiseModel

noise = NoiseModel().add_channel(BitFlipChannel(target_qubit=0, p=0.01))
solver = BasicVQE(
    hamiltonian,
    ansatz=ansatz,
    backend=NumpyBackend(),
    shots=1024,
    noise_model=noise,
)
energy = solver.energy(np.array([0.1]))
counts = solver._last_measurement.counts(-1)   # -1 取末端测量计数
```

### 示例：PauliEstimator 有限 shots 能量

```python
from aicir import NumpyBackend, PauliEstimator

solver = BasicVQE(
    hamiltonian,
    ansatz=ansatz,
    backend=NumpyBackend(),
    energy_estimator=PauliEstimator(NumpyBackend(), shots=4096),
)
energy = solver.energy(np.array([0.1]))
term_stats = solver._last_estimator_result.term_results
```

注意：`energy_estimator="exact"` 使用 full-matrix observable 精确期望；`energy_estimator=PauliEstimator(...)` 使用 Pauli 项拆分、测量基变换、shots 分配和 counts 统计估计能量。非 exact estimator 需要传入 `Circuit` 或 callable ansatz，不支持内置 dense-matrix ansatz 路径。

---

## 3. QAOA 编排

`BasicQAOA` 用于 canonical gate-level QAOA：输入 aicir 标准 `Hamiltonian`，构造可执行的 `Circuit`，并通过 `Measure` 做 exact statevector 或 shots sampling。`Hamiltonian` 可以包含任意实系数 Pauli 项；`I`/`Z`/`ZZ` 的 Ising/QUBO cost 使用快速门路径，含 `X`/`Y` 或更长 Pauli string 的 cost 按 `trotter_order` 用 Trotter/Suzuki product formula 展开。

标准 ansatz 为：

```text
|psi(gamma, beta)> =
prod_l exp(-i beta_l sum_i X_i) exp(-i gamma_l H_C) |+>^n
```

在 gate-level circuit 中对应：

- 初态：每个 qubit 施加 `H`，得到 `|+>^n`。
- 一体 `X`/`Y`/`Z` cost 项：分别生成 `rx`/`ry`/`rz`。
- 二体 `Z_i Z_j` cost 项：生成 `rzz(2 * gamma * J_ij)`。
- 一般 Pauli string：basis change + CNOT parity ladder + `rz` + uncompute。
- mixer：每个 qubit 施加 `rx(2 * beta)`。
- identity offset 不生成量子门，只进入 `bitstring_energy()` 和 `energy()`。
- `trotter_order=1` 时，每层 cost evolution 使用 `[prod_k exp(-i gamma_l c_k P_k / r)]^r`。
- `trotter_order=2` 时，每个 slice 使用对称 Suzuki 公式：先按 Hamiltonian 项顺序施加半步，最后一项施加整步，再按反序施加半步；`trotter_steps=r` 时重复 `r` 个 slice。

### 参数说明

| 参数                    | 说明                                                                                              |
| ----------------------- | ------------------------------------------------------------------------------------------------- |
| `problem_hamiltonian` | `aicir.core.operators.Hamiltonian` 或 dense matrix；`Hamiltonian` 路径支持任意实系数 Pauli 项 |
| `p`                   | QAOA 深度，参数量为`2 * p`                                                                      |
| `n_qubits`            | 可选；传`Hamiltonian` 时必须与 `Hamiltonian.n_qubits` 一致                                    |
| `seed`                | 初始参数采样随机种子                                                                              |
| `trotter_steps`       | Trotter/Suzuki 切片数，必须`>= 1`                                                               |
| `trotter_order`       | Trotter/Suzuki 公式阶数；支持`1` 和 `2`，默认 `1`                                           |
| `mixer_hamiltonian`   | 仅 dense matrix 输入支持；`Hamiltonian` 路径固定使用标准 X mixer                                |
| `cost`                | 可选外部 qfun 代价函数入口；使用该入口时不构造 QAOA circuit                                       |

### 示例：Hamiltonian → gate-level Circuit

```python
import numpy as np
from aicir import Hamiltonian
from aicir.vqc import BasicQAOA

cost = Hamiltonian(
    n_qubits=2,
    terms=[
        ("Z", [0], 0.5),
        ("ZZ", [0, 1], -1.0),
        ("II", 0.2),
    ],
)

qaoa = BasicQAOA(problem_hamiltonian=cost, p=1)
params = np.array([0.3, 0.2])  # [gamma_0, beta_0]

circuit = qaoa.build_circuit(params)
print(circuit.to_gate_dicts())
```

### 示例：exact energy 与 shots sampling

```python
exact_energy = qaoa.energy(params)
counts = qaoa.sample(params, shots=1024, seed=7)
sampled_energy = qaoa.energy(params, shots=1024, seed=7)

print(exact_energy)
print(counts)
print(sampled_energy)
```

`sample()` 返回末端 Z 基测量 counts，例如 `{"00": 503, "11": 521}`。`energy(..., shots=...)` 会用这些 bitstring 的 cost 值估计平均能量。

注意：`energy(..., shots=...)` 仅支持 diagonal I/Z-only cost。含 `X`/`Y` 的非对角 Hamiltonian 可以用 `energy(params)` 做 exact expectation；有限 shots 能量需要 Pauli-term estimator，而不是直接使用最终 Z-basis counts。

### 示例：使用黑盒优化器

```python
import numpy as np
from aicir.optimizer import NelderMead

result = qaoa.run(
    init_params=np.array([0.1, 0.1]),
    optimizer=NelderMead(options={"maxiter": 100}),
)

print(result.energy)
print(result.parameters)
```

QAOA 的常见优化器是无梯度黑盒优化器，例如 `NelderMead`、`COBYLA`、`SPSA`。`run()` 默认提供 finite-difference gradient descent；实际任务通常建议显式传入 optimizer。

### dense matrix 输入

需要 matrix-form Hamiltonian 或 dense 自定义 mixer 时，可以传入 dense matrix `problem_hamiltonian`：

```python
from aicir import NumpyBackend

matrix = cost.to_matrix(NumpyBackend())
dense_qaoa = BasicQAOA(problem_hamiltonian=matrix, p=1)
```

这条路径不生成 gate-level `Circuit`，也不支持 shots-based QAOA energy。需要构造、采样或下发线路时，请传入 `Hamiltonian`。

---

## 4. 验证命令

```bash
PYTHONPATH=. pytest tests/vqc/test_vqe_orchestration.py
PYTHONPATH=. pytest tests/vqc/test_parameter_shift_uses_qml.py
PYTHONPATH=. pytest tests/vqc/test_qaoa_canonical.py tests/vqc/test_qaoa_qfun.py
PYTHONPATH=. pytest tests/optimization/qubo/test_qaoa_helpers.py
```

ansatz 模板（`hea`/`hea_ti`/`uccsd`）的验证命令见 `aicir/ansatze/README.md`。
