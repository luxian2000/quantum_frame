# aicir.qml.grad — 量子机器学习梯度工具包

本模块 (`aicir/qml/grad.py`) 实现了八种用于量子电路参数梯度估计或预条件的方法，覆盖从黑盒数值近似到解析反向传播和量子自然梯度的完整谱系。所有方法均与 `NumpyBackend`、`TorchBackend`、`NPUBackend` 兼容，后端返回的张量（包括自动微分追踪张量、复数标量、加速器设备张量）均可直接作为目标函数返回值，无需手动调用 `float()` 或 `to_numpy()`。

---

## 公共接口一览

| 函数 | 全称 | 复杂度（函数调用次数） | 适用场景 |
|------|------|----------------------|----------|
| `auto` | Automatic Differentiation | 1 次反向传播 | Torch/NPU 后端，自动微分图 |
| `psr` | Parameter-Shift Rule | $2P$ | 通用旋转门，无噪声或含噪声均可 |
| `spsr` | Stochastic PSR | $2K+1$（$K \leq P$ 次采样） | 大参数量随机梯度 |
| `mpsr` | Multi-parameter PSR | $2^M$（$M$ 个坐标的混合偏导数） | 高阶混合偏导 |
| `fd` | Finite Difference | $2P$（中心差）或 $P+1$（单侧） | 任意可微目标，黑盒 |
| `ad` | Adjoint Differentiation | $O(P)$ 次门作用，$O(1)$ 额外存储 | 无噪声态向量模拟，效率最高 |
| `qng` | Quantum Natural Gradient | 梯度调用 + $2P+1$ 次态函数调用 | 用 QFIM 逆预条件梯度，加速 ansatz 优化 |
| `bdqng` | Block-diagonal QNG | 梯度调用 + $2P+1$ 次态函数调用，分块求解 | 按参数块近似 QFIM，适合大参数 ansatz |

$P$：可微参数数量。

---

## 1. 自动微分 `auto`

```python
from aicir.qml import auto
grad = auto(fn, params, *, backend=None)
```

### 原理

`auto` 利用 **PyTorch 反向传播**穿透量子电路的所有门运算，实现"幺正传播的反向传播"（Backpropagation through the Unitary）。其工作方式与经典神经网络的自动微分完全相同：

1. 参数以 `torch.Tensor(requires_grad=True)` 形式传入目标函数
2. 目标函数在 Torch 后端上构建并演化电路（所有操作留在自动微分计算图中）
3. 对期望值标量调用 `.backward()`，读取 `.grad`

$$\frac{\partial \langle O \rangle}{\partial \boldsymbol{\theta}} = \text{torch.autograd.grad}(\langle O \rangle(\boldsymbol{\theta}),\ \boldsymbol{\theta})$$

### NPU 兼容设计

参数张量的 `dtype` 和 `device` 均从 `backend` 中读取（`backend._dtype` → 对应实数类型，`backend._device` → NPU/CUDA 设备），确保：

- 对于 `NPUBackend`，参数从创建起就驻留在 NPU 上，梯度在设备端累积，最后才将 `.grad` 一次性移至主机
- 不会在梯度计算中触发额外的设备↔主机往返

| 精度后端 | 参数 `dtype` |
|--------|-------------|
| `complex64`（默认） | `torch.float32` |
| `complex128` | `torch.float64` |

### 与其他方法的对比

- `auto` 与 `ad` 都只需一次反向遍历，但机制不同：
  - `ad`：手动实现伴随传播，只需 `NumpyBackend`，对电路结构有限制（仅 Pauli 旋转门可微）
  - `auto`：依赖 PyTorch 自动微分，支持任意可微操作（`u3`、任意幺正门等），但需要 Torch/NPU 后端且目标函数不能断开计算图
- 若目标函数调用了 `float()`、`.item()`、`.detach()` 或 `to_numpy()`，计算图将断裂，`auto` 会抛出明确错误

### 示例

```python
from aicir.qml import auto
from aicir import Circuit, State, ry
from aicir.channel.backends.torch_backend import TorchBackend
import numpy as np

bk = TorchBackend(device="cpu")
z   = bk.cast(np.diag([1.0, -1.0]).astype(np.complex64))

def fn(theta):                        # theta: torch.Tensor, requires_grad=True
    circuit = Circuit(ry(theta, 0), n_qubits=1)
    state   = State.zero_state(1, bk).evolve(circuit.unitary(backend=bk))
    return bk.expectation_sv(state.data, z)   # 必须返回 torch 张量，不调用 float()

grad = auto(fn, np.array(0.5), backend=bk)
# grad ≈ -sin(0.5) ≈ -0.4794
```

---

## 2. 参数移位规则 `psr`

```python
from aicir.qml import psr
grad = psr(fn, params, *, shift=π/2, coefficient=0.5)
```

### 原理

对于以 Pauli 旋转门 $U_k(\theta_k) = e^{-i\theta_k G_k/2}$ 参数化的量子电路，可观测量期望 $\langle O \rangle(\theta_k)$ 关于 $\theta_k$ 的梯度满足**精确的**两点公式：

$$\frac{\partial \langle O \rangle}{\partial \theta_k} = \frac{1}{2}\left[\langle O \rangle\!\left(\theta_k + \frac{\pi}{2}\right) - \langle O \rangle\!\left(\theta_k - \frac{\pi}{2}\right)\right]$$

即 `coefficient × [fn(θ+shift) - fn(θ-shift)]`，默认 `shift=π/2, coefficient=0.5`，对应 Pauli 旋转门的标准生成元频谱 $\{±\frac{1}{2}\}$。

### 特点

- **精确**：不依赖步长，无截断误差
- **黑盒**：`fn` 只需返回标量，与电路内部结构无关
- **通用**：支持含噪声线路（通过密度矩阵测量）
- 每个参数需要 **2 次** 函数调用，总共 $2P$ 次

### 示例

```python
import numpy as np
from aicir.qml import psr
from aicir import Circuit, State, NumpyBackend, ry

bk = NumpyBackend()
z  = np.diag([1.0, -1.0])

def objective(params):
    state = State.zero_state(1, bk).evolve(Circuit(ry(params[0], 0), n_qubits=1).unitary())
    return float(state.expectation(z))

grad = psr(objective, np.array([0.5]))
# grad ≈ [-sin(0.5)] ≈ [-0.4794]
```

---

## 3. 随机参数移位规则 `spsr`

```python
from aicir.qml import spsr
grad = spsr(fn, params, *, n_samples=1, rng=None, replace=False,
            shift=π/2, coefficient=0.5, unbiased=True)
```

### 原理

在大参数量场景下，每步只随机选取 $K$ 个参数坐标（`n_samples=K`）执行两点移位，对其他坐标的梯度估计为零。当 `unbiased=True` 时，被采样坐标的梯度乘以无偏缩放因子 $\frac{P}{K}$，使期望值等于完整 `psr` 梯度：

$$\mathbb{E}[\hat{g}_k] = \frac{\partial \langle O \rangle}{\partial \theta_k}$$

### 特点

- 只需 $2K + 1$（`replace=False`）或 $2K$（`replace=True`）次函数调用
- 方差与 $K$ 成反比，但期望无偏（`unbiased=True` 时）
- 适合变分量子本征求解器（VQE）的随机梯度下降优化
- 当 `n_samples=P` 时退化为精确的 `psr`

---

## 4. 多参数混合偏导 `mpsr`

```python
from aicir.qml import mpsr
mixed = mpsr(fn, params, parameter_indices=None, *, shift=π/2, coefficient=0.5)
```

### 原理

通过参数移位计算多参数的**混合高阶偏导数**：

$$\frac{\partial^M \langle O \rangle}{\partial \theta_{k_1} \cdots \partial \theta_{k_M}} = \left(\frac{1}{2}\right)^M \sum_{\mathbf{s} \in \{±1\}^M} \left(\prod_j s_j\right) \langle O \rangle\!\left(\theta + \frac{\pi}{2}\mathbf{s}\odot\mathbf{e}_{k}\right)$$

其中对 $M$ 个选定坐标的所有 $2^M$ 个符号组合求和。

### 特点

- 返回**标量**混合偏导数值
- `parameter_indices` 可使用整数平铺索引或多维元组索引
- 需要 $2^M$ 次函数调用（$M$ 为指定坐标数）

---

## 5. 有限差分 `fd`

```python
from aicir.qml import fd
grad = fd(fn, params, *, eps=1e-3, mode="central")
```

### 原理

对每个参数坐标 $\theta_k$，采用数值差商近似偏导数：

| `mode` | 公式 | 精度阶 | 函数调用次数 |
|--------|------|--------|-------------|
| `"central"` | $\frac{f(\theta_k+\varepsilon)-f(\theta_k-\varepsilon)}{2\varepsilon}$ | $O(\varepsilon^2)$ | $2P$ |
| `"forward"` | $\frac{f(\theta_k+\varepsilon)-f(\theta)}{\varepsilon}$ | $O(\varepsilon)$ | $P+1$ |
| `"backward"` | $\frac{f(\theta)-f(\theta_k-\varepsilon)}{\varepsilon}$ | $O(\varepsilon)$ | $P+1$ |

`"forward"` 和 `"backward"` 模式会复用基点函数值 $f(\theta)$（在所有坐标上只调用一次），因此总调用次数为 $P+1$。

### 步长选择

aicir 默认使用 `complex64`（单精度）模拟，浮点精度约为 $10^{-7}$。最优步长约为 $\varepsilon \approx \delta^{1/2}$（中心差分）或 $\delta^{1/3}$（单侧），其中 $\delta$ 为机器精度。因此：

- **`complex64` / `float32` 目标**（默认）：推荐 `eps=1e-3`（默认值），避免浮点消去误差
- **`complex128` / `float64` 目标**：可使用更小步长，如 `eps=1e-6`

### 特点

- 无需了解电路生成元频谱，适合任意可微目标
- 对**非旋转门**参数（如 `u3`、任意幺正门）有效，而 `psr` 对此类门需要知道频谱
- 存在截断误差与舍入误差的权衡

---

## 6. 伴随微分 `ad`

```python
from aicir.qml import ad
grad = ad(circuit, observable, *, backend=None, return_value=False)
# 返回 np.ndarray，形状为 (可微参数数量,)
# return_value=True 时返回 (grad, expectation_value)
```

### 原理

伴随微分是专为**无噪声态向量模拟器**设计的反向模式微分方法。对于完整电路 $U = U_P \cdots U_1$，期望值 $\langle O \rangle = \langle\psi|O|\psi\rangle$ 关于旋转角 $\theta_k$（对应生成元 $G_k$，$U_k = e^{-i\theta_k G_k/2}$）的梯度为：

$$\frac{\partial \langle O \rangle}{\partial \theta_k} = 2\operatorname{Re}\!\left[\langle \lambda_k | \frac{\partial U_k}{\partial \theta_k} | \psi_{k-1} \rangle\right] = \operatorname{Im}\!\left[\langle \lambda_k | G_k | \psi_k \rangle\right]$$

其中 $|\psi_k\rangle = U_k \cdots U_1|0\rangle$，$|\lambda_k\rangle = U_P^\dagger \cdots U_{k+1}^\dagger O |\psi\rangle$。

### 算法流程

```
前向传播：|ψ_0⟩ = |0⟩  →  |ψ_1⟩  →  ···  →  |ψ_P⟩ = |ψ⟩
初始化：  |λ_P⟩ = O|ψ⟩，|φ⟩ = |ψ⟩（从末端开始反向）
反向传播：对 k = P, P-1, ..., 1：
  若 U_k 可微：grad[k] = Im⟨λ_k | G_k | φ⟩
  更新：|φ⟩ ← U_k^† |φ⟩，|λ⟩ ← U_k^† |λ⟩
```

### 特点

- **时间复杂度**：$O(P)$ 次局部门作用（与参数数量线性相关）
- **空间复杂度**：仅需 $O(1)$ 额外状态存储（$|\lambda\rangle$ 与 $|\phi\rangle$ 各一份），远优于有限差分的 $O(P)$ 额外电路
- 与 `psr` 相比：相同精度，但减少 $\sim P$ 倍的模拟开销
- **结构感知**：直接作用于 `Circuit` 对象，可微门为 `rx/ry/rz/crx/cry/crz/rzz`；其他门（`H`、`cx`、`u3`、任意幺正门）正常传播但不被微分
- **NPU 兼容**：梯度读取 $\operatorname{Im}\langle\lambda|G|\phi\rangle$ 全程在设备端完成，不触发设备→主机拷贝
- `observable` 可为矩阵（numpy/后端张量）或 `Hamiltonian` 对象

### 示例

```python
from aicir.qml import ad
from aicir import Circuit, NumpyBackend, ry, cx
from aicir.channel.operators import Hamiltonian
import numpy as np

bk = NumpyBackend()
circuit = Circuit(ry(0.4, 0), cx(1, [0]), ry(0.7, 1), n_qubits=2)
H = Hamiltonian(n_qubits=2).term(1.0, {"Z": [0]}).term(0.5, {"Z": [1]})

grad = ad(circuit, H, backend=bk)          # 2 个 ry 门，返回长度为 2 的梯度
grad, energy = ad(circuit, H, backend=bk, return_value=True)
```

---

## 7. 量子自然梯度 `qng`

```python
from aicir.qml import qng
natural_grad = qng(fn, state_fn, params, *, damping=1e-6, metric_eps=1e-3)
```

### 原理

量子自然梯度（Quantum Natural Gradient, QNG）使用量子 Fisher 信息矩阵（QFIM）对普通梯度进行预条件：

$$\tilde{g} = (F + \lambda I)^{-1}\nabla L$$

其中 $F$ 是 ansatz 纯态在 Fubini-Study 几何下的 QFIM，$\lambda$ 对应 `damping`，用于处理奇异或病态的 QFIM。优化器可用 `theta -= eta * natural_grad` 更新参数。

`qng` 默认：

- 使用 `psr(fn, params)` 计算普通目标梯度 $\nabla L$
- 使用 `state_fn(params)` 返回的纯态，通过中心有限差分估计 QFIM
- 解 `(F + damping * I) @ natural_grad = grad`

QFIM 的纯态公式为：

$$F_{ij}=4\operatorname{Re}\left[\langle\partial_i\psi|\partial_j\psi\rangle-\langle\partial_i\psi|\psi\rangle\langle\psi|\partial_j\psi\rangle\right]$$

### NPU 兼容设计

`state_fn` 可以返回：

- numpy 态向量
- Torch/NPU 后端张量（如 `state.data`）
- aicir `State` / `StateVector` 对象

设备张量会在 QFIM 求解前 detach 并移动到主机，用 NumPy 完成小规模线性求解，避免依赖 NPU 的复杂矩阵求逆支持；目标函数返回的 NPU/Torch 标量仍可被 `psr` / `fd` / `auto` 路径正常处理。

### 示例

```python
import numpy as np
from aicir import Circuit, NumpyBackend, State, ry
from aicir.qml import qng

bk = NumpyBackend()
z = bk.cast(np.diag([1.0, -1.0]).astype(np.complex64))

def state_fn(theta):
    circuit = Circuit(ry(theta[0], 0), n_qubits=1)
    return State.zero_state(1, bk).evolve(circuit.unitary(backend=bk))

def objective(theta):
    state = state_fn(theta)
    return bk.expectation_sv(state.data, z)

theta = np.array([0.5])
direction, grad, qfim = qng(objective, state_fn, theta, return_gradient=True, return_qfim=True)
# qfim ≈ [[1.0]]
# direction ≈ grad ≈ [-sin(0.5)]
```

若已从其它方法得到普通梯度或 QFIM，也可以直接传入：

```python
direction = qng(
    None,
    None,
    params,
    grad=ordinary_grad,
    qfim=precomputed_qfim,
    damping=1e-5,
)
```

---

## 8. 分块对角量子自然梯度 `bdqng`

```python
from aicir.qml import bdqng
natural_grad = bdqng(fn, state_fn, params, *, blocks=None, block_size=1)
```

### 原理

Block-diagonal QNG 是 QNG 的可扩展近似。它把参数划分为多个 block，只保留同一 block 内的 QFIM 项，并忽略跨 block 的耦合：

$$F \approx \operatorname{blockdiag}(F_{B_1}, F_{B_2}, \ldots, F_{B_m})$$

每个 block 独立求解：

$$\tilde{g}_{B_k} = (F_{B_k} + \lambda I)^{-1}\nabla L_{B_k}$$

与完整 `qng` 相比，`bdqng` 不需要求解一个大的 $P \times P$ 线性系统，而是求解多个小系统。默认 `block_size=1`，即 diagonal QNG；也可以通过 `blocks=[[0, 1], [2, 3]]` 指定更大的参数块。

### 特点

- `blocks=None` 时按 flat 参数顺序用 `block_size` 连续分块
- `blocks` 可显式指定 flat 索引或多维 tuple 索引，且必须覆盖每个参数一次
- 可传入 `qfim_blocks` 复用预计算的 block QFIM
- 支持与 `qng` 相同的普通梯度来源：`psr`、`fd`、`auto` 或外部 `grad`
- `state_fn` 同样可返回 numpy 态向量、Torch/NPU 张量或 aicir `State`

### 示例

```python
import numpy as np
from aicir import Circuit, NumpyBackend, State, ry
from aicir.qml import bdqng

bk = NumpyBackend()
z = bk.cast(np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.complex64))

def state_fn(theta):
    circuit = Circuit(ry(theta[0], 0), ry(theta[1], 1), n_qubits=2)
    return State.zero_state(2, bk).evolve(circuit.unitary(backend=bk))

def objective(theta):
    state = state_fn(theta)
    return bk.expectation_sv(state.data, z)

theta = np.array([0.4, -0.2])
direction = bdqng(objective, state_fn, theta, blocks=[[0], [1]])
```

若一个 ansatz 按 layer 排列参数，可按 layer 分块：

```python
direction, qfim_blocks = bdqng(
    objective,
    state_fn,
    params,
    blocks=[[0, 1, 2], [3, 4, 5]],
    return_qfim_blocks=True,
)
```

---

## 方法选择指南

```
无噪声态向量模拟，含任意可微门（u3 等），已在 Torch/NPU 后端？
  → auto（单次反向传播，支持任意门类型）

含噪声线路 / 硬件执行？
  → psr（含噪声电路亦精确）或 fd（任意目标）

参数量极大（>1000），允许梯度有噪声？
  → spsr（随机采样 K 坐标，2K+1 次调用）

需要混合高阶偏导？
  → mpsr

调试/原型/不清楚生成元频谱？
  → fd（无任何假设，任意后端，eps=1e-3 for float32）

无噪声态向量模拟，参数全部为 Pauli 旋转门？
  → ad（最高效：O(P) 门作用，仅 O(1) 额外状态）

希望优化步长适应量子态空间几何？
  → qng（用 QFIM 逆预条件 psr/fd/auto 或外部梯度）

参数量较大，但仍希望利用量子态空间几何？
  → bdqng（按 block 近似 QFIM，默认 block_size=1）
```

---

## 完整参数说明

### `auto(fn, params, *, backend=None)`

| 参数 | 说明 |
|------|------|
| `fn` | 接受 `torch.Tensor`（`requires_grad=True`）参数，返回可微标量张量的目标函数 |
| `backend` | Torch/NPU 后端（决定参数的 dtype 和 device，默认 CPU TorchBackend） |

### `psr(fn, params, *, shift=π/2, coefficient=0.5)`

| 参数 | 说明 |
|------|------|
| `fn` | 接受完整参数数组、返回标量的目标函数 |
| `params` | 当前参数值，支持标量和任意形状数组 |
| `shift` | 正负移位量（默认 π/2，对应 Pauli 旋转门标准频谱） |
| `coefficient` | 差分系数（默认 0.5） |

### `spsr(fn, params, *, n_samples, rng, replace, shift, coefficient, unbiased)`

| 参数 | 说明 |
|------|------|
| `n_samples` | 每步采样的参数坐标数 |
| `rng` | 随机数生成器（int seed 或 `np.random.Generator`） |
| `replace` | 是否有放回采样 |
| `unbiased` | 是否乘以 P/K 无偏因子（默认 True） |

### `mpsr(fn, params, parameter_indices=None, *, shift, coefficient)`

| 参数 | 说明 |
|------|------|
| `parameter_indices` | 指定混合偏导的坐标：整数、元组索引或其列表；`None` 表示对所有参数 |

### `fd(fn, params, *, eps=1e-3, mode="central")`

| 参数 | 说明 |
|------|------|
| `eps` | 差分步长（float32 模拟推荐 `1e-3`，float64 可用 `1e-6`） |
| `mode` | `"central"`（二阶精度）、`"forward"`、`"backward"` |

### `ad(circuit, observable, *, backend=None, return_value=False)`

| 参数 | 说明 |
|------|------|
| `circuit` | 完全绑定参数的 `Circuit` 对象（含 `rx/ry/rz/crx/cry/crz/rzz` 可微门） |
| `observable` | Hermitian 算符矩阵或 `Hamiltonian` 对象 |
| `backend` | 计算后端（默认 `NumpyBackend`） |
| `return_value` | 若为 `True`，同时返回期望值 `⟨O⟩` |

### `qng(fn, state_fn, params, *, grad=None, qfim=None, gradient_method="psr", gradient_kwargs=None, shift=π/2, coefficient=0.5, metric_eps=1e-3, damping=1e-6, backend=None, return_gradient=False, return_qfim=False)`

| 参数 | 说明 |
|------|------|
| `fn` | 接受完整参数数组、返回标量损失的目标函数；当 `grad` 已提供时可为 `None` |
| `state_fn` | 接受完整参数数组、返回纯态 ansatz 终态；当 `qfim` 已提供时可为 `None` |
| `grad` | 可选的普通梯度；提供后跳过 `fn` 的梯度计算 |
| `qfim` | 可选的 QFIM；提供后跳过 `state_fn` 的 QFIM 估计 |
| `gradient_method` | 未提供 `grad` 时使用的普通梯度方法：`"psr"`、`"fd"` 或 `"auto"` |
| `gradient_kwargs` | 透传给普通梯度方法的额外参数 |
| `shift` / `coefficient` | `gradient_method="psr"` 时的参数移位配置 |
| `metric_eps` | 用于 QFIM 态导数中心差分的步长 |
| `damping` | 加到 QFIM 对角线上的非负阻尼项，缓解奇异或病态矩阵 |
| `backend` | `gradient_method="auto"` 时使用的 Torch/NPU 后端 |
| `return_gradient` | 若为 `True`，额外返回普通梯度 |
| `return_qfim` | 若为 `True`，额外返回 QFIM |

### `bdqng(fn, state_fn, params, *, blocks=None, block_size=1, grad=None, qfim_blocks=None, gradient_method="psr", gradient_kwargs=None, shift=π/2, coefficient=0.5, metric_eps=1e-3, damping=1e-6, backend=None, return_gradient=False, return_qfim_blocks=False)`

| 参数 | 说明 |
|------|------|
| `fn` | 接受完整参数数组、返回标量损失的目标函数；当 `grad` 已提供时可为 `None` |
| `state_fn` | 接受完整参数数组、返回纯态 ansatz 终态；当 `qfim_blocks` 已提供时可为 `None` |
| `blocks` | 显式参数分块，例如 `[[0, 1], [2, 3]]`；每个参数必须出现且只出现一次 |
| `block_size` | 未提供 `blocks` 时按 flat 参数顺序连续分块的大小，默认 `1` |
| `grad` | 可选的普通梯度；提供后跳过 `fn` 的梯度计算 |
| `qfim_blocks` | 可选的 block QFIM 列表；提供后跳过 `state_fn` 的 block QFIM 估计 |
| `gradient_method` | 未提供 `grad` 时使用的普通梯度方法：`"psr"`、`"fd"` 或 `"auto"` |
| `gradient_kwargs` | 透传给普通梯度方法的额外参数 |
| `shift` / `coefficient` | `gradient_method="psr"` 时的参数移位配置 |
| `metric_eps` | 用于 block QFIM 态导数中心差分的步长 |
| `damping` | 加到每个 QFIM block 对角线上的非负阻尼项 |
| `backend` | `gradient_method="auto"` 时使用的 Torch/NPU 后端 |
| `return_gradient` | 若为 `True`，额外返回普通梯度 |
| `return_qfim_blocks` | 若为 `True`，额外返回 QFIM block 列表 |
