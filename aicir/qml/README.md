# aicir.qml — 量子机器学习梯度与梯度无关优化工具包

本模块（`aicir/qml/deriv/` 包，内部按功能拆成 `fn_gradient`/`hessian`/`adjoint`/`qfim`/`qng`/`rotosolve` 等子模块）实现了量子电路参数梯度、高阶导数、QFIM 几何预条件和显式 gradient-free 坐标精确最小化方法。所有方法均与 `NumpyBackend`、`GPUBackend`、`NPUBackend` 兼容，后端返回的张量（包括自动微分追踪张量、复数标量、加速器设备张量）均可直接作为目标函数返回值，无需手动调用 `float()` 或 `to_numpy()`。子模块拆分不影响导入路径：`from aicir.qml import psr` 与 `from aicir.qml.deriv import psr` 都继续可用。

---

## 0. 返回类型契约

- **一律返回 NumPy 数组**：`auto`、`psr`、`psr4`、`spsr`、`spsa`、`mpsr`、`fd`、`ad`、`hessian`、`qfim`/`metric_tensor`、`qfim_diag`、`qfim_blocks`、`qng`、`bdqng`。即便内部为保持设备驻留而用 torch 张量计算（如 `auto`、`qfim` 系在 torch 态输入下），最终都会转换为主机端 NumPy 数组再返回；`qng`/`bdqng` 即便 `state_fn` 返回 torch/NPU 张量，或预先算好的 `grad`/`qfim`/`qfim_blocks` 本身是 torch 张量，也统一转换回 NumPy。
- **入参决定出参设备**：只有 `kqng`、`dqng`、`rotosolve` 三个例外——当传入 `NPUBackend`（或 NPU-family backend），或预先算好的 `grad`/`qfim_diag`/`kfac_factors`（`rotosolve` 则是 `params`）本身是 torch 张量时，它们返回**同设备的 torch 张量**；否则同样返回 NumPy 数组。

调用方若把 `qng`/`bdqng` 的返回值和 `kqng`/`dqng`/`rotosolve` 的返回值混用（例如同一优化循环里既用 `qng` 又用 `dqng`），需要注意前者恒定是 NumPy，后者可能是 torch 张量，不能假设两者返回类型一致。

---

## 1. 方法选择指南

```
无噪声态向量模拟，含任意可微门（u3 等），已在 Torch/NPU 后端？
  → auto（单次反向传播，支持任意门类型）

含噪声线路 / 硬件执行？
  → psr（含噪声电路亦精确）或 fd（任意目标）

参数量极大（>1000），允许梯度有噪声？
  → spsr（随机采样 K 坐标，2K 次调用）

参数量极大，硬件/含噪声目标每步只能承受少量函数调用？
  → spsa（随机全方向扰动，2K 次调用，K=n_samples）

需要混合高阶偏导？
  → mpsr

需要完整二阶导数矩阵？
  → hessian（Pauli 旋转目标用 psr；任意黑盒目标用 fd）

调试/原型/不清楚生成元频谱？
  → fd（无任何假设，任意后端，eps=1e-3 for float32）

无噪声态向量模拟，参数全部为 Pauli 旋转门？
  → ad（最高效：O(P) 门作用，仅 O(1) 额外状态）

希望优化步长适应量子态空间几何？
  → qng（用 QFIM 逆预条件 psr/fd/spsa/auto 或外部梯度）

只需要量子态空间几何本身？
  → qfim / metric_tensor（返回完整 Fubini-Study QFIM）

参数量较大，但仍希望利用量子态空间几何？
  → bdqng（按 block 近似 QFIM，默认 block_size=1）

参数以矩阵/层为自然结构，希望用更紧的低成本几何近似？
  → kqng（Kronecker 因子近似每个 QFIM block）

参数量很大，只能承受最便宜的几何预条件？
  → dqng（只用 QFIM 对角线，便宜但粗糙）

目标关于单个旋转参数满足正弦结构，希望完全不计算梯度？
  → rotosolve（逐坐标解析最小化，gradient-free）
```

---

## 2. 接口与参数说明

### 公共接口一览

| 函数          | 全称                                               | 复杂度（函数调用次数）                      | 适用场景                                      |
| ------------- | -------------------------------------------------- | ------------------------------------------- | --------------------------------------------- |
|               |                                                    |                                             |                                               |
|               |                                                    |                                             |                                               |
| `auto`      | Automatic Differentiation                          | 1 次反向传播                                | Torch/NPU 后端，自动微分图                    |
| `psr`       | Parameter-Shift Rule                               | $2P$                                      | 通用旋转门，无噪声或含噪声均可                |
| `psr4`      | Four-Term PSR                                      | $4P$                                      | 生成元谱为 $\{-1,0,1\}$ 的门（如激发门），标准两项 `psr` 不适用 |
| `spsr`      | Stochastic PSR                                     | $2K$（$K \leq P$ 次采样）               | 大参数量随机坐标梯度                          |
| `spsa`      | Simultaneous Perturbation Stochastic Approximation | $2K$（$K$ 个扰动方向）                  | 极大参数量/硬件噪声下的随机全方向梯度         |
| `mpsr`      | Multi-parameter PSR                                | $2^M$（$M$ 个坐标的混合偏导数）         | 高阶混合偏导                                  |
| `hessian`   | Hessian                                            | $O(P^2)$ 二阶移位或有限差分              | 完整二阶导数矩阵                              |
| `fd`        | Finite Difference                                  | $2P$（中心差）或 $P+1$（单侧）          | 任意可微目标，黑盒                            |
| `ad`        | Adjoint Differentiation                            | $O(P)$ 次门作用，$O(1)$ 额外存储        | 无噪声态向量模拟，效率最高                    |
| `qfim` / `metric_tensor` | Quantum Fisher Information Matrix | $2P+1$ 次态函数调用                    | 直接读取 Fubini-Study 几何矩阵                |
| `qng`       | Quantum Natural Gradient                           | 梯度调用 +$2P+1$ 次态函数调用             | 用 QFIM 逆预条件梯度，加速 ansatz 优化        |
| `bdqng`     | Block-diagonal QNG                                 | 梯度调用 +$2P+1$ 次态函数调用，分块求解   | 按参数块近似 QFIM，适合大参数 ansatz          |
| `kqng`      | KFAC-style QNG                                     | 梯度调用 + 分块因子估计，Kronecker 因子求解 | 用 Kronecker 因子近似 QFIM block              |
| `dqng`      | Diagonal QNG                                       | 梯度调用 +$2P+1$ 次态函数调用，逐坐标除法 | QFIM 对角近似，最便宜但最粗糙                 |
| `rotosolve` | ROTOSOLVE                                          | 每个坐标每 sweep 3 次函数调用               | 正弦结构目标的逐坐标 gradient-free 精确最小化 |

$P$：可微参数数量。$K$：随机采样坐标数或扰动方向数。

---

### 完整参数说明

#### `auto(fn, params, *, backend=None)`

| 参数        | 说明                                                                           |
| ----------- | ------------------------------------------------------------------------------ |
| `fn`      | 接受`torch.Tensor`（`requires_grad=True`）参数，返回可微标量张量的目标函数 |
| `backend` | Torch/NPU 后端（决定参数的 dtype 和 device，默认 CPU GPUBackend）              |

#### `psr(fn, params, *, shift=π/2, coefficient=0.5)`

| 参数            | 说明                                               |
| --------------- | -------------------------------------------------- |
| `fn`          | 接受完整参数数组、返回标量的目标函数               |
| `params`      | 当前参数值，支持标量和任意形状数组                 |
| `shift`       | 正负移位量（默认 π/2，对应 Pauli 旋转门标准频谱） |
| `coefficient` | 差分系数（默认 0.5）                               |

#### `psr4(fn, params, *, shifts=(π/2, 3π/2), coefficients=None)`

| 参数            | 说明                                                                 |
| --------------- | -------------------------------------------------------------------- |
| `fn`          | 接受完整参数数组、返回标量的目标函数                                 |
| `params`      | 当前参数值，支持标量和任意形状数组                                   |
| `shifts`      | 两个正移位量 `(s1, s2)`，默认 `(π/2, 3π/2)`                         |
| `coefficients` | 可选系数对 `(c1, c2)`；`None` 时使用激发门默认值 `((√2+1)/4√2, (√2−1)/4√2)` |

#### `spsr(fn, params, *, n_samples, rng, replace, shift, coefficient, unbiased)`

| 参数          | 说明                                               |
| ------------- | -------------------------------------------------- |
| `n_samples` | 每步采样的参数坐标数                               |
| `rng`       | 随机数生成器（int seed 或`np.random.Generator`） |
| `replace`   | 是否有放回采样                                     |
| `unbiased`  | 是否乘以 P/K 无偏因子（默认 True）                 |

#### `spsa(fn, params, *, eps=1e-3, n_samples=1, rng=None, perturbations=None)`

| 参数              | 说明                                                                  |
| ----------------- | --------------------------------------------------------------------- |
| `eps`           | 同时扰动步长，默认`1e-3`                                            |
| `n_samples`     | 随机扰动方向数量；每个方向只需 2 次函数调用                           |
| `rng`           | 随机数生成器（int seed 或`np.random.Generator`）                    |
| `perturbations` | 可选的固定扰动向量/矩阵；所有元素必须非零，提供后样本数由其第一维决定 |

#### `mpsr(fn, params, parameter_indices=None, *, shift, coefficient)`

| 参数                  | 说明                                                                |
| --------------------- | ------------------------------------------------------------------- |
| `parameter_indices` | 指定混合偏导的坐标：整数、元组索引或其列表；`None` 表示对所有参数 |

#### `hessian(fn, params, *, method="auto", shift=π/2, coefficient=0.5, eps=1e-3)`

| 参数 | 说明 |
| --- | --- |
| `method` | `"auto"`（默认）：先用 psr 二阶公式估计每个对角元并与 fd 对照，若某个对角元不一致（生成元谱不是标准 Pauli 旋转的 $\{-1,0,1\}$）整体降级为 fd，并发出 `RuntimeWarning`；`"psr"`：纯二阶参数移位公式 + `mpsr` 混合偏导，检测到对角元与 fd 不一致时直接抛 `ValueError`（不静默换算法）；`"fd"`：纯中心有限差分，不做 psr 对照 |
| `shift` / `coefficient` | `method="auto"`/`"psr"` 时的参数移位配置 |
| `eps` | 二阶差分步长；`method="fd"` 时用于全部差分，`method="auto"`/`"psr"` 时用于内部一致性对照 |

二阶有限差分需要除以 $\varepsilon^2$，比一阶 `fd`（除以 $\varepsilon$）对浮点消去误差敏感得多：在 aicir 默认的 `complex64` 单精度模拟下，目标函数本身的精度约 `1e-7`，用默认 `eps=1e-3` 做二阶差分的舍入误差可达 `1e-7 / eps² ≈ 0.1` 量级。因此：

- `method="fd"` 是纯差分路径，精度直接受此限制。需要准确结果时显式传入更大的 `eps`（如 `1e-2`），或改用 `method="psr"`（标准 Pauli 旋转门下解析精确）。
- `method="auto"`/`"psr"` 的内部一致性对照会先估计 fd 自身的不确定度（比较 `eps` 与 `2*eps` 两个步长的差分结果），容差取该不确定度与固定容差中的较大者。**只有当 fd 足够可信、却仍与 psr 显著不符时**，才判定生成元谱不是标准 Pauli 旋转并降级/报错——所以标准旋转门电路不会因为 fd 本身不准而被误判，默认 `eps` 下即可放心使用。

#### `fd(fn, params, *, eps=1e-3, mode="central")`

| 参数     | 说明                                                        |
| -------- | ----------------------------------------------------------- |
| `eps`  | 差分步长（float32 模拟推荐`1e-3`，float64 可用 `1e-6`） |
| `mode` | `"central"`（二阶精度）、`"forward"`、`"backward"`    |

#### `ad(circuit, observable, *, backend=None, return_value=False)`

| 参数             | 说明                                                                         |
| ---------------- | ---------------------------------------------------------------------------- |
| `circuit`      | 完全绑定参数的`Circuit` 对象（含 `rx/ry/rz/crx/cry/crz/rzz/rxx` 可微门） |
| `observable`   | Hermitian 算符矩阵或`Hamiltonian` 对象                                     |
| `backend`      | 计算后端（默认`NumpyBackend`）                                             |
| `return_value` | 若为`True`，同时返回期望值 `⟨O⟩`                                       |

#### `qfim(state_fn, params, *, metric_eps=1e-3, backend=None)` / `metric_tensor(...)`

| 参数 | 说明 |
| --- | --- |
| `state_fn` | 接受完整参数数组、返回纯态 ansatz 终态；可返回 `State`、NumPy 数组或 Torch-family 后端张量 |
| `metric_eps` | 态导数中心差分步长 |
| `backend` | `state_fn` 返回 Torch/NPU `State` 或张量时用于保持设备/内积语义 |

`qfim_diag(...)` 返回对角线，`qfim_blocks(..., blocks=..., block_size=...)` 返回 block QFIM 列表。

#### `qng(fn, state_fn, params, *, grad=None, qfim=None, gradient_method="psr", gradient_kwargs=None, shift=π/2, coefficient=0.5, metric_eps=1e-3, damping=1e-6, backend=None, return_gradient=False, return_qfim=False)`

| 参数                        | 说明                                                                               |
| --------------------------- | ---------------------------------------------------------------------------------- |
| `fn`                      | 接受完整参数数组、返回标量损失的目标函数；当`grad` 已提供时可为 `None`         |
| `state_fn`                | 接受完整参数数组、返回纯态 ansatz 终态；当`qfim` 已提供时可为 `None`           |
| `grad`                    | 可选的普通梯度；提供后跳过`fn` 的梯度计算                                        |
| `qfim`                    | 可选的 QFIM；提供后跳过`state_fn` 的 QFIM 估计                                   |
| `gradient_method`         | 未提供`grad` 时使用的普通梯度方法：`"psr"`、`"fd"`、`"spsa"` 或 `"auto"`；传入 `"spsr"` 会抛 `ValueError`（见下） |
| `gradient_kwargs`         | 透传给普通梯度方法的额外参数                                                       |
| `shift` / `coefficient` | `gradient_method="psr"` 时的参数移位配置                                         |
| `metric_eps`              | 用于 QFIM 态导数中心差分的步长                                                     |
| `damping`                 | 加到 QFIM 对角线上的非负阻尼项，缓解奇异或病态矩阵                                 |
| `backend`                 | `gradient_method="auto"` 时使用的 Torch/NPU 后端                                 |
| `return_gradient`         | 若为`True`，额外返回普通梯度                                                     |
| `return_qfim`             | 若为`True`，额外返回 QFIM                                                        |

#### `bdqng(fn, state_fn, params, *, blocks=None, block_size=1, grad=None, qfim_blocks=None, gradient_method="psr", gradient_kwargs=None, shift=π/2, coefficient=0.5, metric_eps=1e-3, damping=1e-6, backend=None, return_gradient=False, return_qfim_blocks=False)`

| 参数                        | 说明                                                                               |
| --------------------------- | ---------------------------------------------------------------------------------- |
| `fn`                      | 接受完整参数数组、返回标量损失的目标函数；当`grad` 已提供时可为 `None`         |
| `state_fn`                | 接受完整参数数组、返回纯态 ansatz 终态；当`qfim_blocks` 已提供时可为 `None`    |
| `blocks`                  | 显式参数分块，例如`[[0, 1], [2, 3]]`；每个参数必须出现且只出现一次               |
| `block_size`              | 未提供`blocks` 时按 flat 参数顺序连续分块的大小，默认 `1`                      |
| `grad`                    | 可选的普通梯度；提供后跳过`fn` 的梯度计算                                        |
| `qfim_blocks`             | 可选的 block QFIM 列表；提供后跳过`state_fn` 的 block QFIM 估计                  |
| `gradient_method`         | 未提供`grad` 时使用的普通梯度方法：`"psr"`、`"fd"`、`"spsa"` 或 `"auto"`；传入 `"spsr"` 会抛 `ValueError`（见下） |
| `gradient_kwargs`         | 透传给普通梯度方法的额外参数                                                       |
| `shift` / `coefficient` | `gradient_method="psr"` 时的参数移位配置                                         |
| `metric_eps`              | 用于 block QFIM 态导数中心差分的步长                                               |
| `damping`                 | 加到每个 QFIM block 对角线上的非负阻尼项                                           |
| `backend`                 | `gradient_method="auto"` 时使用的 Torch/NPU 后端                                 |
| `return_gradient`         | 若为`True`，额外返回普通梯度                                                     |
| `return_qfim_blocks`      | 若为`True`，额外返回 QFIM block 列表                                             |

#### `kqng(fn, state_fn, params, *, blocks=None, factor_shapes=None, block_size=None, grad=None, kfac_factors=None, gradient_method="psr", gradient_kwargs=None, shift=π/2, coefficient=0.5, metric_eps=1e-3, damping=1e-6, backend=None, return_gradient=False, return_kfac_factors=False)`

| 参数                        | 说明                                                                               |
| --------------------------- | ---------------------------------------------------------------------------------- |
| `fn`                      | 接受完整参数数组、返回标量损失的目标函数；当`grad` 已提供时可为 `None`         |
| `state_fn`                | 接受完整参数数组、返回纯态 ansatz 终态；当`kfac_factors` 已提供时可为 `None`   |
| `blocks`                  | 显式参数分块，例如`[[0, 1, 2, 3], [4, 5, 6, 7]]`                                 |
| `factor_shapes`           | 每个 block 的 Kronecker reshape 形状，例如`(2, 2)` 或 `[(2, 2), (2, 2)]`       |
| `block_size`              | 未提供`blocks` / `factor_shapes` 时按 flat 参数顺序连续分块的大小              |
| `grad`                    | 可选的普通梯度；提供后跳过`fn` 的梯度计算                                        |
| `kfac_factors`            | 可选的 Kronecker 因子列表`[(A, B), ...]`；提供后跳过 `state_fn` 的因子估计     |
| `gradient_method`         | 未提供`grad` 时使用的普通梯度方法：`"psr"`、`"fd"`、`"spsa"` 或 `"auto"`；传入 `"spsr"` 会抛 `ValueError`（见下） |
| `gradient_kwargs`         | 透传给普通梯度方法的额外参数                                                       |
| `shift` / `coefficient` | `gradient_method="psr"` 时的参数移位配置                                         |
| `metric_eps`              | 用于 QFIM block 态导数中心差分的步长                                               |
| `damping`                 | Kronecker 因子阻尼；实现中使用`sqrt(damping)` 加到两个因子对角线上               |
| `backend`                 | Torch/NPU 后端；传入`NPUBackend` 时启用不搬回 CPU 的设备端路径                   |
| `return_gradient`         | 若为`True`，额外返回普通梯度                                                     |
| `return_kfac_factors`     | 若为`True`，额外返回 Kronecker 因子列表                                          |

#### `dqng(fn, state_fn, params, *, grad=None, qfim_diag=None, gradient_method="psr", gradient_kwargs=None, shift=π/2, coefficient=0.5, metric_eps=1e-3, damping=1e-6, backend=None, return_gradient=False, return_qfim_diag=False)`

| 参数                        | 说明                                                                               |
| --------------------------- | ---------------------------------------------------------------------------------- |
| `fn`                      | 接受完整参数数组、返回标量损失的目标函数；当`grad` 已提供时可为 `None`         |
| `state_fn`                | 接受完整参数数组、返回纯态 ansatz 终态；当`qfim_diag` 已提供时可为 `None`      |
| `grad`                    | 可选的普通梯度；提供后跳过`fn` 的梯度计算                                        |
| `qfim_diag`               | 可选的 QFIM 对角线；提供后跳过`state_fn` 的 QFIM 对角估计                        |
| `gradient_method`         | 未提供`grad` 时使用的普通梯度方法：`"psr"`、`"fd"`、`"spsa"` 或 `"auto"`；传入 `"spsr"` 会抛 `ValueError`（见下） |
| `gradient_kwargs`         | 透传给普通梯度方法的额外参数                                                       |
| `shift` / `coefficient` | `gradient_method="psr"` 时的参数移位配置                                         |
| `metric_eps`              | 用于 QFIM 对角态导数中心差分的步长                                                 |
| `damping`                 | 加到每个 QFIM 对角项上的非负阻尼项                                                 |
| `backend`                 | Torch/NPU 后端；传入`NPUBackend` 时启用不搬回 CPU 的设备端路径                   |
| `return_gradient`         | 若为`True`，额外返回普通梯度                                                     |
| `return_qfim_diag`        | 若为`True`，额外返回 QFIM 对角线                                                 |

#### `rotosolve(fn, params, *, n_sweeps=1, parameter_indices=None, shift=π/2, atol=1e-12, backend=None, return_value=False)`

| 参数                  | 说明                                                         |
| --------------------- | ------------------------------------------------------------ |
| `fn`                | 接受完整参数数组/张量、返回标量损失的目标函数                |
| `params`            | 当前参数值，支持标量和任意形状数组/张量                      |
| `n_sweeps`          | 坐标扫描轮数，每轮依次更新`parameter_indices` 中的坐标     |
| `parameter_indices` | 可选更新坐标，支持 flat 整数索引、多维 tuple 索引或列表      |
| `shift`             | 两点偏移量，默认`π/2`，对应频率为 1 的标准 ROTOSOLVE 公式 |
| `atol`              | 拟合正弦振幅低于该阈值时认为该坐标平坦，不更新               |
| `backend`           | Torch/NPU 后端；传入后目标值、三角函数和坐标更新保持在设备端 |
| `return_value`      | 若为`True`，同时返回最终目标函数值                         |

## 3. 自动微分 `auto`

```python
from aicir.qml import auto
grad = auto(fn, params, *, backend=None)
```

### 原理

`auto` 利用 **PyTorch 反向传播**穿透量子电路的所有门运算，实现"幺正传播的反向传播"（Backpropagation through the Unitary）。其工作方式与经典神经网络的自动微分完全相同：

1. 参数以 `torch.Tensor(requires_grad=True)` 形式传入目标函数
2. 目标函数在 Torch 后端上构建并演化电路（所有操作留在自动微分计算图中）
3. 对期望值标量调用 `.backward()`，读取 `.grad`

$$
\frac{\partial \langle O \rangle}{\partial \boldsymbol{\theta}} = \text{torch.autograd.grad}(\langle O \rangle(\boldsymbol{\theta}),\ \boldsymbol{\theta})
$$

### NPU 兼容设计

参数张量的 `dtype` 和 `device` 均从 `backend` 中读取（`backend._dtype` → 对应实数类型，`backend._device` → NPU/CUDA 设备），确保：

- 对于 `NPUBackend`，参数从创建起就驻留在 NPU 上，梯度在设备端累积，最后才将 `.grad` 一次性移至主机
- 不会在梯度计算中触发额外的设备↔主机往返

| 精度后端              | 参数`dtype`     |
| --------------------- | ----------------- |
| `complex64`（默认） | `torch.float32` |
| `complex128`        | `torch.float64` |

### 与其他方法的对比

- `auto` 与 `ad` 都只需一次反向遍历，但机制不同：
  - `ad`：手动实现伴随传播，只需 `NumpyBackend`，对电路结构有限制（仅 Pauli 旋转门可微）
  - `auto`：依赖 PyTorch 自动微分，支持任意可微操作（`u3`、任意幺正门等），但需要 Torch/NPU 后端且目标函数不能断开计算图
- 若目标函数调用了 `float()`、`.item()`、`.detach()` 或 `to_numpy()`，计算图将断裂，`auto` 会抛出明确错误

### 示例

```python
from aicir.qml import auto
from aicir import Circuit, State, ry
from aicir.backends.gpu_backend import GPUBackend
import numpy as np

bk = GPUBackend(device="cpu")
z   = bk.cast(np.diag([1.0, -1.0]).astype(np.complex64))

def fn(theta):                        # theta: torch.Tensor, requires_grad=True
    circuit = Circuit(ry(theta, 0), n_qubits=1)
    state   = State.zero_state(1, bk).evolve(circuit.unitary(backend=bk))
    return bk.expectation_sv(state.data, z)   # 必须返回 torch 张量，不调用 float()

grad = auto(fn, np.array(0.5), backend=bk)
# grad ≈ -sin(0.5) ≈ -0.4794
```

---

## 4. 参数移位规则 `psr`

```python
from aicir.qml import psr
grad = psr(fn, params, *, shift=π/2, coefficient=0.5)
```

### 原理

对于以 Pauli 旋转门 $U_k(\theta_k) = e^{-i\theta_k G_k/2}$ 参数化的量子电路，可观测量期望 $\langle O \rangle(\theta_k)$ 关于 $\theta_k$ 的梯度满足**精确的**两点公式：

$$
\frac{\partial \langle O \rangle}{\partial \theta_k} = \frac{1}{2}\left[\langle O \rangle\!\left(\theta_k + \frac{\pi}{2}\right) - \langle O \rangle\!\left(\theta_k - \frac{\pi}{2}\right)\right]
$$

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

## 5. 随机参数移位规则 `spsr`

```python
from aicir.qml import spsr
grad = spsr(fn, params, *, n_samples=1, rng=None, replace=False,
            shift=π/2, coefficient=0.5, unbiased=True)
```

### 原理

在大参数量场景下，每步只随机选取 $K$ 个参数坐标（`n_samples=K`）执行两点移位，对其他坐标的梯度估计为零。当 `unbiased=True` 时，被采样坐标的梯度乘以无偏缩放因子 $\frac{P}{K}$，使期望值等于完整 `psr` 梯度：

$$
\mathbb{E}[\hat{g}_k] = \frac{\partial \langle O \rangle}{\partial \theta_k}
$$

### 特点

- 只需 $2K$ 次函数调用
- 方差与 $K$ 成反比，但期望无偏（`unbiased=True` 时）
- 适合变分量子本征求解器（VQE）的随机梯度下降优化
- 当 `n_samples=P` 时退化为精确的 `psr`

---

## 6. SPSA `spsa`

```python
from aicir.qml import spsa
grad = spsa(fn, params, *, eps=1e-3, n_samples=1, rng=None,
            perturbations=None)
```

### 原理

SPSA（Simultaneous Perturbation Stochastic Approximation）每次采样一个随机扰动向量 $\Delta_k$，同时扰动所有参数，只用两次目标函数调用估计全量梯度：

$$
\hat{g}_k(\theta)=\frac{L(\theta+\epsilon\Delta_k)-L(\theta-\epsilon\Delta_k)}{2\epsilon}\Delta_k^{-1}
$$

默认 $\Delta_k$ 的每个元素从 $\{-1,+1\}$ 中采样，因此 $\Delta_k^{-1}=\Delta_k$。当 `n_samples > 1` 时，`spsa` 会对多个扰动方向的估计取平均以降低方差。

### 特点

- SPSA 是随机梯度估计器，不是 gradient-free 方法；它估计的是 $\nabla L$
- 每个扰动方向只需 2 次函数调用，总调用次数为 $2K$，与参数数量 $P$ 无关
- 对硬件执行、含噪声目标和超大参数量 ansatz 很有用
- 估计通常有偏且噪声较大，适合作为随机优化器的梯度输入
- 可通过 `perturbations` 传入固定扰动向量，便于复现实验和测试

### 示例

```python
import numpy as np
from aicir.qml import spsa

def objective(theta):
    return np.sum(np.cos(theta))

theta = np.array([0.1, 0.2, 0.3])
grad = spsa(objective, theta, eps=1e-3, n_samples=8, rng=123)
```

---

## 6.5 四项参数移位规则 `psr4`

```python
from aicir.qml import psr4
grad = psr4(fn, params, *, shifts=(π/2, 3π/2), coefficients=None)
```

### 原理

标准 `psr` 的两项公式只对生成元谱为 $\{-\frac{1}{2},\frac{1}{2}\}$ 的门（Pauli 旋转门）精确。对于生成元谱为 $\{-1,0,1\}$ 的门（如量子化学激发门 SingleExcitation/DoubleExcitation），梯度需要四个采样点：

$$
\frac{\partial \langle O \rangle}{\partial \theta_k} = c_1\left[\langle O \rangle(\theta_k+s_1) - \langle O \rangle(\theta_k-s_1)\right] - c_2\left[\langle O \rangle(\theta_k+s_2) - \langle O \rangle(\theta_k-s_2)\right]
$$

默认 $(s_1,s_2)=(\pi/2, 3\pi/2)$，$(c_1,c_2)=\left(\frac{\sqrt2+1}{4\sqrt2}, \frac{\sqrt2-1}{4\sqrt2}\right)$，与 PennyLane 的 SingleExcitation/DoubleExcitation 参数移位配方一致。

### 特点

- 每个参数需要 **4 次** 函数调用，总共 $4P$
- `psr4` 已注册进 §15 的 DiffMethod 注册表（`category="fn_gradient"`），可经 `get_diff("psr4")`/`resolve_diff("psr4")` 发现和调用
- `psr4` **不**参与 `select_diff` 的自动优选：调用方需要知道线路门的生成元谱是 $\{-1,0,1\}$ 才应显式选用 `psr4`，`select_diff` 无法从上下文推断门类型
- 若门生成元谱既不是标准 Pauli 旋转的 $\{-\frac{1}{2},\frac{1}{2}\}$，也不是 $\{-1,0,1\}$，应改用 `fd` 或 `auto`

### 示例

```python
import numpy as np
from aicir.qml import psr4

# 生成元谱 {-1,0,1} 的合成目标（模拟激发门期望值随角度变化的三频结构）
def objective(theta):
    return 0.5 * np.cos(theta[0])

grad = psr4(objective, np.array([0.3]))
```

---

## 7. 多参数混合偏导 `mpsr`

```python
from aicir.qml import mpsr
mixed = mpsr(fn, params, parameter_indices=None, *, shift=π/2, coefficient=0.5)
```

### 原理

通过参数移位计算多参数的**混合高阶偏导数**：

$$
\frac{\partial^M \langle O \rangle}{\partial \theta_{k_1} \cdots \partial \theta_{k_M}} = \left(\frac{1}{2}\right)^M \sum_{\mathbf{s} \in \{±1\}^M} \left(\prod_j s_j\right) \langle O \rangle\!\left(\theta + \frac{\pi}{2}\mathbf{s}\odot\mathbf{e}_{k}\right)
$$

其中对 $M$ 个选定坐标的所有 $2^M$ 个符号组合求和。

### 特点

- 返回**标量**混合偏导数值
- `parameter_indices` 可使用整数平铺索引或多维元组索引
- 需要 $2^M$ 次函数调用（$M$ 为指定坐标数）

---

## 8. 有限差分 `fd`

```python
from aicir.qml import fd
grad = fd(fn, params, *, eps=1e-3, mode="central")
```

### 原理

对每个参数坐标 $\theta_k$，采用数值差商近似偏导数：

| `mode`       | 公式                                                                     | 精度阶               | 函数调用次数 |
| -------------- | ------------------------------------------------------------------------ | -------------------- | ------------ |
| `"central"`  | $\frac{f(\theta_k+\varepsilon)-f(\theta_k-\varepsilon)}{2\varepsilon}$ | $O(\varepsilon^2)$ | $2P$       |
| `"forward"`  | $\frac{f(\theta_k+\varepsilon)-f(\theta)}{\varepsilon}$                | $O(\varepsilon)$   | $P+1$      |
| `"backward"` | $\frac{f(\theta)-f(\theta_k-\varepsilon)}{\varepsilon}$                | $O(\varepsilon)$   | $P+1$      |

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

## 9. 伴随微分 `ad`

```python
from aicir.qml import ad
grad = ad(circuit, observable, *, backend=None, return_value=False)
# 返回 np.ndarray，形状为 (可微参数数量,)
# return_value=True 时返回 (grad, expectation_value)
```

### 原理

伴随微分是专为**无噪声态向量模拟器**设计的反向模式微分方法。对于完整电路 $U = U_P \cdots U_1$，期望值 $\langle O \rangle = \langle\psi|O|\psi\rangle$ 关于旋转角 $\theta_k$（对应生成元 $G_k$，$U_k = e^{-i\theta_k G_k/2}$）的梯度为：

$$
\frac{\partial \langle O \rangle}{\partial \theta_k} = 2\operatorname{Re}\!\left[\langle \lambda_k | \frac{\partial U_k}{\partial \theta_k} | \psi_{k-1} \rangle\right] = \operatorname{Im}\!\left[\langle \lambda_k | G_k | \psi_k \rangle\right]
$$

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
- **结构感知**：直接作用于 `Circuit` 对象，可微门为 `rx/ry/rz/crx/cry/crz/rzz/rxx`；其他门（`H`、`cx`、`u3`、任意幺正门）正常传播但不被微分
- **NPU 兼容**：梯度读取 $\operatorname{Im}\langle\lambda|G|\phi\rangle$ 全程在设备端完成，不触发设备→主机拷贝
- `observable` 可为矩阵（numpy/后端张量）或 `Hamiltonian` 对象

### 示例

```python
from aicir.qml import ad
from aicir import Circuit, NumpyBackend, Hamiltonian, ry, cx
import numpy as np

bk = NumpyBackend()
circuit = Circuit(ry(0.4, 0), cx(1, [0]), ry(0.7, 1), n_qubits=2)
H = Hamiltonian([("ZI", 1.0), ("IZ", 0.5)])

grad = ad(circuit, H, backend=bk)          # 2 个 ry 门，返回长度为 2 的梯度
grad, energy = ad(circuit, H, backend=bk, return_value=True)
```

---

## 10. 量子自然梯度 `qng`

```python
from aicir.qml import qng
natural_grad = qng(fn, state_fn, params, *, damping=1e-6, metric_eps=1e-3)
```

### 原理

量子自然梯度（Quantum Natural Gradient, QNG）使用量子 Fisher 信息矩阵（QFIM）对普通梯度进行预条件：

$$
\tilde{g} = (F + \lambda I)^{-1}\nabla L
$$

其中 $F$ 是 ansatz 纯态在 Fubini-Study 几何下的 QFIM，$\lambda$ 对应 `damping`，用于处理奇异或病态的 QFIM。优化器可用 `theta -= eta * natural_grad` 更新参数。

`qng` 默认：

- 使用 `psr(fn, params)` 计算普通目标梯度 $\nabla L$
- 使用 `state_fn(params)` 返回的纯态，通过中心有限差分估计 QFIM
- 解 `(F + damping * I) @ natural_grad = grad`

也可通过 `gradient_method="spsa"` 使用 SPSA 作为普通梯度来源，适合只想用少量目标函数调用估计 QNG 预条件前梯度的场景。

`gradient_method="spsr"` 会被 `qng`/`bdqng`/`kqng`/`dqng` 统一拒绝，抛出 `ValueError`：`spsr` 每次只随机采样部分参数坐标估计梯度，这种局部采样与 QNG 族的整体度规预条件组合尚未验证，叠加后可能得到有偏或不稳定的自然梯度方向。需要随机化梯度时请改用覆盖全部坐标的 `"spsa"`。

QFIM 的纯态公式为：

$$
F_{ij}=4\operatorname{Re}\left[\langle\partial_i\psi|\partial_j\psi\rangle-\langle\partial_i\psi|\psi\rangle\langle\psi|\partial_j\psi\rangle\right]
$$

### NPU 兼容设计

`state_fn` 可以返回：

- numpy 态向量
- Torch/NPU 后端张量（如 `state.data`）
- aicir `State` 对象

设备张量会在 QFIM 求解前 detach 并移动到主机，用 NumPy 完成小规模线性求解，避免依赖 NPU 的复杂矩阵求逆支持；目标函数返回的 NPU/Torch 标量仍可被 `psr` / `fd` / `spsa` / `auto` 路径正常处理。

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

## 11. 分块对角量子自然梯度 `bdqng`

```python
from aicir.qml import bdqng
natural_grad = bdqng(fn, state_fn, params, *, blocks=None, block_size=1)
```

### 原理

Block-diagonal QNG 是 QNG 的可扩展近似。它把参数划分为多个 block，只保留同一 block 内的 QFIM 项，并忽略跨 block 的耦合：

$$
F \approx \operatorname{blockdiag}(F_{B_1}, F_{B_2}, \ldots, F_{B_m})
$$

每个 block 独立求解：

$$
\tilde{g}_{B_k} = (F_{B_k} + \lambda I)^{-1}\nabla L_{B_k}
$$

与完整 `qng` 相比，`bdqng` 不需要求解一个大的 $P \times P$ 线性系统，而是求解多个小系统。默认 `block_size=1`，即 diagonal QNG；也可以通过 `blocks=[[0, 1], [2, 3]]` 指定更大的参数块。

### 特点

- `blocks=None` 时按 flat 参数顺序用 `block_size` 连续分块
- `blocks` 可显式指定 flat 索引或多维 tuple 索引，且必须覆盖每个参数一次
- 可传入 `qfim_blocks` 复用预计算的 block QFIM
- 支持与 `qng` 相同的普通梯度来源：`psr`、`fd`、`spsa`、`auto` 或外部 `grad`
- `state_fn` 同样可返回 numpy 态向量、Torch/NPU 张量或 aicir `State`

### 示例

```python
，，import numpy as np
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

## 12. KFAC-style 量子自然梯度 `kqng`

```python
from aicir.qml import kqng
natural_grad = kqng(fn, state_fn, params, *, factor_shapes=None, damping=1e-6)
```

### 原理

KFAC-style QNG 把每个 QFIM block 近似为 Kronecker 积：

$$
F_B \approx A_B \otimes B_B
$$

若 block 梯度按 `factor_shapes=(m, n)` reshape 为矩阵 $G_B$，则用两个小矩阵求解代替一个大矩阵求解：

$$
A_B^{(\lambda)} X_B (B_B^{(\lambda)})^T = G_B
$$

其中 $A_B^{(\lambda)}$、$B_B^{(\lambda)}$ 是加阻尼后的 Kronecker 因子。`kqng` 会返回 `vec(X_B)` 作为该 block 的自然梯度近似。

### 使用方式

- 可直接传入 `kfac_factors=[(A, B), ...]`，跳过 QFIM 估计
- 可通过 `state_fn` 估计 block QFIM，再从每个 block 提取 Kronecker 因子
- `factor_shapes=(m, n)` 或 `factor_shapes=[(m1, n1), ...]` 指定每个 block 的矩阵形状，要求 `m*n == block_size`
- 若 `params` 是二维数组且未显式指定 block，默认把整个参数矩阵作为一个 KFAC block
- 若参数是一维数组且未显式指定形状，默认退化为 `(P, 1)` 的单列 Kronecker 因子

### NPU 兼容设计

当传入 `NPUBackend`，或 `grad` / `kfac_factors` 是 Torch/NPU 张量时，`kqng` 会走设备端路径：

- 普通梯度保持为 Torch/NPU tensor
- `state_fn` 返回的态向量保持为 Torch/NPU tensor
- QFIM block、Kronecker 因子和 `torch.linalg.solve` 均在设备端完成
- 不调用 `.cpu()`、`to_numpy()` 或 NumPy 线性代数

### 示例

```python
import numpy as np
from aicir.qml import kqng

grad = np.array([[6.0, 10.0], [12.0, 20.0]])
A = np.diag([2.0, 4.0])
B = np.diag([3.0, 5.0])

direction = kqng(
    None,
    None,
    np.zeros((2, 2)),
    grad=grad,
    kfac_factors=[(A, B)],
    damping=0.0,
)
# direction ≈ [[1, 1], [1, 1]]
```

若使用 ansatz 态估计 KFAC 因子：

```python
direction, factors = kqng(
    objective,
    state_fn,
    params,
    blocks=[[0, 1, 2, 3]],
    factor_shapes=[(2, 2)],
    return_kfac_factors=True,
)
```

---

## 13. 对角量子自然梯度 `dqng`

```python
from aicir.qml import dqng
natural_grad = dqng(fn, state_fn, params, *, damping=1e-6, metric_eps=1e-3)
```

### 原理

Diag QNG 是 QNG 最便宜的近似，只保留 QFIM 的对角线：

$$
F \approx \operatorname{diag}(F_{11}, F_{22}, \ldots, F_{PP})
$$

每个参数坐标独立预条件：

$$
\tilde{g}_i = \frac{\nabla_i L}{F_{ii} + \lambda}
$$

它比 `qng` 和 `bdqng` 更便宜，不需要矩阵求逆或分块线性求解；代价是完全忽略参数间耦合，因此近似更粗。

### NPU 兼容设计

当传入 `NPUBackend`，或 `grad` / `qfim_diag` 是 Torch/NPU 张量时，`dqng` 会走设备端路径：

- `fn` 返回的标量保持为 Torch/NPU tensor
- `state_fn` 返回的态向量保持为 Torch/NPU tensor
- QFIM 对角线、阻尼项和最终除法均在设备端完成
- 不调用 `.cpu()`、`to_numpy()` 或 NumPy 线性代数

因此 NPU ansatz 的梯度预条件不会把中间数据搬回 CPU。若用户的 `fn` 或 `state_fn` 自己调用了 `.cpu()` / `to_numpy()`，则会由用户函数自身造成主机搬移，`dqng` 不会额外执行该操作。

### 示例

```python
import numpy as np
from aicir import Circuit, NumpyBackend, State, ry
from aicir.qml import dqng

bk = NumpyBackend()
z = bk.cast(np.diag([1.0, -1.0]).astype(np.complex64))

def state_fn(theta):
    circuit = Circuit(ry(theta[0], 0), n_qubits=1)
    return State.zero_state(1, bk).evolve(circuit.unitary(backend=bk))

def objective(theta):
    state = state_fn(theta)
    return bk.expectation_sv(state.data, z)

theta = np.array([0.5])
direction, qfim_diag = dqng(objective, state_fn, theta, return_qfim_diag=True)
# qfim_diag ≈ [1.0]
# direction ≈ [-sin(0.5)]
```

也可以直接传入预计算的对角 QFIM：

```python
direction = dqng(
    None,
    None,
    params,
    grad=ordinary_grad,
    qfim_diag=precomputed_diag,
    damping=1e-5,
)
```

---

## 14. ROTOSOLVE

```python
from aicir.qml import rotosolve
params = rotosolve(fn, params, *, n_sweeps=1, parameter_indices=None,
                   backend=None, return_value=False)
```

### 原理

ROTOSOLVE 是显式的 **gradient-free** 方法。它不估计梯度，也不使用有限差分梯度；它利用单个旋转参数上的三角结构，直接解出该坐标的最优值：

$$
L(\theta_k)=a\sin(\theta_k+\phi)+c
$$

固定其它参数后，`rotosolve` 对当前坐标 $k$ 评估：

$$
L_0=L(\theta_k),\quad L_+=L(\theta_k+s),\quad L_-=L(\theta_k-s)
$$

默认 $s=\pi/2$。由三点值拟合该坐标上的正弦曲线，然后解析更新到最小点：

$$
\Delta\theta_k=-\frac{\pi}{2}-\operatorname{atan2}(A, B)
$$

其中

$$
A=\frac{L_0-\frac{1}{2}(L_+ + L_-)}{1-\cos s},\qquad
B=\frac{L_+ - L_-}{2\sin s}
$$

因此它执行的是逐坐标精确最小化，而不是沿梯度方向下降。

### NPU 兼容设计

当传入 `backend=NPUBackend`，或 `params` 本身是 Torch/NPU tensor 时，`rotosolve` 会走设备端路径：

- 参数张量、目标函数标量、`atan2` 和坐标更新均保持为 Torch/NPU tensor
- 不调用 `.cpu()`、`.numpy()`、`to_numpy()` 或 NumPy 三角函数处理设备标量
- 返回值也是设备端 tensor；`return_value=True` 时最终 loss 同样保持在设备端

因此 NPU ansatz 的 ROTOSOLVE 优化过程中不会由 `rotosolve` 把中间数据搬回 CPU。若用户的 `fn` 自己调用 `.cpu()` / `to_numpy()`，则主机搬移来自用户函数本身。

### 示例

```python
import numpy as np
from aicir.qml import rotosolve

def objective(theta):
    return 2.0 * np.sin(theta[0] + 0.3) + 0.5 * np.sin(theta[1] - 0.2)

theta = np.array([0.7, -0.9])
theta, value = rotosolve(objective, theta, return_value=True)
# value ≈ -2.5
```

量子线路示例：

```python
import numpy as np
from aicir import Circuit, NumpyBackend, State, ry
from aicir.qml import rotosolve

bk = NumpyBackend()
z = bk.cast(np.diag([1.0, -1.0]).astype(np.complex64))

def objective(theta):
    circuit = Circuit(ry(theta[0], 0), n_qubits=1)
    state = State.zero_state(1, bk).evolve(circuit.unitary(backend=bk))
    return bk.expectation_sv(state.data, z)

theta, value = rotosolve(objective, np.array([0.5]), return_value=True)
# value ≈ -1.0
```

---

## 15. DiffMethod 策略注册表（`aicir.qml.diff`）

上面的 `deriv.py` 把每种梯度方法实现为独立函数；`aicir/qml/diff/` 在其上叠加一层**策略注册表**，把 fn-based 全梯度方法（统一契约 `(fn, params, **kw) -> 梯度向量`）按名字单点登记，便于按字符串解析、按上下文自动选择、以及注册自定义方法。它镜像 `aicir/gates/` 的 `GateSpec` 注册表习惯，全部 API 从 `aicir.qml` 顶层再导出。

注册表按 `category` 索引**全部** 10 个内置方法，分三类（契约各异）：

- `fn_gradient`（`(fn, params) -> 梯度向量`）：`psr` / `fd` / `auto` / `spsa` / `spsr`；**唯一**参与 `resolve_diff`/`select_diff` 自动分发的类别。
- `circuit_gradient`（`(circuit, observable) -> 梯度`）：`ad`（伴随微分）。
- `preconditioner`（`(fn, state_fn, params) -> 方向/度规`）：`qng` / `bdqng` / `kqng` / `dqng`。

`mpsr` 有意**不**纳入注册表（它返回标量混合偏导，不满足任何统一契约），仍可作为 `qml.mpsr` 直接调用。`ad` 与 `qng` 族虽登记入表（供 `get_diff`/`registered_diffs(category=...)` 检索发现），但 `resolve_diff` 仅解析 `fn_gradient`——`resolve_diff('ad'|'qng')` 抛 `ValueError`，确保经典优化器不会拿到签名不兼容的可调用。

### API 一览

| 函数                                                             | 说明                                                                                                                                                         |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `DiffMethod`                                                   | 冻结数据类，描述一个方法：`name`、`fn`、`aliases`、`category`、`exact`、`stochastic`、`requires_torch`、`supports_shots`、`supports_noise` |
| `register_diff(spec, *, overwrite=False)`                      | 注册一个`DiffMethod`；名称/别名冲突时报错                                                                                                                  |
| `unregister_diff(name)`                                        | 注销一个方法（含别名），未注册则静默                                                                                                                         |
| `get_diff(name) -> DiffMethod \| None`                          | 按规范名或别名查 spec，未注册返回`None`                                                                                                                    |
| `registered_diffs(category=None) -> tuple[str, ...]`           | 返回已注册方法的规范名；`category` 非空时按类别过滤                                                                                                        |
| `canonical_diff(name) -> str`                                  | 别名 → 规范名；未注册名原样返回                                                                                                                             |
| `resolve_diff(name) -> Callable`                               | 返回`fn_gradient` 方法的 `fn`；未注册名或非 `fn_gradient` 类别（如 `ad`/`qng`）抛 `ValueError`                                                   |
| `select_diff(*, backend=None, shots=None, noisy=False) -> str` | 按上下文选择方法名（纯函数）                                                                                                                                 |

### `select_diff` 选择策略

纯函数，不执行梯度，只返回方法名。过滤规则：`requires_torch` 的方法仅在 Torch 系后端保留；有 shots 时丢弃 `supports_shots=False`；`noisy=True` 时丢弃 `supports_noise=False`。偏好顺序为 **`auto` → `psr` → `fd`**（`spsa`/`spsr` 不参与自动优选，仅在调用方显式请求时使用）。例如：Torch 后端 + 无 shots + 无噪声 → `auto`；有 shots 或含噪声 → `psr`；非 Torch 后端 → `psr`。

### 示例：解析与选择

```python
from aicir.qml import resolve_diff, select_diff, registered_diffs

registered_diffs()                 # 全部 10 项（含 ad/qng 族）
registered_diffs(category="fn_gradient")  # ('psr', 'fd', 'auto', 'spsa', 'spsr')

grad_fn = resolve_diff("psr")      # 即 deriv.psr 本身
# grad = grad_fn(objective, params)

select_diff()                      # 'psr'（无 Torch 后端）
select_diff(backend=gpu_backend)   # 'auto'（Torch 系后端、无 shots/噪声）
select_diff(backend=gpu_backend, shots=1024)  # 'psr'（auto 不支持 shots）
```

### 示例：经优化器按名调用

`aicir.optimizer.params` 的优化器（`GD`/`Adam`/`ScipyMinimize`）通过注册表分发，因此 `gradient_method` 可用任意已注册方法名：

```python
from aicir.optimizer.params import Adam

opt = Adam(gradient_method="spsr", gradient_kwargs={"rng": 0})
result = opt.minimize(objective, init_params)
```

注意：`auto` 需要 Torch 系后端与连接 autograd 图的目标，无法经经典优化器的黑盒数值目标使用；在该路径上请求 `auto` 会被守卫拦截并给出清晰错误，应改用 `psr`/`fd`/`spsa`/`spsr`。

### 示例：注册自定义方法

```python
from aicir.qml import DiffMethod, register_diff, unregister_diff

def my_grad(fn, params, **kw):
    ...  # 返回与 params 同形状的梯度向量

register_diff(DiffMethod("mygrad", my_grad, exact=False))
# 之后 resolve_diff("mygrad") / Adam(gradient_method="mygrad") 均可用
unregister_diff("mygrad")
```

> `select_diff` 的首个调用方是下面的 `qfun`：`@qfun(..., differential="auto")` 的 `.grad` 即经 `select_diff(backend, shots, noisy)` 自动择优。

## 16. 量子函数 `qfun`（`aicir.qml.qfun`）

`qfun` 把"量子函数 + 设备 + 测量 + 梯度"统一成一个可调用对象，是上面注册表的高层消费方。约定：被装饰的函数**构造并返回一个 `Circuit`**（不依赖全局 tape），观测量在装饰器上用 `observable=` 声明；调用得期望值，`.grad` 得梯度。函数体也可返回 `expval`/`probs`/`sample` 测量对象在体内声明测量意图（见下"测量返回构造器"）。

```python
from aicir import Circuit, Hamiltonian, ry
from aicir.qml import qfun

H = Hamiltonian([("Z", 1.0)])

@qfun(device="numpy", differential="psr", observable=H)
def cost(theta):
    c = Circuit(n_qubits=1)
    c.append(ry(theta, 0))
    return c

cost(0.3)        # 期望值 <H> = cos(0.3)
cost.grad(0.3)   # 梯度 = -sin(0.3)
```

### 装饰器参数

| 参数             | 默认        | 说明                                                                                                                                |
| ---------------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `device`       | `"numpy"` | 后端：`numpy`/`cpu` → `NumpyBackend`，`gpu`/`torch` → `GPUBackend`，`npu` → `NPUBackend`                         |
| `differential` | `"psr"`   | 梯度方法名，经 §15 注册表`resolve_diff` 解析（仅 `fn_gradient`）；`"auto"` 走 `select_diff` 自动择优                       |
| `observable`   | 可选        | 单个可观测量（如`Hamiltonian`）或其列表；列表时 `cost(x)` 返回数组、`grad` 返回 Jacobian。函数体返回 `expval(...)` 时可省略 |
| `shots`        | `None`    | `None`/`0` 为精确期望；正整数走 shot 估计                                                                                       |
| `noise_model`  | `None`    | 提供`NoiseModel` 时把噪声附加到线路、经密度矩阵模拟读取期望；`differential="auto"` 据此以 `noisy=True` 择优                   |

### 行为约定

- 函数体须返回 `Circuit` 或 `expval`/`probs`/`sample` 测量对象，否则抛 `TypeError`；返回线路含未绑定参数抛 `ValueError`。
- `observable=` 不在装饰期强制：返回裸 `Circuit` 且无装饰器 `observable=` 时于**调用期**抛 `ValueError`（函数体改用 `expval(c, H)` 则免）。
- `differential="auto"` 在非 Torch 后端降级为 `psr`，有 shots/噪声时同样回退（见 §15 选择策略）。

### 测量返回构造器（`expval`/`probs`/`sample`）

函数体可返回测量对象表达测量意图（无全局 tape，故显式携带 `circuit`）：

```python
from aicir import Circuit, Hamiltonian, ry
from aicir.qml import qfun, expval, probs, sample

H = Hamiltonian([("Z", 1.0)])

@qfun(differential="psr")            # observable 省略，由 expval 提供
def cost(theta):
    c = Circuit(n_qubits=1)
    c.append(ry(theta, 0))
    return expval(c, H)              # <H>，可微：cost.grad(x)

@qfun()
def dist(theta):
    c = Circuit(n_qubits=1)
    c.append(ry(theta, 0))
    return probs(c, wires=None)      # 概率向量（wires 可限制并边缘化）

@qfun(shots=128)
def shots_counts(theta):
    c = Circuit(n_qubits=1)
    c.append(ry(theta, 0))
    return sample(c)                 # counts 字典；无 shots= 抛 ValueError
```

- `expval(circuit, observable)` → 期望值（float/数组），是唯一可微返回；`probs`/`sample` 调用 `.grad` 抛 `ValueError`。
- `probs(circuit, wires=None)` → 整寄存器概率向量；给 `wires` 时边缘化到对应比特。
- `sample(circuit, wires=None)` → counts；shots 取自装饰器 `shots=`。

### 多参数（单数组参）

单观测量下，`cost`/`cost.grad` 接受标量或一维数组参数，`grad` 返回与输入同形：

```python
H = Hamiltonian([("ZI", 1.0)])      # <Z on q0> = cos(theta[0])

@qfun(observable=H)
def cost(theta):                    # theta 为一维数组
    c = Circuit(n_qubits=2)
    c.append(ry(theta[0], 0))
    c.append(ry(theta[1], 1))
    return c

cost(np.array([0.3, 0.7]))          # cos(0.3)
cost.grad(np.array([0.3, 0.7]))     # [-sin(0.3), 0.0]，形状 (2,)
```

### 多测量（observable 列表 → Jacobian）

`observable=[H1, H2, ...]` 时 `cost(x)` 返回 `(n_obs,)` 数组，`cost.grad(x)` 返回 Jacobian（标量参 → `(n_obs,)`；向量参 → `(n_obs, n_param)`）：

```python
@qfun(observable=[Hamiltonian([("Z", 1.0)]), Hamiltonian([("X", 1.0)])])
def cost(theta):
    c = Circuit(n_qubits=1)
    c.append(ry(theta, 0))
    return c

cost(0.3)        # [cos(0.3), sin(0.3)]
cost.grad(0.3)   # [-sin(0.3), cos(0.3)]，形状 (2,)
```

### 噪声路径

```python
from aicir import NoiseModel, BitFlipChannel

nm = NoiseModel().add_channel(BitFlipChannel(0, 0.25), after_gates=["ry"])

@qfun(observable=Hamiltonian([("Z", 1.0)]), noise_model=nm)
def cost(theta):
    c = Circuit(n_qubits=1)
    c.append(ry(theta, 0))
    return c

cost(0.3)   # (1-2*0.25)*cos(0.3) = 0.5*cos(0.3)
```

### 接入 VQE / QAOA

`BasicVQE`/`BasicQAOA` 可直接以**单观测量** `qfun` 作代价函数（旁路各自的 ansatz/编排，`energy`/梯度委托给 `cost`/`cost.grad`）：

```python
from aicir.vqc import BasicVQE, BasicQAOA

@qfun(observable=Hamiltonian([("Z", 1.0)]))
def cost(theta):
    c = Circuit(n_qubits=1)
    c.append(ry(theta[0], 0))
    return c

BasicVQE(cost=cost, n_params=1).run(max_iters=200, lr=0.3, init_params=np.array([0.1]))
BasicQAOA(cost=cost, p=1).run(max_iters=200, lr=0.3, init_params=np.array([0.1, 0.0]))
```

> VQC 复用：`BasicVQE`/`BasicSSVQE`/`BasicVQD.parameter_shift_gradient()` 均统一调用 `aicir.qml.deriv.psr`（单一真源）。自定义 QNN/VQC 时也建议复用 §4–§7 的 `psr`/`spsr`/`mpsr`，勿各自重写 parameter-shift。

---

## 17. PyTorch 量子层 `QLayer`（`aicir.qml.qlayer`）

`QLayer` 把一个 `QFun` 封装成 `torch.nn.Module`，可**一行嵌入经典 PyTorch 网络**做混合训练（需 `torch`；无 torch 时 `aicir.qml.QLayer is None`）。

它不重写梯度：前向调用 `qfun(params)` 取期望值，反向调用 `qfun.grad(params)` 取参数移位 Jacobian 并接入 torch autograd。因此该层与 `QFun` 的后端解耦——`device="numpy"/"gpu"/"npu"` 皆可用，梯度方法仍走 §15 注册表这一单一真源。经典输入与可训练权重经 `torch.cat` 拼成单个参数向量喂给 `qfun`（`cat` 可微，故梯度同时回流到**前置经典层的输入**与**本层权重**）。

> 说明：`QFun.__call__` 返回主机标量，故量子部分的数值仍经 numpy 往返；`QLayer` 的价值在于把量子层无缝接入 torch 计算图，底层模拟仍由 `QFun` 的后端（含 NPU/GPU）执行。

### 构造与前向

```python
QLayer(qfun, n_weights, *, init=None, dtype=torch.float32)
```

| 参数        | 说明                                                                     |
| ----------- | ------------------------------------------------------------------------ |
| `qfun`      | 已声明设备/观测量/梯度方法的 `QFun`（须为 `expval` 返回，可微）           |
| `n_weights` | 本层可训练权重个数（拼在输入之后传给 `qfun`）；为 `0` 时纯由外部输入驱动 |
| `init`      | 权重初值（数组/张量，长度须为 `n_weights`）；缺省在 `[0, 2π)` 均匀采样   |
| `dtype`     | 权重与输出张量浮点 dtype，默认 `torch.float32`                           |

`forward(inputs=None)`：`inputs is None` → 仅用权重；`inputs` 一维 → 拼 `[inputs, weights]`；`inputs` 二维 `(batch, features)` → 逐行求值堆叠成 `(batch,)` 或 `(batch, n_obs)`。单观测量输出标量、多观测量输出 `(n_obs,)`。

### 示例：纯量子可训练层

```python
import numpy as np, torch
from aicir import Circuit, Hamiltonian, ry
from aicir.qml import qfun, QLayer

@qfun(observable=Hamiltonian([("Z", 1.0)]))     # <Z> = cos(theta)
def cost(theta):
    c = Circuit(n_qubits=1)
    c.append(ry(theta[0], 0))
    return c

layer = QLayer(cost, n_weights=1, init=np.array([0.3]))
opt = torch.optim.SGD(layer.parameters(), lr=0.3)
for _ in range(50):
    opt.zero_grad()
    loss = layer()        # 极小化 <Z> → theta → π
    loss.backward()
    opt.step()
```

### 示例：经典 → 量子混合网络

```python
# Linear 输出作为量子层输入，梯度回流到 Linear.weight 与量子权重
model = torch.nn.Sequential(
    torch.nn.Linear(4, 1),
    QLayer(cost, n_weights=0),   # 全部参数来自输入
)
y = model(torch.randn(8, 4))         # 批量前向 → (8,)
```

---

## 18. 批量量子层 `BatchLayer`（`aicir.qml.qlayer`）

`BatchLayer` 是 `QLayer` 的批量姊妹：要求**固定模板线路**（参数门限单参数的
`rx/ry/rz/crx/cry/crz/rzz/rxx`），换来整批一次演化与端到端 torch autograd——
底层走 `aicir.core.batch.BatchSV`（实部/虚部实张量、逐样本门角度、NPU 安全，
无复数内核），不做逐参数 PSR 循环，反向由 torch 原生求导完成。

```python
BatchLayer(circuit, n_inputs, *, backend, init=None, dtype=torch.float32)
```

- 参数约定与 `QLayer` 的 `cat([inputs, weights])` 同序：模板 `circuit.parameters`
  首用序的前 `n_inputs` 个为数据编码参数（逐样本取 `inputs` 对应列），其余为
  本层可训练权重（`torch.nn.Parameter`）。
- 读出为逐比特 `<Z_q>`：`forward(inputs (batch, n_inputs))` 返回
  `(batch, n_qubits)`；一维输入返回 `(n_qubits,)`。梯度同时回流到权重与
  前置经典层输入。
- `backend` 须为 torch 系（`GPUBackend`/`NPUBackend`），决定 device 与 dtype。

```python
import torch
from aicir import Circuit, Parameter, cx, rx, ry, rzz
from aicir.backends.gpu_backend import GPUBackend
from aicir.qml import BatchLayer

x0, x1 = Parameter("x0"), Parameter("x1")
w0, w1, w2 = (Parameter(f"w{i}") for i in range(3))
template = Circuit(rx(x0, 0), rx(x1, 1),          # 数据编码（逐样本）
                   ry(w0, 0), ry(w1, 1),          # 可训练权重
                   rzz(w2, 0, 1), cx(1, [0]), n_qubits=2)

layer = BatchLayer(template, n_inputs=2, backend=GPUBackend())
model = torch.nn.Sequential(torch.nn.Linear(8, 2), layer)  # 混合网络
out = model(torch.randn(32, 8))                            # (32, 2) 的 <Z_q>
out.sum().backward()                                        # 原生 autograd
```

选型：任意 Python 构线路 / 任意观测量 / 需要 PSR 语义 → `QLayer`；
固定模板 + 大 batch 训练（尤其 NPU）→ `BatchLayer`。

## 19. 端到端量子分类器 `build_classifier`（`aicir.qml.classifier`）

`build_classifier` 把「角度编码 → 硬件高效纠缠层 → 逐比特 `<Z_q>` 读出 →
线性头」组合成一个标准 `torch.nn.Module`，直接用 torch 优化器/损失训练。量子
部分走 `BatchLayer`（整批一次演化、实/虚分离、NPU 安全、原生 autograd），
故大 batch 训练在 NPU/GPU 上高效。

```python
build_classifier(*, n_features, n_classes, backend, n_qubits=None, layers=2, seed=None)
```

- 返回 `nn.Module`：`forward(x (batch, n_features)) -> logits (batch, n_classes)`。
- `n_qubits` 缺省 = `n_features`；模板前 `n_features` 个参数为逐样本数据编码，
  其余为 `layers` 层 `ry` + `rzz`/`cx` 环形纠缠的权重（符合 `BatchLayer` 门集）。
- `backend` 须为 torch 系（`GPUBackend`/`NPUBackend`）。

```python
import torch
from aicir.backends.gpu_backend import GPUBackend
from aicir.qml import build_classifier

model = build_classifier(n_features=2, n_classes=2, backend=GPUBackend(),
                         n_qubits=2, layers=2, seed=0)
opt = torch.optim.Adam(model.parameters(), lr=0.1)
lossfn = torch.nn.CrossEntropyLoss()
for _ in range(100):
    opt.zero_grad()
    lossfn(model(X), y).backward()   # X:(N,2) float, y:(N,) long
    opt.step()
```

可运行示例（XOR 四象限 / sklearn moons，含 `--device npu`）：
`python -m demos.QNN.qnn_classifier_demo`。准确率由 `tests/qml/test_classifier.py`
钉住（XOR 训练/测试 >0.85/>0.80）。

## 20. 量子核 `QuantumKernel`（`aicir.qml.kernel`）

量子核 `K(x, z) = |<Φ(x)|Φ(z)>|²`。`QuantumKernel` 用 `BatchSV` **一次**批量演化
全部 N 个特征态（O(N) 次演化），再以实/虚分离矩阵乘算 gram（全实数 matmul，NPU
安全），替代逐对重演化的 O(N²) 路径。核矩阵可直接喂 sklearn 预计算核 SVM。

```python
from aicir.backends.gpu_backend import GPUBackend
from aicir.qml import QuantumKernel, angle_feature_map

kernel = QuantumKernel(angle_feature_map(n_qubits=3), backend=GPUBackend())
K = kernel.matrix(X)            # (N, N) numpy，X:(N, 3)
Kxz = kernel.matrix(X, Z)       # (N, M) 交叉核
k = kernel(x, z)                # 单对

# 喂经典 SVM（预计算核）
from sklearn.svm import SVC
SVC(kernel="precomputed").fit(K, y)
```

- `feature_map` 为含符号 `Parameter` 的模板，**全部**参数数据驱动（首用序对应输入
  列），参数门限 `rx/ry/rz/crx/cry/crz/rzz/rxx`。
- 非线性特征映射（如 IQP 的 `(π-xᵢ)(π-xⱼ)`）由调用方预先算入输入角度矩阵。
- 对比 `aicir.encoder.IQPEncoder.kernel_matrix`（逐对 O(N²) 演化）：`QuantumKernel`
  O(N) 演化，大 N 与 NPU 上优势显著。
