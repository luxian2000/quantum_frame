# aicir vs WuYueSDK — 量子模拟框架功能对比

本文基于对两套代码库的通读，对 **aicir**（本仓库 `quantum_frame`）与 **WuYueSDK / 五岳**（`/Users/luxian/GitSpace/WuYueSDK`）两款量子电路模拟与量子算法框架做全面的功能梳理与横向对比。

> 阅读方式：先看第 1、2 节了解整体定位与规模，再看第 3 节的能力矩阵速览，后续各节展开细节。
>
> **更新（2026-07-02）**：aicir 新增 `aicir.simulator`（张量网络/单振幅/部分振幅模拟，可微、支持 NPU）与 `aicir.qml.QLayer`（`nn.Module` 封装），本文相应章节（1、3、4、10、19、20、21）已同步刷新。
>
> **更新（2026-07-03）**：aicir 张量网络引擎接入 cotengra（`tn` extra）：`optimize=`/`memory_limit=` 支持 HyperOptimizer 路径与切片执行（执行仍走 backend 原语，NPU/可微性保留）——「cotengra 级路径优化仅 WuYue」的差距条目已消除。

---

## 1. 定位与总体印象

| 维度 | aicir | WuYueSDK |
| --- | --- | --- |
| 出品 | 个人/研究向 from-scratch 框架 | 中移（苏州）软件技术有限公司，MulanPSL-2.0 |
| 核心组成 | 单一包 `aicir`（分层子包） | 双模块：`wuyue`（电路/模拟）+ `qutorch`（量子 AI，仿 PennyLane） |
| 数值底座 | 后端抽象（NumPy / Torch-GPU / Ascend-NPU），自研 | NumPy + `autoray`/`opt_einsum`/`cotengra` + Torch(qutorch) |
| 主打差异 | 后端可插拔（含 **Ascend NPU**）、QAS、丰富 metrics/transpile、多算法编排、**NPU 上可微分的张量网络/单/部分振幅模拟（cotengra 路径+切片）** | 全/密度矩阵/部分/单振幅/张量网络多模拟方法、**经典控制流**、**真实量子云 QPU** |
| 文档语言 | 中文为主 | 中文为主 |

一句话总结：

- **aicir** 是一个"宽而自洽"的研究型框架 —— 后端抽象干净、变分算法/架构搜索/度量/编译/编码器等子系统齐备，且能落到 **昇腾 NPU** 硬件加速。
- **WuYueSDK** 是一个"深在模拟引擎与工程化"的工业级 SDK —— 模拟方法多样（尤其张量网络与单/部分振幅）、支持经典寄存器与 if/while 控制流、并接入**真实量子计算云**，`qutorch` 提供成熟的 PyTorch 自动微分栈。

---

## 2. 规模与工程化

| 指标 | aicir | WuYueSDK |
| --- | --- | --- |
| Python 源文件数 | ~181 | ~111 |
| 代码行数（不含测试） | ~40,100 | ~27,800 |
| 测试文件数 | 128（`tests/`，`pytest`） | 4（`test/`，`pytest`） |
| 打包 | `pyproject.toml`（setuptools，`pip install -e .[all]`） | `setup.py`（`bdist_wheel`） |
| 必需依赖 | 仅 `numpy`（torch/matplotlib/scipy 全部可选，`importorskip` 跳过） | numpy/sympy/tensornetwork/cma/opt_einsum/networkx/scikit-learn/qiskit/pennylane/cirq… 依赖较重 |
| CHANGELOG | 有，持续维护 | README 记录 |

**观察**：aicir 的测试覆盖面（128 个测试文件）与"零硬依赖"设计明显更克制、更利于集成；WuYueSDK 依赖较重（直接绑定 qiskit==1.4.3 / pennylane==0.43.0 / cirq==1.6.1），换来的是开箱即用的第三方互操作与云端能力。

---

## 3. 能力矩阵速览

| 能力 | aicir | WuYueSDK |
| --- | --- | --- |
| 态矢量模拟 | ✅ `StateVector` | ✅ 全振幅 `Sim_full` |
| 密度矩阵/含噪模拟 | ✅ `DensityMatrix` | ✅ `Sim_dm` |
| 张量网络模拟 | ✅ `aicir.simulator.tn_statevector`（cotengra/opt_einsum 路径 + 切片，`tn` extra；NPU 上可微） | ✅ `Sim_tc`（TensorNetwork + cotengra 缩并） |
| 单振幅模拟 | ✅ `aicir.simulator.single_amplitude` | ✅ `Sim_single` |
| 部分振幅模拟 | ✅ `aicir.simulator.partial_amplitude` | ✅ `Sim_partial` |
| GPU 后端 | ✅ Torch/CUDA | ⚠️ 经由 qutorch(torch) 间接 |
| **昇腾 NPU 后端** | ✅ `NPUBackend`（自定义 autograd 算子；张量网络缩并/切片/期望值均可在 NPU 上求导） | ❌ |
| 批量模拟 (batch) | ✅ `core/batch.py` | ✅ `set_batch` |
| 经典寄存器 | ⚠️ 以 measure 标记为主 | ✅ `ClassicalRegister` / `Bit` |
| 经典控制流 if/while | ❌ | ✅ `qif/trueif/falseif/qwhile`（`QuantumProg`） |
| 噪声通道 | 去极化、比特/相位翻转、幅值阻尼、离子阱噪声 | 更丰富：含 `PhaseDamping`/`GAD`/`Reset`/`PauliChannel`/双比特去极化去相位等 Kraus 通道 |
| VQE/QAOA/VQD/SSVQE | ✅ 全部（`aicir/vqc`） | ⚠️ VQE/QAOA（qutorch 示例/组件），无 VQD/SSVQE |
| 量子架构搜索 QAS | ✅ supernet/CRLQAS/PPR_DQL/PPO_RB/MOG-VQE | ❌ |
| QML 梯度方法 | psr/spsr/multipsr/psr4/fd/spsa/qng/ad | 参数移位 + 有限差分 + backprop(autograd) + SPSA |
| PennyLane 风格装饰器 | ✅ `@qfun` + `expval/probs/sample`（`aicir.qml.qfun`） | ✅ `@qfunc` 装饰器 |
| torch `nn.Module` 层封装 | ✅ `QLayer`（`aicir.qml.qlayer`，桥接 `QFun` 到 torch autograd） | ✅ `QuantumLayer(nn.Module)` 可嵌入 PyTorch |
| 量子化学 | ✅ 固定预置 + `build_molecule` 现算流水线（`chem` extra）+ `ansatze.uccsd`（fSWAP 精确激发电路） | ✅ UCCSD、单/双激发、HF 态、excitations 生成 |
| QUBO/组合优化 | ✅ `optimization/qubo` + QAOA 适配 | ⚠️ 以 QAOA 示例为主 |
| 编译/线路重写 transpile | ✅ pass-manager 管线 | ✅ `example/qda`（Clifford+Rz 优化、DAG、模板） |
| 编码器 encoder | ✅ Amplitude/Angle/Basis | ✅ amplitude/angle/basis/**iqp** |
| 度量 metrics（表达力/可训练性） | ✅ 专门子系统（含含噪版本） | ❌ |
| 算法示例库 | demos 为主 | ✅ QFT/Grover/Shor/QPE/QAA 内置 |
| OpenQASM I/O | ✅ QASM 2.0 + 3.0 + JSON 往返 | ⚠️ 仅 QASM2 导入 |
| 第三方互操作 | ✅ Qiskit / PennyLane / **WuYue** IO 模块 | ✅ 云端 runner 内用 qiskit/pennylane/cirq 做 QASM 转换 |
| 真实量子硬件云 | ❌（NPU 是 AI 加速器，非 QPU） | ✅ `WuyueQCloudDevice` / `plugin/ecloud` 提交任务到真实 QPU |
| 可视化 | ✅ 电路/态/密度矩阵/QAS 绘图 | ✅ 电路 text/matplotlib、Bloch 球 |

> ✅ 完整支持 ⚠️ 部分/间接支持 ❌ 未见

---

## 4. 模拟引擎（最大差异点）

### WuYueSDK — 多模拟方法
`wuyue/simulator` + `wuyue/backend` 提供 **5 种** 后端，各有适用场景：

- `Sim_full` / `AmplitudeBackend` — 全振幅态矢量，通用但内存随比特指数增长。
- `Sim_dm` / `DensityMatrixBackend` — 密度矩阵，支持 `apply_noise(noisemodel)` 含噪演化。
- `Sim_tc` / `TensorCircuitBackend` — **张量网络**，基于 `tensornetwork`/`opt_einsum`/`cotengra` 做缩并路径优化，可跑更大但纠缠受限的线路。
- `Sim_single` / `SingleAmpBackend` — **单振幅**，只算某个基态的振幅。
- `Sim_partial` / `PartialAmpBackend` — **部分振幅**，只算感兴趣的振幅子集。

这套"按需选择模拟策略"的设计曾是 WuYue 的核心竞争力。

### aicir — 后端抽象 + NPU + 张量网络（cotengra 路径与切片）
aicir 把"数值框架"抽象成可插拔 `Backend`（`NumpyBackend`/`GPUBackend`/`NPUBackend`，后者用自定义 `torch.autograd.Function` 算子规避 Ascend 复数梯度限制），并在此基础上提供 `aicir.simulator` 子包：

- `tn_statevector` — 精确张量网络全振幅模拟（`Circuit` → 标注张量网络 → 逐对缩并）。
- `single_amplitude` / `partial_amplitude` — 单/部分振幅查询，无需展开完整态矢量。
- `tn_expectation` — 末态期望值，Torch/NPU 后端上**保留自动微分图**（`.backward()` 与参数移位规则数值一致）。
- 收缩路径与切片：`optimize="auto"|"cotengra"|"opt_einsum"|"greedy"` 选路径来源（auto 下大网络自动用 cotengra HyperOptimizer）；`memory_limit=` 设定中间张量内存预算后由 cotengra 规划**切片执行**（含输出指标切片的 one-hot 精确重建），执行仍走 backend 原语，可微性保留（`tn` extra）。
- `Measure.run(circuit, method="tensor")` 把该引擎接入既有 shots/observable/measure 管线（仅纯态、无噪声）。

**结论**：两者都具备张量网络/单/部分振幅模拟且都可用 cotengra 规划路径。aicir 的独有差异：切片执行仍走后端抽象，因此**在昇腾 NPU 上可跑且保持可微**，可直接喂给 VQE/QAOA 等变分训练循环；WuYue 的独有差异在真实 QPU 云与经典控制流。

---

## 5. 电路构建模型

| 方面 | aicir | WuYueSDK |
| --- | --- | --- |
| 核心类 | `Circuit`（dataclass，内部存 **dict**）；`Operation` 类型化 IR | `QuantumCircuit`（gate 对象）；`QuantumProg`（支持控制流） |
| 门表示 | **纯 dict**（工厂函数返回），`gate_to_matrix` 消费 | **门对象**（`Basicgate` 及子类，如 `RX/CU3/RXX`） |
| 寄存器 | 隐式按 qubit 索引 | 显式 `QuantumRegister`/`ClassicalRegister`/`Bit` |
| 参数 | `Parameter` 符号占位 + `bind_parameters` | `Parameters`/`ParameterVector` + `bind_parameters`/`use_parameter` |
| 控制流 | 无（靠 measure 标记） | **`qif/trueif/falseif` + `qwhile`**（依赖测量结果的经典条件） |
| 逆电路 | 有 | `dagger()` |
| 批量 | `core/batch.py` | `set_batch/get_batch_size` |
| 深度/矩阵 | `unitary()` | `depth()`/`get_matrix()` |

WuYue 更接近 Qiskit 的"寄存器 + 门对象 + 经典控制流"范式，工程完整度高（`QuantumProg` 提供真正的测量条件跳转）；aicir 走"dict 门 + 类型化 IR 迁移中"的轻量路线，正在向 `Operation`/`GateSpec` 架构演进（见 `NEXT.md`）。

---

## 6. 门库

- **WuYue**（`element/gate.py`）：I/X/Y/Z/H/S/T/SY、RX/RY/RZ/P、U1/U2/U3/U、CU1/CU2/CU3、CNOT/CX/CY/CZ/DCX、SWAP/CSWAP/ISWAP/ECR、Toffoli/CCX、RXX/RYY/RZZ/IsingZZ、R、以及 `usergate(name, matrix)` 自定义门。门是对象，支持 dagger/control。
- **aicir**（`core/gates.py`）：`pauli_x/cnot/rx/rzz/rxx(ms_gate)/toffoli/u3…` 工厂返回 dict，受控门用 `(target, control_qubits_list)`，支持 `control_states`（0/1，含 0 控制）与多控制；另有**粒子数守恒激发门**（`single_excitation`/Givens、`double_excitation`），面向量子化学。

两者门集覆盖度相当。aicir 的激发门与"控制态 0/1"较有特色；WuYue 的 ECR/DCX/SY 等偏硬件原生门与 `usergate` 较有特色。

---

## 7. 测量与可观测量

- **aicir**（`aicir/measure`）：两套互斥机制（`Measure.run(measure_qubits=...)` 显式指定 vs 电路内嵌 `measure()` 标记自动识别）；`estimator` 做基于 shot 的 Pauli 项能量估计（量子比特级对易分组 QWC、基变换测量、shot 分配）；另有 `sampler/projector/trajectory/aggregate`。新 `primitives` 子包提供 Qiskit 风格 `Sampler`/`Estimator`（Statevector/Shot/Noisy 变体 + BackendSampler/Estimator）。
- **WuYue**：`backend.run(shots)` 返回计数；`qutorch/measure` 与 `qutorch/observable`（`Observable`/`PauliX/Y/Z` + `TensorProduct` `@` 组合）提供期望值；云设备 `WuyueQCloudDevice` 实现 `count/prob/expectation`，内含 **QWC 分组** 与 counts→期望换算。

两者都实现了 QWC 分组的 Pauli 期望估计。aicir 的 primitives/estimator 抽象更系统化。

---

## 8. 噪声模拟

- **WuYue**（`element/noise.py`）通道更丰富：BitFlip、PhaseFlip、PauliChannel、Depolarizing、AmplitudeDamping、**GeneralizedAmplitudeDamping**、**PhaseDamping**、**ResetNoise**、**TwoQubitDepolarizing**、**TwoQubitDephasing**，均为 Kraus 算子形式，配合 `circuit/noise_model.py` 与密度矩阵后端。
- **aicir**（`aicir/noise`）：去极化、比特/相位翻转、幅值阻尼，外加**离子阱噪声**（`ion_trap.py` + 参数文档）与噪声 `analysis/metrics`，全部走密度矩阵路径。

WuYue 的通道种类更全（尤其双比特通道、相位阻尼、reset）；aicir 特色在**离子阱专用噪声建模**与噪声度量分析。

---

## 9. 变分算法（VQA）

- **aicir**（`aicir/vqc`）：`BasicVQE`/`run_vqe` 编排，外加 **QAOA、VQD、SSVQE**，内置 ansatz 模板（HEA、离子阱 `hea_ti`）。所有参数移位统一走 `qml.deriv.psr`（单一真源）。
- **WuYue**：VQE/QAOA 以 `qutorch` 组件 + demo 形式提供（`vqe_h2_*`、`maxcut_problem`、`qaoa/` 有 cost/layer/时间演化），未见 VQD/SSVQE 的成套实现。

aicir 的变分算法家族更完整（含 VQD/SSVQE 激发态方法）。

---

## 10. QML 与自动微分

- **WuYue `qutorch`** 是仿 **PennyLane** 的成熟栈：`@qfunc(device, diff_method="backprop")` 装饰器、`QuantumLayer(torch.nn.Module)` 可嵌入 PyTorch 训练、`device` 抽象（statevector / 云）、`difference`（parameter_shift / finite_difference）、`encoding`（amplitude/angle/basis/iqp）、`layer`（单/双激发）、`optimize`（SPSA）、`workflow`（context/dispatch）。对 QML/混合模型开发者最友好。
- **aicir `qml`**：同样提供 **PennyLane 风格装饰器** `@qfun(device=..., differential="psr", observable=H)`（`aicir/qml/qfun.py`），返回 `QFun`（`__call__` 求值、`.grad` 求梯度），配 `expval`/`probs`/`sample` 测量对象；底层 `deriv.py` 汇集 psr/spsr/multipsr(mpsr)/psr4/fd/auto/ad/spsa/qng 等后端无关梯度，配合 GPUBackend/NPUBackend 直接用 Torch autograd。`QLayer`（`aicir/qml/qlayer.py`）把 `QFun` 桥接进 torch autograd：前向调 `qfun(params)`，反向调 `qfun.grad(params)`（参数移位 Jacobian），经典输入与权重经 `torch.cat` 拼接（可微，梯度回流到前置经典层），可一行嵌入 `torch.nn.Sequential`。

**结论**：两者都有 PennyLane 式装饰器，也都能把量子层塞进 PyTorch `nn.Module` 混合模型（aicir `QLayer` / WuYue `QuantumLayer`）。差别在梯度实现路径：WuYue `QuantumLayer` 走 `qutorch` 自身的 backprop/parameter-shift 栈；aicir `QLayer` 复用 `qml.deriv` 单一梯度真源（含 psr/spsa/qng 等），并可直接用于自定义后端（含 NPU）。

---

## 11. 量子化学

- **WuYue `qutorch/qchem`**：`uccsd`、`fermionic_single/double_excitation`、`hf_state`、`excitations`/`excitations_to_qubits`、`hamiltonian_transforms` —— 具备构造 UCCSD ansatz 与激发算符的能力。
- **aicir `chemistry` + `ansatze`**：固定预置 qubit Hamiltonian（h2/lih/h2o/nh3/n2/beh2）之外，`build_molecule`（`chem` extra，Qiskit Nature/PySCF）现算任意分子 Hamiltonian，JW 映射下附带 HF 占据/激发元数据；`aicir.ansatze.uccsd` 吃纯数据构造 UCCSD（非相邻激发用 fSWAP 网络精确实现，酉矩阵级对照 Qiskit UCC 验证），H2 端到端 VQE 达精确基态 ~1e-7 Ha。

两者量子化学能力相当；aicir 的激发电路与化学解耦（可脱离 qiskit-nature 独立使用），WuYue 与 qutorch 栈耦合更深。

---

## 12. 量子架构搜索（QAS）

- **aicir `qas`**：统一入口 `run(method, **kwargs)` + `config.<method>(...)`，方法覆盖 `supernet`（权重共享）、`CRLQAS`、`PPR_DQL`、`PPO_RB`（RL 系）、以及 `MOG-VQE`（NSGA-II 多目标）。这是 aicir 的独有强项（依赖 torch）。
- **WuYue**：未见架构搜索子系统。

---

## 13. 编译 / 线路优化

- **aicir `transpile`**：pass-manager 管线（`optimize`/`optimize_basic`/`optimize_circuit`），从旧 `optimizer/circuit.py` 整合而来；`optimizer` 现仅保留经典参数优化器（Adam/GD/SPSA/COBYLA/LBFGSB/ScipyMinimize）。
- **WuYue `example/qda`**：`clifford_rz_optimization`、`dag`、`template` —— 面向 Clifford+Rz 的线路优化与 DAG/模板重写。

两者都有线路级优化，路线不同：aicir 走通用 pass 管线，WuYue 走 Clifford+Rz/DAG 专项优化。

---

## 14. 组合优化 / QUBO

- **aicir `optimization`**：`qubo/` 提供 QUBO 建模 + Ising 映射 + QAOA 适配器（`qubo.qaoa`）；`sb/` 为 sample/subspace 占位。与经典优化器 `aicir/optimizer` 明确区分。
- **WuYue**：以 QAOA + maxcut/portfolio demo 体现，未见独立 QUBO 建模层。

---

## 15. 互操作与 I/O

- **aicir `core/io`**：OpenQASM **2.0 + 3.0** 与 JSON 双向往返；`qiskit_io`/`pennylane_io`/**`wuyue_io`** 互操作模块（可与本文对比对象 WuYue 互转）；多控制 `crx/cry/crz` 在 QASM3 下自动分解。
- **WuYue**：`circuit/qasm2.py` 仅 **QASM2 导入**；但云端 `plugin/runner.py` 内置将电路转 QASM 并用 qiskit/pennylane/cirq 处理 U3 分解、pi 表达式替换等，工程化地对接第三方与真机。

aicir 的 QASM/JSON 往返能力更完整（含 3.0 与显式跨框架 IO 模块）；WuYue 的互操作偏"为上真机服务"。

---

## 16. 真实量子硬件 / 云

- **WuYue** 独有：`plugin/ecloud`（auth/client/config/transport/retry）、`plugin/model`（submit/query 任务体）、`WuyueClient.submit_task/query_task_info`、`Runner.run`、以及 `qutorch/device/wyqcloud_device.py` —— 可把线路提交到**真实量子计算云 QPU** 执行并取回结果。
- **aicir**：没有 QPU 云。其 `NPUBackend` 面向**昇腾 AI 加速器**（经典硬件加速模拟/训练），`devices/target.py` 是硬件能力/Target 占位（迁移中）。

要跑真机 → 只有 WuYue 具备。要在国产 AI 芯片上加速模拟 → 只有 aicir 具备。

---

## 17. 算法示例库

- **WuYue `example/algorithm.py`** 内置标准算法：QFT / QFT_dagger、Grover、Shor、QPE(`_qpe_amod15`)、量子振幅放大 QAA。开箱即用的教学/演示价值高。
- **aicir**：`demos/` 以 VQE/QNN 梯度/NPU 验证等为主；`universal/qft.py` 提供 QFT 原语，但 Grover/Shor 等经典算法未成套内置。

---

## 18. 可视化

- **aicir `visual`**：电路图、态、密度矩阵、QAS 结果绘图（`Circuit.plot()/show()`，需 matplotlib）。
- **WuYue `visualization`**：电路 text/matplotlib 绘制、**Bloch 球**（`bloch.py`）、`plot_visual`。

WuYue 多了 Bloch 球可视化；aicir 多了 QAS/密度矩阵专项可视化。

---

## 19. 各自独有能力小结

**仅 aicir 具备：**
- 昇腾 **NPU 后端**（自定义 autograd 算子）+ 干净的后端抽象层
- **NPU 上可微分的张量网络模拟**（含 cotengra 切片执行下保持可微，可直接接入变分训练循环）
- **量子架构搜索**（supernet/CRLQAS/RL/MOG-VQE）
- **表达力/可训练性度量** 子系统（含含噪版本）
- **VQD / SSVQE** 激发态变分算法
- OpenQASM **3.0** 往返 + 显式 Qiskit/PennyLane/**WuYue** IO
- 独立 **QUBO 建模** 层、离子阱噪声建模、128+ 个测试文件的覆盖

**仅 WuYueSDK 具备：**
- **经典控制流**（qif/qwhile）+ 经典寄存器
- **真实量子云 QPU** 接入（ecloud/Runner/WuyueQCloudDevice）
- 更丰富的 Kraus 噪声通道、内置 Grover/Shor/QPE、Bloch 球可视化

**两者都具备（aicir 于 2026-07-02/03 补齐）：**
- 张量网络 / 单振幅 / 部分振幅模拟（`aicir.simulator` vs `Sim_tc`/`Sim_single`/`Sim_partial`），且都用 cotengra 做路径优化（aicir 另支持切片下保持可微/NPU）
- `nn.Module` 量子层封装（aicir `QLayer` vs WuYue `QuantumLayer`）
- UCCSD / 量子化学激发算符（aicir `aicir.ansatze.uccsd` + `build_molecule` 现算流水线 vs WuYue `qutorch/qchem`）

---

## 20. 选型建议

| 你的场景 | 推荐 |
| --- | --- |
| 需要在昇腾 NPU / GPU 上做可微分变分训练 | **aicir** |
| 需要 NPU 加速/可微的张量网络、单/部分振幅模拟（含切片突破内存墙） | **aicir** |
| 需要量子架构搜索 / 表达力可训练性度量 | **aicir** |
| 需要 VQD/SSVQE 激发态、QASM3 往返、跨框架转换 | **aicir** |
| 想用最小依赖（仅 numpy）嵌入到其他项目 | **aicir** |
| 需要模拟大而低纠缠线路 / 只求单个振幅 | 均可（两者都用 cotengra 规划；aicir 切片下仍可微/上 NPU） |
| 需要经典控制流（测量条件跳转、while 循环） | **WuYueSDK** |
| 需要把线路跑到真实量子计算机 | **WuYueSDK** |
| 需要 UCCSD 等量子化学激发算符 | 均可（aicir `ansatze.uccsd`+`build_molecule` / WuYue `qutorch/qchem`） |
| 习惯 PennyLane 装饰器风格（两者都有 `qfun`/`qfunc`） | 均可 |
| 要把量子层作为 `nn.Module` 嵌入 PyTorch 混合网络 | 均可（aicir `QLayer` / WuYue `QuantumLayer`） |
| 需要开箱即用的 Grover/Shor/QPE 教学示例 | **WuYueSDK** |

---

## 21. 结语

两套框架并非简单的"谁更强"，而是**侧重点互补**：

- **aicir** 把赌注押在 *可插拔后端（含国产 NPU）+ 研究型算法广度（QAS/度量/多变分算法）+ 轻依赖与高测试覆盖*，是做量子算法研究与国产硬件加速的合适底座；张量网络引擎（cotengra 路径+切片）与 UCCSD/化学流水线补齐了原本的模拟方法与化学短板，且独有 NPU 上的可微分张量网络能力。
- **WuYueSDK** 把赌注押在 *完整电路语义（寄存器/控制流）+ 真实 QPU 云 + 工程化 QML 栈*，是做工程落地与真机实验的合适底座。

值得注意的是，aicir 已内置 `wuyue_io` 互操作模块，两者电路可互转 —— 在实践中完全可以"用 WuYue 上真机、用 aicir 做 NPU 加速训练与架构搜索"组合使用。
