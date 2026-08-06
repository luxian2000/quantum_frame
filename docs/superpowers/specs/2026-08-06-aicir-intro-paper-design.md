# aicir 介绍论文（arXiv whitepaper）设计

日期：2026-08-06
状态：设计已确认，待评审后进入实施计划

## 1. 目标与约束

| 项 | 决定 |
| --- | --- |
| 形态 | arXiv 软件白皮书（对标 TensorCircuit 2205.10091、MindQuantum 2406.17248） |
| 篇幅 | 正文约 26 页 + 4 个附录 |
| 语言 | 英文（arXiv 惯例；仓库文档为中文，论文附录 A 兼作项目首个英文入口） |
| 论点 | **三支柱**：可移植性、横向扩展、架构搜索 |
| 性能证据 | **实跑跨框架基准**（不复用旧数据） |
| 对照策略 | CPU 同精度对齐 + 昇腾独占 |
| QEC | **不进 v1**，移至 §10 路线图（M2a 正在开发中，会让 §7 描述立即过期） |

### 1.1 三支柱如何构成**一个**论证

支柱不是并列的三个卖点，而是一条因果链——这是本文与"功能巡览式"白皮书的根本区别：

> 窄后端抽象买来了非 CUDA 加速器的可移植性 → 同一套实/虚分解纪律使**分片态**仍可微 → 架构搜索是唯一同时需要这两者的负载。

写作时每一节都要回指这条链，否则退化为特性清单。

### 1.2 明确的非目标

必须在 §1 写清，避免读者误判定位：不接真实 QPU；无脉冲级栈；不是电子结构求解器；`aicir.qrc` / `aicir.wireless` 是占位骨架（附录 D 明确标注）。

## 2. 正文骨架

| § | 标题 | 页 | 职责 |
| --- | --- | --- | --- |
| 1 | Introduction | 2.5 | 加速器单一生态问题；论点链；贡献；非目标 |
| 2 | Design philosophy | 3 | 命名原则（§4 本文档）。前置，使三支柱读作推论而非并列 |
| 3 | Programming model | 3 | 类型化 `Operation` IR、`GateSpec` 注册表、`Parameter`、经典控制流、双测量机制、primitives |
| 4 | **支柱一：后端可移植性** | 4.5 | `Backend` 抽象 + 昇腾作为对抗性验证 |
| 5 | **支柱二：可微分单态分片** | 4.5 | `2^p` 行分片、paired-real autograd、检查点、被拒绝的组合 |
| 6 | **支柱三：架构搜索** | 4 | `SearchStrategy` 注册表、10 方法分类、多卡 safe/aggressive 与数值等价性论证 |
| 7 | Remainder of the stack | 2.5 | TN/MPS/单/部分振幅、VQA 家族、qml diff 注册表、chemistry、noise、metrics、transpile+`Target` |
| 8 | Evaluation | 4 | 基准协议 + 结果（§5 本文档） |
| 9 | Related work | 1.5 | 对照参考文献集定位 |
| 10 | Limitations & roadmap | 1 | 诚实清单：QEC、脉冲、占位子包、单节点上限 |
| 11 | Conclusion | 0.5 | |

### 2.1 附录（上限 4）

- **A — 安装与 API 速查**：项目首个英文入口。
- **B — 昇腾内核缺口目录**：每个缺失的 complex64 算子、失效表现与规避手法。对其他非 CUDA 移植者最具复用价值的产物。
- **C — 基准复现**：命令、版本、硬件、原始清单。
- **D — 子系统清单**：逐子包的 LOC / 测试数 / 成熟度，占位者显式标注。

## 3. 核心特性与突出策略

### 3.1 突出策略总则

**每个支柱以"该设计避免了什么失效"开场，而非以特性列表开场。** 三条具体手法：

1. **对抗性硬件叙事**（支柱一）：昇腾缺 complex64 matmul/add/mul、复数 SVD/QR、复数归约，`aclnnComplex` 限 8 维。把这些当作**抽象层的压力测试**来写——抽象若不够窄，这些缺口会渗透到上层。附录 B 是证据。
2. **机器可核验的证据**（支柱二）：不写"我们实现了分布式自动微分"，而是展示清单字段：`release_gate=PASS`、`fallback_to_cpu=false`、`payload_dtypes=["torch.float32"]`（这一条即"跨卡绝不传复数"的机器证明）、`rank_devices`、HCCL、CANN 8.0.RC3。参考集中少有论文提供这种粒度。
3. **"需要前两者的负载"**（支柱三）：QAS 占 29.5k LOC / 41% 代码量，且无任何对照框架具备。以它作为前两支柱的**消费者**登场，链条闭合。

### 3.2 各支柱要点

**支柱一**：`Backend` 抽象方法清单；real/imag 分解 + 自定义 `autograd.Function` 模式；参数化门用 `cos + i·sin` 拼装使梯度累加落在实数角上；n>8 时改用扁平位置换 gather。

**支柱二**：`2^p` 行分片；一进程一卡、所有 rank 同序完成 collective；`PureStateParam`/`DensityParam`/`StinespringParam` 成对实数叶子（直接传复数 trainable leaf 被显式拒绝）；`grad_checkpoint`；**被拒绝的组合同样是设计**（shots/collapse 在可微路径上拒绝、非 2 幂 world size、中途测量/控制流在分配存储前即拒绝）。

**支柱三**：统一 `run(method, **kwargs)` + `QASResult`；10 方法分类；ranking/training/finetune 三阶段分片；`safe` 与单卡**数值完全等价**（可作为正确性论证），`aggressive` 换吞吐但轨迹不同。

## 4. 设计哲学（§2 的命名原则）

七条，均可在代码中找到对应证据：

1. **窄后端接口是唯一的可移植性缝**。上层（`StateVector`、`Circuit`、`Measure`、VQE、QAS）从不接触数值框架。新增后端只需实现抽象方法。
2. **精度是后端能力，不是全局常量**（2026-08-06 落地）。`NumpyBackend`→complex128（正确性优先，且与 Aer/Qulacs 对齐便于对比）、`GPUBackend`→complex64（torch 生态 float32-native）、`NPUBackend`→complex64 硬锁并对 complex128 显式报错。**默认值故意不同，不得"统一"**。门矩阵在最宽精度构造、由后端在边界窄化——反过来不可逆。
3. **硬件规避手法留在需要它的后端里**。规范例证：昇腾因 `aclnnComplex` 8 维上限必须走 gather，NumPy 无此限制，故 CPU 用跨步视图、NPU 保留 gather，两者以对方为正确性 oracle。抽象层的价值正在于允许它们不同。
4. **单一真源**。`GateSpec` 注册表（门元信息）、`qml.deriv.psr`（参数移位）、`_unitary_axes`（自定义门作用轴）、`aicir/dtypes.py`（精度）。本轮三个 bug 全部源于同一概念散落多处。
5. **结构性保证优于约定**。QEC 运行器逐轮交错模拟与解码，解码器**无法**看到未来轮次；基准运行器在 parity 未通过时**拒绝**产出计时数据。正确性由控制流保证，而非由使用者自律。
6. **证据优于断言**。探针脚本产出 commit 钉死的清单与发布门禁，而非日志。
7. **零硬依赖核心**。仅需 numpy；torch/matplotlib/scipy 全部可选，缺失时 `importorskip` 跳过。

## 5. 基准协议（§8）

### 5.1 有效性控制（本设计最需守住的部分）

1. **parity 先于计时**：各框架必须对同一规格产出同一态矢量。运行器在 parity 失败时拒绝输出计时。
2. **构建与执行分开计时**：引用参考集中的 *Benchmarking the performance of quantum computing software for circuit creation, manipulation and compilation*，预先回应"你们把 transpile 算进去了"的质疑。
3. **精度对齐**：同精度才可比。Qulacs 仅 double，只出现在双精度表并注明是构建限制。
4. **中位数 + IQR，预热剔除**；线程数与 BLAS 实现写入清单。
5. **门数对齐**：适配器的门分解差异会被误记为引擎差距（QFT 曾因此高估 1.9×）。

### 5.2 测量轴

| 轴 | 内容 | 参与者 |
| --- | --- | --- |
| A | 单门吞吐 vs n | 全部 CPU 框架 |
| B | 标准线路 QFT/random/GHZ vs n、depth | 全部 CPU 框架 |
| C | VQE H₂/LiH：单次能量、单次梯度、收敛墙钟 | 全部 CPU 框架 |
| D | 峰值内存 vs n | 全部 CPU 框架 |
| E | aicir NPU vs CPU | 仅 aicir |
| F | 强扩展 W=1/2/4/8；弱扩展（每卡数下可达的最大 n） | 仅 aicir |
| G | 能力矩阵：哪些框架**能跑**每一行 | 全部 |

A–D 建立可信度，E–G 承载论点。**A–D 输掉某一行可以接受并须如实报告；悄悄不报不可接受。**

### 5.3 当前状态（2026-08-06 实测）

harness 已落地于 `scripts/bench/`，aicir / Qiskit 2.4.2 / Cirq 1.7.0 三方 parity 通过。

经本轮两项优化（跨步快路径、`unitary` 修复）后，GHZ 与 QFT 上 aicir 已与 Qiskit 大致持平、在中等比特数下领先；**与 Cirq 在大 n 仍有真实差距**（GHZ n=18 约 3.7×）。

尚缺：`qiskit-aer`（真正的高性能 CPU 基线）未安装；轴 C–G 未实现。

## 6. 风险与开放问题

| 风险 | 处置 |
| --- | --- |
| CPU 性能不及 Cirq，削弱"先可信再独占"的框架 | 先补 `qiskit-aer` 与轴 C–G，再判断 CPU 平价是否真的承载论点；若否，把论述重心移到"可达工作负载"（轴 G） |
| 参考文献元数据有误 | `52-SDK/` 中 `Qulacs_1811.11920`、`NWQSim_2105.01025` 的正文与文件名不符，**每条引用须核对正文后再入参考列表** |
| 论文写作期代码漂移 | 五阶段冻结纪律（§7） |
| 附录膨胀为无人评审的倾倒场 | 上限 4 个，每个须支撑正文某项主张 |

## 7. 阶段纪律

| 阶段 | 工作 | 代码可变？ |
| --- | --- | --- |
| A | 写 §1、§2、§9、§10、§11（论证而非 API，约占正文 40%） | 自由 |
| B | 仅做论点所需的升级；建基准 harness | 这是窗口期 |
| C | 打 `paper-v1` 标签；对冻结 API 写 §3–§7 | 仅分支，不并入 |
| D | 在冻结 SHA 上跑基准，归档 `docs/evidence/benchmarks/<sha>/` | 冻结 |
| E | 写 §8，全文数字一致性复核 | 冻结 |

**arXiv 有版本机制**：v1 钉死一个 SHA，后续里程碑发 v2。因此"再加一个特性就发"的诱惑有原则性答案——那是 v2，§10 是它的等待区。

## 8. 待评审确认项

1. §7 是否需要包含 `aicir.distributed` 之外的多卡模式（QAS supernet 分片、fair-label 队列）？目前计划只在 §6 提及。
2. 轴 C（VQE 端到端）用哪个分子？建议 H₂（所有框架都能跑）+ LiH（区分度更高）。
3. 是否要在 §8 报告 aicir 的 complex64 vs complex128 自比数据？可展示精度/性能权衡，但会增加表格数量。
