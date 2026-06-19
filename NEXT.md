# aicir 架构改进建议

本文记录后续架构演进方向。目标不是把 `aicir` 变成 Qiskit 或 PennyLane 的复刻版，而是吸收两者的架构优点：借鉴 Qiskit 的编译、硬件目标和 primitives 分层，借鉴 PennyLane 的 QNode、设备绑定、自动微分接口和 transform 工作流，同时保留 `aicir` 当前轻量、直接、便于研究原型验证的特点。

## 参考架构

### Qiskit 的优点

- 编译层清晰：通过 transpiler 和 pass manager 把线路验证、门分解、布局、路由、优化、调度拆成可组合 pass。
- 硬件目标明确：用 backend/target 描述设备支持的门集、连接关系和执行约束，编译结果可以面向具体硬件。
- primitives 抽象稳定：`Sampler` 和 `Estimator` 把采样与期望值估计作为算法层的统一执行入口，VQE、QAOA 等算法无需直接处理底层执行细节。

### PennyLane 的优点

- QNode 抽象简洁：把量子函数、设备、测量返回和自动微分策略组合成一个可调用对象，适合量子机器学习。
- 自动微分接口成熟：同一量子程序可以接入 NumPy、Torch、JAX 等经典优化栈。
- transform 机制灵活：线路分解、合并、梯度变换、批处理等功能以可组合 transform 的方式叠加。
- 测量返回模型自然：`expval`、`probs`、`sample` 等返回类型直接表达算法意图。

## aicir 当前状态

`aicir` 已经具备较完整的研究型量子算法框架雏形：

- `aicir.core` 提供 `Circuit`、门构造器、参数绑定和 QASM/JSON 互转。
- `aicir.backends` 提供 NumPy、GPU、NPU 后端；`aicir.noise` 提供噪声模型；`aicir.core.operators` 提供 Pauli/Hamiltonian 算符（顶层 `aicir` 仍再导出 `Hamiltonian`/`PauliOp`/`PauliString`）。
- `aicir.measure` 提供测量、采样和 Pauli 项能量估计。
- `aicir.optimizer` 提供线路结构优化和经典参数优化器。
- `aicir.qml` 提供参数移位、有限差分、自动微分、伴随微分、量子自然梯度等方法。
- `aicir.vqc` 提供 VQE、QAOA、VQD、SSVQE 和 ansatz 模板。
- `aicir.qas` 提供量子架构搜索、搜索空间、奖励和评估工具。

主要短板是：核心线路表示仍以门字典列表为主，类型约束和元信息不足；编译优化还不是 pass pipeline；后端、噪声、测量、梯度、硬件约束之间缺少统一的执行抽象。

## 目标目录结构

后续架构调整建议采用“新增清晰架构层，旧路径保持兼容”的方式推进。目标结构如下：

```text
aicir/
|-- core/              # 保留：Circuit、Parameter、基础构造入口
|   |-- io/            # QASM/JSON 等基础序列化
|   `-- operators.py   # 已落地：PauliOp/PauliString/Hamiltonian（顶层 aicir 再导出）
|-- ir/                # 新增：Operation、Measurement、Observable、CircuitIR
|-- gates/             # 新增：GateSpec 注册表、门元信息、分解规则
|-- transpile/         # 新增：Pass、PassManager、线路编译与优化
|   `-- passes/        # 新增：验证、规范化、消去、合并、分解、布局、路由等 pass
|-- devices/           # 新增：Target、设备能力、连接拓扑、硬件约束
|-- backends/          # 已落地：从 channel/backends 上移到顶层
|-- noise/             # 已落地：从 channel/noise 上移到顶层
|-- primitives/        # 新增：Sampler、Estimator、统一执行结果
|-- qml/               # 保留：梯度方法；diff method 注册表；qfun（PennyLane 风格量子函数）
|-- optimizer/         # 保留：经典参数优化；线路优化逐步迁到 transpile
|-- measure/           # 保留兼容；底层逐步委托 primitives
|-- vqc/               # 保留：VQE/QAOA/VQD/SSVQE，逐步调用 Estimator/QNode
|-- qas/               # 保留：架构搜索，逐步接入 Target/GateSpec
|-- metrics/           # 保留：评分指标，逐步接入 Target/GateSpec
|-- interop/           # 新增：Qiskit、PennyLane、QASM3 互转
|-- visual/
|-- chemistry/
|-- encoder/
|-- optimization/
|-- universal/
`-- wireless/
```

第一批已落地的目录应聚焦底层抽象，不移动旧模块：

```text
aicir/ir/
aicir/gates/
aicir/transpile/
aicir/transpile/passes/
aicir/devices/
aicir/primitives/
```

## 改进方向

### 1. 建立正式的线路中间表示

当前 `Circuit` 的门字典表示灵活，但优化器、绘图、QASM、梯度和 metrics 都需要重复判断字段。建议新增 typed IR：

- `Operation`：门名、作用比特、参数、控制比特、控制态、标签和 metadata。
- `Measurement`：测量类型、测量比特、返回类型。
- `Observable`：Pauli string、Hamiltonian、dense matrix 等可观测量。
- `CircuitIR`：保存 operation 序列、qubit 数、classical bits、metadata。

兼容策略：继续允许用户传入门字典，内部先规范化为 typed IR。公开 API 可长期保留门字典入口，内部实现逐步迁移到 typed IR。

当前状态：`Operation`、`Measurement`、`Observable`、`CircuitIR` 已作为 `aicir.ir` 的 typed IR 落地；`aicir.ir` 还提供统一访问 helper，将 `CircuitIR`、`Circuit.operations` 和旧 `Circuit.gates` 规范化为 typed instruction 序列。`Circuit` 继续保留 `.gates` 门字典公开 surface 以兼容旧代码，同时提供 `.operations` 和 `.ir` typed IR 视图。JSON/QASM/DAG 导出、绘图、测量、Pauli 估计、transpile/optimizer、QML 伴随梯度、metrics、noise 和 QAS 的主要内部读取路径已迁移为优先消费 typed IR，并在需要旧格式的兼容层显式生成门字典视图。

### 2. 新增 `aicir.transpile` 编译层

建议把现有 `aicir.optimizer.circuit` 升级为可组合编译流水线。初始接口可以是：

```python
from aicir.transpile import PassManager

pm = PassManager([
    "validate",
    "canonicalize",
    "cancel_inverse",
    "merge_rotations",
    "commute_single_qubit",
])
optimized = pm.run(circuit)
```

建议拆分的 pass：

- `ValidatePass`：检查 qubit 范围、参数数量、控制比特和目标比特冲突。
- `CanonicalizePass`：统一门名、参数格式和控制态格式。
- `CancelInversePass`：消去自逆门和互逆门。
- `MergeRotationPass`：合并同轴旋转。
- `CommutationPass`：处理安全可交换的局部重排。
- `DecomposePass`：将高级门分解为目标门集。
- `LayoutPass`：选择 logical-to-physical qubit 映射。
- `RoutingPass`：插入 SWAP 或选择等价连接以满足硬件拓扑。

短期可以让 `optimize_circuit` 继续存在，但内部委托给默认 `PassManager`。

当前状态：`aicir.transpile` 提供 `TransformationPass`/`PassManager`/`default_optimization_pipeline`，本地优化 pass `CancelInversePass`/`MergeRotationsPass`/`CommuteSingleQubitPass`，结构 pass `ValidatePass`/`CanonicalizePass`，以及第二批面向硬件目标的 pass：`DecomposePass`（高级门分解到目标门集，内置 `swap→3×cx`、`cz→h·cx·h`、`cy→rz·cx·rz` 标准规则，仅单控制位，规则展开产生 `hadamard`/`rz`，暂不做任意单比特门的 Euler 基底翻译）、`LayoutPass`（显式/平凡 logical→physical 重标号，比特置换意义下等价）、`RoutingPass`（沿耦合图最短路径插 SWAP 并对称复位，整线路完全幺正等价，SWAP 数非最优）。三者消费 `aicir.devices.Target`（门集 + 耦合拓扑）；`PassManager` 字符串名支持 `decompose`/`layout`。尚未开始：基于置换跟踪的最优路由、自动择优布局、任意单比特门的目标基底翻译、`DecomposePass` 的 `GateSpec.decomposition` 字段驱动。

### 3. 引入硬件目标描述 `Target`

当前已有 backend、noise 和 metrics 中的硬件 profile，但还没有统一描述硬件能力的对象。建议新增：

```python
Target(
    n_qubits=5,
    basis_gates=("rx", "ry", "rz", "cx"),
    coupling_map=[(0, 1), (1, 2), (2, 3), (3, 4)],
    supports_shots=True,
    supports_statevector=True,
    supports_density_matrix=False,
    supports_autodiff=False,
)
```

`Target` 应成为以下模块的共同输入：

- `transpile`：决定分解、布局和路由。
- `measure`：判断是否支持 shots、批量采样、指定测量。
- `vqc`：选择 Estimator/Sampler 执行路径。
- `qas`：限制搜索空间和硬件效率评分。
- `metrics`：计算拓扑映射效率、native gate 效率和噪声敏感性。

当前状态：第一片已落地。`aicir.devices` 提供冻结数据类 `Target`，字段含 `n_qubits`/`basis_gates`/`coupling_map` 与四个执行能力标志；`basis_gates` 按 `GateSpec` 规范名归一，`coupling_map` 按无向图处理并在构造时校验比特范围。提供门集查询 `supports(gate)` 与拓扑查询 `coupled(a, b)`/`neighbors(q)`/`fully_connected`。`DecomposePass`/`LayoutPass`/`RoutingPass` 已作为首批消费方。尚未接入 `measure`/`vqc`/`qas`/`metrics`。

### 4. 建立 `Sampler` 和 `Estimator` primitives

建议把测量执行统一为 primitives：

```python
samples = sampler.run(circuits, shots=1024)
values = estimator.run(circuits, observables, parameter_values=params)
```

建议分层：

- `BaseSampler` / `BaseEstimator`：统一接口。
- `StatevectorSampler` / `StatevectorEstimator`：精确模拟。
- `ShotSampler` / `ShotEstimator`：有限 shots 统计。
- `NoisySampler` / `NoisyEstimator`：density matrix 或 noise model 路径。
- `BackendSampler` / `BackendEstimator`：面向真实硬件或远端服务的扩展点。

这样 VQE、QAOA、QAS、metrics 都可以依赖 primitives，而不是各自处理测量、Hamiltonian 和 counts。

当前状态：第一片已落地。`aicir.primitives` 提供 `BaseSampler`/`BaseEstimator` 接口、最小统一结果对象 `SampleResult`/`EstimateResult`（第 9 节切片），以及三个实现：`ShotSampler`（包装 `Measure`，支持显式 `measure_qubits` 与内嵌 measure 门）、`StatevectorEstimator`（精确态向量期望，拒绝 `shots=`）、`ShotEstimator`（包装 `PauliEstimator` 的分组/基变换/shot 分配，并暴露 `estimate()` 直通方法，可直接作 `BasicVQE(energy_estimator=...)` 注入）。约定：接收已绑定参数的电路；单入参返回单结果、序列返回列表；单个可观测量可广播。`Noisy*`/`Backend*` 变体、`parameter_values=` 延迟绑定与 `vqc`/`qas`/`metrics` 的切换尚未开始。

### 5. 增加 PennyLane 风格 `QNode`

建议新增轻量 QNode，用于统一“量子函数 + 设备 + 测量 + 梯度”：

```python
from aicir import Hamiltonian
from aicir.qnode import qnode, expval

H = Hamiltonian([("Z", 1.0)])

@qnode(device="numpy", diff_method="psr")
def cost(theta):
    ry(theta, 0)
    return expval(H)

value = cost(0.1)
grad = cost.grad(0.1)
```

这可以成为 `aicir.qml` 和 `aicir.vqc` 的公共底座。`BasicVQE`、`BasicQAOA` 后续可以接受 QNode 或内部构造 QNode，减少算法类中重复的参数绑定和测量逻辑。

当前状态：第一片已落地，命名为 `qfun`（`aicir/qml/qfun.py`，从 `aicir.qml` 导出 `qfun` 装饰器与 `QFun` 类）。统一"量子函数 + 设备 + 测量 + 梯度"为一个可调用对象：`@qfun(device=..., differential=..., observable=..., shots=None)` 包装一个**返回 `Circuit`** 的函数（不依赖全局 tape，规避门工厂队列化的侵入式改动与误捕获风险——见设计取舍）；调用得期望值 `cost(x)`，`cost.grad(x)` 得梯度。`device` 映射 `numpy`/`cpu`→`NumpyBackend`、`gpu`/`torch`→`GPUBackend`、`npu`→`NPUBackend`；`differential` 经 §6 的 `aicir.qml.diff` 注册表分发，`"auto"` 走 `select_diff(backend, shots, noisy)` 自动优选，其余经 `resolve_diff`（仅 `fn_gradient`）。观测量经 `observable.to_matrix(backend)`，测量走 `Measure.run(..., observables=..., shots=None)` 精确路径。支持单个可训练位置参数（标量或一维数组）。配套 `tests/qfun/test_qfun.py`（`<Z>=cosθ`、`grad=-sinθ` 解析校验、`auto` 选择、契约守卫）。

与 §5 原草图的有意差异：观测量声明在装饰器（`observable=`）而非函数体内 `return expval(H)`；函数体显式 `return Circuit`。因此暂未提供 `expval` 测量帮助器。尚未做：多参数/多测量、`BasicVQE`/`BasicQAOA` 接入 `qfun`、shot 估计与噪声路径的便捷封装。

### 6. 把梯度方法做成策略注册表

`aicir.qml` 已经包含多种梯度方法，但选择逻辑分散。建议新增 `DiffMethod` 注册表：

- `auto`：Torch/NPU 后端且无 shots 时优先。
- `ad`：无噪声态向量模拟且支持伴随微分时优先。
- `psr`：硬件、shots 或噪声线路的通用精确规则。
- `fd`：黑盒 fallback。
- `spsa` / `spsr`：大参数量或高噪声任务。
- `qng` / `bdqng` / `kqng` / `dqng`：作为优化器预条件策略。

QNode、Estimator 和参数优化器可以通过同一策略表选择梯度实现。

当前状态：第一、二片已落地。新增子包 `aicir.qml.diff`（`spec.py` 定义冻结数据类 `DiffMethod`，字段含 `name`/`fn`/`aliases`/`category`/`exact`/`stochastic`/`requires_torch`/`supports_shots`/`supports_noise`；`registry.py` 实现注册表与选择器），并从 `aicir.qml` 顶层再导出。注册表公开 API：`register_diff`/`unregister_diff`/`get_diff`/`registered_diffs(category=None)`/`canonical_diff`/`resolve_diff`；纯函数选择器 `select_diff(*, backend=None, shots=None, noisy=False)` 按 auto → psr → fd 优先级自动选择（`spsa`/`spsr` 不参与自动选择）。

第二片：注册表按 `category` 分三类索引**全部**内置微分方法——`fn_gradient`（`(fn, params) -> 梯度向量`：`psr`/`fd`/`auto`/`spsa`/`spsr`）、`circuit_gradient`（`(circuit, observable) -> 梯度`：`ad` 伴随微分）、`preconditioner`（`(fn, state_fn, params) -> 方向/度规`：`qng`/`bdqng`/`kqng`/`dqng`）。`category` 在 `DiffMethod.__post_init__` 校验。`registered_diffs(category=...)` 可按类别过滤检索。契约安全：`resolve_diff` 与 `select_diff` **只对 `fn_gradient` 生效**——`resolve_diff('ad'|'qng'|...)` 抛 `ValueError`，保证经典优化器分发不会拿到签名不兼容的可调用；`ad`/`qng` 族仅供 `get_diff`/`registered_diffs(category=...)` 检索发现。`mpsr`（返回标量混合偏导）仍**有意不纳入注册表**，作为 `qml.mpsr` 直接可用。capability 字段（`exact`/`stochastic`/`requires_torch`/`supports_*`）只为 `fn_gradient` 的 `select_diff` 优选服务；`ad` 与 `qng` 族均从态向量求值，标注 `supports_shots/noise=False`。

`aicir/optimizer/params.py` 的 `_gradient_from_method` 已改为经 `resolve_diff` 分发，使 `GD`/`Adam`/`ScipyMinimize` 可统一访问所有内置 fn-gradient 方法；对 `requires_torch=True` 的方法（即 `auto`）在经典黑盒目标路径上加守卫并抛出明确错误。`select_diff` 已有单元测试，并已由 §5 的 `qfun` 接入：`@qfun(..., differential="auto")` 的 `.grad` 经 `select_diff(backend, shots, noisy)` 自动选择梯度方法（首个真实调用方）。`aicir/qml/deriv.py` 未改动；`vqc`/`qas` 保持原有 `from ..qml.deriv import psr` 路径，参数移位单一实现不变。

### 7. 建立 GateSpec 注册表

目前新增一个门需要修改矩阵、QASM、绘图、梯度、metrics、QAS 等多处代码。建议引入 `GateSpec`：

```python
GateSpec(
    name="rxx",
    aliases=("ms_gate", "molmer_sorensen"),
    num_qubits=2,
    num_params=1,
    matrix=...,
    inverse=...,
    qasm_name="rxx",
    generator=...,
    decomposition=...,
)
```

收益：

- 新门注册一次，多模块复用。
- `transpile` 可根据 `GateSpec.decomposition` 做目标门集分解。
- `qml` 可根据 generator 自动判断能否用参数移位。
- `visual` 可从 spec 获取显示名称和参数格式。
- `qas` 可从 spec 生成搜索动作空间。

当前状态：第一、二片已落地。`aicir.gates` 提供 `GateSpec`（门名/别名/目标比特数/参数个数/是否受控/QASM 名/显示符号）与注册表 API（`register_gate`/`unregister_gate`/`get_gate_spec`/`registered_gate_names`/`canonical_gate_name`），内置门集已全部注册；`num_qubits`/`num_params` 为 `None` 表示可变（`unitary`、`measure`、`reset`、整寄存器 `identity`）。消费方已接入六处：`aicir.ir.Operation` 构造期按 spec 校验（未注册门名保持宽松）；`ValidatePass` 实质校验；`CanonicalizePass` 别名归一；QASM 导出名由 `GateSpec.qasm_name` 派生；矩阵路径（`gate_to_matrix`/`apply_gate_to_state` 等）入口经 `canonical_gate_name` 归一后按规范名分发；ASCII 与 matplotlib 绘图符号/配色族由 `GateSpec.symbol` 与规范名派生（注册自定义门可携带 symbol 直接显示）。`reset` 已接入线路标记、测量执行路径与绘图；matplotlib 使用与 `measure` 同色、标注 `Reset` 的虚线表示 reset，`Reset` 字号与 `Rz` 门主标签一致且虚线 dash 间距更大；虚线连接测量门与后续同一比特量子门，没有后续门时延伸到线路末端，且不叠加普通实线；遇到无完整方框的后续门时按虚拟方框左边界截断，虚拟方框内恢复普通实线，但 reset 尚未纳入 QASM 互转。尚未迁移：`matrix`/`generator`/`decomposition` 字段；`metrics`/`qas` 评分中的别名容忍集合（`DEFAULT_NATIVE_GATES`、双比特门判定等）属评分语义，留待单独处理。

### 8. 强化跨框架互操作

建议新增互操作层：

- `to_qiskit()` / `from_qiskit()`
- `to_pennylane()` / `from_pennylane()`
- `to_wuyue()` / `from_wuyue()`
- `to_qasm3()` / `from_qasm3()`

目的不是依赖外部框架，而是让用户可以把 `aicir` 的轻量研究线路带入 Qiskit/WuYue 硬件生态，或把 PennyLane/Qiskit/WuYue 的线路导入 `aicir` 做本地模拟、QAS 和自定义指标评估。

当前状态：Qiskit、PennyLane 与 WuYue 第一片已落地。Qiskit 路径位于 `aicir.core.io.qiskit_io`，提供 `circuit_to_qiskit`/`circuit_from_qiskit` 和短别名 `to_qiskit`/`from_qiskit`；PennyLane 路径位于 `aicir.core.io.pennylane_io`，提供 `circuit_to_pennylane`/`circuit_from_pennylane` 和短别名 `to_pennylane`/`from_pennylane`；WuYue 路径位于 `aicir.core.io.wuyue_io`，提供 `circuit_to_wuyue`/`circuit_from_wuyue` 和短别名 `to_wuyue`/`from_wuyue`。三者都从 `aicir.core.io`、`aicir.core`、顶层 `aicir` 导出，并保持对应框架为可选依赖，仅在调用互操作函数时导入。当前支持基础门、参数旋转、受控门、`swap`、`rzz/rxx`、`u2/u3`、`ccx` 等基础幺正门；Qiskit 路径额外支持线路内 `measure` 标记，PennyLane 路径暂不转换 aicir 线路内 `measure`，WuYue 路径按当前 SDK 原生门集支持 `cx/cz`、`rzz`、`swap`、`u2/u3`、`ccx`、`identity` 和线路内 `measure`，暂不转换 `cy`、`crx/cry/crz` 与 `rxx`。更完整的 QASM3 互转仍未开始。

### 9. 统一结果对象和 metadata

建议为执行结果建立统一数据模型：

- `SampleResult`：counts、probs、shots、measured_qubits、metadata。
- `EstimateResult`：values、variances、shots、term_results、metadata。
- `GradientResult`：gradient、method、nfev、metadata。
- `TranspileResult`：circuit、layout、passes、depth_before、depth_after、metadata。

统一结果对象会让 demos、tests、QAS、VQE 和报告生成更稳定。

## 推荐推进顺序

### 第一阶段：统一执行和优化基础

优先级最高，风险较低。

1. 新增 typed IR，保留门字典兼容入口。已落地：`aicir.ir.Operation` 支持与现有门字典互转，`Measurement` 支持测量声明与现有 `measure` 门字典互转，`Observable` 支持包装 Pauli string、Hamiltonian 和 dense matrix，`CircuitIR` 支持从现有 `Circuit` 构造并转回 `Circuit`；`Circuit` 构造、`append`、`extend` 已可接收 `Operation` 和 `Measurement`，并继续保存现有门字典 surface；`Circuit.operations` / `Circuit.ir` typed IR 视图和 `aicir.ir` 访问 helper 已接入 JSON/QASM/DAG、绘图、测量、Pauli 估计、transpile/optimizer、QML 伴随梯度、metrics、noise、QAS 等主要内部路径。
2. 新增 `Pass` / `PassManager`，把现有线路优化规则拆成 pass。已落地：`aicir.transpile` 提供 `TransformationPass`、`PassManager`、`default_optimization_pipeline`，并提供 `CancelInversePass`、`MergeRotationsPass`、`CommuteSingleQubitPass` 等第一批本地优化 pass；`optimize_circuit` 已委托给默认 pipeline。
3. 新增 `Sampler` / `Estimator` primitives，先包装现有 `Measure` 和 `PauliEstimator`。已落地：`aicir.primitives` 提供 `ShotSampler`/`StatevectorEstimator`/`ShotEstimator` 与统一结果对象（见第 4 节当前状态）。
4. 让 `BasicVQE` 和 `BasicQAOA` 优先调用 `Estimator`。部分可用：`BasicVQE(energy_estimator=ShotEstimator(...))` 已可直接注入（estimate 直通契约）；默认路径切换尚未开始。

### 第二阶段：硬件目标和门注册表

主要目标是减少重复逻辑，并为真实硬件或受限模拟器铺路。

1. 新增 `Target`。已落地：`aicir.devices.Target`（见第 3 节当前状态）。
2. 新增 `GateSpec` 注册表。已落地（见第 7 节当前状态）。
3. 让 QASM、visual、qml、metrics、qas 从 `GateSpec` 获取门元信息。部分可用：QASM/visual/IR 校验已接入；metrics/qas 评分语义待迁。
4. 实现基础 `DecomposePass`、`LayoutPass` 和 `RoutingPass`。已落地（见第 2 节当前状态）；消费 `Target` 的门集与耦合拓扑。

### 第三阶段：QNode 和自动微分工作流

主要目标是提升 QML/VQA 用户体验。

1. 新增 `aicir.qnode`。
2. 新增 `expval`、`probs`、`sample` 等测量返回构造器。
3. 新增 `DiffMethod` 注册表。
4. 将 `qml.deriv`、`optimizer.params`、`vqc` 逐步接入 QNode。
5. 可选新增 Torch layer 包装，方便混合神经网络训练。

### 第四阶段：互操作和生态扩展

主要目标是接入外部生态。

1. 完善 QASM3 round-trip。
2. 新增 Qiskit 转换器。
3. 新增 PennyLane 转换器。
4. 为真实硬件或远端执行预留 provider 接口。

## 设计原则

- 保留轻量入口：门字典和 `Circuit(...)` 仍应可直接使用。
- 内部强类型化：优化、绘图、QASM、梯度和执行应逐步依赖 typed IR。
- 分层清晰：core 只负责表达，transpile 负责变换，primitives 负责执行，vqc/qas 负责算法。
- 可组合优先：pass、transform、diff method、estimator 都应能单独测试和替换。
- 渐进迁移：不要一次性重写现有模块，先新增薄抽象，再逐步把算法层切过去。

## 参考资料

- Qiskit transpiler: https://quantum.cloud.ibm.com/docs/en/guides/transpile
- Qiskit primitives: https://quantum.cloud.ibm.com/docs/en/guides/primitives
- PennyLane QNode: https://docs.pennylane.ai/en/stable/introduction/circuits.html
- PennyLane interfaces: https://docs.pennylane.ai/en/stable/introduction/interfaces.html
- PennyLane compilation transforms: https://docs.pennylane.ai/en/stable/introduction/compiling_circuits.html

## QAS 和 QML 待办任务清单 (Pending Tasks)

基于当前进度，以下是 QAS 和 QML 模块待完成的任务：

### 1. 量子架构搜索 (QAS)

#### 1.1 扩展评估指标
`aicir.qas.evaluator.ArchitectureEvaluator` 中的部分高级评估指标目前仍为 `todo` 状态：
*   **Expressibility:** 
    *   `frame_potential`
    *   `entangling_capability`
    *   `transformer_predictor`
*   **Trainability:** 
    *   `gradient_variance`
    *   `gradient_norm`
    *   `barren_plateau_risk`
*   **Noise Robustness:** 
    *   `ideal_noisy_score_gap`
    *   `per_source_ablation`
*   **Hardware Efficiency:** 
    *   `connectivity_penalty`
    *   `calibrated_error_cost`
    *   `latency_cost`

#### 1.2 含噪声的可微 QAS (Noisy Differentiable QAS)
*   目前 `supernet.py` 中的 `NoiseConfig` 仅为占位符。需要完成含噪声信道下解析梯度的计算与评估机制。

#### 1.3 硬件目标集成 (Hardware Target Integration)
*   将 QAS 搜索空间和硬件效率评分迁移至统一的 `Target` 抽象（定义后端能力、拓扑、原生门集），而不是依赖硬编码的 `DEFAULT_NATIVE_GATES` 等代理。

---

### 2. 量子机器学习 (QML)

#### 2.1 PennyLane 风格的 `QNode` 抽象
*   实现 `QNode` 接口（例如 `@qnode(device="numpy", diff_method="psr")`）。
*   统一“量子函数 + 设备 + 测量 + 梯度”，减少算法间重复的参数绑定和后端指定代码。

#### 2.2 整合梯度选择器
*   梯度注册表和自动选择机制 (`select_diff`) 已实现，并已由 `qfun`（`differential="auto"`）接入为首个调用方。
*   **任务**：将 `select_diff` 进一步接入 `Estimator` primitives，以便在不支持 Torch 时自动降级到 `psr` 或 `fd`。

#### 2.3 `GateSpec` 元数据扩充
*   向 `GateSpec` 注册表添加解析梯度所需的 `generator` 和 `decomposition` 字段。
*   允许 QML 梯度方法自动内省门类是否支持解析参数移位等计算，而无需硬编码判断。
