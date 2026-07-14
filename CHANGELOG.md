# Changelog

本文件记录 `aicir` 库的功能新增与重要接口变化。日期使用本地开发日期。

## 2026-07-14

### Changed

- **`Measure.run` 新增 `return_probabilities: bool = True`：完整概率数组可选。** 传 `False` 时 `Result.probabilities` 为 `None` 且全程跳过其计算（exact/shots=1 路径不再调用 `State.probabilities`，`aggregate_avg` 新增 `include_probabilities` 参数同步跳过），供大比特下只取计数/末态的调用方省内存；`Result.most_probable()` 在概率不可用时抛 `ValueError`（附原因）。默认 `True`，既有行为与所有现有调用方（`BasicQAOA.probabilities`、`qml.qfun`、`primitives` sampler 等）完全不变。NPU 探针 capacity 档改用只取计数的最轻组合（`return_state=False` + `return_probabilities=False`）。
  - 所修缺陷：此前 `Result.probabilities` 无条件生成——每次 `run` 都物化完整 2^n float64 概率数组（`to_numpy(...).astype(np.float64)` 还需一次显式拷贝，外加后端 |amp|² 中间量），即使调用方只需要 `counts(-1)`。n=32 时仅此一项约 34 GB，与态向量本身（~34 GB complex64）相当，使峰值内存约翻倍；末端采样自 `sample_terminal_batch` 起已独立计算 Born 分布，不依赖该字段。该行为自统一测量入口引入以来一直存在。

- **`Measure.run` 无中途随机源路径的末端读出改为批量采样（同种子计数流变更）。** 无噪声且无线路内随机源的 shot 路径中，末端 Z 基读出不再逐 shot 调用 `terminal_z_measure`，改为新增的 `projector.sample_terminal_batch`：Born 分布只计算一次、一次性抽取全部 M 个读出结果（O(2^n + M)），坍缩后完整态只按不同读出结果各构造一次、且仅在下游需要聚合末态（`return_state` 或 `observables`）时才构造。分布与逐 shot 逐比特投影严格一致（先按 Born 规则采样全寄存器 index 再读子集比特，与原实现同构），但随机数消费顺序改变——**同一 seed 下的具体计数与此前版本不同**（跨版本 seed 复现不属于既有契约）。含噪声 / in-circuit measure / 控制流的逐轨迹路径不受影响。
  - 所修缺陷：此前该路径对每个 shot 各调用一次 `terminal_z_measure`，每次都重新计算完整 2^n Born 分布、做整段态向量的子集投影与归一化，总代价 O(M·n·2^n) 且逐 shot 产生一份坍缩态副本（即使 `return_state=False` 从不使用）。真机 Ascend NPU 上 n=24、shots=200 的探针 capacity 档为此耗时 633s，其中密度矩阵聚合修复后剩余耗时几乎全部来自这里；该行为自统一测量入口引入以来一直存在。
  - 真机验证：`scripts/npu/measure_agg.sh --n-qubits 24` 严格 NPU（npu:0）4/4 cases 通过（含验证共享纯态向量聚合契约的 `shared_pre_vector_state`），capacity 档由修复前的 633s 降至 8.40s。

- **`Measure.run` 多 shot 共享纯态前态：聚合 `state` 保持向量形态（契约变更）。** 无噪声且无线路内随机源（无 in-circuit measure/控制流）的 `shots>1` 路径中，全部轨迹共享同一纯态前态，`avg(|ψ><ψ|) == |ψ><ψ|`，聚合 `state` 不再折叠为 `(2^n,2^n)` 密度矩阵而是保持向量形态 `State`（`np.asarray(result.state)` 形状由 `(2^n,2^n)` 变为 `(2^n,)`）；无末端测量（`measure_qubits=None`）时 `final_state` 同为向量（`final_state_kind="state_vector"`），全程不构造密度矩阵。有末端测量时 `final_state` 仍为密度矩阵（真混合态，契约不变），但改为按读出结果分组构造（新增 `aggregate.terminal_mixture`，外积次数 = 不同读出结果数 ≤ min(M, 2^k)，而非 shots 数 M）。噪声 / `initial_density_matrix` / in-circuit measure 路径行为完全不变。另：`aggregate_avg` 的快照与轻量概率聚合按对象身份去重，共享轨迹下只计算一次。

  - 修复Bug：此前 `sm="avg"` 聚合对 `shots>1` 一律做密度矩阵平均——即使无噪声、无线路内随机源、全部 M 条轨迹共享同一纯态前态（`Measure.run` 在该路径只演化一次、末端采样 M 次），仍会对同一个态向量做 M 次 `|ψ><ψ|` 外积并累加进两个 `(2^n,2^n)` complex 累加器（pre/post 各一），快照与概率聚合同样逐轨迹重复 M 次。空间 O(4^n)、时间 O(M·4^n)，n≈16–17 起即不可行（64–256 GB），而结果在数学上恒等于单个 `|ψ><ψ|`，属纯粹浪费；该行为自 `aggregate.py` 引入以来一直存在。
- **`Measure.run` 多 shot 路径按需跳过密度矩阵聚合。** `shots>1` 且 `return_state=False`、未传 `observables` 时，`sm="avg"` 聚合不再构造 `(2^n, 2^n)` 的 pre/post 密度矩阵平均（此前即使调用方丢弃聚合态也会构造，无噪声纯态线路下为纯粹浪费）；`probabilities` 改为逐轨迹概率向量的平均，与 `diag(平均密度矩阵)` 数学等价，采样流（种子可复现性）不变。`return_state=True` 或传入 `observables` 时行为完全不变（聚合态仍为密度矩阵）。配套：`aggregate_avg(...)` 新增 `include_states: bool = True` 参数，`False` 时返回字典中 `state`/`final_state` 为 `None`。

  - 修复Bug：此前 `return_state=False` 只在装配 `Result` 时把 `state`/`final_state` 字段置 `None`，密度矩阵本身照常构造——`shots>1` 的 `sm="avg"` 聚合无条件对 M 条轨迹做 pre/post 两个 `(2^n,2^n)` complex 密度矩阵平均，再把结果丢弃。即调用方显式声明不要聚合态（如 `BasicQAOA.sample`/`probabilities` 只取计数与概率），仍要付出 O(4^n) 内存与 O(M·4^n) 计算，n≈16–17 起即不可行；该行为自 `aggregate.py` 引入以来一直存在。

### Added

- **`aicir.encoder.IQPEncoder`：IQP 特征映射编码器。** 实现 Havlicek et al.（Nature 567, 209 (2019)，arXiv:1804.11326）提出的量子增强特征空间编码 `|Phi(x)> = (U_Phi(x) H^n)^reps |0>^n`，其中 `U_Phi(x)` 为 Pauli-Z 基对角相位酉阵（`|S| <= 2`），默认系数取论文选择 `phi_i = x_i`、`phi_ij = (pi - x_i)(pi - x_j)`（可经 `data_map=` 自定义）。对角层用 `rz`/`rzz` 精确实现（`exp(i*phi*Z) == rz(-2*phi)`，无全局相位差）。构造参数：`n_qubits`（可选，数据不足补零）、`reps`（默认 2，同论文）、`entanglement`（`"full"`/`"linear"`/显式配对列表）。除 `BaseEncoder` 的 `encode()` 外另提供 `circuit(data)`（只建线路不演化）与量子核方法 `kernel(x, z)`、`kernel_matrix(xs, zs=None)`（`K = |<Phi(x)|Phi(z)>|^2`，可直接喂经典 SVM）；IQP 映射不可逆，`decode()` 抛 `NotImplementedError`。测试 `tests/circuit/test_iqp_encoder.py` 含与论文定义逐元素对照的独立 NumPy 参考实现。

## 2026-07-12

### Added

- **`aicir.qml.deriv` 拆包（Phase 4，纯代码移动）。** 2660 行单文件 `aicir/qml/deriv.py` 按功能拆为 `deriv/` 包：`_coerce.py`（共享归一化/torch 探测 helper）、`fn_gradient.py`（`psr`/`psr4`/`spsr`/`spsa`/`fd`/`auto`/`mpsr` 及各自 torch/NPU 设备驻留私有变体）、`hessian.py`、`adjoint.py`（`ad`）、`qfim.py`（`qfim`/`metric_tensor`/`qfim_diag`/`qfim_blocks`）、`qng.py`（`qng`/`bdqng`/`kqng`/`dqng`）、`rotosolve.py`。全部 84 个函数体逐字节不变（AST 对照校验）；唯一 dedup 是原文件两份 byte-identical 的 `_real_torch_dtype_from_backend`。包 `__init__.py` 重新聚合原扁平命名空间——拆分前可从 `aicir.qml.deriv` 导入的 91 个名称（含私有 helper）全部保留（`tests/qml/test_deriv_package_layout.py` 钉子测试），numpy/torch 双实现按 NPU 约定不做合并，torch 依旧是可选依赖。
- **`aicir.qml.deriv` array-in/array-out 返回契约文档化。** 包 `__init__.py` docstring 明确：`psr`/`psr4`/`fd`/`auto`/`spsa`/`spsr`/`mpsr`/`ad`/`hessian`/`qfim` 族以及 `qng`/`bdqng` 一律返回 NumPy；仅 `kqng`/`dqng`（NPU-family backend 或 torch `grad`/`qfim_diag`/`kfac_factors` 入参）与 `rotosolve`（torch `params` 或 Torch/NPU backend）有 torch 入参→同设备 torch 出参路径。这是现状的文档化，未改任何返回行为；新增 `tests/qml/test_return_type_contract.py` 钉住契约。
- **`psr4` 顶层导出 + 注册表注册。** `aicir.qml` 顶层现在导出 `psr4`（四项参数移位规则，激发门等 {-1,0,1} 谱生成元）；`aicir/qml/diff/registry.py` 注册为 `fn_gradient` 类别（`exact=True`，与 `psr` 同为解析规则；签名满足 `(fn, params, **kw) -> ndarray` 契约），`get_diff("psr4")`/`resolve_diff("psr4")` 可发现与解析。刻意不加入 `select_diff` 自动优选偏好——仍是 `auto -> psr -> fd`，`psr4` 需按线路生成元谱显式选择。
- **`hessian` 新增 `method="auto"` 显式降级语义。** `hessian` 此前默认 `method="psr"` 但在 psr 二阶对角元与 fd 不一致时静默降级为 fd。现默认改为 `method="auto"`：数值行为与旧默认完全相同，但降级触发时发出 `RuntimeWarning`；`method="psr"` 变为严格模式，原本触发降级的条件现在抛 `ValueError`（不再静默换算法）；`method="fd"` 行为不变。
- **`aicir.chemistry._qiskit_bridge`：收敛两条 Qiskit Nature 生成路径。** `pipeline.build_molecule` 与 `spec.generate_hamiltonian`（`MolecularSpec` 分支）此前各自维护一份 driver 构造/mapper 选择/active-space 变换/`SparsePauliOp -> terms` 转换逻辑，现统一收敛到新的私有模块 `aicir/chemistry/_qiskit_bridge.py`：`build_driver`、`select_mapper`、`apply_active_space`/`apply_active_space_mapping`、`sparse_pauli_to_terms`、`hf_occupation_from_mapper`、`structural_excitations`，以及显式的 `reverse_pauli_labels`（供需要旧镜像比特序的调用方使用，默认路径不调用）。模块声明 `QUBIT_ORDER = "qiskit_label"`：canonical 比特序是 Qiskit `SparsePauliOp.to_list()` 原样标签，不翻转，与已冻结分子预置一致。`qiskit_nature`/`pyscf` 保持可选依赖，仅在函数体内惰性 import。`pipeline.build_molecule` 的公开签名与输出经回归测试钉住为重构前后字节级一致。
- **`GeneratedHamiltonian.to_hamiltonian()`。** 与 `MoleculeHamiltonian.to_hamiltonian()` 同名同型，返回 `aicir.core.operators.Hamiltonian`。两个 dataclass 仍不合并（`GeneratedHamiltonian.terms` 是 `float` 系数、JSON 友好；`MoleculeHamiltonian.terms` 是 `complex` 系数），仅在此方法处汇合。
- **≤6-qubit 预置补齐 `n_electrons`/`hf_occupation`/`excitations` 元数据。** `h2`（ParityMapper 两比特约化）、`h2_jw`、`lih`、`h2o` 四个已做基态能量验证的 preset 现补齐这三个字段，值通过本地 `pyscf`+`qiskit-nature`（现算 `build_molecule` 同几何/基组/mapping，逐项比对 terms 后抄录其元数据输出）交叉验证；`h2_tapered`（`TaperedQubitMapper` 不在 `build_molecule` 支持的三种 mapper 之列，且 tapering 后单比特表示不再与自旋轨道占据数一一对应）与 12–16 qubit 结构守卫 preset（`nh3`/`n2`/`beh2`）保持 `None`，未补齐。冻结的 `terms` 数据本身未改动。这使得文档化的 `uccsd(mol.n_qubits, mol.hf_occupation, mol.excitations)` 桥接现在也能直接吃固定预置，不再必须先跑可选的 `build_molecule` 现算流水线。
- **`aicir.qas`：Phase 3a——统一基础设施（不改变现有 `run()` 分发与各方法返回值，接入留给 3b）。**

  - **`QASResult`**（`aicir/qas/core/results.py`，从 `aicir.qas` 顶层导出）：跨 QAS 方法的统一结果包装（`method`/`value`/`circuit`/`parameters`/`history`/`metadata`/`raw`），满足 `aicir.protocols.AlgorithmResult` 协议；`raw` 保留方法专属原始结果对象（如 `SupernetResult`/`PPRDQLResult`）不丢信息。
  - **QAS 问题输入归一化**（`aicir/qas/core/problem.py`）：`normalize_problem(obj, *, n_qubits=None, kind=None) -> QASProblem` 统一接受 `Hamiltonian`、`State`、方阵/态向量 `ndarray`、带 `to_hamiltonian()` 的对象（duck-type 覆盖 `aicir.chemistry` 的 `MoleculeHamiltonian`/`GeneratedHamiltonian`，不做硬导入）以及 Pauli 项列表。新增 `normalize_terms`/`terms_label_first`/`terms_coeff_first` 显式处理 `aicir/qas/problems/hamiltonians.py` 预置项的 `(coeff, label)` 顺序与 `aicir.core.operators.Hamiltonian` 规范 `(label, coeff)` 顺序之间的转换（按类型消歧，歧义即报错，不做猜测）。方阵输入默认归为哈密顿量矩阵，仅当迹≈1 且半正定时归为密度矩阵，可用 `kind=` 显式覆盖。
  - **后端解析收敛**：新增 `aicir/qas/core/backend_utils.make_torch_backend(device)`，把 `supernet.py`/`qdrats.py`/`dqas.py` 里三份逐字节相同的 `_make_backend`（`device` 以 `"npu"` 开头选 `NPUBackend`，否则 `GPUBackend`，`device=None` 落到 `GPUBackend(device=None)`）收敛为单一实现，三处调用点改为委托，行为不变。`CRLQASConfig` 新增 `device: str | None = None` 字段；`train_crlqas` 通过新增的 `_resolve_crlqas_backend` 解析后端——`device=None` 仍直接用 `NumpyBackend()`（与改造前逐字节一致，含不污染全局 RNG 的保证），非 `None` 时委托 `make_torch_backend`。
  - **配置字段别名**：`core/config.py` 新增 `_FIELD_ALIASES`（方法名 -> {旧字段名: 规范字段名}），在 `_build` 的“未知字段报错”之前应用，命中时发出 `DeprecationWarning`。`PPORollbackConfig`/`PPRDQLConfig` 的 `episode_num` 改名为 `max_episodes`（`CRLQASConfig` 本就用 `max_episodes`，三者统一）；`CRLQASConfig.q_hidden_dim` 改名为 `hidden_dim`（`PPORollbackConfig`/`PPRDQLConfig` 本就用 `hidden_dim`）；旧字段名仍可作为 `config.<method>(...)` 关键字参数使用（直接构造 dataclass 则必须用新名）。`q_learning_rate`/`architecture_learning_rate`/`supernet_steps`/`search_epochs` 等语义不同的相似字段刻意不合并，理由见 `aicir/qas/README.md` §9 词汇对照表。同步更新了受影响的现有测试与 demo 脚本里的直接构造调用点。
- **`aicir.optimization.qubo` 新增强类型 `IsingExport`（Phase 5，接口统一收尾）。** `IsingModel`/`Model` 此前只有 dict-returning 的 `IsingModel.named()`/`Model.to_ising()` 导出 Ising 模型（`h`/`J`/`offset` 裸 dict）；现新增冻结 dataclass `IsingExport`（`linear`/`quadratic`/`offset`/`variable_names`/`variable_metadata`）与对应方法 `IsingModel.to_export()`（索引键，字段与 `self.h`/`self.J` 语义一致）、`Model.to_ising_export()`（变量名键，与 `to_ising()` 字段语义完全一致）。`IsingExport` 从 `aicir.optimization.qubo.modeling`（并经通配导出传导至 `aicir.optimization.qubo`）导出。原 dict-returning 方法保留不破坏兼容，docstring 标注已弃用并指向类型化替代。
- **torch 可选化：`aicir.vqc` numpy 路径无 torch 环境可用。** `aicir/vqc/__init__.py` 此前用 `try/except ModuleNotFoundError` 包裹 `BasicVQE`/`BasicVQD`/`BasicSSVQE` 的导入，误以为这三者依赖 torch；实际排查（`VQE.py`/`VQD.py`/`SSVQE.py` 仅经 `aicir.qml.deriv` 间接引用 torch，而该包的 torch 探测/设备驻留分支本就是函数体内惰性 import，参见 `aicir/qml/deriv/_coerce.py`/`fn_gradient.py`）确认三者在 numpy 路径下从不触发 torch 导入。现移除该 try/except——`from aicir.vqc import BasicVQE` 恒可成功，`aicir/backends/__init__.py`/`aicir/__init__.py` 原有的 GPU/NPU 后端可选降级保持不变。新增 `tests/vqc/test_import_without_torch.py`：用 meta path finder 在子进程内真实阻断 `torch`/`torch_npu` 导入（而非在当前进程伪造 `sys.modules["torch"] = None`，那种手法会被 `import torch` 直接命中缓存产生假绿），验证 `import aicir`、`from aicir.vqc import BasicVQE`、numpy 后端 `BasicVQE.run()` 端到端可跑且能收敛出有限能量。
- **`BasicQAOA.run`/`BasicVQE.run` 新增 `learning_rate` 规范别名。** 两者的 `run()` 此前只有 `lr` 关键字；现新增 `learning_rate: float | None = None`，语义与 `aicir/protocols.py` 词汇表一致（`lr` 为兼容别名，规范名 `learning_rate`）。同时传入且取值冲突（`learning_rate` 与非默认 `lr` 不一致）时抛 `ValueError`；只传 `learning_rate` 时以其为准。`aicir/optimization/qubo/qaoa.py` 的 `run_model_qaoa`/`run_qubo_qaoa`（`BasicQAOA.run` 关键字参数的超集，有回归测试钉住）同步新增并原样转发该参数。

### Fixed

- **README 通读审计中发现的 6 处源码缺陷。**
  - **QAS supernet-native 架构生成器把 Hadamard 门写成不存在的 `"h"` 类型。** `aicir/qas/library/ansatz.py` 的 `architecture_from_supernet_gene` 把单比特门标签 `"h"` 原样塞进门字典的 `"type"` 字段；`GateSpec` 注册表的别名是大写 `"H"`（同其余 Pauli/S/T/I 单字母别名的大小写惯例），`canonical_gate_name("h")` 查不到，落地时对应门无法在本地矩阵路径求值，`fair_vqe._simulate_statevector` 会抛 `ValueError: cannot apply gate 'h' without dense expansion`。现改为写入规范名 `"hadamard"`。
  - **`hessian`/`GPUBackend.inner_product` 之外，另有两处复数矩阵乘法在部分 BLAS 构建（如 Apple Accelerate）上产生无害但会污染日志/触发 `-W error` 的 `RuntimeWarning`（divide/overflow/invalid）。** `aicir/ansatze/hea_ti.py` 的 `global_evolution_unitary`（4 比特以上的 `hea_ti_ansatz` 构造路径）与 `aicir/backends/numpy_backend.py` 的 `matmul`/`expectation_sv`/`expectation_dm`——后者的既有"提升到 complex128 以减少告警"注释并未真正生效。四处均已用 `np.errstate(divide="ignore", over="ignore", invalid="ignore")` 局部抑制，不改变数值结果（已核对幺正误差量级不变）。
  - **`aicir.noise.ion_trap.IonTrapNoiseConfig` 的 `prefer_calibration` 参数从未被读取。** `resolved_parameters`/`build_noise_model`/`idle_dephasing_probability`/`summary` 都接受或透传这个参数，但函数体内没有任何校准数据源可供"优先选择"，参数純属摆设。四处均移除该参数（无外部调用方显式传参，纯净删除，行为不变）。
  - **`aicir.noise.analysis.noise_sensitivity` 的 `n_samples` 参数从未被读取。** 该函数用精确密度矩阵比较理想/含噪保真度，全程无随机采样；调用方 `aicir/qas/core/evaluator.py` 却按其余需要采样的指标（KL/MMD/梯度统计）的统一约定传了 `n_samples=self.n_samples`。现移除该参数并同步更新调用点，行为不变。
  - **测量→reset 之间的量子线本应画成与 measure 同色的虚线，实际渲染成与普通量子线完全相同的实线。** `aicir/visual/plot.py` 的 `_draw_reset_link` 用的是 `wire_color` 且未设 `linestyle`，与相邻普通实线线宽/颜色/zorder 全部相同，视觉上无法区分（尽管周围逻辑专门为它排除了一段普通实线）；函数自身的三处 docstring 都称之为"dash"。现改为量测边缘色的虚线（`linestyle=(0, (4, 3))`，间距大于 matplotlib 默认虚线）。顺带清理了因此永远走不到的死配色 `_PALETTE["t_gate"]`（非 Clifford 门统一走 `_PALETTE["hadamard"]` 配色，T 门/Toffoli 从未用过该条目）。
  - **`train_crlqas`/`crlqas` 未暴露已实现的 `estimator=` 注入点。** `_energy_of_gates` 内部早已支持可选 `estimator`（`aicir.primitives` `BaseEstimator` 契约）参数，但公开的 `train_crlqas`/`crlqas` 函数签名没有对应关键字，外部调用方无法使用。现两者新增 `estimator: Any = None` 关键字参数并透传给训练结束后的兜底能量求值（Adam-SPSA 热循环按既有设计始终走 `estimator=None` 的直接态向量路径，不受影响，热路径性能不变）。
- **`qng` 族对 `gradient_method="spsr"` 显式报错。** `qng`/`bdqng`/`kqng`/`dqng` 的普通梯度分发（numpy 与 torch/NPU 两条路径）此前对 `"spsr"` 落入泛化的 "must be 'psr', 'fd', 'spsa', or 'auto'" 报错；现改为专门的 `ValueError` 中文信息，说明原因（spsr 的随机坐标采样与 QNG 度规预条件组合未验证/不支持）并列出受支持方法。
- **`spec.py` Breaking：修复与 `pipeline.py`/冻结预置相反的比特序。** `spec._sparse_pauli_terms`（原 ~244-253 行）此前对 Qiskit label 做 `[::-1]` 镜像翻转，与 `pipeline.build_molecule`（不翻转，见其内联注释）及已冻结分子预置的比特序方向相反——对同一分子，`generate_hamiltonian(MolecularSpec(...))` 与 `build_molecule(...)` 会产出镜像对称但不等价的 Hamiltonian。现改为委托 `_qiskit_bridge`，删除该翻转，两条路径的比特序统一为 canonical `qiskit_label`（`GeneratedHamiltonian.metadata["qubit_order"] == "qiskit_label"` 可查）。**迁移说明**：如确实需要旧的镜像比特序，显式调用 `aicir.chemistry._qiskit_bridge.reverse_pauli_labels(terms)` 自行翻转；`spec.py` 的 `PresetSpec`/`PauliTermsSpec` 分支不受影响（前者本就直接透传预置 terms 未翻转，后者是字面量输入）。影响面很小：`spec.py` 的 preset 导入路径在 Phase 0 之前本已因 `ModuleNotFoundError` 断链（见上一节 Fixed），`MolecularSpec` 分支此前几乎不可能被下游依赖到镜像比特序这一具体细节。

### Breaking

- **`aicir.qas`：Phase 3b——QAS 调度迁移到 Phase 3a 基础设施，`run()` 统一返回 `QASResult`。** 承接上一节 Phase 3a（新增基础设施但不改行为）；本次是接线：
  - **`SearchStrategy` 注册表收官。** `core/config._FACTORIES` 的全部 10 个方法（`supernet`/`supernet_classification`/`supernet_h2`/`pporb`/`pprdql`/`crlqas`/`qdrats`/`dqas`/`vqe_loop`/新增的 `mogvqe`）现在都在 `core/strategies.py` 注册为 `SearchStrategy`（实现拆到新增的 `core/adapters.py`，`strategies.py` 只保留注册）。`runner.py` 的旧 `_Spec`/`_TABLE`/`_load` 分发表已删除；`run()` 只查策略注册表。`available_qas_methods()` 现在派生自注册表（`registered_strategies()`，字典序），不再是 `_FACTORIES` 的插入序。
  - **`run()` 对全部方法统一返回 `QASResult`。** 之前 `pporb` 返回裸 `(theta, circuit)` 元组、`supernet`/`dqas`/`qdrats`/`crlqas`/`pprdql`/`vqe_loop` 各自返回方法专属结果对象（`SupernetResult`/`DQASResult`/`QDRATSResult`/`CRLQASResult`/`PPRDQLResult`/`ClosedLoopResult`）；现在全部统一包装为 `QASResult`（`method`/`value`/`circuit`/`parameters`/`history`/`metadata`/`raw`），原始对象保留在 `raw` 不丢信息。迁移表：
    | 方法                                                     | 旧用法                                                                   | 新用法                                                                                                                                                                                |
    | -------------------------------------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | `pporb`                                                | `theta, circ = run("pporb", ...)`                                      | `r = run("pporb", ...)`；`r.parameters`（旧 `theta`）、`r.circuit`；`r.value` 恒为 `None`（`ppo_rb_qas` 本就不返回保真度，重算需重跑环境仿真，3b 不做）                 |
    | `supernet`/`supernet_classification`/`supernet_h2` | `result.best_circuit`/`result.best_score`/`result.ranking_records` | `r.circuit`（= 旧 `best_circuit`）、`r.value`（= 旧 `best_score`）；`ranking_records`/`best_architecture`/`final_metrics` 移到 `r.metadata`，或整份旧对象走 `r.raw` |
    | `dqas`/`qdrats`                                      | `result.circuit`/`result.parameters`/`result.minimum_energy`       | `r.circuit`、`r.parameters` 字段名不变；`result.minimum_energy` -> `r.value`；`search_log` -> `r.history`；`finetune_log`/架构相关字段移到 `r.metadata`               |
    | `crlqas`                                               | `result.circuit`/`result.parameters`/`result.minimum_energy`       | 同上：`r.circuit`/`r.parameters` 不变，`minimum_energy` -> `r.value`，`episode_best_energies` -> `r.history`                                                              |
    | `pprdql`                                               | `result.circuit`/`result.best_fidelity`/`result.policy`            | `r.circuit` 不变，`best_fidelity` -> `r.value`，`policy` -> `r.parameters`，`episode_rewards` -> `r.history`                                                            |
    | `vqe_loop`                                             | `result` 直接是 `ClosedLoopResult`（文件路径）                       | `r.raw` 是原 `ClosedLoopResult`；`r.value`/`r.circuit`/`r.parameters` 恒为 `None`（该方法产出 benchmark-table CSV，没有内存态最优解，解析 CSV 不是廉价操作，3b 不做）     |
  - **`run(method, problem=..., **kwargs)`：统一任务输入参数（新增，向后兼容）。** `QASRunConfig` 新增 `problem: Any = None` 字段，接受 `normalize_problem()` 支持的任意形态（`Hamiltonian`/`State`/矩阵/态向量/Pauli 项列表/`QASProblem`）。旧字段 `hamiltonian=`/`target_state=`/`target_density_matrix=` 继续可用并路由到同一个归一化问题；`problem=` 与旧字段同时传入、或两个旧字段同时传入，均报 `ValueError`。`mogvqe` 额外新增 `initial_ansatz` 字段（拓扑起点，不属于 problem/config 语义）。
  - **可选 `estimator=` 注入点（新增，不改变默认行为）。** `crlqas._energy_of_gates(..., estimator=None)` 与 `vqe_loop.fair_vqe._evaluate_pauli_state_energy`/`evaluate_vqe_energy(..., estimator=None)`：默认 `None` 时数值路径与改造前逐字节一致；传入实现 `aicir.primitives` `BaseEstimator.run(circuit, observable) -> EstimateResult` 契约的对象时，energy 改走 `estimator.run(...).value`。未打通到 `run()` 的 adapter 层（仅在函数级别可用），vqe_loop 的 `optimize_vqe_energy` 内部 COBYLA 目标函数闭包同样未接入。
  - **清理**：`aicir/qas/algorithms/{supernet,qdrats,dqas}.py` 删除未使用的 `NPUBackend` 死导入（各自委托 `core.backend_utils.make_torch_backend` 后不再直接引用该类型）；`crlqas.py` 的 `_resolve_hamiltonian_matrix(..., backend: NumpyBackend)` 类型注解更正为 `backend: Backend`（该函数经 `_resolve_crlqas_backend` 实际可能收到 `GPUBackend`/`NPUBackend`）。

## 2026-07-11

### Added

- **`aicir.protocols`：跨层结果/优化器协议（Phase 1 跨层契约统一）。** 新增 `AlgorithmResult`（`value`/`parameters`/`history`/`metadata` 只读协议，`runtime_checkable`）、`Optimizer`（`minimize(fn, init_params, ...)` 协议）与 `HistoryRecord`（`step`/`fun`/`grad_norm`/`learning_rate`/`extras`，支持 `record["fun"]`/`record.get(...)`/`"key" in record` 兼容旧 dict 键访问）。模块零重依赖（仅 `typing`/`dataclasses`），不从顶层 `aicir/__init__.py` 导出。
- **Result 词汇表对齐 `value`/`parameters`/`history`/`metadata`。** `VQEResult`/`QAOAResult` 新增 `value`（等价 `energy`）与 `history`（等价 `energy_history`）别名属性；`VQDResult`/`SSVQEResult` 新增 `value`（分别取基态 `energies[0]` 与联合子空间目标 `weighted_cost`）、`history`（对应 `objective_histories[0]`/`cost_history`）与 `metadata`（原无该字段，返回 `{}`）；`OptimizationResult` 补齐 `metadata` 别名（同样返回 `{}`，与既有 `.parameters`/`.value` 别名对齐）；`QAOAResult.parameters` 已是 `concatenate([gammas, betas])`（gammas 在前），天然满足协议语义，未另加同名属性。`aicir.primitives.results.EstimateResult` 新增 `energy`（等价 `value`）以及 `parameters`/`history`（单次期望值估计无此语义，均返回 `None`），使其满足 `AlgorithmResult`。
- **estimator `estimate()` 改为委托 `run()`。** `aicir.primitives.estimator` 的 `StatevectorEstimator`/`NoisyEstimator`/`ShotEstimator` 的 `estimate()` 均重新实现为对 `run()` 的薄委托（同一次底层数值计算，非重复求值/采样），仅重新打包为旧调用方期望的类型（`_EnergyResult` 或 `PauliEstimateResult`）以保持向后兼容；`_EnergyResult` docstring 标注已弃用，新代码应消费 `run()` 返回的 `EstimateResult`。`BasicVQE._evaluate_circuit` 的注入 `energy_estimator` 消费路径改为优先调用 `run()`（含默认的 `StatevectorEstimator` 内部路径），仅当对象只暴露 `estimate()`（如原生 `PauliEstimator`）时才退回旧契约；`_resolve_energy_estimator` 相应放宽为接受 `run`/`estimate` 任一方法。
- **optimizer history 改用 `HistoryRecord`。** `GD`/`Adam`/`SPSA`/`ScipyMinimize` 的逐步历史记录从裸 dict 改为 `aicir.protocols.HistoryRecord`（SPSA 的 `perturbation` 等专属字段进入 `extras`）；`HistoryRecord` 的 dict 风格访问（`["fun"]`/`.get(...)`/`in`）保持旧消费方代码不变。`aicir.optimizer.params.minimize()` 调度器的类型检查由 `hasattr(optimizer, "minimize")` 改为 `isinstance(optimizer, Optimizer)`（`aicir.protocols.Optimizer`，效果等价，错误信息不变）；`BasicVQE._run_with_optimizer` 读取 `opt_result.history` 时的过滤条件相应从 `isinstance(entry, dict)` 放宽为 `hasattr(entry, "get")`，以兼容 dict 与 `HistoryRecord` 两种历史记录形态。

### Fixed

- **`aicir.chemistry.spec` preset 路径修复。** `_generated_from_preset` 内部导入的 `from .molecule import get_molecule` 指向已废弃的单文件模块，preset 分支必然 `ModuleNotFoundError`；改为 `from .molecules import get_molecule`。
- **`run_model_qaoa` 补齐转发参数。** 新增显式关键字参数 `optimizer=`/`shots=`/`backend=`/`method=`/`grad_method=`（与 `BasicQAOA.run` 默认值一致）并原样转发；原有 `seed=` 同时用于构造 `BasicQAOA`（初始参数采样）与转发给 `BasicQAOA.run`（shots 采样等运行期随机性）。`run_qubo_qaoa` 别名不变。
- **`SPSA.minimize` 补齐 `gradient_fn=`。** 与 `GD`/`Adam`/`ScipyMinimize` 共享的 `minimize(fn, init_params, *, gradient_fn=None, callback=None)` 签名对齐；提供 `gradient_fn` 时使用其梯度估计替代 SPSA 同步扰动估计，不提供时行为与之前完全一致。
- **`crlqas` 局部化随机数。** `train_crlqas` 不再调用 `np.random.seed`/`random.seed` 污染全局状态，改为仅使用局部 `random.Random(cfg.seed)`/`np.random.default_rng(cfg.seed)`（原本模块内部随机数已全部走这两个局部生成器，全局 seeding 属冗余污染）；`torch.manual_seed` 因 `nn.Linear` 默认初始化没有干净的局部生成器路径而保留，但收窄到紧邻网络构造之前。**同 seed 的数值序列会发生变化**（不再与之前的全局污染路径等价）。
- **QAS config 报错信息补充字段名。** `aicir.qas.config` 的 `_build` 在实例化前比对 `dataclasses.fields`，未知字段的 `TypeError` 现在包含具体字段名与合法字段集合；其余 `TypeError`（如字段值类型错误）保留原始异常信息（`from exc` 串联）。

## 2026-07-09

### Added

- **QML 二阶与几何公开接口。** 新增 `aicir.qml.hessian`，支持 Pauli 旋转目标的二阶参数移位/`mpsr` 混合偏导，以及任意黑盒目标的有限差分 Hessian；新增 `qfim`/`metric_tensor`、`qfim_diag`、`qfim_blocks`，复用 QNG 内部 Fubini-Study QFIM 估计路径并支持 NumPy 与 Torch-family 状态返回。
- **Supernet QAS 补全 evolutionary ranking 与 noisy VQE 路径。** `SupernetConfig` 新增 `ranking_generations`、`ranking_mutation_rate`；`ranking_strategy="evolutionary"` 现在用采样种群、突变和截断选择生成候选，再走与随机 ranking 相同的 scoring/record schema。`NoiseConfig` 由占位扩展为可用配置，`noise_mode="depolarizing"` / `"amplitude_damping"` 在 VQE/H2 目标下走密度矩阵期望。
- **Chemistry mapper metadata parity。** `build_molecule(..., mapping="parity"|"bravyi_kitaev")` 现在填充 `n_electrons`、mapper-derived `hf_occupation` 和结构化 `excitations` 元数据；Parity two-qubit reduction 也返回可校验的 HF 元数据。Parity/BK excitation 元数据用于结构桥接，不宣称 mapper-correct 化学 UCCSD。

## 2026-07-06

### Added

- **MPS 引擎支持 Ascend NPU 前向 + 参数移位梯度。** `NPUBackend.svd` 由
  `NotImplementedError` 改为 real-embedding 实现（`[[Re,-Im],[Im,Re]]` 实块跑 NPU 原生实数
  SVD 后重建复数因子）；新增 `Backend.mul`/`div` 原语（numpy/gpu 为
  `*` 和 `/`，NPU 走 real/imag 分解，`_NpuMulFn` 自定义 autograd Function）；`mps.py` 的复数
  乘/除改走 backend 原语。NPU 上 `mps_statevector`/`mps_expectation`/`MPSEstimator` 前向可用，
  `MPSEstimator.gradient` 默认走 `psr` 参数移位；直接 autograd（`mps_expectation.backward`）
  仅 CPU/GPU 支持，Ascend 缺少 complex64 梯度累加内核。配套 `demos/demo_npu_mps.py`
  （真机验收）与 NPU 门控测试。CPU/GPU 结果不变。
- **`aicir.simulator` MPS（矩阵乘积态）近似模拟引擎（Spec 2）：`mps_statevector` /
  `mps_expectation`，并为 `Measure.run` 增加 `method="mps"`。** bond 截断由
  `max_bond_dim`（硬上限）+ `cutoff`（相对奇异值阈值，默认 1e-10）共同控制；正交
  中心 + SVD 的 TEBD 式演化，单比特门就地作用、相邻双比特门 SVD 截断、非相邻双比特
  门自动 SWAP 并跟踪逻辑↔物理置换。新增 `Backend.svd` 原语（NumPy/GPU 实现；NPU
  后续由 real-embedding 支持前向，见上条）。`mps_expectation` 对 `Hamiltonian`/
  `PauliString` 走 transfer 收缩不稠密化、GPU 上对参数门可微。NPU 梯度经
  `MPSEstimator.gradient(method="psr")` 参数移位。新增
  `aicir.primitives.MPSEstimator`（可注入 `BasicVQE(energy_estimator=...)`）。仅纯态、
  无噪声、1/2 比特门（≥3 比特门先经 `DecomposePass`）。

## 2026-07-05

### QAOA 稀疏化与解析梯度

- 门级 `BasicQAOA` 精确能量（`shots=None`）改走稀疏逐项期望 `Σ_j c_j⟨ψ|P_j|ψ⟩`，移除稠密 `to_matrix` 与 `2^n` Python 循环（对角/非对角统一）。
- `build_circuit` 重构为门磁带（`_qaoa_tape`/`_circuit_from_tape`），作为前向与梯度的单一事实来源。
- 新增可选解析参数移位梯度：`BasicQAOA.analytic_gradient(...)` 与 `run(..., grad_method="analytic")`（逐门 π/2 移位 + 链式法则聚合，对 Trotter 化线路解析精确）。默认梯度仍为有限差分（`grad_method="fd"`）。

### Changed

- **`BasicQAOA` 升级为 canonical gate-level QAOA。** `aicir.vqc.BasicQAOA` 现在可直接接收 aicir 标准 `Hamiltonian` 作为 `problem_hamiltonian`，支持任意实系数 Pauli cost 项；主路径会构造 `Circuit`（`H` 初态、`rz`/`rzz` 快速路径、一般 Pauli string 的一阶/二阶 Trotter-Suzuki product formula、`rx` mixer layer）。新增 `trotter_steps>=1` 与 `trotter_order=1/2`（默认 `1`）；diagonal I/Z-only cost 继续支持 bitstring energy 和基于最终 Z-basis counts 的 shots energy，非对角 cost 支持 exact expectation，shots energy 暂要求后续接 Pauli-term estimator。旧 dense matrix 输入仍作为 exact-simulator 兼容路径保留；QUBO helper 默认改为传递 `Hamiltonian`，显式 dense custom mixer 时才回退到 matrix 路径。

## 2026-07-04

### Changed

- **Breaking:** `Circuit.gates` 现在返回 typed instruction 列表，而不是可变门字典列表。元素类型包括 `Operation`、`Measurement` 和 `ControlFlow`；`Circuit.operations` 是同一 typed 视图的 tuple 形式。typed gate 仍支持旧字典键的只读访问（如 `gate["type"]`、`gate.get("parameter")`）以及与旧 dict 的 `==` 比较，但 `gate[...] = ...` 会抛 `TypeError`。
- 新增/稳定旧字典互操作 API：`Circuit.legacy_gates` 与 `Circuit.to_gate_dicts()` 返回 detached legacy dict 列表，供 JSON/QASM/第三方互操作和旧代码修改参数使用。迁移规则：把会修改门的 `copy.deepcopy(circuit.gates)` / `list(circuit.gates)` 改为 `circuit.to_gate_dicts()`；把序列化输出改为消费 `to_gate_dicts()`；新 typed 代码优先使用 `gate.name`、`gate.qubits`、`gate.params`。
- 内部消费端已迁移：控制流 body/else-body、QAS VQE 参数绑定、supernet 跨 rank payload 均显式走 legacy dict 快照，避免 typed immutable gate 泄漏到需要可变 dict 或 JSON-like payload 的边界。
- 内部运行时进一步 typed 化：`core.circuit`、`core.gates`、`core.batch`、`measure.trajectory`、`simulator.network`、`qml.deriv`、`metrics`、`visual.plot`、MoG-VQE、VQE-QAS preparation/fair binding 以及主要 transpile passes 不再读取 `gate["..."]` / `gate.get(...)` 业务字段，而是走 `aicir.ir` accessors。新增静态守卫测试禁止非边界生产代码重新依赖 legacy gate dict 字段。保留 dict 的边界限定为 IR dict round-trip、core IO/第三方互操作、legacy transpile rewrite API 和 CRLQAS action-space DTO。

### Added

- **显式 Nelder-Mead 优化器 API。** 新增 `aicir.optimizer.NelderMead`，作为 `ScipyMinimize(method="Nelder-Mead")` 的薄封装，便于对无梯度黑盒 VQE/VQA 目标直接选择 Nelder-Mead 单纯形法；已有 `scipy_minimize(..., method="Nelder-Mead")` 路径保持不变。
- **typed IR 迁移的 NPU 回归脚本矩阵。** 新增/整理 `scripts/npu/{smoke,backend,ops,capacity,typed_ir,circuit,deriv,qml,qaoa,tensor,qas,demos,run_all}.sh` 与统一 runner，支持 `--strict-npu`、`--fail-fast`、`--dry-run`、`--pytest-arg ...`。其中 `typed_ir_deriv_probe.sh --section all` 是轻量硬件探针：覆盖 `Circuit.gates` typed surface、legacy dict 互操作、JSON/QASM round-trip、metrics/transpile、typed `Observable`、`qml.auto`/`qml.ad`/`psr`/`fd` deriv 路径；`qaoa_probe.py` 覆盖 gate-level QAOA 的 `Hamiltonian` 输入、Trotter order 1/2、exact energy、diagonal sampling 和短优化路径。
- **多卡 NPU strict probe。** 新增 `scripts/npu/multi_card.sh`（包装 `python -m torch.distributed.run`）与 `multi_card_probe.py`：验证 HCCL 初始化、rank→NPU 绑定、`shard_context`、object `all_gather`、实数 NPU tensor 的 `broadcast_parameters`/`all_reduce_mean`，并在每个 rank 上跑 typed IR/deriv 小图与 supernet `safe`/`aggressive` 小规模多卡任务。设备绑定对齐 `demos/BeH2/BeH2_npu.py`：`LOCAL_RANK -> npu:{LOCAL_RANK}`。
- **4 卡 NPU QNN demo。** 新增 `demos/demo_npu_qnn_4card.py` 与 `scripts/npu/qnn_4card.sh`：用 typed `Operation` 构造 2-qubit teacher/student QNN，在 `torchrun` 4 rank 下做数据并行训练；每个 rank 在本地 NPU 上完成 statevector 前向与 `NPUBackend.expectation_sv` 反传，只通过 HCCL 同步实数参数/梯度。该 demo 验证 typed IR + autodiff + 多卡实数 collective 的实际训练路径，不宣称单个 statevector 已被切分到多 NPU。
- **8 卡 NPU QAOA probe。** 新增 `scripts/npu/qaoa_8card.sh`、`run_qaoa_8card.py` 与 `qaoa_8card_probe.py`：用 `torchrun --nproc_per_node 8` 验证 rank-local gate-level `BasicQAOA`、diagonal sampling、非对角 `trotter_order=1/2` exact energy、rank 0 参数广播、有限差分 QAOA 梯度的 HCCL 实数均值同步，以及 synthetic problem shards 覆盖检查。该 probe 不宣称单个 statevector 已被切分到 8 张 NPU。
- **NPU suite 目标漂移守卫。** `tests/scripts/test_npu_test_runner.py` 现在校验每个 suite 的 pytest target 与 probe script 都真实存在，避免文件改名后 `run_all.sh` 在远端 NPU 跑到中后段才因 `file or directory not found` 失败。

### Fixed

- **Ascend NPU complex64 限制导致的 typed-IR deriv/backend 问题。** 本轮迁移确认真实 NPU 上不能把 CPU/GPU 的任意 complex64 torch 图视为等价契约；`NPUBackend` 路径必须继续经 real/imag 分解或自定义 autograd，避免 `aclnnMatmul` / `aclnnInplaceAdd` / `aclnnAdd(DT_COMPLEX64)` 等不支持路径。相关说明已写入项目记忆/文档；真实 NPU deriv 以 `typed_ir_deriv_probe.sh --section deriv` 和 `scripts/npu/deriv.sh --strict-npu` 为准，通用 full-matrix complex autograd 仅保留 CPU/fallback 合同。
- **`NPUBackend.inner_product` 的共轭内积语义。** 远端 NPU 测试暴露 `torch.dot(torch.conj(bra), ket)` 期望与 workaround 结果不一致；修正后 NPU-safe 路径按 `<bra|ket>` 语义计算，且测试避免把 unsupported `torch.dot` 当作真实 NPU workaround 的实现依赖。
- **`gloo` 分布式测试遇到 NPU tensor。** `tests/test_supernet_sharding_dist.py` 是 CPU-gloo reproducibility 测试，但真实 NPU 环境下 supernet 参数/梯度会落在 `npu` tensor 上，直接 `dist.broadcast` / `dist.all_reduce` 触发 `RuntimeError: No backend type associated with device type npu`。`aicir.qas.core.sharding` 现仅在「非 CPU tensor + 非 HCCL process group」时 CPU staging 后 collective，再 copy 回原设备；真实多 NPU HCCL 路径仍走原生 NPU collective。
- **多卡 NPU probe 设备选择对齐 BeH2 demo。** 远端验证表明在该 Ascend runtime 上显式设置 `ASCEND_RT_VISIBLE_DEVICES=0,5,6,7` 会让 `torch.npu.set_device(...)` 同时拒绝逻辑 ID（`npu:1/2/3`）和物理 ID（`npu:5/6/7`）。因此 `multi_card.sh` 不再导出 Ascend visible-device 变量，`--devices` 仅保留为兼容 no-op；多卡验证应像 BeH2 一样只用 `torchrun --nproc_per_node=N`，由 `LOCAL_RANK` 绑定 `npu:{LOCAL_RANK}`。
- **NPU runner 文件名漂移。** `circuit` suite 中旧目标 `tests/circuit/test_typed_gates_api.py` 更新为实际文件 `tests/circuit/test_circuit_typed_gates_api.py`，并由 suite target 存在性测试防回退。

### Validation

- 真实 Ascend NPU 上已完成单卡 strict sweep：`typed_ir_deriv_probe.sh --section all`、`typed_ir.sh --strict-npu --pytest-arg -q`、`deriv.sh --strict-npu --pytest-arg -q`、`backend.sh --strict-npu --pytest-arg -q`、`capacity.sh --strict-npu --pytest-arg -q` 均通过。
- 真实 Ascend NPU 上已完成 QAOA strict suite：`scripts/npu/qaoa.sh --strict-npu --pytest-arg -q` 通过，`qaoa_probe.py` 在 `NPUBackend(dtype=torch.complex64, device=npu:0, npu_available=True)` 上完成 diagonal gate-level energy/sampling、非对角 `trotter_order=1` exact energy、非对角 `trotter_order=2` exact energy 和短 `BasicQAOA.run(...)`；随后 `tests/vqc/test_qaoa_canonical.py`、`tests/vqc/test_qaoa_qfun.py`、`tests/optimization/qubo/test_qaoa_helpers.py` 共 25 项通过。
- 真实 Ascend NPU 上 `scripts/npu/run_all.sh --strict-npu --fail-fast --pytest-arg -q --pytest-arg --durations=20` 已从 `smoke` 跑到 `demos` 全套完成，无 failure/error；覆盖 typed IR、Circuit.gates API、deriv、NPUBackend workaround、capacity/sharding、QML、tensor simulator、QAS 和 demos。当前结论是：typed `Circuit.gates` API 迁移在单卡真实 NPU 上通过全量 NPU 脚本验证。
- 真实 4 卡 Ascend NPU / HCCL 上已完成多卡 strict probe：`scripts/npu/multi_card.sh --nproc-per-node 4 --section collectives` 通过 rank→`npu:0..3` 绑定、HCCL 初始化、`shard_context`、object `all_gather`、实数 NPU tensor `broadcast_parameters` 与 `all_reduce_mean`；`scripts/npu/multi_card.sh --nproc-per-node 4 --section all` 进一步通过每 rank typed IR、deriv，以及 supernet `safe`/`aggressive` 小规模分片。supernet safe 四 rank 能量一致为 `-1.0197104215621948`，aggressive 四 rank 能量一致为 `-0.9999998211860657`。
- 真实 8 卡 Ascend NPU / HCCL 上已完成 QAOA 多卡 probe：`scripts/npu/qaoa_8card.sh --nproc-per-node 8 --samples 16 --steps 20` 通过，rank→`npu:0..7` 绑定、`WORLD_SIZE=8`、backend=`hccl`；16 个 synthetic QAOA problem shards 被 8 个 rank 各自持有 2 个并完整覆盖，rank-local gate-level `BasicQAOA` 跑通 diagonal sampling、非对角 `trotter_order=1/2` exact energy、rank 0 参数广播和 HCCL 实数有限差分梯度平均。全局 loss 从 `0.0082067298` 降到 `-0.5388664603`，最终梯度范数为 `0.0024492152`。该 probe 验证的是 rank-local QAOA + HCCL 实数同步，不宣称单个 statevector 已被切分到 8 张 NPU。
- 真实 4 卡 Ascend NPU / HCCL 上已完成 QNN typed-IR 训练 demo：`scripts/npu/qnn_4card.sh --nproc-per-node 4` 通过，rank→`npu:0..3`、backend=`hccl`，32 samples / 12 steps 的 mean loss 从 `0.0357317636` 降到 `0.0031882407`；`scripts/npu/qnn_4card.sh --nproc-per-node 4 --steps 24 --samples 64` 通过，64 samples / 24 steps 的 mean loss 从 `0.0357317654` 降到 `0.0021315088`。这验证了 typed `Operation` QNN、rank-local NPU statevector 前向、`NPUBackend.expectation_sv` autodiff，以及 HCCL 实数梯度平均可在 4 卡真实 NPU 上闭环运行。

## 2026-07-03

### Added

- **`aicir.simulator` 接入 cotengra（新 `tn` extra）**：`optimize="auto"|"cotengra"|"opt_einsum"|"greedy"` 选择收缩路径来源；`memory_limit=` 设定中间张量内存预算后由 cotengra 规划切片，执行侧逐切片固定指标（新 backend 原语 `take`）、同一 pairwise `tensordot` 收缩、`add` 累加——NPU 复数分解与 torch autograd 全程保留（`_NpuTakeFn`/`_NpuAddFn` 规避 aclnnAdd DT_COMPLEX64）。四个公共函数（`tn_statevector`/`single_amplitude`/`partial_amplitude`/`tn_expectation`）透传两参数。
- **经典控制流（if/while）：`ClassicalRegister`/`Bit`/`Condition`（`aicir.core.classical`）+ `measure(..., creg=/cbits=)` + `if_`/`while_`（`aicir.core.circuit`）。** `ClassicalRegister(size, name)` 的位（`reg[i]`）与整寄存器（`reg`）都支持 `==`/`!=` 构造 `Condition`；`measure(qubits, creg=reg)` 按序、`measure(qubits, cbits=[...])` 显式指定，把 Z 基投影结果写入经典位（有经典目标时仅支持 Z 基，`creg`/`cbits` 互斥）。`if_(condition, body, else_body=None)` 与 `while_(condition, body, *, max_iterations)`（`max_iterations` 必填，超限仍满足条件抛 `RuntimeError`）产出 `ControlFlow` 指令（`aicir.ir.control_flow`），`body`/`else_body` 的 `n_qubits` 须与外层一致。控制流只走 `Measure.run` 测量轨迹路径逐轨迹求值/递归执行；`Circuit.unitary()` 与张量网络引擎（`aicir.simulator`）遇到 `ControlFlow` 一律抛 `ValueError`，QASM3 导出推迟。`Result.classical_counts(reg)` 统计各轨迹末尾经典寄存器整数取值分布（LSB=`reg[0]`，从未写入的轨迹计 0）。含 `ControlFlow`/`Condition` 的电路 JSON 序列化往返后执行结果与原电路一致。

### Changed

- **Breaking:** `aicir.vqc.ansatz` moved to `aicir.ansatze` (top-level package). `hea`/`hea_parameter_count`/`hea_ti`/`hea_ti_parameter_count`/`uccsd`/`uccsd_parameter_count`/`entangling_edges`/`hardware_efficient_ansatz`/`power_law_couplings` now import from `aicir.ansatze`, not `aicir.vqc.ansatz`. `aicir.vqc` no longer re-exports `ansatz`. Reason: ansatz has no dependency on `vqc` and is already consumed by `aicir.qas`, `aicir.optimization.qubo`, and `aicir.chemistry` (via decoupled data) — nesting it under `vqc` implied a coupling that didn't exist. No backward-compatible alias (see CLAUDE.md's "old long aliases are intentionally not kept" convention).

## 2026-07-02

### Added

- **`aicir.chemistry.build_molecule`：电子结构现算流水线（`chem` extra：`qiskit-nature` +
  `pyscf`），与固定预置并列。** 给定任意分子几何/基组/映射（`jordan_wigner`/`parity`/
  `bravyi_kitaev`，可选 `active_electrons`/`active_orbitals` 做 active-space 裁剪），驱动
  `PySCFDriver` + Qiskit Nature mapper 现算 qubit Hamiltonian，返回与预置同构的
  `MoleculeHamiltonian`。仅 Jordan-Wigner 映射额外填充 `n_electrons`/`hf_occupation`/
  `excitations` 三个字段（HF 参考态占据 + singles/doubles 费米子激发，qubit 索引与
  `terms` 同一比特序），用于桥接 `aicir.vqc.ansatz.uccsd`。未安装 `chem` extra 时抛
  `ImportError` 并提示安装命令，不影响核心 `numpy`-only 依赖。配套
  `tests/chemistry/test_pipeline.py`、`test_pipeline_guard.py`、`test_molecule_metadata.py`
  与 `aicir/chemistry/README.md` §3。
- **`aicir.vqc.ansatz.uccsd`：UCCSD 化学 ansatz 模板，吃纯数据、与 `aicir.chemistry`
  解耦。** `uccsd(n_qubits, hf_occupation, excitations, reps=1, ...)` 用 HF 占据位铺
  `pauli_x` 参考态，再按激发列表逐个施加单/双激发门；非相邻 orbital 间的激发通过
  fSWAP 网络（`ansatz/_excitation.py`：`fswap_ops`/`single_excitation_ops`/
  `double_excitation_ops`）精确实现，双激发的创生/湮灭对角色由参数*位置*（而非数值
  大小）决定，正确处理 HF 占据/未占据在比特序上交错分布的情形（如 H2 4-qubit JW：
  占据 {1,3}、未占据 {0,2}）。参数顺序为先 reps 外层、后激发内层。配套
  `uccsd_parameter_count`、`tests/vqc/test_uccsd_ansatz.py`、
  `tests/vqc/test_excitation_circuits.py`（JW 生成元 expm oracle，覆盖交错角色配对）、
  以及端到端集成测试 `tests/vqc/test_uccsd_vqe_integration.py`（`build_molecule` →
  `uccsd` → `BasicVQE`，H2 收敛到基态能量，VQE 能量与精确对角化基态相差 ~1e-7）。
- **`aicir.qml.QLayer`：把 `QFun` 封装成 `torch.nn.Module` 量子层，可一行嵌入
  PyTorch 混合网络。** 前向调用 `qfun(params)`、反向调用 `qfun.grad(params)`（参数移位
  Jacobian）接入 torch autograd，与 `QFun` 后端解耦（`device="numpy"/"gpu"/"npu"` 皆可），
  梯度方法仍走 `aicir.qml.diff` 注册表单一真源。经典输入与可训练权重经 `torch.cat` 拼成
  单参数向量喂给 `qfun`，梯度同时回流到前置经典层与本层权重；支持批量输入与多观测量输出。
  `torch` 为可选依赖，缺失时 `aicir.qml.QLayer is None`。配套 `tests/qml/test_qlayer.py`
  与 `aicir/qml/README.md` §17。
- **`aicir.simulator` 精确张量网络模拟引擎：`tn_statevector` / `single_amplitude` /
  `partial_amplitude` / `tn_expectation`，并为 `Measure.run` 增加 `method="tensor"`。**
  收缩建立在新的 `Backend` 原语（`tensordot/transpose/reshape/conj`）之上，NPU 的
  `tensordot` 复用 autograd-safe 复数 matmul（real/imag 分解），期望值在 torch/NPU 后端
  可微；收缩路径用 opt_einsum（可选）或内置贪心。仅纯态、无噪声。配套
  `demos/demo_npu_tensor.py`（远程 NPU 验证）与 `aicir/simulator/README.md`。MPS 截断另立 Spec 2。

## 2026-07-01

### Added

- `split_cores`（Billionnet-Jaumard 分解）：将二次 QUBO 经 posiform + 蕴含图 + Tarjan SCC 拆分为相互独立的 hard core，`min f == Σ min f_i`，可细分连通块。
- **`aicir.chemistry` 改为每分子一个模块的 `molecules/` 包（公共 API 不变）。**
  `chemistry/molecule.py` 拆为 `chemistry/molecules/`：`_base.py`（`MoleculeHamiltonian`
  数据类 + `MOLECULES` 注册表 + `register_molecule` + 访问器），每个分子一个自注册模块，
  **文件名用分子式大小写**（`H2.py`/`LiH.py`/`H2O.py`/`NH3.py`/`N2.py`/`BeH2.py`），
  canonical 名称统一小写。新增预置：`lih`(4q,2e/2o)、`h2o`(6q,4e/3o)、`nh3`(12q,6e/6o)、
  `n2`(14q,10e/7o)、`beh2`(16q,6e/8o,3-21G)，系数取自各自 demo 的 PySCF/Qiskit Nature 结果。
  公共接口（`get_molecule`/`molecule_hamiltonian`/`molecule_matrix`/`available_molecules`/…
  从 `aicir.chemistry` 导入）完全不变，另新增 `register_molecule` 与各分子常量。配套
  `tests/chemistry/test_molecules.py`：≤6 qubit 小分子有 dense-matrix 基态能量守卫
  （`h2o` 对上 `-6.1596636772`）；12–16 qubit 的 nh3/n2/beh2（dense 构造过慢/过大）走
  结构守卫，系数由上游 PySCF/Qiskit Nature 保证。

### Changed

- **GateSpec.matrix Approach A：门矩阵两条路径统一经注册表分发（分支 `gatespec`）。**
  `aicir.core.gates` 的 `gate_to_matrix`（全矩阵）与 `apply_gate_to_state`（局部）不再
  各自硬编码 if/elif 分发链，改为经新增 helper `_gate_local_matrix(gate)` 从
  `GateSpec.matrix` 注册表取局部矩阵（受控门取底门局部矩阵，`_CONTROLLED_BASE_GATE`
  映射 + `_controlled_local_from_base` 包裹），再分别经 `_expand_local_matrix_to_full` /
  `_apply_local_matrix_to_state` 施加。门局部矩阵**唯一来源**即注册表。
  - `gate_to_matrix` 从 ~260 行双路（numpy/backend）折叠为 ~10 行。
  - **新能力**：自定义**不受控**门注册 `matrix=` 后现在也能走快速局部路径
    （`apply_gate_to_state` 旧实现对未硬编码门返回 `None`，只能回退全矩阵）。
  - 移除因此产生的 14 个死构造器（`_hadamard`/`_crx`/`_cy`/`_cz`/`_crz`/`_toffoli`/
    `_swap_backend`/`_single_qubit_from_base_backend` 等）。
  - 行为保持：全量既有测试通过（VQE 能量、Bell 态、QASM round-trip 均经此路径）。
  - 配套 `tests/gates/test_matrix_dispatch_consistency.py`、`test_matrix_autograd.py`、
    `test_custom_gate_dispatch.py`（均按 numpy/torch/**npu** 后端参数化）。
  - **NPU 平台验证**：三个测试文件在 Ascend NPU 上 `135 passed`（`npu` 参数化分支实际
    命中设备，非 CPU 回退）——两路一致、autograd 对有限差分、自定义门两路一致均在真实
    NPU 后端通过。
  - **收尾：受控自定义门 + 多比特目标受控门接入。** `_gate_local_matrix` 改为「任意
    携带 `control_qubits` 的门」都走受控路径，底门名取 `_CONTROLLED_BASE_GATE.get(type, type)`
    （内置受控门映射到不同底门，自定义门底门即自身），目标轴由 `_gate_axes(gate)` 取得
    （`axes = controls + target_axes`）。`_controlled_local_from_base` 泛化为接受
    `2^k × 2^k` 底门（`k` 个目标比特），底门维度须为 2 的幂且与目标比特数一致。因此
    `{"type": <自定义门>, "control_qubits": [...]}`、受控 `swap`（Fredkin/CSWAP）、受控
    自定义 2 比特门现在两条路径都能模拟。配套 `test_custom_gate_dispatch.py` 增加受控
    单比特 / 受控 swap / 受控自定义 2 比特门用例。

### Fixed

- **`tests/circuit/io/test_json_qasm_io.py` torch 未导入。**
  `test_json_serializes_torch_tensor_parameters_as_numeric_values` 直接用 `torch.tensor(...)`
  但文件未 import torch（`NameError`，主分支上一直失败）。改为顶层 `try: import torch`
  守卫 + `@unittest.skipIf(torch is None, ...)`；有 torch 时运行、无 torch 时干净跳过。

## 2026-06-30

### Added

- **DQAS / Differentiable Quantum Architecture Search 接入统一 QAS 入口。**
  新增 `aicir/qas/algorithms/dqas.py`：按 Zhang et al. (2022) DQAS 的 independent categorical
  概率模型采样线路结构，维护 `alpha[p, c]` 架构 logits 与 `theta[p, c, l]` 参数池，
  用 score-function/REINFORCE 估计架构梯度，并用 autograd 更新线路参数。公共入口新增
  `config.dqas(...)`、`run("dqas", hamiltonian=..., config=...)`，并导出 `DQASConfig`、
  `DQASResult`、`DifferentiableQAS`、`dqas`、`train_dqas`。`dqas` 已注册为
  `DQASStrategy`（`requires_torch=True`）。配套 `tests/algorithms/test_dqas.py`。
- **DQAS 搜索门池 API：`gate_pool` / `pool` / legacy `operation_pool`。**
  `DQASConfig` 新增 `gate_pool`（默认 `"generic"`）、`two_qubit_pairs`、`pool` 别名，并保留
  `operation_pool` 兼容别名。支持门名 `identity`、`rx`、`ry`、`rz`、`rzryrz`、`cx`；
  `"generic"` 等价于 `("identity", "rzryrz", "cx")`。`cx` 默认展开为所有有向非自环连接，
  也可通过 `two_qubit_pairs=((control, target), ...)` 限制。tuple/list 保持用户顺序；
  set 形式会按固定门序规范化以避免 sampled index 不可复现。另新增
  `gate_pool="excitation"`：全局 operation pool 展开为 identity、给定
  `single_excitations`、给定 `double_excitations`，并支持 `hf_occupied_qubits`
  前置 Hartree-Fock 初态制备。
- **H2O DQAS NPU demo。**
  新增 `demos/H2O/H2O_dqas.py`，默认 `device="npu:0"`，在 H2O 6-qubit active-space
  Hamiltonian 上用 DQAS + excitation pool 搜索基态 ansatz，写出文本报告、NPU 命名的
  `H2O_dqas_npu_cir.py` 和 `H2O_dqas_npu_cir.png`。脚本支持 `--device cpu` dry run 以便
  在无 Ascend 环境验证搜索、保存和绘图路径。
- **transpile merge-rotations 支持 excitation 门。** `single_excitation`/`double_excitation`
  为固定生成元的旋转门，角度可加（`G(θ1)·G(θ2)=G(θ1+θ2)`）；`MergeRotationsPass`/
  `optimize` 现把相邻、同操作数（同顺序）的两个 excitation 门按角度相加合并，角度抵消
  为 0 时整对消去。配套 `tests/transpile/test_excitation_merge.py`。
- **QDRATS 可配置门池 `gate_pool`（支持 excitation ansatz）。**
  `QDRATSConfig` 新增 `gate_pool`（`"generic"` 默认 / `"excitation"`）、`single_excitations`、
  `double_excitations`、`hf_occupied_qubits`。QuantumDARTS 重构为 slot/candidate 抽象：
  - `"generic"`（默认）：每个 target 比特一个 slot，候选 `{rz·ry·rz, identity, cx_*}`，
    张量形状/RNG 与旧实现一致，**数值不变**。
  - `"excitation"`：从 closed-shell HF 参考态出发，每个激发算符一个 slot，候选
    `{single_excitation/double_excitation(θ), identity}`，离散化线路前置 HF 的 `pauli_x`
    制备。与 supernet 同款粒子数/自旋保持 ansatz，改由 QDRATS 可微搜索。
  - 配套 `tests/algorithms/test_qdrats.py`（generic 向后兼容 + excitation 行为）。
  - demo `demos/H2O/H2O_qdrats.py` 切换为 excitation 池（复用 H2O singles/doubles/HF）。
- **QAS `SearchStrategy` 协议 + 策略注册表（QAS README §2.1 落地）。**
  新增 `aicir.qas.core.strategy.SearchStrategy`（抽象基类，`run(request) -> 结果`）、
  `aicir.qas.core.registry`（`StrategySpec` + `register_strategy`/`unregister_strategy`/
  `get_strategy`/`get_spec`/`registered_strategies`，均从 `aicir.qas.core` 再导出）、
  `aicir.qas.core.strategies`（内置策略适配并注册，import 副作用）。`run(method, ...)`
  先查注册表命中则走 `strategy.run`，未命中回落到 `runner` 的 `_Spec` 分发表。当前
  `supernet` 迁移为 `SupernetStrategy`，`dqas` 迁移为 `DQASStrategy`（均
  `requires_torch=True`），其余方法行为不变；`run("supernet", ...)` 调用方式与返回值
  完全不变。配套 `tests/qas/test_strategy_registry.py`。
- **`LayoutPass` 自动布局 `initial_layout="auto"`（NEXT.md §2 transpile 硬化）。**
  按双比特门交互频率降序，贪心地把高频交互的逻辑对放到耦合图相邻的物理比特上，
  减少后续 `RoutingPass` 的 SWAP。需 `target`；全连接时退为恒等。贪心启发式（非全局最优）。
  配套 `tests/transpile/test_layout_pass.py`。
- **`DecomposePass` 单比特 ZYZ 基底翻译（NEXT.md §2 transpile 硬化）。**
  基底含 `rz` 与 `ry` 时，任意不受控单比特门经 `GateSpec.matrix` 取 2x2 矩阵、
  Euler 分解为 `rz·ry·rz`（等价**至全局相位**）。基底不含 `rz`/`ry` 时按旧行为原样保留。
  配套 `tests/transpile/test_decompose_pass.py`。

### Changed

- **`RoutingPass` 升级为置换跟踪路由（NEXT.md §2 transpile 硬化）。**
  - 插入 SWAP 后**不再复位**：把由路由产生的比特置换向前携带，后续门（含单比特门）
    按当前位置重新映射。整线路与原线路等价**至最终比特置换**，SWAP 数较旧“插入-复位”
    方案大致减半，且相邻化的比特对在后续门上无需再插 SWAP（跨门复用）。
  - 新增运行后属性 `final_layout`（`logical -> physical wire`，覆盖全部物理线，未移动者恒等）
    与 `last_layout`（镜像，恒等时 `None`）。
  - `PassManager.run_with_result` 现按 pass 顺序**组合**各 `last_layout`，使
    `LayoutPass → RoutingPass` 链式得到 `composed[q] = routing[layout[q]]` 并记入
    `TranspileResult.layout`。
  - **接口变化**：路由输出不再与原线路完全幺正等价，而是等价至 `final_layout` 置换；
    读测量结果时需据此还原比特顺序。
  - 配套 `tests/transpile/test_routing_pass.py`、`tests/transpile/test_transpile_result.py`。

## 2026-06-29

### Added

- **`TranspileResult` 统一结果对象 + `PassManager.run_with_result`（NEXT.md §9 收尾）。**

  - `aicir.transpile.TranspileResult`（`aicir.transpile.result`）：字段 `circuit`/`layout`/`passes`/
    `depth_before`/`depth_after`/`metadata`；置于 transpile 域内，避免 primitives↔transpile 耦合。
  - `PassManager.run_with_result(circuit)`：运行流水线并返回 `TranspileResult`，深度经
    `depth_proxy`（ASAP 层数），`layout` 取自含 `last_layout` 的 pass（如 `LayoutPass`，无则 `None`）。
    `LayoutPass.run` 现记录 `last_layout`。`PassManager.run` 仍返回 `Circuit` 不变。
  - 配套 `tests/transpile/test_transpile_result.py`。
- **qfun 测量返回构造器 `expval`/`probs`/`sample`（NEXT.md §5 收尾）。** 函数体可返回这些对象在体内
  声明测量意图（无全局 tape，显式携带 circuit），替代/补充装饰器 `observable=`：

  - `expval(circuit, observable)` → 期望值（唯一可微返回）；`probs(circuit, wires=None)` → 概率向量
    （`wires` 边缘化）；`sample(circuit, wires=None)` → counts（shots 取自装饰器，缺省抛 `ValueError`）。
  - `.grad` 仅对 expval 有效；probs/sample 调用 `.grad` 抛 `ValueError`。均从 `aicir.qml` 导出。
  - 配套 `tests/qfun/test_qfun.py`。
- **`GateSpec.matrix` 字段 + `gate_matrix()` 访问器 + `gate_to_matrix` 自定义门回退（NEXT.md §7 收尾）。**

  - `GateSpec.matrix`：后端感知局部矩阵构造器 `(params, backend) -> 局部矩阵`，已为全部不受控标准门
    填充（复用 core 局部矩阵原语，于 `aicir.core.gates` 导入时附加）；受控门、`measure`/`reset`/`unitary` 为 `None`。
  - `aicir.gates.gate_matrix(name, params=(), backend=None)` 读取局部矩阵；`set_gate_matrix` 附加构造器。
  - `gate_to_matrix` 未硬编码门的回退经 `GateSpec.matrix` 构造局部矩阵并嵌入，使自定义**不受控**门注册
    `matrix=` 后即可模拟，无需改 core。内置门分发与 autograd/local-apply 热路径不变。
  - 配套 `tests/gates/test_gatespec_matrix.py`（含 numpy/torch 一致性漂移护栏）。
- **粒子数守恒激发门 `single_excitation`（别名 `givens`）与 `double_excitation`。**

  - `single_excitation(θ, qubit_1, qubit_2)`：实 Givens 旋转，作用于 |01⟩↔|10⟩ 子空间，
    保持粒子数守恒；`givens` 为规范别名。`GateSpec.generator=None`、`qasm_name=None`、
    `shift_rule=None`（两参数移位需配合 `psr4`）。
  - `double_excitation(θ, q0, q1, q2, q3)`：四比特粒子数守恒激发门，作用于 |0011⟩↔|1100⟩
    子空间；`GateSpec.generator=None`、`qasm_name=None`、`shift_rule="four_term"`。
  - 两门均在顶层 `aicir` 导出。
  - 配套 `tests/gates/test_excitation_gates.py`。
- **`aicir.qml.deriv.psr4`：四项参数移位规则（激发门 {−1, 0, 1} 特征谱）。**

  - 移位点 ±π/2、±3π/2，权重按四项规则推导，适用于 `single_excitation`/`double_excitation`
    等特征值谱为 {−1, 0, 1} 的生成元。
  - `psr4` 已加入 `aicir.qml.deriv.__all__`。
  - 配套 `tests/qml/test_psr4.py`。
- **`GateSpec.num_controls`：受控门的控制位数量作为注册表数据。**

  - `aicir.gates.GateSpec` 新增 `num_controls: int = 0`（受控门的控制位数量）。
    内置门：`cx`/`cy`/`cz`/`crx`/`cry`/`crz`→`1`、`toffoli`→`2`、其余→`0`；
    对所有注册门满足不变量 `controlled == (num_controls > 0)`。加性、非破坏。
  - 配套 `tests/gates/test_gatespec_num_controls.py`。

### Changed

- **`BasicVQE` 默认精确能量改经 `StatevectorEstimator` primitive（NEXT.md §4 phase-1 item 4 收尾）。**
  未显式注入 `energy_estimator`、|0⟩ 起点精确态向量（无 shots/噪声/密度矩阵/自定义初始态、非
  `return_state`）时，能量经新增 `BasicVQE._default_estimator()`→`StatevectorEstimator` 求值，与注入路径
  统一。数值与原 `Measure` 精确路径一致。shots/噪声/密度矩阵/初始态等仍退回 `Measure`（primitives 仍可
  经 `energy_estimator=` 注入或 `target=` 选中）；`BasicQAOA`（稠密无线路）不接入。配套
  `tests/vqc/test_vqe_orchestration.py`。
- **qfun `observable=` 不再于装饰期强制。** 函数体可经 `expval(...)` 自带观测量；"缺少 observable" 的错误
  下移到**调用期**，且仅在返回裸 `Circuit` 且无装饰器 `observable=` 时触发（向后兼容现有用法）。
- **`aicir.qas` supernet 改用 aicir 门，移除模块内自定义门定义。**

  - `aicir/qas/algorithms/supernet.py` 删除其本地 `GateSpec` 数据类与
    `_SINGLE_QUBIT_GATES` / `_TWO_QUBIT_GATES` builder 字典；改为按 `aicir.gates`
    注册表校验 token、读取 `num_params` / `num_controls`，并经 `aicir.core` 门工厂
    （`hadamard`/`rx`/`ry`/`rz`/`cx`/`rzz`）构造门。双比特门控制/目标拆分由注册表
    `num_controls` 驱动，不再硬编码。
  - 纯内部重构、行为保持：发出的门字典逐字节不变（门名、比特字段、`control_states`、
    参数位置）；搜索空间 token（`i`/`h`）与公开配置面不变。
  - 配套 `tests/test_supernet_gate_pool.py`（行为锁定特征测试）。

## 2026-06-27

### Added

- **`select_diff` 接入 Estimator primitives：`BaseEstimator.gradient(...)`（NEXT.md §6 / QML todo 2.2）。**
  - 新增 `BaseEstimator.gradient(circuit, observable, *, parameter_values, shots=None, method="auto")`，
    以 estimator 自身执行路径为目标函数 `params -> <H>`，经 `aicir.qml.diff` 注册表分发梯度规则：
    `method="auto"` 时调用 `select_diff(backend, shots, noisy)` 自动优选——不支持 Torch 后端、
    带 shots 或带噪声时降级到 `psr`/`fd`；其余按名经 `resolve_diff` 解析。
  - `NoisyEstimator` 置 `_noisy=True`，使噪声路径选到 `supports_noise` 的方法。
  - 新增统一结果对象 `aicir.primitives.GradientResult`（NEXT.md §9，含 `gradient`/`method`/
    `nfev`/`metadata`）。
  - 配套 `tests/primitives/test_estimator_gradient.py`。
- **`GateSpec` 元数据扩充：`generator` / `decomposition` 字段（NEXT.md §7）。**
  - `aicir.gates.GateSpec` 新增 `generator`（单参数旋转门的 Pauli 生成元标签，
    `U = exp(-i θ G / 2)`：`rx`→`"X"`、`ry`→`"Y"`、`rz`→`"Z"`、`crx/cry/crz` 记目标位
    生成元、`rzz`→`"ZZ"`、`rxx`→`"XX"`）与 `decomposition`（分解规则可调用，签名
    `(qubits, controls, control_states, params) -> list[dict] | None`）。
  - 新增注册表 helper `gate_generator` / `parametric_pauli_gates` / `gate_decomposition`，
    从 `aicir.gates` 导出；分解规则置于自包含的 `aicir/gates/decompositions.py`
    （仅构造纯门字典，避开 `gates`↔`ir` 循环导入）。
  - 配套 `tests/gates/test_gatespec_metadata.py`。
- **`Target` 接入 primitives / vqc / metrics（NEXT.md §3）。**
  - `aicir.primitives.estimator_for_target(target, *, backend=None, noise_model=None, shots=None)`
    按 `Target` 能力选择 `Statevector`/`Shot`/`Noisy` 估计器；无可用执行路径抛 `ValueError`。
  - `aicir.metrics.HardwareProfile.from_target(target, **overrides)` 从 Target 取
    `native_gates`（空门集回退 `DEFAULT_NATIVE_GATES`）与 `coupling_map`。
  - `aicir.primitives.StatevectorEstimator` 新增 `estimate()` 直通方法，满足
    `BasicVQE(energy_estimator=...)` 注入契约。
  - 配套 `tests/devices/test_target_integration.py`。
- **`BasicVQE(..., target=Target(...))`：按设备能力注入 Estimator（NEXT.md §4 phase-1 item 4）。**
  - 未显式注入 `energy_estimator` 时，经 `estimator_for_target` 注入估计器，使能量
    求值走 primitives；显式 `energy_estimator` 优先。`BasicQAOA` 默认路径为稠密线性
    代数（无线路），不在本次范围。
  - 配套 `tests/vqc/test_vqe_target.py`。

### Changed

- **`DecomposePass` 改为 `GateSpec.decomposition` 字段驱动。**
  - `aicir.transpile.DecomposePass` 的分解规则从注册表 `gate_decomposition` 读取，
    不再硬编码；注册自定义门携带 `decomposition` 即被自动识别。行为对内置
    `swap`/`cz`/`cy` 不变。
- **`aicir.qml.deriv` 可微门集改为从 `GateSpec` 自省。**
  - `_AD_PAULI_GENERATOR` / `_AD_DIFFERENTIABLE` 改为从 `parametric_pauli_gates()` /
    `gate_generator()` 派生（与旧硬编码值逐一致），注册新 Pauli 旋转门即自动可伴随微分。

## 2026-06-23

### Added

- **QDRATS / QuantumDARTS 宏观量子架构搜索。**
  - 新增 `aicir/qas/algorithms/qdrats.py`：实现 Wu et al. (ICML 2023) QuantumDARTS
    宏观搜索流程，使用 Gumbel-Softmax 在每个 qubit-layer 位置采样真实量子门，
    交替优化架构权重与 `Rz-Ry-Rz` 旋转参数，最终离散化为 `aicir.Circuit` 并微调参数。
  - 候选门集按论文设定展开为 `RzRyRz`、identity、以及每个目标位对应的
    `cx(control -> target)` 控制位变体；二比特门按“当前位置 qubit 为 target，
    subscript/候选项决定 control”的规则生成。
  - 配置放入 `aicir/qas/core/config.py`：新增 `QDRATSConfig`、`config.qdrats(...)`，
    并支持 `qdrats` / `qdarts` / `quantumdarts` / `quantum_darts` 方法别名。
  - 公共入口接入 `aicir.qas.run("qdrats", hamiltonian=..., config=...)`，
    并导出 `QDRATSConfig`、`QDRATSResult`、`QuantumDARTS`、`qdrats`、`train_qdrats`。
  - 配套 `tests/algorithms/test_qdrats.py` 与 runner 方法列表测试；已验证
    `pytest -q tests/algorithms/test_qdrats.py tests/test_qas_runner.py`。

### Changed

- **QAS 内置策略适配迁移到 `aicir.qas.core.strategies`。**
  - 原 `aicir/qas/algorithms/strategies.py` 只做 `SearchStrategy` 适配与注册，不是具体算法；
    现移至 `aicir/qas/core/strategies.py`，与 `core/strategy.py`、`core/registry.py`、
    `core/runner.py` 同层。
  - `runner.py` 改为导入 `aicir.qas.core.strategies` 触发内置策略注册；`SupernetStrategy`
    内部懒导入 `train_supernet`，避免 `core` 模块顶层过早加载具体算法。
  - README 与策略注册表测试同步更新。

## 2026-06-22

### Added

- **NPU 局部量子门 autograd 安全路径：`_NpuLocalGateApplyFn` + `NPUBackend.apply_flat_gate`。**

  - 根因：`_apply_local_matrix_to_state_flat`（n_qubits > 8 时 NPU 路径）在 npu_complex 分支中：
    `flat`（complex64）被 `.real` 和 `.imag` 各读一次（2 条 backward 路径），
    `updated = matmul(M, gathered)` 同样被 `.real`/`.imag` 各读一次（再 2 条）。
    backward 累加 complex64 梯度 → `aclnnAdd(DT_COMPLEX64)` → Ascend 崩溃。
    BeH2 用 16 比特 > 8，走 flat 路径，故 `--gradient ad` 经过 `_NpuHamiltonianExpectationFn`
    修复后仍在 `loss.backward()` 崩溃（第二处 fan-out）。
  - 修复：`_NpuLocalGateApplyFn`（`aicir/backends/npu_backend.py`）把 gather→matmul→scatter
    包成单一 `torch.autograd.Function`；`flat` 和 `local_matrix` 各只出现一次；
    backward 全程 float32 实/虚部运算，最后 `torch.complex(...)` 一次性构造——无 complex64 add。
  - `NPUBackend.apply_flat_gate(flat, local_matrix, indices)` 新方法；
    `_apply_local_matrix_to_state_flat`（`aicir/core/gates.py`）在 `npu_complex` 且
    `hasattr(backend, "apply_flat_gate")` 时优先调用，原有路径保留作为 fallback。
  - 实数门兼容：NPU 上 RY/H/X 等实数门的 `local_matrix` 是 float32（非 complex64，
    `torch.imag` 会报错），`_NpuLocalGateApplyFn` 经 `_mat_re_im` 判别 dtype，
    实数分支跳过虚部 matmul 并返回实数 `grad_local`，复数门走完整四实数 matmul。
  - 配套测试：`tests/backends/test_npu_hamiltonian_grad.py` 新增 2 项
    （前向等价性、autodiff vs 有限差分梯度一致性 < 1e-3）；套件合计 6/6 pass，
    全量 847 passed（含原有预期失败不变）。
- **NPU complex64-free autodiff：`_NpuHamiltonianExpectationFn` + `NPUBackend.hamiltonian_expectation_pauli`。**

  - 根因：`Supernet._hamiltonian_expectation` Pauli 循环中 `state`（complex64）被 1313 次用作期望
    值的 bra/ket，backward 需 1313 次累加 `state.grad`（complex64 add → `aclnnAdd(DT_COMPLEX64)`
    → Ascend 缺失该算子，`--gradient ad` 必然崩溃）。
  - 修复：`_NpuHamiltonianExpectationFn`（新 `aicir/backends/npu_backend.py`）把整个 Pauli 循环
    包进 `torch.autograd.Function`；state 在图中只出现一次；1313 次梯度累加在 backward 内以
    float32 完成（`grad_re += ...` / `grad_im += ...`），最后 `torch.complex(grad_re, grad_im)`
    一次性构造返回——全程无 complex64 add。
  - `NPUBackend.hamiltonian_expectation_pauli(state, basis_indices, pauli_cache)` 新方法；
    `Supernet._hamiltonian_expectation` Pauli 路径优先调用（`hasattr` duck-typing，
    CPU/GPU backend 不受影响，仍走原循环路径）。
  - 实测验证：`demos/check_npu_autograd.py` 已确认 float32 autograd 在 Ascend 全量可用；
    梯度公式 `grad_state = 2·go·(H|ψ⟩)` 与 psr 梯度误差 < 1e-4（见新测试）。
  - 配套测试：`tests/backends/test_npu_hamiltonian_grad.py`（4 项：符号正确性、前向等价、
    autodiff vs psr 梯度一致性）。

## 2026-06-21

### Added

- **`NPUBackend.caps(capabilities)`：消费 `npu_probe` 能力 sheet + 单设备 sizing guard。**
  - 新增 classmethod `NPUBackend.caps(caps, *, device=None, dtype=None, fallback_to_cpu=True)`，
    显式注入 `NpuCapabilities`（`device` 缺省取 `caps.device`），读 `max_qubits` 与
    `needs_real_imag_decomp` 填执行参数；本身不探测（探测仍由 `probe_npu` 负责）。
  - 新增 `ensure_capacity(n_qubits)`：`n_qubits` 超过 `max_qubits` 时抛 `ValueError`，
    `zeros_state` 分配 `2^n` 态向量前调用，防超容 OOM/SIGKILL；`max_qubits=None`（裸构造）不守卫。
  - 裸 `NPUBackend()` 保持现状默认（`_max_qubits=None` / `_needs_real_imag=True`），零行为变化。
  - dtype 路径分支（按 `needs_real_imag_decomp` 切换 real/imag 与原生复数）有意推迟：
    Ascend 恒需 real/imag，分支为现存硬件上的死代码，待复数能力 NPU 出现再按
    「原生支持时改用 GPUBackend」重设计。配套 `tests/backends/test_npu_backend_caps.py`。

### Fixed

- **Supernet QAS 在 16 比特（BeH2）被 `SIGKILL`（cgroup OOM，~238 GB/rank）的根因修复。**
  `Supernet` 构造时枚举整个单比特布局空间 `product(single_qubit_gates, repeat=n_qubits)`
  = `gates ** n_qubits`（BeH2 即 `5 ** 16 ≈ 1.5e11`），并为每个布局预建共享参数——只在测试用的
  极小 `n_qubits` 下可行，16 比特撑爆主机内存。现改为：布局按下标采样+解码（与旧 `choice(枚举列表)`
  **字节等价**，rng 序列不变，golden 测试不受影响），共享参数**首次访问懒建**，每 supernet 优化器
  懒建并经 `add_param_group` 增长；共享参数内存降为 `O(supernet_steps × layers × n_qubits)`。
  每个被访问架构的参数在**所有** supernet 上创建（仅评估分片，参数创建不分片），保持 `safe`/
  `aggressive` 分片下各 rank 的共享参数键集一致（梯度 all-reduce / `broadcast_parameters` 依赖此不变量）。
  配套 `tests/test_supernet_lazy_layouts.py`（含 16 比特秒级构造回归）。

## 2026-06-20

### Added

- **`qfun` 第二片（NEXT.md §5）：多参数 + 多测量 + `BasicVQE` 接入。**

  - 多参数：单观测量时 `cost(x)`/`cost.grad(x)` 接受单数组参（vector），梯度返回同形数组。
  - 多测量：`observable=[H1, H2, ...]` → `cost(x)` 返回 `(n_obs,)` 数组，`cost.grad(x)`
    返回 Jacobian（标量参→`(n_obs,)`，向量参→`(n_obs, n_param)`），逐观测量经同一
    `aicir.qml.deriv.psr` 求梯度（单一真源）。
  - `BasicVQE(cost=<qfun>, n_params=...)`（须单观测量 qfun）旁路 ansatz/hamiltonian 编排，
    `energy`/`parameter_shift_gradient` 直接委托 `cost`/`cost.grad`，`metadata["mode"]="qfun"`；
    `hamiltonian` 改为可选位置参（cost 模式下不需要）。配套 `tests/vqc/test_vqe_qfun.py`、
    扩充 `tests/qfun/test_qfun.py`。
- **`qfun` 第三片（NEXT.md §5）：`BasicQAOA` 接入 + 噪声路径封装。**

  - `BasicQAOA(cost=<qfun>, p=...)`（须单观测量 qfun）旁路稠密矩阵 ansatz，`energy`/梯度委托
    `cost`/`cost.grad`，`n_params=2p`；`problem_hamiltonian` 改为可选，`QAOAResult.statevector`
    允许 `None`（cost 模式）。配套 `tests/vqc/test_qaoa_qfun.py`。
  - `@qfun(..., noise_model=NoiseModel)` 把噪声附加到线路、经 `Measure.run` 走密度矩阵模拟读取
    期望值；`differential="auto"` 在有噪声时以 `noisy=True` 走 `select_diff`。
- **`aicir.primitives` 第 4 节主体落地：补齐 Sampler/Estimator 全变体 + 延迟绑定 + 扩展点。**

  - 采样新增 `StatevectorSampler`（精确解析概率，拒绝 `shots=`）、`NoisySampler`
    （`noise_model` 附加到线路走密度矩阵采样）。
  - 估计新增 `NoisyEstimator`（密度矩阵期望，`shots=None` 确定性 / `shots>=1` 叠加散粒），
    暴露 `estimate()`，可作 `BasicVQE(energy_estimator=...)` 注入。
  - 扩展点 `BackendSampler`/`BackendEstimator`：包装用户注入的 `runner`（counts / 期望值 /
    现成结果对象），面向真实硬件或远端服务。
  - 全部 `run(...)` 新增 `parameter_values=` 延迟绑定（对模板电路；单电路 → 一维数组、
    序列 → 数组序列）。
  - 下游加性集成：`BasicVQE` 经 `energy_estimator=` 直接消费 `ShotEstimator`/`NoisyEstimator`，
    未改 VQE 内部（已端到端测试）。扩充 `tests/primitives/test_primitives.py`。
- **`aicir.backends.npu_probe` 模块：Ascend NPU 硬件能力探测与缓存。**

  - 公开 API：`probe_npu(backend=None, *, allow_cpu_fallback=False, refresh=False)` 探测 NPU
    dtype / 算子 / 张量维度 / 内存能力，缓存到磁盘（`~/.cache/aicir/`，支持 `AICIR_CACHE_DIR` 覆盖），
    缓存键为 `device | torch_version | torch_npu_version`。
  - `NpuCapabilities` 不可变数据类：探测结果容器，含 `supports_complex_*`、`max_ndim`、`max_qubits`
    （单卡/分片）、`total_memory`，可序列化/反序列化（`to_dict` / `from_dict`）。
  - `target_from_npu(caps, n_qubits=None) -> Target`：把能力映射为电路 Target 标志。
  - 脚本 `demos/demo_npu_probe.py`：命令行工具，支持 `--allow-cpu-fallback` / `--refresh` 旗标，
    打印能力表并构建 Target（若可能）。配套 `tests/backends/test_npu_probe.py`（单元测试 + 集成测试）。

### Fixed

- **`qfun` 单元素一维参数梯度。** `QFun.grad` 旧逻辑把任意 `size==1` 数组折叠成标量，导致
  `theta[0]` 索引的单参向量函数报错并触发 NumPy `ndim>0 → scalar` 弃用告警；现仅对 0 维
  （真标量）折叠，一维数组按原样保形传入。
- **`npu_probe` 算子探测在 Ascend 上无法逐个测量。** `_probe_op_support` 旧逻辑用
  `torch.ones(complex64)` 构造测试张量，而 Ascend 的填充算子 `aclnnInplaceOne` 不支持
  complex64，构造即抛错，使 `supports_complex_matmul/conj/add` 三标志全退化为 `False`
  （非逐个测量）。改用 `torch.complex(实数, 实数)` 构造，绕开填充算子，使 matmul/conj/add
  各自独立测量；构造本身失败才三者皆 `False` 并记 `construct complex64`。

## 2026-06-19

### Added

- **QAS 模块化第一片：`SearchStrategy` 协议 + 策略注册表，开始取代 `runner.py` 的硬编码 `if` 分发链。**
  新增 `aicir.qas.core.strategy.SearchStrategy`（抽象基类，契约 `run(request) -> 结果`）与
  `aicir.qas.core.registry`（`StrategySpec` 冻结数据类 + `register_strategy`/`get_strategy`/
  `get_spec`/`registered_strategies`/`unregister_strategy`，镜像 `DiffMethod`/`GateSpec` 习惯）。
  内置策略在 `aicir.qas.core.strategies` 适配并注册。`run(method, ...)` 先查注册表、命中
  走 `strategy.run`，未命中回落旧分支。**当前仅 `supernet` 迁移**为 `SupernetStrategy`；其余方法
  （`ppo_rb`/`ppr_dql`/`crlqas`/`supernet_classification`/`supernet_h2`）仍走旧分支，对用户行为
  与返回值不变。配套 `tests/qas/test_strategy_registry.py`。
- **新增 PennyLane 风格量子函数 `qfun`（NEXT.md §5 第一片）。** 新模块 `aicir/qml/qfun.py`
  导出 `qfun` 装饰器与 `QFun` 类，统一"量子函数 + 设备 + 测量 + 梯度"：
  `@qfun(device=..., differential=..., observable=..., shots=None)` 包装一个**返回
  `Circuit`** 的函数；`cost(x)` 得期望值，`cost.grad(x)` 得梯度。`device` 映射
  `numpy`/`cpu`/`gpu`/`torch`/`npu` 后端；`differential` 经 `aicir.qml.diff` 注册表分发
  （`"auto"` 走 `select_diff`——`qfun` 是 `select_diff` 的首个真实调用方），观测量经
  `observable.to_matrix(backend)`、测量走 `Measure.run` 精确路径。支持单个可训练位置
  参数（标量/一维数组）。设计上观测量声明在装饰器、函数体显式返回 `Circuit`（不依赖
  全局 tape，规避门队列化的侵入与误捕获）；故暂不提供 `expval`。配套 `tests/qfun/test_qfun.py`。
- **`DiffMethod` 注册表第二片（NEXT.md §6）：按 `category` 索引全部内置微分方法。**
  - `DiffMethod` 新增 `category` 字段（`__post_init__` 校验），取值
    `fn_gradient`（`(fn, params) -> 梯度向量`）/ `circuit_gradient`
    （`(circuit, observable) -> 梯度`）/ `preconditioner`
    （`(fn, state_fn, params) -> 方向/度规`）。
  - 注册表新增内置项：`ad`（`circuit_gradient`，伴随微分）与 `qng`/`bdqng`/
    `kqng`/`dqng`（`preconditioner`，量子自然梯度族）；连同原有
    `psr`/`fd`/`auto`/`spsa`/`spsr` 共十项。
  - `registered_diffs(category=None)` 支持按类别过滤检索。
  - 契约安全：`resolve_diff` 与 `select_diff` **仅对 `fn_gradient` 生效**——
    `resolve_diff('ad'|'qng'|...)` 抛 `ValueError`，避免经典优化器拿到签名
    不兼容的可调用；`ad`/`qng` 族仅经 `get_diff`/`registered_diffs(category=...)`
    发现。`mpsr` 仍有意不纳入。
  - capability 字段（`exact`/`stochastic`/`requires_torch`/`supports_*`）只服务
    `fn_gradient` 的 `select_diff` 优选；`ad`/`qng` 族从态向量求值，标注
    `supports_shots/noise=False`。

### Changed

- **QAS 结构简化（安全片）：解散误命名的 `aicir.qas.primitives`，并消除重名 `sharding`。**
  - `qas/primitives/` 与顶层 `aicir.primitives`（Sampler/Estimator）名称冲突、内容也非「primitives」，整体解散：
    `ansatz.py` → `qas/library/`，`backend_utils.py` 与 NPU `sharding.py` → `qas/core/`。
  - `qas/vqe_loop/sharding.py`（fair-label 队列分片 CLI 调度器）改名 `shard_scheduler.py`，与
    `core/sharding.py`（NPU 集合通信原语）区分；CLI 命令相应改为
    `python -m aicir.qas.vqe_loop.shard_scheduler`。
  - 纯文件搬迁 + import 路径更新，无行为变化；导入改为
    `from aicir.qas.library.ansatz import ...` / `from aicir.qas.core.backend_utils import ...` /
    `from aicir.qas.core.sharding import ...`。
- **线路结构优化统一收归 `aicir.transpile`，并移除与 `aicir.optimizer` 的重复实现。**
  - `aicir.optimizer.circuit` 模块整体移除；其中重复的本地化简规则（与
    `transpile/passes/_local_rewrite.py` 逐行重复）不再保留，门字典列表的不动点
    化简统一为 `_local_rewrite.optimize_gates`（规则单一来源）。
  - `optimize_basic` / `optimize_circuit`（多格式：dict / OpenQASM 文本 / DAG）迁至
    `aicir.transpile.rewrite`，并由 `aicir.transpile` 导出。`aicir.optimizer` 现仅提供
    经典参数优化器（`Adam`/`SPSA`/`minimize` 等）。
  - `aicir.transpile.default_optimization_pipeline()` 重命名为
    `aicir.transpile.optimize(circuit) -> Circuit`（直接返回优化后的新线路，
    等价于旧的 `default_optimization_pipeline().run(circuit)`）；不再保留旧名。
    `optimize_circuit(circuit)` 为其 Circuit 专用别名。
  - 受影响导入：`from aicir.optimizer import optimize_basic/optimize_circuit`
    → `from aicir.transpile import optimize_basic/optimize_circuit`；
    `default_optimization_pipeline` → `optimize`。

## 2026-06-18

### Added

- 新增硬件目标描述 `Target`（NEXT.md 第 3 节第一片）：子包 `aicir.devices` 提供冻结数据类 `Target`（字段含 `n_qubits`/`basis_gates`/`coupling_map`/`supports_shots`/`supports_statevector`/`supports_density_matrix`/`supports_autodiff`），`basis_gates` 按 `GateSpec` 规范名归一，`coupling_map` 按无向图处理；提供门集查询 `supports(gate)` 与拓扑查询 `coupled(a, b)`/`neighbors(q)`/`fully_connected`。配套 `tests/devices/test_target.py`。
- 新增 `aicir.transpile` 编译 pass（NEXT.md 第 2 节、推进顺序第二阶段第 4 项）：
  - `DecomposePass(basis_gates=..., target=..., skip_unsupported=False)`：把高级双比特门分解到目标门集，内置经数值验证的标准规则 `swap→3×cx`、`cz→h·cx·h`、`cy→rz·cx·rz`（仅单控制位）；门集内的门原样保留，不在门集且无规则的双比特门默认抛 `ValueError`。规则展开产生 `hadamard`/`rz`，暂不对任意单比特门做 Euler 基底翻译。
  - `LayoutPass(initial_layout=..., target=...)`：按逻辑->物理映射（`dict` 或序列）重标号比特，不插门，比特置换意义下与原线路等价；`None` 为平凡恒等布局；映射须单射，可由 `Target` 限定物理位宽与范围。
  - `RoutingPass(target=...)`：沿耦合图最短路径插入 SWAP 使每个双比特门作用在相邻物理比特上，施加后按相反顺序插回 SWAP 复位，因此整条线路与原线路**完全幺正等价**（SWAP 数非最优，基于置换跟踪的最优路由留待后续）；全连接 `Target` 时为恒等，>2 比特门抛 `NotImplementedError`。
  - 三者从 `aicir.transpile` 导出；`PassManager` 字符串名新增 `"decompose"`（默认 `cx` 门集）与 `"layout"`（平凡布局）。配套 `tests/transpile/test_decompose_pass.py`、`test_layout_pass.py`、`test_routing_pass.py`。
- `aicir.qas` supernet 支持单次搜索的多 NPU 分片：新增 `SupernetConfig.shard_mode`
  与 `supernet_qas(..., mode="safe"|"aggressive")`。仅在分布式 NPU 运行下生效，
  分片 training / ranking / finetune 三个阶段；`safe` 与单卡数值等价，`aggressive` 为数据并行。
  `demos/BeH2/BeH2_npu.py` 改为同种子单次分片搜索并新增 `--mode`。

### Fixed

- **BeH2 16 比特 NPU supernet QAS 运行被 `SIGKILL` 终止的排查与内存修复**：
  远端执行 `torchrun --nproc_per_node=4 demos/BeH2/BeH2_npu.py` 时，4 个 rank 均成功初始化 NPU 后进入
  `start sharded supernet search`，约 12 分钟后 torch elastic 报告 rank 3 `exitcode: -9`
  / `Signal 9 (SIGKILL)`；Ascend TBE 后台线程随后输出的 `EOFError`、`ConnectionResetError` 判定为子进程被杀后的连带现象，而不是首要异常。该任务规模为 16 qubits、1313 个 Pauli 项、默认 `layers=4`、`supernet_num=6`、`supernet_steps=300`、`ranking_num=120`、`finetune_steps=500`，单次精确能量评估和训练反向图都很重。
  - 第一次尝试：检查 `BeH2_npu.py` 与 `supernet.py` 后确认 `Hamiltonian` 没有走稠密矩阵路径，而是逐 Pauli 项计算期望；初步判断为 Pauli 项缓存、autograd 图与多 rank 并发共同导致内存压力，真正失败点是系统层 `SIGKILL`。结果：定位方向成立，但尚未改代码。
  - 第二次尝试：把 Pauli 项缓存从每项常驻 `mapped_indices + phase_real + phase_imag` 改为 `mapped_indices + int8 phase_code`，希望减少两个 float phase 向量。结果：相关 supernet 测试通过，但本地 CPU 回退最小 BeH2 smoke 仍被 `exit code 137` 杀掉，说明仅压缩 phase 不够。
  - 第三次尝试：进一步删除每项常驻 `mapped_indices`，改为共享 basis index 加每项 `flip_mask` 临时生成映射索引。结果：相关测试通过，但本地 smoke 仍被杀，说明除常驻索引外，纯评估/训练中的 autograd 图仍是重要内存来源。
  - 第四次尝试：将 `demos/BeH2/BeH2_npu.py` 默认梯度切为参数移位（新增 `--gradient psr|ad`，默认 `psr`），并让参数移位黑盒评估与 `finetune_steps=0` 纯评估路径进入 `torch.no_grad()`，避免对 1313 个 Pauli 项构建不用的反向图；baseline 的 `finetune_steps=0` 也不再强制做一次 backward。结果：`pytest tests/test_supernet_sharding.py tests/test_vqa_qas.py -q` 通过（28 项）；本地 CPU 回退 16 比特 BeH2 smoke 仍可能因环境资源不足被 `exit code 137` 杀掉，但已消除两个明确的内存放大因素。
  - 第五次尝试：将 Pauli 期望缓存进一步压缩为每项仅保存 `flip_mask`、`sign_mask`、`y_count mod 4` 与系数，完全不常驻 Pauli phase/index 大张量；求值时用共享 `basis_indices` 临时计算奇偶符号。结果：相关 supernet 测试继续通过；当前建议远端 NPU 先用默认 `--gradient psr` 重跑正式命令，若仍被杀，先用 `--layers 1 --supernet-num 1 --supernet-steps 1 --ranking-num 1 --finetune-steps 0` 做链路验证，再逐步放大规模。

## 2026-06-17

### Changed

- **`Measure.run` 的 `measure_qubits` 语义反转、移除 `tm` 参数**（破坏性接口变更）：
  `measure_qubits` 成为末端读出的唯一控制项——`None`=不做末端测量；空 `()`（新默认）/`[]`=读出全部比特；`[q0,…]`=读出子集（保留顺序）。原 `tm` 布尔参数删除（原 `tm=False` 等价 `measure_qubits=None`，原 `tm=True` 即默认全测）。exact 模式（`shots∈{None,0}`）下 `measure_qubits` 被忽略、不再报错。`ShotSampler.run` 默认 `measure_qubits=()`。涉及 `aicir/measure/measure.py`、`aicir/primitives/sampler.py`、`aicir/measure/result.py` 及 README §4。

## 2026-06-16

### Fixed

- **昇腾 NPU 上 8+ 量子比特态矢量演化崩溃**：`aicir/core/gates.py` 的 `_apply_local_matrix_to_state` 原先将态矢量整形为 `(2,)*n_qubits` 的高阶张量再 `permute`/`contiguous`，秩等于 n。当 n>8 时超出昇腾 ACL 算子最多 8 维的限制，`.contiguous()` 触发 `aclnnInplaceCopy failed ... the self tensor cannot be larger than 8 dimensions`（20 比特线路必现）。现改为把相邻的非目标比特合并成单个「空闲段」维度，工作张量的秩降至至多 `2*len(axes)+1`（单比特门 3 维，Toffoli 也仅 7 维），与量子比特总数无关。改动为后端无关、按位精确等价（`err=0`，144 项门测试全通过），不改变基排序与端序。GPU/CPU 后端原先因 PyTorch 允许至多 64 维而未受影响，本次修改对其结果不变、性能等同或略优（`contiguous` 仍只拷贝 2^n 个元素，但每元素的步幅索引计算维度更少）。
- **密度矩阵 `partial_trace` 的同类高维崩溃**：`npu_backend.py` 与 `gpu_backend.py` 的 `partial_trace` 原先将密度矩阵一次性整形为 `[2]*n + [2]*n`（秩 2n）再 `permute`/`einsum`，NPU 在 n>4、CUDA 在 n>32 时即超限。现改为逐比特求迹，每步整形为 `(L, 2, R, L, 2, R)` 的秩-6 张量并对该比特的行/列同一指标求和，工作张量的秩恒为 6，与量子比特总数无关。按降序求迹保持保留比特的原有（升序）次序，结果与原实现按位等价（已对 5 比特全部 keep 子集校验 `err<1e-10`，相关测试全通过）。

## 2026-06-15

### Added

- `cx` / `cnot` 工厂的第一个参数支持目标位列表：`cx([t1, t2, ...], [controls])` 对每个目标施加同一组控制位，等价于多个共享控制位的单目标 CX（彼此对易）。单目标 `cx(t, [c])` 行为不变；`control_states` 同样适用。多目标门以单个 `Operation`（`qubits=(t1, t2, ...)`）存储，态向量演化、`Circuit.unitary()` 与 OpenQASM 导出在内部按目标自动展开。`cx` 的 `GateSpec.num_qubits` 由 `1` 放宽为 `None`。配套新增 `tests/gates/test_cx_multi_target.py`。
- 新增 `DiffMethod` 策略注册表（NEXT.md 第 6 节第一片）：子包 `aicir.qml.diff` 提供冻结数据类 `DiffMethod`（字段含 `name`/`fn`/`aliases`/`exact`/`stochastic`/`requires_torch`/`supports_shots`/`supports_noise`）与注册表 API（`register_diff`/`unregister_diff`/`get_diff`/`registered_diffs`/`canonical_diff`/`resolve_diff`），并从 `aicir.qml` 顶层再导出。
- 新增纯函数选择器 `select_diff(*, backend=None, shots=None, noisy=False)`，按 auto → psr → fd 优先级自动推断梯度方法（`spsa`/`spsr` 不参与自动选择）；已有单元测试覆盖，暂未接入调用方（保留给后续 QNode）。
- 内置注册 fn-based 全梯度方法：`psr`/`fd`/`auto`/`spsa`/`spsr`；`mpsr`（返回标量混合偏导而非梯度向量）有意排除在注册表之外，仍作为 `qml.mpsr` 直接可用；基于线路的 `ad` 与预条件策略 `qng` 同样不纳入注册表。
- 新增 `State.__array__` numpy 数组协议：`np.asarray(state)` / `np.allclose(a, state)` / `backend.cast(state)` 等隐式转换可用（向量导出 `(2^n,)`、密度导出 `(2^n, 2^n)`，返回副本以保持不可变风格）。

### Changed

- `aicir/optimizer/params.py` 的 `_gradient_from_method` 改为经 `resolve_diff` 分发，`GD`/`Adam`/`ScipyMinimize` 现可统一访问所有内置梯度方法；对 `requires_torch=True` 的方法（即 `auto`）在经典黑盒目标路径上新增守卫，传入时抛出明确错误。
- `aicir/qml/deriv.py` 未改动；`vqc`/`qas` 保持原有 `from ..qml.deriv import psr` 路径，参数移位单一实现不变。
- **（破坏性）** `aicir.measure.Result` 的 `state`/`final_state`/`snap()`/`snapshot_states` 现返回统一 `State` 对象（而非裸 numpy 数组），与全局 State 迁移一致；`.ket`/`.array`/`.matrix`/`.is_density`/`.probabilities()` 可直接使用。迁移指引：`result.state.reshape(-1)` → `result.state.array`，`result.final_state.reshape(2, 2)` → `result.final_state.matrix`；`np.asarray(result.state)`、`np.allclose(...)`、`backend.cast(...)` 经 `State.__array__` 仍兼容。`reduce()` 仍返回约化密度 numpy 数组，`final_state_kind` 字段保留。

## 2026-06-14

### Added

- **批量态矢量路径 `BatchSV`**（`aicir.core.batch`，已在 `aicir` / `aicir.core` 顶层导出）：一次演化一批态矢量，供把变分线路当作神经网络一层的场景使用。特性：
  - 旋转门角度可逐样本（per-sample）不同（如数据编码角度依赖输入），亦兼容 0 维标量张量参数（保留 autograd 梯度）；
  - 全程以实部/虚部两个实张量表示，只用实数乘加，**NPU 安全**（规避 Ascend 缺失的 complex64 `aclnnAdd`/`aclnnMul`），反向不触发复数累加；
  - 端到端可微；门矩阵复用单态路径 `_single_qubit_base_for_gate` 的定义（常量门/标量旋转门），逐样本张量角度按相同公式构造批量基矩阵；
  - 端序与单态路径一致（qubit 0 为最高位）；提供 `apply_gate`、`probabilities`、`z_expectations`（逐比特 `<Z_q>`）。
  - 受控单比特门（`crx/cry/crz` 等）通过控制比特掩码通用实现。

### Changed（破坏性变更）

- **后端、噪声、算符上移到顶层（`aicir.channel` 包移除）**：`aicir.channel.backends` → `aicir.backends`、`aicir.channel.noise` → `aicir.noise`、`aicir.channel.operators` → `aicir.operators`。旧 `aicir.channel` 包及其 `__init__` 已删除，不保留兼容垫片。顶层再导出（`Backend`/`NumpyBackend`/`GPUBackend`/`NPUBackend`、`NoiseModel`/`NoiseChannel`/各噪声信道、`PauliOp`/`PauliString`/`Hamiltonian`）不变，`from aicir import ...` 无需改动；仅子模块路径导入（如 `from aicir.channel.backends.numpy_backend import NumpyBackend`）需改为新路径。
  - 顺带修复 `aicir/__init__.py` 中 `measure` 门构造器被 `aicir.measure` 子包属性覆盖的隐患（旧 `channel/__init__` 早期导入 `aicir.measure` 恰好掩盖了该问题）：现在导入 measure 子包后显式恢复门构造器绑定。
- **`measure`/`reset` 量子比特参数改为「单个 int 或列表」**：`measure([0,1], ...)` / `reset([0,1])`；不再支持 `measure(0, 1)` 多位置形式（请改用列表）。
- **统一测量模型**：删除"机制一/机制二互斥"框架；线路内嵌 `measure`/`reset` 与末端读出（`tm`/`measure_qubits`）现可共存，两者是同一模型的两个正交部分。
- **线路内 `measure(basis, id)` 改为投影测量**：由非坍缩边缘采样标记改为真正的两结果联合 Pauli 投影（`basis∈{Z,X,Y}`，同子空间保留相干）；新增 `id` 参数使 `result.output("id")` 可用；`basis`/`id` 字段在 IR 中作为一等字段序列化（JSON 往返保真）。
- **`reset` 改为无前置条件的信道**：删除"必须先有同一比特 `measure`"的限制；`reset` 可出现在任意位置；对纠缠目标施加 reset 使该轨迹升级为密度矩阵。
- **新增 `run()` 参数 `tm`/`sm`/`seed`**：`tm=True`（默认）控制末端测量；`sm="avg"` 为多轨迹聚合模式（`"shot"/"cond"` 待实现，传入报 `NotImplementedError`）；`seed` 单一 RNG 种子贯穿全部随机操作。
- **`output`/`counts`/`prob` 改为方法**：`result.output(i)`/`result.counts(i)`/`result.prob(i)` 按操作下标（或字符串 `id`）索引；末端结果用 `output(-1)`/`counts(-1)`。原字段访问方式（`result.output`/`result.counts`）已移除。
- **`state`/`final_state` 语义更新**：`state` = 末端测量前完整态（`ρ_pre`）；`final_state` = 末端测量后的态（`ρ_post`）；`shots>1` 时两者均为平均密度矩阵 `(2^n,2^n)`。
- **`shots=0` ≡ `None`（exact 模式）且覆盖 `tm`**：exact 模式下不执行末端测量（`final_state==state`，`output(-1)` 报错）；线路内 `measure()` 仍走一条随机轨迹并坍缩。这是对上传规则 23（`shots=0` 非法）与规则 36（`shots=None` 仍取末端结果）的显式偏差。
- **`Circuit.unitary()` 遇 measure/reset 改为报错**：默认对 `measure`/`reset` 一律 `raise`（原行为：`measure` 静默跳过）；新增 `ignore_nonunitary=True` 选项以丢弃非酉操作并返回酉部分。
- **QASM 导出限制**：联合（多比特）/ 非 Z 基 / 带 `id` 的 `measure` 导出时报 `NotImplementedError`；普通单比特 Z `measure` 仍按标准 QASM 2.0/3.0 导出。
- **末端输出顺序保留输入顺序**：`measure_qubits` 原始输入顺序保留到 `output(-1)` 列顺序，不做内部排序。
- **已移除**：`run_density_matrix`、`run_batch`、`scan_parameters` 旧入口；噪声通过 `circuit.noise_model` / `initial_density_matrix` 传入新 `run()`。
- **`sm="shot"/"cond"` 待实现**：传入时报 `NotImplementedError`，后续版本实现。

## 2026-06-12

### Added

- 新增 Qiskit 互操作入口（NEXT.md 第 8 节第一片）：`circuit_to_qiskit`/`circuit_from_qiskit` 以及短别名 `to_qiskit`/`from_qiskit`，位于 `aicir.core.io.qiskit_io` 并从 `aicir.core.io`、`aicir.core`、顶层 `aicir` 导出。`qiskit` 为可选依赖，仅在调用互操作函数时导入；当前支持基础门、参数旋转、受控门、`swap`、`rzz/rxx`、`u2/u3`、`ccx` 和线路内 `measure` 标记。
- 新增 PennyLane 互操作入口（NEXT.md 第 8 节第一片）：`circuit_to_pennylane`/`circuit_from_pennylane` 以及短别名 `to_pennylane`/`from_pennylane`，位于 `aicir.core.io.pennylane_io` 并从 `aicir.core.io`、`aicir.core`、顶层 `aicir` 导出。`pennylane` 为可选依赖，仅在调用互操作函数时导入；当前支持基础门、参数旋转、受控门、`swap`、`rzz/rxx`（PennyLane `IsingZZ`/`IsingXX`）、`u2/u3`、`ccx` 和 `identity`。
- 新增 WuYue 互操作入口（NEXT.md 第 8 节第一片）：`circuit_to_wuyue`/`circuit_from_wuyue` 以及短别名 `to_wuyue`/`from_wuyue`，位于 `aicir.core.io.wuyue_io` 并从 `aicir.core.io`、`aicir.core`、顶层 `aicir` 导出。`wuyue` 为可选依赖，仅在调用互操作函数时导入；当前支持 WuYue 原生基础门、参数旋转、`cx/cz`、`swap`、`rzz`（WuYue `IsingZZ`）、`u2/u3`、`ccx`、`identity` 和线路内 `measure` 标记。
- `Result` 新增 `state` 字段：始终返回测量前的完整末态（SV 路径为态向量，DM 路径为 flatten 密度矩阵），不受采样影响。
- `Result` 新增 `output` 字段：`shots=1` 单次测量结果——被测比特上 Z⊗…⊗Z 关联投影测量的本征值（±1，实现为对各被测比特分别做 Z 基投影后取乘积，与联合宇称测量的 ±1 分布一致且保证其余比特为纯态）；坍缩到的具体基态见 `counts` / `final_state`。
- `result.metadata` 新增 `final_state_kind`（`'state_vector'`/`'density_matrix'`/`None`）与 `final_state_qubits`（`final_state` 所描述的比特下标）。
- 新增线路内 `reset(*qubits)` 标记，与 `measure(*qubits)` 参数格式一致；每个 reset 目标必须先有同一比特的 `measure`，且二者之间不能有任何量子门作用于该比特。`Measure.run()`/`run_density_matrix()` 支持 reset 执行语义；matplotlib 线路图以与 `measure` 同色、标注 `Reset` 的虚线表示 reset，`Reset` 字号与 `Rz` 门主标签一致且虚线 dash 间距更大；虚线区间不叠加普通实线，没有后续量子门时延伸到线路末端；遇到 CNOT/SWAP 等无完整方框的后续门时，虚线停在虚拟方框左边界，虚拟方框内部恢复普通实线。
- 新增 `demos.reset_demo`，通过 `H(0) -> cnot(1,0) -> cnot(2,1) -> measure(1) -> reset(1) -> cnot(1,2)` 三比特线路的 `result.snap(...)` 记录 reset 前后中间态，验证 `reset(1)` 将已测量比特从 `|1>` 重置为 `|0>`，且后续门从重置后的态继续演化。

### Changed

- **破坏性**：`Measure.run`/`run_density_matrix` 的 `shots` 默认值由 `None` 改为 `1`；`shots=None`/`0` 表示不测量（无论是否传 `measure_qubits`）。
- **破坏性**：`result.final_state` 语义由"演化末态"改为"测量后的态"：`shots=None`/`0` 时与 `state` 相同；`shots=1` 时为坍缩后的态（子集读出时仅含未被测比特）；`shots>1` 时为对被测比特求偏迹后的约化密度矩阵（SV 路径为 `(2^m, 2^m)` 二维，DM 路径为 flatten；读出全部比特时无剩余比特，为 `None`）。需要演化末态请改用 `result.state`（内部消费方 `BasicVQE.ansatz_state`、`StatevectorEstimator` 已随之切换）。
- `shots` 为负数时显式抛出 `ValueError`。
- **破坏性**：统一量子态表示为单一 `State` 类，删除 `StateVector` 与 `DensityMatrix` 两个公开名称（顶层 `aicir` 与 `aicir.core` 不再导出它们）。
  - 新增 `State.from_matrix(...)` 从密度矩阵构造；`from_array`/`from_matrix`/`zero_state` 的 `backend` 改为可选（默认 `NumpyBackend`），`from_array`/`from_matrix` 可省略 `n_qubits`（按长度/形状推断）。
  - 新增属性 `.array`（纯态振幅向量，混合态为 `None`）、`.matrix`（密度矩阵）、`.ket`（Dirac 记号：纯态超叠加、混合态 Σρ_ij|i><j| 展开），均可直接打印。
  - 新增 `.is_density` 判定属性；密度矩阵方法（`purity`/`partial_trace`/`eigenvalues`/`von_neumann_entropy`/`is_pure`/`maximally_mixed`）并入 `State`。
  - 迁移指引：`isinstance(x, DensityMatrix)` 改用 `x.is_density`；`DensityMatrix(...)` 改用 `State.from_matrix(...)`；`StateVector(...)` 改用 `State(...)` / `State.from_array(...)`。

## 2026-06-11

### Added

- 新增 `aicir.gates` GateSpec 注册表（NEXT.md 第 7 节第一片）：`GateSpec`（门名/别名/目标比特数/参数个数/是否受控/QASM 名）、`register_gate`/`unregister_gate`/`get_gate_spec`/`registered_gate_names`；内置门集已全部注册。`num_qubits`/`num_params` 为 `None` 表示可变（`unitary`、`measure`、整寄存器 `identity`）。
- `Operation` 构造期接入 GateSpec 校验：已注册门检查目标比特数、参数个数与控制位要求，未注册门名保持宽松（自定义门不受限）。
- `aicir.transpile.ValidatePass` 升级为实质校验：qubit/控制位越界（相对 `n_qubits`）、目标与控制比特冲突、重复比特；原先仅做 round-trip 规范化。
- 新增 `aicir.gates.canonical_gate_name`：把别名门名（`X`/`cnot`/`ccnot`/`measurement` 等）解析为规范名，未注册名称原样返回。
- `aicir.transpile.CanonicalizePass` 升级为实质规范化：把门字典中的别名 `type` 重写为 GateSpec 规范名；原先仅做 round-trip 复制。
- QASM 导出门名改以 GateSpec 注册表为单一来源：`core/io/qasm.py` 的导出表由 `GateSpec.qasm_name` 派生，别名键（`X`/`cnot`/`ccnot` 等）从表中移除，导出时先经 `canonical_gate_name` 归一；导入表由导出表反推。导出结果与旧版完全一致（有回归测试钉住）。
- `GateSpec` 新增 `symbol` 字段（绘图显示符号，受控门为目标位符号），ASCII 与 matplotlib 绘图的符号/配色族查询改以注册表为单一来源：`_single_gate_symbol`/`_controlled_target_symbol` 由 `GateSpec.symbol` 派生（注册自定义门可携带 symbol 直接显示），`visual/plot.py` 的 `_FAMILY` 配色表只保留规范名键。
- 矩阵路径统一别名处理：`gate_to_matrix`/`apply_gate_to_state`/`_single_qubit_base_for_gate` 等在入口经 `canonical_gate_name` 归一后分发，全部 `["pauli_x", "X"]`/`["cnot", "cx"]`/`["toffoli", "ccnot"]` 式别名分支收敛为规范名（别名与规范名共享矩阵缓存）。行为与旧版完全一致（有别名等价回归测试钉住）。
- 新增 `aicir.primitives`（NEXT.md 第 4 节第一片）：`BaseSampler`/`BaseEstimator` 接口与最小统一结果对象 `SampleResult`/`EstimateResult`（第 9 节切片）；`ShotSampler` 包装 `Measure`，`StatevectorEstimator` 提供精确态向量期望，`ShotEstimator` 包装 `PauliEstimator` 并暴露 `estimate()` 直通方法（可直接作 `BasicVQE(energy_estimator=...)` 注入）。约定：接收已绑定参数的电路，单入参返回单结果、序列返回列表，单个可观测量可广播。

### Changed

- 门工厂函数（`pauli_x`/`hadamard`/`rx`/`cx`/`swap`/`rzz`/`u3`/`u2` 等）**签名与参数顺序完全不变**，但返回值由裸门字典升级为类型化 `Operation`；`measure(...)` 返回 `Measurement`。构造期即校验（量子比特下标、控制位/控制态长度等）。`Circuit` 内部存储的门字典与旧版完全一致，下游消费方无需改动。
- `Operation`/`Measurement` 新增旧门字典**只读**兼容层：支持 `gate["type"]`、`.get()`、`in`、`dict(gate)`、`len`/迭代等读取，以及与旧门字典的双向 `==` 比较；写入（`gate[...] = ...`）抛出 `TypeError`（对象不可变）。
- `aicir.visual` 的 `plot(...)`/`show(...)` 来源归一化现接受类型化指令（单个 `Operation`/`Measurement` 或其序列）。

## 2026-06-10

### Added

- `Operation` 新增显式 `label` 字段（默认 `None`），对齐 NEXT.md typed IR 规格；门字典中的 `label` 键现在提升为该字段而不再落入 `metadata`，`to_dict`/`from_dict` 保持 round-trip。
- 新增第一批架构演进目录占位：`aicir.ir`、`aicir.gates`、`aicir.transpile`、`aicir.transpile.passes`、`aicir.devices`、`aicir.primitives`，用于后续 typed IR、GateSpec、pass pipeline、Target 和 Sampler/Estimator primitives 迁移。
- `NEXT.md` 记录 `aicir` 目标目录结构和第一批已落地目录。
- 新增 typed IR `Operation`：支持从现有门字典构造、转换回门字典、通过 `normalize_gate` 兼容旧入口；`Circuit` 构造、`append`、`extend` 现在可接收 `Operation`，同时内部继续保存现有门字典格式。
- 补齐 typed IR 第一阶段剩余对象：`Measurement` 支持测量声明与现有 `measure` 门字典互转，`Observable` 支持包装 `PauliString`、`Hamiltonian` 和 dense matrix，`CircuitIR` 支持从现有 `Circuit` 构造、转回 `Circuit`，并保留 operation 序列、量子比特数、经典比特和 metadata。
- 新增 typed IR 访问 helper：`aicir.ir.circuit_instructions`、`circuit_gate_dicts`、`instruction_name`、`instruction_qubits`、`instruction_controls`、`instruction_parameter` 等，用于内部模块统一消费 `CircuitIR`、`Circuit.operations` 和旧 `Circuit.gates`。
- `Circuit` 新增 `.operations` 与 `.ir` typed IR 视图；`.gates` 继续保留为旧门字典公开 surface。
- 新增 `aicir.transpile` pass pipeline：提供 `TransformationPass`、`PassManager`、`default_optimization_pipeline`，以及 `ValidatePass`、`CanonicalizePass`、`CancelInversePass`、`MergeRotationsPass`、`CommuteSingleQubitPass`；`optimize_circuit` 和 `optimize_basic(Circuit)` 继续保留旧接口并委托给默认 pipeline。
- 新增 `aicir.optimizer.optimize_circuit` 公开入口，用于直接优化 `Circuit` 对象并保留 `n_qubits` 与 backend。
- 扩展 `aicir.optimizer.circuit` 的 dict/Circuit 路径：支持有限安全重排，可跨过不同量子比特的单比特门，以及已知可交换的 CNOT 模式来消去冗余门或合并 `rx/ry/rz`。

### Changed

- JSON/QASM/DAG 导出、绘图、测量、Pauli 估计、transpile/optimizer、QML 伴随梯度、metrics、noise 和 QAS 的主要内部读取路径迁移为优先消费 typed IR；需要旧门字典格式的矩阵、渲染和本地 rewrite 兼容层会显式从 typed instruction 生成门字典视图。

### Tests

- 新增 `tests/circuit/test_typed_ir_internal_migration.py`，覆盖 `CircuitIR` 直接进入 JSON/QASM、绘图、metrics、测量、QML 伴随梯度、transpile/optimizer 和核心门矩阵路径。

## 2026-06-09

### Fixed

- 修复 `aicir.qas.algorithms.supernet` 计算 `Hamiltonian` 期望能量时的形状广播 bug：当态向量为 `(2^n, 1)` 列向量（训练/微调路径的默认形状）时，会与一维 phase/index 向量广播成 `(2^n, 2^n)` 并 `sum()`，导致能量被放大 `2^n` 倍。受影响时损失停在 `真实能量 × 2^n`、梯度趋近 0、架构排序失效，supernet 无法收敛到基态。现统一把态向量展平为一维后再计算，能量幅值正确，VQE/QAS 可正常收敛（MaxCut 等对角哈密顿量近似比从约 0.6 提升到 1.0）。
- 修正 MaxCut demo 的结果判读：区分能量期望对应的 `expected_cut` 与最佳显著读出比特串对应的 `achieved_cut`，避免把非基态叠加态误报为已完全收敛；同时将默认 ansatz 深度从 4 层提高到 6 层，并重新生成示例线路，使 5 节点示例的 `expected_cut` 接近精确最大割。
- MaxCut demo 新增 `--disable-rzz` 参数，可在 `supernet_qas` 搜索中禁用 `rzz`，只保留 `cx` 双比特门。
- 修正线路图中文字自适应：方块内文字渲染宽度限制在方块宽度的 0.8 倍以内，所有门内/门下文字高度限制在方块高度的 0.7 倍以内，方块下方参数文字宽度限制在方块宽度以内，避免长角度标签溢出。

### Added

- 新增 `demos/MaxCut/maxcut.py`：随机图 → MaxCut Ising 哈密顿量 → `supernet_qas` 搜索 VQE 基态线路 → 把哈密顿量与线路写入 `maxcut_hamiltonian.py`，并把随机图与线路一起绘制到 `maxcut_hamiltonian.png`。

## 2026-06-05

### Added

- 新增测量机制二：`measure(*qubits)` 门工厂，可在构造 `Circuit` 时内嵌测量标记，调用 `Measure.run()` 时自动仅读出标记比特，无需额外传入比特参数。
- `Measure.run()` 与 `Measure.run_density_matrix()` 自动检测线路中的 `measure` 门并跳过幺正演化；演化结束后计算指定比特子集的边际概率分布，输出对应多比特计数字典（MSB 顺序）。
- `Measure.run()` 结果的 `metadata["measured_qubits"]` 字段记录被读出的比特下标列表（机制二）；机制一下为 `None`。
- `Measure.run()` 与 `Measure.run_density_matrix()` 新增 `measure_qubits` 参数，可在机制一下显式指定要读出的比特子集（`metadata["measured_qubits"]` 会随之记录）；`run_batch` / `scan_parameters` 可通过 `per_circuit_options` 传递。
- 两种测量机制互斥：当电路已内嵌 `measure()` 门（机制二）时再传入 `measure_qubits`（机制一）会抛出 `ValueError`，避免两种读出方式相互冲突。
- `measure` 函数从 `aicir` 顶层导出，支持 `from aicir import measure`；`aicir.measure` 子包仍可通过 `from aicir.measure.measure import Measure` 访问。
- 线路图支持 `measure(q1, q2, ...)` 多比特测量门绘制：每个被测比特线上绘制独立测量框，测量框右侧不再延伸导线（符合量子线路惯例）。
- 新增 `Circuit.plot(...)` 语法，用于直接从电路对象输出线路图；默认文件位置为调用该方法的 `.py` 文件所在目录。
- 新增 `rxx(θ, q1, q2)` 双比特 XX 旋转门，并提供 `ms_gate` / `molmer_sorensen` 作为 Mølmer-Sørensen gate 别名。
- `rxx` 支持矩阵构造、逐门态演化、Torch autograd、QASM 导入导出、QML adjoint gradient、HEA entangler、metrics/QAS/noise 统计路径。
- 新增 `rxx` 与 Mølmer-Sørensen gate 的单元测试，覆盖矩阵定义、别名、QASM round-trip、Torch 梯度、绘图和 HEA/QML 路径。

### Changed

- `VQA_QAS` 的参数移位（parameter-shift）梯度改为复用 `aicir.qml.deriv.psr`，与 VQE/SSVQE/VQD 一致，将移位规则收敛到单一实现；梯度数值保持不变（标准 Pauli 旋转规则，shift=π/2、coefficient=0.5）。
- 改进线路图 layer packing，确保后出现且跨越相同 wire span 的门不会被绘制到前序多比特门之前。
- 线路图中 `Rzz`/`Rxx` 使用完整门名显示；`rzz`/`rxx` 参数值显示在两个对应门框内部。
- 线路图中 `rx`/`ry`/`rz` 及受控旋转门的参数值移入对应门框内部，显示在门名下方。
- 线路图中 `u2`/`u3` 参数值显示在门框下方的小字号文本中；`u2` 参数单行显示，`u3` 参数两行显示。
- README 和子模块文档补充 `Circuit.plot(...)`、`rxx`/Mølmer-Sørensen gate、QASM、Torch autograd、QML AD 和 HEA entangler 说明。

### Tests

- 全量测试通过：`env PYTHONPATH=. pytest`，共 299 项。
- 视觉绘图测试通过：`env PYTHONPATH=. pytest tests/visual/test_visual.py`，共 24 项。

## 2026-06-04

### Added

- 新增 `aicir.vqc.ansatz.hea`：标准 hardware efficient ansatz，支持 `Circuit`、`Parameter`、多种旋转门、entangler 和 topology。
- 新增 `aicir.vqc.ansatz.hea_ti`：trapped-ion HEA-TI ansatz，包含 TFIM/XY 全局演化、power-law 耦合和 HEA-TI 参数数量工具。
- 新增 `aicir.optimizer.params`：VQE/VQA 参数优化器，包括 `GD`、`Adam`、`SPSA`、`COBYLA`、`LBFGSB`、`ScipyMinimize`、`minimize`。
- 新增 `aicir.measure.estimator`：shot-based Pauli-term energy estimator，支持 Hamiltonian Pauli 项拆分、qubit-wise commuting 分组、基变换测量、shots 分配和能量方差统计。
- 新增通用 VQE orchestration：`BasicVQE` 支持 `Circuit`/callable ansatz、`Parameter` 绑定、`Hamiltonian`、backend、`Measure`、shots、density_matrix noise、初态配置、外部 optimizer 和可配置 `energy_estimator`。
- 新增 `aicir.chemistry.molecule`：预置已确认系数的 H2 qubit Hamiltonian，包括 parity 2-qubit、Jordan-Wigner 4-qubit 和 tapered 1-qubit 版本。
- 新增 `aicir.chemistry.README`：记录 chemistry 子包的当前接口、预置分子、示例和设计约束。
- 新增 `aicir.vqc.README`：记录 VQE orchestration、HEA、HEA-TI 和梯度工具配合方式。
- 新增 `demos.vqe_h2_demo`：从 H2 Hamiltonian preset、HEA ansatz、VQE 编排和 optimizer 出发，演示氢分子基态能量求解。

### Changed

- `aicir.optimizer.basic` 改名为 `aicir.optimizer.circuit`，用于线路结构优化相关工具。
- 删除 `optimizer/basic.py`，不再保留旧模块文件。
- 参数优化器只保留简短名称，不保留 `AdamOptimizer`、`SPSAOptimizer` 等长别名。
- chemistry molecule preset 使用 `h2`、`h2_jw`、`h2_tapered` 等简短 canonical 名称，不保留旧长名称或额外别名。
- 将 VQE 文档和注释中的 `dense-matrix` / `dense matrix` 统一为 `dense_matrix`。

### Tests

- 新增 HEA、HEA-TI、参数优化器、PauliEstimator、VQE orchestration、chemistry molecule preset 相关测试。
