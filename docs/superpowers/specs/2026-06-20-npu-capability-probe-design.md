# NPU 硬件能力探测（npu_probe）设计

日期：2026-06-20
状态：已批准设计，待实现

## 目标

提供一个 **NPU 专属的运行时硬件能力探测器**，把 Ascend NPU 实际支持的能力探测成
结构化参数（支持的数据类型/算子、张量维度与尺寸上限、设备内存），缓存到磁盘供后续脚本
复用，并映射为 `aicir.devices.Target` 的执行能力标志。让 NPU 适配代码基于**探测到的事实**
而非硬编码假设来选择执行路径。

非目标（本 spec 不做）：

- 通用跨后端探测（仅 NPU；Numpy/GPU 不在范围）。
- 吞吐量/GFLOPS 基准测试（属性能报告，非正确性输入）。
- 状态向量分片执行内核（独立的更大项目，本探测只产出其所需的*尺寸输入*，不移动振幅）。

## 背景

- `aicir/devices/target.py` 的 `Target` 已是**静态**能力描述（门集、连接拓扑、执行标志），
  但不含任何运行时探测到的 dtype/维度/内存信息。
- `aicir/backends/npu_backend.py` 已处理 Ascend 无复数算子的事实（`_np_conj_T`、
  real/imag 分解的 matmul），但这些是**硬编码**假设，未经探测。
- 多 NPU 当前为任务/数据并行（`owned_indices` 按电路切分），无状态向量分片执行。

## 架构

三个单一职责单元 + 一个脚本：

```
aicir/backends/npu_probe.py     # 新增
  NpuCapabilities                       # 冻结数据类：探测结果（纯数据）
  probe_npu(backend=None, *, allow_cpu_fallback=False, refresh=False) -> NpuCapabilities
  target_from_npu(caps, n_qubits) -> Target   # 纯映射

demos/demo_npu_probe.py          # 新增：可运行探测脚本
tests/backends/test_npu_probe.py # 新增
```

职责分离：`probe_npu`（运行时、有副作用的微算子探测）→ `NpuCapabilities`（纯数据）→
`target_from_npu`（纯映射）。脚本把三者串起来供人使用。`Target` 本身不改动。

## 数据模型 `NpuCapabilities`

冻结数据类。**静态**字段入磁盘缓存，**实时**内存查询不缓存。

```
# 身份 / 失效键
device: str                 # "npu:0" 或 "cpu"（回退）
available: bool             # is_npu_available()
torch_version: str
torch_npu_version: str | None
# dtype / 算子支持（探测）
complex_dtype: str          # "complex64"
supports_complex_matmul: bool
supports_complex_conj: bool
supports_complex_add: bool
needs_real_imag_decomp: bool   # 派生：并非所有复数算子可用
# 尺寸上限（探测/派生）
max_ndim: int | None
max_elements: int | None       # 由 total_memory + dtype 字节数派生
max_qubits: int | None         # floor(log2(max_elements))，单设备
max_qubits_sharded: int | None # max_qubits + floor(log2(world_size))，供未来 SV 分片规划
# 内存
total_memory: int | None       # 字节（静态）
# 运行时
world_size: int                # 取自 NPURuntimeContext
# 元信息
probe_errors: tuple[str, ...]  # 优雅失败的探测项
```

`free_memory()` 是**辅助函数**而非缓存字段（内存随分配变化，实时查询）。

## 探测机制与失败处理

- **dtype/算子**：在设备上分配极小 complex64 张量，各自 `try/except` 跑
  `matmul`/`conj`/`add`；成功 → `True`，异常 → `False` 并追加到 `probe_errors`。
  （这正是 Ascend complex64 缺口当前暴露的方式。）
- **max_ndim**：用递增轴数的 `torch.empty`（微张量）试到抛错为止；开销极小。
- **尺寸上限**：由 `total_memory / complex64 字节数` 派生，**不**做 allocate-until-OOM
  （避免拖垮机器）。`max_qubits = floor(log2(max_elements))`。
- **内存**：`torch.npu.mem_get_info()`（或 CUDA/`torch_npu` 等价物）；不可用则 `None`。
- 任何探测都不会让调用方崩溃——每个失败降级为 `None`/`False` + 一条 `probe_errors`。

## 缓存

- 路径：`$AICIR_CACHE_DIR/npu_caps.json`（默认 `~/.cache/aicir/`），env 可覆盖。
- `probe_npu(..., refresh=False)`：`refresh=False` 且缓存条目失效键
  （`device + torch_version + torch_npu_version`）匹配时加载缓存；否则探测并写回。
- 仅持久化静态字段。失效键不匹配 → 静默重新探测。

## Target 映射

- `target_from_npu(caps, n_qubits)` 用 caps 设置 `Target` 既有执行标志
  （`supports_autodiff=True`、`supports_statevector=True`、density 视 caps 而定）；
  `n_qubits` 取调用方入参或 `caps.max_qubits`。
- **`Target` 不新增字段**——丰富的 dtype/内存细节留在 `NpuCapabilities`，
  `Target` 只消费它能建模的子集。

## 脚本

`demos/demo_npu_probe.py`：跑 `probe_npu`、打印能力表、构建并打印 `Target`；
`--allow-cpu-fallback` 与 `--refresh` 标志，镜像 `demos/demo_npu.py` 的风格。

## 测试

`tests/backends/test_npu_probe.py`，CPU 回退路径（CI 无 NPU）：

- `probe_npu(allow_cpu_fallback=True)` 返回填充好的 `NpuCapabilities`，不崩溃。
- 缓存往返：写 → 读回同一对象；失效键不匹配强制重探。
- `target_from_npu` 正确映射执行标志与 `n_qubits`。
- 失败算子探测记录 `probe_errors` 并降级。
- NPU-only 断言用 `importorskip` / 无设备时跳过。

## 与状态向量分片的关系

本探测是 SV 分片的**前置但不充分**条件：它产出分片所需的尺寸输入
（`max_qubits`、`max_qubits_sharded`、`total_memory`、`world_size`、是否需 real/imag 分解），
但**不**实现分片执行内核（local/global qubit 处理 + 振幅交换集合通信）。后者是独立 spec。
