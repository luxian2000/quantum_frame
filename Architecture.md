# 量子模拟器架构（基于仓库当前骨架）

```text
nexq/
├── channel/                  # 后端抽象与数值内核
│   ├── backends/
│   │   ├── base.py           # Backend 抽象基类
│   │   ├── numpy_backend.py  # 纯 NumPy/CPU 实现
│   │   ├── torch_backend.py  # PyTorch 实现（CPU/CUDA/torch tensors）
│   │   └── npu_backend.py    # NPU-first 后端（Ascend torch_npu 集成 + NPU 特定兼容）
│   ├── states/               # 态表示与转换（StateVector / DensityMatrix）
│   │   ├── state_vector.py
│   │   └── density_matrix.py
│   ├── noise/                # 噪声通道与模型
│   └── operators.py          # PauliOp / PauliString / Hamiltonian 工具
│
├── circuit/                  # 电路构建与 IO（独立于后端）
│   ├── gates.py              # 门矩阵构造（NumPy）
│   ├── model.py              # Circuit 类（门序列、unitary()）
│   └── io/
│       ├── json_io.py        # 电路序列化
│       └── qasm.py           # OpenQASM 2.0/3.0 导入/导出（子集）
│
├── measure/                  # 测量、采样与结果封装
│   ├── measure.py            # 流程调度：构建初态 -> 应用 unitary -> 概率/采样/期望
│   ├── sampler.py            # 概率采样工具
│   └── result.py             # Result 对象（概率、counts、期望、方差、末态）
│
├── algorithms/               # 算法层骨架（占位，按需实现）
│   ├── universal/            # QFT, Grover, QPE 等通用量子算法
│   ├── qml/                  # 量子机器学习
│   ├── variational/          # 变分算法骨架（vqe, qaoa, ansatz）
│   ├── qas/                  # 量子架构搜索
│   ├── chemistry/            # 量子化学工具
│   ├── optimizers/           # 优化器（qubo, sb 子模块）
│   └── wireless/             # 无线通信量子算法
│
├── encoder/                  # 态编码器（angle/amplitude/basis 等）
│
├── execution/                # 执行/调度（当前为空包）
│
├── tests/                    # 单元与集成测试
└── demo.py                   # 小型示例 / NPU 后端验证脚本
```

-- 设计要点（摘要）

- 模块职责分离：

  - `circuit` 负责门定义与矩阵生成，使用 NumPy 生成电路的全局 `unitary()`，与具体后端解耦；
  - `channel` 提供后端实现（`TorchBackend` / `NumpyBackend` / `NPUBackend`），上层通过 `Backend` 抽象调用线性代数与态演化接口；
  - `measure` 负责将电路、后端和采样串联成可重复使用的测量流水线；
  - `algorithms` 与 `encoder` 为高层功能占位，逐步向库中注入 VQE、QAOA、QML、化学映射与编码器实现；
- 可移植性与互操作性：

  - `circuit/io/qasm.py` 支持 OpenQASM 2.0/3.0 的导入导出（子集），便于与外部工具互通；
  - `channel.backends.torch_backend` 与 `channel.backends.numpy_backend` 提供可替换的张量后端；
- NPU 支持与兼容策略（重要）:

  - `NPUBackend` 负责解析分布式环境（`WORLD_SIZE` / `RANK` / `LOCAL_RANK`）并选择 `npu:{LOCAL_RANK}`，提供 `from_distributed_env()` 构造器；
  - 由于目标 Ascend NPU 的 `torch_npu` 在一定版本上对 `complex64` 部分算子（如 `matmul`、`abs`、`trace`、`kron`、`einsum` 等）不直接支持，项目在 `nexq/channel/backends/npu_backend.py` 中引入了 NPU 专属的兼容性 workaround：
    - 当 `device.type == "npu"` 且张量为复数 (`torch.is_complex`) 时，针对性地使用实部/虚部拆分重组来实现以下算子：
      - `matmul`, `apply_unitary`
      - `abs_sq`（|z|^2）、`measure_probs`
      - `kron`（Kronecker 积）
      - `dagger`（共轭转置）、`trace`
      - `inner_product`、`partial_trace`
      - `expectation_sv`、`expectation_dm`
    - 这些改写仅在 NPU 分支启用，CPU/CUDA 路径仍保持 `TorchBackend` 的行为，从而将兼容性改动限制在最小范围并降低回归风险。
- 顶层导出与使用：

  - `from nexq import NPUBackend, TorchBackend, Circuit, Measure, Hamiltonian, StateVector, ...` 可以方便地构建端到端工作流；
  - `demo.py` 用作 NPU 可用性验证脚本（默认严格 NPU 模式），可通过 `--allow-cpu-fallback` 开启 CPU 回退以便本地调试。

-- 当前限制与后续工作建议

- `algorithms` 与 `encoder` 包目前为骨架，建议逐步实现：

  - `algorithms.variational.vqe`、`qaoa`、`ansatz`：提供参数化电路模板、能量测量聚合器与训练循环；
  - `algorithms.optimizers`：实现量子/经典混合优化器（如 SPSA、Adam wrapper、样本高效器）；
  - `encoder`：实现 `BaseEncoder` 抽象，并提供 `AngleEncoder`、`AmplitudeEncoder` 等实现；
- NPU 层面的改进：

  - 持续关注 `torch_npu` 内核对 `complex64` 的支持情况，若未来内核原生支持，应逐步移除 workaround 并验证数值兼容；
  - 增加更多 NPU 上的集成测试（CI 或在可访问的 Ascend 环境中跑回归用例）。

-- 快速开始（示例）

```bash
# 运行 NPU 验证脚本（严格模式，NPU 必须可用）
python demo.py

# 允许回退到 CPU（本地调试）
python demo.py --allow-cpu-fallback
```

```python
from nexq import NPUBackend, Circuit, Measure, hadamard, cnot

backend = NPUBackend.from_distributed_env(fallback_to_cpu=True)
cir = Circuit(hadamard(0), cnot(1, [0]), n_qubits=2)
measure = Measure(backend)
res = measure.run(cir, shots=1024)
print(res.summary())
```

```

```
