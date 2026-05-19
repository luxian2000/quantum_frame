# 量子模拟器架构（目录树版）

```text
nexq/
├── __init__.py
├── algorithms/                         # 算法层（QAS/QML/VQC/Universal 等）
│   ├── __init__.py
│   ├── chemistry/
│   │   └── __init__.py
│   ├── optimization/
│   │   ├── __init__.py
│   │   ├── qubo/
│   │   │   └── __init__.py
│   │   └── sb/
│   │       └── __init__.py
│   ├── qas/
│   │   ├── __init__.py
│   │   ├── README.md
│   │   ├── expressibility.py           # 量子线路在 Haar 测度下的 KL / MMD 指标
│   │   ├── PPO_RB.py                   # PPO with rollback
│   │   ├── PPR_DQL.py                  # Probabilistic Policy Reuse + DQL
│   │   ├── CRLQAS.py                   # Curriculum RL QAS（DDQN + Adam-SPSA）
│   │   └── demo/                       # QAS demo 脚本与 demo 生成文件
│   │       ├── PPO_RB_demo_ghz4.py
│   │       ├── PPO_RB_demo_w3.py
│   │       ├── PPO_RB_demo_dicke3.py
│   │       ├── PPR_DQL_demo_ghz3.py
│   │       ├── CRLQAS_demo_h2.py
│   │       ├── ppo_rb_ghz3_circuit.qasm
│   │       ├── ppo_rb_ghz4_circuit.qasm
│   │       ├── ppo_rb_w3_circuit.qasm
│   │       ├── ppo_rb_dicke3_circuit.qasm
│   │       ├── ppr_dql_ghz3_circuit.qasm
│   │       └── crlqas_h2_circuit.qasm
│   ├── qml/
│   │   └── __init__.py
│   ├── universal/
│   │   └── __init__.py
│   ├── vqc/
│   │   ├── __init__.py
│   │   ├── ansatz/
│   │   │   └── __init__.py
│   │   ├── QAOA.py
│   │   ├── SSVQE.py
│   │   ├── VQD.py
│   │   └── VQE.py
│   └── wireless/
│       └── __init__.py
│
├── channel/                            # 后端抽象、噪声与算符
│   ├── __init__.py
│   ├── operators.py
│   ├── backends/
│   │   ├── __init__.py
│   │   ├── base.py                     # Backend 抽象协议
│   │   ├── numpy_backend.py            # NumPy / CPU
│   │   ├── torch_backend.py            # Torch CPU / CUDA
│   │   └── npu_backend.py              # Ascend NPU + complex64 workaround
│   ├── noise/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── channels.py
│   │   └── model.py
│   └── states/                         # 历史状态层目录（当前主要实现位于 core/state.py）
│
├── core/                               # 电路模型、量子态、门矩阵、IO
│   ├── __init__.py
│   ├── state.py                        # 纯态表示 |ψ>（支持自动归一化）
│   ├── density.py                      # 密度矩阵表示 ρ
│   ├── gates.py                        # gate_to_matrix(..., backend=None)
│   ├── circuit.py                      # Circuit(..., backend=None), unitary(backend=...)
│   └── io/
│       ├── __init__.py
│       ├── dag.py                      # 量子线路的有向图表示
│       ├── json_io.py
│       └── qasm.py                     # 量子线路与 OpenQASM 2.0/3.0 互转
│
├── encoder/
│   ├── __init__.py
│   ├── abstract.py
│   ├── amplitude.py
│   ├── angle.py
│   ├── basis.py
│   └── demo/
│       └── encode_1234_demo.py
│
├── optimizer/                          # 量子线路优化器（包名为 nexq.optimizer）
│   ├── __init__.py
│   ├── basic.py
│   └── README.md
│
└── measure/                            # 执行与测量入口
  ├── __init__.py
  ├── measure.py                      # 优先 circuit.backend，再回退 Measure.backend
  ├── sampler.py
  └── result.py

```

## 执行路径说明

- 旧路径：`Circuit(...)` + `Measure(backend).run(circuit)`
- 新路径：`Circuit(..., backend=backend)` + `Measure(...).run(circuit)`
  - 在新路径中，前端矩阵组装与后端执行优先保持同一 XPU。

## NPU 兼容策略（摘要）

- `NPUBackend` 对 NPU 上部分 `complex64` 算子缺失做了拆分回退（real/imag）。
- 仓库根目录 smoke 脚本：`smoke_npu_new_path.py`，覆盖：
  - single gate
  - controlled gate
  - parametric gate
  - density-matrix 路径
