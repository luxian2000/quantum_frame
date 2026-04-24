# 量子模拟器架构（目录树版）

```text
nexq/
├── __init__.py
├── algorithms/                         # 算法层（QAS/QML/Variational/Universal 等）
│   ├── __init__.py
│   ├── chemistry/
│   ├── optimizers/
│   │   ├── qubo/
│   │   └── sb/
│   ├── qas/
│   │   └── expressibility.py
│   ├── qml/
│   ├── universal/
│   ├── variational/
│   │   ├── ansatz/
│   │   ├── qaoa/
│   │   └── vqe/
│   └── wireless/
│
├── channel/                            # 后端抽象与量子态/噪声/算符
│   ├── __init__.py
│   ├── operators.py
│   ├── backends/
│   │   ├── __init__.py
│   │   ├── base.py                     # Backend 抽象协议
│   │   ├── numpy_backend.py            # NumPy/CPU
│   │   ├── torch_backend.py            # Torch CPU/CUDA
│   │   └── npu_backend.py              # Ascend NPU + complex64 兼容 workaround
│   ├── noise/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── channels.py
│   │   └── model.py
│   └── states/
│       ├── __init__.py
│       ├── state_vector.py
│       └── density_matrix.py
│
├── circuit/                            # 电路模型、门矩阵、IO
│   ├── __init__.py
│   ├── gates.py                        # gate_to_matrix(..., backend=None)
│   ├── model.py                        # Circuit(..., backend=None), unitary(backend=...)
│   └── io/
│       ├── __init__.py
│       ├── dag.py
│       ├── json_io.py
│       └── qasm.py
│
├── encoder/
│   ├── __init__.py
│   ├── base.py
│   └── encoders.py
│
├── execution/                          # 预留执行编排层（当前为空）
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
- 新增 smoke 脚本：仓库根目录 `smoke_npu_new_path.py`，覆盖：
  - single gate
  - controlled gate
  - parametric gate
  - density-matrix 路径
