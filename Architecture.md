# 量子模拟器构架

```text
quantum_sim/
├── core/
│   ├── backends/             # 计算后端解耦
│   │   ├── base.py           # 后端抽象基类
│   │   ├── numpy_backend.py  # 基础 CPU 实现
│   │   └── torch_backend.py  # 与深度学习深度集成的后端
│   ├── states/               # 物理实体描述
│   │   ├── state_vector.py
│   │   ├── density_matrix.py
│   │   └── mps.py            # 矩阵乘积态
│   ├── gates/                # 门操作实现
│   ├── noise/                # 噪声模型
│   ├── operators.py          # 泡利算符类、哈密顿量构建
│   └── tensor_network/
│
├── circuit/
│   ├── dag/                  # 有向无环图表示，方便电路优化
│   ├── compiler/             # 核心编译流水线
│   │   ├── passes/           # 优化 Pass：合并相邻门、抵消恒等变换
│   │   ├── transpiler.py     # 针对特定基门集的转换
│   │   └── scheduler.py      # 门操作执行排序
│   ├── circuit.py            # 用户接口类
│   └── io/                   # 持久化与转换
│       ├── qasm.py           # OpenQASM 2.0/3.0 支持
│       └── json_io.py        # 电路序列化
│
├── measure/                  # 测量与结果生成模块
│   ├── measure.py            # 调度后端执行逻辑并产生测量结果
│   ├── result.py             # 统一的测量结果对象（均值、方差、样本）
│   └── sampler.py            # 采样器逻辑
│
├── algorithms/
│   ├── universal/            # QFT, Search, QPE, HHL, QW, QSVT ...
│   ├── QML/
│   │   ├── traditional QML/  # regression, SVM, kernel ...
│   │   ├── deep QML/         # neural networks, reservior ...
│   │   └── QRL/
│   ├── variational/
│   │   ├── VQE
│   │   ├── QAOA
│   │   └── ansatz/           # 预定义的变分线路模板 (HEA 等)
│   ├── QAS/                  # 量子架构搜索
│   ├── chemistry/            # 特定领域：费米子映射、分子哈密顿量
│   └── optimizers/           # QUBO/HUBO/SB 等优化算法
│
├── interface/                # 跨框架生态适配
│   ├── torch_connector.py    # 将量子电路包装为 Torch Layer
│   └── qiskit_adapter.py     # 与 Qiskit 互操作
│
├── utils/
├── paper_codes/
├── benchmarks/               # 性能测试基准（对比其他仿真器）
└── tests/
```
