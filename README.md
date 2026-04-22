# my_frame

一个用于量子门矩阵构造与简单量子线路计算的 Python 仓库，提供两套实现：

- PyTorch 实现
- MindSpore 实现

当前仓库中的命名风格已按 PEP 8 统一（函数与变量采用 snake_case，常量采用 UPPER_SNAKE_CASE）。

## 项目结构

- [basic_torch.py](basic_torch.py)
  - PyTorch 版本的基础量子门与矩阵工具。
  - 包含单比特门、多比特受控门、门到矩阵转换等底层能力。
- [define_torch.py](define_torch.py)
  - PyTorch 版本的线路层封装。
  - 包含线路组合、期望值计算、门字典构造函数（如 hadamard、cnot、u3）。
- [basic_mind.py](basic_mind.py)
  - MindSpore 版本的基础量子门与矩阵工具。
- [define_mind.py](define_mind.py)
  - MindSpore 版本的线路层封装。
- [run_tests.py](run_tests.py)
  - 统一测试入口脚本。
  - 负责集中执行 Torch 与 MindSpore 路径的示例测试。

## 功能概览

### 1. 基础矩阵与态工具

- matrix_product: 多矩阵连乘
- tensor_product: 张量积（Kronecker 积）
- dagger: 共轭转置
- identity: n 比特单位矩阵

### 2. 基础量子门（底层）

- 单比特门：_pauli_x、_pauli_y、_pauli_z、_hadamard、_rx、_ry、_rz、_s_gate、_t_gate、_u2、_u3
- 多比特门：_cx、_cy、_cz、_crx、_cry、_crz、_swap、_toffoli、_rzz
- 门映射：gate_to_matrix

### 3. 线路层封装

- phi_0: 生成全零态
- expectation: 计算期望值
- circuit: 按门序列生成整体线路矩阵
- 门构造函数：pauli_x、hadamard、cx/cnot、u3 等

## 运行环境

- Python 3.10+
- 必选依赖（Torch 路径）：
  - torch
  - numpy
- 可选依赖（MindSpore 路径）：
  - mindspore

说明：

- [basic_mind.py](basic_mind.py) 与 [define_mind.py](define_mind.py) 中默认设置了 Ascend 设备上下文。
- 如果本机没有对应 MindSpore/Ascend 环境，MindSpore 路径可能无法运行。

## 安装依赖

仅运行 Torch 路径时：

    pip install torch numpy

若需要 MindSpore 路径，请按官方文档安装与你设备匹配的 MindSpore 版本。

## 快速开始

在仓库根目录执行统一测试：

    python run_tests.py

测试行为：

- 会先执行 Torch 相关测试。
- 会尝试执行 MindSpore 相关测试。
- 若缺少 mindspore，脚本会提示跳过 MindSpore 测试，不影响 Torch 测试结果。

## 使用示例（Torch 线路层）

下面示例展示如何构造 2 比特 Bell 线路并计算期望值：

    from define_torch import phi_0, circuit, hadamard, cnot, expectation, identity
    import torch

    psi0 = phi_0(2)
    bell_u = circuit(
        hadamard(0),
        cnot(1, [0]),
        num_qubits=2,
    )
    bell_state = torch.matmul(bell_u, psi0)
    exp_i = expectation(bell_state, identity(2))
    print(exp_i)

## 开发说明

- 测试代码已从业务脚本中抽离，统一维护在 [run_tests.py](run_tests.py)。
- 建议新增功能时：
  - 在基础层实现门矩阵与映射逻辑
  - 在定义层补充门构造函数或线路接口
  - 在 [run_tests.py](run_tests.py) 中补充对应测试段

## 常见问题

1. 运行 MindSpore 相关代码时报错 No module named mindspore

- 原因：当前环境未安装 MindSpore。
- 解决：安装与你硬件和系统匹配的 MindSpore 版本，或仅使用 Torch 路径。

2. MindSpore 安装后仍报设备相关错误

- 原因：默认上下文为 Ascend 设备，与你本机环境不匹配。
- 解决：根据你的运行环境调整 [basic_mind.py](basic_mind.py) 与 [define_mind.py](define_mind.py) 的 context.set_context 配置。
