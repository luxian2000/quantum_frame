# aicir.optimization.qubo

本包提供纯 Python 实现的 QUBO（二次无约束二值优化）建模工具，用于组合优化问题。

建模 API 改编自独立的 `qubo-modeling` 原型，刻意与电路模拟器后端解耦。无需
导入 GPU/NPU 后端或安装 Torch 即可使用。

## 基本用法

```python
from aicir.optimization.qubo import Model, ModelContext, one_hot

ctx = ModelContext()
x = ctx.binary_array("x", 3)

model = Model(ctx.zero())
model.add_constraint(one_hot(x, penalty=5.0))

qubo, offset = model.to_qubo()
ising = model.to_ising()
terms, qaoa_offset, variable_names = model.to_qaoa_terms()
```

## Hamiltonian 适配器

```python
from aicir.optimization.qubo import model_to_hamiltonian

hamiltonian = model_to_hamiltonian(model)
```

该适配器将一、二比特的 Z 项转换为 `aicir.operators.Hamiltonian`。默认会将
Ising 全局偏置量保留为单位算符项，以保持能量精确一致；若只需要 QAOA 相位算符
（可差一个全局相位），可传入 `include_offset=False` 省略偏置量。

## BasicQAOA 辅助函数

```python
from aicir.optimization.qubo import model_to_qaoa_matrix, run_qubo_qaoa

matrix, n_qubits = model_to_qaoa_matrix(model)
result = run_qubo_qaoa(model, p=1, max_iters=50, lr=0.05, seed=123)
```

这些辅助函数复用现有的 `aicir.vqc.BasicQAOA` 稠密矩阵路径，仅适合小规模示例
与初步集成测试。大规模 QUBO 实例应避免稠密的 `2^n x 2^n` 矩阵，改用电路级或
稀疏 QAOA 实现。

可运行的端到端示例：

```bash
python demos/qubo_qaoa_demo.py
```

## 分解（Split）

当目标多项式的变量间存在天然独立的子结构时，可以将其拆分为若干互不相关的
子多项式分别求解，再把各子问题的最优值相加得到全局最优值。本包提供两种粒度
不同的拆分方法：

### `split_polynomial` —— 按连通分量拆分

`split_polynomial` 把一个 `Polynomial`（可由 `Model.polynomial()` 得到）按变量
共现关系拆成若干**变量互不相交**的子多项式。任意两个变量只要出现在同一个单
项式中就视为相连；由此得到的连通分量彼此独立，因此各分量的最小值之和就是原
多项式的最小值：

```python
from aicir.optimization.qubo import split_polynomial

pieces = split_polynomial(model.polynomial())
# min(f) == sum(min(piece) for piece in pieces)
```

- 每个单项式恰好落入一个分量，各分量精确划分了原多项式的全部单项式（含常数
  项，作为独立一块返回）。
- 支持任意阶多项式（不局限于二次）。
- 若某个连通块内部本身存在“冲突”（例如一对变量同时被惩罚同为 0 和同为 1），
  `split_polynomial` 无法进一步拆开它——这正是 `split_cores` 要解决的问题。

### `split_cores` —— Billionnet–Jaumard 二次核心分解

`split_cores` 是更强的、感知符号结构的拆分方法，可以把 `split_polynomial`
留下的单个连通块继续拆细。它仅适用于**二次（degree ≤ 2）**多项式：

```python
from aicir.optimization.qubo import split_cores

cores = split_cores(model.polynomial())
# min(f) == sum(min(core) for core in cores)
```

实现流程：

1. 将多项式经初等互补（`x = 1 - x̄`）转换为 posiform（各系数非负、外加一个常数）；
2. 构建 posiform 对应的 2-SAT 蕴含图（变量与其补的字面量为图节点）；
3. 用 Tarjan 算法求强连通分量；
4. 保留同时含有某变量及其补的“自对偶”分量（即真正冲突、无法再分解的
   *hard core*），把每个分量还原为 x 空间的子多项式；其余（一致可满足的）单
   项式可证明最小值为 0，直接舍弃。

与 `split_polynomial` 相比，`split_cores` 的约定**更弱**：它只保证
`min(f) == sum(min(core) for core in cores)`，**不保证**对原多项式单项式的
精确划分（被舍弃的一致可满足部分不会出现在返回结果中）。

## 覆盖范围

本次首批集成已包含：

- 二值变量与稀疏多项式表达式
- 通过 `ModelContext` 实现的模型级变量注册表
- 常见约束：one-hot、cardinality、assignment、带松弛变量的线性不等式
- 结构化目标函数
- 底层 `QuboBuilder`
- TSP、图着色、背包问题的构造器
- QUBO、Ising、稀疏矩阵、QAOA 项的导出
- 转换为 `aicir.operators.Hamiltonian`
- `aicir.vqc.BasicQAOA` 的稠密矩阵辅助函数
- 解的解码与小规模模型的暴力枚举分析
- 按连通分量拆分（`split_polynomial`）与 Billionnet–Jaumard 二次核心分解
  （`split_cores`）

本次未包含：

- PyQUBO / Fixstars Amplify 基准测试脚本
- 独立打包构建文件
- 面向大规模 QUBO 的可扩展稀疏或电路级 QAOA 执行

下一步集成应为更大规模 QUBO 实例添加可扩展的稀疏或电路级 QAOA 路径。
