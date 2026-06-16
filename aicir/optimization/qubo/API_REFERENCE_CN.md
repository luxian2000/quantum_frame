# QUBO 建模工具包中文 API 接口文档

本文档面向第一版 `qubo-modeling` 软件包，覆盖 `qubo_modeling.__all__`
中导出的公开类、函数和常用方法。除特别说明外，推荐用户通过顶层包导入：

```python
from qubo_modeling import Model, ModelContext, one_hot
```

## 1. 建模主流程

典型建模流程如下：

```python
from qubo_modeling import Model, ModelContext, linear_objective, one_hot

ctx = ModelContext()
x = ctx.binary_array("x", 3)

model = Model(ctx.zero())
model.add_constraint(one_hot(x, penalty=10.0, label="choose_one"))
model.add_objective(linear_objective([(1.0, x[0]), (2.0, x[1]), (-1.0, x[2])]))

qubo, offset = model.to_qubo_indices()
```

推荐优先使用 `ModelContext` 创建变量。它会维护独立的变量注册表，避免不同实验之间的变量名和变量编号相互污染。

## 2. 变量注册与上下文

### `VariableMetadata`

```python
VariableMetadata(
    name: str,
    kind: str = "binary",
    role: str = "decision",
    source: str | None = None,
)
```

变量元数据。

- `name`：变量名，例如 `"x[0]"`。
- `kind`：变量类型，当前主要使用 `"binary"`。
- `role`：变量角色，常见值为 `"decision"` 或 `"auxiliary"`。
- `source`：变量来源说明，常用于记录 slack/auxiliary 变量由哪个约束生成。

### `VariableRegistry`

```python
VariableRegistry()
```

变量注册表，负责维护变量名、变量编号和变量元数据之间的映射。

#### `get_or_create`

```python
registry.get_or_create(
    name: str,
    kind: str = "binary",
    role: str = "decision",
    source: str | None = None,
) -> int
```

获取变量编号。如果变量名不存在，则创建一个新变量。

#### `name`

```python
registry.name(var_id: int) -> str
```

根据变量编号返回变量名。

#### `names`

```python
registry.names() -> list[str]
```

返回所有变量名，顺序与变量编号一致。

#### `metadata`

```python
registry.metadata(var_id: int) -> VariableMetadata
```

返回指定变量编号的元数据。

#### `variables_by_role`

```python
registry.variables_by_role(role: str) -> list[str]
```

返回指定角色的变量名列表。

#### `auxiliary_names`

```python
registry.auxiliary_names() -> list[str]
```

返回所有辅助变量名。

### `ModelContext`

```python
ModelContext(registry: VariableRegistry = <factory>)
```

建模上下文，封装变量注册表，并提供创建变量、整数变量、Builder 和解码结果的便捷方法。

#### `binary`

```python
ctx.binary(name: str)
```

创建一个二进制决策变量，返回 `Polynomial`。

#### `auxiliary_binary`

```python
ctx.auxiliary_binary(name: str, source: str | None = None)
```

创建一个二进制辅助变量，变量角色为 `"auxiliary"`。

#### `binary_array`

```python
ctx.binary_array(prefix: str, shape: int | tuple[int, ...]) -> list
```

创建一维或多维二进制变量数组。

示例：

```python
x = ctx.binary_array("x", 3)        # x[0], x[1], x[2]
y = ctx.binary_array("y", (2, 3))   # y[0,0] ... y[1,2]
```

#### `integer`

```python
ctx.integer(
    name: str,
    lower_bound: int = 0,
    upper_bound: int | None = None,
    encoding: str = "log",
    role: str = "decision",
    source: str | None = None,
)
```

创建编码整数变量，返回 `EncodedInteger`。`encoding` 支持 `"log"` 和 `"unary"`。

#### `auxiliary_integer`

```python
ctx.auxiliary_integer(
    name: str,
    lower_bound: int = 0,
    upper_bound: int | None = None,
    encoding: str = "log",
    source: str | None = None,
)
```

创建辅助整数变量，常用于不等式 slack 变量。

#### `zero`

```python
ctx.zero()
```

返回当前上下文中的零多项式，常用于初始化空目标模型：

```python
model = Model(ctx.zero())
```

#### `qubo_builder`

```python
ctx.qubo_builder()
```

创建绑定当前变量注册表的 `QuboBuilder`。

#### `decode_solution`

```python
ctx.decode_solution(
    assignment,
    integers=None,
    include_auxiliary: bool = False,
)
```

根据当前上下文的变量注册表解码求解结果。

## 3. 多项式、二进制变量与线性表达式

### `Polynomial`

```python
Polynomial(
    terms: dict[tuple[int, ...], float],
    registry: VariableRegistry = <factory>,
)
```

稀疏多项式表达式。内部以变量编号元组为 key，例如：

- `()` 表示常数项。
- `(0,)` 表示线性项。
- `(0, 1)` 表示二次项。

二进制变量满足 `x^2 = x`，因此同一变量重复相乘会被规约。

支持运算：

```python
p + q
p - q
-p
p * q
p * 2.0
p ** 2
```

#### `Polynomial.constant`

```python
Polynomial.constant(value, registry=GLOBAL_REGISTRY) -> Polynomial
```

创建常数多项式。

#### `Polynomial.variable`

```python
Polynomial.variable(
    name: str,
    registry=GLOBAL_REGISTRY,
    role: str = "decision",
    source: str | None = None,
) -> Polynomial
```

创建二进制变量多项式。

#### `clean`

```python
poly.clean(eps: float = 1e-12) -> Polynomial
```

移除绝对值小于 `eps` 的近零系数。

#### `degree`

```python
poly.degree() -> int
```

返回多项式最高次数。

#### `variables`

```python
poly.variables() -> list[str]
```

返回多项式中出现的变量名。

#### `to_qubo_indices`

```python
poly.to_qubo_indices() -> tuple[dict[tuple[int, int], float], float]
```

导出以变量编号表示的 QUBO 系数和常数偏移。

#### `to_qubo`

```python
poly.to_qubo() -> tuple[dict[tuple[str, str], float], float]
```

导出以变量名表示的 QUBO 系数和常数偏移。

### `Binary`

```python
Binary(
    name: str,
    registry: VariableRegistry = GLOBAL_REGISTRY,
    role: str = "decision",
    source: str | None = None,
) -> Polynomial
```

创建一个二进制变量。对于正式模型，推荐使用 `ctx.binary(name)`。

### `binary_array`

```python
binary_array(
    prefix: str,
    shape: int | tuple[int, ...],
    registry: VariableRegistry = GLOBAL_REGISTRY,
    role: str = "decision",
    source: str | None = None,
) -> list
```

创建二进制变量数组。对于正式模型，推荐使用 `ctx.binary_array(...)`。

### `Sum`

```python
Sum(values: Iterable[Polynomial | int | float]) -> Polynomial
```

对一组多项式或数值求和。

### `LinearExpression`

```python
LinearExpression(
    terms: dict[int, float],
    offset: float = 0.0,
    registry: VariableRegistry = <factory>,
)
```

线性表达式，适合用于等式/不等式约束中的加权和。

#### `LinearExpression.from_terms`

```python
LinearExpression.from_terms(
    weighted_terms: Iterable[tuple[float, Polynomial]],
    offset: float = 0.0,
) -> LinearExpression
```

从 `(系数, 变量)` 列表构造线性表达式。

#### `LinearExpression.from_integer`

```python
LinearExpression.from_integer(
    value: EncodedInteger,
    scale: float = 1.0,
) -> LinearExpression
```

从编码整数变量构造线性表达式。

#### `weighted_terms`

```python
linear.weighted_terms() -> list[tuple[float, Polynomial]]
```

返回 `(系数, 变量)` 形式的线性项。

#### `expression`

```python
linear.expression() -> Polynomial
```

转换为 `Polynomial`。

### `Linear`

```python
Linear(
    weighted_terms: Iterable[tuple[float, Polynomial]] | EncodedInteger,
    offset: float = 0.0,
) -> LinearExpression
```

创建线性表达式的便捷函数。

## 4. 模型与目标函数

### `Model`

```python
Model(
    objective: Polynomial,
    constraints: list[Constraint] = <factory>,
    objective_fragments: list[ObjectiveFragment] = <factory>,
)
```

QUBO 模型对象。包含目标函数、约束和结构化目标片段。

#### `add_constraint`

```python
model.add_constraint(constraint: Constraint) -> None
```

添加一个约束。

#### `add_constraints`

```python
model.add_constraints(constraints: list[Constraint]) -> None
```

批量添加约束。

#### `add_objective`

```python
model.add_objective(objective: ObjectiveFragment | Polynomial) -> None
```

添加目标函数片段。若传入 `ObjectiveFragment`，可走高效 Builder 路径。

#### `polynomial`

```python
model.polynomial() -> Polynomial
```

返回完整惩罚后目标多项式。

#### `to_qubo`

```python
model.to_qubo() -> tuple[dict[tuple[str, str], float], float]
```

导出变量名版本 QUBO。

#### `to_qubo_indices`

```python
model.to_qubo_indices(
    clean: bool = True,
    copy: bool = True,
) -> tuple[dict[tuple[int, int], float], float]
```

导出变量编号版本 QUBO。`clean=True` 会删除近零项，`copy=True` 会返回副本以避免外部修改内部数据。

#### `to_sparse_matrix`

```python
model.to_sparse_matrix(
    symmetric: bool = False,
    compact: bool = True,
) -> SparseMatrixCOO
```

导出 COO 稀疏矩阵。

#### `to_qubo_builder`

```python
model.to_qubo_builder() -> QuboBuilder
```

将模型编译为底层 `QuboBuilder`。

#### `to_ising_indices`

```python
model.to_ising_indices(compact: bool = True) -> IsingModel
```

导出以变量编号表示的 Ising 模型。

#### `to_ising`

```python
model.to_ising() -> dict[str, object]
```

导出以变量名表示的 Ising 字典。

#### `to_qaoa_terms`

```python
model.to_qaoa_terms(
    compact: bool = True,
) -> tuple[list[QAOATerm], float, list[str] | None]
```

导出 QAOA 成本哈密顿量项、常数偏移和变量名列表。

### `ObjectiveFragment`

```python
ObjectiveFragment(
    expression: Polynomial | None = None,
    builder_action = None,
    expression_factory = None,
    registry: VariableRegistry | None = None,
)
```

结构化目标片段。普通用户通常不需要直接构造它，而是使用 `linear_objective` 或 `quadratic_objective`。

#### `expression`

```python
fragment.expression() -> Polynomial
```

返回目标片段对应的多项式。

#### `add_to_builder`

```python
fragment.add_to_builder(builder: QuboBuilder) -> None
```

将目标片段直接写入 Builder。

### `linear_objective`

```python
linear_objective(
    weighted_variables: Iterable[tuple[float, Polynomial]],
) -> ObjectiveFragment
```

构造线性目标函数片段。

示例：

```python
model.add_objective(linear_objective([(3.0, x[0]), (-1.0, x[1])]))
```

### `quadratic_objective`

```python
quadratic_objective(
    weighted_pairs: Iterable[tuple[float, Polynomial, Polynomial]],
) -> ObjectiveFragment
```

构造二次目标函数片段。

示例：

```python
model.add_objective(quadratic_objective([(2.0, x[0], x[1])]))
```

## 5. 约束 API

### `Constraint`

```python
Constraint(
    expression: Polynomial | None = None,
    penalty: float = 1.0,
    label: str | None = None,
    builder_action = None,
    expression_factory = None,
    registry: VariableRegistry | None = None,
)
```

约束对象。约束最终会以惩罚项形式加入 QUBO。

普通用户通常不直接构造 `Constraint`，而是使用下面的约束函数。

#### `expression`

```python
constraint.expression() -> Polynomial
```

返回未乘惩罚系数或已封装好的约束多项式。

#### `as_penalty`

```python
constraint.as_penalty() -> Polynomial
```

返回用于加入目标函数的惩罚多项式。

#### `add_to_builder`

```python
constraint.add_to_builder(builder: QuboBuilder) -> None
```

将约束直接写入 Builder。

### `one_hot`

```python
one_hot(
    variables: Iterable[Polynomial],
    penalty: float = 1.0,
    label: str | None = None,
) -> Constraint
```

构造 one-hot 约束：变量中恰好有一个取 1。

数学形式：

```text
penalty * (sum(x_i) - 1)^2
```

### `cardinality`

```python
cardinality(
    variables: Iterable[Polynomial],
    count: int,
    penalty: float = 1.0,
    label: str | None = None,
) -> Constraint
```

构造基数约束：变量中恰好 `count` 个取 1。

数学形式：

```text
penalty * (sum(x_i) - count)^2
```

### `at_most_one`

```python
at_most_one(
    variables: Iterable[Polynomial],
    penalty: float = 1.0,
    label: str | None = None,
) -> Constraint
```

构造至多选一个约束：`sum(x_i) <= 1`。

### `at_least_one`

```python
at_least_one(
    variables: Iterable[Polynomial],
    slack_prefix: str = "at_least_slack",
    penalty: float = 1.0,
    label: str | None = None,
) -> tuple[Constraint, list[Polynomial]]
```

构造至少选一个约束：`sum(x_i) >= 1`。

返回 `(constraint, slack_variables)`。该约束会自动引入 slack 辅助变量。

### `weighted_equality`

```python
weighted_equality(
    weighted_variables: Iterable[tuple[float, Polynomial]] | LinearExpression,
    target: float,
    penalty: float = 1.0,
    label: str | None = None,
) -> Constraint
```

构造加权等式约束。

数学形式：

```text
penalty * (sum(w_i x_i) - target)^2
```

### `linear_inequality`

```python
linear_inequality(
    weighted_variables: Iterable[tuple[float, Polynomial]] | LinearExpression,
    upper_bound: int,
    slack_prefix: str = "slack",
    penalty: float = 1.0,
    label: str | None = None,
) -> tuple[Constraint, list[Polynomial]]
```

构造线性不等式约束：

```text
sum(w_i x_i) <= upper_bound
```

返回 `(constraint, slack_variables)`。当前版本假设权重主要为非负整数场景。

### `integer_equality`

```python
integer_equality(
    expression: LinearExpression | EncodedInteger | Iterable[tuple[float, Polynomial]],
    target: float,
    penalty: float = 1.0,
    label: str | None = None,
) -> Constraint
```

构造整数表达式等式约束。

### `integer_less_equal`

```python
integer_less_equal(
    expression: LinearExpression | EncodedInteger | Iterable[tuple[float, Polynomial]],
    upper_bound: int,
    slack_prefix: str = "integer_le_slack",
    penalty: float = 1.0,
    label: str | None = None,
) -> tuple[Constraint, list[Polynomial]]
```

构造整数表达式小于等于约束，并自动引入 slack 辅助变量。

### `one_hot_rows`

```python
one_hot_rows(
    matrix: Iterable[Iterable[Polynomial]],
    penalty: float = 1.0,
    label: str | None = None,
) -> list[Constraint]
```

对矩阵每一行添加 one-hot 约束。

### `one_hot_columns`

```python
one_hot_columns(
    matrix: Iterable[Iterable[Polynomial]],
    penalty: float = 1.0,
    label: str | None = None,
) -> list[Constraint]
```

对矩阵每一列添加 one-hot 约束。

### `assignment_matrix`

```python
assignment_matrix(
    matrix: Iterable[Iterable[Polynomial]],
    penalty: float = 1.0,
    label: str | None = None,
) -> list[Constraint]
```

对矩阵同时添加行 one-hot 和列 one-hot 约束，适合任务分配类问题。

### `permutation`

```python
permutation(
    matrix: Iterable[Iterable[Polynomial]],
    penalty: float = 1.0,
    label: str | None = None,
) -> list[Constraint]
```

排列矩阵约束，当前语义与 `assignment_matrix` 类似，适合 TSP 等排列编码。

## 6. 整数变量 API

### `EncodedInteger`

```python
EncodedInteger(
    name: str,
    lower_bound: int,
    upper_bound: int,
    bits: list[Polynomial],
    weights: list[int],
    registry: VariableRegistry,
)
```

编码整数变量。它本身不是单个 QUBO 变量，而是由若干二进制 bit 变量表示。

#### `expression`

```python
integer.expression() -> Polynomial
```

返回整数变量对应的多项式表达式。

#### `weighted_terms`

```python
integer.weighted_terms(scale: float = 1.0) -> list[tuple[float, Polynomial]]
```

返回整数变量的 bit 加权项。

#### `linear_expression`

```python
integer.linear_expression(scale: float = 1.0)
```

返回整数变量对应的 `LinearExpression`。

#### `bit_names`

```python
integer.bit_names() -> list[str]
```

返回该整数变量底层 bit 变量名。

### `Integer`

```python
Integer(
    name: str,
    lower_bound: int = 0,
    upper_bound: int | None = None,
    encoding: str = "log",
    registry: VariableRegistry = GLOBAL_REGISTRY,
    role: str = "decision",
    source: str | None = None,
) -> EncodedInteger
```

创建整数变量。`encoding="log"` 表示对数编码，`encoding="unary"` 表示一元编码。

### `LogEncodedInteger`

```python
LogEncodedInteger(
    name: str,
    lower_bound: int = 0,
    upper_bound: int | None = None,
    registry: VariableRegistry = GLOBAL_REGISTRY,
    role: str = "decision",
    source: str | None = None,
) -> EncodedInteger
```

创建对数编码整数变量。通常是推荐默认选择。

### `UnaryEncodedInteger`

```python
UnaryEncodedInteger(
    name: str,
    lower_bound: int = 0,
    upper_bound: int | None = None,
    registry: VariableRegistry = GLOBAL_REGISTRY,
    role: str = "decision",
    source: str | None = None,
) -> EncodedInteger
```

创建一元编码整数变量。适合范围较小、希望编码结构更直观的场景。

## 7. QUBO Builder 底层接口

### `QuboBuilder`

```python
QuboBuilder(
    registry: VariableRegistry = <factory>,
    qubo: dict[tuple[int, int], float] = <factory>,
    offset: float = 0.0,
)
```

性能导向的 QUBO 系数写入器。它直接维护：

- `qubo`：以变量编号为 key 的 QUBO 系数字典。
- `offset`：常数偏移。
- `registry`：变量注册表。

如果已经知道问题结构，使用 Builder 可以减少中间 `Polynomial` 对象的创建。

#### `add_offset`

```python
builder.add_offset(value: float) -> None
```

添加常数偏移。

#### `add_linear`

```python
builder.add_linear(var_id: int, coeff: float) -> None
```

添加线性项 `coeff * x_i`。

#### `add_quadratic`

```python
builder.add_quadratic(left_id: int, right_id: int, coeff: float) -> None
```

添加二次项 `coeff * x_i * x_j`。若 `left_id == right_id`，按二进制变量性质作为线性项处理。

#### `add_linear_terms`

```python
builder.add_linear_terms(weighted_ids: Iterable[tuple[float, int]]) -> None
```

批量添加线性项。

#### `add_quadratic_terms`

```python
builder.add_quadratic_terms(
    weighted_ids: Iterable[tuple[float, int, int]],
) -> None
```

批量添加二次项。

#### `add_polynomial`

```python
builder.add_polynomial(
    polynomial: Polynomial,
    scale: float = 1.0,
) -> None
```

将多项式写入 Builder。多项式最高次数必须不超过 2。

#### `add_cardinality_penalty`

```python
builder.add_cardinality_penalty(
    var_ids: Iterable[int],
    count: int,
    penalty: float = 1.0,
) -> None
```

直接添加基数约束惩罚项。

#### `add_at_most_one_penalty`

```python
builder.add_at_most_one_penalty(
    var_ids: Iterable[int],
    penalty: float = 1.0,
) -> None
```

直接添加至多选一个惩罚项。

#### `add_at_least_one_penalty`

```python
builder.add_at_least_one_penalty(
    var_ids: Iterable[int],
    slack_prefix: str = "at_least_slack",
    penalty: float = 1.0,
) -> list[int]
```

直接添加至少选一个惩罚项，并返回新创建的 slack 变量编号。

#### `add_weighted_equality_penalty`

```python
builder.add_weighted_equality_penalty(
    weighted_ids: Iterable[tuple[float, int]],
    target: float,
    penalty: float = 1.0,
) -> None
```

直接添加加权等式惩罚项。

#### `add_linear_inequality_penalty`

```python
builder.add_linear_inequality_penalty(
    weighted_ids: Iterable[tuple[float, int]],
    upper_bound: int,
    slack_prefix: str = "slack",
    penalty: float = 1.0,
) -> list[int]
```

直接添加线性不等式惩罚项，并返回 slack 变量编号列表。

#### `to_qubo_indices`

```python
builder.to_qubo_indices(
    clean: bool = True,
    copy: bool = True,
) -> tuple[dict[tuple[int, int], float], float]
```

导出变量编号版本 QUBO。

#### `to_qubo`

```python
builder.to_qubo() -> tuple[dict[tuple[str, str], float], float]
```

导出变量名版本 QUBO。

#### `to_sparse_matrix`

```python
builder.to_sparse_matrix(
    symmetric: bool = False,
    compact: bool = True,
) -> SparseMatrixCOO
```

导出 COO 稀疏矩阵。

#### `to_ising_indices`

```python
builder.to_ising_indices(compact: bool = True) -> IsingModel
```

导出 Ising 模型。

#### `to_qaoa_terms`

```python
builder.to_qaoa_terms(
    compact: bool = True,
) -> tuple[list[QAOATerm], float, list[str] | None]
```

导出 QAOA 成本哈密顿量项。

## 8. 导出后端与数据结构

### `SparseMatrixCOO`

```python
SparseMatrixCOO(
    row: list[int],
    col: list[int],
    data: list[float],
    shape: tuple[int, int],
    offset: float = 0.0,
    variable_names: list[str] | None = None,
    variable_metadata: list[VariableMetadata] | None = None,
)
```

COO 格式稀疏矩阵。

- `row`：非零元素行索引。
- `col`：非零元素列索引。
- `data`：非零元素值。
- `shape`：矩阵形状。
- `offset`：QUBO 常数偏移。

#### `to_dense`

```python
matrix.to_dense() -> list[list[float]]
```

转换为普通二维列表。

### `IsingModel`

```python
IsingModel(
    h: dict[int, float],
    J: dict[tuple[int, int], float],
    offset: float = 0.0,
    variable_names: list[str] | None = None,
    variable_metadata: list[VariableMetadata] | None = None,
)
```

Ising 模型：

```text
E(z) = offset + sum_i h_i z_i + sum_{i<j} J_ij z_i z_j
```

其中 `z_i ∈ {-1, +1}`。

#### `named`

```python
ising.named() -> dict[str, object]
```

返回变量名版本的 Ising 字典。

#### `to_qaoa_terms`

```python
ising.to_qaoa_terms() -> list[QAOATerm]
```

转换为 QAOA 项列表。

### `QAOATerm`

```python
QAOATerm(
    qubits: tuple[int, ...],
    coefficient: float,
)
```

QAOA 成本哈密顿量中的一项。

- `qubits=(i,)` 表示 `Z_i` 项。
- `qubits=(i, j)` 表示 `Z_i Z_j` 项。

#### `pauli`

```python
term.pauli() -> str
```

返回 Pauli 字符串描述，例如 `"Z0"` 或 `"Z0 Z1"`。

### `qubo_to_ising_indices`

```python
qubo_to_ising_indices(
    qubo: dict[tuple[int, int], float],
    offset: float = 0.0,
    variable_names: list[str] | None = None,
    variable_metadata: list[VariableMetadata] | None = None,
) -> IsingModel
```

将 QUBO 转换为 Ising 模型。

当前转换约定与 QAOA 成本矩阵一致，使用二进制变量到自旋变量的映射。

## 9. 求解结果解码与调试分析

### `DecodedSolution`

```python
DecodedSolution(
    binary: dict[str, int],
    integers: dict[str, int],
    auxiliary: dict[str, int],
)
```

解码后的求解结果。

- `binary`：决策二进制变量。
- `integers`：解码后的整数变量。
- `auxiliary`：辅助变量。

#### `decisions`

```python
solution.decisions() -> dict[str, int]
```

返回决策变量与整数变量的合并结果，不包含辅助变量。

### `decode_solution`

```python
decode_solution(
    assignment,
    registry: VariableRegistry,
    integers: Sequence[EncodedInteger] | None = None,
    include_auxiliary: bool = False,
) -> DecodedSolution
```

根据变量注册表解码 bit assignment。

`assignment` 可使用变量编号或变量名作为 key。

### `decode_integer`

```python
decode_integer(
    integer: EncodedInteger,
    assignment,
) -> int
```

根据 bit assignment 解码一个整数变量。

### `qubo_energy`

```python
qubo_energy(
    qubo: Mapping[tuple[int, int], float],
    assignment,
    offset: float = 0.0,
) -> float
```

计算指定 bit assignment 下的 QUBO 能量。

### `BruteForceResult`

```python
BruteForceResult(
    best_energy: float,
    best_assignments: list[tuple[int, ...]],
    energies: dict[tuple[int, ...], float],
)
```

小规模穷举求解结果。

- `best_energy`：最小能量。
- `best_assignments`：所有达到最小能量的 bitstring。
- `energies`：全部 bitstring 对应的能量。

### `brute_force_qubo`

```python
brute_force_qubo(
    qubo: Mapping[tuple[int, int], float],
    offset: float = 0.0,
    variable_count: int | None = None,
    max_variables: int = 20,
    atol: float = 1e-9,
) -> BruteForceResult
```

对小规模 QUBO 做穷举求解。`max_variables` 用于防止误对大规模问题做指数级枚举。

### `brute_force_model`

```python
brute_force_model(
    model,
    max_variables: int = 20,
    clean: bool = True,
    atol: float = 1e-9,
) -> BruteForceResult
```

对 `Model` 做穷举求解。

### `brute_force_builder`

```python
brute_force_builder(
    builder,
    max_variables: int = 20,
    clean: bool = True,
    atol: float = 1e-9,
) -> BruteForceResult
```

对 `QuboBuilder` 做穷举求解。

### `decode_best_solutions`

```python
decode_best_solutions(
    result: BruteForceResult,
    registry: VariableRegistry,
    integers: Sequence[EncodedInteger] | None = None,
    include_auxiliary: bool = False,
) -> list[DecodedSolution]
```

将穷举得到的最优 bitstring 列表解码为变量名结果。

## 10. 标准问题构造器

### `tsp_model`

```python
tsp_model(
    distances: Sequence[Sequence[float]],
    penalty: float = 10.0,
    prefix: str = "x",
    registry: VariableRegistry = GLOBAL_REGISTRY,
) -> Model
```

构造旅行商问题 TSP 的高层 `Model`。

- `distances[i][j]` 表示城市 `i` 到城市 `j` 的距离。
- `x[city, position]` 表示城市 `city` 是否放在路径位置 `position`。

### `tsp_qubo_builder`

```python
tsp_qubo_builder(
    distances: Sequence[Sequence[float]],
    penalty: float = 10.0,
    prefix: str = "x",
    registry: VariableRegistry = GLOBAL_REGISTRY,
) -> QuboBuilder
```

构造 TSP 的底层 `QuboBuilder`。

### `graph_coloring_model`

```python
graph_coloring_model(
    node_count: int,
    edges: Sequence[tuple[int, int]],
    color_count: int,
    penalty: float = 10.0,
    prefix: str = "x",
    registry: VariableRegistry = GLOBAL_REGISTRY,
) -> Model
```

构造图着色问题的高层 `Model`。

- `node_count`：节点数量。
- `edges`：图边列表。
- `color_count`：颜色数量。
- `x[node, color]` 表示节点 `node` 是否选择颜色 `color`。

### `graph_coloring_qubo_builder`

```python
graph_coloring_qubo_builder(
    node_count: int,
    edges: Sequence[tuple[int, int]],
    color_count: int,
    penalty: float = 10.0,
    prefix: str = "x",
    registry: VariableRegistry = GLOBAL_REGISTRY,
) -> QuboBuilder
```

构造图着色问题的底层 `QuboBuilder`。

### `knapsack_model`

```python
knapsack_model(
    values: Sequence[float],
    weights: Sequence[int],
    capacity: int,
    penalty: float = 10.0,
    item_prefix: str = "x",
    slack_prefix: str = "s",
    registry: VariableRegistry = GLOBAL_REGISTRY,
) -> Model
```

构造 0-1 背包问题的高层 `Model`。

- `values[i]`：物品 `i` 的价值。
- `weights[i]`：物品 `i` 的重量。
- `capacity`：背包容量。
- 目标通常等价于最大化价值；在 QUBO 中通过最小化负价值实现。

### `knapsack_qubo_builder`

```python
knapsack_qubo_builder(
    values: Sequence[float],
    weights: Sequence[int],
    capacity: int,
    penalty: float = 10.0,
    item_prefix: str = "x",
    slack_prefix: str = "s",
    registry: VariableRegistry = GLOBAL_REGISTRY,
) -> QuboBuilder
```

构造背包问题的底层 `QuboBuilder`。

## 11. 推荐使用层级

对于一般研究建模，推荐顺序是：

1. 使用 `ModelContext` 创建变量。
2. 使用 `Model` 组织目标和约束。
3. 使用 `one_hot`、`weighted_equality`、`linear_inequality` 等高层约束函数。
4. 使用 `linear_objective`、`quadratic_objective` 添加结构化目标。
5. 使用 `to_qubo_indices()`、`to_ising_indices()`、`to_qaoa_terms()` 导出到后端。

当你已经非常清楚 QUBO 系数结构，并且希望提高建模速度时，可以直接使用 `QuboBuilder`。

## 12. 当前版本限制

- QUBO 导出要求多项式最高次数不超过 2。
- 整数变量下界要求为非负整数。
- `linear_inequality` 主要面向非负权重的线性约束。
- `brute_force_*` 系列仅适合小规模模型。
- 团队量子线路框架适配器应作为后续独立任务维护。

## 13. 逐接口使用示例

本节按公开 API 名称给出最小使用示例。示例默认从顶层包导入：

```python
from qubo_modeling import *
```

### 13.1 变量、上下文和元数据

#### `VariableMetadata`

```python
meta = VariableMetadata(
    name="slack[0]",
    kind="binary",
    role="auxiliary",
    source="capacity_constraint",
)
print(meta.name, meta.role, meta.source)
```

用于描述变量来源，特别适合审查 slack/auxiliary 变量。

#### `VariableRegistry`

```python
registry = VariableRegistry()
x0_id = registry.get_or_create("x[0]")
x1_id = registry.get_or_create("x[1]")

print(x0_id)                 # 0
print(registry.name(x1_id))  # x[1]
print(registry.names())      # ['x[0]', 'x[1]']
```

#### `VariableRegistry.get_or_create`

```python
registry = VariableRegistry()
decision_id = registry.get_or_create("x")
aux_id = registry.get_or_create(
    "s[0]",
    kind="binary",
    role="auxiliary",
    source="linear_inequality",
)
```

#### `VariableRegistry.name`

```python
registry = VariableRegistry()
var_id = registry.get_or_create("x")
print(registry.name(var_id))  # x
```

#### `VariableRegistry.names`

```python
registry = VariableRegistry()
registry.get_or_create("x")
registry.get_or_create("y")
print(registry.names())  # ['x', 'y']
```

#### `VariableRegistry.metadata`

```python
registry = VariableRegistry()
var_id = registry.get_or_create("aux", role="auxiliary", source="demo")
print(registry.metadata(var_id))
```

#### `VariableRegistry.variables_by_role`

```python
registry = VariableRegistry()
registry.get_or_create("x", role="decision")
registry.get_or_create("s", role="auxiliary")
print(registry.variables_by_role("decision"))   # ['x']
print(registry.variables_by_role("auxiliary"))  # ['s']
```

#### `VariableRegistry.auxiliary_names`

```python
registry = VariableRegistry()
registry.get_or_create("x")
registry.get_or_create("s[0]", role="auxiliary")
print(registry.auxiliary_names())  # ['s[0]']
```

#### `ModelContext`

```python
ctx = ModelContext()
x = ctx.binary("x")
y = ctx.binary("y")

model = Model(x + y)
qubo, offset = model.to_qubo()
```

#### `ModelContext.binary`

```python
ctx = ModelContext()
x = ctx.binary("x")
print(x)  # 1*x
```

#### `ModelContext.auxiliary_binary`

```python
ctx = ModelContext()
s = ctx.auxiliary_binary("s[0]", source="manual_slack")
print(ctx.registry.auxiliary_names())  # ['s[0]']
```

#### `ModelContext.binary_array`

```python
ctx = ModelContext()
x = ctx.binary_array("x", (2, 3))
print(x[0][0], x[1][2])
print(ctx.registry.names())
```

#### `ModelContext.integer`

```python
ctx = ModelContext()
n = ctx.integer("n", lower_bound=0, upper_bound=5, encoding="log")
print(n.bit_names())
print(n.expression())
```

#### `ModelContext.auxiliary_integer`

```python
ctx = ModelContext()
s = ctx.auxiliary_integer("slack", lower_bound=0, upper_bound=3, source="capacity")
print(s.bit_names())
print(ctx.registry.auxiliary_names())
```

#### `ModelContext.zero`

```python
ctx = ModelContext()
model = Model(ctx.zero())
print(model.polynomial())  # 0
```

#### `ModelContext.qubo_builder`

```python
ctx = ModelContext()
x = ctx.binary("x")
builder = ctx.qubo_builder()
builder.add_polynomial(2.0 * x)
print(builder.to_qubo())
```

#### `ModelContext.decode_solution`

```python
ctx = ModelContext()
x = ctx.binary_array("x", 2)
solution = ctx.decode_solution({0: 1, 1: 0})
print(solution.decisions())  # {'x[0]': 1, 'x[1]': 0}
```

### 13.2 多项式、二进制变量和线性表达式

#### `Polynomial`

```python
registry = VariableRegistry()
x_id = registry.get_or_create("x")
y_id = registry.get_or_create("y")

poly = Polynomial({(x_id,): 1.0, (x_id, y_id): 2.0}, registry)
print(poly)
```

#### `Polynomial.constant`

```python
ctx = ModelContext()
c = Polynomial.constant(3.0, registry=ctx.registry)
print(c.to_qubo_indices())  # ({}, 3.0)
```

#### `Polynomial.variable`

```python
registry = VariableRegistry()
x = Polynomial.variable("x", registry=registry)
print(x.variables())  # ['x']
```

#### `Polynomial.clean`

```python
ctx = ModelContext()
x = ctx.binary("x")
poly = x + 1e-14 * x
print(poly.clean())
```

#### `Polynomial.degree`

```python
ctx = ModelContext()
x = ctx.binary("x")
y = ctx.binary("y")
print((x + y).degree())      # 1
print((x * y + 1).degree())  # 2
```

#### `Polynomial.variables`

```python
ctx = ModelContext()
x = ctx.binary("x")
y = ctx.binary("y")
print((x * y + 2).variables())  # ['x', 'y']
```

#### `Polynomial.to_qubo_indices`

```python
ctx = ModelContext()
x = ctx.binary("x")
y = ctx.binary("y")
qubo, offset = (3 * x + 2 * x * y + 5).to_qubo_indices()
print(qubo, offset)
```

#### `Polynomial.to_qubo`

```python
ctx = ModelContext()
x = ctx.binary("x")
y = ctx.binary("y")
qubo, offset = (3 * x + 2 * x * y + 5).to_qubo()
print(qubo)  # {('x', 'x'): 3.0, ('x', 'y'): 2.0}
```

#### `Binary`

```python
registry = VariableRegistry()
x = Binary("x", registry=registry)
model = Model(x)
print(model.to_qubo())
```

#### `binary_array`

```python
registry = VariableRegistry()
x = binary_array("x", (2, 2), registry=registry)
print(x[0][0], x[1][1])
```

#### `Sum`

```python
ctx = ModelContext()
x = ctx.binary_array("x", 3)
expr = Sum(x)
print(expr)  # x[0] + x[1] + x[2]
```

#### `LinearExpression`

```python
ctx = ModelContext()
x = ctx.binary_array("x", 2)
linear = LinearExpression.from_terms([(2.0, x[0]), (3.0, x[1])], offset=1.0)
print(linear.expression())
```

#### `LinearExpression.from_terms`

```python
ctx = ModelContext()
x = ctx.binary_array("x", 2)
linear = LinearExpression.from_terms([(4.0, x[0]), (-1.0, x[1])])
print(linear.weighted_terms())
```

#### `LinearExpression.from_integer`

```python
ctx = ModelContext()
n = ctx.integer("n", upper_bound=7)
linear = LinearExpression.from_integer(n)
print(linear.expression())
```

#### `LinearExpression.weighted_terms`

```python
ctx = ModelContext()
x = ctx.binary_array("x", 2)
linear = Linear([(2.0, x[0]), (5.0, x[1])])
print(linear.weighted_terms())
```

#### `LinearExpression.expression`

```python
ctx = ModelContext()
x = ctx.binary_array("x", 2)
linear = Linear([(2.0, x[0]), (5.0, x[1])], offset=3.0)
poly = linear.expression()
print(poly.to_qubo())
```

#### `Linear`

```python
ctx = ModelContext()
x = ctx.binary_array("x", 2)
linear = Linear([(1.0, x[0]), (2.0, x[1])], offset=1.0)
constraint = weighted_equality(linear, target=2.0, penalty=10.0)
```

### 13.3 模型和目标函数

#### `Model`

```python
ctx = ModelContext()
x = ctx.binary_array("x", 2)

model = Model(ctx.zero())
model.add_constraint(one_hot(x, penalty=5.0))
model.add_objective(linear_objective([(1.0, x[0]), (-1.0, x[1])]))

print(model.to_qubo())
```

#### `Model.add_constraint`

```python
ctx = ModelContext()
x = ctx.binary_array("x", 2)
model = Model(ctx.zero())
model.add_constraint(one_hot(x, penalty=5.0, label="choose_one"))
```

#### `Model.add_constraints`

```python
ctx = ModelContext()
x = ctx.binary_array("x", (2, 2))
model = Model(ctx.zero())
model.add_constraints(permutation(x, penalty=10.0))
```

#### `Model.add_objective`

```python
ctx = ModelContext()
x = ctx.binary_array("x", 2)
model = Model(ctx.zero())
model.add_objective(linear_objective([(2.0, x[0]), (3.0, x[1])]))
```

#### `Model.polynomial`

```python
ctx = ModelContext()
x = ctx.binary_array("x", 2)
model = Model(ctx.zero())
model.add_constraint(one_hot(x, penalty=5.0))
poly = model.polynomial()
print(poly.degree())
```

#### `Model.to_qubo`

```python
ctx = ModelContext()
x = ctx.binary("x")
model = Model(2.0 * x)
qubo, offset = model.to_qubo()
print(qubo, offset)
```

#### `Model.to_qubo_indices`

```python
ctx = ModelContext()
x = ctx.binary("x")
model = Model(2.0 * x)
qubo, offset = model.to_qubo_indices()
print(qubo, offset)
```

#### `Model.to_sparse_matrix`

```python
ctx = ModelContext()
x = ctx.binary_array("x", 2)
model = Model(x[0] + 2 * x[0] * x[1])
matrix = model.to_sparse_matrix()
print(matrix.row, matrix.col, matrix.data, matrix.shape)
```

#### `Model.to_qubo_builder`

```python
ctx = ModelContext()
x = ctx.binary_array("x", 2)
model = Model(x[0] + x[0] * x[1])
builder = model.to_qubo_builder()
print(builder.qubo)
```

#### `Model.to_ising_indices`

```python
ctx = ModelContext()
x = ctx.binary_array("x", 2)
model = Model(x[0] + 2 * x[0] * x[1])
ising = model.to_ising_indices()
print(ising.h, ising.J, ising.offset)
```

#### `Model.to_ising`

```python
ctx = ModelContext()
x = ctx.binary_array("x", 2)
model = Model(x[0] + 2 * x[0] * x[1])
named_ising = model.to_ising()
print(named_ising)
```

#### `Model.to_qaoa_terms`

```python
ctx = ModelContext()
x = ctx.binary_array("x", 2)
model = Model(x[0] + 2 * x[0] * x[1])
terms, offset, names = model.to_qaoa_terms()
print(terms, offset, names)
```

#### `ObjectiveFragment`

```python
ctx = ModelContext()
x = ctx.binary("x")
fragment = ObjectiveFragment(3.0 * x)
model = Model(ctx.zero())
model.add_objective(fragment)
```

#### `ObjectiveFragment.expression`

```python
ctx = ModelContext()
x = ctx.binary("x")
fragment = linear_objective([(2.0, x)])
print(fragment.expression)
```

#### `ObjectiveFragment.add_to_builder`

```python
ctx = ModelContext()
x = ctx.binary("x")
fragment = linear_objective([(2.0, x)])
builder = ctx.qubo_builder()
fragment.add_to_builder(builder)
print(builder.to_qubo())
```

#### `linear_objective`

```python
ctx = ModelContext()
x = ctx.binary_array("x", 3)
objective = linear_objective([(1.0, x[0]), (0.5, x[1]), (-2.0, x[2])])
model = Model(ctx.zero())
model.add_objective(objective)
```

#### `quadratic_objective`

```python
ctx = ModelContext()
x = ctx.binary_array("x", 2)
objective = quadratic_objective([(3.0, x[0], x[1])])
model = Model(ctx.zero())
model.add_objective(objective)
```

### 13.4 约束 API

#### `Constraint`

```python
ctx = ModelContext()
x = ctx.binary_array("x", 2)
constraint = Constraint((x[0] + x[1] - 1) ** 2, penalty=5.0, label="manual_one_hot")
model = Model(ctx.zero())
model.add_constraint(constraint)
```

#### `Constraint.expression`

```python
ctx = ModelContext()
x = ctx.binary_array("x", 2)
constraint = one_hot(x, penalty=5.0)
print(constraint.expression)
```

#### `Constraint.as_penalty`

```python
ctx = ModelContext()
x = ctx.binary_array("x", 2)
constraint = one_hot(x, penalty=5.0)
penalty_poly = constraint.as_penalty()
print(penalty_poly)
```

#### `Constraint.add_to_builder`

```python
ctx = ModelContext()
x = ctx.binary_array("x", 2)
constraint = one_hot(x, penalty=5.0)
builder = ctx.qubo_builder()
constraint.add_to_builder(builder)
print(builder.to_qubo())
```

#### `one_hot`

```python
ctx = ModelContext()
x = ctx.binary_array("x", 3)
model = Model(ctx.zero())
model.add_constraint(one_hot(x, penalty=10.0, label="choose_one"))
```

#### `cardinality`

```python
ctx = ModelContext()
x = ctx.binary_array("x", 4)
model = Model(ctx.zero())
model.add_constraint(cardinality(x, count=2, penalty=10.0, label="choose_two"))
```

#### `at_most_one`

```python
ctx = ModelContext()
x = ctx.binary_array("x", 3)
model = Model(ctx.zero())
model.add_constraint(at_most_one(x, penalty=5.0))
```

#### `at_least_one`

```python
ctx = ModelContext()
x = ctx.binary_array("x", 3)
constraint, slack = at_least_one(x, slack_prefix="alo_s", penalty=5.0)
model = Model(ctx.zero())
model.add_constraint(constraint)
print([var.variables() for var in slack])
```

#### `weighted_equality`

```python
ctx = ModelContext()
x = ctx.binary_array("x", 3)
constraint = weighted_equality(
    [(2.0, x[0]), (3.0, x[1]), (1.0, x[2])],
    target=4.0,
    penalty=10.0,
)
model = Model(ctx.zero())
model.add_constraint(constraint)
```

#### `linear_inequality`

```python
ctx = ModelContext()
x = ctx.binary_array("x", 3)
constraint, slack = linear_inequality(
    [(2.0, x[0]), (3.0, x[1]), (1.0, x[2])],
    upper_bound=4,
    slack_prefix="cap_s",
    penalty=10.0,
)
model = Model(ctx.zero())
model.add_constraint(constraint)
print(ctx.registry.auxiliary_names())
```

#### `integer_equality`

```python
ctx = ModelContext()
n = ctx.integer("n", upper_bound=7)
constraint = integer_equality(n, target=3.0, penalty=10.0)
model = Model(ctx.zero())
model.add_constraint(constraint)
```

#### `integer_less_equal`

```python
ctx = ModelContext()
n = ctx.integer("n", upper_bound=7)
constraint, slack = integer_less_equal(n, upper_bound=5, penalty=10.0)
model = Model(ctx.zero())
model.add_constraint(constraint)
print(ctx.registry.auxiliary_names())
```

#### `one_hot_rows`

```python
ctx = ModelContext()
x = ctx.binary_array("x", (2, 2))
row_constraints = one_hot_rows(x, penalty=10.0, label="row")
model = Model(ctx.zero())
model.add_constraints(row_constraints)
```

#### `one_hot_columns`

```python
ctx = ModelContext()
x = ctx.binary_array("x", (2, 2))
col_constraints = one_hot_columns(x, penalty=10.0, label="col")
model = Model(ctx.zero())
model.add_constraints(col_constraints)
```

#### `assignment_matrix`

```python
ctx = ModelContext()
x = ctx.binary_array("x", (2, 2))
constraints = assignment_matrix(x, penalty=10.0, label="assign")
model = Model(ctx.zero())
model.add_constraints(constraints)
```

#### `permutation`

```python
ctx = ModelContext()
x = ctx.binary_array("x", (3, 3))
model = Model(ctx.zero())
model.add_constraints(permutation(x, penalty=10.0, label="perm"))
```

### 13.5 整数变量 API

#### `EncodedInteger`

```python
ctx = ModelContext()
n = ctx.integer("n", upper_bound=5)
print(n.name, n.lower_bound, n.upper_bound)
print(n.bits)
print(n.weights)
```

#### `EncodedInteger.expression`

```python
ctx = ModelContext()
n = ctx.integer("n", upper_bound=5)
expr = n.expression()
print(expr)
```

#### `EncodedInteger.weighted_terms`

```python
ctx = ModelContext()
n = ctx.integer("n", upper_bound=5)
print(n.weighted_terms(scale=2.0))
```

#### `EncodedInteger.linear_expression`

```python
ctx = ModelContext()
n = ctx.integer("n", upper_bound=5)
linear = n.linear_expression()
print(linear.expression())
```

#### `EncodedInteger.bit_names`

```python
ctx = ModelContext()
n = ctx.integer("n", upper_bound=5)
print(n.bit_names())
```

#### `Integer`

```python
ctx = ModelContext()
n = Integer("n", lower_bound=0, upper_bound=5, encoding="log", registry=ctx.registry)
print(n.expression())
```

#### `LogEncodedInteger`

```python
ctx = ModelContext()
n = LogEncodedInteger("n", lower_bound=0, upper_bound=7, registry=ctx.registry)
print(n.bit_names())
```

#### `UnaryEncodedInteger`

```python
ctx = ModelContext()
n = UnaryEncodedInteger("n", lower_bound=0, upper_bound=3, registry=ctx.registry)
print(n.bit_names())
```

### 13.6 `QuboBuilder` 底层接口

#### `QuboBuilder`

```python
registry = VariableRegistry()
x_id = registry.get_or_create("x")
builder = QuboBuilder(registry=registry)
builder.add_linear(x_id, 2.0)
print(builder.to_qubo())
```

#### `QuboBuilder.add_offset`

```python
builder = QuboBuilder()
builder.add_offset(3.0)
print(builder.to_qubo_indices())  # ({}, 3.0)
```

#### `QuboBuilder.add_linear`

```python
registry = VariableRegistry()
x_id = registry.get_or_create("x")
builder = QuboBuilder(registry=registry)
builder.add_linear(x_id, 2.0)
```

#### `QuboBuilder.add_quadratic`

```python
registry = VariableRegistry()
x_id = registry.get_or_create("x")
y_id = registry.get_or_create("y")
builder = QuboBuilder(registry=registry)
builder.add_quadratic(x_id, y_id, 5.0)
```

#### `QuboBuilder.add_linear_terms`

```python
registry = VariableRegistry()
x_id = registry.get_or_create("x")
y_id = registry.get_or_create("y")
builder = QuboBuilder(registry=registry)
builder.add_linear_terms([(1.0, x_id), (2.0, y_id)])
```

#### `QuboBuilder.add_quadratic_terms`

```python
registry = VariableRegistry()
x_id = registry.get_or_create("x")
y_id = registry.get_or_create("y")
builder = QuboBuilder(registry=registry)
builder.add_quadratic_terms([(3.0, x_id, y_id)])
```

#### `QuboBuilder.add_polynomial`

```python
ctx = ModelContext()
x = ctx.binary_array("x", 2)
builder = ctx.qubo_builder()
builder.add_polynomial((x[0] + x[1] - 1) ** 2, scale=5.0)
```

#### `QuboBuilder.add_cardinality_penalty`

```python
registry = VariableRegistry()
ids = [registry.get_or_create(f"x[{i}]") for i in range(3)]
builder = QuboBuilder(registry=registry)
builder.add_cardinality_penalty(ids, count=1, penalty=10.0)
```

#### `QuboBuilder.add_at_most_one_penalty`

```python
registry = VariableRegistry()
ids = [registry.get_or_create(f"x[{i}]") for i in range(3)]
builder = QuboBuilder(registry=registry)
builder.add_at_most_one_penalty(ids, penalty=10.0)
```

#### `QuboBuilder.add_at_least_one_penalty`

```python
registry = VariableRegistry()
ids = [registry.get_or_create(f"x[{i}]") for i in range(3)]
builder = QuboBuilder(registry=registry)
slack_ids = builder.add_at_least_one_penalty(ids, slack_prefix="alo_s", penalty=10.0)
print(slack_ids, registry.auxiliary_names())
```

#### `QuboBuilder.add_weighted_equality_penalty`

```python
registry = VariableRegistry()
x_id = registry.get_or_create("x")
y_id = registry.get_or_create("y")
builder = QuboBuilder(registry=registry)
builder.add_weighted_equality_penalty([(2.0, x_id), (3.0, y_id)], target=3.0, penalty=10.0)
```

#### `QuboBuilder.add_linear_inequality_penalty`

```python
registry = VariableRegistry()
x_id = registry.get_or_create("x")
y_id = registry.get_or_create("y")
builder = QuboBuilder(registry=registry)
slack_ids = builder.add_linear_inequality_penalty(
    [(2.0, x_id), (3.0, y_id)],
    upper_bound=4,
    slack_prefix="cap_s",
    penalty=10.0,
)
```

#### `QuboBuilder.to_qubo_indices`

```python
registry = VariableRegistry()
x_id = registry.get_or_create("x")
builder = QuboBuilder(registry=registry)
builder.add_linear(x_id, 2.0)
qubo, offset = builder.to_qubo_indices()
print(qubo, offset)
```

#### `QuboBuilder.to_qubo`

```python
registry = VariableRegistry()
x_id = registry.get_or_create("x")
builder = QuboBuilder(registry=registry)
builder.add_linear(x_id, 2.0)
print(builder.to_qubo())
```

#### `QuboBuilder.to_sparse_matrix`

```python
registry = VariableRegistry()
x_id = registry.get_or_create("x")
y_id = registry.get_or_create("y")
builder = QuboBuilder(registry=registry)
builder.add_quadratic(x_id, y_id, 3.0)
matrix = builder.to_sparse_matrix()
print(matrix.row, matrix.col, matrix.data)
```

#### `QuboBuilder.to_ising_indices`

```python
registry = VariableRegistry()
x_id = registry.get_or_create("x")
builder = QuboBuilder(registry=registry)
builder.add_linear(x_id, 2.0)
ising = builder.to_ising_indices()
print(ising.h, ising.offset)
```

#### `QuboBuilder.to_qaoa_terms`

```python
registry = VariableRegistry()
x_id = registry.get_or_create("x")
builder = QuboBuilder(registry=registry)
builder.add_linear(x_id, 2.0)
terms, offset, names = builder.to_qaoa_terms()
print(terms, offset, names)
```

### 13.7 导出后端与数据结构

#### `SparseMatrixCOO`

```python
matrix = SparseMatrixCOO(
    row=[0, 0],
    col=[0, 1],
    data=[1.0, 2.0],
    shape=(2, 2),
    offset=3.0,
    variable_names=["x", "y"],
)
print(matrix)
```

#### `SparseMatrixCOO.to_dense`

```python
matrix = SparseMatrixCOO(row=[0, 1], col=[0, 1], data=[1.0, 2.0], shape=(2, 2))
print(matrix.to_dense())  # [[1.0, 0.0], [0.0, 2.0]]
```

#### `IsingModel`

```python
ising = IsingModel(
    h={0: -0.5},
    J={(0, 1): 1.25},
    offset=2.0,
    variable_names=["x", "y"],
)
print(ising)
```

#### `IsingModel.named`

```python
ising = IsingModel(h={0: -0.5}, J={(0, 1): 1.25}, variable_names=["x", "y"])
print(ising.named())
```

#### `IsingModel.to_qaoa_terms`

```python
ising = IsingModel(h={0: -0.5}, J={(0, 1): 1.25}, variable_names=["x", "y"])
terms = ising.to_qaoa_terms()
print(terms)
```

#### `QAOATerm`

```python
single = QAOATerm(qubits=(0,), coefficient=-0.5)
pair = QAOATerm(qubits=(0, 1), coefficient=1.25)
print(single, pair)
```

#### `QAOATerm.pauli`

```python
term = QAOATerm(qubits=(0, 1), coefficient=1.25)
print(term.pauli())  # Z0 Z1
```

#### `qubo_to_ising_indices`

```python
qubo = {(0, 0): 1.0, (0, 1): 2.0}
ising = qubo_to_ising_indices(qubo, offset=0.0, variable_names=["x", "y"])
print(ising.h, ising.J, ising.offset)
```

### 13.8 解码、穷举和调试分析

#### `DecodedSolution`

```python
solution = DecodedSolution(
    binary={"x": 1},
    integers={"n": 3},
    auxiliary={"s[0]": 0},
)
print(solution)
```

#### `DecodedSolution.decisions`

```python
solution = DecodedSolution(binary={"x": 1}, integers={"n": 3}, auxiliary={"s": 0})
print(solution.decisions())  # {'x': 1, 'n': 3}
```

#### `decode_solution`

```python
ctx = ModelContext()
x = ctx.binary_array("x", 2)
decoded = decode_solution({0: 1, 1: 0}, registry=ctx.registry)
print(decoded.binary)
```

#### `decode_integer`

```python
ctx = ModelContext()
n = ctx.integer("n", upper_bound=3)
assignment = {name: 1 for name in n.bit_names()}
print(decode_integer(n, assignment))
```

#### `qubo_energy`

```python
qubo = {(0, 0): -1.0, (1, 1): 2.0, (0, 1): 3.0}
energy = qubo_energy(qubo, assignment=(1, 0), offset=0.5)
print(energy)
```

#### `BruteForceResult`

```python
result = BruteForceResult(
    best_energy=-1.0,
    best_assignments=[(1, 0)],
    energies={(0, 0): 0.0, (1, 0): -1.0},
)
print(result.best_energy)
```

#### `brute_force_qubo`

```python
qubo = {(0, 0): -1.0, (1, 1): 2.0}
result = brute_force_qubo(qubo, variable_count=2)
print(result.best_energy, result.best_assignments)
```

#### `brute_force_model`

```python
ctx = ModelContext()
x = ctx.binary_array("x", 2)
model = Model(-x[0] + 2 * x[1])
result = brute_force_model(model)
print(result.best_assignments)
```

#### `brute_force_builder`

```python
registry = VariableRegistry()
x_id = registry.get_or_create("x")
builder = QuboBuilder(registry=registry)
builder.add_linear(x_id, -1.0)
result = brute_force_builder(builder)
print(result.best_assignments)
```

#### `decode_best_solutions`

```python
ctx = ModelContext()
x = ctx.binary_array("x", 2)
model = Model(-x[0] + 2 * x[1])
result = brute_force_model(model)
solutions = decode_best_solutions(result, ctx.registry)
print([solution.decisions() for solution in solutions])
```

### 13.9 标准问题构造器

#### `tsp_model`

```python
distances = [
    [0.0, 1.0, 2.0],
    [1.0, 0.0, 3.0],
    [2.0, 3.0, 0.0],
]
ctx = ModelContext()
model = tsp_model(distances, penalty=10.0, prefix="tsp", registry=ctx.registry)
qubo, offset = model.to_qubo()
```

#### `tsp_qubo_builder`

```python
distances = [
    [0.0, 1.0, 2.0],
    [1.0, 0.0, 3.0],
    [2.0, 3.0, 0.0],
]
registry = VariableRegistry()
builder = tsp_qubo_builder(distances, penalty=10.0, prefix="tsp", registry=registry)
qubo, offset = builder.to_qubo()
```

#### `graph_coloring_model`

```python
ctx = ModelContext()
model = graph_coloring_model(
    node_count=3,
    edges=[(0, 1), (1, 2)],
    color_count=2,
    penalty=10.0,
    prefix="color",
    registry=ctx.registry,
)
result = brute_force_model(model)
```

#### `graph_coloring_qubo_builder`

```python
registry = VariableRegistry()
builder = graph_coloring_qubo_builder(
    node_count=3,
    edges=[(0, 1), (1, 2)],
    color_count=2,
    penalty=10.0,
    prefix="color",
    registry=registry,
)
qubo, offset = builder.to_qubo()
```

#### `knapsack_model`

```python
ctx = ModelContext()
model = knapsack_model(
    values=[4.0, 3.0, 2.0],
    weights=[2, 1, 3],
    capacity=3,
    penalty=10.0,
    item_prefix="item",
    slack_prefix="slack",
    registry=ctx.registry,
)
result = brute_force_model(model)
solutions = decode_best_solutions(result, ctx.registry)
```

#### `knapsack_qubo_builder`

```python
registry = VariableRegistry()
builder = knapsack_qubo_builder(
    values=[4.0, 3.0, 2.0],
    weights=[2, 1, 3],
    capacity=3,
    penalty=10.0,
    item_prefix="item",
    slack_prefix="slack",
    registry=registry,
)
qubo, offset = builder.to_qubo()
```

