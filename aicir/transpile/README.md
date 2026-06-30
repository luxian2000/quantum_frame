# aicir.transpile 使用手册

量子线路编译与结构优化框架——提供 pass pipeline 架构，用于线路验证、规范化和本地优化。

---

## 目录

| 文件 / 目录 | 说明 |
| --- | --- |
| `base.py` | `TransformationPass` 抽象基类 |
| `passmanager.py` | `PassManager` 流水线管理器 + `optimize`（默认优化入口） |
| `rewrite.py` | 多格式前端 `optimize_basic` / `optimize_circuit`（dict/qasm/dag） |
| `passes/` | 内置 pass 实现 |
| `passes/basic.py` | `ValidatePass` / `CanonicalizePass` |
| `passes/cancel_inverse.py` | `CancelInversePass` |
| `passes/merge_rotations.py` | `MergeRotationsPass` |
| `passes/commute_single_qubit.py` | `CommuteSingleQubitPass` |
| `passes/decompose.py` | `DecomposePass`（高级门分解到目标门集） |
| `passes/layout.py` | `LayoutPass`（logical→physical 重标号） |
| `passes/routing.py` | `RoutingPass`（沿耦合图插 SWAP） |
| `passes/_local_rewrite.py` | 底层重写引擎（内部模块） |

> 面向硬件目标的 `DecomposePass` / `LayoutPass` / `RoutingPass` 消费 `aicir.devices.Target`（门集 + 耦合拓扑），详见第 4.6–4.8 节。

---

## 1  快速上手

### 1.1  一行默认优化

```python
from aicir import Circuit, hadamard, pauli_x, rx
from aicir.transpile import optimize

circuit = Circuit(
    hadamard(0),
    hadamard(0),       # 与前一个 H 互逆 → 消去
    rx(0.1, 1),
    rx(0.2, 1),        # 与前一个 rx 同轴 → 合并为 rx(0.3, 1)
    pauli_x(2),
    pauli_x(2),        # X·X = I → 消去
    n_qubits=3,
)

optimized = optimize(circuit)   # 返回优化后的新 Circuit
# optimized 只剩 rx(0.3, 1)
```

### 1.2  自定义流水线

```python
from aicir.transpile import PassManager

pm = PassManager(
    ["validate", "canonicalize", "cancel_inverse", "merge_rotations", "commute_single_qubit"],
    fixed_point=True,     # 重复运行直到线路不再变化
    max_rounds=64,        # 最大迭代轮次
)

optimized = pm.run(circuit)
```

---

## 2  PassManager

`PassManager` 是 pass 流水线的核心调度器。

### 2.1  构造参数

| 参数 | 类型 | 默认 | 说明 |
| --- | --- | --- | --- |
| `passes` | `Iterable[str \| TransformationPass]` | — | pass 序列（支持字符串名称和 pass 实例混用） |
| `fixed_point` | `bool` | `False` | 是否重复运行 pass 序列直到线路不再变化 |
| `max_rounds` | `int` | `64` | `fixed_point=True` 时的最大迭代轮次 |

### 2.2  字符串名称映射

| 字符串 | Pass 类 |
| --- | --- |
| `"validate"` | `ValidatePass` |
| `"canonicalize"` | `CanonicalizePass` |
| `"cancel_inverse"` / `"cancel"` | `CancelInversePass` |
| `"merge_rotations"` / `"merge_rotation"` | `MergeRotationsPass` |
| `"commute_single_qubit"` / `"commute"` | `CommuteSingleQubitPass` |
| `"decompose"` | `DecomposePass`（默认目标门集 `("cx",)`） |
| `"layout"` | `LayoutPass`（平凡布局） |

> `RoutingPass` 需要 `Target`，不能用字符串名构造；请直接传入实例。

### 2.3  运行

```python
optimized = pm.run(circuit)   # 返回变换后的 Circuit 副本
```

`run` 接收 `Circuit` 对象（或任何兼容 `CircuitIR` 接口的对象），返回一个新的 `Circuit`，原始线路不被修改。

---

## 3  optimize — 默认优化入口

线路结构优化的统一入口。对线路应用预配置的默认本地优化流水线，返回优化后的新线路：

```python
from aicir.transpile import optimize

optimized = optimize(circuit, max_rounds=64, max_reorder_hops=8)
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `max_rounds` | `64` | 不动点迭代最大轮次 |
| `max_reorder_hops` | `8` | `CommuteSingleQubitPass` 的最大回看步数 |

等价于运行以下流水线：

```python
PassManager(
    [
        CancelInversePass(),
        MergeRotationsPass(),
        CommuteSingleQubitPass(max_reorder_hops=8),
    ],
    fixed_point=True,
    max_rounds=64,
).run(circuit)
```

`optimize_circuit(circuit)` 是 Circuit 专用别名，等价于 `optimize(circuit)`；
`optimize_basic(obj)` 是多格式前端（dict / OpenQASM 文本 / DAG），按输入类型分派并保持输出类型不变（详见第 5 节）。

---

## 4  内置 Pass 详解

### 4.1  ValidatePass — 结构校验

检查线路的结构正确性（需要线路上下文 `n_qubits`）：

- **量子比特越界**：目标位 / 控制位超出 `[0, n_qubits)` 范围
- **重复比特**：同一门中目标位或控制位出现重复
- **目标与控制冲突**：目标位与控制位存在交集

> 门的目标比特数 / 参数个数 / 控制位要求已在 IR 构造时由 `GateSpec` 自动校验；`ValidatePass` 负责需要 `n_qubits` 上下文的补充检查。

```python
from aicir.transpile import ValidatePass

ValidatePass().run(circuit)   # 校验不通过则抛 ValueError
```

### 4.2  CanonicalizePass — 门名规范化

将别名门名重写为 `GateSpec` 规范名：

| 别名 | → 规范名 |
| --- | --- |
| `X` | `pauli_x` |
| `cnot` | `cx` |
| `ccnot` | `toffoli` |
| … | … |

未注册的门名原样保留。同时经过 `Circuit` 构造器完成字典格式归一。

```python
from aicir.transpile import CanonicalizePass

canonical = CanonicalizePass().run(circuit)
```

### 4.3  CancelInversePass — 相邻逆门消去

消去相邻的自逆门对：

| 可消去的门对 | 说明 |
| --- | --- |
| `X · X` | 泡利门自逆 |
| `Y · Y` | 泡利门自逆 |
| `Z · Z` | 泡利门自逆 |
| `H · H` | Hadamard 自逆 |
| `CX · CX` | 同控制位 / 目标位 / 控制态的 CNOT 自逆 |
| `S · S†` / `S† · S` | S 门与其共轭转置互逆 |

```python
from aicir.transpile import CancelInversePass

reduced = CancelInversePass().run(circuit)
```

### 4.4  MergeRotationsPass — 同轴旋转合并

合并作用于同一量子比特的相邻同类旋转门：

```
rx(θ₁, q) · rx(θ₂, q)  →  rx(θ₁ + θ₂, q)
ry(θ₁, q) · ry(θ₂, q)  →  ry(θ₁ + θ₂, q)
rz(θ₁, q) · rz(θ₂, q)  →  rz(θ₁ + θ₂, q)
```

- 仅合并无控制位的单比特旋转门。
- 合并后角度接近零（`|θ| < 1e-15`）时直接移除该门。

```python
from aicir.transpile import MergeRotationsPass

merged = MergeRotationsPass().run(circuit)
```

### 4.5  CommuteSingleQubitPass — 交换律优化

通过有限回看（lookback），跨越可交换的中间门来消去或合并单比特门。

```python
from aicir.transpile import CommuteSingleQubitPass

optimized = CommuteSingleQubitPass(max_reorder_hops=8).run(circuit)
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `max_reorder_hops` | `8` | 最大回看步数（向前跳过的门数上限） |

**交换律规则**：单比特门在满足以下条件时可安全跨越 CNOT 门：

| 单比特门 | 可跨越的位置 |
| --- | --- |
| `Z` / `S` / `S†` / `RZ` | CNOT 的**控制位**上 |
| `X` / `RX` | CNOT 的**目标位**上 |

跨越后若找到可消去 / 可合并的同比特门，则执行消去或合并；否则保持原位。

### 4.6  DecomposePass — 高级门分解

把高级双比特门分解到目标门集，保持幺正等价。分解规则由 `GateSpec.decomposition` 字段**驱动**（`aicir.gates.gate_decomposition`），内置经数值验证的标准规则：

| 高级门 | 分解 |
| --- | --- |
| `swap(a, b)` | `cx(a,[b]) · cx(b,[a]) · cx(a,[b])` |
| `cz(t,[c])` | `h(t) · cx(t,[c]) · h(t)` |
| `cy(t,[c])` | `rz(-π/2,t) · cx(t,[c]) · rz(π/2,t)` |

```python
from aicir.transpile import DecomposePass
from aicir.devices import Target

# 显式门集
out = DecomposePass(basis_gates=("cx",)).run(circuit)
# 或从 Target 取门集
out = DecomposePass(target=Target(n_qubits=4, basis_gates=("cx", "hadamard", "rz"))).run(circuit)
```

> 规则取自注册表而非硬编码：注册自定义门时携带 `decomposition`（签名 `(qubits, controls, control_states, params) -> list[dict] | None`），`DecomposePass` 无需改动即可识别。受控形式仅支持单控制位；规则展开产生的单比特门为 `hadamard`/`rz`，暂不做任意单比特门的 Euler 基底翻译。

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `basis_gates` | `("cx",)` | 目标原生门集（支持别名）；门集内的门保留不动 |
| `target` | `None` | 传入 `Target` 时从中取 `basis_gates`（与 `basis_gates` 二选一） |
| `skip_unsupported` | `False` | `True` 时对不在门集且无规则的门保持原样；`False` 时这类双比特门抛 `ValueError` |

- 受控形式仅支持单控制位（`control_states=(1,)`）。
- 注册表规则展开产生 `hadamard`/`rz`。此外：基底含 `rz` 与 `ry` 时，任意**不受控单比特门**经 **ZYZ（Euler）翻译**为 `rz·ry·rz`（经 `GateSpec.matrix` 取 2x2 矩阵，等价**至全局相位**——对不受控单比特门物理无关）。基底不含 `rz`/`ry` 时单比特门按旧行为原样保留。

### 4.7  LayoutPass — 逻辑→物理布局

重新标号比特，不插入任何门，线路在比特置换意义下与原线路等价。支持显式 / 平凡 / 自动三种模式。

```python
from aicir.transpile import LayoutPass
from aicir.devices import Target

target = Target(n_qubits=4, coupling_map=[(0, 1), (1, 2), (2, 3)])
out = LayoutPass(initial_layout={0: 2, 1: 3}, target=target).run(circuit)  # 显式；输出 n_qubits=4
out = LayoutPass([3, 1]).run(circuit)        # 序列形式：逻辑 0→3, 逻辑 1→1
out = LayoutPass().run(circuit)               # None=平凡恒等布局
out = LayoutPass("auto", target=target).run(circuit)  # 自动：按交互频率贪心相邻化
```

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `initial_layout` | `None` | `dict`（`logical→physical`）/ 序列（下标=逻辑位，值=物理位）/ `"auto"` / `None`=恒等 |
| `target` | `None` | 给出时输出 `n_qubits` 取 `target.n_qubits` 并校验物理位范围；`"auto"` 必填 |

- 映射必须单射（不同逻辑位不能映射到同一物理位）。
- `"auto"`：按双比特门交互频率降序，贪心地把高频交互的逻辑对放到耦合图相邻的物理比特上，减少后续 `RoutingPass` 的 SWAP；需 `target`，全连接时退为恒等。为贪心启发式（非全局最优），按噪声择优留待后续。

### 4.8  RoutingPass — 拓扑路由

沿 `Target.coupling_map` 最短路径插入 SWAP，使每个双比特门作用在相邻物理比特上。采用**置换跟踪**：施加该门后**不复位** SWAP，而是把由此产生的比特置换向前携带，后续门按当前位置重新映射。因此整条线路与原线路等价**至最终比特置换**（`final_layout`），SWAP 数量比“插入-复位”方案大致减半，且相邻化的比特对在后续门上无需再插 SWAP。

```python
from aicir.transpile import RoutingPass
from aicir.devices import Target

target = Target(n_qubits=4, coupling_map=[(0, 1), (1, 2), (2, 3)])
rp = RoutingPass(target=target)
out = rp.run(circuit)
rp.final_layout   # {logical -> physical wire}，覆盖全部物理线，未移动者恒等
```

| 参数 | 说明 |
| --- | --- |
| `target` | 提供 `coupling_map` 的 `Target`（必填）；全连接时本 pass 为恒等 |

- 假设线路比特已是物理比特（通常先经 `LayoutPass`）。
- 仅支持单比特门与恰好 2 个不同比特的双比特门；更高阶门（如 `toffoli`）抛 `NotImplementedError`。
- 运行后 `final_layout` 给出最终 `logical -> physical wire` 置换；`last_layout` 镜像它（恒等时为 `None`），供 `PassManager.run_with_result` 记入 `TranspileResult.layout`，并与前置 `LayoutPass` 链式组合（`composed[q] = routing[layout[q]]`）。读测量结果时需按该置换还原比特顺序。
- SWAP 数仍非全局最优（贪心、未跨门优化插入顺序）；基于代价的最优路由留待后续。
- 典型组合：`LayoutPass → RoutingPass → DecomposePass`（把插入的 `swap` 再分解为 `cx`）。

---

## 5  编写自定义 Pass

继承 `TransformationPass` 并实现 `run` 方法即可：

```python
from aicir.transpile import TransformationPass
from aicir.core.circuit import Circuit

class MyCustomPass(TransformationPass):
    """移除所有 identity 门。"""

    def run(self, circuit: Circuit) -> Circuit:
        from aicir.ir import circuit_gate_dicts
        gates = [g for g in circuit_gate_dicts(circuit) if g["type"] != "identity"]
        return Circuit(*gates, n_qubits=circuit.n_qubits)
```

自定义 pass 可直接传入 `PassManager`：

```python
from aicir.transpile import PassManager, CancelInversePass

pm = PassManager([
    CancelInversePass(),
    MyCustomPass(),
])
optimized = pm.run(circuit)
```

---

## 6  入口选择

线路结构优化已统一收归本模块（旧的 `aicir.optimizer.circuit` /
`default_optimization_pipeline` 已移除）。

| 场景 | 推荐方式 |
| --- | --- |
| 优化一个 `Circuit` | `optimize(circuit)` 或等价的 `optimize_circuit(circuit)` |
| 优化 dict / OpenQASM / DAG | `optimize_basic(obj)` |
| 需要自定义编译流程 | `PassManager([...]).run(circuit)` |

---

## 7  公共 API 速查

### 从 `aicir.transpile` 直接导入

| 名称 | 类型 | 说明 |
| --- | --- | --- |
| `TransformationPass` | 抽象基类 | 自定义 pass 的基类 |
| `PassManager` | 类 | 流水线调度器 |
| `optimize` | 函数 | 默认本地优化入口，返回优化后的新 `Circuit` |
| `optimize_circuit` | 函数 | `optimize` 的 Circuit 专用别名 |
| `optimize_basic` | 函数 | 多格式前端（dict / OpenQASM / DAG） |
| `ValidatePass` | Pass | 结构校验 |
| `CanonicalizePass` | Pass | 门名规范化 |
| `CancelInversePass` | Pass | 相邻逆门消去 |
| `MergeRotationsPass` | Pass | 同轴旋转合并 |
| `CommuteSingleQubitPass` | Pass | 交换律回看优化 |
| `DecomposePass` | Pass | 高级门分解到目标门集 |
| `LayoutPass` | Pass | logical→physical 比特重标号 |
| `RoutingPass` | Pass | 沿耦合图插入 SWAP 满足拓扑 |

> `Target`（硬件目标描述）从 `aicir.devices` 导入，是 `DecomposePass`/`LayoutPass`/`RoutingPass` 的共同输入。
