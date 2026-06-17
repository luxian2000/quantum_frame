# measure_qubits 语义反转 + 移除 tm — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `Measure.run` 的 `measure_qubits` 成为末端读出的唯一控制项，反转空/None 语义（`None`=不测，`[]`=全测，默认 `[]`），并移除 `tm` 参数。

**Architecture:** 改动集中在 `aicir/measure/measure.py` 的 `run()` 末端读出解析逻辑：删除 `tm`/`mq_explicit`/exact 冲突分支，改为「exact → 永不测量且忽略 measure_qubits；shot 模式 → None=off、[]=全部、[list]=子集」。下游 `sampler.py`、`result.py`、README、CHANGELOG 与测试同步。`trajectory.py` 不动（上层始终传具体列表或 `tm=False`）。

**Tech Stack:** Python 3、numpy、pytest（`PYTHONPATH=.` 从仓库根运行）。注释/文档为中文。

**关键规则速查（新语义）：**

| 调用 | 结果 |
|---|---|
| `run(cir)` / `run(cir, shots≥1)` | shot 模式，读出**全部**比特 |
| `run(cir, shots≥1, measure_qubits=None)` | shot 模式，**不**做末端测量 |
| `run(cir, shots≥1, measure_qubits=[0])` | shot 模式，仅读出 qubit 0 |
| `run(cir, shots=None)` / `shots=0` | exact，off |
| `run(cir, shots=None, measure_qubits=任意)` | exact，off（**不报错**） |

---

### Task 1: 在 measure.py 中反转 run() 的末端读出语义并移除 tm

**Files:**
- Modify: `aicir/measure/measure.py`（`run()` 签名 122–125、docstring 126–142、末端解析 161–174）
- Test: `tests/measure/test_unified_run.py`

本任务是核心。先迁移/新增测试，再改实现。

- [ ] **Step 1: 改写 test_unified_run.py 以表达新语义**

把 `tests/measure/test_unified_run.py` 中以下三处替换/新增（其余测试保持不变）。

将第 33–37 行的 `test_incircuit_measure_collapses_and_output_indexed` 中
`run(cir, shots=16, tm=False)` 改为 `run(cir, shots=16, measure_qubits=None)`：

```python
def test_incircuit_measure_collapses_and_output_indexed():
    cir = Circuit(hadamard(0), cnot(1, [0]), measure([0, 1]), n_qubits=2)  # op2 = measure
    r = run(cir, shots=16, measure_qubits=None)
    assert r.output(2).shape == (16, 1)
    assert set(np.unique(r.output(2))) <= {1}
```

删除旧的冲突用例（第 47–56 行的 `test_conflict_tm_false_with_measure_qubits`
与 `test_conflict_exact_mode_with_explicit_measure_qubits`），替换为新语义用例：

```python
def test_measure_qubits_none_disables_terminal_in_shot_mode():
    cir = Circuit(hadamard(0), n_qubits=1)
    r = run(cir, shots=8, measure_qubits=None)
    assert r.terminal_qubits is None
    with pytest.raises(ValueError):
        r.output(-1)


def test_measure_qubits_empty_reads_all_in_shot_mode():
    cir = Circuit(hadamard(0), cnot(1, [0]), n_qubits=2)
    r = run(cir, shots=8, measure_qubits=[])
    assert r.terminal_qubits == [0, 1]
    assert r.output(-1).shape == (8, 2)


def test_exact_mode_ignores_measure_qubits_no_error():
    cir = Circuit(hadamard(0), n_qubits=1)
    for mq in (None, [], [0]):
        r = run(cir, shots=None, measure_qubits=mq)
        assert r.terminal_qubits is None
        with pytest.raises(ValueError):
            r.output(-1)
```

将第 71–75 行 `test_seed_reproducible` 与第 78–81 行 `test_duplicate_id_raises`
中的 `tm=False` 改为 `measure_qubits=None`：

```python
def test_seed_reproducible():
    cir = Circuit(hadamard(0), measure(0), n_qubits=1)
    a = run(cir, shots=32, seed=7, measure_qubits=None).output(1)
    b = run(cir, shots=32, seed=7, measure_qubits=None).output(1)
    assert np.array_equal(a, b)


def test_duplicate_id_raises():
    cir = Circuit(measure(0, id="m"), measure(0, id="m"), n_qubits=1)
    with pytest.raises(ValueError):
        run(cir, shots=4, measure_qubits=None)
```

- [ ] **Step 2: 运行测试，确认新用例失败**

Run: `PYTHONPATH=. pytest tests/measure/test_unified_run.py -q`
Expected: FAIL —`test_measure_qubits_empty_reads_all_in_shot_mode` 等失败
（当前 `[]` 表示不测、`tm=False` 关键字仍存在但新断言不符），且
`run(..., tm=...)` 已被移除前仍以旧行为运行。

- [ ] **Step 3: 改写 run() 签名与 docstring**

在 `aicir/measure/measure.py`，把第 122–142 行的签名与 docstring 改为
（删除 `tm`，`measure_qubits` 默认 `[]`）：

```python
    def run(self, circuit, shots=1, measure_qubits=[], snap=None,
            sm="avg", seed=None, *,
            initial_state=None, initial_density_matrix=None,
            observables=None, return_state=True) -> Result:
        """统一测量入口。

        参数:
            circuit:                 待测电路（需具备 n_qubits 属性）
            shots:                   采样次数；None 或 0 表示 exact 模式（单条精确轨迹，
                                     不做末端测量，且忽略 measure_qubits）；≥1 表示 M 条轨迹按 sm 聚合
            measure_qubits:          末端读出比特控制（仅 shot 模式生效）：
                                     None=不做末端测量；[]（默认）=读出全部比特；
                                     [q0, q1, …]=读出该子集（保留输入顺序）。
                                     exact 模式下该参数被忽略、不报错
            snap:                    需记录完整态快照的操作下标集合
            sm:                      多轨迹聚合模式，目前仅支持 'avg'（'shot'/'cond' 暂未实现）
            seed:                    随机种子（用于复现）
            initial_state:           初始态（None 表示 |0...0>）
            initial_density_matrix:  初始密度矩阵（提供时以密度矩阵模式初始化量子态；
                                     与 initial_state 互斥）
            observables:             可观测量字典 {name: operator_matrix}
            return_state:            是否在结果中附带 state / final_state
        """
```

注意：可变默认值 `[]` 安全，因为下方仅通过 `_normalize_measure_qubits`
读取它并返回**新列表**，绝不就地修改。

- [ ] **Step 4: 改写末端读出解析逻辑**

把第 161–174 行（从注释 `# 末端测量解析...` 到 `terminal_qubits = ...` 那行）
整段替换为：

```python
        # 末端读出解析：exact 模式永不测量且忽略 measure_qubits；
        # shot 模式下 None=off、[]=全部、[list]=子集。
        if exact or measure_qubits is None:
            do_terminal = False
            terminal_qubits = None
        else:
            norm_mq = self._normalize_measure_qubits(measure_qubits, n)
            terminal_qubits = norm_mq if len(norm_mq) > 0 else list(range(n))
            do_terminal = True
```

（`_normalize_measure_qubits` 对 `[]` 直接返回 `[]`，对越界/重复仍抛 ValueError。）

- [ ] **Step 5: 运行测试，确认通过**

Run: `PYTHONPATH=. pytest tests/measure/test_unified_run.py -q`
Expected: PASS（全部用例）。

- [ ] **Step 6: 提交**

```bash
git add aicir/measure/measure.py tests/measure/test_unified_run.py
git commit -m "feat(measure): 反转 measure_qubits 语义并移除 tm 参数

None=不测, []=全测(默认), [list]=子集; exact 模式忽略 measure_qubits

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: 更新 ShotSampler 默认 measure_qubits

**Files:**
- Modify: `aicir/primitives/sampler.py:93`
- Test: `tests/measure/test_unified_run.py`（复用）或现有 sampler 测试

`ShotSampler` 总是 shot 模式，默认 `measure_qubits=None` 会导致无 counts。改为 `[]`（全测）。

- [ ] **Step 1: 写失败测试**

在 `tests/measure/test_unified_run.py` 末尾追加：

```python
def test_shotsampler_defaults_to_full_readout():
    from aicir.primitives.sampler import ShotSampler
    cir = Circuit(hadamard(0), cnot(1, [0]), n_qubits=2)
    res = ShotSampler(shots=128).run(cir)
    counts = res.counts(-1) if hasattr(res, "counts") else res.counts
    assert sum(counts.values()) == 128
    assert set(counts) <= {"00", "11"}
```

- [ ] **Step 2: 运行，确认失败**

Run: `PYTHONPATH=. pytest tests/measure/test_unified_run.py::test_shotsampler_defaults_to_full_readout -q`
Expected: FAIL — 默认 `measure_qubits=None` 导致无末端测量，`counts(-1)` 报错。

- [ ] **Step 3: 改默认值**

在 `aicir/primitives/sampler.py` 第 93 行，把
`def run(self, circuits, *, shots: int | None = None, measure_qubits=None):`
改为：

```python
    def run(self, circuits, *, shots: int | None = None, measure_qubits=[]):
```

- [ ] **Step 4: 运行，确认通过**

Run: `PYTHONPATH=. pytest tests/measure/test_unified_run.py::test_shotsampler_defaults_to_full_readout -q`
Expected: PASS。

若 `_sample_result_from_measure` 返回对象的 counts 访问方式与断言不符，
按其真实接口调整断言（读 `aicir/primitives/sampler.py` 顶部 `_sample_result_from_measure`），但不改默认值结论。

- [ ] **Step 5: 提交**

```bash
git add aicir/primitives/sampler.py tests/measure/test_unified_run.py
git commit -m "feat(sampler): ShotSampler 默认 measure_qubits=[] 读出全部比特

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: 更新 Result.output 的报错文案

**Files:**
- Modify: `aicir/measure/result.py:63`

- [ ] **Step 1: 改文案**

把第 63 行：

```python
                raise ValueError("未执行末端测量：output(-1) 不可用（tm=False / measure_qubits=[] / shots∈{None,0}）")
```

改为：

```python
                raise ValueError("未执行末端测量：output(-1) 不可用（measure_qubits=None / shots∈{None,0}）")
```

- [ ] **Step 2: 运行受影响测试**

Run: `PYTHONPATH=. pytest tests/measure -q`
Expected: PASS（文案不被断言匹配，仅触发 ValueError 类型；保持绿）。

- [ ] **Step 3: 提交**

```bash
git add aicir/measure/result.py
git commit -m "docs(measure): 更新 output(-1) 报错文案以匹配新 measure_qubits 语义

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: 迁移 test_measure_run_semantics.py

**Files:**
- Modify: `tests/measure/test_measure_run_semantics.py`

- [ ] **Step 1: 更新模块顶部语义注释**

把第 23 行：

```python
- measure_qubits     → 与 tm=False 或 exact 模式互斥（均抛 ValueError）
```

改为：

```python
- measure_qubits     → None=不测；[]=全测（默认）；[list]=子集；
                        exact 模式下忽略 measure_qubits（不报错）
```

- [ ] **Step 2: 替换 exact 冲突用例为「忽略」用例**

把第 70–74 行的 `test_exact_mode_rejects_explicit_measure_qubits` 替换为：

```python
def test_exact_mode_ignores_measure_qubits(m):
    # exact 模式忽略 measure_qubits（None/[]/[list] 均不测、不报错）
    for shots in (None, 0):
        for mq in (None, [], [0]):
            result = m.run(bell_circuit(), shots=shots, measure_qubits=mq)
            assert result.terminal_qubits is None
            with pytest.raises(ValueError):
                result.output(-1)
```

- [ ] **Step 3: 运行，确认通过**

Run: `PYTHONPATH=. pytest tests/measure/test_measure_run_semantics.py -q`
Expected: PASS（其余用例如 `test_single_shot_subset_measure_qubits`、
`test_multi_shot_subset_output_and_density` 用非空子集，行为不变）。

- [ ] **Step 4: 提交**

```bash
git add tests/measure/test_measure_run_semantics.py
git commit -m "test(measure): 迁移 run 语义测试到新 measure_qubits 约定

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: 迁移 test_measure.py 与 test_readme_measure_examples.py

**Files:**
- Modify: `tests/measure/test_measure.py:138-144`
- Modify: `tests/docs/test_readme_measure_examples.py`（第 26、41、68 行的 `tm=False`）

- [ ] **Step 1: 在 test_measure.py 增加 []=全测 用例**

把第 138–144 行 `test_explicit_measure_qubits_with_all_qubits` 之后追加一个
等价于「默认/全测」的 `[]` 用例（保留原 `[0, 1]` 用例不动）：

```python
    def test_empty_measure_qubits_reads_all(self):
        # measure_qubits=[] 表示读出全部比特
        result = self.measure.run(self.bell, shots=200, measure_qubits=[])

        self.assertEqual(result.terminal_qubits, [0, 1])
        counts = result.counts(-1)
        self.assertTrue(all(len(k) == 2 for k in counts))
```

- [ ] **Step 2: 把 test_readme_measure_examples.py 的 tm=False 改为 measure_qubits=None**

第 26 行 `r = run(cir, shots=None, tm=False)` → `r = run(cir, shots=None, measure_qubits=None)`
第 41 行 `r = run(cir, shots=8, tm=False)` → `r = run(cir, shots=8, measure_qubits=None)`
第 68 行 `r = run(cir, shots=None, tm=False)` → `r = run(cir, shots=None, measure_qubits=None)`

- [ ] **Step 3: 运行，确认通过**

Run: `PYTHONPATH=. pytest tests/measure/test_measure.py tests/docs/test_readme_measure_examples.py -q`
Expected: PASS。

- [ ] **Step 4: 提交**

```bash
git add tests/measure/test_measure.py tests/docs/test_readme_measure_examples.py
git commit -m "test(measure): 迁移剩余测试到新 measure_qubits 语义

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: 更新 README §4 文档

**Files:**
- Modify: `README.md`（§4 开头 529、§4.1 538–557、§4.5 636–641、§4.6 657）

- [ ] **Step 1: §4 开头与 §4.1 代码块/参数表**

第 529 行去掉 `tm`：

```markdown
aicir 采用**统一测量模型**：线路内的 `measure`/`reset` 是操作序列中的正式操作，末端读出由 `measure_qubits` 控制；线路内操作与末端读出可共存。
```

第 538–541 行代码块改为（删除 `tm` 行、更新两处注释）：

```python
    shots=1,             # 采样次数；None/0 = exact 模式（不做末端测量，忽略 measure_qubits）
    measure_qubits=[],   # 末端读出：None=不测；[]=全部(默认)；[q0,…]=子集(保留顺序)
    snap=None,           # 记录完整态快照的操作下标列表
```

第 555 行参数表行改为：

```markdown
| `measure_qubits` | `[]` | `None` = 不做末端测量；`[]` = 全部比特 `[0..n−1]`；`[q0,…]` = 子集（保留顺序）；exact 模式忽略本参数 |
```

删除第 557 行的 `tm` 表行。

- [ ] **Step 2: §4.5 末端测量**

把第 636–641 行改为：

```markdown
电路全部显式操作执行完后，shot 模式（`shots≥1`）且 `measure_qubits` 非 `None` 时，对所列比特执行**逐比特 Z 基**投影测量：

- **输入顺序保留**：`measure_qubits=[1, 0]` 时 `output(-1)` 列顺序为 `[qubit1, qubit0]`，不做内部排序。
- `output(-1)` 形状：`shots=1` 时 `(1, k)`，`shots=M` 时 `(M, k)`（`k=len(measure_qubits)`，`[]` 时 `k=n`）。
- `measure_qubits=[]`（默认）：读出全部 `n` 个比特。
- `measure_qubits=None`：不做末端测量。
```

- [ ] **Step 3: §4.6 exact 模式表行**

把第 657 行改为：

```markdown
| `None` 或 `0` | exact 模式 | 单条精确轨迹；忽略 `measure_qubits`、不执行末端测量；`output(-1)` / `counts(i)` / `prob(i)` 报错 |
```

- [ ] **Step 4: 校验 README 示例仍可运行**

Run: `PYTHONPATH=. pytest tests/docs/test_readme_measure_examples.py -q`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add README.md
git commit -m "docs(readme): 更新 §4 测量模型为新 measure_qubits 语义、移除 tm

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: 新增 CHANGELOG 条目并全量回归

**Files:**
- Modify: `CHANGELOG.md`（在第 4 行 `## 2026-06-16` 之前插入新日期段）

- [ ] **Step 1: 插入 2026-06-17 条目**

在 `CHANGELOG.md` 第 4 行（`## 2026-06-16` 之前）插入：

```markdown
## 2026-06-17

### Changed

- **`Measure.run` 的 `measure_qubits` 语义反转、移除 `tm` 参数**（破坏性接口变更）：
  `measure_qubits` 成为末端读出的唯一控制项——`None`=不做末端测量；`[]`（新默认）=读出全部比特；`[q0,…]`=读出子集（保留顺序）。原 `tm` 布尔参数删除（原 `tm=False` 等价 `measure_qubits=None`，原 `tm=True` 即默认 `[]`）。exact 模式（`shots∈{None,0}`）下 `measure_qubits` 被忽略、不再报错。`ShotSampler.run` 默认 `measure_qubits=[]`。涉及 `aicir/measure/measure.py`、`aicir/primitives/sampler.py`、`aicir/measure/result.py` 及 README §4。

```

- [ ] **Step 2: 全量回归**

Run: `PYTHONPATH=. pytest tests/measure tests/docs/test_readme_measure_examples.py -q`
Expected: PASS（全绿）。

- [ ] **Step 3: 全仓库搜索遗留的 `tm=` 调用**

Run: `grep -rn "tm=" aicir tests demos --include="*.py" | grep -v "run_trajectory" | grep -v "def run_trajectory"`
Expected: 仅 `trajectory.py` 内部及其直接测试 `tests/measure/test_trajectory.py` 出现 `tm=`（这是 `run_trajectory` 的底层参数，**保留**）。若 `Measure().run(...)` 或 README 仍有 `tm=`，回到对应任务修正。

- [ ] **Step 4: 提交**

```bash
git add CHANGELOG.md
git commit -m "docs(changelog): 记录 measure_qubits 语义反转与 tm 移除

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## 验收标准

- `PYTHONPATH=. pytest tests/measure tests/docs/test_readme_measure_examples.py` 全绿。
- `Measure.run` 与 `ShotSampler.run` 不再接受 `tm`；`measure_qubits` 默认 `[]`。
- README §4、CHANGELOG 与代码行为一致。
- `grep "tm="` 仅命中 `trajectory.py` 底层与其测试。
