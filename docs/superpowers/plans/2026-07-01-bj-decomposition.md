# Billionnet–Jaumard Core Decomposition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `split_cores(poly)` — the Billionnet–Jaumard (B–J) decomposition — as the documented extension point behind the existing connected-components `split_polynomial`, splitting a *connected* quadratic QUBO into independent hard cores whose minima sum to the global minimum.

**Architecture:** Three focused modules. `posiform.py` converts a `Polynomial` to a posiform (nonnegative coefficients over literals + a constant) by elementary complementation (`x = 1 − x̄`, no roof duality). `implication.py` builds the 2-SAT implication graph from the posiform and computes strongly connected components (iterative Tarjan). `split.py` gains `split_cores`, which groups posiform monomials by self-dual SCC (B–J Thm 3.1), expands each group back to an `x`-space `Polynomial`, and returns the hard cores plus the posiform constant. The consistent "leftover" is provably `min = 0` (Aspvall–Plass–Tarjan Thm 2.1) so it is dropped.

**Tech Stack:** Pure Python 3.11 (no new dependencies), existing `aicir.optimization.qubo.modeling` package, `pytest`.

## Global Constraints

- Run everything from repo root with `PYTHONPATH=.` (no installed package required).
- No new third-party dependencies — pure Python only (`torch`/`scipy`/`matplotlib` must not be imported).
- Quadratic-only: any input with `poly.degree() > 2` raises `ValueError` with the substring `"degree <= 2"` (matches `Polynomial.to_qubo_indices`).
- Docstrings/comments in **English** — the `modeling/` subpackage uses English throughout (unlike the Chinese-commented core).
- Public function names are **two words**, `verb_noun` (never three-word `xxx_yyy_zzz`): `to_posiform`, `build_graph`, `strong_components`, `split_cores`.
- Literal encoding is fixed: variable id `v` → positive literal node `2*v`, complement node `2*v + 1`; `complement(node) == node ^ 1`.
- `split_cores` contract is **weaker than `split_polynomial`**: it preserves only the *minimum* (`min f == sum(min(p) for p in pieces)`), NOT the term partition (the min-0 leftover is discarded). This must be stated in its docstring.

---

### Task 1: Posiform conversion

**Files:**
- Create: `aicir/optimization/qubo/modeling/posiform.py`
- Test: `tests/optimization/qubo/test_posiform.py`

**Interfaces:**
- Consumes: `Polynomial` (`terms: dict[tuple[int,...], float]`, `.degree()`, `.registry`) from `modeling/polynomial.py`; `VariableRegistry`/`GLOBAL_REGISTRY` from `modeling/registry.py`.
- Produces:
  - `class Posiform` — frozen dataclass with `terms: dict[tuple[int, ...], float]` (keys are sorted tuples of 1–2 literal-node ints, all coeffs > 0), `constant: float`, `registry: VariableRegistry`.
  - `to_posiform(poly: Polynomial) -> Posiform`.

- [ ] **Step 1: Write the failing tests**

Create `tests/optimization/qubo/test_posiform.py`:

```python
from aicir.optimization.qubo import Binary
from aicir.optimization.qubo.modeling import VariableRegistry
from aicir.optimization.qubo.modeling.posiform import to_posiform


def test_positive_quadratic_passes_through():
    reg = VariableRegistry()
    a = Binary("a", registry=reg)  # id 0
    b = Binary("b", registry=reg)  # id 1
    posi = to_posiform(a * b)      # terms {(0, 1): 1}
    assert posi.terms == {(0, 2): 1.0}   # literal nodes 2*0=0, 2*1=2
    assert posi.constant == 0.0


def test_negative_linear_complements():
    reg = VariableRegistry()
    a = Binary("a", registry=reg)  # id 0
    posi = to_posiform(-1 * a)     # -x0 = -1 + x̄0
    assert posi.terms == {(1,): 1.0}     # x̄0 is node 1
    assert posi.constant == -1.0


def test_negative_quadratic_complements():
    reg = VariableRegistry()
    a = Binary("a", registry=reg)  # id 0
    b = Binary("b", registry=reg)  # id 1
    posi = to_posiform(-1 * a * b) # -x0 x1 = -1 + x̄0 + x0 x̄1
    assert posi.terms == {(1,): 1.0, (0, 3): 1.0}   # x̄0 node 1; x0 x̄1 nodes (0, 3)
    assert posi.constant == -1.0


def test_rejects_cubic():
    reg = VariableRegistry()
    a = Binary("a", registry=reg)
    b = Binary("b", registry=reg)
    c = Binary("c", registry=reg)
    try:
        to_posiform(a * b * c)
    except ValueError as exc:
        assert "degree <= 2" in str(exc)
    else:
        raise AssertionError("Expected cubic input to be rejected.")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. python -m pytest tests/optimization/qubo/test_posiform.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'aicir.optimization.qubo.modeling.posiform'`

- [ ] **Step 3: Write the implementation**

Create `aicir/optimization/qubo/modeling/posiform.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field

from .polynomial import Polynomial
from .registry import GLOBAL_REGISTRY, VariableRegistry

# Literal encoding: variable id v -> positive literal node 2v, complement node 2v+1.
# complement(node) == node ^ 1.


@dataclass(frozen=True)
class Posiform:
    """A quadratic pseudo-Boolean function as a posiform.

    Terms are products of one or two literals with strictly positive
    coefficients, plus a separate constant. A literal is a variable or its
    Boolean complement, encoded as an integer node (see module header).
    """

    terms: dict[tuple[int, ...], float]
    constant: float
    registry: VariableRegistry = field(default_factory=lambda: GLOBAL_REGISTRY)


def to_posiform(poly: Polynomial) -> Posiform:
    """Rewrite a degree <= 2 polynomial as a posiform via x = 1 - x_bar.

    Negative coefficients are moved onto complemented literals so that every
    monomial coefficient becomes nonnegative, at the cost of an additive
    constant. The choice x_i x_j = x_i - x_i x_bar_j is fixed (deterministic);
    a stronger posiform would require roof duality, which is intentionally not
    used here.
    """
    if poly.degree() > 2:
        raise ValueError("to_posiform only supports degree <= 2 polynomials.")

    terms: dict[tuple[int, ...], float] = {}
    constant = 0.0

    def add(key: tuple[int, ...], coeff: float) -> None:
        terms[key] = terms.get(key, 0.0) + coeff

    for term, coeff in poly.terms.items():
        if len(term) == 0:
            constant += coeff
        elif len(term) == 1:
            v = term[0]
            if coeff >= 0:
                add((2 * v,), coeff)
            else:
                constant += coeff
                add((2 * v + 1,), -coeff)
        else:
            i, j = term
            if coeff >= 0:
                add((2 * i, 2 * j), coeff)
            else:
                # coeff < 0: c*xi*xj = c + (-c)*x_bar_i + (-c)*xi*x_bar_j
                constant += coeff
                add((2 * i + 1,), -coeff)
                add((2 * i, 2 * j + 1), -coeff)

    cleaned = {key: value for key, value in terms.items() if abs(value) > 1e-12}
    return Posiform(cleaned, constant, poly.registry)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. python -m pytest tests/optimization/qubo/test_posiform.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add aicir/optimization/qubo/modeling/posiform.py tests/optimization/qubo/test_posiform.py
git commit -m "新增 QUBO 多项式到 posiform 的转换（B-J 分解前置）"
```

---

### Task 2: Implication graph + Tarjan SCC

**Files:**
- Create: `aicir/optimization/qubo/modeling/implication.py`
- Test: `tests/optimization/qubo/test_implication.py`

**Interfaces:**
- Consumes: `Posiform` (`.terms` keyed by 1–2 literal-node tuples) from Task 1.
- Produces:
  - `build_graph(posiform: Posiform) -> dict[int, list[int]]` — adjacency list over literal nodes; every referenced node is present as a key (sinks map to `[]`). Arc rules encode each monomial `m_k = 0`: a unit literal `(a,)` adds `a -> a^1`; a pair `(a, b)` adds `a -> b^1` and `b -> a^1`.
  - `strong_components(adjacency: dict[int, list[int]]) -> list[list[int]]` — iterative Tarjan; returns SCCs as lists of nodes.

- [ ] **Step 1: Write the failing tests**

Create `tests/optimization/qubo/test_implication.py`:

```python
from aicir.optimization.qubo import Binary
from aicir.optimization.qubo.modeling import VariableRegistry
from aicir.optimization.qubo.modeling.posiform import to_posiform
from aicir.optimization.qubo.modeling.implication import build_graph, strong_components


def test_build_graph_pair_arcs():
    reg = VariableRegistry()
    a = Binary("a", registry=reg)  # id 0
    b = Binary("b", registry=reg)  # id 1
    posi = to_posiform(a * b)      # terms {(0, 2): 1}
    graph = build_graph(posi)
    # pair (0, 2) -> arcs 0 -> 2^1=3 and 2 -> 0^1=1
    assert set(graph[0]) == {3}
    assert set(graph[2]) == {1}
    assert graph[1] == []
    assert graph[3] == []


def test_build_graph_unit_arc():
    reg = VariableRegistry()
    a = Binary("a", registry=reg)  # id 0
    posi = to_posiform(-1 * a)     # terms {(1,): 1}
    graph = build_graph(posi)
    # unit (1,) -> arc 1 -> 1^1=0
    assert set(graph[1]) == {0}
    assert graph[0] == []


def test_strong_components_cycle_and_singleton():
    adjacency = {0: [1], 1: [2], 2: [0], 3: []}
    components = strong_components(adjacency)
    normalized = sorted(sorted(component) for component in components)
    assert normalized == [[0, 1, 2], [3]]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. python -m pytest tests/optimization/qubo/test_implication.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'aicir.optimization.qubo.modeling.implication'`

- [ ] **Step 3: Write the implementation**

Create `aicir/optimization/qubo/modeling/implication.py`:

```python
from __future__ import annotations

from .posiform import Posiform


def build_graph(posiform: Posiform) -> dict[int, list[int]]:
    """Build the 2-SAT implication graph of the posiform's monomials.

    Each monomial must be 0 at the minimum. A pair m = a*b contributes the
    clause (~a or ~b), i.e. arcs a -> complement(b) and b -> complement(a). A
    unit m = a contributes the clause (~a), i.e. arc a -> complement(a).
    """
    adjacency: dict[int, list[int]] = {}

    def ensure(node: int) -> None:
        if node not in adjacency:
            adjacency[node] = []

    def arc(src: int, dst: int) -> None:
        ensure(src)
        ensure(dst)
        adjacency[src].append(dst)

    for key in posiform.terms:
        if len(key) == 1:
            a = key[0]
            arc(a, a ^ 1)
        else:
            a, b = key
            arc(a, b ^ 1)
            arc(b, a ^ 1)
    return adjacency


def strong_components(adjacency: dict[int, list[int]]) -> list[list[int]]:
    """Tarjan's strongly connected components, iterative to avoid deep recursion."""
    index_counter = [0]
    index: dict[int, int] = {}
    lowlink: dict[int, int] = {}
    on_stack: dict[int, bool] = {}
    stack: list[int] = []
    result: list[list[int]] = []

    def connect(root: int) -> None:
        work = [(root, 0)]
        while work:
            node, cursor = work[-1]
            if cursor == 0:
                index[node] = index_counter[0]
                lowlink[node] = index_counter[0]
                index_counter[0] += 1
                stack.append(node)
                on_stack[node] = True
            recursed = False
            neighbors = adjacency[node]
            for offset in range(cursor, len(neighbors)):
                nxt = neighbors[offset]
                if nxt not in index:
                    work[-1] = (node, offset + 1)
                    work.append((nxt, 0))
                    recursed = True
                    break
                if on_stack.get(nxt):
                    lowlink[node] = min(lowlink[node], index[nxt])
            if recursed:
                continue
            if lowlink[node] == index[node]:
                component: list[int] = []
                while True:
                    member = stack.pop()
                    on_stack[member] = False
                    component.append(member)
                    if member == node:
                        break
                result.append(component)
            work.pop()
            if work:
                parent = work[-1][0]
                lowlink[parent] = min(lowlink[parent], lowlink[node])

    for node in adjacency:
        if node not in index:
            connect(node)
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=. python -m pytest tests/optimization/qubo/test_implication.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add aicir/optimization/qubo/modeling/implication.py tests/optimization/qubo/test_implication.py
git commit -m "新增蕴含图构建与 Tarjan 强连通分量（B-J 分解核心）"
```

---

### Task 3: `split_cores` orchestration + export

**Files:**
- Modify: `aicir/optimization/qubo/modeling/split.py` (append `_literal_poly` and `split_cores`; update the "Extension point" docstring note)
- Modify: `aicir/optimization/qubo/modeling/__init__.py` (import + `__all__`)
- Modify: `CHANGELOG.md` (add a dated entry)
- Test: `tests/optimization/qubo/test_split_cores.py`

**Interfaces:**
- Consumes: `to_posiform` (Task 1); `build_graph`, `strong_components` (Task 2); `Polynomial` from `modeling/polynomial.py`.
- Produces: `split_cores(poly: Polynomial) -> list[Polynomial]` — hard-core subpolynomials over disjoint variable sets, plus one constant piece `Polynomial.constant(K)` when the posiform constant `K != 0`. Guarantees `min f == sum(min(p) for p in pieces)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/optimization/qubo/test_split_cores.py`:

```python
import itertools
import random

from aicir.optimization.qubo import Binary, split_cores
from aicir.optimization.qubo.modeling import VariableRegistry
from aicir.optimization.qubo.modeling.polynomial import Polynomial


def _eval(poly, bits):
    total = 0.0
    for term, coeff in poly.terms.items():
        product = 1
        for var_id in term:
            product *= bits[var_id]
        total += coeff * product
    return total


def _min_energy(poly):
    var_ids = sorted({var_id for term in poly.terms for var_id in term})
    if not var_ids:
        return _eval(poly, {})
    return min(
        _eval(poly, dict(zip(var_ids, combo)))
        for combo in itertools.product((0, 1), repeat=len(var_ids))
    )


def test_split_cores_single_frustrated_core():
    reg = VariableRegistry()
    a = Binary("a", registry=reg)
    b = Binary("b", registry=reg)
    f = a * b - 2 * a - 2 * b + 5   # global minimum is 2

    pieces = split_cores(f)

    assert _min_energy(f) == 2
    assert abs(_min_energy(f) - sum(_min_energy(p) for p in pieces)) < 1e-9


def test_split_cores_preserves_minimum_random():
    rng = random.Random(0)
    for _ in range(50):
        reg = VariableRegistry()
        xs = [Binary(f"x{i}", registry=reg) for i in range(4)]
        f = Polynomial.constant(0.0, reg)
        for i in range(4):
            if rng.random() < 0.5:
                f = f + rng.randint(-3, 3) * xs[i]
        for i in range(4):
            for j in range(i + 1, 4):
                if rng.random() < 0.5:
                    f = f + rng.randint(-3, 3) * xs[i] * xs[j]
        pieces = split_cores(f)
        assert abs(_min_energy(f) - sum(_min_energy(p) for p in pieces)) < 1e-9


def test_split_cores_produces_disjoint_cores():
    reg = VariableRegistry()
    a = Binary("a", registry=reg)
    b = Binary("b", registry=reg)
    c = Binary("c", registry=reg)
    d = Binary("d", registry=reg)
    f = (a * b - 2 * a - 2 * b + 5) + (c * d - 2 * c - 2 * d + 5)

    non_constant = [p for p in split_cores(f) if any(term for term in p.terms)]
    var_sets = [{v for term in p.terms for v in term} for p in non_constant]

    for i in range(len(var_sets)):
        for j in range(i + 1, len(var_sets)):
            assert var_sets[i].isdisjoint(var_sets[j])


def test_split_cores_rejects_cubic():
    reg = VariableRegistry()
    a = Binary("a", registry=reg)
    b = Binary("b", registry=reg)
    c = Binary("c", registry=reg)
    try:
        split_cores(a * b * c)
    except ValueError as exc:
        assert "degree <= 2" in str(exc)
    else:
        raise AssertionError("Expected cubic input to be rejected.")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=. python -m pytest tests/optimization/qubo/test_split_cores.py -q`
Expected: FAIL with `ImportError: cannot import name 'split_cores'`

- [ ] **Step 3: Append the implementation to `split.py`**

Add these imports near the top of `aicir/optimization/qubo/modeling/split.py` (below the existing `from .polynomial import Polynomial, Term`):

```python
from .implication import build_graph, strong_components
from .posiform import to_posiform
```

Append to the end of `aicir/optimization/qubo/modeling/split.py`:

```python
def _literal_poly(node: int, registry) -> Polynomial:
    var_id = node >> 1
    variable = Polynomial({(var_id,): 1.0}, registry)
    if node & 1:  # complemented literal x_bar = 1 - x
        return Polynomial.constant(1.0, registry) - variable
    return variable


def split_cores(poly: Polynomial) -> list[Polynomial]:
    """Split a quadratic polynomial into Billionnet-Jaumard hard cores.

    Converts the polynomial to a posiform, builds the 2-SAT implication graph,
    and isolates the strongly connected components that contain both a variable
    and its complement (the "hard cores", B-J Thm 3.1). Each core is expanded
    back to an x-space subpolynomial; the remaining (consistent) monomials form
    a posiform whose minimum is provably 0 (Aspvall-Plass-Tarjan Thm 2.1) and
    are dropped. The posiform constant is returned as a trailing constant piece.

    Unlike :func:`split_polynomial`, this preserves only the *minimum*, not the
    term partition::

        min f == sum(min f_i for f_i in split_cores(f))

    It is quadratic-only; degree >= 3 raises ``ValueError``. This is the
    posiform/implication-graph refinement referenced in ``split_polynomial``:
    it can break a single *connected* quadratic block into several independent
    cores, which connected-component splitting cannot.
    """
    if poly.degree() > 2:
        raise ValueError("split_cores only supports degree <= 2 polynomials.")

    posiform = to_posiform(poly)
    adjacency = build_graph(posiform)
    components = strong_components(adjacency)

    comp_id = {node: i for i, comp in enumerate(components) for node in comp}
    self_dual: set[int] = set()
    for i, comp in enumerate(components):
        members = set(comp)
        if any((node ^ 1) in members for node in members):
            self_dual.add(i)

    cores: dict[int, Polynomial] = {}
    for key, coeff in posiform.terms.items():
        ids = {comp_id[node] for node in key}
        if len(ids) != 1:
            continue  # spans components -> consistent leftover (min 0)
        (cid,) = ids
        if cid not in self_dual:
            continue  # consistent-only component -> leftover (min 0)
        monomial = Polynomial.constant(coeff, poly.registry)
        for node in key:
            monomial = monomial * _literal_poly(node, poly.registry)
        cores[cid] = cores.get(cid, Polynomial({}, poly.registry)) + monomial

    pieces = [cores[cid] for cid in sorted(cores)]
    if abs(posiform.constant) > 1e-12:
        pieces.append(Polynomial.constant(posiform.constant, poly.registry))
    return pieces
```

Then update the `Extension point` paragraph inside the existing `split_polynomial` docstring so it points at the new function. Replace the sentence beginning `A stronger, sign-aware split ...` with:

```
    A stronger, sign-aware split is implemented in :func:`split_cores`
    (Billionnet-Jaumard): it converts to a posiform, builds the literal
    implication graph, runs Tarjan SCC, and peels off the components holding a
    variable and its complement. It is quadratic-only and can break a single
    connected block that this connected-components split leaves whole.
```

- [ ] **Step 4: Export `split_cores` from the package**

In `aicir/optimization/qubo/modeling/__init__.py`, change the split import line:

```python
from .split import split_polynomial
```

to:

```python
from .split import split_cores, split_polynomial
```

and add `"split_cores"` to `__all__`, immediately before `"split_polynomial"`:

```python
    "split_cores",
    "split_polynomial",
```

- [ ] **Step 5: Run the new tests to verify they pass**

Run: `PYTHONPATH=. python -m pytest tests/optimization/qubo/test_split_cores.py -q`
Expected: PASS (4 passed)

- [ ] **Step 6: Run the full qubo suite to check nothing regressed**

Run: `PYTHONPATH=. python -m pytest tests/optimization/qubo/ -q`
Expected: PASS (all tests, including the pre-existing `test_split.py` and the two new modules)

- [ ] **Step 7: Add a CHANGELOG entry**

Prepend a dated bullet under the most recent heading in `CHANGELOG.md`:

```markdown
- 2026-07-01 新增 `split_cores`（Billionnet-Jaumard 分解）：将二次 QUBO 经 posiform + 蕴含图 + Tarjan SCC 拆分为相互独立的 hard core，`min f == Σ min f_i`，可细分连通块。
```

- [ ] **Step 8: Commit**

```bash
git add aicir/optimization/qubo/modeling/split.py \
        aicir/optimization/qubo/modeling/__init__.py \
        tests/optimization/qubo/test_split_cores.py \
        CHANGELOG.md
git commit -m "新增 split_cores：Billionnet-Jaumard hard core 分解"
```

---

## Deferred (not in this plan)

- **Minimizer recovery for the leftover.** This plan returns the decomposition and the exact minimum value; it does not reconstruct the assignment of the consistent (min-0) variables. That needs the reduced graph + reverse-topological SCC labelling (Aspvall–Plass–Tarjan / Hansen–Jaumard), and would land as a follow-on `solve_cores`/assignment helper. Deferred per YAGNI until a caller needs the full argmin.
- **Roof-duality posiform.** The elementary posiform here is correct but not the strongest split; a roof-duality conversion (Hammer–Hansen–Simeone) would tighten it. Optional, deferred.

## Self-Review

- **Spec coverage:** posiform conversion (Task 1), implication graph + SCC (Task 2), hard-core grouping + expansion + constant handling + export + changelog (Task 3), quadratic-only guard (all three tasks), `min f == Σ min f_i` (Task 3 tests). Minimizer recovery and roof duality are explicitly scoped out above.
- **Placeholder scan:** every code and test step contains complete code; no TODO/TBD/"handle edge cases".
- **Type consistency:** `to_posiform -> Posiform`; `Posiform.terms` keys are literal-node tuples consumed identically by `build_graph`; `build_graph -> dict[int, list[int]]` consumed by `strong_components -> list[list[int]]`; `split_cores` uses `comp_id`/`self_dual` over those components and returns `list[Polynomial]`. Node encoding `2*v` / `2*v+1` / `^1` is used identically across Tasks 1–3.
