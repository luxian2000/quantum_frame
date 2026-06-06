"""Shared helpers: NP-problem QUBO/Ising formulations for supernet demos.

This module turns the Hamiltonian-cycle / Hamiltonian-path formulations from

    Andrew Lucas, "Ising formulations of many NP problems",
    Frontiers in Physics 2 (2014), Section 7.1, Eq. (56).

into an aicir :class:`~aicir.channel.operators.Hamiltonian` whose ground state
encodes the solution, so that :mod:`aicir.qas.supernet` can search a circuit that
prepares it.

Pipeline
--------
1. Build the cost as a QUBO (quadratic polynomial in binary variables
   ``x_{v,i} = 1`` iff vertex ``v`` occupies order position ``i``).
2. Map each binary variable to a qubit via ``x = (1 - Z) / 2`` (standard
   QUBO -> Ising substitution; with aicir's ``Z|0> = +|0>``, ``Z|1> = -|1>``
   this means ``x = 1`` corresponds to qubit state ``|1>``).
3. Minimise ``<psi|H|psi>`` with supernet (``task="vqe"``); the optimal basis
   state is read back and decoded into the vertex ordering.

The formulations here are diagonal (only ``I`` and ``Z`` terms), so the ground
state is a computational-basis state and a brute-force check over the QUBO is
cheap for the small instances used by the demos.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Dict, Iterable, List, Sequence, Tuple

from ...channel.operators import Hamiltonian

# A binary variable is identified by (vertex, position).
Var = Tuple[int, int]
Edge = Tuple[int, int]


# ──────────────────────────────────────────────────────────────────────────────
# QUBO accumulator
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class QUBO:
    """A quadratic pseudo-Boolean cost over binary variables (x in {0, 1}).

    Stored as a constant, linear coefficients, and (unordered) quadratic
    coefficients. ``x_i^2 = x_i`` simplification is applied automatically.
    """

    const: float = 0.0
    linear: Dict[Var, float] = field(default_factory=dict)
    quad: Dict[frozenset, float] = field(default_factory=dict)

    def add_const(self, c: float) -> None:
        self.const += c

    def add_linear(self, v: Var, c: float) -> None:
        self.linear[v] = self.linear.get(v, 0.0) + c

    def add_quad(self, i: Var, j: Var, c: float) -> None:
        if i == j:  # x_i * x_i = x_i
            self.add_linear(i, c)
            return
        key = frozenset((i, j))
        self.quad[key] = self.quad.get(key, 0.0) + c

    def add_squared(self, weight: float, const: float, coefs: Dict[Var, float]) -> None:
        """Add ``weight * (const + sum_v coefs[v] * x_v) ** 2`` (with x^2 = x)."""
        self.add_const(weight * const * const)
        for v, cv in coefs.items():
            # cross term 2*const*cv*x_v plus cv^2 * x_v^2 (= cv^2 * x_v)
            self.add_linear(v, weight * (2.0 * const * cv + cv * cv))
        items = list(coefs.items())
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                (vi, ci), (vj, cj) = items[a], items[b]
                self.add_quad(vi, vj, weight * 2.0 * ci * cj)

    def pin(self, fixed: Dict[Var, int]) -> "QUBO":
        """Substitute fixed 0/1 values, returning a QUBO over the free vars."""
        out = QUBO(const=self.const)
        for v, c in self.linear.items():
            if v in fixed:
                out.const += c * fixed[v]
            else:
                out.add_linear(v, c)
        for key, c in self.quad.items():
            i, j = tuple(key)
            fi, fj = i in fixed, j in fixed
            if fi and fj:
                out.const += c * fixed[i] * fixed[j]
            elif fi:
                if fixed[i]:
                    out.add_linear(j, c)
            elif fj:
                if fixed[j]:
                    out.add_linear(i, c)
            else:
                out.add_quad(i, j, c)
        return out

    def variables(self) -> List[Var]:
        seen = set(self.linear)
        for key in self.quad:
            seen.update(key)
        return sorted(seen)

    def energy(self, assignment: Dict[Var, int]) -> float:
        value = self.const
        for v, c in self.linear.items():
            value += c * assignment[v]
        for key, c in self.quad.items():
            i, j = tuple(key)
            value += c * assignment[i] * assignment[j]
        return value

    def brute_force_minimum(self) -> Tuple[float, Dict[Var, int]]:
        """Exhaustively minimise the QUBO (only for small variable counts)."""
        variables = self.variables()
        best_energy = float("inf")
        best_assignment: Dict[Var, int] = {}
        for bits in product((0, 1), repeat=len(variables)):
            assignment = dict(zip(variables, bits))
            energy = self.energy(assignment)
            if energy < best_energy - 1e-12:
                best_energy = energy
                best_assignment = assignment
        return best_energy, best_assignment


# ──────────────────────────────────────────────────────────────────────────────
# Hamiltonian-cycle / Hamiltonian-path QUBO (Lucas 2014, Eq. 56)
# ──────────────────────────────────────────────────────────────────────────────

def ordering_qubo(
    n_vertices: int,
    edges: Iterable[Edge],
    *,
    weight: float = 1.0,
    cyclic: bool = True,
    pin_first_vertex: bool = False,
) -> QUBO:
    """Build the QUBO of Lucas (2014), Eq. (56).

    Variables ``x_{v,i}`` (``v`` = vertex, ``i`` = order position, both in
    ``0..n_vertices-1``) satisfy three penalty terms:

    1. every vertex appears in exactly one position,
    2. every position holds exactly one vertex,
    3. consecutive positions ``i`` and ``i+1`` must be joined by an edge.

    Args:
        n_vertices: Number of graph vertices ``N``.
        edges: Undirected edges as ``(u, v)`` pairs.
        weight: Penalty scale ``A`` (any ``A > 0`` works; the optimum is 0).
        cyclic: ``True`` for Hamiltonian cycles (position ``N`` wraps to ``0``);
            ``False`` for Hamiltonian paths (no wrap-around).
        pin_first_vertex: For cycles only, pin vertex ``0`` to position ``0``
            (the ``(N-1)^2`` spin reduction the paper recommends, since a cycle
            can always be rotated to start at vertex 0).

    Returns:
        The :class:`QUBO`; ground-state energy ``0`` iff a cycle/path exists.
    """
    edge_set = {frozenset(e) for e in edges}
    qubo = QUBO()

    # Term 1: each vertex occupies exactly one position.
    for v in range(n_vertices):
        qubo.add_squared(weight, 1.0, {(v, i): -1.0 for i in range(n_vertices)})

    # Term 2: each position is occupied by exactly one vertex.
    for i in range(n_vertices):
        qubo.add_squared(weight, 1.0, {(v, i): -1.0 for v in range(n_vertices)})

    # Term 3: penalise consecutive non-adjacent vertices.
    n_transitions = n_vertices if cyclic else n_vertices - 1
    for i in range(n_transitions):
        j = (i + 1) % n_vertices
        for u in range(n_vertices):
            for w in range(n_vertices):
                if u == w:
                    continue
                if frozenset((u, w)) not in edge_set:
                    qubo.add_quad((u, i), (w, j), weight)

    if pin_first_vertex and cyclic:
        fixed: Dict[Var, int] = {(0, 0): 1}
        for i in range(1, n_vertices):
            fixed[(0, i)] = 0
        for v in range(1, n_vertices):
            fixed[(v, 0)] = 0
        qubo = qubo.pin(fixed)

    return qubo


# ──────────────────────────────────────────────────────────────────────────────
# QUBO -> Ising Hamiltonian
# ──────────────────────────────────────────────────────────────────────────────

def qubo_to_hamiltonian(
    qubo: QUBO,
) -> Tuple[Hamiltonian, Dict[Var, int]]:
    """Convert a QUBO to a diagonal Ising :class:`Hamiltonian` (x = (1 - Z)/2).

    Returns the Hamiltonian and the ``variable -> qubit index`` map (variables
    sorted, assigned to qubits ``0..n-1``).
    """
    variables = qubo.variables()
    var_to_qubit = {v: q for q, v in enumerate(variables)}
    n_qubits = len(variables)

    const = qubo.const
    z_coef: Dict[int, float] = {}
    zz_coef: Dict[frozenset, float] = {}

    # linear: a * x = a/2 - (a/2) Z
    for v, a in qubo.linear.items():
        q = var_to_qubit[v]
        const += a / 2.0
        z_coef[q] = z_coef.get(q, 0.0) - a / 2.0

    # quadratic: b * x_i x_j = b/4 (1 - Z_i - Z_j + Z_i Z_j)
    for key, b in qubo.quad.items():
        i, j = tuple(key)
        qi, qj = var_to_qubit[i], var_to_qubit[j]
        const += b / 4.0
        z_coef[qi] = z_coef.get(qi, 0.0) - b / 4.0
        z_coef[qj] = z_coef.get(qj, 0.0) - b / 4.0
        zz_coef[frozenset((qi, qj))] = zz_coef.get(frozenset((qi, qj)), 0.0) + b / 4.0

    terms = [("I", complex(const), [0])]
    for q, c in sorted(z_coef.items()):
        if abs(c) > 1e-12:
            terms.append(("Z", complex(c), [q]))
    for key, c in sorted(zz_coef.items(), key=lambda kv: sorted(kv[0])):
        if abs(c) > 1e-12:
            qi, qj = sorted(key)
            terms.append(("ZZ", complex(c), [qi, qj]))

    return Hamiltonian(n_qubits=n_qubits, terms=terms), var_to_qubit


# ──────────────────────────────────────────────────────────────────────────────
# Decoding
# ──────────────────────────────────────────────────────────────────────────────

def bitstring_to_assignment(
    basis_index: int,
    var_to_qubit: Dict[Var, int],
    n_qubits: int,
    pinned: Dict[Var, int] | None = None,
) -> Dict[Var, int]:
    """Decode a computational-basis index into variable values.

    aicir uses the MSB convention: qubit ``q`` is bit ``(idx >> (n-1-q)) & 1``.
    """
    assignment: Dict[Var, int] = dict(pinned or {})
    for var, qubit in var_to_qubit.items():
        assignment[var] = (basis_index >> (n_qubits - 1 - qubit)) & 1
    return assignment


def assignment_to_order(assignment: Dict[Var, int], n_vertices: int) -> List[int | None]:
    """Turn an ``x_{v,i}`` assignment into ``order[i] = vertex at position i``.

    Returns ``None`` at any position that is not occupied by exactly one vertex.
    """
    order: List[int | None] = [None] * n_vertices
    for i in range(n_vertices):
        occupants = [v for v in range(n_vertices) if assignment.get((v, i), 0) == 1]
        order[i] = occupants[0] if len(occupants) == 1 else None
    return order


def is_valid_ordering(
    order: Sequence[int | None],
    edges: Iterable[Edge],
    *,
    cyclic: bool,
) -> bool:
    """Check that ``order`` is a permutation whose consecutive vertices are edges."""
    n = len(order)
    if any(v is None for v in order) or sorted(order) != list(range(n)):  # type: ignore[arg-type]
        return False
    edge_set = {frozenset(e) for e in edges}
    n_transitions = n if cyclic else n - 1
    for i in range(n_transitions):
        u, w = order[i], order[(i + 1) % n]
        if frozenset((u, w)) not in edge_set:  # type: ignore[arg-type]
            return False
    return True


def format_order(order: Sequence[int | None], *, cyclic: bool) -> str:
    parts = [str(v) if v is not None else "?" for v in order]
    if cyclic and order and order[0] is not None:
        parts.append(str(order[0]))
    return " -> ".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Ground-state search with supernet (multi-restart)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class QASSolution:
    """Outcome of a supernet ground-state search."""

    energy: float
    probabilities: "object"        # backend probability vector (numpy array)
    best_index: int                # argmax computational-basis index
    result: "object"               # the winning SupernetResult
    seed: int                      # restart seed that won
    attempts: List[Tuple[int, float]]  # (seed, energy) for every restart


def solve_ground_state_qas(
    hamiltonian,
    n_qubits: int,
    *,
    restarts: int = 4,
    layers: int = 2,
    supernet_steps: int = 150,
    ranking_num: int = 30,
    finetune_steps: int = 200,
    learning_rate: float = 0.12,
    finetune_learning_rate: float = 0.1,
    two_qubit_pairs: Sequence[Tuple[int, int]] | None = None,
    base_seed: int = 0,
    verbose: bool = True,
) -> QASSolution:
    """Search a circuit whose ground state minimises ``hamiltonian`` via supernet.

    VQE is a heuristic, so combinatorial Ising landscapes have local minima; we
    run several independent restarts (different seeds) and keep the lowest-energy
    result. Imports of the differentiable stack are deferred so that the QUBO
    utilities above stay dependency-free.
    """
    import numpy as np

    from ..supernet import SupernetConfig, train_supernet
    from ...measure import Measure

    if two_qubit_pairs is None:
        two_qubit_pairs = tuple((i, i + 1) for i in range(n_qubits - 1))

    best: QASSolution | None = None
    attempts: List[Tuple[int, float]] = []
    for offset in range(restarts):
        seed = base_seed + offset
        config = SupernetConfig(
            n_qubits=n_qubits,
            layers=layers,
            single_qubit_gates=("ry",),
            two_qubit_pairs=tuple(two_qubit_pairs),
            search_single_qubit_gates=False,
            search_two_qubit_gates=True,
            supernet_steps=supernet_steps,
            ranking_num=ranking_num,
            finetune_steps=finetune_steps,
            learning_rate=learning_rate,
            finetune_learning_rate=finetune_learning_rate,
            seed=seed,
            task="vqe",
        )
        result = train_supernet(None, config=config, hamiltonian=hamiltonian)
        energy = float(result.final_metrics["fine_tuned_energy"])
        attempts.append((seed, energy))
        if verbose:
            print(f"    restart seed={seed}: energy={energy:+.6f}")

        if best is None or energy < best.energy:
            backend = result.best_circuit.backend
            probs = np.asarray(Measure(backend).run(result.best_circuit, shots=0).probabilities)
            best = QASSolution(
                energy=energy,
                probabilities=probs,
                best_index=int(np.argmax(probs)),
                result=result,
                seed=seed,
                attempts=attempts,
            )
        else:
            best.attempts = attempts

    assert best is not None
    return best
