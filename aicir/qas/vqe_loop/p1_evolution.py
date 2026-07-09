"""P1 evolution operators for mutation-driven VQE-QAS.

This module mutates architecture-gene rows only. It does not run oracle
screening, E2/E5 proxies, or fair VQE labels.

Responsibilities:

- parent selection from completed fair-label rows;
- supernet-native mutation: gate, connectivity, layer, and depth changes;
- chemistry-excitation mutation: insert, delete, swap, and change excitations;
- operator-sequence mutation: insert, delete, swap, big mutation, and growth;
- optional operator-growth/A1-A2 evaluator for ADAPT-style pool selection;
- conversion of ``MutationResult`` objects back into benchmark-compatible rows;
- ``generate_mutation_children`` as the main P1 child-generation entry point.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
import re
from typing import Any, Callable, Mapping, Sequence

from aicir.qas.library.ansatz import ChemistryExcitationAnsatzGene, OperatorSequenceAnsatzGene, SupernetAnsatzGene
from aicir.qas.vqe_loop.benchmark_table import decoded_ansatz_gene_payload
from aicir.qas.vqe_loop.benchmark_table import row_hamiltonian_terms
from aicir.qas.vqe_loop.benchmark_table import as_float as _as_float


DEFAULT_SINGLE_QUBIT_GATES = ("i", "h", "rx", "ry", "rz")
DEFAULT_TWO_QUBIT_GATES = ("none", "cx", "rzz")
SUPERNET_MUTATION_TYPES = ("gate_mutation", "connectivity_mutation", "layer_mutation", "depth_mutation")
CHEMISTRY_MUTATION_TYPES = (
    "chemistry_insert",
    "chemistry_delete",
    "chemistry_swap",
    "chemistry_change",
    "chemistry_adapt_growth",
)
OPERATOR_MUTATION_TYPES = (
    "operator_insert",
    "operator_delete",
    "operator_swap",
    "operator_big_mutation",
    "operator_adapt_growth",
)
MUTATION_TYPES = SUPERNET_MUTATION_TYPES + OPERATOR_MUTATION_TYPES + CHEMISTRY_MUTATION_TYPES
CROSSOVER_TYPES = ("layer_crossover",)
MutableGene = SupernetAnsatzGene | OperatorSequenceAnsatzGene | ChemistryExcitationAnsatzGene


class MutationUnavailable(ValueError):
    """Raised when a requested variation operator cannot change the gene."""




OperatorGrowthEvaluator = Callable[[OperatorSequenceAnsatzGene, str], float | Mapping[str, Any]]
ChemistryGrowthEvaluator = Callable[[ChemistryExcitationAnsatzGene, Mapping[str, Any]], float | Mapping[str, Any]]


def _default_operator_pool(n_qubits: int) -> tuple[str, ...]:
    n = int(n_qubits)
    pool: list[str] = []
    for qubit in range(n):
        for axis in ("X", "Y", "Z"):
            pauli = ["I"] * n
            pauli[qubit] = axis
            pool.append("".join(pauli))
    for left in range(n):
        for right in range(left + 1, n):
            for axis in ("X", "Y", "Z"):
                pauli = ["I"] * n
                pauli[left] = axis
                pauli[right] = axis
                pool.append("".join(pauli))
    return tuple(pool)


def _normalize_operator_pool(operator_pool: Sequence[str] | None, n_qubits: int) -> tuple[str, ...]:
    raw_pool = tuple(operator_pool or _default_operator_pool(n_qubits))
    normalized: list[str] = []
    for operator in raw_pool:
        pauli = str(operator).strip().upper()
        if not pauli:
            continue
        normalized.append(pauli)
    if not normalized:
        raise MutationUnavailable("operator_insert requires a non-empty operator_pool")
    return tuple(normalized)


def _finite_difference_operator_growth_score(
    gene: OperatorSequenceAnsatzGene,
    candidate_operator: str,
    terms: Sequence[tuple[float, str]],
    *,
    prefix_parameters: Sequence[float] | None = None,
    epsilon: float = 0.05,
) -> float:
    from aicir.qas.library.ansatz import architecture_from_operator_sequence_gene
    from aicir.qas.problems.hamiltonians import VQEProblem
    from aicir.qas.vqe_loop.fair_vqe import evaluate_vqe_energy

    base_architecture = architecture_from_operator_sequence_gene(gene)
    trial = OperatorSequenceAnsatzGene(
        n_qubits=gene.n_qubits,
        operators=gene.operators + (str(candidate_operator).upper(),),
        name=gene.name,
    )
    trial_architecture = architecture_from_operator_sequence_gene(trial)
    problem = VQEProblem(
        name="operator_adapt_growth_proxy",
        n_qubits=gene.n_qubits,
        hamiltonian=tuple((float(coeff), str(pauli).upper()) for coeff, pauli in terms),
        reference_energy=float("nan"),
    )
    parsed_prefix = [float(value) for value in tuple(prefix_parameters or ())]
    prefix = parsed_prefix[: gene.layers]
    if len(prefix) < gene.layers:
        prefix.extend([0.0] * (gene.layers - len(prefix)))
    try:
        base = evaluate_vqe_energy(base_architecture, problem, parameters=prefix)
        plus = evaluate_vqe_energy(trial_architecture, problem, parameters=prefix + [float(epsilon)])
        minus = evaluate_vqe_energy(trial_architecture, problem, parameters=prefix + [-float(epsilon)])
    except Exception:
        return float("inf")
    gradient = abs((float(plus) - float(minus)) / (2.0 * float(epsilon)))
    energy_drop = max(0.0, float(base) - min(float(plus), float(minus)))
    return -(gradient + energy_drop)

def _best_parameters_from_row(row: Mapping[str, Any]) -> tuple[float, ...]:
    raw = row.get("best_trace")
    if raw is None or str(raw).strip() == "":
        return ()
    try:
        trace = json.loads(str(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        return ()
    if not isinstance(trace, Sequence):
        return ()
    best_energy: float | None = None
    best_parameters: tuple[float, ...] = ()
    for item in trace:
        if not isinstance(item, Mapping):
            continue
        params = item.get("best_parameters")
        if not isinstance(params, Sequence) or isinstance(params, (str, bytes)):
            continue
        try:
            parsed = tuple(float(value) for value in params)
        except (TypeError, ValueError):
            continue
        energy = _as_float(item.get("energy"))
        if best_energy is None or (energy is not None and energy < best_energy):
            best_energy = energy
            best_parameters = parsed
    return best_parameters


def _operator_pool_from_row_hamiltonian(row: Mapping[str, Any], n_qubits: int) -> tuple[str, ...] | None:
    pool: list[str] = []
    try:
        terms = row_hamiltonian_terms(row)
    except ValueError:
        return None
    for coeff, pauli in terms:
        candidate = str(pauli).strip().upper()
        if len(candidate) != int(n_qubits):
            continue
        if all(symbol == "I" for symbol in candidate):
            continue
        if any(symbol not in {"I", "X", "Y", "Z"} for symbol in candidate):
            continue
        if float(coeff) == 0.0:
            continue
        if candidate not in pool:
            pool.append(candidate)
    return tuple(pool) if pool else None


def _operator_growth_evaluator_from_row(row: Mapping[str, Any]) -> OperatorGrowthEvaluator | None:
    terms = row_hamiltonian_terms(row)
    if not terms:
        return None
    prefix_parameters = _best_parameters_from_row(row)

    def evaluate(gene: OperatorSequenceAnsatzGene, candidate_operator: str) -> float:
        return _finite_difference_operator_growth_score(
            gene,
            candidate_operator,
            terms,
            prefix_parameters=prefix_parameters,
        )

    return evaluate


def _growth_score_value(result: float | Mapping[str, Any]) -> float:
    if isinstance(result, Mapping):
        if "score" in result:
            return float(result["score"])
        if "gradient_proxy" in result or "energy_drop" in result:
            gradient = abs(float(result.get("gradient_proxy", 0.0)))
            energy_drop = max(0.0, float(result.get("energy_drop", 0.0)))
            return -(gradient + energy_drop)
        for key in ("energy", "value", "delta_energy"):
            if key in result:
                return float(result[key])
        raise ValueError(
            "operator_growth_evaluator mapping must contain score, energy/value/delta_energy, "
            "or gradient_proxy/energy_drop"
        )
    return float(result)

def _select_adapt_growth_operator(
    gene: OperatorSequenceAnsatzGene,
    pool: Sequence[str],
    evaluator: OperatorGrowthEvaluator | None,
) -> str:
    candidates = [str(operator).strip().upper() for operator in pool]
    if not candidates:
        raise MutationUnavailable("operator_adapt_growth requires a non-empty operator_pool")
    if evaluator is None:
        existing = set(gene.operators)
        for candidate in candidates:
            if candidate not in existing:
                return candidate
        return candidates[0]
    existing = set(gene.operators)
    novel_candidates = [candidate for candidate in candidates if candidate not in existing]
    if novel_candidates:
        candidates = novel_candidates
    scored: list[tuple[float, int, str]] = []
    for index, candidate in enumerate(candidates):
        scored.append((_growth_score_value(evaluator(gene, candidate)), index, candidate))
    scored.sort(key=lambda item: (item[0], item[1]))
    return scored[0][2]


@dataclass(frozen=True)
class MutationResult:
    parent: MutableGene
    child: MutableGene
    mutation_type: str


def _gene_from_row(row: Mapping[str, Any]) -> MutableGene:
    raw = row.get("ansatz_gene")
    if isinstance(raw, (SupernetAnsatzGene, OperatorSequenceAnsatzGene, ChemistryExcitationAnsatzGene)):
        return raw
    payload = decoded_ansatz_gene_payload(row)
    if payload is None:
        raise ValueError("row requires ansatz_gene for P1 variation")
    kind = str(payload.get("kind", "")).lower()
    if kind == "operator_sequence":
        return OperatorSequenceAnsatzGene.from_jsonable(payload)
    if kind == "chemistry_excitation":
        return ChemistryExcitationAnsatzGene.from_jsonable(payload)
    return SupernetAnsatzGene.from_jsonable(payload)

def _rng(seed: int | None = None, rng: random.Random | None = None) -> random.Random:
    return rng if rng is not None else random.Random(seed)


def _choose_alternative(current: str, choices: Sequence[str], rng: random.Random) -> str:
    alternatives = [str(choice).lower() for choice in choices if str(choice).lower() != str(current).lower()]
    if not alternatives:
        raise ValueError(f"no alternative gate available for {current!r}")
    return rng.choice(alternatives)


def _random_single_layer(
    n_qubits: int,
    rng: random.Random,
    single_qubit_gates: Sequence[str],
) -> tuple[str, ...]:
    return tuple(rng.choice(tuple(single_qubit_gates)) for _ in range(int(n_qubits)))


def _random_two_layer(
    width: int,
    rng: random.Random,
    two_qubit_gates: Sequence[str],
) -> tuple[str, ...]:
    return tuple(rng.choice(tuple(two_qubit_gates)) for _ in range(int(width)))


def _canonical_hash(gene: MutableGene) -> str:
    return json.dumps(gene.to_jsonable(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _safe_id(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return text.strip("._") or "parent"


def _child_architecture_id(parent_id: str, mutation_type: str, gene: MutableGene) -> str:
    digest = hashlib.sha256(_canonical_hash(gene).encode("utf-8")).hexdigest()[:12]
    return f"p1_child_{_safe_id(parent_id)}_{mutation_type}_{digest}"


def _architecture_metadata(gene: MutableGene) -> dict[str, str]:
    from aicir.qas.library.ansatz import (
        architecture_from_chemistry_excitation_gene,
        architecture_from_operator_sequence_gene,
        architecture_from_supernet_gene,
    )

    if isinstance(gene, OperatorSequenceAnsatzGene):
        architecture = architecture_from_operator_sequence_gene(gene)
    elif isinstance(gene, ChemistryExcitationAnsatzGene):
        architecture = architecture_from_chemistry_excitation_gene(gene)
    else:
        architecture = architecture_from_supernet_gene(gene)
    family = str(architecture.metadata.get("family") or "")
    topology = str(architecture.metadata.get("topology") or "")
    return {
        "family": family,
        "entangler_type": topology,
        "topology": topology,
        "n_params": str(int(architecture.parameter_count)),
        "two_q_count": str(int(architecture.two_qubit_gate_count)),
    }

def _gene_distance(left_row: Mapping[str, Any], right_rows: Sequence[Mapping[str, Any]]) -> float:
    from aicir.qas.vqe_loop.oracle import gene_aware_distance

    left_gene = _gene_from_row(left_row)
    distances: list[float] = []
    for row in right_rows:
        try:
            distances.append(gene_aware_distance(left_gene, _gene_from_row(row)))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    return min(distances) if distances else 0.0


def select_parent_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    count: int,
    diversity_count: int = 0,
) -> list[dict[str, Any]]:
    """Select parent rows from completed fair labels.

    Default behavior is energy top-P.  ``diversity_count`` reserves the tail of
    the parent set for farthest labeled architectures.
    """

    scored: list[tuple[float, str, dict[str, Any]]] = []
    for index, row in enumerate(rows):
        fair_energy = _as_float(row.get("fair_best_energy"))
        if fair_energy is None:
            continue
        identifier = str(row.get("architecture_id") or f"row:{index}")
        scored.append((fair_energy, identifier, dict(row)))
    scored.sort(key=lambda item: (item[0], item[1]))
    target = max(0, int(count))
    if target <= 0:
        return []
    diversity_slots = max(0, min(int(diversity_count), max(0, target - 1)))
    energy_slots = target - diversity_slots
    selected = [row for _energy, _identifier, row in scored[:energy_slots]]
    remaining = [row for _energy, _identifier, row in scored[energy_slots:]]
    while len(selected) < target and remaining:
        chosen = max(
            remaining,
            key=lambda row: (
                _gene_distance(row, selected),
                -float(_as_float(row.get("fair_best_energy")) or 0.0),
                str(row.get("architecture_id", "")),
            ),
        )
        selected.append(chosen)
        remaining = [row for row in remaining if row.get("architecture_id") != chosen.get("architecture_id")]
    return [dict(row) for row in selected]


def _ensure_supernet_gene(gene: MutableGene, mutation_type: str) -> SupernetAnsatzGene:
    if not isinstance(gene, SupernetAnsatzGene):
        raise MutationUnavailable(f"{mutation_type} requires a supernet-native gene")
    return gene


def _ensure_chemistry_gene(gene: MutableGene, mutation_type: str) -> ChemistryExcitationAnsatzGene:
    if not isinstance(gene, ChemistryExcitationAnsatzGene):
        raise MutationUnavailable(f"{mutation_type} requires a chemistry-excitation gene")
    return gene


def _ensure_operator_gene(gene: MutableGene, mutation_type: str) -> OperatorSequenceAnsatzGene:
    if not isinstance(gene, OperatorSequenceAnsatzGene):
        raise MutationUnavailable(f"{mutation_type} requires an operator-sequence gene")
    return gene


def _mutate_single_gate(
    gene: SupernetAnsatzGene,
    rng: random.Random,
    single_qubit_gates: Sequence[str],
) -> SupernetAnsatzGene:
    layer_index = rng.randrange(gene.layers)
    qubit_index = rng.randrange(gene.n_qubits)
    single_layers = [list(layer) for layer in gene.single_qubit_layers]
    single_layers[layer_index][qubit_index] = _choose_alternative(
        single_layers[layer_index][qubit_index],
        single_qubit_gates,
        rng,
    )
    return SupernetAnsatzGene(
        n_qubits=gene.n_qubits,
        single_qubit_layers=tuple(tuple(layer) for layer in single_layers),
        two_qubit_layers=gene.two_qubit_layers,
        two_qubit_pairs=gene.two_qubit_pairs,
    )


def _mutate_two_qubit_gate(
    gene: SupernetAnsatzGene,
    rng: random.Random,
    two_qubit_gates: Sequence[str],
) -> SupernetAnsatzGene:
    if not gene.two_qubit_pairs:
        raise MutationUnavailable("connectivity_mutation requires at least one two-qubit pair")
    layer_index = rng.randrange(gene.layers)
    pair_index = rng.randrange(len(gene.two_qubit_pairs))
    two_layers = [list(layer) for layer in gene.two_qubit_layers]
    two_layers[layer_index][pair_index] = _choose_alternative(
        two_layers[layer_index][pair_index],
        two_qubit_gates,
        rng,
    )
    return SupernetAnsatzGene(
        n_qubits=gene.n_qubits,
        single_qubit_layers=gene.single_qubit_layers,
        two_qubit_layers=tuple(tuple(layer) for layer in two_layers),
        two_qubit_pairs=gene.two_qubit_pairs,
    )


def _mutate_layer(
    gene: SupernetAnsatzGene,
    rng: random.Random,
    single_qubit_gates: Sequence[str],
    two_qubit_gates: Sequence[str],
) -> SupernetAnsatzGene:
    layer_index = rng.randrange(gene.layers)
    single_layers = list(gene.single_qubit_layers)
    two_layers = list(gene.two_qubit_layers)
    for _attempt in range(16):
        new_single = _random_single_layer(gene.n_qubits, rng, single_qubit_gates)
        new_two = _random_two_layer(len(gene.two_qubit_pairs), rng, two_qubit_gates)
        if new_single != single_layers[layer_index] or new_two != two_layers[layer_index]:
            break
    if new_single == single_layers[layer_index] and new_two == two_layers[layer_index]:
        copied = list(new_single)
        copied[0] = _choose_alternative(copied[0], single_qubit_gates, rng)
        new_single = tuple(copied)
    single_layers[layer_index] = new_single
    two_layers[layer_index] = new_two
    return SupernetAnsatzGene(
        n_qubits=gene.n_qubits,
        single_qubit_layers=tuple(single_layers),
        two_qubit_layers=tuple(two_layers),
        two_qubit_pairs=gene.two_qubit_pairs,
    )


def _mutate_depth(
    gene: SupernetAnsatzGene,
    rng: random.Random,
    single_qubit_gates: Sequence[str],
    two_qubit_gates: Sequence[str],
    *,
    min_layers: int,
    max_layers: int,
) -> SupernetAnsatzGene:
    min_depth = max(1, int(min_layers))
    max_depth = max(min_depth, int(max_layers))
    can_delete = gene.layers > min_depth
    can_add = gene.layers < max_depth
    if not can_add and not can_delete:
        raise MutationUnavailable("depth_mutation cannot change depth under current min/max layers")
    operation = "add" if not can_delete else "delete" if not can_add else rng.choice(("add", "delete"))
    single_layers = list(gene.single_qubit_layers)
    two_layers = list(gene.two_qubit_layers)
    if operation == "delete":
        layer_index = rng.randrange(gene.layers)
        del single_layers[layer_index]
        del two_layers[layer_index]
    else:
        insert_index = rng.randrange(gene.layers + 1)
        single_layers.insert(insert_index, _random_single_layer(gene.n_qubits, rng, single_qubit_gates))
        two_layers.insert(insert_index, _random_two_layer(len(gene.two_qubit_pairs), rng, two_qubit_gates))
    return SupernetAnsatzGene(
        n_qubits=gene.n_qubits,
        single_qubit_layers=tuple(single_layers),
        two_qubit_layers=tuple(two_layers),
        two_qubit_pairs=gene.two_qubit_pairs,
    )



def _finite_difference_chemistry_growth_score(
    gene: ChemistryExcitationAnsatzGene,
    candidate_excitation: Mapping[str, Any],
    terms: Sequence[tuple[float, str]],
    *,
    prefix_parameters: Sequence[float] | None = None,
    epsilon: float = 0.05,
) -> float:
    from aicir.qas.library.ansatz import architecture_from_chemistry_excitation_gene
    from aicir.qas.problems.hamiltonians import VQEProblem
    from aicir.qas.vqe_loop.fair_vqe import evaluate_vqe_energy

    base_architecture = architecture_from_chemistry_excitation_gene(gene)
    trial = ChemistryExcitationAnsatzGene(
        n_qubits=gene.n_qubits,
        hf_occupied_qubits=gene.hf_occupied_qubits,
        excitations=gene.excitations + (dict(candidate_excitation),),
        active_electrons=gene.active_electrons,
        active_spatial_orbitals=gene.active_spatial_orbitals,
        name=gene.name,
    )
    trial_architecture = architecture_from_chemistry_excitation_gene(trial)
    problem = VQEProblem(
        name="chemistry_adapt_growth_proxy",
        n_qubits=gene.n_qubits,
        hamiltonian=tuple((float(coeff), str(pauli).upper()) for coeff, pauli in terms),
        reference_energy=float("nan"),
    )
    parsed_prefix = [float(value) for value in tuple(prefix_parameters or ())]
    prefix = parsed_prefix[: gene.layers]
    if len(prefix) < gene.layers:
        prefix.extend([0.0] * (gene.layers - len(prefix)))
    try:
        base = evaluate_vqe_energy(base_architecture, problem, parameters=prefix)
        plus = evaluate_vqe_energy(trial_architecture, problem, parameters=prefix + [float(epsilon)])
        minus = evaluate_vqe_energy(trial_architecture, problem, parameters=prefix + [-float(epsilon)])
    except Exception:
        return float("inf")
    gradient = abs((float(plus) - float(minus)) / (2.0 * float(epsilon)))
    energy_drop = max(0.0, float(base) - min(float(plus), float(minus)))
    return -(gradient + energy_drop)


def _chemistry_growth_evaluator_from_row(row: Mapping[str, Any]) -> ChemistryGrowthEvaluator | None:
    terms = row_hamiltonian_terms(row)
    if not terms:
        return None
    prefix_parameters = _best_parameters_from_row(row)

    def evaluate(gene: ChemistryExcitationAnsatzGene, candidate_excitation: Mapping[str, Any]) -> float:
        return _finite_difference_chemistry_growth_score(
            gene,
            candidate_excitation,
            terms,
            prefix_parameters=prefix_parameters,
        )

    return evaluate


def _chemistry_excitation_pool(gene: ChemistryExcitationAnsatzGene) -> tuple[dict[str, Any], ...]:
    if gene.active_electrons is not None and gene.active_spatial_orbitals is not None:
        from aicir.qas.vqe_loop.p0_chemistry_excitation import closed_shell_excitation_pools

        _hf, singles, doubles = closed_shell_excitation_pools(
            int(gene.active_electrons),
            int(gene.active_spatial_orbitals),
        )
        pool = [{"type": "single_excitation", "qubits": list(pair)} for pair in singles]
        pool.extend({"type": "double_excitation", "qubits": list(group)} for group in doubles)
        return tuple(pool)
    return tuple(dict(excitation) for excitation in gene.excitations)



def _canonical_excitation(excitation: Mapping[str, Any]) -> tuple[str, tuple[int, ...]]:
    return str(excitation.get("type", "")).strip().lower(), tuple(int(qubit) for qubit in excitation.get("qubits", ()))


def _limit_chemistry_candidates_by_type(
    candidates: Sequence[Mapping[str, Any]],
    pool_limit: int | None,
) -> list[dict[str, Any]]:
    items = [dict(candidate) for candidate in candidates]
    if pool_limit is None or int(pool_limit) <= 0 or len(items) <= int(pool_limit):
        return items
    limit = int(pool_limit)
    singles = [item for item in items if str(item.get("type", "")).lower() == "single_excitation"]
    doubles = [item for item in items if str(item.get("type", "")).lower() == "double_excitation"]
    if not singles or not doubles or limit < 2:
        return items[:limit]
    single_quota = max(1, limit // 2)
    double_quota = max(1, limit - single_quota)
    selected = singles[:single_quota] + doubles[:double_quota]
    selected_keys = {_canonical_excitation(item) for item in selected}
    for item in items:
        if len(selected) >= limit:
            break
        key = _canonical_excitation(item)
        if key not in selected_keys:
            selected.append(dict(item))
            selected_keys.add(key)
    return selected[:limit]

def _select_adapt_growth_excitations(
    gene: ChemistryExcitationAnsatzGene,
    pool: Sequence[Mapping[str, Any]],
    evaluator: ChemistryGrowthEvaluator | None,
    *,
    count: int = 1,
    pool_limit: int | None = None,
) -> tuple[dict[str, Any], ...]:
    candidates = [dict(item) for item in pool]
    if not candidates:
        raise MutationUnavailable("chemistry_adapt_growth requires a non-empty chemistry excitation pool")
    existing = {_canonical_excitation(excitation) for excitation in gene.excitations}
    novel_candidates = [candidate for candidate in candidates if _canonical_excitation(candidate) not in existing]
    if novel_candidates:
        candidates = novel_candidates
    candidates = _limit_chemistry_candidates_by_type(candidates, pool_limit)
    take = max(1, min(int(count), len(candidates)))
    if evaluator is None:
        return tuple(dict(candidate) for candidate in candidates[:take])
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for index, candidate in enumerate(candidates):
        scored.append((_growth_score_value(evaluator(gene, candidate)), index, dict(candidate)))
    scored.sort(key=lambda item: (item[0], item[1]))
    return tuple(dict(item[2]) for item in scored[:take])


def _select_adapt_growth_excitation(
    gene: ChemistryExcitationAnsatzGene,
    pool: Sequence[Mapping[str, Any]],
    evaluator: ChemistryGrowthEvaluator | None,
) -> dict[str, Any]:
    return dict(_select_adapt_growth_excitations(gene, pool, evaluator, count=1)[0])


def _mutate_chemistry_excitation(
    gene: ChemistryExcitationAnsatzGene,
    mutation_type: str,
    rng: random.Random,
    *,
    chemistry_growth_evaluator: ChemistryGrowthEvaluator | None,
    chemistry_adapt_append_k: int = 1,
    chemistry_adapt_pool_limit: int | None = None,
    min_layers: int = 1,
    max_layers: int | None = None,
) -> ChemistryExcitationAnsatzGene:
    excitations = [dict(excitation) for excitation in gene.excitations]
    pool = list(_chemistry_excitation_pool(gene))
    if not pool:
        raise MutationUnavailable(f"{mutation_type} requires a non-empty chemistry excitation pool")
    min_depth = max(0, int(min_layers))
    max_depth = None if max_layers is None else max(min_depth, int(max_layers))
    if mutation_type == "chemistry_adapt_growth":
        if max_depth is not None and len(excitations) >= max_depth:
            raise MutationUnavailable("chemistry_adapt_growth cannot exceed max_layers")
        append_count = max(1, int(chemistry_adapt_append_k))
        if max_depth is not None:
            append_count = min(append_count, max_depth - len(excitations))
        if append_count <= 0:
            raise MutationUnavailable("chemistry_adapt_growth cannot exceed max_layers")
        child = list(excitations)
        child.extend(
            _select_adapt_growth_excitations(
                gene,
                pool,
                chemistry_growth_evaluator,
                count=append_count,
                pool_limit=chemistry_adapt_pool_limit,
            )
        )
    elif mutation_type == "chemistry_insert":
        if max_depth is not None and len(excitations) >= max_depth:
            raise MutationUnavailable("chemistry_insert cannot exceed max_layers")
        child = list(excitations)
        child.insert(rng.randrange(len(child) + 1), dict(rng.choice(pool)))
    elif mutation_type == "chemistry_delete":
        if len(excitations) <= min_depth:
            raise MutationUnavailable("chemistry_delete cannot go below min_layers")
        child = list(excitations)
        del child[rng.randrange(len(child))]
    elif mutation_type == "chemistry_swap":
        if len(excitations) < 2:
            raise MutationUnavailable("chemistry_swap requires at least two excitations")
        child = list(excitations)
        left, right = rng.sample(range(len(child)), 2)
        child[left], child[right] = child[right], child[left]
    elif mutation_type == "chemistry_change":
        if not excitations:
            raise MutationUnavailable("chemistry_change requires at least one excitation")
        child = list(excitations)
        index = rng.randrange(len(child))
        alternatives = [item for item in pool if item != child[index]]
        if not alternatives:
            raise MutationUnavailable("chemistry_change requires an alternative excitation")
        child[index] = dict(rng.choice(alternatives))
    else:
        raise ValueError(f"unsupported chemistry mutation_type: {mutation_type}")
    return ChemistryExcitationAnsatzGene(
        n_qubits=gene.n_qubits,
        hf_occupied_qubits=gene.hf_occupied_qubits,
        excitations=tuple(child),
        active_electrons=gene.active_electrons,
        active_spatial_orbitals=gene.active_spatial_orbitals,
        name=gene.name,
    )


def mutate_gene(
    gene: MutableGene,
    *,
    mutation_type: str,
    seed: int | None = None,
    rng: random.Random | None = None,
    single_qubit_gates: Sequence[str] = DEFAULT_SINGLE_QUBIT_GATES,
    two_qubit_gates: Sequence[str] = DEFAULT_TWO_QUBIT_GATES,
    operator_pool: Sequence[str] | None = None,
    operator_growth_evaluator: OperatorGrowthEvaluator | None = None,
    chemistry_growth_evaluator: ChemistryGrowthEvaluator | None = None,
    chemistry_adapt_append_k: int = 1,
    chemistry_adapt_pool_limit: int | None = None,
    min_layers: int = 1,
    max_layers: int | None = None,
) -> MutationResult:
    """Apply one named mutation to an architecture ansatz gene."""

    local_rng = _rng(seed, rng)
    normalized = str(mutation_type).strip().lower()
    if normalized not in MUTATION_TYPES:
        raise ValueError(f"unsupported mutation_type: {mutation_type}")

    if normalized in CHEMISTRY_MUTATION_TYPES:
        chemistry_gene = _ensure_chemistry_gene(gene, normalized)
        child = _mutate_chemistry_excitation(
            chemistry_gene,
            normalized,
            local_rng,
            chemistry_growth_evaluator=chemistry_growth_evaluator,
            chemistry_adapt_append_k=int(chemistry_adapt_append_k),
            chemistry_adapt_pool_limit=chemistry_adapt_pool_limit,
            min_layers=int(min_layers),
            max_layers=max_layers,
        )
        if child == gene:
            raise MutationUnavailable(f"{normalized} produced no structural change")
        return MutationResult(parent=gene, child=child, mutation_type=normalized)

    if normalized in OPERATOR_MUTATION_TYPES:
        operator_gene = _ensure_operator_gene(gene, normalized)
        child = _mutate_operator_sequence(
            operator_gene,
            normalized,
            local_rng,
            operator_pool=operator_pool,
            operator_growth_evaluator=operator_growth_evaluator,
            min_layers=int(min_layers),
            max_layers=max_layers,
        )
        if child == gene:
            raise MutationUnavailable(f"{normalized} produced no structural change")
        return MutationResult(parent=gene, child=child, mutation_type=normalized)

    supernet_gene = _ensure_supernet_gene(gene, normalized)
    max_depth = int(max_layers) if max_layers is not None else max(supernet_gene.layers + 1, int(min_layers))
    if normalized == "gate_mutation":
        child = _mutate_single_gate(supernet_gene, local_rng, single_qubit_gates)
    elif normalized == "connectivity_mutation":
        child = _mutate_two_qubit_gate(supernet_gene, local_rng, two_qubit_gates)
    elif normalized == "layer_mutation":
        child = _mutate_layer(supernet_gene, local_rng, single_qubit_gates, two_qubit_gates)
    else:
        child = _mutate_depth(
            supernet_gene,
            local_rng,
            single_qubit_gates,
            two_qubit_gates,
            min_layers=int(min_layers),
            max_layers=max_depth,
        )
    if child == gene:
        raise MutationUnavailable(f"{normalized} produced no structural change")
    return MutationResult(parent=gene, child=child, mutation_type=normalized)

def _mutate_operator_sequence(
    gene: OperatorSequenceAnsatzGene,
    mutation_type: str,
    rng: random.Random,
    *,
    operator_pool: Sequence[str] | None,
    operator_growth_evaluator: OperatorGrowthEvaluator | None,
    min_layers: int,
    max_layers: int | None,
) -> OperatorSequenceAnsatzGene:
    min_depth = max(0, int(min_layers))
    max_depth = None if max_layers is None else max(min_depth, int(max_layers))
    operators = list(gene.operators)
    if mutation_type == "operator_insert":
        if max_depth is not None and len(operators) >= max_depth:
            raise MutationUnavailable("operator_insert cannot exceed max_layers")
        pool = _normalize_operator_pool(operator_pool, gene.n_qubits)
        inserted = rng.choice(pool)
        child_ops = list(operators)
        child_ops.insert(rng.randrange(len(child_ops) + 1), inserted)
    elif mutation_type == "operator_adapt_growth":
        if max_depth is not None and len(operators) >= max_depth:
            raise MutationUnavailable("operator_adapt_growth cannot exceed max_layers")
        pool = _normalize_operator_pool(operator_pool, gene.n_qubits)
        inserted = _select_adapt_growth_operator(gene, pool, operator_growth_evaluator)
        child_ops = list(operators)
        child_ops.append(inserted)
    elif mutation_type == "operator_delete":
        if len(operators) <= min_depth:
            raise MutationUnavailable("operator_delete cannot go below min_layers")
        child_ops = list(operators)
        del child_ops[rng.randrange(len(child_ops))]
    elif mutation_type == "operator_swap":
        if len(operators) < 2:
            raise MutationUnavailable("operator_swap requires at least two operators")
        left, right = rng.sample(range(len(operators)), 2)
        child_ops = list(operators)
        child_ops[left], child_ops[right] = child_ops[right], child_ops[left]
    elif mutation_type == "operator_big_mutation":
        steps = max(2, min(8, len(operators) + 2))
        child = gene
        for _step in range(steps):
            possible = ["operator_insert"]
            if child.layers > min_depth:
                possible.append("operator_delete")
            if child.layers >= 2:
                possible.append("operator_swap")
            if max_depth is not None and child.layers >= max_depth and "operator_insert" in possible:
                possible.remove("operator_insert")
            child = _mutate_operator_sequence(
                child,
                rng.choice(possible),
                rng,
                operator_pool=operator_pool,
                operator_growth_evaluator=operator_growth_evaluator,
                min_layers=min_depth,
                max_layers=max_depth,
            )
        return child
    else:
        raise ValueError(f"unsupported operator mutation_type: {mutation_type}")
    return OperatorSequenceAnsatzGene(n_qubits=gene.n_qubits, operators=tuple(child_ops), name=gene.name)


def layer_crossover(
    left: SupernetAnsatzGene,
    right: SupernetAnsatzGene,
    *,
    cut: int | None = None,
    seed: int | None = None,
    rng: random.Random | None = None,
) -> MutationResult:
    """Splice layers from two compatible supernet-native parent genes."""

    if left.n_qubits != right.n_qubits:
        raise ValueError("layer_crossover requires the same n_qubits")
    if left.two_qubit_pairs != right.two_qubit_pairs:
        raise ValueError("layer_crossover requires identical two_qubit_pairs")
    max_cut = min(left.layers, right.layers)
    if max_cut < 2:
        raise MutationUnavailable("layer_crossover requires at least two layers")
    local_rng = _rng(seed, rng)
    split = int(cut) if cut is not None else local_rng.randrange(1, max_cut)
    split = max(1, min(split, max_cut - 1))
    child = SupernetAnsatzGene(
        n_qubits=left.n_qubits,
        single_qubit_layers=left.single_qubit_layers[:split] + right.single_qubit_layers[split:],
        two_qubit_layers=left.two_qubit_layers[:split] + right.two_qubit_layers[split:],
        two_qubit_pairs=left.two_qubit_pairs,
    )
    if child == left or child == right:
        raise MutationUnavailable("layer_crossover produced no structural change")
    return MutationResult(parent=left, child=child, mutation_type="layer_crossover")


def mutation_result_to_row(
    parent_row: Mapping[str, Any],
    result: MutationResult,
    *,
    crossover_parent_row: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert a mutation result into a queue-compatible child row."""

    parent_id = str(parent_row.get("architecture_id") or "parent")
    child_payload = result.child.to_jsonable()
    metadata = _architecture_metadata(result.child)
    row = dict(parent_row)
    for field in ("fair_best_energy", "fair_mean_energy", "fair_median_energy", "label_status"):
        row.pop(field, None)
    row.update(
        {
            "architecture_id": _child_architecture_id(parent_id, result.mutation_type, result.child),
            "canonical_arch_hash": _canonical_hash(result.child),
            "source": "p1_mutation",
            "parent_architecture_id": parent_id,
            "crossover_parent_architecture_id": ""
            if crossover_parent_row is None
            else str(crossover_parent_row.get("architecture_id") or ""),
            "mutation_type": result.mutation_type,
            "n_qubits": str(result.child.n_qubits),
            "depth_group": f"L{result.child.layers}",
            "ansatz_gene": json.dumps(child_payload, ensure_ascii=False),
            **metadata,
        }
    )
    return row


def _weighted_choice(
    mutation_cycle: Sequence[str],
    mutation_weights: Mapping[str, float] | None,
    rng: random.Random,
) -> str:
    if not mutation_weights:
        raise ValueError("mutation_weights is empty")
    weights = [max(0.0, float(mutation_weights.get(item, 0.0))) for item in mutation_cycle]
    if sum(weights) <= 0:
        raise ValueError("mutation_weights must contain at least one positive weight")
    return rng.choices(list(mutation_cycle), weights=weights, k=1)[0]


def _mutation_order(
    mutation_cycle: Sequence[str],
    *,
    parent_index: int,
    child_index: int,
    children_per_parent: int,
    rng: random.Random,
    mutation_weights: Mapping[str, float] | None,
) -> list[str]:
    if mutation_weights:
        first = _weighted_choice(mutation_cycle, mutation_weights, rng)
    else:
        first = mutation_cycle[(parent_index * int(children_per_parent) + child_index) % len(mutation_cycle)]
    return [first] + [item for item in mutation_cycle if item != first]


def generate_mutation_children(
    parent_rows: Sequence[Mapping[str, Any]],
    *,
    children_per_parent: int,
    mutation_types: Sequence[str] = MUTATION_TYPES,
    seed: int = 0,
    single_qubit_gates: Sequence[str] = DEFAULT_SINGLE_QUBIT_GATES,
    two_qubit_gates: Sequence[str] = DEFAULT_TWO_QUBIT_GATES,
    operator_pool: Sequence[str] | None = None,
    operator_growth_evaluator: OperatorGrowthEvaluator | None = None,
    chemistry_growth_evaluator: ChemistryGrowthEvaluator | None = None,
    chemistry_adapt_append_k: int = 1,
    chemistry_adapt_pool_limit: int | None = None,
    min_layers: int = 1,
    max_layers: int | None = None,
    mutation_weights: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Generate child rows from parent rows with explicit mutation types."""

    if int(children_per_parent) <= 0:
        return []
    mutation_cycle = tuple(mutation_types) or MUTATION_TYPES
    local_rng = random.Random(int(seed))
    children: list[dict[str, Any]] = []
    for parent_index, parent_row in enumerate(parent_rows):
        parent_gene = _gene_from_row(parent_row)
        row_growth_evaluator = operator_growth_evaluator or _operator_growth_evaluator_from_row(parent_row)
        row_chemistry_growth_evaluator = chemistry_growth_evaluator or _chemistry_growth_evaluator_from_row(parent_row)
        row_operator_pool = operator_pool or _operator_pool_from_row_hamiltonian(parent_row, parent_gene.n_qubits)
        for child_index in range(int(children_per_parent)):
            result: MutationResult | None = None
            for mutation_type in _mutation_order(
                mutation_cycle,
                parent_index=parent_index,
                child_index=child_index,
                children_per_parent=int(children_per_parent),
                rng=local_rng,
                mutation_weights=mutation_weights,
            ):
                try:
                    result = mutate_gene(
                        parent_gene,
                        mutation_type=mutation_type,
                        rng=local_rng,
                        single_qubit_gates=single_qubit_gates,
                        two_qubit_gates=two_qubit_gates,
                        operator_pool=row_operator_pool,
                        operator_growth_evaluator=row_growth_evaluator,
                        chemistry_growth_evaluator=row_chemistry_growth_evaluator,
                        chemistry_adapt_append_k=int(chemistry_adapt_append_k),
                        chemistry_adapt_pool_limit=chemistry_adapt_pool_limit,
                        min_layers=min_layers,
                        max_layers=max_layers,
                    )
                    break
                except MutationUnavailable:
                    continue
            if result is None:
                raise MutationUnavailable("no configured mutation type could change the parent gene")
            children.append(mutation_result_to_row(parent_row, result))
    return children


def generate_crossover_children(
    parent_rows: Sequence[Mapping[str, Any]],
    *,
    count: int,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Generate layer-crossover child rows from pairs of parent rows."""

    if int(count) <= 0:
        return []
    if len(parent_rows) < 2:
        raise MutationUnavailable("layer_crossover requires at least two parent rows")
    local_rng = random.Random(int(seed))
    rows = [dict(row) for row in parent_rows]
    children: list[dict[str, Any]] = []
    attempts = 0
    while len(children) < int(count) and attempts < int(count) * max(4, len(rows)):
        attempts += 1
        left_index, right_index = local_rng.sample(range(len(rows)), 2)
        left_row = rows[left_index]
        right_row = rows[right_index]
        try:
            result = layer_crossover(_gene_from_row(left_row), _gene_from_row(right_row), rng=local_rng)
        except MutationUnavailable:
            continue
        children.append(mutation_result_to_row(left_row, result, crossover_parent_row=right_row))
    if len(children) < int(count):
        raise MutationUnavailable("could not generate requested crossover children")
    return children


__all__ = [
    "DEFAULT_SINGLE_QUBIT_GATES",
    "DEFAULT_TWO_QUBIT_GATES",
    "CROSSOVER_TYPES",
    "CHEMISTRY_MUTATION_TYPES",
    "MUTATION_TYPES",
    "MutationUnavailable",
    "MutationResult",
    "OperatorGrowthEvaluator",
    "ChemistryGrowthEvaluator",
    "generate_crossover_children",
    "generate_mutation_children",
    "layer_crossover",
    "mutate_gene",
    "mutation_result_to_row",
    "select_parent_rows",
]









