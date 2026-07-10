"""基于多目标遗传算法的变分量子本征求解器 (MoG-VQE)。

- 论文: MoG-VQE: Multiobjective genetic variational quantum eigensolver
- 参考: arXiv:2007.04424

本实现以 aicir 原生的形式遵循了 MoG-VQE 的核心思想：
- 使用 block-based hardware-efficient ansatz 表示候选线路拓扑。
- 将每个候选线路表示为二量子比特 block 的有序列表。
- 通过插入 block、删除 block 和大尺度变异来修改线路拓扑。
- 使用 NSGA-II 按两个目标选择下一代种群：能量和 CNOT 数量，二者均最小化。
- 对固定拓扑优化旋转角参数，并输出修改后的 aicir Circuit。

输入 (Inputs):
- initial_ansatz: MOGVQEIndividual | Circuit | Sequence[MOGVQEBlock]。初始 block-based
  HEA 拓扑；传入 Circuit 时会从其中的 CNOT 连接提取 block 拓扑。
- hamiltonian: Optional[Hamiltonian | np.ndarray]。目标哈密顿量；提供后使用精确态向量
  能量评估。
- energy_evaluator: Optional[Callable[[Circuit], float]]。自定义能量评估函数；可用于接入
  外部 CMA-ES/VQE 优化流程。
- config: Optional[MOGVQEConfig]。NSGA-II、拓扑变异和参数优化的超参数配置。
- backend: Optional[Any]。用于构造和仿真线路的 aicir 后端。

输出 (Outputs):
- mogvqe 返回一个 MOGVQEResult 数据类，包含：
  * best_individual: 搜索到的最优 block 拓扑。
  * best_circuit: 搜索到并绑定最优参数后的 aicir Circuit。
  * best_energy: 最优线路达到的能量值。
  * best_parameters: 最优线路对应的连续旋转参数。
  * pareto_front: 最终种群中的非支配 Pareto 前沿。
  * population: 最终 NSGA-II 种群。
  * history: 每一代的最优能量、CNOT 数量和 Pareto 前沿摘要。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Any, Callable, Literal, Sequence

import numpy as np

from ...backends.numpy_backend import NumpyBackend
from ...core.circuit import Circuit, cx, ry, rz
from ...core.operators import Hamiltonian
from ...ir import instruction_controls, instruction_name, instruction_qubits


Edge = tuple[int, int]
EnergyEvaluator = Callable[[Circuit], float]
ParameterOptimizerName = Literal["separable_es", "random", "none"]

_GENERALIZED_CNOT = "generalized_cnot"
_GENERALIZED_TWO_QUBIT = "generalized_two_qubit"
_BLOCK_ALIASES = {
    "generalized_cnot": _GENERALIZED_CNOT,
    "gcnot": _GENERALIZED_CNOT,
    "cnot": _GENERALIZED_CNOT,
    "generalized_two_qubit": _GENERALIZED_TWO_QUBIT,
    "gtwo": _GENERALIZED_TWO_QUBIT,
    "two_cnot": _GENERALIZED_TWO_QUBIT,
}


@dataclass(frozen=True)
class MOGVQEBlock:
    """One MoG-VQE topology block acting on an ordered qubit pair."""

    control: int
    target: int
    block_type: str = _GENERALIZED_CNOT

    def __post_init__(self) -> None:
        control = int(self.control)
        target = int(self.target)
        if control == target:
            raise ValueError("A MoG-VQE block cannot connect a qubit to itself")
        object.__setattr__(self, "control", control)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "block_type", _normalize_block_type(self.block_type))

    @property
    def parameter_count(self) -> int:
        if self.block_type == _GENERALIZED_CNOT:
            return 4
        if self.block_type == _GENERALIZED_TWO_QUBIT:
            return 5
        raise ValueError(f"Unsupported block type: {self.block_type}")

    @property
    def cnot_count(self) -> int:
        return 1 if self.block_type == _GENERALIZED_CNOT else 2


@dataclass(frozen=True)
class MOGVQEIndividual:
    """A candidate MoG-VQE ansatz topology."""

    n_qubits: int
    blocks: tuple[MOGVQEBlock, ...]
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        n_qubits = int(self.n_qubits)
        if n_qubits <= 0:
            raise ValueError("n_qubits must be positive")
        normalized_blocks = tuple(
            block if isinstance(block, MOGVQEBlock) else MOGVQEBlock(*block)  # type: ignore[arg-type]
            for block in self.blocks
        )
        for block in normalized_blocks:
            if not (0 <= block.control < n_qubits and 0 <= block.target < n_qubits):
                raise ValueError("Block qubit index is outside [0, n_qubits)")
        object.__setattr__(self, "n_qubits", n_qubits)
        object.__setattr__(self, "blocks", normalized_blocks)

    @property
    def parameter_count(self) -> int:
        return sum(block.parameter_count for block in self.blocks)

    @property
    def cnot_count(self) -> int:
        return sum(block.cnot_count for block in self.blocks)

    def to_circuit(
        self,
        parameters: Sequence[float] | np.ndarray | None = None,
        *,
        backend: Any = None,
    ) -> Circuit:
        """Expand this block topology into an ``aicir`` ``Circuit``.

        ``parameters`` are consumed in block order. If omitted, zero angles are
        used, which exposes the modified CNOT topology while keeping the return
        value directly simulatable.
        """

        params = _normalize_parameters(parameters, self.parameter_count)
        cursor = 0
        gates: list[Any] = []
        for block in self.blocks:
            width = block.parameter_count
            block_params = params[cursor : cursor + width]
            cursor += width
            _append_block_gates(gates, block, block_params)
        return Circuit(*gates, n_qubits=self.n_qubits, backend=backend)


@dataclass
class MOGVQEConfig:
    """Configuration for MoG-VQE NSGA-II topology search."""

    population_size: int = 16
    generations: int = 10
    mutation_insert_weight: float = 2.0
    mutation_delete_weight: float = 1.0
    mutation_big_weight: float = 0.25
    big_mutation_steps: int = 10
    min_blocks: int = 0
    max_blocks: int | None = None
    block_type: str = _GENERALIZED_CNOT
    allowed_edges: tuple[Edge, ...] | None = None
    parameter_optimizer: ParameterOptimizerName = "separable_es"
    parameter_generations: int = 8
    parameter_population_size: int = 8
    parameter_sigma: float = 0.5
    parameter_bounds: tuple[float, float] = (-math.pi, math.pi)
    seed: int | None = None


@dataclass
class MOGVQECandidate:
    """Evaluated individual with NSGA-II metadata."""

    individual: MOGVQEIndividual
    energy: float
    cnot_count: int
    parameters: np.ndarray
    circuit: Circuit
    rank: int = 0
    crowding_distance: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def objectives(self) -> tuple[float, float]:
        return (float(self.energy), float(self.cnot_count))


@dataclass
class MOGVQEResult:
    """Result of a MoG-VQE topology-search run."""

    best_individual: MOGVQEIndividual
    best_circuit: Circuit
    best_energy: float
    best_parameters: np.ndarray
    pareto_front: list[MOGVQECandidate]
    population: list[MOGVQECandidate]
    history: list[dict[str, Any]]
    config: MOGVQEConfig


def block_hardware_efficient_ansatz(
    n_qubits: int,
    layers: int = 1,
    *,
    topology: str | Sequence[Edge] = "linear",
    block_type: str = _GENERALIZED_CNOT,
) -> MOGVQEIndividual:
    """Create the block-based HEA topology used as MoG-VQE input.

    Each layer applies the selected MoG-VQE block on every edge in the chosen
    topology. Supported topology strings are ``"linear"``, ``"ring"``,
    ``"all_to_all"`` and ``"full"``.
    """

    n_qubits = _validate_n_qubits(n_qubits)
    layers = int(layers)
    if layers < 0:
        raise ValueError("layers must be non-negative")
    edges = _topology_edges(n_qubits, topology)
    blocks = [
        MOGVQEBlock(control, target, block_type)
        for _ in range(layers)
        for control, target in edges
    ]
    return MOGVQEIndividual(
        n_qubits=n_qubits,
        blocks=tuple(blocks),
        metadata={"family": "MoG-VQE block HEA", "layers": layers, "topology": topology},
    )


def extract_blocks_from_circuit(
    circuit: Circuit,
    *,
    block_type: str = _GENERALIZED_CNOT,
) -> MOGVQEIndividual:
    """Extract a MoG-VQE block topology from the CNOTs in an existing circuit."""

    blocks: list[MOGVQEBlock] = []
    for instruction in circuit.gates:
        gate_type = str(instruction_name(instruction)).lower()
        if gate_type not in {"cx", "cnot"}:
            continue
        controls = list(instruction_controls(instruction))
        if len(controls) != 1:
            continue
        targets = instruction_qubits(instruction)
        for target in targets:
            blocks.append(MOGVQEBlock(int(controls[0]), int(target), block_type))
    return MOGVQEIndividual(
        n_qubits=int(circuit.n_qubits),
        blocks=tuple(blocks),
        metadata={"source": "Circuit CNOT topology"},
    )


def mogvqe(
    initial_ansatz: MOGVQEIndividual | Circuit | Sequence[MOGVQEBlock],
    *,
    hamiltonian: Hamiltonian | np.ndarray | None = None,
    energy_evaluator: EnergyEvaluator | None = None,
    config: MOGVQEConfig | None = None,
    backend: Any = None,
) -> MOGVQEResult:
    """Run MoG-VQE NSGA-II topology search and return a modified circuit.

    Pass either ``energy_evaluator(circuit) -> float`` or a dense/``Hamiltonian``
    object. If a Hamiltonian is provided, this function performs exact
    state-vector energy evaluation on the supplied backend after its built-in
    parameter optimizer chooses angles for each topology.
    """

    cfg = _validated_config(config or MOGVQEConfig())
    initial = _coerce_individual(initial_ansatz)
    resolved_backend = backend if backend is not None else NumpyBackend()
    evaluator = _resolve_energy_evaluator(
        hamiltonian=hamiltonian,
        energy_evaluator=energy_evaluator,
        backend=resolved_backend,
    )

    py_rng = random.Random(cfg.seed)
    np_rng = np.random.default_rng(cfg.seed)
    edges = cfg.allowed_edges or _all_ordered_edges(initial.n_qubits)

    raw_population = _initial_population(initial, cfg, edges, py_rng)
    population = [_evaluate_individual(item, evaluator, cfg, resolved_backend, np_rng) for item in raw_population]
    population = nsga_ii_select(population, cfg.population_size)
    history = [_history_entry(0, population)]

    for generation in range(1, cfg.generations + 1):
        offspring_topologies = [
            mutate_individual(candidate.individual, cfg, edges=edges, rng=py_rng)
            for candidate in population
        ]
        offspring = [
            _evaluate_individual(item, evaluator, cfg, resolved_backend, np_rng)
            for item in offspring_topologies
        ]
        population = nsga_ii_select(population + offspring, cfg.population_size)
        history.append(_history_entry(generation, population))

    front = pareto_front(population)
    best = min(front, key=lambda candidate: (candidate.energy, candidate.cnot_count))
    return MOGVQEResult(
        best_individual=best.individual,
        best_circuit=best.circuit,
        best_energy=float(best.energy),
        best_parameters=best.parameters.copy(),
        pareto_front=front,
        population=population,
        history=history,
        config=cfg,
    )


run_mog_vqe = mogvqe


def mutate_individual(
    individual: MOGVQEIndividual,
    config: MOGVQEConfig | None = None,
    *,
    edges: Sequence[Edge] | None = None,
    rng: random.Random | None = None,
) -> MOGVQEIndividual:
    """Apply the paper's insert/delete/big-mutation topology operator."""

    cfg = _validated_config(config or MOGVQEConfig())
    local_rng = rng or random.Random(cfg.seed)
    allowed_edges = tuple(edges or cfg.allowed_edges or _all_ordered_edges(individual.n_qubits))
    weights = (cfg.mutation_insert_weight, cfg.mutation_delete_weight, cfg.mutation_big_weight)
    operation = local_rng.choices(("insert", "delete", "big"), weights=weights, k=1)[0]
    if operation == "insert":
        return _insert_block(individual, cfg, allowed_edges, local_rng)
    if operation == "delete":
        return _delete_block(individual, cfg, local_rng)

    mutated = individual
    for _ in range(cfg.big_mutation_steps):
        if local_rng.random() < 0.5:
            mutated = _insert_block(mutated, cfg, allowed_edges, local_rng)
        else:
            mutated = _delete_block(mutated, cfg, local_rng)
    return mutated


def pareto_front(population: Sequence[MOGVQECandidate]) -> list[MOGVQECandidate]:
    """Return the rank-0 Pareto front for minimized ``(energy, CNOT count)``."""

    fronts = non_dominated_sort(list(population))
    return fronts[0] if fronts else []


def non_dominated_sort(population: Sequence[MOGVQECandidate]) -> list[list[MOGVQECandidate]]:
    """Fast-enough NSGA-II non-dominated sorting for small QAS populations."""

    items = list(population)
    n_items = len(items)
    dominates_list: list[list[int]] = [[] for _ in range(n_items)]
    dominated_count = [0 for _ in range(n_items)]
    fronts: list[list[int]] = [[]]

    for i, left in enumerate(items):
        for j, right in enumerate(items):
            if i == j:
                continue
            if _dominates(left, right):
                dominates_list[i].append(j)
            elif _dominates(right, left):
                dominated_count[i] += 1
        if dominated_count[i] == 0:
            left.rank = 0
            fronts[0].append(i)

    rank = 0
    while rank < len(fronts) and fronts[rank]:
        next_front: list[int] = []
        for i in fronts[rank]:
            for j in dominates_list[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    items[j].rank = rank + 1
                    next_front.append(j)
        rank += 1
        if next_front:
            fronts.append(next_front)

    return [[items[index] for index in front] for front in fronts if front]


def nsga_ii_select(
    population: Sequence[MOGVQECandidate],
    population_size: int,
) -> list[MOGVQECandidate]:
    """Select the next population using rank and crowding distance."""

    selected: list[MOGVQECandidate] = []
    for front in non_dominated_sort(population):
        _assign_crowding_distance(front)
        if len(selected) + len(front) <= population_size:
            selected.extend(front)
            continue
        remaining = population_size - len(selected)
        selected.extend(
            sorted(front, key=lambda candidate: candidate.crowding_distance, reverse=True)[:remaining]
        )
        break
    return selected


def count_cnot_gates(circuit: Circuit) -> int:
    """Count CNOT/CX gates in a generated ``Circuit``."""

    return sum(1 for instruction in circuit.gates if str(instruction_name(instruction)).lower() in {"cx", "cnot"})


def _append_block_gates(gates: list[Any], block: MOGVQEBlock, theta: np.ndarray) -> None:
    control, target = block.control, block.target
    if block.block_type == _GENERALIZED_CNOT:
        theta1, theta2, theta3, theta4 = theta
        gates.extend(
            [
                rz(theta1, control),
                ry(theta2, control),
                rz(theta3, target),
                ry(theta4, target),
                cx(target, [control]),
                ry(-theta4, target),
                rz(-theta3, target),
                ry(-theta2, control),
                rz(-theta1, control),
            ]
        )
        return

    if block.block_type == _GENERALIZED_TWO_QUBIT:
        theta1, theta2, theta3, theta4, theta5 = theta
        gates.extend(
            [
                rz(theta1, control),
                ry(theta2, control),
                rz(theta3, target),
                ry(theta4, target),
                cx(target, [control]),
                ry(-theta4, target),
                rz(-0.5 * theta3 - 0.5 * theta5, target),
                cx(target, [control]),
                rz(-0.5 * theta3 + 0.5 * theta5, target),
                ry(-theta2, control),
                rz(-theta1, control),
            ]
        )
        return

    raise ValueError(f"Unsupported block type: {block.block_type}")


def _evaluate_individual(
    individual: MOGVQEIndividual,
    evaluator: EnergyEvaluator,
    config: MOGVQEConfig,
    backend: Any,
    rng: np.random.Generator,
) -> MOGVQECandidate:
    parameters, energy, metadata = _optimize_parameters(individual, evaluator, config, backend, rng)
    circuit = individual.to_circuit(parameters, backend=backend)
    return MOGVQECandidate(
        individual=individual,
        energy=float(energy),
        cnot_count=individual.cnot_count,
        parameters=parameters,
        circuit=circuit,
        metadata=metadata,
    )


def _optimize_parameters(
    individual: MOGVQEIndividual,
    evaluator: EnergyEvaluator,
    config: MOGVQEConfig,
    backend: Any,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    count = individual.parameter_count
    if count == 0 or config.parameter_optimizer == "none":
        params = np.zeros(count, dtype=float)
        return params, float(evaluator(individual.to_circuit(params, backend=backend))), {"optimizer": "none"}

    low, high = config.parameter_bounds
    if config.parameter_optimizer == "random":
        total_trials = max(1, config.parameter_generations) * max(1, config.parameter_population_size)
        return _random_parameter_search(individual, evaluator, backend, rng, count, low, high, total_trials)

    if config.parameter_optimizer != "separable_es":
        raise ValueError("parameter_optimizer must be 'separable_es', 'random', or 'none'")

    population_size = max(2, int(config.parameter_population_size))
    generations = max(1, int(config.parameter_generations))
    elite_count = max(1, population_size // 2)
    mean = rng.uniform(low, high, size=count)
    std = np.full(count, float(config.parameter_sigma), dtype=float)
    best_params: np.ndarray | None = None
    best_energy = math.inf
    evaluations = 0

    for _ in range(generations):
        samples = rng.normal(loc=mean, scale=std, size=(population_size, count))
        samples = np.clip(samples, low, high)
        scored: list[tuple[float, np.ndarray]] = []
        for sample in samples:
            energy = float(evaluator(individual.to_circuit(sample, backend=backend)))
            evaluations += 1
            scored.append((energy, sample))
            if energy < best_energy:
                best_energy = energy
                best_params = sample.copy()

        scored.sort(key=lambda item: item[0])
        elites = np.asarray([sample for _, sample in scored[:elite_count]], dtype=float)
        mean = elites.mean(axis=0)
        if elite_count > 1:
            std = np.maximum(elites.std(axis=0), 1e-3)
        std = np.minimum(std * 0.95 + 0.05 * float(config.parameter_sigma), high - low)

    assert best_params is not None
    return (
        best_params,
        best_energy,
        {"optimizer": "separable_es", "evaluations": evaluations},
    )


def _random_parameter_search(
    individual: MOGVQEIndividual,
    evaluator: EnergyEvaluator,
    backend: Any,
    rng: np.random.Generator,
    count: int,
    low: float,
    high: float,
    total_trials: int,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    best_params = np.zeros(count, dtype=float)
    best_energy = math.inf
    for trial in range(total_trials):
        params = rng.uniform(low, high, size=count)
        energy = float(evaluator(individual.to_circuit(params, backend=backend)))
        if energy < best_energy:
            best_energy = energy
            best_params = params.copy()
    return best_params, best_energy, {"optimizer": "random", "evaluations": total_trials}


def _resolve_energy_evaluator(
    *,
    hamiltonian: Hamiltonian | np.ndarray | None,
    energy_evaluator: EnergyEvaluator | None,
    backend: Any,
) -> EnergyEvaluator:
    if energy_evaluator is not None:
        return lambda circuit: float(energy_evaluator(circuit))
    if hamiltonian is None:
        raise ValueError("mogvqe requires either hamiltonian or energy_evaluator")

    if isinstance(hamiltonian, Hamiltonian):
        matrix = hamiltonian.to_matrix(backend)
        n_qubits = int(hamiltonian.n_qubits)
    else:
        matrix_np = np.asarray(hamiltonian, dtype=np.complex128)
        if matrix_np.ndim != 2 or matrix_np.shape[0] != matrix_np.shape[1]:
            raise ValueError("hamiltonian must be a square matrix")
        n_qubits = _infer_n_qubits(matrix_np.shape[0])
        matrix = backend.cast(matrix_np)

    def exact_energy(circuit: Circuit) -> float:
        if int(circuit.n_qubits) != n_qubits:
            raise ValueError(f"circuit.n_qubits={circuit.n_qubits} does not match Hamiltonian width {n_qubits}")
        state = backend.zeros_state(n_qubits)
        unitary = circuit.unitary(backend=backend)
        evolved = backend.apply_unitary(state, unitary)
        return float(backend.expectation_sv(evolved, matrix))

    return exact_energy


def _initial_population(
    initial: MOGVQEIndividual,
    config: MOGVQEConfig,
    edges: Sequence[Edge],
    rng: random.Random,
) -> list[MOGVQEIndividual]:
    population = [initial]
    while len(population) < config.population_size:
        if rng.random() < 0.5:
            base = initial
        else:
            base = _random_individual(initial.n_qubits, config, edges, rng)
        population.append(mutate_individual(base, config, edges=edges, rng=rng))
    return population


def _random_individual(
    n_qubits: int,
    config: MOGVQEConfig,
    edges: Sequence[Edge],
    rng: random.Random,
) -> MOGVQEIndividual:
    if rng.random() < 0.5 and n_qubits > 1:
        linear = [(q, q + 1) for q in range(n_qubits - 1)]
        blocks = [MOGVQEBlock(c, t, config.block_type) for c, t in linear]
    else:
        n_init = rng.randint(n_qubits, max(n_qubits, 4 * n_qubits))
        blocks = [
            MOGVQEBlock(*rng.choice(tuple(edges)), block_type=config.block_type)
            for _ in range(n_init)
        ]
    if config.max_blocks is not None:
        blocks = blocks[: config.max_blocks]
    return MOGVQEIndividual(n_qubits=n_qubits, blocks=tuple(blocks), metadata={"source": "random"})


def _insert_block(
    individual: MOGVQEIndividual,
    config: MOGVQEConfig,
    edges: Sequence[Edge],
    rng: random.Random,
) -> MOGVQEIndividual:
    if config.max_blocks is not None and len(individual.blocks) >= config.max_blocks:
        return individual
    if not edges:
        return individual
    control, target = rng.choice(tuple(edges))
    block = MOGVQEBlock(control, target, config.block_type)
    blocks = list(individual.blocks)
    position = rng.randint(0, len(blocks))
    blocks.insert(position, block)
    return MOGVQEIndividual(individual.n_qubits, tuple(blocks), metadata=dict(individual.metadata))


def _delete_block(
    individual: MOGVQEIndividual,
    config: MOGVQEConfig,
    rng: random.Random,
) -> MOGVQEIndividual:
    if len(individual.blocks) <= config.min_blocks:
        return individual
    blocks = list(individual.blocks)
    del blocks[rng.randrange(len(blocks))]
    return MOGVQEIndividual(individual.n_qubits, tuple(blocks), metadata=dict(individual.metadata))


def _history_entry(generation: int, population: Sequence[MOGVQECandidate]) -> dict[str, Any]:
    front = pareto_front(population)
    best = min(population, key=lambda candidate: (candidate.energy, candidate.cnot_count))
    return {
        "generation": int(generation),
        "best_energy": float(best.energy),
        "best_cnot_count": int(best.cnot_count),
        "pareto_front": [
            {
                "energy": float(candidate.energy),
                "cnot_count": int(candidate.cnot_count),
                "n_blocks": len(candidate.individual.blocks),
            }
            for candidate in front
        ],
    }


def _dominates(left: MOGVQECandidate, right: MOGVQECandidate) -> bool:
    left_obj = left.objectives
    right_obj = right.objectives
    return all(l <= r for l, r in zip(left_obj, right_obj)) and any(
        l < r for l, r in zip(left_obj, right_obj)
    )


def _assign_crowding_distance(front: Sequence[MOGVQECandidate]) -> None:
    if not front:
        return
    for candidate in front:
        candidate.crowding_distance = 0.0
    if len(front) <= 2:
        for candidate in front:
            candidate.crowding_distance = math.inf
        return

    for objective_index in range(2):
        ordered = sorted(front, key=lambda candidate: candidate.objectives[objective_index])
        ordered[0].crowding_distance = math.inf
        ordered[-1].crowding_distance = math.inf
        min_value = ordered[0].objectives[objective_index]
        max_value = ordered[-1].objectives[objective_index]
        if max_value == min_value:
            continue
        scale = max_value - min_value
        for index in range(1, len(ordered) - 1):
            previous_value = ordered[index - 1].objectives[objective_index]
            next_value = ordered[index + 1].objectives[objective_index]
            ordered[index].crowding_distance += (next_value - previous_value) / scale


def _coerce_individual(value: MOGVQEIndividual | Circuit | Sequence[MOGVQEBlock]) -> MOGVQEIndividual:
    if isinstance(value, MOGVQEIndividual):
        return value
    if isinstance(value, Circuit):
        return extract_blocks_from_circuit(value)
    blocks = tuple(value)
    if not blocks:
        raise ValueError("initial_ansatz block sequence cannot be empty")
    n_qubits = max(max(block.control, block.target) for block in blocks) + 1
    return MOGVQEIndividual(n_qubits=n_qubits, blocks=blocks)


def _normalize_parameters(parameters: Sequence[float] | np.ndarray | None, expected: int) -> np.ndarray:
    if parameters is None:
        return np.zeros(expected, dtype=float)
    if isinstance(parameters, (str, bytes)):
        raise TypeError("parameters must be a non-string sequence")
    values = np.asarray(parameters, dtype=float).reshape(-1)
    if values.size != expected:
        raise ValueError(f"Expected {expected} parameter value(s), got {values.size}")
    return values


def _normalize_block_type(block_type: str) -> str:
    key = str(block_type).strip().lower().replace("-", "_")
    try:
        return _BLOCK_ALIASES[key]
    except KeyError as exc:
        allowed = ", ".join(sorted(_BLOCK_ALIASES))
        raise ValueError(f"Unsupported MoG-VQE block type {block_type!r}. Supported values: {allowed}") from exc


def _validated_config(config: MOGVQEConfig) -> MOGVQEConfig:
    if config.population_size <= 0:
        raise ValueError("population_size must be positive")
    if config.generations < 0:
        raise ValueError("generations must be non-negative")
    if config.min_blocks < 0:
        raise ValueError("min_blocks must be non-negative")
    if config.max_blocks is not None and config.max_blocks < config.min_blocks:
        raise ValueError("max_blocks must be >= min_blocks")
    if config.big_mutation_steps < 0:
        raise ValueError("big_mutation_steps must be non-negative")
    if sum((config.mutation_insert_weight, config.mutation_delete_weight, config.mutation_big_weight)) <= 0:
        raise ValueError("At least one mutation weight must be positive")
    if config.parameter_generations < 0:
        raise ValueError("parameter_generations must be non-negative")
    if config.parameter_population_size <= 0:
        raise ValueError("parameter_population_size must be positive")
    if config.parameter_sigma <= 0:
        raise ValueError("parameter_sigma must be positive")
    low, high = config.parameter_bounds
    if not low < high:
        raise ValueError("parameter_bounds must be an increasing (low, high) pair")
    normalized_edges = None
    if config.allowed_edges is not None:
        normalized_edges = tuple(_validate_edge(edge) for edge in config.allowed_edges)
    return MOGVQEConfig(
        population_size=int(config.population_size),
        generations=int(config.generations),
        mutation_insert_weight=float(config.mutation_insert_weight),
        mutation_delete_weight=float(config.mutation_delete_weight),
        mutation_big_weight=float(config.mutation_big_weight),
        big_mutation_steps=int(config.big_mutation_steps),
        min_blocks=int(config.min_blocks),
        max_blocks=None if config.max_blocks is None else int(config.max_blocks),
        block_type=_normalize_block_type(config.block_type),
        allowed_edges=normalized_edges,
        parameter_optimizer=config.parameter_optimizer,
        parameter_generations=int(config.parameter_generations),
        parameter_population_size=int(config.parameter_population_size),
        parameter_sigma=float(config.parameter_sigma),
        parameter_bounds=(float(low), float(high)),
        seed=config.seed,
    )


def _validate_n_qubits(n_qubits: int) -> int:
    value = int(n_qubits)
    if value <= 0:
        raise ValueError("n_qubits must be positive")
    return value


def _validate_edge(edge: Edge) -> Edge:
    if len(edge) != 2:
        raise ValueError(f"Edge must contain two qubit indices, got {edge!r}")
    control, target = int(edge[0]), int(edge[1])
    if control == target:
        raise ValueError("Edge cannot connect a qubit to itself")
    return control, target


def _topology_edges(n_qubits: int, topology: str | Sequence[Edge]) -> tuple[Edge, ...]:
    n_qubits = _validate_n_qubits(n_qubits)
    if not isinstance(topology, str):
        return tuple(_validate_edge(edge) for edge in topology)
    key = topology.strip().lower()
    if key == "linear":
        return tuple((q, q + 1) for q in range(n_qubits - 1))
    if key == "ring":
        edges = [(q, q + 1) for q in range(n_qubits - 1)]
        if n_qubits > 2:
            edges.append((n_qubits - 1, 0))
        return tuple(edges)
    if key in {"all_to_all", "full"}:
        return tuple((i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits))
    raise ValueError("topology must be 'linear', 'ring', 'all_to_all', 'full', or an edge sequence")


def _all_ordered_edges(n_qubits: int) -> tuple[Edge, ...]:
    return tuple((i, j) for i in range(n_qubits) for j in range(n_qubits) if i != j)


def _infer_n_qubits(dim: int) -> int:
    if dim <= 0:
        raise ValueError("Hamiltonian dimension must be positive")
    n_qubits = int(round(math.log2(dim)))
    if (1 << n_qubits) != dim:
        raise ValueError("Hamiltonian dimension must be a power of 2")
    return n_qubits


__all__ = [
    "MOGVQEBlock",
    "MOGVQECandidate",
    "MOGVQEConfig",
    "MOGVQEIndividual",
    "MOGVQEResult",
    "block_hardware_efficient_ansatz",
    "count_cnot_gates",
    "extract_blocks_from_circuit",
    "mutate_individual",
    "non_dominated_sort",
    "nsga_ii_select",
    "pareto_front",
    "mogvqe",
    "run_mog_vqe",
]
