"""Supernet-based QAS for variational quantum algorithms.

This module implements the QAS workflow from Du et al., npj Quantum
Information 2022, in an aicir-native form:

1. set up a supernet-indexed ansatz pool,
2. share weights by layer-level single-qubit gate layout,
3. optimize sampled architectures in one stage,
4. rank sampled architectures with trained shared weights,
5. fine-tune the selected ansatz with independent parameters.

The differentiable path intentionally uses aicir ``Circuit`` objects,
``GPUBackend`` (or ``NPUBackend`` on Ascend), and aicir gate matrix/state
evolution helpers. No external quantum SDK is used.

Memory note — root cause of the BeH2 16-qubit ``SIGKILL`` (cgroup OOM, ~238 GB/rank):
an earlier implementation eagerly materialized the entire single-qubit layout space
``product(single_qubit_gates, repeat=n_qubits)`` (= ``gates ** n_qubits``; for BeH2
that is ``5 ** 16 ≈ 1.5e11`` tuples) at construction and pre-created one shared
parameter set per distinct layout. Feasible only at the tiny ``n_qubits`` the tests
use; at 16 qubits it exhausts host RAM long before training. Architectures are now
sampled **per slot** and shared parameters created **lazily on first visit**, bounding
shared-parameter memory to ``O(supernet_steps × layers × n_qubits)``. Each visited
architecture's parameters are created for *all* supernets on *every* rank so the
``safe`` sharded grad all-reduce keeps an identical key set across ranks (only the
expensive evaluation is sharded, not parameter creation). Do not reintroduce
full-product layout enumeration.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import inspect
import io
import math
import random
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from ...backends.gpu_backend import GPUBackend
from ...backends.npu_backend import NPUBackend
from ...core.operators import Hamiltonian
from ...core.circuit import (
    Circuit,
    cx,
    double_excitation,
    hadamard,
    pauli_x,
    rx,
    ry,
    rz,
    rzz,
    single_excitation,
)
from ...core.gates import apply_gate_to_state, gate_to_matrix
from ...gates import canonical_gate_name, get_gate_spec
from ...ir import circuit_instructions, instruction_name
from ...qml.deriv import psr
from ...noise import DepolarizingChannel, AmplitudeDampingChannel, NoiseModel
from ..core.sharding import (
    shard_context,
    owned_indices,
    all_gather,
    all_reduce_mean,
    broadcast_parameters,
)


ObjectiveFn = Callable[..., torch.Tensor | float]
# Trainable-parameter address inside the supernet. The leading tag ("sq" / "tq")
# separates single-qubit from two-qubit parameters so the two weight-sharing
# signatures never collide; the layout tuple is the per-layer gate layout that
# the weight-sharing rule keys on; param_index keeps multi-angle gates
# addressable (the scoped gate set only uses single-angle rotations, so it is
# always 0 here, but the structure generalizes to u2/u3).
ParameterKey = tuple[str, int, int, tuple[str, ...], int, str, int]

# Sentinel for "no two-qubit gate on this pair" (replaces the old boolean False).
_NO_TWO_QUBIT = "none"


# 可搜索的门 token：搜索空间字母表（SupernetConfig 公开面），非门定义。
_SINGLE_QUBIT_TOKENS = ("i", "h", "rx", "ry", "rz")
_TWO_QUBIT_TOKENS = ("cx", "rzz", "single_excitation")
_FOUR_QUBIT_TOKENS = ("double_excitation",)

# 搜索 token → aicir 规范门名。"i" 表示"该槽不放门"，故不在表内。
_TOKEN_CANONICAL: dict[str, str] = {
    "h": "hadamard",
    "rx": "rx",
    "ry": "ry",
    "rz": "rz",
    "cx": "cx",
    "rzz": "rzz",
    "single_excitation": "single_excitation",
    "givens": "single_excitation",
    "double_excitation": "double_excitation",
}

# 单比特旋转 token → aicir.core 工厂；门语义全部来自 aicir，本模块不定义门。
_SINGLE_ROTATION_FACTORY = {"rx": rx, "ry": ry, "rz": rz}


def _token_n_params(token: str) -> int:
    """该 token 的可训练角度数；取自 aicir.gates 注册表（"i" 无门、0 个）。"""
    if token == "i":
        return 0
    return int(get_gate_spec(_TOKEN_CANONICAL[token]).num_params)


def _build_single_gate(token: str, params: Sequence[Any], qubit: int) -> dict[str, Any] | None:
    """构造单比特门（经 aicir.core 工厂）；"i" 不放门，返回 None。"""
    if token == "i":
        return None
    canonical = _TOKEN_CANONICAL[token]
    if canonical == "hadamard":
        return hadamard(int(qubit))
    return _SINGLE_ROTATION_FACTORY[canonical](params[0], target_qubit=int(qubit))


def _build_two_qubit_gate(
    token: str, params: Sequence[Any], control: int, target: int
) -> dict[str, Any]:
    """构造双比特门（经 aicir.core 工厂）；控制/目标拆分由注册表 num_controls 驱动。"""
    canonical = _TOKEN_CANONICAL[token]
    if canonical == "single_excitation":
        return single_excitation(params[0], qubit_1=int(control), qubit_2=int(target))
    n_controls = int(get_gate_spec(canonical).num_controls)
    ordered = (int(control), int(target))
    controls = ordered[:n_controls]
    targets = ordered[n_controls:]
    if canonical == "cx":
        return cx(target_qubit=targets[0], control_qubits=list(controls))
    return rzz(params[0], qubit_1=targets[0], qubit_2=targets[1])


def _build_four_qubit_gate(
    token: str, params: Sequence[Any], q0: int, q1: int, q2: int, q3: int
) -> dict[str, Any]:
    canonical = _TOKEN_CANONICAL[token]
    if canonical == "double_excitation":
        return double_excitation(params[0], int(q0), int(q1), int(q2), int(q3))
    raise ValueError(f"unsupported four-qubit supernet token {token!r}")


def _normalize_single_gate(name: str) -> str:
    gate = str(name).strip().lower()
    if gate not in _SINGLE_QUBIT_TOKENS:
        raise ValueError(
            f"supernet single-qubit gates are {tuple(_SINGLE_QUBIT_TOKENS)}; got {name!r}"
        )
    if gate != "i" and get_gate_spec(_TOKEN_CANONICAL[gate]) is None:
        raise ValueError(
            f"supernet single-qubit token {name!r} maps to an unregistered aicir gate"
        )
    return gate


def _normalize_two_qubit_gate(name: str) -> str:
    gate = str(name).strip().lower()
    gate = _TOKEN_CANONICAL.get(gate, canonical_gate_name(gate))
    if gate not in _TWO_QUBIT_TOKENS:
        raise ValueError(
            f"supernet two-qubit gates are {tuple(_TWO_QUBIT_TOKENS)}; got {name!r}"
        )
    if get_gate_spec(_TOKEN_CANONICAL[gate]) is None:
        raise ValueError(
            f"supernet two-qubit token {name!r} maps to an unregistered aicir gate"
        )
    return gate


def _normalize_four_qubit_gate(name: str) -> str:
    gate = str(name).strip().lower()
    gate = _TOKEN_CANONICAL.get(gate, canonical_gate_name(gate))
    if gate not in _FOUR_QUBIT_TOKENS:
        raise ValueError(
            f"supernet four-qubit gates are {tuple(_FOUR_QUBIT_TOKENS)}; got {name!r}"
        )
    if get_gate_spec(_TOKEN_CANONICAL[gate]) is None:
        raise ValueError(
            f"supernet four-qubit token {name!r} maps to an unregistered aicir gate"
        )
    return gate


def _normalize_four_qubit_choice(value: Any) -> str:
    name = str(value).strip().lower()
    if name in ("", _NO_TWO_QUBIT, "i", "identity"):
        return _NO_TWO_QUBIT
    return _normalize_four_qubit_gate(name)


def _normalize_two_qubit_choice(value: Any) -> str:
    """Normalize a per-pair choice; accepts ``bool`` for backward compatibility.

    Historically a layer's two-qubit structure was a ``tuple[bool, ...]`` mask
    over CNOT pairs. ``True`` now maps to ``"cx"`` and ``False`` to ``"none"`` so
    existing call sites and tests keep working unchanged.
    """
    if isinstance(value, bool):
        return "cx" if value else _NO_TWO_QUBIT
    name = str(value).strip().lower()
    if name in ("", _NO_TWO_QUBIT, "i", "identity"):
        return _NO_TWO_QUBIT
    return _normalize_two_qubit_gate(name)


@dataclass(frozen=True)
class NoiseConfig:
    """Differentiable supernet 的简化噪声配置。"""

    enabled: bool = False
    probability: float = 0.01
    channel: str = "depolarizing"
    after_gates: tuple[str, ...] | None = None


@dataclass
class SupernetConfig:
    n_qubits: int = 3
    layers: int = 3
    single_qubit_gates: tuple[str, ...] = ("i", "h", "rx", "ry", "rz")
    two_qubit_gates: tuple[str, ...] = ("cx", "rzz")
    two_qubit_pairs: tuple[tuple[int, int], ...] = ((0, 1), (0, 2), (1, 2))
    four_qubit_gates: tuple[str, ...] = ()
    four_qubit_groups: tuple[tuple[int, int, int, int], ...] = ()
    hf_occupied_qubits: tuple[int, ...] = ()
    search_single_qubit_gates: bool = True
    search_two_qubit_gates: bool = True
    search_four_qubit_gates: bool = True
    supernet_num: int = 1
    supernet_steps: int = 100
    ranking_num: int = 50
    finetune_steps: int = 20
    learning_rate: float = 0.05
    finetune_learning_rate: float = 0.03
    seed: int = 42
    device: str = "cpu"
    task: str = "classification"
    log_interval: int = 0
    use_parameter_shift: bool = False
    track_best_validation: bool = True
    ranking_strategy: str = "random"
    use_evolutionary_ranking: bool = False
    ranking_generations: int = 4
    ranking_mutation_rate: float = 0.2
    noise_mode: str = "none"
    shard_mode: str = "safe"
    noise_config: NoiseConfig | None = None


@dataclass(frozen=True)
class LayerArchitecture:
    single_qubit_gates: tuple[str, ...]
    two_qubit_choices: tuple[str, ...]
    four_qubit_choices: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        single = tuple(_normalize_single_gate(gate) for gate in self.single_qubit_gates)
        choices = tuple(_normalize_two_qubit_choice(choice) for choice in self.two_qubit_choices)
        four_choices = tuple(_normalize_four_qubit_choice(choice) for choice in self.four_qubit_choices)
        object.__setattr__(self, "single_qubit_gates", single)
        object.__setattr__(self, "two_qubit_choices", choices)
        object.__setattr__(self, "four_qubit_choices", four_choices)

    @property
    def two_qubit_mask(self) -> tuple[bool, ...]:
        """Backward-compatible view: ``True`` where a two-qubit gate is present."""
        return tuple(choice != _NO_TWO_QUBIT for choice in self.two_qubit_choices)


@dataclass(frozen=True)
class Architecture:
    layers: tuple[LayerArchitecture, ...]


@dataclass
class SupernetResult:
    best_architecture: Architecture
    best_circuit: Circuit
    best_score: float
    best_supernet_id: int
    ranking_records: list[dict[str, Any]]
    supernet_log: list[dict[str, Any]]
    finetune_log: list[dict[str, Any]]
    final_metrics: dict[str, Any]
    config: SupernetConfig


def _make_backend(device: str | torch.device | None) -> GPUBackend:
    """Pick the torch-based backend matching the requested device.

    Ascend NPU lacks complex64 kernels (matmul/kron/conj/...), so NPU devices
    must use ``NPUBackend``, whose overrides decompose those ops into real
    arithmetic. CPU/CUDA use the plain ``GPUBackend``. On a machine without a
    real NPU, ``NPUBackend`` transparently falls back to CPU and behaves like
    ``GPUBackend``.
    """
    if str(device).lower().startswith("npu"):
        return NPUBackend(device=device)
    return GPUBackend(device=device)


def _as_torch_scalar(value: torch.Tensor | float, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.reshape(())
    return torch.tensor(float(value), dtype=torch.float32, device=device)


def _float_value(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(()))
    return float(value)


def _unique_tensors(tensors: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    seen: set[int] = set()
    unique: list[torch.Tensor] = []
    for tensor in tensors:
        ident = id(tensor)
        if ident not in seen:
            seen.add(ident)
            unique.append(tensor)
    return unique


def _circuit_diagram(circuit: Circuit) -> str:
    stream = io.StringIO()
    circuit.show(file=stream)
    return stream.getvalue().rstrip()


class Supernet:
    """One-stage supernet QAS for VQA ansatz selection."""

    def __init__(self, config: SupernetConfig | None = None):
        config = config or SupernetConfig()
        normalized_single = tuple(_normalize_single_gate(gate) for gate in config.single_qubit_gates)
        normalized_two = tuple(_normalize_two_qubit_gate(gate) for gate in config.two_qubit_gates)
        normalized_four = tuple(_normalize_four_qubit_gate(gate) for gate in config.four_qubit_gates)
        config = replace(
            config,
            single_qubit_gates=normalized_single,
            two_qubit_gates=normalized_two,
            four_qubit_gates=normalized_four,
        )
        self.config = config
        self._validate_config()

        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        self._rng = random.Random(config.seed)
        self._np_rng = np.random.default_rng(config.seed)

        self.backend = _make_backend(config.device)
        self.device = self.backend._device
        # Index alphabet for the per-pair two-qubit choice: "none" first so the
        # absent option keeps index 0, then the configured two-qubit gates.
        self._two_qubit_choice_alphabet: tuple[str, ...] = (_NO_TWO_QUBIT,) + tuple(config.two_qubit_gates)
        self._four_qubit_choice_alphabet: tuple[str, ...] = (_NO_TWO_QUBIT,) + tuple(config.four_qubit_gates)
        self._readout_index_cache: dict[int, torch.Tensor] = {}
        self._hamiltonian_cache: dict[int, torch.Tensor] = {}
        self._basis_index_cache: dict[int, torch.Tensor] = {}
        self._pauli_expectation_cache: dict[int, list[tuple[int, int, int, float, float]]] = {}
        self._cached_noise_model: NoiseModel | None | bool = False

        # 共享参数与每 supernet 优化器都按需懒建（见模块顶部内存说明）：构造时不枚举
        # gates**n_qubits 的布局空间。优化器初始为 None，首个参数出现时建立、之后
        # 经 add_param_group 增长（Adam 不接受空参数列表）。
        self.shared_parameters: dict[ParameterKey, torch.nn.Parameter] = {}
        self._best_shared_parameter_values: dict[ParameterKey, torch.Tensor] | None = None
        self._best_validation_accuracy = -math.inf
        self._best_validation_loss = math.inf
        self._optimizers: list[torch.optim.Adam | None] = [None] * config.supernet_num

    def _validate_config(self) -> None:
        cfg = self.config
        ranking_strategy = str(cfg.ranking_strategy).strip().lower()
        if ranking_strategy not in {"random", "evolutionary"}:
            raise ValueError("ranking_strategy must be 'random' or 'evolutionary'")
        if cfg.ranking_generations < 0:
            raise ValueError("ranking_generations must be non-negative")
        if not (0.0 <= float(cfg.ranking_mutation_rate) <= 1.0):
            raise ValueError("ranking_mutation_rate must be in [0, 1]")
        noise_mode = str(cfg.noise_mode).strip().lower()
        if noise_mode not in {"none", "depolarizing", "amplitude_damping"}:
            raise ValueError("noise_mode must be 'none', 'depolarizing', or 'amplitude_damping'")
        if cfg.noise_config is not None:
            p = float(cfg.noise_config.probability)
            if not (0.0 <= p <= 1.0):
                raise ValueError("noise_config.probability must be in [0, 1]")
        shard_mode = str(cfg.shard_mode).strip().lower()
        if shard_mode not in {"safe", "aggressive"}:
            raise ValueError("shard_mode must be 'safe' or 'aggressive'")
        if cfg.n_qubits <= 0:
            raise ValueError("n_qubits must be positive")
        if cfg.layers <= 0:
            raise ValueError("layers must be positive")
        if cfg.supernet_num <= 0:
            raise ValueError("supernet_num must be positive")
        if cfg.supernet_steps < 0:
            raise ValueError("supernet_steps must be non-negative")
        if cfg.ranking_num <= 0:
            raise ValueError("ranking_num must be positive")
        if cfg.finetune_steps < 0:
            raise ValueError("finetune_steps must be non-negative")
        if cfg.learning_rate <= 0 or cfg.finetune_learning_rate <= 0:
            raise ValueError("learning rates must be positive")
        if not cfg.single_qubit_gates:
            raise ValueError("single_qubit_gates must be non-empty")
        for control, target in cfg.two_qubit_pairs:
            if not (0 <= int(control) < cfg.n_qubits and 0 <= int(target) < cfg.n_qubits):
                raise ValueError("two_qubit_pairs contain a qubit outside [0, n_qubits)")
            if int(control) == int(target):
                raise ValueError("two_qubit_pairs entries must use two different qubits")
        for group in cfg.four_qubit_groups:
            if len(group) != 4:
                raise ValueError("four_qubit_groups entries must contain exactly four qubits")
            qubits = tuple(int(q) for q in group)
            if any(q < 0 or q >= cfg.n_qubits for q in qubits):
                raise ValueError("four_qubit_groups contain a qubit outside [0, n_qubits)")
            if len(set(qubits)) != 4:
                raise ValueError("four_qubit_groups entries must not repeat qubits")
        for qubit in cfg.hf_occupied_qubits:
            if not (0 <= int(qubit) < cfg.n_qubits):
                raise ValueError("hf_occupied_qubits contain a qubit outside [0, n_qubits)")

    def _new_shared_parameter(self) -> torch.nn.Parameter:
        value = float(self._np_rng.uniform(-0.05, 0.05))
        return torch.nn.Parameter(torch.tensor(value, dtype=torch.float32, device=self.device))

    def _register_optimizer_param(self, supernet_id: int, tensor: torch.nn.Parameter) -> None:
        """把新建的共享参数注册到该 supernet 的优化器；首参数时建立优化器
        （Adam 不接受空参数列表），其后经 add_param_group 增长。"""
        optimizer = self._optimizers[supernet_id]
        if optimizer is None:
            self._optimizers[supernet_id] = torch.optim.Adam(
                [tensor], lr=self.config.learning_rate
            )
        else:
            optimizer.add_param_group({"params": [tensor]})

    def _shared_param(self, key: ParameterKey, supernet_id: int) -> torch.nn.Parameter:
        """按需取/建一个共享参数（按 layer 单/双比特布局共享，符合论文规则）。"""
        tensor = self.shared_parameters.get(key)
        if tensor is None:
            tensor = self._new_shared_parameter()
            self.shared_parameters[key] = tensor
            self._register_optimizer_param(supernet_id, tensor)
        return tensor

    def _ensure_architecture_params(self, architecture: "Architecture") -> None:
        """为该架构在**所有** supernet 上懒建共享参数，使各 rank 的共享参数键集一致。

        ``safe`` 分片的梯度 all-reduce 按 ``sorted(shared_parameters.keys())`` 对齐，
        要求每个 rank 持有相同键集；评估分片到各 rank（只跑自己拥有的 supernet），但
        参数创建对所有 supernet 进行，故不破坏该不变量。零参数门（i/h/cx）不建参数。
        """
        for supernet_id in range(self.config.supernet_num):
            for layer_id, layer in enumerate(architecture.layers):
                single_layout = tuple(_normalize_single_gate(g) for g in layer.single_qubit_gates)
                for qubit_id, gate_type in enumerate(single_layout):
                    for param_index in range(_token_n_params(gate_type)):
                        key = self.single_parameter_key(
                            supernet_id, layer_id, single_layout, qubit_id, gate_type, param_index
                        )
                        self._shared_param(key, supernet_id)
                two_layout = tuple(_normalize_two_qubit_choice(c) for c in layer.two_qubit_choices)
                for pair_index, choice in enumerate(two_layout):
                    if choice == _NO_TWO_QUBIT:
                        continue
                    for param_index in range(_token_n_params(choice)):
                        key = self.two_qubit_parameter_key(
                            supernet_id, layer_id, two_layout, pair_index, choice, param_index
                        )
                        self._shared_param(key, supernet_id)
                four_layout = tuple(_normalize_four_qubit_choice(c) for c in layer.four_qubit_choices)
                for group_index, choice in enumerate(four_layout):
                    if choice == _NO_TWO_QUBIT:
                        continue
                    for param_index in range(_token_n_params(choice)):
                        key = self.four_qubit_parameter_key(
                            supernet_id, layer_id, four_layout, group_index, choice, param_index
                        )
                        self._shared_param(key, supernet_id)

    def single_parameter_key(
        self,
        supernet_id: int,
        layer_id: int,
        single_layout: Sequence[str],
        qubit_id: int,
        gate_type: str,
        param_index: int = 0,
    ) -> ParameterKey:
        return (
            "sq",
            int(supernet_id),
            int(layer_id),
            tuple(_normalize_single_gate(gate) for gate in single_layout),
            int(qubit_id),
            _normalize_single_gate(gate_type),
            int(param_index),
        )

    def two_qubit_parameter_key(
        self,
        supernet_id: int,
        layer_id: int,
        two_qubit_layout: Sequence[str],
        pair_index: int,
        gate_type: str,
        param_index: int = 0,
    ) -> ParameterKey:
        return (
            "tq",
            int(supernet_id),
            int(layer_id),
            tuple(_normalize_two_qubit_choice(choice) for choice in two_qubit_layout),
            int(pair_index),
            _normalize_two_qubit_gate(gate_type),
            int(param_index),
        )

    def four_qubit_parameter_key(
        self,
        supernet_id: int,
        layer_id: int,
        four_qubit_layout: Sequence[str],
        group_index: int,
        gate_type: str,
        param_index: int = 0,
    ) -> ParameterKey:
        return (
            "fq",
            int(supernet_id),
            int(layer_id),
            tuple(_normalize_four_qubit_choice(choice) for choice in four_qubit_layout),
            int(group_index),
            _normalize_four_qubit_gate(gate_type),
            int(param_index),
        )

    def layer_search_space_size(self) -> int:
        cfg = self.config
        single = len(cfg.single_qubit_gates) ** cfg.n_qubits if cfg.search_single_qubit_gates else 1
        width = len(cfg.two_qubit_pairs)
        if width == 0:
            two = 1
        elif cfg.search_two_qubit_gates:
            two = len(self._two_qubit_choice_alphabet) ** width
        else:
            two = 1
        four_width = len(cfg.four_qubit_groups)
        if four_width == 0:
            four = 1
        elif cfg.search_four_qubit_gates:
            four = len(self._four_qubit_choice_alphabet) ** four_width
        else:
            four = 1
        return single * two * four

    def logical_search_space_size(self) -> int:
        return self.layer_search_space_size() ** self.config.layers

    @staticmethod
    def _decode_layout(idx: int, alphabet: tuple[str, ...], length: int) -> tuple[str, ...]:
        # 复现 product(alphabet, repeat=length) 的第 idx 个元组（首元素变化最慢），
        # 不物化布局列表。与旧实现 self._rng.choice(枚举列表) 字节等价（见模块顶部内存说明）。
        base = len(alphabet)
        return tuple(alphabet[(idx // (base ** (length - 1 - pos))) % base] for pos in range(length))

    def sample_architecture(self) -> Architecture:
        # 采样下标再解码，等价于旧实现对 product 枚举列表的 choice（相同 rng 消耗与结果），
        # 但不枚举/物化 gates**n_qubits 的布局空间（见模块顶部内存说明）。
        cfg = self.config
        layers: list[LayerArchitecture] = []
        width = len(cfg.two_qubit_pairs)
        four_width = len(cfg.four_qubit_groups)
        n_single = len(cfg.single_qubit_gates) ** cfg.n_qubits
        n_two = len(self._two_qubit_choice_alphabet) ** width if width else 1
        n_four = len(self._four_qubit_choice_alphabet) ** four_width if four_width else 1
        for _ in range(cfg.layers):
            if cfg.search_single_qubit_gates:
                single_layout = self._decode_layout(
                    self._rng.randrange(n_single), cfg.single_qubit_gates, cfg.n_qubits
                )
            else:
                self._rng.randrange(1)  # 匹配旧 choice([fixed]) 的 rng 消耗
                single_layout = tuple(cfg.single_qubit_gates[0] for _ in range(cfg.n_qubits))
            if width == 0:
                self._rng.randrange(1)  # 匹配旧 choice([()]) 的 rng 消耗
                two_layout: tuple[str, ...] = ()
            elif cfg.search_two_qubit_gates:
                two_layout = self._decode_layout(
                    self._rng.randrange(n_two), self._two_qubit_choice_alphabet, width
                )
            else:
                self._rng.randrange(1)  # 匹配旧 choice([fixed]) 的 rng 消耗
                fixed = cfg.two_qubit_gates[0] if cfg.two_qubit_gates else _NO_TWO_QUBIT
                two_layout = tuple(fixed for _ in range(width))
            if four_width == 0:
                self._rng.randrange(1)
                four_layout: tuple[str, ...] = ()
            elif cfg.search_four_qubit_gates:
                four_layout = self._decode_layout(
                    self._rng.randrange(n_four), self._four_qubit_choice_alphabet, four_width
                )
            else:
                self._rng.randrange(1)
                fixed_four = cfg.four_qubit_gates[0] if cfg.four_qubit_gates else _NO_TWO_QUBIT
                four_layout = tuple(fixed_four for _ in range(four_width))
            layers.append(LayerArchitecture(single_layout, two_layout, four_layout))
        return Architecture(tuple(layers))

    def encode_architecture(self, architecture: Architecture) -> tuple[tuple[int, ...], ...]:
        encoded: list[tuple[int, ...]] = []
        for layer in architecture.layers:
            single_indices = tuple(
                self.config.single_qubit_gates.index(_normalize_single_gate(gate))
                for gate in layer.single_qubit_gates
            )
            two_indices = tuple(
                self._two_qubit_choice_alphabet.index(_normalize_two_qubit_choice(choice))
                for choice in layer.two_qubit_choices
            )
            four_indices = tuple(
                self._four_qubit_choice_alphabet.index(_normalize_four_qubit_choice(choice))
                for choice in layer.four_qubit_choices
            )
            encoded.append(single_indices + two_indices + four_indices)
        return tuple(encoded)

    def decode_architecture(self, indices: Sequence[Sequence[int]] | Sequence[int]) -> Architecture:
        width = (
            self.config.n_qubits
            + len(self.config.two_qubit_pairs)
            + len(self.config.four_qubit_groups)
        )
        if len(indices) == self.config.layers and all(hasattr(item, "__len__") for item in indices):
            layer_indices = [tuple(int(v) for v in item) for item in indices]  # type: ignore[arg-type]
        else:
            flat = tuple(int(v) for v in indices)  # type: ignore[arg-type]
            if len(flat) != width * self.config.layers:
                raise ValueError("flat architecture index list has the wrong length")
            layer_indices = [flat[i : i + width] for i in range(0, len(flat), width)]

        layers: list[LayerArchitecture] = []
        for raw in layer_indices:
            if len(raw) != width:
                raise ValueError("layer architecture index list has the wrong length")
            single = tuple(self.config.single_qubit_gates[int(idx)] for idx in raw[: self.config.n_qubits])
            two_end = self.config.n_qubits + len(self.config.two_qubit_pairs)
            choices = tuple(self._two_qubit_choice_alphabet[int(v)] for v in raw[self.config.n_qubits : two_end])
            four_choices = tuple(self._four_qubit_choice_alphabet[int(v)] for v in raw[two_end:])
            layers.append(LayerArchitecture(single, choices, four_choices))
        return Architecture(tuple(layers))

    def cnot_count(self, architecture: Architecture) -> int:
        """Number of CNOT (``cx``) gates — the paper's noise-relevant metric."""
        return sum(
            1 for layer in architecture.layers for choice in layer.two_qubit_choices if choice == "cx"
        )

    def two_qubit_count(self, architecture: Architecture) -> int:
        """Number of searched two-qubit gates, including single excitations."""
        return sum(
            1 for layer in architecture.layers for choice in layer.two_qubit_choices if choice != _NO_TWO_QUBIT
        )

    def four_qubit_count(self, architecture: Architecture) -> int:
        return sum(
            1 for layer in architecture.layers for choice in layer.four_qubit_choices if choice != _NO_TWO_QUBIT
        )

    def excitation_count(self, architecture: Architecture) -> int:
        singles = sum(
            1
            for layer in architecture.layers
            for choice in layer.two_qubit_choices
            if _normalize_two_qubit_choice(choice) == "single_excitation"
        )
        return singles + self.four_qubit_count(architecture)

    def build_circuit(
        self,
        architecture: Architecture,
        supernet_id: int = 0,
        parameters: Mapping[ParameterKey, torch.Tensor | float] | None = None,
    ) -> tuple[Circuit, list[ParameterKey], list[torch.Tensor]]:
        if len(architecture.layers) != self.config.layers:
            raise ValueError("architecture layer count does not match config.layers")

        gates: list[dict[str, Any]] = [
            pauli_x(int(qubit)).to_dict()
            for qubit in self.config.hf_occupied_qubits
        ]
        active_keys: list[ParameterKey] = []
        active_tensors: list[torch.Tensor] = []

        for layer_id, layer in enumerate(architecture.layers):
            if len(layer.single_qubit_gates) != self.config.n_qubits:
                raise ValueError("single_qubit_gates length does not match n_qubits")
            if len(layer.two_qubit_choices) != len(self.config.two_qubit_pairs):
                raise ValueError("two_qubit_choices length does not match two_qubit_pairs")
            if len(layer.four_qubit_choices) != len(self.config.four_qubit_groups):
                raise ValueError("four_qubit_choices length does not match four_qubit_groups")

            single_layout = tuple(_normalize_single_gate(gate) for gate in layer.single_qubit_gates)
            for qubit_id, gate_type in enumerate(single_layout):
                params: list[torch.Tensor | float] = []
                for param_index in range(_token_n_params(gate_type)):
                    key = self.single_parameter_key(
                        supernet_id, layer_id, single_layout, qubit_id, gate_type, param_index
                    )
                    theta = self._shared_param(key, supernet_id) if parameters is None else parameters[key]
                    params.append(theta)
                    active_keys.append(key)
                    if isinstance(theta, torch.Tensor):
                        active_tensors.append(theta)
                gate = _build_single_gate(gate_type, params, qubit_id)
                if gate is not None:
                    gates.append(gate)

            two_layout = tuple(_normalize_two_qubit_choice(choice) for choice in layer.two_qubit_choices)
            for pair_index, choice in enumerate(two_layout):
                if choice == _NO_TWO_QUBIT:
                    continue
                control, target = self.config.two_qubit_pairs[pair_index]
                params = []
                for param_index in range(_token_n_params(choice)):
                    key = self.two_qubit_parameter_key(
                        supernet_id, layer_id, two_layout, pair_index, choice, param_index
                    )
                    theta = self._shared_param(key, supernet_id) if parameters is None else parameters[key]
                    params.append(theta)
                    active_keys.append(key)
                    if isinstance(theta, torch.Tensor):
                        active_tensors.append(theta)
                gate = _build_two_qubit_gate(choice, params, control, target)
                if gate is not None:
                    gates.append(gate)

            four_layout = tuple(_normalize_four_qubit_choice(choice) for choice in layer.four_qubit_choices)
            for group_index, choice in enumerate(four_layout):
                if choice == _NO_TWO_QUBIT:
                    continue
                q0, q1, q2, q3 = self.config.four_qubit_groups[group_index]
                params = []
                for param_index in range(_token_n_params(choice)):
                    key = self.four_qubit_parameter_key(
                        supernet_id, layer_id, four_layout, group_index, choice, param_index
                    )
                    theta = self._shared_param(key, supernet_id) if parameters is None else parameters[key]
                    params.append(theta)
                    active_keys.append(key)
                    if isinstance(theta, torch.Tensor):
                        active_tensors.append(theta)
                gate = _build_four_qubit_gate(choice, params, q0, q1, q2, q3)
                if gate is not None:
                    gates.append(gate)

        return Circuit(*gates, n_qubits=self.config.n_qubits, backend=self.backend), active_keys, active_tensors

    def _simulate_gates(self, gates: Sequence[dict[str, Any]], initial_state: torch.Tensor | None = None) -> torch.Tensor:
        state = self.backend.zeros_state(self.config.n_qubits) if initial_state is None else initial_state
        for gate in gates:
            evolved = apply_gate_to_state(gate, state, self.config.n_qubits, self.backend)
            if evolved is None:
                matrix = gate_to_matrix(gate, cir_qubits=self.config.n_qubits, backend=self.backend)
                evolved = self.backend.apply_unitary(state, matrix)
            state = evolved
        return state

    def simulate_state(self, circuit: Circuit, initial_state: torch.Tensor | None = None) -> torch.Tensor:
        return self._simulate_gates(circuit_instructions(circuit), initial_state=initial_state)

    def _noise_model(self) -> NoiseModel | None:
        if self._cached_noise_model is not False:
            return self._cached_noise_model
        cfg = self.config
        mode = str(cfg.noise_mode).strip().lower()
        noise_cfg = cfg.noise_config
        enabled = mode != "none" or (noise_cfg is not None and noise_cfg.enabled)
        if not enabled:
            self._cached_noise_model = None
            return None
        if noise_cfg is None:
            noise_cfg = NoiseConfig(enabled=True, channel=mode)
        channel_name = mode if mode != "none" else str(noise_cfg.channel).strip().lower()
        probability = float(noise_cfg.probability)
        model = NoiseModel()
        after_gates = noise_cfg.after_gates
        for qubit in range(self.config.n_qubits):
            if channel_name == "depolarizing":
                channel = DepolarizingChannel(target_qubit=qubit, p=probability)
            elif channel_name == "amplitude_damping":
                channel = AmplitudeDampingChannel(target_qubit=qubit, gamma=probability)
            else:
                raise ValueError("unsupported noise channel")
            model.add_channel(channel, after_gates=after_gates)
        self._cached_noise_model = model
        return model

    def _has_noise(self) -> bool:
        return self._noise_model() is not None

    def _simulate_density(self, circuit: Circuit) -> torch.Tensor:
        state = self.backend.zeros_state(self.config.n_qubits)
        rho = self.backend.matmul(state, self.backend.dagger(state))
        noise_model = self._noise_model()
        for gate in circuit_instructions(circuit):
            unitary = gate_to_matrix(gate, cir_qubits=self.config.n_qubits, backend=self.backend)
            rho = self.backend.matmul(self.backend.matmul(unitary, rho), self.backend.dagger(unitary))
            if noise_model is not None:
                rho = noise_model.apply(
                    rho,
                    n_qubits=self.config.n_qubits,
                    backend=self.backend,
                    gate_type=instruction_name(gate),
                    gate=gate,
                )
        return rho

    def _readout_indices(self, qubit: int = 0) -> torch.Tensor:
        qubit = int(qubit)
        if qubit not in self._readout_index_cache:
            indices = [
                index
                for index in range(1 << self.config.n_qubits)
                if ((index >> (self.config.n_qubits - qubit - 1)) & 1) == 1
            ]
            self._readout_index_cache[qubit] = torch.tensor(indices, dtype=torch.long, device=self.device)
        return self._readout_index_cache[qubit]

    def _probability_qubit_one(self, state: torch.Tensor, qubit: int = 0) -> torch.Tensor:
        probs = self.backend.measure_probs(state)
        return probs.index_select(0, self._readout_indices(qubit)).sum()

    def _probability_qubit_zero(self, state: torch.Tensor, qubit: int = 0) -> torch.Tensor:
        return torch.as_tensor(1.0, dtype=torch.float32, device=self.device) - self._probability_qubit_one(state, qubit)

    def _classification_predictions(self, circuit: Circuit, dataset: dict[str, tuple[torch.Tensor, torch.Tensor]], split: str) -> torch.Tensor:
        features, _ = dataset.get(split, dataset["train"])
        outputs: list[torch.Tensor] = []
        for row in features:
            encoding = [ry(row[qubit], target_qubit=qubit) for qubit in range(self.config.n_qubits)]
            state = self._simulate_gates(encoding)
            state = self.simulate_state(circuit, initial_state=state)
            outputs.append(self._probability_qubit_zero(state, qubit=self.config.n_qubits - 1))
        return torch.stack(outputs)

    def _classification_loss(self, circuit: Circuit, dataset: dict[str, tuple[torch.Tensor, torch.Tensor]], split: str) -> torch.Tensor:
        predictions = self._classification_predictions(circuit, dataset, split)
        _, labels = dataset.get(split, dataset["train"])
        return torch.mean((predictions - labels) ** 2)

    def _classification_accuracy(self, circuit: Circuit, dataset: dict[str, tuple[torch.Tensor, torch.Tensor]], split: str) -> float:
        with torch.no_grad():
            predictions = self._classification_predictions(circuit, dataset, split)
            _, labels = dataset.get(split, dataset["train"])
            pred_labels = (predictions >= 0.5).to(dtype=labels.dtype)
            return float((pred_labels == labels).float().mean().detach().cpu())

    def _hamiltonian_matrix(self, hamiltonian: Hamiltonian | np.ndarray | torch.Tensor | None) -> torch.Tensor:
        if hamiltonian is None:
            hamiltonian = h2_hamiltonian()
        cache_key = id(hamiltonian)
        if cache_key in self._hamiltonian_cache:
            return self._hamiltonian_cache[cache_key]
        if isinstance(hamiltonian, Hamiltonian):
            matrix = hamiltonian.to_matrix(self.backend)
        elif isinstance(hamiltonian, torch.Tensor):
            matrix = hamiltonian.to(dtype=self.backend._dtype, device=self.device)
        else:
            matrix = self.backend.cast(np.asarray(hamiltonian, dtype=np.complex64))
        self._hamiltonian_cache[cache_key] = matrix
        return matrix

    def _pauli_term_cache(
        self,
        hamiltonian: Hamiltonian,
    ) -> list[tuple[int, int, int, float, float]]:
        cache_key = id(hamiltonian)
        if cache_key in self._pauli_expectation_cache:
            return self._pauli_expectation_cache[cache_key]

        n_qubits = int(hamiltonian.n_qubits)
        cached_terms: list[tuple[int, int, int, float, float]] = []
        for term in hamiltonian._terms:
            flip_mask = 0
            sign_mask = 0
            y_count = 0
            labels = term.qubit_labels
            for qubit, label in enumerate(labels):
                bit_mask = 1 << (n_qubits - qubit - 1)
                if label in {"X", "Y"}:
                    flip_mask ^= bit_mask
                if label in {"Y", "Z"}:
                    sign_mask ^= bit_mask
                if label == "Y":
                    y_count += 1

            coefficient = complex(term.coefficient)
            cached_terms.append(
                (
                    flip_mask,
                    sign_mask,
                    y_count % 4,
                    float(coefficient.real),
                    float(coefficient.imag),
                )
            )
        self._pauli_expectation_cache[cache_key] = cached_terms
        return cached_terms

    def _basis_indices(self, dim: int) -> torch.Tensor:
        cached = self._basis_index_cache.get(dim)
        if cached is None or cached.device != self.device:
            cached = torch.arange(dim, dtype=torch.long, device=self.device)
            self._basis_index_cache[dim] = cached
        return cached

    def _pauli_signs(self, basis_indices: torch.Tensor, sign_mask: int) -> torch.Tensor | None:
        if sign_mask == 0:
            return None
        parity = torch.zeros_like(basis_indices, dtype=torch.bool)
        bit = 0
        mask = int(sign_mask)
        while mask:
            if mask & 1:
                parity = torch.logical_xor(parity, ((basis_indices >> bit) & 1).to(torch.bool))
            mask >>= 1
            bit += 1
        ones = torch.ones_like(basis_indices, dtype=torch.float32)
        return torch.where(parity, -ones, ones)

    def _hamiltonian_expectation(self, state: torch.Tensor, hamiltonian: Hamiltonian | np.ndarray | torch.Tensor | None) -> torch.Tensor:
        if hamiltonian is None:
            hamiltonian = h2_hamiltonian()
        if isinstance(hamiltonian, Hamiltonian):
            # 态向量可能是 (2^n, 1) 列向量；展平成一维，避免与一维 phase/index
            # 向量做广播（(2^n,1)*(2^n,) -> (2^n,2^n)）导致能量被放大 2^n 倍。
            state = state.reshape(-1)
            basis_indices = self._basis_indices(state.numel())
            pauli_cache = self._pauli_term_cache(hamiltonian)
            # NPU autodiff 安全路径：state 在图中只出现一次，梯度累加全程 float32，
            # 避免 1313 次 complex64 grad add（Ascend 缺 aclnnAdd(DT_COMPLEX64)）。
            if hasattr(self.backend, "hamiltonian_expectation_pauli"):
                return self.backend.hamiltonian_expectation_pauli(state, basis_indices, pauli_cache)
            state_real = torch.real(state)
            state_imag = torch.imag(state)
            energy = torch.zeros((), dtype=torch.float32, device=self.device)
            for flip_mask, sign_mask, y_phase, coefficient_real, coefficient_imag in pauli_cache:
                if flip_mask:
                    mapped_indices = torch.bitwise_xor(basis_indices, flip_mask)
                    mapped_real = state_real.index_select(0, mapped_indices)
                    mapped_imag = state_imag.index_select(0, mapped_indices)
                else:
                    mapped_real = state_real
                    mapped_imag = state_imag
                overlap_real = mapped_real * state_real + mapped_imag * state_imag
                overlap_imag = mapped_real * state_imag - mapped_imag * state_real
                signs = self._pauli_signs(basis_indices, sign_mask)
                if signs is not None:
                    overlap_real = overlap_real * signs
                    overlap_imag = overlap_imag * signs
                if y_phase == 0:
                    term_real = overlap_real.sum()
                    term_imag = overlap_imag.sum()
                elif y_phase == 1:
                    term_real = -overlap_imag.sum()
                    term_imag = overlap_real.sum()
                elif y_phase == 2:
                    term_real = -overlap_real.sum()
                    term_imag = -overlap_imag.sum()
                else:
                    term_real = overlap_imag.sum()
                    term_imag = -overlap_real.sum()
                energy = energy + coefficient_real * term_real - coefficient_imag * term_imag
            return energy
        return self.backend.expectation_sv(state, self._hamiltonian_matrix(hamiltonian))

    def _h2_energy(self, circuit: Circuit, hamiltonian: Hamiltonian | np.ndarray | torch.Tensor | None) -> torch.Tensor:
        if self._has_noise():
            rho = self._simulate_density(circuit)
            operator = self._hamiltonian_matrix(hamiltonian)
            return self.backend.expectation_dm(rho, operator)
        state = self.simulate_state(circuit)
        return self._hamiltonian_expectation(state, hamiltonian)

    def _external_objective_loss(
        self,
        objective: ObjectiveFn,
        *,
        architecture: Architecture,
        circuit: Circuit,
        supernet_id: int,
        active_parameter_keys: list[ParameterKey],
        active_parameter_tensors: list[torch.Tensor],
        dataset: Any,
        hamiltonian: Any,
        split: str,
    ) -> torch.Tensor:
        kwargs = {
            "qas": self,
            "architecture": architecture,
            "circuit": circuit,
            "backend": self.backend,
            "supernet_id": supernet_id,
            "active_parameter_keys": active_parameter_keys,
            "active_parameter_tensors": active_parameter_tensors,
            "dataset": dataset,
            "hamiltonian": hamiltonian,
            "split": split,
            "simulate_state": self.simulate_state,
        }

        signature = inspect.signature(objective)
        parameters = signature.parameters
        if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
            value = objective(**kwargs)
        else:
            filtered = {
                name: kwargs[name]
                for name, param in parameters.items()
                if name in kwargs
                and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            }
            if filtered:
                value = objective(**filtered)
            else:
                value = objective(circuit)
        return _as_torch_scalar(value, self.device)

    def _loss(
        self,
        architecture: Architecture,
        supernet_id: int,
        objective: ObjectiveFn | str | None,
        dataset: Any = None,
        hamiltonian: Any = None,
        *,
        split: str = "train",
        parameters: Mapping[ParameterKey, torch.Tensor | float] | None = None,
    ) -> tuple[torch.Tensor, Circuit, list[ParameterKey], list[torch.Tensor]]:
        circuit, active_keys, active_tensors = self.build_circuit(
            architecture,
            supernet_id=supernet_id,
            parameters=parameters,
        )

        task = self.config.task.lower()
        objective_name = objective.lower() if isinstance(objective, str) else None
        if objective_name in {"classification", "binary_classification"} or (
            objective is None and task in {"classification", "binary_classification"}
        ):
            loss = self._classification_loss(circuit, dataset, split)
        elif objective_name in {"h2", "h2_vqe", "vqe"} or (
            objective is None and task in {"h2", "h2_vqe", "vqe"}
        ):
            loss = self._h2_energy(circuit, hamiltonian)
        elif callable(objective):
            loss = self._external_objective_loss(
                objective,
                architecture=architecture,
                circuit=circuit,
                supernet_id=supernet_id,
                active_parameter_keys=active_keys,
                active_parameter_tensors=active_tensors,
                dataset=dataset,
                hamiltonian=hamiltonian,
                split=split,
            )
        else:
            raise ValueError("objective must be callable, a known objective name, or None")
        return _as_torch_scalar(loss, self.device), circuit, active_keys, active_tensors

    def select_supernet(
        self,
        architecture: Architecture,
        objective: ObjectiveFn | str | None,
        dataset: Any = None,
        hamiltonian: Any = None,
        *,
        split: str = "train",
    ) -> tuple[int, list[float]]:
        losses: list[float] = []
        with torch.no_grad():
            for supernet_id in range(self.config.supernet_num):
                loss, _, _, _ = self._loss(
                    architecture,
                    supernet_id,
                    objective,
                    dataset,
                    hamiltonian,
                    split=split,
                )
                losses.append(_float_value(loss))
        selected = min(range(len(losses)), key=losses.__getitem__)
        return selected, losses

    def _ranking_record(
        self,
        candidate_index: int,
        architecture: Architecture,
        selected_id: int,
        losses: list[float],
    ) -> dict[str, Any]:
        return {
            "candidate_index": candidate_index,
            "architecture": architecture,
            "architecture_indices": self.encode_architecture(architecture),
            "selected_supernet_id": selected_id,
            "score": losses[selected_id],
            "candidate_losses": losses,
            "cnot_count": self.cnot_count(architecture),
            "two_qubit_count": self.two_qubit_count(architecture),
            "four_qubit_count": self.four_qubit_count(architecture),
            "excitation_count": self.excitation_count(architecture),
        }

    def _mutate_architecture(self, architecture: Architecture) -> Architecture:
        cfg = self.config
        rows = [list(row) for row in self.encode_architecture(architecture)]
        ranges: list[tuple[int, int, int]] = []
        two_start = cfg.n_qubits
        four_start = cfg.n_qubits + len(cfg.two_qubit_pairs)
        for layer_id in range(cfg.layers):
            if cfg.search_single_qubit_gates:
                for slot in range(cfg.n_qubits):
                    ranges.append((layer_id, slot, len(cfg.single_qubit_gates)))
            if cfg.search_two_qubit_gates:
                for offset in range(len(cfg.two_qubit_pairs)):
                    ranges.append((layer_id, two_start + offset, len(self._two_qubit_choice_alphabet)))
            if cfg.search_four_qubit_gates:
                for offset in range(len(cfg.four_qubit_groups)):
                    ranges.append((layer_id, four_start + offset, len(self._four_qubit_choice_alphabet)))
        if not ranges:
            return architecture

        rate = float(cfg.ranking_mutation_rate)
        mutated = False
        for layer_id, slot, alphabet_size in ranges:
            if self._rng.random() > rate:
                continue
            current = rows[layer_id][slot]
            if alphabet_size > 1:
                choices = [idx for idx in range(alphabet_size) if idx != current]
                rows[layer_id][slot] = self._rng.choice(choices)
            mutated = True
        if not mutated and rate > 0.0:
            layer_id, slot, alphabet_size = self._rng.choice(ranges)
            if alphabet_size > 1:
                current = rows[layer_id][slot]
                choices = [idx for idx in range(alphabet_size) if idx != current]
                rows[layer_id][slot] = self._rng.choice(choices)
        return self.decode_architecture(tuple(tuple(row) for row in rows))

    def _evolutionary_candidates(
        self,
        objective: ObjectiveFn | str | None,
        dataset: Any,
        hamiltonian: Any,
        split: str,
        candidates: Sequence[Architecture] | None,
    ) -> list[Architecture]:
        population = list(candidates) if candidates is not None else [
            self.sample_architecture() for _ in range(self.config.ranking_num)
        ]
        if not population:
            raise ValueError("ranking requires at least one candidate architecture")

        for _ in range(self.config.ranking_generations):
            scored = []
            for architecture in population:
                selected_id, losses = self.select_supernet(
                    architecture, objective, dataset, hamiltonian, split=split
                )
                scored.append((losses[selected_id], architecture))
            scored.sort(key=lambda item: item[0])
            survivor_count = max(1, min(len(scored), math.ceil(len(population) / 2)))
            survivors = [architecture for _, architecture in scored[:survivor_count]]
            next_population = list(survivors)
            while len(next_population) < self.config.ranking_num:
                parent = self._rng.choice(survivors)
                next_population.append(self._mutate_architecture(parent))
            population = next_population
        return population[: self.config.ranking_num]

    def _snapshot_shared_parameters(self) -> dict[ParameterKey, torch.Tensor]:
        return {key: value.detach().clone() for key, value in self.shared_parameters.items()}

    def _restore_shared_parameters(self, values: Mapping[ParameterKey, torch.Tensor]) -> None:
        with torch.no_grad():
            for key, value in values.items():
                self.shared_parameters[key].copy_(value)

    def _track_validation_best(
        self,
        architecture: Architecture,
        supernet_id: int,
        objective: ObjectiveFn | str | None,
        dataset: Any,
        hamiltonian: Any,
    ) -> dict[str, float]:
        loss, circuit, _, _ = self._loss(
            architecture,
            supernet_id,
            objective,
            dataset,
            hamiltonian,
            split="validation",
        )
        validation_loss = _float_value(loss)
        validation_accuracy = self._classification_accuracy(circuit, dataset, "validation")
        if (
            validation_accuracy > self._best_validation_accuracy
            or (
                math.isclose(validation_accuracy, self._best_validation_accuracy)
                and validation_loss < self._best_validation_loss
            )
        ):
            self._best_validation_accuracy = validation_accuracy
            self._best_validation_loss = validation_loss
            self._best_shared_parameter_values = self._snapshot_shared_parameters()
        return {
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "best_validation_loss": self._best_validation_loss,
            "best_validation_accuracy": self._best_validation_accuracy,
        }

    def _grad_norm(self, parameters: Sequence[torch.Tensor]) -> float:
        total = 0.0
        for parameter in parameters:
            if parameter.grad is None:
                continue
            grad = parameter.grad.detach()
            total += float(torch.sum(grad * grad).detach().cpu())
        return math.sqrt(total)

    def _parameter_shift_update(
        self,
        parameters: Sequence[torch.Tensor],
        optimizer: torch.optim.Optimizer,
        loss_closure: Callable[[], torch.Tensor],
    ) -> float:
        # Delegate the parameter-shift rule to aicir.qml.deriv.psr so the
        # gradient definition lives in one place. The active gate angles are
        # optimized as torch Parameters (read by ``loss_closure`` via object
        # identity), so we expose a black-box objective that writes a candidate
        # angle vector into those tensors and returns the resulting scalar loss.
        # psr then shifts one coordinate by ±π/2 at a time (the standard
        # Pauli-rotation rule, coefficient 0.5).
        active = _unique_tensors(parameters)
        optimizer.zero_grad(set_to_none=True)

        base_values = np.array([_float_value(parameter) for parameter in active], dtype=float)

        def _write_values(values: np.ndarray) -> None:
            with torch.no_grad():
                for parameter, value in zip(active, np.asarray(values).reshape(-1)):
                    parameter.copy_(
                        torch.as_tensor(
                            float(value), dtype=parameter.dtype, device=parameter.device
                        ).reshape_as(parameter)
                    )

        def objective(theta: np.ndarray) -> float:
            _write_values(theta)
            with torch.no_grad():
                return _float_value(loss_closure())

        gradients = np.asarray(psr(objective, base_values), dtype=float).reshape(-1)

        # psr leaves the tensors at the last shifted point; restore the base
        # angles before recording the gradient for the optimizer step.
        _write_values(base_values)
        for parameter, grad in zip(active, gradients):
            parameter.grad = torch.as_tensor(
                float(grad), dtype=parameter.dtype, device=parameter.device
            ).reshape_as(parameter)

        grad_norm = self._grad_norm(active)
        optimizer.step()
        return grad_norm

    def _sharded_select(self, architecture, objective, dataset, hamiltonian, split, ctx):
        """仅评估本 rank 拥有的 supernet id，all-gather 后返回完整损失向量。"""
        owned = set(owned_indices(self.config.supernet_num, ctx.rank, ctx.world_size))
        local = []
        with torch.no_grad():
            for supernet_id in range(self.config.supernet_num):
                if supernet_id not in owned:
                    continue
                loss, _, _, _ = self._loss(
                    architecture, supernet_id, objective, dataset, hamiltonian, split=split,
                )
                local.append((supernet_id, _float_value(loss)))
        losses = [math.inf] * self.config.supernet_num
        for part in all_gather(local):
            for supernet_id, value in part:
                losses[supernet_id] = value
        selected = min(range(len(losses)), key=losses.__getitem__)
        return selected, losses

    def _aggressive_step(self, objective, dataset, hamiltonian, ctx):
        # 所有 rank 以相同顺序采样 world_size 个架构；本 rank 负责 arch[rank]。
        archs = [self.sample_architecture() for _ in range(ctx.world_size)]
        # 各 rank 处理不同架构（arch[rank]），但下面对 sorted(shared_parameters) 做
        # all_reduce 要求各 rank 键集一致：故所有 rank 为**全部** archs 懒建参数。
        for arch in archs:
            self._ensure_architecture_params(arch)
        architecture = archs[ctx.rank]
        selected_id, candidate_losses = self.select_supernet(
            architecture, objective, dataset, hamiltonian, split="train",
        )
        loss, _, active_keys, active_tensors = self._loss(
            architecture, selected_id, objective, dataset, hamiltonian, split="train",
        )
        # 清零所有共享参数的梯度，对本 rank 的 loss 做反向传播。
        for param in self.shared_parameters.values():
            param.grad = None
        if loss.requires_grad:
            loss.backward()
        ordered_keys = sorted(self.shared_parameters.keys(), key=str)
        grads = [
            (self.shared_parameters[k].grad
             if self.shared_parameters[k].grad is not None
             else torch.zeros_like(self.shared_parameters[k]))
            for k in ordered_keys
        ]
        all_reduce_mean(grads)
        for key, grad in zip(ordered_keys, grads):
            self.shared_parameters[key].grad = grad
        # 步进所有被某个 rank 选中的 supernet 的优化器。
        selected_ids = set()
        for ids in all_gather([selected_id]):
            selected_ids.update(ids)
        for supernet_id in sorted(selected_ids):
            optimizer = self._optimizers[supernet_id]
            if optimizer is not None:  # 全零参数架构（仅 i/h/cx）无参数、无优化器
                optimizer.step()
        return architecture, selected_id, candidate_losses, _float_value(loss), active_keys

    def optimize_supernet(
        self,
        objective: ObjectiveFn | str | None,
        dataset: Any = None,
        hamiltonian: Any = None,
    ) -> list[dict[str, Any]]:
        ctx = shard_context(self.backend)
        safe_sharded = ctx.is_sharded and self.config.shard_mode.lower() == "safe"
        aggressive_sharded = ctx.is_sharded and self.config.shard_mode.lower() == "aggressive"
        log: list[dict[str, Any]] = []
        for step in range(self.config.supernet_steps):
            if aggressive_sharded:
                architecture, selected_id, candidate_losses, loss_float, active_keys = (
                    self._aggressive_step(objective, dataset, hamiltonian, ctx)
                )
                grad_norm = 0.0
                log.append({
                    "step": step, "architecture": architecture,
                    "architecture_indices": self.encode_architecture(architecture),
                    "selected_supernet_id": selected_id,
                    "candidate_losses": candidate_losses, "loss": loss_float,
                    "gradient_norm": grad_norm,
                    "active_parameter_count": len(active_keys),
                    "cnot_count": self.cnot_count(architecture),
                    "two_qubit_count": self.two_qubit_count(architecture),
                    "four_qubit_count": self.four_qubit_count(architecture),
                    "excitation_count": self.excitation_count(architecture),
                })
                continue
            architecture = self.sample_architecture()
            # 为该架构在所有 supernet 上懒建参数：保证 safe 分片 broadcast_parameters
            # 各 rank 键集一致，并确保被选中 supernet 的优化器已建立。
            self._ensure_architecture_params(architecture)
            if safe_sharded:
                selected_id, candidate_losses = self._sharded_select(
                    architecture, objective, dataset, hamiltonian, "train", ctx
                )
            else:
                selected_id, candidate_losses = self.select_supernet(
                    architecture,
                    objective,
                    dataset,
                    hamiltonian,
                    split="train",
                )
            optimizer = self._optimizers[selected_id]

            def loss_closure() -> torch.Tensor:
                loss_value, _, _, _ = self._loss(
                    architecture,
                    selected_id,
                    objective,
                    dataset,
                    hamiltonian,
                    split="train",
                )
                return loss_value

            loss, _, active_keys, active_tensors = self._loss(
                architecture,
                selected_id,
                objective,
                dataset,
                hamiltonian,
                split="train",
            )
            active_tensors = _unique_tensors(active_tensors)
            # safe_sharded 模式下仅 rank 0 执行梯度步，其余 rank 跳过优化器更新；
            # 全零参数架构（仅 i/h/cx）无参数、无优化器，同样跳过。
            if optimizer is None or (safe_sharded and ctx.rank != 0):
                grad_norm = 0.0
            elif self.config.use_parameter_shift or self._has_noise():
                grad_norm = self._parameter_shift_update(active_tensors, optimizer, loss_closure)
            else:
                optimizer.zero_grad(set_to_none=True)
                if loss.requires_grad:
                    loss.backward()
                grad_norm = self._grad_norm(active_tensors)
                optimizer.step()
            if safe_sharded:
                # rank 0 完成梯度步后广播共享参数，保持各 rank 权重一致
                broadcast_parameters(self.shared_parameters, src=0)

            record = {
                "step": step,
                "architecture": architecture,
                "architecture_indices": self.encode_architecture(architecture),
                "selected_supernet_id": selected_id,
                "candidate_losses": candidate_losses,
                "loss": _float_value(loss),
                "gradient_norm": grad_norm,
                "active_parameter_count": len(active_keys),
                "cnot_count": self.cnot_count(architecture),
                "two_qubit_count": self.two_qubit_count(architecture),
                "four_qubit_count": self.four_qubit_count(architecture),
                "excitation_count": self.excitation_count(architecture),
            }
            if self.config.track_best_validation and self.config.task.lower() in {"classification", "binary_classification"}:
                record.update(self._track_validation_best(architecture, selected_id, objective, dataset, hamiltonian))
            log.append(record)
            if self.config.log_interval and (step + 1) % self.config.log_interval == 0:
                print(
                    f"[supernet] step={step + 1} supernet={selected_id} "
                    f"loss={record['loss']:.6f} cnot={record['cnot_count']}"
                )
        return log

    def rank_architectures(
        self,
        objective: ObjectiveFn | str | None,
        dataset: Any = None,
        hamiltonian: Any = None,
        *,
        candidates: Sequence[Architecture] | None = None,
        split: str = "validation",
    ) -> list[dict[str, Any]]:
        strategy = "evolutionary" if self.config.use_evolutionary_ranking else self.config.ranking_strategy.lower()
        if strategy == "evolutionary":
            candidates = self._evolutionary_candidates(
                objective,
                dataset,
                hamiltonian,
                split,
                candidates,
            )
        elif strategy != "random":
            raise ValueError("ranking_strategy must be 'random' or 'evolutionary'")
        elif candidates is None:
            candidates = [self.sample_architecture() for _ in range(self.config.ranking_num)]
        if not candidates:
            raise ValueError("ranking requires at least one candidate architecture")

        ctx = shard_context(self.backend)
        owned = (
            set(owned_indices(len(candidates), ctx.rank, ctx.world_size))
            if ctx.is_sharded
            else set(range(len(candidates)))
        )

        local_records: list[dict[str, Any]] = []
        for index, architecture in enumerate(candidates):
            if index not in owned:
                continue
            selected_id, losses = self.select_supernet(
                architecture, objective, dataset, hamiltonian, split=split,
            )
            local_records.append(self._ranking_record(index, architecture, selected_id, losses))

        if ctx.is_sharded:
            merged: list[dict[str, Any]] = []
            for part in all_gather(local_records):
                merged.extend(part)
            records = merged
        else:
            records = local_records
        records.sort(key=lambda item: (item["score"], item["candidate_index"]))
        for rank, record in enumerate(records, start=1):
            record["rank"] = rank
        return records

    def _finetune_parameters(self, architecture: Architecture, supernet_id: int) -> dict[ParameterKey, torch.nn.Parameter]:
        _, active_keys, active_tensors = self.build_circuit(architecture, supernet_id=supernet_id)
        parameters: dict[ParameterKey, torch.nn.Parameter] = {}
        for key, tensor in zip(active_keys, active_tensors):
            if key not in parameters:
                parameters[key] = torch.nn.Parameter(tensor.detach().clone())
        return parameters

    def finetune_architecture(
        self,
        architecture: Architecture,
        supernet_id: int,
        objective: ObjectiveFn | str | None,
        dataset: Any = None,
        hamiltonian: Any = None,
    ) -> tuple[Circuit, dict[ParameterKey, torch.nn.Parameter], list[dict[str, Any]], float]:
        parameters = self._finetune_parameters(architecture, supernet_id)
        params_list = list(parameters.values())
        # 当架构全为零参数门（identity/Hadamard）时，params_list 为空，
        # torch.optim.Adam([]) 会抛出 ValueError；此时跳过优化器创建和训练循环，
        # 仅评估一次损失即可，与 finetune_steps==0 的路径保持一致。
        if params_list:
            optimizer: torch.optim.Optimizer | None = torch.optim.Adam(
                params_list, lr=self.config.finetune_learning_rate
            )
        else:
            optimizer = None
        log: list[dict[str, Any]] = []
        best_loss = math.inf
        best_values = {key: value.detach().clone() for key, value in parameters.items()}

        def loss_closure() -> torch.Tensor:
            loss_value, _, _, _ = self._loss(
                architecture,
                supernet_id,
                objective,
                dataset,
                hamiltonian,
                split="train",
                parameters=parameters,
            )
            return loss_value

        if self.config.finetune_steps == 0 or optimizer is None:
            with torch.no_grad():
                loss = loss_closure()
            best_loss = _float_value(loss)

        for step in range(self.config.finetune_steps if optimizer is not None else 0):
            loss = loss_closure()
            active = list(parameters.values())
            if self.config.use_parameter_shift or self._has_noise():
                grad_norm = self._parameter_shift_update(active, optimizer, loss_closure)
            else:
                optimizer.zero_grad(set_to_none=True)
                if loss.requires_grad:
                    loss.backward()
                grad_norm = self._grad_norm(active)
                optimizer.step()
            loss_float = _float_value(loss)
            if loss_float < best_loss:
                best_loss = loss_float
                best_values = {key: value.detach().clone() for key, value in parameters.items()}
            log.append(
                {
                    "step": step,
                    "loss": loss_float,
                    "gradient_norm": grad_norm,
                    "active_parameter_count": len(parameters),
                }
            )

        with torch.no_grad():
            for key, value in best_values.items():
                parameters[key].copy_(value)
        numeric_parameters = {key: float(value.detach().cpu()) for key, value in parameters.items()}
        circuit, _, _ = self.build_circuit(architecture, supernet_id=supernet_id, parameters=numeric_parameters)
        return circuit, parameters, log, float(best_loss)

    def _final_metrics(
        self,
        architecture: Architecture,
        circuit: Circuit,
        finetune_parameters: Mapping[ParameterKey, torch.Tensor | float],
        selected_supernet_id: int,
        objective: ObjectiveFn | str | None,
        dataset: Any,
        hamiltonian: Any,
        ranking_best_score: float,
    ) -> dict[str, Any]:
        task = self.config.task.lower()
        metrics: dict[str, Any] = {
            "selected_ansatz": architecture,
            "selected_cnot_count": self.cnot_count(architecture),
            "selected_two_qubit_count": self.two_qubit_count(architecture),
            "selected_four_qubit_count": self.four_qubit_count(architecture),
            "selected_excitation_count": self.excitation_count(architecture),
            "selected_circuit_ascii": _circuit_diagram(circuit),
            "fine_tuned_parameters": {
                "|".join(str(part) for part in key): _float_value(value)
                for key, value in finetune_parameters.items()
            },
        }
        if task in {"classification", "binary_classification"}:
            for split in ("train", "validation", "test"):
                loss, _, _, _ = self._loss(
                    architecture,
                    selected_supernet_id,
                    objective,
                    dataset,
                    hamiltonian,
                    split=split,
                    parameters=finetune_parameters,
                )
                metrics[f"{split}_loss"] = _float_value(loss)
                metrics[f"{split}_accuracy"] = self._classification_accuracy(circuit, dataset, split)
            metrics["qas_ranked_best_loss"] = float(ranking_best_score)
        elif task in {"h2", "h2_vqe", "vqe"}:
            energy, _, _, _ = self._loss(
                architecture,
                selected_supernet_id,
                objective,
                dataset,
                hamiltonian,
                parameters=finetune_parameters,
            )
            metrics["qas_ranked_best_energy"] = float(ranking_best_score)
            metrics["fine_tuned_energy"] = _float_value(energy)
            metrics["baseline_vqe_energy"] = self._run_fixed_h2_vqe_baseline(hamiltonian)
        else:
            loss, _, _, _ = self._loss(
                architecture,
                selected_supernet_id,
                objective,
                dataset,
                hamiltonian,
                parameters=finetune_parameters,
            )
            metrics["final_loss"] = _float_value(loss)
            metrics["qas_ranked_best_score"] = float(ranking_best_score)
        return metrics

    def _run_fixed_h2_vqe_baseline(self, hamiltonian: Any) -> float:
        # 固定参考线路（Fig. 3a）：每层对每个量子比特施加所有已配置的单比特旋转类型，
        # 再对每对量子比特施加一个固定纠缠门。门参数数及语义全部来自 aicir.gates 注册表。
        single_types = tuple(dict.fromkeys(self.config.single_qubit_gates))
        if self.config.two_qubit_gates:
            entangler = "cx" if "cx" in self.config.two_qubit_gates else self.config.two_qubit_gates[0]
            entangler_n_params = _token_n_params(entangler)
        else:
            entangler = None
            entangler_n_params = 0
        if self.config.four_qubit_gates:
            four_entangler = self.config.four_qubit_gates[0]
            four_entangler_n_params = _token_n_params(four_entangler)
        else:
            four_entangler = None
            four_entangler_n_params = 0

        per_layer = sum(_token_n_params(g) for g in single_types) * self.config.n_qubits
        if entangler is not None:
            per_layer += entangler_n_params * len(self.config.two_qubit_pairs)
        if four_entangler is not None:
            per_layer += four_entangler_n_params * len(self.config.four_qubit_groups)
        total = per_layer * self.config.layers

        parameters = [self._new_shared_parameter() for _ in range(total)]
        optimizer = torch.optim.Adam(parameters, lr=self.config.finetune_learning_rate) if parameters else None
        steps = int(self.config.finetune_steps)
        best_energy = math.inf

        def energy_closure() -> torch.Tensor:
            gates: list[dict[str, Any]] = [
                pauli_x(int(qubit)).to_dict()
                for qubit in self.config.hf_occupied_qubits
            ]
            cursor = 0
            for _ in range(self.config.layers):
                for qubit in range(self.config.n_qubits):
                    for gate_type in single_types:
                        n = _token_n_params(gate_type)
                        slot = parameters[cursor : cursor + n]
                        cursor += n
                        gate = _build_single_gate(gate_type, slot, qubit)
                        if gate is not None:
                            gates.append(gate)
                if entangler is not None:
                    for control, target in self.config.two_qubit_pairs:
                        slot = parameters[cursor : cursor + entangler_n_params]
                        cursor += entangler_n_params
                        gate = _build_two_qubit_gate(entangler, slot, int(control), int(target))
                        if gate is not None:
                            gates.append(gate)
                if four_entangler is not None:
                    for q0, q1, q2, q3 in self.config.four_qubit_groups:
                        slot = parameters[cursor : cursor + four_entangler_n_params]
                        cursor += four_entangler_n_params
                        gate = _build_four_qubit_gate(
                            four_entangler, slot, int(q0), int(q1), int(q2), int(q3)
                        )
                        if gate is not None:
                            gates.append(gate)
            state = self._simulate_gates(gates)
            return self._hamiltonian_expectation(state, hamiltonian)

        if steps == 0 or optimizer is None:
            return _float_value(energy_closure())

        for _ in range(steps):
            loss = energy_closure()
            best_energy = min(best_energy, _float_value(loss))
            optimizer.zero_grad(set_to_none=True)
            if self.config.use_parameter_shift or self._has_noise():
                self._parameter_shift_update(parameters, optimizer, energy_closure)
            else:
                loss.backward()
                optimizer.step()
        return float(best_energy)

    def train(
        self,
        objective: ObjectiveFn | str | None = None,
        dataset: Any = None,
        hamiltonian: Any = None,
    ) -> SupernetResult:
        task = self.config.task.lower()
        if task in {"classification", "binary_classification"}:
            dataset = prepare_classification_dataset(dataset, self.config, self.device)
            ranking_split = "validation"
        elif task in {"h2", "h2_vqe", "vqe"}:
            hamiltonian = h2_hamiltonian() if hamiltonian is None else hamiltonian
            ranking_split = "train"
        else:
            ranking_split = "validation"

        supernet_log = self.optimize_supernet(objective, dataset=dataset, hamiltonian=hamiltonian)
        if (
            self.config.track_best_validation
            and task in {"classification", "binary_classification"}
            and self._best_shared_parameter_values is not None
        ):
            self._restore_shared_parameters(self._best_shared_parameter_values)
        ranking_records = self.rank_architectures(
            objective,
            dataset=dataset,
            hamiltonian=hamiltonian,
            split=ranking_split,
        )
        # 分片感知的 Top-K 并行微调：每个 rank 负责一个候选架构，
        # 全局汇聚后选出得分最低的最优架构。单卡路径等价于原逻辑。
        ctx = shard_context(self.backend)
        if ctx.is_sharded:
            cand_index = ctx.rank
        else:
            cand_index = 0

        if cand_index < len(ranking_records):
            local_record = ranking_records[cand_index]
            local_arch = local_record["architecture"]
            local_supernet_id = int(local_record["selected_supernet_id"])
            local_circuit, local_params, local_finetune_log, local_score = (
                self.finetune_architecture(
                    local_arch, local_supernet_id, objective,
                    dataset=dataset, hamiltonian=hamiltonian,
                )
            )
            local_payload = {
                "score": float(local_score),
                "supernet_id": local_supernet_id,
                "ranking_index": cand_index,
                "n_qubits": int(local_circuit.n_qubits),
                "gates": local_circuit.to_gate_dicts(),
                # 使用真实 ParameterKey（纯 tuple，可序列化）作为键，保证 all_gather
                # 后各 rank 能用胜出架构的键直接索引 _final_metrics/_loss。
                "numeric_parameters": {key: float(value.detach()) for key, value in local_params.items()},
            }
        else:
            local_payload = None  # rank 数多于候选架构数
            local_params = {}
            local_finetune_log: list[dict[str, Any]] = []

        if ctx.is_sharded:
            payloads = [p for p in all_gather(local_payload) if p is not None]
            best_payload = min(payloads, key=lambda p: p["score"])
        else:
            best_payload = local_payload

        best_record = ranking_records[best_payload["ranking_index"]]
        best_architecture = best_record["architecture"]
        selected_supernet_id = int(best_payload["supernet_id"])
        best_circuit = Circuit(
            *best_payload["gates"], n_qubits=best_payload["n_qubits"], backend=self.backend,
        )
        # 使用全局胜出 payload 中的参数（真实 ParameterKey → float），确保所有 rank
        # 的 _final_metrics 都基于同一套胜出参数计算，避免非胜出 rank 用自己的参数
        # 与胜出架构错配。单卡路径下 best_payload == local_payload，语义不变。
        finetune_parameters = best_payload["numeric_parameters"]
        # finetune_log 仅在本 rank 产出胜出结果时为真实日志，否则为空列表。
        finetune_log = local_finetune_log if (best_payload["ranking_index"] == cand_index) else []
        finetune_score = best_payload["score"]
        final_metrics = self._final_metrics(
            best_architecture,
            best_circuit,
            finetune_parameters,
            selected_supernet_id,
            objective,
            dataset,
            hamiltonian,
            ranking_best_score=float(best_record["score"]),
        )
        final_metrics["ranking_score_distribution"] = [float(record["score"]) for record in ranking_records]
        if task in {"classification", "binary_classification"} and self._best_validation_accuracy > -math.inf:
            final_metrics["best_supernet_validation_accuracy"] = float(self._best_validation_accuracy)
            final_metrics["best_supernet_validation_loss"] = float(self._best_validation_loss)

        if task in {"classification", "binary_classification"}:
            best_score = float(final_metrics["validation_loss"])
        elif task in {"h2", "h2_vqe", "vqe"}:
            best_score = float(final_metrics["fine_tuned_energy"])
        else:
            best_score = float(final_metrics.get("final_loss", finetune_score))

        return SupernetResult(
            best_architecture=best_architecture,
            best_circuit=best_circuit,
            best_score=best_score,
            best_supernet_id=selected_supernet_id,
            ranking_records=ranking_records,
            supernet_log=supernet_log,
            finetune_log=finetune_log,
            final_metrics=final_metrics,
            config=self.config,
        )


def prepare_classification_dataset(
    dataset: Any,
    config: SupernetConfig,
    device: torch.device,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    if dataset is None:
        x, y = _synthetic_quantum_kernel_dataset(config.seed, config.n_qubits)
        return {
            "train": _to_torch_pair(x[:100], y[:100], device),
            "validation": _to_torch_pair(x[100:200], y[100:200], device),
            "test": _to_torch_pair(x[200:300], y[200:300], device),
        }

    if not isinstance(dataset, Mapping):
        raise TypeError("classification dataset must be a mapping or None")

    if all(key in dataset for key in ("train", "validation", "test")):
        return {
            "train": _to_torch_pair(*dataset["train"], device),
            "validation": _to_torch_pair(*dataset["validation"], device),
            "test": _to_torch_pair(*dataset["test"], device),
        }
    required = ("x_train", "y_train", "x_validation", "y_validation", "x_test", "y_test")
    if all(key in dataset for key in required):
        return {
            "train": _to_torch_pair(dataset["x_train"], dataset["y_train"], device),
            "validation": _to_torch_pair(dataset["x_validation"], dataset["y_validation"], device),
            "test": _to_torch_pair(dataset["x_test"], dataset["y_test"], device),
        }
    raise ValueError("dataset must provide train/validation/test pairs or x_*/y_* arrays")


def _ry_matrix(theta: float) -> np.ndarray:
    half = float(theta) / 2.0
    return np.array(
        [[math.cos(half), -math.sin(half)], [math.sin(half), math.cos(half)]],
        dtype=np.complex64,
    )


def _apply_single_qubit_matrix_numpy(state: np.ndarray, matrix: np.ndarray, n_qubits: int, qubit: int) -> np.ndarray:
    tensor = state.reshape((2,) * n_qubits)
    moved = np.moveaxis(tensor, qubit, 0).reshape(2, -1)
    updated = matrix @ moved
    return np.moveaxis(updated.reshape((2,) + (2,) * (n_qubits - 1)), 0, qubit).reshape(-1)


def _apply_cnot_numpy(state: np.ndarray, n_qubits: int, control: int, target: int) -> np.ndarray:
    updated = np.zeros_like(state)
    for index, amplitude in enumerate(state):
        if ((index >> (n_qubits - control - 1)) & 1) == 1:
            index ^= 1 << (n_qubits - target - 1)
        updated[index] += amplitude
    return updated


def _synthetic_quantum_kernel_dataset(seed: int, n_qubits: int) -> tuple[np.ndarray, np.ndarray]:
    if n_qubits != 3:
        raise ValueError("The built-in Supplementary synthetic classification dataset requires n_qubits=3.")

    rng = np.random.default_rng(seed)
    theta_star = rng.uniform(0.0, 2.0 * math.pi, size=(3, 3))
    features: list[np.ndarray] = []
    labels: list[float] = []
    max_trials = 200_000

    for _ in range(max_trials):
        x = rng.uniform(-math.pi, math.pi, size=3).astype(np.float32)
        state = np.zeros(8, dtype=np.complex64)
        state[0] = 1.0
        for qubit in range(3):
            state = _apply_single_qubit_matrix_numpy(state, _ry_matrix(float(x[qubit])), 3, qubit)
        for layer in range(3):
            for qubit in range(3):
                state = _apply_single_qubit_matrix_numpy(state, _ry_matrix(float(theta_star[layer, qubit])), 3, qubit)
            state = _apply_cnot_numpy(state, 3, control=0, target=1)
            state = _apply_cnot_numpy(state, 3, control=1, target=2)

        prob_last_qubit_zero = 0.0
        for index, amplitude in enumerate(state):
            if ((index >> 0) & 1) == 0:
                prob_last_qubit_zero += float(np.real(np.conj(amplitude) * amplitude))
        if prob_last_qubit_zero >= 0.75:
            features.append(x)
            labels.append(1.0)
        elif prob_last_qubit_zero <= 0.25:
            features.append(x)
            labels.append(0.0)
        if len(features) == 300:
            return np.stack(features).astype(np.float32), np.asarray(labels, dtype=np.float32)

    raise RuntimeError("Could not generate 300 accepted synthetic classification samples.")


def _to_torch_pair(features: Any, labels: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.as_tensor(features, dtype=torch.float32, device=device)
    y = torch.as_tensor(labels, dtype=torch.float32, device=device).reshape(-1)
    if x.ndim != 2:
        raise ValueError("features must be a 2D array")
    if x.shape[0] != y.shape[0]:
        raise ValueError("features and labels must contain the same number of samples")
    return x, y


def h2_hamiltonian() -> Hamiltonian:
    """Return the four-qubit molecular hydrogen Hamiltonian from Supplementary Eq. (19)."""

    g = -0.042
    z = {
        0: 0.178,
        1: 0.178,
        2: -0.243,
        3: -0.243,
    }
    zz = {
        (0, 1): 0.171,
        (0, 2): 0.123,
        (0, 3): 0.168,
        (1, 2): 0.168,
        (1, 3): 0.123,
        (2, 3): 0.176,
    }
    quartic = {
        "YXXY": 0.045,
        "YYXX": -0.045,
        "XXYY": -0.045,
        "XYYX": 0.045,
    }

    terms = [("I", g, [0])]
    for qubit, coefficient in z.items():
        terms.append(("Z", coefficient, [qubit]))
    for (q0, q1), coefficient in zz.items():
        terms.append(("ZZ", coefficient, [q0, q1]))
    terms.extend([
        ("YXXY", quartic["YXXY"]),
        ("YYXX", quartic["YYXX"]),
        ("XXYY", quartic["XXYY"]),
        ("XYYX", quartic["XYYX"]),
    ])
    return Hamiltonian(n_qubits=4, terms=terms)


def train_supernet(
    objective: ObjectiveFn | str | None = None,
    config: SupernetConfig | None = None,
    dataset: Any = None,
    hamiltonian: Any = None,
) -> SupernetResult:
    return Supernet(config).train(objective, dataset=dataset, hamiltonian=hamiltonian)


def classification_supernet(config: SupernetConfig | None = None) -> SupernetResult:
    if config is None:
        config = SupernetConfig(
            n_qubits=3,
            layers=3,
            single_qubit_gates=("i", "h", "rx", "ry", "rz"),
            two_qubit_gates=("cx", "rzz"),
            two_qubit_pairs=((0, 1), (0, 2), (1, 2)),
            search_single_qubit_gates=True,
            search_two_qubit_gates=True,
            task="classification",
        )
    else:
        config = replace(config, task="classification")
    return train_supernet(None, config=config)


def h2_vqe_supernet(config: SupernetConfig | None = None) -> SupernetResult:
    if config is None:
        config = SupernetConfig(
            n_qubits=4,
            layers=3,
            single_qubit_gates=("i", "h", "rx", "ry", "rz"),
            two_qubit_gates=("cx", "rzz"),
            two_qubit_pairs=((0, 1), (1, 2), (2, 3)),
            search_single_qubit_gates=True,
            search_two_qubit_gates=True,
            supernet_steps=500,
            ranking_num=500,
            finetune_steps=50,
            task="h2_vqe",
        )
    else:
        config = replace(config, task="h2_vqe")
    if config.n_qubits != 4:
        raise ValueError("h2_vqe_supernet requires n_qubits=4")
    return train_supernet(None, config=config, hamiltonian=h2_hamiltonian())


def _default_two_qubit_pairs(n_qubits: int) -> tuple[tuple[int, int], ...]:
    """Nearest- + next-nearest-neighbour pairs on a linear chain of ``n_qubits``.

    A reasonable general-purpose entangler connectivity: every pair ``(i, j)``
    with ``1 <= j - i <= 2``. This matches the connectivity used by the LiH/H2O
    VQE demos.
    """
    return tuple(
        (i, j)
        for i in range(n_qubits)
        for j in range(i + 1, n_qubits)
        if j - i <= 2
    )


def supernet_qas(
    hamiltonian: Any,
    layers: int = 6,
    supernet_num: int = 5,
    supernet_steps: int = 250,
    finetune_steps: int = 250,
    *,
    n_qubits: int | None = None,
    ranking_num: int = 80,
    single_qubit_gates: tuple[str, ...] = ("i", "h", "rx", "ry", "rz"),
    two_qubit_gates: tuple[str, ...] = ("cx", "rzz"),
    two_qubit_pairs: tuple[tuple[int, int], ...] | None = None,
    four_qubit_gates: tuple[str, ...] = (),
    four_qubit_groups: tuple[tuple[int, int, int, int], ...] = (),
    hf_occupied_qubits: tuple[int, ...] = (),
    learning_rate: float = 0.1,
    finetune_learning_rate: float = 0.05,
    seed: int = 2,
    device: str = "cpu",
    use_parameter_shift: bool = False,
    mode: str = "safe",
    **config_overrides: Any,
) -> SupernetResult:
    """Search and fine-tune a ground-state-preparing ansatz for ``hamiltonian``.

    High-level, VQE-focused wrapper over :class:`Supernet` that exposes the knobs
    you usually tune directly, so callers do not have to assemble a full
    :class:`SupernetConfig`. It builds the weight-shared supernet, optimises and
    ranks the sampled ansatze, then fine-tunes the best one.

    The five primary parameters (``hamiltonian``, ``layers``, ``supernet_num``,
    ``supernet_steps``, ``finetune_steps``) are the levers usually swept; the rest
    have sensible VQE defaults. ``n_qubits`` and ``two_qubit_pairs`` are derived
    from the Hamiltonian when omitted. Any other :class:`SupernetConfig` field can
    still be set through ``config_overrides``.

    Args:
        hamiltonian: Target Hamiltonian (must expose ``n_qubits`` or pass it).
        layers: Ansatz depth ``L``.
        supernet_num: Number of weight-shared supernets ``W``.
        supernet_steps: Supernet optimisation steps.
        finetune_steps: Fine-tuning steps for the selected circuit.
        n_qubits: Qubit count; inferred from ``hamiltonian.n_qubits`` if ``None``.
        ranking_num: Number of sampled candidate ansatze to rank.
        single_qubit_gates: Single-qubit gate pool searched per site.
        two_qubit_gates: Two-qubit gate pool (e.g. ``cx``/``rzz``).
        two_qubit_pairs: Entangler connectivity; nearest + next-nearest by default.
        four_qubit_gates: Four-qubit gate pool, currently used for ``double_excitation``.
        four_qubit_groups: Qubit quadruples searched by four-qubit gates.
        hf_occupied_qubits: Qubits flipped before the searched ansatz to prepare HF.
        learning_rate: Adam learning rate for the supernet phase.
        finetune_learning_rate: Adam learning rate for fine-tuning.
        seed: Random seed.
        device: Torch device string (``"cpu"``, ``"cuda"``, ``"npu:0"`` ...).
        use_parameter_shift: Use parameter-shift gradients instead of autograd.
        mode: 分片模式，转发至 ``SupernetConfig.shard_mode``；可选值 ``"safe"`` 或 ``"aggressive"``。
        **config_overrides: Any remaining :class:`SupernetConfig` field.

    Returns:
        SupernetResult: ``final_metrics["fine_tuned_energy"]`` and
        ``final_metrics["baseline_vqe_energy"]`` hold the key energies, and
        ``best_circuit`` is the selected, fine-tuned ansatz.
    """
    if n_qubits is None:
        n_qubits = getattr(hamiltonian, "n_qubits", None)
    if n_qubits is None:
        raise ValueError(
            "n_qubits could not be inferred from the Hamiltonian; pass n_qubits explicitly."
        )
    if two_qubit_pairs is None:
        two_qubit_pairs = _default_two_qubit_pairs(int(n_qubits))
    task = config_overrides.pop("task", "vqe")
    config_overrides.setdefault("shard_mode", mode)

    config = SupernetConfig(
        n_qubits=int(n_qubits),
        layers=layers,
        single_qubit_gates=single_qubit_gates,
        two_qubit_gates=two_qubit_gates,
        two_qubit_pairs=two_qubit_pairs,
        four_qubit_gates=four_qubit_gates,
        four_qubit_groups=four_qubit_groups,
        hf_occupied_qubits=hf_occupied_qubits,
        supernet_num=supernet_num,
        supernet_steps=supernet_steps,
        ranking_num=ranking_num,
        finetune_steps=finetune_steps,
        learning_rate=learning_rate,
        finetune_learning_rate=finetune_learning_rate,
        seed=seed,
        device=device,
        task=task,
        use_parameter_shift=use_parameter_shift,
        **config_overrides,
    )
    return Supernet(config).train(hamiltonian=hamiltonian)


__all__ = [
    "NoiseConfig",
    "SupernetConfig",
    "LayerArchitecture",
    "Architecture",
    "SupernetResult",
    "Supernet",
    "train_supernet",
    "classification_supernet",
    "h2_vqe_supernet",
    "supernet_qas",
]
