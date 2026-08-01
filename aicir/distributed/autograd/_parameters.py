"""Physical parameter containers backed only by real-valued tensor leaves."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
import hashlib
import json
from numbers import Integral
import math
from typing import Any

import torch

from .._contracts import PARAMETER_STRUCTURE_ERROR, synchronize_autograd_failure
from ._pair import _Pair


_BUCKET_ALIAS_ATTRIBUTE = "_aicir_gradient_bucket_alias"


@dataclass(frozen=True)
class _ParameterEntry:
    """One deterministic replicated-leaf descriptor.

    The component field deliberately distinguishes paired-real physical values:
    their real and imaginary adjoints occupy independent ranges in the one
    float32 transport buffer.
    """

    name: str
    value: torch.Tensor
    component: str = "real"


def _normalize_parameter_entries(parameters) -> tuple[_ParameterEntry, ...]:
    """Normalize public/internal bucket inputs without changing their order."""

    entries = []
    for index, item in enumerate(parameters):
        if isinstance(item, _ParameterEntry):
            entry = item
        elif isinstance(item, torch.Tensor):
            entry = _ParameterEntry(f"parameter[{index}]", item)
        elif isinstance(item, tuple) and len(item) == 3:
            name, value, component = item
            entry = _ParameterEntry(str(name), value, str(component))
        else:
            raise TypeError("梯度桶参数必须是 torch.Tensor 或 (name, tensor, component)")
        if not isinstance(entry.value, torch.Tensor):
            raise TypeError("梯度桶参数必须是 torch.Tensor")
        entries.append(entry)
    return tuple(entries)


def _parameter_structure_digest(parameters) -> bytes:
    """Return the fixed digest used for collective parameter preflight."""

    entries = _normalize_parameter_entries(parameters)
    fields_for_digest = [
        {
            "name": entry.name,
            "shape": tuple(int(axis) for axis in entry.value.shape),
            "dtype": str(entry.value.dtype),
            "requires_grad": bool(entry.value.requires_grad),
            "paired_component": entry.component,
        }
        for entry in entries
    ]
    encoded = json.dumps(fields_for_digest, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).digest()


def _preflight_parameter_structure(parameters, *, communicator) -> bytes:
    """Agree the ordered replicated parameter schema before any data work."""

    digest = _parameter_structure_digest(parameters)
    if int(getattr(communicator, "world_size", 1)) <= 1:
        return digest
    local = torch.tensor(
        list(digest), dtype=torch.float32, device=communicator.device
    )
    gather = getattr(communicator, "all_gather_real", None)
    if not callable(gather):
        # Existing communicator implementations expose ``all_gather``; it is
        # still a real float32 control-plane payload here.
        gather = communicator.all_gather
    gathered = gather(local)
    values = {
        bytes(item)
        if isinstance(item, (bytes, bytearray))
        else bytes(int(value) for value in item.detach().cpu().reshape(-1).tolist())
        for item in gathered
    }
    if len(values) != 1:
        synchronize_autograd_failure(communicator)
        raise ValueError(PARAMETER_STRUCTURE_ERROR)
    return digest


class _GradientBucketFn(torch.autograd.Function):
    """One autograd node that synchronizes all replicated real adjoints."""

    @staticmethod
    def forward(ctx, communicator, *values: torch.Tensor):
        if not values:
            raise ValueError("梯度桶至少需要一个参数")
        for value in values:
            if value.dtype != torch.float32 or torch.is_complex(value):
                raise TypeError("梯度桶参数必须是实数 torch.float32")
        ctx.communicator = communicator
        ctx.shapes = tuple(tuple(value.shape) for value in values)
        ctx.devices = tuple(value.device for value in values)
        # clone establishes distinct internal aliases without touching caller
        # leaves or their immutable typed instructions.
        return tuple(value.clone() for value in values)

    @staticmethod
    def backward(ctx, *gradients):
        packed_parts = []
        for gradient, shape, device in zip(gradients, ctx.shapes, ctx.devices):
            if gradient is None:
                gradient = torch.zeros(shape, dtype=torch.float32, device=device)
            elif gradient.dtype != torch.float32 or torch.is_complex(gradient):
                raise TypeError("梯度桶只能规约实数 torch.float32 梯度")
            packed_parts.append(gradient.reshape(-1))
        packed = torch.cat(packed_parts)
        synchronized = ctx.communicator.all_reduce_sum_real(packed)
        offset = 0
        unpacked = []
        for shape in ctx.shapes:
            size = math.prod(shape)
            unpacked.append(synchronized.narrow(0, offset, size).reshape(shape))
            offset += size
        return (None, *unpacked)


def _bucket_parameters(parameters, *, communicator) -> tuple[torch.Tensor, ...]:
    """Return deterministic differentiable aliases for replicated leaves.

    This helper is deliberately the only route that creates gradient bucket
    aliases.  It performs the schema all-gather before forwarding any circuit
    data and creates exactly one custom autograd node for the complete list.
    """

    candidates = _normalize_parameter_entries(parameters)
    # Every rank enters this preflight, including ranks with no candidate or
    # only detached candidates.  Otherwise rank-local filtering could let one
    # rank enter a gradient collective while another returns early.
    _preflight_parameter_structure(candidates, communicator=communicator)
    entries = tuple(entry for entry in candidates if entry.value.requires_grad)
    if not entries:
        return ()
    aliases = _GradientBucketFn.apply(communicator, *(entry.value for entry in entries))
    if isinstance(aliases, torch.Tensor):
        aliases = (aliases,)
    for alias in aliases:
        setattr(alias, _BUCKET_ALIAS_ATTRIBUTE, True)
    return tuple(aliases)


def _replace_trainable_aliases(value: Any, mapping: dict[int, torch.Tensor]):
    """Recursively rebuild typed payloads while retaining non-trainable data."""

    alias = mapping.get(id(value))
    if alias is not None:
        return alias
    if isinstance(value, _Pair):
        return _Pair(
            _replace_trainable_aliases(value.real, mapping),
            _replace_trainable_aliases(value.imag, mapping),
        )
    if isinstance(value, tuple):
        return tuple(_replace_trainable_aliases(item, mapping) for item in value)
    if isinstance(value, list):
        return [_replace_trainable_aliases(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: _replace_trainable_aliases(item, mapping) for key, item in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        replacements = {
            field.name: _replace_trainable_aliases(getattr(value, field.name), mapping)
            for field in fields(value)
        }
        return replace(value, **replacements)
    return value


def _bind_trainable_aliases(circuit, mapping) -> Any:
    """Rebuild a :class:`Circuit` and optional noise model without mutation."""

    from ...core.circuit import Circuit
    from ...ir import circuit_instructions, instruction_parameter, instruction_with_parameter

    instructions = []
    for instruction in circuit_instructions(circuit):
        parameter = instruction_parameter(instruction)
        instructions.append(
            instruction_with_parameter(
                instruction, _replace_trainable_aliases(parameter, mapping)
            )
            if parameter is not None
            else instruction
        )
    rebound = Circuit(*instructions, n_qubits=circuit.n_qubits, backend=getattr(circuit, "backend", None))
    if hasattr(circuit, "noise_model"):
        rebound.noise_model = _replace_trainable_aliases(circuit.noise_model, mapping)
        # Cached Kraus pairs belong to the caller's old parameter graph.  The
        # rebuilt internal model must derive them lazily from bucket aliases.
        if hasattr(rebound.noise_model, "_kraus_cache"):
            rebound.noise_model._kraus_cache = {}
    return rebound


def _walk_replicated_parameters(value, *, name: str, component="real"):
    """Yield replicated circuit/noise candidate leaves, never initial state."""

    if isinstance(value, torch.Tensor):
        yield _ParameterEntry(name, value, component)
        return
    if isinstance(value, _Pair):
        yield from _walk_replicated_parameters(value.real, name=f"{name}.real", component="real")
        yield from _walk_replicated_parameters(value.imag, name=f"{name}.imag", component="imag")
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _walk_replicated_parameters(item, name=f"{name}[{index}]")
        return
    if isinstance(value, dict):
        for key in sorted(value, key=str):
            yield from _walk_replicated_parameters(value[key], name=f"{name}.{key}")
        return
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            # Dataclass-backed noise models retain derived runtime state (for
            # example ``NoiseModel._kraus_cache``).  Those tensors are not
            # caller-owned parameters and may already be stale from a prior
            # forward pass, so never make them part of the replicated schema.
            if field.name.startswith("_"):
                continue
            yield from _walk_replicated_parameters(getattr(value, field.name), name=f"{name}.{field.name}")


def _replicated_parameter_entries(circuit) -> tuple[_ParameterEntry, ...]:
    """Collect first-use ordered circuit/noise/Stinespring candidates once."""

    from ...ir import circuit_instructions, instruction_parameter

    entries = []
    seen = set()
    for index, instruction in enumerate(circuit_instructions(circuit)):
        for entry in _walk_replicated_parameters(
            instruction_parameter(instruction), name=f"circuit[{index}].parameter"
        ):
            if id(entry.value) not in seen:
                seen.add(id(entry.value))
                entries.append(entry)
    if hasattr(circuit, "noise_model"):
        for entry in _walk_replicated_parameters(circuit.noise_model, name="noise_model"):
            if id(entry.value) not in seen:
                seen.add(id(entry.value))
                entries.append(entry)
    return tuple(entries)


def _bind_replicated_gradient_bucket(circuit, *, communicator):
    """Preflight and bind the private native-engine replicated bucket."""

    candidates = _replicated_parameter_entries(circuit)
    aliases = _bucket_parameters(candidates, communicator=communicator)
    if not aliases:
        return circuit
    entries = tuple(entry for entry in candidates if entry.value.requires_grad)
    return _bind_trainable_aliases(
        circuit, {id(entry.value): alias for entry, alias in zip(entries, aliases)}
    )


class _ReplicatedParameterFn(torch.autograd.Function):
    """Identity forward with one real global parameter-adjoint reduction."""

    @staticmethod
    def forward(ctx, value: torch.Tensor, communicator):
        ctx.communicator = communicator
        return value

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return ctx.communicator.all_reduce_sum_real(gradient), None


def replicated_parameter(value, *, communicator):
    """Mark a real trainable gate leaf as replicated across all ranks.

    This wrapper belongs at the gate parameter leaf, before the parameter is
    expanded into multiple matrix entries.  Reducing individual matrix parts
    would count a shared angle once per real/imaginary entry.
    """

    if not isinstance(value, torch.Tensor) or not value.requires_grad:
        return value
    if value.dtype != torch.float32 or torch.is_complex(value):
        raise TypeError("分布式 statevector 参数必须是实数 torch.float32")
    if communicator.world_size == 1:
        return value
    return _ReplicatedParameterFn.apply(value, communicator)


@dataclass(frozen=True)
class PureStateParam:
    """Unconstrained paired-real amplitudes for a normalized pure state."""

    real: torch.Tensor
    imag: torch.Tensor

    def _raw_pair(self) -> _Pair:
        return _Pair(self.real, self.imag)

    def parameters(self) -> tuple[torch.Tensor, ...]:
        return (self.real, self.imag)

    def normalized_pair(self) -> _Pair:
        """Return amplitudes normalized by one global real Euclidean norm."""

        pair = self._raw_pair()
        norm = torch.sqrt(pair.abs_sq().sum())
        if float(norm.detach().cpu()) == 0.0:
            raise ValueError("纯态参数的范数必须大于 0")
        return pair.div_real(norm)


@dataclass(frozen=True)
class DensityParam:
    """Paired-real factor whose normalized ``L L^H`` is a density matrix."""

    real: torch.Tensor
    imag: torch.Tensor

    def _raw_pair(self) -> _Pair:
        return _Pair(self.real, self.imag)

    def parameters(self) -> tuple[torch.Tensor, ...]:
        return (self.real, self.imag)

    def density_pair(self) -> _Pair:
        """Build a positive semidefinite, trace-one paired-real density matrix."""

        density = self._raw_pair().matmul(self._raw_pair().dagger())
        trace = torch.diagonal(density.real, dim1=-2, dim2=-1).sum()
        if float(trace.detach().cpu()) == 0.0:
            raise ValueError("密度矩阵因子的迹必须大于 0")
        return density.div_real(trace)


@dataclass(frozen=True)
class StinespringParam:
    """Raw paired-real square-channel Stinespring parameters.

    ``target_qubits`` defaults to the first ``log2(input_dim)`` logical
    qubits.  It may instead name any unique non-negative logical targets; the
    active distributed state validates the upper bound before planning.
    """

    input_dim: int
    output_dim: int
    environment_dim: int
    real: torch.Tensor
    imag: torch.Tensor
    target_qubits: tuple[int, ...] | None = None

    def __post_init__(self):
        dimensions = (self.input_dim, self.output_dim, self.environment_dim)
        if any(isinstance(value, bool) or not isinstance(value, Integral) for value in dimensions):
            raise TypeError("Stinespring 维度必须是正整数")
        if any(int(value) <= 0 for value in dimensions):
            raise ValueError("Stinespring 维度必须为正整数")
        if int(self.input_dim) != int(self.output_dim):
            raise ValueError("Stinespring 要求 input_dim == output_dim")
        qubits = int(math.log2(int(self.input_dim)))
        if (1 << qubits) != int(self.input_dim):
            raise ValueError("Stinespring input_dim 必须是 2 的幂")
        _Pair(self.real, self.imag)
        dimension = int(self.output_dim) * int(self.environment_dim)
        if tuple(self.real.shape) != (dimension, dimension):
            raise ValueError("Stinespring 原始参数 shape 必须为 (output_dim * environment_dim, output_dim * environment_dim)")
        targets = tuple(range(qubits)) if self.target_qubits is None else tuple(self.target_qubits)
        if len(targets) != qubits:
            raise ValueError("Stinespring target_qubits 数量必须等于 log2(input_dim)")
        if any(isinstance(target, bool) or not isinstance(target, Integral) or int(target) < 0 for target in targets):
            raise ValueError("Stinespring target_qubits 必须是互异非负整数")
        targets = tuple(int(target) for target in targets)
        if len(set(targets)) != len(targets):
            raise ValueError("Stinespring target_qubits 必须互异")
        object.__setattr__(self, "input_dim", int(self.input_dim))
        object.__setattr__(self, "output_dim", int(self.output_dim))
        object.__setattr__(self, "environment_dim", int(self.environment_dim))
        object.__setattr__(self, "target_qubits", targets)

    def parameters(self) -> tuple[torch.Tensor, ...]:
        return (self.real, self.imag)
