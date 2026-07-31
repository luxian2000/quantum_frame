"""Gate planning and streaming sharded-statevector kernels."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
import torch

from ..core.gates import (
    _apply_local_matrix_to_state,
    _cast_local_matrix,
    _gate_local_matrix,
    _unitary_parameter_matrix,
)
from ..gates import canonical_gate_name
from ..ir import as_instruction, instruction_controls, instruction_name, instruction_n_qubits, instruction_parameter, instruction_qubits, instruction_with_parameter
from .autograd._parameters import replicated_parameter
from .autograd._pair import _Pair
from .state import DistState


@dataclass(frozen=True)
class _GatePlan:
    instruction_index: int
    local_matrix: object
    logical_axes: tuple[int, ...]
    storage_axes: tuple[int, ...]
    distributed_storage_axes: tuple[int, ...]
    local_storage_axes: tuple[int, ...]
    partner_masks: tuple[int, ...]
    distributed_axes: int

    def partner_for(self, *, rank: int, mask: int) -> int:
        if int(mask) not in self.partner_masks:
            raise ValueError(f"mask={mask} 不属于该门的 partner_masks")
        return int(rank) ^ int(mask)

    def tag(self, mask: int) -> int:
        return int(self.instruction_index) * 4096 + int(mask)


class _AutogradExecutionContext:
    """Bounded cache for one native-autograd circuit execution.

    Reuse this object for every planner participating in one circuit; a
    physical leaf is then wrapped once and reduced once after all uses.
    Contexts are intentionally caller-owned, never process-global.
    """

    def __init__(self):
        self.parameter_cache = {}


def _trainable_pair_matrix(instruction, gate_type):
    """Build parameterized matrices directly as independent real tensors."""

    parameter = instruction_parameter(instruction)
    values = tuple(parameter) if isinstance(parameter, (list, tuple)) else (parameter,)
    if not any(isinstance(value, torch.Tensor) and value.requires_grad for value in values):
        return None
    ref = next(value for value in values if isinstance(value, torch.Tensor))
    zero, one = torch.zeros((), dtype=torch.float32, device=ref.device), torch.ones((), dtype=torch.float32, device=ref.device)
    def mat(real, imag): return _Pair(torch.stack([torch.stack(row) for row in real]), torch.stack([torch.stack(row) for row in imag]))
    t = values[0]
    c, s = torch.cos(t / 2.0), torch.sin(t / 2.0)
    if gate_type in {"rx", "ry", "rz", "crx", "cry", "crz"}:
        if gate_type.endswith("x"):
            base = mat([[c, zero], [zero, c]], [[zero, -s], [-s, zero]])
        elif gate_type.endswith("y"):
            base = mat([[c, -s], [s, c]], [[zero, zero], [zero, zero]])
        else:
            base = mat([[c, zero], [zero, c]], [[ -s, zero], [zero, s]])
        if instruction_controls(instruction):
            # Controls precede targets in the established planner axis order.
            return _Pair(
                torch.stack([torch.stack([one if row == col and row < 2 else base.real[row - 2, col - 2] if row >= 2 and col >= 2 else zero for col in range(4)]) for row in range(4)]),
                torch.stack([torch.stack([base.imag[row - 2, col - 2] if row >= 2 and col >= 2 else zero for col in range(4)]) for row in range(4)]),
            )
        return base
    if gate_type == "rzz":
        return mat([[c,zero,zero,zero],[zero,c,zero,zero],[zero,zero,c,zero],[zero,zero,zero,c]], [[-s,zero,zero,zero],[zero,s,zero,zero],[zero,zero,s,zero],[zero,zero,zero,-s]])
    if gate_type == "rxx":
        return mat([[c,zero,zero,zero],[zero,c,zero,zero],[zero,zero,c,zero],[zero,zero,zero,c]], [[zero,zero,zero,-s],[zero,zero,-s,zero],[zero,-s,zero,zero],[-s,zero,zero,zero]])
    if gate_type == "u2":
        phi, lam = values; q = one / math.sqrt(2.0)
        return mat([[q,-q*torch.cos(lam)],[q*torch.cos(phi),q*torch.cos(phi+lam)]], [[zero,-q*torch.sin(lam)],[q*torch.sin(phi),q*torch.sin(phi+lam)]])
    if gate_type == "u3":
        theta, phi, lam = values; c, s = torch.cos(theta/2), torch.sin(theta/2)
        return mat([[c,-s*torch.cos(lam)],[s*torch.cos(phi),c*torch.cos(phi+lam)]], [[zero,-s*torch.sin(lam)],[s*torch.sin(phi),c*torch.sin(phi+lam)]])
    return None


class _GatePlanner:
    """Resolve a typed instruction into storage axes and rank partners."""

    def __init__(self, backend, layout, n_qubits: int, *, execution_context=None):
        self._backend = backend
        self._layout = layout
        self._n_qubits = int(n_qubits)
        # A planner is one bounded circuit-planning context.  Sharing the
        # wrapper here makes a repeated physical tensor one autograd node, so
        # its accumulated adjoint crosses the collective exactly once.
        self._execution_context = execution_context
        if layout.n_qubits != self._n_qubits:
            raise ValueError("layout 与 n_qubits 不一致")

    def plan(self, gate, instruction_index: int) -> _GatePlan:
        instruction = as_instruction(gate)
        gate_type = canonical_gate_name(instruction_name(instruction))
        parameter = instruction_parameter(instruction)
        if parameter is not None and gate_type != "unitary":
            if getattr(parameter, "requires_grad", False) or any(
                getattr(value, "requires_grad", False)
                for value in (parameter if isinstance(parameter, (tuple, list)) else ())
            ):
                if self._execution_context is None:
                    raise RuntimeError(
                        "trainable distributed gate planning requires an explicit _AutogradExecutionContext"
                    )
            def _wrap(value):
                if isinstance(value, tuple):
                    return tuple(_wrap(item) for item in value)
                if isinstance(value, list):
                    return [_wrap(item) for item in value]
                if not getattr(value, "requires_grad", False):
                    return value
                key = id(value)
                wrapped = self._execution_context.parameter_cache.get(key)
                if wrapped is None:
                    wrapped = replicated_parameter(
                        value,
                        communicator=self._backend.communicator,
                    )
                    self._execution_context.parameter_cache[key] = wrapped
                return wrapped

            instruction = instruction_with_parameter(instruction, _wrap(parameter))
        pair_matrix = _trainable_pair_matrix(instruction, gate_type)
        if pair_matrix is not None:
            return self.plan_matrix(
                pair_matrix,
                tuple(instruction_controls(instruction)) + tuple(instruction_qubits(instruction)),
                instruction_index=instruction_index,
            )
        if gate_type == "unitary":
            parameter = instruction_parameter(instruction)
            if isinstance(parameter, _Pair) and (
                parameter.real.requires_grad or parameter.imag.requires_grad
            ):
                if self._execution_context is None:
                    raise RuntimeError(
                        "trainable distributed gate planning requires an explicit _AutogradExecutionContext"
                    )
                def _wrap_component(value):
                    key = id(value)
                    cached = self._execution_context.parameter_cache.get(key)
                    if cached is None:
                        cached = replicated_parameter(value, communicator=self._backend.communicator)
                        self._execution_context.parameter_cache[key] = cached
                    return cached
                parameter = _Pair(
                    _wrap_component(parameter.real), _wrap_component(parameter.imag)
                )
            if isinstance(parameter, torch.Tensor) and parameter.requires_grad:
                if torch.is_complex(parameter):
                    raise TypeError(
                        "原生 distributed autograd 不接受 requires_grad complex unitary；"
                        "请提供 _Pair(real, imag) 或在 CPU 参考路径构造该矩阵"
                    )
                raise TypeError("trainable unitary 必须以 _Pair(real, imag) 提供")
            matrix = parameter if isinstance(parameter, _Pair) else _unitary_parameter_matrix(parameter, self._backend)
            shape = tuple(int(dim) for dim in (matrix.real.shape if isinstance(matrix, _Pair) else matrix.shape))
            if len(shape) != 2 or shape[0] != shape[1] or shape[0] <= 0:
                raise ValueError("unitary 门参数必须是正方阵")
            inferred = int(round(math.log2(shape[0])))
            if (1 << inferred) != shape[0]:
                raise ValueError("unitary 门矩阵维度必须是 2 的幂")
            gate_qubits = int(instruction_n_qubits(instruction, inferred))
            if gate_qubits != inferred:
                raise ValueError("unitary 门的 n_qubits 与矩阵维度不一致")
            return self.plan_matrix(
                matrix,
                tuple(range(gate_qubits)),
                instruction_index=instruction_index,
            )
        local, logical_axes, cache_key = _gate_local_matrix(
            instruction,
            gate_type,
            self._backend,
        )
        if local is None or logical_axes is None:
            raise ValueError(
                f"指令 {gate_type!r} 没有可用于分布式执行的局部门矩阵"
            )

        matrix = _cast_local_matrix(
            self._backend,
            local,
            cache_key=cache_key,
        )
        return self.plan_matrix(
            matrix,
            logical_axes,
            instruction_index=instruction_index,
        )

    def plan_matrix(
        self,
        local_matrix,
        logical_axes,
        *,
        instruction_index: int,
    ) -> _GatePlan:
        logical_axes = tuple(int(axis) for axis in logical_axes)
        if len(set(logical_axes)) != len(logical_axes):
            raise ValueError("局部矩阵的逻辑量子比特不能重复")
        if any(
            logical < 0 or logical >= self._n_qubits
            for logical in logical_axes
        ):
            raise ValueError("局部矩阵的逻辑量子比特超出范围")
        storage_axes = tuple(
            self._layout.logical_to_storage[logical]
            for logical in logical_axes
        )
        distributed_storage_axes = tuple(
            sorted(
                storage
                for storage in storage_axes
                if storage < self._layout.distributed_axes
            )
        )
        local_storage_axes = tuple(
            storage
            for storage in storage_axes
            if storage >= self._layout.distributed_axes
        )

        rank_bit_masks = tuple(
            1 << (self._layout.distributed_axes - 1 - storage)
            for storage in distributed_storage_axes
        )
        partner_masks = []
        for count in range(1, len(rank_bit_masks) + 1):
            for combination in itertools.combinations(rank_bit_masks, count):
                mask = 0
                for bit in combination:
                    mask ^= bit
                partner_masks.append(mask)

        if isinstance(local_matrix, torch.Tensor) and torch.is_complex(local_matrix) and local_matrix.requires_grad:
            raise TypeError("plan_matrix 不接受 requires_grad complex matrix；请提供 _Pair(real, imag)")
        matrix = (
            local_matrix
            if isinstance(local_matrix, _Pair)
            else self._backend.cast_local_matrix(local_matrix)
        )
        dimension = 1 << len(logical_axes)
        matrix_shape = matrix.real.shape if isinstance(matrix, _Pair) else matrix.shape
        if tuple(int(axis) for axis in matrix_shape) != (
            dimension,
            dimension,
        ):
            raise ValueError(
                f"局部门矩阵形状必须是 ({dimension}, {dimension})"
            )
        return _GatePlan(
            instruction_index=int(instruction_index),
            local_matrix=matrix,
            logical_axes=logical_axes,
            storage_axes=storage_axes,
            distributed_storage_axes=distributed_storage_axes,
            local_storage_axes=local_storage_axes,
            partner_masks=tuple(sorted(partner_masks)),
            distributed_axes=self._layout.distributed_axes,
        )


def _rank_axis_bit(rank: int, storage_axis: int, distributed_axes: int) -> int:
    shift = int(distributed_axes) - 1 - int(storage_axis)
    return (int(rank) >> shift) & 1


def _matrix_block(
    matrix,
    storage_axes,
    *,
    output_rank: int,
    input_rank: int,
    distributed_axes: int,
):
    """Fix distributed output/input bits and retain local target axes."""

    storage_axes = tuple(int(axis) for axis in storage_axes)
    arity = len(storage_axes)
    shaped = matrix.reshape((2,) * (2 * arity))
    selectors = [slice(None)] * (2 * arity)
    local_count = 0
    for position, storage_axis in enumerate(storage_axes):
        if storage_axis < distributed_axes:
            selectors[position] = _rank_axis_bit(
                output_rank,
                storage_axis,
                distributed_axes,
            )
            selectors[arity + position] = _rank_axis_bit(
                input_rank,
                storage_axis,
                distributed_axes,
            )
        else:
            local_count += 1
    block = shaped[tuple(selectors)]
    local_dimension = 1 << local_count
    return block.reshape(local_dimension, local_dimension)


class _VectorKernel:
    """Apply local-matrix blocks to one statevector shard."""

    def __init__(self, backend):
        self._backend = backend

    def _apply_source(self, source, plan: _GatePlan, source_rank: int):
        block = _matrix_block(
            plan.local_matrix,
            plan.storage_axes,
            output_rank=self._backend.rank,
            input_rank=source_rank,
            distributed_axes=plan.distributed_axes,
        )
        local_axes = tuple(
            storage - plan.distributed_axes
            for storage in plan.storage_axes
            if storage >= plan.distributed_axes
        )
        if not local_axes:
            return self._backend.mul(source, block.reshape(()))
        local_n_qubits = int(source.shape[0]).bit_length() - 1
        return _apply_local_matrix_to_state(
            source,
            block,
            local_axes,
            local_n_qubits,
            self._backend,
        )

    def apply(self, state: DistState, plan: _GatePlan) -> DistState:
        if state.kind != "vector":
            raise TypeError("_VectorKernel 仅接受 vector DistState")

        output = self._apply_source(
            state.local_data,
            plan,
            self._backend.rank,
        )
        for mask in plan.partner_masks:
            peer = plan.partner_for(rank=self._backend.rank, mask=mask)
            incoming = self._backend.communicator.exchange(
                state.local_data,
                peer=peer,
                tag=plan.tag(mask),
            )
            contribution = self._apply_source(incoming, plan, peer)
            output = self._backend.add(output, contribution)

        return DistState.from_local(
            output,
            spec=state.spec,
            backend=self._backend,
            bit_order=state.bit_order,
        )
