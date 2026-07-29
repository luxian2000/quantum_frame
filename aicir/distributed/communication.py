"""Deterministic communication primitives for distributed state shards."""

from __future__ import annotations

import torch


class _Communicator:
    """Small process-group wrapper with an NPU-safe complex transport."""

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        device,
        group=None,
        supports_complex: bool = True,
        dist_module=None,
    ):
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.device = torch.device(device)
        self.group = group
        self.supports_complex = bool(supports_complex)
        self._dist = torch.distributed if dist_module is None else dist_module

    def _require_process_group(self) -> None:
        if self.world_size <= 1:
            return
        if not self._dist.is_available() or not self._dist.is_initialized():
            raise RuntimeError(
                "world_size > 1 时必须先初始化 torch.distributed process group"
            )

    def _exchange_tensor(self, tensor, peer: int, tag: int):
        self._require_process_group()
        receive = torch.empty_like(tensor)
        operations = [
            self._dist.P2POp(
                self._dist.isend,
                tensor.contiguous(),
                int(peer),
                group=self.group,
                tag=int(tag),
            ),
            self._dist.P2POp(
                self._dist.irecv,
                receive,
                int(peer),
                group=self.group,
                tag=int(tag),
            ),
        ]
        for work in self._dist.batch_isend_irecv(operations):
            work.wait()
        return receive

    def exchange(self, tensor, *, peer: int, tag: int):
        if not 0 <= int(peer) < self.world_size:
            raise ValueError(
                f"peer={peer} 必须位于 [0, {self.world_size})"
            )
        if torch.is_complex(tensor) and not self.supports_complex:
            real = self._exchange_tensor(
                torch.real(tensor).contiguous(),
                int(peer),
                int(tag) * 2,
            )
            imag = self._exchange_tensor(
                torch.imag(tensor).contiguous(),
                int(peer),
                int(tag) * 2 + 1,
            )
            return torch.complex(real, imag).to(dtype=tensor.dtype)
        return self._exchange_tensor(tensor, int(peer), int(tag))

    def all_reduce_sum(self, tensor):
        if self.world_size == 1:
            return tensor.clone()
        self._require_process_group()
        if torch.is_complex(tensor) and not self.supports_complex:
            real = torch.real(tensor).contiguous()
            imag = torch.imag(tensor).contiguous()
            self._dist.all_reduce(
                real,
                op=self._dist.ReduceOp.SUM,
                group=self.group,
            )
            self._dist.all_reduce(
                imag,
                op=self._dist.ReduceOp.SUM,
                group=self.group,
            )
            return torch.complex(real, imag).to(dtype=tensor.dtype)
        result = tensor.clone()
        self._dist.all_reduce(
            result,
            op=self._dist.ReduceOp.SUM,
            group=self.group,
        )
        return result

    def gather_to_root(self, tensor, *, root: int = 0):
        if not 0 <= int(root) < self.world_size:
            raise ValueError(
                f"root={root} 必须位于 [0, {self.world_size})"
            )
        if self.world_size == 1:
            return [tensor.clone()]
        self._require_process_group()
        gathered = (
            [torch.empty_like(tensor) for _ in range(self.world_size)]
            if self.rank == int(root)
            else None
        )
        self._dist.gather(
            tensor.contiguous(),
            gather_list=gathered,
            dst=int(root),
            group=self.group,
        )
        return gathered

    def scatter_from_root(self, tensors, *, root: int = 0, shape=None, dtype=None):
        if not 0 <= int(root) < self.world_size:
            raise ValueError(
                f"root={root} 必须位于 [0, {self.world_size})"
            )
        if self.world_size == 1:
            if tensors is None or len(tensors) != 1:
                raise ValueError("单 rank scatter 需要一个输入张量")
            return tensors[0].clone()
        self._require_process_group()
        if shape is None or dtype is None:
            raise ValueError("多 rank scatter 必须提供 shape 和 dtype")
        receive = torch.empty(tuple(shape), dtype=dtype, device=self.device)
        self._dist.scatter(
            receive,
            scatter_list=tensors if self.rank == int(root) else None,
            src=int(root),
            group=self.group,
        )
        return receive
