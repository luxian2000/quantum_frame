"""Deterministic communication primitives for distributed state shards."""

from __future__ import annotations

from collections import Counter
import time

import torch


# ``ProcessGroup::send`` 把 tag 绑定成 32 位 C++ int，超出即抛
# ``TypeError: send(): incompatible function arguments``。上层按
# (指令, 噪声规则, Kraus 项, partner mask) 组合出的 tag 会随门数线性膨胀，
# 因此在传输边界统一取模。取模基数是偶数，`exchange` 的 (2t, 2t+1)
# 实部/虚部配对在取模后仍然相邻且奇偶不变。
_TAG_MODULUS = 1 << 30


def _wrap_tag(tag: int) -> int:
    """Fold one logical tag into the transport's 32-bit signed tag space.

    分布式执行里每个 rank 按同一顺序发起同一组 P2P 交换，且每次交换在下一次
    之前完成等待，因此同一 (peer, tag) 上的消息按发起顺序匹配，取模造成的
    重复 tag 不会错配。
    """

    tag = int(tag)
    if tag < 0:
        raise ValueError("P2P tag 必须非负")
    return tag % _TAG_MODULUS


class _P2PWorkGroup:
    """One logical P2P operation backed by its send and receive works."""

    def __init__(self, communicator, works, *, on_complete=None):
        self._communicator = communicator
        self._works = tuple(works)
        self._on_complete = on_complete
        self._released = False

    def is_completed(self) -> bool:
        return all(bool(work.is_completed()) for work in self._works)

    def wait(self) -> None:
        started = time.perf_counter()
        first_error = None
        try:
            for work in self._works:
                try:
                    work.wait()
                except Exception as error:  # noqa: BLE001 - drain paired work first
                    if first_error is None:
                        first_error = error
        finally:
            if not self._released:
                for work in self._works:
                    self._communicator._release_work_handle(work)
                self._released = True
                self._communicator._p2p_wait_seconds += time.perf_counter() - started
        if first_error is not None:
            raise first_error
        if self._on_complete is not None:
            callback, self._on_complete = self._on_complete, None
            callback()


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
        self._communication_records = []
        self._work_handles = []
        self._p2p_wait_seconds = 0.0
        # The public forward-only simulator does not consult this setting.
        # Native-autograd callers opt in through the private primitive only.
        self._autograd_communication_mode = "baseline"

    @property
    def autograd_communication_mode(self) -> str:
        return self._autograd_communication_mode

    def set_autograd_communication_mode(self, mode: str) -> None:
        if mode not in {"baseline", "reuse", "overlap"}:
            raise ValueError("通信模式必须是 baseline、reuse 或 overlap")
        self._autograd_communication_mode = mode

    @property
    def communication_records(self):
        """Immutable real-transport evidence for distributed autograd probes."""

        return tuple(dict(record) for record in self._communication_records)

    @property
    def communication_counters(self):
        """Count real transport endpoints by dtype, peer, tag, and bytes.

        ``bytes`` is the logical payload volume observed at this rank, not a
        backend wire-byte estimate.  A P2P exchange counts one sent and one
        received component; root scatter/gather count all root chunks, while
        a non-root endpoint counts its one local chunk.
        """

        records = self._communication_records
        return {
            "dtype": dict(Counter(record["dtype"] for record in records)),
            "peer": dict(Counter(record["peer"] for record in records)),
            "tag": dict(Counter(record["tag"] for record in records)),
            "bytes": sum(record["bytes"] for record in records),
            "p2p_wait_ms": self._p2p_wait_seconds * 1000.0,
        }

    def clear_communication_records(self) -> None:
        self._communication_records.clear()
        self._p2p_wait_seconds = 0.0

    def register_work_handle(self, work) -> None:
        """Register an asynchronous transport handle until its owner releases it.

        Real gradient buckets use synchronous all-reduce and therefore leave
        this list empty.  P2P paths register every returned work object before
        waiting, which keeps probe evidence tied to actual communicator state
        rather than a fabricated completed-handle flag.
        """

        self._work_handles.append(work)

    def _release_work_handle(self, work) -> None:
        try:
            self._work_handles.remove(work)
        except ValueError:
            pass

    @property
    def work_handle_status(self):
        """Observed outstanding and incomplete asynchronous work handles."""

        unfinished = 0
        for work in self._work_handles:
            completed = getattr(work, "is_completed", None)
            if not callable(completed) or not bool(completed()):
                unfinished += 1
        return {
            "outstanding_work_handles": len(self._work_handles),
            "unfinished_work_handles": unfinished,
            "all_handles_complete": unfinished == 0,
        }

    @staticmethod
    def _require_real_float32(tensor) -> None:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("autograd 通信负载必须是 torch.Tensor")
        if tensor.dtype != torch.float32 or torch.is_complex(tensor):
            raise TypeError("autograd 通信负载必须是实数 torch.float32")

    def _record_real_transport(self, tensor, *, kind, peer=None, tag=None, copies=1):
        self._communication_records.append(
            {
                "kind": str(kind),
                "dtype": str(tensor.dtype),
                "peer": None if peer is None else int(peer),
                "tag": None if tag is None else int(tag),
                "bytes": int(tensor.numel() * tensor.element_size() * copies),
            }
        )

    def _require_process_group(self) -> None:
        if self.world_size <= 1:
            return
        if not self._dist.is_available() or not self._dist.is_initialized():
            raise RuntimeError(
                "world_size > 1 时必须先初始化 torch.distributed process group"
            )

    def _exchange_tensor_async(self, tensor, peer: int, tag: int, *, receive=None):
        self._require_process_group()
        # P2P receives require a contiguous destination on Gloo/HCCL.  Keep
        # contiguity at this transport boundary instead of forcing every
        # density-kernel output to materialize a full local copy.
        if receive is None:
            receive = torch.empty(
                tuple(tensor.shape), dtype=tensor.dtype, device=tensor.device
            )
        elif (
            not isinstance(receive, torch.Tensor)
            or receive.dtype != tensor.dtype
            or receive.device != tensor.device
            or tuple(receive.shape) != tuple(tensor.shape)
            or not receive.is_contiguous()
        ):
            raise ValueError("P2P receive buffer 必须与发送张量同 shape、dtype、device 且连续")
        send = tensor.contiguous()
        tag = _wrap_tag(tag)
        operations = [
            self._dist.P2POp(
                self._dist.isend,
                send,
                int(peer),
                group=self.group,
                tag=tag,
            ),
            self._dist.P2POp(
                self._dist.irecv,
                receive,
                int(peer),
                group=self.group,
                tag=tag,
            ),
        ]
        works = self._dist.batch_isend_irecv(operations)
        for work in works:
            self.register_work_handle(work)
        return receive, _P2PWorkGroup(self, works)

    def _exchange_tensor(self, tensor, peer: int, tag: int):
        receive, work = self._exchange_tensor_async(tensor, peer, tag)
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

    def exchange_real(self, tensor, *, peer: int, tag: int):
        """Exchange a paired-real autograd component using an explicit tag."""

        self._require_real_float32(tensor)
        peer = int(peer)
        if not 0 <= peer < self.world_size or peer == self.rank:
            raise ValueError(
                f"peer={peer} 必须是有效且非本 rank 的通信对端"
            )
        result = self._exchange_tensor(tensor, peer, int(tag))
        self._record_real_transport(
            tensor,
            kind="exchange",
            peer=peer,
            tag=tag,
            copies=2,
        )
        return result

    def exchange_real_async(self, tensor, *, peer: int, tag: int, receive=None):
        """Launch a real float32 P2P exchange into an optional reusable buffer.

        The returned work owns both underlying send and receive handles.  Its
        ``wait`` method releases them only after both have completed, which is
        essential when a paired-real buffer is eligible for reuse.
        """

        self._require_real_float32(tensor)
        peer = int(peer)
        if not 0 <= peer < self.world_size or peer == self.rank:
            raise ValueError(
                f"peer={peer} 必须是有效且非本 rank 的通信对端"
            )
        if receive is not None:
            self._require_real_float32(receive)
        result, work = self._exchange_tensor_async(
            tensor,
            peer,
            int(tag),
            receive=receive,
        )
        work._on_complete = lambda: self._record_real_transport(
            tensor,
            kind="exchange",
            peer=peer,
            tag=tag,
            copies=2,
        )
        return result, work

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

    def all_reduce_sum_real(self, tensor):
        """All-reduce an autograd transport tensor without complex kernels."""

        self._require_real_float32(tensor)
        if self.world_size == 1:
            result = tensor.clone()
        else:
            self._require_process_group()
            result = tensor.clone()
            self._dist.all_reduce(
                result,
                op=self._dist.ReduceOp.SUM,
                group=self.group,
            )
        self._record_real_transport(tensor, kind="all_reduce")
        return result

    def _all_gather_tensor(self, tensor):
        gathered = [
            torch.empty_like(tensor) for _ in range(self.world_size)
        ]
        self._dist.all_gather(
            gathered,
            tensor.contiguous(),
            group=self.group,
        )
        return gathered

    def all_gather(self, tensor):
        if self.world_size == 1:
            return [tensor.clone()]
        self._require_process_group()
        if torch.is_complex(tensor):
            real_parts = self._all_gather_tensor(
                torch.real(tensor).contiguous()
            )
            imag_parts = self._all_gather_tensor(
                torch.imag(tensor).contiguous()
            )
            return [
                torch.complex(real, imag).to(dtype=tensor.dtype)
                for real, imag in zip(real_parts, imag_parts)
            ]
        return self._all_gather_tensor(tensor)

    def all_gather_real(self, tensor):
        """All-gather a float32 autograd transport tensor."""

        self._require_real_float32(tensor)
        if self.world_size == 1:
            gathered = [tensor.clone()]
        else:
            self._require_process_group()
            gathered = self._all_gather_tensor(tensor)
        self._record_real_transport(
            tensor,
            kind="all_gather",
            copies=self.world_size,
        )
        return gathered

    def broadcast(self, tensor, *, root: int = 0):
        root = int(root)
        if not 0 <= root < self.world_size:
            raise ValueError(
                f"root={root} 必须位于 [0, {self.world_size})"
            )
        if self.world_size == 1:
            return tensor.clone()
        self._require_process_group()
        if torch.is_complex(tensor):
            real = torch.real(tensor).contiguous()
            imag = torch.imag(tensor).contiguous()
            self._dist.broadcast(real, src=root, group=self.group)
            self._dist.broadcast(imag, src=root, group=self.group)
            return torch.complex(real, imag).to(dtype=tensor.dtype)
        result = tensor.clone()
        self._dist.broadcast(result, src=root, group=self.group)
        return result

    def _gather_tensor(self, tensor, root: int):
        gathered = (
            [torch.empty_like(tensor) for _ in range(self.world_size)]
            if self.rank == root
            else None
        )
        self._dist.gather(
            tensor.contiguous(),
            gather_list=gathered,
            dst=root,
            group=self.group,
        )
        return gathered

    def gather_to_root(self, tensor, *, root: int = 0):
        root = int(root)
        if not 0 <= root < self.world_size:
            raise ValueError(
                f"root={root} 必须位于 [0, {self.world_size})"
            )
        if self.world_size == 1:
            return [tensor.clone()]
        self._require_process_group()
        if torch.is_complex(tensor):
            real_parts = self._gather_tensor(
                torch.real(tensor).contiguous(),
                root,
            )
            imag_parts = self._gather_tensor(
                torch.imag(tensor).contiguous(),
                root,
            )
            if self.rank != root:
                return None
            return [
                torch.complex(real, imag).to(dtype=tensor.dtype)
                for real, imag in zip(real_parts, imag_parts)
            ]
        return self._gather_tensor(tensor, root)

    def gather_to_root_real(self, tensor, *, root: int = 0):
        """Gather one float32 paired-real component to ``root``."""

        self._require_real_float32(tensor)
        root = int(root)
        if not 0 <= root < self.world_size:
            raise ValueError(f"root={root} 必须位于 [0, {self.world_size})")
        if self.world_size == 1:
            gathered = [tensor.clone()]
        else:
            self._require_process_group()
            gathered = self._gather_tensor(tensor, root)
        self._record_real_transport(
            tensor,
            kind="gather",
            peer=root,
            copies=self.world_size if self.rank == root else 1,
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
        if dtype in (torch.complex64, torch.complex128):
            real_dtype = (
                torch.float32 if dtype == torch.complex64 else torch.float64
            )
            real_tensors = (
                [torch.real(tensor).contiguous() for tensor in tensors]
                if self.rank == int(root)
                else None
            )
            imag_tensors = (
                [torch.imag(tensor).contiguous() for tensor in tensors]
                if self.rank == int(root)
                else None
            )
            real = self.scatter_from_root(
                real_tensors,
                root=root,
                shape=shape,
                dtype=real_dtype,
            )
            imag = self.scatter_from_root(
                imag_tensors,
                root=root,
                shape=shape,
                dtype=real_dtype,
            )
            return torch.complex(real, imag).to(dtype=dtype)
        receive = torch.empty(tuple(shape), dtype=dtype, device=self.device)
        self._dist.scatter(
            receive,
            scatter_list=tensors if self.rank == int(root) else None,
            src=int(root),
            group=self.group,
        )
        return receive

    def scatter_from_root_real(self, tensors, *, root: int = 0, shape=None):
        """Scatter one float32 paired-real component from ``root``."""

        root = int(root)
        if not 0 <= root < self.world_size:
            raise ValueError(f"root={root} 必须位于 [0, {self.world_size})")
        if shape is None:
            raise ValueError("autograd scatter 必须提供 local shape")
        if self.rank == root:
            if tensors is None or len(tensors) != self.world_size:
                raise ValueError("root 必须提供每个 rank 的 float32 scatter 分片")
            for tensor in tensors:
                self._require_real_float32(tensor)
                if tuple(tensor.shape) != tuple(shape):
                    raise ValueError("autograd scatter 分片 shape 与 local shape 不一致")
        if self.world_size == 1:
            if tensors is None:
                raise ValueError("单 rank scatter 需要一个输入张量")
            result = tensors[0].clone()
        else:
            self._require_process_group()
            result = torch.empty(
                tuple(shape),
                dtype=torch.float32,
                device=self.device,
            )
            self._dist.scatter(
                result,
                scatter_list=tensors if self.rank == root else None,
                src=root,
                group=self.group,
            )
        self._record_real_transport(
            result,
            kind="scatter",
            peer=root,
            copies=self.world_size if self.rank == root else 1,
        )
        return result
