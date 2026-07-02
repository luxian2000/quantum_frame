"""
aicir/backends/npu_backend.py

Ascend NPU backend built on top of PyTorch + torch_npu.

Design goals:
- Reuse GPUBackend math kernels to keep behavior consistent.
- Prefer NPU automatically when available.
- Allow graceful CPU fallback for environments without torch_npu.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

import torch

from .gpu_backend import GPUBackend

try:
    import torch_npu  # noqa: F401
    _HAS_TORCH_NPU = hasattr(torch, "npu")
except ImportError:
    _HAS_TORCH_NPU = False


def is_npu_available() -> bool:
    """Return True if torch_npu is importable and runtime NPU is available."""
    return bool(_HAS_TORCH_NPU and hasattr(torch, "npu") and torch.npu.is_available())


@dataclass(frozen=True)
class NPURuntimeContext:
    """Runtime context inferred from distributed environment variables."""

    world_size: int
    rank: int
    local_rank: int
    distributed: bool
    process_group_initialized: bool = False
    process_group_backend: str | None = None


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got: {raw!r}") from exc


def npu_runtime_context_from_env() -> NPURuntimeContext:
    """Parse WORLD_SIZE/RANK/LOCAL_RANK from env, with safe defaults for single-process runs."""
    world_size = max(1, _env_int("WORLD_SIZE", 1))
    rank = max(0, _env_int("RANK", 0))
    local_rank = max(0, _env_int("LOCAL_RANK", 0))
    return NPURuntimeContext(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        distributed=world_size > 1,
    )


def _has_complete_distributed_env() -> bool:
    return all(os.environ.get(name) for name in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK"))


def _np_is_complex(x) -> bool:
    return isinstance(x, torch.Tensor) and torch.is_complex(x)


def _np_conj_T(x: torch.Tensor) -> torch.Tensor:
    """Conjugate transpose, NPU-safe (no torch.conj on complex64)."""
    xt = torch.transpose(x, -2, -1)
    if _np_is_complex(xt):
        return torch.complex(torch.real(xt), -torch.imag(xt))
    return xt


class _NpuMatmulFn(torch.autograd.Function):
    """Complex matmul on NPU with an autograd-safe custom backward.

    NPU has no complex64 kernels, so the forward decomposes into real matmuls
    (via ``NPUBackend._matmul_forward``). The catch is autograd *backward*: if
    that decomposition is recorded in the traced graph, the engine accumulates
    complex gradients with ``aclnnAdd``—which NPU also lacks. By computing both
    forward and backward *inside* this Function (untraced), the traced graph
    sees only one node with complex in/out used linearly, so no complex-tensor
    gradient accumulation ever happens. Gradients match native complex matmul.
    """

    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return NPUBackend._matmul_forward(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_a = grad_b = None
        if ctx.needs_input_grad[0]:
            ga = NPUBackend._matmul_forward(grad_output, _np_conj_T(b))
            grad_a = ga if _np_is_complex(a) else (torch.real(ga) if _np_is_complex(ga) else ga)
        if ctx.needs_input_grad[1]:
            gb = NPUBackend._matmul_forward(_np_conj_T(a), grad_output)
            grad_b = gb if _np_is_complex(b) else (torch.real(gb) if _np_is_complex(gb) else gb)
        return grad_a, grad_b


class _NpuExpectationFn(torch.autograd.Function):
    """``Re⟨ψ|O|ψ⟩`` on NPU with an autograd-safe custom backward.

    The state appears twice in ``s^H O s`` (bra and ket); in a plain graph its
    gradient would be accumulated (complex add → unsupported on NPU). Wrapping
    the whole expectation keeps the state consumed exactly once in the traced
    graph, so its gradient is assigned, not accumulated. For real loss
    ``E = Re(s^H O s)``, ``dE/ds`` (PyTorch convention) is ``(O + O^H) s``.
    """

    @staticmethod
    def forward(ctx, state, operator):
        ctx.save_for_backward(state, operator)
        s = state.reshape(-1, 1)
        hs = NPUBackend._matmul_forward(operator, s)
        val = NPUBackend._matmul_forward(_np_conj_T(s), hs)
        return torch.real(val).reshape(())

    @staticmethod
    def backward(ctx, grad_output):
        state, operator = ctx.saved_tensors
        s = state.reshape(-1, 1)
        hs = NPUBackend._matmul_forward(operator, s)            # O s
        hhs = NPUBackend._matmul_forward(_np_conj_T(operator), s)  # O^H s
        go = grad_output  # real scalar
        # Combine via real parts to avoid a complex64 add on NPU.
        grad_real = (torch.real(hs) + torch.real(hhs)) * go
        grad_imag = (torch.imag(hs) + torch.imag(hhs)) * go
        grad_state = torch.complex(grad_real, grad_imag).reshape(state.shape)
        return grad_state, None


def _pauli_signs_npu(basis_indices: torch.Tensor, sign_mask: int) -> torch.Tensor | None:
    """(-1)^{popcount(basis_index & sign_mask)}，float32，NPU 安全。"""
    if not sign_mask:
        return None
    parity = torch.zeros_like(basis_indices, dtype=torch.bool)
    mask, bit = int(sign_mask), 0
    while mask:
        if mask & 1:
            parity = torch.logical_xor(parity, ((basis_indices >> bit) & 1).bool())
        mask >>= 1
        bit += 1
    ones = torch.ones_like(basis_indices, dtype=torch.float32)
    return torch.where(parity, -ones, ones)


def _pauli_apply_npu(
    state_re: torch.Tensor,
    state_im: torch.Tensor,
    basis_indices: torch.Tensor,
    flip_mask: int,
    sign_mask: int,
    y_phase: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """计算 (P_k |ψ⟩) 的实/虚部（全程 float32，无 complex64 节点）。"""
    if flip_mask:
        idx = torch.bitwise_xor(basis_indices, flip_mask)
        mr, mi = state_re.index_select(0, idx), state_im.index_select(0, idx)
    else:
        mr, mi = state_re, state_im
    if sign_mask:
        s = _pauli_signs_npu(basis_indices, sign_mask)
        mr, mi = mr * s, mi * s
    # 乘以 i^y_phase
    if y_phase == 0:
        return mr, mi
    if y_phase == 1:   # ×i
        return -mi, mr
    if y_phase == 2:   # ×(−1)
        return -mr, -mi
    return mi, -mr     # ×(−i)


class _NpuHamiltonianExpectationFn(torch.autograd.Function):
    """Re⟨ψ|H|ψ⟩（Pauli 字符串求和 Hamiltonian）的 NPU 安全自动微分包装。

    核心思路：state 在自动微分图中只出现一次（作为本 Function 的输入），
    1313 次 Pauli 项的梯度累加全部在本 Function 的 backward 内以 float32 完成，
    最终一次性组装成 complex64 梯度返回——没有跨 autograd 节点的 complex64 add，
    彻底绕开 Ascend 缺失的 aclnnAdd(DT_COMPLEX64)。

    backward 公式：grad_state = 2 · go · (H |ψ⟩)，其中
        (H|ψ⟩)[m] = Σ_k c_k (P_k|ψ⟩)[m]
    实/虚部分别在 float32 累加。
    """

    @staticmethod
    def forward(ctx, state, basis_indices, pauli_cache):
        ctx.save_for_backward(state, basis_indices)
        ctx.pauli_cache = pauli_cache
        s_re = torch.real(state)
        s_im = torch.imag(state)
        energy = torch.zeros((), dtype=torch.float32, device=state.device)
        for flip_mask, sign_mask, y_phase, c_re, c_im in pauli_cache:
            if flip_mask:
                idx = torch.bitwise_xor(basis_indices, flip_mask)
                m_re = s_re.index_select(0, idx)
                m_im = s_im.index_select(0, idx)
            else:
                m_re, m_im = s_re, s_im
            ov_re = m_re * s_re + m_im * s_im
            ov_im = m_re * s_im - m_im * s_re
            if sign_mask:
                sg = _pauli_signs_npu(basis_indices, sign_mask)
                ov_re, ov_im = ov_re * sg, ov_im * sg
            if y_phase == 0:
                t_re, t_im = ov_re.sum(), ov_im.sum()
            elif y_phase == 1:
                t_re, t_im = -ov_im.sum(), ov_re.sum()
            elif y_phase == 2:
                t_re, t_im = -ov_re.sum(), -ov_im.sum()
            else:
                t_re, t_im = ov_im.sum(), -ov_re.sum()
            energy = energy + c_re * t_re - c_im * t_im
        return energy

    @staticmethod
    def backward(ctx, go):
        state, basis_indices = ctx.saved_tensors
        s_re = torch.real(state)
        s_im = torch.imag(state)
        dim = state.numel()
        dev = state.device
        g_re = torch.zeros(dim, dtype=torch.float32, device=dev)
        g_im = torch.zeros(dim, dtype=torch.float32, device=dev)
        go2 = go * 2.0  # d(Re⟨ψ|H|ψ⟩)/dψ = 2·Re(H|ψ⟩), PyTorch 复数梯度惯例
        for flip_mask, sign_mask, y_phase, c_re, c_im in ctx.pauli_cache:
            pk_re, pk_im = _pauli_apply_npu(s_re, s_im, basis_indices, flip_mask, sign_mask, y_phase)
            ck_pk_re = c_re * pk_re - c_im * pk_im
            ck_pk_im = c_re * pk_im + c_im * pk_re
            g_re = g_re + go2 * ck_pk_re   # float32 累加，无 complex64 add
            g_im = g_im + go2 * ck_pk_im
        # 一次性组装 complex64——不是累加，是赋值
        return torch.complex(g_re, g_im).reshape(state.shape), None, None


class _NpuLocalGateApplyFn(torch.autograd.Function):
    """局部量子门（gather→matmul→scatter）的 NPU 自动微分安全包装。

    ``_apply_local_matrix_to_state_flat`` 对 NPU 复数态做：
        gathered = complex(real(flat)[idx], imag(flat)[idx])  # flat 用两次
        updated  = matmul(local_matrix, gathered)
        out_re[idx] = real(updated)                          # updated 用两次
        out_im[idx] = imag(updated)

    flat 和 updated 各在图中出现两次（.real + .imag），backward 需对 complex64
    张量累加两次梯度 → aclnnAdd(DT_COMPLEX64) 报错。

    本 Function 将整个流程包成一次原子操作：flat 和 local_matrix 各只出现一次，
    所有 gather/matmul/scatter 全程以 float32 实/虚部拆解完成，backward 最后
    一次性组装 complex64 梯度——无跨节点的 complex64 累加。
    """

    @staticmethod
    def _mat_re_im(local_matrix):
        """拆分门矩阵实/虚部；实数门（如 RY/H/X）矩阵本身是 float32，虚部为 0。"""
        if torch.is_complex(local_matrix):
            return torch.real(local_matrix), torch.imag(local_matrix)
        return local_matrix, None

    @staticmethod
    def forward(ctx, flat, local_matrix, indices):
        # flat: (dim,) complex64; local_matrix: (k,k) complex64 或实数 float32; indices: (k, base) long
        ctx.save_for_backward(flat, local_matrix, indices)
        idx = indices.reshape(-1)

        s_re = torch.real(flat)
        s_im = torch.imag(flat)
        g_re = s_re[idx].reshape(indices.shape)
        g_im = s_im[idx].reshape(indices.shape)

        m_re, m_im = _NpuLocalGateApplyFn._mat_re_im(local_matrix)
        if m_im is None:
            upd_re = torch.matmul(m_re, g_re)
            upd_im = torch.matmul(m_re, g_im)
        else:
            upd_re = torch.matmul(m_re, g_re) - torch.matmul(m_im, g_im)
            upd_im = torch.matmul(m_re, g_im) + torch.matmul(m_im, g_re)

        dim = flat.shape[0]
        out_re = torch.empty(dim, dtype=torch.float32, device=flat.device)
        out_im = torch.empty(dim, dtype=torch.float32, device=flat.device)
        out_re[idx] = upd_re.reshape(-1)
        out_im[idx] = upd_im.reshape(-1)
        return torch.complex(out_re, out_im)

    @staticmethod
    def backward(ctx, grad_out):
        flat, local_matrix, indices = ctx.saved_tensors
        idx = indices.reshape(-1)

        # grad_out 是 complex64，但 .real/.imag 是 float32 提取（在 backward 内，无追踪）
        grad_upd_re = torch.real(grad_out)[idx].reshape(indices.shape)
        grad_upd_im = torch.imag(grad_out)[idx].reshape(indices.shape)

        # 重新 gather 用于 grad_local 计算（saved flat，不追踪）
        s_re = torch.real(flat)
        s_im = torch.imag(flat)
        g_re = s_re[idx].reshape(indices.shape)
        g_im = s_im[idx].reshape(indices.shape)

        m_re, m_im = _NpuLocalGateApplyFn._mat_re_im(local_matrix)

        # ∂L/∂m_re = grad_upd_re @ g_re^T + grad_upd_im @ g_im^T
        # ∂L/∂m_im = −grad_upd_re @ g_im^T + grad_upd_im @ g_re^T
        grad_m_re = (
            torch.matmul(grad_upd_re, g_re.t()) + torch.matmul(grad_upd_im, g_im.t())
        )
        if m_im is None:
            # 实数门：梯度也为实数（虚部分量不存在）
            grad_local = grad_m_re
            grad_g_re = torch.matmul(m_re.t(), grad_upd_re)
            grad_g_im = torch.matmul(m_re.t(), grad_upd_im)
        else:
            grad_m_im = (
                -torch.matmul(grad_upd_re, g_im.t()) + torch.matmul(grad_upd_im, g_re.t())
            )
            grad_local = torch.complex(grad_m_re, grad_m_im)  # 一次构造，非累加

            # ∂L/∂g_re = m_re^T @ grad_upd_re + m_im^T @ grad_upd_im
            # ∂L/∂g_im = −m_im^T @ grad_upd_re + m_re^T @ grad_upd_im
            grad_g_re = torch.matmul(m_re.t(), grad_upd_re) + torch.matmul(m_im.t(), grad_upd_im)
            grad_g_im = (
                torch.matmul(-m_im.t(), grad_upd_re) + torch.matmul(m_re.t(), grad_upd_im)
            )

        # scatter 回 flat 空间（float32 scatter_add，无 complex64 add）
        dim = flat.shape[0]
        grad_flat_re = torch.zeros(dim, dtype=torch.float32, device=flat.device)
        grad_flat_im = torch.zeros(dim, dtype=torch.float32, device=flat.device)
        grad_flat_re.scatter_add_(0, idx, grad_g_re.reshape(-1))
        grad_flat_im.scatter_add_(0, idx, grad_g_im.reshape(-1))

        grad_flat = torch.complex(grad_flat_re, grad_flat_im)  # 一次构造，非累加
        return grad_flat, grad_local, None  # indices 无梯度


class NPUBackend(GPUBackend):
    """NPU-first backend for Ascend devices, compatible with GPUBackend API."""

    def __init__(self, dtype=None, device=None, fallback_to_cpu: bool = True):
        """
        Args:
            dtype: torch complex dtype, default torch.complex64.
            device: target device. If None, auto-selects npu:0 when available.
            fallback_to_cpu: if True, fall back to cpu when NPU is unavailable.
        """
        resolved = self._resolve_device(device=device, fallback_to_cpu=fallback_to_cpu)

        if self._is_npu_device(resolved) and hasattr(torch, "npu") and hasattr(torch.npu, "set_device"):
            # Keep behavior aligned with torch_npu best practice.
            torch.npu.set_device(resolved)

        super().__init__(dtype=dtype, device=resolved)
        self._requested_device = device
        self._fallback_to_cpu = bool(fallback_to_cpu)
        self._runtime_context = None
        self._local_matrix_cache = {}
        # 能力 sheet 派生的执行参数；裸构造保持现状安全默认（见 NPUBackend.caps）。
        self._max_qubits: int | None = None
        self._needs_real_imag: bool = True

    @classmethod
    def from_distributed_env(
        cls,
        dtype=None,
        fallback_to_cpu: bool = True,
        init_process_group: bool = True,
        process_group_backend: str | None = None,
    ):
        """
        Create a backend instance from distributed env variables.

        Env variables used:
            WORLD_SIZE, RANK, LOCAL_RANK, MASTER_ADDR, MASTER_PORT

        Behavior:
            - Preferred device is npu:{LOCAL_RANK}
            - Falls back to CPU when NPU is unavailable and fallback_to_cpu=True
            - Initializes torch.distributed when WORLD_SIZE > 1 and torchrun-style
              rendezvous env variables are present.
        """
        ctx = npu_runtime_context_from_env()
        backend = cls(dtype=dtype, device=f"npu:{ctx.local_rank}", fallback_to_cpu=fallback_to_cpu)
        pg_initialized = False
        pg_backend = None

        if ctx.distributed and init_process_group and _has_complete_distributed_env():
            if not torch.distributed.is_available():
                raise RuntimeError("torch.distributed is not available, cannot initialize distributed NPU runtime")
            if not torch.distributed.is_initialized():
                pg_backend = process_group_backend
                if pg_backend is None:
                    pg_backend = "hccl" if getattr(backend._device, "type", None) == "npu" else "gloo"
                torch.distributed.init_process_group(
                    backend=pg_backend,
                    rank=ctx.rank,
                    world_size=ctx.world_size,
                )
            else:
                pg_backend = process_group_backend
            pg_initialized = torch.distributed.is_initialized()

        backend._runtime_context = replace(
            ctx,
            process_group_initialized=pg_initialized,
            process_group_backend=pg_backend,
        )
        return backend

    @classmethod
    def caps(cls, capabilities, *, device=None, dtype=None, fallback_to_cpu: bool = True):
        """从 ``npu_probe`` 的能力 sheet 构造后端。

        显式注入：读 ``capabilities``（``NpuCapabilities``）填充执行参数
        （``max_qubits`` 用于 sizing guard，``needs_real_imag_decomp`` 备用），
        本身不探测。``device`` 缺省取 ``capabilities.device``。
        """
        backend = cls(
            dtype=dtype,
            device=device if device is not None else capabilities.device,
            fallback_to_cpu=fallback_to_cpu,
        )
        backend._max_qubits = capabilities.max_qubits
        backend._needs_real_imag = bool(capabilities.needs_real_imag_decomp)
        return backend

    def ensure_capacity(self, n_qubits: int) -> None:
        """单设备容量预检：``n_qubits`` 超过能力 sheet 的 ``max_qubits`` 时抛错。

        ``max_qubits`` 为 ``None``（裸构造或无内存数据）时不守卫。防止超容分配
        触发 OOM/SIGKILL。
        """
        if self._max_qubits is not None and int(n_qubits) > self._max_qubits:
            raise ValueError(
                f"n_qubits={n_qubits} 超过该 NPU 单设备容量 max_qubits={self._max_qubits}"
                f"（2^n complex64 态向量放不下）"
            )

    @property
    def distributed_initialized(self) -> bool:
        """True when this backend has a usable torch.distributed process group."""
        return bool(torch.distributed.is_available() and torch.distributed.is_initialized())

    @property
    def distributed_world_size(self) -> int:
        ctx = self._runtime_context
        return 1 if ctx is None else int(ctx.world_size)

    @property
    def distributed_rank(self) -> int:
        ctx = self._runtime_context
        return 0 if ctx is None else int(ctx.rank)

    def should_run_batch_index(self, index: int) -> bool:
        """Return True when this rank owns a batch item under task-parallel execution."""
        if not self.distributed_initialized:
            return True
        return int(index) % self.distributed_world_size == self.distributed_rank

    def gather_indexed_results(self, indexed_results):
        """
        Gather per-rank `(index, result)` pairs and return a globally ordered list.

        This enables real multi-NPU utilization for embarrassingly parallel
        workloads such as circuit batches and parameter scans. It does not shard
        a single state vector across devices.
        """
        if not self.distributed_initialized:
            return sorted(indexed_results, key=lambda item: item[0])

        gathered = [None for _ in range(self.distributed_world_size)]
        torch.distributed.all_gather_object(gathered, list(indexed_results))
        merged = []
        for rank_items in gathered:
            if rank_items:
                merged.extend(rank_items)
        return sorted(merged, key=lambda item: item[0])

    @staticmethod
    def _is_npu_device(device) -> bool:
        try:
            return torch.device(device).type == "npu"
        except Exception:
            return str(device).startswith("npu")

    @staticmethod
    def _resolve_device(device=None, fallback_to_cpu: bool = True):
        if device is None:
            if is_npu_available():
                return torch.device("npu:0")
            if fallback_to_cpu:
                return torch.device("cpu")
            raise RuntimeError(
                "NPU is not available. Install torch_npu and configure Ascend runtime correctly."
            )

        if isinstance(device, str):
            dev_str = device.strip().lower()
            if dev_str.startswith("npu") and not is_npu_available():
                if fallback_to_cpu:
                    return torch.device("cpu")
                raise RuntimeError(
                    "Requested NPU device, but NPU is unavailable. "
                    "Install torch_npu and verify torch.npu.is_available()."
                )
            return torch.device(device)

        dev = device
        try:
            dev_type = torch.device(dev).type
        except Exception:
            dev_type = str(dev).split(":", 1)[0].lower()

        if dev_type == "npu" and not is_npu_available():
            if fallback_to_cpu:
                return torch.device("cpu")
            raise RuntimeError(
                "Requested NPU device, but NPU is unavailable. "
                "Install torch_npu and verify torch.npu.is_available()."
            )
        return dev

    @property
    def name(self) -> str:
        return (
            f"NPUBackend(dtype={self._dtype}, device={self._device}, "
            f"npu_available={is_npu_available()})"
        )

    @staticmethod
    def _is_complex_tensor(value) -> bool:
        return isinstance(value, torch.Tensor) and torch.is_complex(value)

    def _is_npu_complex(self, tensor) -> bool:
        """Return True when on NPU device and tensor is complex — triggers workaround path."""
        return getattr(self._device, "type", None) == "npu" and self._is_complex_tensor(tensor)

    def _should_use_complex_matmul_workaround(self, a, b) -> bool:
        return (
            getattr(self._device, "type", None) == "npu"
            and self._is_complex_tensor(a)
            and self._is_complex_tensor(b)
        )

    @staticmethod
    def _complex_matmul_workaround(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_real, a_imag = torch.real(a), torch.imag(a)
        b_real, b_imag = torch.real(b), torch.imag(b)
        real = torch.matmul(a_real, b_real) - torch.matmul(a_imag, b_imag)
        imag = torch.matmul(a_real, b_imag) + torch.matmul(a_imag, b_real)
        return torch.complex(real, imag)

    @staticmethod
    def _real_complex_matmul(real_matrix: torch.Tensor, complex_matrix: torch.Tensor) -> torch.Tensor:
        real = torch.matmul(real_matrix, torch.real(complex_matrix))
        imag = torch.matmul(real_matrix, torch.imag(complex_matrix))
        return torch.complex(real, imag).to(dtype=complex_matrix.dtype)

    @staticmethod
    def _complex_real_matmul(complex_matrix: torch.Tensor, real_matrix: torch.Tensor) -> torch.Tensor:
        real = torch.matmul(torch.real(complex_matrix), real_matrix)
        imag = torch.matmul(torch.imag(complex_matrix), real_matrix)
        return torch.complex(real, imag).to(dtype=complex_matrix.dtype)

    @staticmethod
    def _matmul_forward(a, b):
        """NPU-safe matmul (no autograd graph), dispatching by real/complex mix.

        This is the raw forward kernel shared by the autograd-friendly
        ``_NpuMatmulFn`` (both its forward and backward) and is what keeps the
        complex decomposition out of the traced graph.
        """
        a_complex = _np_is_complex(a)
        b_complex = _np_is_complex(b)
        if not a_complex and b_complex:
            return NPUBackend._real_complex_matmul(a, b)
        if a_complex and not b_complex:
            return NPUBackend._complex_real_matmul(a, b)
        if a_complex and b_complex:
            return NPUBackend._complex_matmul_workaround(a, b)
        return torch.matmul(a, b)

    def matmul(self, a, b):
        if (
            getattr(self._device, "type", None) == "npu"
            and isinstance(a, torch.Tensor)
            and isinstance(b, torch.Tensor)
            and (torch.is_complex(a) or torch.is_complex(b))
        ):
            # Route complex matmul through the custom autograd Function so that
            # backward stays NPU-safe (no complex64 gradient-accumulation adds).
            return _NpuMatmulFn.apply(a, b)
        return super().matmul(a, b)

    def cast_local_matrix(self, matrix, cache_key=None, real_if_possible: bool = True):
        """
        Cast a small gate matrix once per backend/device.

        On NPU, real-valued gate matrices are kept as real tensors so local
        gate application can use two real GEMMs instead of the generic four
        real-GEMM complex workaround.
        """
        if cache_key is not None and cache_key in self._local_matrix_cache:
            return self._local_matrix_cache[cache_key]

        if isinstance(matrix, torch.Tensor):
            value = matrix.to(device=self._device)
            if value.is_complex() and not (
                real_if_possible
                and getattr(self._device, "type", None) == "npu"
                and not bool(torch.any(torch.imag(value)).detach().cpu().item())
            ):
                value = value.to(dtype=self._dtype)
            elif value.is_complex():
                real_dtype = torch.float32 if self._dtype == torch.complex64 else torch.float64
                value = torch.real(value).to(dtype=real_dtype, device=self._device)
        else:
            import numpy as np

            if hasattr(matrix, "detach") and hasattr(matrix, "cpu"):
                matrix = matrix.detach().cpu()
            array = np.asarray(matrix)
            keep_real = (
                real_if_possible
                and getattr(self._device, "type", None) == "npu"
                and np.allclose(np.imag(array), 0.0)
            )
            if keep_real:
                real_dtype = torch.float32 if self._dtype == torch.complex64 else torch.float64
                value = torch.tensor(np.real(array), dtype=real_dtype, device=self._device)
            else:
                value = torch.tensor(array, dtype=self._dtype, device=self._device)

        if cache_key is not None:
            self._local_matrix_cache[cache_key] = value
        return value

    def apply_local_matrix(self, local_matrix, state_block):
        """
        Apply a small local gate matrix to a flattened state block.

        This path avoids the generic complex workaround for common real-valued
        gates. A real local matrix times a complex state needs only two real
        matmuls; a genuinely complex local matrix still uses the compatible
        four-real-matmul decomposition on NPU.
        """
        return self.matmul(local_matrix, state_block)

    def apply_flat_gate(
        self,
        flat: "torch.Tensor",
        local_matrix: "torch.Tensor",
        indices: "torch.Tensor",
    ) -> "torch.Tensor":
        """局部量子门 gather→matmul→scatter 的 NPU autograd 安全路径。

        替代 _apply_local_matrix_to_state_flat 中的 npu_complex 分支：
        flat 和 local_matrix 各只作为图的一条输入边，backward 全程 float32，
        消除 aclnnAdd(DT_COMPLEX64) 错误。
        """
        return _NpuLocalGateApplyFn.apply(flat, local_matrix, indices)

    def eye(self, dim: int):
        """NPU workaround: build complex identity from real eye when complex eye kernel is unsupported."""
        if getattr(self._device, "type", None) == "npu" and self._dtype in (torch.complex64, torch.complex128):
            real_dtype = torch.float32 if self._dtype == torch.complex64 else torch.float64
            real_eye = torch.eye(dim, dtype=real_dtype, device=self._device)
            imag_eye = torch.zeros_like(real_eye)
            return torch.complex(real_eye, imag_eye).to(dtype=self._dtype)
        return super().eye(dim)

    def zeros_state(self, n_qubits: int):
        """NPU workaround: avoid complex in-place write path when initializing |0...0>."""
        self.ensure_capacity(n_qubits)
        if getattr(self._device, "type", None) == "npu" and self._dtype in (torch.complex64, torch.complex128):
            dim = 1 << n_qubits
            real_dtype = torch.float32 if self._dtype == torch.complex64 else torch.float64
            head = torch.ones((1, 1), dtype=real_dtype, device=self._device)
            tail = torch.zeros((dim - 1, 1), dtype=real_dtype, device=self._device)
            real = torch.cat([head, tail], dim=0)
            imag = torch.zeros_like(real)
            return torch.complex(real, imag).to(dtype=self._dtype)
        return super().zeros_state(n_qubits)

    def apply_unitary(self, state, unitary):
        return self.matmul(unitary, state)

    def kron(self, a, b):
        """NPU workaround: Kronecker product via real/imag decomposition."""
        if self._is_npu_complex(a) and self._is_npu_complex(b):
            ar, ai = torch.real(a), torch.imag(a)
            br, bi = torch.real(b), torch.imag(b)
            real = torch.kron(ar, br) - torch.kron(ai, bi)
            imag = torch.kron(ar, bi) + torch.kron(ai, br)
            return torch.complex(real, imag)
        return super().kron(a, b)

    def tensordot(self, a, b, axes):
        if self._is_npu_complex(a) or self._is_npu_complex(b):
            from ._contract import tensordot_via_matmul
            return tensordot_via_matmul(self, a, b, axes)
        return super().tensordot(a, b, axes)

    def transpose(self, a, axes):
        if self._is_npu_complex(a):
            perm = [int(x) for x in axes]
            return torch.complex(torch.real(a).permute(*perm), torch.imag(a).permute(*perm))
        return super().transpose(a, axes)

    def reshape(self, a, shape):
        shape = tuple(int(s) for s in shape)
        if self._is_npu_complex(a):
            return torch.complex(torch.real(a).reshape(shape), torch.imag(a).reshape(shape))
        return super().reshape(a, shape)

    def conj(self, a):
        if self._is_npu_complex(a):
            return torch.complex(torch.real(a), -torch.imag(a))
        return super().conj(a)

    def dagger(self, matrix):
        """NPU workaround: conjugate transpose via real/imag split (avoids torch.conj on complex64)."""
        if self._is_npu_complex(matrix):
            t = torch.transpose(matrix, -2, -1).contiguous()
            return torch.complex(torch.real(t), -torch.imag(t))
        return super().dagger(matrix)

    def trace(self, matrix):
        """NPU workaround: trace via real/imag split (avoids torch.trace on complex64)."""
        if self._is_npu_complex(matrix):
            return torch.complex(
                torch.trace(torch.real(matrix)),
                torch.trace(torch.imag(matrix)),
            )
        return super().trace(matrix)

    def inner_product(self, bra, ket):
        """NPU workaround: inner product via real/imag dot products (avoids torch.dot on complex64)."""
        if self._is_npu_complex(bra) and self._is_npu_complex(ket):
            b = bra.reshape(-1)
            k = ket.reshape(-1)
            br, bi = torch.real(b), torch.imag(b)
            kr, ki = torch.real(k), torch.imag(k)
            real = torch.dot(br, kr) + torch.dot(bi, ki)
            imag = torch.dot(br, ki) - torch.dot(bi, kr)
            return torch.complex(real, imag)
        return super().inner_product(bra, ket)

    @staticmethod
    def _partial_trace_real(rho_real: torch.Tensor, keep: list, n_qubits: int) -> torch.Tensor:
        """Real-valued partial trace kernel; shared by the NPU complex workaround."""
        keep = sorted(set(int(k) for k in keep))
        trace_out = [i for i in range(n_qubits) if i not in keep]
        if not trace_out:
            return rho_real.clone()
        # 逐比特求迹，每步把密度矩阵整形为 (L, 2, R, L, 2, R) 的秩-6 张量并对该
        # 比特的行/列同一指标求和。避免一次性整形为 [2]*n + [2]*n（秩 2n），后者
        # 在昇腾 NPU 上 n>4 即超出 ACL 算子最多 8 维的限制。工作张量的秩恒为 6，
        # 与量子比特总数无关；按降序求迹使保留比特维持原有（升序）次序。
        remaining = list(range(n_qubits))
        cur = rho_real
        for qubit in sorted(trace_out, reverse=True):
            pos = remaining.index(qubit)
            m = len(remaining)
            left = 1 << pos
            right = 1 << (m - pos - 1)
            block = cur.reshape(left, 2, right, left, 2, right)
            cur = block[:, 0, :, :, 0, :] + block[:, 1, :, :, 1, :]
            cur = cur.reshape(left * right, left * right)
            remaining.pop(pos)
        return cur

    def partial_trace(self, rho, keep, n_qubits):
        """NPU workaround: partial trace via real/imag split (avoids torch.einsum on complex64)."""
        if self._is_npu_complex(rho):
            rho_r = self._partial_trace_real(torch.real(rho), keep, n_qubits)
            rho_i = self._partial_trace_real(torch.imag(rho), keep, n_qubits)
            return torch.complex(rho_r, rho_i)
        return super().partial_trace(rho, keep, n_qubits)

    def expectation_sv(self, state, operator):
        """NPU workaround: ⟨ψ|O|ψ⟩ via a custom autograd Function.

        Wrapping the whole expectation keeps the (fan-out) state consumed once in
        the traced graph, so backward never accumulates complex gradients
        (aclnnAdd on complex64). Equivalent value/grad to the parent on CPU/CUDA.
        """
        if self._is_npu_complex(state):
            return _NpuExpectationFn.apply(state, operator)
        return super().expectation_sv(state, operator)

    def expectation_dm(self, rho, operator):
        """NPU workaround: Tr(ρO) via self.matmul/trace (avoids torch.trace/matmul on complex64)."""
        if self._is_npu_complex(rho):
            prod = self.matmul(rho, operator)
            val = self.trace(prod)
            return torch.real(val)
        return super().expectation_dm(rho, operator)

    def abs_sq(self, tensor):
        """NPU workaround: |z|² = real² + imag² (avoids aclnnAbs on complex64)."""
        if self._is_npu_complex(tensor):
            return torch.real(tensor) ** 2 + torch.imag(tensor) ** 2
        return super().abs_sq(tensor)

    def measure_probs(self, state):
        """NPU workaround: compute probabilities via real²+imag² instead of abs()."""
        if self._is_npu_complex(state):
            flat = state.reshape(-1)
            probs = torch.real(flat) ** 2 + torch.imag(flat) ** 2
            total = probs.sum()
            if total > 0:
                probs = probs / total
            return probs
        return super().measure_probs(state)

    def hamiltonian_expectation_pauli(
        self,
        state: torch.Tensor,
        basis_indices: torch.Tensor,
        pauli_cache: list,
    ) -> torch.Tensor:
        """Re⟨ψ|H|ψ⟩（Pauli 项列表形式 Hamiltonian），NPU autodiff 安全。

        用 _NpuHamiltonianExpectationFn 包装，使 state 在图中只出现一次，
        避免 1313 次 complex64 梯度累加（aclnnAdd 缺失）。
        """
        return _NpuHamiltonianExpectationFn.apply(state.reshape(-1), basis_indices, pauli_cache)

    @property
    def runtime_context(self):
        """Distributed runtime context if created via from_distributed_env, else None."""
        return self._runtime_context
