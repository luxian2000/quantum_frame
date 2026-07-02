"""设备无关的 tensordot：仅用 backend.transpose/reshape/matmul 实现。

NPUBackend 复数张量走此路径，复用其 autograd-safe 复数 matmul，从而收缩全程
无跨节点 complex64 累加，且保留自动微分图。"""

from __future__ import annotations

import math


def _prod(dims) -> int:
    out = 1
    for d in dims:
        out *= int(d)
    return out


def tensordot_via_matmul(backend, a, b, axes):
    ax_a = [int(x) for x in axes[0]]
    ax_b = [int(x) for x in axes[1]]
    nda, ndb = len(a.shape), len(b.shape)
    ax_a = [x % nda for x in ax_a]
    ax_b = [x % ndb for x in ax_b]
    free_a = [i for i in range(nda) if i not in ax_a]
    free_b = [i for i in range(ndb) if i not in ax_b]

    a2 = backend.transpose(a, free_a + ax_a)
    b2 = backend.transpose(b, ax_b + free_b)

    fa = [int(a.shape[i]) for i in free_a]
    ca = [int(a.shape[i]) for i in ax_a]
    fb = [int(b.shape[i]) for i in free_b]

    a2 = backend.reshape(a2, (_prod(fa), _prod(ca)))
    b2 = backend.reshape(b2, (_prod(ca), _prod(fb)))
    out = backend.matmul(a2, b2)
    return backend.reshape(out, tuple(fa) + tuple(fb))
