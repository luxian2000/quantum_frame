"""张量网络收缩：求路径（opt_einsum 可选 / 内置贪心），逐对经 backend.tensordot 执行。"""

from __future__ import annotations


def _pair_contract(t1, id1, t2, id2, backend):
    shared = [x for x in id1 if x in id2]
    a1 = [id1.index(x) for x in shared]
    a2 = [id2.index(x) for x in shared]
    out = backend.tensordot(t1, t2, (a1, a2))
    new_ids = tuple(x for x in id1 if x not in shared) + tuple(x for x in id2 if x not in shared)
    return out, new_ids


def _greedy_path(indices):
    """贪心成对路径：优先共享标签最多、结果秩最小的一对。

    返回 (i, j) 列表，位置指向**收缩中不断缩小**的列表：每步移除 i、j，把结果追加到末尾。
    """
    idx = [set(t) for t in indices]
    path = []
    while len(idx) > 1:
        best = None
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                shared = idx[i] & idx[j]
                result_rank = len(idx[i] ^ idx[j]) if shared else len(idx[i]) + len(idx[j])
                key = (-len(shared), result_rank)
                if best is None or key < best[0]:
                    best = (key, i, j)
        _, i, j = best
        merged = idx[i] ^ idx[j]
        for k in sorted((i, j), reverse=True):
            idx.pop(k)
        idx.append(merged)
        path.append((i, j))
    return path


def _opt_einsum_path(indices, open_indices):
    import opt_einsum

    ids = sorted({x for tup in indices for x in tup})
    sym = {x: opt_einsum.get_symbol(i) for i, x in enumerate(ids)}
    subs = ",".join("".join(sym[x] for x in tup) for tup in indices)
    out = "".join(sym[x] for x in open_indices)
    eq = f"{subs}->{out}"
    shapes = [tuple(2 for _ in tup) for tup in indices]
    path, _info = opt_einsum.contract_path(eq, *shapes, shapes=True, optimize="auto")
    return list(path)


def _contraction_path(indices, open_indices):
    try:
        return _opt_einsum_path(indices, open_indices)
    except ImportError:
        return _greedy_path(indices)


def contract(tensors, indices, open_indices, backend):
    tens = list(tensors)
    idx = [tuple(t) for t in indices]
    for step in _contraction_path(idx, open_indices):
        if len(step) < 2:
            # opt_einsum 对无需收缩的单张量网络返回 (0,) —— 无操作步，跳过
            continue
        i, j = step[0], step[1]
        t, ids = _pair_contract(tens[i], idx[i], tens[j], idx[j], backend)
        for k in sorted((i, j), reverse=True):
            tens.pop(k)
            idx.pop(k)
        tens.append(t)
        idx.append(ids)

    result, ids = tens[0], idx[0]
    if open_indices:
        perm = [ids.index(x) for x in open_indices]
        if perm != list(range(len(perm))):
            result = backend.transpose(result, perm)
    return result
