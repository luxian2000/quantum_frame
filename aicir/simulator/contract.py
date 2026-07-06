"""张量网络收缩：求路径（cotengra / opt_einsum 可选 / 内置贪心），逐对经 backend.tensordot 执行。

cotengra 只负责“规划”（路径 + 切片指标集合），执行仍走 `_pair_contract`/`backend.tensordot`
（逐对张量收缩），从而保留 NPU/自动微分兼容性——不把张量交给 cotengra 自身的执行后端。
"""

from __future__ import annotations

import itertools

_COTENGRA_HINT = '需要 cotengra；请安装可选依赖：pip install -e ".[tn]"'
_AUTO_COTENGRA_MIN_TENSORS = 24  # auto 下小网络不值得 HyperOptimizer 的搜索开销


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


def _cotengra_path(indices, open_indices, memory_limit):
    """cotengra 规划：返回 (path, sliced_labels)。仅规划，不执行。

    path 为 opt_einsum 兼容的线性 (i, j) 格式（`tree.get_path()`，区别于 SSA 格式的
    `tree.get_ssa_path()`）；已用 3 张量探针验证按此格式逐对收缩与直接收缩结果一致。
    """
    import cotengra as ctg

    ids = sorted({x for tup in indices for x in tup})
    sym = {x: f"i{k}" for k, x in enumerate(ids)}
    inputs = [tuple(sym[x] for x in tup) for tup in indices]
    output = tuple(sym[x] for x in open_indices)
    size_dict = {s: 2 for s in sym.values()}
    kwargs = {"progbar": False, "parallel": False}
    if memory_limit is not None:
        kwargs["slicing_opts"] = {"target_size": int(memory_limit)}
    opt = ctg.HyperOptimizer(**kwargs)
    tree = opt.search(inputs, output, size_dict)
    rev = {v: k for k, v in sym.items()}
    sliced = tuple(rev[s] for s in tree.sliced_inds)
    return list(tree.get_path()), sliced


def _plan(indices, open_indices, optimize, memory_limit):
    if optimize not in ("auto", "cotengra", "opt_einsum", "greedy"):
        raise ValueError(f"未知 optimize: {optimize!r}（可选 auto/cotengra/opt_einsum/greedy）")
    if memory_limit is not None and optimize in ("opt_einsum", "greedy"):
        raise ValueError("memory_limit（切片）只有 cotengra 规划器支持，不能与 optimize="
                         f"{optimize!r} 组合")
    if optimize == "cotengra" or memory_limit is not None:
        try:
            return _cotengra_path(indices, open_indices, memory_limit)
        except ImportError:
            raise ImportError(_COTENGRA_HINT)
    if optimize == "auto":
        if len(indices) >= _AUTO_COTENGRA_MIN_TENSORS:
            try:
                return _cotengra_path(indices, open_indices, None)
            except ImportError:
                pass
        try:
            return _opt_einsum_path(indices, open_indices), ()
        except ImportError:
            return _greedy_path(indices), ()
    if optimize == "opt_einsum":
        return _opt_einsum_path(indices, open_indices), ()  # 缺失时 ImportError 原样抛出
    return _greedy_path(indices), ()


def _execute(tens, idx, path, backend):
    """按 path 逐对收缩（原 contract 主循环提取）。"""
    for step in path:
        if len(step) < 2:
            continue  # 单张量网络的 (0,) 无操作步
        i, j = step[0], step[1]
        t, ids = _pair_contract(tens[i], idx[i], tens[j], idx[j], backend)
        for k in sorted((i, j), reverse=True):
            tens.pop(k)
            idx.pop(k)
        tens.append(t)
        idx.append(ids)
    return tens[0], idx[0]


def contract(tensors, indices, open_indices, backend, *, optimize="auto", memory_limit=None):
    idx = [tuple(t) for t in indices]
    path, sliced = _plan(idx, open_indices, optimize, memory_limit)

    if not sliced:
        result, ids = _execute(list(tensors), list(idx), path, backend)
    else:
        # cotengra 的切片指标可能同时含“内部（bond）标签”与“开放（open）标签”：
        # - bond 标签切片后在最终结果中彻底消失（本就是被收缩掉的求和指标）——
        #   对各取值的贡献直接 backend.add 求和即可；
        # - open 标签切片后仍要作为输出的一根腿保留——若也直接求和会把它错误地
        #   收缩掉（实测：3 比特网络在 memory_limit=4 下，全态本身已是 8 元素，
        #   仅切内部指标无法压到预算内，cotengra 必然连开放指标一起切；用
        #   cotengra 自带的 tree.contract() 验证过其重建方式是“按位置摆放”而非
        #   求和）。这里用已有原语实现等价的“摆放”：backend.eye(2)+backend.take
        #   取出该取值对应的 one-hot 行向量，与切片贡献做外积
        #   （backend.tensordot(..., axes=([], []))）补回该轴，不同取值的外积
        #   支撑互不相交，因此对它们 backend.add 求和即等价于拼接摆放。
        sliced_open = [s for s in sliced if s in open_indices]
        sliced_bond = [s for s in sliced if s not in open_indices]
        eye2 = backend.eye(2)
        result = ids = None
        for open_assign in itertools.product((0, 1), repeat=len(sliced_open)):
            piece = piece_ids = None
            for bond_assign in itertools.product((0, 1), repeat=len(sliced_bond)):
                ts, ix = [], []
                for t, tup in zip(tensors, idx):
                    for lab, val in itertools.chain(
                        zip(sliced_open, open_assign), zip(sliced_bond, bond_assign)
                    ):
                        if lab in tup:
                            t = backend.take(t, tup.index(lab), val)
                            tup = tuple(x for x in tup if x != lab)
                    ts.append(t)
                    ix.append(tup)
                r, r_ids = _execute(ts, ix, path, backend)
                piece = r if piece is None else backend.add(piece, r)
                piece_ids = r_ids
            for lab, val in zip(sliced_open, open_assign):
                one_hot = backend.take(eye2, 0, val)
                piece = backend.tensordot(piece, one_hot, ([], []))
                piece_ids = piece_ids + (lab,)
            result = piece if result is None else backend.add(result, piece)
            ids = piece_ids

    if open_indices:
        perm = [ids.index(x) for x in open_indices]
        if perm != list(range(len(perm))):
            result = backend.transpose(result, perm)
    return result
