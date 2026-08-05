"""稳定子码的 GF(2) 辛表示核心。

约定：n 比特 Pauli 表示为 (x|z) ∈ F₂^{2n}，x_i=1 当 P_i∈{X,Y}，z_i=1 当 P_i∈{Z,Y}。
辛积 = (a_x·b_z + a_z·b_x) mod 2，0 表示对易、1 表示反对易。

本模块是**纯代数**层：不构造线路、不接触后端、不依赖 aicir.core 之外的东西。
"""

from __future__ import annotations

from itertools import combinations, product

import numpy as np

_LABEL_TO_XZ = {"I": (0, 0), "X": (1, 0), "Y": (1, 1), "Z": (0, 1)}
_XZ_TO_LABEL = {(0, 0): "I", (1, 0): "X", (1, 1): "Y", (0, 1): "Z"}


def _as_label_string(item) -> str:
    """接受 str 或 aicir.core.operators.PauliString，统一取出标签串。"""
    if isinstance(item, str):
        return item.strip().upper()
    labels = getattr(item, "qubit_labels", None)
    if labels is None:
        raise TypeError(f"无法解析为 Pauli 串：{item!r}")
    return "".join(labels)


def pauli_to_gf2(item, n_qubits: int | None = None) -> np.ndarray:
    """Pauli 标签串 → (2n,) uint8 的 (x|z) 向量。"""
    label = _as_label_string(item)
    n = len(label) if n_qubits is None else int(n_qubits)
    if len(label) != n:
        raise ValueError(f"Pauli 串长度 {len(label)} 与 n_qubits {n} 不符")
    vec = np.zeros(2 * n, dtype=np.uint8)
    for i, ch in enumerate(label):
        if ch not in _LABEL_TO_XZ:
            raise ValueError(f"未知泡利标签 {ch!r}，只支持 I/X/Y/Z")
        x, z = _LABEL_TO_XZ[ch]
        vec[i] = x
        vec[n + i] = z
    return vec


def gf2_to_pauli(vec: np.ndarray) -> str:
    """(2n,) uint8 → Pauli 标签串。"""
    vec = np.asarray(vec, dtype=np.uint8).ravel()
    if vec.size % 2:
        raise ValueError("向量长度必须是偶数（x 块 + z 块）")
    n = vec.size // 2
    return "".join(_XZ_TO_LABEL[(int(vec[i]), int(vec[n + i]))] for i in range(n))


def symplectic_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """辛积 (a_x·b_z + a_z·b_x) mod 2。0 = 对易，1 = 反对易。

    a 可为 (2n,) 或 (m, 2n)；b 为 (2n,)。返回标量或 (m,)。

    标量 vs. 数组由 a 输入时的维度决定（1-D → 标量，2-D → (m,)），
    不能用输出 size 判断——否则 m=1 的批量输入会被错误折叠成标量。
    """
    a_arr = np.asarray(a, dtype=np.uint8)
    was_1d = a_arr.ndim == 1
    a2 = np.atleast_2d(a_arr)
    b = np.asarray(b, dtype=np.uint8).ravel()
    n = b.size // 2
    out = (a2[:, :n] @ b[n:] + a2[:, n:] @ b[:n]) % 2
    return np.uint8(out[0]) if was_1d else out.astype(np.uint8)


def _gf2_rank(rows: np.ndarray) -> int:
    """GF(2) 高斯消元求秩。"""
    mat = np.array(rows, dtype=np.uint8, copy=True)
    rank = 0
    n_cols = mat.shape[1]
    for col in range(n_cols):
        pivot = next((r for r in range(rank, mat.shape[0]) if mat[r, col]), None)
        if pivot is None:
            continue
        mat[[rank, pivot]] = mat[[pivot, rank]]
        for r in range(mat.shape[0]):
            if r != rank and mat[r, col]:
                mat[r] ^= mat[rank]
        rank += 1
        if rank == mat.shape[0]:
            break
    return rank


def _in_span(rows: np.ndarray, vec: np.ndarray) -> bool:
    """vec 是否在 rows 的 GF(2) 张成空间内。"""
    if rows.shape[0] == 0:
        return not np.asarray(vec, dtype=np.uint8).any()
    base = _gf2_rank(rows)
    augmented = np.vstack([rows, np.asarray(vec, dtype=np.uint8)[None, :]])
    return _gf2_rank(augmented) == base


class StabilizerCode:
    """稳定子码：生成元 + 逻辑算符的 GF(2) 辛表示。"""

    def __init__(self, n, generators, signs, logical_x, logical_z, name, coords=None):
        self.n = int(n)
        self.generators = np.asarray(generators, dtype=np.uint8).reshape(-1, 2 * self.n)
        self.signs = np.asarray(signs, dtype=np.uint8).ravel()
        self.logical_x = np.asarray(logical_x, dtype=np.uint8).reshape(-1, 2 * self.n)
        self.logical_z = np.asarray(logical_z, dtype=np.uint8).reshape(-1, 2 * self.n)
        self.name = str(name)
        self.coords = dict(coords) if coords else {}
        if self.logical_x.shape[0] != self.logical_z.shape[0]:
            raise ValueError("logical_x 与 logical_z 数量必须相同")
        if self.signs.size != self.generators.shape[0]:
            raise ValueError("signs 长度必须与生成元数一致")

    @property
    def m(self) -> int:
        """稳定子生成元个数。"""
        return int(self.generators.shape[0])

    @property
    def k(self) -> int:
        """逻辑比特数。"""
        return int(self.logical_x.shape[0])

    @classmethod
    def from_paulis(cls, generators, *, logical_x, logical_z, name,
                    coords=None, signs=None) -> "StabilizerCode":
        """从 Pauli 串（str 或 PauliString）构造。"""
        gen_labels = [_as_label_string(g) for g in generators]
        n = len(gen_labels[0])
        if any(len(g) != n for g in gen_labels):
            raise ValueError("所有生成元的 Pauli 串长度必须一致")
        gens = np.stack([pauli_to_gf2(g, n) for g in gen_labels])
        lx = np.stack([pauli_to_gf2(_as_label_string(p), n) for p in logical_x])
        lz = np.stack([pauli_to_gf2(_as_label_string(p), n) for p in logical_z])
        sg = np.zeros(gens.shape[0], dtype=np.uint8) if signs is None else signs
        return cls(n, gens, sg, lx, lz, name, coords)

    def validate(self) -> None:
        """校验码的合法性；任何一项不成立都抛 ValueError 并指名违规对象。"""
        for i, j in combinations(range(self.m), 2):
            if symplectic_product(self.generators[i], self.generators[j]):
                raise ValueError(
                    f"[{self.name}] 生成元 {i} ({gf2_to_pauli(self.generators[i])}) 与 "
                    f"{j} ({gf2_to_pauli(self.generators[j])}) 不对易"
                )
        rank = _gf2_rank(self.generators)
        if rank != self.m:
            raise ValueError(
                f"[{self.name}] 生成元线性相关：{self.m} 个生成元的 GF(2) 秩只有 {rank}"
            )
        if self.n - rank != self.k:
            raise ValueError(
                f"[{self.name}] 逻辑比特数不符：n−rank = {self.n - rank}，但给出了 {self.k} 组 logical"
            )
        for kind, rows in (("logical_x", self.logical_x), ("logical_z", self.logical_z)):
            for i, row in enumerate(rows):
                bad = np.nonzero(symplectic_product(self.generators, row))[0]
                if bad.size:
                    raise ValueError(
                        f"[{self.name}] {kind}[{i}] ({gf2_to_pauli(row)}) 与生成元 "
                        f"{int(bad[0])} 反对易，不在 normalizer 内"
                    )
        for i in range(self.k):
            for j in range(self.k):
                got = int(symplectic_product(self.logical_x[i], self.logical_z[j]))
                want = 1 if i == j else 0
                if got != want:
                    raise ValueError(
                        f"[{self.name}] logical_x[{i}] 与 logical_z[{j}] 的辛积应为 {want}，实际 {got}"
                    )

    def syndrome(self, error) -> np.ndarray:
        """错误 Pauli 的综合征：与各生成元的辛积，shape (m,)。"""
        vec = error if isinstance(error, np.ndarray) else pauli_to_gf2(error, self.n)
        return np.atleast_1d(symplectic_product(self.generators, vec)).astype(np.uint8)

    def _is_logical(self, vec: np.ndarray) -> bool:
        """与所有生成元对易，但不属于稳定子群 → 是非平凡逻辑算符。"""
        if self.syndrome(vec).any():
            return False
        return not _in_span(self.generators, vec)

    def distance(self, max_weight: int | None = None) -> int:
        """码距：最小权非平凡逻辑算符的权重。

        max_weight=None 表示不设上限、搜到为止（n ≲ 15 可行）；
        给定整数则搜到该权重为止，未找到时抛 ValueError——**绝不返回下界猜测**。
        """
        limit = self.n if max_weight is None else int(max_weight)
        for weight in range(1, limit + 1):
            for support in combinations(range(self.n), weight):
                for labels in product("XYZ", repeat=weight):
                    chars = ["I"] * self.n
                    for q, ch in zip(support, labels):
                        chars[q] = ch
                    vec = pauli_to_gf2("".join(chars), self.n)
                    if self._is_logical(vec):
                        return weight
        raise ValueError(
            f"[{self.name}] 在 max_weight={limit} 内未找到非平凡逻辑算符；"
            f"提高 max_weight 或检查码定义（不返回下界猜测）"
        )

    def logical_class(self, residual) -> np.ndarray:
        """残余 Pauli 落在哪个逻辑陪集。

        返回 (k, 2) uint8：第 i 行 = 第 i 个逻辑比特上的 (X 分量, Z 分量)。
        X 分量由 residual 与 logical_z[i] 的辛积给出，Z 分量由与 logical_x[i] 的辛积给出。
        全零表示残余落在稳定子群内。
        """
        vec = residual if isinstance(residual, np.ndarray) else pauli_to_gf2(residual, self.n)
        out = np.zeros((self.k, 2), dtype=np.uint8)
        for i in range(self.k):
            out[i, 0] = symplectic_product(self.logical_z[i], vec)
            out[i, 1] = symplectic_product(self.logical_x[i], vec)
        return out

    def verdict(self, residual) -> str:
        """把 logical_class 结果翻译成判定字符串。

        全零 → "corrected"；k=1 → "logical_x"/"logical_z"/"logical_y"；
        k>1 → "logical_q{i}_{x|y|z}"，多个逻辑比特出错时按下标升序以 "+" 连接。
        """
        cls = self.logical_class(residual)
        if not cls.any():
            return "corrected"
        parts = []
        for i, (has_x, has_z) in enumerate(cls):
            if not (has_x or has_z):
                continue
            letter = "y" if (has_x and has_z) else ("x" if has_x else "z")
            parts.append(f"logical_{letter}" if self.k == 1 else f"logical_q{i}_{letter}")
        return "+".join(parts)

    def __repr__(self) -> str:
        return f"StabilizerCode({self.name}, n={self.n}, k={self.k}, m={self.m})"
