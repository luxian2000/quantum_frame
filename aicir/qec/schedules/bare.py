"""裸 ancilla syndrome 提取：每个生成元一个 ancilla。

单轮结构：H(ancilla) → 从 ancilla 向生成元 support 上各比特施加受控 P → H(ancilla)
→ measure 写入经典寄存器 → reset(ancilla)。

轮间靠 reset 复用 ancilla，这是表面码 d=3 保持在 9+8 而非逐轮新增 ancilla 的原因。
"""

from __future__ import annotations

from aicir.core.circuit import Circuit, cx, cy, cz, hadamard, measure, reset
from aicir.core.classical import ClassicalRegister

from ..code import gf2_to_pauli
from . import ReadoutCircuit, RoundCircuit, register_schedule

_CONTROLLED = {"X": cx, "Y": cy, "Z": cz}


class BareAncillaSchedule:
    """默认调度：裸 ancilla，标准 CNOT 顺序。"""

    name = "bare"

    def _split(self, code):
        data = tuple(range(code.n))
        ancilla = tuple(range(code.n, code.n + code.m))
        return data, ancilla, code.n + code.m

    def build_encode(self, code, logical_state: str = "0"):
        """制备逻辑初态。

        M1 只支持 |0…0> 作为码空间内的物理初态：对 CSS 码与本仓库内置的五个码，
        全 |0> 已是 Z 型稳定子的 +1 本征态；X 型稳定子的投影由第一轮 syndrome 提取
        完成（这正是轮 0 参考值由 reference 向量给出的原因）。
        """
        state = str(logical_state)
        if state not in ("0", "+"):
            raise ValueError(f"M1 只支持逻辑初态 '0' 或 '+'，收到 {state!r}")
        _, _, n_total = self._split(code)
        cir = Circuit(n_qubits=n_total)
        if state == "+":
            for q in range(code.n):
                cir.append(hadamard(q))
        return cir

    def build_round(self, code, round_index: int, *, creg_name: str = "syn") -> RoundCircuit:
        data, ancilla, n_total = self._split(code)
        reg = ClassicalRegister(code.m, creg_name)
        cir = Circuit(n_qubits=n_total)
        for j in range(code.m):
            anc = ancilla[j]
            labels = gf2_to_pauli(code.generators[j])
            cir.append(hadamard(anc))
            for q, ch in enumerate(labels):
                if ch != "I":
                    cir.append(_CONTROLLED[ch](q, [anc]))
            cir.append(hadamard(anc))
        cir.append(measure(list(ancilla), creg=reg))
        cir.append(reset(list(ancilla)))
        return RoundCircuit(
            circuit=cir, creg_name=creg_name, ancilla_qubits=ancilla,
            data_qubits=data, record_offset=int(round_index) * code.m,
        )

    def build_readout(self, code, logical_state: str = "0") -> ReadoutCircuit:
        """末端逻辑读出：在制备基下逐个测量 data 比特。

        读出基必须与制备基匹配：'0'/'1' → 逻辑 Z 基，'+'/'-' → 逻辑 X 基。
        """
        state = str(logical_state)
        if state not in ("0", "1", "+", "-"):
            raise ValueError(f"未知逻辑初态 {state!r}；读出基无法与制备基匹配")
        data, _, n_total = self._split(code)
        reg = ClassicalRegister(code.n, "readout")
        cir = Circuit(n_qubits=n_total)
        if state in ("+", "-"):
            for q in data:
                cir.append(hadamard(q))          # X 基 → 先转到 Z 基再测
        cir.append(measure(list(data), creg=reg))
        records = tuple(tuple(range(code.n)) for _ in range(code.k))
        return ReadoutCircuit(circuit=cir, creg_name="readout", observable_records=records)


register_schedule("bare", BareAncillaSchedule)
