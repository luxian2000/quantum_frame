import pytest

from aicir.qec.codes import get_code
from aicir.qec.schedules import (
    BareAncillaSchedule, build_layout, resolve_schedule, verify_schedule,
)

CASES = [
    ("repetition", {"d": 3, "basis": "Z"}),
    ("five_qubit", {}),
    ("steane", {}),
    ("shor", {}),
    ("surface", {"d": 3}),
]


@pytest.mark.parametrize("name,kwargs", CASES)
@pytest.mark.parametrize("rounds", [1, 3])
def test_detectors_are_deterministic_without_noise(name, kwargs, rounds):
    """无噪声下每个 detector 必须恒为 0——这是提取调度唯一最有力的结构性检验。"""
    code = get_code(name, **kwargs)
    verify_schedule(code, BareAncillaSchedule(), rounds)


def test_layout_shape_accounts_for_partial_round_zero():
    """Shor 码 8 个生成元中只有 6 个纯 Z 型 → 轮 0 只建 6 个 detector。

    刻意选 Shor 而非 Steane/surface：后两者 X/Z 型各占一半，
    len(round0) 恰等于 m/2，m*rounds 与正确值在某些轮数下会巧合相等。
    """
    code = get_code("shor")
    layout = build_layout(code, BareAncillaSchedule(), rounds=3)
    assert layout.n_stabilizers == code.m == 8
    assert layout.n_rounds == 3
    assert layout.round0_stabilizers == (0, 1, 2, 3, 4, 5)
    assert layout.n_detectors == 6 + 8 * 2 == 22
    assert len(layout.round_slice(0)) == 6
    assert len(layout.round_slice(1)) == 8


@pytest.mark.parametrize("name,kwargs,expected", [
    ("repetition", {"d": 3, "basis": "Z"}, (0, 1)),
    ("five_qubit", {}, ()),                      # 非 CSS：轮 0 无任何确定生成元
    ("steane", {}, (3, 4, 5)),
    ("surface", {"d": 3}, (4, 5, 6, 7)),
])
def test_deterministic_round0_matches_measured_values(name, kwargs, expected):
    """轮 0 确定的生成元集合 —— 数值已在真机上逐码实测确认。"""
    from aicir.qec.schedules import deterministic_round0
    code = get_code(name, **kwargs)
    assert deterministic_round0(code, "0") == expected


def test_round_circuit_uses_data_then_ancilla_numbering():
    code = get_code("repetition", d=3, basis="Z")
    rc = BareAncillaSchedule().build_round(code, 0)
    assert rc.data_qubits == (0, 1, 2)
    assert rc.ancilla_qubits == (3, 4)
    assert rc.circuit.n_qubits == 5


def test_readout_basis_must_match_preparation():
    code = get_code("steane")
    sched = BareAncillaSchedule()
    sched.build_readout(code, "0")     # Z 基制备 → Z 基读出，正常
    with pytest.raises(ValueError, match="基"):
        sched.build_readout(code, "?")


def test_resolve_schedule_accepts_name_and_instance():
    assert isinstance(resolve_schedule("bare"), BareAncillaSchedule)
    inst = BareAncillaSchedule()
    assert resolve_schedule(inst) is inst
    with pytest.raises(KeyError, match="bare"):
        resolve_schedule("no_such_schedule")


def test_verify_schedule_reports_offending_detector():
    """人为破坏调度（漏掉 ancilla reset）必须被 verify_schedule 抓住。

    此处刻意用 Steane 而非 repetition：ancilla-as-control 的相位反冲提取电路下，
    漏 reset 造成的读数是 raw[t] = XOR_{i=0}^{t} s_i（s_i 为该生成元第 i 轮的真实
    本征值）。repetition(Z 基) 全部生成元是纯 Z 型，在 |0…0⟩ 上 s_i 恒为 0（非概率
    性地恒为 0，不是 50/50 随机），于是这条 XOR 链无论轮数、无论 shots 数目都精确
    抵消为 0 —— 该 bug 在这一个码上是**数学上不可观测的**，不是实现问题。
    Steane 有 X 型生成元（轮 0 读数真随机），坏调度下 raw[偶轮]=s、raw[奇轮]=0，
    只要某个 X 型生成元某个 shot 的 s=1，相邻轮 XOR 就会非零；默认 4 个 shots
    （seed=0..3）在 rounds=3 下已验证第 3 个 shot 必然触发。"""
    code = get_code("steane")

    class BrokenSchedule(BareAncillaSchedule):
        def build_round(self, code, round_index, *, creg_name="syn"):
            rc = super().build_round(code, round_index, creg_name=creg_name)
            # 去掉所有 reset 指令 → 第二轮起 ancilla 带着上一轮的值，detector 不再恒 0
            kept = [g for g in rc.circuit.gates
                    if getattr(g, "measurement_type", None) != "reset"]
            rc.circuit.gates[:] = kept
            return rc

    with pytest.raises(ValueError, match="detector"):
        verify_schedule(code, BrokenSchedule(), rounds=3)
