import pytest

from aicir.qec.code import StabilizerCode
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


def test_detector_records_are_correctly_indexed():
    """Detector.records 必须是真实的 record 下标，而非只有个数正确。

    只断言 n_detectors/round_slice 长度的测试抓不住转置类 bug（例如把
    cur = t*m+s 误写成 s*rounds+t）——数量与 verify_schedule 都不受影响
    （verify_schedule 直接读 raw 数组，不经过 records），只有直接比对
    records 内容才能发现。"""
    code = get_code("steane")
    m = code.m
    layout = build_layout(code, BareAncillaSchedule(), rounds=3)
    # 轮 0 detector 只含单个 record，下标为 0*m+s。
    for s in layout.round0_stabilizers:
        det = layout.detector_at(s, 0)
        assert det.records == (0 * m + s,)
    # 轮 2 detector 含两个 record：上一轮 (1*m+s) 与本轮 (2*m+s)，顺序固定。
    for s in range(m):
        det = layout.detector_at(s, 2)
        assert det.records == (1 * m + s, 2 * m + s)


def test_observable_records_follow_logical_operator_support():
    """Observable.records 必须取逻辑算符自身的支持，而非笼统的「全部 n 个 data
    比特」——否则 k>1 时不同逻辑比特会共用同一组 record、彼此不可区分。

    对 k=1 的内置码，「全 n 比特奇偶」恰好与逻辑算符的实际支持稳定子等价，
    数值可能巧合相同；repetition(d=3, Z 基) 的 logical_z="ZII" 支持只有
    qubit 0，用它来暴露「全 n 比特」与「真实支持」的差异。
    """
    code = get_code("repetition", d=3, basis="Z")
    layout = build_layout(code, BareAncillaSchedule(), rounds=1, logical_state="0")
    base = 1 * code.m
    assert layout.observables[0].records == (base + 0,)

    # k=2：[[4,2,2]] iceberg 码（Task 10 将注册的同一构造）。两个逻辑比特的
    # logical_z 支持不同（"ZIZI" 对 qubit 0,2；"ZZII" 对 qubit 0,1），
    # records 必须彼此不同，否则解码器无法区分两个逻辑比特。
    code42 = StabilizerCode.from_paulis(
        generators=["XXXX", "ZZZZ"],
        logical_x=["XXII", "XIXI"],
        logical_z=["ZIZI", "ZZII"],
        name="iceberg_422",
    )
    code42.validate()
    layout42 = build_layout(code42, BareAncillaSchedule(), rounds=1, logical_state="0")
    base42 = 1 * code42.m
    assert layout42.observables[0].records == (base42 + 0, base42 + 2)
    assert layout42.observables[1].records == (base42 + 0, base42 + 1)
    assert layout42.observables[0].records != layout42.observables[1].records
