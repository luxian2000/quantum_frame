import numpy as np
import pytest

from aicir.qec.code import (
    StabilizerCode, gf2_to_pauli, pauli_to_gf2, symplectic_product,
)

# 五比特完美码 [[5,1,3]]（生成元与 logical 已逐对手工验证对易关系）
FIVE_GENS = ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]


def test_pauli_gf2_roundtrip():
    v = pauli_to_gf2("XYZI")
    # x 块：X->1, Y->1, Z->0, I->0 ；z 块：X->0, Y->1, Z->1, I->0
    assert np.array_equal(v, np.array([1, 1, 0, 0, 0, 1, 1, 0], dtype=np.uint8))
    assert gf2_to_pauli(v) == "XYZI"


def test_symplectic_product_basic():
    x = pauli_to_gf2("X")
    z = pauli_to_gf2("Z")
    y = pauli_to_gf2("Y")
    assert symplectic_product(x, z) == 1      # X 与 Z 反对易
    assert symplectic_product(x, y) == 1      # X 与 Y 反对易
    assert symplectic_product(x, x) == 0      # 自对易
    assert symplectic_product(pauli_to_gf2("XX"), pauli_to_gf2("ZZ")) == 0  # 重叠 2 → 对易


def test_symplectic_product_broadcasts_over_rows():
    rows = np.stack([pauli_to_gf2("XI"), pauli_to_gf2("ZI")])
    out = symplectic_product(rows, pauli_to_gf2("ZI"))
    assert out.shape == (2,)
    assert list(out) == [1, 0]


def test_five_qubit_code_validates():
    code = StabilizerCode.from_paulis(
        FIVE_GENS, logical_x=["XXXXX"], logical_z=["ZZZZZ"], name="five_qubit",
    )
    code.validate()
    assert (code.n, code.k, code.m) == (5, 1, 4)


def test_validate_rejects_anticommuting_generators():
    with pytest.raises(ValueError, match="不对易"):
        StabilizerCode.from_paulis(
            ["XI", "ZI"], logical_x=["IX"], logical_z=["IZ"], name="bad",
        ).validate()


def test_validate_rejects_dependent_generators():
    # ZZI 与 IZZ 独立，但第三个 ZIZ = 前两者之积 → 秩亏
    with pytest.raises(ValueError, match="线性相关"):
        StabilizerCode.from_paulis(
            ["ZZI", "IZZ", "ZIZ"], logical_x=["XXX"], logical_z=["ZII"], name="bad",
        ).validate()


def test_validate_rejects_logical_not_in_normalizer():
    # XII 与稳定子 ZZI 反对易 → 不是合法 logical
    with pytest.raises(ValueError, match="logical"):
        StabilizerCode.from_paulis(
            ["ZZI", "IZZ"], logical_x=["XII"], logical_z=["ZII"], name="bad",
        ).validate()


def test_symplectic_product_single_row_batch_stays_array():
    # 回归测试：m=1 时 (1,2n) 批量输入不能被折叠成 0-d 标量
    rows = np.stack([pauli_to_gf2("ZZ")])
    out = symplectic_product(rows, pauli_to_gf2("XI"))
    assert out.shape == (1,)


def test_validate_with_single_generator():
    # 回归测试：m=1 的合法码应能通过 validate()，此前会在 normalizer
    # 检查处因 symplectic_product 把单行批量折叠成 0-d 标量而崩溃。
    # ZZ 与自身对易；XX、ZI 都与 ZZ 对易（在 normalizer 内）；
    # XX 与 ZI 反对易（辛积=1），满足 logical_x/logical_z 配对要求。
    code = StabilizerCode.from_paulis(
        ["ZZ"], logical_x=["XX"], logical_z=["ZI"], name="m1test",
    )
    code.validate()
    assert (code.n, code.k, code.m) == (2, 1, 1)


def test_syndrome_of_single_qubit_errors():
    code = StabilizerCode.from_paulis(
        FIVE_GENS, logical_x=["XXXXX"], logical_z=["ZZZZZ"], name="five_qubit",
    )
    # 距离 3 的码，任何权重 1 错误都必须给出非零综合征
    for q in range(5):
        for p in "XYZ":
            label = "I" * q + p + "I" * (4 - q)
            assert code.syndrome(pauli_to_gf2(label)).any()


def test_syndrome_of_stabilizer_is_zero():
    code = StabilizerCode.from_paulis(
        FIVE_GENS, logical_x=["XXXXX"], logical_z=["ZZZZZ"], name="five_qubit",
    )
    assert not code.syndrome(pauli_to_gf2(FIVE_GENS[0])).any()


def test_distance_of_five_qubit_code_is_three():
    code = StabilizerCode.from_paulis(
        FIVE_GENS, logical_x=["XXXXX"], logical_z=["ZZZZZ"], name="five_qubit",
    )
    assert code.distance() == 3


def test_distance_raises_at_cutoff_instead_of_guessing():
    code = StabilizerCode.from_paulis(
        FIVE_GENS, logical_x=["XXXXX"], logical_z=["ZZZZZ"], name="five_qubit",
    )
    with pytest.raises(ValueError, match="max_weight"):
        code.distance(max_weight=2)


def test_logical_class_and_verdict():
    code = StabilizerCode.from_paulis(
        FIVE_GENS, logical_x=["XXXXX"], logical_z=["ZZZZZ"], name="five_qubit",
    )
    assert code.verdict(pauli_to_gf2("IIIII")) == "corrected"
    # 稳定子元素仍算 corrected（残余落在稳定子群内）
    assert code.verdict(pauli_to_gf2(FIVE_GENS[0])) == "corrected"
    # 逻辑 X 算符本身 → logical_x
    assert code.verdict(pauli_to_gf2("XXXXX")) == "logical_x"
    assert code.verdict(pauli_to_gf2("ZZZZZ")) == "logical_z"
    cls = code.logical_class(pauli_to_gf2("XXXXX"))
    assert cls.shape == (1, 2)
    assert list(cls[0]) == [1, 0]
