import pytest

from aicir.qec.code import pauli_to_gf2
from aicir.qec.codes import CODES, get_code, register_code

# (名字, 构造 kwargs, 期望 n, 期望 k, 期望 distance())
BUILTINS = [
    ("repetition", {"d": 3, "basis": "Z"}, 3, 1, 1),   # 见下方注释：重复码码距是 1
    ("repetition", {"d": 5, "basis": "Z"}, 5, 1, 1),
    ("repetition", {"d": 3, "basis": "X"}, 3, 1, 1),
    ("five_qubit", {}, 5, 1, 3),
    ("steane", {}, 7, 1, 3),
    ("shor", {}, 9, 1, 3),
    ("surface", {"d": 3}, 9, 1, 3),
]


@pytest.mark.parametrize("name,kwargs,n,k,dist", BUILTINS)
def test_builtin_code_validates_and_has_expected_shape(name, kwargs, n, k, dist):
    code = get_code(name, **kwargs)
    code.validate()
    assert (code.n, code.k) == (n, k)
    # 重复码的 distance() 是 1：它的稳定子全是 Z 型，logical_z = Z₀ 权重为 1，
    # 对 Z 错误无保护。参数 d 只是它针对 X 噪声的有效距离。
    assert code.distance() == dist


@pytest.mark.parametrize("name,kwargs,n,k,dist", BUILTINS)
def test_builtin_code_weight_one_errors_are_detected_in_protected_basis(name, kwargs, n, k, dist):
    code = get_code(name, **kwargs)
    if name == "repetition":
        # 重复码只保护一个基：ZZ 型稳定子检测 X 错误，XX 型稳定子检测 Z 错误。
        # 同型错误与稳定子对易，必然漏检——已实测：basis="X" 时 X 错误在 0 个比特上被检测到。
        bases = ["X"] if kwargs["basis"] == "Z" else ["Z"]
    else:
        bases = ["X", "Y", "Z"]
    for q in range(code.n):
        for p in bases:
            label = "I" * q + p + "I" * (code.n - q - 1)
            assert code.syndrome(pauli_to_gf2(label)).any(), f"{name} 漏检 {label}"


def test_surface_code_carries_coords():
    code = get_code("surface", d=3)
    assert len(code.coords) == 9
    assert code.coords[0] == (0, 0)
    assert code.coords[8] == (2, 2)


def test_registry_roundtrip():
    def _builder():
        from aicir.qec.code import StabilizerCode
        return StabilizerCode.from_paulis(
            ["ZZ"], logical_x=["XX"], logical_z=["ZI"], name="tiny",
        )

    register_code("tiny_test_code", _builder)
    assert "tiny_test_code" in CODES
    assert get_code("tiny_test_code").n == 2


def test_unknown_code_raises_listing_available():
    with pytest.raises(KeyError, match="five_qubit"):
        get_code("no_such_code")
