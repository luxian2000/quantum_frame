"""生成元感知的 select_diff（qml 成熟化 #3）。

激发门（single/double_excitation）生成元谱 {-1,0,1} 需四项移位规则 psr4；
两项旋转门（rx/ry/rz/rzz/...）用 psr。此前 auto 对激发线路静默选 psr →
梯度错误。select_diff 现按线路门的 GateSpec shift_rule 把 psr 升级为 psr4。
"""

import numpy as np
import pytest

from aicir import Circuit, Hamiltonian, PauliString, cx, hadamard, rx, ry, rzz
from aicir import single_excitation, double_excitation
from aicir.backends.numpy_backend import NumpyBackend
from aicir.qml import qfun, select_diff
from aicir.qml.diff import circuit_shift_rule


def test_circuit_shift_rule_classifies():
    rot = Circuit(rx(0.1, 0), ry(0.2, 1), rzz(0.3, 0, 1), cx(1, [0]), n_qubits=2)
    assert circuit_shift_rule(rot) == "two_term"

    exc = Circuit(single_excitation(0.3, 0, 1), n_qubits=2)
    assert circuit_shift_rule(exc) == "four_term"

    dexc = Circuit(double_excitation(0.2, 0, 1, 2, 3), n_qubits=4)
    assert circuit_shift_rule(dexc) == "four_term"

    mixed = Circuit(rx(0.1, 0), single_excitation(0.3, 0, 1), n_qubits=2)
    assert circuit_shift_rule(mixed) == "mixed"

    const = Circuit(hadamard(0), cx(1, [0]), n_qubits=2)
    assert circuit_shift_rule(const) is None


def test_select_diff_upgrades_psr_to_psr4_for_excitation():
    exc = Circuit(single_excitation(0.3, 0, 1), n_qubits=2)
    assert select_diff(backend=NumpyBackend(), circuit=exc) == "psr4"


def test_select_diff_keeps_psr_for_rotations():
    rot = Circuit(rx(0.1, 0), ry(0.2, 1), n_qubits=2)
    assert select_diff(backend=NumpyBackend(), circuit=rot) == "psr"


def test_select_diff_mixed_falls_to_fd():
    mixed = Circuit(rx(0.1, 0), single_excitation(0.3, 0, 1), n_qubits=2)
    assert select_diff(backend=NumpyBackend(), circuit=mixed) == "fd"


def test_select_diff_backward_compatible_without_circuit():
    # 不传 circuit：行为不变（numpy 无 shots → psr）
    assert select_diff(backend=NumpyBackend()) == "psr"


def test_auto_differential_uses_psr4_for_excitation():
    # 判别电路（同 test_psr4）：|00> ─H(q1)→ (|00>+|01>)/√2，再 single_excitation，
    # 观测量 X⊗I 暴露半频分量，使两项 psr 与四项 psr4 显著不同。
    obs = Hamiltonian([PauliString("X", n_qubits=2, qubits=[0])])

    def body(theta, diff):
        @qfun(device="numpy", differential=diff, observable=obs)
        def f(t):
            return Circuit(hadamard(1), single_excitation(t[0], 0, 1), n_qubits=2)
        return f

    x = np.array([0.6])
    g_auto = float(np.asarray(body(x, "auto").grad(x)).reshape(-1)[0])
    g_psr4 = float(np.asarray(body(x, "psr4").grad(x)).reshape(-1)[0])
    g_psr = float(np.asarray(body(x, "psr").grad(x)).reshape(-1)[0])

    # auto 现应等于 psr4（正确），且与 psr（错误）显著不同
    assert abs(g_auto - g_psr4) < 1e-6
    assert abs(g_auto - g_psr) > 1e-3
