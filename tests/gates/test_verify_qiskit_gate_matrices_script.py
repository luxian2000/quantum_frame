from pathlib import Path

import pytest

pytest.importorskip("qiskit")

from tests.gates import verify_qiskit_gate_matrices as verifier


def test_verify_qiskit_gate_matrices_writes_matrix_report(tmp_path):
    output_path = tmp_path / "gate_matrices.txt"

    exit_code = verifier.main(["--output", str(output_path)])

    text = output_path.read_text(encoding="utf-8")
    assert exit_code == 0
    assert "Gate: pauli_x" in text
    assert "Qiskit matrix:" in text
    assert "aicir matrix:" in text
    assert "All 26 aicir basic gate matrix checks match Qiskit." in text
