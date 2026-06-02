"""Demo: encode (1, 2, 3, 4) with three basic encoders."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

# Allow running this file directly from the repository root.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aicir.encoder.amplitude import AmplitudeEncoder
from aicir.encoder.angle import AngleEncoder
from aicir.encoder.basis import BasisEncoder
from aicir.core.io.qasm import circuit_to_qasm3
from aicir.optimizer import optimize_basic


def _angle_equivalent(a, b, atol=1e-6):
    """Check angular equivalence under 2*pi periodicity."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = (a - b + np.pi) % (2 * np.pi) - np.pi
    return np.allclose(diff, 0.0, atol=atol), diff


def demo_amplitude(data):
    encoder = AmplitudeEncoder()
    circuit, state = encoder.encode(data, cir="dict")
    decoded = encoder.decode(state)

    print("=== AmplitudeEncoder ===")
    print("n_qubits:", state.n_qubits)
    print("circuit gates:", circuit.gates)
    print("state:", state.format())
    print("decoded amplitudes(real):", np.round(decoded, 6))
    print()


def demo_angle(data):
    encoder = AngleEncoder()
    circuit, state = encoder.encode(data, cir="dict")
    decoded = encoder.decode(state)
    ok, diff = _angle_equivalent(np.asarray(data, dtype=float), decoded)

    print("=== AngleEncoder ===")
    print("n_qubits:", state.n_qubits)
    print("circuit gates:", circuit.gates)
    print("state:", state.format())
    print("decoded angles:", np.round(decoded, 6))
    print("2*pi periodic equivalence:", ok)
    print("wrapped angle diff:", np.round(diff, 6))
    print(
        "note: current decode uses marginal |1> probability inversion, "
        "which is not one-to-one for all angles."
    )
    print()


def demo_basis(data, redundant=False):
    encoder = BasisEncoder(redundant=redundant)
    circuit, state = encoder.encode(data, cir="dict")
    # Optimize the generated circuit before exporting
    optimized_circuit = optimize_basic(circuit)
    qasm3 = circuit_to_qasm3(optimized_circuit)
    # Run one more optimization pass on exported QASM text
    qasm3 = optimize_basic(qasm3, input_type="qasm")
    qasm_path = pathlib.Path(__file__).with_name(
        f"encode_1234_demo_redundant_{str(redundant).lower()}.qasm"
    )
    # Always (re)write the QASM file on each run: remove existing file then write
    if qasm_path.exists():
        try:
            qasm_path.unlink()
        except Exception:
            pass
    qasm_path.write_text(qasm3, encoding="utf-8")
    decoded = encoder.decode(state)
    probs = state.probabilities()

    print(f"=== BasisEncoder (redundant={redundant}) ===")
    print("input array:", data)
    print("n_qubits:", state.n_qubits)
    print("original circuit gates:", circuit.gates)
    print("optimized circuit gates:", getattr(optimized_circuit, 'gates', optimized_circuit))
    print("state:", state.format())
    print("probabilities:", np.round(probs, 6))
    print("OpenQASM 3.0 file:", qasm_path, "(written/overwritten)")
    print("decoded bits:", decoded)
    print()


def main():
    data = (1, 1, 2, 4, 4, 3)
    print("Input data:", data)
    print()

    demo_amplitude(data)
    demo_angle(data)
    demo_basis(data, redundant=False)
    demo_basis(data, redundant=True)


if __name__ == "__main__":
    main()
