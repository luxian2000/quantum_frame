import math
import torch

from define_torch import *


def build_qft_circuit(num_qubits):
    """Build an n-qubit QFT circuit unitary using define_torch gate definitions."""
    gates = []

    for target in range(num_qubits):
        gates.append(hadamard(target))
        for control in range(target + 1, num_qubits):
            angle = math.pi / (2 ** (control - target))
            gates.append(crz(angle, target_qubit=target, control_qubits=[control]))

    for i in range(num_qubits // 2):
        gates.append(swap(i, num_qubits - 1 - i))

    return circuit(*gates, num_qubits=num_qubits), gates


def prepare_basis_state(bitstring):
    """Prepare a computational basis state |bitstring> from |0...0>."""
    num_qubits = len(bitstring)
    state = phi_0(num_qubits)

    prep_gates = []
    for qubit, bit in enumerate(bitstring):
        if bit == "1":
            prep_gates.append(pauli_x(qubit))

    if prep_gates:
        prep_u = circuit(*prep_gates, num_qubits=num_qubits)
        state = torch.matmul(prep_u, state)

    return state


def print_state(state, title, tol=1e-6):
    """Pretty-print amplitudes with basis labels."""
    num_qubits = int(round(math.log2(state.shape[0])))
    print(f"\n{title}")
    for idx in range(state.shape[0]):
        amp = state[idx, 0]
        if torch.abs(amp) > tol:
            basis = format(idx, f"0{num_qubits}b")
            print(f"|{basis}>: {amp}")


def main():
    num_qubits = 3
    input_bitstring = "101"

    input_state = prepare_basis_state(input_bitstring)
    qft_u, qft_gates = build_qft_circuit(num_qubits)
    output_state = torch.matmul(qft_u, input_state)

    print(f"QFT demo with {num_qubits} qubits")
    print(f"Input basis state: |{input_bitstring}>")
    print(f"Number of gates in QFT circuit: {len(qft_gates)}")

    print_state(input_state, "Input state amplitudes:")
    print_state(output_state, "Output state amplitudes after QFT:")

    # Verify unitarity: U^\dagger U = I
    unitary_check = matrix_product(dagger(qft_u), qft_u)
    is_unitary = torch.allclose(unitary_check, identity(num_qubits), atol=1e-5)
    print(f"\nUnitary check (U^dagger U == I): {is_unitary}")

    # Demonstrate inverse via U^\dagger and reconstruct the input state
    recovered_state = torch.matmul(dagger(qft_u), output_state)
    reconstruction_error = torch.max(torch.abs(recovered_state - input_state)).item()
    print(f"Reconstruction max error after inverse QFT: {reconstruction_error:.3e}")


if __name__ == "__main__":
    main()
