"""Factor 15 with a QFT period-finding demo.

This is a compact Shor-style example focused on how aicir ``State`` and
``Circuit`` work together:

1. Build a periodic counting-register state as ``State``.
2. Build the QFT as a ``Circuit``.
3. Evolve the state by the circuit unitary.
4. Recover the period from QFT peaks and compute the factors of 15.

The modular exponentiation oracle is summarized classically so the demo stays
small and keeps the spotlight on the State/Circuit interaction.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from aicir import NumpyBackend
from aicir.core import State
from aicir.universal import qft_circuit


def modular_order(a: int, n: int) -> int:
    """Return the smallest r > 0 such that a**r = 1 mod n."""

    value = 1
    for r in range(1, n + 1):
        value = (value * a) % n
        if value == 1:
            return r
    raise ValueError(f"could not find an order for a={a} modulo n={n}")


def periodic_state(n_qubits: int, period: int, offset: int, backend: NumpyBackend) -> State:
    """Create sum_k |offset + k*period> over the counting register."""

    dim = 1 << n_qubits
    amplitudes = np.zeros(dim, dtype=np.complex64)
    amplitudes[offset::period] = 1.0
    return State.from_array(amplitudes, n_qubits=n_qubits, backend=backend)


def sorted_probability_peaks(state: State, atol: float = 1e-8) -> list[tuple[int, float]]:
    """Return nonzero basis-state probabilities sorted by probability."""

    probs = np.asarray(state.probabilities(), dtype=float).reshape(-1)
    peaks = [(idx, float(prob)) for idx, prob in enumerate(probs) if prob > atol]
    return sorted(peaks, key=lambda item: item[1], reverse=True)


def recover_period_from_qft(
    peaks: list[tuple[int, float]],
    *,
    q: int,
    a: int,
    n: int,
) -> int:
    """Use continued fractions on QFT peaks and verify candidates."""

    for y, _prob in peaks:
        if y == 0:
            continue
        fraction = Fraction(y, q).limit_denominator(n)
        r = fraction.denominator
        if r > 0 and pow(a, r, n) == 1:
            return r
    raise ValueError("could not recover a valid period from QFT peaks")


def factors_from_period(a: int, r: int, n: int) -> tuple[int, int]:
    """Classical Shor post-processing from an even period."""

    if r % 2 != 0:
        raise ValueError(f"period must be even, got r={r}")

    half_power = pow(a, r // 2, n)
    left = math.gcd(half_power - 1, n)
    right = math.gcd(half_power + 1, n)
    if left in {1, n} or right in {1, n}:
        raise ValueError(f"period r={r} did not produce non-trivial factors")
    return tuple(sorted((left, right)))


def main() -> None:
    n = 15
    a = 2
    n_counting_qubits = 3
    q = 1 << n_counting_qubits
    backend = NumpyBackend()

    true_period = modular_order(a, n)
    state = periodic_state(
        n_qubits=n_counting_qubits,
        period=true_period,
        offset=0,
        backend=backend,
    )

    qft = qft_circuit(n_counting_qubits).bind_backend(backend)
    qft_state = state.evolve(qft.unitary())
    peaks = sorted_probability_peaks(qft_state)
    recovered_period = recover_period_from_qft(peaks, q=q, a=a, n=n)
    factors = factors_from_period(a, recovered_period, n)

    print("=== aicir QFT demo: factor 15 ===")
    print(f"N = {n}, a = {a}")
    print(f"Classical modular values a^x mod N: {[pow(a, x, n) for x in range(q)]}")
    print()

    print("Initial State for the counting register:")
    print(f"  {state.ket}")
    print()

    print("QFT Circuit:")
    qft.show()
    print()

    print("State after applying Circuit.unitary() to State.evolve(...):")
    print(f"  {qft_state.ket}")
    print()

    print("QFT probability peaks:")
    for y, prob in peaks:
        phase = Fraction(y, q)
        print(f"  y={y:>2}  y/Q={phase}  probability={prob:.3f}")
    print()

    print("Example direct State measurement after QFT:")
    print(f"  {qft_state.measure(shots=32)}")
    print()

    print(f"Recovered period r = {recovered_period}")
    print(f"Factors from gcd({a}^(r/2) +/- 1, {n}) = {factors[0]} * {factors[1]}")


if __name__ == "__main__":
    main()
