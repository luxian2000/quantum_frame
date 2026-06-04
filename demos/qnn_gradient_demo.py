"""Demo: QNN training with gradient and gradient-free methods.

Trains a 2-qubit variational quantum circuit to minimize the expectation value
of the ZZ observable, comparing seven gradient / gradient-free strategies from
aicir.qml:

  Gradient methods  : psr, fd, spsa, ad (adjoint)
  Natural gradients : qng, dqng
  Gradient-free     : rotosolve

Run from the repository root:
    python -m demos.qnn_gradient_demo
    python -m demos.qnn_gradient_demo --show
"""

from __future__ import annotations

import argparse

import numpy as np

from aicir import Circuit, NumpyBackend, State, cx, ry
from aicir.qml import ad, dqng, fd, psr, qng, rotosolve, spsa

from ._visual_demo_utils import add_common_visual_args, configure_matplotlib, save_figure

# ── Problem setup ────────────────────────────────────────────────────────────

N_QUBITS = 2
N_PARAMS = 4
N_STEPS = 60
LR = 0.25
RNG_SEED = 42

# Observable: H = Z⊗Z, eigenvalues {-1, +1}, ground state energy = -1
ZZ = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.complex64)


def build_circuit(theta: np.ndarray) -> Circuit:
    """Two-layer Ry ansatz with one entangling CNOT."""
    return Circuit(
        ry(float(theta[0]), 0),
        ry(float(theta[1]), 1),
        cx(1, [0]),  # CNOT: control=0, target=1
        ry(float(theta[2]), 0),
        ry(float(theta[3]), 1),
        n_qubits=N_QUBITS,
    )


def expectation(theta: np.ndarray, backend: NumpyBackend) -> float:
    sv = (
        State.zero_state(N_QUBITS, backend)
        .evolve(build_circuit(theta).unitary())
        .to_numpy()
        .reshape(-1)
    )
    return float(np.real(sv.conj() @ ZZ @ sv))


def state_vector(theta: np.ndarray, backend: NumpyBackend) -> np.ndarray:
    return (
        State.zero_state(N_QUBITS, backend)
        .evolve(build_circuit(theta).unitary())
        .to_numpy()
        .reshape(-1)
    )


# ── Training loops ───────────────────────────────────────────────────────────


def run_gradient_descent(
    grad_fn,
    objective,
    init: np.ndarray,
    n_steps: int = N_STEPS,
    lr: float = LR,
) -> np.ndarray:
    theta = init.copy()
    history = [objective(theta)]
    for _ in range(n_steps):
        theta = theta - lr * grad_fn(theta)
        history.append(objective(theta))
    return np.array(history)


def run_rotosolve(objective, init: np.ndarray, n_steps: int = N_STEPS) -> np.ndarray:
    theta = init.copy()
    history = [objective(theta)]
    for _ in range(n_steps):
        theta = rotosolve(objective, theta, n_sweeps=1)
        history.append(objective(theta))
    return np.array(history)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="QNN gradient method comparison demo.")
    add_common_visual_args(parser)
    parser.add_argument("--steps", type=int, default=N_STEPS, help="Optimisation steps.")
    parser.add_argument("--lr", type=float, default=LR, help="Learning rate for gradient methods.")
    args = parser.parse_args()

    backend = NumpyBackend()
    rng = np.random.default_rng(RNG_SEED)
    init = rng.uniform(-np.pi, np.pi, N_PARAMS)

    objective = lambda theta: expectation(theta, backend)
    state_fn = lambda theta: state_vector(theta, backend)

    print("=== QNN gradient method comparison ===")
    print(f"  Circuit : 2-qubit, {N_PARAMS} Ry parameters, 1 CNOT")
    print(f"  Observable: Z⊗Z  (ground state energy = -1.0)")
    print(f"  Steps   : {args.steps}   LR: {args.lr}")
    print(f"  Initial energy: {objective(init):.4f}")
    print()

    histories: dict[str, np.ndarray] = {}

    # 1. Parameter-shift rule (exact gradient)
    print("Running PSR ...", end=" ", flush=True)
    histories["PSR"] = run_gradient_descent(
        lambda t: psr(objective, t), objective, init, args.steps, args.lr
    )
    print(f"final E = {histories['PSR'][-1]:.4f}")

    # 2. Finite differences (central, approximate gradient)
    print("Running FD  ...", end=" ", flush=True)
    histories["FD"] = run_gradient_descent(
        lambda t: fd(objective, t, eps=1e-3), objective, init, args.steps, args.lr
    )
    print(f"final E = {histories['FD'][-1]:.4f}")

    # 3. SPSA (stochastic, 4 perturbation samples per step)
    print("Running SPSA...", end=" ", flush=True)
    spsa_rng = np.random.default_rng(RNG_SEED)
    histories["SPSA"] = run_gradient_descent(
        lambda t: spsa(objective, t, eps=0.05, n_samples=4, rng=spsa_rng),
        objective,
        init,
        args.steps,
        args.lr,
    )
    print(f"final E = {histories['SPSA'][-1]:.4f}")

    # 4. Adjoint differentiation (one forward + one backward pass)
    print("Running AD  ...", end=" ", flush=True)
    histories["AD"] = run_gradient_descent(
        lambda t: ad(build_circuit(t), ZZ, backend=backend),
        objective,
        init,
        args.steps,
        args.lr,
    )
    print(f"final E = {histories['AD'][-1]:.4f}")

    # 5. Quantum Natural Gradient (full QFIM)
    print("Running QNG ...", end=" ", flush=True)
    histories["QNG"] = run_gradient_descent(
        lambda t: qng(objective, state_fn, t, damping=1e-4),
        objective,
        init,
        args.steps,
        args.lr,
    )
    print(f"final E = {histories['QNG'][-1]:.4f}")

    # 6. Diagonal QNG (diagonal QFIM approximation)
    print("Running DQNG...", end=" ", flush=True)
    histories["DQNG"] = run_gradient_descent(
        lambda t: dqng(objective, state_fn, t, damping=1e-4),
        objective,
        init,
        args.steps,
        args.lr,
    )
    print(f"final E = {histories['DQNG'][-1]:.4f}")

    # 7. Rotosolve (gradient-free, exact coordinate minimization)
    print("Running Rotosolve...", end=" ", flush=True)
    histories["Rotosolve"] = run_rotosolve(objective, init, args.steps)
    print(f"final E = {histories['Rotosolve'][-1]:.4f}")

    print()
    print(f"  Ground state energy (exact): -1.0000")

    plt = configure_matplotlib(args.show)
    import matplotlib.pyplot as mpl_plt

    colors = {
        "PSR": "#2196F3",
        "FD": "#4CAF50",
        "SPSA": "#FF9800",
        "AD": "#9C27B0",
        "QNG": "#F44336",
        "DQNG": "#E91E63",
        "Rotosolve": "#795548",
    }
    linestyles = {
        "PSR": "-",
        "FD": "--",
        "SPSA": ":",
        "AD": "-.",
        "QNG": "-",
        "DQNG": "--",
        "Rotosolve": (0, (3, 1, 1, 1)),
    }

    fig, ax = mpl_plt.subplots(figsize=(8, 5))
    for name, hist in histories.items():
        ax.plot(
            hist,
            label=name,
            color=colors[name],
            linestyle=linestyles[name],
            linewidth=1.8,
        )
    ax.axhline(-1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5, label="Ground state (−1)")
    ax.set_xlabel("Optimisation step")
    ax.set_ylabel("Energy ⟨Z⊗Z⟩")
    ax.set_title("QNN training: gradient vs gradient-free methods")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.set_xlim(0, args.steps)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    saved = save_figure(fig, args.output_dir, "qnn_gradient_demo.png")
    print(f"\nSaved figure: {saved}")

    if args.show:
        mpl_plt.show()
    else:
        mpl_plt.close("all")


if __name__ == "__main__":
    main()
