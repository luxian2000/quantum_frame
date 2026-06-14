"""Noisy expressibility metrics for parameterized quantum circuits."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..backends.base import Backend
from ..noise.analysis import default_plus_state, estimate_noise_strength, evolve_density_gatewise
from ..noise.model import NoiseModel
from ..core.circuit import Circuit
from ..core.state import State
from .expressibility import (
    KL_Haar_divergence,
    MMD_relative,
    _count_total_parameters,
    _gaussian_kernel,
    _get_parametrized_gate_indices,
    _haar_population_distribution,
    _replace_circuit_parameters,
)


def _sample_circuit_probability_vectors(
    circuit: Circuit,
    backend: Backend,
    noise_model: Optional[NoiseModel],
    n_samples: int = 500,
    initial_state: Optional[State] = None,
) -> np.ndarray:
    dim = 1 << circuit.n_qubits
    if initial_state is None:
        initial_state = default_plus_state(circuit.n_qubits, backend)

    param_indices = _get_parametrized_gate_indices(circuit)
    total_params = _count_total_parameters(circuit, param_indices)
    samples = np.zeros((n_samples, dim), dtype=np.float64)

    for sample_index in range(n_samples):
        sampled_circuit = circuit
        if total_params > 0:
            params = np.random.uniform(0, 2 * np.pi, total_params)
            sampled_circuit = _replace_circuit_parameters(circuit, params)
        rho = evolve_density_gatewise(sampled_circuit, backend, initial_state, noise_model=noise_model)
        samples[sample_index] = rho.probabilities()

    return samples


def _apply_noise_to_fidelities(fidelities: np.ndarray, noise_strength: float = 0.01) -> np.ndarray:
    uniform_noise = np.random.uniform(size=fidelities.shape)
    noisy_fidelities = fidelities * (1 - noise_strength) + uniform_noise * noise_strength
    return np.clip(noisy_fidelities, 0, 1)


def KL_Haar_noisy(
    circuit: Circuit,
    backend: Optional[Backend] = None,
    n_samples: int = 500,
    n_bins: int = 1000,
    noise_model: Optional[NoiseModel] = None,
    initial_state: Optional[State] = None,
    use_density_matrix: bool = True,
) -> float:
    """Compute KL divergence between noisy PQC output distribution and Haar distribution."""
    if backend is None:
        from ..backends.numpy_backend import NumpyBackend

        backend = NumpyBackend()

    if circuit.n_qubits > 10:
        raise ValueError(
            f"n_qubits={circuit.n_qubits} too large for expressibility evaluation. "
            "Max recommended: 10 qubits."
        )

    dim = 1 << circuit.n_qubits
    p_haar = _haar_population_distribution(dim, n_bins)
    noisy_probability_vectors = _sample_circuit_probability_vectors(
        circuit,
        backend,
        noise_model if use_density_matrix else None,
        n_samples=n_samples,
        initial_state=initial_state,
    )
    noisy_probs = np.max(noisy_probability_vectors, axis=1)
    if noise_model is not None and not use_density_matrix:
        noisy_probs = _apply_noise_to_fidelities(noisy_probs, estimate_noise_strength(noise_model))

    hist_counts, _ = np.histogram(noisy_probs, bins=np.linspace(0, 1, n_bins + 1))
    p_pqc = hist_counts / n_samples
    epsilon = 1e-10
    p_pqc_safe = np.clip(p_pqc, epsilon, 1 - epsilon)
    p_haar_safe = np.clip(p_haar, epsilon, 1 - epsilon)
    p_pqc_safe = p_pqc_safe / np.sum(p_pqc_safe)
    p_haar_safe = p_haar_safe / np.sum(p_haar_safe)
    return float(np.sum(p_pqc_safe * np.log(p_pqc_safe / p_haar_safe)))


def MMD_noisy(
    circuit: Circuit,
    backend: Optional[Backend] = None,
    n_samples: int = 500,
    sigma: float = 0.1,
    noise_model: Optional[NoiseModel] = None,
    initial_state: Optional[State] = None,
    use_density_matrix: bool = True,
) -> float:
    """Compute MMD between noisy PQC probability vectors and Haar-like samples."""
    if backend is None:
        from ..backends.numpy_backend import NumpyBackend

        backend = NumpyBackend()

    dim = 1 << circuit.n_qubits
    noisy_probs = _sample_circuit_probability_vectors(
        circuit,
        backend,
        noise_model if use_density_matrix else None,
        n_samples=n_samples,
        initial_state=initial_state,
    )
    expo = np.random.exponential(scale=1.0, size=(n_samples, dim))
    haar_samples = expo / np.sum(expo, axis=1, keepdims=True)
    k_xx = _gaussian_kernel(noisy_probs, noisy_probs, sigma)
    k_xz = _gaussian_kernel(noisy_probs, haar_samples.astype(np.float64), sigma)
    k_zz = _gaussian_kernel(haar_samples.astype(np.float64), haar_samples.astype(np.float64), sigma)
    mmd_sq = np.mean(k_xx) - 2 * np.mean(k_xz) + np.mean(k_zz)
    return float(max(0.0, mmd_sq))


def comparative_expressibility(
    circuit: Circuit,
    backend: Optional[Backend] = None,
    n_samples: int = 500,
    noise_model: Optional[NoiseModel] = None,
    initial_state: Optional[State] = None,
) -> Dict[str, float]:
    """Compute ideal and noisy expressibility metrics for comparison."""
    if backend is None:
        from ..backends.numpy_backend import NumpyBackend

        backend = NumpyBackend()

    kl_ideal = KL_Haar_divergence(circuit, samples=n_samples, backend=backend)
    mmd_ideal = 1.0 - MMD_relative(circuit, samples=n_samples, backend=backend)
    result = {"kl_ideal": kl_ideal, "mmd_ideal": mmd_ideal}
    if noise_model is not None:
        kl_noisy = KL_Haar_noisy(circuit, backend=backend, n_samples=n_samples, noise_model=noise_model, initial_state=initial_state)
        mmd_noisy = MMD_noisy(circuit, backend=backend, n_samples=n_samples, noise_model=noise_model, initial_state=initial_state)
        result["kl_noisy"] = kl_noisy
        result["mmd_noisy"] = mmd_noisy
        result["noise_degradation_kl"] = (kl_noisy - kl_ideal) / kl_ideal if kl_ideal > 0 else 0.0
        result["noise_degradation_mmd"] = (mmd_noisy - mmd_ideal) / mmd_ideal if mmd_ideal > 0 else 0.0
    return result


def expressibility_score(
    circuit: Circuit,
    backend: Optional[Backend] = None,
    n_samples: int = 500,
    noise_model: Optional[NoiseModel] = None,
    method: str = "auto",
    initial_state: Optional[State] = None,
) -> float:
    """Unified ideal/noisy expressibility distance interface."""
    if backend is None:
        from ..backends.numpy_backend import NumpyBackend

        backend = NumpyBackend()

    if method == "auto":
        method = "kl" if (1 << circuit.n_qubits) <= 256 else "mmd"
    if noise_model is not None:
        if method == "kl":
            return KL_Haar_noisy(circuit, backend=backend, n_samples=n_samples, noise_model=noise_model, initial_state=initial_state)
        return MMD_noisy(circuit, backend=backend, n_samples=n_samples, noise_model=noise_model, initial_state=initial_state)
    if method == "kl":
        return KL_Haar_divergence(circuit, samples=n_samples, backend=backend)
    return 1.0 - MMD_relative(circuit, samples=n_samples, backend=backend)


__all__ = [
    "KL_Haar_noisy",
    "MMD_noisy",
    "comparative_expressibility",
    "expressibility_score",
]
