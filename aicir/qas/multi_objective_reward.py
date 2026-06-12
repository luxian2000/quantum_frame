"""
aicir/qas/multi_objective_reward.py

Multi-objective reward function for noise-aware quantum architecture search.

This module provides a comprehensive reward function that balances:
1. Expressibility - circuit's ability to generate diverse states
2. Trainability - gradient behavior (avoiding barren plateaus)
3. Noise robustness - stability under noise
4. Hardware efficiency - compatibility with target hardware

References:
- Sim et al., arXiv:1905.10876 (expressibility)
- Holmes et al., PRX Quantum 2022 (barren plateaus)
- Zhang et al., arXiv:2405.18837 (ML-based evaluation)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, List

import numpy as np

from ..core.circuit import Circuit
from ..channel.backends.base import Backend
from ..channel.noise.model import NoiseModel
from ..ir import (
    circuit_instruction_count,
    circuit_instructions,
    instruction_name,
    instruction_parameter,
)

from ..metrics.expressibility import (
    KL_Haar_divergence,
    MMD_relative,
)
from ..metrics.noisy_expressibility import (
    KL_Haar_noisy,
    MMD_noisy,
)
from ..channel.noise.analysis import NoiseSensitivityResult, noise_sensitivity


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class RewardWeights:
    """
    Weights for multi-objective reward function.

    All weights should be non-negative. They will be automatically normalized
    to sum to 1.0 unless normalize=False.
    """
    expressibility: float = 0.25
    trainability: float = 0.25
    noise_robustness: float = 0.25
    hardware_efficiency: float = 0.25
    normalize: bool = True

    def __post_init__(self):
        """Validate and normalize weights."""
        if self.expressibility < 0 or self.trainability < 0:
            raise ValueError("Weights must be non-negative")
        if self.noise_robustness < 0 or self.hardware_efficiency < 0:
            raise ValueError("Weights must be non-negative")

        if self.normalize:
            total = self.expressibility + self.trainability
            total += self.noise_robustness + self.hardware_efficiency
            if total > 0:
                self.expressibility /= total
                self.trainability /= total
                self.noise_robustness /= total
                self.hardware_efficiency /= total

    def to_dict(self) -> Dict[str, float]:
        """Return as dictionary."""
        return {
            "expressibility": self.expressibility,
            "trainability": self.trainability,
            "noise_robustness": self.noise_robustness,
            "hardware_efficiency": self.hardware_efficiency,
        }


# =============================================================================
# Individual Score Classes
# =============================================================================

class ExpressibilityScore:
    """
    Compute expressibility score for a circuit.

    Lower KL/MMD divergence from Haar = higher expressibility.
    We convert to a [0, 1] score where 1 = most expressive.
    """

    def __init__(
        self,
        n_samples: int = 300,
        method: str = "auto",
        n_bins: int = 1000,
    ):
        """
        Args:
            n_samples: Number of samples for Monte Carlo estimation
            method: "kl", "mmd", or "auto"
            n_bins: Number of bins for KL divergence
        """
        self.n_samples = n_samples
        self.method = method
        self.n_bins = n_bins

    def compute(
        self,
        circuit: Circuit,
        backend: Backend,
    ) -> float:
        """
        Compute expressibility score [0, 1].

        Returns:
            float: Score where 1 = most expressive (closest to Haar)
        """
        if not any(instruction_parameter(gate) is not None for gate in circuit_instructions(circuit)):
            return self._structural_proxy(circuit)

        try:
            distance = KL_Haar_divergence(
                circuit,
                samples=self.n_samples,
                n_bins=self.n_bins,
                backend=backend,
            )
            score = np.exp(-distance)
        except Exception:
            try:
                score = MMD_relative(
                    circuit,
                    samples=self.n_samples,
                    backend=backend,
                )
            except Exception:
                score = self._structural_proxy(circuit)

        return float(np.clip(score, 0, 1))

    def _structural_proxy(self, circuit: Circuit) -> float:
        """Fast fallback for non-parameterized or invalid intermediate circuits."""
        n_gates = circuit_instruction_count(circuit)
        if n_gates == 0:
            return 0.0
        two_qubit = sum(
            1
            for gate in circuit_instructions(circuit)
            if instruction_name(gate) in {"cx", "cnot", "cy", "cz", "crx", "cry", "crz", "swap", "toffoli", "ccnot", "rzz", "rxx"}
        )
        entangling_ratio = two_qubit / n_gates
        depth_proxy = n_gates / max(1, circuit.n_qubits)
        return float(np.clip(0.6 * entangling_ratio + 0.4 * (1.0 - np.exp(-depth_proxy / 4.0)), 0.0, 1.0))

    def compute_noisy(
        self,
        circuit: Circuit,
        backend: Backend,
        noise_model: NoiseModel,
    ) -> float:
        """Compute expressibility under noise."""
        try:
            distance = KL_Haar_noisy(
                circuit,
                backend=backend,
                n_samples=self.n_samples,
                noise_model=noise_model,
                use_density_matrix=True,
            )
            score = np.exp(-distance)
        except Exception:
            try:
                distance = MMD_noisy(
                    circuit,
                    backend=backend,
                    n_samples=self.n_samples,
                    noise_model=noise_model,
                )
                score = np.exp(-distance)
            except Exception:
                score = self._structural_proxy(circuit)
        return float(np.clip(score, 0, 1))


class TrainabilityScore:
    """
    Compute trainability score based on gradient behavior.

    This is a simplified estimator for barren plateau phenomenon.
    In practice, full gradient analysis requires expensive computation.

    Reference: Holmes et al., PRX Quantum 2022
    """

    def __init__(
        self,
        n_samples: int = 50,
        param_init_range: float = 0.1,
    ):
        """
        Args:
            n_samples: Number of random parameter samples
            param_init_range: Initial parameter range
        """
        self.n_samples = n_samples
        self.param_init_range = param_init_range

    def compute(
        self,
        circuit: Circuit,
        backend: Backend,
    ) -> float:
        """
        Estimate trainability score [0, 1].

        This is a heuristic based on circuit properties:
        - Shallow circuits score higher
        - Hardware-efficient circuits score higher
        - Too many 2-qubit gates -> lower score

        Returns:
            float: Score where 1 = most trainable
        """
        n_qubits = circuit.n_qubits
        n_gates = circuit_instruction_count(circuit)

        # Count gate types
        single_qubit_ops = 0
        two_qubit_ops = 0

        for gate in circuit_instructions(circuit):
            gate_type = instruction_name(gate)
            if gate_type in {"cx", "cnot", "cy", "cz", "crx", "cry", "crz",
                            "swap", "toffoli", "ccnot", "rzz", "rxx"}:
                two_qubit_ops += 1
            else:
                single_qubit_ops += 1

        # Heuristic 1: Circuit depth factor
        # Deeper circuits are harder to train (barren plateau)
        # Use gate_count / n_qubits as proxy for depth
        depth_proxy = n_gates / n_qubits if n_qubits > 0 else 1
        depth_score = np.exp(-depth_proxy / 10)  # Decay with depth

        # Heuristic 2: 2-qubit gate density
        # More 2-qubit gates = more entanglement = potential barren plateau
        if n_gates > 0:
            two_qubit_ratio = two_qubit_ops / n_gates
        else:
            two_qubit_ratio = 0
        entanglement_score = np.exp(-2 * two_qubit_ratio)

        # Heuristic 3: Parameter count efficiency
        # Fewer parameters per qubit = easier to train
        if n_qubits > 0:
            params_per_qubit = single_qubit_ops / n_qubits
        else:
            params_per_qubit = 0
        param_score = np.exp(-params_per_qubit / 5)

        # Combine scores
        score = 0.4 * depth_score + 0.4 * entanglement_score + 0.2 * param_score

        return float(np.clip(score, 0, 1))

    def estimate_gradient_variance(
        self,
        circuit: Circuit,
        backend: Backend,
    ) -> float:
        """
        Estimate variance of gradients (expensive, not used in quick evaluation).

        This would require computing gradients for random parameter initializations.
        """
        # Placeholder for expensive gradient computation
        # In practice, would use parameter shift rule or finite differences
        return self.compute(circuit, backend)


class NoiseRobustnessScore:
    """
    Compute noise robustness score.

    This measures how well a circuit maintains performance under noise.
    """

    def __init__(
        self,
        noise_model: Optional[NoiseModel] = None,
        n_samples: int = 200,
        fidelity_threshold: float = 0.9,
    ):
        """
        Args:
            noise_model: Default noise model to use
            n_samples: Samples for noisy expressibility
            fidelity_threshold: Threshold for considering circuit "functional"
        """
        self.noise_model = noise_model
        self.n_samples = n_samples
        self.fidelity_threshold = fidelity_threshold

    def compute(
        self,
        circuit: Circuit,
        backend: Backend,
        noise_model: Optional[NoiseModel] = None,
    ) -> float:
        """
        Compute noise robustness score [0, 1].

        Score = 1 if circuit is equally expressive under noise
              = 0 if noise completely destroys expressibility

        Args:
            circuit: Circuit to evaluate
            backend: Backend to use
            noise_model: Override default noise model

        Returns:
            float: Score where 1 = most robust
        """
        nm = noise_model if noise_model is not None else self.noise_model

        if nm is None:
            # No noise model - assume ideal robustness
            return 1.0

        try:
            # Compare ideal vs noisy expressibility
            ideal_kl = KL_Haar_divergence(circuit, samples=self.n_samples, backend=backend)

            # Estimate noisy KL
            # This uses a simplified model for speed
            noise_strength = self._estimate_noise_strength(nm)
            noisy_kl = ideal_kl * (1 + noise_strength * 5)  # Simplified model

            # Score based on degradation
            if ideal_kl > 0:
                degradation = (noisy_kl - ideal_kl) / ideal_kl
            else:
                degradation = 0

            # Convert to score: less degradation = higher score
            score = np.exp(-max(0, degradation))

        except Exception:
            # Fallback to sensitivity-based score
            try:
                sensitivity = noise_sensitivity(
                    circuit, backend, nm, n_samples=100
                )
                # Lower fidelity loss = higher score
                score = 1 - sensitivity.avg_fidelity_loss
            except Exception:
                score = 0.5  # Default moderate score

        return float(np.clip(score, 0, 1))

    def _estimate_noise_strength(self, noise_model: NoiseModel) -> float:
        """Extract typical noise strength from model."""
        total = 0.0
        count = 0
        for rule in noise_model.rules:
            channel = rule.channel
            p = getattr(channel, 'p', None)
            if p is not None:
                total += p
                count += 1
            gamma = getattr(channel, 'gamma', None)
            if gamma is not None:
                total += gamma
                count += 1
        return total / count if count > 0 else 0.01


class HardwareEfficiencyScore:
    """
    Compute hardware efficiency score.

    This measures how well a circuit maps to target hardware constraints.
    """

    def __init__(
        self,
        native_gates: Optional[List[str]] = None,
        max_depth: int = 100,
        connectivity: Optional[Dict[int, List[int]]] = None,
    ):
        """
        Args:
            native_gates: List of native gate types for target hardware
            max_depth: Maximum allowed circuit depth
            connectivity: Dict mapping qubit -> list of connected qubits
        """
        # Default: generic universal set
        self.native_gates = native_gates or [
            "hadamard", "rx", "ry", "rz", "cx", "cnot"
        ]
        self.max_depth = max_depth
        # Full connectivity by default
        self.connectivity = connectivity

    def compute(
        self,
        circuit: Circuit,
        backend: Backend,
    ) -> float:
        """
        Compute hardware efficiency score [0, 1].

        Factors:
        - Native gate usage (no decomposed gates)
        - Depth within hardware limits
        - Connectivity-aware gate placement

        Returns:
            float: Score where 1 = most hardware-efficient
        """
        # Factor 1: Native gate ratio
        native_count = 0
        for gate in circuit_instructions(circuit):
            gate_type = instruction_name(gate)
            if gate_type in self.native_gates:
                native_count += 1

        n_gates = circuit_instruction_count(circuit)
        if n_gates > 0:
            native_ratio = native_count / n_gates
        else:
            native_ratio = 1.0

        # Factor 2: Depth efficiency
        depth_proxy = n_gates / circuit.n_qubits if circuit.n_qubits > 0 else 1
        depth_score = min(1.0, self.max_depth / max(1, depth_proxy * 10))

        # Factor 3: 2-qubit gate efficiency
        # Fewer 2-qubit gates = better for most hardware
        two_qubit_count = sum(
            1 for g in circuit_instructions(circuit)
            if instruction_name(g) in {"cx", "cnot", "cy", "cz", "crx", "cry", "crz",
                                       "swap", "toffoli", "ccnot", "rzz", "rxx"}
        )

        if circuit.n_qubits > 0:
            two_qubit_ratio = two_qubit_count / circuit.n_qubits
        else:
            two_qubit_ratio = 0

        # Ideal: 1-2 CNOTs per qubit pair
        efficiency_score = np.exp(-two_qubit_ratio / 3)

        # Combine
        score = 0.4 * native_ratio + 0.3 * depth_score + 0.3 * efficiency_score

        return float(np.clip(score, 0, 1))


# =============================================================================
# Multi-Objective Reward Function
# =============================================================================

class MultiObjectiveReward:
    """
    Multi-objective reward function for quantum architecture search.

    Combines expressibility, trainability, noise robustness, and hardware
    efficiency into a single reward signal for RL agents.
    """

    def __init__(
        self,
        weights: Optional[RewardWeights] = None,
        expressibility_score: Optional[ExpressibilityScore] = None,
        trainability_score: Optional[TrainabilityScore] = None,
        noise_robustness_score: Optional[NoiseRobustnessScore] = None,
        hardware_efficiency_score: Optional[HardwareEfficiencyScore] = None,
        fidelity_weight: float = 0.2,
        penalty_weight: float = 0.05,
    ):
        """
        Initialize multi-objective reward function.

        Args:
            weights: Reward weight configuration
            expressibility_score: Expressibility evaluator
            trainability_score: Trainability evaluator
            noise_robustness_score: Noise robustness evaluator
            hardware_efficiency_score: Hardware efficiency evaluator
            fidelity_weight: Weight for task fidelity (if applicable)
            penalty_weight: Base penalty for circuit complexity
        """
        self.weights = weights or RewardWeights()

        self.expressibility_score = expressibility_score or ExpressibilityScore()
        self.trainability_score = trainability_score or TrainabilityScore()
        self.noise_robustness_score = noise_robustness_score or NoiseRobustnessScore()
        self.hardware_efficiency_score = hardware_efficiency_score or HardwareEfficiencyScore()

        self.fidelity_weight = fidelity_weight
        self.penalty_weight = penalty_weight

    def __call__(
        self,
        circuit: Circuit,
        backend: Backend,
        fidelity: Optional[float] = None,
        noise_model: Optional[NoiseModel] = None,
    ) -> float:
        """
        Compute multi-objective reward for a circuit.

        Args:
            circuit: Circuit to evaluate
            backend: Backend to use
            fidelity: Task fidelity (if target state is defined)
            noise_model: Noise model for robustness evaluation

        Returns:
            float: Combined reward value
        """
        # Compute individual scores
        expr = self.expressibility_score.compute(circuit, backend)
        train = self.trainability_score.compute(circuit, backend)

        # Noise robustness (with optional noise model)
        if noise_model is not None:
            noise_rob = self.noise_robustness_score.compute(
                circuit, backend, noise_model
            )
        else:
            noise_rob = self.noise_robustness_score.compute(
                circuit, backend, self.noise_robustness_score.noise_model
            )

        hw_eff = self.hardware_efficiency_score.compute(circuit, backend)

        # Combine using weights
        reward = (
            self.weights.expressibility * expr +
            self.weights.trainability * train +
            self.weights.noise_robustness * noise_rob +
            self.weights.hardware_efficiency * hw_eff
        )

        # Add fidelity component if provided
        if fidelity is not None:
            reward += self.fidelity_weight * fidelity

        # Add complexity penalty (prefer simpler circuits)
        complexity_penalty = self._compute_complexity_penalty(circuit)
        reward -= self.penalty_weight * complexity_penalty

        return float(reward)

    def _compute_complexity_penalty(self, circuit: Circuit) -> float:
        """
        Compute complexity penalty to favor simpler circuits.

        Penalizes:
        - Large number of gates
        - Large circuit depth
        - Many 2-qubit operations
        """
        n_gates = circuit_instruction_count(circuit)

        # Normalize by qubit count
        if circuit.n_qubits > 0:
            depth_proxy = n_gates / circuit.n_qubits
        else:
            depth_proxy = 0

        # Count 2-qubit gates
        two_qubit = sum(
            1 for g in circuit_instructions(circuit)
            if instruction_name(g) in {"cx", "cnot", "cy", "cz", "crx", "cry", "crz",
                                       "swap", "toffoli", "ccnot", "rzz", "rxx"}
        )

        # Combined penalty
        penalty = 0.3 * (n_gates / 20) + 0.4 * (depth_proxy / 5) + 0.3 * (two_qubit / 5)

        return float(penalty)

    def get_scores_detail(
        self,
        circuit: Circuit,
        backend: Backend,
        noise_model: Optional[NoiseModel] = None,
    ) -> Dict[str, float]:
        """
        Get detailed scores for all objectives.

        Returns:
            Dict with individual scores and total
        """
        scores = {
            "expressibility": self.expressibility_score.compute(circuit, backend),
            "trainability": self.trainability_score.compute(circuit, backend),
            "hardware_efficiency": self.hardware_efficiency_score.compute(circuit, backend),
        }

        if noise_model is not None:
            scores["noise_robustness"] = self.noise_robustness_score.compute(
                circuit, backend, noise_model
            )
        else:
            scores["noise_robustness"] = self.noise_robustness_score.compute(
                circuit, backend, self.noise_robustness_score.noise_model
            )

        # Weighted total
        scores["total"] = (
            self.weights.expressibility * scores["expressibility"] +
            self.weights.trainability * scores["trainability"] +
            self.weights.noise_robustness * scores["noise_robustness"] +
            self.weights.hardware_efficiency * scores["hardware_efficiency"]
        )

        return scores

    def get_weights(self) -> Dict[str, float]:
        """Get current weights as dictionary."""
        return self.weights.to_dict()

    def set_weights(self, weights: RewardWeights):
        """Update weights."""
        self.weights = weights


# =============================================================================
# Integration Wrapper for state_qas.py
# =============================================================================

class QASRewardWrapper:
    """
    Wrapper to integrate MultiObjectiveReward with state_qas.py RL framework.

    This wrapper adapts the multi-objective reward function to work with
    the QuantumStateSearchEnvCore environment.
    """

    def __init__(
        self,
        multi_objective_reward: MultiObjectiveReward,
        fidelity_threshold: float = 0.95,
        use_noise: bool = False,
    ):
        """
        Args:
            multi_objective_reward: The reward function to use
            fidelity_threshold: Threshold for considering task complete
            use_noise: Whether to include noise in evaluation
        """
        self.reward_func = multi_objective_reward
        self.fidelity_threshold = fidelity_threshold
        self.use_noise = use_noise

    def compute_reward(
        self,
        circuit: Circuit,
        backend: Backend,
        fidelity: float,
        noise_model: Optional[NoiseModel] = None,
    ) -> float:
        """
        Compute reward for a circuit in the RL loop.

        Args:
            circuit: Current circuit being evaluated
            backend: Backend for computation
            fidelity: Fidelity to target state
            noise_model: Noise model (if use_noise=True)

        Returns:
            float: Reward value
        """
        # Use noise model from reward function if available
        if self.use_noise and noise_model is None:
            noise_model = self.reward_func.noise_robustness_score.noise_model

        # Compute multi-objective reward
        reward = self.reward_func(
            circuit=circuit,
            backend=backend,
            fidelity=fidelity,
            noise_model=noise_model,
        )

        # Add completion bonus if fidelity threshold met
        if fidelity >= self.fidelity_threshold:
            reward += 0.1  # Bonus for successful completion

        return reward


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "RewardWeights",
    "ExpressibilityScore",
    "TrainabilityScore",
    "NoiseRobustnessScore",
    "HardwareEfficiencyScore",
    "MultiObjectiveReward",
    "QASRewardWrapper",
]
