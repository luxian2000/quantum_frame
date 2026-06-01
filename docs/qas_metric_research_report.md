# QAS Metric Research Report

This note records the P0.5 metric-upgrade research for QAS validation. The goal is
to replace placeholder architecture priors with metrics that better predict
task-level performance after parameter optimization.

## Long-Term TODO Placement

### P0.5: Metric Correction Before Strong Search Claims

1. Replace the current trainability `structure_proxy` with task-aware gradient
   diagnostics:
   - `gradient_norm`
   - `gradient_variance`
   - optional barren-plateau risk score from gradient scale, observable locality,
     and circuit depth

2. Replace the current hardware-efficiency heuristic with a profile-aware score:
   - native gate coverage
   - two-qubit gate count and depth
   - connectivity/routing overhead estimate
   - calibrated gate-error and readout-error cost when a hardware profile is
     available

### P1: Search Upgrade After Metrics Are Credible

1. Add random and mutation-based candidate generation.
2. Add beam search guided by the corrected QAS prior score.
3. Integrate RL search only after the reward uses reliable metrics or direct
   task-level feedback. Existing RL modules are useful starting points, but
   they should not optimize the current placeholder score unchanged.

## Trainability Research

### Main Observation

Trainability should not be judged only by depth and parameter count. The most
relevant literature diagnoses trainability through gradients of a task cost:
small mean absolute gradients and exponentially small gradient variance indicate
barren-plateau risk. For our QAS runner, the metric should therefore sample
parameters, evaluate the task objective, estimate parameter-shift gradients, and
summarize gradient norm and variance.

### Candidate Sources

| # | Source | What It Contributes | Code |
|---|---|---|---|
| 1 | McClean et al., "Barren plateaus in quantum neural network training landscapes", Nature Communications 2018, https://www.nature.com/articles/s41467-018-07090-4 | Introduces barren plateaus as exponentially vanishing gradients in parametrized quantum circuits. | No canonical repo found. |
| 2 | Cerezo et al., "Cost function dependent barren plateaus in shallow parametrized quantum circuits", Nature Communications 2021, https://www.nature.com/articles/s41467-021-21728-w | Shows global vs local cost functions change gradient-variance scaling. Important for task-aware metrics. | No canonical repo found. |
| 3 | Holmes et al., "Connecting Ansatz Expressibility to Gradient Magnitudes and Barren Plateaus", PRX Quantum 2022, https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.3.010313 | Connects high expressibility to smaller gradients; warns against maximizing expressibility alone. | No canonical repo found. |
| 4 | Uvarov and Biamonte, "On barren plateaus and cost function locality in variational quantum algorithms", arXiv 2020, https://arxiv.org/abs/2011.10530 | Relates gradient-variance lower bounds to causal-cone width of local Hamiltonian terms. | No canonical repo found. |
| 5 | "Efficient estimation of trainability for variational quantum circuits", PRX Quantum 2023, https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.4.040335 | Provides scalable trainability estimation through gradient and variance computation for broad VQC classes. | No canonical repo found. |
| 6 | Abbas et al., "The power of quantum neural networks", Nature Computational Science 2021 / arXiv, repo https://github.com/amyami187/effective_dimension | Uses effective dimension / Fisher information as capacity-trainability evidence. Useful later, heavier than gradient metrics. | https://github.com/amyami187/effective_dimension |
| 7 | Arrasmith et al., "Equivalence of cost concentration and gradient vanishing for quantum circuits", Quantum Science and Technology 2024, https://rqs.umd.edu/publications/equivalence-cost-concentration-and-gradient-vanishing-quantum-circuits-elementary | Connects cost concentration and gradient variance; supports using variance as a practical diagnostic. | No canonical repo found. |
| 8 | "Noise-induced barren plateaus in variational quantum algorithms", Nature Communications 2021, https://www.nature.com/articles/s41467-021-27045-6 | Shows noise can flatten gradients; relevant when QAS adds noisy evaluation. | No canonical repo found. |
| 9 | "Barren plateaus in quantum tensor network optimization", arXiv 2022, https://arxiv.org/abs/2209.00292 | Compares qMPS/qTTN/qMERA-style structures and gradient decay; useful for architecture families. | No canonical repo found. |
| 10 | QuAIRKit barren plateau tutorial, https://quairkit.com/QuAIRKit/latest/tutorials/research/bp.html | Practical gradient-analysis tutorial for QNNs. | QuAIRKit examples. |
| 11 | TensorCircuit barren plateau tutorial, https://tensorcircuit.readthedocs.io/en/latest/tutorials/barren_plateaus.html | Practical vectorized gradient-variance demonstration. | TensorCircuit examples. |
| 12 | Barren Plateaus Survey repo, https://github.com/junzhuang-code/Barren_Plateaus_Survey/ | Literature index and mitigation taxonomy. | https://github.com/junzhuang-code/Barren_Plateaus_Survey/ |

### Recommended Method

Use a direct, task-aware parameter-shift estimator first.

For an architecture `A`, problem objective `C(theta)`, and sampled parameter
vectors `theta_s`:

1. For each parameter `theta_i`, estimate
   `dC/dtheta_i = (C(theta + pi/2 e_i) - C(theta - pi/2 e_i)) / 2`.
2. Compute per-sample gradient vector `g_s`.
3. Report:
   - `mean_gradient_norm = mean(||g_s||_2)`
   - `gradient_variance = mean(var_s(g_s[i]))`
   - `mean_abs_gradient = mean(abs(g_s))`
   - `zero_gradient_fraction = fraction(abs(g_s[i]) < eps)`
4. Convert to a higher-is-better score with a saturating transform, for example:
   `score = 1 - exp(-mean_gradient_norm / scale)`, with a variance bonus and
   excessive-depth penalty.

This should be implemented in `nexq/metrics/trainability.py` and wired into
`ArchitectureEvaluator` as:

```python
active_metrics={"trainability": "gradient_variance"}
```

The metric should accept a task objective callback. If no task is supplied, keep
`structure_proxy` as the fallback.

## Hardware Efficiency Research

### Main Observation

Hardware efficiency should not be a generic "shorter is better" score. NISQ
hardware efficiency depends on whether a logical circuit matches the target
native gate set, topology, calibrated gate errors, readout errors, and scheduling
constraints. For this repo, the first implementation should be dependency-light:
define a `HardwareProfile` dataclass and estimate hardware cost directly from
the NexQ circuit. Later we can add optional Qiskit/TKET transpilation adapters.

### Candidate Sources

| # | Source | What It Contributes | Code |
|---|---|---|---|
| 1 | Wang et al., "QuantumNAS: Noise-Adaptive Search for Robust Quantum Circuits", HPCA 2022, project https://www.hanruiwang.com/projects/quantumnas | Co-searches circuit and qubit mapping under noise; directly relevant to hardware-aware QAS. | Project links paper; code availability should be checked before reuse. |
| 2 | Du et al., "Quantum circuit architecture search for variational quantum algorithms", npj Quantum Information 2022, https://www.nature.com/articles/s41534-022-00570-y | QAS for VQAs; provides architecture-search framing and open code. | https://github.com/yuxuan-du/Quantum_architecture_search/ |
| 3 | "Quantum Circuit Architecture Search on a Superconducting Processor", https://pmc.ncbi.nlm.nih.gov/articles/PMC11726871/ | Demonstrates topology-compatible ansatz search on superconducting hardware. | Check article supplementary/code links. |
| 4 | Li et al., "Tackling the Qubit Mapping Problem for NISQ-Era Quantum Devices" / SABRE, https://sites.cs.ucsb.edu/~yufeiding/publication/ASPLOS2019.pdf | Standard heuristic for layout/routing; motivates SWAP overhead and distance costs. | Qiskit includes SABRE passes. |
| 5 | "A Hardware-Aware Heuristic for the Qubit Mapping Problem in the NISQ Era", https://arxiv.org/abs/2010.03397 | Uses calibration data to improve fidelity under mapping choices. | Example implementation: https://github.com/Kaustuvi/quantum-qubit-mapping |
| 6 | Murali et al., "Formal constraint-based compilation for noisy intermediate-scale quantum systems", IBM Research, https://research.ibm.com/publications/formal-constraint-based-compilation-for-noisy-intermediate-scale-quantum-systems | Compilation under NISQ limits: connectivity, short coherence windows, and gate constraints. | No simple drop-in repo found. |
| 7 | "Full-stack quantum computing systems in the NISQ era", https://arxiv.org/abs/2204.06369 | Survey of algorithm-driven and hardware-aware compilation techniques. | Survey; not an implementation. |
| 8 | "Connectivity-aware Synthesis of Quantum Algorithms", https://arxiv.org/abs/2501.14020 | Optimizes gate count/depth across topology families; useful for connectivity-aware scoring. | Check authors' release before reuse. |
| 9 | Qiskit transpiler documentation, https://quantum.cloud.ibm.com/docs/api/qiskit/1.0/transpiler | Defines coupling maps, layout/routing/transpilation stages that can produce compiled depth/gate counts. | https://github.com/Qiskit/qiskit |
| 10 | Qiskit IBM Transpiler AIRouting docs, https://qiskit.qotlabs.org/docs/api/qiskit-ibm-transpiler/ai-ai-routing | AI layout/routing pass; useful later for optional hardware-aware compilation. | https://github.com/Qiskit/qiskit-ibm-transpiler |
| 11 | TKET compiler, https://github.com/CQCL/tket | Open compiler for architecture-aware circuit optimization. | https://github.com/CQCL/tket |
| 12 | MQT QMAP, https://portal.fis.tum.de/en/publications/mqt-qmap-a-tool-for-quantum-circuit-compilation/ | Dedicated quantum circuit mapping tool; useful for future optional adapter. | https://github.com/cda-tum/mqt-qmap |
| 13 | "Beyond Logical Circuits: Hardware-Aware Analysis of Expressibility and Trainability in VQAs", https://arxiv.org/abs/2605.25552 | Recent evidence that transpilation changes expressibility and gradient variance; validates hardware-aware metrics. | No repo found yet. |
| 14 | "Scaling of QAOA on superconducting qubit based hardware", IBM Research, https://research.ibm.com/publications/scaling-of-the-quantum-approximate-optimization-algorithm-on-superconducting-qubit-based-hardware | Shows QAOA scalability is limited by hardware connectivity and transpiler settings. | Uses Qiskit runtime ecosystem. |

### Recommended Method

Use a profile-aware logical estimator first, then add optional transpilation.

Recommended `HardwareProfile` fields:

```python
native_gates: set[str]
coupling_map: set[tuple[int, int]]
single_qubit_error: dict[str, float] | float
two_qubit_error: dict[tuple[int, int], float] | float
readout_error: dict[int, float] | float
gate_durations: dict[str, float]
max_depth: int | None
```

Recommended metrics:

1. `native_gate_ratio`: fraction of gates already in the native set.
2. `two_qubit_gate_count`: two-qubit gates dominate current NISQ error.
3. `depth_proxy`: scheduling/decoherence proxy.
4. `connectivity_violation_count`: two-qubit gates not supported by the coupling map.
5. `routing_distance_cost`: shortest-path distance minus one for nonlocal
   interactions; approximates required SWAP overhead.
6. `estimated_error_cost`: additive first-order error approximation:
   `sum(single_gate_errors) + sum(two_qubit_errors) + sum(readout_errors)`.
7. Higher-is-better score:
   `exp(-alpha * error_cost - beta * routing_cost - gamma * depth_norm)`.

This avoids adding a heavyweight compiler dependency now, but leaves a clean path
for future Qiskit/TKET/QMAP adapters that compute actual transpiled depth,
SWAP count, and compiled error cost.

## Decision For This Repo

1. Implement trainability first because it is closest to the current failure
   mode: QAS prior score does not predict post-optimization task performance.
2. Use parameter-shift gradient sampling on the same `ProblemInstance`
   objective used by task validation.
3. Implement hardware profiles second, using dependency-light logical estimates.
4. Do not integrate existing RL search yet. RL should be connected after the
   reward is reliable; otherwise it will optimize placeholder metrics faster,
   not better.

## Proposed Implementation Order

1. Add task-aware gradient metric helpers in `nexq/metrics/trainability.py`.
2. Add `task_trainability` support in `ArchitectureEvaluator`.
3. Pass `ProblemInstance` or objective callback from the validation runner into
   the evaluator for QAS ranking.
4. Add tests comparing a parameterized ansatz against a no-parameter circuit.
5. Add `HardwareProfile` and profile-aware details in `nexq/metrics/hardware.py`.
6. Add hardware profile tests for native gates, nonlocal edges, and estimated
   error cost.
7. Re-run multi-seed benchmark with:

```python
SearchConfig(
    active_metrics={
        "trainability": "gradient_variance",
        "hardware_efficiency": "profile_error_cost",
    }
)
```
