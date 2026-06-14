# Ion Trap Noise Parameters

This file is the single source of truth for the default ion-trap noise model.
ion_trap.py reads this file, normalizes the numbers, and builds a runtime NoiseModel.

# Formula profile (source: "离子阱量子计算系统噪声分析与模型设计", 2025)
formula_profile: ion_trap_doc_2025

# Circuit / experiment setup (kept consistent with si1000_ion.yaml)
rounds: 25
basis: z
twoq_gate: zz_opt
data_qubits: [0, 1, 6, 3, 4, 2, 7]
ancillas: [5]
logical_label_mode: parity

# Noise switches
# Set True/False here to enable or disable each noise source at runtime.
enable_initialization_noise: True
enable_measurement_noise: True
enable_oneq_gate_noise: True
enable_twoq_gate_noise: True
enable_idle_dephasing_noise: True
enable_crosstalk_noise: True

# Directly used runtime parameters
T2: 0.2
meas_bitflip: 2.5e-4
reset_bitflip: 2.5e-4
oneq_gate_time: 0.0001
twoq_gate_time: 0.0006
reset_time: 1.0e-5
meas_time: 1.0e-3

# Idle dephasing / noise 3 (document formula)
# p_idle = 1/2 * (1 - exp(-t / T2))
# t selection strategy in this project:
# - after a single-qubit gate, use t = oneq_gate_time
# - after a two-qubit gate, use t = twoq_gate_time
# Because trapped-ion gates are serial, when one gate executes, idle dephasing
# is applied to all other qubits. The active gate qubits are excluded.
idle_time_strategy: switch_by_gate_arity

# Raw calibration values used by formula derivation
# Single-qubit: p_s = 3/2 * deltaF^(1)
oneq_avg_infidelity_deltaF1: 0.001

# Two-qubit: p_MS = 5/4 * deltaF
# Kept consistent with si1000_ion.yaml: twoq_depol = 5/4 * 0.75 * 1.0e-2
twoq_avg_infidelity_deltaF2: 0.0075

# Crosstalk / noise 6
# Uniform correlated crosstalk error probability.
# Source formula: p_c = sin^2(epsilon_CT * theta / 2), with theta = pi / 2.
# Representative epsilon_CT ~= 0.00213 gives p_c ~= 2.8e-6.
cross_talk: 2.8e-06
