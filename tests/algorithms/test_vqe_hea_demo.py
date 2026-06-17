import math
import unittest

from aicir.qas._types import ArchitectureSpec
from aicir.qas import (
    HEAMask,
    THETA_INIT_RANDOM_UNIFORM_PI,
    adaptive_fair_n_starts,
    analyze_hamiltonian,
    architecture_from_hea_mask,
    derive_priority_seed_masks,
    diagnose_theta_randomness,
    evaluate_h2_energy,
    exact_ground_energy,
    get_structure_family,
    hamiltonian_aware_mask_preferences,
    is_hamiltonian_favored_family,
    is_b1_improvement_valid,
    ising4_demo_problem,
    mutate_hea_mask,
    mutate_hea_mask_hamiltonian_aware,
    rotation_only_ansatz,
    run_ising4_b2_reliability_experiment,
    run_ising4_fair_vqe_stability_experiment,
    run_ising4_final_multiseed_validation,
    run_ising4_full_enumeration_baseline,
    run_ising4_trainability_prior_demo,
    run_vqe_hea_demo,
    run_vqe_ising4_demo,
    resolve_qas_backend,
    stratified_stage1_pool,
    tfim_open_chain_free_fermion_ground_energy,
    update_beam,
    validate_tfim_reference_alignment,
    v3_final_maxfev,
    v3_screening_maxfev,
    v3_top_k,
)
from aicir.qas.task_evaluation import parameter_count
from aicir.qas.vqe_hea_demo import VQEDemoProblem, evaluate_vqe_energy


class TestVQEHEADemo(unittest.TestCase):
    def test_mutate_hea_mask_changes_exactly_one_dimension(self):
        import numpy as np

        mask = HEAMask(
            n_qubits=2,
            layers=1,
            rotation_block="ry",
            entangler="cx",
            final_rotation="ry",
            entangle_pattern="linear",
        )
        mutated = mutate_hea_mask(mask, np.random.default_rng(7))

        self.assertEqual(mutated.n_qubits, mask.n_qubits)
        changed = sum(left != right for left, right in zip(mask.key()[1:], mutated.key()[1:]))
        self.assertEqual(changed, 1)

    def test_h2_energy_evaluates_hea_architecture(self):
        architecture = architecture_from_hea_mask(HEAMask(n_qubits=2, layers=1))
        n_params = parameter_count(architecture.circuit)
        energy = evaluate_h2_energy(architecture, [0.0] * n_params)

        self.assertTrue(math.isfinite(energy))

    def test_vqe_energy_uses_unmeasured_state_for_x_terms(self):
        architecture = ArchitectureSpec.from_gates(
            name="one_ry",
            gates=[{"type": "ry", "target_qubit": 0, "parameter": 0.0}],
            n_qubits=1,
        )
        problem = VQEDemoProblem(
            name="minus_x",
            n_qubits=1,
            hamiltonian=[(-1.0, "X")],
            reference_energy=-1.0,
        )

        energy = evaluate_vqe_energy(architecture, problem, [math.pi / 2])

        self.assertAlmostEqual(energy, -1.0, places=6)

    def test_v3_protocol_helpers_freeze_budget_and_top_k_rules(self):
        self.assertEqual(v3_screening_maxfev(3), 500)
        self.assertEqual(v3_screening_maxfev(10), 800)
        self.assertEqual(v3_final_maxfev(3), 1000)
        self.assertEqual(v3_final_maxfev(10), 2000)
        self.assertEqual(v3_top_k(0), 0)
        self.assertEqual(v3_top_k(8), 8)
        self.assertEqual(v3_top_k(108), 11)

    def test_theta_randomness_diagnostic_uses_independent_uniform_pi_starts(self):
        architecture = architecture_from_hea_mask(HEAMask(n_qubits=4, layers=1, rotation_block="ry"))
        report = diagnose_theta_randomness(
            architecture,
            ising4_demo_problem(),
            n_trials=3,
            seed=101,
            maxfev=2,
            init_mode=THETA_INIT_RANDOM_UNIFORM_PI,
        )

        self.assertEqual(report["theta_init_mode"], THETA_INIT_RANDOM_UNIFORM_PI)
        self.assertTrue(report["passes_randomness_guard"])
        self.assertGreater(report["init_l2_std"], 0.0)

    def test_resolve_qas_backend_defaults_to_numpy(self):
        backend = resolve_qas_backend()

        self.assertIn("NumpyBackend", backend.name)

    def test_resolve_qas_backend_supports_complex128_final_dtype(self):
        backend = resolve_qas_backend(dtype="complex128")

        self.assertIn("complex128", backend.name)

    def test_vqe_hea_demo_runs_small_pipeline(self):
        report = run_vqe_hea_demo(candidate_limit=8, stage1_keep_top=4, sa_steps=2, seed=7)

        self.assertTrue(report.stage1_rows)
        self.assertTrue(any(row.kept for row in report.stage1_rows))
        self.assertTrue(any(not row.kept for row in report.stage1_rows))
        self.assertTrue(report.sa_trace)
        self.assertTrue(report.final_results)
        self.assertIn("metric | min | p25 | max", "\n".join(report.summary_lines()))
        self.assertIn("Final VQE validation", "\n".join(report.summary_lines()))

    def test_ising4_problem_has_exact_reference(self):
        problem = ising4_demo_problem()

        self.assertEqual(problem.n_qubits, 4)
        self.assertAlmostEqual(problem.reference_energy, exact_ground_energy(problem.hamiltonian), places=10)

    def test_tfim_reference_alignment_matches_dense_for_v3_scales(self):
        report = validate_tfim_reference_alignment(scales=(4, 6, 8))

        self.assertTrue(report.passed)
        self.assertTrue(all(row.abs_diff < 1e-9 for row in report.rows))

    def test_tfim_open_chain_free_fermion_matches_ising4_reference(self):
        problem = ising4_demo_problem()

        self.assertAlmostEqual(
            tfim_open_chain_free_fermion_ground_energy(problem.n_qubits),
            problem.reference_energy,
            places=10,
        )

    def test_tfim_reference_alignment_rejects_pbc_without_parity_sector(self):
        with self.assertRaises(NotImplementedError):
            validate_tfim_reference_alignment(scales=(4,), periodic=True)

    def test_hamiltonian_analyzer_biases_ising_mutation_rules(self):
        import numpy as np

        problem = ising4_demo_problem()
        profile = analyze_hamiltonian(problem.hamiltonian)
        preferences = hamiltonian_aware_mask_preferences(profile)
        mask = HEAMask(n_qubits=4, layers=1, rotation_block="ry", entangler="cx")
        mutated = mutate_hea_mask_hamiltonian_aware(mask, profile, np.random.default_rng(3), bias_probability=1.0)

        self.assertGreater(profile.zz_weight, profile.xx_weight)
        self.assertEqual(preferences["entangler"][0], "rzz")
        self.assertIn("rx_ry_rz", preferences["rotation_block"])
        self.assertEqual(mutated.n_qubits, mask.n_qubits)
        self.assertEqual(sum(left != right for left, right in zip(mask.key()[1:], mutated.key()[1:])), 1)

    def test_update_beam_reserves_diverse_family_slots(self):
        candidates = [
            architecture_from_hea_mask(HEAMask(n_qubits=4, layers=1, rotation_block="ry", entangler="cx")),
            architecture_from_hea_mask(HEAMask(n_qubits=4, layers=2, rotation_block="ry", entangler="cx")),
            architecture_from_hea_mask(HEAMask(n_qubits=4, layers=1, rotation_block="rx_ry_rz", entangler="rzz")),
            architecture_from_hea_mask(HEAMask(n_qubits=4, layers=2, rotation_block="rx_ry_rz", entangler="rzz")),
        ]
        scores = {candidate.name: float(index) for index, candidate in enumerate(candidates)}
        beam = update_beam(candidates, scores, beam_width=4)
        first_half_families = [get_structure_family(candidate) for candidate in beam[:2]]

        self.assertEqual(len(beam), 4)
        self.assertEqual(len(set(first_half_families)), 2)

    def test_vqe_ising4_demo_runs_small_pipeline(self):
        report = run_vqe_ising4_demo(candidate_limit=8, stage1_keep_top=4, sa_steps=2, seed=11)

        self.assertIn("tfim_chain_4q", "\n".join(report.summary_lines()))
        self.assertTrue(report.stage1_rows)
        self.assertTrue(report.sa_trace)
        self.assertTrue(report.final_results)

    def test_ising4_b2_reliability_experiment_runs_small_pipeline(self):
        report = run_ising4_b2_reliability_experiment(
            seed=31,
            top_k=3,
            candidate_limit=8,
            stage1_keep_top=5,
            b1_max_evaluations=3,
            b2_n_starts=1,
            b2_max_evaluations=4,
            fair_n_starts=1,
            fair_evals_per_param=1,
            fair_min_evaluations=3,
            include_aware_neighbors=False,
        )
        summary = "\n".join(report.summary_lines())

        self.assertTrue(report.rows)
        self.assertIn("B2 reliability experiment", summary)
        self.assertIn("spearman_b2_vs_fair", summary)
        self.assertIn("spearman_improvement_vs_fair", summary)
        self.assertIn("improvement", summary)

    def test_priority_seed_derivation_includes_tfim_rzz_rx_deep_family(self):
        profile = analyze_hamiltonian(ising4_demo_problem().hamiltonian)
        seeds = derive_priority_seed_masks(profile, max_layers=3)

        self.assertTrue(
            any(
                mask.layers == 3
                and mask.rotation_block == "rx_ry_rz"
                and mask.entangler == "rzz"
                and mask.entangle_pattern == "linear"
                for mask in seeds
            )
        )

    def test_adaptive_fair_starts_and_b1_validity_helpers(self):
        architecture = architecture_from_hea_mask(
            HEAMask(
                n_qubits=4,
                layers=3,
                rotation_block="rx_ry_rz",
                entangler="rzz",
                final_rotation="ry",
                entangle_pattern="linear",
            )
        )

        self.assertGreaterEqual(adaptive_fair_n_starts(architecture, min_starts=3, params_per_start=15), 4)
        self.assertFalse(is_b1_improvement_valid(-3.0001, floor_energy=-3.0, tolerance=0.01))
        self.assertTrue(is_b1_improvement_valid(-3.2, floor_energy=-3.0, tolerance=0.01))

    def test_stratified_stage1_pool_keeps_l3_and_hamiltonian_favored(self):
        from aicir.qas import enumerate_hea_masks

        problem = ising4_demo_problem()
        profile = analyze_hamiltonian(problem.hamiltonian)
        candidates = [architecture_from_hea_mask(mask) for mask in enumerate_hea_masks(problem.n_qubits)]
        rows = stratified_stage1_pool(candidates, profile, pool_size=12, n_samples=2)
        kept = [row.architecture for row in rows if row.kept]
        kept_masks = [architecture.metadata["hea_mask"] for architecture in kept]

        self.assertTrue(any(mask[1] == 3 for mask in kept_masks))
        self.assertTrue(any(is_hamiltonian_favored_family(architecture, profile) for architecture in kept))

    def test_ising4_b2_reliability_uses_per_param_budget_and_bad_baseline(self):
        report = run_ising4_b2_reliability_experiment(
            seed=37,
            top_k=3,
            candidate_limit=8,
            stage1_keep_top=5,
            b1_max_evaluations=3,
            b2_n_starts=1,
            b2_evals_per_param=2,
            b2_max_evaluations=40,
            fair_n_starts=1,
            fair_evals_per_param=1,
            fair_min_evaluations=3,
            include_aware_neighbors=False,
            include_bad_baseline=True,
            include_priority_seeds=False,
            improvement_floor=0.0,
        )
        b2_evals = {row.b2_evaluations for row in report.rows}
        bad_rows = [row for row in report.rows if row.architecture.metadata.get("source") == "bad_baseline"]

        self.assertGreater(len(b2_evals), 1)
        self.assertTrue(bad_rows)
        self.assertIn("fair_energy_spread", "\n".join(report.summary_lines()))

    def test_ising4_b2_reliability_improvement_floor_removes_bad_baseline(self):
        report = run_ising4_b2_reliability_experiment(
            seed=43,
            top_k=3,
            candidate_limit=8,
            stage1_keep_top=5,
            b1_max_evaluations=3,
            b2_n_starts=1,
            b2_evals_per_param=2,
            b2_max_evaluations=40,
            fair_n_starts=1,
            fair_evals_per_param=1,
            fair_min_evaluations=3,
            include_aware_neighbors=False,
            include_bad_baseline=True,
            include_priority_seeds=False,
            improvement_floor=1.0,
        )

        self.assertFalse(any(row.architecture.metadata.get("source") == "bad_baseline" for row in report.rows))

    def test_ising4_full_enumeration_baseline_runs_small_pipeline(self):
        report = run_ising4_full_enumeration_baseline(
            seed=41,
            candidate_limit=4,
            fair_n_starts=1,
            fair_evals_per_param=1,
            fair_min_evaluations=3,
            include_bad_baseline=True,
        )
        summary = "\n".join(report.summary_lines(top_k=3))

        self.assertTrue(report.results)
        self.assertIn("full enumeration baseline", summary)
        self.assertIn("Top fair VQE candidates", summary)

    def test_ising4_fair_vqe_stability_experiment_runs_small_pipeline(self):
        report = run_ising4_fair_vqe_stability_experiment(
            seed=47,
            top_k=2,
            repeats=2,
            candidate_limit=4,
            fair_n_starts=1,
            fair_evals_per_param=1,
            fair_min_evaluations=3,
            adaptive_fair_starts=False,
        )
        summary = "\n".join(report.summary_lines())

        self.assertTrue(report.rows)
        self.assertIn("fair VQE stability", summary)
        self.assertIn("std", summary)

    def test_ising4_final_multiseed_validation_runs_small_pipeline(self):
        report = run_ising4_final_multiseed_validation(
            seed=53,
            repeats=1,
            candidate_limit=6,
            stage1_keep_top=4,
            b1_max_evaluations=3,
            b2_evals_per_param=1,
            b2_max_evaluations=5,
            fair_n_starts=1,
            fair_evals_per_param=1,
            fair_min_evaluations=3,
            adaptive_fair_starts=False,
            include_aware_neighbors=False,
            include_priority_seeds=False,
        )
        summary = "\n".join(report.summary_lines())

        self.assertTrue(report.rows)
        self.assertIn("fair VQE stability", summary)
        self.assertEqual(report.metadata["candidate_source"], "stage1_priority_b2_diagnostics")

    def test_rotation_only_ansatz_has_no_two_qubit_gates(self):
        architecture = rotation_only_ansatz(n_qubits=4)

        self.assertEqual(architecture.two_qubit_gate_count, 0)
        self.assertEqual(architecture.metadata["source"], "bad_baseline")

    def test_ising4_trainability_prior_demo_runs(self):
        report = run_ising4_trainability_prior_demo(
            seed=29,
            candidate_limit=8,
            stage1_keep_top=6,
            trainability_top_k=3,
            sa_steps=2,
            search_max_evaluations=6,
            final_n_starts=1,
            fair_evals_per_param=2,
            fair_min_evaluations=4,
        )
        summary = "\n".join(report.summary_lines())

        self.assertTrue(report.trainability_top_results)
        self.assertTrue(report.sa_trace)
        self.assertTrue(report.baseline_results)
        self.assertIn("Trainability top fair final", summary)
        self.assertIn("Diagnostic: SA final vs baselines", summary)


if __name__ == "__main__":
    unittest.main()
