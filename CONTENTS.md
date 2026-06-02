# quantum_frame 目录结构

```text
.
├── CHANGELOG.md
├── CONTENTS.md
├── README.md
├── comparison.md
├── demo_npu.py
├── docs/
├── smoke_npu_new_path.py
├── aicir/
│   ├── __init__.py
│   ├── channel/
│   │   ├── __init__.py
│   │   ├── operators.py
│   │   ├── backends/
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── npu_backend.py
│   │   │   ├── numpy_backend.py
│   │   │   └── torch_backend.py
│   │   ├── noise/
│   │   │   ├── __init__.py
│   │   │   ├── analysis.py
│   │   │   ├── base.py
│   │   │   ├── channels.py
│   │   │   ├── ion_trap.py
│   │   │   ├── ion_trap_noise_params.md
│   │   │   ├── metrics.py
│   │   │   └── model.py
│   │   └── states/
│   ├── chemistry/
│   │   └── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── circuit.py
│   │   ├── density.py
│   │   ├── gates.py
│   │   ├── state.py
│   │   └── io/
│   │       ├── __init__.py
│   │       ├── README.md
│   │       ├── dag.py
│   │       ├── json_io.py
│   │       └── qasm.py
│   ├── encoder/
│   │   ├── __init__.py
│   │   ├── abstract.py
│   │   ├── amplitude.py
│   │   ├── angle.py
│   │   ├── basis.py
│   │   └── demo/
│   │       ├── encode_1234_demo.py
│   │       ├── encode_1234_demo_redundant_false.qasm
│   │       └── encode_1234_demo_redundant_true.qasm
│   ├── measure/
│   │   ├── __init__.py
│   │   ├── measure.py
│   │   ├── result.py
│   │   └── sampler.py
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── _utils.py
│   │   ├── expressibility.py
│   │   ├── hardware.py
│   │   ├── noisy_expressibility.py
│   │   └── trainability.py
│   ├── optimization/
│   │   ├── __init__.py
│   │   ├── qubo/
│   │   │   └── __init__.py
│   │   └── sb/
│   │       └── __init__.py
│   ├── optimizer/
│   │   ├── __init__.py
│   │   ├── README.md
│   │   └── basic.py
│   ├── qas/
│   │   ├── __init__.py
│   │   ├── _types.py
│   │   ├── _utils.py
│   │   ├── CRLQAS.py
│   │   ├── PPO_RB.py
│   │   ├── PPR_DQL.py
│   │   ├── README.md
│   │   ├── architecture_candidates.md
│   │   ├── architecture_candidates.py
│   │   ├── architecture_metrics.md
│   │   ├── architecture_search.py
│   │   ├── candidates.py
│   │   ├── evaluator.py
│   │   ├── multi_objective_reward.py
│   │   ├── reward.py
│   │   ├── search_env.py
│   │   └── demo/
│   │       ├── CRLQAS_demo_h2.py
│   │       ├── PPO_RB_demo_dicke3.py
│   │       ├── PPO_RB_demo_ghz4.py
│   │       ├── PPO_RB_demo_w3.py
│   │       ├── PPR_DQL_demo_ghz3.py
│   │       ├── PPR_DQL_noise_adaptive_demo.py
│   │       ├── architecture_scoring_demo.py
│   │       ├── architecture_scoring_results.txt
│   │       ├── crlqas_h2_circuit.qasm
│   │       ├── ppo_rb_dicke3_circuit.qasm
│   │       ├── ppo_rb_ghz3_circuit.qasm
│   │       ├── ppo_rb_ghz4_circuit.qasm
│   │       ├── ppo_rb_w3_circuit.qasm
│   │       └── ppr_dql_ghz3_circuit.qasm
│   ├── qml/
│   │   ├── __init__.py
│   │   └── grad.py
│   ├── universal/
│   │   ├── __init__.py
│   │   └── qft.py
│   ├── vqc/
│   │   ├── __init__.py
│   │   ├── QAOA.py
│   │   ├── SSVQE.py
│   │   ├── VQD.py
│   │   ├── VQE.py
│   │   └── ansatz/
│   │       └── __init__.py
│   └── wireless/
│       └── __init__.py
└── tests/
    ├── __init__.py
    ├── print_qft_3.py
    ├── algorithms/
    │   ├── test_architecture_candidates.py
    │   ├── test_architecture_evaluation.py
    │   ├── test_crlqas.py
    │   ├── test_ion_trap_noise_config.py
    │   └── test_ppr_dql.py
    ├── backends/
    │   ├── __init__.py
    │   └── test_npu_backend.py
    ├── circuit/
    │   ├── __init__.py
    │   ├── test_basis_encoder.py
    │   ├── test_circuit_backend_unitary.py
    │   ├── test_circuit_show.py
    │   ├── test_optimizer_basic.py
    │   ├── test_parameterized_circuit.py
    │   ├── test_state.py
    │   └── io/
    │       ├── __init__.py
    │       └── test_json_qasm_io.py
    ├── execution/
    ├── measure/
    │   └── test_measure.py
    ├── noise/
    │   ├── __init__.py
    │   └── test_noise_model.py
    ├── qml/
    │   └── test_gradient.py
    ├── universal/
    │   └── test_qft.py
    └── vqc/
        └── test_parameter_shift_uses_qml.py
```
