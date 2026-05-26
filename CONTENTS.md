# quantum_frame 目录结构

```text
.
├── CHANGELOG.md
├── comparison.md
├── CONTENTS.md
├── README.md
├── demo_npu.py
├── smoke_npu_new_path.py
├── docs/
├── nexq/
│   ├── __init__.py
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── chemistry/
│   │   │   └── __init__.py
│   │   ├── metrics/
│   │   │   ├── __init__.py
│   │   │   ├── _utils.py
│   │   │   ├── expressibility.py
│   │   │   ├── hardware.py
│   │   │   ├── noisy_expressibility.py
│   │   │   └── trainability.py
│   │   ├── optimization/
│   │   │   ├── __init__.py
│   │   │   ├── qubo/
│   │   │   └── sb/
│   │   ├── qas/
│   │   │   ├── __init__.py
│   │   │   ├── _types.py
│   │   │   ├── _utils.py
│   │   │   ├── architecture_candidates.md
│   │   │   ├── architecture_candidates.py
│   │   │   ├── architecture_metrics.md
│   │   │   ├── architecture_search.py
│   │   │   ├── candidates.py
│   │   │   ├── CRLQAS.py
│   │   │   ├── evaluator.py
│   │   │   ├── multi_objective_reward.py
│   │   │   ├── PPO_RB.py
│   │   │   ├── PPR_DQL.py
│   │   │   ├── README.md
│   │   │   ├── reward.py
│   │   │   ├── search_env.py
│   │   │   └── demo/
│   │   ├── qml/
│   │   │   └── __init__.py
│   │   ├── universal/
│   │   │   ├── __init__.py
│   │   │   └── QFT.py
│   │   ├── vqc/
│   │   │   ├── __init__.py
│   │   │   ├── QAOA.py
│   │   │   ├── SSVQE.py
│   │   │   ├── VQD.py
│   │   │   ├── VQE.py
│   │   │   └── ansatz/
│   │   └── wireless/
│   │       └── __init__.py
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
│   │   │   ├── ion_trap_noise_params.md
│   │   │   ├── ion_trap.py
│   │   │   ├── metrics.py
│   │   │   └── model.py
│   │   └── states/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── circuit.py
│   │   ├── density.py
│   │   ├── gates.py
│   │   ├── state.py
│   │   └── io/
│   │       ├── __init__.py
│   │       ├── dag.py
│   │       ├── json_io.py
│   │       ├── qasm.py
│   │       └── README.md
│   ├── encoder/
│   │   ├── __init__.py
│   │   ├── abstract.py
│   │   ├── amplitude.py
│   │   ├── angle.py
│   │   ├── basis.py
│   │   └── demo/
│   │       ├── encode_1234_demo_redundant_false.qasm
│   │       ├── encode_1234_demo_redundant_true.qasm
│   │       └── encode_1234_demo.py
│   ├── measure/
│   │   ├── __init__.py
│   │   ├── measure.py
│   │   ├── result.py
│   │   └── sampler.py
│   └── optimizer/
│       ├── __init__.py
│       ├── basic.py
│       └── README.md
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
    │   ├── test_state.py
    │   └── io/
    │       └── ...
    ├── execution/
    ├── measure/
    │   └── test_measure.py
    └── noise/
        ├── __init__.py
        └── test_noise_model.py
```
