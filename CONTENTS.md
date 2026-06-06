# aicir 目录树

```text
aicir/
├── __init__.py
├── channel/
│   ├── __init__.py
│   ├── backends/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── npu_backend.py
│   │   ├── numpy_backend.py
│   │   └── torch_backend.py
│   ├── noise/
│   │   ├── __init__.py
│   │   ├── analysis.py
│   │   ├── base.py
│   │   ├── channels.py
│   │   ├── ion_trap.py
│   │   ├── ion_trap_noise_params.md
│   │   ├── metrics.py
│   │   └── model.py
│   └── operators.py
├── chemistry/
│   ├── README.md
│   ├── __init__.py
│   └── molecule.py
├── core/
│   ├── __init__.py
│   ├── circuit.py
│   ├── density.py
│   ├── gates.py
│   ├── io/
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── dag.py
│   │   ├── json_io.py
│   │   └── qasm.py
│   └── state.py
├── encoder/
│   ├── __init__.py
│   ├── abstract.py
│   ├── amplitude.py
│   ├── angle.py
│   ├── basis.py
│   └── demos/
│       ├── encode_1234_demo.py
│       ├── encode_1234_demo_redundant_false.qasm
│       └── encode_1234_demo_redundant_true.qasm
├── measure/
│   ├── __init__.py
│   ├── estimator.py
│   ├── measure.py
│   ├── result.py
│   └── sampler.py
├── metrics/
│   ├── README.md
│   ├── __init__.py
│   ├── _utils.py
│   ├── expressibility.py
│   ├── hardware.py
│   ├── noisy_expressibility.py
│   └── trainability.py
├── optimization/
│   ├── __init__.py
│   ├── qubo/
│   │   └── __init__.py
│   └── sb/
│       └── __init__.py
├── optimizer/
│   ├── README.md
│   ├── __init__.py
│   ├── circuit.py
│   └── params.py
├── qas/
│   ├── README.md
│   ├── CRLQAS.py
│   ├── PPO_RB.py
│   ├── PPR_DQL.py
│   ├── supernet.py
│   ├── __init__.py
│   ├── _types.py
│   ├── _utils.py
│   ├── architecture_candidates.md
│   ├── architecture_candidates.py
│   ├── architecture_metrics.md
│   ├── architecture_search.py
│   ├── candidates.py
│   ├── config.py
│   ├── demos/
│   │   ├── CRLQAS_demo_h2.py
│   │   ├── PPO_RB_demo_dicke3.py
│   │   ├── PPO_RB_demo_ghz4.py
│   │   ├── PPO_RB_demo_w3.py
│   │   ├── PPR_DQL_demo_ghz3.py
│   │   ├── PPR_DQL_noise_adaptive_demo.py
│   │   ├── VQA_QAS_demo_hamiltonian_cycle.py
│   │   ├── VQA_QAS_demo_hamiltonian_path.py
│   │   ├── _np_ising_utils.py
│   │   ├── architecture_scoring_demo.py
│   │   ├── architecture_scoring_results.txt
│   │   ├── crlqas_h2_circuit.qasm
│   │   ├── ppo_rb_dicke3_circuit.qasm
│   │   ├── ppo_rb_ghz3_circuit.qasm
│   │   ├── ppo_rb_ghz4_circuit.qasm
│   │   ├── ppo_rb_w3_circuit.qasm
│   │   ├── ppr_dql_ghz3_circuit.qasm
│   │   ├── vqa_qas_hamiltonian_cycle_circuit.qasm
│   │   └── vqa_qas_hamiltonian_path_circuit.qasm
│   ├── evaluator.py
│   ├── multi_objective_reward.py
│   ├── reward.py
│   ├── runner.py
│   └── search_env.py
├── qml/
│   ├── README.md
│   ├── __init__.py
│   └── deriv.py
├── universal/
│   ├── __init__.py
│   └── qft.py
├── visual/
│   ├── __init__.py
│   ├── circuit.py
│   ├── density.py
│   ├── plot.py
│   ├── qas.py
│   ├── state.py
│   └── utils.py
├── vqc/
│   ├── README.md
│   ├── QAOA.py
│   ├── SSVQE.py
│   ├── VQD.py
│   ├── VQE.py
│   ├── __init__.py
│   └── ansatz/
│       ├── __init__.py
│       ├── hea.py
│       └── hea_ti.py
└── wireless/
    └── __init__.py
```
