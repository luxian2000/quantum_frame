# aicir.optimization.qubo

This package provides pure-Python QUBO modeling tools for combinatorial
optimization problems.

The modeling API is adapted from the standalone `qubo-modeling` prototype and
is intentionally independent from the circuit simulator backend. It can be used
without importing GPU/NPU backends or installing Torch.

## Basic Usage

```python
from aicir.optimization.qubo import Model, ModelContext, one_hot

ctx = ModelContext()
x = ctx.binary_array("x", 3)

model = Model(ctx.zero())
model.add_constraint(one_hot(x, penalty=5.0))

qubo, offset = model.to_qubo()
ising = model.to_ising()
terms, qaoa_offset, variable_names = model.to_qaoa_terms()
```

## Hamiltonian Adapter

```python
from aicir.optimization.qubo import model_to_hamiltonian

hamiltonian = model_to_hamiltonian(model)
```

The adapter converts one- and two-qubit Z terms into
`aicir.channel.operators.Hamiltonian`. By default it keeps the global Ising
offset as an identity term, which preserves exact energies. Pass
`include_offset=False` when the offset should be omitted, for example when only
the QAOA phase operator up to a global phase is needed.

## BasicQAOA Helper

```python
from aicir.optimization.qubo import model_to_qaoa_matrix, run_qubo_qaoa

matrix, n_qubits = model_to_qaoa_matrix(model)
result = run_qubo_qaoa(model, p=1, max_iters=50, lr=0.05, seed=123)
```

These helpers use the existing `aicir.vqc.BasicQAOA` dense-matrix path. They are
intended for small examples and first integration tests. Large QUBO instances
should avoid dense `2^n x 2^n` matrices and use a circuit-level or sparse QAOA
implementation instead.

A runnable end-to-end demo is available at:

```bash
python demos/qubo_qaoa_demo.py
```

## Scope

Included in this first integration pass:

- binary variables and sparse polynomial expressions
- model-scoped variable registry via `ModelContext`
- common constraints such as one-hot, cardinality, assignment, and linear
  inequality with slack variables
- structured objectives
- low-level `QuboBuilder`
- TSP, graph coloring, and knapsack builders
- QUBO, Ising, sparse matrix, and QAOA term exports
- conversion to `aicir.channel.operators.Hamiltonian`
- dense-matrix helper for `aicir.vqc.BasicQAOA`
- solution decoding and small-model brute-force analysis

Not included in this pass:

- PyQUBO / Fixstars Amplify benchmark scripts
- standalone package build files
- scalable sparse or circuit-level QAOA execution for large QUBOs

The next integration step should add a scalable sparse or circuit-level QAOA
path for larger QUBO instances.
