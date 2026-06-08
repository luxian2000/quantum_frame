"""Small QUBO-to-QAOA demo.

Run from the repository root:

    python demos/qubo_qaoa_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aicir.optimization.qubo import (
    Model,
    ModelContext,
    linear_objective,
    most_likely_qaoa_assignment,
    one_hot,
    run_qubo_qaoa,
)


def main() -> None:
    ctx = ModelContext()
    x = ctx.binary_array("x", 2)

    model = Model(ctx.zero())
    model.add_constraint(one_hot(x, penalty=5.0, label="choose_one"))
    model.add_objective(linear_objective([(0.0, x[0]), (-1.0, x[1])]))

    result = run_qubo_qaoa(model, p=1, max_iters=20, lr=0.05, seed=7)
    _, _, variable_names = model.to_qaoa_terms()
    decoded = most_likely_qaoa_assignment(result.statevector, variable_names)

    print("=== QUBO QAOA Demo ===")
    print(f"best energy : {result.energy:.6f}")
    print(f"bitstring   : {decoded.bitstring}")
    print(f"probability : {decoded.probability:.6f}")
    print(f"assignment  : {decoded.assignment}")
    print(f"history     : {[round(value, 6) for value in result.energy_history[:5]]} ...")


if __name__ == "__main__":
    main()
