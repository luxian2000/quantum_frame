import pytest

from aicir import cx, hadamard, pauli_x
from aicir.distributed import DistNPUBackend
from aicir.distributed.gates import _GatePlanner
from aicir.distributed.layout import _Layout


def _backend(monkeypatch, *, world_size=4, rank=1):
    monkeypatch.setenv("WORLD_SIZE", str(world_size))
    monkeypatch.setenv("RANK", str(rank))
    monkeypatch.setenv("LOCAL_RANK", str(rank))
    return DistNPUBackend.from_env(
        fallback_to_cpu=True,
        init_process_group=False,
    )


def test_local_gate_plan_has_no_partners(monkeypatch):
    backend = _backend(monkeypatch)
    layout = _Layout.explicit(
        (0, 1, 2, 3),
        n_qubits=4,
        distributed_axes=2,
    )
    planner = _GatePlanner(backend, layout, n_qubits=4)

    plan = planner.plan(hadamard(3), instruction_index=0)

    assert plan.logical_axes == (3,)
    assert plan.storage_axes == (3,)
    assert plan.distributed_storage_axes == ()
    assert plan.partner_masks == ()


def test_distributed_axis_plan_uses_rank_xor_partner(monkeypatch):
    backend = _backend(monkeypatch)
    layout = _Layout.explicit(
        (0, 1, 2, 3),
        n_qubits=4,
        distributed_axes=2,
    )
    planner = _GatePlanner(backend, layout, n_qubits=4)

    q0 = planner.plan(pauli_x(0), instruction_index=1)
    q1 = planner.plan(pauli_x(1), instruction_index=2)

    assert q0.partner_masks == (2,)
    assert q0.partner_for(rank=1, mask=2) == 3
    assert q1.partner_masks == (1,)
    assert q1.partner_for(rank=1, mask=1) == 0


def test_two_distributed_axes_visit_all_partner_masks(monkeypatch):
    backend = _backend(monkeypatch)
    layout = _Layout.explicit(
        (0, 1, 2, 3),
        n_qubits=4,
        distributed_axes=2,
    )
    planner = _GatePlanner(backend, layout, n_qubits=4)

    plan = planner.plan(
        cx(target_qubit=1, control_qubits=(0,)),
        instruction_index=3,
    )

    assert plan.distributed_storage_axes == (0, 1)
    assert plan.partner_masks == (1, 2, 3)


def test_planner_rejects_nonunitary_instruction(monkeypatch):
    from aicir import measure

    backend = _backend(monkeypatch)
    layout = _Layout.explicit(
        (0, 1, 2, 3),
        n_qubits=4,
        distributed_axes=2,
    )
    planner = _GatePlanner(backend, layout, n_qubits=4)

    with pytest.raises(ValueError, match="局部门矩阵"):
        planner.plan(measure(0), instruction_index=0)

