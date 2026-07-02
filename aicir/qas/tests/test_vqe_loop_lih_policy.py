from pathlib import Path

from aicir.qas.vqe_loop.benchmark_table import decide_next_round_quotas
from aicir.qas.vqe_loop.p0_bootstrap_fair import (
    P0BootstrapConfig,
    run_p0_bootstrap_fair,
)

def test_closed_loop_rejects_legacy_stage0_preparation_path(monkeypatch, tmp_path):
    calls = []

    def fake_run_module(module, args, *, cwd):
        calls.append((module, list(args)))

    monkeypatch.setattr("aicir.qas.vqe_loop.p0_bootstrap_fair._run_module", fake_run_module)

    try:
        run_p0_bootstrap_fair(
            P0BootstrapConfig(
                output_dir=tmp_path,
                n_qubits=2,
                hamiltonian_terms=[(1.0, "ZI")],
                hamiltonian_id="lih_sto3g_jw_r15",
                hamiltonian_class="molecular_lih",
                rounds=0,
                initial_labels=1,
                include_layerwise=False,
            )
        )
    except ValueError as exc:
        assert "P0 bootstrap" in str(exc)
    else:
        raise AssertionError("P0 bootstrap should reject legacy preparation path")

    assert calls == []

def test_weak_lih_oracle_with_failed_local_prefers_sparse_and_keeps_control_sentinels():
    decision = decide_next_round_quotas(
        n_qubits=12,
        base_quotas=(3, 2, 2, 1),
        calibration={
            "k_min": 3,
            "tr_in_count": 7,
            "tr_in_mae": 0.5658,
            "tr_out_mae": 0.7873,
            "sparse_abstain_rate": 1.0,
            "passes": {"overall": False, "sparse_abstain": True},
        },
        local_improved=False,
    )

    assert decision.mode == "sparse_explore"
    assert (decision.local, decision.boundary, decision.sparse, decision.control) == (1, 1, 4, 2)
    assert decision.sparse > decision.boundary
    assert decision.control == 2


