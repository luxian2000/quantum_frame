"""Static contracts for the strict channel probe's task-7 metrics."""

from __future__ import annotations

import inspect

from scripts.npu import distributed_autograd_probe as probe


def test_noise_probe_includes_every_builtin_probability_error_in_pass_gate():
    source = inspect.getsource(probe._noise_section)
    for channel in ("bit_flip", "phase_flip", "depolarizing", "amplitude_damping"):
        assert channel in source
    assert "probability_errors" in source
    assert "max_abs_error" in source
    assert "probability_errors.values()" in source


def test_stinespring_probe_computes_actual_positivity_and_nonzero_target_metrics():
    source = inspect.getsource(probe._stinespring_section)
    assert "np.linalg.eigvalsh" in source
    assert '"positivity_error": positivity_error' in source
    assert "target_qubits" in source
