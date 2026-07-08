def test_demo_npu_mps_run_checks_cpu_fallback():
    from demos.demo_npu_mps import run_checks

    assert run_checks(allow_cpu_fallback=True) is True
