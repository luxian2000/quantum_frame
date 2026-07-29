def test_public_api_uses_explicit_names_only():
    import aicir.distributed as distributed

    assert distributed.__all__ == [
        "DistNPUBackend",
        "DistState",
        "DistSimulator",
        "DistResult",
    ]
    assert not hasattr(distributed, "Backend")
    assert not hasattr(distributed, "State")
    assert not hasattr(distributed, "Simulator")
    assert not hasattr(distributed, "Result")


def test_distributed_types_are_not_top_level_exports():
    import aicir

    assert not hasattr(aicir, "DistNPUBackend")
    assert not hasattr(aicir, "DistState")
    assert not hasattr(aicir, "DistSimulator")
    assert not hasattr(aicir, "DistResult")
