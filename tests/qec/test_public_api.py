import importlib

import pytest


def test_public_names_are_exported():
    qec = importlib.import_module("aicir.qec")
    expected = {
        "StabilizerCode", "pauli_to_gf2", "gf2_to_pauli", "symplectic_product",
        "CODES", "get_code", "register_code",
        "Detector", "Observable", "DetectorLayout",
        "BareAncillaSchedule", "register_schedule", "resolve_schedule", "verify_schedule",
        "ErrorEvent", "PauliErrorModel",
        "DecodeStep", "LookupDecoder", "register_decoder", "resolve_decoder",
        "QECShotRecord", "QECResult", "run", "TimingModel",
    }
    missing = expected - set(qec.__all__)
    assert not missing, f"__all__ 缺少：{sorted(missing)}"
    for name in expected:
        assert hasattr(qec, name), f"aicir.qec 缺少属性 {name}"


def test_qec_core_has_no_optional_dependencies():
    """qec 核心只能依赖 numpy —— 不得 import torch/scipy/matplotlib/stim/pymatching。"""
    import pathlib
    import re

    root = pathlib.Path("aicir/qec")
    banned = re.compile(r"^\s*(?:import|from)\s+(torch|scipy|matplotlib|stim|pymatching)\b",
                        re.MULTILINE)
    for path in root.rglob("*.py"):
        hits = banned.findall(path.read_text(encoding="utf-8"))
        assert not hits, f"{path} 引入了禁止的依赖：{hits}"


def test_readme_exists():
    import pathlib
    assert pathlib.Path("aicir/qec/README.md").is_file()
