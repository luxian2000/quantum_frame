import importlib.util
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts" / "npu" / "run_npu_tests.py"


def load_runner_module():
    spec = importlib.util.spec_from_file_location("run_npu_tests", RUNNER)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_runner(*args):
    return subprocess.run(
        [sys.executable, str(RUNNER), *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_npu_runner_lists_layered_suites():
    result = run_runner("--list")

    assert result.returncode == 0, result.stderr
    for suite in (
        "smoke",
        "backend",
        "ops",
        "capacity",
        "typed_ir",
        "circuit",
        "deriv",
        "qml",
        "qaoa",
        "tensor",
        "qas",
        "demos",
    ):
        assert f"{suite}:" in result.stdout


def test_npu_runner_dry_run_prints_selected_pytest_commands():
    result = run_runner(
        "--dry-run",
        "--suite",
        "smoke",
        "--suite",
        "qml",
        "--pytest-arg",
        "-q",
    )

    assert result.returncode == 0, result.stderr
    assert "-m pytest" in result.stdout
    assert "tests/backends/test_npu_backend.py" in result.stdout
    assert "tests/qml" in result.stdout
    assert "-q" in result.stdout


def test_npu_runner_dry_run_prints_typed_ir_and_deriv_probe_commands():
    result = run_runner(
        "--dry-run",
        "--suite",
        "typed_ir",
        "--suite",
        "deriv",
        "--pytest-arg",
        "-q",
    )

    assert result.returncode == 0, result.stderr
    assert "scripts/npu/typed_ir_deriv_probe.py" in result.stdout
    assert "--section typed-ir" in result.stdout
    assert "--section deriv" in result.stdout
    assert "tests/circuit/test_typed_ir_internal_migration.py" in result.stdout
    assert "tests/primitives/test_estimator_gradient.py" in result.stdout
    assert "tests/qml" in result.stdout


def test_npu_runner_dry_run_prints_qaoa_probe_and_pytest_commands():
    result = run_runner(
        "--dry-run",
        "--suite",
        "qaoa",
        "--pytest-arg",
        "-q",
    )

    assert result.returncode == 0, result.stderr
    assert "scripts/npu/qaoa_probe.py" in result.stdout
    assert "tests/vqc/test_qaoa_canonical.py" in result.stdout
    assert "tests/vqc/test_qaoa_qfun.py" in result.stdout
    assert "tests/optimization/qubo/test_qaoa_helpers.py" in result.stdout
    assert "-q" in result.stdout


def test_npu_runner_strict_dry_run_includes_real_npu_check():
    result = run_runner("--dry-run", "--strict-npu", "--suite", "smoke")

    assert result.returncode == 0, result.stderr
    assert "AICIR_REQUIRE_REAL_NPU=1" in result.stdout
    assert "is_npu_available" in result.stdout


def test_npu_shell_entrypoints_exist_and_are_executable():
    for filename in (
        "run_all.sh",
        "multi_card.sh",
        "qnn_4card.sh",
        "qaoa_8card.sh",
        "smoke.sh",
        "backend.sh",
        "ops.sh",
        "capacity.sh",
        "typed_ir.sh",
        "circuit.sh",
        "deriv.sh",
        "qml.sh",
        "qaoa.sh",
        "tensor.sh",
        "qas.sh",
        "demos.sh",
        "typed_ir_deriv_probe.sh",
    ):
        path = ROOT / "scripts" / "npu" / filename
        assert path.exists(), filename
        assert os.access(path, os.X_OK), filename


def test_npu_multicard_dry_run_prints_torchrun_command():
    script = ROOT / "scripts" / "npu" / "multi_card.sh"
    result = subprocess.run(
        [
            str(script),
            "--dry-run",
            "--nproc-per-node",
            "4",
            "--devices",
            "0,5,6,7",
            "--section",
            "collectives",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0, result.stderr
    assert "--devices is ignored" in result.stdout
    assert "AICIR_NPU_DEVICES=0,5,6,7" not in result.stdout
    assert "ASCEND_RT_VISIBLE_DEVICES=0,5,6,7" not in result.stdout
    assert "-m torch.distributed.run" in result.stdout
    assert "--nproc_per_node 4" in result.stdout
    assert "scripts/npu/multi_card_probe.py" in result.stdout
    assert "--section collectives" in result.stdout


def test_npu_qnn_4card_dry_run_prints_torchrun_command():
    script = ROOT / "scripts" / "npu" / "qnn_4card.sh"
    result = subprocess.run(
        [
            str(script),
            "--dry-run",
            "--nproc-per-node",
            "4",
            "--steps",
            "2",
            "--samples",
            "8",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0, result.stderr
    assert "ASCEND_RT_VISIBLE_DEVICES" not in result.stdout
    assert "-m torch.distributed.run" in result.stdout
    assert "--nproc_per_node 4" in result.stdout
    assert "demos/demo_npu_qnn_4card.py" in result.stdout
    assert "--steps 2" in result.stdout
    assert "--samples 8" in result.stdout


def test_npu_qaoa_8card_dry_run_prints_torchrun_command():
    script = ROOT / "scripts" / "npu" / "qaoa_8card.sh"
    result = subprocess.run(
        [
            str(script),
            "--dry-run",
            "--nproc-per-node",
            "8",
            "--steps",
            "2",
            "--samples",
            "16",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0, result.stderr
    assert "ASCEND_RT_VISIBLE_DEVICES" not in result.stdout
    assert "-m torch.distributed.run" in result.stdout
    assert "--nproc_per_node 8" in result.stdout
    assert "scripts/npu/qaoa_8card_probe.py" in result.stdout
    assert "--expected-world-size 8" in result.stdout
    assert "--steps 2" in result.stdout
    assert "--samples 16" in result.stdout


def test_npu_runner_suite_targets_exist():
    runner = load_runner_module()

    for suite in runner.SUITES.values():
        for script in suite.scripts:
            script_path = ROOT / script[0]
            assert script_path.exists(), f"{suite.name}: {script[0]}"
        for target in suite.targets:
            target_path = ROOT / target.split("::", 1)[0]
            assert target_path.exists(), f"{suite.name}: {target}"


def test_typed_ir_deriv_probe_help_lists_sections():
    probe = ROOT / "scripts" / "npu" / "typed_ir_deriv_probe.py"
    result = subprocess.run(
        [sys.executable, str(probe), "--help"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0, result.stderr
    assert "typed-ir" in result.stdout
    assert "deriv" in result.stdout


def test_qaoa_probe_help_lists_options():
    probe = ROOT / "scripts" / "npu" / "qaoa_probe.py"
    result = subprocess.run(
        [sys.executable, str(probe), "--help"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0, result.stderr
    assert "--allow-cpu-fallback" in result.stdout
    assert "--shots" in result.stdout


def test_multi_card_probe_help_lists_sections():
    probe = ROOT / "scripts" / "npu" / "multi_card_probe.py"
    result = subprocess.run(
        [sys.executable, str(probe), "--help"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0, result.stderr
    assert "collectives" in result.stdout
    assert "supernet" in result.stdout


def test_demo_npu_qnn_4card_help_lists_training_options():
    demo = ROOT / "demos" / "demo_npu_qnn_4card.py"
    result = subprocess.run(
        [sys.executable, str(demo), "--help"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0, result.stderr
    assert "--steps" in result.stdout
    assert "--layers" in result.stdout
    assert "--allow-cpu-fallback" in result.stdout


def test_qaoa_8card_probe_help_lists_training_options():
    probe = ROOT / "scripts" / "npu" / "qaoa_8card_probe.py"
    result = subprocess.run(
        [sys.executable, str(probe), "--help"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0, result.stderr
    assert "--expected-world-size" in result.stdout
    assert "--steps" in result.stdout
    assert "--samples" in result.stdout
    assert "--allow-cpu-fallback" in result.stdout
