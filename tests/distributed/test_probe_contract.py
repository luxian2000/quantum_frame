from pathlib import Path


PROBE = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "npu"
    / "distributed_state_probe.py"
)
API_PROBE = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "npu"
    / "distributed_api_probe.py"
)
MANUAL = (
    Path(__file__).resolve().parents[2]
    / "aicir"
    / "distributed"
    / "README.md"
)
EXPECTED_SECTIONS = {
    "state",
    "layout",
    "continuation",
    "noise",
    "observable",
    "measure",
    "result",
    "communication",
    "contract",
}


def test_probe_is_strict_and_covers_distributed_contract():
    source = PROBE.read_text(encoding="utf-8")

    assert "fallback_to_cpu=False" in source
    assert "local_rank" in source
    assert 'device.type != "npu"' in source
    assert "local_gate" in source
    assert "communicating_gate" in source
    assert "density" in source
    assert "AmplitudeDampingChannel" in source
    assert "expectation" in source
    assert "shots=" in source
    assert "local_tensor_sizes" in source
    assert "if backend.rank == 0" in source
    assert "json.dumps" in source


def test_npu_launch_instructions_preserve_cann_pythonpath():
    for path in (PROBE, MANUAL):
        source = path.read_text(encoding="utf-8")
        launch_lines = [
            line.strip()
            for line in source.splitlines()
            if line.strip().startswith("PYTHONPATH=")
            and "torchrun" in line
        ]
        assert launch_lines
        assert all(
            line.startswith("PYTHONPATH=.:${PYTHONPATH} torchrun")
            for line in launch_lines
        )


def test_full_api_probe_has_sectioned_strict_contract():
    source = API_PROBE.read_text(encoding="utf-8")

    assert "fallback_to_cpu=False" in source
    assert "failed_invariants" in source
    assert "EXPECTED_SECTIONS" in source
    assert ".kraus_operators(" not in source
    assert "._local_kraus(" not in source
    for token in (
        "AmplitudeDampingChannel",
        "BitFlipChannel",
        "DepolarizingChannel",
        "Hamiltonian",
        "Observable.matrix",
        "PhaseFlipChannel",
        "amplitude_damping_error",
        "bit_flip_error",
        "depolarizing_error",
        "initial_density_matrix",
        "hamiltonian_error",
        "local_dense_error",
        "logical_to_storage",
        "noise_density_error",
        "noise_sequence_error",
        "phase_flip_error",
        "rule_selection_error",
        "targeted_distributed_axes",
        "continuation_vector_error",
        "continuation_density_error",
        "local_tensor_sizes",
        "collapse=True",
        "measure_qubits",
        "collapsed_support_error",
        "return_state",
        "return_probabilities",
        "four_return_combinations",
        "local_p2p_delta",
        "distributed_p2p_delta",
        "peer_mask",
        "paired_transport_tags",
        "UNSUPPORTED_AS_DESIGNED",
        "EXPECTED_ERROR",
    ):
        assert token in source
    assert '"local_gate": True' not in source
    assert '"communicating_gate": True' not in source
    assert (
        "def _expected_error(call, expected_message, expected_type):"
        in source
    )
    assert "str(match) in message" not in source
    for section in EXPECTED_SECTIONS:
        assert f'"{section}"' in source


def test_full_api_probe_launch_preserves_cann_pythonpath():
    source = API_PROBE.read_text(encoding="utf-8")
    launch_lines = [
        line.strip()
        for line in source.splitlines()
        if line.strip().startswith("PYTHONPATH=")
        and "torchrun" in line
    ]

    assert launch_lines
    assert all(
        line.startswith("PYTHONPATH=.:${PYTHONPATH} torchrun")
        for line in launch_lines
    )


def test_manual_documents_full_npu_api_acceptance_contract():
    source = MANUAL.read_text(encoding="utf-8")
    normalized = " ".join(source.split())

    for token in (
        "完整 NPU API 验收",
        "source /usr/local/Ascend/cann/set_env.sh",
        "torchrun --nproc-per-node=2 scripts/npu/distributed_api_probe.py --section all",
        "torchrun --nproc-per-node=4 scripts/npu/distributed_api_probe.py --section all",
        "`state`",
        "`layout`",
        "`continuation`",
        "`noise`",
        "`observable`",
        "`measure`",
        "`result`",
        "`communication`",
        "`contract`",
        "`status`",
        "`failed_invariants`",
        "`local_tensor_sizes`",
        "`fallback_to_cpu=False`",
        "DistSimulator 首期仅支持前向模拟，不支持自动微分",
        "Gloo",
        "不能作为 NPU 验收",
    ):
        assert token in source
    assert (
        "`sections.<name>.status` | 正常支持项只能为 `PASS` 或 `FAIL`"
        in source
    )
    assert "`sections.contract.metrics.case_statuses`" in source
    assert (
        "`sections.contract.metrics.case_evidence[*].status`"
        in source
    )
    assert (
        "正常支持项必须为 `PASS`；契约中的预期拒绝项记录为 "
        "`EXPECTED_ERROR` 或 `UNSUPPORTED_AS_DESIGNED`"
        not in source
    )
    assert (
        "独立数值 oracle 的最终演算和判定在 rank 0 执行；各 rank "
        "仍可能使用 NumPy 构造或整理探针输入。 NumPy 不参与被测的"
        "分布式状态演化，也不是模拟 fallback。"
        in normalized
    )
