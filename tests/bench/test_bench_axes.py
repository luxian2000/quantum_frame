"""基准轴 C–G 的契约测试。

轴 A/B（单门吞吐、标准线路）在 `test_bench_harness.py` 覆盖。这里覆盖：

- C：VQE 端到端（单次能量、单次梯度）——变分负载才是本框架的主场；
- D：峰值内存 vs n；
- E：aicir NPU vs CPU（无真机时跳过，但代码路径必须可导入）；
- F：多卡强/弱扩展（同上）；
- G：能力矩阵——哪些框架**能跑**每一行，是三支柱论证的直接证据。

轴 E/F 在没有昇腾硬件的机器上只做结构性校验：能力申报正确、跳过原因被记录。
"""

import numpy as np
import pytest

from scripts.bench.core.spec import build_spec
from scripts.bench.axes import (
    capability_matrix,
    measure_peak_memory,
    run_vqe_axis,
    strong_scaling_plan,
)


class TestCapabilityMatrix:
    """轴 G：能力矩阵。论文的核心论证靠它，不能靠手写表格。"""

    def test_lists_every_known_framework(self):
        matrix = capability_matrix()
        assert "aicir" in matrix
        # 未安装的框架也要出现，标为不可用，而不是从表里消失。
        assert "qulacs" in matrix

    def test_records_availability_and_capabilities(self):
        matrix = capability_matrix()
        entry = matrix["aicir"]
        assert entry["available"] is True
        for key in ("statevector", "single_state_sharding", "non_cuda_accelerator", "architecture_search"):
            assert key in entry["capabilities"]

    def test_only_aicir_claims_sharding_and_ascend(self):
        """这是论文最强的一条主张，必须由数据而非散文支撑。"""

        matrix = capability_matrix()
        for name, entry in matrix.items():
            if name == "aicir":
                continue
            assert entry["capabilities"].get("single_state_sharding") is False
            assert entry["capabilities"].get("non_cuda_accelerator") is False

    def test_capability_claims_are_marked_with_evidence(self):
        """每条 True 都要能指向证据，否则就是自说自话。"""

        entry = capability_matrix()["aicir"]
        assert entry["evidence"], "aicir 的能力主张必须附证据引用"


class TestMemoryAxis:
    """轴 D：峰值内存。"""

    def test_reports_bytes_and_scales_with_qubits(self):
        small = measure_peak_memory("aicir", build_spec("ghz", n_qubits=8))
        large = measure_peak_memory("aicir", build_spec("ghz", n_qubits=14))
        assert small["peak_bytes"] > 0
        # 2^14 比 2^8 大 64 倍，峰值必须体现出增长。
        assert large["peak_bytes"] > small["peak_bytes"]

    def test_records_theoretical_minimum_for_comparison(self):
        record = measure_peak_memory("aicir", build_spec("ghz", n_qubits=12))
        # 理论下界 = 2^n * sizeof(complex)；实测远超它说明有额外拷贝。
        assert record["theoretical_bytes"] == (1 << 12) * 16
        assert record["overhead_ratio"] >= 1.0


class TestVQEAxis:
    """轴 C：变分负载端到端。"""

    def test_reports_energy_and_gradient_timings_separately(self):
        result = run_vqe_axis("aicir", n_qubits=4, layers=1, repeats=2, warmup=0)
        assert "energy_eval" in result
        assert "gradient_eval" in result
        assert result["energy_eval"]["median"] > 0

    def test_energy_is_physically_sane(self):
        """能量必须落在哈密顿量谱范围内——计时再快，算错了也没用。"""

        result = run_vqe_axis("aicir", n_qubits=4, layers=1, repeats=1, warmup=0)
        assert result["n_parameters"] > 0
        assert np.isfinite(result["energy"])
        assert abs(result["energy"]) <= result["spectral_bound"] + 1e-9

    def test_gradient_length_matches_parameter_count(self):
        result = run_vqe_axis("aicir", n_qubits=4, layers=1, repeats=1, warmup=0)
        assert result["gradient_length"] == result["n_parameters"]


class TestScalingPlan:
    """轴 F：多卡扩展。无真机时只校验计划本身自洽。"""

    def test_strong_scaling_requires_power_of_two_world_sizes(self):
        plan = strong_scaling_plan(n_qubits=20, world_sizes=(1, 2, 4, 8))
        assert [row["world_size"] for row in plan] == [1, 2, 4, 8]

    def test_rejects_non_power_of_two(self):
        with pytest.raises(ValueError):
            strong_scaling_plan(n_qubits=20, world_sizes=(3,))

    def test_rejects_world_size_exceeding_qubit_budget(self):
        # world_size = 2^p 要求 n_qubits >= p；n=2 时 W=8 (p=3) 不合法。
        with pytest.raises(ValueError):
            strong_scaling_plan(n_qubits=2, world_sizes=(8,))

    def test_weak_scaling_grows_qubits_with_world_size(self):
        plan = strong_scaling_plan(n_qubits=20, world_sizes=(1, 2, 4), mode="weak")
        qubits = [row["n_qubits"] for row in plan]
        assert qubits == [20, 21, 22], "弱扩展下每翻倍一次卡数，应多装一个比特"
