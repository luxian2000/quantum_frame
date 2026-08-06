"""跨框架基准测试脚手架的契约测试。

这里不测"谁更快"——测的是**基准本身是否有效**：

- 线路规格必须可复现（同 seed 同线路），否则不同框架跑的根本不是同一条线路；
- 各框架适配器必须产出**同一个态矢量**（含比特序归一化：Qiskit 小端、Cirq 大端），
  否则计时表比较的是不同的计算；
- 计时必须区分构建与执行、剔除预热、取中位数与 IQR 而非均值；
- 清单必须钉住 commit 与框架版本，否则数字无法追溯。
"""

import numpy as np
import pytest

from scripts.bench.core.spec import CircuitSpec, available_families, build_spec
from scripts.bench.core.timing import TimingStats, time_callable
from scripts.bench.adapters import available_adapters, get_adapter


class TestCircuitSpec:
    """线路规格：框架无关的声明式描述，是所有适配器的唯一输入。"""

    def test_ghz_spec_is_deterministic(self):
        first = build_spec("ghz", n_qubits=4)
        second = build_spec("ghz", n_qubits=4)
        assert first.operations == second.operations

    def test_random_spec_is_reproducible_under_same_seed(self):
        first = build_spec("random", n_qubits=4, depth=3, seed=7)
        second = build_spec("random", n_qubits=4, depth=3, seed=7)
        assert first.operations == second.operations

    def test_random_spec_differs_under_different_seed(self):
        first = build_spec("random", n_qubits=4, depth=3, seed=7)
        second = build_spec("random", n_qubits=4, depth=3, seed=8)
        assert first.operations != second.operations

    def test_families_cover_the_benchmark_axes(self):
        families = set(available_families())
        assert {"ghz", "qft", "random", "layered_ansatz"} <= families

    def test_spec_records_its_own_identity(self):
        spec = build_spec("random", n_qubits=5, depth=2, seed=3)
        assert isinstance(spec, CircuitSpec)
        assert spec.n_qubits == 5
        assert spec.family == "random"
        assert spec.seed == 3


class TestTiming:
    """计时统计：中位数 + IQR，预热不计入。"""

    def test_reports_median_not_mean(self):
        # 每次重复读两次时钟（起、止），故给出成对读数，得到时长 [1, 1, 1, 100]。
        readings = iter([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 100.0])

        stats = time_callable(lambda: None, repeats=4, warmup=0, _clock=lambda: next(readings))
        # 一个 100 秒的离群点不得污染中位数（均值会是 25.75）。
        assert stats.median == pytest.approx(1.0)
        assert stats.maximum == pytest.approx(100.0)

    def test_warmup_runs_are_excluded_from_statistics(self):
        calls = []

        stats = time_callable(lambda: calls.append(1), repeats=3, warmup=2)
        assert len(calls) == 5  # 2 次预热 + 3 次计时
        assert stats.repeats == 3

    def test_reports_interquartile_range(self):
        stats = time_callable(lambda: None, repeats=8, warmup=0)
        assert isinstance(stats, TimingStats)
        assert stats.iqr >= 0.0
        assert stats.minimum <= stats.median


class TestAdapterParity:
    """有效性核心：所有可用框架对同一规格必须给出同一个态矢量。"""

    SPECS = [
        ("ghz", dict(n_qubits=4)),
        ("qft", dict(n_qubits=4)),
        ("random", dict(n_qubits=4, depth=3, seed=11)),
        ("layered_ansatz", dict(n_qubits=4, depth=2, seed=5)),
    ]

    @pytest.mark.parametrize("family,kwargs", SPECS, ids=[s[0] for s in SPECS])
    def test_all_available_adapters_agree_with_aicir(self, family, kwargs):
        spec = build_spec(family, **kwargs)
        reference = get_adapter("aicir").statevector(spec)
        assert reference.shape == (1 << spec.n_qubits,)

        others = [n for n in available_adapters() if n != "aicir"]
        if not others:
            pytest.skip("没有可用的对照框架")

        for name in others:
            got = get_adapter(name).statevector(spec)
            # 允许全局相位差异——物理态相同即可。
            overlap = abs(np.vdot(reference, got))
            assert overlap == pytest.approx(1.0, abs=1e-5), f"{name} 与 aicir 的态不一致 (|<a|b>|={overlap})"

    def test_parity_suite_can_actually_detect_a_bit_order_bug(self):
        """元测试：parity 套件必须真的有检测力。

        ``ghz`` 与 ``qft`` 从 |0…0⟩ 出发得到的末态在比特反转下**不变**
        （GHZ 是 |00…0⟩+|11…1⟩；QFT|0⟩ 是均匀叠加），因此它们对比特序错误完全
        失明。只有 ``random``/``layered_ansatz`` 有区分力。

        这条测试防止后来者把 parity 规格"精简"成只剩 ghz/qft——那样比特序 bug
        会安静地溜过去，而计时表比较的将是不同的线路。
        """

        from scripts.bench.adapters.base import reverse_qubit_order

        discriminating = []
        for family, kwargs in self.SPECS:
            spec = build_spec(family, **kwargs)
            state = get_adapter("aicir").statevector(spec)
            flipped = reverse_qubit_order(state, spec.n_qubits)
            if abs(abs(np.vdot(state, flipped)) - 1.0) > 1e-5:
                discriminating.append(family)

        assert discriminating, "parity 套件里没有任何一条线路能区分比特序错误"

    def test_aicir_adapter_is_always_available(self):
        assert "aicir" in available_adapters()

    def test_unknown_adapter_raises(self):
        with pytest.raises(KeyError):
            get_adapter("definitely_not_a_framework")


class TestManifest:
    """清单：钉住 commit 与框架版本，否则数字不可追溯。"""

    def test_manifest_records_commit_and_framework_versions(self):
        from scripts.bench.core.manifest import build_manifest

        manifest = build_manifest(results=[], failed_conditions=[])
        assert "commit" in manifest
        assert "run_id" in manifest
        assert "frameworks" in manifest
        assert manifest["release_gate"] in {"PASS", "FAIL"}

    def test_manifest_fails_gate_when_conditions_failed(self):
        from scripts.bench.core.manifest import build_manifest

        manifest = build_manifest(results=[], failed_conditions=["parity_mismatch"])
        assert manifest["release_gate"] == "FAIL"

    def test_manifest_records_threading_environment(self):
        from scripts.bench.core.manifest import build_manifest

        manifest = build_manifest(results=[], failed_conditions=[])
        # 线程数不报告，跨机器的 CPU 计时就无法解释。
        assert "threads" in manifest["environment"]
