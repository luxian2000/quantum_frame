import numpy as np
import pytest

from aicir.qec.detectors import Detector, DetectorLayout, Observable


def _layout(n_rounds=3, n_stab=2, round0=(0,)):
    """稳定子 0 在轮 0 确定（有 detector），稳定子 1 不确定（轮 0 无 detector）。"""
    dets, idx = [], 0
    for r in range(n_rounds):
        for s in range(n_stab):
            if r == 0 and s not in round0:
                continue                     # 轮 0 只对确定的生成元建 detector
            recs = (r * n_stab + s,) if r == 0 else ((r - 1) * n_stab + s, r * n_stab + s)
            dets.append(Detector(index=idx, records=recs, stabilizer=s, round_index=r))
            idx += 1
    obs = [Observable(index=0, records=(n_rounds * n_stab,))]
    return DetectorLayout(
        n_detectors=idx, n_rounds=n_rounds, n_stabilizers=n_stab,
        detectors=tuple(dets), observables=tuple(obs), coords={},
        round0_stabilizers=tuple(round0),
    )


def test_layout_shape_and_lookup():
    layout = _layout()
    # 轮 0 只有 1 个 detector（稳定子 0），轮 1/2 各 2 个 → 共 5
    assert layout.n_detectors == 5
    d = layout.detector_at(stabilizer=1, round_index=2)
    assert d.stabilizer == 1 and d.round_index == 2 and d.index == 4


def test_round_slice_returns_that_rounds_detectors():
    layout = _layout()
    assert layout.round_slice(0) == (0,)          # 轮 0 只有确定的那一个
    assert layout.round_slice(2) == (3, 4)


def test_detection_events_round_zero_masks_nondeterministic_stabilizers():
    """轮 0 读数随机的生成元不构成 detector，其事件必须被掩为 0。"""
    layout = _layout()
    raw = np.array([[1, 1], [1, 0], [1, 1]], dtype=np.uint8)
    ref = np.array([0, 0], dtype=np.uint8)
    ev0 = layout.detection_events(raw, 0, ref)
    # 稳定子 0 确定：1 ^ 0 = 1 ；稳定子 1 不确定：掩为 0（尽管原始读数是 1）
    assert list(ev0) == [1, 0]


def test_detection_events_later_rounds_are_differences_unmasked():
    layout = _layout()
    raw = np.array([[1, 1], [1, 0], [1, 1]], dtype=np.uint8)
    ref = np.array([0, 0], dtype=np.uint8)
    # 轮 1：稳定子 1 由 1 变 0 → 事件；轮 1 不掩码
    assert list(layout.detection_events(raw, 1, ref)) == [0, 1]
    assert list(layout.detection_events(raw, 2, ref)) == [0, 1]


def test_all_deterministic_layout_masks_nothing_at_round_zero():
    layout = _layout(round0=(0, 1))
    raw = np.array([[1, 1], [1, 1], [1, 1]], dtype=np.uint8)
    ref = np.array([0, 0], dtype=np.uint8)
    assert list(layout.detection_events(raw, 0, ref)) == [1, 1]


def test_detector_at_rejects_unknown_pair():
    layout = _layout()
    with pytest.raises(KeyError):
        layout.detector_at(stabilizer=9, round_index=0)
    with pytest.raises(KeyError):
        layout.detector_at(stabilizer=1, round_index=0)   # 轮 0 该生成元无 detector
