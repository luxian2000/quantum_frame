from unittest.mock import Mock

import torch

from aicir.distributed import DistResult


def test_result_construction_never_implicitly_gathers():
    state = Mock()
    result = DistResult(
        state=state,
        local_probabilities=torch.tensor([0.5, 0.5]),
        expectations={"z": 0.0},
        counts={"0": 1},
        rank=0,
        world_size=2,
    )

    assert result.state is state
    assert result.is_root
    state.gather.assert_not_called()
    state.to_numpy.assert_not_called()


def test_non_root_result_has_no_counts():
    result = DistResult(
        state=None,
        local_probabilities=None,
        expectations={},
        counts=None,
        rank=1,
        world_size=2,
    )

    assert not result.is_root
    assert result.counts is None

