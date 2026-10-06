import pytest

from mena_mlops.deployment.canary import TrafficSplit, split_for_candidate


def test_candidate_split() -> None:
    assert split_for_candidate(20) == TrafficSplit(stable=80, candidate=20)


def test_invalid_split_is_rejected() -> None:
    with pytest.raises(ValueError):
        split_for_candidate(101)
