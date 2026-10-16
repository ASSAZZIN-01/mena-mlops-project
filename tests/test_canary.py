import pytest

from mena_mlops.deployment.canary import (
    TrafficSplit,
    docker_compose_command,
    split_for_candidate,
)


def test_candidate_split() -> None:
    assert split_for_candidate(20) == TrafficSplit(stable=80, candidate=20)


def test_invalid_split_is_rejected() -> None:
    with pytest.raises(ValueError):
        split_for_candidate(101)


def test_compose_command_is_configurable(monkeypatch) -> None:
    monkeypatch.setenv("DOCKER_COMPOSE_COMMAND", "docker-compose")

    assert docker_compose_command() == ["docker-compose"]
