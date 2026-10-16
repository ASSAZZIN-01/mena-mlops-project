"""Canary traffic split validation and application."""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class TrafficSplit:
    stable: int
    candidate: int

    def __post_init__(self) -> None:
        if not 0 <= self.candidate <= 100:
            raise ValueError("candidate percentage must be between 0 and 100")
        if self.stable + self.candidate != 100:
            raise ValueError("stable and candidate percentages must total 100")


def docker_compose_command() -> list[str]:
    """Return the configured Docker Compose executable and arguments."""

    return shlex.split(os.getenv("DOCKER_COMPOSE_COMMAND", "docker compose"))


def split_for_candidate(candidate: int) -> TrafficSplit:
    return TrafficSplit(stable=100 - candidate, candidate=candidate)


def apply_split(split: TrafficSplit) -> None:
    environment = os.environ.copy()
    environment["STABLE_WEIGHT"] = str(split.stable)
    environment["CANDIDATE_WEIGHT"] = str(split.candidate)
    subprocess.run(
        [*docker_compose_command(), "rm", "-sf", "nginx"],
        env=environment,
        check=True,
    )
    subprocess.run(
        [*docker_compose_command(), "up", "-d", "--no-deps", "nginx"],
        env=environment,
        check=True,
    )
