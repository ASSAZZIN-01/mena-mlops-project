"""Canary traffic split validation and application."""

from __future__ import annotations

import os
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


def split_for_candidate(candidate: int) -> TrafficSplit:
    return TrafficSplit(stable=100 - candidate, candidate=candidate)


def apply_split(split: TrafficSplit) -> None:
    environment = os.environ.copy()
    environment["STABLE_WEIGHT"] = str(split.stable)
    environment["CANDIDATE_WEIGHT"] = str(split.candidate)
    subprocess.run(
        [
            "docker",
            "compose",
            "up",
            "-d",
            "--force-recreate",
            "nginx",
        ],
        env=environment,
        check=True,
    )
