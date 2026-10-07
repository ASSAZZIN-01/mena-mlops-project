"""Control the local Nginx canary split and recreate only its gateway."""

from __future__ import annotations

import argparse

from mena_mlops.deployment.canary import apply_split, split_for_candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument(
        "--candidate",
        type=int,
        help="Set candidate traffic percentage",
    )
    actions.add_argument(
        "--rollback",
        action="store_true",
        help="Route 100%% of traffic to stable",
    )
    args = parser.parse_args()
    candidate = 0 if args.rollback else args.candidate
    if candidate is None:
        parser.error("--candidate is required unless --rollback is used")
    split = split_for_candidate(candidate)
    print(f"Applying stable={split.stable}% candidate={split.candidate}%")
    apply_split(split)


if __name__ == "__main__":
    main()
