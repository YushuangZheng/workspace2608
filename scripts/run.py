#!/usr/bin/env python3
"""Command-line entry point for the DynaMAC and MiDiGaP implementations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from essay2608.data import load_demonstrations  # noqa: E402
from essay2608.policy import BimanualDynaMAC, DynaMAC, DynaMACConfig  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/dynamac_demos.npz"))
    parser.add_argument(
        "--config", type=Path, default=Path("configs/dynamac_smoke.json")
    )
    commands = parser.add_subparsers(dest="command", required=True)

    fit = commands.add_parser("fit", help="Fit the bundled demonstrations and save a checkpoint")
    fit.add_argument("--task", choices=("single", "bimanual"), required=True)
    fit.add_argument("--output", type=Path, required=True)

    inspect = commands.add_parser("inspect", help="Inspect a DynaMAC checkpoint without modifying it")
    inspect.add_argument("checkpoint", type=Path)

    commands.add_parser(
        "verify",
        help="Fit the bundled single-arm and bimanual data and print a summary",
    )
    return parser.parse_args()


def load_config(path: Path) -> DynaMACConfig:
    return DynaMACConfig(**json.loads(path.read_text(encoding="utf-8")))


def compact_summary(policy: DynaMAC) -> dict:
    return {
        "fingerprint": policy.fingerprint(),
        "frames": list(policy.frame_names),
        "map_modal_path": list(policy._select_mode_path("map")),
        "skills": [
            {
                "label": skill.label,
                "duration": skill.duration,
                "modes": len(skill.mode_priors),
                "selected_frames": list(skill.selected_frames),
                "linked_frames": [
                    name for name, values in skill.link_diagnostics.items() if values["linked"]
                ],
            }
            for skill in policy.skills
        ],
    }


def main() -> None:
    args = arguments()
    if args.command == "inspect":
        policy = DynaMAC.load(args.checkpoint)
        print(json.dumps(policy.summary(), ensure_ascii=False, indent=2))
        return

    bundle = load_demonstrations(args.data)
    config = load_config(args.config)
    if args.command == "fit" and args.task == "single":
        policy = DynaMAC(config).fit(bundle.single_arm)
        policy.save(args.output)
        print(json.dumps(compact_summary(policy), ensure_ascii=False, indent=2))
        return
    if args.command == "fit" and args.task == "bimanual":
        policy = BimanualDynaMAC(config=config).fit(bundle.left_arm, bundle.right_arm)
        args.output.mkdir(parents=True, exist_ok=True)
        policy.left.save(args.output / "left.npz")
        policy.right.save(args.output / "right.npz")
        print(
            json.dumps(
                {"left": compact_summary(policy.left), "right": compact_summary(policy.right)},
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    bimanual = BimanualDynaMAC(config=config).fit(bundle.left_arm, bundle.right_arm)
    single = DynaMAC(config).fit(bundle.single_arm)
    print(
        json.dumps(
            {
                "data_schema": bundle.metadata["schema"],
                "single": compact_summary(single),
                "bimanual_left": compact_summary(bimanual.left),
                "bimanual_right": compact_summary(bimanual.right),
                "claim_boundary": (
                    "Structural verification only; the bundled data do not produce paper "
                    "benchmark results or replace simulator task data."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
