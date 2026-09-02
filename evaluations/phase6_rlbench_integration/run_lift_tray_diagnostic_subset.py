"""Run one immutable LiftTray evaluation episode as a diagnostic subset.

The formal evaluator intentionally accepts only the complete 200-episode set.
This helper selects one already sealed plan by index without regenerating it and
marks the output as non-paper-comparable diagnostic evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from integrations.rlbench.rlbench_dynamac.core.records import atomic_json
from integrations.rlbench.rlbench_dynamac.eval import direct_evaluate
from integrations.rlbench.rlbench_dynamac.eval.eval_set import (
    FIXED_EVAL_EPISODES,
    GLOBAL_EVAL_SEED_START,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diagnostics-dir", type=Path, required=True)
    parser.add_argument("--policy-python", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=400)
    return parser


def _diagnostic_loader(episode_index: int):
    original = direct_evaluate._load_fixed_motion_plans

    def load(args: Any):
        requested_seed = args.seed
        requested_episodes = args.episodes
        args.seed = GLOBAL_EVAL_SEED_START
        args.episodes = FIXED_EVAL_EPISODES
        try:
            manifest, selected = original(args)
        finally:
            args.seed = requested_seed
            args.episodes = requested_episodes
        subset = dict(selected)
        subset["plans"] = [selected["plans"][episode_index]]
        return manifest, subset

    return load


def _mark_diagnostic(path: Path, episode_index: int) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["diagnostic_subset"] = {
        "schema": "rlbench-fixed-eval-read-only-diagnostic-subset-v1",
        "episode_index": episode_index,
        "episode_seed": GLOBAL_EVAL_SEED_START + episode_index,
        "formal_result": False,
        "paper_comparable": False,
        "plan_regenerated": False,
    }
    payload["evaluation_protocol_id"] = (
        f"{payload['evaluation_protocol_id']}+diagnostic-subset-v1"
    )
    payload["fixed_eval_set"][
        "formal_access"
    ] = "canonical_id_read_only_diagnostic_subset"
    payload["scenario_protocol"]["paper_comparable"] = False
    atomic_json(path, payload)


def main() -> int:
    args = build_parser().parse_args()
    if not 0 <= args.episode_index < FIXED_EVAL_EPISODES:
        raise ValueError("episode index lies outside the sealed 200-episode set")
    if args.horizon < 1:
        raise ValueError("horizon must be positive")
    direct_evaluate._load_fixed_motion_plans = _diagnostic_loader(args.episode_index)
    result = direct_evaluate.main(
        [
            "--task",
            "bimanual_lift_tray",
            "--models-dir",
            "integrations/rlbench/models/v4",
            "--policy-type",
            "closed_loop_multistream",
            "--closed-loop-models-dir",
            "integrations/rlbench/models/closed_loop_phase6_v1",
            "--policy-diagnostics-dir",
            str(args.diagnostics_dir),
            "--controller-profile",
            "auto",
            "--policy-python",
            str(args.policy_python),
            "--episodes",
            "1",
            "--seed",
            str(GLOBAL_EVAL_SEED_START + args.episode_index),
            "--horizon",
            str(args.horizon),
            "--scenario",
            "static",
            "--eval-set-id",
            "rlbench_eval_v2",
            "--release",
            "v4",
            "--headless",
            "--output",
            str(args.output),
        ]
    )
    if result == 0:
        _mark_diagnostic(args.output, args.episode_index)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
