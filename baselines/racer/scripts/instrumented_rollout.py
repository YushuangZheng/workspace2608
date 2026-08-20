#!/usr/bin/env python3
"""Run official RACER while recording gate-reset point-cloud evidence."""

from __future__ import annotations

import sys
from pathlib import Path

UPSTREAM_ROOT = Path(__file__).resolve().parents[1] / "upstream"
sys.path.insert(0, str(UPSTREAM_ROOT))

from point_cloud_evidence import evidence_recording_simulator
from racer.evaluation import rollout


def main() -> int:
    args = rollout.make_args()
    rollout.args = args
    rollout.RLBenchSim = evidence_recording_simulator(rollout.RLBenchSim)
    evaluator = None
    try:
        evaluator = rollout.Evaluator(args)
        evaluator.eval()
    finally:
        if evaluator is not None and hasattr(evaluator, "env"):
            evaluator.env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
