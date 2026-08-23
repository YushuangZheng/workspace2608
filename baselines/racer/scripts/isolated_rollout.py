#!/usr/bin/env python3
"""Run the unmodified RACER evaluator with only RLBenchSim process-isolated."""

from __future__ import annotations

import sys
from pathlib import Path

UPSTREAM_ROOT = Path(__file__).resolve().parents[1] / "upstream"
sys.path.insert(0, str(UPSTREAM_ROOT))
from isolated_simulator_proxy import SpawnIsolatedRLBenchSim
from point_cloud_evidence import evidence_recording_simulator
from racer.evaluation import rollout


def main() -> int:
    # The upstream module refers to its module-global ``args`` from eval().
    args = rollout.make_args()
    rollout.args = args
    rollout.RLBenchSim = evidence_recording_simulator(SpawnIsolatedRLBenchSim)
    evaluator = None
    try:
        evaluator = rollout.Evaluator(args)
        evaluator.eval()
    finally:
        if evaluator is not None and hasattr(evaluator, "env"):
            # A successful isolation gate requires this explicit close and the
            # worker's natural status 0; proxy.close() enforces both.
            evaluator.env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
