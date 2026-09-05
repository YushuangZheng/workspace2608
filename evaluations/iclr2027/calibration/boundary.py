"""Calibrate Full-method boundary runtime parameters from five success demos.

This is the ICLR task-registry adapter around the already validated phase-four
normal-demonstration calibration.  It deliberately reads neither the A-only
monitor-calibration rollouts nor any fault/sealed-test artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from essay2608.policy import (
    BimanualDynaMAC,
    DynaMAC,
    synchronized_bimanual_demonstrations,
)
from essay2608.policy.closed_loop import (
    ClosedLoopMultiStreamPolicy,
    ClosedLoopTaskModelBuilder,
    ClosedLoopTaskModelConfig,
)
from evaluations.development.phase23_component_ab.run import ArmCase
from evaluations.development.phase4_boundary_calibration.run import run
from integrations.rlbench.iclr2027.build_assets import (
    CLOSED_LOOP_ROOT,
    DATA_ROOT,
    DYNAMAC_ROOT,
    PHASE6_DATA_ROOT,
    PHASE6_DYNAMAC_ROOT,
    PHASE6_SEGMENTATION_CONFIG,
    SEGMENTATION_CONFIG,
)
from integrations.rlbench.iclr2027.task_registry import experiment_task
from integrations.rlbench.rlbench_dynamac.core.paths import REPOSITORY_ROOT
from integrations.rlbench.rlbench_dynamac.data.demo_adapter import (
    load_low_dim_obs_pickles,
    make_bimanual_demonstrations,
    make_unimanual_demonstrations,
)
from integrations.rlbench.rlbench_dynamac.data.direct_policy import (
    demonstration_paths,
)
from integrations.rlbench.rlbench_dynamac.data.tapas_segmentation import (
    load_rlbench_segmentation_config,
)

DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "evaluations/iclr2027/configs/shared/normal_task_boundary_calibration.json"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "evaluations/iclr2027/artifacts/calibration/normal_task_boundaries/v1"
)
TASK_MODEL_CONFIG = REPOSITORY_ROOT / "configs/closed_loop_task_model.json"


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def load_cases(task_id: str, demonstration_count: int) -> list[ArmCase]:
    """Load the exact A1 models and their originating successful demos."""

    task = experiment_task(task_id)
    data_root = PHASE6_DATA_ROOT if task.spec.bimanual else DATA_ROOT
    base_root = PHASE6_DYNAMAC_ROOT if task.spec.bimanual else DYNAMAC_ROOT
    if demonstration_count < 1:
        raise ValueError("calibration requires at least one normal replay")
    paths = tuple(demonstration_paths(data_root, task_id, demonstration_count))
    episodes = load_low_dim_obs_pickles(paths)
    names = [path.parent.name for path in paths]
    segmentation = load_rlbench_segmentation_config(
        PHASE6_SEGMENTATION_CONFIG if task.spec.bimanual else SEGMENTATION_CONFIG
    )
    converted = (
        make_bimanual_demonstrations(
            episodes,
            task.spec,
            names=names,
            config=segmentation,
        )
        if task.spec.bimanual
        else make_unimanual_demonstrations(
            episodes,
            task.spec,
            names=names,
            config=segmentation,
        )
    )
    builder = ClosedLoopTaskModelBuilder(
        ClosedLoopTaskModelConfig(**_json(TASK_MODEL_CONFIG))
    )
    model_root = base_root / task_id
    if task.spec.bimanual:
        base = BimanualDynaMAC(
            left=DynaMAC.load(model_root / "left.npz"),
            right=DynaMAC.load(model_root / "right.npz"),
        )
        base_policies = {"left": base.left, "right": base.right}
    else:
        base = DynaMAC.load(model_root / "model.npz")
        base_policies = {"single": base}
    closed = ClosedLoopMultiStreamPolicy.load(
        CLOSED_LOOP_ROOT / task_id,
        base_policies=base_policies,
    )
    if not task.spec.bimanual:
        demonstrations = converted.demonstrations
        return [
            ArmCase(
                task_id,
                "single",
                base,
                closed.task_models["single"],
                demonstrations,
                builder._align_demonstrations(base, demonstrations),
                task.spec.recoverable_relation_frames,
                paths,
                model_root / "model.npz",
            )
        ]

    left_demos, right_demos = synchronized_bimanual_demonstrations(
        converted.left_demonstrations,
        converted.right_demonstrations,
    )
    result = []
    for arm, policy, demonstrations in (
        ("left", base.left, left_demos),
        ("right", base.right, right_demos),
    ):
        result.append(
            ArmCase(
                task_id,
                arm,
                policy,
                closed.task_models[arm],
                demonstrations,
                builder._align_demonstrations(policy, demonstrations),
                task.spec.recoverable_relation_frames,
                paths,
                model_root / f"{arm}.npz",
            )
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    run(
        arguments.config,
        arguments.output,
        case_loader=load_cases,
        include_skill_entry_cycles=True,
    )


if __name__ == "__main__":
    main()
