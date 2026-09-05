"""Build A1 DynaMAC and closed-loop task models from five demonstrations."""

from __future__ import annotations

import argparse
from pathlib import Path

from integrations.rlbench.iclr2027.task_registry import TASKS, experiment_task
from integrations.rlbench.rlbench_closed_loop.build_models import (
    build_task as build_closed_loop,
)
from integrations.rlbench.rlbench_dynamac.core.paths import (
    INTEGRATION_ROOT,
    REPOSITORY_ROOT,
)
from integrations.rlbench.rlbench_dynamac.core.task_specs import (
    TRAINING_MANIFEST_SCHEMA_STATIC_V1,
)
from integrations.rlbench.rlbench_dynamac.data.direct_policy import train_task
from integrations.rlbench.rlbench_dynamac.data.tapas_segmentation import (
    load_rlbench_segmentation_config,
)

DATA_ROOT = INTEGRATION_ROOT / "data" / "iclr2027" / "demonstrations"
MODEL_ROOT = INTEGRATION_ROOT / "models" / "iclr2027"
DYNAMAC_ROOT = MODEL_ROOT / "dynamac"
CLOSED_LOOP_ROOT = MODEL_ROOT / "closed_loop"
PHASE6_DATA_ROOT = INTEGRATION_ROOT / "data" / "training" / "main"
PHASE6_DYNAMAC_ROOT = INTEGRATION_ROOT / "models" / "phase6_v1"
PHASE6_SEGMENTATION_CONFIG = INTEGRATION_ROOT / "configs" / "tapas_segmentation.json"
SEGMENTATION_CONFIG = (
    INTEGRATION_ROOT / "configs" / "iclr2027" / "tapas_segmentation.json"
)
UNCALIBRATED_BOUNDARY_CONFIG = (
    INTEGRATION_ROOT / "configs" / "iclr2027" / "boundary_uncalibrated.json"
)


def build_dynamac(task_id: str) -> Path:
    task = experiment_task(task_id)
    if task.spec.bimanual:
        raise ValueError(f"{task_id} is reused from the authenticated phase-six assets")
    config = load_rlbench_segmentation_config(SEGMENTATION_CONFIG)
    train_task(
        task_id,
        data_root=DATA_ROOT,
        models_dir=DYNAMAC_ROOT,
        config_path=INTEGRATION_ROOT / "configs" / "dynamac_rlbench_v3.json",
        demonstration_count=5,
        task_spec=task.spec,
        manifest_schema=TRAINING_MANIFEST_SCHEMA_STATIC_V1,
        segmentation_config=config,
    )
    return DYNAMAC_ROOT / task_id / "training.json"


def build_closed(
    task_id: str,
    *,
    output_root: Path = CLOSED_LOOP_ROOT,
) -> Path:
    task = experiment_task(task_id)
    config = load_rlbench_segmentation_config(
        PHASE6_SEGMENTATION_CONFIG if task.spec.bimanual else SEGMENTATION_CONFIG
    )
    return build_closed_loop(
        task_id,
        data_root=PHASE6_DATA_ROOT if task.spec.bimanual else DATA_ROOT,
        base_models=PHASE6_DYNAMAC_ROOT if task.spec.bimanual else DYNAMAC_ROOT,
        output_root=output_root,
        demonstration_count=5,
        task_model_config=REPOSITORY_ROOT / "configs" / "closed_loop_task_model.json",
        belief_config=REPOSITORY_ROOT / "configs" / "closed_loop_belief.json",
        execution_config=REPOSITORY_ROOT / "configs" / "closed_loop_execution.json",
        recovery_config=REPOSITORY_ROOT / "configs" / "closed_loop_recovery.json",
        boundary_root=REPOSITORY_ROOT / "configs" / "closed_loop_boundary",
        task_spec=task.spec,
        boundary_config=UNCALIBRATED_BOUNDARY_CONFIG,
        segmentation_config=config,
    )


def build_parser() -> argparse.ArgumentParser:
    available = sorted(TASKS)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", action="append", choices=available + ["all"])
    parser.add_argument(
        "--component", choices=("dynamac", "closed_loop", "all"), default="all"
    )
    parser.add_argument("--closed-loop-output", type=Path, default=CLOSED_LOOP_ROOT)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    available = sorted(TASKS)
    selected = available if not args.task or args.task == ["all"] else args.task
    for task_id in selected:
        if args.component in {"dynamac", "all"}:
            if TASKS[task_id].spec.bimanual:
                if args.component == "dynamac":
                    raise ValueError(
                        f"{task_id} reuses its authenticated phase-six DynaMAC"
                    )
            else:
                print(f"{task_id}: {build_dynamac(task_id)}", flush=True)
        if args.component in {"closed_loop", "all"}:
            print(
                f"{task_id}: "
                f"{build_closed(task_id, output_root=args.closed_loop_output)}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
