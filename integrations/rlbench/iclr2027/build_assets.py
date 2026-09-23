"""Build DynaMAC backbones and TSF task models from five demonstrations."""

from __future__ import annotations

import argparse
from pathlib import Path

from integrations.rlbench.iclr2027.task_registry import TASKS, experiment_task
from integrations.rlbench.rlbench_tsf.build_models import build_task as build_tsf
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
TSF_MODEL_ROOT = MODEL_ROOT / "tsf"
BIMANUAL_DATA_ROOT = INTEGRATION_ROOT / "data" / "training" / "main"
DYNAMAC_BACKBONE_ROOT = INTEGRATION_ROOT / "models" / "dynamac_backbone_v1"
BIMANUAL_SEGMENTATION_CONFIG = (
    INTEGRATION_ROOT / "configs" / "tapas_segmentation.json"
)
SEGMENTATION_CONFIG = (
    INTEGRATION_ROOT / "configs" / "iclr2027" / "tapas_segmentation.json"
)
UNCALIBRATED_BOUNDARY_CONFIG = (
    INTEGRATION_ROOT / "configs" / "iclr2027" / "boundary_uncalibrated.json"
)


def build_dynamac(task_id: str) -> Path:
    task = experiment_task(task_id)
    if task.spec.bimanual:
        raise ValueError(f"{task_id} reuses the authenticated bimanual backbone")
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


def build_tsf_task(
    task_id: str,
    *,
    output_root: Path = TSF_MODEL_ROOT,
) -> Path:
    task = experiment_task(task_id)
    config = load_rlbench_segmentation_config(
        BIMANUAL_SEGMENTATION_CONFIG if task.spec.bimanual else SEGMENTATION_CONFIG
    )
    return build_tsf(
        task_id,
        data_root=BIMANUAL_DATA_ROOT if task.spec.bimanual else DATA_ROOT,
        base_models=DYNAMAC_BACKBONE_ROOT if task.spec.bimanual else DYNAMAC_ROOT,
        output_root=output_root,
        demonstration_count=5,
        task_model_config=REPOSITORY_ROOT / "configs" / "tsf_task_model.json",
        belief_config=REPOSITORY_ROOT / "configs" / "tsf_inference.json",
        execution_config=REPOSITORY_ROOT / "configs" / "tsf_execution.json",
        recovery_config=REPOSITORY_ROOT / "configs" / "tsf_recovery.json",
        boundary_root=REPOSITORY_ROOT / "configs" / "tsf_boundaries",
        task_spec=task.spec,
        boundary_config=UNCALIBRATED_BOUNDARY_CONFIG,
        segmentation_config=config,
    )


def build_parser() -> argparse.ArgumentParser:
    available = sorted(TASKS)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", action="append", choices=available + ["all"])
    parser.add_argument(
        "--component", choices=("dynamac", "tsf", "all"), default="all"
    )
    parser.add_argument("--tsf-output", type=Path, default=TSF_MODEL_ROOT)
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
                        f"{task_id} reuses its authenticated bimanual DynaMAC"
                    )
            else:
                print(f"{task_id}: {build_dynamac(task_id)}", flush=True)
        if args.component in {"tsf", "all"}:
            print(
                f"{task_id}: "
                f"{build_tsf_task(task_id, output_root=args.tsf_output)}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
