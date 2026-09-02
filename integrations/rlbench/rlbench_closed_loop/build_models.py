"""Build reproducible closed-loop sidecars beside, never inside, V4 models."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Sequence

from essay2608.policy import BimanualDynaMAC, DynaMAC
from essay2608.policy.closed_loop import (
    ClosedLoopMultiStreamPolicy,
    ClosedLoopPolicyConfig,
    ClosedLoopTaskModelBuilder,
    ClosedLoopTaskModelConfig,
)
from integrations.rlbench.rlbench_dynamac.core.paths import REPOSITORY_ROOT
from integrations.rlbench.rlbench_dynamac.core.records import atomic_json
from integrations.rlbench.rlbench_dynamac.core.task_specs import (
    get_task_spec,
    load_task_specs,
)
from integrations.rlbench.rlbench_dynamac.data.demo_adapter import (
    load_low_dim_obs_pickles,
    make_bimanual_demonstrations,
    make_unimanual_demonstrations,
)
from integrations.rlbench.rlbench_dynamac.data.direct_policy import demonstration_paths
from integrations.rlbench.rlbench_dynamac.protocols.store_bottle_semantics import (
    STORE_BOTTLE_TASK_NAME,
    make_store_bottle_semantic_demonstrations,
    store_bottle_semantic_task_spec,
)


DEFAULT_DATA_ROOT = REPOSITORY_ROOT / "integrations/rlbench/data/training/main"
DEFAULT_BASE_MODELS = REPOSITORY_ROOT / "integrations/rlbench/models/phase6_v1"
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT / "integrations/rlbench/models/closed_loop_phase6_v1"
)
DEFAULT_TASK_CONFIG = REPOSITORY_ROOT / "configs/closed_loop_task_model.json"
DEFAULT_BELIEF_CONFIG = REPOSITORY_ROOT / "configs/closed_loop_belief.json"
DEFAULT_EXECUTION_CONFIG = REPOSITORY_ROOT / "configs/closed_loop_execution.json"
DEFAULT_RECOVERY_CONFIG = REPOSITORY_ROOT / "configs/closed_loop_recovery.json"
DEFAULT_BOUNDARY_ROOT = REPOSITORY_ROOT / "configs/closed_loop_boundary"
BUILD_SCHEMA = "essay2608.rlbench.closed_loop_build.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"配置必须是 JSON 对象：{path}")
    return value


def build_task(
    task: str,
    *,
    data_root: Path,
    base_models: Path,
    output_root: Path,
    demonstration_count: int,
    task_model_config: Path,
    belief_config: Path,
    execution_config: Path,
    recovery_config: Path,
    boundary_root: Path,
) -> Path:
    output = output_root / task
    if output.exists():
        raise FileExistsError(f"拒绝覆盖既有闭环模型：{output}")
    paths = tuple(demonstration_paths(data_root, task, demonstration_count))
    episodes = load_low_dim_obs_pickles(paths)
    names = [path.parent.name for path in paths]
    spec = (
        store_bottle_semantic_task_spec()
        if task == STORE_BOTTLE_TASK_NAME
        else get_task_spec(task)
    )
    converted = (
        make_store_bottle_semantic_demonstrations(episodes, names=names)
        if task == STORE_BOTTLE_TASK_NAME
        else (
            make_bimanual_demonstrations(episodes, spec, names=names)
            if spec.bimanual
            else make_unimanual_demonstrations(episodes, spec, names=names)
        )
    )
    builder = ClosedLoopTaskModelBuilder(
        ClosedLoopTaskModelConfig(**_json(task_model_config))
    )
    task_dir = base_models / task
    checkpoints: tuple[Path, ...]
    if spec.bimanual:
        base = BimanualDynaMAC(
            left=DynaMAC.load(task_dir / "left.npz"),
            right=DynaMAC.load(task_dir / "right.npz"),
        )
        left, right = builder.build_bimanual(
            base,
            converted.left_demonstrations,
            converted.right_demonstrations,
            recoverable_frames=spec.recoverable_relation_frames,
        )
        models = {"left": left, "right": right}
        checkpoints = (task_dir / "left.npz", task_dir / "right.npz")
    else:
        base_policy = DynaMAC.load(task_dir / "model.npz")
        model = builder.build(
            base_policy,
            converted.demonstrations,
            arm_id="single",
            recoverable_frames=spec.recoverable_relation_frames,
        )
        models = {"single": model}
        checkpoints = (task_dir / "model.npz",)

    boundary_config = boundary_root / f"{task}.json"
    config = ClosedLoopPolicyConfig.from_files(
        belief=belief_config,
        execution=execution_config,
        boundary=boundary_config,
        recovery=recovery_config,
    )
    policy = ClosedLoopMultiStreamPolicy(models, config)
    output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{task}.staging-", dir=output_root))
    try:
        policy.save(staging)
        provenance_paths = (
            task_model_config,
            belief_config,
            execution_config,
            recovery_config,
            boundary_config,
            *checkpoints,
            *paths,
        )
        atomic_json(
            staging / "build.json",
            {
                "schema": BUILD_SCHEMA,
                "task": task,
                "bimanual": spec.bimanual,
                "demonstrations": names,
                "base_model_root": str(base_models.resolve()),
                "closed_loop_output_root": str(output_root.resolve()),
                "files": {
                    str(path.resolve().relative_to(REPOSITORY_ROOT.resolve())): _sha256(
                        path
                    )
                    for path in provenance_paths
                },
                "policy_summary": policy.summary(),
            },
        )
        os.rename(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return output / "policy.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--base-models", type=Path, default=DEFAULT_BASE_MODELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--demonstrations", type=int, default=5)
    parser.add_argument("--task-model-config", type=Path, default=DEFAULT_TASK_CONFIG)
    parser.add_argument("--belief-config", type=Path, default=DEFAULT_BELIEF_CONFIG)
    parser.add_argument(
        "--execution-config", type=Path, default=DEFAULT_EXECUTION_CONFIG
    )
    parser.add_argument("--recovery-config", type=Path, default=DEFAULT_RECOVERY_CONFIG)
    parser.add_argument("--boundary-root", type=Path, default=DEFAULT_BOUNDARY_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.demonstrations < 1:
        raise ValueError("示范数量必须为正整数")
    tasks = args.task or sorted(load_task_specs())
    unknown = set(tasks).difference(load_task_specs())
    if unknown:
        raise ValueError(f"未知任务：{sorted(unknown)}")
    for task in tasks:
        path = build_task(
            task,
            data_root=args.data_root,
            base_models=args.base_models,
            output_root=args.output,
            demonstration_count=args.demonstrations,
            task_model_config=args.task_model_config,
            belief_config=args.belief_config,
            execution_config=args.execution_config,
            recovery_config=args.recovery_config,
            boundary_root=args.boundary_root,
        )
        print(f"{task}: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
