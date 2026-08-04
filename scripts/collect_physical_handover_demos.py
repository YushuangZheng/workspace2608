"""Collect an exact, non-selective physical-handover seed batch via frozen evaluator workers."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from essay2608.data.physical_handover import audit_physical_handover_dataset
from essay2608.eval.physical_handover_audit import TASK_ID, V3_SOURCE_SHA256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--max_steps", type=int, default=1400)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--expected_source_sha256", default=V3_SOURCE_SHA256)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _convert_trace(
    trace_path: Path,
    output_path: Path,
    *,
    seed: int,
    trial: dict,
) -> None:
    with np.load(trace_path, allow_pickle=False) as source:
        arrays = {key: source[key].copy() for key in source.files}
    steps = len(arrays["state"])
    target_orientation = np.zeros((steps, 4), dtype=np.float32)
    target_orientation[:, 0] = 1.0
    arrays.update(
        {
            "time": np.arange(steps, dtype=np.float32)
            * float(np.asarray(arrays["control_dt"]).item()),
            "left_ee_pose": np.concatenate(
                (arrays["left_ee_position"], arrays["left_ee_orientation"]), axis=-1
            ).astype(np.float32),
            "right_ee_pose": np.concatenate(
                (arrays["right_ee_position"], arrays["right_ee_orientation"]), axis=-1
            ).astype(np.float32),
            "object_pose": np.concatenate(
                (arrays["object_position"], arrays["object_orientation"]), axis=-1
            ).astype(np.float32),
            "target_pose": np.concatenate(
                (arrays["target_position"], target_orientation), axis=-1
            ).astype(np.float32),
            "seed": np.asarray(seed, dtype=np.int64),
            "source_sha256": np.asarray(trial["source_sha256"]),
            "experiment_fingerprint": np.asarray(trial["experiment_fingerprint"]),
            "both_duration_s": np.asarray(trial["both_duration_s"], dtype=np.float32),
            "maximum_object_height_m": np.asarray(
                trial["maximum_object_height_m"], dtype=np.float32
            ),
            "final_xy_error_m": np.asarray(trial["final_xy_error_m"], dtype=np.float32),
            "object_on_support": np.asarray(trial["object_on_support"], dtype=bool),
            "stable": np.asarray(trial["stable"], dtype=bool),
            "settling_displacement_m": np.asarray(
                trial["settling_displacement_m"], dtype=np.float32
            ),
            "quaternion_order": np.asarray("wxyz"),
            "coordinate_frame": np.asarray("local_environment"),
        }
    )
    np.savez_compressed(output_path, **arrays)


def main() -> None:
    args = parse_args()
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("采集 seed 不得重复")
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise RuntimeError(f"拒绝覆盖已有数据目录：{output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    repository = Path(__file__).resolve().parents[1]
    evaluator = repository / "scripts/eval_physical_handover.py"

    with tempfile.TemporaryDirectory(
        prefix=".physical_handover_",
        dir=output_dir.parent,
    ) as temporary:
        temporary = Path(temporary)
        evaluation_dir = temporary / "evaluation"
        command = [
            sys.executable,
            str(evaluator),
            "--headless",
            "--device",
            args.device,
            "--seeds",
            *(str(seed) for seed in args.seeds),
            "--max_steps",
            str(args.max_steps),
            "--output_dir",
            str(evaluation_dir),
        ]
        result = subprocess.run(command, check=False)
        summary_path = evaluation_dir / "summary.json"
        if not summary_path.is_file():
            raise RuntimeError("物理评测 worker 未生成 summary，数据采集整体拒绝")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        failed = [trial for trial in summary["trials"] if not trial.get("success")]
        if result.returncode or failed:
            failures = [(trial.get("seed"), trial.get("failure_reason")) for trial in failed]
            raise RuntimeError(f"预注册 seed 批次存在失败，整体不生成数据集：{failures!r}")
        if summary.get("source_sha256") != args.expected_source_sha256:
            raise RuntimeError("评测源码指纹与物理数据协议不一致")
        if tuple(summary.get("seeds", [])) != tuple(args.seeds):
            raise RuntimeError("评测 summary 的 seed 顺序与采集请求不一致")

        staging = temporary / "dataset"
        staging.mkdir()
        trial_dir = staging / "trials"
        trial_dir.mkdir()
        entries = []
        for index, (seed, trial) in enumerate(zip(args.seeds, summary["trials"], strict=True)):
            stem = f"scripted_physical_handover__seed_{seed}"
            output_path = staging / f"demo_{index:03d}.npz"
            _convert_trace(
                evaluation_dir / "trials" / f"{stem}.npz",
                output_path,
                seed=seed,
                trial=trial,
            )
            result_path = trial_dir / f"demo_{index:03d}.json"
            result_path.write_text(
                json.dumps(trial, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            entries.append(
                {
                    "file": output_path.name,
                    "sha256": _sha256(output_path),
                    "seed": seed,
                    "trial_result": str(result_path.relative_to(staging)),
                    "experiment_fingerprint": trial["experiment_fingerprint"],
                }
            )
        manifest = {
            "task_id": TASK_ID,
            "dataset_schema_version": 1,
            "num_demos": len(entries),
            "requested_seeds": list(args.seeds),
            "max_steps": args.max_steps,
            "source_sha256": args.expected_source_sha256,
            "evaluator_sha256": _sha256(evaluator),
            "collection_mode": "exact_seed_batch_no_replacement",
            "relation_source": "physical_contact_proximity_relative_motion_not_phase",
            "quaternion_order": "wxyz",
            "coordinate_frame": "local_environment",
            "demos": entries,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        audit = audit_physical_handover_dataset(
            staging,
            expected_seeds=args.seeds,
            expected_source_sha256=args.expected_source_sha256,
        )
        manifest["pre_freeze_acceptance"] = {
            key: value for key, value in audit.items() if key != "entries"
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        staging.replace(output_dir)
    print(f"物理交接数据完整批次采集通过：{output_dir}")


if __name__ == "__main__":
    main()
