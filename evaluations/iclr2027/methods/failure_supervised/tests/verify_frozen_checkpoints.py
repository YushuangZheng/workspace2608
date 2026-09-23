"""Read-only numerical/selection audit of all frozen M4 checkpoints.

No fitting, thresholding, test evaluation, or checkpoint rewriting. Uses only
the authorized train split and development fixture. Writes a separate report.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from evaluations.iclr2027.interfaces.failure_train import (
    load_failure_train_manifest,
    load_failure_train_sequence,
    select_failure_train_rows,
)
from evaluations.iclr2027.interfaces.feature_schema import FEATURE_SCHEMA
from evaluations.iclr2027.interfaces.runtime_monitor import EpisodeContext
from evaluations.iclr2027.methods.failure_supervised.queue import jobs
from evaluations.iclr2027.methods.failure_supervised.runtime import (
    CanonicalFailureSupervisedMonitor,
    sha256,
)
from evaluations.iclr2027.methods.failure_supervised.train_cli import (
    CHECKPOINTS,
    CONFIG,
    DATASET,
    MANIFEST,
    ROOT,
    TRAINING,
    cache_dir,
    config,
    run_id,
    write_json,
)


def main() -> None:
    torch.set_num_threads(1)
    rows = load_failure_train_manifest(MANIFEST)
    cfg = config()
    all_jobs = sum(jobs(), [])
    caches, views = {}, {}
    reports = []
    for job in all_jobs:
        task, seed = job["task"], job["seed"]
        family, budget = job.get("held_out_family"), job.get("budget")
        identity = run_id(task, seed, budget, family)
        checkpoint = CHECKPOINTS / identity / "model.pt"
        checkpoint_hash = sha256(checkpoint)
        done = json.loads((TRAINING / "runs" / identity / "complete.json").read_text())
        if checkpoint_hash != done["checkpoint_sha256"] or sha256(CONFIG) != done["config_sha256"]:
            raise ValueError("checkpoint/config checksum mismatch")
        selected = select_failure_train_rows(rows, task=task, budget=budget, held_out_family=family)
        if task not in caches:
            with np.load(cache_dir(task) / "raw.npz", allow_pickle=False) as data:
                caches[task] = {key: data[key] for key in data.files}
        cache = caches[task]
        key = (task, budget, family)
        if key not in views:
            index = {eid: i for i, eid in enumerate(cache["episode_ids"].tolist())}
            selected_x = np.concatenate(
                [
                    cache["x"][
                        cache["offsets"][index[r["episode_id"]]] : cache["offsets"][
                            index[r["episode_id"]] + 1
                        ]
                    ]
                    for r in selected
                ]
            )
            mean = selected_x.mean(0, dtype=np.float64).astype(np.float32)
            std = np.maximum(
                selected_x.std(0, dtype=np.float64), cfg["normalizer"]["std_floor"]
            ).astype(np.float32)
            sequences = [
                load_failure_train_sequence(DATASET, row) for row in (selected[0], selected[-1])
            ]
            views[key] = (mean, std, sequences)
        mean, std, sequences = views[key]
        monitor = CanonicalFailureSupervisedMonitor(checkpoint, CONFIG)
        if not np.array_equal(mean, monitor.mean) or not np.array_equal(std, monitor.std):
            raise ValueError("normalizer is not fitted to the exact budget/LOFO subset")
        training_identity = json.loads(
            (TRAINING / "runs" / identity / "training_identity.json").read_text()
        )
        if training_identity["selected_episode_ids"] != [r["episode_id"] for r in selected]:
            raise ValueError("selected episodes disagree with canonical reader")
        sequence_reports = []
        for seq in sequences:
            context = EpisodeContext(
                seq.episode_id,
                task,
                "M4",
                len(monitor.layout.arms) == 2,
                1000,
                FEATURE_SCHEMA,
                monitor.config_hash,
                monitor.checkpoint_hash,
            )
            encoded = np.stack([monitor.layout.encode_validated(record) for record in seq.features])
            normalized = np.clip((encoded - mean) / std, -monitor.clip, monitor.clip)
            with torch.inference_mode():
                logits, _ = monitor.scorer.model(torch.from_numpy(normalized)[None])
                full = logits.sigmoid()[0].numpy()
            monitor.reset(context)
            streamed = []
            for record in seq.features:
                monitor.observe_record(record)
                streamed.append(monitor.score()[cfg["score_name"]])
                assert monitor.output.threshold is None and monitor.output.persistence_count == 0
                assert not monitor.alarm()
            error = float(np.max(np.abs(full - np.asarray(streamed))))
            if not np.allclose(full, streamed, atol=2e-5, rtol=2e-5):
                raise ValueError(
                    f"full versus streaming probabilities disagree: {identity}: {error}"
                )
            monitor.reset(context)
            monitor.observe_record(seq.features[0])
            if monitor.score()[cfg["score_name"]] != streamed[0]:
                raise ValueError("episode reset did not reproduce its first prediction")
            sequence_reports.append(
                {
                    "episode_id": seq.episode_id,
                    "cycles": len(seq.features),
                    "max_probability_error": error,
                }
            )
        if sha256(checkpoint) != checkpoint_hash:
            raise ValueError("audit changed checkpoint bytes")
        reports.append(
            {
                **job,
                "checkpoint_sha256": checkpoint_hash,
                "status": "pass",
                "exact_train_subset_and_normalizer": True,
                "sequences": sequence_reports,
            }
        )
        print(
            json.dumps(
                {
                    "completed": len(reports),
                    "total": len(all_jobs),
                    "run": identity,
                    "max_error": max(r["max_probability_error"] for r in sequence_reports),
                }
            ),
            flush=True,
        )
    report = {
        "status": "pass",
        "scope": "training_input_and_full_streaming_inference_consistency",
        "formal_evaluation": False,
        "weights_updated": False,
        "thresholds_calibrated": False,
        "checkpoints": len(reports),
        "full_episodes_replayed": 2 * len(reports),
        "config_sha256": sha256(CONFIG),
        "audit_code_sha256": sha256(Path(__file__)),
        "torch": str(torch.__version__),
        "device": "cpu",
        "checks": reports,
    }
    path = TRAINING / "INFERENCE_VALIDATION.json"
    write_json(path, report)
    print(json.dumps({"status": "pass", "report": str(path.relative_to(ROOT))}), flush=True)


if __name__ == "__main__":
    main()
