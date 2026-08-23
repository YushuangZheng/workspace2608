#!/usr/bin/env python3
"""Summarize the bounded Transport logpZO protocol without paper-equivalence claims."""

import argparse
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


PROTOCOL_LABEL = "upstream_release_external_dp_checkpoint"


def atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def atomic_json(path, payload):
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def wilson_interval(successes, total, z=1.959963984540054):
    if total <= 0:
        return [None, None]
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return [center - margin, center + margin]


def load_rollouts(path, modify, limit):
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    expected = {
        "schema": "dynamac-fail-detect-logpzo-rollouts-v1",
        "protocol_label": PROTOCOL_LABEL,
        "task": "transport",
        "policy_type": "diffusion",
        "modify": modify,
        "num_inference_steps": 70,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError("{} has incompatible {}".format(path, key))
    records = payload.get("episodes", [])[:limit]
    if len(records) != limit:
        raise RuntimeError("{} has {} records; need {}".format(path, len(records), limit))
    expected_seeds = list(range(payload["start_seed"], payload["start_seed"] + limit))
    if [record.get("seed") for record in records] != expected_seeds:
        raise RuntimeError("{} seeds are not the expected contiguous range".format(path))
    lengths = {len(record.get("logpzo", [])) for record in records}
    if len(lengths) != 1 or next(iter(lengths)) <= 0:
        raise RuntimeError("{} score trajectories do not have one positive length".format(path))
    for record in records:
        if record.get("action_steps") != 8:
            raise RuntimeError("{} contains a non-8 action step rollout".format(path))
        values = record["logpzo"]
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
            raise RuntimeError("{} contains a non-finite score".format(path))
        if not isinstance(record.get("success"), bool):
            raise RuntimeError("{} contains a non-boolean success label".format(path))
    return payload, records, next(iter(lengths))


def policy_stats(records):
    successes = sum(record["success"] for record in records)
    total = len(records)
    return {
        "episodes": total,
        "successes": successes,
        "failures": total - successes,
        "success_rate": successes / total,
        "success_rate_wilson_95": wilson_interval(successes, total),
    }


def build_upper_band(upstream, successful_records, calibration_successes, alpha):
    import numpy as np

    if len(successful_records) < calibration_successes:
        raise RuntimeError(
            "need {} successful ID trajectories for calibration; found {}".format(
                calibration_successes, len(successful_records)
            )
        )
    selected = successful_records[:calibration_successes]
    mean_count = max(1, int(calibration_successes * 0.3))
    band_count = calibration_successes - mean_count
    if band_count < 1:
        raise RuntimeError("calibration split needs at least one band trajectory")
    sys.path.insert(0, str(upstream / "UQ_test"))
    from timeseries_cp.methods.functional_predictor import FunctionalPredictor, ModulationType
    from timeseries_cp.utils.data_utils import RegressionType

    matrix = np.asarray([record["logpzo"] for record in selected], dtype=np.float64)
    predictor = FunctionalPredictor(
        modulation_type=ModulationType.Tfunc,
        regression_type=RegressionType.Mean,
    )
    band = predictor.get_one_sided_prediction_band(
        matrix[:mean_count], matrix[mean_count:], alpha=alpha, lower_bound=False
    ).reshape(-1)
    if band.shape != (matrix.shape[1],) or not np.all(np.isfinite(band)):
        raise RuntimeError("released conformal predictor produced an invalid upper band")
    return band, selected, mean_count, band_count


def detection_stats(records, band):
    outcomes = []
    for record in records:
        crossings = [index for index, value in enumerate(record["logpzo"]) if value >= band[index]]
        predicted_failure = bool(crossings)
        true_failure = not record["success"]
        outcomes.append({
            "seed": record["seed"],
            "domain": record["domain"],
            "true_failure": true_failure,
            "predicted_failure": predicted_failure,
            "first_detection_step": crossings[0] * 8 if crossings else None,
        })

    tp = sum(item["true_failure"] and item["predicted_failure"] for item in outcomes)
    tn = sum(not item["true_failure"] and not item["predicted_failure"] for item in outcomes)
    fp = sum(not item["true_failure"] and item["predicted_failure"] for item in outcomes)
    fn = sum(item["true_failure"] and not item["predicted_failure"] for item in outcomes)
    if tp + fn == 0 or tn + fp == 0:
        raise RuntimeError("detection test set must contain both success and failure classes")
    detection_steps = [
        item["first_detection_step"] for item in outcomes
        if item["true_failure"] and item["predicted_failure"]
    ]
    if detection_steps:
        mean_step = sum(detection_steps) / len(detection_steps)
        if len(detection_steps) == 1:
            standard_error = 0.0
        else:
            mean = mean_step
            sample_variance = sum((value - mean) ** 2 for value in detection_steps) / (len(detection_steps) - 1)
            standard_error = math.sqrt(sample_variance / len(detection_steps))
    else:
        mean_step = None
        standard_error = None
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    return {
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "true_positive_rate": tpr,
        "true_positive_rate_wilson_95": wilson_interval(tp, tp + fn),
        "true_negative_rate": tnr,
        "true_negative_rate_wilson_95": wilson_interval(tn, tn + fp),
        "balanced_accuracy": (tpr + tnr) / 2.0,
        "balanced_accuracy_interval": None,
        "balanced_accuracy_interval_note": "No single Wilson interval is reported for the mean of two class-conditional proportions.",
        "true_positive_detection_step_mean": mean_step,
        "true_positive_detection_step_standard_error": standard_error,
        "outcomes": outcomes,
    }


def render_markdown(summary):
    id_stats = summary["policy"]["id"]
    ood_stats = summary["policy"]["ood"]
    detection = summary["detection"]
    confusion = detection["confusion"]
    tpr_ci = detection["true_positive_rate_wilson_95"]
    tnr_ci = detection["true_negative_rate_wilson_95"]
    detection_step = detection["true_positive_detection_step_mean"]
    detection_step_text = "n/a" if detection_step is None else "{:.1f}".format(detection_step)
    return """# FAIL-Detect bounded Transport result

- Protocol: `{protocol}`
- Claim: official external Diffusion Policy checkpoint; **not** the FAIL-Detect paper policy/checkpoint or full paper protocol.
- Rollouts: {limit} ID + {limit} OOD; paired initial seeds.
- ID policy success: {id_success}/{limit} ({id_rate:.3f})
- OOD policy success: {ood_success}/{limit} ({ood_rate:.3f})
- logpZO calibration: {calibration} successful ID trajectories ({mean_count} mean / {band_count} band), alpha={alpha}
- Detection test confusion: TP={tp}, TN={tn}, FP={fp}, FN={fn}
- TPR: {tpr:.3f} (Wilson 95% CI [{tpr_low:.3f}, {tpr_high:.3f}])
- TNR: {tnr:.3f} (Wilson 95% CI [{tnr_low:.3f}, {tnr_high:.3f}])
- Balanced accuracy: {balanced:.3f}; no single Wilson interval is assigned to this mean of two class-conditional proportions.
- Mean true-positive detection step: {detection_step}
- Gate decision: `{gate_decision}`
""".format(
        protocol=summary["protocol_label"],
        limit=summary["rollouts_per_domain"],
        id_success=id_stats["successes"],
        id_rate=id_stats["success_rate"],
        ood_success=ood_stats["successes"],
        ood_rate=ood_stats["success_rate"],
        calibration=summary["calibration"]["successful_id_trajectories"],
        mean_count=summary["calibration"]["mean_count"],
        band_count=summary["calibration"]["band_count"],
        alpha=summary["calibration"]["alpha"],
        tp=confusion["tp"],
        tn=confusion["tn"],
        fp=confusion["fp"],
        fn=confusion["fn"],
        tpr=detection["true_positive_rate"],
        tpr_low=tpr_ci[0],
        tpr_high=tpr_ci[1],
        tnr=detection["true_negative_rate"],
        tnr_low=tnr_ci[0],
        tnr_high=tnr_ci[1],
        balanced=detection["balanced_accuracy"],
        detection_step=detection_step_text,
        gate_decision=summary["gate"]["decision"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--id-rollouts", type=Path, required=True)
    parser.add_argument("--ood-rollouts", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-provenance", type=Path, required=True)
    parser.add_argument("--artifact-lock", type=Path, required=True)
    parser.add_argument("--input-validation", type=Path, required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--calibration-successes", type=int, required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--minimum-id-success-rate", type=float, default=0.7)
    args = parser.parse_args()
    if args.limit <= 0 or args.calibration_successes < 2:
        parser.error("limit must be positive and calibration-successes must be at least 2")
    if not 0.0 < args.alpha < 1.0:
        parser.error("alpha must be between zero and one")

    repo_root = args.repo_root.resolve()
    upstream = repo_root / "baselines/fail_detect/upstream"
    with args.artifact_lock.open("r", encoding="utf-8") as handle:
        artifact_lock = json.load(handle)
    with args.input_validation.open("r", encoding="utf-8") as handle:
        input_validation = json.load(handle)
    id_payload, id_records, id_length = load_rollouts(args.id_rollouts, False, args.limit)
    ood_payload, ood_records, ood_length = load_rollouts(args.ood_rollouts, True, args.limit)
    if id_length != ood_length:
        raise RuntimeError("ID and OOD score trajectories have different lengths")
    if [item["seed"] for item in id_records] != [item["seed"] for item in ood_records]:
        raise RuntimeError("ID and OOD seeds are not paired")
    for key in ("policy_checkpoint_sha256", "dataset_sha256", "logpzo_checkpoint_sha256"):
        if id_payload.get(key) != ood_payload.get(key):
            raise RuntimeError("ID/OOD provenance mismatch for {}".format(key))
    expected_hashes = {
        "policy_checkpoint_sha256": artifact_lock["artifacts"]["transport_ph_dp_checkpoint"]["sha256"],
        "dataset_sha256": artifact_lock["artifacts"]["transport_image_abs"]["sha256"],
        "logpzo_checkpoint_sha256": input_validation["detector_sha256"],
    }
    for key, expected in expected_hashes.items():
        if id_payload.get(key) != expected:
            raise RuntimeError("rollout provenance does not match runtime validation for {}".format(key))

    successful_id = [record for record in id_records if record["success"]]
    band, calibration, mean_count, band_count = build_upper_band(
        upstream, successful_id, args.calibration_successes, args.alpha
    )
    calibration_seeds = {record["seed"] for record in calibration}
    detection_records = []
    for record in id_records:
        if record["seed"] not in calibration_seeds:
            detection_records.append(dict(record, domain="id"))
    detection_records.extend(dict(record, domain="ood") for record in ood_records)
    detection = detection_stats(detection_records, band)
    id_stats = policy_stats(id_records)
    ood_stats = policy_stats(ood_records)

    reasons = []
    if args.gate and id_stats["success_rate"] < args.minimum_id_success_rate:
        reasons.append(
            "ID policy success {:.3f} is below {:.3f}".format(
                id_stats["success_rate"], args.minimum_id_success_rate
            )
        )
    if not args.gate:
        decision = "not_applied"
    else:
        decision = "pass" if not reasons else "stop"
    summary = {
        "schema": "dynamac-fail-detect-quant-summary-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_label": PROTOCOL_LABEL,
        "claim_boundary": "Bounded logpZO evaluation using an official external Diffusion Policy checkpoint; not a paper-result reproduction.",
        "rollouts_per_domain": args.limit,
        "paired_start_seed": id_payload["start_seed"],
        "score_points_per_rollout": id_length,
        "policy": {"id": id_stats, "ood": ood_stats},
        "calibration": {
            "source": "first successful ID trajectories",
            "successful_id_trajectories": args.calibration_successes,
            "mean_count": mean_count,
            "band_count": band_count,
            "alpha": args.alpha,
            "released_predictor": "FunctionalPredictor(Tfunc, Mean), upper band",
            "seeds": sorted(calibration_seeds),
        },
        "detection_test_episodes": len(detection_records),
        "detection": detection,
        "gate": {
            "enabled": args.gate,
            "minimum_id_success_rate": args.minimum_id_success_rate if args.gate else None,
            "decision": decision,
            "reasons": reasons,
            "performance_threshold_applied_to_detector": False,
        },
        "provenance": {
            "upstream_commit": id_payload["upstream_commit"],
            "policy_checkpoint_sha256": id_payload["policy_checkpoint_sha256"],
            "dataset_sha256": id_payload["dataset_sha256"],
            "logpzo_checkpoint_sha256": id_payload["logpzo_checkpoint_sha256"],
        },
    }
    atomic_json(args.output_json, summary)
    atomic_text(args.output_md, render_markdown(summary))
    archive = artifact_lock["artifacts"]["robomimic_image"]
    checkpoint = artifact_lock["artifacts"]["transport_ph_dp_checkpoint"]
    dataset = artifact_lock["artifacts"]["transport_image_abs"]
    provenance = {
        "schema": "dynamac-fail-detect-quant-provenance-v1",
        "generated_at": summary["created_at"],
        "protocol_label": PROTOCOL_LABEL,
        "claim_boundary": summary["claim_boundary"],
        "upstream": {
            "url": "https://github.com/CXU-TRI/FAIL-Detect.git",
            "commit": id_payload["upstream_commit"],
        },
        "official_artifacts": {
            "robomimic_image_archive": {
                "url": archive["url"],
                "bytes": archive["bytes"],
                "sha256": archive["sha256"],
                "remote": archive["remote"],
            },
            "transport_image_abs": {
                "archive_member": dataset["archive_member"],
                "bytes": dataset["bytes"],
                "sha256": dataset["sha256"],
            },
            "transport_ph_dp_checkpoint": {
                "url": checkpoint["url"],
                "bytes": checkpoint["bytes"],
                "sha256": checkpoint["sha256"],
                "remote": checkpoint["remote"],
            },
        },
        "generated_artifacts": {
            "features": {
                "sha256": input_validation["feature_sha256"],
                **input_validation["features"],
            },
            "logpzo_detector": {
                "sha256": input_validation["detector_sha256"],
                **input_validation["detector"],
            },
        },
        "evaluation": {
            "rollouts_per_domain": args.limit,
            "paired_start_seed": id_payload["start_seed"],
            "num_inference_steps": id_payload["num_inference_steps"],
            "calibration_successes": args.calibration_successes,
            "calibration_mean_count": mean_count,
            "calibration_band_count": band_count,
            "alpha": args.alpha,
            "gate_decision": decision,
        },
    }
    atomic_json(args.output_provenance, provenance)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.gate and decision != "pass":
        raise SystemExit(4)


if __name__ == "__main__":
    main()
