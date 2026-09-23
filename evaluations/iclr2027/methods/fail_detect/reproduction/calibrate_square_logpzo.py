"""Reproduce FAIL-Detect's functional CP band and alarm decisions."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

import numpy as np

from evaluations.iclr2027.methods.fail_detect.conformal import (
    TimeVaryingConformalBand,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nominal", type=pathlib.Path, nargs="+", required=True)
    parser.add_argument("--ood", type=pathlib.Path, nargs="+", required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.025)
    parser.add_argument("--num-train", type=int, default=300)
    parser.add_argument("--num-cal", type=int, default=700)
    parser.add_argument("--num-test", type=int, default=1000)
    return parser.parse_args()


def _load(paths: list[pathlib.Path], expected_condition: str) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "pass" or payload.get("condition") != expected_condition:
            raise ValueError(f"unexpected rollout artifact {path}")
        episodes.extend(payload["episodes"])
    episodes.sort(key=lambda item: int(item["seed"]))
    seeds = [int(item["seed"]) for item in episodes]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"duplicate {expected_condition} rollout seeds")
    horizons = {len(item["logpzo"]) for item in episodes}
    if len(horizons) != 1:
        raise ValueError(f"inconsistent {expected_condition} score horizons: {horizons}")
    if (
        not episodes
        or not np.isfinite(
            np.asarray([item["logpzo"] for item in episodes], dtype=np.float64)
        ).all()
    ):
        raise ValueError(f"invalid {expected_condition} scores")
    return episodes


def _fit_upstream_split(
    episodes: list[dict[str, Any]],
    *,
    alpha: float,
    num_train: int,
    num_cal: int,
) -> tuple[TimeVaryingConformalBand, dict[str, Any]]:
    split_size = num_train + num_cal
    candidates = episodes[:split_size]
    successful = [item for item in candidates if int(item["success"]) == 1]
    mean_count = int(len(successful) * num_train / split_size)
    width_count = len(successful) - mean_count
    if mean_count <= 0 or width_count <= 0:
        raise ValueError("not enough successful trajectories for both CP splits")
    mean_scores = np.asarray([item["logpzo"] for item in successful[:mean_count]], dtype=np.float64)
    width_scores = np.asarray(
        [item["logpzo"] for item in successful[-width_count:]], dtype=np.float64
    )
    band = TimeVaryingConformalBand.fit(
        mean_scores,
        width_scores,
        alpha=alpha,
        modulation_kind="tfunc",
    )
    return band, {
        "candidate_episodes": split_size,
        "successful_candidates": len(successful),
        "mean_episodes": mean_count,
        "width_episodes": width_count,
        "candidate_seed_start": int(candidates[0]["seed"]),
        "candidate_seed_end": int(candidates[-1]["seed"]),
    }


def _decisions(
    episodes: list[dict[str, Any]],
    band: TimeVaryingConformalBand,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    decisions = []
    true_positive = false_positive = true_negative = false_negative = 0
    failure_delays = []
    for item in episodes:
        scores = np.asarray(item["logpzo"], dtype=np.float64)
        exceeds = scores >= band.upper
        alarm = bool(np.any(exceeds))
        first_index = int(np.argmax(exceeds)) if alarm else None
        failure = int(item["success"]) == 0
        if failure and alarm:
            true_positive += 1
            assert first_index is not None
            failure_delays.append(first_index * 8)
        elif failure:
            false_negative += 1
        elif alarm:
            false_positive += 1
        else:
            true_negative += 1
        decisions.append(
            {
                "seed": int(item["seed"]),
                "success": int(item["success"]),
                "failure": int(failure),
                "alarm": int(alarm),
                "first_alarm_score_index": first_index,
                "first_alarm_env_step": None if first_index is None else first_index * 8,
            }
        )

    def _ratio(numerator: int, denominator: int) -> float | None:
        return None if denominator == 0 else numerator / denominator

    recall = _ratio(true_positive, true_positive + false_negative)
    specificity = _ratio(true_negative, true_negative + false_positive)
    precision = _ratio(true_positive, true_positive + false_positive)
    if recall is None or precision is None or recall + precision == 0:
        f1 = None
    else:
        f1 = 2 * recall * precision / (recall + precision)
    balanced_accuracy = (
        None if recall is None or specificity is None else (recall + specificity) / 2
    )
    metrics = {
        "episodes": len(episodes),
        "failures": true_positive + false_negative,
        "successes": true_negative + false_positive,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "mean_detected_failure_delay_env_steps": (
            None if not failure_delays else float(np.mean(failure_delays))
        ),
    }
    return metrics, decisions


def main() -> None:
    args = _parse_args()
    required = args.num_train + args.num_cal + args.num_test
    nominal = _load(args.nominal, "id_nominal")
    ood = _load(args.ood, "ood_modify")
    if len(nominal) != required or len(ood) != required:
        raise ValueError(
            f"expected exactly {required} rollouts per condition, "
            f"got nominal={len(nominal)} ood={len(ood)}"
        )

    nominal_band, nominal_split = _fit_upstream_split(
        nominal,
        alpha=args.alpha,
        num_train=args.num_train,
        num_cal=args.num_cal,
    )
    ood_band, ood_split = _fit_upstream_split(
        ood,
        alpha=args.alpha,
        num_train=args.num_train,
        num_cal=args.num_cal,
    )
    test_start = args.num_train + args.num_cal
    nominal_metrics, nominal_decisions = _decisions(nominal[test_start:], nominal_band)
    upstream_ood_metrics, upstream_ood_decisions = _decisions(ood[test_start:], ood_band)
    frozen_ood_metrics, frozen_ood_decisions = _decisions(ood[test_start:], nominal_band)

    result = {
        "status": "pass",
        "scope": "official_square_flow_logpzo_functional_cp_and_alarm",
        "official_commit": "b758e55f7c0c988188f2e4876ffc03ae8a3c30ed",
        "alpha": args.alpha,
        "official_split": {
            "num_train": args.num_train,
            "num_cal": args.num_cal,
            "num_test": args.num_test,
            "nominal": nominal_split,
            "ood": ood_split,
        },
        "nominal_band": nominal_band.to_dict(),
        "upstream_conditionwise_ood_band": ood_band.to_dict(),
        "evaluations": {
            "nominal_upstream_conditionwise": {
                "metrics": nominal_metrics,
                "decisions": nominal_decisions,
            },
            "ood_upstream_conditionwise": {
                "metrics": upstream_ood_metrics,
                "decisions": upstream_ood_decisions,
            },
            "ood_nominal_frozen_band": {
                "metrics": frozen_ood_metrics,
                "decisions": frozen_ood_decisions,
            },
        },
        "notes": [
            "The condition-wise rows reproduce the public plotting script's "
            "300/700/1000 split behavior after filtering calibration candidates "
            "to successful trajectories.",
            "The nominal-frozen OOD row is also reported because it avoids fitting "
            "a detector on OOD trajectories; neither artifact is a paper threshold.",
            "Server A must regenerate formal calibration artifacts from its private "
            "Main-10 normal-calibration split.",
        ],
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(
        json.dumps(
            {
                "nominal": nominal_metrics,
                "ood_upstream": upstream_ood_metrics,
                "ood_nominal_frozen": frozen_ood_metrics,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
