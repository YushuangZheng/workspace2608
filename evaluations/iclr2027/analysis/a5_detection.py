"""Generate A5 E2 shadow-detection and recovery-conversion artifacts."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping

import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, precision_recall_curve

from evaluations.iclr2027.analysis.rebuild_a5_audit import load_current_reaudit


ROOT = Path(__file__).resolve().parents[3]
RESULT_ROOT = ROOT / "evaluations/iclr2027/results/controlled/e1_e2"
DERIVED = RESULT_ROOT / "derived"
ATTRIBUTION_MANIFEST = (
    ROOT / "evaluations/iclr2027/manifests/ablation4.jsonl"
)
SHADOWS = {
    "m2": ("Trajectory-Likelihood", "standardized_nll"),
    "m3": ("FAIL-Detect", "logpzo"),
    "m4_seed1103": ("Failure-Supervised [1103]", "violation_probability"),
    "m4_seed2207": ("Failure-Supervised [2207]", "violation_probability"),
    "m4_seed3301": ("Failure-Supervised [3301]", "violation_probability"),
    "m6": ("Ours-Monitor", "task_state_mismatch"),
}
RECOVERY_METHODS = {
    "m2": "Trajectory-Likelihood + Retry",
    "m3": "FAIL-Detect + Retry",
    "m4_seed1103": "Failure-Supervised + Retry [1103]",
    "m4_seed2207": "Failure-Supervised + Retry [2207]",
    "m4_seed3301": "Failure-Supervised + Retry [3301]",
    "m6": "Ours-Monitor + Retry",
    "retry_same_state": "Ours-Monitor + Relation Repair + Same-State Resume",
    "m5": "Full method",
}
FIGURE3_RECOVERY_METHODS = {"m6", "retry_same_state", "m5"}
RELATION_FAULTS = {"missed_interaction", "relation_loss"}
TEMPORAL_FAULTS = {"actuation_delay", "coordination_delay"}


_CURRENT_REAUDIT: dict[tuple[str, str], dict[str, Any]] | None = None


def _current_audit(key: str, episode: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the sole current A5 event audit.

    Nominal rows contain no scheduled event and therefore retain their empty
    embedded audit.  Perturbed rows must resolve through the versioned
    re-audit artifact; falling back to a legacy embedded audit is forbidden.
    """

    if episode.get("condition") != "perturbed":
        return episode.get("audit", {})
    if key == "retry_same_state":
        audit = episode.get("audit", {})
        if audit.get("schema") != "essay2608.iclr2027.physical-event-audit.v1":
            # A queue-level ITT record is emitted when both infrastructure
            # attempts end before a simulator episode exists.  Such a record
            # has no cycle evidence and therefore cannot contain a versioned
            # physical-event audit; it is necessarily untriggered.  Accept
            # only that exact case rather than falling back to an unversioned
            # audit for an executed episode.
            empty_fields = (
                "physically_triggered",
                "violation_onset_cycle",
                "violation_end_cycle",
                "relation_restored_cycle",
            )
            if (
                episode.get("termination_reason") == "infrastructure_error"
                and episode.get("fault_protocol") is None
                and not bool(audit.get("physically_triggered"))
                and all(audit.get(field) is None for field in empty_fields[1:])
            ):
                return {
                    "schema": "essay2608.iclr2027.physical-event-audit.v1",
                    "eligible": False,
                    "physically_triggered": False,
                    "violation_onset_cycle": None,
                    "violation_end_cycle": None,
                    "relation_restored_cycle": None,
                    "legal_reentry_cycle": None,
                    "target_objects": [],
                    "source": "queue_level_infrastructure_failure",
                }
            raise RuntimeError(
                "retry_same_state does not carry the current physical-event audit"
            )
        return audit
    global _CURRENT_REAUDIT
    if _CURRENT_REAUDIT is None:
        _CURRENT_REAUDIT = load_current_reaudit()
    identity = (key, str(episode["episode_id"]))
    if identity not in _CURRENT_REAUDIT:
        raise RuntimeError(f"missing current A5 re-audit row: {identity}")
    return _CURRENT_REAUDIT[identity]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_jsonl_gzip(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _m0_episode(condition: str, episode_id: str) -> dict[str, Any]:
    safe = episode_id.replace("/", "__")
    path = RESULT_ROOT / "end_to_end" / "m0" / condition / "episodes" / f"{safe}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _rising_edges(cycles: Iterable[int]) -> int:
    values = sorted(set(int(value) for value in cycles))
    return sum(index == 0 or value != values[index - 1] + 1 for index, value in enumerate(values))


def _episode_shadow_rows(key: str, condition: str) -> list[dict[str, Any]]:
    index_path = RESULT_ROOT / "shadow" / key / condition / "score_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("episodes") != 2000 or not index.get("action_passthrough_verified"):
        raise RuntimeError(f"incomplete or action-changing shadow artifact: {key}/{condition}")
    score_name = SHADOWS[key][1]
    rows = []
    for entry in index["files"]:
        episode_id = str(entry["episode_id"])
        scores = _load_jsonl_gzip(ROOT / entry["score_path"])
        alarm_cycles = [int(row["cycle"]) for row in scores if row["alarm"]]
        values = [float(row["scores"][score_name]) for row in scores]
        episode = _m0_episode(condition, episode_id)
        audit = _current_audit("m0", episode)
        onset = audit.get("violation_onset_cycle")
        triggered = bool(audit.get("physically_triggered", False))
        valid_alarm_cycles = (
            []
            if onset is None
            else [cycle for cycle in alarm_cycles if cycle >= int(onset)]
        )
        rows.append(
            {
                "episode_id": episode_id,
                "task": episode["task"],
                "condition": condition,
                "fault_family": episode.get("fault_family"),
                "cycles": len(scores),
                "scorable": bool(scores),
                "triggered": triggered,
                "onset": onset,
                "episode_score": max(values) if values else -math.inf,
                "predicted": bool(alarm_cycles),
                "detected": bool(triggered and valid_alarm_cycles),
                "first_valid_alarm": valid_alarm_cycles[0] if valid_alarm_cycles else None,
                "alarm_rising_edges": _rising_edges(alarm_cycles),
            }
        )
    return rows


def generate_detection() -> dict[str, Any]:
    complete_rows = []
    pr_rows = []
    delay_rows = []
    for key, (method_name, _score_name) in SHADOWS.items():
        nominal_all = _episode_shadow_rows(key, "nominal")
        perturbed_all = _episode_shadow_rows(key, "perturbed")
        nominal = [row for row in nominal_all if row["scorable"]]
        perturbed = [row for row in perturbed_all if row["scorable"]]
        combined = nominal + perturbed
        labels = np.asarray([row["triggered"] for row in combined], dtype=np.int8)
        predictions = np.asarray(
            [row["detected"] if row["triggered"] else row["predicted"] for row in combined],
            dtype=np.int8,
        )
        scores = np.asarray([row["episode_score"] for row in combined], dtype=np.float64)
        triggered = [row for row in perturbed if row["triggered"]]
        detected = [row for row in triggered if row["detected"]]
        false_alarm_episodes = [row for row in combined if row["predicted"] and not row["detected"]]
        true_positive = len(detected)
        false_positive = len(false_alarm_episodes)
        false_negative = len(triggered) - true_positive
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / len(triggered) if triggered else math.nan
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        delays = [int(row["first_valid_alarm"]) - int(row["onset"]) for row in detected]
        nominal_cycles = sum(int(row["cycles"]) for row in nominal)
        nominal_interventions = sum(int(row["alarm_rising_edges"]) for row in nominal)
        false_per_1000 = 1000.0 * nominal_interventions / nominal_cycles
        complete_rows.append(
            {
                "method_key": key,
                "monitor": method_name,
                "scheduled_episodes": len(nominal_all) + len(perturbed_all),
                "episodes": len(combined),
                "unscored_infrastructure_episodes": (
                    len(nominal_all) + len(perturbed_all) - len(combined)
                ),
                "physically_triggered_events": len(triggered),
                "true_positive_events": true_positive,
                "false_positive_episode_events": false_positive,
                "false_negative_events": false_negative,
                "event_precision": precision,
                "event_recall": recall,
                "event_f1": f1,
                "episode_balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
                "episode_auprc": float(average_precision_score(labels, scores)),
                "median_detection_delay_cycles": median(delays) if delays else math.nan,
                "delay_q1_cycles": float(np.quantile(delays, 0.25)) if delays else math.nan,
                "delay_q3_cycles": float(np.quantile(delays, 0.75)) if delays else math.nan,
                "nominal_false_interventions": nominal_interventions,
                "nominal_cycles": nominal_cycles,
                "false_interventions_per_1000_cycles": false_per_1000,
            }
        )
        delay_rows.append(
            {
                "method_key": key,
                "monitor": method_name,
                "event_f1": f1,
                "median_detection_delay_cycles": median(delays) if delays else math.nan,
                "false_interventions_per_1000_nominal_cycles": false_per_1000,
            }
        )
        precision_curve, recall_curve, thresholds = precision_recall_curve(labels, scores)
        for index, threshold in enumerate(thresholds):
            pr_rows.append(
                {
                    "method_key": key,
                    "monitor": method_name,
                    "threshold": float(threshold),
                    "precision": float(precision_curve[index]),
                    "recall": float(recall_curve[index]),
                }
            )
    _write_csv(DERIVED / "appendix_detection_complete.csv", complete_rows)
    _write_csv(DERIVED / "fig3_delay_false_alarm.csv", delay_rows)
    _write_csv(DERIVED / "fig3_monitor_pr.csv", pr_rows)
    by_fault_rows = []
    for key, (method_name, _score_name) in SHADOWS.items():
        scheduled = _episode_shadow_rows(key, "perturbed")
        families = sorted({str(row["fault_family"]) for row in scheduled if row["fault_family"]})
        for family in families:
            family_rows = [row for row in scheduled if row["fault_family"] == family and row["scorable"]]
            triggered = [row for row in family_rows if row["triggered"]]
            detected = [row for row in triggered if row["detected"]]
            false_alarm = [row for row in family_rows if row["predicted"] and not row["detected"]]
            true_positive = len(detected)
            false_positive = len(false_alarm)
            recall = true_positive / len(triggered) if triggered else math.nan
            precision = (
                true_positive / (true_positive + false_positive)
                if true_positive + false_positive
                else math.nan
            )
            f1 = (
                2.0 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            delays = [int(row["first_valid_alarm"]) - int(row["onset"]) for row in detected]
            by_fault_rows.append(
                {
                    "method_key": key,
                    "monitor": method_name,
                    "fault_family": family,
                    "scheduled_episodes": len(family_rows),
                    "physically_triggered_events": len(triggered),
                    "detected_events": true_positive,
                    "false_alarm_episode_events": false_positive,
                    "event_precision_within_scheduled_family": precision,
                    "event_recall": recall,
                    "event_f1_within_scheduled_family": f1,
                    "median_detection_delay_cycles": median(delays) if delays else math.nan,
                }
            )
    _write_csv(DERIVED / "fig3_detection_by_fault.csv", by_fault_rows)
    _write_csv(DERIVED / "appendix_detection_by_fault.csv", by_fault_rows)
    return {
        "monitors": len(SHADOWS),
        "rows": len(complete_rows),
        "by_fault_rows": len(by_fault_rows),
    }


def _endpoint_episode_path(key: str, condition: str, episode_id: str) -> Path:
    safe = episode_id.replace("/", "__")
    if key == "retry_same_state":
        return (
            ROOT
            / "evaluations/iclr2027/results/controlled/e5/fine/retry_same_state"
            / condition
            / "episodes"
            / f"{safe}.json"
        )
    return RESULT_ROOT / "end_to_end" / key / condition / "episodes" / f"{safe}.json"


def _endpoint_cycles(key: str, episode: Mapping[str, Any]) -> list[dict[str, Any]]:
    condition = str(episode["condition"])
    episode_path = _endpoint_episode_path(key, condition, str(episode["episode_id"]))
    cycle_path = (episode_path.parent / str(episode["cycle_file"])).resolve()
    return _load_jsonl_gzip(cycle_path)


def _method_events(
    key: str,
    episode: Mapping[str, Any],
    *,
    audit: Mapping[str, Any] | None = None,
    full_method: bool | None = None,
) -> dict[str, Any]:
    cycles = _endpoint_cycles(key, episode)
    audit = _current_audit(key, episode) if audit is None else audit
    policy_owned_recovery = key in {"m5", "retry_same_state"}
    full_method = policy_owned_recovery if full_method is None else full_method
    onset = audit.get("violation_onset_cycle")
    alarm_cycles = []
    retry_cycles = []
    policy_resume_cycles = []
    for row in cycles:
        cycle = int(row["cycle"])
        if full_method:
            alarm = bool(row.get("feature", {}).get("policy_state", {}).get("monitor", {}).get("alarm", False))
        else:
            alarm = bool(row.get("execution", {}).get("monitor", {}) and row["execution"]["monitor"].get("alarm", False))
        if alarm:
            alarm_cycles.append(cycle)
        retry = row.get("execution", {}).get("retry")
        if isinstance(retry, dict) and retry.get("requested"):
            retry_cycles.append(cycle)
        policy_arms = (
            row.get("execution", {}).get("policy_audit", {}).get("arms", {})
        )
        if any(
            isinstance(value, Mapping) and value.get("reentry_committed")
            for value in policy_arms.values()
        ):
            policy_resume_cycles.append(cycle)
    valid_alarms = [] if onset is None else [cycle for cycle in alarm_cycles if cycle >= int(onset)]
    valid_retry_cycles = (
        [] if onset is None else [cycle for cycle in retry_cycles if cycle >= int(onset)]
    )
    valid_policy_resume_cycles = (
        policy_resume_cycles
        if onset is None
        else [cycle for cycle in policy_resume_cycles if cycle >= int(onset)]
    )
    if key == "m5":
        resume_cycle = episode.get("legal_reentry_cycle")
    elif key == "retry_same_state":
        # The shared episode schema retains the historical field name
        # ``legal_reentry_cycle`` for every policy-owned recovery commit.
        # This ablation performs no legal-state search, so derive its resume
        # event from the causal cycle log and never treat that summary field
        # as evidence of legal re-entry.
        resume_cycle = (
            valid_policy_resume_cycles[0] if valid_policy_resume_cycles else None
        )
    else:
        resume_cycle = valid_retry_cycles[0] if valid_retry_cycles else None
    family = episode.get("fault_family")
    if family in RELATION_FAULTS:
        restored = audit.get("relation_restored_cycle")
    elif family in TEMPORAL_FAULTS:
        restored = audit.get("violation_end_cycle")
    else:
        restored = resume_cycle if resume_cycle is not None else (episode["cycles"] if episode["success"] else None)
    resume_kind = (
        "legal_reentry"
        if key == "m5"
        else "same_state_resume"
        if key == "retry_same_state"
        else "skill_retry"
    )
    return {
        "detected": bool(valid_alarms),
        "first_alarm": valid_alarms[0] if valid_alarms else None,
        "condition_restored_cycle": restored,
        "condition_restored": restored is not None,
        "relation_applicable": family in RELATION_FAULTS,
        "relation_restored": audit.get("relation_restored_cycle") is not None,
        "execution_resumed": resume_cycle is not None,
        "resume_kind": resume_kind,
        "legal_reentry": bool(key == "m5" and resume_cycle is not None),
        "post_resume_completion": bool(resume_cycle is not None and episode["success"]),
        # Retained for the M5-specific recovery-gap analysis.  It is true only
        # for Full; generic Skill-Retry and forced same-state resume are not
        # mislabeled as legal re-entry.
        "post_reentry_completion": bool(
            key == "m5" and resume_cycle is not None and episode["success"]
        ),
    }


def generate_recovery() -> dict[str, Any]:
    attribution_rows = [
        json.loads(line)
        for line in ATTRIBUTION_MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    attribution_keys = {
        (str(row["task"]), int(row["seed"])) for row in attribution_rows
    }
    if len(attribution_keys) != 800:
        raise RuntimeError("Figure 3 recovery attribution manifest is not 800 unique episodes")
    decomposition = []
    funnel = []
    categories = []
    for key, method_name in RECOVERY_METHODS.items():
        root = _endpoint_episode_path(key, "perturbed", "placeholder").parent
        episodes = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(root.glob("*.json"))]
        source_expected = 800 if key == "retry_same_state" else 2000
        if len(episodes) != source_expected:
            raise RuntimeError(
                f"incomplete endpoint recovery cell: {key}; "
                f"expected {source_expected}, found {len(episodes)}"
            )
        episodes = [
            episode
            for episode in episodes
            if (str(episode["task"]), int(episode["seed"]))
            in attribution_keys
        ]
        if len(episodes) != 800:
            raise RuntimeError(
                f"recovery system {key} is not paired on the 800-episode Stress-4 population"
            )
        triggered = [
            episode
            for episode in episodes
            if _current_audit(key, episode).get("physically_triggered")
        ]
        events = [_method_events(key, episode) for episode in triggered]
        detected_events = [event for event in events if event["detected"]]
        detected = len(detected_events)
        restored_events = [
            event
            for event in detected_events
            if event["condition_restored"]
            and int(event["condition_restored_cycle"]) >= int(event["first_alarm"])
        ]
        restored = len(restored_events)
        relation_events = [event for event in events if event["relation_applicable"]]
        relation_detected = [event for event in relation_events if event["detected"]]
        relation_restored = sum(
            event["relation_restored"]
            and int(event["condition_restored_cycle"]) >= int(event["first_alarm"])
            for event in relation_detected
        )
        resumed_events = [event for event in restored_events if event["execution_resumed"]]
        resumed = len(resumed_events)
        post = sum(event["post_resume_completion"] for event in resumed_events)
        legal = sum(event["legal_reentry"] for event in restored_events)
        final = sum(bool(episode["success"]) for episode in triggered)
        recovery_cycles = [int(episode.get("recovery_cycles", 0)) for episode in triggered]
        decomposition.append(
            {
                "method_key": key,
                "method": method_name,
                "physically_triggered": len(triggered),
                "detected": detected,
                "task_condition_restored": restored,
                "relation_applicable": len(relation_events),
                "relation_detected": len(relation_detected),
                "relation_restored": relation_restored,
                "resume_semantics": events[0]["resume_kind"] if events else "none",
                "execution_resumed": resumed,
                "legal_reentry": legal if key == "m5" else "not_applicable",
                "post_resume_completion": post,
                "final_success": final,
                "mean_recovery_cycles": float(np.mean(recovery_cycles)) if recovery_cycles else math.nan,
                "median_recovery_cycles": median(recovery_cycles) if recovery_cycles else math.nan,
            }
        )
        stages = (
            ("detected", detected, len(triggered)),
            ("task_condition_restored", restored, detected),
            ("execution_resumed", resumed, restored),
            ("final_success", final, len(triggered)),
        )
        if key in FIGURE3_RECOVERY_METHODS:
            for stage, numerator, denominator in stages:
                funnel.append(
                    {
                        "method_key": key,
                        "method": method_name,
                        "stage": stage,
                        "numerator": numerator,
                        "denominator": denominator,
                        "rate": numerator / denominator if denominator else math.nan,
                    }
                )
        counts = {
            "missed_or_late_detection": 0,
            "detected_but_condition_not_restored": 0,
            "restored_but_no_execution_resume": 0,
            "post_resume_policy_failure": 0,
            "successful_after_resume": 0,
        }
        for episode, event in zip(triggered, events):
            restored_after_detection = bool(
                event["detected"]
                and event["condition_restored"]
                and int(event["condition_restored_cycle"]) >= int(event["first_alarm"])
            )
            if not event["detected"]:
                category = "missed_or_late_detection"
            elif not restored_after_detection:
                category = "detected_but_condition_not_restored"
            elif not event["execution_resumed"]:
                category = "restored_but_no_execution_resume"
            elif not episode["success"]:
                category = "post_resume_policy_failure"
            else:
                category = "successful_after_resume"
            counts[category] += 1
        for category, count in counts.items():
            categories.append(
                {
                    "method_key": key,
                    "method": method_name,
                    "category": category,
                    "count": count,
                    "rate_among_physically_triggered": count / len(triggered) if triggered else math.nan,
                }
            )
    _write_csv(DERIVED / "appendix_recovery_decomposition.csv", decomposition)
    _write_csv(DERIVED / "fig3_recovery_funnel.csv", funnel)
    _write_csv(DERIVED / "appendix_recovery_failure_categories.csv", categories)
    return {"methods": len(RECOVERY_METHODS), "rows": len(decomposition)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("detection", "recovery", "all"))
    args = parser.parse_args(argv)
    result = {}
    if args.command in {"detection", "all"}:
        result["detection"] = generate_detection()
    if args.command in {"recovery", "all"}:
        result["recovery"] = generate_recovery()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
