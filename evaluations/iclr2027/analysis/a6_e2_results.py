"""Finalize Figure 3 after the same-state relation-repair cell is complete."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from evaluations.iclr2027.analysis import a5_detection


ROOT = Path(__file__).resolve().parents[3]
RESULT_ROOT = ROOT / "evaluations/iclr2027/results/controlled/e1_e2"
E5_RETRY = ROOT / "evaluations/iclr2027/results/controlled/e5/fine/retry_same_state"
DERIVED = RESULT_ROOT / "derived"
OUTPUT = DERIVED / "E2_FIGURE3_ANALYSIS.json"
ATTRIBUTION_OUTPUT = DERIVED / "fig3_system_attribution_audit.json"
PRIMARY = {"m6", "retry_same_state", "m5"}
ATTRIBUTION_MANIFEST = ROOT / "evaluations/iclr2027/manifests/ablation4.jsonl"
METHOD_CONFIGS = {
    "m6": ROOT / "evaluations/iclr2027/configs/methods/m6_ours_monitor_retry.json",
    "retry_same_state": ROOT
    / "evaluations/iclr2027/configs/methods/ablation_retry_same_state.json",
    "m5": ROOT / "evaluations/iclr2027/configs/methods/m5_full.json",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _episode_root(method: str, condition: str) -> Path:
    if method == "retry_same_state":
        return E5_RETRY / condition / "episodes"
    return RESULT_ROOT / "end_to_end" / method / condition / "episodes"


def _episode_index(method: str, condition: str) -> dict[tuple[str, int], dict[str, Any]]:
    index: dict[tuple[str, int], dict[str, Any]] = {}
    for path in sorted(_episode_root(method, condition).glob("*.json")):
        episode = json.loads(path.read_text(encoding="utf-8"))
        identity = (str(episode["task"]), int(episode["seed"]))
        if identity in index:
            raise RuntimeError(
                f"duplicate {method}/{condition} episode identity: {identity}"
            )
        index[identity] = episode
    return index


def _paired_system_attribution_audit() -> dict[str, Any]:
    """Verify paired inputs and the intended staged recovery interventions.

    Figure 3(c) is a paired-seed end-to-end intervention experiment, not a
    shadow replay.  Once any controller intervenes, its observation history and
    later alarm times may legitimately diverge.  Causal attribution therefore
    comes from the shared frozen task-state inference implementation plus
    explicit feature-profile differences, not from requiring separately run
    simulator trajectories to remain byte-identical.
    """

    from source.policy.tsf.ablation import TSFFeatureProfile

    attribution_rows = [
        json.loads(line)
        for line in ATTRIBUTION_MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    identities = {(str(row["task"]), int(row["seed"])) for row in attribution_rows}
    if len(identities) != 800:
        raise RuntimeError("Figure 3 attribution manifest is not 800 unique episodes")
    indexes = {
        condition: {
            method: _episode_index(method, condition) for method in sorted(PRIMARY)
        }
        for condition in ("nominal", "perturbed")
    }
    nominal_identities = set(indexes["nominal"]["retry_same_state"])
    if len(nominal_identities) != 400:
        raise RuntimeError("Figure 3 middle-system nominal population is not 400 episodes")
    expected_identities = {
        "nominal": nominal_identities,
        "perturbed": identities,
    }
    source_counts = {
        condition: {method: len(index) for method, index in methods.items()}
        for condition, methods in indexes.items()
    }
    matched_counts = {
        condition: {
            method: len(expected_identities[condition].intersection(index))
            for method, index in methods.items()
        }
        for condition, methods in indexes.items()
    }
    episode_fields = (
        "task",
        "seed",
        "condition",
        "fault_family",
        "fault_severity",
        "trigger_stage",
        "variation",
    )
    paired_metadata_mismatches: list[dict[str, Any]] = []
    for condition, expected in expected_identities.items():
        for identity in sorted(expected):
            if not all(identity in indexes[condition][method] for method in PRIMARY):
                continue
            values = {
                method: {
                    field: indexes[condition][method][identity].get(field)
                    for field in episode_fields
                }
                for method in sorted(PRIMARY)
            }
            if len({json.dumps(value, sort_keys=True) for value in values.values()}) != 1:
                if len(paired_metadata_mismatches) < 20:
                    paired_metadata_mismatches.append(
                        {
                            "condition": condition,
                            "task": identity[0],
                            "seed": identity[1],
                            "values": values,
                        }
                    )

    configs = {
        method: json.loads(path.read_text(encoding="utf-8"))
        for method, path in METHOD_CONFIGS.items()
    }
    profiles = {
        method: TSFFeatureProfile.named(str(config["feature_profile"]))
        for method, config in configs.items()
    }
    shared_fields = (
        "dynamic_frame_roles",
        "relation_scene_boundary_guards",
        "active_relation_verification",
        "complete_state_progress_evidence",
        "belief_driven_progress",
        "boundary_gated_advancement",
        "relation_progress_evidence",
        "scene_progress_evidence",
        "control_equivalence_aggregation",
    )
    shared_profile_values = {
        field: {method: bool(getattr(profile, field)) for method, profile in profiles.items()}
        for field in shared_fields
    }
    shared_profile_pass = all(
        len(set(method_values.values())) == 1 and next(iter(method_values.values()))
        for method_values in shared_profile_values.values()
    )
    runtime_pass = len(
        {json.dumps(config["runtime"], sort_keys=True) for config in configs.values()}
    ) == 1
    intervention_pass = (
        profiles["m6"].state_aware_recovery_reentry is False
        and configs["m6"]["recovery"]["kind"] == "skill_retry"
        and profiles["retry_same_state"].state_aware_recovery_reentry is True
        and profiles["retry_same_state"].legal_reentry_selection is False
        and configs["retry_same_state"]["recovery"]["kind"]
        == "relation_repair_and_same_state_reentry"
        and profiles["m5"].state_aware_recovery_reentry is True
        and profiles["m5"].legal_reentry_selection is True
        and configs["m5"]["recovery"]["kind"]
        == "relation_repair_and_legal_reentry"
    )
    population_pass = (
        matched_counts["nominal"] == {method: 400 for method in sorted(PRIMARY)}
        and matched_counts["perturbed"] == {method: 800 for method in sorted(PRIMARY)}
        and not paired_metadata_mismatches
    )
    passed = population_pass and shared_profile_pass and runtime_pass and intervention_pass
    result = {
        "schema": "essay2608.iclr2027.figure3-system-attribution-audit.v1",
        "status": "PASS" if passed else "FAIL",
        "design": "paired-seed end-to-end intervention ablation",
        "paired_population": {"nominal": 400, "perturbed": 800},
        "source_episode_counts": source_counts,
        "matched_episode_counts": matched_counts,
        "paired_metadata_fields": list(episode_fields),
        "paired_metadata_mismatch_examples": paired_metadata_mismatches,
        "shared_task_state_inference_profile": shared_profile_values,
        "shared_runtime_config": runtime_pass,
        "staged_interventions": {
            "m6": "shared TSF inference and active verification, then Skill-Retry",
            "retry_same_state": (
                "shared TSF inference, active verification, and relation repair; "
                "resume the frozen pre-recovery task state without legal-state search"
            ),
            "m5": (
                "shared TSF inference, active verification, relation repair, and "
                "support-constrained legal re-entry"
            ),
            "verified": intervention_pass,
        },
        "trajectory_identity_required": False,
        "trajectory_identity_rationale": (
            "These are separate paired simulator rollouts. After an intervention, and "
            "occasionally near a numerical state threshold before one, observation "
            "histories may diverge; byte-identical traces are required only for the "
            "shadow-monitor comparison in Figure 3(a-b), not this end-to-end recovery "
            "ablation."
        ),
    }
    _atomic_json(ATTRIBUTION_OUTPUT, result)
    if not passed:
        raise RuntimeError(
            "Figure 3(c) systems failed the paired attribution audit; "
            f"see {ATTRIBUTION_OUTPUT}"
        )
    return result


def main() -> int:
    counts = {
        condition: len(list((E5_RETRY / condition / "episodes").glob("*.json")))
        for condition in ("nominal", "perturbed")
    }
    if counts != {"nominal": 400, "perturbed": 800}:
        raise RuntimeError(f"incomplete Figure 3 middle system: {counts}")

    attribution_audit = _paired_system_attribution_audit()
    detection = a5_detection.generate_detection()
    recovery = a5_detection.generate_recovery()
    funnel_path = DERIVED / "fig3_recovery_funnel.csv"
    with funnel_path.open(newline="", encoding="utf-8") as stream:
        funnel_keys = {row["method_key"] for row in csv.DictReader(stream)}
    if funnel_keys != PRIMARY:
        raise RuntimeError(f"Figure 3(c) carries the wrong systems: {sorted(funnel_keys)}")

    outputs = [
        "appendix_detection_complete.csv",
        "appendix_detection_by_fault.csv",
        "fig3_monitor_pr.csv",
        "fig3_delay_false_alarm.csv",
        "fig3_detection_by_fault.csv",
        "appendix_recovery_decomposition.csv",
        "appendix_recovery_failure_categories.csv",
        "fig3_recovery_funnel.csv",
        "fig3_system_attribution_audit.json",
    ]
    result = {
        "schema": "essay2608.iclr2027.a6-e2-figure3-analysis.v1",
        "status": "PASS",
        "new_physical_episodes": 1200,
        "middle_system": {
            "method_key": "retry_same_state",
            "paper_name": "TSF-Monitor + Relation Repair + Same-State Resume",
            "not_generic_skill_retry": True,
            "episodes": counts,
        },
        "figure3c_systems": [
            "TSF-Monitor + Skill-Retry (M6)",
            "TSF-Monitor + Relation Repair + Same-State Resume",
            "Full TSF with Legal Re-entry (M5)",
        ],
        "detection": detection,
        "recovery": recovery,
        "system_attribution_audit": attribution_audit,
        "outputs": outputs,
        "output_sha256": {name: _sha256(DERIVED / name) for name in outputs},
    }
    _atomic_json(OUTPUT, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
