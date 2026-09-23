"""Final integrity gate for A6 formal experiments and paper artifacts.

This is a single end-of-stage audit.  It does not run episodes, alter formal
results, or substitute a partial PASS when an experiment is missing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from evaluations.iclr2027.analysis.a6_endpoint_integrity import (
    audit as audit_current_endpoints,
)


ROOT = Path(__file__).resolve().parents[3]
EVAL_ROOT = ROOT / "evaluations" / "iclr2027"
RESULTS = EVAL_ROOT / "results"
OUTPUT = RESULTS / "a6_acceptance"
CURRENT_ENDPOINT_INTEGRITY = (
    RESULTS / "a6_execution" / "A6_CURRENT_ENDPOINT_INTEGRITY.json"
)

ANALYSES = {
    "e2": RESULTS / "controlled" / "e1_e2" / "derived" / "E2_FIGURE3_ANALYSIS.json",
    "e3_ab": RESULTS / "controlled" / "e3" / "derived" / "E3_AB_ANALYSIS.json",
    "e3_c": RESULTS / "controlled" / "e3" / "derived" / "E3C_ANALYSIS.json",
    "e3_c_oracle": RESULTS / "controlled" / "e3" / "derived" / "E3C_ORACLE_FEASIBILITY.json",
    "e4": RESULTS / "controlled" / "e4" / "derived" / "E4_ANALYSIS.json",
    "e5_core": RESULTS / "controlled" / "e5" / "derived" / "E5_CORE_ANALYSIS.json",
    "e5_fine": RESULTS / "controlled" / "e5" / "derived" / "E5_FINE_ANALYSIS.json",
    "e5_threshold": RESULTS / "controlled" / "e5" / "derived" / "E5_THRESHOLD_SENSITIVITY.json",
    "e5_control_equivalence": RESULTS / "controlled" / "e5" / "derived" / "control_equivalence" / "CONTROL_EQUIVALENCE_ANALYSIS.json",
    "e6": RESULTS / "native_v3" / "derived" / "E6_ANALYSIS.json",
}
EXPECTED_COUNTS = {
    "e2_relation_repair_same_state": 1200,
    "e3_budget_lofo_shadow": 19200,
    "e3_c": 12000,
    "e3_c_oracle": 600,
    "e4": 9600,
    "e5_core": 3600,
    "e5_fine_new": 7200,
    "e5_fine_evaluated": 8400,
    "e5_control_equivalence_log_episodes": 2400,
    "e5_control_equivalence_pairs": 1200,
    "e6_all_reported_cells": 3600,
}

PAPER_ARTIFACTS = {
    "table1_csv": RESULTS / "controlled" / "e1_e2" / "derived" / "table1_shared_backbone.csv",
    "table1_tex": RESULTS / "controlled" / "e1_e2" / "derived" / "table1_shared_backbone.tex",
    "table2_csv": RESULTS / "native_v3" / "derived" / "table2_native_systems.csv",
    "table2_tex": RESULTS / "native_v3" / "derived" / "table2_native_systems.tex",
    "table3_csv": RESULTS / "controlled" / "e5" / "derived" / "table3_core_ablation.csv",
    "table3_tex": RESULTS / "controlled" / "e5" / "derived" / "table3_core_ablation.tex",
    "figure3_monitor_pr": RESULTS / "controlled" / "e1_e2" / "derived" / "fig3_monitor_pr.csv",
    "figure3_delay_false_alarm": RESULTS / "controlled" / "e1_e2" / "derived" / "fig3_delay_false_alarm.csv",
    "figure3_detection_by_fault": RESULTS / "controlled" / "e1_e2" / "derived" / "fig3_detection_by_fault.csv",
    "figure3_recovery_funnel": RESULTS / "controlled" / "e1_e2" / "derived" / "fig3_recovery_funnel.csv",
    "figure4_failure_budget": RESULTS / "controlled" / "e3" / "derived" / "fig4_failure_budget.csv",
    "figure4_lofo": RESULTS / "controlled" / "e3" / "derived" / "fig4_leave_one_family_out.csv",
    "figure4_severity_composition": RESULTS / "controlled" / "e3" / "derived" / "fig4_severity_composition.csv",
    "figure4_oracle_feasibility": RESULTS / "controlled" / "e3" / "derived" / "fig4_oracle_feasibility.csv",
    "figure5_stage_count": RESULTS / "controlled" / "e4" / "derived" / "fig5_success_by_stage_count.csv",
    "figure5_scheduled_opportunities": RESULTS / "controlled" / "e4" / "derived" / "fig5_success_by_scheduled_opportunities.csv",
    "figure5_actual_trigger_appendix": RESULTS / "controlled" / "e4" / "derived" / "appendix_success_by_actual_trigger_count.csv",
    "figure5_remaining_completion": RESULTS / "controlled" / "e4" / "derived" / "fig5_remaining_completion.csv",
    "fine_ablation": RESULTS / "controlled" / "e5" / "derived" / "appendix_fine_ablation.csv",
    "threshold_sensitivity": RESULTS / "controlled" / "e5" / "derived" / "appendix_threshold_sensitivity.csv",
    "control_equivalence_cycle_metrics": RESULTS / "controlled" / "e5" / "derived" / "control_equivalence" / "appendix_control_equivalence_cycle_metrics.csv",
    "control_equivalence_paired_effects": RESULTS / "controlled" / "e5" / "derived" / "control_equivalence" / "appendix_control_equivalence_paired_effects.csv",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _validate_hashes(name: str, path: Path, summary: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    root = path.parent
    outputs = summary.get("outputs", {})
    hashes = summary.get("output_sha256", {})
    if isinstance(outputs, Mapping):
        expected = dict(outputs)
    elif isinstance(outputs, list) and isinstance(hashes, Mapping):
        expected = {str(item): str(hashes.get(str(item), "")) for item in outputs}
    else:
        return [f"{name}: malformed output hash index"]
    for relative, expected_hash in expected.items():
        output = root / str(relative)
        if not output.is_file():
            errors.append(f"{name}: missing output {relative}")
        elif _sha256(output) != expected_hash:
            errors.append(f"{name}: output hash mismatch {relative}")
    return errors


def status() -> dict[str, Any]:
    missing = [str(path.relative_to(ROOT)) for path in ANALYSES.values() if not path.is_file()]
    if not CURRENT_ENDPOINT_INTEGRITY.is_file():
        missing.append(str(CURRENT_ENDPOINT_INTEGRITY.relative_to(ROOT)))
    completed = [name for name, path in ANALYSES.items() if path.is_file()]
    if CURRENT_ENDPOINT_INTEGRITY.is_file():
        completed.append("current_endpoint_integrity")
    return {
        "stage": "A6",
        "ready_for_acceptance": not missing,
        "completed_analyses": completed,
        "missing_analyses": missing,
    }


def accept() -> dict[str, Any]:
    errors: list[str] = []
    # Recompute rather than merely trusting the earlier PASS record, so any
    # endpoint or re-audit drift after Table/Figure generation is detected at
    # the final A6 gate.
    audit_current_endpoints()
    missing = [path for path in ANALYSES.values() if not path.is_file()]
    if missing:
        raise RuntimeError("A6 analyses are incomplete: " + ", ".join(str(path.relative_to(ROOT)) for path in missing))

    a5_path = RESULTS / "controlled" / "e1_e2" / "A5_ACCEPTANCE.json"
    a5 = _json(a5_path)
    if not str(a5.get("status", "")).startswith("PASS"):
        errors.append("A5 acceptance is not PASS")
    end_to_end = a5.get("end_to_end", {})
    for field, expected in (("episode_results", 36000), ("cycle_files", 36000), ("identity_errors", 0), ("pairing_errors", 0), ("temporary_files", 0)):
        if int(end_to_end.get(field, -1)) != expected:
            errors.append(f"A5 {field} != {expected}")
    if int(a5.get("shadow_errors", -1)) != 0:
        errors.append("A5 shadow errors are nonzero")
    reaudit = a5.get("re_audit", {})
    if reaudit.get("status") != "CURRENT" or int(reaudit.get("errors", -1)) != 0:
        errors.append("A5 current physical re-audit is not valid")

    if not CURRENT_ENDPOINT_INTEGRITY.is_file():
        errors.append("A6 current composite endpoint integrity is missing")
        current_endpoint = {}
    else:
        current_endpoint = _json(CURRENT_ENDPOINT_INTEGRITY)
        if current_endpoint.get("status") != "PASS":
            errors.append("A6 current composite endpoint integrity is not PASS")
        endpoint_totals = current_endpoint.get("end_to_end") or {}
        for field, expected in (
            ("episode_results", 36000),
            ("cycle_files", 36000),
            ("identity_errors", 0),
            ("pairing_errors", 0),
            ("temporary_files", 0),
        ):
            if int(endpoint_totals.get(field, -1)) != expected:
                errors.append(f"A6 current endpoint {field} != {expected}")
        current_reaudit = current_endpoint.get("re_audit") or {}
        if (
            int(current_reaudit.get("episodes", -1)) != 18000
            or int(current_reaudit.get("errors", -1)) != 0
        ):
            errors.append("A6 current endpoint re-audit is invalid")

    summaries = {name: _json(path) for name, path in ANALYSES.items()}
    for name, summary in summaries.items():
        if not str(summary.get("status", "")).startswith("PASS"):
            errors.append(f"{name}: analysis is not PASS")
        errors.extend(_validate_hashes(name, ANALYSES[name], summary))

    observed_counts = {
        "e2_relation_repair_same_state": int(summaries["e2"].get("new_physical_episodes", -1)),
        "e3_budget_lofo_shadow": int(summaries["e3_ab"].get("new_shadow_replays", -1)),
        "e3_c": int(summaries["e3_c"].get("new_episodes", -1)),
        "e3_c_oracle": int(summaries["e3_c_oracle"].get("episodes", -1)),
        "e4": int(summaries["e4"].get("episodes", -1)),
        "e5_core": int(summaries["e5_core"].get("new_episodes", -1)),
        "e5_fine_new": int(summaries["e5_fine"].get("new_episodes", -1)),
        "e5_fine_evaluated": int(
            summaries["e5_fine"].get("evaluated_episodes", -1)
        ),
        "e5_control_equivalence_log_episodes": int(
            summaries["e5_control_equivalence"].get("cycle_metric_episodes", -1)
        ),
        "e5_control_equivalence_pairs": int(
            summaries["e5_control_equivalence"].get("episode_pairs", -1)
        ),
        "e6_all_reported_cells": int(summaries["e6"].get("episodes", -1)),
    }
    for name, expected in EXPECTED_COUNTS.items():
        if observed_counts[name] != expected:
            errors.append(f"{name}: {observed_counts[name]} != {expected}")

    e6_delivery_path = RESULTS / "b_delivery" / "A_ACCEPTANCE_E6_V3_FORMAL.json"
    if not e6_delivery_path.is_file():
        errors.append("E6 v3 B formal delivery acceptance is missing")
        e6_delivery = {}
    else:
        e6_delivery = _json(e6_delivery_path)
        if e6_delivery.get("status") != "PASS" or int(e6_delivery.get("results_verified", -1)) != 1200:
            errors.append("E6 v3 B formal delivery acceptance is not a 1,200-result PASS")

    paper_hashes = {}
    for name, path in PAPER_ARTIFACTS.items():
        if not path.is_file():
            errors.append(f"missing paper artifact: {path.relative_to(ROOT)}")
        else:
            paper_hashes[name] = {
                "path": str(path.relative_to(ROOT)),
                "sha256": _sha256(path),
            }

    qualitative = EVAL_ROOT / "configs" / "shared" / "qualitative_replay_selection.json"
    if not qualitative.is_file():
        errors.append("qualitative replay selection is missing")
    else:
        selection = _json(qualitative)
        if not selection.get("selection_rule"):
            errors.append("qualitative replay selection has no deterministic rule")

    formal_roots = (
        RESULTS / "controlled" / "e3",
        RESULTS / "controlled" / "e4",
        RESULTS / "controlled" / "e5",
        RESULTS / "native_v3" / "ours",
    )
    temporary_files = [
        str(path.relative_to(ROOT))
        for result_root in formal_roots
        for path in result_root.rglob("*.tmp")
    ]
    if temporary_files:
        errors.append(f"formal result roots contain {len(temporary_files)} temporary files")

    if errors:
        raise RuntimeError("A6 acceptance failed:\n- " + "\n- ".join(errors))

    record = {
        "schema": "essay2608.iclr2027.a6-acceptance.v1",
        "status": "PASS_AWAITING_USER_CONFIRMATION_FOR_A7",
        "a5": {"path": str(a5_path.relative_to(ROOT)), "sha256": _sha256(a5_path), "episode_results": 36000},
        "current_endpoint_integrity": {
            "path": str(CURRENT_ENDPOINT_INTEGRITY.relative_to(ROOT)),
            "sha256": _sha256(CURRENT_ENDPOINT_INTEGRITY),
            "episode_results": 36000,
        },
        "new_formal_episode_counts": observed_counts,
        "analysis_records": {name: {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path), "status": summaries[name]["status"]} for name, path in ANALYSES.items()},
        "e6_b_formal_delivery": {
            "path": str(e6_delivery_path.relative_to(ROOT)),
            "sha256": _sha256(e6_delivery_path),
            "results_verified": 1200,
        },
        "paper_artifacts": paper_hashes,
        "infrastructure_failures_remain_intention_to_treat": True,
        "temporary_files": 0,
        "user_confirmation_for_a7": False,
    }
    _atomic_json(OUTPUT / "A6_ACCEPTANCE.json", record)
    lines = [
        "# A6 验收结果",
        "",
        "状态：`PASS_AWAITING_USER_CONFIRMATION_FOR_A7`。基础设施失败继续保留在 ITT 分母中。",
        "",
        "| 实验单元 | 新增/汇总回合数 |",
        "|---|---:|",
    ]
    labels = {
        "e2_relation_repair_same_state": "E2 relation repair plus same-state resume",
        "e3_budget_lofo_shadow": "E3 budget/LOFO no-op shadow replay",
        "e3_c": "E3-C severity/composition/stage",
        "e3_c_oracle": "E3-C oracle feasibility diagnostic",
        "e4": "E4 long horizon",
        "e5_core": "E5 core ablations",
        "e5_fine_new": "E5 fine ablations newly executed",
        "e5_fine_evaluated": "E5 fine ablations evaluated (including E2 reuse)",
        "e6_all_reported_cells": "E6 Table 2 all reported cells",
    }
    lines.extend(f"| {labels[key]} | {observed_counts[key]:,} |" for key in labels)
    lines.extend(("", "Table 1--3 与 Figure 3--5 的数据文件均已生成并写入 SHA256 索引。", ""))
    _atomic_text(OUTPUT / "RESULTS.md", "\n".join(lines))
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("status", "accept"))
    args = parser.parse_args(argv)
    value = status() if args.command == "status" else accept()
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
