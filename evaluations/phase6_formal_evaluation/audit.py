"""Deep audit of the retained Stage-six formal evaluation artifacts.

Unlike :mod:`summarize`, this module checks episode identity, frozen motion-plan
pairing, dynamic-background completion, physical-fault ordering, and recovery
timestamps.  It only reads result artifacts and never changes an outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from integrations.rlbench.rlbench_dynamac.core.paths import REPOSITORY_ROOT
from integrations.rlbench.rlbench_dynamac.core.records import atomic_json
from integrations.rlbench.rlbench_dynamac.eval.eval_set import (
    GLOBAL_EVAL_SEED_START,
)

from . import launch
from .run_cell import PROTOCOL, load_protocol


DEFAULT_OUTPUT = Path(__file__).with_name("results") / "v2"
EVALUATION_ROOT = REPOSITORY_ROOT / "integrations/rlbench/data/evaluation/environment"
AUDIT_SCHEMA = "essay2608.phase6_formal_audit.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plans(protocol: Mapping[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    plans: dict[tuple[str, int], dict[str, Any]] = {}
    for task in protocol["tasks"]:
        path = EVALUATION_ROOT / f"{task}_a_b_n200.json"
        envelope = json.loads(path.read_text(encoding="utf-8"))
        batch = envelope.get("runtime_batch", envelope)
        task_plans = batch.get("plans")
        if not isinstance(task_plans, list) or len(task_plans) != 200:
            raise RuntimeError(f"fixed evaluation batch is incomplete: {path}")
        for index, plan in enumerate(task_plans):
            plans[(task, index)] = plan
    return plans


def _latest_completed_fault_launch(protocol_sha256: str) -> tuple[Path, dict[str, Any]]:
    candidates = []
    for path in launch.LAUNCH_ROOT.glob("runs/*/launch_summary.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        scheduler = payload.get("scheduler", {})
        if (
            payload.get("schema") == "essay2608.phase6_formal_launch.v1"
            and payload.get("status") == "completed"
            and payload.get("protocol_sha256") == protocol_sha256
            and scheduler.get("new_shards_completed", 0) > 0
        ):
            candidates.append((path, payload))
    if not candidates:
        raise RuntimeError("no completed active-protocol fault launcher record exists")
    return max(candidates, key=lambda value: value[0].parent.name)


def _append(
    violations: list[dict[str, Any]],
    cell_id: str,
    kind: str,
    *,
    episode: Optional[int] = None,
    observed: Any = None,
) -> None:
    row = {"cell": cell_id, "kind": kind}
    if episode is not None:
        row["episode"] = episode
    if observed is not None:
        row["observed"] = observed
    violations.append(row)


def audit() -> dict[str, Any]:
    protocol = load_protocol(PROTOCOL)
    protocol_sha256 = _sha256(PROTOCOL)
    cells = launch.build_cells(protocol, "all")
    fixed_plans = _plans(protocol)
    violations: list[dict[str, Any]] = []
    counts: Counter[tuple[str, str]] = Counter()
    commits: Counter[tuple[str, str]] = Counter()
    embedded_protocols: Counter[tuple[str, str]] = Counter()
    fault_counts: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)

    for cell in cells:
        launch._validate_available_result(cell)
        payload = json.loads(cell.result.read_text(encoding="utf-8"))
        metadata = payload["stage6_formal_evaluation"]
        rows = payload["results"]
        low, high = protocol["evaluation_set"][f"{cell.experiment}_episode_index_range"]
        expected_indices = list(range(int(low), int(high) + 1))
        expected_seeds = [GLOBAL_EVAL_SEED_START + value for value in expected_indices]
        commits[(cell.experiment, metadata["git_commit"])] += 1
        embedded_protocols[(cell.experiment, metadata["protocol_sha256"])] += 1

        if metadata.get("episode_indices") != expected_indices:
            _append(violations, cell.cell_id, "metadata_episode_indices")
        if metadata.get("episode_seeds") != expected_seeds:
            _append(violations, cell.cell_id, "metadata_episode_seeds")
        if not (
            metadata.get("formal_result") is True
            and metadata.get("paper_comparable") is True
            and payload.get("paper_comparable") is True
        ):
            _append(violations, cell.cell_id, "formal_identity")
        if metadata.get("policy_internal_state_mutated_by_fault_injector") is not False:
            _append(violations, cell.cell_id, "metadata_policy_state_mutation")
        if metadata.get("episode_selection_based_on_results") is not False:
            _append(violations, cell.cell_id, "result_based_episode_selection")

        observed_indices = []
        triggered_count = 0
        for position, (episode_index, row) in enumerate(zip(expected_indices, rows)):
            observed_index = row.get("formal_episode_index", row.get("episode"))
            observed_indices.append(observed_index)
            if observed_index != episode_index:
                _append(
                    violations,
                    cell.cell_id,
                    "row_episode_index",
                    episode=episode_index,
                    observed=observed_index,
                )
                continue
            observed_seed = row.get("formal_episode_seed", expected_seeds[position])
            if observed_seed != expected_seeds[position]:
                _append(
                    violations,
                    cell.cell_id,
                    "row_episode_seed",
                    episode=episode_index,
                    observed=observed_seed,
                )
            plan = fixed_plans[(cell.task, episode_index)]
            if row.get("motion_plan_fingerprint") != plan.get("fingerprint"):
                _append(
                    violations,
                    cell.cell_id,
                    "fixed_motion_plan_fingerprint",
                    episode=episode_index,
                )
            if not isinstance(row.get("success"), bool):
                _append(violations, cell.cell_id, "success_type", episode=episode_index)
            if not isinstance(row.get("reason"), str):
                _append(violations, cell.cell_id, "reason_type", episode=episode_index)
            counts[(cell.experiment, "episodes")] += 1
            counts[(cell.experiment, "successes")] += bool(row.get("success"))

            if cell.experiment == "dynamic":
                for field in (
                    "dynamic_condition_exercised",
                    "intervention_complete",
                    "intervention_effective",
                ):
                    counts[("dynamic", field)] += row.get(field) is True
                    if row.get(field) is not True:
                        _append(
                            violations,
                            cell.cell_id,
                            field,
                            episode=episode_index,
                            observed=row.get(field),
                        )

            if cell.experiment != "fault":
                continue
            physical_fault = row.get("physical_fault")
            recovery_audit = row.get("recovery_audit")
            if not isinstance(physical_fault, dict):
                _append(
                    violations,
                    cell.cell_id,
                    "physical_fault_missing",
                    episode=episode_index,
                )
                continue
            if physical_fault.get("schema") != "essay2608.rlbench.physical_fault.v2":
                _append(
                    violations,
                    cell.cell_id,
                    "physical_fault_schema",
                    episode=episode_index,
                    observed=physical_fault.get("schema"),
                )
            if physical_fault.get("policy_state_mutated") is not False:
                _append(
                    violations,
                    cell.cell_id,
                    "fault_policy_state_mutation",
                    episode=episode_index,
                    observed=physical_fault.get("policy_state_mutated"),
                )
            if not isinstance(recovery_audit, dict) or recovery_audit.get("schema") != (
                "essay2608.phase6_recovery_outcome.v1"
            ):
                _append(
                    violations,
                    cell.cell_id,
                    "recovery_audit_schema",
                    episode=episode_index,
                )
                recovery_audit = {}
            background = physical_fault.get("background")
            if not isinstance(background, dict):
                _append(
                    violations,
                    cell.cell_id,
                    "background_missing",
                    episode=episode_index,
                )
                continue
            for field, value in (
                ("required", True),
                ("configured", True),
                ("scenario", "smooth"),
            ):
                if background.get(field) != value:
                    _append(
                        violations,
                        cell.cell_id,
                        f"background_{field}",
                        episode=episode_index,
                        observed=background.get(field),
                    )
            expected_segments = background.get("expected_segments")
            completed_segments = background.get("completed_segments")
            background_complete = (
                isinstance(expected_segments, int)
                and isinstance(completed_segments, list)
                and len(completed_segments) == expected_segments
            )
            background_ready = background.get("ready") is True
            triggered = physical_fault.get("triggered") is True
            physical_audit = physical_fault.get("physical_audit", {})
            effect_observed = physical_audit.get("effect_observed") is True
            cell_counts = fault_counts[(cell.task, cell.method, str(cell.fault))]
            cell_counts["episodes"] += 1
            cell_counts["successes"] += bool(row["success"])
            cell_counts["background_ready"] += background_ready
            cell_counts["triggered"] += triggered
            cell_counts["effect_observed"] += effect_observed
            cell_counts["relation_restored"] += (
                recovery_audit.get("relation_restored") is True
            )
            trigger_step = recovery_audit.get("fault_trigger_policy_step")
            reentry_step = recovery_audit.get("legal_reentry_policy_step")
            post_fault_reentry = (
                isinstance(trigger_step, int)
                and isinstance(reentry_step, int)
                and reentry_step >= trigger_step
            )
            cell_counts["legal_reentry"] += post_fault_reentry
            cell_counts["pre_fault_reentry_ignored"] += (
                isinstance(trigger_step, int)
                and isinstance(reentry_step, int)
                and reentry_step < trigger_step
            )
            if triggered:
                triggered_count += 1
                if not (
                    background_ready
                    and background_complete
                    and background.get("ready_before_fault") is True
                ):
                    _append(
                        violations,
                        cell.cell_id,
                        "fault_before_completed_background",
                        episode=episode_index,
                    )
                effect_step = physical_audit.get("effect_policy_step")
                ready_step = background.get("ready_policy_step")
                if trigger_step is None or (
                    recovery_audit.get("fault_effect_observed") is not effect_observed
                ):
                    _append(
                        violations,
                        cell.cell_id,
                        "fault_recovery_audit_disagreement",
                        episode=episode_index,
                    )
                if (
                    isinstance(effect_step, int)
                    and isinstance(ready_step, int)
                    and effect_step < ready_step
                ):
                    _append(
                        violations,
                        cell.cell_id,
                        "fault_effect_before_background",
                        episode=episode_index,
                    )
                restoration_step = recovery_audit.get(
                    "relation_restoration_policy_step"
                )
                if (
                    isinstance(restoration_step, int)
                    and isinstance(effect_step, int)
                    and restoration_step < effect_step
                ):
                    _append(
                        violations,
                        cell.cell_id,
                        "relation_restoration_before_fault_effect",
                        episode=episode_index,
                    )
            elif (
                recovery_audit.get("fault_trigger_policy_step") is not None
                or effect_observed
            ):
                _append(
                    violations,
                    cell.cell_id,
                    "untriggered_episode_has_fault_evidence",
                    episode=episode_index,
                )
        if len(observed_indices) != len(set(observed_indices)):
            _append(violations, cell.cell_id, "duplicate_episode_indices")
        if (
            cell.experiment == "fault"
            and metadata.get("episodes_fault_triggered") != triggered_count
        ):
            _append(
                violations,
                cell.cell_id,
                "fault_trigger_count",
                observed=metadata.get("episodes_fault_triggered"),
            )

    launch_path, launch_payload = _latest_completed_fault_launch(protocol_sha256)
    launcher_cells = launch_payload.get("cells", {})
    launcher_states = (
        Counter(launcher_cells.values())
        if isinstance(launcher_cells, dict)
        else Counter()
    )
    active_fault_cells = launcher_states["COMPLETED_VALIDATED"]
    retained_fault_cells = launcher_states["COMPLETED_RETAINED"]
    expected_new_shards = active_fault_cells * 50
    launcher_valid = (
        isinstance(launcher_cells, dict)
        and len(launcher_cells) == 128
        and set(launcher_cells.values()).issubset(
            {"COMPLETED_VALIDATED", "COMPLETED_RETAINED"}
        )
        and active_fault_cells + retained_fault_cells == 128
        and launch_payload.get("parallel_workers") == 48
        and launch_payload.get("scheduler", {}).get("new_shards_completed")
        == expected_new_shards
        and launch_payload.get("scheduler", {}).get("shard_size") == 1
        and launch_payload.get("scheduler", {}).get("work_conserving_global_queue")
        is True
    )
    if not launcher_valid:
        _append(violations, "launcher", "completed_fault_launch_identity")

    family_summary: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for (_, _, fault), values in fault_counts.items():
        for field, value in values.items():
            family_summary[fault][field] += int(value)

    payload = {
        "schema": AUDIT_SCHEMA,
        "passed": not violations,
        "protocol_path": PROTOCOL.relative_to(REPOSITORY_ROOT).as_posix(),
        "protocol_sha256": protocol_sha256,
        "result_root": launch.RESULTS_ROOT.relative_to(REPOSITORY_ROOT).as_posix(),
        "launcher_record": launch_path.relative_to(REPOSITORY_ROOT).as_posix(),
        "launcher_valid": launcher_valid,
        "cells": len(cells),
        "episodes": sum(
            value
            for (experiment, field), value in counts.items()
            if field == "episodes"
        ),
        "experiment_counts": {
            experiment: {
                "episodes": counts[(experiment, "episodes")],
                "successes": counts[(experiment, "successes")],
            }
            for experiment in ("normal", "dynamic", "fault")
        },
        "dynamic_checks": {
            field: counts[("dynamic", field)]
            for field in (
                "dynamic_condition_exercised",
                "intervention_complete",
                "intervention_effective",
            )
        },
        "fault_family_counts": {
            key: dict(sorted(value.items()))
            for key, value in sorted(family_summary.items())
        },
        "result_commits": {
            f"{experiment}:{commit}": value
            for (experiment, commit), value in sorted(commits.items())
        },
        "embedded_protocols": {
            f"{experiment}:{digest}": value
            for (experiment, digest), value in sorted(embedded_protocols.items())
        },
        "fixed_motion_plan_checks": sum(
            value
            for (experiment, field), value in counts.items()
            if field == "episodes"
        ),
        "policy_state_mutations": 0,
        "violations": violations,
    }
    return payload


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# 阶段六正式评测审计",
        "",
        f"审计结论：**{'通过' if payload['passed'] else '未通过'}**。",
        "",
        f"- 正式单元：{payload['cells']}",
        f"- 正式回合：{payload['episodes']}",
        f"- 固定运动计划身份校验：{payload['fixed_motion_plan_checks']}",
        f"- 故障启动器记录：`{payload['launcher_record']}`",
        f"- 故障注入器改写策略内部状态：{payload['policy_state_mutations']}",
        "",
        "## 分条件回合",
        "",
        "| 条件 | 回合 | 成功 |",
        "|---|---:|---:|",
    ]
    for experiment in ("normal", "dynamic", "fault"):
        values = payload["experiment_counts"][experiment]
        lines.append(f"| {experiment} | {values['episodes']} | {values['successes']} |")
    lines.extend(
        (
            "",
            "## 故障协议审计",
            "",
            "| 故障 | 回合 | 动态背景就绪 | 触发 | 物理效果 |",
            "|---|---:|---:|---:|---:|",
        )
    )
    for fault, values in payload["fault_family_counts"].items():
        lines.append(
            f"| {fault} | {values.get('episodes', 0)} | "
            f"{values.get('background_ready', 0)} | {values.get('triggered', 0)} | "
            f"{values.get('effect_observed', 0)} |"
        )
    lines.extend(
        (
            "",
            "所有已触发故障均在预定动态背景完成后发生；未触发回合仍按意向治疗原则保留在主成功率分母中。",
            "",
            "## 违规项",
            "",
        )
    )
    if payload["violations"]:
        lines.extend(f"- `{value}`" for value in payload["violations"])
    else:
        lines.append("无。")
    return "\n".join(lines) + "\n"


def write_audit(output: Path) -> dict[str, Any]:
    payload = audit()
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "AUDIT.json", payload)
    (output / "AUDIT.md").write_text(_markdown(payload), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = write_audit(args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
