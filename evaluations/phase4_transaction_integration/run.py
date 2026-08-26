"""Real-model integration validation for phase-four joint transactions."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from essay2608.policy.closed_loop import (
    BeliefUpdater,
    BeliefUpdaterConfig,
    BoundaryRuntimeConfig,
    ClosedLoopExecutionController,
    MultiArmBoundaryController,
    ProgressEstimate,
    ProgressStatus,
)
from essay2608.policy.dynamac import DynaMACObservation
from evaluations.phase23_component_ab.run import (
    BELIEF_CONFIG_PATH,
    REPOSITORY_ROOT,
    _load_cases,
    _mode_by_skill,
    _runtime_observation,
)
from evaluations.phase4_boundary_calibration.run import _raw_samples, _reset_updater

SCHEMA = "essay2608-phase4-transaction-integration-v1"


def _read_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema",
        "claim_boundary",
        "task",
        "demonstration_indices",
        "source_skill",
        "target_skill",
        "leading_arm",
        "delayed_arm",
        "delayed_cycles",
        "terminal_hold_cycles",
        "runtime_config",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("联合事务集成配置字段不完整或包含未知字段")
    if value["schema"] != SCHEMA:
        raise ValueError("联合事务集成配置 schema 不匹配")
    demonstrations = tuple(int(item) for item in value["demonstration_indices"])
    if not demonstrations or min(demonstrations) < 0:
        raise ValueError("示范索引不能为空且必须非负")
    if int(value["target_skill"]) != int(value["source_skill"]) + 1:
        raise ValueError("本验证要求相邻技能边界")
    delayed_cycles = int(value["delayed_cycles"])
    hold_cycles = int(value["terminal_hold_cycles"])
    if delayed_cycles < 1 or hold_cycles <= delayed_cycles:
        raise ValueError("终端保持周期必须严格大于正的延迟周期")
    if value["leading_arm"] == value["delayed_arm"]:
        raise ValueError("先就绪臂和延迟臂不能相同")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_text(state: Any) -> str:
    return f"k{state.skill_index}:t{state.local_index}"


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"不能写入空结果 {path.name}")
    names: list[str] = []
    for row in rows:
        for name in row:
            if name not in names:
                names.append(name)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=names, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _git_value(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _terminal_context(
    cases_by_arm: Mapping[str, Any],
    demonstration: int,
    belief_config: BeliefUpdaterConfig,
    source_skill: int,
    target_skill: int,
) -> tuple[
    int,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Mapping[int, int]],
]:
    samples_by_arm = {
        arm: _raw_samples(case, demonstration) for arm, case in cases_by_arm.items()
    }
    lengths = {len(samples) for samples in samples_by_arm.values()}
    if len(lengths) != 1:
        raise RuntimeError("LiftTray 双臂示范控制周期未同步")
    sample_count = lengths.pop()
    modes = {
        arm: _mode_by_skill(case.policy, demonstration)
        for arm, case in cases_by_arm.items()
    }
    updaters = {
        arm: BeliefUpdater(case.model, belief_config)
        for arm, case in cases_by_arm.items()
    }
    previous_beliefs: dict[str, Any | None] = {arm: None for arm in cases_by_arm}
    for arm, case in cases_by_arm.items():
        _reset_updater(
            case,
            updaters[arm],
            samples_by_arm[arm][0],
            0,
            modes[arm],
            None,
        )

    for tick in range(1, sample_count - 1):
        beliefs: dict[str, Any] = {}
        reset_occurred = False
        for arm, case in cases_by_arm.items():
            previous = samples_by_arm[arm][tick - 1]
            current = samples_by_arm[arm][tick]
            if current.state_id.skill_index != previous.state_id.skill_index:
                _reset_updater(
                    case,
                    updaters[arm],
                    current,
                    tick,
                    modes[arm],
                    previous_beliefs[arm],
                )
                reset_occurred = True
                continue
            belief = updaters[arm].update(
                _runtime_observation(tick, current, previous),
                executed_reference_state=previous.state_id,
                mode_by_skill=modes[arm],
            )
            beliefs[arm] = belief
            previous_beliefs[arm] = belief
        if reset_occurred or set(beliefs) != set(cases_by_arm):
            continue
        if all(
            samples_by_arm[arm][tick].state_id.skill_index == source_skill
            and samples_by_arm[arm][tick + 1].state_id.skill_index == target_skill
            for arm in cases_by_arm
        ):
            frozen = {arm: samples[tick] for arm, samples in samples_by_arm.items()}
            return tick, frozen, updaters, beliefs, modes
    raise RuntimeError(
        f"示范 {demonstration} 未找到同步边界 {source_skill}->{target_skill}"
    )


def _delayed_progress_belief(belief: Any, delayed_state: Any) -> Any:
    progress = ProgressEstimate(
        prior={delayed_state: 1.0},
        posterior={delayed_state: 1.0},
        nominal_state=delayed_state,
        estimated_state=delayed_state,
        confidence=1.0,
        entropy=0.0,
        best_explanation_score=belief.progress.best_explanation_score,
        status=ProgressStatus.ALIGNED,
    )
    return replace(
        belief,
        progress=progress,
        local_candidates=(delayed_state,),
        expanded_candidates=(),
    )


def _validate_one(
    *,
    config: Mapping[str, Any],
    cases_by_arm: Mapping[str, Any],
    runtime_config: BoundaryRuntimeConfig,
    belief_config: BeliefUpdaterConfig,
    demonstration: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_skill = int(config["source_skill"])
    target_skill = int(config["target_skill"])
    leading_arm = str(config["leading_arm"])
    delayed_arm = str(config["delayed_arm"])
    delayed_cycles = int(config["delayed_cycles"])
    hold_cycles = int(config["terminal_hold_cycles"])
    models = {arm: case.model for arm, case in cases_by_arm.items()}
    boundary_tick, frozen, updaters, _, modes = _terminal_context(
        cases_by_arm,
        demonstration,
        belief_config,
        source_skill,
        target_skill,
    )

    boundary_models = {
        arm: next(
            boundary
            for boundary in model.boundaries.values()
            if boundary.source_skill == source_skill
            and boundary.target_skill == target_skill
        )
        for arm, model in models.items()
    }
    groups = {boundary.transaction_group for boundary in boundary_models.values()}
    if len(groups) != 1 or None in groups:
        raise AssertionError("LiftTray 真实边界未形成唯一联合事务组")
    group = next(iter(groups))

    cloned = {arm: copy.deepcopy(updater) for arm, updater in updaters.items()}
    real_hold_beliefs: list[dict[str, Any]] = []
    for hold_cycle in range(1, hold_cycles + 1):
        tick = boundary_tick + hold_cycle
        real_hold_beliefs.append(
            {
                arm: cloned[arm].update(
                    _runtime_observation(tick, sample, sample),
                    executed_reference_state=sample.state_id,
                    mode_by_skill=modes[arm],
                )
                for arm, sample in frozen.items()
            }
        )

    controllers = {
        arm: ClosedLoopExecutionController(model) for arm, model in models.items()
    }
    source_states = {
        arm: model.skill_states[source_skill][-1] for arm, model in models.items()
    }
    target_states = {
        arm: model.skill_states[target_skill][0] for arm, model in models.items()
    }
    for arm, case in cases_by_arm.items():
        observation = DynaMACObservation(frozen[arm].ee_pose, frozen[arm].frames)
        case.policy.reset(observation, mode_strategy="map")
        controllers[arm].reset(source_states[arm])
    boundary_controller = MultiArmBoundaryController(
        models, controllers, runtime_config
    )
    delayed_state = models[delayed_arm].skill_states[source_skill][0]
    rows: list[dict[str, Any]] = []

    for hold_cycle, real_beliefs in enumerate(real_hold_beliefs, start=1):
        beliefs = dict(real_beliefs)
        if hold_cycle <= delayed_cycles:
            beliefs[delayed_arm] = _delayed_progress_belief(
                real_beliefs[delayed_arm], delayed_state
            )
        ticks = {belief.tick for belief in beliefs.values()}
        shared_snapshot = len(ticks) == 1
        references_before = {
            arm: controller.cursor.reference_state
            for arm, controller in controllers.items()
        }
        execution_results = {}
        for arm, controller in controllers.items():
            execution_results[arm] = controller.update(
                beliefs[arm],
                DynaMACObservation(frozen[arm].ee_pose, frozen[arm].frames),
                mode_by_skill=modes[arm],
                successor_ready=False,
            )
        cycle = boundary_controller.update(
            beliefs,
            mode_by_arm_skill=modes,
        )
        if cycle.transaction is None:
            raise AssertionError("真实联合边界周期没有生成事务结果")
        transaction = cycle.transaction
        committed_arms = {request.arm_id for request in transaction.committed}
        references_after = {
            arm: controller.cursor.reference_state
            for arm, controller in controllers.items()
        }
        expected_joint_commit = hold_cycle == hold_cycles
        if hold_cycle < hold_cycles:
            atomic_invariant = (
                not committed_arms
                and references_after == references_before == source_states
                and transaction.held_transaction_groups == (group,)
            )
        else:
            atomic_invariant = (
                committed_arms == set(models)
                and references_before == source_states
                and references_after == target_states
                and not transaction.held_transaction_groups
            )
        if not shared_snapshot or not atomic_invariant:
            raise AssertionError(
                f"示范 {demonstration} 周期 {hold_cycle} 违反同快照原子事务语义"
            )
        if expected_joint_commit != bool(committed_arms):
            raise AssertionError("联合提交时机与正式连续确认周期不一致")

        row: dict[str, Any] = {
            "demonstration": demonstration,
            "hold_cycle": hold_cycle,
            "tick": cycle.tick,
            "transaction_group": group,
            "controlled_progress_delay": int(hold_cycle <= delayed_cycles),
            "shared_pre_action_snapshot": int(shared_snapshot),
            "held_transaction_group": "|".join(transaction.held_transaction_groups),
            "committed_arms": "|".join(sorted(committed_arms)),
            "partial_commit": int(
                bool(committed_arms) and committed_arms != set(models)
            ),
            "atomic_invariant": int(atomic_invariant),
        }
        for arm in sorted(models):
            request = cycle.requests[arm]
            local = cycle.local_completion[arm]
            execution = execution_results[arm]
            row.update(
                {
                    f"{arm}_progress_state": _state_text(
                        beliefs[arm].progress.estimated_state
                    ),
                    f"{arm}_local_score": local.score,
                    f"{arm}_local_streak": local.consecutive_cycles,
                    f"{arm}_local_done": int(local.done),
                    f"{arm}_permitted": int(request.permitted),
                    f"{arm}_execution_decision": execution.decision.value,
                    f"{arm}_reference_before": _state_text(references_before[arm]),
                    f"{arm}_reference_after": _state_text(references_after[arm]),
                }
            )
        rows.append(row)

    one_arm_ready = rows[delayed_cycles - 1]
    final = rows[-1]
    summary = {
        "demonstration": demonstration,
        "transaction_group": group,
        "leading_arm": leading_arm,
        "delayed_arm": delayed_arm,
        "leading_arm_ready_while_delayed": int(
            bool(one_arm_ready[f"{leading_arm}_permitted"])
            and not bool(one_arm_ready[f"{delayed_arm}_permitted"])
        ),
        "both_cursors_held_when_one_arm_ready": int(
            one_arm_ready[f"{leading_arm}_reference_after"]
            == _state_text(source_states[leading_arm])
            and one_arm_ready[f"{delayed_arm}_reference_after"]
            == _state_text(source_states[delayed_arm])
        ),
        "partial_commit_count": sum(int(row["partial_commit"]) for row in rows),
        "joint_commit_cycle": int(final["hold_cycle"]),
        "joint_commit_tick": int(final["tick"]),
        "joint_committed_arms": str(final["committed_arms"]),
        "both_targets_reached": int(
            all(
                final[f"{arm}_reference_after"] == _state_text(target_states[arm])
                for arm in models
            )
        ),
        "accepted": int(
            bool(one_arm_ready[f"{leading_arm}_permitted"])
            and not bool(one_arm_ready[f"{delayed_arm}_permitted"])
            and bool(one_arm_ready["held_transaction_group"])
            and not bool(one_arm_ready["committed_arms"])
            and not any(int(row["partial_commit"]) for row in rows)
            and set(str(final["committed_arms"]).split("|")) == set(models)
            and all(
                final[f"{arm}_reference_after"] == _state_text(target_states[arm])
                for arm in models
            )
        ),
    }
    if not summary["accepted"]:
        raise AssertionError(f"示范 {demonstration} 联合事务集成验证未通过")
    return rows, summary


def _report(config: Mapping[str, Any], summaries: Sequence[Mapping[str, Any]]) -> str:
    accepted = sum(int(row["accepted"]) for row in summaries)
    no_partial = sum(int(row["partial_commit_count"]) == 0 for row in summaries)
    lines = [
        "# 阶段四联合事务集成验证结果",
        "",
        f"- 任务：`{config['task']}`",
        f"- 正常示范：{len(summaries)} 条",
        f"- 真实联合事务端到端通过：{accepted}/{len(summaries)}",
        f"- 一臂先就绪时双方均保持：{sum(int(row['both_cursors_held_when_one_arm_ready']) for row in summaries)}/{len(summaries)}",
        f"- 部分提交为零：{no_partial}/{len(summaries)}",
        f"- 双臂同 tick 原子提交到目标技能：{sum(int(row['both_targets_reached']) for row in summaries)}/{len(summaries)}",
        "",
        "| 示范 | 一臂先就绪 | 双方保持 | 部分提交 | 联合提交周期 | 联合提交 tick | 结果 |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summaries:
        lines.append(
            "| {demo} | {leading} | {held} | {partial} | {cycle} | {tick} | {result} |".format(
                demo=row["demonstration"],
                leading=row["leading_arm_ready_while_delayed"],
                held=row["both_cursors_held_when_one_arm_ready"],
                partial=row["partial_commit_count"],
                cycle=row["joint_commit_cycle"],
                tick=row["joint_commit_tick"],
                result="通过" if row["accepted"] else "失败",
            )
        )
    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            "本验证证明真实 LiftTray 模型与正式阶段四参数能够贯通同快照联合入口守卫和原子游标提交。前两个周期对右臂进度后验施加的是受控异步延迟，用于验证一臂先就绪分支；它不是新的训练数据，也不代表完整 RLBench 在线动作执行成功率。",
            "",
        ]
    )
    return "\n".join(lines)


def run(config_path: Path, output: Path) -> None:
    config = _read_config(config_path)
    if output.exists():
        raise FileExistsError(f"输出目录已存在：{output}")
    output.mkdir(parents=True)
    runtime_path = REPOSITORY_ROOT / str(config["runtime_config"])
    runtime_config = BoundaryRuntimeConfig.from_json(runtime_path)
    belief_config = BeliefUpdaterConfig.from_json(BELIEF_CONFIG_PATH)
    demonstrations = tuple(int(item) for item in config["demonstration_indices"])
    cases = _load_cases(str(config["task"]), max(demonstrations) + 1)
    cases_by_arm = {case.arm: case for case in cases}
    expected_arms = {str(config["leading_arm"]), str(config["delayed_arm"])}
    if set(cases_by_arm) != expected_arms:
        raise RuntimeError("配置机械臂与真实 LiftTray 模型不一致")

    trace_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for demonstration in demonstrations:
        print(
            f"[phase4-transaction] replay demonstration {demonstration}",
            flush=True,
        )
        rows, summary = _validate_one(
            config=config,
            cases_by_arm=cases_by_arm,
            runtime_config=runtime_config,
            belief_config=belief_config,
            demonstration=demonstration,
        )
        trace_rows.extend(rows)
        summaries.append(summary)

    _write_csv(output / "transaction_cycles.csv", trace_rows)
    _write_csv(output / "transaction_trials.csv", summaries)
    (output / "summary.json").write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "task": config["task"],
                "trials": len(summaries),
                "accepted": sum(int(row["accepted"]) for row in summaries),
                "partial_commits": sum(
                    int(row["partial_commit_count"]) for row in summaries
                ),
                "all_accepted": all(bool(row["accepted"]) for row in summaries),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    report = _report(config, summaries)
    (output / "RESULTS.md").write_text(report, encoding="utf-8")
    shutil.copy2(config_path, output / "config.json")
    shutil.copy2(runtime_path, output / "runtime_config.json")
    (output / "environment.json").write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "python": sys.version,
                "platform": platform.platform(),
                "git_commit": _git_value("rev-parse", "HEAD"),
                "git_branch": _git_value("branch", "--show-current"),
                "git_status": _git_value("status", "--short"),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    files = sorted(path for path in output.iterdir() if path.name != "SHA256SUMS")
    (output / "SHA256SUMS").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in files),
        encoding="utf-8",
    )
    print(report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    run(arguments.config.resolve(), arguments.output.resolve())


if __name__ == "__main__":
    main()
