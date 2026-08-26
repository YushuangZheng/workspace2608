"""Audit phase-five recovery metadata against all frozen V4 task models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from essay2608.policy.closed_loop import (
    EpisodeLinkAnchorRegistry,
    RelationDecision,
    RelationGoalKind,
    RelationGoalPlanner,
    RelationRecoveryIntent,
    UnlinkMetadataRepository,
)
from evaluations.phase23_component_ab.run import REPOSITORY_ROOT, _load_cases

SCHEMA = "essay2608-phase5-recovery-acceptance-config-v1"


def _read_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema",
        "tasks",
        "demonstration_count",
        "covariance_inflation",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("阶段五验收配置字段不完整或包含未知字段")
    if value["schema"] != SCHEMA:
        raise ValueError("阶段五验收配置 schema 不匹配")
    if not value["tasks"] or int(value["demonstration_count"]) < 2:
        raise ValueError("阶段五验收任务或示范数量无效")
    if float(value["covariance_inflation"]) < 0.0:
        raise ValueError("阶段五验收协方差放宽量不能为负")
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("阶段五验收结果不能为空")
    names: list[str] = []
    for row in rows:
        for name in row:
            if name not in names:
                names.append(name)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=names, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audit_case(case: Any, covariance_inflation: float) -> dict[str, Any]:
    model = case.model
    registry = EpisodeLinkAnchorRegistry(model)
    unlink_repository = UnlinkMetadataRepository(model)
    planner = RelationGoalPlanner(registry, unlink_repository)

    resolved_origins = 0
    for key, event_id in model.link_origins.items():
        anchor = registry.resolve(key.frame_id, key.state_id, key.mode)
        if anchor.origin_event_id != event_id or anchor.source != "offline_link":
            raise RuntimeError(f"{case.key} 的正式 link_origin 解析错误")
        resolved_origins += 1

    link_goals = 0
    instantiated_waypoints = 0
    identity = np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    for event_id, offline in model.link_anchors.items():
        source = offline.linked_entry_states[0]
        goal = planner.plan(
            (
                RelationRecoveryIntent(
                    model.arm_id,
                    event_id.frame_id,
                    RelationDecision.LINKED,
                    RelationDecision.EXTERNAL,
                ),
            ),
            source_state=source,
            mode=event_id.mode,
        )[0]
        if (
            goal.kind != RelationGoalKind.LINK
            or goal.link_anchor is None
            or goal.link_anchor.origin_event_id != event_id
        ):
            raise RuntimeError(f"{case.key} 的 LINK 目标未解析到正确事件锚点")
        waypoints = registry.instantiate(
            goal.link_anchor,
            identity,
            covariance_inflation,
        )
        if len(waypoints) != len(offline.local_means):
            raise RuntimeError(f"{case.key} 的 LINK 锚点实例化长度错误")
        link_goals += 1
        instantiated_waypoints += len(waypoints)

    pending_goals = 0
    for event_id, candidate in model.link_pending_events.items():
        if event_id in model.link_origins.values():
            raise RuntimeError(f"{case.key} 的 Pending 被错误传播为离线 link_origin")
        registry.reset()
        runtime = registry.activate_pending(event_id)
        goal = planner.plan(
            (
                RelationRecoveryIntent(
                    model.arm_id,
                    candidate.frame_id,
                    RelationDecision.LINKED,
                    RelationDecision.EXTERNAL,
                ),
            ),
            source_state=candidate.candidate_state,
            mode=event_id.mode,
        )[0]
        if (
            runtime.source != "verified_pending"
            or goal.link_anchor is None
            or goal.link_anchor.origin_event_id != event_id
        ):
            raise RuntimeError(f"{case.key} 的 Pending episode 激活错误")
        pending_goals += 1
    registry.reset()

    unlink_goals = 0
    legal_reentry_states = 0
    for event_id, metadata in model.unlink_events.items():
        resolved = unlink_repository.resolve(
            event_id.frame_id,
            metadata.release_state,
            event_id.mode,
        )
        if resolved.event_id != event_id:
            raise RuntimeError(f"{case.key} 的 UNLINK 元数据解析错误")
        goal = planner.plan(
            (
                RelationRecoveryIntent(
                    model.arm_id,
                    event_id.frame_id,
                    RelationDecision.EXTERNAL,
                    RelationDecision.LINKED,
                ),
            ),
            source_state=metadata.release_state,
            mode=event_id.mode,
        )[0]
        if goal.kind != RelationGoalKind.UNLINK or goal.unlink_metadata is None:
            raise RuntimeError(f"{case.key} 的 UNLINK 目标构建错误")
        unknown = set(goal.legal_reentry_states).difference(model.states)
        if unknown or not goal.legal_reentry_states:
            raise RuntimeError(f"{case.key} 的 UNLINK 合法重入状态无效")
        unlink_goals += 1
        legal_reentry_states += len(goal.legal_reentry_states)

    return {
        "task": case.task,
        "arm": case.arm,
        "formal_link_anchors": len(model.link_anchors),
        "pending_candidates": len(model.link_pending_events),
        "unlink_events": len(model.unlink_events),
        "resolved_link_origins": resolved_origins,
        "constructed_link_goals": link_goals,
        "activated_pending_goals": pending_goals,
        "constructed_unlink_goals": unlink_goals,
        "instantiated_link_waypoints": instantiated_waypoints,
        "unlink_legal_reentry_states": legal_reentry_states,
        "accepted": 1,
    }


def run(config_path: Path, output: Path) -> None:
    config = _read_config(config_path)
    if output.exists():
        raise FileExistsError(f"输出目录已存在：{output}")
    output.mkdir(parents=True)
    rows = []
    for task in config["tasks"]:
        print(f"[phase5-acceptance] loading {task}", flush=True)
        for case in _load_cases(task, int(config["demonstration_count"])):
            rows.append(_audit_case(case, float(config["covariance_inflation"])))
    if any(not row["accepted"] for row in rows):
        raise RuntimeError("至少一个阶段五真实模型元数据审计失败")

    (output / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(output / "metadata_audit.csv", rows)
    environment = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "dirty_worktree_expected": True,
    }
    (output / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    totals = {
        key: sum(int(row[key]) for row in rows)
        for key in (
            "formal_link_anchors",
            "pending_candidates",
            "unlink_events",
            "resolved_link_origins",
            "constructed_link_goals",
            "activated_pending_goals",
            "constructed_unlink_goals",
            "instantiated_link_waypoints",
            "unlink_legal_reentry_states",
        )
    }
    report = [
        "# 阶段五真实 V4 恢复元数据验收",
        "",
        f"- 任务/机械臂模型：{len(rows)}",
        f"- 正式 LINK 锚点：{totals['formal_link_anchors']}",
        f"- LINK_PENDING 候选：{totals['pending_candidates']}",
        f"- UNLINK 事件：{totals['unlink_events']}",
        f"- 已解析状态级 link_origin：{totals['resolved_link_origins']}",
        f"- 已实例化 LINK 路点：{totals['instantiated_link_waypoints']}",
        f"- UNLINK 合法重入状态：{totals['unlink_legal_reentry_states']}",
        "- 结果：全部通过。",
        "",
        "本验收只证明阶段五能够正确消费冻结 V4 正常示范生成的事件元数据；",
        "不把它表述为 RLBench 故障恢复成功率实验，后者需要阶段六完成运行时集成后执行。",
        "",
    ]
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    files = sorted(path for path in output.iterdir() if path.name != "SHA256SUMS")
    (output / "SHA256SUMS").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in files),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.config, arguments.output)


if __name__ == "__main__":
    main()
