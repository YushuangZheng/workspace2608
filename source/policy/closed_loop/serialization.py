"""Save and load closed-loop sidecars without duplicating DynaMAC arrays."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ..dynamac import DynaMAC
from .config import ClosedLoopPolicyConfig
from .task_model import ClosedLoopTaskModel

BUNDLE_SCHEMA = "essay2608.closed_loop_policy_bundle.v1"


def save_policy_bundle(
    directory: str | Path,
    *,
    task_models: Mapping[str, ClosedLoopTaskModel],
    config: ClosedLoopPolicyConfig,
    summary: Mapping[str, Any],
) -> Path:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    arms = sorted(task_models)
    for arm in arms:
        task_models[arm].save(root / f"{arm}.npz")
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "arms": arms,
        "base_policy_fingerprints": {
            arm: task_models[arm].base_policy.fingerprint() for arm in arms
        },
        "config": config.to_dict(),
        "summary": dict(summary),
    }
    path = root / "policy.json"
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def load_policy_bundle(
    directory: str | Path,
    *,
    base_policies: Mapping[str, DynaMAC],
) -> tuple[dict[str, ClosedLoopTaskModel], ClosedLoopPolicyConfig, dict[str, Any]]:
    root = Path(directory)
    manifest = json.loads((root / "policy.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema") != BUNDLE_SCHEMA:
        raise ValueError("不支持的阶段六策略 bundle schema")
    arms = tuple(manifest.get("arms", ()))
    if set(arms) != set(base_policies):
        raise ValueError("阶段六策略 bundle 与基础策略机械臂集合不一致")
    expected = manifest.get("base_policy_fingerprints", {})
    if any(base_policies[arm].fingerprint() != expected.get(arm) for arm in arms):
        raise ValueError("阶段六策略 bundle 绑定了不同的 DynaMAC checkpoint")
    models = {
        arm: ClosedLoopTaskModel.load(root / f"{arm}.npz", base_policies[arm])
        for arm in arms
    }
    config = ClosedLoopPolicyConfig.from_mapping(manifest["config"])
    summary = manifest.get("summary", {})
    if not isinstance(summary, dict):
        raise TypeError("阶段六策略 bundle summary 必须为对象")
    return models, config, summary


__all__ = ["BUNDLE_SCHEMA", "load_policy_bundle", "save_policy_bundle"]
