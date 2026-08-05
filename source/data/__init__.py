"""论文自身演示数据的加载入口。

RoboDojo 的任务、资产、GUI 和上游 policy 适配统一位于顶层
``robodojo_adapter``；本模块只保留项目自己的 NPZ 演示包格式。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from essay2608.policy.dynamac import DynaMACDemonstration


@dataclass(frozen=True)
class DemonstrationBundle:
    """仓库随附的五条单臂与五条真实接触双臂演示。"""

    single_arm: tuple[DynaMACDemonstration, ...]
    left_arm: tuple[DynaMACDemonstration, ...]
    right_arm: tuple[DynaMACDemonstration, ...]
    metadata: dict


def _required(archive: np.lib.npyio.NpzFile, keys: set[str]) -> None:
    missing = keys.difference(archive.files)
    if missing:
        raise ValueError(f"打包演示缺少字段：{sorted(missing)}")


def load_demonstrations(path: str | Path) -> DemonstrationBundle:
    """读取无 pickle、单文件的 DynaMAC 演示包。"""

    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        _required(archive, {"metadata_json", "single_count", "bimanual_count"})
        metadata = json.loads(str(archive["metadata_json"].item()))
        if metadata.get("schema") != "essay2608.dynamac.demonstrations.v1":
            raise ValueError("不支持的演示包 schema")
        single_count = int(archive["single_count"].item())
        bimanual_count = int(archive["bimanual_count"].item())
        single = []
        left = []
        right = []
        for index in range(single_count):
            prefix = f"single_{index}_"
            keys = {
                prefix + name
                for name in ("ee_pose", "action", "object_pose", "target_pose", "skill")
            }
            _required(archive, keys)
            action = archive[prefix + "action"].copy()
            single.append(
                DynaMACDemonstration(
                    ee_pose=archive[prefix + "ee_pose"].copy(),
                    action_pose=action[:, :7],
                    gripper=action[:, 7:8],
                    frames={
                        "object": archive[prefix + "object_pose"].copy(),
                        "target": archive[prefix + "target_pose"].copy(),
                    },
                    skill=archive[prefix + "skill"].copy(),
                    name=f"single_{index:03d}",
                )
            )
        for index in range(bimanual_count):
            prefix = f"bimanual_{index}_"
            keys = {
                prefix + name
                for name in (
                    "left_ee_pose",
                    "right_ee_pose",
                    "object_pose",
                    "target_pose",
                    "action",
                    "skill",
                )
            }
            _required(archive, keys)
            left_pose = archive[prefix + "left_ee_pose"].copy()
            right_pose = archive[prefix + "right_ee_pose"].copy()
            object_pose = archive[prefix + "object_pose"].copy()
            target_pose = archive[prefix + "target_pose"].copy()
            action = archive[prefix + "action"].copy()
            skill = archive[prefix + "skill"].copy()
            left.append(
                DynaMACDemonstration(
                    ee_pose=left_pose,
                    action_pose=action[:, :7],
                    gripper=action[:, 7:8],
                    frames={
                        "object": object_pose,
                        "target": target_pose,
                        "right_ee": right_pose,
                    },
                    skill=skill,
                    name=f"bimanual_left_{index:03d}",
                )
            )
            right.append(
                DynaMACDemonstration(
                    ee_pose=right_pose,
                    action_pose=action[:, 8:15],
                    gripper=action[:, 15:16],
                    frames={
                        "object": object_pose,
                        "target": target_pose,
                        "left_ee": left_pose,
                    },
                    skill=skill,
                    name=f"bimanual_right_{index:03d}",
                )
            )
    return DemonstrationBundle(tuple(single), tuple(left), tuple(right), metadata)


__all__ = [
    "DemonstrationBundle",
    "load_demonstrations",
]
