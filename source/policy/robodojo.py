"""将本项目策略桥接到 RoboDojo WebSocket 评测协议。"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Literal

import numpy as np

from .diffusion_policy import DiffusionPolicy
from .dynamac import BimanualDynaMAC, DynaMAC, DynaMACAction, DynaMACObservation
from .midigap import TaskParameterizedMiDiGaP

PolicyName = Literal["dp", "midigap", "dynamac"]
ArmMode = Literal["single", "bimanual"]


def _pose(value, field: str) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64).reshape(-1)
    if pose.shape != (7,) or not np.all(np.isfinite(pose)):
        raise ValueError(f"RoboDojo 位姿字段 {field} 必须是有限的 xyz+wxyz 七维向量")
    return pose


def _gripper(value, field: str) -> np.ndarray:
    gripper = np.asarray(value, dtype=np.float64).reshape(-1)
    if gripper.shape != (1,) or not np.all(np.isfinite(gripper)):
        raise ValueError(f"RoboDojo 夹爪字段 {field} 必须是一维有限向量")
    return gripper


def _checkpoint_file(checkpoint: Path, side: str | None = None) -> Path:
    path = checkpoint if side is None else checkpoint / f"{side}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"策略 checkpoint 不存在：{path}")
    return path


def _load_single(name: PolicyName, checkpoint: Path):
    if name == "dp":
        return DiffusionPolicy.load(checkpoint)
    if name == "midigap":
        return TaskParameterizedMiDiGaP.load(checkpoint)
    if name == "dynamac":
        return DynaMAC.load(checkpoint)
    raise ValueError(f"未知策略：{name}")


class RoboDojoPolicyModel:
    """XPolicyLab 模型接口；只消费真值状态和仿真器注入的任务物体位姿。"""

    def __init__(
        self,
        policy_name: PolicyName,
        arm_mode: ArmMode,
        checkpoint: str | Path,
    ) -> None:
        self.policy_name = policy_name
        self.arm_mode = arm_mode
        self.checkpoint = Path(checkpoint).resolve()
        if arm_mode == "single":
            self.policy = _load_single(policy_name, _checkpoint_file(self.checkpoint))
            self.left_policy = None
            self.right_policy = None
        elif arm_mode == "bimanual":
            self.policy = None
            self.left_policy = _load_single(policy_name, _checkpoint_file(self.checkpoint, "left"))
            self.right_policy = _load_single(
                policy_name, _checkpoint_file(self.checkpoint, "right")
            )
            if policy_name in {"midigap", "dynamac"}:
                self.policy = BimanualDynaMAC(self.left_policy, self.right_policy)
        else:
            raise ValueError(f"未知机械臂模式：{arm_mode}")
        self._observation: dict | None = None
        self._needs_policy_reset = True
        self._last_action: dict | None = None

    @staticmethod
    def _task_frames(observation: dict) -> dict[str, np.ndarray]:
        task = observation.get("task")
        if not isinstance(task, dict) or task.get("source") not in {
            "robodojo_simulator_ground_truth",
            "rgbd_pose_estimator",
        }:
            raise ValueError("观测缺少项目适配层注入的 RoboDojo 任务位姿")
        values = task.get("object_poses")
        if not isinstance(values, dict):
            raise ValueError("观测 task.object_poses 不是字典")
        return {name: _pose(value, f"task.object_poses.{name}") for name, value in values.items()}

    @staticmethod
    def _policy_observation(
        policy,
        ee_pose: np.ndarray,
        all_frames: dict[str, np.ndarray],
    ) -> DynaMACObservation:
        missing = sorted(set(policy.frame_names).difference(all_frames))
        if missing:
            raise ValueError(f"checkpoint 所需任务参数未出现在 RoboDojo 观测中：{missing}")
        return DynaMACObservation(
            ee_pose=ee_pose,
            frames={name: all_frames[name] for name in policy.frame_names},
        )

    def update_obs(self, observation: dict) -> None:
        self._observation = observation

    def update_obs_batch(self, observations: list[dict]) -> None:
        if len(observations) != 1:
            raise ValueError("论文协议固定 GUI 单环境评测，不接受批量环境")
        self.update_obs(observations[0])

    def reset(self) -> None:
        # RoboDojo 在首个观测之前调用 reset；底层策略需要首帧才能冻结虚拟任务参数。
        self._observation = None
        self._needs_policy_reset = True
        self._last_action = None

    def _single_action(self, observation: dict) -> dict:
        state = observation["state"]
        ee_pose = _pose(state["ee_pose"], "state.ee_pose")
        frames = self._task_frames(observation)
        policy_observation = self._policy_observation(self.policy, ee_pose, frames)
        if self._needs_policy_reset:
            self.policy.reset(policy_observation)
            self._needs_policy_reset = False
        if self.policy.complete:
            assert self._last_action is not None
            return self._last_action
        result: DynaMACAction = self.policy.act(policy_observation)
        action = {
            "arm_ee_pose": result.pose.astype(np.float32),
            "ee_joint_state": result.gripper.astype(np.float32),
        }
        self._last_action = action
        return action

    def _bimanual_action(self, observation: dict) -> dict:
        state = observation["state"]
        left_pose = _pose(state["left_ee_pose"], "state.left_ee_pose")
        right_pose = _pose(state["right_ee_pose"], "state.right_ee_pose")
        frames = self._task_frames(observation)
        left_frames = {**frames, "right_ee": right_pose}
        right_frames = {**frames, "left_ee": left_pose}
        left_observation = self._policy_observation(self.left_policy, left_pose, left_frames)
        right_observation = self._policy_observation(self.right_policy, right_pose, right_frames)
        if self._needs_policy_reset:
            if self.policy_name in {"midigap", "dynamac"}:
                self.policy.reset(left_observation, right_observation)
            else:
                self.left_policy.reset(left_observation)
                self.right_policy.reset(right_observation)
            self._needs_policy_reset = False

        complete = self.left_policy.complete and self.right_policy.complete
        if complete:
            assert self._last_action is not None
            return self._last_action
        if self.policy_name in {"midigap", "dynamac"}:
            if self.left_policy.complete != self.right_policy.complete:
                raise RuntimeError("并发 MiDiGaP/DynaMAC 左右臂技能时长不同步")
            result = self.policy.act(left_observation, right_observation)
            left_action, right_action = result.left, result.right
            left_pose_value = left_action.pose
            left_gripper_value = left_action.gripper
            right_pose_value = right_action.pose
            right_gripper_value = right_action.gripper
        else:
            if self.left_policy.complete:
                assert self._last_action is not None
                left_pose_value = self._last_action["left_ee_pose"]
                left_gripper_value = self._last_action["left_ee_joint_state"]
            else:
                left_action = self.left_policy.act(left_observation)
                left_pose_value = left_action.pose
                left_gripper_value = left_action.gripper
            if self.right_policy.complete:
                assert self._last_action is not None
                right_pose_value = self._last_action["right_ee_pose"]
                right_gripper_value = self._last_action["right_ee_joint_state"]
            else:
                right_action = self.right_policy.act(right_observation)
                right_pose_value = right_action.pose
                right_gripper_value = right_action.gripper
        action = {
            "left_ee_pose": np.asarray(left_pose_value, dtype=np.float32),
            "left_ee_joint_state": np.asarray(left_gripper_value, dtype=np.float32),
            "right_ee_pose": np.asarray(right_pose_value, dtype=np.float32),
            "right_ee_joint_state": np.asarray(right_gripper_value, dtype=np.float32),
        }
        self._last_action = action
        return action

    def get_action(self) -> list[dict]:
        if self._observation is None:
            raise RuntimeError("get_action() 在 update_obs() 之前被调用")
        action = (
            self._single_action(self._observation)
            if self.arm_mode == "single"
            else self._bimanual_action(self._observation)
        )
        return [action]

    def get_action_batch(self, env_idx_list=None) -> list[list[dict]]:
        del env_idx_list
        return [self.get_action()]


class RoboDojoReplayCaptureModel:
    """在 GUI 中回放官方 EE 动作，同时以 JSONL 补采任务帧真值。"""

    def __init__(
        self,
        episode_path: str | Path,
        output_path: str | Path,
        arm_mode: ArmMode,
        active_side: Literal["auto", "left", "right"] = "auto",
    ) -> None:
        try:
            import h5py
        except ImportError as error:
            raise RuntimeError("回放官方演示需要 h5py") from error
        self.episode_path = Path(episode_path).resolve()
        self.output_path = Path(output_path).resolve()
        self.arm_mode = arm_mode
        with h5py.File(self.episode_path, "r") as archive:
            self.actions = {
                side: {
                    "joint": archive[f"action/{side}_arm_joint_states"][()].astype(
                        np.float64
                    ),
                    "pose": archive[f"action/{side}_ee_poses"][()].astype(np.float64),
                    "gripper": archive[f"action/{side}_ee_joint_states"][()].astype(np.float64),
                }
                for side in ("left", "right")
            }
            self.instruction = archive["instruction"][()].decode("utf-8")
            self.frequency = int(archive["additional_info/frequency"][()])
        lengths = {side: len(values["pose"]) for side, values in self.actions.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"左右臂回放长度不一致：{lengths}")
        if active_side == "auto":
            travel = {
                side: float(np.linalg.norm(np.diff(values["pose"][:, :3], axis=0), axis=1).sum())
                for side, values in self.actions.items()
            }
            active_side = max(travel, key=travel.get)
        self.active_side = active_side
        self.length = next(iter(lengths.values()))
        self._index = 0
        self._observation: dict | None = None

    def reset(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema": "essay2608.robodojo.gui_capture.v1",
            "episode": str(self.episode_path),
            "instruction": self.instruction,
            "frequency": self.frequency,
            "arm_mode": self.arm_mode,
            "active_side": self.active_side,
            "steps": self.length,
            "replay_action_type": "joint",
        }
        self.output_path.write_text(
            json.dumps({"type": "metadata", **metadata}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        self._index = 0
        self._observation = None

    def update_obs(self, observation: dict) -> None:
        self._observation = observation

    def update_obs_batch(self, observations: list[dict]) -> None:
        if len(observations) != 1:
            raise ValueError("GUI 补采固定为单环境")
        self.update_obs(observations[0])

    def _record(self, action: dict) -> None:
        assert self._observation is not None
        state = self._observation["state"]
        task = self._observation.get("task", {})
        state_poses = {
            key: np.asarray(value).tolist()
            for key, value in state.items()
            if key.endswith("ee_pose")
        }
        frames = {
            name: np.asarray(value).tolist() for name, value in task.get("object_poses", {}).items()
        }
        payload = {
            "type": "step",
            "index": self._index,
            "state_ee_poses": state_poses,
            "frames": frames,
            "action": {name: np.asarray(value).tolist() for name, value in action.items()},
        }
        with self.output_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def get_action(self) -> list[dict]:
        if self._observation is None:
            raise RuntimeError("回放 get_action() 在 update_obs() 之前被调用")
        step = min(self._index, self.length - 1)
        if self.arm_mode == "single":
            values = self.actions[self.active_side]
            action = {
                "arm_joint_state": values["joint"][step].astype(np.float32),
                "ee_joint_state": values["gripper"][step].astype(np.float32),
            }
        else:
            action = {}
            for side in ("left", "right"):
                values = self.actions[side]
                action[f"{side}_arm_joint_state"] = values["joint"][step].astype(np.float32)
                action[f"{side}_ee_joint_state"] = values["gripper"][step].astype(np.float32)
        if self._index < self.length:
            self._record(action)
        self._index += 1
        return [action]

    def get_action_batch(self, env_idx_list=None) -> list[list[dict]]:
        del env_idx_list
        return [self.get_action()]


def serve_robodojo_policy(
    policy_name: PolicyName,
    arm_mode: ArmMode,
    checkpoint: str | Path,
    host: str = "127.0.0.1",
    port: int = 19000,
    xpolicylab_root: str | Path | None = None,
) -> None:
    """以前台进程启动兼容 RoboDojo 的策略服务器。"""

    if xpolicylab_root is None:
        project_root = Path(__file__).resolve().parents[2]
        xpolicylab_root = project_root / "third_party" / "RoboDojo" / "XPolicyLab"
    xpolicylab_root = Path(xpolicylab_root).resolve()
    if str(xpolicylab_root) not in sys.path:
        sys.path.insert(0, str(xpolicylab_root))
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig

    model = RoboDojoPolicyModel(policy_name, arm_mode, checkpoint)
    server = PolicyServer(
        model,
        PolicyServerConfig(
            host=host,
            port=port,
            ws_ping_interval_s=None,
            ws_ping_timeout_s=None,
        ),
    )
    print(
        f"[策略服务器] policy={policy_name} arm_mode={arm_mode} "
        f"checkpoint={Path(checkpoint).resolve()} ws://{host}:{port}",
        flush=True,
    )
    asyncio.run(server.serve_forever())


def serve_robodojo_replay_capture(
    episode_path: str | Path,
    output_path: str | Path,
    arm_mode: ArmMode,
    active_side: Literal["auto", "left", "right"] = "auto",
    host: str = "127.0.0.1",
    port: int = 19000,
    xpolicylab_root: str | Path | None = None,
) -> None:
    """启动官方动作 GUI 回放与真值补采服务器。"""

    if xpolicylab_root is None:
        project_root = Path(__file__).resolve().parents[2]
        xpolicylab_root = project_root / "third_party" / "RoboDojo" / "XPolicyLab"
    xpolicylab_root = Path(xpolicylab_root).resolve()
    if str(xpolicylab_root) not in sys.path:
        sys.path.insert(0, str(xpolicylab_root))
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig

    model = RoboDojoReplayCaptureModel(episode_path, output_path, arm_mode, active_side)
    server = PolicyServer(
        model,
        PolicyServerConfig(
            host=host,
            port=port,
            ws_ping_interval_s=None,
            ws_ping_timeout_s=None,
        ),
    )
    print(
        f"[GUI 补采服务器] episode={Path(episode_path).resolve()} "
        f"output={Path(output_path).resolve()} ws://{host}:{port}",
        flush=True,
    )
    asyncio.run(server.serve_forever())


__all__ = [
    "RoboDojoPolicyModel",
    "RoboDojoReplayCaptureModel",
    "serve_robodojo_policy",
    "serve_robodojo_replay_capture",
]
