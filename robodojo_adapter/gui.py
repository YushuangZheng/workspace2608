"""只允许 GUI 的 RoboDojo 评测客户端，并注入真值任务参数。"""

from __future__ import annotations

import os
import runpy
import sys
import types
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
from types import MethodType

import numpy as np

from .data import ROBODOJO_RUNTIME_ROOT
from .pose import RGBDPoseEstimator, estimate_rgbd_pose

_PERTURBATION_TARGET = {
    "push_T": "target_t",
    "pour_liquid_into_cup": "cup",
    "sweep_blocks": "broom_shovel",
}

_VIEWPORT_PRESETS = {
    "push_T": ((0.82, -1.00, 1.30), (0.15, -0.08, 0.82)),
    "pour_liquid_into_cup": ((-0.82, -1.00, 1.30), (-0.15, -0.08, 0.82)),
    "sweep_blocks": ((0.98, -1.28, 1.44), (0.0, -0.06, 0.82)),
}


def _viewport_camera_pose(task_name: str):
    """返回适合坐在屏幕前观察操作的近景，而非 Isaac Lab 的 7.5 m 默认远景。"""

    return _VIEWPORT_PRESETS.get(task_name, ((0.98, -1.28, 1.44), (0.0, -0.06, 0.82)))


def _patch_viewport_camera(task_name: str) -> None:
    from env.environment.base_env import BaseEnv

    original = BaseEnv.setup_sim_cfg
    if getattr(original, "_essay2608_close_view", False):
        return
    eye, lookat = _viewport_camera_pose(task_name)

    def setup_sim_cfg(self, config):
        original(self, config)
        self.sim_cfg.viewer.eye = eye
        self.sim_cfg.viewer.lookat = lookat
        self.sim_cfg.viewer.origin_type = "env"
        self.sim_cfg.viewer.env_index = 0

    setup_sim_cfg._essay2608_close_view = True
    BaseEnv.setup_sim_cfg = setup_sim_cfg


def _install_project_deploy_module(policy_name: str) -> None:
    """给项目策略注册上游期望的通用逐步评测循环。"""

    if policy_name not in {
        "essay2608_dp",
        "essay2608_midigap",
        "essay2608_dynamac",
        "essay2608_replay_capture",
    }:
        return
    module_name = f"XPolicyLab.policy.{policy_name}.deploy"
    package_name = f"XPolicyLab.policy.{policy_name}"
    package = types.ModuleType(package_name)
    package.__path__ = []
    deploy = types.ModuleType(module_name)

    def eval_one_episode(TASK_ENV, model_client):
        model_client.call(func_name="reset")
        while not TASK_ENV.is_episode_end():
            observation = TASK_ENV.get_obs()
            model_client.call(func_name="update_obs", obs=observation)
            actions = model_client.call(func_name="get_action")
            for action_index, action in enumerate(actions):
                TASK_ENV.take_action(action)
                if TASK_ENV.is_episode_end() or action_index + 1 == len(actions):
                    break
                observation = TASK_ENV.get_obs()
                model_client.call(func_name="update_obs", obs=observation)

    deploy.eval_one_episode = eval_one_episode
    package.deploy = deploy
    sys.modules[package_name] = package
    sys.modules[module_name] = deploy


def _numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value, dtype=np.float64)


def _labels(layout, env_idx: int) -> Iterable[str]:
    seen: set[str] = set()
    for object_type in (
        "Rigid",
        "Dynamic",
        "Geometry",
        "Articulation",
        "Garment",
        "Fluid",
    ):
        records = layout.object_records_by_type.get(object_type)
        if records is None:
            continue
        for record in records.layout_records_by_env[env_idx]:
            label = record.get("label")
            if label and label not in seen:
                seen.add(label)
                yield label


def _patch_ground_truth_task_parameters(
    task_name: str,
    observation_mode: str = "oracle_pose",
) -> None:
    from env.observation_manager.obs_manager import ObsManager

    original = ObsManager.get_obs
    if getattr(original, "_essay2608_task_pose", False):
        return
    if observation_mode not in {"oracle_pose", "rgbd_pose"}:
        raise ValueError(f"未知任务位姿观测模式：{observation_mode}")
    rgbd_estimator = RGBDPoseEstimator.from_environment() if observation_mode == "rgbd_pose" else None

    def get_obs(self, env_idx_list=None):
        observations = original(self, env_idx_list=env_idx_list)
        layout = self.env.scene_manager.layout_manager
        for env_idx, observation in observations.items():
            if observation_mode == "oracle_pose":
                object_poses: dict[str, np.ndarray] = {}
                for label in _labels(layout, int(env_idx)):
                    position, orientation = layout.get_instance_pose(
                        env_idx=int(env_idx), label=label, relative=True
                    )
                    if position is None or orientation is None:
                        continue
                    object_poses[label] = np.concatenate((_numpy(position), _numpy(orientation)))
                source = "robodojo_simulator_ground_truth"
                estimator_name = "robodojo_layout_manager"
            else:
                assert rgbd_estimator is not None
                object_poses = estimate_rgbd_pose(rgbd_estimator, task_name, observation)
                source = "rgbd_pose_estimator"
                estimator_name = rgbd_estimator.name
            observation["task"] = {
                "object_poses": object_poses,
                "pose_convention": "xyz+wxyz_relative_to_environment",
                "source": source,
                "estimator": estimator_name,
            }
        return observations

    get_obs._essay2608_task_pose = True
    ObsManager.get_obs = get_obs


def _install_dynamic_perturbation(env, task_name: str, condition: str) -> None:
    if condition == "static":
        return
    if condition not in {"smooth", "teleport"}:
        raise ValueError(f"未知动态条件：{condition}")
    target_label = _PERTURBATION_TARGET[task_name]
    original = env.take_action_batch
    base_pose: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    counters: dict[int, int] = {}

    def perturb(self, actions_list, env_idx_list=None):
        indices = list(range(self.num_envs)) if env_idx_list is None else list(env_idx_list)
        layout = self.scene_manager.layout_manager
        for env_idx in indices:
            if env_idx not in base_pose or self.take_action_cnt[env_idx] == 0:
                position, orientation = layout.get_instance_pose(
                    env_idx=env_idx, label=target_label, relative=True
                )
                base_pose[env_idx] = (_numpy(position), _numpy(orientation))
                counters[env_idx] = 0
            position, orientation = base_pose[env_idx]
            step = counters[env_idx]
            offset = np.zeros(3, dtype=np.float64)
            if condition == "smooth":
                phase = min(step, 160) / 160.0
                offset[:2] = (0.12 * np.sin(np.pi * phase), 0.08 * np.sin(2.0 * np.pi * phase))
            elif step >= 80:
                offset[:2] = (0.14, -0.10)
            instance = layout.get_instance_name(env_idx, target_label)
            obj = layout.get_scene_object(env_idx, instance)
            if obj is None:
                raise RuntimeError(f"无法定位动态扰动目标：env={env_idx}, label={target_label}")
            obj.set_local_pose(translation=position + offset, orientation=orientation)
            if hasattr(obj, "set_linear_velocity"):
                obj.set_linear_velocity(np.zeros(3, dtype=np.float32))
            if hasattr(obj, "set_angular_velocity"):
                obj.set_angular_velocity(np.zeros(3, dtype=np.float32))
            counters[env_idx] += 1
        return original(actions_list, env_idx_list=env_idx_list)

    env.take_action_batch = MethodType(perturb, env)


def _patch_single_arm_ee_action_key(env) -> None:
    """兼容上游单臂校验用 ``ee_pose``、控制路径用 ``arm_ee_pose`` 的不一致。"""

    if len(env.robot_action_dim_info["arm_dim"]) != 1:
        return
    original = env.validate_action_dict

    def validate_action_dict(self, action_dict):
        normalized = dict(action_dict)
        if "arm_ee_pose" in normalized:
            normalized["ee_pose"] = normalized.pop("arm_ee_pose")
        return original(normalized)

    env.validate_action_dict = MethodType(validate_action_dict, env)


def _patch_single_arm_gripper_restore(env) -> None:
    """修复上游把单臂 ``ee_joint_state`` 错误还原为 ``arm`` 的问题。"""

    if len(env.robot_action_dim_info["arm_dim"]) != 1:
        return
    manager = env.robot_manager
    original = manager.restore_name

    def restore_name(self, processed_name):
        if processed_name == "ee_joint_state":
            return "ee"
        return original(processed_name)

    manager.restore_name = MethodType(restore_name, manager)


def run_gui_eval_client() -> None:
    if "--headless" in sys.argv:
        raise RuntimeError("本项目禁止 RoboDojo 评测使用 --headless；所有正式评测必须开 GUI")
    if not os.environ.get("DISPLAY"):
        raise RuntimeError("未检测到 DISPLAY，不能满足全程 GUI 评测协议")
    runtime_root = Path(os.environ.get("ESSAY2608_ROBODOJO_RUNTIME", ROBODOJO_RUNTIME_ROOT))
    main_path = runtime_root / "src" / "eval_client" / "main.py"
    if not main_path.is_file():
        raise RuntimeError("RoboDojo 运行覆盖层未准备，请先运行 robodojo prepare")

    namespace = runpy.run_path(str(main_path), run_name="essay2608_robodojo_upstream")
    # runpy 返回执行全局字典的副本；必须改函数实际持有的 ``__globals__``，
    # 否则替换 create_eval_env 看似成功但 main() 仍调用上游原函数。
    upstream_globals = namespace["main"].__globals__
    observation_mode = os.environ.get("ESSAY2608_OBSERVATION_MODE", "oracle_pose")
    _patch_ground_truth_task_parameters(namespace["args_cli"].task_name, observation_mode)
    _install_project_deploy_module(namespace["args_cli"].policy_name)
    original_create = upstream_globals["create_eval_env"]
    condition = os.environ.get("ESSAY2608_PERTURBATION", "static")
    task_name = namespace["args_cli"].task_name
    _patch_viewport_camera(task_name)

    def create_eval_env(*args, **kwargs):
        config = args[0] if args else kwargs["config"]
        # GUI 首次编译 shader/CuRobo 可能数分钟没有策略请求。连接在本机，禁用
        # keepalive 可避免把正常的阻塞初始化误判为网络断开。
        config.deploy_cfg["ws_ping_interval_s"] = None
        config.deploy_cfg["ws_ping_timeout_s"] = None
        env = original_create(*args, **kwargs)
        source_layout = os.environ.get("ESSAY2608_SOURCE_LAYOUT")
        if source_layout:
            import json

            layout_path = Path(source_layout).resolve()
            layout = json.loads(layout_path.read_text(encoding="utf-8"))
            env.scene_manager.layout_manager.replay = True
            env.seed_manager.get_seed_scene_info = lambda _seed: deepcopy(layout)
        if os.environ.get("ESSAY2608_PROCEDURAL_LAYOUT") == "1":
            env.scene_manager.layout_manager.replay = False
            env.seed_manager.get_seed_scene_info = lambda _seed: None
        _patch_single_arm_gripper_restore(env)
        if len(env.robot_action_dim_info["arm_dim"]) == 1:
            restored = env.robot_manager.restore_name("ee_joint_state")
            if restored != "ee":
                raise RuntimeError(f"单臂夹爪名称适配未生效：{restored}")
        _patch_single_arm_ee_action_key(env)
        _install_dynamic_perturbation(env, task_name, condition)
        return env

    upstream_globals["create_eval_env"] = create_eval_env
    namespace["main"]()


if __name__ == "__main__":
    run_gui_eval_client()
