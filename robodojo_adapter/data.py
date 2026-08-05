"""RoboDojo 上游、资产与论文实验协议适配。

RoboDojo 本体固定为 Git 子模块；大体积资产和运行期覆盖配置不进入 Git。这个模块只维护
项目需要的薄适配层，避免复制上游任务实现或修改子模块工作树。
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Literal

from .tapas import TapasSegmentationConfig, tapas_skill_labels

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROBODOJO_ROOT = PROJECT_ROOT / "third_party" / "RoboDojo"
# 官方代码、Assets 和选择性 data 统一归属于唯一的 RoboDojo 根目录；项目只在
# 该根目录的 .cache/essay2608 保存下载清单和 HF 根文件，不再创建第二个 RoboDojo。
ROBODOJO_OFFICIAL_ROOT = ROBODOJO_ROOT  # 兼容旧调用名；不再指向独立目录
ROBODOJO_ASSET_ROOT = ROBODOJO_ROOT / "Assets"
ROBODOJO_DEMO_ROOT = ROBODOJO_ROOT / "data" / "RoboDojo"
ROBODOJO_META_ROOT = ROBODOJO_ROOT / ".cache" / "essay2608"
ROBODOJO_RUNTIME_ROOT = PROJECT_ROOT / ".runtime" / "robodojo"
ROBODOJO_RESULT_ROOT = PROJECT_ROOT / "results" / "robodojo" / "raw"
ROBODOJO_CAPTURE_ROOT = PROJECT_ROOT / "results" / "robodojo" / "captures"
ROBODOJO_SOURCE_LAYOUT_ROOT = PROJECT_ROOT / "results" / "robodojo" / "source_layouts"


COMMON_ASSET_PATTERNS = (
    "Assets/Background/brown_photostudio_02_4k.hdr",
    "Assets/Material/material_0122/**",
    "Assets/Material/material_0564/**",
    "Assets/Object/RoboDojo/Geometry/camera_stand/**",
    "Assets/Robots/franka/**",
    "Assets/Robots/x5/**",
    "Assets/Room/Simple_Room_nolight/**",
    "Assets/Sensor/Camera/**",
)


TASK_ASSET_PATTERNS = {
    "push_T": (
        "Assets/Object/RoboDojo/Rigid/t/**",
        "Assets/Object/RoboDojo/Geometry/t_cushion/**",
    ),
    "pour_liquid_into_cup": (
        "Assets/Object/RoboDojo/Rigid/wuliangye/**",
        "Assets/Object/RoboDojo/Rigid/mug/**",
        "Assets/Object/RoboDojo/Rigid/goblet/**",
        "Assets/Object/RoboDojo/Fluid/wuliangye/**",
        "Assets/Material/Fluid/**",
    ),
    "sweep_blocks": (
        "Assets/Object/RoboDojo/Rigid/broom/**",
        "Assets/Object/RoboDojo/Rigid/broom_shovel/**",
        "Assets/Object/RoboDojo/Rigid/small_cube/**",
    ),
}


def task_asset_patterns(
    task_name: str,
    paths: RoboDojoPaths | None = None,
    include_layout: bool = True,
) -> tuple[str, ...]:
    """按任务配置自动推导物体资产；论文任务保留更窄的精确白名单。"""

    paths = paths or RoboDojoPaths()
    if task_name in TASK_ASSET_PATTERNS:
        patterns = list(TASK_ASSET_PATTERNS[task_name])
    else:
        task_path = paths.upstream_root / "task" / "RoboDojo" / "config" / f"{task_name}.yml"
        # ``label`` 是布局中的实例名（如 ``cube0``、``target``），不是资产
        # 目录名；这里只按 category.name 解析真实可下载的物体资产。
        object_names = _task_category_names(_yaml_load(task_path))
        patterns = [f"Assets/Object/RoboDojo/**/{name}/**" for name in object_names]
    if include_layout:
        patterns.append(f"Assets/Eval_Layout/RoboDojo/arx_x5/*/{task_name}_*.json")
    return tuple(dict.fromkeys(patterns))


@dataclass(frozen=True)
class RoboDojoTaskCandidate:
    """与 DynaMAC 论文任务对应的 RoboDojo 候选。"""

    name: str
    arm_mode: Literal["single", "bimanual"]
    paper_analogues: tuple[str, ...]
    task_frames: tuple[str, ...]
    eval_episodes: int
    rationale: str
    scene_config: str = "default"
    robot_config: str = "dual_x5"
    camera_config: str = "camera_config"
    dimension: str | None = None
    variant: str = "standard"
    arm_count: int = 2


@dataclass(frozen=True)
class RoboDojoRobotCandidate:
    """上游机器人配置及其可控目标臂数量。"""

    name: str
    path: str
    arm_count: int
    target_arm_count: int
    robot_names: tuple[str, ...]


@dataclass(frozen=True)
class RoboDojoSceneCandidate:
    """上游静态场景配置。"""

    name: str
    path: str


@dataclass(frozen=True)
class RoboDojoPolicyCandidate:
    """上游 XPolicyLab policy adapter。"""

    name: str
    path: str
    protocol: str
    has_model: bool
    has_server: bool


def _yaml_load(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("RoboDojo 配置解析需要安装 pyyaml：pip install pyyaml") from error
    if not path.is_file():
        return {}
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _task_config_names(config: dict[str, Any]) -> tuple[str, ...]:
    """从任务布局配置提取对象/目标标签，避免复制 54 个任务的元数据。"""

    names: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key in ("name", "label"):
                candidate = value.get(key)
                if isinstance(candidate, str) and candidate and candidate not in names:
                    names.append(candidate)
                elif isinstance(candidate, list):
                    for item in candidate:
                        if isinstance(item, str) and item and item not in names:
                            names.append(item)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(config)
    return tuple(names)


def _task_category_names(config: dict[str, Any]) -> tuple[str, ...]:
    """提取任务配置中 ``category`` 下的真实资产目录名。"""

    names: list[str] = []

    def walk(value: Any, in_category: bool = False) -> None:
        if isinstance(value, dict):
            category = value.get("category")
            if isinstance(category, list):
                walk(category, True)
            if in_category:
                candidate = value.get("name")
                if isinstance(candidate, str) and candidate and candidate not in names:
                    names.append(candidate)
            for key, child in value.items():
                if key != "category":
                    walk(child, in_category)
        elif isinstance(value, list):
            for child in value:
                walk(child, in_category)

    walk(config)
    return tuple(names)


def _task_defaults(paths: RoboDojoPaths) -> dict[str, dict[str, Any]]:
    common_file = paths.upstream_root / "task" / "RoboDojo" / "config" / "_task.yml"
    payload = _yaml_load(common_file)
    common = payload.get("common", {}) if isinstance(payload.get("common"), dict) else {}
    overrides = payload.get("tasks", {}) if isinstance(payload.get("tasks"), dict) else {}
    result: dict[str, dict[str, Any]] = {}
    inventory = robodojo_task_catalog(paths)
    for item in inventory.get("tasks", []):
        if not item.get("runnable"):
            continue
        name = str(item["name"])
        values = dict(common)
        if isinstance(overrides.get(name), dict):
            values.update(overrides[name])
        config_path = paths.upstream_root / "task" / "RoboDojo" / "config" / f"{name}.yml"
        values["task_frames"] = _task_config_names(_yaml_load(config_path))
        values["dimension"] = item.get("dimension")
        values["variant"] = item.get("variant", "standard")
        result[name] = values
    # 单元测试/最小部署可能只提供项目 fixture 而没有完整上游子模块；保留论文
    # 三任务的静态注册，不能因此让结果汇总失效。
    if not result and "TASK_CANDIDATES" in globals():
        for name, candidate in TASK_CANDIDATES.items():
            result[name] = {
                "scene_config": candidate.scene_config,
                "robot_config": candidate.robot_config,
                "camera_config": candidate.camera_config,
                "task_frames": candidate.task_frames,
                "eval_nums": candidate.eval_episodes,
            }
    return result


def _robot_candidates(paths: RoboDojoPaths) -> dict[str, RoboDojoRobotCandidate]:
    robot_dirs = [paths.upstream_root / "env_cfg" / "robot"]
    runtime_robot_dir = paths.runtime_root / "env_cfg" / "robot"
    if runtime_robot_dir.is_dir():
        robot_dirs.append(runtime_robot_dir)
    result: dict[str, RoboDojoRobotCandidate] = {}
    for robot_dir in robot_dirs:
        for path in sorted(robot_dir.glob("*.yml")):
            if path.stem in result:
                continue
            payload = _yaml_load(path)
            records = payload.get("robots", [])
            if not isinstance(records, list):
                continue
            names: list[str] = []
            target_count = 0
            for record in records:
                if not isinstance(record, dict):
                    continue
                robot_name = record.get("robot_name")
                if isinstance(robot_name, str):
                    names.append(robot_name)
                if record.get("type", "target") == "target":
                    target_count += 2 if record.get("coupled", False) else 1
            result[path.stem] = RoboDojoRobotCandidate(
                name=path.stem,
                path=str(path),
                arm_count=len(names),
                target_arm_count=target_count,
                robot_names=tuple(names),
            )
    return result


def _scene_candidates(paths: RoboDojoPaths) -> dict[str, RoboDojoSceneCandidate]:
    scene_dir = paths.upstream_root / "env_cfg" / "scene"
    return {
        path.stem: RoboDojoSceneCandidate(path.stem, str(path))
        for path in sorted(scene_dir.glob("*.yml"))
    }


def _policy_candidates(paths: RoboDojoPaths) -> dict[str, RoboDojoPolicyCandidate]:
    policy_dir = paths.upstream_root / "XPolicyLab" / "policy"
    result: dict[str, RoboDojoPolicyCandidate] = {}
    if not policy_dir.is_dir():
        return result
    for directory in sorted(path for path in policy_dir.iterdir() if path.is_dir()):
        deploy = directory / "deploy.yml"
        if not deploy.is_file():
            continue
        payload = _yaml_load(deploy)
        result[directory.name] = RoboDojoPolicyCandidate(
            name=directory.name,
            path=str(directory),
            protocol=str(payload.get("protocol", "ws")),
            has_model=(directory / "model.py").is_file(),
            has_server=(directory / "setup_eval_policy_server.sh").is_file(),
        )
    return result


def robodojo_resource_catalog(paths: Any = None) -> dict[str, Any]:
    """统一发现任务、场景、机器人、环境配置、资产和 policy。"""

    paths = paths or RoboDojoPaths()
    defaults = _task_defaults(paths)
    robots = _robot_candidates(paths)
    scenes = _scene_candidates(paths)
    policies = _policy_candidates(paths)
    tasks: dict[str, RoboDojoTaskCandidate] = {}
    for name, values in defaults.items():
        robot_name = str(values.get("robot_config", "dual_x5"))
        robot = robots.get(robot_name)
        target_count = robot.target_arm_count if robot else 0
        arm_mode: Literal["single", "bimanual"] = (
            "single" if target_count == 1 else "bimanual"
        )
        tasks[name] = RoboDojoTaskCandidate(
            name=name,
            arm_mode=arm_mode,
            paper_analogues=(),
            task_frames=tuple(values.get("task_frames", ())),
            eval_episodes=int(values.get("eval_nums", 50)),
            rationale="上游 RoboDojo 任务；论文近邻需由实验注册表另行标注。",
            scene_config=str(values.get("scene_config", "default")),
            robot_config=robot_name,
            camera_config=str(values.get("camera_config", "camera_config")),
            dimension=values.get("dimension"),
            variant=str(values.get("variant", "standard")),
            arm_count=target_count,
        )
    # 论文子集的近邻、任务帧和单/双臂评测配置保留为可复现覆盖层。
    for name, paper in TASK_CANDIDATES.items():
        if name not in tasks:
            continue
        discovered = tasks[name]
        tasks[name] = RoboDojoTaskCandidate(
            **{
                **asdict(discovered),
                "arm_mode": paper.arm_mode,
                "paper_analogues": paper.paper_analogues,
                "task_frames": paper.task_frames,
                "rationale": paper.rationale,
                "robot_config": discovered.robot_config,
                "eval_episodes": paper.eval_episodes,
            }
        )
    env_dir = paths.upstream_root / "env_cfg"
    env_configs = sorted(path.stem for path in env_dir.glob("*.yml"))
    return {
        "tasks": tasks,
        "scenes": scenes,
        "robots": robots,
        "policies": policies,
        "env_configs": tuple(env_configs),
        "asset_robots": sorted(
            path.name for path in (paths.asset_root / "Robots").glob("*") if path.is_dir()
        ),
    }


def robodojo_task_candidates(paths: Any = None) -> dict[str, RoboDojoTaskCandidate]:
    paths = paths or RoboDojoPaths()
    return robodojo_resource_catalog(paths)["tasks"]


TASK_CANDIDATES = {
    "push_T": RoboDojoTaskCandidate(
        name="push_T",
        arm_mode="single",
        paper_analogues=("SweepDust",),
        task_frames=("t", "target_t"),
        eval_episodes=50,
        rationale="单活动臂接触推动并精确对齐目标，接近 SweepDust 的平面接触控制。",
    ),
    "pour_liquid_into_cup": RoboDojoTaskCandidate(
        name="pour_liquid_into_cup",
        arm_mode="single",
        paper_analogues=("StoreBottle",),
        task_frames=("bottle", "cup"),
        eval_episodes=50,
        rationale="单活动臂操纵瓶子满足相对容器位姿，接近 StoreBottle 的对象—容器关系。",
    ),
    "sweep_blocks": RoboDojoTaskCandidate(
        name="sweep_blocks",
        arm_mode="bimanual",
        paper_analogues=("HandOver", "SweepDust"),
        task_frames=("broom", "broom_shovel", "cube*"),
        eval_episodes=50,
        rationale="任务定义明确包含扫帚跨手交接及双手扫入簸箕。",
    ),
}


@dataclass(frozen=True)
class RoboDojoPaths:
    project_root: Path = PROJECT_ROOT
    upstream_root: Path = ROBODOJO_ROOT
    asset_root: Path = ROBODOJO_ASSET_ROOT
    runtime_root: Path = ROBODOJO_RUNTIME_ROOT
    result_root: Path = ROBODOJO_RESULT_ROOT

    def as_json(self) -> dict[str, str]:
        return {name: str(value) for name, value in asdict(self).items()}


@dataclass(frozen=True)
class RoboDojoPolicyDemonstrations:
    """从冻结官方演示构造的策略输入，以及不可省略的数据来源说明。"""

    single_arm: tuple[object, ...]
    left_arm: tuple[object, ...]
    right_arm: tuple[object, ...]
    metadata: dict


def _materialize_curobo_configs(asset_root: Path) -> list[Path]:
    """由官方模板生成含本机绝对路径的 CuRobo 配置。"""

    assets_container = asset_root.parent.resolve()
    generated = []
    robot_root = asset_root / "Robots"
    if not robot_root.is_dir():
        return generated
    for template in sorted(robot_root.rglob("*_tmp.yml")):
        target = template.with_name(template.name.replace("_tmp.yml", ".yml"))
        content = template.read_text(encoding="utf-8")
        content = content.replace("${ASSETS_PATH}", str(assets_container))
        content = content.replace("$ASSETS_PATH", str(assets_container))
        target.write_text(content, encoding="utf-8")
        generated.append(target)
    return generated


def _link_directory(link: Path, target: Path) -> None:
    """创建可重复验证的相对符号链接，不覆盖未知文件。"""

    relative_target = Path(os.path.relpath(target, link.parent))
    if link.is_symlink():
        if Path(os.readlink(link)) == relative_target:
            return
        # 运行层是可丢弃的适配目录；官方下载物现在与上游代码共用唯一
        # RoboDojo 根目录，因此更新旧的外置资产路径链接。
        link.unlink()
        link.symlink_to(relative_target, target_is_directory=True)
        return
    if link.exists():
        raise RuntimeError(f"运行期路径已存在且不是受管链接：{link}")
    link.symlink_to(relative_target, target_is_directory=True)


def _write_project_robot_configs(runtime_root: Path) -> None:
    """在运行覆盖层中提供可直接组合的 X5/Franka 单臂与双臂配置。"""

    robot_dir = runtime_root / "env_cfg" / "robot"
    robot_dir.mkdir(parents=True, exist_ok=True)
    for obsolete in (
        robot_dir / "single_x5.yml",
        runtime_root / "env_cfg" / "essay2608_single_x5.yml",
    ):
        if obsolete.is_file():
            obsolete.unlink()
    configs = {
        "essay2608_single_x5_left": (("x5", [-0.3, -0.45, 0.765]),),
        "essay2608_single_x5_right": (("x5", [0.3, -0.45, 0.765]),),
        "essay2608_single_franka_left": (("franka", [-0.3, -0.45, 0.765]),),
        "essay2608_single_franka_right": (("franka", [0.3, -0.45, 0.765]),),
        "essay2608_dual_x5": (
            ("x5", [-0.3, -0.45, 0.765]),
            ("x5", [0.3, -0.45, 0.765]),
        ),
        "essay2608_dual_franka": (
            ("franka", [-0.3, -0.45, 0.765]),
            ("franka", [0.3, -0.45, 0.765]),
        ),
    }
    robot_dims = {"x5": 6, "franka": 7}
    robot_info_path = robot_dir / "_robot_info.json"
    robot_info = json.loads(robot_info_path.read_text(encoding="utf-8"))
    for config_name, records in configs.items():
        robot_records = []
        for index, (robot_name, root_position) in enumerate(records):
            grasp_direction = "top_down_little_right" if index == 0 else "top_down_little_left"
            robot_records.append(
                {
                    "robot_type": "arm",
                    "robot_name": robot_name,
                    "coupled": False,
                    "default_root_pos": root_position,
                    "default_root_rot": [0.707, 0, 0, 0.707],
                    "grasp_perfect_direction": grasp_direction,
                }
            )
        (robot_dir / f"{config_name}.yml").write_text(
            "robots:\n"
            + "".join(
                f"  - robot_type: arm\n"
                f"    robot_name: {record['robot_name']}\n"
                f"    coupled: false\n"
                f"    default_root_pos: {record['default_root_pos']}\n"
                f"    default_root_rot: [0.707, 0, 0, 0.707]\n"
                f"    grasp_perfect_direction: \"{record['grasp_perfect_direction']}\"\n"
                for record in robot_records
            ),
            encoding="utf-8",
        )
        dims = [robot_dims[record["robot_name"]] for record in robot_records]
        robot_info[config_name] = {"arm_dim": dims, "ee_dim": [1] * len(dims)}
    robot_info_path.write_text(
        json.dumps(robot_info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def ensure_runtime_env_config(
    runtime_root: Path,
    *,
    task_name: str,
    scene_config: str,
    robot_config: str,
    observation_mode: Literal["oracle_pose", "rgbd_pose"] = "oracle_pose",
    camera_config: str = "camera_config",
) -> str:
    """生成任意任务/场景/机器人组合的项目运行配置。"""

    if observation_mode not in {"oracle_pose", "rgbd_pose"}:
        raise ValueError(f"未知观测模式：{observation_mode}")
    scene_path = runtime_root / "env_cfg" / "scene" / f"{scene_config}.yml"
    robot_path = runtime_root / "env_cfg" / "robot" / f"{robot_config}.yml"
    # ``arx_x5`` 是 RoboDojo 旧版顶层环境配置名，而不是 robot/ 下的文件。
    # 接受这类上游环境别名，并解析其 config.robot，保证 --robot arx_x5 与
    # 直接使用上游 env_cfg/arx_x5.yml 的语义一致。
    robot_reference = robot_config
    if not robot_path.is_file():
        alias_path = runtime_root / "env_cfg" / f"{robot_config}.yml"
        alias_payload = _yaml_load(alias_path)
        alias_config = alias_payload.get("config", {})
        if isinstance(alias_config, dict) and isinstance(alias_config.get("robot"), str):
            robot_reference = alias_config["robot"]
            robot_path = runtime_root / "env_cfg" / "robot" / f"{robot_reference}.yml"
    camera_path = runtime_root / "env_cfg" / "camera" / f"{camera_config}.yml"
    for path, label in ((scene_path, "场景"), (robot_path, "机器人"), (camera_path, "相机")):
        if not path.is_file():
            raise FileNotFoundError(f"{label}配置不存在：{path}")
    suffix = "oracle" if observation_mode == "oracle_pose" else "rgbd"
    # 机器人配置名可能来自项目覆盖层（本身已经带 essay2608_ 前缀），也
    # 可能直接来自上游（如 arx_x5）。统一只保留一个项目前缀，避免生成
    # ``essay2608_essay2608_...`` 这种难以在命令行复用的配置名。
    robot_label = robot_config.removeprefix("essay2608_")
    config_name = f"essay2608_{robot_label}_{scene_config}_{suffix}"
    vision = {
        "approximate_depth": False,
        "depth": observation_mode == "rgbd_pose",
        "intrinsic_matrix": observation_mode == "rgbd_pose",
        "extrinsic_matrix": observation_mode == "rgbd_pose",
        "shape": True,
    }
    (runtime_root / "env_cfg" / f"{config_name}.yml").write_text(
        f"""config_name: {config_name}

config:
  sim: sim_config
  scene: {scene_config}
  robot: {robot_reference}
  camera: {camera_config}

observation:
  collect_freq: 25
  robot:
    joint_states: true
    world_ee_state: true
  vision:
    approximate_depth: {str(vision['approximate_depth']).lower()}
    depth: {str(vision['depth']).lower()}
    intrinsic_matrix: {str(vision['intrinsic_matrix']).lower()}
    extrinsic_matrix: {str(vision['extrinsic_matrix']).lower()}
    shape: true
""",
        encoding="utf-8",
    )
    # 上游 EvalEnv 按 env 配置名查找布局目录。布局文件描述的是物体/相机
    # 初始状态，与机械臂型号正交；对新组合复用官方 arx_x5 布局，同时在
    # 配置旁写明来源，避免误把它当作针对新机械臂重新标定的布局。
    layout_root = runtime_root / "Assets" / "Eval_Layout" / "RoboDojo"
    layout_source = layout_root / "arx_x5"
    layout_link = layout_root / config_name
    if layout_source.is_dir() and not layout_link.exists():
        _link_directory(layout_link, layout_source)
    return config_name


def _write_legacy_project_env_configs(runtime_root: Path) -> None:
    """保留论文三任务的旧配置名，并使其走统一配置生成器。"""

    for config_name in ("essay2608_single_x5_left", "essay2608_single_x5_right"):
        (runtime_root / "env_cfg" / f"{config_name}.yml").write_text(
            f"""config_name: {config_name}

config:
  sim: sim_config
  scene: default
  robot: {config_name}
  camera: camera_config

observation:
  collect_freq: 25
  robot:
    joint_states: true
    world_ee_state: true
  vision:
    approximate_depth: false
    depth: false
    intrinsic_matrix: false
    extrinsic_matrix: false
    shape: true
""",
            encoding="utf-8",
        )


def prepare_robodojo_runtime(paths: RoboDojoPaths = RoboDojoPaths()) -> Path:
    """生成可丢弃的运行根目录，上游子模块保持只读。"""

    required = (
        paths.upstream_root / "src" / "eval_client" / "main.py",
        paths.upstream_root / "task" / "RoboDojo" / "task_registry.py",
        paths.upstream_root / "XPolicyLab" / "policy" / "demo_policy" / "deploy.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(
            "RoboDojo 子模块不完整，请先执行 git submodule update --init --recursive："
            + ", ".join(missing)
        )

    paths.runtime_root.mkdir(parents=True, exist_ok=True)
    # 上游 ``env/global_configs.py`` 根据 ``__file__/..`` 解析 Assets 与 env_cfg。
    # 若 env 是目录链接，操作系统会沿链接回到子模块，越过我们的运行期覆盖层。
    # env 仅 1 MB 左右，因此在可丢弃运行目录中复制它；其余大目录继续链接。
    runtime_env = paths.runtime_root / "env"
    expected_env_link = Path(os.path.relpath(paths.upstream_root / "env", runtime_env.parent))
    if runtime_env.is_symlink():
        if Path(os.readlink(runtime_env)) != expected_env_link:
            raise RuntimeError(
                f"运行期 env 链接指向异常：{runtime_env} -> {os.readlink(runtime_env)}"
            )
        runtime_env.unlink()
    elif runtime_env.exists() and not runtime_env.is_dir():
        raise RuntimeError(f"运行期 env 路径不是目录：{runtime_env}")
    shutil.copytree(paths.upstream_root / "env", runtime_env, dirs_exist_ok=True)

    # 管理脚本也作为只读入口挂入运行层；这样可以从统一的
    # ``.runtime/robodojo/scripts`` 调用上游下载、服务端和结果汇总工具，
    # 同时不把脚本复制成项目的第二份实现。
    for name in ("task", "src", "utils", "scripts", "XPolicyLab"):
        _link_directory(paths.runtime_root / name, paths.upstream_root / name)

    env_cfg = paths.runtime_root / "env_cfg"
    shutil.copytree(paths.upstream_root / "env_cfg", env_cfg, dirs_exist_ok=True)
    _write_project_robot_configs(paths.runtime_root)
    _write_legacy_project_env_configs(paths.runtime_root)

    paths.asset_root.mkdir(parents=True, exist_ok=True)
    _materialize_curobo_configs(paths.asset_root)
    layout_root = paths.asset_root / "Eval_Layout" / "RoboDojo"
    layout_root.mkdir(parents=True, exist_ok=True)
    obsolete_layout = layout_root / "essay2608_single_x5"
    if obsolete_layout.is_symlink():
        obsolete_layout.unlink()
    elif obsolete_layout.exists():
        raise RuntimeError(f"旧单臂布局路径不是受管链接：{obsolete_layout}")
    for config_name in ("essay2608_single_x5_left", "essay2608_single_x5_right"):
        _link_directory(layout_root / config_name, layout_root / "arx_x5")
    paths.result_root.mkdir(parents=True, exist_ok=True)
    _link_directory(paths.runtime_root / "Assets", paths.asset_root)
    _link_directory(paths.runtime_root / "eval_result", paths.result_root)

    resources = robodojo_resource_catalog(paths)
    manifest = {
        "schema": "essay2608.robodojo.runtime.v1",
        "upstream_commit": _git_commit(paths.upstream_root),
        "paths": paths.as_json(),
        "gui_required": True,
        "task_candidates": {
            name: asdict(task) for name, task in resources["tasks"].items()
        },
        "resources": {
            "scenes": sorted(resources["scenes"]),
            "robots": sorted(resources["robots"]),
            "policies": sorted(resources["policies"]),
        },
    }
    (paths.runtime_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return paths.runtime_root


def _resolve_task_names(
    task_names: tuple[str, ...] | None,
    paths: RoboDojoPaths,
) -> tuple[str, ...]:
    names = tuple(task_names) if task_names is not None else tuple(robodojo_task_candidates(paths))
    known = robodojo_task_candidates(paths)
    unknown = sorted(set(names).difference(known))
    if unknown:
        raise ValueError(f"未知 RoboDojo 任务：{unknown}")
    return names


def download_robodojo_assets(
    task_names: tuple[str, ...] | None = None,
    paths: RoboDojoPaths = RoboDojoPaths(),
    revision: str = "main",
) -> Path:
    """从官方 Hugging Face 数据集下载指定任务的资产；不传任务即覆盖全部可运行任务。"""

    task_names = _resolve_task_names(task_names, paths)
    # RoboDojo 资产仓库启用了 Xet。匿名并发请求很容易触发 Xet token 接口限流，
    # 因此资产库初始化固定使用普通 HTTP，并以单线程、可恢复方式下载。
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    try:
        from huggingface_hub import HfApi, constants, hf_hub_download
        from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError
        from requests import RequestException
    except ImportError as error:
        raise RuntimeError("下载资产需要先安装 huggingface_hub") from error
    constants.HF_HUB_DISABLE_XET = True

    patterns = list(COMMON_ASSET_PATTERNS)
    for task_name in task_names:
        patterns.extend(task_asset_patterns(task_name, paths))
    api = HfApi()
    resolved_revision = api.dataset_info("RoboDojo-Benchmark/RoboDojo", revision=revision).sha
    files: set[str] = set()
    # 原先按每个模式递归请求一次远端目录；54 个任务会产生上百次分页请求，
    # 在匿名 HF/Xet 连接下容易看起来像“卡住”。Assets 文件树只有约 1.5 万项，
    # 一次读取后在本地匹配模式，既减少请求也保证 --all 的可重复性。
    remote_assets = {}
    for item in api.list_repo_tree(
        "RoboDojo-Benchmark/RoboDojo",
        path_in_repo="Assets",
        repo_type="dataset",
        revision=resolved_revision,
        recursive=True,
    ):
        # ``RepoFile`` 在不同 huggingface_hub 版本中不一定从顶层导出；
        # 文件项稳定地带有 ``size``，目录项则没有。忽略网页预览缩略图。
        if hasattr(item, "size") and ".thumbs" not in Path(item.path).parts:
            remote_assets[item.path] = int(item.size or 0)
    for pattern in sorted(set(patterns)):
        if not any(character in pattern for character in "*?["):
            if pattern in remote_assets:
                files.add(pattern)
            continue
        files.update(
            path for path in remote_assets if fnmatchcase(path, pattern)
        )
    if not files:
        raise RuntimeError("候选任务没有解析出任何 RoboDojo 资产文件")

    local_root = paths.asset_root.parent
    local_root.mkdir(parents=True, exist_ok=True)

    def download(filename: str) -> str:
        local_path = local_root / filename
        if local_path.is_file() and local_path.stat().st_size > 0:
            return filename
        for attempt in range(8):
            try:
                hf_hub_download(
                    repo_id="RoboDojo-Benchmark/RoboDojo",
                    repo_type="dataset",
                    revision=resolved_revision,
                    filename=filename,
                    local_dir=local_root,
                )
                return filename
            except (RequestException, HfHubHTTPError, LocalEntryNotFoundError):
                if attempt == 7:
                    raise
                wait_seconds = min(2 ** (attempt + 1), 120)
                print(
                    f"下载请求受限，{wait_seconds} 秒后重试：{filename}",
                    flush=True,
                )
                time.sleep(wait_seconds)
        raise AssertionError("不可达")

    ordered_files = sorted(files)
    # 匿名 HF 连接对批量 HEAD/GET 较敏感；五路请求配合“已有文件跳过”和
    # 指数退避在大文件阶段能提高吞吐。重新运行会从中断处续传。
    with ThreadPoolExecutor(max_workers=5) as executor:
        for index, _ in enumerate(executor.map(download, ordered_files), start=1):
            if index == 1 or index % 25 == 0 or index == len(ordered_files):
                print(f"RoboDojo 资产：{index}/{len(ordered_files)}", flush=True)
    paths.asset_root.mkdir(parents=True, exist_ok=True)
    planner_configs = _materialize_curobo_configs(paths.asset_root)
    manifest = {
        "schema": "essay2608.robodojo.assets.v1",
        "repository": "RoboDojo-Benchmark/RoboDojo",
        "requested_revision": revision,
        "resolved_revision": resolved_revision,
        "tasks": list(task_names),
        "patterns": sorted(set(patterns)),
        "file_count": len(files),
        "files": sorted(files),
        "generated_curobo_configs": [str(path.relative_to(local_root)) for path in planner_configs],
    }
    meta_root = paths.asset_root.parent / ".cache" / "essay2608"
    meta_root.mkdir(parents=True, exist_ok=True)
    (meta_root / "assets_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return paths.asset_root


def download_robodojo_demonstrations(
    task_names: tuple[str, ...] | None = None,
    episode_count: int = 5,
    revision: str = "main",
    paths: RoboDojoPaths = RoboDojoPaths(),
) -> Path:
    """按冻结数量下载指定任务的官方 HDF5 专家演示。"""

    task_names = _resolve_task_names(task_names, paths)
    if episode_count < 1:
        raise ValueError("episode_count 必须为正")
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as error:
        raise RuntimeError("下载演示需要先安装 huggingface_hub") from error
    filenames = [
        f"data/RoboDojo/{task_name}/arx_x5/data/episode_{index:07d}.hdf5"
        for task_name in task_names
        for index in range(episode_count)
    ]
    local_root = paths.asset_root.parent
    local_root.mkdir(parents=True, exist_ok=True)
    resolved_revision = HfApi().dataset_info("RoboDojo-Benchmark/RoboDojo", revision=revision).sha
    # 这里都是精确文件名。逐文件请求可避免 ``snapshot_download`` 为筛选少量
    # HDF5 而先遍历整个多 TB 数据集的文件树。
    records = []
    for filename in filenames:
        local_path = Path(
            hf_hub_download(
                repo_id="RoboDojo-Benchmark/RoboDojo",
                repo_type="dataset",
                revision=resolved_revision,
                filename=filename,
                local_dir=local_root,
            )
        )
        digest = hashlib.sha256()
        with local_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        records.append(
            {"path": filename, "bytes": local_path.stat().st_size, "sha256": digest.hexdigest()}
        )
    manifest = {
        "schema": "essay2608.robodojo.demonstrations.v1",
        "repository": "RoboDojo-Benchmark/RoboDojo",
        "requested_revision": revision,
        "resolved_revision": resolved_revision,
        "tasks": list(task_names),
        "episode_count": episode_count,
        "files": records,
        "claim_boundary": (
            "RoboDojo HDF5 不含本项目真值任务帧；必须经 GUI 回放补采后才能训练 DynaMAC/MiDiGaP"
        ),
    }
    meta_root = paths.asset_root.parent / ".cache" / "essay2608"
    meta_root.mkdir(parents=True, exist_ok=True)
    (meta_root / "data_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return paths.asset_root.parent / "data" / "RoboDojo"


def sync_robodojo_official_snapshot(
    paths: RoboDojoPaths = RoboDojoPaths(),
    revision: str = "main",
) -> Path:
    """同步官方 HF 根文件和全部任务资产，明确跳过 ``data/``、``ckpt/``。

    专家 HDF5 不在这里隐式全量下载；调用方再用
    :func:`download_robodojo_demonstrations` 选择任务和条数。这样本地目录始终能从
    清单区分“官方快照”与“项目训练数据”。
    """

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as error:
        raise RuntimeError("同步官方 RoboDojo 快照需要先安装 huggingface_hub") from error
    official_root = paths.asset_root.parent
    official_root.mkdir(parents=True, exist_ok=True)
    meta_root = official_root / ".cache" / "essay2608"
    hf_root = meta_root / "hf_root"
    hf_root.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    resolved_revision = api.dataset_info(
        "RoboDojo-Benchmark/RoboDojo", revision=revision
    ).sha
    root_files = (".gitattributes", ".gitignore", "README.md")
    for filename in root_files:
        hf_hub_download(
            repo_id="RoboDojo-Benchmark/RoboDojo",
            repo_type="dataset",
            revision=resolved_revision,
            filename=filename,
            local_dir=hf_root,
        )
    download_robodojo_assets(task_names=None, paths=paths, revision=revision)
    data_manifest = meta_root / "data_manifest.json"
    selected_data = json.loads(data_manifest.read_text(encoding="utf-8")) if data_manifest.is_file() else None
    manifest = {
        "schema": "essay2608.robodojo.official_snapshot.v1",
        "repository": "RoboDojo-Benchmark/RoboDojo",
        "requested_revision": revision,
        "resolved_revision": resolved_revision,
        "downloaded_prefixes": ["Assets/"],
        "downloaded_root_files": list(root_files),
        "excluded_prefixes": ["data/", "ckpt/"],
        "selected_data_manifest": selected_data,
    }
    target = meta_root / "official_snapshot_manifest.json"
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return official_root


def _three_phase_skill_labels(length: int):
    """兼容旧调用名。

    旧的固定三段切分不再用于正式加载路径；保留这个私有符号是为了避免旧脚本
    导入时报错，并明确要求调用方提供真实轨迹后使用 :func:`_tapas_labels`。
    """

    if length < 6:
        raise ValueError("RoboDojo 演示过短，无法进行技能分割")
    raise ValueError("固定长度接口无法进行 TAPAS 分割，请传入末端轨迹")


def _tapas_labels(poses, gripper):
    config = _tapas_config(len(poses))
    return tapas_skill_labels(poses, gripper, config)


def _tapas_config(length: int) -> TapasSegmentationConfig:
    """返回当前实验冻结的、随轨迹长度轻微自适应的 TAPAS 参数。"""

    return TapasSegmentationConfig(
        minimum_skill_length=max(8, min(24, length // 12)),
        maximum_skills=3,
    )


def _tapas_provenance() -> dict[str, Any]:
    return {
        "backend": "kinematic_velocity_valleys_v1",
        "minimum_skill_length": "max(8,min(24,T//12))",
        "maximum_skills": 3,
        "velocity_quantile": TapasSegmentationConfig.velocity_quantile,
        "smoothing_window": TapasSegmentationConfig.smoothing_window,
        "gripper_change_quantile": TapasSegmentationConfig.gripper_change_quantile,
    }


def _capture_path(capture_root: str | Path | None, task_name: str, index: int) -> Path | None:
    """解析 GUI 补采 JSONL 路径；同时支持根目录和任务目录两种布局。"""

    if capture_root is None:
        return None
    root = Path(capture_root).expanduser().resolve()
    filename = f"episode_{index:07d}.jsonl"
    candidates = [root / task_name / filename, root / filename]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _load_captured_frames(
    capture_root: str | Path | None,
    task_name: str,
    index: int,
    expected_length: int,
    require_audit: bool = False,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """读取一次 GUI 回放的逐时刻任务帧。

    如果显式指定了 ``capture_root``，文件存在但帧为空或长度不匹配时直接报错，
    防止把失败的补采悄悄降级成静态布局。
    """

    import numpy as np

    path = _capture_path(capture_root, task_name, index)
    if path is None:
        if capture_root is not None:
            raise FileNotFoundError(
                f"GUI 补采不存在：{Path(capture_root) / task_name / f'episode_{index:07d}.jsonl'}"
            )
        return None, {"source": "not_provided"}
    audit_path = path.with_suffix(".audit.json")
    audit: dict[str, Any] | None = None
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if not audit.get("accepted_for_training", False):
            raise ValueError(f"GUI 补采未通过训练门禁：{audit_path}")
    elif require_audit:
        raise FileNotFoundError(f"GUI 补采验收记录不存在：{audit_path}")
    metadata: dict[str, Any] = {}
    steps: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"GUI 补采 JSONL 无法解析：{path}:{line_number}") from error
            if record.get("type") == "metadata":
                metadata = record
            elif record.get("type") == "step":
                steps.append(record)
    if len(steps) != expected_length:
        raise ValueError(f"GUI 补采步数与 HDF5 不一致：{path} ({len(steps)} != {expected_length})")
    if [step.get("index") for step in steps] != list(range(expected_length)):
        raise ValueError(f"GUI 补采 step index 不连续：{path}")
    names = tuple(sorted(steps[0].get("frames", {})))
    if not names:
        raise ValueError(f"GUI 补采没有任务真值帧：{path}")
    frames: dict[str, Any] = {}
    for name in names:
        values = []
        for step in steps:
            if tuple(sorted(step.get("frames", {}))) != names:
                raise ValueError(f"GUI 补采各时刻任务帧集合不一致：{path}")
            value = np.asarray(step["frames"][name], dtype=np.float64)
            if value.shape != (7,) or not np.all(np.isfinite(value)):
                raise ValueError(f"GUI 补采帧 {name} 不是有限的 xyz+wxyz：{path}")
            values.append(value)
        frames[name] = np.stack(values)
    return frames, {
        "source": "gui_capture",
        "path": str(path),
        "sha256": _sha256_file(path),
        "schema": metadata.get("schema"),
        "observation_source": metadata.get("observation_source"),
        "audit_path": str(audit_path) if audit is not None else None,
        "audit_sha256": _sha256_file(audit_path) if audit is not None else None,
    }


def _layout_pose(record: dict, field: str):
    import numpy as np

    pose = np.asarray([*record["default_pos"], *record["default_ori"]], dtype=np.float64)
    if pose.shape != (7,) or not np.all(np.isfinite(pose)):
        raise ValueError(f"RoboDojo 布局字段 {field} 不是有限的 xyz+wxyz 位姿")
    return pose


def _load_generic_robodojo_policy_demonstrations(
    task_name: str,
    episode_count: int,
    paths: RoboDojoPaths,
    capture_root: str | Path | None = None,
) -> RoboDojoPolicyDemonstrations:
    """加载任意 RoboDojo 任务的机器人轨迹。

    上游 HDF5 对所有任务都提供双 X5 的 state/action；未完成 RGB-D/GUI 物体帧补采时，
    这里只构造末端与对侧末端条件，并把限制写入元数据。它用于打通训练入口，不把
    缺少任务物体真值的轨迹伪装成 DynaMAC 论文正式数据。
    """

    try:
        import h5py
        import numpy as np
    except ImportError as error:
        raise RuntimeError("读取 RoboDojo 官方演示需要 h5py 和 NumPy") from error
    from essay2608.policy.dynamac import DynaMACDemonstration

    candidate = robodojo_task_candidates(paths)[task_name]
    single_arm: list[DynaMACDemonstration] = []
    left_arm: list[DynaMACDemonstration] = []
    right_arm: list[DynaMACDemonstration] = []
    episode_records = []
    for index in range(episode_count):
        episode = (
            paths.project_root
            / "data"
            / "robodojo"
            / "data"
            / "RoboDojo"
            / task_name
            / "arx_x5"
            / "data"
            / f"episode_{index:07d}.hdf5"
        )
        if not episode.is_file():
            raise FileNotFoundError(f"冻结的 RoboDojo 演示不存在：{episode}")
        with h5py.File(episode) as archive:
            required = [
                f"{group}/{side}_ee_poses"
                for group in ("state", "action")
                for side in ("left", "right")
            ] + [
                f"{group}/{side}_ee_joint_states"
                for group in ("state", "action")
                for side in ("left", "right")
            ]
            missing = [name for name in required if name not in archive]
            if missing:
                raise ValueError(f"官方演示字段不完整：{missing}")
            state = {
                side: archive[f"state/{side}_ee_poses"][()].astype(np.float64)
                for side in ("left", "right")
            }
            action = {
                side: archive[f"action/{side}_ee_poses"][()].astype(np.float64)
                for side in ("left", "right")
            }
            gripper = {
                side: archive[f"action/{side}_ee_joint_states"][()].astype(np.float64)
                for side in ("left", "right")
            }
        lengths = {len(value) for values in (state, action, gripper) for value in values.values()}
        if len(lengths) != 1:
            raise ValueError(f"官方演示长度不一致：{episode}")
        length = lengths.pop()
        for side in ("left", "right"):
            if not np.allclose(action[side][:-1], state[side][1:], atol=2.0e-4):
                raise ValueError(f"官方演示 action[t] != state[t+1]：{episode} {side}")
        captured_frames, capture_record = _load_captured_frames(
            capture_root, task_name, index, length, require_audit=capture_root is not None
        )
        if candidate.arm_mode == "single":
            active = max(
                ("left", "right"),
                key=lambda side: float(
                    np.linalg.norm(np.diff(state[side][:, :3], axis=0), axis=1).sum()
                ),
            )
            skill = _tapas_labels(state[active], gripper[active])
            frames = captured_frames or {"active_ee": state[active].copy()}
            single_arm.append(
                DynaMACDemonstration(
                    ee_pose=state[active],
                    action_pose=action[active],
                    gripper=gripper[active],
                    frames=frames,
                    skill=skill,
                    name=f"robodojo_{task_name}_{index:03d}_{active}",
                )
            )
        else:
            skill = _tapas_labels(state["left"], gripper["left"])
            left_frames = dict(captured_frames or {})
            left_frames.setdefault("right_ee", state["right"].copy())
            right_frames = dict(captured_frames or {})
            right_frames.setdefault("left_ee", state["left"].copy())
            left_arm.append(
                DynaMACDemonstration(
                    ee_pose=state["left"],
                    action_pose=action["left"],
                    gripper=gripper["left"],
                    frames=left_frames,
                    skill=skill,
                    name=f"robodojo_{task_name}_{index:03d}_left",
                )
            )
            right_arm.append(
                DynaMACDemonstration(
                    ee_pose=state["right"],
                    action_pose=action["right"],
                    gripper=gripper["right"],
                    frames=right_frames,
                    skill=skill,
                    name=f"robodojo_{task_name}_{index:03d}_right",
                )
            )
        episode_records.append(
            {
                "episode": str(episode),
                "sha256": _sha256_file(episode),
                "steps": length,
                "frame_source": capture_record,
            }
        )
    return RoboDojoPolicyDemonstrations(
        tuple(single_arm),
        tuple(left_arm),
        tuple(right_arm),
        {
            "schema": "essay2608.robodojo.policy_demonstrations.v1",
            "task": task_name,
            "task_arm_mode": candidate.arm_mode,
            "source_embodiment": "arx_x5",
            "episode_count": episode_count,
            "skill_segmentation": "tapas_velocity_valleys_v1",
            "skill_segmentation_config": _tapas_provenance(),
            "observation_frames": sorted(
                single_arm[0].frames if single_arm else left_arm[0].frames
            ),
            "limitations": (
                "使用 GUI 逐时刻任务帧；视觉候选生成仍由 RoboDojo/外部感知提供。"
                if capture_root is not None
                else "未提供 GUI 物体帧，使用末端退化条件；不能作为正式任务参数结果。"
            ),
            "episodes": episode_records,
        },
    )


def load_robodojo_policy_demonstrations(
    task_name: str,
    episode_count: int = 5,
    paths: RoboDojoPaths = RoboDojoPaths(),
    capture_root: str | Path | None = None,
) -> RoboDojoPolicyDemonstrations:
    """把冻结官方演示转换为 DynaMAC/MiDiGaP/DP 共用的真值状态输入。

    ``capture_root`` 指向 ``robodojo capture`` 生成的 JSONL 根目录时，优先使用
    GUI 每一时刻补采的任务帧；未提供时才回退到静态布局/对侧末端条件。
    """

    try:
        import h5py
        import numpy as np
    except ImportError as error:
        raise RuntimeError("读取 RoboDojo 官方演示需要 h5py 和 NumPy") from error
    from essay2608.policy.dynamac import DynaMACDemonstration

    if task_name not in robodojo_task_candidates(paths):
        raise ValueError(f"RoboDojo 任务不存在或不可运行：{task_name}")
    if episode_count < 1:
        raise ValueError("episode_count 必须为正整数")

    if task_name not in {"push_T", "sweep_blocks"}:
        return _load_generic_robodojo_policy_demonstrations(
            task_name, episode_count, paths, capture_root
        )

    single_arm = []
    left_arm = []
    right_arm = []
    episode_records = []
    for index in range(episode_count):
        episode = (
            paths.project_root
            / "data"
            / "robodojo"
            / "data"
            / "RoboDojo"
            / task_name
            / "arx_x5"
            / "data"
            / f"episode_{index:07d}.hdf5"
        )
        if not episode.is_file():
            raise FileNotFoundError(f"冻结的 RoboDojo 演示不存在：{episode}")
        with h5py.File(episode, "r") as archive:
            required = [
                f"{group}/{side}_{field}"
                for group in ("state", "action")
                for side in ("left", "right")
                for field in ("arm_joint_states", "ee_poses", "ee_joint_states")
            ]
            missing = [name for name in required if name not in archive]
            if missing:
                raise ValueError(f"官方演示字段不完整：{missing}")
            state = {
                side: archive[f"state/{side}_ee_poses"][()].astype(np.float64)
                for side in ("left", "right")
            }
            action = {
                side: archive[f"action/{side}_ee_poses"][()].astype(np.float64)
                for side in ("left", "right")
            }
            gripper = {
                side: archive[f"action/{side}_ee_joint_states"][()].astype(np.float64)
                for side in ("left", "right")
            }
            joint_state = {
                side: archive[f"state/{side}_arm_joint_states"][()].astype(np.float64)
                for side in ("left", "right")
            }
            joint_action = {
                side: archive[f"action/{side}_arm_joint_states"][()].astype(np.float64)
                for side in ("left", "right")
            }
        lengths = {
            len(value)
            for value in [
                *state.values(),
                *action.values(),
                *gripper.values(),
                *joint_state.values(),
                *joint_action.values(),
            ]
        }
        if len(lengths) != 1:
            raise ValueError(f"官方演示左右臂/状态动作长度不一致：{episode}")
        length = lengths.pop()
        for side in ("left", "right"):
            exact_joint_shift = np.array_equal(joint_action[side][:-1], joint_state[side][1:])
            ee_shift_error = float(np.max(np.abs(action[side][:-1] - state[side][1:])))
            if not exact_joint_shift or ee_shift_error > 2.0e-4:
                raise ValueError(f"官方演示 action[t] != state[t+1]：{episode} {side}")
        record = {"episode": str(episode), "sha256": _sha256_file(episode), "steps": length}
        captured_frames, capture_record = _load_captured_frames(
            capture_root, task_name, index, length, require_audit=capture_root is not None
        )
        record["frame_source"] = capture_record

        if task_name == "push_T":
            skill = _tapas_labels(state["right"], gripper["right"])
            if captured_frames is not None:
                frames = captured_frames
                record["frame_source"] = capture_record
            else:
                layout_path = (
                    paths.project_root
                    / "results"
                    / "robodojo"
                    / "source_layouts"
                    / "push_T"
                    / f"episode_{index:07d}.json"
                )
                evidence_path = layout_path.with_suffix(".reconstruction.json")
                if not layout_path.is_file() or not evidence_path.is_file():
                    raise FileNotFoundError(f"push_T RGB 源布局重建不存在：{layout_path}")
                evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
                if evidence.get("source_sha256") != record["sha256"]:
                    raise ValueError(f"源布局与官方演示哈希不匹配：{layout_path}")
                layout = json.loads(layout_path.read_text(encoding="utf-8"))
                initial_t = _layout_pose(layout["Rigid"]["t"][0], "Rigid.t")
                target_t = _layout_pose(
                    layout["Geometry"]["t_cushion"][0], "Geometry.t_cushion"
                )
                frames = {
                    "t": np.repeat(initial_t[None], length, axis=0),
                    "target_t": np.repeat(target_t[None], length, axis=0),
                }
                record.update(
                    {
                        "layout": str(layout_path),
                        "layout_sha256": _sha256_file(layout_path),
                        "layout_method": evidence.get("method"),
                    }
                )
            single_arm.append(
                DynaMACDemonstration(
                    ee_pose=state["right"],
                    action_pose=action["right"],
                    gripper=gripper["right"],
                    frames=frames,
                    skill=skill,
                    name=f"robodojo_push_T_{index:03d}_right",
                )
            )
            record.update(
                {
                    "active_side": "right",
                    "frame_names": sorted(frames),
                }
            )
        else:
            skill = _tapas_labels(state["left"], gripper["left"])
            left_frames = dict(captured_frames or {})
            left_frames.setdefault("right_ee", state["right"].copy())
            right_frames = dict(captured_frames or {})
            right_frames.setdefault("left_ee", state["left"].copy())
            left_arm.append(
                DynaMACDemonstration(
                    ee_pose=state["left"],
                    action_pose=action["left"],
                    gripper=gripper["left"],
                    frames=left_frames,
                    skill=skill,
                    name=f"robodojo_sweep_blocks_{index:03d}_left",
                )
            )
            right_arm.append(
                DynaMACDemonstration(
                    ee_pose=state["right"],
                    action_pose=action["right"],
                    gripper=gripper["right"],
                    frames=right_frames,
                    skill=skill,
                    name=f"robodojo_sweep_blocks_{index:03d}_right",
                )
            )
        episode_records.append(record)

    metadata = {
        "schema": "essay2608.robodojo.policy_demonstrations.v1",
        "task": task_name,
        "task_arm_mode": robodojo_task_candidates()[task_name].arm_mode,
        "source_embodiment": "arx_x5",
        "episode_count": episode_count,
        "skill_segmentation": "tapas_velocity_valleys_v1",
        "skill_segmentation_config": _tapas_provenance(),
        "observation_frames": sorted(
            single_arm[0].frames if single_arm else left_arm[0].frames
        ),
        "limitations": (
            "使用 GUI 逐时刻任务帧；视觉候选生成仍由 RoboDojo/外部感知提供。"
            if capture_root is not None
            else (
                "RGB 重建的初始/目标位姿；原轨迹在当前仿真版本未通过接触确定性回放"
                if task_name == "push_T"
                else "首轮双臂基线没有物体真值，只建模左右末端协调关系"
            )
        ),
        "episodes": episode_records,
    }
    return RoboDojoPolicyDemonstrations(
        tuple(single_arm), tuple(left_arm), tuple(right_arm), metadata
    )


def _git_commit(repository: Path) -> str:
    head = repository / ".git"
    if not head.exists():
        return "unknown"
    result = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def robodojo_task_catalog(paths: RoboDojoPaths = RoboDojoPaths()) -> dict:
    """读取上游自己的任务清单，避免在本项目复制 54 个任务名称。"""

    inventory = paths.upstream_root / "scripts" / "internal" / "task_inventory.py"
    if not inventory.is_file():
        return {
            "available": False,
            "counts": {"tasks": 0, "runnable": 0},
            "tasks": [],
        }
    result = subprocess.run(
        [sys.executable, str(inventory), "--format", "json"],
        cwd=paths.upstream_root,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    return {
        "available": True,
        "upstream_commit": _git_commit(paths.upstream_root),
        "counts": payload["counts"],
        "tasks": payload["tasks"],
    }


def robodojo_status(paths: RoboDojoPaths = RoboDojoPaths()) -> dict:
    """返回不启动仿真的接入状态。"""

    required_asset_dirs = ("Robots", "Object", "Material", "Eval_Layout")
    asset_state = {name: (paths.asset_root / name).is_dir() for name in required_asset_dirs}
    robot_assets = (
        sorted(path.name for path in (paths.asset_root / "Robots").iterdir() if path.is_dir())
        if asset_state["Robots"]
        else []
    )
    common_ready = all(
        _asset_pattern_present(paths.asset_root, item) for item in COMMON_ASSET_PATTERNS
    ) and all(
        (paths.asset_root / "Robots" / robot / "curobo.yml").is_file() for robot in ("franka", "x5")
    )
    tasks = robodojo_task_candidates(paths)
    task_asset_state = {
        task_name: all(
            _asset_pattern_present(paths.asset_root, item)
            for item in task_asset_patterns(task_name, paths)
        )
        for task_name in tasks
    }
    resources = robodojo_resource_catalog(paths)
    catalog = robodojo_task_catalog(paths)
    return {
        "upstream_present": (paths.upstream_root / "README.md").is_file(),
        "upstream_commit": _git_commit(paths.upstream_root),
        "runtime_present": (paths.runtime_root / "manifest.json").is_file(),
        "assets": asset_state,
        "common_assets_ready": common_ready,
        "task_assets": task_asset_state,
        "assets_ready": common_ready and all(task_asset_state.values()),
        "robot_asset_library": robot_assets,
        "gui_required": True,
        "candidate_pool": {
            "available": catalog["available"],
            "counts": catalog["counts"],
        },
        "tasks": {name: asdict(task) for name, task in tasks.items()},
        "scenes": {name: asdict(scene) for name, scene in resources["scenes"].items()},
        "robots": {name: asdict(robot) for name, robot in resources["robots"].items()},
        "policies": {name: asdict(policy) for name, policy in resources["policies"].items()},
        "env_configs": list(resources["env_configs"]),
        "paths": paths.as_json(),
    }


def _asset_pattern_present(asset_root: Path, pattern: str) -> bool:
    relative = pattern.removeprefix("Assets/")
    # 不同 Python 版本对末尾 ``/**`` 的 glob 结果不同（有的只返回目录）。
    # 先匹配包含内部 ``**`` 的目录，再递归检查其文件，避免把通配符当字面
    # 目录，也避免完整下载后误报任务资产缺失。
    if relative.endswith("/**"):
        directory_pattern = relative.removesuffix("/**")
        return any(
            path.is_dir() and any(child.is_file() for child in path.rglob("*"))
            for path in asset_root.glob(directory_pattern)
        )
    return any(path.is_file() for path in asset_root.glob(relative))


def environment_config_for(
    task_name: str,
    *,
    scene_config: str | None = None,
    robot_config: str | None = None,
    observation_mode: Literal["oracle_pose", "rgbd_pose"] = "oracle_pose",
    runtime_root: Path | None = None,
) -> str:
    """返回任务/场景/机器人组合的运行配置名。"""

    candidates = robodojo_task_candidates()
    if task_name not in candidates:
        raise ValueError(f"RoboDojo 任务不存在或不可运行（不在候选集）：{task_name}")
    # 论文三任务的默认组合保持不变；传入任意覆盖项即走通用组合器。
    if scene_config is None and robot_config is None and observation_mode == "oracle_pose":
        if task_name == "push_T":
            return "essay2608_single_x5_right"
        if task_name == "pour_liquid_into_cup":
            return "essay2608_single_x5_left"
        if task_name == "sweep_blocks":
            return "arx_x5"
    task = candidates[task_name]
    scene_config = scene_config or task.scene_config
    if robot_config is None and task_name == "push_T":
        robot_config = "essay2608_single_x5_right"
    elif robot_config is None and task_name == "pour_liquid_into_cup":
        robot_config = "essay2608_single_x5_left"
    else:
        robot_config = robot_config or task.robot_config
    runtime_root = runtime_root or ROBODOJO_RUNTIME_ROOT
    if not (runtime_root / "env_cfg" / "robot").is_dir():
        runtime_root = prepare_robodojo_runtime()
    return ensure_runtime_env_config(
        runtime_root,
        task_name=task_name,
        scene_config=scene_config,
        robot_config=robot_config,
        observation_mode=observation_mode,
        camera_config=task.camera_config,
    )


def demonstration_environment_config_for(task_name: str) -> str:
    """返回官方演示的原始机器人配置，而不是论文评测时的活动臂配置。

    当前冻结演示全部来自上游 ``data/RoboDojo/<task>/arx_x5``。即使论文把
    ``push_T`` 和 ``pour_liquid_into_cup`` 归为单活动臂任务，回放补采也必须
    保留采集时的双 X5 实体、初始状态与碰撞环境，随后再提取活动臂训练样本。
    """

    if task_name not in robodojo_task_candidates():
        raise ValueError(f"任务不在 RoboDojo 可运行候选集：{task_name}")
    return "arx_x5"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_json(value) -> bool:
    if isinstance(value, dict):
        return all(_finite_json(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_json(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def audit_robodojo_capture(
    capture_path: str | Path,
    result_path: str | Path,
    task_name: str,
    layout_seed: int,
    layout_kind: str = "official_eval",
) -> dict:
    """审计 GUI 回放补采；只有原生任务成功的完整轨迹才可进入训练。"""

    candidates = robodojo_task_candidates()
    if task_name not in candidates:
        raise ValueError(f"任务不在 RoboDojo 可运行候选集中：{task_name}")
    capture_path = Path(capture_path).resolve()
    result_path = Path(result_path).resolve()
    reasons = []
    records = []
    if not capture_path.is_file() or capture_path.stat().st_size == 0:
        reasons.append("未生成非空 JSONL 补采文件")
    else:
        try:
            records = [
                json.loads(line)
                for line in capture_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            reasons.append(f"JSONL 无法解析：{error}")
    metadata = records[0] if records else {}
    steps = records[1:] if records else []
    if metadata.get("type") != "metadata" or metadata.get("schema") != (
        "essay2608.robodojo.gui_capture.v1"
    ):
        reasons.append("缺少受支持的补采元数据")
    if not steps:
        reasons.append("没有补采到任何仿真步")
    expected_indices = list(range(len(steps)))
    if [item.get("index") for item in steps] != expected_indices:
        reasons.append("补采步号不连续")
    required_patterns = candidates[task_name].task_frames
    observed_frames: set[str] = set()
    for item in steps:
        if item.get("type") != "step":
            reasons.append("JSONL 含非 step 记录")
            break
        frames = item.get("frames")
        if not isinstance(frames, dict) or not frames:
            reasons.append("至少一个仿真步缺少任务帧")
            break
        observed_frames.update(frames)
        if not _finite_json(item):
            reasons.append("补采轨迹含非有限数值")
            break
    missing_patterns = [
        pattern
        for pattern in required_patterns
        if not any(fnmatchcase(frame, pattern) for frame in observed_frames)
    ]
    if missing_patterns:
        reasons.append(f"缺少任务帧：{missing_patterns}")

    result = {}
    if not result_path.is_file():
        reasons.append("缺少 RoboDojo 原生结果文件")
    else:
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            reasons.append(f"结果文件无法解析：{error}")
    details = result.get("details", {}) if isinstance(result, dict) else {}
    native_success = (
        result.get("eval_time") == 1
        and len(details) == 1
        and all(bool(item.get("success")) for item in details.values())
    )
    if not native_success:
        reasons.append("RoboDojo 原生任务成功判据未通过")

    audit = {
        "schema": "essay2608.robodojo.capture_audit.v1",
        "accepted_for_training": not reasons,
        "task": task_name,
        "layout_kind": layout_kind,
        "layout_seed": int(layout_seed),
        "capture_path": str(capture_path),
        "capture_sha256": _sha256_file(capture_path) if capture_path.is_file() else None,
        "result_path": str(result_path),
        "result_sha256": _sha256_file(result_path) if result_path.is_file() else None,
        "records": len(records),
        "steps": len(steps),
        "observed_frames": sorted(observed_frames),
        "native_success": native_success,
        "reasons": reasons,
    }
    audit_path = capture_path.with_suffix(".audit.json")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit


def _push_t_mask_world_points(image, intrinsic, extrinsic, hue_minimum: int) -> tuple:
    """把头部相机中的紫色 T 块像素反投影到桌面平面。"""

    import cv2
    import numpy as np

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.asarray([hue_minimum, 150, 130], dtype=np.uint8),
        np.asarray([145, 255, 255], dtype=np.uint8),
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    candidates = [
        index
        for index in range(1, count)
        if 40 <= stats[index, cv2.CC_STAT_AREA] <= 2500
        and 100 < stats[index, cv2.CC_STAT_LEFT] < 600
        and 140 < stats[index, cv2.CC_STAT_TOP] < 340
    ]
    if not candidates:
        raise ValueError("画面中没有可靠的紫色 T 块连通域")
    component = max(candidates, key=lambda index: stats[index, cv2.CC_STAT_AREA])
    rows, columns = np.nonzero(labels == component)
    rays = np.stack(
        (
            (columns - intrinsic[0, 2]) / intrinsic[0, 0],
            -(rows - intrinsic[1, 2]) / intrinsic[1, 1],
            -np.ones(len(columns)),
        ),
        axis=1,
    )
    origin = extrinsic[:3, 3]
    directions = rays @ extrinsic[:3, :3].T
    scales = (0.7725 - origin[2]) / directions[:, 2]
    world_xy = (origin + directions * scales[:, None])[:, :2]
    return world_xy, int(stats[component, cv2.CC_STAT_AREA])


def _push_t_mask_pose(image, intrinsic, extrinsic, hue_minimum: int) -> tuple:
    import numpy as np

    world_xy, pixels = _push_t_mask_world_points(
        image, intrinsic, extrinsic, hue_minimum
    )
    centre = np.mean(world_xy, axis=0)
    centered = world_xy - centre
    _, eigenvectors = np.linalg.eigh(centered.T @ centered / len(centered))
    axis = eigenvectors[:, -1]
    projection = centered @ axis
    # T 块细长杆一侧相对面积质心伸得更远，用它消除 180° 歧义。
    if np.percentile(projection, 99) < -np.percentile(projection, 1):
        axis = -axis
    yaw = math.atan2(float(axis[1]), float(axis[0]))
    return centre, yaw, pixels, world_xy


def _fit_push_t_template(world_xy, local_template, centre, yaw) -> tuple:
    """用截尾双向 Chamfer 距离把官方资产轮廓模板配准到演示像素。"""

    import numpy as np
    from scipy.optimize import minimize
    from scipy.spatial import cKDTree

    observed = np.asarray(world_xy, dtype=np.float64)[:: max(len(world_xy) // 350, 1)]
    template = np.asarray(local_template, dtype=np.float64)[:: max(len(local_template) // 350, 1)]

    def transform(parameters):
        x, y, angle = parameters
        cosine, sine = math.cos(angle), math.sin(angle)
        rotation = np.asarray(((cosine, -sine), (sine, cosine)))
        return template @ rotation.T + np.asarray((x, y))

    def trimmed_mean(values):
        cutoff = np.percentile(values, 82)
        retained = values[values <= cutoff]
        return float(np.mean(retained)) if len(retained) else float("inf")

    def objective(parameters):
        predicted = transform(parameters)
        observed_to_predicted = cKDTree(predicted).query(observed, workers=1)[0]
        predicted_to_observed = cKDTree(observed).query(predicted, workers=1)[0]
        return trimmed_mean(observed_to_predicted) + trimmed_mean(predicted_to_observed)

    result = minimize(
        objective,
        np.asarray((*centre, yaw), dtype=np.float64),
        method="Powell",
        bounds=(
            (float(centre[0]) - 0.025, float(centre[0]) + 0.025),
            (float(centre[1]) - 0.025, float(centre[1]) + 0.025),
            (yaw - 0.45, yaw + 0.45),
        ),
        options={"xtol": 1.0e-6, "ftol": 1.0e-6, "maxiter": 120},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"T 块轮廓配准失败：{result.message}")
    return result.x[:2], float(result.x[2]), float(result.fun)


def reconstruct_push_t_source_layout(
    episode_path: str | Path,
    output_path: str | Path,
    template_path: str | Path | None = None,
    calibration_image_path: str | Path | None = None,
    calibration_layout_path: str | Path | None = None,
    calibration_video_path: str | Path | None = None,
    calibration_capture_path: str | Path | None = None,
    calibration_frame: int = 250,
) -> tuple[Path, Path]:
    """从官方 RGB+标定重建 push_T 演示源布局，并保存独立证据侧车。"""

    try:
        import cv2
        import h5py
        import numpy as np
    except ImportError as error:
        raise RuntimeError("push_T 源布局重建需要 h5py、OpenCV 和 NumPy") from error

    episode_path = Path(episode_path).resolve()
    output_path = Path(output_path).resolve()
    if not episode_path.is_file():
        raise FileNotFoundError(f"官方演示不存在：{episode_path}")
    if template_path is None:
        template_path = (
            ROBODOJO_ASSET_ROOT
            / "Eval_Layout"
            / "RoboDojo"
            / "arx_x5"
            / "0"
            / "push_T_0.json"
        )
    template_path = Path(template_path).resolve()
    if not template_path.is_file():
        raise FileNotFoundError(f"push_T 布局模板不存在：{template_path}")

    observations = []
    with h5py.File(episode_path, "r") as archive:
        required = (
            "vision/cam_head/colors",
            "vision/cam_head/intrinsic_matrix",
            "vision/cam_head/extrinsic_matrix",
        )
        missing = [name for name in required if name not in archive]
        if missing:
            raise ValueError(f"官方演示缺少布局重建字段：{missing}")
        colors = archive["vision/cam_head/colors"]
        intrinsic = archive["vision/cam_head/intrinsic_matrix"][()]
        extrinsics = archive["vision/cam_head/extrinsic_matrix"][()]
        initial_indices = range(0, min(20, len(colors)), 3)
        target_indices = range(max(len(colors) * 2 // 3, 0), len(colors))
        for phase, indices in (("initial", initial_indices), ("target", target_indices)):
            for index in indices:
                image = cv2.imdecode(
                    np.frombuffer(colors[index], dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if image is None:
                    continue
                for hue_minimum in (115, 118, 120):
                    try:
                        centre, yaw, pixels, points = _push_t_mask_pose(
                            image, intrinsic, extrinsics[index], hue_minimum
                        )
                    except ValueError:
                        continue
                    if not (-0.5 < centre[0] < 0.5 and -0.3 < centre[1] < -0.05):
                        continue
                    observations.append(
                        {
                            "phase": phase,
                            "frame": int(index),
                            "hue_minimum": hue_minimum,
                            "position": centre,
                            "yaw": yaw,
                            "pixels": pixels,
                            "points": points,
                        }
                    )
    initial = [item for item in observations if item["phase"] == "initial"]
    target = [item for item in observations if item["phase"] == "target"]
    # GUI 标定模板同样使用 H>=118；拟合时固定这一阈值，避免 H=115 把蓝色桌面高光
    # 并入轮廓。其余阈值只用于前面的可检测性检查。
    preferred_initial = [item for item in initial if item["hue_minimum"] == 118]
    preferred_target = [item for item in target if item["hue_minimum"] == 118]
    if len(preferred_initial) >= 3:
        initial = preferred_initial
    if len(preferred_target) >= 3:
        target = preferred_target
    if len(initial) < 3 or len(target) < 3:
        raise RuntimeError(
            f"push_T 源布局重建证据不足：initial={len(initial)}, target={len(target)}"
        )
    # 遮挡会让 T 轮廓主轴偶尔翻转约 180°。先以像素面积加权选择占优角度簇，再在簇内
    # 取面积最大的 15 个样本；末段的这些样本对应 T 块完全显露且已落在目标垫上。
    def dominant_orientation_cluster(items):
        def angular_distance(left, right):
            return abs(math.atan2(math.sin(left - right), math.cos(left - right)))

        anchor = max(
            items,
            key=lambda candidate: sum(
                item["pixels"]
                for item in items
                if angular_distance(item["yaw"], candidate["yaw"]) < 0.35
            ),
        )
        cluster = [
            item
            for item in items
            if angular_distance(item["yaw"], anchor["yaw"]) < 0.35
        ]
        # 多个 HSV 阈值来自同一画面，不把它们伪装成独立重复测量。
        best_by_frame = {}
        for item in cluster:
            current = best_by_frame.get(item["frame"])
            if current is None or item["pixels"] > current["pixels"]:
                best_by_frame[item["frame"]] = item
        return sorted(
            best_by_frame.values(), key=lambda item: item["pixels"], reverse=True
        )[:15]

    initial = dominant_orientation_cluster(initial)
    target = dominant_orientation_cluster(target)

    calibration = None
    legacy_calibration = calibration_image_path is not None or calibration_layout_path is not None
    replay_calibration = calibration_video_path is not None or calibration_capture_path is not None
    if legacy_calibration and replay_calibration:
        raise ValueError("静态图标定与 GUI 回放标定不能同时使用")
    if replay_calibration:
        if calibration_video_path is None or calibration_capture_path is None:
            raise ValueError("calibration_video_path 与 calibration_capture_path 必须同时提供")
        calibration_video_path = Path(calibration_video_path).resolve()
        calibration_capture_path = Path(calibration_capture_path).resolve()
        if not calibration_video_path.is_file() or not calibration_capture_path.is_file():
            raise FileNotFoundError("GUI 标定视频或对应真值补采文件不存在")
        video = cv2.VideoCapture(str(calibration_video_path))
        video.set(cv2.CAP_PROP_POS_FRAMES, calibration_frame)
        decoded, calibration_image = video.read()
        video.release()
        if not decoded or calibration_image is None:
            raise ValueError(f"GUI 标定视频无法读取第 {calibration_frame} 帧")
        calibration_records = [
            json.loads(line)
            for line in calibration_capture_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        matching = [
            record
            for record in calibration_records
            if record.get("type") == "step" and record.get("index") == calibration_frame
        ]
        if len(matching) != 1:
            raise ValueError(
                f"GUI 真值补采中第 {calibration_frame} 帧数量异常：{len(matching)}"
            )
        reference_pose = matching[0].get("frames", {}).get("t")
        if not isinstance(reference_pose, list) or len(reference_pose) != 7:
            raise ValueError("GUI 真值补采标定帧缺少七维 t 位姿")
        reference_position = np.asarray(reference_pose[:2], dtype=np.float64)
        reference_orientation = reference_pose[3:]
        calibration_provenance = {
            "video": str(calibration_video_path),
            "video_sha256": _sha256_file(calibration_video_path),
            "capture": str(calibration_capture_path),
            "capture_sha256": _sha256_file(calibration_capture_path),
            "frame": calibration_frame,
            "native_success_required": False,
            "usage": "仅用仿真真值标定官方资产轮廓，不作为训练演示",
        }
    elif legacy_calibration:
        if calibration_image_path is None or calibration_layout_path is None:
            raise ValueError("calibration_image_path 与 calibration_layout_path 必须同时提供")
        calibration_image_path = Path(calibration_image_path).resolve()
        calibration_layout_path = Path(calibration_layout_path).resolve()
        calibration_image = cv2.imread(str(calibration_image_path), cv2.IMREAD_COLOR)
        if calibration_image is None or not calibration_layout_path.is_file():
            raise FileNotFoundError("T 块 GUI 轮廓标定图或对应布局不存在")
        calibration_layout = json.loads(calibration_layout_path.read_text(encoding="utf-8"))
        reference = calibration_layout["Rigid"]["t"][0]
        reference_position = np.asarray(reference["default_pos"][:2], dtype=np.float64)
        reference_orientation = reference["default_ori"]
        calibration_provenance = {
            "image": str(calibration_image_path),
            "image_sha256": _sha256_file(calibration_image_path),
            "layout": str(calibration_layout_path),
            "layout_sha256": _sha256_file(calibration_layout_path),
        }

    if legacy_calibration or replay_calibration:
        calibration_world, _ = _push_t_mask_world_points(
            calibration_image, intrinsic, extrinsics[0], 118
        )
        reference_yaw = 2.0 * math.atan2(reference_orientation[3], reference_orientation[0])
        cosine, sine = math.cos(reference_yaw), math.sin(reference_yaw)
        reference_rotation = np.asarray(((cosine, -sine), (sine, cosine)))
        local_template = (calibration_world - reference_position) @ reference_rotation
        joint_fits = {}
        for phase, items in (("initial", initial), ("target", target)):
            seed_position = np.median(
                np.stack([item["position"] for item in items]), axis=0
            )
            seed_yaws = np.asarray([item["yaw"] for item in items])
            seed_yaw = math.atan2(
                float(np.mean(np.sin(seed_yaws))), float(np.mean(np.cos(seed_yaws)))
            )
            position, fitted_yaw, loss = _fit_push_t_template(
                np.concatenate([item["points"] for item in items], axis=0),
                local_template,
                seed_position,
                seed_yaw,
            )
            joint_fits[phase] = {
                "position": position,
                "yaw": fitted_yaw,
                "loss": loss,
            }
        for item in [*initial, *target]:
            position, fitted_yaw, loss = _fit_push_t_template(
                item["points"], local_template, item["position"], item["yaw"]
            )
            item["position"] = position
            item["yaw"] = fitted_yaw
            item["fit_loss"] = loss
        calibration = {
            **calibration_provenance,
            "joint_fit_loss": {
                phase: values["loss"] for phase, values in joint_fits.items()
            },
        }

    def aggregate(items):
        positions = np.stack([item["position"] for item in items])
        yaws = np.asarray([item["yaw"] for item in items])
        position = np.median(positions, axis=0)
        yaw = math.atan2(float(np.mean(np.sin(yaws))), float(np.mean(np.cos(yaws))))
        return position, yaw, np.std(positions, axis=0), float(np.std(np.unwrap(yaws)))

    initial_position, initial_yaw, initial_std, initial_yaw_std = aggregate(initial)
    target_position, target_yaw, target_std, target_yaw_std = aggregate(target)
    if calibration is not None:
        initial_position = joint_fits["initial"]["position"]
        initial_yaw = joint_fits["initial"]["yaw"]
        target_position = joint_fits["target"]["position"]
        target_yaw = joint_fits["target"]["yaw"]
    layout = json.loads(template_path.read_text(encoding="utf-8"))
    rigid = layout["Rigid"]["t"][0]
    cushion = layout["Geometry"]["t_cushion"][0]
    # 保留官方布局模板选择的资产类别。HDF5 只提供轨迹和图像，重建只应替换
    # 位姿；覆盖 category_idx 会把演示换成另一套官方 T 资产，破坏接触动力学。
    rigid["default_pos"] = [*initial_position.tolist(), 0.7725]
    rigid["default_ori"] = [
        math.cos(initial_yaw / 2.0),
        0.0,
        0.0,
        math.sin(initial_yaw / 2.0),
    ]
    cushion["default_pos"] = [*target_position.tolist(), 0.765]
    cushion["default_ori"] = [
        math.cos(target_yaw / 2.0),
        0.0,
        0.0,
        math.sin(target_yaw / 2.0),
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(layout, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    evidence = {
        "schema": "essay2608.robodojo.source_layout_reconstruction.v1",
        "task": "push_T",
        "source_episode": str(episode_path),
        "source_sha256": _sha256_file(episode_path),
        "template": str(template_path),
        "method": (
            "gui_asset_silhouette_se2_registration"
            if calibration is not None
            else "head_rgb_hsv_mask_and_table_plane_backprojection"
        ),
        "calibration": calibration,
        "initial": {
            "position_xy": initial_position.tolist(),
            "yaw": initial_yaw,
            "position_std": initial_std.tolist(),
            "yaw_std": initial_yaw_std,
            "samples": len(initial),
        },
        "target": {
            "position_xy": target_position.tolist(),
            "yaw": target_yaw,
            "position_std": target_std.tolist(),
            "yaw_std": target_yaw_std,
            "samples": len(target),
            "frames": sorted({item["frame"] for item in target}),
        },
        "training_admissibility": (
            "本文件只是源布局估计；只有对应 GUI 回放的 capture_audit "
            "accepted_for_training=true 后才可训练"
        ),
    }
    evidence_path = output_path.with_suffix(".reconstruction.json")
    evidence_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output_path, evidence_path


def _parse_additional_info(value: str) -> dict[str, str]:
    fields = {}
    for item in value.split(","):
        if "=" in item:
            key, field_value = item.split("=", 1)
            fields[key] = field_value
    return fields


def collect_robodojo_results(paths: RoboDojoPaths = RoboDojoPaths()) -> list[dict]:
    """读取所有已注册 RoboDojo 任务的原生 ``_result.json``。"""

    rows = []
    tasks = robodojo_task_candidates(paths)
    benchmark_root = paths.result_root / "RoboDojo"
    if not benchmark_root.is_dir():
        return rows
    for result_path in sorted(benchmark_root.glob("*/*/*/*/*/_result.json")):
        relative = result_path.relative_to(benchmark_root)
        task_name, policy_directory, config_name, seed_info, run_id, _ = relative.parts
        if task_name not in tasks:
            continue
        seed_text, separator, additional = seed_info.partition("_")
        if not separator:
            continue
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        details = payload.get("details", {})
        successes = sum(
            bool(item.get("success")) for item in details.values() if isinstance(item, dict)
        )
        fields = _parse_additional_info(additional)
        rows.append(
            {
                "task": task_name,
                "arm_mode": tasks[task_name].arm_mode,
                "method": fields.get("method", policy_directory.removeprefix("essay2608_")),
                "condition": fields.get("condition", "unknown"),
                "observation_mode": fields.get("observation", "unknown"),
                "seed": int(seed_text),
                "checkpoint": fields.get("ckpt", "unknown"),
                "gui": fields.get("gui") == "true",
                "config": config_name,
                "run_id": run_id,
                "episodes": int(payload.get("eval_time", len(details))),
                "successes": int(successes),
                "success_rate": float(payload.get("success_rate", 0.0)),
                "score": float(payload.get("score", 0.0)),
                "result_path": str(result_path),
            }
        )
    return rows


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    z = 1.959963984540054
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    radius = (
        z / denominator * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total))
    )
    return center - radius, center + radius


def write_robodojo_paper_table(
    output_directory: str | Path = PROJECT_ROOT / "results" / "robodojo",
    paths: RoboDojoPaths = RoboDojoPaths(),
) -> tuple[Path, Path]:
    """生成逐次 CSV 和三种子论文表；不完整实验明确显示为待评测。"""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    rows = collect_robodojo_results(paths)
    csv_path = output_directory / "summary.csv"
    fieldnames = [
        "task",
        "arm_mode",
        "method",
        "condition",
        "observation_mode",
        "seed",
        "checkpoint",
        "gui",
        "config",
        "run_id",
        "episodes",
        "successes",
        "success_rate",
        "score",
        "result_path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    markdown_path = output_directory / "paper_table.md"
    lines = [
        "# RoboDojo GUI 仿真对比表",
        "",
        "协议：每个任务、方法使用种子 0/1/2，每种子 50 回合；成功率按种子报告均值±样本标准差，另给 150 回合合并 Wilson 95% CI。所有纳入结果必须标记 `gui=true`。",
        "",
        "| 机械臂 | 任务 | 方法 | 条件 | 成功率（%） | RoboDojo 分数 | 95% CI（%） | 种子/回合 | 状态 |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for task_name, candidate in TASK_CANDIDATES.items():
        for method in ("dp", "midigap", "dynamac"):
            matching = [
                row
                for row in rows
                if row["task"] == task_name
                and row["method"] == method
                and row["condition"] == "static"
                and row["gui"]
                and row["episodes"] >= candidate.eval_episodes
            ]
            by_seed: dict[int, list[dict]] = {}
            for row in matching:
                by_seed.setdefault(row["seed"], []).append(row)
            unambiguous = all(len(by_seed.get(seed, [])) == 1 for seed in (0, 1, 2))
            if unambiguous:
                selected = [by_seed[seed][0] for seed in (0, 1, 2)]
                rates = [100.0 * row["success_rate"] for row in selected]
                scores = [row["score"] for row in selected]
                successes = sum(row["successes"] for row in selected)
                episodes = sum(row["episodes"] for row in selected)
                low, high = _wilson_interval(successes, episodes)
                rate_text = f"{statistics.mean(rates):.1f} ± {statistics.stdev(rates):.1f}"
                score_text = f"{statistics.mean(scores):.1f} ± {statistics.stdev(scores):.1f}"
                ci_text = f"[{100.0 * low:.1f}, {100.0 * high:.1f}]"
                sample_text = f"3/{episodes}"
                state = "完整"
            else:
                complete_seeds = sum(len(by_seed.get(seed, [])) == 1 for seed in (0, 1, 2))
                duplicates = sum(max(len(values) - 1, 0) for values in by_seed.values())
                rate_text = score_text = ci_text = "—"
                sample_text = f"{complete_seeds}/3"
                state = "待评测" if duplicates == 0 else f"重复运行待裁决（{duplicates}）"
            lines.append(
                f"| {candidate.arm_mode} | `{task_name}` | {method} | static | "
                f"{rate_text} | {score_text} | {ci_text} | {sample_text} | {state} |"
            )
    lines.extend(
        [
            "",
            "动态 `smooth`/`teleport` 结果作为本论文扩展表单独聚合，不与 RoboDojo 原生静态布局混算。双臂 DP 当前为左右臂独立 state DP；在联合动作 DP 完成前不得把它标成论文同配置基线。",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, markdown_path


__all__ = [
    "ROBODOJO_ASSET_ROOT",
    "ROBODOJO_CAPTURE_ROOT",
    "ROBODOJO_DEMO_ROOT",
    "ROBODOJO_OFFICIAL_ROOT",
    "ROBODOJO_RESULT_ROOT",
    "ROBODOJO_ROOT",
    "ROBODOJO_RUNTIME_ROOT",
    "ROBODOJO_SOURCE_LAYOUT_ROOT",
    "RoboDojoPaths",
    "RoboDojoPolicyDemonstrations",
    "RoboDojoPolicyCandidate",
    "RoboDojoRobotCandidate",
    "RoboDojoSceneCandidate",
    "RoboDojoTaskCandidate",
    "TASK_CANDIDATES",
    "audit_robodojo_capture",
    "download_robodojo_assets",
    "download_robodojo_demonstrations",
    "ensure_runtime_env_config",
    "robodojo_resource_catalog",
    "robodojo_task_candidates",
    "task_asset_patterns",
    "demonstration_environment_config_for",
    "environment_config_for",
    "load_robodojo_policy_demonstrations",
    "collect_robodojo_results",
    "prepare_robodojo_runtime",
    "reconstruct_push_t_source_layout",
    "robodojo_task_catalog",
    "sync_robodojo_official_snapshot",
    "robodojo_status",
    "write_robodojo_paper_table",
]
