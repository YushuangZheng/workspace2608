#!/usr/bin/env python3
"""论文复现实验的唯一命令行入口。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlparse

from essay2608.data import (
    ROBODOJO_CAPTURE_ROOT,
    ROBODOJO_DEMO_ROOT,
    ROBODOJO_SOURCE_LAYOUT_ROOT,
    TASK_CANDIDATES,
    audit_robodojo_capture,
    demonstration_environment_config_for,
    download_robodojo_assets,
    download_robodojo_demonstrations,
    environment_config_for,
    load_demonstrations,
    load_robodojo_policy_demonstrations,
    prepare_robodojo_runtime,
    reconstruct_push_t_source_layout,
    robodojo_resource_catalog,
    robodojo_status,
    robodojo_task_candidates,
    robodojo_task_catalog,
    sync_robodojo_official_snapshot,
    write_robodojo_paper_table,
)
from essay2608.policy import (
    BimanualDynaMAC,
    DiffusionPolicy,
    DynaMAC,
    DynaMACConfig,
    TaskParameterizedMiDiGaP,
    serve_robodojo_policy,
    serve_robodojo_replay_capture,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/dynamac_demos.npz"))
    parser.add_argument("--config", type=Path, default=Path("configs/dynamac.json"))
    commands = parser.add_subparsers(dest="command", required=True)

    fit = commands.add_parser("fit", help="从演示拟合并保存 checkpoint")
    fit.add_argument("--policy", choices=("dp", "midigap", "dynamac"), required=True)
    fit.add_argument("--task", choices=("single", "bimanual"), required=True)
    fit.add_argument("--output", type=Path, required=True)

    inspect = commands.add_parser("inspect", help="只读检查 checkpoint")
    inspect.add_argument("--policy", choices=("midigap", "dynamac"), required=True)
    inspect.add_argument("checkpoint", type=Path)

    commands.add_parser("verify", help="拟合随附数据并打印结构摘要，不保存模型")

    robodojo = commands.add_parser("robodojo", help="RoboDojo 接入、资产和 GUI 评测")
    robodojo_commands = robodojo.add_subparsers(dest="robodojo_command", required=True)
    robodojo_commands.add_parser("status", help="只读检查上游、资产和运行层")
    robodojo_commands.add_parser("catalog", help="列出上游全部可运行任务候选")
    robodojo_commands.add_parser("resources", help="列出任务、场景、机器人和 policy 资源")
    robodojo_commands.add_parser("prepare", help="生成可丢弃运行层")
    official_sync = robodojo_commands.add_parser(
        "official-sync", help="同步官方 HF 根文件与 Assets，排除 data/ckpt"
    )
    official_sync.add_argument("--revision", default="main")
    robodojo_commands.add_parser("table", help="从原始结果生成论文级 CSV/Markdown 表")
    assets = robodojo_commands.add_parser("assets", help="按任务下载官方资产；--all 覆盖全部任务")
    assets.add_argument("--tasks", nargs="+")
    assets.add_argument("--all", action="store_true", dest="all_tasks")
    demos = robodojo_commands.add_parser("demos", help="按任务下载冻结的官方专家演示")
    demos.add_argument("--tasks", nargs="+")
    demos.add_argument("--all", action="store_true", dest="all_tasks")
    demos.add_argument("--episodes", type=int, default=5)
    reconstruct = robodojo_commands.add_parser(
        "reconstruct-push", help="从官方 RGB 与标定重建 push_T 演示源布局"
    )
    reconstruct.add_argument("--episode", type=Path, required=True)
    reconstruct.add_argument("--output", type=Path, required=True)
    reconstruct.add_argument("--calibration-image", type=Path)
    reconstruct.add_argument("--calibration-layout", type=Path)
    reconstruct.add_argument("--calibration-video", type=Path)
    reconstruct.add_argument("--calibration-capture", type=Path)
    reconstruct.add_argument("--calibration-frame", type=int, default=250)

    robodojo_fit = robodojo_commands.add_parser(
        "fit", help="从冻结官方演示拟合任务专用策略"
    )
    robodojo_fit.add_argument("--policy", choices=("dp", "midigap", "dynamac"), required=True)
    robodojo_fit.add_argument("--task", required=True)
    robodojo_fit.add_argument(
        "--episodes", type=int, default=5, help="使用前 N 条演示（默认 5；DP 正式训练可设为 100）"
    )
    robodojo_fit.add_argument(
        "--capture-root",
        type=Path,
        help="GUI 补采 JSONL 根目录；提供后使用每时刻 Oracle/RGB-D 任务帧",
    )
    robodojo_fit.add_argument("--output", type=Path, required=True)

    capture_server = robodojo_commands.add_parser("capture-server", help=argparse.SUPPRESS)
    capture_server.add_argument("--episode", type=Path, required=True)
    capture_server.add_argument("--output", type=Path, required=True)
    capture_server.add_argument("--arm-mode", choices=("single", "bimanual"), required=True)
    capture_server.add_argument("--active-side", choices=("auto", "left", "right"), default="auto")
    capture_server.add_argument(
        "--replay-action", choices=("ee_pose", "joint"), default="ee_pose",
        help="回放 HDF5 中的末端位姿或关节动作（默认 ee_pose）",
    )
    capture_server.add_argument("--host", default="127.0.0.1")
    capture_server.add_argument("--port", type=int, required=True)

    capture = robodojo_commands.add_parser("capture", help="GUI 回放官方动作并同步补采任务帧真值")
    capture.add_argument("--task", required=True)
    capture.add_argument("--episode", type=Path, required=True)
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--source-layout", type=Path, required=True)
    capture.add_argument("--seed", type=int, default=0)
    capture.add_argument("--active-side", choices=("auto", "left", "right"), default="auto")
    capture.add_argument(
        "--replay-action", choices=("ee_pose", "joint"), default="ee_pose",
        help="回放 HDF5 中的末端位姿或关节动作（默认 ee_pose）",
    )
    capture.add_argument("--device-id", type=int, default=0)
    capture.add_argument(
        "--observation-mode", choices=("oracle_pose", "rgbd_pose"), default="oracle_pose"
    )

    capture_batch = robodojo_commands.add_parser(
        "capture-batch", help="自动逐条 GUI 回放官方演示并验收真值帧"
    )
    capture_batch.add_argument("--task", required=True)
    capture_batch.add_argument("--episodes", type=int, default=5)
    capture_batch.add_argument("--start-index", type=int, default=0)
    capture_batch.add_argument("--output-root", type=Path, default=ROBODOJO_CAPTURE_ROOT)
    capture_batch.add_argument(
        "--source-layout-root", type=Path, default=ROBODOJO_SOURCE_LAYOUT_ROOT
    )
    capture_batch.add_argument(
        "--auto-reconstruct",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="push_T 缺布局时自动从 HDF5 RGB 重建（默认开启）",
    )
    capture_batch.add_argument("--seed", type=int, default=0)
    capture_batch.add_argument("--active-side", choices=("auto", "left", "right"), default="auto")
    capture_batch.add_argument(
        "--replay-action", choices=("ee_pose", "joint"), default="ee_pose",
        help="回放 HDF5 中的末端位姿或关节动作（默认 ee_pose）",
    )
    capture_batch.add_argument("--device-id", type=int, default=0)
    capture_batch.add_argument(
        "--observation-mode", choices=("oracle_pose", "rgbd_pose"), default="oracle_pose"
    )
    capture_batch.add_argument(
        "--continue-on-error", action="store_true", help="记录失败并继续后续演示"
    )

    server = robodojo_commands.add_parser("policy-server", help=argparse.SUPPRESS)
    _add_policy_eval_arguments(server, include_eval=False)
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, required=True)

    evaluate = robodojo_commands.add_parser("eval", help="启动策略服务器和 GUI 正式评测")
    _add_policy_eval_arguments(evaluate, include_eval=True)
    external = robodojo_commands.add_parser(
        "external-eval", help="连接任意 XPolicyLab policy server，使用项目 GUI 客户端评测"
    )
    external.add_argument("--policy-name", required=True)
    external.add_argument("--policy-server-url", required=True)
    external.add_argument("--task", required=True)
    external.add_argument("--env-cfg")
    external.add_argument("--scene")
    external.add_argument("--robot")
    external.add_argument(
        "--observation-mode", choices=("oracle_pose", "rgbd_pose"), default="oracle_pose"
    )
    external.add_argument("--seed", type=int, default=0)
    external.add_argument("--episodes", type=int, default=1)
    external.add_argument("--device-id", type=int, default=0)
    external.add_argument("--additional-info", default="external=true")
    return parser.parse_args()


def _add_policy_eval_arguments(parser: argparse.ArgumentParser, include_eval: bool) -> None:
    parser.add_argument("--policy", choices=("dp", "midigap", "dynamac"), required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scene")
    parser.add_argument("--robot")
    parser.add_argument(
        "--observation-mode", choices=("oracle_pose", "rgbd_pose"), default="oracle_pose"
    )
    if include_eval:
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--episodes", type=int)
        parser.add_argument("--device-id", type=int, default=0)
        parser.add_argument(
            "--condition", choices=("static", "smooth", "teleport"), default="static"
        )


def load_config(path: Path) -> DynaMACConfig:
    return DynaMACConfig(**json.loads(path.read_text(encoding="utf-8")))


def compact_summary(policy: DynaMAC) -> dict:
    return {
        "fingerprint": policy.fingerprint(),
        "frames": list(policy.frame_names),
        "map_modal_path": list(policy._select_mode_path("map")),
        "skills": [
            {
                "label": skill.label,
                "duration": skill.duration,
                "modes": len(skill.mode_priors),
                "selected_frames": list(skill.selected_frames),
                "linked_frames": [
                    name for name, values in skill.link_diagnostics.items() if values["linked"]
                ],
            }
            for skill in policy.skills
        ],
    }


def _fit_policy(args: argparse.Namespace) -> None:
    bundle = load_demonstrations(args.data)
    config = load_config(args.config)
    policy_class = {
        "dynamac": DynaMAC,
        "midigap": TaskParameterizedMiDiGaP,
        "dp": DiffusionPolicy,
    }[args.policy]
    policy_config = config if args.policy != "dp" else None

    def create_policy():
        return policy_class(policy_config) if policy_config is not None else policy_class()

    if args.task == "single":
        policy = create_policy().fit(bundle.single_arm)
        policy.save(args.output)
        summary = (
            compact_summary(policy)
            if args.policy != "dp"
            else {
                "policy": policy.name,
                "frames": list(policy.frame_names),
                "skills": list(policy.skill_sequence),
                "scope": "独立 state U-Net DP 复现",
            }
        )
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        if args.policy == "dynamac":
            policy = BimanualDynaMAC(config=config).fit(bundle.left_arm, bundle.right_arm)
            left, right = policy.left, policy.right
        else:
            left = create_policy().fit(bundle.left_arm)
            right = create_policy().fit(bundle.right_arm)
        left.save(args.output / "left.npz")
        right.save(args.output / "right.npz")
        summary = {
            "left": compact_summary(left) if args.policy != "dp" else left.name,
            "right": compact_summary(right) if args.policy != "dp" else right.name,
            "dual_dp_protocol": (
                "左右臂独立 state DP；正式论文表必须单列，不能冒充联合动作 DP"
                if args.policy == "dp"
                else None
            ),
        }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _checkpoint_digest(path: Path, arm_mode: str) -> str:
    files = [path] if arm_mode == "single" else [path / "left.npz", path / "right.npz"]
    digest = hashlib.sha256()
    for file_path in files:
        if not file_path.is_file():
            raise FileNotFoundError(f"checkpoint 不存在：{file_path}")
        with file_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()[:12]


def _run_robodojo_fit(args: argparse.Namespace) -> None:
    bundle = load_robodojo_policy_demonstrations(
        args.task,
        episode_count=args.episodes,
        capture_root=args.capture_root,
    )
    arm_mode = robodojo_task_candidates()[args.task].arm_mode
    config = load_config(args.config)
    policy_class = {
        "dp": DiffusionPolicy,
        "midigap": TaskParameterizedMiDiGaP,
        "dynamac": DynaMAC,
    }[args.policy]

    def create_policy():
        return policy_class() if args.policy == "dp" else policy_class(config)

    output = args.output.resolve()
    if arm_mode == "single":
        policy = create_policy().fit(bundle.single_arm)
        policy.save(output)
        summary = (
            compact_summary(policy)
            if args.policy != "dp"
            else {
                "policy": policy.name,
                "frames": list(policy.frame_names),
                "skills": list(policy.skill_sequence),
            }
        )
        provenance_path = output.with_suffix(".training.json")
    else:
        output.mkdir(parents=True, exist_ok=True)
        if args.policy == "dynamac":
            coupled = BimanualDynaMAC(config=config).fit(bundle.left_arm, bundle.right_arm)
            left, right = coupled.left, coupled.right
        else:
            left = create_policy().fit(bundle.left_arm)
            right = create_policy().fit(bundle.right_arm)
        left.save(output / "left.npz")
        right.save(output / "right.npz")
        summary = {
            "left": compact_summary(left) if args.policy != "dp" else left.name,
            "right": compact_summary(right) if args.policy != "dp" else right.name,
            "dual_dp_protocol": (
                "左右臂独立 state DP；不标成联合动作 DP" if args.policy == "dp" else None
            ),
        }
        provenance_path = output / "training.json"

    provenance = {
        **bundle.metadata,
        "policy": args.policy,
        "episodes": args.episodes,
        "capture_root": str(args.capture_root.resolve()) if args.capture_root else None,
        "checkpoint": str(output),
        "checkpoint_id": _checkpoint_digest(output, arm_mode),
        "summary": summary,
    }
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(provenance, ensure_ascii=False, indent=2))


def _free_port() -> int:
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        return int(server.getsockname()[1])


def _wait_for_server(process: subprocess.Popen, port: int, timeout: float = 120.0) -> None:
    from websockets.sync.client import connect

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"策略服务器提前退出，returncode={process.returncode}")
        try:
            with connect(
                f"ws://127.0.0.1:{port}",
                open_timeout=0.5,
                close_timeout=0.5,
                proxy=None,
            ):
                return
        except Exception:
            time.sleep(0.25)
    raise TimeoutError("等待策略服务器超时")


def _validate_layout_seed(runtime: Path, env_config: str, seed: int) -> None:
    """在启动 Isaac Sim 前给出可读的布局种子错误。"""

    layout_dir = runtime / "Assets" / "Eval_Layout" / "RoboDojo" / env_config / str(seed)
    if layout_dir.is_dir():
        return
    parent = layout_dir.parent
    available = sorted(
        int(path.name) for path in parent.glob("*") if path.is_dir() and path.name.isdigit()
    ) if parent.is_dir() else []
    hint = f"；可用种子：{available}" if available else "；该环境没有已下载布局"
    raise ValueError(f"任务布局种子不存在：env_cfg={env_config}, seed={seed}{hint}")


def _run_robodojo_eval(args: argparse.Namespace) -> None:
    if not os.environ.get("DISPLAY"):
        raise RuntimeError("未检测到 DISPLAY；本项目禁止无头正式评测")
    if args.episodes is not None and args.episodes < 1:
        raise ValueError("episodes 必须为正")
    runtime = prepare_robodojo_runtime()
    candidates = robodojo_task_candidates()
    if args.task not in candidates:
        raise ValueError(f"RoboDojo 任务不存在或不可运行：{args.task}")
    status = robodojo_status()
    if not status["common_assets_ready"] or not status["task_assets"].get(args.task, False):
        raise RuntimeError(
            f"任务 {args.task} 的 RoboDojo 资产未准备完整，请先运行 robodojo assets --tasks {args.task}"
        )
    candidate = candidates[args.task]
    env_config = environment_config_for(
        args.task,
        scene_config=args.scene,
        robot_config=args.robot,
        observation_mode=args.observation_mode,
        runtime_root=runtime,
    )
    _validate_layout_seed(runtime, env_config, args.seed)
    checkpoint = args.checkpoint.resolve()
    checkpoint_id = _checkpoint_digest(checkpoint, candidate.arm_mode)
    episodes = args.episodes or candidate.eval_episodes
    port = _free_port()
    server_command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "robodojo",
        "policy-server",
        "--policy",
        args.policy,
        "--task",
        args.task,
        "--checkpoint",
        str(checkpoint),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.device_id),
            "EVAL_NUM": str(episodes),
            "ESSAY2608_PERTURBATION": args.condition,
            "ESSAY2608_OBSERVATION_MODE": args.observation_mode,
            "ESSAY2608_ROBODOJO_RUNTIME": str(runtime),
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "PYTHONUNBUFFERED": "1",
        }
    )
    server = subprocess.Popen(server_command, cwd=PROJECT_ROOT, env=environment)
    try:
        _wait_for_server(server, port)
        additional_info = (
            f"method={args.policy},condition={args.condition},"
            f"observation={args.observation_mode},scene={args.scene or candidate.scene_config},"
            f"robot={args.robot or candidate.robot_config},ckpt={checkpoint_id},gui=true"
        )
        client_command = [
            sys.executable,
            "-m",
            "essay2608.data.robodojo_gui",
            "--task_name",
            args.task,
            "--num_envs",
            "1",
            "--env_cfg_type",
            env_config,
            "--enable_cameras",
            "--device_id",
            "0",
            "--policy_name",
            f"essay2608_{args.policy}",
            "--port",
            str(port),
            "--host",
            "127.0.0.1",
            "--protocol",
            "ws",
            "--policy_server_url",
            f"ws://127.0.0.1:{port}",
            "--additional_info",
            additional_info,
            "--seed",
            str(args.seed),
            "--kit_args",
            " --enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera",
        ]
        print(
            f"[GUI 评测] task={args.task} policy={args.policy} seed={args.seed} "
            f"episodes={episodes} condition={args.condition}",
            flush=True,
        )
        subprocess.run(client_command, cwd=runtime, env=environment, check=True)
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()


def _run_robodojo_external_eval(args: argparse.Namespace) -> None:
    """连接上游/用户自己的 WebSocket policy server，但始终使用项目 GUI 客户端。"""

    if not os.environ.get("DISPLAY"):
        raise RuntimeError("未检测到 DISPLAY；本项目禁止无头正式评测")
    if args.episodes < 1:
        raise ValueError("episodes 必须为正")
    parsed = urlparse(args.policy_server_url)
    if parsed.scheme != "ws" or not parsed.hostname or parsed.port is None:
        raise ValueError("policy-server-url 必须是 ws://host:port")
    candidates = robodojo_task_candidates()
    if args.task not in candidates:
        raise ValueError(f"RoboDojo 任务不存在或不可运行：{args.task}")
    runtime = prepare_robodojo_runtime()
    candidate = candidates[args.task]
    env_cfg = args.env_cfg or environment_config_for(
        args.task,
        scene_config=args.scene,
        robot_config=args.robot,
        observation_mode=args.observation_mode,
        runtime_root=runtime,
    )
    _validate_layout_seed(runtime, env_cfg, args.seed)
    environment = os.environ.copy()
    environment.update(
        {
            "EVAL_NUM": str(args.episodes),
            "ESSAY2608_OBSERVATION_MODE": args.observation_mode,
            "ESSAY2608_ROBODOJO_RUNTIME": str(runtime),
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "PYTHONUNBUFFERED": "1",
        }
    )
    additional_info = (
        f"{args.additional_info},observation={args.observation_mode},"
        f"scene={args.scene or candidate.scene_config},robot={args.robot or candidate.robot_config},gui=true"
    )
    client_command = [
        sys.executable,
        "-m",
        "essay2608.data.robodojo_gui",
        "--task_name",
        args.task,
        "--num_envs",
        "1",
        "--env_cfg_type",
        env_cfg,
        "--enable_cameras",
        "--device_id",
        str(args.device_id),
        "--policy_name",
        args.policy_name,
        "--port",
        str(parsed.port),
        "--host",
        parsed.hostname,
        "--protocol",
        "ws",
        "--policy_server_url",
        args.policy_server_url,
        "--additional_info",
        additional_info,
        "--seed",
        str(args.seed),
        "--kit_args",
        " --enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera",
    ]
    print(
        f"[GUI 外部评测] task={args.task} policy={args.policy_name} "
        f"episodes={args.episodes} observation={args.observation_mode} env_cfg={env_cfg}",
        flush=True,
    )
    subprocess.run(client_command, cwd=runtime, env=environment, check=True)


def _run_robodojo_capture(args: argparse.Namespace) -> None:
    if not os.environ.get("DISPLAY"):
        raise RuntimeError("未检测到 DISPLAY；RoboDojo 回放补采必须开 GUI")
    runtime = prepare_robodojo_runtime()
    status = robodojo_status()
    if not status["common_assets_ready"] or not status["task_assets"].get(args.task, False):
        raise RuntimeError(
            f"任务 {args.task} 的 RoboDojo 资产未准备完整，请先运行 robodojo assets --tasks {args.task}"
        )
    episode = args.episode.resolve()
    if not episode.is_file():
        raise FileNotFoundError(f"官方演示不存在：{episode}")
    output = args.output.resolve()
    source_layout = args.source_layout.resolve()
    if not source_layout.is_file():
        raise FileNotFoundError(f"演示源布局不存在：{source_layout}")
    # 冻结的官方演示均由 arx_x5 双臂场景采集。这里必须按数据来源回放，
    # 不能把“论文单活动臂分类”误当成演示的物理实体配置。
    arm_mode = "bimanual"
    source_env_config = demonstration_environment_config_for(args.task)
    result_pattern = (
        robodojo_status()["paths"]["result_root"]
    )
    result_root = Path(result_pattern).resolve()
    result_files_before = set(result_root.rglob("_result.json"))
    port = _free_port()
    server_command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "robodojo",
        "capture-server",
        "--episode",
        str(episode),
        "--output",
        str(output),
        "--arm-mode",
        arm_mode,
        "--active-side",
        args.active_side,
        "--replay-action",
        args.replay_action,
        "--port",
        str(port),
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.device_id),
            "EVAL_NUM": "1",
            "ESSAY2608_OBSERVATION_MODE": getattr(args, "observation_mode", "oracle_pose"),
            "ESSAY2608_ROBODOJO_RUNTIME": str(runtime),
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "PYTHONUNBUFFERED": "1",
            "ESSAY2608_SOURCE_LAYOUT": str(source_layout),
        }
    )
    server = subprocess.Popen(server_command, cwd=PROJECT_ROOT, env=environment)
    try:
        _wait_for_server(server, port)
        client_command = [
            sys.executable,
            "-m",
            "essay2608.data.robodojo_gui",
            "--task_name",
            args.task,
            "--num_envs",
            "1",
            "--env_cfg_type",
            source_env_config,
            "--enable_cameras",
            "--device_id",
            "0",
            "--policy_name",
            "essay2608_replay_capture",
            "--port",
            str(port),
            "--host",
            "127.0.0.1",
            "--protocol",
            "ws",
            "--policy_server_url",
            f"ws://127.0.0.1:{port}",
            "--additional_info",
            "capture=true,gui=true,layout=reconstructed_demo,source_embodiment=arx_x5",
            "--seed",
            str(args.seed),
            "--kit_args",
            " --enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera",
        ]
        print(
            f"[GUI 回放补采] task={args.task} seed={args.seed} episode={episode} "
            f"source_env={source_env_config}",
            flush=True,
        )
        subprocess.run(client_command, cwd=runtime, env=environment, check=True)
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError("GUI 回放结束但没有生成任务帧补采文件")
        result_files_after = set(result_root.rglob("_result.json"))
        new_result_files = sorted(result_files_after.difference(result_files_before))
        if len(new_result_files) != 1:
            raise RuntimeError(
                f"无法唯一定位本次 RoboDojo 结果：新增 {len(new_result_files)} 个 _result.json"
            )
        audit = audit_robodojo_capture(
            output,
            new_result_files[0],
            args.task,
            args.seed,
            layout_kind="reconstructed_demo",
        )
        if not audit["accepted_for_training"]:
            raise RuntimeError(
                "GUI 回放未通过训练数据门禁：" + "；".join(audit["reasons"])
            )
        print(f"[GUI 回放补采验收通过] {output}", flush=True)
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()


def _run_robodojo_capture_batch(args: argparse.Namespace) -> None:
    """批量自动回放演示；用户不需要在每条轨迹中手动操纵机器人。"""

    if args.episodes < 1 or args.start_index < 0:
        raise ValueError("episodes 必须为正数，start-index 不能为负数")
    task_root = ROBODOJO_DEMO_ROOT / args.task / "arx_x5" / "data"
    output_root = args.output_root.resolve() / args.task
    layout_root = args.source_layout_root.resolve() / args.task
    records: list[dict[str, object]] = []
    for offset in range(args.episodes):
        index = args.start_index + offset
        episode = task_root / f"episode_{index:07d}.hdf5"
        output = output_root / f"episode_{index:07d}.jsonl"
        layout = layout_root / f"episode_{index:07d}.json"
        record: dict[str, object] = {
            "index": index,
            "episode": str(episode),
            "output": str(output),
            "observation_mode": args.observation_mode,
        }
        try:
            if not episode.is_file():
                raise FileNotFoundError(f"官方演示不存在：{episode}")
            if args.task == "push_T" and args.auto_reconstruct:
                evidence = layout.with_suffix(".reconstruction.json")
                if not layout.is_file() or not evidence.is_file():
                    reconstruct_push_t_source_layout(episode, layout)
            if not layout.is_file():
                raise FileNotFoundError(
                    f"任务 {args.task} 的源布局不存在：{layout}；"
                    "非 push_T 任务请先提供与演示对应的 source layout"
                )
            single_args = argparse.Namespace(
                task=args.task,
                episode=episode,
                output=output,
                source_layout=layout,
                seed=args.seed,
                active_side=args.active_side,
                replay_action=args.replay_action,
                device_id=args.device_id,
                observation_mode=args.observation_mode,
            )
            _run_robodojo_capture(single_args)
            record["status"] = "accepted_for_training"
        except Exception as error:
            record["status"] = "failed"
            record["error"] = str(error)
            records.append(record)
            if not args.continue_on_error:
                summary_path = output_root / "capture_batch.json"
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                summary_path.write_text(
                    json.dumps(
                        {
                            "schema": "essay2608.robodojo.capture_batch.v1",
                            "task": args.task,
                            "records": records,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                raise
        else:
            records.append(record)
    summary_path = output_root / "capture_batch.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(
            {
                "schema": "essay2608.robodojo.capture_batch.v1",
                "task": args.task,
                "episodes": args.episodes,
                "start_index": args.start_index,
                "observation_mode": args.observation_mode,
                "records": records,
                "accepted": sum(item.get("status") == "accepted_for_training" for item in records),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(summary_path)


def _run_robodojo(args: argparse.Namespace) -> None:
    if args.robodojo_command == "status":
        print(json.dumps(robodojo_status(), ensure_ascii=False, indent=2))
    elif args.robodojo_command == "catalog":
        print(json.dumps(robodojo_task_catalog(), ensure_ascii=False, indent=2))
    elif args.robodojo_command == "resources":
        resources = robodojo_resource_catalog()
        print(
            json.dumps(
                {
                    "tasks": {name: asdict(task) for name, task in resources["tasks"].items()},
                    "scenes": {name: asdict(item) for name, item in resources["scenes"].items()},
                    "robots": {name: asdict(item) for name, item in resources["robots"].items()},
                    "policies": {
                        name: asdict(item) for name, item in resources["policies"].items()
                    },
                    "env_configs": list(resources["env_configs"]),
                    "asset_robots": resources["asset_robots"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.robodojo_command == "prepare":
        print(prepare_robodojo_runtime())
    elif args.robodojo_command == "official-sync":
        print(sync_robodojo_official_snapshot(revision=args.revision))
        print(prepare_robodojo_runtime())
    elif args.robodojo_command == "assets":
        tasks = None if args.all_tasks else tuple(args.tasks or TASK_CANDIDATES)
        print(download_robodojo_assets(tasks))
        print(prepare_robodojo_runtime())
    elif args.robodojo_command == "demos":
        tasks = None if args.all_tasks else tuple(args.tasks or TASK_CANDIDATES)
        print(download_robodojo_demonstrations(tasks, args.episodes))
    elif args.robodojo_command == "fit":
        _run_robodojo_fit(args)
    elif args.robodojo_command == "reconstruct-push":
        print(
            "\n".join(
                str(path)
                for path in reconstruct_push_t_source_layout(
                    args.episode,
                    args.output,
                    calibration_image_path=args.calibration_image,
                    calibration_layout_path=args.calibration_layout,
                    calibration_video_path=args.calibration_video,
                    calibration_capture_path=args.calibration_capture,
                    calibration_frame=args.calibration_frame,
                )
            )
        )
    elif args.robodojo_command == "capture-server":
        serve_robodojo_replay_capture(
            args.episode,
            args.output,
            args.arm_mode,
            args.active_side,
            args.replay_action,
            host=args.host,
            port=args.port,
        )
    elif args.robodojo_command == "capture":
        _run_robodojo_capture(args)
    elif args.robodojo_command == "capture-batch":
        _run_robodojo_capture_batch(args)
    elif args.robodojo_command == "table":
        paths = write_robodojo_paper_table()
        print("\n".join(str(path) for path in paths))
    elif args.robodojo_command == "policy-server":
        arm_mode = robodojo_task_candidates()[args.task].arm_mode
        serve_robodojo_policy(
            args.policy, arm_mode, args.checkpoint, host=args.host, port=args.port
        )
    elif args.robodojo_command == "eval":
        _run_robodojo_eval(args)
    elif args.robodojo_command == "external-eval":
        _run_robodojo_external_eval(args)


def main() -> None:
    args = arguments()
    if args.command == "robodojo":
        _run_robodojo(args)
        return
    if args.command == "inspect":
        policy_class = DynaMAC if args.policy == "dynamac" else TaskParameterizedMiDiGaP
        policy = policy_class.load(args.checkpoint)
        print(json.dumps(policy.summary(), ensure_ascii=False, indent=2))
        return
    if args.command == "fit":
        _fit_policy(args)
        return

    bundle = load_demonstrations(args.data)
    config = load_config(args.config)
    bimanual = BimanualDynaMAC(config=config).fit(bundle.left_arm, bundle.right_arm)
    single = DynaMAC(config).fit(bundle.single_arm)
    print(
        json.dumps(
            {
                "data_schema": bundle.metadata["schema"],
                "single": compact_summary(single),
                "bimanual_left": compact_summary(bimanual.left),
                "bimanual_right": compact_summary(bimanual.right),
                "claim_boundary": (
                    "算法结构验证；随附脚本数据不是 RoboDojo 专家演示，"
                    "不得把该摘要写成 RoboDojo 基准结果"
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
