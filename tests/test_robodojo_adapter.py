from __future__ import annotations

import json
from pathlib import Path

import pytest
from essay2608.data.robodojo import (
    TASK_CANDIDATES,
    RoboDojoPaths,
    audit_robodojo_capture,
    demonstration_environment_config_for,
    environment_config_for,
    prepare_robodojo_runtime,
    robodojo_resource_catalog,
    robodojo_status,
    robodojo_task_catalog,
    write_robodojo_paper_table,
)
from essay2608.data.robodojo_gui import (
    _patch_single_arm_ee_action_key,
    _patch_single_arm_gripper_restore,
    _viewport_camera_pose,
)
from essay2608.policy.robodojo import RoboDojoReplayCaptureModel


def test_paper_task_mapping_distinguishes_arm_modes() -> None:
    assert environment_config_for("push_T") == "essay2608_single_x5_right"
    assert environment_config_for("pour_liquid_into_cup") == "essay2608_single_x5_left"
    assert environment_config_for("sweep_blocks") == "arx_x5"
    assert TASK_CANDIDATES["push_T"].paper_analogues == ("SweepDust",)
    assert demonstration_environment_config_for("push_T") == "arx_x5"
    assert demonstration_environment_config_for("pour_liquid_into_cup") == "arx_x5"
    assert "HandOver" in TASK_CANDIDATES["sweep_blocks"].paper_analogues
    with pytest.raises(ValueError, match="候选集"):
        environment_config_for("unknown")


def test_upstream_task_catalog_is_the_candidate_pool() -> None:
    catalog = robodojo_task_catalog()
    assert catalog["available"] is True
    assert catalog["counts"]["runnable"] == 54
    names = {task["name"] for task in catalog["tasks"]}
    assert set(TASK_CANDIDATES).issubset(names)


def test_resource_registry_exposes_all_combinations() -> None:
    resources = robodojo_resource_catalog()
    assert len(resources["tasks"]) == 54
    assert {"default", "conveyor"}.issubset(resources["scenes"])
    assert "dual_x5" in resources["robots"]
    assert "dual_x5_and_franka_competition" in resources["robots"]
    # arx_x5 是上游顶层环境别名，项目组合器应解析到 dual_x5，而不是报机器人不存在。
    config_name = environment_config_for("hang_mugs", robot_config="arx_x5")
    assert config_name == "essay2608_arx_x5_default_oracle"


def test_runtime_overlay_keeps_upstream_read_only(tmp_path: Path) -> None:
    upstream = tmp_path / "upstream"
    for file_name in (
        "src/eval_client/main.py",
        "task/RoboDojo/task_registry.py",
        "XPolicyLab/policy/demo_policy/deploy.py",
    ):
        path = upstream / file_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n", encoding="utf-8")
    (upstream / "env").mkdir()
    (upstream / "utils").mkdir()
    (upstream / "env_cfg" / "robot").mkdir(parents=True)
    (upstream / "env_cfg" / "robot" / "_robot_info.json").write_text(
        json.dumps({"dual_x5": {"arm_dim": [6, 6], "ee_dim": [1, 1]}}),
        encoding="utf-8",
    )
    (upstream / ".git").mkdir()

    paths = RoboDojoPaths(
        project_root=tmp_path,
        upstream_root=upstream,
        asset_root=tmp_path / "assets",
        runtime_root=tmp_path / "runtime",
        result_root=tmp_path / "results",
    )

    # 测试夹具不是完整 Git 仓库；只在测试中替换提交读取。
    import essay2608.data.robodojo as adapter

    original = adapter._git_commit
    adapter._git_commit = lambda _: "fixture"
    try:
        runtime = prepare_robodojo_runtime(paths)
        prepare_robodojo_runtime(paths)
    finally:
        adapter._git_commit = original

    assert (runtime / "env").is_dir()
    assert not (runtime / "env").is_symlink()
    assert (runtime / "Assets").is_symlink()
    assert not (upstream / "env_cfg" / "essay2608_single_x5_right.yml").exists()
    single = json.loads((runtime / "env_cfg" / "robot" / "_robot_info.json").read_text())
    assert single["essay2608_single_x5_right"] == {"arm_dim": [6], "ee_dim": [1]}
    manifest = json.loads((runtime / "manifest.json").read_text())
    assert manifest["gui_required"] is True


def test_status_is_explicit_about_missing_assets(tmp_path: Path) -> None:
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    (upstream / ".git").mkdir()
    (upstream / "README.md").write_text("fixture", encoding="utf-8")
    paths = RoboDojoPaths(
        project_root=tmp_path,
        upstream_root=upstream,
        asset_root=tmp_path / "assets",
        runtime_root=tmp_path / "runtime",
        result_root=tmp_path / "results",
    )
    import essay2608.data.robodojo as adapter

    original = adapter._git_commit
    adapter._git_commit = lambda _: "fixture"
    try:
        status = robodojo_status(paths)
    finally:
        adapter._git_commit = original
    assert status["upstream_present"] is True
    assert status["assets_ready"] is False
    assert status["gui_required"] is True


def test_paper_table_requires_three_complete_gui_seeds(tmp_path: Path) -> None:
    result_root = tmp_path / "raw"
    for seed in (0, 1, 2):
        run = (
            result_root
            / "RoboDojo"
            / "push_T"
            / "essay2608_dynamac"
            / "essay2608_single_x5"
            / f"{seed}_method=dynamac,condition=static,ckpt=abc,gui=true"
            / f"run_{seed}"
        )
        run.mkdir(parents=True)
        details = {
            str(index): {"layout_id": index, "success": index < 40, "score": 1.0}
            for index in range(50)
        }
        (run / "_result.json").write_text(
            json.dumps({"success_rate": 0.8, "score": 80.0, "eval_time": 50, "details": details}),
            encoding="utf-8",
        )
    paths = RoboDojoPaths(
        project_root=tmp_path,
        upstream_root=tmp_path / "upstream",
        asset_root=tmp_path / "assets",
        runtime_root=tmp_path / "runtime",
        result_root=result_root,
    )
    _, markdown = write_robodojo_paper_table(tmp_path / "table", paths)
    table = markdown.read_text(encoding="utf-8")
    assert "80.0 ± 0.0" in table
    assert "3/150 | 完整" in table


def test_single_arm_action_key_adapter_resolves_upstream_mismatch() -> None:
    class FakeEnvironment:
        robot_action_dim_info = {"arm_dim": [6], "ee_dim": [1]}

        def __init__(self) -> None:
            self.validated = None

        def validate_action_dict(self, action):
            self.validated = action

    environment = FakeEnvironment()
    _patch_single_arm_ee_action_key(environment)
    environment.validate_action_dict({"arm_ee_pose": [0, 0, 0, 1, 0, 0, 0], "ee_joint_state": [0]})
    assert "ee_pose" in environment.validated
    assert "arm_ee_pose" not in environment.validated


def test_single_arm_gripper_restore_keeps_ee_name() -> None:
    class FakeManager:
        def restore_name(self, name):
            return f"upstream:{name}"

    class FakeEnvironment:
        robot_action_dim_info = {"arm_dim": [6]}

        def __init__(self) -> None:
            self.robot_manager = FakeManager()

    environment = FakeEnvironment()
    _patch_single_arm_gripper_restore(environment)
    assert environment.robot_manager.restore_name("ee_joint_state") == "ee"
    assert environment.robot_manager.restore_name("arm_joint_state") == ("upstream:arm_joint_state")


def test_gui_viewport_is_a_close_task_view() -> None:
    eye, lookat = _viewport_camera_pose("push_T")
    assert sum(value * value for value in eye) ** 0.5 < 3.0
    assert lookat[2] > 0.7


def test_replay_capture_selects_the_actually_moving_arm(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    episode = tmp_path / "episode.hdf5"
    with h5py.File(episode, "w") as archive:
        action = archive.create_group("action")
        left_pose = [[0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]]
        right_pose = [[0, 0, 0, 1, 0, 0, 0], [0.2, 0, 0, 1, 0, 0, 0]]
        action["left_ee_poses"] = left_pose
        action["right_ee_poses"] = right_pose
        action["left_arm_joint_states"] = [[0] * 6, [0] * 6]
        action["right_arm_joint_states"] = [[0] * 6, [0.2] * 6]
        action["left_ee_joint_states"] = [[1], [1]]
        action["right_ee_joint_states"] = [[1], [0]]
        archive["instruction"] = "fixture"
        additional = archive.create_group("additional_info")
        additional["frequency"] = 25
    output = tmp_path / "capture.jsonl"
    model = RoboDojoReplayCaptureModel(episode, output, "single")
    assert model.active_side == "right"
    model.reset()
    model.update_obs(
        {
            "state": {"ee_pose": [0, 0, 0, 1, 0, 0, 0]},
            "task": {"object_poses": {"target": [1, 2, 3, 1, 0, 0, 0]}},
        }
    )
    action = model.get_action()[0]
    assert "arm_joint_state" in action
    assert "arm_ee_pose" not in action
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert records[1]["frames"]["target"][:3] == [1, 2, 3]

    bimanual_output = tmp_path / "capture_bimanual.jsonl"
    bimanual = RoboDojoReplayCaptureModel(episode, bimanual_output, "bimanual")
    bimanual.reset()
    bimanual.update_obs(
        {
            "state": {
                "left_ee_pose": [0, 0, 0, 1, 0, 0, 0],
                "right_ee_pose": [0, 0, 0, 1, 0, 0, 0],
            },
            "task": {"object_poses": {}},
        }
    )
    bimanual_action = bimanual.get_action()[0]
    assert set(bimanual_action) == {
        "left_arm_joint_state",
        "left_ee_joint_state",
        "right_arm_joint_state",
        "right_ee_joint_state",
    }

    ee_model = RoboDojoReplayCaptureModel(
        episode, tmp_path / "capture_ee.jsonl", "bimanual", replay_action_type="ee_pose"
    )
    ee_model.reset()
    ee_model.update_obs(
        {
            "state": {
                "left_ee_pose": [0, 0, 0, 1, 0, 0, 0],
                "right_ee_pose": [0, 0, 0, 1, 0, 0, 0],
            },
            "task": {"object_poses": {}},
        }
    )
    assert set(ee_model.get_action()[0]) == {
        "left_ee_pose",
        "left_ee_joint_state",
        "right_ee_pose",
        "right_ee_joint_state",
    }


def test_capture_audit_rejects_native_failure(tmp_path: Path) -> None:
    capture = tmp_path / "capture.jsonl"
    capture.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "type": "metadata",
                        "schema": "essay2608.robodojo.gui_capture.v1",
                    }
                ),
                json.dumps(
                    {
                        "type": "step",
                        "index": 0,
                        "frames": {
                            "t": [0, 0, 0, 1, 0, 0, 0],
                            "target_t": [0, 0, 0, 1, 0, 0, 0],
                        },
                        "action": {"arm_ee_pose": [0, 0, 0, 1, 0, 0, 0]},
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    result = tmp_path / "_result.json"
    result.write_text(
        json.dumps(
            {
                "success_rate": 0,
                "eval_time": 1,
                "details": {"0": {"success": False}},
            }
        ),
        encoding="utf-8",
    )
    audit = audit_robodojo_capture(capture, result, "push_T", 0)
    assert audit["accepted_for_training"] is False
    assert audit["native_success"] is False
    assert "原生任务成功判据" in audit["reasons"][-1]
    assert capture.with_suffix(".audit.json").is_file()


def test_capture_audit_accepts_complete_native_success(tmp_path: Path) -> None:
    capture = tmp_path / "capture.jsonl"
    capture.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "type": "metadata",
                        "schema": "essay2608.robodojo.gui_capture.v1",
                    }
                ),
                json.dumps(
                    {
                        "type": "step",
                        "index": 0,
                        "frames": {
                            "t": [0, 0, 0, 1, 0, 0, 0],
                            "target_t": [0, 0, 0, 1, 0, 0, 0],
                        },
                        "action": {"arm_ee_pose": [0, 0, 0, 1, 0, 0, 0]},
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )
    result = tmp_path / "_result.json"
    result.write_text(
        json.dumps(
            {
                "success_rate": 1,
                "eval_time": 1,
                "details": {"0": {"success": True}},
            }
        ),
        encoding="utf-8",
    )
    audit = audit_robodojo_capture(capture, result, "push_T", 2)
    assert audit["accepted_for_training"] is True
    assert audit["native_success"] is True
    assert audit["layout_seed"] == 2
