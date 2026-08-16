from __future__ import annotations

import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
from essay2608.policy.dynamac import (
    BimanualDynaMAC,
    DynaMAC,
    DynaMACConfig,
    SkillModel,
    StreamModel,
)

from integrations.rlbench.rlbench_dynamac.direct_evaluate import (
    _apply_scenario,
    _learned_policy_steps,
    _noop_action,
)
from integrations.rlbench.rlbench_dynamac.direct_evaluate import (
    build_parser as build_evaluation_parser,
)
from integrations.rlbench.rlbench_dynamac.direct_policy import (
    PolicyServer,
    _validate_published_model,
    _WireObservation,
    demonstration_paths,
)
from integrations.rlbench.rlbench_dynamac.records import atomic_json, reserve_output
from integrations.rlbench.rlbench_dynamac.runtime import (
    ScenarioController,
    bimanual_observations_from_rlbench,
    execute_joint_target_control,
)


class _InvalidActionForTest(Exception):
    pass


class _JointArm:
    def __init__(self, positions) -> None:
        self._positions = iter(positions)

    def get_joint_positions(self):
        return next(self._positions)


class _StepScene:
    def __init__(self) -> None:
        self.steps = 0

    def step(self) -> None:
        self.steps += 1


class _LoadedPolicyForValidation:
    def __init__(self, config, fingerprint, identity) -> None:
        self.config = config
        self._fingerprint = fingerprint
        self._identity = dict(identity)

    def fingerprint(self):
        return self._fingerprint

    def summary(self):
        return dict(self._identity)


def _bimanual_validation_case(monkeypatch):
    base = DynaMACConfig(eq6_empty_selection="keep_argmax")
    derived = BimanualDynaMAC(config=base)
    identity = {
        "model_schema_version": 13,
        "selection_semantics_id": (
            "eq5_skill_majority_mask_before_eq6_time_state_position3d_unimodal_v1"
        ),
        "tapas_reference_commit": "52e35214b9baa7b190b87196c36b9e98f4006149",
    }
    policies = {
        "left.npz": _LoadedPolicyForValidation(
            derived.left.config, "left-fingerprint", identity
        ),
        "right.npz": _LoadedPolicyForValidation(
            derived.right.config, "right-fingerprint", identity
        ),
    }
    monkeypatch.setattr(
        DynaMAC,
        "load",
        staticmethod(lambda path: policies[path.name]),
    )
    manifest = {
        "task": "bimanual_handover_item",
        "bimanual": True,
        "config": asdict(base),
        "left": {
            "config": asdict(policies["left.npz"].config),
            "fingerprint": policies["left.npz"].fingerprint(),
        },
        "right": {
            "config": asdict(policies["right.npz"].config),
            "fingerprint": policies["right.npz"].fingerprint(),
        },
    }
    return manifest, policies


def test_direct_training_discovers_episodes_in_numeric_order(tmp_path) -> None:
    root = tmp_path / "task" / "all_variations" / "episodes"
    for number in (10, 2, 0, 1, 11):
        episode = root / f"episode{number}"
        episode.mkdir(parents=True)
        (episode / "low_dim_obs.pkl").touch()

    paths = demonstration_paths(tmp_path, "task", count=4)

    assert [path.parent.name for path in paths] == [
        "episode0",
        "episode1",
        "episode2",
        "episode10",
    ]


def test_result_path_is_exclusively_reserved_and_atomically_published(tmp_path) -> None:
    output = tmp_path / "result.json"

    with reserve_output(output):
        with np.testing.assert_raises(FileExistsError):
            with reserve_output(output):
                pass
        atomic_json(output, {"complete": True})

    assert json.loads(output.read_text(encoding="utf-8")) == {"complete": True}
    assert not output.with_name(output.name + ".lock").exists()
    with np.testing.assert_raises(FileExistsError):
        with reserve_output(output):
            pass


def test_wire_observation_converts_rlbench_xyzw_to_core_wxyz() -> None:
    pose = [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]
    observation = _WireObservation(
        {
            "left": {"gripper_pose": pose},
            "right": {"gripper_pose": pose},
            "task_low_dim_state": pose * 2,
        }
    )

    left, right = bimanual_observations_from_rlbench(
        observation, "bimanual_lift_tray"
    )

    assert np.allclose(left.ee_pose, [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0])
    assert np.allclose(right.ee_pose, left.ee_pose)
    assert list(left.frames) == ["item", "tray"]


def test_invalid_action_noop_uses_right_first_current_state() -> None:
    right_pose = np.asarray([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])
    left_pose = np.asarray([4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0])
    observation = SimpleNamespace(
        right=SimpleNamespace(gripper_pose=right_pose, gripper_open=0.9),
        left=SimpleNamespace(gripper_pose=left_pose, gripper_open=0.1),
    )

    action = _noop_action(observation)

    assert action.shape == (18,)
    assert np.allclose(action[:7], right_pose)
    assert np.allclose(action[9:16], left_pose)
    assert np.allclose(action[[7, 8, 16, 17]], [1.0, 0.0, 0.0, 0.0])


def test_evaluation_defaults_to_200_episodes() -> None:
    args = build_evaluation_parser().parse_args(
        ["--task", "bimanual_handover_item"]
    )

    assert args.episodes == 200
    assert args.headless is True
    assert args.policy_timeout == 120.0
    assert args.scenario == "static"
    assert args.scenario_trigger_fraction == 1.0 / 3.0
    assert args.scenario_steps == 10
    assert args.scenario_reference_steps is None


def test_dynamic_scenario_refreshes_observation_before_policy_action() -> None:
    original = object()
    refreshed = object()

    class Controller:
        def apply(self, task_environment, *, step, horizon):
            assert task_environment is environment
            assert step == 4
            assert horizon == 12
            return {"applied": True, "trigger_step": 4}

    class Environment:
        def get_observation(self):
            return refreshed

    environment = Environment()
    observation, event = _apply_scenario(
        Controller(), environment, original, step=4, horizon=12
    )

    assert observation is refreshed
    assert event["policy_observation_refreshed"] is True


def test_smooth_scenario_applies_the_upstream_endpoint() -> None:
    class Root:
        def __init__(self) -> None:
            self.pose = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

        def get_pose(self):
            return self.pose.copy()

        def set_pose(self, value):
            self.pose = np.asarray(value, dtype=np.float64).copy()

    class Task:
        def __init__(self, root) -> None:
            self.root = root

        def boundary_root(self):
            return self.root

    class Scene:
        def __init__(self) -> None:
            self.root = Root()
            self.task = Task(self.root)
            self._move_task_smoothly_state = None

        def move_task_smoothly(self, total_steps, **_kwargs):
            state = self._move_task_smoothly_state
            if state is None:
                state = {
                    "source_pose": self.root.get_pose(),
                    "goal_pose": np.asarray(
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
                    ),
                    "current_step": 0,
                }
            fraction = state["current_step"] / total_steps
            self.root.set_pose(
                state["source_pose"]
                + fraction * (state["goal_pose"] - state["source_pose"])
            )
            state["current_step"] += 1
            if state["current_step"] >= total_steps:
                del self._move_task_smoothly_state
                return True
            self._move_task_smoothly_state = state
            return False

    class TaskEnvironment:
        def __init__(self) -> None:
            self._scene = Scene()

        def get_observation(self):
            x = float(self._scene.root.pose[0])
            return SimpleNamespace(task_low_dim_state=np.asarray([x]))

    environment = TaskEnvironment()
    controller = ScenarioController(
        "smooth_task_motion", trigger_fraction=0.0, total_steps=2
    )

    first = controller.apply(environment, step=0, horizon=2)
    second = controller.apply(environment, step=1, horizon=2)

    assert first["complete"] is False
    assert second["complete"] is True
    assert second["endpoint_applied"] is True
    assert second["endpoint_fraction"] == 1.0
    assert environment._scene.root.pose[0] == 1.0


def test_learned_policy_steps_uses_longer_arm_clock(tmp_path) -> None:
    model = tmp_path / "task"
    model.mkdir()
    (model / "training.json").write_text(
        '{"left":{"durations":[10,20]},"right":{"durations":[12,24]}}',
        encoding="utf-8",
    )

    assert _learned_policy_steps(tmp_path, "task") == 36


def test_policy_ping_binds_results_to_the_loaded_checkpoint(tmp_path) -> None:
    config = DynaMACConfig(
        covariance_estimation_method="diagonal_empirical_ridge",
        default_mode_strategy="map",
    )
    policy = DynaMAC(config)
    policy.frame_names = ("wine_bottle",)
    policy.skill_sequence = (7,)
    duration = 3
    mean = np.repeat(
        np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])[None, None],
        duration,
        axis=1,
    )
    covariance = np.repeat((np.eye(6) * 1.0e-3)[None, None], duration, axis=1)
    policy.skills = [
        SkillModel(
            label=7,
            duration=duration,
            selected_frames=("virtual_skill_7",),
            mode_priors=np.ones(1),
            streams={
                "virtual_skill_7": StreamModel(
                    "virtual_skill_7",
                    mean,
                    covariance,
                )
            },
            gripper=np.zeros((1, duration, 1)),
            mode_demonstration_indices=((0,),),
        )
    ]
    expected_fingerprint = policy.fingerprint()
    expected_summary = policy.summary()

    model_dir = tmp_path / "models" / "stack_wine"
    model_dir.mkdir(parents=True)
    policy.save(model_dir / "model.npz")
    (model_dir / "training.json").write_text(
        json.dumps(
            {
                "manifest_schema": "dynamac-direct-training-v2",
                "task": "stack_wine",
                "bimanual": False,
                "config": asdict(config),
                "fingerprint": expected_fingerprint,
            }
        ),
        encoding="utf-8",
    )

    response = PolicyServer("stack_wine", tmp_path / "models").handle(
        {"command": "ping"}
    )

    assert response["ok"] is True
    assert response["ready"] is True
    assert response["task"] == "stack_wine"
    assert response["bimanual"] is False
    assert response["policy_steps"] == duration
    identity = response["model_identity"]
    assert set(identity) == {
        "model_schema_version",
        "selection_semantics_id",
        "tapas_reference_commit",
        "config",
        "fingerprint",
        "training_manifest_schema",
        "manifest_authenticated",
        "training_config",
    }
    assert identity["model_schema_version"] == expected_summary["model_schema_version"]
    assert identity["selection_semantics_id"] == expected_summary[
        "selection_semantics_id"
    ]
    assert identity["config"] == asdict(config)
    assert identity["config"]["covariance_estimation_method"] == (
        "diagonal_empirical_ridge"
    )
    assert identity["fingerprint"] == expected_fingerprint
    assert identity["manifest_authenticated"] is True
    assert identity["training_config"] == asdict(config)


def test_bimanual_manifest_base_config_is_bound_to_both_checkpoints(
    tmp_path, monkeypatch
) -> None:
    manifest, _policies = _bimanual_validation_case(monkeypatch)
    _validate_published_model("bimanual_handover_item", tmp_path, manifest)

    manifest["config"] = {**manifest["config"], "tau_omega": 0.25}

    with pytest.raises(RuntimeError, match="not derived from the base config"):
        _validate_published_model("bimanual_handover_item", tmp_path, manifest)


@pytest.mark.parametrize(
    "field",
    ("model_schema_version", "selection_semantics_id", "tapas_reference_commit"),
)
def test_bimanual_manifest_rejects_mixed_checkpoint_identity(
    tmp_path, monkeypatch, field
) -> None:
    manifest, policies = _bimanual_validation_case(monkeypatch)
    policies["right.npz"]._identity[field] = "tampered"

    with pytest.raises(RuntimeError, match="mismatched model identity"):
        _validate_published_model("bimanual_handover_item", tmp_path, manifest)


def test_joint_target_control_ignores_task_terminal_until_combined_action_finishes() -> None:
    scene = _StepScene()
    # No task object is present: consulting scene.task.success() here would fail.
    arm = _JointArm(([0.0], [0.5], [1.0]))

    status = execute_joint_target_control(
        scene,
        ((arm, np.asarray([1.0])),),
        max_steps=3,
        invalid_action_error=_InvalidActionForTest,
    )

    assert status == "reached"
    assert scene.steps == 3


def test_joint_target_control_cap_exhaustion_is_an_invalid_action() -> None:
    scene = _StepScene()
    right = _JointArm(([0.0], [0.1], [0.2], [0.3]))
    left = _JointArm(([0.0], [0.1], [0.2], [0.3]))

    with np.testing.assert_raises_regex(
        _InvalidActionForTest,
        "bimanual command timed out",
    ):
        execute_joint_target_control(
            scene,
            (
                (right, np.asarray([10.0])),
                (left, np.asarray([10.0])),
            ),
            max_steps=4,
            invalid_action_error=_InvalidActionForTest,
            error_message="bimanual command timed out",
        )

    assert scene.steps == 4
