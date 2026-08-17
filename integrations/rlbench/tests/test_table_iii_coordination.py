from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
from types import ModuleType, SimpleNamespace

import pytest
from essay2608.policy import DynaMACConfig

from integrations.rlbench.rlbench_dynamac import table_iii_coordination as coordination


class _ArmPolicy:
    def __init__(self, config, fingerprint, durations):
        self.config = config
        self._fingerprint = fingerprint
        self.skill_sequence = tuple(range(len(durations)))
        self.skills = [SimpleNamespace(duration=value) for value in durations]

    def fingerprint(self):
        return self._fingerprint


def test_coordination_training_summary_is_v2_and_fingerprint_bound(tmp_path):
    base = DynaMACConfig(eq6_empty_selection="keep_argmax")
    policy = SimpleNamespace(
        left=_ArmPolicy(replace(base, random_seed=11), "left-fingerprint", (3, 4)),
        right=_ArmPolicy(replace(base, random_seed=22), "right-fingerprint", (5, 6)),
    )
    converted = SimpleNamespace(
        audit={"schema": "rlbench-dynamac-demo-adapter-v2"},
        segmentation=SimpleNamespace(audit={"coordination": "shared_union"}),
    )

    summary = coordination._training_summary(
        policy=policy,
        converted=converted,
        names=["episode0", "episode1"],
        policy_config=base,
        debug_plot=tmp_path / "segmentation.png",
    )

    assert summary["manifest_schema"] == "dynamac-direct-training-v2"
    assert summary["task"] == coordination.POLICY_TASK
    assert summary["bimanual"] is True
    assert summary["config"] == asdict(base)
    assert summary["left"]["config"] == asdict(policy.left.config)
    assert summary["right"]["config"] == asdict(policy.right.config)
    assert summary["left"]["fingerprint"] == "left-fingerprint"
    assert summary["right"]["fingerprint"] == "right-fingerprint"
    assert summary["adapter"] == converted.audit
    assert summary["segmentation_debug_plot"] == "segmentation.png"


def test_coordination_training_is_staged_validated_and_never_overwritten(
    tmp_path, monkeypatch
):
    seen = {}

    def fake_train_into(args, staging):
        assert args.models_dir == tmp_path
        assert staging.parent == tmp_path
        (staging / "marker.txt").write_text("complete", encoding="utf-8")
        return {"left": {"durations": [3]}, "right": {"durations": [4]}}

    def fake_validate(task, staging, summary):
        seen["task"] = task
        seen["staging"] = staging
        seen["summary"] = summary
        assert (staging / "marker.txt").read_text(encoding="utf-8") == "complete"

    monkeypatch.setattr(coordination, "_train_into", fake_train_into)
    monkeypatch.setattr(coordination, "_validate_published_model", fake_validate)
    args = SimpleNamespace(models_dir=tmp_path)

    summary = coordination.train(args)
    output = tmp_path / coordination.POLICY_TASK

    assert summary == seen["summary"]
    assert seen["task"] == coordination.POLICY_TASK
    assert seen["staging"] != output
    assert (output / "marker.txt").read_text(encoding="utf-8") == "complete"
    assert not output.with_name(output.name + ".lock").exists()
    assert not list(tmp_path.glob(f".{coordination.POLICY_TASK}.staging-*"))
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        coordination.train(args)


def test_coordination_evaluation_is_reserved_atomic_and_identity_tagged(
    tmp_path, monkeypatch
):
    environment_module = ModuleType("rlbench.environment")
    rlbench_module = ModuleType("rlbench")
    rlbench_module.__path__ = []
    action_mode = SimpleNamespace(
        arm_action_mode=SimpleNamespace(
            diagnostics=lambda: {"joint_target_reached": 2}
        )
    )
    lifecycle = {"launched": False, "shutdown": False, "worker_closed": False}

    class Environment:
        def __init__(self, *, action_mode, obs_config, headless, robot_setup):
            assert action_mode is globals_action_mode
            assert obs_config is observation_config
            assert headless is True
            assert robot_setup == "dual_panda"

        def launch(self):
            lifecycle["launched"] = True

        def get_task(self, task_class):
            assert task_class is dynamic_task
            return object()

        def shutdown(self):
            lifecycle["shutdown"] = True

    class Worker:
        def __init__(self, python, task, models_dir, timeout):
            assert python == "policy-python"
            assert task == coordination.POLICY_TASK
            assert models_dir == tmp_path / "models"
            assert timeout == 7.0
            self.policy_steps = 37
            self.model_identity = {"manifest_authenticated": True, "fingerprint": "x"}
            self.policy_clock_semantics_id = (
                "policy-tick-transaction-commit-on-primary-action-success-v1"
            )

        def close(self):
            lifecycle["worker_closed"] = True

    def run_episode(
        task_environment,
        worker,
        *,
        episode,
        seed,
        horizon,
        arm,
        trigger,
        max_primary_action_attempts,
    ):
        assert worker.policy_steps == 37
        assert seed == 5
        assert horizon == 1000
        assert arm == "left"
        assert trigger == 12
        assert max_primary_action_attempts == 3
        return {
            "episode": episode,
            "seed": seed + episode,
            "success": episode == 0,
            "steps": 10,
            "reason": "success" if episode == 0 else "policy_complete",
            "invalid_actions": 1,
            "perturbed_steps": 2,
        }

    globals_action_mode = action_mode
    observation_config = object()
    dynamic_task = type("DynamicTask", (), {})
    environment_module.Environment = Environment
    monkeypatch.setitem(sys.modules, "rlbench", rlbench_module)
    monkeypatch.setitem(sys.modules, "rlbench.environment", environment_module)
    monkeypatch.setattr(coordination, "_make_action_mode", lambda: action_mode)
    monkeypatch.setattr(coordination, "_observation_config", lambda: observation_config)
    monkeypatch.setattr(coordination, "_task_class", lambda: dynamic_task)
    monkeypatch.setattr(coordination, "PolicyProcess", Worker)
    monkeypatch.setattr(coordination, "_run_episode", run_episode)
    output = tmp_path / "coordination.json"
    args = SimpleNamespace(
        output=output,
        arm="left",
        models_dir=tmp_path / "models",
        policy_python="policy-python",
        policy_timeout=7.0,
        headless=True,
        episodes=2,
        seed=5,
        horizon=1000,
    )

    payload = coordination.evaluate(args)
    stored = json.loads(output.read_text(encoding="utf-8"))

    assert lifecycle == {"launched": True, "shutdown": True, "worker_closed": True}
    assert stored == payload
    assert payload["evaluation_protocol_id"] == coordination.EVALUATION_PROTOCOL_ID
    assert payload["model_identity"]["manifest_authenticated"] is True
    assert payload["controller"]["joint_target_max_steps"] == 200
    assert payload["controller"]["policy_clock_rollback"] is True
    assert payload["controller"]["policy_clock_semantics_id"] == (
        "policy-tick-transaction-commit-on-primary-action-success-v1"
    )
    assert payload["ik_execution_diagnostics"] == {"joint_target_reached": 2}
    assert payload["learned_policy_steps"] == 37
    assert payload["successes"] == 1
    assert payload["success_rate"] == 0.5
    assert not output.with_name(output.name + ".lock").exists()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        coordination.evaluate(args)
