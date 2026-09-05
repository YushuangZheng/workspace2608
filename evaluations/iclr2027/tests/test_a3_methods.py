from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from essay2608.policy.closed_loop import ClosedLoopFeatureProfile
from evaluations.iclr2027.interfaces.runtime_monitor import EpisodeContext
from evaluations.iclr2027.methods.ours_monitor import OursTaskStateMonitor
from evaluations.iclr2027.methods.registry import build_monitor, load_method_spec
from evaluations.iclr2027.methods.restart import NoProgressMonitor
from evaluations.iclr2027.methods.trajectory_likelihood import (
    TrajectoryLikelihoodMonitor,
)
from integrations.rlbench.rlbench_closed_loop.policy_server import (
    ClosedLoopPolicyServer,
)
from integrations.rlbench.rlbench_dynamac.data.direct_policy import PolicyServer

ROOT = Path(__file__).resolve().parents[3]
METHODS = (
    "m0_dynamac",
    "m1_restart",
    "m2_trajectory_likelihood",
    "m5_full",
    "m6_ours_monitor_retry",
    "ablation_motion_only",
    "ablation_open_loop_progress",
    "ablation_generic_retry",
)


def _context(method: str = "test") -> EpisodeContext:
    return EpisodeContext("episode", "task", method, False, 1000, "schema", "hash")


def test_a_owned_method_configs_are_complete_and_share_retry_budget() -> None:
    specs = {name: load_method_spec(name) for name in METHODS}
    assert {spec.method_id for spec in specs.values()} == set(METHODS)
    retry_specs = [
        specs[name]
        for name in (
            "m1_restart",
            "m2_trajectory_likelihood",
            "m6_ours_monitor_retry",
            "ablation_generic_retry",
        )
    ]
    assert {spec.recovery["budget_cycles"] for spec in retry_specs} == {400}
    assert {spec.recovery["maximum_retries"] for spec in retry_specs} == {1}
    assert isinstance(build_monitor(specs["m1_restart"]), NoProgressMonitor)
    assert isinstance(
        build_monitor(specs["m2_trajectory_likelihood"]),
        TrajectoryLikelihoodMonitor,
    )
    assert isinstance(
        build_monitor(specs["m6_ours_monitor_retry"]),
        OursTaskStateMonitor,
    )


def test_no_progress_rule_compares_prior_demand_with_causal_motion() -> None:
    monitor = NoProgressMonitor(consecutive_stopped_cycles=2)
    monitor.reset(_context())
    observation = {
        "arms": {
            "single": {
                "ee_pose_xyzw": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            }
        }
    }
    action = {"action": [0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0]}
    monitor.observe(observation, action, {})
    assert not monitor.alarm()
    monitor.observe(observation, action, {})
    assert not monitor.alarm()
    monitor.observe(observation, action, {})
    assert monitor.alarm()
    moved = {
        "arms": {
            "single": {
                "ee_pose_xyzw": [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            }
        }
    }
    monitor.observe(moved, action, {})
    assert not monitor.alarm()


def test_trajectory_likelihood_uses_active_stream_covariance_and_poe_weight() -> None:
    monitor = TrajectoryLikelihoodMonitor(
        threshold=0.05,
        persistence_cycles=2,
    )
    monitor.reset(_context("m2"))
    observation = {
        "arms": {
            "single": {
                "ee_pose_xyzw": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                "gripper_open": 1.0,
            }
        }
    }
    policy_state = {
        "stream_metadata": {
            "active_streams": ["object", "inactive"],
            "poe_weights": {"object": 1.0, "inactive": 1000.0},
            "marginal_means": {
                "object": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                "inactive": [100.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            },
            "marginal_covariances": {
                "object": np.eye(6).tolist(),
                "inactive": np.eye(6).tolist(),
            },
        }
    }
    # Remove the second stream from the authoritative active mask: its very
    # large PoE weight and residual must have no effect.
    policy_state["stream_metadata"]["active_streams"] = ["object"]
    monitor.observe(observation, {}, policy_state)
    assert np.isclose(monitor.score()["standardized_nll"], 1.0 / 12.0)
    assert not monitor.alarm()
    monitor.observe(observation, {}, policy_state)
    assert monitor.alarm()


def test_uncalibrated_trajectory_monitor_never_intervenes() -> None:
    spec = load_method_spec("m2_trajectory_likelihood")
    monitor = build_monitor(spec)
    assert isinstance(monitor, TrajectoryLikelihoodMonitor)
    monitor.reset(_context("m2"))
    monitor.observe({"arms": {}}, {}, {"stream_metadata": {}})
    assert monitor.threshold is None
    assert not monitor.alarm()


def test_m6_reads_exact_closed_loop_alarm_without_a_second_detector() -> None:
    monitor = OursTaskStateMonitor()
    monitor.reset(_context("m6"))
    monitor.observe({}, {}, {"monitor": {"alarm": True, "reasons": ["x"]}})
    assert monitor.alarm()
    assert monitor.score() == {"task_state_mismatch": 1.0, "trigger_reasons": 1.0}


def test_core_ablation_profiles_switch_only_prespecified_authority() -> None:
    full = ClosedLoopFeatureProfile.named("full")
    motion = ClosedLoopFeatureProfile.named("motion_only")
    clock = ClosedLoopFeatureProfile.named("open_loop_progress")
    retry = ClosedLoopFeatureProfile.named("generic_retry")
    assert full.complete_state_progress_evidence
    assert not motion.complete_state_progress_evidence
    assert not motion.dynamic_frame_roles
    assert not motion.relation_scene_boundary_guards
    assert not clock.belief_driven_progress
    assert not clock.boundary_gated_advancement
    assert clock.auxiliary_verification_recovery
    assert retry.complete_state_progress_evidence
    assert retry.belief_driven_progress
    assert retry.boundary_gated_advancement
    assert not retry.auxiliary_verification_recovery


def test_both_policy_servers_expose_dormant_generic_retry_endpoint() -> None:
    baseline = object.__new__(PolicyServer)
    baseline._pending_transaction = None
    baseline.bimanual = False
    baseline.policy = SimpleNamespace(
        restart_current_skill_reference=lambda: (2, 0)
    )
    response = baseline.handle({"command": "retry_current_skill"})
    assert response["reference_entries"] == {
        "single": {"skill": 2, "progress": 0}
    }

    closed = object.__new__(ClosedLoopPolicyServer)
    closed._pending = None
    closed.policy = SimpleNamespace(
        restart_current_skill_reference=lambda: {
            "single": SimpleNamespace(skill_index=3, local_index=0)
        }
    )
    response = closed.handle({"command": "retry_current_skill"})
    assert response["reference_entries"] == {
        "single": {"skill": 3, "progress": 0}
    }


def test_horizon3_has_all_levels_manifests_and_both_model_families() -> None:
    expected = {
        "place_cups_1",
        "place_cups_2",
        "place_cups_3",
        "remove_cups_1",
        "remove_cups_2",
        "push_buttons_1",
        "push_buttons_2",
        "push_buttons_3",
    }
    for filename in ("horizon3_per_stage.jsonl", "horizon3_single_event.jsonl"):
        path = ROOT / "evaluations" / "iclr2027" / "manifests" / filename
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        assert len(rows) == 1600
        assert {row["task"] for row in rows} == expected
    for task in expected:
        assert (
            ROOT / "integrations" / "rlbench" / "models" / "iclr2027"
            / "dynamac" / task / "model.npz"
        ).is_file()
        assert (
            ROOT / "integrations" / "rlbench" / "models" / "iclr2027"
            / "closed_loop" / task / "policy.json"
        ).is_file()
