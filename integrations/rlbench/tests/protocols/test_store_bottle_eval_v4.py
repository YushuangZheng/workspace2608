from __future__ import annotations

import copy
from argparse import Namespace

import numpy as np
import pytest

from integrations.rlbench.rlbench_dynamac.core import runtime
from integrations.rlbench.rlbench_dynamac.eval import direct_evaluate
from integrations.rlbench.rlbench_dynamac.protocols import (
    store_bottle_eval_v4 as protocol,
)


SOURCE_POSE = (0.1, -0.2, 0.8, 0.0, 0.0, 0.0, 1.0)


def _plan(
    index: int,
    *,
    base_seed: int = 2_608_000_000,
    diagnostic_mode=None,
):
    mode = diagnostic_mode or protocol.store_mode_for_episode(index)
    moved = protocol.V4_STORE_MODE_MOVED_ENTITIES[mode]
    candidate_seed = 123_000 + index
    entities = []
    for name in protocol.V4_STORE_ENTITY_ORDER:
        goal = (
            protocol.sample_v4_store_entity_goal_pose(
                SOURCE_POSE,
                candidate_seed,
                entity=name,
            )
            if name in moved
            else np.asarray(SOURCE_POSE, dtype=np.float64)
        )
        entities.append(
            protocol.StoreBottleEntityMotion(
                name=name,
                root_name=protocol.V4_STORE_ENTITY_ROOTS[name],
                frame_name=protocol.V4_STORE_ENTITY_FRAMES[name],
                source_pose=SOURCE_POSE,
                goal_pose=tuple(goal),
                moved=name in moved,
                candidate_seed=candidate_seed if name in moved else None,
            )
        )
    motion = protocol.load_v4_store_motion_source_protocol(verify_semantics_file=False)
    intervention = protocol.load_v4_store_intervention_protocol(
        verify_evidence_files=False
    )
    return protocol.StoreBottleMultiEntityPlan(
        task_name=protocol.STORE_BOTTLE_TASK_NAME,
        episode_index=index,
        episode_seed=base_seed + index,
        variation=0,
        mode=mode,
        entities=tuple(entities),
        source_low_dim_state=SOURCE_POSE + SOURCE_POSE,
        validation={
            "schema": protocol.V4_STORE_PLAN_VALIDATION_SCHEMA,
            "source_seed": base_seed + index,
            "source_waypoint_validated": True,
            "goal_waypoint_validated": True,
            "goal_sampling_max_attempts": (
                protocol.V4_STORE_GOAL_SAMPLING_MAX_ATTEMPTS
            ),
            "sampling_attempts": 1,
            "motion_source_fingerprint": motion["fingerprint"],
            "intervention_fingerprint": intervention["fingerprint"],
            "policy_result_fields_read": False,
        },
    )


def test_store_protocol_freezes_two_preinteraction_triggers_from_v4_evidence():
    value = protocol.load_v4_store_intervention_protocol()
    evidence = value["evidence"]
    assert value["triggers"]["bottle"]["global_tick"] == 60
    assert value["triggers"]["fridge"]["global_tick"] == 45
    assert all(
        trigger < close
        for trigger, close in zip(
            evidence["expert_projected_trigger_indices"]["bottle"],
            evidence["expert_first_gripper_close_indices"]["left"],
        )
    )
    assert all(
        trigger < close
        for trigger, close in zip(
            evidence["expert_projected_trigger_indices"]["fridge"],
            evidence["expert_first_gripper_close_indices"]["right"],
        )
    )
    authentication = protocol.v4_store_trigger_authentication(275)
    assert authentication["triggers"]["fridge"]["trigger_step"] == 45
    assert authentication["triggers"]["bottle"]["trigger_step"] == 60


def test_store_mode_schedule_has_preregistered_n200_counts():
    schedule = protocol.load_v4_store_motion_source_protocol(
        verify_semantics_file=False
    )["episode_mode_schedule"]
    assert schedule == {
        "formula": "fixed_bottle_only",
        "mode": "bottle_only",
        "formal_n200_counts": {
            "bottle_only": 200,
            "fridge_only": 0,
            "both": 0,
        },
    }
    values = [protocol.store_mode_for_episode(index) for index in range(200)]
    assert {name: values.count(name) for name in protocol.V4_STORE_MODE_ORDER} == {
        "bottle_only": 200,
        "fridge_only": 0,
        "both": 0,
    }
    assert all(_plan(index).moved_entities == ("bottle",) for index in range(3))


@pytest.mark.parametrize(
    ("entity", "minimum", "maximum", "maximum_yaw"),
    (("bottle", 0.03, 0.10, 0.10), ("fridge", 0.02, 0.05, 0.05)),
)
def test_entity_candidates_are_source_relative_and_within_limits(
    entity, minimum, maximum, maximum_yaw
):
    first = protocol.sample_v4_store_entity_goal_pose(SOURCE_POSE, 99, entity=entity)
    second = protocol.sample_v4_store_entity_goal_pose(SOURCE_POSE, 99, entity=entity)
    geometry = protocol.store_entity_geometry(SOURCE_POSE, first)
    assert np.array_equal(first, second)
    assert minimum <= geometry["xy_radius_m"] <= maximum
    assert geometry["z_delta_m"] == pytest.approx(0.0, abs=1e-12)
    assert abs(geometry["yaw_delta_rad"]) <= maximum_yaw
    assert geometry["relative_rotation_xy_norm"] == pytest.approx(0.0, abs=1e-12)


def test_store_batch_and_task_scoped_loader_roundtrip():
    plans = [_plan(index, base_seed=50) for index in range(6)]
    inner = protocol.store_bottle_motion_plan_batch(
        base_seed=50,
        variations=[0] * 6,
        plans=plans,
    )
    assert protocol.load_v4_store_motion_plan_batch(inner) == plans
    envelope = protocol.build_v4_store_task_scoped_plan_batch(
        base_seed=50,
        variations=[0] * 6,
        plans=plans,
    )
    assert envelope["runtime_loader"] == protocol.V4_STORE_RUNTIME_LOADER_ID
    assert envelope["runtime_batch"]["mode_counts"] == {
        "bottle_only": 6,
        "fridge_only": 0,
        "both": 0,
    }
    assert protocol.v4_store_runtime_loaders() == {
        protocol.V4_STORE_RUNTIME_LOADER_ID: protocol.load_v4_store_motion_plan_batch
    }
    mutated = dict(inner)
    mutated["mode_counts"] = dict(inner["mode_counts"], both=3)
    with pytest.raises(ValueError, match="authentication"):
        protocol.load_v4_store_motion_plan_batch(mutated)


def test_loader_tolerates_only_cross_python_roundoff_in_legacy_derived_geometry():
    inner = protocol.store_bottle_motion_plan_batch(
        base_seed=80,
        variations=[0],
        plans=[_plan(0, base_seed=80)],
    )
    legacy = copy.deepcopy(inner)
    entity = legacy["plans"][0]["entities"]["bottle"]
    derived = protocol.store_entity_geometry(entity["source_pose"], entity["goal_pose"])
    entity["geometry"] = {
        key: value + (5.0e-12 if key == "xy_radius_m" else 0.0)
        for key, value in derived.items()
    }
    plan_body = {
        key: value for key, value in legacy["plans"][0].items() if key != "fingerprint"
    }
    legacy["plans"][0]["fingerprint"] = protocol.canonical_fingerprint(plan_body)
    batch_body = {
        key: value for key, value in legacy.items() if key != "batch_fingerprint"
    }
    legacy["batch_fingerprint"] = protocol.canonical_fingerprint(batch_body)
    assert len(protocol.load_v4_store_motion_plan_batch(legacy)) == 1

    drifted = copy.deepcopy(legacy)
    drifted_entity = drifted["plans"][0]["entities"]["bottle"]
    drifted_entity["geometry"]["xy_radius_m"] += 1.0e-6
    plan_body = {
        key: value for key, value in drifted["plans"][0].items() if key != "fingerprint"
    }
    drifted["plans"][0]["fingerprint"] = protocol.canonical_fingerprint(plan_body)
    batch_body = {
        key: value for key, value in drifted.items() if key != "batch_fingerprint"
    }
    drifted["batch_fingerprint"] = protocol.canonical_fingerprint(batch_body)
    with pytest.raises(ValueError, match="derived entity geometry"):
        protocol.load_v4_store_motion_plan_batch(drifted)


class _Root:
    def __init__(self, pose):
        self.pose = np.asarray(pose, dtype=np.float64)

    def get_pose(self):
        return self.pose.copy()

    def set_pose(self, pose):
        self.pose = np.asarray(pose, dtype=np.float64)


class _TaskEnvironment:
    def __init__(self):
        self.roots = {name: _Root(SOURCE_POSE) for name in ("bottle", "fridge")}
        self._scene = Namespace(task=object(), robot=object())
        self.observation_count = 0

    def get_observation(self):
        self.observation_count += 1
        return {"observation": self.observation_count}


def test_controller_applies_fridge_then_bottle_as_independent_events(monkeypatch):
    plan = _plan(2, base_seed=70, diagnostic_mode="both")
    environment = _TaskEnvironment()
    monkeypatch.setattr(
        protocol,
        "bind_v4_store_source_plan",
        lambda *args, **kwargs: {"formal_source_bound": True},
    )
    monkeypatch.setattr(protocol, "_semantic_roots", lambda task: environment.roots)
    monkeypatch.setattr(protocol, "_semantic_tree_state", lambda task, roots: [])
    monkeypatch.setattr(
        protocol,
        "_compare_semantic_tree",
        lambda before, after: {"matched": True},
    )
    monkeypatch.setattr(
        protocol, "_low_dim_frame_audit", lambda task: {"matched": True}
    )
    monkeypatch.setattr(runtime, "_robot_external_collision_pairs", lambda *args: ())
    controller = protocol.StoreBottleMultiEntityController(
        plan=plan,
        scenario="teleport",
    )
    controller.bind_source(
        environment,
        descriptions=["fixture"],
        fresh_task_generation={"fixture": True},
    )
    observation = {"observation": 0}
    observation, early = controller.apply(environment, observation, policy_step=44)
    assert early == []
    observation, fridge = controller.apply(environment, observation, policy_step=45)
    assert [event["entity"] for event in fridge] == ["fridge"]
    assert np.allclose(
        environment.roots["bottle"].get_pose(), plan.entity("bottle").source_pose
    )
    observation, middle = controller.apply(environment, observation, policy_step=59)
    assert middle == []
    observation, bottle = controller.apply(environment, observation, policy_step=60)
    assert [event["entity"] for event in bottle] == ["bottle"]
    assert environment.observation_count == 2


def test_controller_smoothly_reuses_same_authenticated_entity_plan(monkeypatch):
    plan = _plan(0, base_seed=70, diagnostic_mode="bottle_only")
    environment = _TaskEnvironment()
    monkeypatch.setattr(
        protocol,
        "bind_v4_store_source_plan",
        lambda *args, **kwargs: {"formal_source_bound": True},
    )
    monkeypatch.setattr(protocol, "_semantic_roots", lambda task: environment.roots)
    monkeypatch.setattr(protocol, "_semantic_tree_state", lambda task, roots: [])
    monkeypatch.setattr(
        protocol,
        "_compare_semantic_tree",
        lambda before, after: {"matched": True},
    )
    monkeypatch.setattr(
        protocol, "_low_dim_frame_audit", lambda task: {"matched": True}
    )
    monkeypatch.setattr(runtime, "_robot_external_collision_pairs", lambda *args: ())
    controller = protocol.StoreBottleMultiEntityController(
        plan=plan,
        scenario="smooth",
        total_steps=2,
    )
    controller.bind_source(
        environment,
        descriptions=["fixture"],
        fresh_task_generation={"fixture": True},
    )

    observation = {"observation": 0}
    observation, first = controller.apply(environment, observation, policy_step=60)
    observation, second = controller.apply(environment, observation, policy_step=61)

    assert first[0]["kind"] == "smooth_store_entity"
    assert first[0]["smooth_call"] == 1
    assert first[0]["complete"] is False
    assert second[0]["smooth_call"] == 2
    assert second[0]["complete"] is True
    assert np.allclose(
        environment.roots["bottle"].get_pose(), plan.entity("bottle").goal_pose
    )


def test_direct_evaluator_store_dispatch_is_v4_only():
    assert direct_evaluate._is_v4_store(
        Namespace(release="v4", task=protocol.STORE_BOTTLE_TASK_NAME)
    )
    assert not direct_evaluate._is_v4_store(
        Namespace(release="v3", task=protocol.STORE_BOTTLE_TASK_NAME)
    )
    assert not direct_evaluate._is_v4_store(
        Namespace(release="v4", task="bimanual_lift_tray")
    )
