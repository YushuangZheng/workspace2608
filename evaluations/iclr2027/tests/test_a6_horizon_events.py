from types import SimpleNamespace

from evaluations.iclr2027.audit.horizon_events import (
    EVENT_SCHEDULE,
    InteractionStageFaultEnvironment,
)
from evaluations.iclr2027.runners.a6_horizon_episode import (
    _annotate_horizon_summary,
    _per_interaction_steps,
)


class _Condition:
    def __init__(self) -> None:
        self.met = False

    def condition_met(self):
        return self.met, False


class _BaseEnvironment:
    def __init__(self, conditions):
        task = SimpleNamespace(
            iclr2027_ordered_interaction_conditions=lambda: conditions
        )
        self._scene = SimpleNamespace(task=task)

    def step(self, action):
        if action.get("complete") is not None:
            action["complete"].met = True
        return action


class _OneShotEnvironment:
    def __init__(self, environment, stage):
        self._environment = environment
        self.stage = stage
        self.steps = 0
        self.triggered = False
        self.events = []

    def __getattr__(self, name):
        return getattr(self._environment, name)

    def step(self, action):
        if not self.triggered:
            self.triggered = True
            self.events.append(
                {
                    "kind": "test_fault",
                    "policy_step": self.steps,
                    "protocol_effective": True,
                }
            )
        result = self._environment.step(action)
        self.steps += 1
        return result

    def record_committed_fallback(self):
        self.steps += 1

    def protocol_metadata(self):
        return {
            "triggered": self.triggered,
            "events": self.events,
            "physical_effect_observed": self.triggered,
            "target_arm": "single",
            "target_objects": [f"object{self.stage}"],
            "relation_restored": None,
            "relation_restoration_policy_step": None,
        }


def test_per_interaction_wrapper_opens_only_after_physical_completion():
    conditions = (_Condition(), _Condition(), _Condition())
    base = _BaseEnvironment(conditions)
    built = []

    def builder(environment, _task, **_kwargs):
        wrapped = _OneShotEnvironment(environment, len(built))
        built.append(wrapped)
        return wrapped

    wrapped = InteractionStageFaultEnvironment(
        base,
        SimpleNamespace(task_level=3),
        family="actuation_delay",
        severity="medium",
        trigger_stage="early",
        per_interaction_policy_steps=10,
        config={},
        fault_builder=builder,
    )

    wrapped.step({"complete": None})
    assert len(built) == 1
    wrapped.step({"complete": conditions[0]})
    assert len(built) == 2
    wrapped.step({"complete": None})
    assert len(built) == 2
    wrapped.step({"complete": conditions[1]})
    assert len(built) == 3
    wrapped.step({"complete": None})

    metadata = wrapped.protocol_metadata()
    assert metadata["event_schedule"] == EVENT_SCHEDULE
    assert metadata["interaction_count"] == 3
    assert metadata["interaction_stages_reached"] == 3
    assert metadata["interaction_stages_completed"] == 2
    assert metadata["triggered_event_count"] == 3
    assert [event["interaction_index"] for event in metadata["events"]] == [0, 1, 2]
    assert [event["policy_step"] for event in metadata["events"]] == [0, 2, 4]


def test_fallback_advances_only_the_active_stage_clock():
    condition = _Condition()
    base = _BaseEnvironment((condition,))
    built = []

    def builder(environment, _task, **_kwargs):
        wrapped = _OneShotEnvironment(environment, len(built))
        built.append(wrapped)
        return wrapped

    wrapped = InteractionStageFaultEnvironment(
        base,
        SimpleNamespace(task_level=1),
        family="actuation_delay",
        severity="medium",
        trigger_stage="early",
        per_interaction_policy_steps=10,
        config={},
        fault_builder=builder,
    )
    wrapped.record_committed_fallback()
    assert wrapped.protocol_metadata()["policy_steps_observed"] == 1
    assert built[0].steps == 1


def test_per_interaction_horizon_and_summary_annotation(tmp_path):
    assert _per_interaction_steps(613, 3) == 204
    episode_id = "horizon3_per_stage/place_cups_3/0000"
    episode_path = (
        tmp_path / "episodes" / "horizon3_per_stage__place_cups_3__0000.json"
    )
    episode_path.parent.mkdir(parents=True)
    episode_path.write_text(
        __import__("json").dumps(
            {
                "episode_id": episode_id,
                "final_success": True,
                "audit": {"physically_triggered": False},
                "fault_protocol": {
                    "event_schedule": EVENT_SCHEDULE,
                    "interaction_count": 3,
                    "interaction_stages_reached": 3,
                    "interaction_stages_completed": 3,
                    "triggered_event_count": 2,
                    "completed_interactions": [
                        {"interaction_index": 0, "completion_policy_step": 20},
                        {"interaction_index": 1, "completion_policy_step": 50},
                        {"interaction_index": 2, "completion_policy_step": 80},
                    ],
                    "components": [
                        {
                            "interaction_index": 0,
                            "relation_restored": True,
                            "relation_restoration_policy_step": 15,
                        },
                        {
                            "interaction_index": 1,
                            "relation_restored": False,
                            "relation_restoration_policy_step": None,
                        },
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    value = _annotate_horizon_summary(
        tmp_path, {"episode_id": episode_id}
    )
    audit = value["horizon_audit"]
    assert audit["stage_count"] == 3
    assert audit["event_opportunity_count"] == 3
    assert audit["actual_triggered_event_count"] == 2
    assert audit["remaining_stages_after_repair"] == [
        {
            "interaction_index": 0,
            "restoration_policy_step": 15,
            "completed_interactions_at_restoration": 0,
            "remaining_stages_after_repair": 3,
            "completed_episode_after_repair": True,
        }
    ]
    assert value["audit"]["physically_triggered"] is True
