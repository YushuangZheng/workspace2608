from types import SimpleNamespace

from evaluations.iclr2027.audit.horizon_nested_events import (
    NESTED_EVENT_SCHEDULE,
    NestedInteractionFaultEnvironment,
)


class _Condition:
    def __init__(self) -> None:
        self.met = False

    def condition_met(self):
        return self.met, False


class _BaseEnvironment:
    def __init__(self, conditions, release=None):
        task = SimpleNamespace(
            iclr2027_ordered_interaction_conditions=lambda: conditions,
            _success_conditions=(() if release is None else (release,)),
        )
        self._scene = SimpleNamespace(task=task)
        self.actions = []

    def step(self, action):
        self.actions.append(dict(action))
        if action.get("complete") is not None:
            action["complete"].met = True
        return {"base_step": len(self.actions), "token": action.get("token")}


class _DelayedOneShot:
    def __init__(self, environment, stage, trigger_after):
        self._environment = environment
        self.stage = stage
        self.trigger_after = trigger_after
        self.steps = 0
        self.triggered = False
        self.events = []

    def __getattr__(self, name):
        return getattr(self._environment, name)

    def step(self, action):
        if not self.triggered and self.steps >= self.trigger_after:
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
            "events": list(self.events),
            "physical_effect_observed": self.triggered,
            "target_arm": "single",
            "target_objects": ["object%d" % self.stage],
            "relation_restored": None,
            "relation_restoration_policy_step": None,
        }


def _wrapper(level, conditions, trigger_after=(2, 0, 0), release=None):
    base = _BaseEnvironment(conditions, release=release)
    built = []

    def builder(environment, _task, **_kwargs):
        stage = len(built)
        wrapped = _DelayedOneShot(environment, stage, trigger_after[stage])
        built.append(wrapped)
        return wrapped

    wrapped = NestedInteractionFaultEnvironment(
        base,
        SimpleNamespace(task_level=level),
        family="actuation_delay",
        severity="medium",
        trigger_stage="middle",
        paired_policy_steps=100,
        later_stage_policy_steps=50,
        config={},
        fault_builder=builder,
    )
    return wrapped, base, built


def test_nested_prefix_has_only_the_paired_environment():
    conditions = (_Condition(), _Condition(), _Condition())
    wrapped, base, built = _wrapper(3, conditions)

    first = wrapped.step({"token": "a", "complete": None})
    second = wrapped.step({"token": "b", "complete": None})
    assert first == {"base_step": 1, "token": "a"}
    assert second == {"base_step": 2, "token": "b"}
    assert len(built) == 1
    assert base.actions == [
        {"token": "a", "complete": None},
        {"token": "b", "complete": None},
    ]

    # The third action produces the exact paired first event.  A later event
    # is still not opened until the current physical interaction completes.
    wrapped.step({"token": "c", "complete": None})
    assert len(built) == 1
    assert wrapped.protocol_metadata()["paired_first_event_interaction_index"] == 0


def test_nested_opportunities_open_only_after_first_event_and_later_boundary():
    conditions = (_Condition(), _Condition(), _Condition())
    wrapped, _base, built = _wrapper(3, conditions)
    wrapped.step({"token": "a", "complete": conditions[0]})
    assert len(built) == 1  # boundary preceded the paired first event
    wrapped.step({"token": "b", "complete": None})
    wrapped.step({"token": "c", "complete": None})
    assert wrapped.protocol_metadata()["paired_first_event_interaction_index"] == 1
    assert len(built) == 1

    wrapped.step({"token": "d", "complete": conditions[1]})
    assert len(built) == 2
    wrapped.step({"token": "e", "complete": None})
    metadata = wrapped.protocol_metadata()
    assert metadata["event_schedule"] == NESTED_EVENT_SCHEDULE
    assert metadata["event_opportunity_count"] == 2
    assert metadata["later_event_opportunities"] == 1
    assert metadata["triggered_event_count"] == 2
    assert [event["interaction_index"] for event in metadata["events"]] == [1, 2]


def test_one_stage_nested_wrapper_is_physically_the_single_event_wrapper():
    condition = _Condition()
    wrapped, base, built = _wrapper(1, (condition,))
    for index in range(5):
        wrapped.step(
            {
                "token": index,
                "complete": condition if index == 4 else None,
            }
        )
    metadata = wrapped.protocol_metadata()
    assert len(built) == 1
    assert len(base.actions) == 5
    assert metadata["event_opportunity_count"] == 1
    assert metadata["later_event_opportunities"] == 0
    assert metadata["triggered_event_count"] == 1


def test_untriggered_paired_event_never_opens_later_faults():
    conditions = (_Condition(), _Condition(), _Condition())
    wrapped, base, built = _wrapper(3, conditions, trigger_after=(100, 0, 0))
    for index, condition in enumerate(conditions):
        wrapped.step({"token": index, "complete": condition})
    metadata = wrapped.protocol_metadata()
    assert len(built) == 1
    assert len(base.actions) == 3
    assert metadata["paired_first_event_triggered"] is False
    assert metadata["event_opportunity_count"] == 1
    assert metadata["later_event_opportunities"] == 0
    assert metadata["triggered_event_count"] == 0


class NothingGrasped(_Condition):
    pass


def test_nested_stage_waits_for_native_completion_and_public_release():
    conditions = (_Condition(), _Condition())
    release = NothingGrasped()
    wrapped, _base, built = _wrapper(
        2, conditions, trigger_after=(0, 0, 0), release=release
    )

    wrapped.step({"token": "fault", "complete": None})
    wrapped.step({"token": "native", "complete": conditions[0]})
    assert len(built) == 1
    assert wrapped.protocol_metadata()["interaction_stages_completed"] == 0

    wrapped.step({"token": "release", "complete": release})
    metadata = wrapped.protocol_metadata()
    assert len(built) == 2
    assert metadata["release_qualified_interactions"] is True
    assert metadata["interaction_stages_completed"] == 1
