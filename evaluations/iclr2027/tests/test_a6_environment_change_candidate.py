from __future__ import annotations

from evaluations.iclr2027.analysis.probes.a6_environment_change_candidate import (
    DIRECTLY_OPERATED_ROLES,
    PASSIVE_SUPPORT_ROLES,
    ReferenceState,
    continuous_relation_dependency_factors,
    extend_external_reference_lifetimes,
    find_public_release_condition,
    minimal_semantic_physical_root,
    release_qualified_conditions,
    semantic_pose_object,
)
from source.policy.tsf.model.scene_factors import FactorId


class _Condition:
    def __init__(self, met: bool, terminate: bool = False):
        self.met = met
        self.terminate = terminate

    def condition_met(self):
        return self.met, self.terminate


NothingGrasped = type("NothingGrasped", (_Condition,), {})


class _ConditionSet:
    def __init__(self, *conditions):
        self._conditions = conditions


class _Object:
    _next_handle = 1

    def __init__(self, name: str, parent=None, *, model=False, pose=None):
        self.name = name
        self.parent = parent
        self.model = model
        self.handle = _Object._next_handle
        _Object._next_handle += 1
        self.children = []
        self.pose = pose
        if parent is not None:
            parent.children.append(self)

    def get_handle(self):
        return self.handle

    def get_parent(self):
        return self.parent

    def is_model(self):
        return self.model

    def get_objects_in_tree(self, *, exclude_base=False):
        values = [] if exclude_base else [self]
        for child in self.children:
            values.extend(child.get_objects_in_tree(exclude_base=False))
        return values

    def get_pose(self):
        return self.pose

    def set_position(self, value):
        self.pose = [*value, *self.pose[3:]]


def test_release_qualified_stage_waits_for_unlink():
    native = _Condition(True)
    release = _Condition(False)
    (condition,) = release_qualified_conditions((native,), release)
    assert condition.condition_met() == (False, False)

    release.met = True
    assert condition.condition_met() == (True, False)


def test_release_does_not_complete_an_unfinished_native_goal():
    (condition,) = release_qualified_conditions((_Condition(False),), _Condition(True))
    assert condition.condition_met() == (False, False)


def test_release_condition_is_reused_from_public_task_conditions():
    release = NothingGrasped(True)
    assert find_public_release_condition((_ConditionSet(_Condition(True), release),)) is release


def test_minimal_root_stops_before_sibling_operated_object():
    base = _Object("task")
    broad_boundary = _Object("tree_boundary", base)
    holder = _Object("holder", broad_boundary)
    spoke = _Object("spoke", holder)
    completed_cup = _Object("cup0", broad_boundary, model=True)

    root = minimal_semantic_physical_root(
        spoke,
        base,
        independently_operated_roots=(completed_cup,),
    )
    assert root is holder


def test_minimal_root_keeps_a_self_contained_model():
    base = _Object("task")
    reference_model = _Object("reference_model", base, model=True)
    proxy = _Object("proxy", reference_model)
    independent = _Object("independent", base, model=True)

    root = minimal_semantic_physical_root(
        proxy,
        base,
        independently_operated_roots=(independent,),
    )
    assert root is reference_model


def test_only_passive_payloads_are_support_preserved():
    assert "carried_object" in PASSIVE_SUPPORT_ROLES
    assert "scene_entity" in PASSIVE_SUPPORT_ROLES
    assert "operated_object" not in PASSIVE_SUPPORT_ROLES
    assert "operated_object" in DIRECTLY_OPERATED_ROLES
    assert "cooperatively_operated_object" in DIRECTLY_OPERATED_ROLES


def test_semantic_pose_resolution_rejects_colocated_rotated_boundary():
    base = _Object("task", pose=[0, 0, 0, 0, 0, 0, 1])
    boundary = _Object("spawn_boundary", base, pose=[1, 2, 3, 0, 0, 1, 0])
    lid = _Object("lid", base, pose=[1, 2, 3, 0, 0, 0, 1])
    assert semantic_pose_object(base, lid.pose) is lid
    assert boundary is not lid


def test_semantic_pose_resolution_uses_existing_graspable_tie_breaker():
    base = _Object("task", pose=[0, 0, 0, 0, 0, 0, 1])
    boundary = _Object("mug_boundary", base, pose=[1, 2, 3, 0, 0, 0, 1])
    cup = _Object("cup", base, pose=[1, 2, 3, 0, 0, 0, 1])
    assert semantic_pose_object(base, cup.pose, preferred_roots=(cup,)) is cup
    assert boundary is not cup


def test_reference_lifetime_is_contiguous_and_relation_bounded():
    states = (
        ReferenceState(0, "holding_lid", True, frozenset({"target_jar"})),
        ReferenceState(1, "holding_lid", True, frozenset()),
        ReferenceState(2, "holding_lid", True, frozenset()),
        ReferenceState(3, "released_lid", False, frozenset()),
        ReferenceState(4, "new_relation", True, frozenset({"other_target"})),
    )
    assert extend_external_reference_lifetimes(states) == (
        frozenset({"target_jar"}),
        frozenset({"target_jar"}),
        frozenset({"target_jar"}),
        frozenset(),
        frozenset({"other_target"}),
    )


def test_reference_lifetime_never_crosses_a_relation_change():
    states = (
        ReferenceState(0, "relation_a", True, frozenset({"frame_a"})),
        ReferenceState(1, "relation_b", True, frozenset()),
    )
    assert extend_external_reference_lifetimes(states)[1] == frozenset()


class _Node:
    def __init__(self, relevant, priors):
        self.mode_action_relevant_frames = (tuple(relevant),)
        self.demo_relation_priors = {
            name: [[1.0 - probability, probability]]
            for name, probability in priors.items()
        }


def test_dependency_factor_crosses_skill_boundary_with_same_relation():
    states = {
        (3, 0): _Node(("virtual_skill_3",), {"lid": 0.7, "jar": 0.3}),
        (4, 0): _Node(("jar",), {"lid": 0.7, "jar": 0.3}),
        (5, 0): _Node(("virtual_skill_5",), {"lid": 0.7, "jar": 0.3}),
    }
    assert continuous_relation_dependency_factors(
        states,
        (5, 0),
        0,
        available_frames=("lid", "jar"),
        relation_link_threshold=0.7,
        relation_unlink_threshold=0.3,
    ) == (FactorId("edge", "lid", target="jar"),)


def test_dependency_factor_is_not_added_while_reference_is_still_active():
    states = {
        (0, 0): _Node(("jar",), {"lid": 0.7, "jar": 0.3}),
        (1, 0): _Node(("jar",), {"lid": 0.7, "jar": 0.3}),
    }
    assert continuous_relation_dependency_factors(
        states,
        (1, 0),
        0,
        available_frames=("lid", "jar"),
        relation_link_threshold=0.7,
        relation_unlink_threshold=0.3,
    ) == ()


def test_dependency_factor_does_not_cross_reference_replacement():
    states = {
        (0, 0): _Node(("old_target",), {"object": 0.7, "old_target": 0.3, "new_target": 0.3}),
        (1, 0): _Node(("new_target",), {"object": 0.7, "old_target": 0.3, "new_target": 0.3}),
    }
    assert continuous_relation_dependency_factors(
        states,
        (1, 0),
        0,
        available_frames=("object", "old_target", "new_target"),
        relation_link_threshold=0.7,
        relation_unlink_threshold=0.3,
    ) == ()


def test_dependency_factor_stops_at_changed_relation_signature():
    states = {
        (0, 0): _Node(("target_a",), {"object_a": 0.7, "target_a": 0.3}),
        (1, 0): _Node(("virtual_skill_1",), {"object_b": 0.7, "target_a": 0.3}),
    }
    assert continuous_relation_dependency_factors(
        states,
        (1, 0),
        0,
        available_frames=("object_a", "object_b", "target_a"),
        relation_link_threshold=0.7,
        relation_unlink_threshold=0.3,
    ) == ()
