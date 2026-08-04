from __future__ import annotations

import numpy as np

from essay2608.eval.bimanual_relation import PhysicalRelationTracker


ZERO = np.zeros((2, 3), dtype=np.float64)
CONTACT = np.asarray([[0.0, 0.0, 0.4], [0.0, 0.0, -0.4]], dtype=np.float64)


def update(tracker: PhysicalRelationTracker, left=ZERO, right=ZERO):
    return tracker.update(
        left_ee_position=np.asarray([0.0, 0.0, 0.0]),
        right_ee_position=np.asarray([0.1, 0.0, 0.0]),
        object_position=np.asarray([0.05, 0.0, 0.0]),
        left_finger_forces=left,
        right_finger_forces=right,
    )


def test_relation_lifecycle_is_derived_from_independent_contact_edges() -> None:
    tracker = PhysicalRelationTracker(motion_window_steps=2, confirmation_steps=2, release_steps=2)
    assert update(tracker).label == "none"
    update(tracker, left=CONTACT)
    assert update(tracker, left=CONTACT).label == "left_only"
    update(tracker, left=CONTACT, right=CONTACT)
    assert update(tracker, left=CONTACT, right=CONTACT).label == "both"
    update(tracker, right=CONTACT)
    assert update(tracker, right=CONTACT).label == "right_only"
    update(tracker)
    assert update(tracker).label == "none"


def test_single_finger_force_does_not_create_relation() -> None:
    tracker = PhysicalRelationTracker(motion_window_steps=2, confirmation_steps=1)
    one_finger = CONTACT.copy()
    one_finger[1] = 0.0
    for _ in range(4):
        truth = update(tracker, left=one_finger)
    assert truth.label == "none"
    assert not truth.left.both_fingers_contact


def test_established_contact_survives_transient_relative_settling() -> None:
    tracker = PhysicalRelationTracker(
        motion_window_steps=2,
        confirmation_steps=1,
        release_steps=2,
    )
    update(tracker, left=CONTACT)
    assert update(tracker, left=CONTACT).label == "left_only"

    truth = tracker.update(
        left_ee_position=np.asarray([0.0, 0.0, 0.0]),
        right_ee_position=np.asarray([0.1, 0.0, 0.0]),
        object_position=np.asarray([0.12, 0.0, 0.0]),
        left_finger_forces=CONTACT,
        right_finger_forces=ZERO,
    )
    assert not truth.left.relative_motion_consistent
    assert truth.left.connected
    assert truth.label == "left_only"

    update(tracker)
    assert update(tracker).label == "none"
