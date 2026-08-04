from __future__ import annotations

import numpy as np
import pytest

from essay2608.eval.perturbations import RECOVERY_CONDITIONS, recovery_condition_parameters


def test_recovery_protocol_declares_exact_seven_conditions() -> None:
    assert RECOVERY_CONDITIONS == (
        "drop_lift_early",
        "drop_transport_middle",
        "drop_before_lower",
        "miss_small_shift",
        "miss_large_shift",
        "edge_grasp",
        "normal_no_failure",
    )


@pytest.mark.parametrize(
    ("condition", "phase", "phase_step"),
    [
        ("drop_lift_early", 4, 8),
        ("drop_transport_middle", 5, 12),
        ("drop_before_lower", 5, 22),
    ],
)
def test_drop_timing_and_seed_variants_are_frozen(condition: str, phase: int, phase_step: int) -> None:
    distances = []
    directions = set()
    forced_open = set()
    for seed in range(6500, 6520):
        config = recovery_condition_parameters(condition, seed)
        distances.append(config["distance_m"])
        directions.add(config["direction"])
        forced_open.add(config["force_open_steps"])
        assert config["trigger_phase"] == phase
        assert config["trigger_phase_step"] == phase_step
        assert np.isclose(np.linalg.norm(config["shift_m"]), config["distance_m"])
    assert set(distances) == {0.05, 0.10, 0.15, 0.20}
    assert directions == {"front", "back", "left", "right"}
    assert forced_open == {0, 3}


@pytest.mark.parametrize(
    ("condition", "distance"),
    [
        ("miss_small_shift", 0.030),
        ("miss_large_shift", 0.100),
        ("edge_grasp", 0.018),
    ],
)
def test_miss_geometry_is_fixed_before_held_out_run(condition: str, distance: float) -> None:
    config = recovery_condition_parameters(condition, 6500)
    assert config["kind"] == "miss"
    assert config["trigger_phase"] == 3
    assert config["trigger_phase_step"] == 0
    assert np.isclose(config["distance_m"], distance)


def test_normal_condition_has_no_event() -> None:
    assert recovery_condition_parameters("normal_no_failure", 6500) == {
        "condition": "normal_no_failure",
        "seed": 6500,
        "kind": "none",
    }
