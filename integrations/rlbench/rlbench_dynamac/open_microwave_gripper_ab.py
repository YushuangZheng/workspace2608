"""Paired OpenMicrowave gripper-timing diagnostic for the V4 controller.

This is an independent development diagnostic, not a formal evaluation cell.
It never reads either sealed evaluation set and it never writes a paper table.
For every development seed, variant A executes the current V3 policy command;
variant B changes only gripper scalar 7 to ``0`` while targeting committed
global policy tick 113.  Failed primary actions abort the transaction and retry
the same committed tick, so the intervention cannot advance on an invalid
attempt.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .records import atomic_json, reserve_output
from .runtime import (
    DEFAULT_FINAL_SETTLING_PHYSICS_STEPS,
    DEFAULT_MAX_PRIMARY_ACTION_ATTEMPTS,
)
from .unimanual_evaluate import (
    INTEGRATION_ROOT,
    DEFAULT_MODELS_DIR,
    DEFAULT_POLICY_PYTHON,
    PolicyProcess,
    _make_action_mode,
    _observation_payload,
    _prepare_low_dim_headless_scene,
    _run_episode,
)

TASK_NAME = "open_microwave"
TASK_MODULE = "rlbench.tasks.open_microwave"
TASK_CLASS = "OpenMicrowave"
SCENARIO = "static"
VARIATION = 0
DEV_BASE_SEED = 4_104_100_000
DEV_EPISODES = 40
HORIZON = 1_000
EARLY_CLOSE_TICK = 113
GRIPPER_ACTION_INDEX = 7
ACTION_DIMENSION = 9
OPEN_GRIPPER = 1.0
CLOSED_GRIPPER = 0.0
DIAGNOSTIC_SCHEMA = "dynamac-open-microwave-gripper-timing-ab-v4"
DIAGNOSTIC_PROTOCOL_ID = (
    "open-microwave-paired-static-dev-g-t-vs-close-at-committed-tick113-v4"
)
STATUS = "PROVISIONAL"
COMPARABILITY = "NON_COMPARABLE"
DEFAULT_OUTPUT_DIR = (
    INTEGRATION_ROOT
    / "results"
    / "v4"
    / "diagnostics"
    / "open_microwave_gripper_timing_ab"
)
DEFAULT_JSON_NAME = "summary.json"
DEFAULT_CSV_NAME = "paired_episodes.csv"


def _canonical_fingerprint(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def paired_execution_order(episode):
    """Alternate AB/BA while keeping each episode's seed paired."""

    if isinstance(episode, bool) or not isinstance(episode, int) or episode < 0:
        raise ValueError("paired diagnostic episode must be a non-negative integer")
    return ("A", "B") if episode % 2 == 0 else ("B", "A")


def apply_gripper_timing_variant(
    action,
    *,
    variant,
    committed_tick,
    early_close_tick=EARLY_CLOSE_TICK,
):
    """Return one 9D action with, at most, scalar 7 changed for variant B."""

    if variant not in {"A", "B"}:
        raise ValueError("gripper-timing variant must be A or B")
    if (
        isinstance(committed_tick, bool)
        or not isinstance(committed_tick, int)
        or committed_tick < 0
    ):
        raise ValueError("committed policy tick must be a non-negative integer")
    array = np.asarray(action, dtype=np.float64)
    if array.shape != (ACTION_DIMENSION,) or not np.all(np.isfinite(array)):
        raise ValueError("OpenMicrowave diagnostic requires one finite 9D action")
    emitted = array.copy()
    if variant == "B" and committed_tick == early_close_tick:
        emitted[GRIPPER_ACTION_INDEX] = CLOSED_GRIPPER
    differences = np.flatnonzero(emitted != array).tolist()
    if any(index != GRIPPER_ACTION_INDEX for index in differences):
        raise RuntimeError("gripper-timing wrapper changed a non-gripper scalar")
    return emitted


class CommittedTickGripperWorker:
    """Policy proxy whose clock advances only after the base worker commits."""

    def __init__(self, worker, variant, *, early_close_tick=EARLY_CLOSE_TICK):
        if variant not in {"A", "B"}:
            raise ValueError("gripper-timing variant must be A or B")
        self._worker = worker
        self.variant = variant
        self.early_close_tick = int(early_close_tick)
        self.committed_tick = 0
        self._pending = None
        self.attempts = []
        self.aborted_attempts = []
        self.committed_actions = []

    def __getattr__(self, name):
        return getattr(self._worker, name)

    @property
    def primary_action_pending(self):
        return self._pending is not None

    def request(self, command, observation=None, **fields):
        if command == "reset":
            if self._pending is not None:
                raise RuntimeError("cannot reset with a pending policy transaction")
            self.committed_tick = 0
            self.attempts = []
            self.aborted_attempts = []
            self.committed_actions = []
            return self._worker.request(command, observation, **fields)
        if command == "act":
            if self._pending is not None:
                raise RuntimeError(
                    "policy returned a second action before commit/abort"
                )
            response = self._worker.request(command, observation, **fields)
            action = response.get("action")
            if action is None:
                return response
            transaction_id = response.get("transaction_id")
            if not isinstance(transaction_id, int) or isinstance(transaction_id, bool):
                raise RuntimeError("policy action has no integer transaction ID")
            original = np.asarray(action, dtype=np.float64)
            emitted = apply_gripper_timing_variant(
                original,
                variant=self.variant,
                committed_tick=self.committed_tick,
                early_close_tick=self.early_close_tick,
            )
            differences = np.flatnonzero(emitted != original).tolist()
            record = {
                "attempt": len(self.attempts) + 1,
                "transaction_id": transaction_id,
                "committed_tick_target": self.committed_tick,
                "original_gripper": float(original[GRIPPER_ACTION_INDEX]),
                "emitted_gripper": float(emitted[GRIPPER_ACTION_INDEX]),
                "differing_indices": differences,
                "original_action_fingerprint": _canonical_fingerprint(
                    original.tolist()
                ),
                "emitted_action_fingerprint": _canonical_fingerprint(
                    emitted.tolist()
                ),
            }
            self.attempts.append(record)
            self._pending = {**record, "emitted_action": emitted.tolist()}
            if differences:
                response = dict(response)
                response["action"] = emitted.tolist()
            return response
        if command == "abort":
            transaction_id = fields.get("transaction_id")
            self._require_pending_transaction(transaction_id)
            response = self._worker.request(command, observation, **fields)
            self.aborted_attempts.append(
                {
                    key: value
                    for key, value in self._pending.items()
                    if key != "emitted_action"
                }
            )
            self._pending = None
            # Deliberately no committed-tick increment here.
            return response
        if command == "commit":
            transaction_id = fields.get("transaction_id")
            self._require_pending_transaction(transaction_id)
            response = self._worker.request(command, observation, **fields)
            self.committed_actions.append(dict(self._pending))
            self._pending = None
            self.committed_tick += 1
            return response
        return self._worker.request(command, observation, **fields)

    def _require_pending_transaction(self, transaction_id):
        if self._pending is None:
            raise RuntimeError("policy transaction is not pending")
        if transaction_id != self._pending["transaction_id"]:
            raise RuntimeError("policy transaction ID does not match pending action")

    def metadata(self):
        close_ticks = [
            row["committed_tick_target"]
            for row in self.committed_actions
            if row["emitted_gripper"] < 0.5
        ]
        original_close_ticks = [
            row["committed_tick_target"]
            for row in self.committed_actions
            if row["original_gripper"] < 0.5
        ]

        def committed_at(tick):
            return next(
                (
                    {
                        key: value
                        for key, value in row.items()
                        if key != "emitted_action"
                    }
                    for row in self.committed_actions
                    if row["committed_tick_target"] == tick
                ),
                None,
            )

        target_attempts = [
            row
            for row in self.attempts
            if row["committed_tick_target"] == self.early_close_tick
        ]
        invalid_differences = [
            row
            for row in self.attempts
            if any(
                index != GRIPPER_ACTION_INDEX
                for index in row["differing_indices"]
            )
            or (
                row["differing_indices"]
                and not (
                    self.variant == "B"
                    and row["committed_tick_target"] == self.early_close_tick
                )
            )
        ]
        return {
            "variant": self.variant,
            "committed_policy_steps": self.committed_tick,
            "early_close_tick": self.early_close_tick,
            "gripper_action_index": GRIPPER_ACTION_INDEX,
            "command_close_tick": close_ticks[0] if close_ticks else None,
            "original_policy_close_tick": (
                original_close_ticks[0] if original_close_ticks else None
            ),
            "target_tick_attempt_count": len(target_attempts),
            "target_tick_committed": committed_at(self.early_close_tick),
            "next_tick_committed": committed_at(self.early_close_tick + 1),
            "aborted_attempt_count": len(self.aborted_attempts),
            "aborted_attempts": list(self.aborted_attempts),
            "only_allowed_scalar_changed": not invalid_differences,
            "protocol_mutation_exercised": any(
                row["differing_indices"] == [GRIPPER_ACTION_INDEX]
                for row in target_attempts
            ),
        }


class DoorGripperMetrics:
    """Observe physical gripper transitions and microwave-door motion."""

    def __init__(self, door_position, gripper_open, *, observation=None):
        self._door_position = door_position
        self._gripper_open = gripper_open
        self.door_samples = []
        self.gripper_samples = []
        self.close_transitions = []
        self.sample(
            observation=observation,
            committed_tick=0,
            phase="initial",
        )
        self.initial_door_position = self.door_samples[0]["position_rad"]

    def _read_gripper(self, observation):
        value = getattr(observation, "gripper_open", None)
        if value is None:
            value = self._gripper_open()
        array = np.asarray(value, dtype=np.float64).reshape(-1)
        if not array.size or not np.all(np.isfinite(array)):
            raise RuntimeError("diagnostic gripper-open measurement is invalid")
        return float(np.mean(array))

    def sample(self, *, observation, committed_tick, phase):
        door = float(self._door_position())
        gripper = self._read_gripper(observation)
        if not math.isfinite(door) or not math.isfinite(gripper):
            raise RuntimeError("diagnostic physical measurement is non-finite")
        sample_index = len(self.door_samples)
        self.door_samples.append(
            {
                "sample": sample_index,
                "committed_tick": int(committed_tick),
                "phase": str(phase),
                "position_rad": door,
            }
        )
        previous = (
            self.gripper_samples[-1]["open_amount"]
            if self.gripper_samples
            else None
        )
        self.gripper_samples.append(
            {
                "sample": sample_index,
                "committed_tick": int(committed_tick),
                "phase": str(phase),
                "open_amount": gripper,
            }
        )
        if previous is not None and previous >= 0.5 and gripper < 0.5:
            self.close_transitions.append(
                {
                    "sample": sample_index,
                    "committed_tick": int(committed_tick),
                    "phase": str(phase),
                    "previous_open_amount": previous,
                    "open_amount": gripper,
                }
            )

    def result(self, *, committed_tick):
        self.sample(
            observation=None,
            committed_tick=committed_tick,
            phase="final",
        )
        positions = [row["position_rad"] for row in self.door_samples]
        displacements = [
            abs(position - self.initial_door_position) for position in positions
        ]
        peak_index = int(np.argmax(displacements))
        transition = self.close_transitions[0] if self.close_transitions else None
        return {
            "actual_gripper_close_transition": transition,
            "actual_gripper_close_transition_tick": (
                transition["committed_tick"] if transition is not None else None
            ),
            "door_joint_initial_position_rad": self.initial_door_position,
            "door_joint_peak_position_rad": positions[peak_index],
            "door_joint_peak_displacement_rad": displacements[peak_index],
            "door_joint_final_position_rad": positions[-1],
            "door_joint_final_displacement_rad": displacements[-1],
            "measurement_samples": len(positions),
        }


class _MeasuredScene:
    def __init__(self, scene, metrics, worker):
        self._scene = scene
        self._metrics = metrics
        self._worker = worker

    def __getattr__(self, name):
        return getattr(self._scene, name)

    def step(self):
        value = self._scene.step()
        self._metrics.sample(
            observation=None,
            committed_tick=self._worker.committed_tick,
            phase="raw_settling",
        )
        return value


class MeasuredTaskEnvironment:
    """Transparent TaskEnvironment proxy used only by this diagnostic."""

    def __init__(self, task_environment, metrics, worker):
        self._task_environment = task_environment
        self._metrics = metrics
        self._worker = worker
        self._scene = _MeasuredScene(task_environment._scene, metrics, worker)

    def __getattr__(self, name):
        return getattr(self._task_environment, name)

    def get_observation(self):
        observation = self._task_environment.get_observation()
        self._metrics.sample(
            observation=observation,
            committed_tick=self._worker.committed_tick,
            phase="get_observation",
        )
        return observation

    def step(self, action):
        phase = (
            "primary_action"
            if self._worker.primary_action_pending
            else "retry_noop"
        )
        try:
            result = self._task_environment.step(action)
        except Exception:
            self._metrics.sample(
                observation=None,
                committed_tick=self._worker.committed_tick,
                phase=f"{phase}_exception",
            )
            raise
        self._metrics.sample(
            observation=result[0],
            committed_tick=self._worker.committed_tick,
            phase=phase,
        )
        return result


def _door_and_gripper_getters(task_environment):
    from pyrep.objects.joint import Joint

    door = Joint("microwave_door_joint")
    gripper = task_environment._scene.robot.gripper

    def gripper_open():
        return gripper.get_open_amount()

    return door.get_joint_position, gripper_open


def _fresh_generation_summary(evidence):
    return {
        "fingerprint": _canonical_fingerprint(evidence),
        "generation_index": evidence.get("generation_index"),
        "episode_seed": evidence.get("episode_seed"),
        "variation": evidence.get("variation"),
        "task_name": evidence.get("task_name"),
    }


def _run_variant(
    *,
    environment,
    task_class,
    worker,
    action_mode,
    args,
    episode,
    seed,
    variant,
    execution_order,
):
    from .runtime import initialize_fresh_task_generation

    before_diagnostics = action_mode.arm_action_mode.diagnostics()
    (
        task_environment,
        descriptions,
        observation,
        fresh_task_generation,
    ) = initialize_fresh_task_generation(
        environment,
        task_class,
        episode_seed=seed,
        variation=VARIATION,
        verify_instance=False,
    )
    initial_fingerprint = _canonical_fingerprint(_observation_payload(observation))
    wrapped_worker = CommittedTickGripperWorker(worker, variant)
    door_position, gripper_open = _door_and_gripper_getters(task_environment)
    metrics = DoorGripperMetrics(
        door_position,
        gripper_open,
        observation=observation,
    )
    measured_environment = MeasuredTaskEnvironment(
        task_environment,
        metrics,
        wrapped_worker,
    )
    result = _run_episode(
        measured_environment,
        wrapped_worker,
        args,
        episode,
        motion_plan=None,
        descriptions=descriptions,
        observation=observation,
        fresh_task_generation=fresh_task_generation,
    )
    after_diagnostics = action_mode.arm_action_mode.diagnostics()
    diagnostic_delta = {
        key: after_diagnostics[key] - before_diagnostics.get(key, 0)
        for key in after_diagnostics
        if isinstance(after_diagnostics[key], (int, float))
        and isinstance(before_diagnostics.get(key, 0), (int, float))
    }
    worker_metadata = wrapped_worker.metadata()
    physical = metrics.result(committed_tick=wrapped_worker.committed_tick)
    return {
        "episode": episode,
        "seed": seed,
        "variation": VARIATION,
        "variant": variant,
        "execution_order": execution_order,
        "success": bool(result["success"]),
        "reason": result["reason"],
        "invalid": int(result.get("invalid_actions", 0)) > 0,
        "invalid_actions": int(result.get("invalid_actions", 0)),
        "steps": int(result.get("steps", 0)),
        "control_attempts": int(result.get("control_attempts", 0)),
        "committed_policy_steps": int(result.get("committed_policy_steps", 0)),
        "initial_observation_fingerprint": initial_fingerprint,
        "command": worker_metadata,
        "physical": physical,
        "ik_execution_diagnostics_delta": diagnostic_delta,
        "final_settling": result.get("final_settling"),
        "fresh_task_generation": _fresh_generation_summary(
            fresh_task_generation
        ),
    }


def _pair_outcome(a_success, b_success):
    if a_success and b_success:
        return "both_success"
    if not a_success and not b_success:
        return "both_failure"
    if not a_success and b_success:
        return "rescue"
    return "regression"


def build_pair_record(*, episode, seed, order, runs):
    a = runs["A"]
    b = runs["B"]
    initial_match = (
        a["initial_observation_fingerprint"]
        == b["initial_observation_fingerprint"]
    )
    a_target = a["command"]["target_tick_committed"]
    b_target = b["command"]["target_tick_committed"]
    target_original_match = (
        a_target is not None
        and b_target is not None
        and a_target["original_action_fingerprint"]
        == b_target["original_action_fingerprint"]
    )
    allowed = (
        a["command"]["only_allowed_scalar_changed"]
        and b["command"]["only_allowed_scalar_changed"]
    )
    target_command_valid = bool(
        b_target is not None
        and b_target["differing_indices"] == [GRIPPER_ACTION_INDEX]
        and b_target["original_gripper"] >= 0.5
        and b_target["emitted_gripper"] == CLOSED_GRIPPER
        and b["command"]["command_close_tick"] == EARLY_CLOSE_TICK
    )
    protocol_valid = bool(
        initial_match
        and allowed
        and target_original_match
        and target_command_valid
        and b["command"]["protocol_mutation_exercised"]
    )
    return {
        "episode": episode,
        "seed": seed,
        "variation": VARIATION,
        "execution_order": list(order),
        "initial_state_match": initial_match,
        "target_tick_original_action_match": target_original_match,
        "target_tick_command_valid": target_command_valid,
        "only_allowed_scalar_changed": allowed,
        "single_variable_exercised": bool(
            b["command"]["protocol_mutation_exercised"]
        ),
        "protocol_pair_valid": protocol_valid,
        "outcome": _pair_outcome(a["success"], b["success"]),
        "runs": {"A": a, "B": b},
    }


def _mcnemar_exact(rescues, regressions):
    discordant = int(rescues) + int(regressions)
    if discordant == 0:
        return 1.0
    tail = min(int(rescues), int(regressions))
    numerator = sum(math.comb(discordant, value) for value in range(tail + 1))
    return min(1.0, 2.0 * numerator / float(2**discordant))


def paired_statistics(pairs):
    valid = [pair for pair in pairs if pair.get("protocol_pair_valid") is True]
    counts = {
        outcome: sum(pair["outcome"] == outcome for pair in valid)
        for outcome in (
            "both_success",
            "both_failure",
            "rescue",
            "regression",
        )
    }
    rescues = counts["rescue"]
    regressions = counts["regression"]
    a_successes = counts["both_success"] + regressions
    b_successes = counts["both_success"] + rescues
    denominator = len(valid)
    exercised = [pair for pair in valid if pair["single_variable_exercised"]]
    return {
        "planned_pairs": len(pairs),
        "protocol_valid_pairs": denominator,
        "protocol_invalid_pairs": len(pairs) - denominator,
        "single_variable_exercised_pairs": len(exercised),
        "both_success": counts["both_success"],
        "both_failure": counts["both_failure"],
        "paired_rescues_a_fail_b_success": rescues,
        "paired_regressions_a_success_b_fail": regressions,
        "discordant_pairs": rescues + regressions,
        "a_successes": a_successes,
        "b_successes": b_successes,
        "a_success_rate": a_successes / float(denominator) if denominator else None,
        "b_success_rate": b_successes / float(denominator) if denominator else None,
        "paired_success_rate_delta_b_minus_a": (
            (b_successes - a_successes) / float(denominator)
            if denominator
            else None
        ),
        "mcnemar_exact_two_sided_p": _mcnemar_exact(rescues, regressions),
        "mcnemar_method": "two_sided_exact_binomial_discordant_pairs",
    }


def _csv_rows(pairs):
    for pair in pairs:
        a = pair["runs"]["A"]
        b = pair["runs"]["B"]
        yield {
            "status": STATUS,
            "comparability": COMPARABILITY,
            "formal_table_eligible": False,
            "episode": pair["episode"],
            "seed": pair["seed"],
            "variation": pair["variation"],
            "execution_order": "".join(pair["execution_order"]),
            "protocol_pair_valid": pair["protocol_pair_valid"],
            "initial_state_match": pair["initial_state_match"],
            "single_variable_exercised": pair["single_variable_exercised"],
            "outcome": pair["outcome"],
            "a_success": a["success"],
            "b_success": b["success"],
            "a_reason": a["reason"],
            "b_reason": b["reason"],
            "a_invalid": a["invalid"],
            "b_invalid": b["invalid"],
            "a_invalid_actions": a["invalid_actions"],
            "b_invalid_actions": b["invalid_actions"],
            "a_command_close_tick": a["command"]["command_close_tick"],
            "b_command_close_tick": b["command"]["command_close_tick"],
            "a_actual_gripper_transition_tick": a["physical"][
                "actual_gripper_close_transition_tick"
            ],
            "b_actual_gripper_transition_tick": b["physical"][
                "actual_gripper_close_transition_tick"
            ],
            "a_door_initial_position_rad": a["physical"][
                "door_joint_initial_position_rad"
            ],
            "b_door_initial_position_rad": b["physical"][
                "door_joint_initial_position_rad"
            ],
            "a_door_peak_position_rad": a["physical"][
                "door_joint_peak_position_rad"
            ],
            "b_door_peak_position_rad": b["physical"][
                "door_joint_peak_position_rad"
            ],
            "a_door_peak_displacement_rad": a["physical"][
                "door_joint_peak_displacement_rad"
            ],
            "b_door_peak_displacement_rad": b["physical"][
                "door_joint_peak_displacement_rad"
            ],
            "a_door_final_position_rad": a["physical"][
                "door_joint_final_position_rad"
            ],
            "b_door_final_position_rad": b["physical"][
                "door_joint_final_position_rad"
            ],
            "a_door_final_displacement_rad": a["physical"][
                "door_joint_final_displacement_rad"
            ],
            "b_door_final_displacement_rad": b["physical"][
                "door_joint_final_displacement_rad"
            ],
        }


def _atomic_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError("paired diagnostic CSV requires at least one row")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-",
        dir=str(path.parent),
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_outputs(summary, output_dir=DEFAULT_OUTPUT_DIR):
    output_dir = Path(output_dir)
    json_path = output_dir / DEFAULT_JSON_NAME
    csv_path = output_dir / DEFAULT_CSV_NAME
    with reserve_output(json_path), reserve_output(csv_path):
        atomic_json(json_path, summary)
        _atomic_csv(csv_path, _csv_rows(summary["pairs"]))
    return json_path, csv_path


def _diagnostic_args():
    return SimpleNamespace(
        task=TASK_NAME,
        scenario=SCENARIO,
        seed=DEV_BASE_SEED,
        variation=VARIATION,
        horizon=HORIZON,
        trigger_fraction=1.0 / 3.0,
        trigger_step=None,
        smooth_steps=10,
        intervention_attempts=100,
        max_primary_action_attempts=DEFAULT_MAX_PRIMARY_ACTION_ATTEMPTS,
        final_settling_steps=DEFAULT_FINAL_SETTLING_PHYSICS_STEPS,
    )


def run_diagnostic(
    *,
    models_dir=DEFAULT_MODELS_DIR,
    policy_python=DEFAULT_POLICY_PYTHON,
    policy_timeout=120.0,
    headless=True,
    output_dir=DEFAULT_OUTPUT_DIR,
):
    """Run the independent 40-seed paired cohort and write JSON plus CSV."""

    import importlib
    import rlbench.environment as environment_module
    from rlbench.observation_config import ObservationConfig

    if Path(models_dir).resolve() != DEFAULT_MODELS_DIR.resolve():
        raise ValueError(
            "OpenMicrowave diagnostic is frozen to the unchanged V3 checkpoint"
        )
    task_class = getattr(importlib.import_module(TASK_MODULE), TASK_CLASS)
    observation_config = ObservationConfig()
    observation_config.set_all(False)
    observation_config.gripper_open = True
    observation_config.gripper_pose = True
    observation_config.task_low_dim_state = True
    action_mode = _make_action_mode()
    environment = environment_module.Environment(
        action_mode=action_mode,
        obs_config=observation_config,
        headless=headless,
    )
    restore_scene, scene_launch = _prepare_low_dim_headless_scene(
        environment_module,
        enabled=headless,
    )
    args = _diagnostic_args()
    worker = None
    launched = False
    pairs = []
    try:
        worker = PolicyProcess(
            policy_python,
            TASK_NAME,
            models_dir,
            timeout=policy_timeout,
        )
        if worker.policy_steps <= EARLY_CLOSE_TICK + 1:
            raise RuntimeError("OpenMicrowave policy ends before diagnostic tick 114")
        environment.launch()
        launched = True
        for episode in range(DEV_EPISODES):
            seed = DEV_BASE_SEED + episode
            order = paired_execution_order(episode)
            runs = {}
            for execution_order, variant in enumerate(order):
                runs[variant] = _run_variant(
                    environment=environment,
                    task_class=task_class,
                    worker=worker,
                    action_mode=action_mode,
                    args=args,
                    episode=episode,
                    seed=seed,
                    variant=variant,
                    execution_order=execution_order,
                )
            pair = build_pair_record(
                episode=episode,
                seed=seed,
                order=order,
                runs=runs,
            )
            pairs.append(pair)
            print(
                f"OpenMicrowave paired dev {episode + 1}/{DEV_EPISODES}: "
                f"{pair['outcome']} ({''.join(order)})",
                flush=True,
            )
        summary = {
            "schema": DIAGNOSTIC_SCHEMA,
            "protocol_id": DIAGNOSTIC_PROTOCOL_ID,
            "status": STATUS,
            "comparability": COMPARABILITY,
            "paper_comparable": False,
            "formal_table_eligible": False,
            "task": TASK_NAME,
            "scenario": SCENARIO,
            "variation": VARIATION,
            "cohort": {
                "kind": "independent_development_cohort",
                "base_seed": DEV_BASE_SEED,
                "episodes": DEV_EPISODES,
                "seed_schedule": "base_seed + episode",
                "paired_same_seed_and_variation": True,
                "pair_order": "AB_even_episode_BA_odd_episode",
                "evaluation_set_access": "none",
            },
            "intervention": {
                "variant_a": "current_v3_gripper_command_g[t]",
                "variant_b": (
                    "set_only_action_index7_to_closed_at_committed_global_tick113"
                ),
                "early_close_tick": EARLY_CLOSE_TICK,
                "gripper_action_index": GRIPPER_ACTION_INDEX,
                "pose_action_indices": list(range(7)),
                "ignore_collisions_action_index": 8,
                "retry_clock": "abort_does_not_advance_committed_tick",
            },
            "execution": {
                "horizon": HORIZON,
                "max_primary_action_attempts": DEFAULT_MAX_PRIMARY_ACTION_ATTEMPTS,
                "final_settling_steps": DEFAULT_FINAL_SETTLING_PHYSICS_STEPS,
                "ik": "current_v4_collision_aware_unimanual_action_mode",
                "scene_launch": scene_launch,
                "model_source": "unchanged_v3_open_microwave_checkpoint",
                "models_dir": str(DEFAULT_MODELS_DIR),
            },
            "model_identity": worker.model_identity,
            "statistics": paired_statistics(pairs),
            "ik_execution_diagnostics": action_mode.arm_action_mode.diagnostics(),
            "claim_boundary": (
                "PROVISIONAL/NON_COMPARABLE single-variable development diagnostic. "
                "It is not rlbench_eval_v2, not rlbench_fixed_v1, and must not be "
                "included in any formal Table I-III result."
            ),
            "pairs": pairs,
        }
    finally:
        try:
            if worker is not None:
                worker.close()
            if launched:
                environment.shutdown()
        finally:
            restore_scene()
    paths = write_outputs(summary, output_dir)
    return summary, paths


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-python", type=Path, default=DEFAULT_POLICY_PYTHON)
    parser.add_argument("--policy-timeout", type=float, default=120.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--headless", dest="headless", action="store_true")
    display.add_argument("--no-headless", dest="headless", action="store_false")
    parser.set_defaults(headless=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.policy_timeout <= 0.0:
        raise ValueError("policy timeout must be positive")
    resolved_output = args.output_dir.resolve()
    expected_root = DEFAULT_OUTPUT_DIR.resolve()
    try:
        resolved_output.relative_to(expected_root)
    except ValueError as error:
        raise ValueError(
            "OpenMicrowave diagnostic output must remain below its V4 directory"
        ) from error
    summary, paths = run_diagnostic(
        models_dir=DEFAULT_MODELS_DIR,
        policy_python=args.policy_python,
        policy_timeout=args.policy_timeout,
        headless=args.headless,
        output_dir=args.output_dir,
    )
    print(f"wrote {paths[0]}")
    print(f"wrote {paths[1]}")
    return 0 if summary["statistics"]["protocol_invalid_pairs"] == 0 else 2


__all__ = [
    "COMPARABILITY",
    "CommittedTickGripperWorker",
    "DEFAULT_OUTPUT_DIR",
    "DEV_BASE_SEED",
    "DEV_EPISODES",
    "DIAGNOSTIC_PROTOCOL_ID",
    "DIAGNOSTIC_SCHEMA",
    "DoorGripperMetrics",
    "EARLY_CLOSE_TICK",
    "GRIPPER_ACTION_INDEX",
    "MeasuredTaskEnvironment",
    "STATUS",
    "apply_gripper_timing_variant",
    "build_pair_record",
    "paired_execution_order",
    "paired_statistics",
    "run_diagnostic",
    "write_outputs",
]


if __name__ == "__main__":
    raise SystemExit(main())
