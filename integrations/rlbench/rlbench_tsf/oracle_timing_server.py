"""Diagnostic-only TSF server with oracle violation timing.

The server keeps the frozen Full policy, normal task model, relation inference,
repair planner, and legal re-entry logic unchanged.  Its only extra input is
the independently audited cycle at which a physical violation first became
observable.  For a bounded window after that cycle, an already-present raw
task-state mismatch may bypass the normal persistence delay.  The policy must
still infer the mismatch direction, repair target, and re-entry state itself.

This module is intentionally isolated from the formal M5 server.  Importing
``integrations.rlbench.rlbench_tsf.policy_server`` continues to select
the frozen production implementation.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from integrations.rlbench.rlbench_tsf.policy_server import (
    TSFPolicyServer,
)
from integrations.rlbench.rlbench_dynamac.core.task_specs import load_task_specs
from essay2608.policy.tsf import TSFFeatureProfile


ORACLE_TIMING_SCHEMA = "essay2608.iclr2027.oracle-timing-diagnostic.v1"
DEFAULT_ORACLE_WINDOW_CYCLES = 20


class OracleTimingPolicyServer(TSFPolicyServer):
    """Expose audited onset timing without exposing disturbance semantics."""

    def __init__(self, *args: Any, oracle_window_cycles: int = DEFAULT_ORACLE_WINDOW_CYCLES, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if int(oracle_window_cycles) < 1:
            raise ValueError("oracle timing window must be positive")
        self.oracle_window_cycles = int(oracle_window_cycles)
        self._oracle_onset_cycle: int | None = None
        self._oracle_armed_tick: int | None = None
        self._oracle_window_end_tick: int | None = None
        self._oracle_triggered_tick: int | None = None
        self._oracle_expired = False
        self._base_mismatch_configs = {
            arm: controller.mismatch_tracker.config
            for arm, controller in self.policy.execution_controllers.items()
        }
        self._base_recovery_trigger_config = self.policy.recovery_trigger.config
        self.model_identity["oracle_timing_diagnostic"] = {
            "schema": ORACLE_TIMING_SCHEMA,
            "input_fields": ["violation_onset_cycle"],
            "window_cycles": self.oracle_window_cycles,
            "fault_identity_visible": False,
            "repair_target_visible": False,
            "reentry_state_visible": False,
            "formal_m5_path_modified": False,
        }

    def _oracle_status(self) -> dict[str, Any]:
        return {
            "schema": ORACLE_TIMING_SCHEMA,
            "violation_onset_cycle": self._oracle_onset_cycle,
            "armed_tick": self._oracle_armed_tick,
            "window_end_tick": self._oracle_window_end_tick,
            "triggered_tick": self._oracle_triggered_tick,
            "expired": bool(self._oracle_expired),
            "active": bool(
                self._oracle_armed_tick is not None
                and self._oracle_triggered_tick is None
                and not self._oracle_expired
            ),
        }

    def _reset_oracle(self) -> None:
        self._restore_persistence()
        self._oracle_onset_cycle = None
        self._oracle_armed_tick = None
        self._oracle_window_end_tick = None
        self._oracle_triggered_tick = None
        self._oracle_expired = False

    def _enable_immediate_mismatch_emission(self) -> None:
        for arm, controller in self.policy.execution_controllers.items():
            base = self._base_mismatch_configs[arm]
            controller.mismatch_tracker.config = replace(
                base,
                no_plausible_cycles=1,
                relation_mismatch_cycles=1,
            )
        self.policy.recovery_trigger.config = replace(
            self._base_recovery_trigger_config,
            boundary_relation_mismatch_cycles=1,
        )

    def _restore_persistence(self) -> None:
        for arm, controller in self.policy.execution_controllers.items():
            controller.mismatch_tracker.config = self._base_mismatch_configs[arm]
        self.policy.recovery_trigger.config = self._base_recovery_trigger_config

    def _arm_oracle(self, request: Mapping[str, Any]) -> dict[str, Any]:
        if set(request) != {"command", "violation_onset_cycle"}:
            raise ValueError("oracle_event accepts only violation_onset_cycle")
        if self._pending is not None:
            raise RuntimeError("oracle timing must be supplied after action commit")
        onset = request["violation_onset_cycle"]
        if isinstance(onset, bool) or not isinstance(onset, int) or onset < 0:
            raise TypeError("violation_onset_cycle must be a non-negative integer")
        # The evaluator audits cycle n immediately after committing that
        # action, so the server clock must already point to n + 1.
        if onset + 1 != self._tick:
            raise RuntimeError("oracle timing does not match the committed policy clock")
        if self._oracle_onset_cycle is not None:
            if onset != self._oracle_onset_cycle:
                raise RuntimeError("one episode may expose only one violation onset")
            return {"ok": True, "oracle_timing": self._oracle_status()}
        self._oracle_onset_cycle = onset
        self._oracle_armed_tick = self._tick
        self._oracle_window_end_tick = self._tick + self.oracle_window_cycles - 1
        return {"ok": True, "oracle_timing": self._oracle_status()}

    def _act(self, payload: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
        active = bool(
            self._oracle_armed_tick is not None
            and self._oracle_triggered_tick is None
            and not self._oracle_expired
        )
        if active and self._oracle_window_end_tick is not None and self._tick > self._oracle_window_end_tick:
            self._oracle_expired = True
            active = False
        if active:
            self._enable_immediate_mismatch_emission()
        try:
            response = super()._act(payload, **kwargs)
        finally:
            self._restore_persistence()
        monitor = response.get("policy_state", {}).get("monitor", {})
        if bool(monitor.get("alarm", False)) and active:
            self._oracle_triggered_tick = self._tick
        response["oracle_timing"] = self._oracle_status()
        return response

    def handle(self, request: Mapping[str, Any]) -> dict[str, Any]:
        command = request.get("command")
        if command == "oracle_event":
            return self._arm_oracle(request)
        if command == "oracle_status":
            if set(request) != {"command"}:
                raise ValueError("oracle_status accepts no fields")
            return {"ok": True, "oracle_timing": self._oracle_status()}
        if command == "reset":
            self._reset_oracle()
        return super().handle(request)


def serve(
    task: str,
    models_dir: Path,
    base_models_dir: Path,
    *,
    diagnostics_dir: Path | None = None,
    feature_profile: str = "full",
    task_spec: Any = None,
    boundary_config: Path | None = None,
    oracle_window_cycles: int = DEFAULT_ORACLE_WINDOW_CYCLES,
) -> int:
    server = OracleTimingPolicyServer(
        task,
        models_dir,
        base_models_dir,
        diagnostics_dir=diagnostics_dir,
        feature_profile=feature_profile,
        task_spec=task_spec,
        boundary_config=boundary_config,
        oracle_window_cycles=oracle_window_cycles,
    )
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            response = server.handle(request)
        except Exception as exc:
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(response, separators=(",", ":")), flush=True)
        if response.get("closed"):
            return 0
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("serve",))
    parser.add_argument("--task", required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--base-models-dir", type=Path, required=True)
    parser.add_argument("--diagnostics-dir", type=Path)
    parser.add_argument("--task-specs", type=Path)
    parser.add_argument("--boundary-config", type=Path)
    parser.add_argument(
        "--feature-profile",
        choices=TSFFeatureProfile.names(),
        default="full",
    )
    parser.add_argument(
        "--oracle-window-cycles",
        type=int,
        default=DEFAULT_ORACLE_WINDOW_CYCLES,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    task_spec = (
        None
        if args.task_specs is None
        else load_task_specs(args.task_specs)[args.task]
    )
    return serve(
        args.task,
        args.models_dir,
        args.base_models_dir,
        diagnostics_dir=args.diagnostics_dir,
        feature_profile=args.feature_profile,
        task_spec=task_spec,
        boundary_config=args.boundary_config,
        oracle_window_cycles=args.oracle_window_cycles,
    )


if __name__ == "__main__":
    raise SystemExit(main())
