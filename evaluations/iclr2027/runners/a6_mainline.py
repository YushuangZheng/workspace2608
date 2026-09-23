"""Run the current A6 experiment sequence with explicit completion gates."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[3]
EVAL_ROOT = ROOT / "evaluations" / "iclr2027"
RESULTS = EVAL_ROOT / "results"
STATE_ROOT = RESULTS / "a6_execution"
STATE_PATH = STATE_ROOT / "A6_MAINLINE_STATUS.json"
LOG_PATH = STATE_ROOT / "A6_MAINLINE.log"


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _record(step: str, status: str, **extra: Any) -> None:
    previous = _read_json(STATE_PATH) if STATE_PATH.is_file() else {}
    history = list(previous.get("history", []))
    history.append(
        {
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "step": step,
            "status": status,
            **extra,
        }
    )
    _atomic_json(
        STATE_PATH,
        {
            "schema": "essay2608.iclr2027.a6-mainline-status.v1",
            "status": status,
            "current_step": step,
            "history": history,
        },
    )


def _run(step: str, module: str, *args: str, allowed: Iterable[int] = (0,)) -> None:
    _record(step, "RUNNING")
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as stream:
        stream.write(f"\n[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] START {step}\n")
        stream.flush()
        completed = subprocess.run(
            [sys.executable, "-m", module, *args],
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        stream.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] END {step} rc={completed.returncode}\n")
    if completed.returncode not in set(allowed):
        _record(step, "FAILED", returncode=completed.returncode)
        raise SystemExit(completed.returncode)
    _record(step, "PASS", returncode=completed.returncode)


def _assert_e6_development() -> None:
    records = RESULTS / "native_v3" / "ours" / "development" / "records"
    if len(list(records.glob("*.json"))) != 180:
        raise RuntimeError("E6 Ours development is not 180/180")
    gate = _read_json(records.parent / "GATE_AMENDMENT_5.json")
    if gate.get("status") != "PASS":
        raise RuntimeError("E6 Ours development gate did not pass")


def _assert_e6_formal() -> None:
    records = RESULTS / "native_v3" / "ours" / "formal" / "records"
    if len(list(records.glob("*.json"))) != 600:
        raise RuntimeError("E6 Ours formal is not 600/600")


def _is_pass(path: Path) -> bool:
    return path.is_file() and str(_read_json(path).get("status", "")).startswith("PASS")


def main() -> int:
    # E2, E3-C, the original E4 cells, E6, and E5 core have already run.
    # This entry point now performs only the outcome-independent post-E4
    # refresh, the corrected nested-paired E4 panel, the remaining appendix
    # ablations, and the final re-aggregation.  It must never revive the
    # cancelled M4+Retry physical study or the superseded independent E4
    # per-stage protocol.
    _record("post_e4_remaining_mainline", "PASS")

    main10_promotion = RESULTS / "controlled" / "e1_e2" / "post_e4_m5_refresh" / "PROMOTION.json"
    if not _is_pass(main10_promotion):
        _run("post_e4_main10_refresh", "evaluations.iclr2027.runners.a6_post_e4_main10_refresh", "run", allowed=(0, 2))
        _run("post_e4_main10_promote", "evaluations.iclr2027.analysis.promote_post_e4_main10_refresh")
    else:
        _record("post_e4_main10_refresh_reused", "PASS")

    lift_tray_promotion = (
        RESULTS
        / "controlled"
        / "e1_e2"
        / "m5_lift_tray_perturbed_refresh"
        / "PROMOTION.json"
    )
    if not _is_pass(lift_tray_promotion):
        _run(
            "m5_lift_tray_perturbed_refresh",
            "evaluations.iclr2027.runners.a6_m5_lift_tray_perturbed_refresh",
            "run",
            allowed=(0, 2),
        )
        _run(
            "m5_lift_tray_perturbed_promote",
            "evaluations.iclr2027.analysis.promote_m5_lift_tray_perturbed_refresh",
        )
    else:
        _record("m5_lift_tray_perturbed_refresh_reused", "PASS")

    handover_promotion = RESULTS / "controlled" / "post_e4_handover_refresh" / "PROMOTION.json"
    if not _is_pass(handover_promotion):
        _run("post_e4_handover_refresh", "evaluations.iclr2027.runners.a6_post_e4_handover_refresh", "run", allowed=(0, 2))
        _run("post_e4_handover_promote", "evaluations.iclr2027.analysis.promote_post_e4_handover_refresh")
    else:
        _record("post_e4_handover_refresh_reused", "PASS")

    # The post-E4 promotions replace complete retained M5/M6 cells.  Rebuild
    # the versioned physical-event audit from those promoted bytes before any
    # Figure 3 statistics are regenerated; otherwise the event rows would
    # still describe the superseded trajectories under the same episode IDs.
    _run("a5_physical_reaudit_refresh", "evaluations.iclr2027.analysis.rebuild_a5_audit")
    _run("a6_current_endpoint_integrity", "evaluations.iclr2027.analysis.a6_endpoint_integrity")
    _run("table1_refresh", "evaluations.iclr2027.analysis.a5_results", "endpoints")
    _run("e1_complete_results", "evaluations.iclr2027.analysis.a6_e1_complete_results")
    _run("e2_figure3_refresh", "evaluations.iclr2027.analysis.a6_e2_results")
    _run("e3c_refresh", "evaluations.iclr2027.analysis.a6_e3c_results")
    _run("e3c_oracle_refresh", "evaluations.iclr2027.analysis.a6_e3c_oracle_results")
    _run("e5_core_refresh", "evaluations.iclr2027.analysis.a6_e5_results")

    e6_analysis = RESULTS / "native_v3" / "derived" / "E6_ANALYSIS.json"
    if not _is_pass(e6_analysis):
        raise RuntimeError("accepted E6 analysis is missing; this remaining-work runner must not rerun E6")
    _assert_e6_development()
    _assert_e6_formal()
    _record("e6_accepted_results_reused", "PASS")

    e4_final = RESULTS / "controlled" / "e4" / "A6_E4_FINAL_ACCEPTANCE.json"
    if not _is_pass(e4_final):
        gate = RESULTS / "controlled" / "e4" / "nested_per_stage" / "E4_NESTED_DEVELOPMENT_GATE.json"
        if not _is_pass(gate):
            _run("e4_nested_development", "evaluations.iclr2027.runners.a6_e4_nested_development", "run", allowed=(0, 2))
            _run("e4_nested_development_validate", "evaluations.iclr2027.runners.a6_e4_nested_development", "validate")
        _run("e4_nested_formal", "evaluations.iclr2027.runners.a6_e4_nested", "run", allowed=(0, 2))
        _run("e4_nested_analysis", "evaluations.iclr2027.analysis.a6_e4_nested_results")
        _run("e4_nested_promote", "evaluations.iclr2027.analysis.promote_e4_nested_results")
    else:
        _record("e4_nested_results_reused", "PASS")

    _run("e5_fine", "evaluations.iclr2027.runners.a6_e5_fine", "run", allowed=(0, 2))
    _run("e5_fine_analysis", "evaluations.iclr2027.analysis.a6_e5_fine_results")
    _run("e5_threshold_sensitivity", "evaluations.iclr2027.analysis.a6_e5_threshold_sensitivity")

    _run("a6_acceptance", "evaluations.iclr2027.analysis.a6_acceptance", "accept")
    _record("a6_experiments_complete", "PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
