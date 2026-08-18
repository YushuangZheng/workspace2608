"""Build the authenticated V3 RLBench Table II comparison."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from .direct_policy import TRAINING_MANIFEST_SCHEMA_V3, V3_ADAPTER_PROTOCOL
from .eval_set import GLOBAL_EVAL_SEED_START
from .paper_comparison import (
    EXPECTED_LOCAL_CONFIG,
    EXPECTED_MODEL_SCHEMA_VERSION,
    EXPECTED_SELECTION_SEMANTICS_ID,
    EXPECTED_TAPAS_COMMIT,
    LocalRun,
    _valid_v3_static_protocol,
    expected_evaluation_protocol_id,
)
from .v3_protocol import resolve_authenticated_v3_trigger

INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = INTEGRATION_ROOT / "results" / "v3" / "table_ii"
TASKS = (
    ("bimanual_put_bottle_in_fridge", "StoreBottle", 0.82),
    ("bimanual_handover_item", "HandOver", 0.97),
    ("bimanual_sweep_to_dustpan", "SweepDust", 1.00),
    ("bimanual_lift_tray", "LiftTray", 1.00),
)


def _validate_v3_identity(payload: dict[str, object], task: str, path: Path) -> None:
    identity = payload.get("model_identity")
    if not isinstance(identity, dict):
        raise ValueError(f"result model identity is missing: {path}")
    valid = (
        identity.get("manifest_authenticated") is True
        and identity.get("training_config") == EXPECTED_LOCAL_CONFIG
        and identity.get("model_schema_version") == EXPECTED_MODEL_SCHEMA_VERSION
        and identity.get("selection_semantics_id")
        == EXPECTED_SELECTION_SEMANTICS_ID
        and identity.get("tapas_reference_commit") == EXPECTED_TAPAS_COMMIT
        and identity.get("training_manifest_schema")
        == TRAINING_MANIFEST_SCHEMA_V3
        and identity.get("training_adapter_protocol") == V3_ADAPTER_PROTOCOL
        and bool(identity.get("checkpoint_trigger_audit_fingerprint"))
        and bool(identity.get("left_fingerprint"))
        and bool(identity.get("right_fingerprint"))
        and payload.get("evaluation_protocol_id")
        == expected_evaluation_protocol_id(task, release="v3")
    )
    if valid:
        try:
            resolve_authenticated_v3_trigger(identity, task=task)
        except (KeyError, RuntimeError, TypeError, ValueError):
            valid = False
    if not valid:
        raise ValueError(f"result v3 protocol/model identity mismatch: {path}")


def result_path(
    results_dir: Path,
    task: str,
    *,
    seed: int,
    episodes: int,
    horizon: int,
) -> Path:
    return results_dir / f"{task}_static_seed{seed}_n{episodes}_h{horizon}.json"


def load_rows(
    results_dir: Path,
    *,
    seed: int,
    episodes: int,
    horizon: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for task, label, paper_rate in TASKS:
        path = result_path(
            results_dir,
            task,
            seed=seed,
            episodes=episodes,
            horizon=horizon,
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected_identity = {
            "task": task,
            "scenario": "static",
            "seed": seed,
            "episodes": episodes,
            "horizon": horizon,
        }
        actual_identity = {key: payload.get(key) for key in expected_identity}
        if actual_identity != expected_identity:
            raise ValueError(f"result identity mismatch: {path}")
        _validate_v3_identity(payload, task, path)
        episode_results = payload.get("results")
        if not isinstance(episode_results, list) or len(episode_results) != episodes:
            raise ValueError(f"result episode count mismatch: {path}")
        raw_rate = payload.get("success_rate")
        raw_successes = payload.get("successes")
        if (
            not isinstance(raw_rate, (int, float))
            or isinstance(raw_rate, bool)
            or not math.isfinite(float(raw_rate))
            or not 0.0 <= float(raw_rate) <= 1.0
            or not isinstance(raw_successes, int)
            or isinstance(raw_successes, bool)
            or not 0 <= raw_successes <= episodes
        ):
            raise ValueError(f"result success totals are invalid: {path}")
        local_rate = float(raw_rate)
        successes = raw_successes
        if successes != sum(bool(item.get("success")) for item in episode_results):
            raise ValueError(f"result success count mismatch: {path}")
        if abs(local_rate - successes / float(episodes)) > 1.0e-9:
            raise ValueError(f"result success rate mismatch: {path}")
        run = LocalRun(
            path=path,
            task=task,
            scenario="static",
            seed=seed,
            episodes=episodes,
            horizon=horizon,
            variation=int(payload.get("variation", 0)),
            successes=successes,
            success_rate=local_rate,
            payload=payload,
        )
        if not _valid_v3_static_protocol(run):
            raise ValueError(f"result v3 static protocol evidence mismatch: {path}")
        termination_reasons = Counter(str(item.get("reason")) for item in episode_results)
        rows.append(
            {
                "task": task,
                "label": label,
                "paper_rate": paper_rate,
                "local_rate": local_rate,
                "difference": local_rate - paper_rate,
                "episodes": episodes,
                "successes": successes,
                "invalid_actions": sum(
                    int(item.get("invalid_actions", 0)) for item in episode_results
                ),
                "termination_reasons": dict(sorted(termination_reasons.items())),
                "path": path,
            }
        )
    return rows


def markdown(rows: Sequence[dict[str, object]], *, seed: int) -> str:
    local_average = sum(float(row["local_rate"]) for row in rows) / len(rows)
    paper_average = sum(float(row["paper_rate"]) for row in rows) / len(rows)
    lines = [
        "# DynaMAC RLBench Table II Comparison",
        "",
        f"Local static evaluation seed: `{seed}`. Paper values are from "
        "Table II of arXiv:2607.22119v1. Every local row is authenticated "
        "against the frozen V3 model, evaluator, trigger-evidence, committed-clock, "
        "and final-settling protocols.",
        "",
        "| Task | Paper DynaMAC | Local implementation | Difference |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['label']} | {float(row['paper_rate']):.2f} | "
            f"{float(row['local_rate']):.3f} | {float(row['difference']):+.3f} |"
        )
    lines.append(
        f"| **Average** | **{paper_average:.2f}** | **{local_average:.3f}** | "
        f"**{local_average - paper_average:+.3f}** |"
    )
    lines.extend(
        [
            "",
            "## Execution diagnostics",
            "",
            "| Task | Successful episodes | Invalid actions | Termination reasons |",
            "|---|---:|---:|---|",
        ]
    )
    for row in rows:
        reasons = ", ".join(
            f"{reason}={count}"
            for reason, count in dict(row["termination_reasons"]).items()
        )
        lines.append(
            f"| {row['label']} | {int(row['successes'])}/{int(row['episodes'])} | "
            f"{int(row['invalid_actions'])} | {reasons} |"
        )
    lines.extend(
        [
            "",
            "The local demonstrations are not the paper's original cohort. Remaining "
            "configuration ambiguities are documented in "
            "`integrations/rlbench/IMPLEMENTATION_NOTES.md`.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--seed", type=int, default=GLOBAL_EVAL_SEED_START)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "table_ii_comparison.md",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = load_rows(
        args.results_dir,
        seed=args.seed,
        episodes=args.episodes,
        horizon=args.horizon,
    )
    text = markdown(rows, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
