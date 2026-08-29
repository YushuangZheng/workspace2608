"""Validate and summarize completed preregistered Stage-six result cells."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
from scipy.stats import binomtest

from integrations.rlbench.rlbench_dynamac.core.records import atomic_json

from . import launch
from .run_cell import PROTOCOL, load_protocol

DEFAULT_OUTPUT = Path(__file__).with_name("results") / "v1"
CONTRASTS = (
    ("progress_only", "dynamac_v4"),
    ("progress_dynamic_roles", "progress_only"),
    ("full", "progress_dynamic_roles"),
    ("full", "dynamac_v4"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wilson(
    successes: int, total: int, confidence: float = 0.95
) -> tuple[float, float]:
    if total < 1 or not 0 <= successes <= total:
        raise ValueError("invalid binomial count")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return center - radius, center + radius


def _holm(p_values: Sequence[float]) -> list[float]:
    count = len(p_values)
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [1.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * float(p_values[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty formal summary: {path}")
    fields = []
    for row in rows:
        for name in row:
            if name not in fields:
                fields.append(name)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _load_cell(cell: launch.FormalCell) -> tuple[dict[str, Any], dict[int, bool]]:
    launch._validate_result(cell)
    payload = json.loads(cell.result.read_text(encoding="utf-8"))
    metadata = payload["stage6_formal_evaluation"]
    indices = metadata["episode_indices"]
    rows = payload["results"]
    outcomes = {}
    for expected, row in zip(indices, rows):
        episode = int(row.get("episode", expected))
        if episode != int(expected) or episode in outcomes:
            raise RuntimeError(f"formal episode identity is invalid: {cell.result}")
        if not isinstance(row.get("success"), bool):
            raise RuntimeError(f"formal episode success is not boolean: {cell.result}")
        outcomes[episode] = row["success"]
    triggered = None
    if cell.fault is not None:
        triggered = sum(row["physical_fault"].get("triggered") is True for row in rows)
    return payload, outcomes


def _cell_rows(
    cells: Sequence[launch.FormalCell],
) -> tuple[
    list[dict[str, Any]], dict[tuple[str, Optional[str], str, str], dict[int, bool]]
]:
    rows = []
    outcomes = {}
    commits = set()
    for cell in cells:
        payload, values = _load_cell(cell)
        metadata = payload["stage6_formal_evaluation"]
        successes = sum(values.values())
        low, high = _wilson(successes, len(values))
        triggered = metadata["episodes_fault_triggered"]
        triggered_successes: Any = ""
        conditional_recovery_rate: Any = ""
        if cell.fault is not None:
            triggered_rows = [
                row
                for row in payload["results"]
                if row["physical_fault"].get("triggered") is True
            ]
            triggered_successes = sum(row["success"] for row in triggered_rows)
            conditional_recovery_rate = (
                "" if not triggered_rows else triggered_successes / len(triggered_rows)
            )
        commits.add(metadata["git_commit"])
        rows.append(
            {
                "experiment": cell.experiment,
                "fault": cell.fault or "",
                "task": cell.task,
                "method": cell.method,
                "episodes": len(values),
                "successes": successes,
                "success_rate": successes / len(values),
                "wilson95_low": low,
                "wilson95_high": high,
                "fault_triggered": "" if triggered is None else triggered,
                "fault_trigger_rate": (
                    "" if triggered is None else triggered / len(values)
                ),
                "triggered_successes": triggered_successes,
                "conditional_recovery_rate": conditional_recovery_rate,
                "git_commit": metadata["git_commit"],
                "result_path": cell.result.as_posix(),
                "result_sha256": _sha256(cell.result),
            }
        )
        outcomes[(cell.experiment, cell.fault, cell.task, cell.method)] = values
    if len(commits) != 1:
        raise RuntimeError("formal cells were produced by different Git commits")
    return rows, outcomes


def _comparisons(
    outcomes: Mapping[tuple[str, Optional[str], str, str], Mapping[int, bool]],
) -> list[dict[str, Any]]:
    groups = sorted({(key[0], key[1], key[2]) for key in outcomes})
    rows = []
    for experiment, fault, task in groups:
        for proposed, reference in CONTRASTS:
            a = outcomes[(experiment, fault, task, proposed)]
            b = outcomes[(experiment, fault, task, reference)]
            if set(a) != set(b):
                raise RuntimeError("paired formal cells have different episode indices")
            proposed_only = sum(a[index] and not b[index] for index in a)
            reference_only = sum(b[index] and not a[index] for index in a)
            discordant = proposed_only + reference_only
            p_value = (
                1.0
                if discordant == 0
                else float(
                    binomtest(
                        proposed_only, discordant, p=0.5, alternative="two-sided"
                    ).pvalue
                )
            )
            rows.append(
                {
                    "experiment": experiment,
                    "fault": fault or "",
                    "task": task,
                    "proposed": proposed,
                    "reference": reference,
                    "episodes": len(a),
                    "proposed_rate": sum(a.values()) / len(a),
                    "reference_rate": sum(b.values()) / len(b),
                    "rate_difference": (sum(a.values()) - sum(b.values())) / len(a),
                    "proposed_only_success": proposed_only,
                    "reference_only_success": reference_only,
                    "mcnemar_exact_p": p_value,
                }
            )
    adjusted = _holm([row["mcnemar_exact_p"] for row in rows])
    for row, value in zip(rows, adjusted):
        row["holm_adjusted_p"] = value
    return rows


def _macro_bootstrap(
    outcomes: Mapping[tuple[str, Optional[str], str, str], Mapping[int, bool]],
    *,
    draws: int = 10_000,
    seed: int = 2_608_000_000,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    experiment_faults = sorted({(key[0], key[1]) for key in outcomes})
    rows = []
    for experiment, fault in experiment_faults:
        tasks = sorted(
            {key[2] for key in outcomes if key[0] == experiment and key[1] == fault}
        )
        for proposed, reference in CONTRASTS:
            task_differences = []
            task_pairs = []
            for task in tasks:
                a = outcomes[(experiment, fault, task, proposed)]
                b = outcomes[(experiment, fault, task, reference)]
                indices = np.asarray(sorted(a), dtype=np.int64)
                av = np.asarray([a[int(index)] for index in indices], dtype=np.float64)
                bv = np.asarray([b[int(index)] for index in indices], dtype=np.float64)
                task_differences.append(float(np.mean(av - bv)))
                task_pairs.append((av, bv))
            samples = np.empty(draws, dtype=np.float64)
            for draw in range(draws):
                selected_tasks = rng.integers(0, len(tasks), size=len(tasks))
                differences = []
                for task_index in selected_tasks:
                    av, bv = task_pairs[int(task_index)]
                    selected_episodes = rng.integers(0, len(av), size=len(av))
                    differences.append(
                        float(np.mean(av[selected_episodes] - bv[selected_episodes]))
                    )
                samples[draw] = float(np.mean(differences))
            rows.append(
                {
                    "experiment": experiment,
                    "fault": fault or "",
                    "proposed": proposed,
                    "reference": reference,
                    "tasks": len(tasks),
                    "macro_rate_difference": float(np.mean(task_differences)),
                    "bootstrap95_low": float(np.quantile(samples, 0.025)),
                    "bootstrap95_high": float(np.quantile(samples, 0.975)),
                    "bootstrap_draws": draws,
                }
            )
    return rows


def _markdown(
    cell_rows: Sequence[Mapping[str, Any]],
    macro_rows: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# 阶段六正式评测结果",
        "",
        "本文件由冻结协议结果自动生成。所有成功率均以预注册 episode 为分母。",
        "",
        "## 单元结果",
        "",
        "| 实验 | 故障 | 任务 | 方法 | 成功 | 成功率 | 95% CI | 触发 |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in cell_rows:
        triggered = row["fault_triggered"]
        rendered = dict(row)
        rendered["fault"] = rendered["fault"] or "-"
        rendered["triggered"] = "-" if triggered == "" else triggered
        lines.append(
            "| {experiment} | {fault} | {task} | {method} | {successes}/{episodes} "
            "| {success_rate:.3f} | [{wilson95_low:.3f}, {wilson95_high:.3f}] | {triggered} |".format(
                **rendered
            )
        )
    lines.extend(
        (
            "",
            "## 跨任务宏平均差值",
            "",
            "| 实验 | 故障 | 方法 - 参照 | 差值 | 成对 bootstrap 95% CI |",
            "|---|---|---|---:|---:|",
        )
    )
    for row in macro_rows:
        rendered = dict(row)
        rendered["fault"] = rendered["fault"] or "-"
        lines.append(
            "| {experiment} | {fault} | {proposed} - {reference} | "
            "{macro_rate_difference:.3f} | [{bootstrap95_low:.3f}, {bootstrap95_high:.3f}] |".format(
                **rendered
            )
        )
    return "\n".join(lines) + "\n"


def summarize(section: str, output: Path) -> dict[str, Any]:
    protocol = load_protocol(PROTOCOL)
    cells = launch.build_cells(protocol, section)
    missing = [cell.cell_id for cell in cells if not cell.result.exists()]
    if missing:
        raise RuntimeError(
            f"formal summary requires every {section} cell; missing {len(missing)}"
        )
    cell_rows, outcomes = _cell_rows(cells)
    comparisons = _comparisons(outcomes)
    macro = _macro_bootstrap(outcomes)
    noninferiority_margin = float(
        protocol["statistics"]["normal_noninferiority_margin"]
    )
    noninferiority = None
    if section in {"normal", "all"}:
        row = next(
            value
            for value in macro
            if value["experiment"] == "normal"
            and value["fault"] == ""
            and value["proposed"] == "full"
            and value["reference"] == "dynamac_v4"
        )
        noninferiority = {
            "proposed": "full",
            "reference": "dynamac_v4",
            "margin": noninferiority_margin,
            "bootstrap95_low": row["bootstrap95_low"],
            "passed": row["bootstrap95_low"] > -noninferiority_margin,
        }
    output.mkdir(parents=True, exist_ok=False)
    _write_csv(output / "cell_summary.csv", cell_rows)
    _write_csv(output / "paired_comparisons.csv", comparisons)
    _write_csv(output / "macro_bootstrap.csv", macro)
    payload = {
        "schema": "essay2608.phase6_formal_summary.v1",
        "section": section,
        "protocol_path": PROTOCOL.as_posix(),
        "protocol_sha256": _sha256(PROTOCOL),
        "cell_count": len(cells),
        "episode_count": sum(cell.episodes for cell in cells),
        "cell_results": cell_rows,
        "paired_comparisons": comparisons,
        "macro_bootstrap": macro,
        "normal_noninferiority": noninferiority,
    }
    atomic_json(output / "summary.json", payload)
    (output / "RESULTS.md").write_text(_markdown(cell_rows, macro), encoding="utf-8")
    artifacts = sorted(path for path in output.iterdir() if path.is_file())
    (output / "SHA256SUMS").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in artifacts),
        encoding="utf-8",
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", choices=("normal", "fault", "all"), default="all")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    payload = summarize(args.section, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
