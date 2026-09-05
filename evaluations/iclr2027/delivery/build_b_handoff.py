"""Index canonical A-to-B handoff files without duplicating their contents."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

from evaluations.iclr2027.runners.episode_io import (
    load_episode,
    resolve_cycle_file,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EVALUATION_ROOT = REPOSITORY_ROOT / "evaluations" / "iclr2027"
DEFAULT_OUTPUT_ROOT = EVALUATION_ROOT / "results" / "a2_acceptance"
FAILURE_TRAIN_ROOT = EVALUATION_ROOT / "datasets" / "failure_train"
FORBIDDEN_TOKENS = (
    "normal_calibration",
    "sealed",
    "main10_nominal",
    "main10_perturbed",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files(root: Path, patterns: Iterable[str]) -> list[Path]:
    result: list[Path] = []
    for pattern in patterns:
        result.extend(path for path in root.glob(pattern) if path.is_file())
    return sorted(set(result))


def interface_sources() -> list[Path]:
    sources = _files(EVALUATION_ROOT / "interfaces", ("*.py",))
    sources += [
        EVALUATION_ROOT / "configs" / "shared" / "feature_schema.json",
        EVALUATION_ROOT / "configs" / "shared" / "faults.json",
        EVALUATION_ROOT / "configs" / "shared" / "artifact_contract.json",
        EVALUATION_ROOT / "configs" / "shared" / "b_interface_protocol.json",
        EVALUATION_ROOT / "configs" / "shared" / "b_delivery_contract.json",
        EVALUATION_ROOT / "configs" / "shared" / "monitor_calibration.json",
    ]
    fixture_root = EVALUATION_ROOT / "tests" / "fixtures" / "development_examples"
    if fixture_root.exists():
        sources += _files(fixture_root, ("**/*",))
    return sorted(set(sources))


def failure_train_sources(result_root: Path = FAILURE_TRAIN_ROOT) -> list[Path]:
    manifest = EVALUATION_ROOT / "manifests" / "main10_failure_train.jsonl"
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 2000:
        raise ValueError(f"failure-train manifest has {len(rows)} rows, expected 2000")
    sources = [manifest]
    for row in rows:
        safe = str(row["episode_id"]).replace("/", "__")
        result_path = result_root / "episodes" / (safe + ".json")
        result = load_episode(result_path)
        if result.get("reason") == "infrastructure_error":
            raise RuntimeError(f"infrastructure-failed training episode: {row['episode_id']}")
        cycle_path = resolve_cycle_file(result_path, result)
        if not cycle_path.is_file():
            raise FileNotFoundError(cycle_path)
        if _sha256(cycle_path) != result["cycle_file_sha256"]:
            raise ValueError(f"cycle hash mismatch: {row['episode_id']}")
        sources.extend((result_path, cycle_path))
    return sorted(set(sources))


def build_handoff_index(kind: str, sources: Iterable[Path]) -> dict:
    entries = []
    for source in sorted(set(Path(path).resolve() for path in sources)):
        relative = source.relative_to(REPOSITORY_ROOT)
        if any(token in str(relative) for token in FORBIDDEN_TOKENS):
            raise ValueError(f"forbidden A-only file in B handoff: {relative}")
        entries.append(
            {
                "path": str(relative),
                "bytes": source.stat().st_size,
                "sha256": _sha256(source),
            }
        )
    return {
        "schema": "essay2608.iclr2027.canonical-handoff-index.v1",
        "kind": kind,
        "transfer_mode": "canonical_paths_no_persistent_copy",
        "repository_relative_paths": True,
        "files": entries,
        "total_files": len(entries),
        "total_bytes": sum(entry["bytes"] for entry in entries),
        "contains_normal_calibration": False,
        "contains_sealed_test": False,
    }


def write_handoff_index(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("interface", "failure_train"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--result-root", type=Path, default=FAILURE_TRAIN_ROOT)
    args = parser.parse_args(argv)

    if args.kind == "interface":
        sources = interface_sources()
        filename = "B_INTERFACE_HANDOFF.json"
    else:
        sources = failure_train_sources(args.result_root.resolve())
        filename = "B_FAILURE_TRAIN_HANDOFF.json"
    payload = build_handoff_index(args.kind, sources)
    output = args.output or (DEFAULT_OUTPUT_ROOT / filename)
    write_handoff_index(output, payload)
    print(
        json.dumps(
            {
                "kind": payload["kind"],
                "files": payload["total_files"],
                "bytes": payload["total_bytes"],
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
