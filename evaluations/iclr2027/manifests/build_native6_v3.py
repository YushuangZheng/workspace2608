"""Materialize isolated event-grounded E6 manifests without touching E1/E6 v2."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from evaluations.iclr2027.manifests.native6_v3 import (
    build_development_rows,
    build_formal_perturbed_rows,
    validate_rows,
)
from integrations.rlbench.rlbench_dynamac.core.paths import REPOSITORY_ROOT


ROOT = REPOSITORY_ROOT / "evaluations" / "iclr2027" / "manifests"
SOURCE_NOMINAL = ROOT / "main10_nominal.jsonl"
SOURCE_NATIVE6 = ROOT / "native6_perturbed.jsonl"
DEVELOPMENT_PATH = ROOT / "native6_v3_development.jsonl"
FORMAL_PATH = ROOT / "native6_v3_perturbed.jsonl"
INDEX_PATH = ROOT / "NATIVE6_V3_MANIFEST_INDEX.json"
CONFIG_ROOT = REPOSITORY_ROOT / "evaluations" / "iclr2027" / "configs" / "shared"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _atomic(
        path,
        "".join(
            json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
    )


def materialize() -> dict[str, Any]:
    source_hashes = {
        str(SOURCE_NOMINAL.relative_to(REPOSITORY_ROOT)): _sha(SOURCE_NOMINAL),
        str(SOURCE_NATIVE6.relative_to(REPOSITORY_ROOT)): _sha(SOURCE_NATIVE6),
    }
    development = build_development_rows(_load_jsonl(SOURCE_NOMINAL))
    formal = build_formal_perturbed_rows(_load_jsonl(SOURCE_NATIVE6))
    validation = validate_rows(development, formal)
    _jsonl(DEVELOPMENT_PATH, development)
    _jsonl(FORMAL_PATH, formal)
    if source_hashes != {
        str(SOURCE_NOMINAL.relative_to(REPOSITORY_ROOT)): _sha(SOURCE_NOMINAL),
        str(SOURCE_NATIVE6.relative_to(REPOSITORY_ROOT)): _sha(SOURCE_NATIVE6),
    }:
        raise RuntimeError("source manifest changed during Native-6 v3 materialization")
    files = {
        str(DEVELOPMENT_PATH.relative_to(REPOSITORY_ROOT)): {
            "rows": len(development),
            "sha256": _sha(DEVELOPMENT_PATH),
        },
        str(FORMAL_PATH.relative_to(REPOSITORY_ROOT)): {
            "rows": len(formal),
            "sha256": _sha(FORMAL_PATH),
        },
    }
    config_paths = (
        CONFIG_ROOT / "native6_physical_protocol_v3.json",
        CONFIG_ROOT / "native6_contract_v3.json",
        CONFIG_ROOT / "native6_result_schema_v3.json",
        CONFIG_ROOT / "native6_b_return_contract_v3.json",
    )
    index = {
        "schema": "essay2608.iclr2027.native6-v3-manifest-index.v1",
        "materialized_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifests": source_hashes,
        "source_results_read": False,
        "legacy_results_reused": False,
        "legacy_trigger_stage_used": False,
        "manifests": files,
        "configs": {
            str(path.relative_to(REPOSITORY_ROOT)): _sha(path) for path in config_paths
        },
        "validation": validation,
    }
    _atomic(
        INDEX_PATH,
        json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return index


if __name__ == "__main__":
    print(json.dumps(materialize(), indent=2, sort_keys=True))
