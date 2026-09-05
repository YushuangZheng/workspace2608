"""Crash-safe, one-episode-per-job result storage."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from evaluations.iclr2027.interfaces.feature_schema import (
    EPISODE_SCHEMA,
    validate_feature_record,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class EpisodeWriter:
    """Write separated causal features and audit labels for one episode.

    A temporary gzip stream is made visible only when :meth:`finalize` commits
    the terminal episode JSON.  A crash therefore leaves a detectable ``.tmp``
    artifact and can never masquerade as a completed episode.
    """

    def __init__(self, output_root: Path, episode_id: str) -> None:
        safe = episode_id.replace("/", "__")
        self.output_root = Path(output_root)
        self.cycles_path = self.output_root / "cycles" / (safe + ".jsonl.gz")
        self.episode_path = self.output_root / "episodes" / (safe + ".json")
        self.temporary_cycles = self.cycles_path.with_name(
            self.cycles_path.name + ".tmp"
        )
        self.temporary_cycles.parent.mkdir(parents=True, exist_ok=True)
        self._stream = gzip.open(self.temporary_cycles, "wt", encoding="utf-8")
        self._count = 0

    def write_cycle(
        self,
        feature: Mapping[str, Any],
        audit: Mapping[str, Any],
        *,
        execution: Mapping[str, Any],
    ) -> None:
        validate_feature_record(feature)
        if int(feature["cycle"]) != int(audit["cycle"]):
            raise ValueError("feature and audit timestamps differ")
        row = {
            "cycle": int(feature["cycle"]),
            "feature": dict(feature),
            "audit": dict(audit),
            "execution": dict(execution),
        }
        self._stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        self._count += 1

    def finalize(self, summary: Mapping[str, Any]) -> Path:
        if summary.get("schema") != EPISODE_SCHEMA:
            raise ValueError("episode summary uses the wrong schema")
        self._stream.close()
        os.replace(str(self.temporary_cycles), str(self.cycles_path))
        payload = dict(summary)
        payload["cycle_records"] = self._count
        payload["cycle_file"] = os.path.relpath(
            self.cycles_path.resolve(), self.episode_path.parent.resolve()
        )
        payload["cycle_file_location"] = "episode_relative"
        payload["cycle_file_sha256"] = _sha256(self.cycles_path)
        _atomic_json(self.episode_path, payload)
        return self.episode_path

    def abort(self) -> None:
        if not self._stream.closed:
            self._stream.close()


def load_cycles(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_episode(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema") != EPISODE_SCHEMA:
        raise ValueError("invalid episode result schema")
    return value


def resolve_cycle_file(
    episode_path: Path, episode: Mapping[str, Any]
) -> Path:
    """Resolve portable and legacy cycle references from an episode summary."""

    reference = Path(str(episode["cycle_file"]))
    location = episode.get("cycle_file_location")
    if reference.is_absolute():
        return reference
    if location == "episode_relative":
        return (Path(episode_path).parent / reference).resolve()
    if location == "repository_relative":
        return (REPOSITORY_ROOT / reference).resolve()
    repository_candidate = (REPOSITORY_ROOT / reference).resolve()
    if repository_candidate.exists():
        return repository_candidate
    return (Path(episode_path).parent / reference).resolve()


def completed_episode_ids(output_root: Path) -> set[str]:
    result = set()
    for path in (Path(output_root) / "episodes").glob("*.json"):
        value = load_episode(path)
        result.add(str(value["episode_id"]))
    return result


__all__ = [
    "EpisodeWriter",
    "completed_episode_ids",
    "load_cycles",
    "load_episode",
    "resolve_cycle_file",
]
