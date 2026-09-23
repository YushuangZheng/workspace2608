"""Strictly admit the audited A6 ablation-only extension to frozen A5."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[3]
EXTENSION = (
    ROOT
    / "evaluations/iclr2027/results/a6_execution/A6_ABLATION_EXTENSION.json"
)
POST_E4_FREEZE = (
    ROOT
    / "evaluations/iclr2027/results/a6_execution/M5_POST_E4_FREEZE.json"
)
CURRENT_M5_FREEZE = (
    ROOT
    / "evaluations/iclr2027/results/a6_execution/M5_PENDING_DIRECT_IDENTITY_FREEZE.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def assert_frozen_a5_with_ablation_extension(
    plan_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify every A5 file, allowing only the signed additive extension."""

    plan = _read_json(plan_path)
    extension = _read_json(EXTENSION)
    if extension.get("status") != "PASS_FULL_SEMANTICS_PRESERVED":
        raise RuntimeError("A6 ablation extension is not accepted")
    if int(extension.get("validation", {}).get("failed", -1)) != 0:
        raise RuntimeError("A6 ablation extension regression validation failed")

    allowed = {
        str(row["path"]): row
        for row in extension.get("allowed_source_changes", [])
    }
    a5_entries = {
        str(row["path"]): row
        for row in plan["endpoint_code_identity"]["entries"]
    }
    if set(allowed).difference(a5_entries):
        raise RuntimeError("A6 extension references a file outside frozen A5")

    observed_changes = set()
    for path_text, entry in a5_entries.items():
        path = ROOT / path_text
        current = _sha256(path)
        if current == entry["sha256"]:
            continue
        amendment = allowed.get(path_text)
        if amendment is None:
            raise RuntimeError("frozen A5 endpoint changed: " + path_text)
        if amendment["a5_sha256"] != entry["sha256"]:
            raise RuntimeError("A6 extension has the wrong A5 origin: " + path_text)
        if amendment["a6_sha256"] != current:
            raise RuntimeError("A6 extension file drifted after validation: " + path_text)
        observed_changes.add(path_text)
    if observed_changes != set(allowed):
        raise RuntimeError("A6 extension source set does not match its signed record")

    from source.policy.tsf.ablation import TSFFeatureProfile

    full = TSFFeatureProfile.named("full")
    required: Mapping[str, Any] = extension["full_profile_required"]
    for field, expected in required.items():
        if getattr(full, field) != expected:
            raise RuntimeError("frozen Full profile semantics changed: " + field)
    return plan, extension


def assert_frozen_m5_post_e4() -> dict[str, Any]:
    """Verify the M5 identity accepted after the complete E4 rerun.

    The completed E4 episodes retain their original candidate-config hash.
    Later experiments use ``m5_full.json`` as a path-only alias to the same
    byte-identical model tree.  This gate verifies both aliases, every model
    artifact, and every algorithm source file recorded at the freeze point.
    """

    record = _read_json(POST_E4_FREEZE)
    if record.get("status") != "PASS":
        raise RuntimeError("post-E4 M5 freeze is not PASS")

    for field in ("candidate_config", "official_config_alias"):
        entry = record[field]
        path = ROOT / str(entry["path"])
        if not path.is_file() or _sha256(path) != entry["sha256"]:
            raise RuntimeError(f"post-E4 M5 {field} changed")

    model_root = (
        ROOT / "integrations/rlbench/models/iclr2027/tsf"
    )
    expected_models = {
        str(entry["path"]): str(entry["sha256"])
        for entry in record["model_tree_identity"]["entries"]
    }
    observed_models = {
        path.relative_to(model_root).as_posix(): _sha256(path)
        for path in model_root.glob("*/*")
        if path.is_file()
    }
    if observed_models != expected_models:
        raise RuntimeError("post-E4 M5 model tree changed")

    source_entries = record["algorithm_source_identity"]["entries"]
    for entry in source_entries:
        path = ROOT / str(entry["path"])
        if not path.is_file() or _sha256(path) != entry["sha256"]:
            raise RuntimeError(
                "post-E4 M5 algorithm source changed: " + str(entry["path"])
            )
    return record


def assert_frozen_m5_current() -> dict[str, Any]:
    """Verify the latest accepted M5 source, models, config, and evidence."""

    record = _read_json(CURRENT_M5_FREEZE)
    if record.get("status") != "PASS":
        raise RuntimeError("current M5 freeze is not PASS")
    acceptance = record["development_acceptance"]
    acceptance_path = ROOT / str(acceptance["path"])
    if (
        not acceptance_path.is_file()
        or _sha256(acceptance_path) != acceptance["sha256"]
        or _read_json(acceptance_path).get("status") != "PASS"
    ):
        raise RuntimeError("current M5 development acceptance changed")
    config = record["official_config"]
    config_path = ROOT / str(config["path"])
    if not config_path.is_file() or _sha256(config_path) != config["sha256"]:
        raise RuntimeError("current M5 method config changed")

    model_root = ROOT / "integrations/rlbench/models/iclr2027/tsf"
    expected_models = {
        str(entry["path"]): str(entry["sha256"])
        for entry in record["model_tree_identity"]["entries"]
    }
    observed_models = {
        path.relative_to(model_root).as_posix(): _sha256(path)
        for path in model_root.glob("*/*")
        if path.is_file()
    }
    if observed_models != expected_models:
        raise RuntimeError("current M5 model tree changed")

    for entry in record["algorithm_source_identity"]["entries"]:
        path = ROOT / str(entry["path"])
        if not path.is_file() or _sha256(path) != entry["sha256"]:
            raise RuntimeError("current M5 source changed: " + str(entry["path"]))
    return record


__all__ = [
    "assert_frozen_a5_with_ablation_extension",
    "assert_frozen_m5_current",
    "assert_frozen_m5_post_e4",
]
