"""Verify one scoped server-B delivery after it is copied to canonical paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONTRACT = (
    REPOSITORY_ROOT
    / "evaluations/iclr2027/configs/shared/b_delivery_contract.json"
)
DEFAULT_ARTIFACT_CONTRACT = (
    REPOSITORY_ROOT
    / "evaluations/iclr2027/configs/shared/artifact_contract.json"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("delivery path must be a non-empty string")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ValueError(f"delivery path is not canonical and relative: {value!r}")
    return value


def _allowed(path: str, contract: Mapping[str, Any]) -> bool:
    if any(path.startswith(str(prefix)) for prefix in contract["forbidden_prefixes"]):
        return False
    if path in {str(value) for value in contract["allowed_method_configs"]}:
        return True
    prefixes = tuple(
        str(value)
        for key in ("allowed_code_prefixes", "allowed_artifact_prefixes")
        for value in contract[key]
    )
    return path.startswith(prefixes)


def _verify_checkpoint_manifest(
    path: Path,
    artifact_contract: Mapping[str, Any],
    repository_root: Path,
) -> dict[str, Any]:
    value = _json(path)
    required = artifact_contract["required_checkpoint_manifest"]
    if value.get("schema") != required["schema"]:
        raise ValueError(f"unsupported checkpoint manifest: {path}")
    for field in required:
        if field not in value:
            raise ValueError(f"checkpoint manifest missing {field}: {path}")
    if value["feature_schema"] != artifact_contract["feature_schema"]:
        raise ValueError(f"checkpoint feature schema mismatch: {path}")
    for field in ("checkpoint_relative_path", "config_relative_path"):
        _safe_relative(value[field])
    for field in ("checkpoint_sha256", "config_sha256"):
        if not isinstance(value[field], str) or not SHA256_RE.fullmatch(value[field]):
            raise ValueError(f"invalid {field}: {path}")
    if (
        isinstance(value["training_budget"], bool)
        or not isinstance(value["training_budget"], int)
        or value["training_budget"] < 0
    ):
        raise ValueError(f"invalid training_budget: {path}")
    if isinstance(value["training_seed"], bool) or not isinstance(
        value["training_seed"], int
    ):
        raise ValueError(f"invalid training_seed: {path}")
    for relative_field, hash_field in (
        ("checkpoint_relative_path", "checkpoint_sha256"),
        ("config_relative_path", "config_sha256"),
    ):
        target = repository_root / value[relative_field]
        if not target.is_file() or _sha256(target) != value[hash_field]:
            raise ValueError(f"checkpoint identity mismatch for {target}")
    return value


def verify_b_delivery(
    manifest_path: Path,
    *,
    contract_path: Path = DEFAULT_CONTRACT,
    artifact_contract_path: Path = DEFAULT_ARTIFACT_CONTRACT,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    """Authenticate paths and bytes without copying or overwriting anything."""

    manifest = _json(manifest_path)
    contract = _json(contract_path)
    artifact_contract = _json(artifact_contract_path)
    if manifest.get("schema") != contract["delivery_manifest_schema"]:
        raise ValueError("unsupported B-to-A delivery manifest")
    for field in contract["required_manifest_fields"]:
        if field not in manifest:
            raise ValueError(f"delivery manifest missing {field}")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("delivery manifest must list at least one file")
    seen: set[str] = set()
    checked = []
    for entry in files:
        if not isinstance(entry, Mapping):
            raise TypeError("delivery file entry must be an object")
        if any(field not in entry for field in contract["required_file_fields"]):
            raise ValueError("delivery file entry is incomplete")
        relative = _safe_relative(entry["path"])
        if relative in seen:
            raise ValueError(f"duplicate delivery path: {relative}")
        seen.add(relative)
        if not _allowed(relative, contract):
            raise ValueError(f"delivery path is outside B ownership: {relative}")
        if any(token in relative for token in contract["forbidden_path_tokens"]):
            raise ValueError(f"delivery path exposes A-only data: {relative}")
        expected_size = entry["bytes"]
        expected_hash = entry["sha256"]
        if (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size < 0
            or not isinstance(expected_hash, str)
            or not SHA256_RE.fullmatch(expected_hash)
        ):
            raise ValueError(f"invalid file identity: {relative}")
        path = repository_root / relative
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"delivery file missing or is a symlink: {relative}")
        if path.stat().st_size != expected_size or _sha256(path) != expected_hash:
            raise ValueError(f"delivery file identity mismatch: {relative}")
        checked.append(relative)
    checkpoint_manifests = manifest.get("checkpoint_manifests", [])
    if not isinstance(checkpoint_manifests, list):
        raise TypeError("checkpoint_manifests must be a list")
    checkpoints = []
    for relative in checkpoint_manifests:
        relative = _safe_relative(relative)
        if relative not in seen:
            raise ValueError("checkpoint manifest must also appear in files")
        checkpoint = _verify_checkpoint_manifest(
            repository_root / relative,
            artifact_contract,
            repository_root,
        )
        for key in ("checkpoint_relative_path", "config_relative_path"):
            if checkpoint[key] not in seen:
                raise ValueError(f"{key} must also appear in delivery files")
        checkpoints.append(checkpoint)
    return {
        "schema": "essay2608.iclr2027.b-delivery-verification.v1",
        "status": "PASS",
        "delivery_manifest": str(manifest_path.resolve()),
        "delivery_manifest_sha256": _sha256(manifest_path),
        "files_verified": len(checked),
        "checkpoint_manifests_verified": len(checkpoints),
        "contains_a_owned_paths": False,
        "copy_or_overwrite_performed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument(
        "--artifact-contract", type=Path, default=DEFAULT_ARTIFACT_CONTRACT
    )
    args = parser.parse_args(argv)
    result = verify_b_delivery(
        args.manifest,
        contract_path=args.contract,
        artifact_contract_path=args.artifact_contract,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
