"""Build/verify local, manifest-scoped component handoffs. Never transfers files.

Uses A's imported delivery contract without modifying it. The A4 draft clearly
distinguishes component portability from A's integration and formal scoring.
Native drafts describe development, not formal E6.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from ..runtime import sha256
from .acceptance import BASE, ROOT

COMMON_BASE = "e7bc53fa847a5e20a7b55308b55612b823210dc9"
TRAIN = BASE / "artifacts/training/m4"
REVIEW = TRAIN / "handoff"
CONTRACT = BASE / "configs/shared/b_delivery_contract.json"
PORTABILITY = REVIEW / "PORTABLE_DELIVERY_VALIDATION.json"


def payload_identity(entries: list[dict]) -> str:
    """Stable byte identity excluding this check's own report to avoid a hash cycle."""
    excluded = str(PORTABILITY.relative_to(ROOT))
    payload = sorted((e for e in entries if e["path"] != excluded), key=lambda e: e["path"])
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def allowed(path: str, contract: dict) -> bool:
    parsed = PurePosixPath(path)
    if (
        not path
        or parsed.is_absolute()
        or ".." in parsed.parts
        or "\\" in path
        or str(parsed) != path
    ):
        return False
    if any(path.startswith(p) for p in contract["forbidden_prefixes"]) or any(
        token in path for token in contract["forbidden_path_tokens"]
    ):
        return False
    return path in contract["allowed_method_configs"] or any(
        path.startswith(prefix)
        for prefix in contract["allowed_code_prefixes"] + contract["allowed_artifact_prefixes"]
    )


def file_entry(path: Path, contract: dict) -> dict:
    relative = str(path.relative_to(ROOT))
    if not allowed(relative, contract):
        raise ValueError(f"outside B delivery whitelist: {relative}")
    if not path.is_file() or path.is_symlink() or ROOT not in path.resolve().parents:
        raise ValueError(f"noncanonical/missing payload: {path}")
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)}


def method_files(name: str, native: bool = False) -> list[Path]:
    root = BASE / ("native_systems" if native else "methods") / name
    files = []
    for path in root.rglob("*"):
        if any(
            part.startswith(".") or part == "__pycache__" for part in path.relative_to(root).parts
        ):
            continue
        if not path.is_file():
            continue
        if path.suffix in (".py", ".md", ".yaml", ".yml", ".toml", ".txt") or path.name in (
            "OFFICIAL_SOURCES.json",
            "square_dataset.json",
        ):
            files.append(path)
    return files


def immutable_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        if sha256(source) != sha256(destination):
            raise ValueError(f"existing provenance copy differs: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def resolve_manifest_path(path: Path) -> Path:
    path = Path(path)
    if ".." in path.parts:
        raise ValueError("manifest path must not traverse parents")
    if not path.is_absolute():
        path = ROOT / path
    if path.is_symlink() or ROOT not in path.resolve().parents:
        raise ValueError("manifest path must stay inside the repository")
    return path.resolve()


def verify_manifest(path: Path) -> dict:
    path = resolve_manifest_path(path)
    contract = json.loads(CONTRACT.read_text())
    d = json.loads(path.read_text())
    if d.get("schema") != contract["delivery_manifest_schema"]:
        raise ValueError("wrong delivery manifest schema")
    for key in contract["required_manifest_fields"]:
        if key not in d:
            raise ValueError(f"missing manifest field: {key}")
    entries = {}
    for entry in d["files"]:
        rel = entry["path"]
        if rel in entries or not allowed(rel, contract):
            raise ValueError(f"duplicate/forbidden path: {rel}")
        actual = file_entry(ROOT / rel, contract)
        if entry != actual:
            raise ValueError(f"file size/hash mismatch: {rel}")
        entries[rel] = entry
    required = json.loads((BASE / "configs/shared/artifact_contract.json").read_text())[
        "required_checkpoint_manifest"
    ]
    for manifest in d["checkpoint_manifests"]:
        if set(required) - set(manifest):
            raise ValueError("incomplete checkpoint manifest")
        if manifest["schema"] != required["schema"]:
            raise ValueError("wrong checkpoint schema")
        for kind in ("checkpoint", "config"):
            rel = manifest[kind + "_relative_path"]
            if rel not in entries or entries[rel]["sha256"] != manifest[kind + "_sha256"]:
                raise ValueError(f"unbound {kind}: {rel}")
    return {
        "manifest": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "files": len(entries),
        "bytes": sum(e["bytes"] for e in entries.values()),
        "checkpoints": len(d["checkpoint_manifests"]),
        "status": "pass",
        "delivery_status": d["delivery_status"],
        "transferred": False,
    }


def build(
    name: str,
    paths: list[Path],
    checkpoints: list[dict],
    *,
    status: str,
    scope: str,
    pending: list[str],
) -> dict:
    contract = json.loads(CONTRACT.read_text())
    paths = sorted(set(paths))
    manifest = {
        "schema": contract["delivery_manifest_schema"],
        "delivery_id": name.removesuffix(".json") + "_20260905",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "producer": "server_B_zhengyukun-1",
        "base_commit": COMMON_BASE,
        "base_commit_basis": "parent_of_first_B_scaffold_commit_5bcd01855_and_verified_ancestor_on_A; not_a_claim_about_unrecorded_dirty_snapshot_bytes",
        "delivery_status": status,
        "scope": scope,
        "transferred": False,
        "complete_A4_gate": False,
        "formal_evaluation": False,
        "pending": pending,
        "received_interface_handoff_sha256": sha256(
            BASE / "results/a2_acceptance/B_INTERFACE_HANDOFF.json"
        ),
        "received_failure_handoff_sha256": sha256(
            BASE / "results/a2_acceptance/B_FAILURE_TRAIN_HANDOFF.json"
        ),
        "delivery_contract_sha256": sha256(CONTRACT),
        "files": [file_entry(p, contract) for p in paths],
        "checkpoint_manifests": checkpoints,
    }
    output = BASE / "results/b_delivery" / name
    write_json(output, manifest)
    return verify_manifest(output)


def prepare() -> dict:
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", COMMON_BASE, "HEAD"], cwd=ROOT, check=True
    )
    first_parent = subprocess.check_output(
        ["git", "rev-parse", "5bcd01855^"], cwd=ROOT, text=True
    ).strip()
    if first_parent != COMMON_BASE:
        raise ValueError("common-base evidence changed")
    acceptance = json.loads((REVIEW / "ACCEPTANCE.json").read_text())
    inference = json.loads((TRAIN / "INFERENCE_VALIDATION.json").read_text())
    if (
        acceptance["status"] != "pass"
        or acceptance["checkpoints_verified"] != 114
        or inference["checkpoints"] != 114
    ):
        raise ValueError("M4 inference acceptance incomplete")
    immutable_copy(
        TRAIN / "queue_20260909_current_executor/packages.json",
        REVIEW / "ENVIRONMENT_PACKAGES.json",
    )
    manifests = sorted((BASE / "artifacts/checkpoints/m4").glob("*/*/*/checkpoint_manifest.json"))
    if len(manifests) != 114:
        raise ValueError("expected 114 M4 checkpoint manifests")
    main, e3, main_ck, e3_ck = [], [], [], []
    for path in manifests:
        m = json.loads(path.read_text())
        identity = path.parent.relative_to(BASE / "artifacts/checkpoints/m4")
        target, ck_target = (
            (main, main_ck)
            if m["training_budget"] == 200 and m["held_out_family"] is None
            else (e3, e3_ck)
        )
        run = TRAIN / "runs" / identity
        done = json.loads((run / "complete.json").read_text())
        if (
            done["status"] != "pass"
            or done["epochs"] != 100
            or done["checkpoint_sha256"] != m["checkpoint_sha256"]
        ):
            raise ValueError("training is incomplete or checkpoint identity changed")
        for code_path, digest in done["code_sha256"].items():
            if sha256(ROOT / code_path) != digest:
                raise ValueError(f"frozen training source differs: {code_path}")
        target.extend(
            [
                path,
                ROOT / m["checkpoint_relative_path"],
                BASE / "artifacts/development_golden/m4" / identity / "golden.json",
            ]
        )
        target.extend(
            run / name
            for name in (
                "complete.json",
                "training_identity.json",
                "environment.json",
                "epochs.jsonl",
            )
        )
        ck_target.append(m)
    if len(main_ck) != 30 or len(e3_ck) != 84:
        raise ValueError("Main-10/E3 checkpoint counts do not match plan")
    from evaluations.iclr2027.methods.fail_detect.tools.validate_adapter import (
        OUTPUT as m3_contract_path,
    )
    from evaluations.iclr2027.methods.fail_detect.tools.validate_adapter import (
        evaluate_contract,
        verify_contract,
    )

    # Contract golden is explicitly synthetic, never proof of a bound Main-10
    # score model. Re-run it before listing the artifact in a handoff draft.
    verify_contract(json.loads(m3_contract_path.read_text()), evaluate_contract())
    m3_official_path = BASE / "artifacts/development_golden/m3/official_square_score_parity.json"
    m3_official = json.loads(m3_official_path.read_text())
    if (
        m3_official["status"] != "pass"
        or m3_official["scope"] != "official_square_only_not_Main10_golden"
        or m3_official["m3_formal_scorer_bound"]
        or m3_official["complete_A4_delivery"]
        or m3_official["records"] != 16
        or m3_official["max_absolute_error"] != 0.0
        or m3_official["wrapper_sha256"] != sha256(BASE / "methods/fail_detect/adapter.py")
    ):
        raise ValueError("official-domain parity evidence or its declared scope differs")
    shared = method_files("fail_detect") + method_files("failure_supervised")
    shared += [
        BASE / "configs/methods/m3_fail_detect.json",
        BASE / "configs/methods/m4_failure_supervised.json",
        REVIEW / "ACCEPTANCE.json",
        REVIEW / "ENVIRONMENT_PACKAGES.json",
        TRAIN / "INFERENCE_VALIDATION.json",
        m3_contract_path,
        m3_official_path,
    ]
    contract = json.loads(CONTRACT.read_text())
    expected_payloads = {
        name: payload_identity([file_entry(path, contract) for path in sorted(set(paths + shared))])
        for name, paths in (("B_TO_A_DELIVERY.json", main), ("B_TO_A_DELIVERY_e3_m4.json", e3))
    }
    portability = json.loads(PORTABILITY.read_text()) if PORTABILITY.exists() else {}
    portable = (
        portability.get("status") == "pass"
        and portability.get("scope") == "manifest_scoped_component_portability_only"
        and portability.get("payload_identities") == expected_payloads
    )
    if portable:
        shared.append(PORTABILITY)
    reports = [
        build(
            "B_TO_A_DELIVERY.json",
            main + shared,
            main_ck,
            status="prepared_for_A_component_integration"
            if portable
            else "local_component_portability_check_pending",
            scope="B_M3_adapter_verification_and_Main10_M4_200_artifacts_not_A_formal_acceptance",
            pending=[
                "A_integration_acceptance_under_section_14_7",
                "project_M3_real_scorer_binding_not_verified_by_adapter_test_golden",
                "A_only_formal_calibration_and_experiments",
            ],
        ),
        build(
            "B_TO_A_DELIVERY_e3_m4.json",
            e3 + shared,
            e3_ck,
            status="prepared_for_A_component_integration"
            if portable
            else "local_component_portability_check_pending",
            scope="E3_M4_budget_and_LOFO_training_artifacts_only",
            pending=["A_checkpoint_integration_and_A_only_formal_calibration_before_evaluation"],
        ),
    ]
    from evaluations.iclr2027.native_systems.racer.reproduction.native6_development import (
        OUTPUTS,
        TASKS,
        validate_result,
    )

    native_pending = {}
    for system in ("rvt", "racer"):
        existing = [OUTPUTS[system] / f"{task}.json" for task in TASKS]
        missing = [p.stem for p in existing if not p.is_file()]
        if missing:
            native_pending[system] = missing
            continue  # never hash a result that is still being written
        checks = [validate_result(p, system, p.stem) for p in existing]
        summary_path = OUTPUTS[system] / "HANDOFF_DEVELOPMENT_STATUS.json"
        write_json(
            summary_path,
            {
                "status": "pass",
                "scope": "nominal_development_only",
                "formal_evaluation": False,
                "tasks": checks,
                "episodes": 150,
                "pending": "A_Native6_episode_initialization_and_physical_fault_contract",
            },
        )
        paths = method_files(system, native=True) + existing + [summary_path]
        audit_src = (
            ROOT.parent / "_runs/native_systems/native6_development_20260905/source_audit.json"
        )
        audit_dst = OUTPUTS[system] / "SOURCE_AUDIT.json"
        immutable_copy(audit_src, audit_dst)
        paths.append(audit_dst)
        # The six JSON result files are final; progress traces are omitted because
        # they are operational logs and may still be append-open in a live queue.
        reports.append(
            build(
                f"B_TO_A_DELIVERY_e6_{system}.json",
                paths,
                [],
                status="local_development_draft_not_formal_E6",
                scope=f"{system}_native6_nominal_development_only",
                pending=[
                    "A_Native6_frozen_initialization_fault_audit_and_episode_contract",
                    "10_nominal_plus_10_perturbed_physical_gate",
                    "formal_E6_100_plus_100_per_task",
                ],
            )
        )
    report = {
        "status": "B_components_prepared_for_A_integration"
        if portable
        else "local_component_portability_check_pending",
        "local_component_portability_verified": portable,
        "transferred": False,
        "complete_A4_or_E6_claimed": False,
        "base_commit": COMMON_BASE,
        "manifests": reports,
        "native_nominal_missing_results": native_pending,
        "M4_training_and_local_inference_complete": True,
        "M3_adapter_contract_examples_verified": 18,
        "M3_official_Square_score_parity_verified": 16,
        "M3_project_scorer_binding_and_real_golden_ready": False,
        "M3_delivery_scope": "adapter_code_reproduction_and_explicit_test_golden_not_a_new_training_assignment",
        "new_Main10_M3_training_assigned_to_B": False,
    }
    write_json(REVIEW / "LOCAL_HANDOFF_STATUS.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            verify_manifest(args.verify) if args.verify else prepare(), indent=2, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
