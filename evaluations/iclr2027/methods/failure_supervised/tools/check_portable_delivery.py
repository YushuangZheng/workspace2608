"""Check the B component delivery using only its manifest and A's public interface.

This creates an isolated temporary copy, never contacts A, trains nothing and
does not claim the analytic M3 test backend is a formal experiment model.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .prepare_handoff import BASE, ROOT, payload_identity, sha256, verify_manifest, write_json

OUTPUT = BASE / "artifacts/training/m4/handoff/PORTABLE_DELIVERY_VALIDATION.json"
MANIFEST_NAMES = ("B_TO_A_DELIVERY.json", "B_TO_A_DELIVERY_e3_m4.json")

BOOTSTRAP = r"""
import contextlib, importlib.util, io, json, pathlib, runpy, sys
root = pathlib.Path.cwd().resolve()
module, *args = sys.argv[1:]
spec = importlib.util.find_spec(module)
if module.startswith("evaluations.") and root not in pathlib.Path(spec.origin).resolve().parents:
    raise RuntimeError("entrypoint resolved outside the isolated delivery")
sys.argv = [module, *args]
buffer = io.StringIO()
with contextlib.redirect_stdout(buffer):
    try:
        runpy.run_module(module, run_name="__main__")
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise RuntimeError(buffer.getvalue()) from exc
origins = {}
for name, value in tuple(sys.modules.items()):
    if not name.startswith("evaluations.") or not getattr(value, "__file__", None):
        continue
    path = pathlib.Path(value.__file__).resolve()
    if root not in path.parents:
        raise RuntimeError("project import escaped isolated delivery: " + str(path))
    origins[name] = str(path.relative_to(root))
if not origins:
    raise RuntimeError("no project modules were exercised")
print(json.dumps({"module": module, "stdout": buffer.getvalue(), "module_origins": origins}))
"""


def copy_entry(entry, target):
    relative = Path(entry["path"])
    if relative.is_absolute() or ".." in relative.parts or "\\" in entry["path"]:
        raise ValueError("noncanonical copy path")
    source = ROOT / relative
    if source.is_symlink() or ROOT not in source.resolve().parents:
        raise ValueError("copy source escapes project")
    if source.stat().st_size != entry["bytes"] or sha256(source) != entry["sha256"]:
        raise ValueError(f"copy source identity differs: {relative}")
    destination = target / relative
    if destination.exists():
        if sha256(destination) != entry["sha256"]:
            raise ValueError("batches disagree on a shared file")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if sha256(destination) != entry["sha256"]:
        raise ValueError("copied byte identity differs")


def run_entrypoint(target, module, *args):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    process = subprocess.run(
        [sys.executable, "-s", "-c", BOOTSTRAP, module, *args],
        cwd=target,
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
    )
    if process.returncode:
        raise RuntimeError(f"isolated {module} failed:\n{process.stdout}\n{process.stderr}")
    return json.loads(process.stdout)


def check():
    manifests = []
    for name in MANIFEST_NAMES:
        path = BASE / "results/b_delivery" / name
        verify_manifest(path)
        manifests.append(json.loads(path.read_text()))
    interface = json.loads((BASE / "results/a2_acceptance/B_INTERFACE_HANDOFF.json").read_text())
    if interface["contains_normal_calibration"] or interface["contains_sealed_test"]:
        raise ValueError("interface fixture must not include private evaluation data")
    prefixes = (
        "evaluations/iclr2027/interfaces/",
        "evaluations/iclr2027/configs/shared/",
        "evaluations/iclr2027/tests/fixtures/development_examples/",
    )
    if any(not e["path"].startswith(prefixes) for e in interface["files"]):
        raise ValueError("unexpected A input outside the public interface/development list")
    stages = []
    with tempfile.TemporaryDirectory(
        prefix="b_component_delivery_", dir=ROOT.parent / "_runs"
    ) as directory:
        target = Path(directory).resolve()
        for entry in interface["files"]:
            copy_entry(entry, target)
        for number, manifest in enumerate(manifests):
            for entry in manifest["files"]:
                copy_entry(entry, target)
            if (target / "evaluations/iclr2027/datasets").exists():
                raise ValueError("an isolated inference test must not contain a training pool")
            m4_run = run_entrypoint(
                target, "evaluations.iclr2027.methods.failure_supervised.tools.acceptance", "--all"
            )
            m4 = json.loads(m4_run.pop("stdout"))
            if m4["checkpoints_verified"] != (30 if number == 0 else 114):
                raise ValueError("isolated delivery checkpoint coverage differs")
            result = {"batch": MANIFEST_NAMES[number], "M4": m4, "import_check": m4_run}
            if number == 0:
                m3_run = run_entrypoint(
                    target, "evaluations.iclr2027.methods.fail_detect.tools.validate_adapter"
                )
                result["M3_adapter_contract"] = json.loads(m3_run.pop("stdout"))
                result["M3_import_check"] = m3_run
                tests = run_entrypoint(
                    target,
                    "pytest",
                    "-q",
                    "--import-mode=importlib",
                    "-p",
                    "no:cacheprovider",
                    "evaluations/iclr2027/methods/fail_detect/tests",
                    "evaluations/iclr2027/methods/failure_supervised/tests",
                )
                result["method_tests"] = tests
            stages.append(result)
    return {
        "schema": "essay2608.iclr2027.b-component-portability.v1",
        "status": "pass",
        "scope": "manifest_scoped_component_portability_only",
        "payload_identities": {
            name: payload_identity(m["files"]) for name, m in zip(MANIFEST_NAMES, manifests)
        },
        "A_public_files_used": len(interface["files"]),
        "training_pool_copied": False,
        "training_performed": False,
        "calibration_or_sealed_copied": False,
        "transferred_to_A": False,
        "M3_formal_scorer_binding_verified": False,
        "A_integration_gate_verified": False,
        "stages": stages,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    report = check()
    if args.write_report:
        write_json(OUTPUT, report)
    print(json.dumps({k: v for k, v in report.items() if k != "stages"}, indent=2))


if __name__ == "__main__":
    main()
