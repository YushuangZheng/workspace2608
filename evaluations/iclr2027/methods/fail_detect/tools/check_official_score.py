"""Compare the wrapper with the pinned official function using real Square weights.

This is a non-training official-domain regression check, not a DynaMAC/Main-10
model binding. The large external checkpoint stays outside the delivery tree.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import subprocess
import sys
from pathlib import Path

import torch

from ..adapter import TorchLogpZOScorer
from ..runtime import _sha256
from .validate_adapter import BASE, ROOT

COMMIT = "b758e55f7c0c988188f2e4876ffc03ae8a3c30ed"
OUTPUT = BASE / "artifacts/development_golden/m3/official_square_score_parity.json"


def evaluate(official_root: Path, run: Path):
    official_root = official_root.resolve()
    if (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=official_root, text=True).strip()
        != COMMIT
    ):
        raise ValueError("official source commit differs")
    subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=official_root, check=True)
    checkpoint = run / "logpzo/square_flow.ckpt"
    features = run / "square_data_flow.pt"
    completed = json.loads((run / "final_reproduction_result.json").read_text())
    if completed["status"] != "pass" or completed["official_commit"] != COMMIT:
        raise ValueError("official reproduction has not passed")
    hashes = {}
    for path in (checkpoint, features):
        relative = str(path.relative_to(run))
        digest = _sha256(path)
        if digest != completed["artifacts"][relative]:
            raise ValueError("official reproduction artifact hash differs")
        hashes[relative] = digest
    for path in (official_root, official_root / "UQ_baselines", official_root / "UQ_test"):
        sys.path.insert(0, str(path))
    get_unet = importlib.import_module("CFM.net_CFM").get_unet
    official = importlib.import_module("eval_load_baseline")
    torch.set_num_threads(1)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload["epoch"] != 200:
        raise ValueError("expected the completed 200-epoch official score model")
    model = get_unet(10)
    model.load_state_dict(payload["model"], strict=True)
    model.eval().requires_grad_(False)
    del payload
    feature_payload = torch.load(features, map_location="cpu", weights_only=True)
    selected = feature_payload["X"][:16].detach().to(dtype=torch.float32).clone()
    del feature_payload
    if selected.shape != (16, 274) or not torch.isfinite(selected).all():
        raise ValueError("not the pinned Square visual-policy representation")
    versions = [p._version for p in model.parameters()]
    scorer = TorchLogpZOScorer(model, input_dim=10, device="cpu")
    outputs = []
    with torch.inference_mode():
        for index, tensor in enumerate(selected):
            reference = float(official.logpZO_UQ(model, tensor[None, :], task_name="square")[0])
            actual = scorer(tensor.numpy()[None, :])
            if not math.isfinite(actual) or not math.isclose(
                actual, reference, abs_tol=1e-5, rel_tol=1e-6
            ):
                raise ValueError(f"official logpZO parity failed for feature {index}")
            outputs.append(
                {
                    "official_feature_index": index,
                    "input_sha256": hashlib.sha256(
                        tensor.numpy().astype("<f4").tobytes()
                    ).hexdigest(),
                    "official_score": reference,
                    "adapter_score": actual,
                    "absolute_error": abs(actual - reference),
                }
            )
    if versions != [p._version for p in model.parameters()]:
        raise ValueError("inference changed model tensor versions")
    return {
        "schema": "essay2608.iclr2027.m3-official-square-score-parity.v1",
        "status": "pass",
        "scope": "official_square_only_not_Main10_golden",
        "m3_formal_scorer_bound": False,
        "complete_A4_delivery": False,
        "official_commit": COMMIT,
        "artifacts_sha256": hashes,
        "official_function_sha256": _sha256(official_root / "UQ_test/eval_load_baseline.py"),
        "wrapper_sha256": _sha256(BASE / "methods/fail_detect/adapter.py"),
        "torch": str(torch.__version__),
        "device": "cpu",
        "dtype": "float32",
        "records": len(outputs),
        "training_performed": False,
        "model_parameters_unchanged": True,
        "Main10_data_read": False,
        "max_absolute_error": max(row["absolute_error"] for row in outputs),
        "outputs": outputs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-root", type=Path, default=ROOT.parent / "_external/FAIL-Detect")
    parser.add_argument(
        "--official-run",
        type=Path,
        default=ROOT.parent / "_runs/fail_detect/square_flow_seed1103_full_20260904",
    )
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    report = evaluate(args.official_root, args.official_run)
    if OUTPUT.exists():
        if json.loads(OUTPUT.read_text()) != report:
            raise ValueError(
                "existing official parity artifact differs; review, do not silently replace"
            )
    elif args.write:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with OUTPUT.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    else:
        raise FileNotFoundError(
            "official parity artifact missing; use --write for initial creation"
        )
    print(json.dumps({k: v for k, v in report.items() if k != "outputs"}, indent=2))


if __name__ == "__main__":
    main()
