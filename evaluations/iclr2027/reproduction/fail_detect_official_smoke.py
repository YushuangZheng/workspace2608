"""Run an algorithm-level smoke against the pinned official FAIL-Detect code.

This does not claim an official task result: the public repository does not
ship the Robomimic dataset, policy checkpoint, exported feature tensors, or a
trained logpZO checkpoint.  It verifies that the official network can train
for one step and that this repository's score/CP adapters agree with the
official implementations on deterministic synthetic data.
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from evaluations.iclr2027.monitors import (
    TimeVaryingConformalBand,
    TorchLogpZOScorer,
    prepare_logpzo_input,
)

PINNED_COMMIT = "b758e55f7c0c988188f2e4876ffc03ae8a3c30ed"
DEFAULT_OFFICIAL_ROOT = Path("/home/ubuntu/workspace/_external/FAIL-Detect")


def _official_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_official_modules(root: Path):
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "UQ_baselines" / "CFM"))
    sys.path.insert(0, str(root / "UQ_test"))
    net_cfm = importlib.import_module("net_CFM")
    functional = importlib.import_module("timeseries_cp.methods.functional_predictor")
    data_utils = importlib.import_module("timeseries_cp.utils.data_utils")
    return net_cfm, functional, data_utils


def run_smoke(official_root: Path, device: str) -> dict[str, object]:
    import torch

    commit = _official_commit(official_root)
    if commit != PINNED_COMMIT:
        raise RuntimeError(f"official checkout drifted: expected {PINNED_COMMIT}, got {commit}")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access it")

    net_cfm, functional, data_utils = _load_official_modules(official_root)
    torch.manual_seed(2608)
    np.random.seed(2608)

    model = net_cfm.get_unet(10).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4)
    clean_feature = torch.randn(2, 28, 10, device=device)
    noise = torch.randn_like(clean_feature)
    continuous_time = torch.rand(2, 1, 1, device=device)
    interpolated = clean_feature + continuous_time * (noise - clean_feature)
    predicted_velocity = model(interpolated, (continuous_time.reshape(-1) * 100).long())
    train_loss = (predicted_velocity - (noise - clean_feature)).square().mean()
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    square_feature = np.linspace(-1.0, 1.0, 274, dtype=np.float32).reshape(1, -1)
    adjusted = prepare_logpzo_input(square_feature, input_dim=10)
    scorer = TorchLogpZOScorer(model, input_dim=10, device=device)
    logpzo_score = scorer(square_feature)

    mean_scores = np.asarray(
        [[0.2, 0.3, 0.7], [0.3, 0.5, 0.8], [0.1, 0.4, 0.6], [0.4, 0.6, 0.9]],
        dtype=np.float64,
    )
    width_scores = np.asarray(
        [[0.3, 0.7, 1.0], [0.2, 0.6, 0.9], [0.5, 0.8, 1.1]],
        dtype=np.float64,
    )
    alpha = 0.2
    official_predictor = functional.FunctionalPredictor(
        modulation_type=functional.ModulationType.Tfunc,
        regression_type=data_utils.RegressionType.Mean,
    )
    official_upper = official_predictor.get_one_sided_prediction_band(
        mean_scores,
        width_scores,
        alpha=alpha,
        lower_bound=False,
    ).reshape(-1)
    adapter_band = TimeVaryingConformalBand.fit(
        mean_scores,
        width_scores,
        alpha=alpha,
    )
    cp_max_abs_error = float(np.max(np.abs(official_upper - adapter_band.upper)))
    if cp_max_abs_error > 1.0e-12:
        raise AssertionError(f"conformal adapter mismatch: {cp_max_abs_error}")
    if not np.isfinite(logpzo_score):
        raise AssertionError("logpZO smoke returned a non-finite score")

    return {
        "status": "pass",
        "scope": "algorithm_only_synthetic_data",
        "official_commit": commit,
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "device": device,
        "official_unet_input_shape": [1, *adjusted.shape],
        "train_loss": float(train_loss.detach().cpu().item()),
        "logpzo_score": logpzo_score,
        "conformal_max_abs_error": cp_max_abs_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-root", type=Path, default=DEFAULT_OFFICIAL_ROOT)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.official_root.resolve(), args.device), indent=2))


if __name__ == "__main__":
    main()
