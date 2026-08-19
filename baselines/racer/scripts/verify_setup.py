#!/usr/bin/env python3
"""Fail-fast, read-only preflight for the frozen RACER checkpoint run."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


EXPECTED_GIT = {
    "upstream": "df8cb2beec2e2061392ef0c4be93bda916dfd51e",
    "upstream/Open-LLaVA-NeXT": "cff008a5cc15d153e89314fe778251a2a8fbd271",
    "upstream/libs/PyRep": "231a1ac6b0a179cff53c1d403d379260b9f05f2f",
    "upstream/libs/RLbench": "84cd2691742001b1ad66ee195c6e95aa42d1248a",
    "upstream/libs/YARR": "d17ed930d2f0768cd89d82aafef81822fd056109",
    "upstream/libs/peract_colab/peract_colab": "3d1e573aa688e5989a82e3c5ac93d83f0f37accb",
    "upstream/racer/peract": "5c2988edb961d67d7a921cbbc638f69947debff8",
}

EXPECTED_FILES = {
    "checkpoints/racer-visuomotor-policy-rich/model_17.pth": 147540672,
    "checkpoints/racer-llava-llama3-lora-rich/adapter_model.safetensors": 708925520,
    "checkpoints/racer-llava-llama3-lora-rich/non_lora_trainables.bin": 41961648,
    "checkpoints/llama3-llava-next-8b/model-00001-of-00004.safetensors": 4976706872,
    "checkpoints/llama3-llava-next-8b/model-00002-of-00004.safetensors": 4999802616,
    "checkpoints/llama3-llava-next-8b/model-00003-of-00004.safetensors": 4915916080,
    "checkpoints/llama3-llava-next-8b/model-00004-of-00004.safetensors": 1817174536,
    "checkpoints/t5-11b/pytorch_model.bin": 45229452544,
}

EXPECTED_HF_REVISIONS = {
    "checkpoints/racer-visuomotor-policy-rich/.cache/huggingface/download/model_17.pth.metadata": "3d387f2627fac0c3988905cb70850afd873124f0",
    "checkpoints/racer-llava-llama3-lora-rich/.cache/huggingface/download/adapter_model.safetensors.metadata": "fa2f84f609d1c0861e3b7b7a7544c4ae5406f920",
    "checkpoints/llama3-llava-next-8b/.cache/huggingface/download/model-00001-of-00004.safetensors.metadata": "6d6dcb8d7948364bebef9c11aba6b0e6c8835391",
    "checkpoints/t5-11b/.cache/huggingface/download/pytorch_model.bin.metadata": "90f37703b3334dfe9d2b009bfcbfbf1ac9d28ea3",
}

TASKS = (
    "place_cups",
    "place_wine_at_rack_location",
    "sweep_to_dustpan_of_size",
)


class Checks:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}
        self.failures: list[str] = []

    def record(self, name: str, passed: bool, detail: object) -> None:
        self.values[name] = {"ok": bool(passed), "detail": detail}
        if not passed:
            self.failures.append(f"{name}: {detail}")


def command_output(command: list[str], cwd: Path | None = None,
                   env: dict[str, str] | None = None) -> tuple[bool, str]:
    process = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return process.returncode == 0, process.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Path to baselines/racer.",
    )
    parser.add_argument("--skip-imports", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    checks = Checks()

    for relative, expected in EXPECTED_GIT.items():
        path = root / relative
        ok, head = command_output(["git", "-C", str(path), "rev-parse", "HEAD"])
        checks.record(f"git:{relative}", ok and head == expected, head or "unreadable")

    for relative, expected_size in EXPECTED_FILES.items():
        path = root / relative
        actual = path.stat().st_size if path.is_file() else None
        checks.record(f"file:{relative}", actual == expected_size, actual)

    for relative, expected_revision in EXPECTED_HF_REVISIONS.items():
        path = root / relative
        actual = path.read_text(encoding="utf-8").splitlines()[0] if path.is_file() else None
        checks.record(f"hf:{relative}", actual == expected_revision, actual)

    index_path = root / "checkpoints/llama3-llava-next-8b/model.safetensors.index.json"
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        referenced = sorted(set(index["weight_map"].values()))
    except Exception as exc:  # noqa: BLE001 - report exact preflight failure
        referenced = []
        checks.record("llava_weight_index", False, repr(exc))
    else:
        expected = sorted(path.name for path in (root / "checkpoints/llama3-llava-next-8b").glob("model-*.safetensors"))
        checks.record("llava_weight_index", referenced == expected, referenced)

    test_root = root / "datasets/rlbench/test"
    for task in TASKS:
        episodes = test_root / task / "all_variations" / "episodes"
        episode_dirs = sorted(path for path in episodes.glob("episode*") if path.is_dir())
        required = ("low_dim_obs.pkl", "variation_number.pkl", "variation_descriptions.pkl")
        complete = len(episode_dirs) == 25 and all(
            all((episode / filename).is_file() for filename in required)
            for episode in episode_dirs
        )
        checks.record(f"dataset:{task}", complete, len(episode_dirs))

    adapters = {
        "actor_data_symlink": (
            root / "upstream/racer/data/rlbench/test",
            test_root,
        ),
        "actor_checkpoint_symlink": (
            root / "upstream/racer/runs/racer-visuomotor-policy-rich",
            root / "checkpoints/racer-visuomotor-policy-rich",
        ),
    }
    for name, (link, target) in adapters.items():
        actual = link.resolve() if link.exists() else None
        checks.record(name, link.is_symlink() and actual == target.resolve(), str(actual))

    coppelia = Path("/data/yukun/essay2608/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04")
    checks.record("coppeliasim", (coppelia / "libcoppeliaSim.so.1").is_file(), str(coppelia))
    user_settings = coppelia / "system/usrset.txt"
    setting_text = user_settings.read_text(encoding="utf-8", errors="replace") if user_settings.is_file() else ""
    checks.record("coppeliasim_old_edu_release", "allowOldEduRelease=7501" in setting_text, str(user_settings))

    xvfb = Path("/data/yukun/.cache/racer/xvfb-ubuntu-root/usr/bin/Xvfb")
    checks.record("user_xvfb", xvfb.is_file() and os.access(xvfb, os.X_OK), str(xvfb))

    vision_cache = Path(
        "/data/yukun/.cache/huggingface-racer/llava-runtime/hub/"
        "models--openai--clip-vit-large-patch14-336"
    )
    vision_revision = "ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
    vision_ref = vision_cache / "refs/main"
    actual_ref = vision_ref.read_text(encoding="utf-8").strip() if vision_ref.is_file() else None
    checks.record("vision_tower_revision", actual_ref == vision_revision, actual_ref)
    vision_weight = vision_cache / "snapshots" / vision_revision / "pytorch_model.bin"
    vision_size = vision_weight.resolve().stat().st_size if vision_weight.exists() else None
    checks.record("vision_tower_weight", vision_size == 1711974081, vision_size)

    rn50 = Path("/data/yukun/.cache/clip/RN50.pt")
    rn50_size = rn50.stat().st_size if rn50.is_file() else None
    checks.record("clip_rn50_weight", rn50_size == 255827503, rn50_size)

    actor_python = Path("/data/yukun/miniconda3/envs/dynamac-racer/bin/python")
    llava_python = Path("/data/yukun/miniconda3/envs/dynamac-racer-llava/bin/python")
    checks.record("actor_python", actor_python.is_file(), str(actor_python))
    checks.record("llava_python", llava_python.is_file(), str(llava_python))

    if not args.skip_imports and actor_python.is_file():
        actor_env = os.environ.copy()
        actor_env["COPPELIASIM_ROOT"] = str(coppelia)
        actor_env["LD_LIBRARY_PATH"] = str(coppelia)
        code = (
            "import torch,pytorch3d,pyrep,rlbench,tensorflow,fastapi,racer;"
            "print(torch.__version__,pytorch3d.__version__,pyrep.__version__,"
            "rlbench.__version__,tensorflow.__version__,fastapi.__version__)"
        )
        ok, output = command_output(
            [str(actor_python), "-c", code], cwd=root / "upstream", env=actor_env
        )
        checks.record("actor_imports", ok, output[-1000:])

    if not args.skip_imports and llava_python.is_file():
        code = (
            "import torch,transformers,accelerate,peft;"
            "from llava.model.builder import load_pretrained_model;"
            "print(torch.__version__,transformers.__version__,accelerate.__version__,peft.__version__)"
        )
        ok, output = command_output(
            [str(llava_python), "-c", code],
            cwd=root / "upstream/Open-LLaVA-NeXT",
        )
        checks.record("llava_imports", ok, output[-1000:])

    report = {
        "root": str(root),
        "ok": not checks.failures,
        "checks": checks.values,
        "failures": checks.failures,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not checks.failures else 1


if __name__ == "__main__":
    sys.exit(main())
