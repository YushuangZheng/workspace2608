"""Measure RACER's released language and VLM service components on server B."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

OPEN_LLAVA_COMMIT = "cff008a5cc15d153e89314fe778251a2a8fbd271"
CLIP_RN50_SHA256 = (
    "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762"
)
T5_11B_SHA256 = (
    "5fdc64177b14b0f72437fea171a752c626733f67755842528c75b90cacd1c807"
)
LLAVA_BASE_SHARDS = {
    "model-00001-of-00004.safetensors": (
        "eda352b5dd159390824859f8a31fb1c015bdf161ae44678602e88e3b4b631e1b"
    ),
    "model-00002-of-00004.safetensors": (
        "639c39656a3ab2c3982d6ec389318efab7fee75296dc765335a4ff9a1a93101e"
    ),
    "model-00003-of-00004.safetensors": (
        "7db41d8a6f73eb6cbb276cad770889c3e13fac731130127c7a60dbc8b1c82a36"
    ),
    "model-00004-of-00004.safetensors": (
        "026b551b6165ea5cd9867f8b640ee819186fbdc032fb721c1304f261b34d436b"
    ),
}
LLAVA_ADAPTER_SHA256 = (
    "30b24a887c78e0c80d84d52953598dde200a73570a851b1f87a85dfb6c324989"
)
LLAVA_NON_LORA_SHA256 = (
    "f7e55b4c84d5dce58e6c8991d0fb671be72510e6b4b3284b8c475c6ece170ee3"
)
LLAVA_VISION_TOWER_SHA256 = (
    "c6032c2e0caae3dc2d4fba35535fa6307dbb49df59c7e182b1bc4b3329b81801"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_sha256(path: Path, expected: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(f"checksum mismatch for {path}: {actual}")


def _git_head(repository: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()


@contextmanager
def _local_llava_adapter(
    source: Path, base: Path, vision_tower: Path
) -> Iterator[Path]:
    """Expose official adapter files with only external resource paths changed."""
    with tempfile.TemporaryDirectory(prefix="racer-llava-adapter-") as raw_dir:
        target = Path(raw_dir)
        for item in source.iterdir():
            output = target / item.name
            if item.name == "config.json":
                config = json.loads(item.read_text(encoding="utf-8"))
                config["mm_vision_tower"] = str(vision_tower)
                output.write_text(
                    json.dumps(config, indent=2) + "\n", encoding="utf-8"
                )
            elif item.name == "adapter_config.json":
                config = json.loads(item.read_text(encoding="utf-8"))
                config["base_model_name_or_path"] = str(base)
                output.write_text(
                    json.dumps(config, indent=2) + "\n", encoding="utf-8"
                )
            else:
                output.symlink_to(item.resolve())
        yield target


def _begin_measurement() -> float:
    import torch

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    return time.perf_counter()


def _end_measurement(started: float) -> dict[str, float]:
    import torch

    torch.cuda.synchronize()
    return {
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "allocated_mib": round(torch.cuda.memory_allocated() / 1024**2, 3),
        "peak_memory_mib": round(
            torch.cuda.max_memory_allocated() / 1024**2, 3
        ),
    }


def _versions() -> dict[str, str]:
    import accelerate
    import bitsandbytes
    import peft
    import tokenizers
    import torch
    import transformers

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda_build": str(torch.version.cuda),
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "accelerate": accelerate.__version__,
        "peft": peft.__version__,
        "bitsandbytes": bitsandbytes.__version__,
    }


def _device_metadata() -> dict[str, Any]:
    import torch

    return {
        "device": "cuda:0",
        "device_name": torch.cuda.get_device_name(0),
        "compute_capability": ".".join(
            str(part) for part in torch.cuda.get_device_capability(0)
        ),
    }


def run_clip(args: argparse.Namespace) -> dict[str, Any]:
    import clip
    import torch

    checkpoint = args.clip_checkpoint.resolve()
    _assert_sha256(checkpoint, CLIP_RN50_SHA256)
    started = _begin_measurement()
    model, _ = clip.load(str(checkpoint), device="cuda:0")
    model.eval()
    tokens = clip.tokenize(args.text).to("cuda:0")
    with torch.inference_mode():
        hidden = model.token_embedding(tokens).type(model.dtype)
        hidden = hidden + model.positional_embedding.type(model.dtype)
        hidden = model.transformer(hidden.permute(1, 0, 2)).permute(1, 0, 2)
        hidden = model.ln_final(hidden).type(model.dtype)
    output_finite = bool(torch.isfinite(hidden).all().item())
    if not output_finite:
        raise RuntimeError("CLIP language encoder produced non-finite values")
    measurement = _end_measurement(started)
    result = {
        "component": "clip_language_encoder",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": CLIP_RN50_SHA256,
        "output_shape": list(hidden.shape),
        "output_finite": output_finite,
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        **measurement,
    }
    del hidden, model
    return result


def run_t5(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer, T5EncoderModel

    model_dir = args.t5_model.resolve()
    _assert_sha256(model_dir / "pytorch_model.bin", T5_11B_SHA256)
    started = _begin_measurement()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = T5EncoderModel.from_pretrained(
        model_dir,
        device_map={"": "cuda:0"},
        local_files_only=True,
    )
    model.eval()
    encoded = tokenizer(args.text, return_tensors="pt").to("cuda:0")
    with torch.inference_mode():
        hidden = model(**encoded).last_hidden_state
    output_finite = bool(torch.isfinite(hidden).all().item())
    if not output_finite:
        raise RuntimeError("T5-11B language encoder produced non-finite values")
    measurement = _end_measurement(started)
    result = {
        "component": "t5_11b_language_encoder",
        "model_dir": str(model_dir),
        "checkpoint_sha256": T5_11B_SHA256,
        "output_shape": list(hidden.shape),
        "output_finite": output_finite,
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        **measurement,
    }
    del encoded, hidden, model, tokenizer
    return result


def run_llava(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.model.builder import load_pretrained_model
    from PIL import Image

    repository = args.llava_repository.resolve()
    commit = _git_head(repository)
    if commit != OPEN_LLAVA_COMMIT:
        raise RuntimeError(f"Open-LLaVA-NeXT commit mismatch: {commit}")
    base_dir = args.llava_base.resolve()
    adapter_dir = args.llava_adapter.resolve()
    vision_tower = args.llava_vision_tower.resolve()
    for filename, expected in LLAVA_BASE_SHARDS.items():
        _assert_sha256(base_dir / filename, expected)
    _assert_sha256(adapter_dir / "adapter_model.safetensors", LLAVA_ADAPTER_SHA256)
    _assert_sha256(
        adapter_dir / "non_lora_trainables.bin", LLAVA_NON_LORA_SHA256
    )
    _assert_sha256(
        vision_tower / "pytorch_model.bin", LLAVA_VISION_TOWER_SHA256
    )

    started = _begin_measurement()
    with _local_llava_adapter(adapter_dir, base_dir, vision_tower) as local_adapter:
        tokenizer, model, image_processor, context_length = load_pretrained_model(
            model_path=str(local_adapter),
            model_base=str(base_dir),
            model_name="llava_llama3_lora",
            device="cuda",
            device_map="auto",
        )
    model.eval()
    image = Image.new("RGB", (336, 336), color=(127, 127, 127))
    prompt = f"{DEFAULT_IMAGE_TOKEN}\nDescribe the robot scene briefly."
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to("cuda:0")
    image_tensor = process_images([image], image_processor, model.config)
    if isinstance(image_tensor, list):
        image_tensor = [
            item.to(dtype=torch.float16, device="cuda:0")
            for item in image_tensor
        ]
    else:
        image_tensor = image_tensor.to(dtype=torch.float16, device="cuda:0")
    with torch.inference_mode():
        generated = model.generate(
            inputs=input_ids,
            images=image_tensor,
            image_sizes=[image.size],
            do_sample=False,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )
    generated_tokens = generated[:, input_ids.shape[1] :]
    if generated_tokens.shape[1] == 0:
        raise RuntimeError("LLaVA returned no generated tokens")
    text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    measurement = _end_measurement(started)
    result = {
        "component": "llava_vlm",
        "official_commit": commit,
        "base_dir": str(base_dir),
        "adapter_dir": str(adapter_dir),
        "vision_tower": str(vision_tower),
        "compatibility_override": (
            "resource paths only: local model base and vision tower"
        ),
        "precision": str(model.dtype),
        "context_length": context_length,
        "generated_token_count": int(generated_tokens.shape[1]),
        "generated_text_nonempty": bool(text.strip()),
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        **measurement,
    }
    del generated, generated_tokens, image_tensor, input_ids, model, tokenizer
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("component", choices=("clip", "t5", "llava"))
    parser.add_argument(
        "--clip-checkpoint",
        type=Path,
        default=Path("/home/ubuntu/workspace/_models/racer/clip/RN50.pt"),
    )
    parser.add_argument(
        "--t5-model",
        type=Path,
        default=Path("/home/ubuntu/workspace/_models/racer/t5-11b"),
    )
    parser.add_argument(
        "--llava-repository",
        type=Path,
        default=Path("/home/ubuntu/workspace/_external/Open-LLaVA-NeXT"),
    )
    parser.add_argument(
        "--llava-base",
        type=Path,
        default=Path(
            "/home/ubuntu/workspace/_models/racer/llama3-llava-next-8b"
        ),
    )
    parser.add_argument(
        "--llava-adapter",
        type=Path,
        default=Path("/home/ubuntu/workspace/_models/racer/llava-lora-rich"),
    )
    parser.add_argument(
        "--llava-vision-tower",
        type=Path,
        default=Path(
            "/home/ubuntu/workspace/_models/racer/clip-vit-large-patch14-336"
        ),
    )
    parser.add_argument("--text", default="close the red jar")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output is not None:
        args.output = args.output.resolve()
    runners = {"clip": run_clip, "t5": run_t5, "llava": run_llava}
    component_result = runners[args.component](args)
    result = {
        "status": "pass",
        "scope": "official_racer_service_component_gpu_smoke",
        "versions": _versions(),
        **_device_metadata(),
        **component_result,
        "formal_evaluation": "not_run_requires_server_a_native6_manifest",
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
