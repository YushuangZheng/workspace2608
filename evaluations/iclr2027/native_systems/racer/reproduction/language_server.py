"""Local-path version of RACER's released language HTTP service."""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class TextInput(BaseModel):
    text: str
    model: str


def create_app(args: argparse.Namespace) -> Any:
    import torch
    from fastapi import FastAPI
    from transformers import AutoTokenizer, T5EncoderModel

    model_path = args.t5_model.resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = T5EncoderModel.from_pretrained(
        model_path,
        device_map={"": args.device},
        local_files_only=True,
    )
    model.eval()
    app = FastAPI()

    @lru_cache(maxsize=args.cache_size)
    def encode_cached(text: str) -> tuple[list[Any], int]:
        encoded = tokenizer(text, return_tensors="pt")
        input_ids = encoded.input_ids.to(args.device)
        with torch.inference_mode():
            embeddings = model(input_ids=input_ids).last_hidden_state
        token_len = int(input_ids.shape[1])
        return embeddings.float().cpu().numpy().tolist(), token_len

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model": "t5-11b",
            "model_path": str(model_path),
            "device": args.device,
        }

    @app.post("/encode/")
    async def encode_text(request: TextInput) -> dict[str, Any]:
        if request.model != "t5-11b":
            return {"error": f"Model {request.model} not loaded."}
        embeddings, token_len = encode_cached(request.text)
        return {"embeddings": embeddings, "token_len": token_len}

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache-size", type=int, default=512)
    parser.add_argument(
        "--t5-model",
        type=Path,
        default=Path("/home/ubuntu/workspace/_models/racer/t5-11b"),
    )
    return parser.parse_args()


def main() -> None:
    import uvicorn

    args = parse_args()
    uvicorn.run(create_app(args), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
