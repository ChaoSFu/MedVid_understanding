from __future__ import annotations

from ..config import TARGET_MODELS
from .qwen_hf import HuggingFaceQwenRunner


def make_qwen3vl_runner(model_key: str, model_path: str | None = None, model_name: str | None = None, **kwargs) -> HuggingFaceQwenRunner:
    if model_key not in {"qwen3vl_4b", "qwen3vl_8b"}:
        raise ValueError(f"Unsupported Qwen3-VL target key: {model_key}")
    path = model_path or TARGET_MODELS[model_key]
    return HuggingFaceQwenRunner(model_path=path, model_name=model_name or path, backend_kind="qwen3vl", **kwargs)
