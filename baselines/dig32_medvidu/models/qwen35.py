from __future__ import annotations

from ..config import TARGET_MODELS
from .qwen_hf import HuggingFaceQwenRunner


def make_qwen35_runner(model_path: str | None = None, model_name: str | None = None, **kwargs) -> HuggingFaceQwenRunner:
    path = model_path or TARGET_MODELS["qwen35_4b"]
    return HuggingFaceQwenRunner(model_path=path, model_name=model_name or path, backend_kind="qwen35", **kwargs)
