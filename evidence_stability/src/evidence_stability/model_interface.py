from __future__ import annotations

import base64
import hashlib
import mimetypes
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseVideoVLM(ABC):
    model_name: str
    model_revision: str | None

    @abstractmethod
    def infer(self, image_paths: list[str], prompt: str) -> str:
        raise NotImplementedError

    def fingerprint(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "model_class": self.__class__.__name__,
        }

    def generation_config(self) -> dict[str, Any]:
        return {}

    def gpu_memory_stats(self) -> dict[str, Any]:
        return {}


class DummyVideoVLM(BaseVideoVLM):
    """Deterministic frozen dummy model for smoke/integration tests."""

    def __init__(self, model_name: str = "dummy-video-vlm", model_revision: str | None = "v1") -> None:
        self.model_name = model_name
        self.model_revision = model_revision

    def infer(self, image_paths: list[str], prompt: str) -> str:
        payload = "\n".join(image_paths) + "\n" + prompt
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return "YES" if int(digest[:2], 16) % 3 == 0 else "NO"

    def generation_config(self) -> dict[str, Any]:
        return {"deterministic_dummy": True}


class OpenAICompatibleVideoVLM(BaseVideoVLM):
    """OpenAI-compatible image-list VLM adapter.

    This keeps Phase C model access behind the same frozen interface. It does
    not perform training, adaptation, or learned selection.
    """

    def __init__(
        self,
        model_name: str,
        model_revision: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_new_tokens: int = 8,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install openai to use OpenAICompatibleVideoVLM") from exc

        self.model_name = model_name
        self.model_revision = model_revision
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.client = OpenAI(
            base_url=base_url or os.environ.get("OPENAI_BASE_URL"),
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"),
        )

    @staticmethod
    def _image_to_data_url(path: str) -> str:
        mime = mimetypes.guess_type(path)[0] or "image/jpeg"
        data = Path(path).read_bytes()
        encoded = base64.b64encode(data).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    def infer(self, image_paths: list[str], prompt: str) -> str:
        content: list[dict[str, Any]] = []
        for image_path in image_paths:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._image_to_data_url(image_path)},
                }
            )
        content.append({"type": "text", "text": prompt})
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": content}],
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )
        return response.choices[0].message.content or ""

    def generation_config(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
        }


def build_model(args: Any) -> BaseVideoVLM:
    if args.model_backend == "dummy":
        return DummyVideoVLM(model_name=args.model_name, model_revision=args.model_revision)
    if args.model_backend == "openai_compatible":
        return OpenAICompatibleVideoVLM(
            model_name=args.model_name,
            model_revision=args.model_revision,
            base_url=args.base_url,
            api_key=args.api_key,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )
    if args.model_backend in {"qwen3_vl", "local_hf_vlm"}:
        from evidence_stability.models.qwen3_vl import Qwen3VLVideoWindowModel

        return Qwen3VLVideoWindowModel(
            model_path=args.model_path,
            model_name=(Path(args.model_path).name if args.model_backend == "local_hf_vlm" else None),
            device=args.device,
            device_map=getattr(args, "device_map", None),
            max_memory=getattr(args, "max_memory", None),
            dtype=args.dtype,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            processor_min_pixels=args.processor_min_pixels,
            processor_max_pixels=args.processor_max_pixels,
        )
    raise ValueError(f"Unsupported model_backend: {args.model_backend}")
