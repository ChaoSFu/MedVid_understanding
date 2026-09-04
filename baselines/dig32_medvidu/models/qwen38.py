from __future__ import annotations

import time
from typing import Any

from ..config import DecodeConfig, PixelConfig, TARGET_MODELS
from .base import (
    build_frame_prompt,
    decode_config_for_task,
    open_selected_frames,
    pil_to_data_url,
    pixel_config_dict,
    prediction_row,
)
from baselines.videoitg_qwen35.models.qwen_sglang import build_sglang_messages, extract_message_text


class Qwen38SGLangRunner:
    def __init__(
        self,
        model_path: str | None = None,
        model_name: str | None = None,
        base_url: str = "http://127.0.0.1:30000/v1",
        api_key: str = "EMPTY",
        media_schema: str = "image_sequence",
        decode: DecodeConfig | None = None,
        pixel: PixelConfig | None = None,
        request_timeout: float = 900.0,
    ) -> None:
        self.model_path = model_path or TARGET_MODELS["qwen38_27b"]
        self.model_name = model_name or self.model_path
        self.base_url = base_url
        self.api_key = api_key
        self.media_schema = media_schema
        self.decode = decode or DecodeConfig()
        self.pixel = pixel or PixelConfig()
        self.request_timeout = request_timeout
        self.model_fingerprint = {
            "checkpoint": self.model_path,
            "model_name": self.model_name,
            "backend": "SGLang OpenAI-compatible API",
            "base_url": self.base_url,
            "served_model": self.model_path,
            "processor_class": "server-side",
            "model_class": "server-side",
            "config_model_type": None,
            "architectures": None,
            "thinking_support": {"extra_body_enable_thinking": True, "configured_thinking": self.decode.thinking},
            "multi_image_support": True,
        }

    def preflight(self) -> dict[str, Any]:
        return self.model_fingerprint

    def build_messages(self, prompt: str, frame_data_urls: list[str], timestamps: list[float]) -> list[dict[str, Any]]:
        px = pixel_config_dict(self.pixel)
        return build_sglang_messages(
            prompt=prompt,
            frame_data_urls=frame_data_urls,
            timestamps=timestamps,
            media_schema=self.media_schema,
            min_pixels=px.get("min_pixels"),
            max_pixels=px.get("max_pixels"),
        )

    async def generate(self, manifest_row: dict[str, Any], selection_row: dict[str, Any], selector_manifest_sha256: str) -> dict[str, Any]:
        from openai import AsyncOpenAI
        from ..tasks import get_adapter

        adapter = get_adapter(str(manifest_row["qa_type"]))
        prompt, timestamps = build_frame_prompt(manifest_row, selection_row)
        frames, image_sizes = open_selected_frames(selection_row)
        frame_data_urls = [pil_to_data_url(image) for image in frames]
        messages = self.build_messages(prompt, frame_data_urls, timestamps)
        decode_cfg = decode_config_for_task(adapter, self.decode)
        client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.request_timeout, max_retries=0)
        start = time.time()
        response = await client.chat.completions.create(
            model=self.model_path,
            messages=messages,
            temperature=0.0,
            top_p=1.0,
            max_tokens=decode_cfg["max_new_tokens"],
            stream=False,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": self.decode.thinking},
                "top_k": 1,
                "repetition_penalty": 1.0,
            },
        )
        elapsed = time.time() - start
        message = response.choices[0].message if response.choices else None
        answer, answer_source = extract_message_text(message) if message is not None else ("", "empty")
        usage = {}
        if getattr(response, "usage", None) is not None:
            usage_obj = response.usage
            usage = {
                "prompt_tokens": getattr(usage_obj, "prompt_tokens", None),
                "completion_tokens": getattr(usage_obj, "completion_tokens", None),
                "total_tokens": getattr(usage_obj, "total_tokens", None),
            }
        return prediction_row(
            manifest_row=manifest_row,
            selection_row=selection_row,
            answer=answer.strip(),
            model_name=self.model_name,
            backend="SGLang OpenAI-compatible API",
            selector_manifest_sha256=selector_manifest_sha256,
            model_fingerprint=self.model_fingerprint,
            prompt=prompt,
            decode_config=decode_cfg,
            pixel_config=pixel_config_dict(self.pixel),
            image_sizes=image_sizes,
            elapsed_sec=elapsed,
            extra_info={"base_url": self.base_url, "media_schema": self.media_schema, "usage": usage, "answer_source": answer_source},
        )
