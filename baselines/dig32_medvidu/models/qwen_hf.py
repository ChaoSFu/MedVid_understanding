from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..config import DecodeConfig, PixelConfig
from .base import (
    build_frame_prompt,
    decode_config_for_task,
    open_selected_frames,
    pixel_config_dict,
    prediction_row,
)


def build_image_sequence_messages(prompt: str, frames: list[Any], timestamps: list[float], pixel_config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for i, (image, timestamp) in enumerate(zip(frames, timestamps), start=1):
        content.append({"type": "text", "text": f"Frame {i}: {timestamp:.3f} seconds"})
        item: dict[str, Any] = {"type": "image", "image": image}
        if pixel_config:
            for key in ("min_pixels", "max_pixels"):
                if pixel_config.get(key) is not None:
                    item[key] = pixel_config[key]
        content.append(item)
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def build_model_fingerprint(
    checkpoint: str,
    model_name: str,
    processor: Any,
    model: Any,
    transformers_version: str,
    torch_version: str,
    dtype: str,
    device_map: str,
    torch_dtype: str,
    attention_implementation: str | None,
) -> dict[str, Any]:
    config = getattr(model, "config", None)
    vision_config = getattr(config, "vision_config", None)
    if hasattr(vision_config, "to_dict"):
        vision_config = vision_config.to_dict()
    return {
        "checkpoint": checkpoint,
        "model_name": model_name,
        "processor_class": processor.__class__.__name__,
        "model_class": model.__class__.__name__,
        "config_model_type": getattr(config, "model_type", None),
        "architectures": getattr(config, "architectures", None),
        "transformers_version": transformers_version,
        "torch_version": torch_version,
        "dtype": dtype,
        "attention_implementation": attention_implementation,
        "device_map": device_map,
        "torch_dtype_arg": torch_dtype,
        "vision_config": vision_config,
        "thinking_support": {
            "apply_chat_template_enable_thinking": hasattr(processor, "apply_chat_template"),
            "configured_thinking": False,
        },
        "multi_image_support": True,
    }


class HuggingFaceQwenRunner:
    def __init__(
        self,
        model_path: str,
        model_name: str,
        backend_kind: str,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        attention_implementation: str | None = None,
        decode: DecodeConfig | None = None,
        pixel: PixelConfig | None = None,
    ) -> None:
        self.model_path = model_path
        self.model_name = model_name
        self.backend_kind = backend_kind
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.attention_implementation = attention_implementation
        self.decode = decode or DecodeConfig()
        self.pixel = pixel or PixelConfig()
        self._loaded = False
        self.model_fingerprint: dict[str, Any] = {}

    def load(self) -> None:
        if self._loaded:
            return
        import torch
        import transformers
        from transformers import AutoModelForImageTextToText, AutoProcessor

        model_kwargs: dict[str, Any] = {"device_map": self.device_map, "torch_dtype": self.torch_dtype}
        if self.attention_implementation:
            model_kwargs["attn_implementation"] = self.attention_implementation

        processor = AutoProcessor.from_pretrained(self.model_path)
        model_cls: Any = AutoModelForImageTextToText
        if self.backend_kind == "qwen3vl":
            model_cls = getattr(transformers, "Qwen3VLForConditionalGeneration", AutoModelForImageTextToText)
        model = model_cls.from_pretrained(self.model_path, **model_kwargs).eval()
        self.torch = torch
        self.transformers = transformers
        self.processor = processor
        self.model = model
        self.model_fingerprint = build_model_fingerprint(
            checkpoint=self.model_path,
            model_name=self.model_name,
            processor=processor,
            model=model,
            transformers_version=transformers.__version__,
            torch_version=torch.__version__,
            dtype=str(next(model.parameters()).dtype),
            device_map=self.device_map,
            torch_dtype=self.torch_dtype,
            attention_implementation=self.attention_implementation,
        )
        self._loaded = True

    def preflight(self) -> dict[str, Any]:
        self.load()
        return self.model_fingerprint

    def _build_inputs(self, messages: list[dict[str, Any]]):
        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info(messages)
        kwargs = {}
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.decode.thinking,
        )
        pixel_cfg = pixel_config_dict(self.pixel)
        if pixel_cfg.get("total_pixels") is not None:
            kwargs["total_pixels"] = pixel_cfg["total_pixels"]
        for key in ("min_pixels", "max_pixels"):
            if pixel_cfg.get(key) is not None:
                kwargs[key] = pixel_cfg[key]
        inputs = self.processor(text=text, images=image_inputs, videos=video_inputs, return_tensors="pt", **kwargs)
        try:
            return inputs.to(self.model.device)
        except Exception:
            return inputs.to("cuda" if self.torch.cuda.is_available() else "cpu")

    def generate(self, manifest_row: dict[str, Any], selection_row: dict[str, Any], selector_manifest_sha256: str) -> dict[str, Any]:
        self.load()
        from ..tasks import get_adapter

        adapter = get_adapter(str(manifest_row["qa_type"]))
        prompt, timestamps = build_frame_prompt(manifest_row, selection_row)
        frames, image_sizes = open_selected_frames(selection_row)
        messages = build_image_sequence_messages(prompt, frames, timestamps, pixel_config_dict(self.pixel))
        inputs = self._build_inputs(messages)
        decode_cfg = decode_config_for_task(adapter, self.decode)
        start = time.time()
        if self.torch.cuda.is_available():
            self.torch.cuda.reset_peak_memory_stats()
        with self.torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=decode_cfg["max_new_tokens"],
                do_sample=False,
                num_beams=1,
                use_cache=True,
            )
        elapsed = time.time() - start
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        answers = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        memory = {}
        if self.torch.cuda.is_available():
            memory = {
                "peak_allocated_bytes": int(self.torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(self.torch.cuda.max_memory_reserved()),
            }
        return prediction_row(
            manifest_row=manifest_row,
            selection_row=selection_row,
            answer=(answers[0].strip() if answers else ""),
            model_name=self.model_name,
            backend=self.model_fingerprint.get("model_class") or "HuggingFace",
            selector_manifest_sha256=selector_manifest_sha256,
            model_fingerprint=self.model_fingerprint,
            prompt=prompt,
            decode_config=decode_cfg,
            pixel_config=pixel_config_dict(self.pixel),
            image_sizes=image_sizes,
            elapsed_sec=elapsed,
            extra_info={"input_tokens": int(inputs.input_ids.shape[-1]), "gpu_memory": memory},
        )
