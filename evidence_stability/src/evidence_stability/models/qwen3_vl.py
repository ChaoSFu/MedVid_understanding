from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from evidence_stability.model_interface import BaseVideoVLM


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _torch_dtype(torch_module: Any, dtype_name: str) -> Any:
    normalized = dtype_name.lower()
    if normalized in {"bfloat16", "bf16"}:
        return torch_module.bfloat16
    if normalized in {"float16", "fp16", "half"}:
        return torch_module.float16
    if normalized in {"float32", "fp32"}:
        return torch_module.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def _resolve_model_class(transformers_module: Any, config: dict[str, Any]) -> type:
    architectures = config.get("architectures") or []
    for architecture in architectures:
        cls = getattr(transformers_module, architecture, None)
        if cls is not None and hasattr(cls, "from_pretrained"):
            return cls

    candidate_names = [
        "AutoModelForImageTextToText",
        "Qwen3VLForConditionalGeneration",
        "AutoModelForVision2Seq",
        "AutoModelForCausalLM",
    ]
    for name in candidate_names:
        cls = getattr(transformers_module, name, None)
        if cls is not None and hasattr(cls, "from_pretrained"):
            return cls
    raise RuntimeError("No supported Qwen3-VL Transformers model class is available.")


class Qwen3VLVideoWindowModel(BaseVideoVLM):
    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        max_new_tokens: int = 8,
        do_sample: bool = False,
        processor_min_pixels: int | None = None,
        processor_max_pixels: int | None = None,
    ) -> None:
        try:
            import torch
            import transformers
            from PIL import Image
            from transformers import AutoProcessor
        except ImportError as exc:
            raise ImportError("Install torch, transformers, and Pillow to use Qwen3-VL backend.") from exc

        self.torch = torch
        self.transformers = transformers
        self.Image = Image
        self.model_path = str(Path(model_path))
        self.device = device
        self.dtype_name = dtype
        self.torch_dtype = _torch_dtype(torch, dtype)
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.processor_min_pixels = processor_min_pixels
        self.processor_max_pixels = processor_max_pixels
        self.last_processor_metadata: dict[str, Any] = {}
        self.last_debug_metadata: dict[str, Any] = {}

        model_dir = Path(self.model_path)
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Model directory not found: {self.model_path}")

        config = _read_json_if_exists(model_dir / "config.json")
        model_class = _resolve_model_class(transformers, config)
        processor_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "local_files_only": True,
        }
        if processor_min_pixels is not None:
            processor_kwargs["min_pixels"] = processor_min_pixels
        if processor_max_pixels is not None:
            processor_kwargs["max_pixels"] = processor_max_pixels

        self.processor = AutoProcessor.from_pretrained(self.model_path, **processor_kwargs)
        self.model = model_class.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model.to(self.device)
        self.model.eval()

        self.model_name = "qwen3_vl_8b"
        self.model_revision = self.fingerprint()["model_identity_hash"]

    def fingerprint(self) -> dict[str, Any]:
        model_dir = Path(self.model_path)
        config = _read_json_if_exists(model_dir / "config.json")
        generation_config = _read_json_if_exists(model_dir / "generation_config.json")
        payload = {
            "model_path": self.model_path,
            "config_sha256": _sha256_file(model_dir / "config.json"),
            "generation_config_sha256": _sha256_file(model_dir / "generation_config.json"),
            "architectures": config.get("architectures"),
            "model_type": config.get("model_type"),
            "processor_class": self.processor.__class__.__name__,
            "model_class": self.model.__class__.__name__,
            "transformers_version": self.transformers.__version__,
            "torch_version": self.torch.__version__,
            "dtype": self.dtype_name,
            "device": self.device,
            "do_sample": self.do_sample,
            "max_new_tokens": self.max_new_tokens,
            "enable_thinking": False,
            "processor_min_pixels": self.processor_min_pixels,
            "processor_max_pixels": self.processor_max_pixels,
        }
        payload["model_identity_hash"] = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return payload

    def generation_config(self) -> dict[str, Any]:
        return {
            "do_sample": self.do_sample,
            "max_new_tokens": self.max_new_tokens,
            "enable_thinking": False,
        }

    def gpu_memory_stats(self) -> dict[str, Any]:
        if not str(self.device).startswith("cuda") or not self.torch.cuda.is_available():
            return {
                "gpu_name": None,
                "peak_allocated_bytes": None,
                "peak_reserved_bytes": None,
                "current_allocated_bytes": None,
                "current_reserved_bytes": None,
            }
        device_obj = self.torch.device(self.device)
        return {
            "gpu_name": self.torch.cuda.get_device_name(device_obj),
            "peak_allocated_bytes": self.torch.cuda.max_memory_allocated(device_obj),
            "peak_reserved_bytes": self.torch.cuda.max_memory_reserved(device_obj),
            "current_allocated_bytes": self.torch.cuda.memory_allocated(device_obj),
            "current_reserved_bytes": self.torch.cuda.memory_reserved(device_obj),
        }

    def _processor_call(self, messages: list[dict[str, Any]], images: list[Any]) -> Any:
        try:
            return self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=False,
            )
        except TypeError:
            try:
                return self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )
            except TypeError:
                try:
                    text = self.processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                except TypeError:
                    text = self.processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                return self.processor(text=[text], images=images, return_tensors="pt")

    @staticmethod
    def _to_device(inputs: Any, device: str) -> Any:
        if hasattr(inputs, "to"):
            return inputs.to(device)
        return {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    def _collect_processor_metadata(self, inputs: Any, frame_count: int, image_sizes: list[tuple[int, int]]) -> dict[str, Any]:
        data = inputs if isinstance(inputs, dict) else dict(inputs)
        metadata: dict[str, Any] = {
            "input_frame_count": frame_count,
            "image_sizes": [list(size) for size in image_sizes],
            "input_keys": sorted(data.keys()),
        }
        input_ids = data.get("input_ids")
        if input_ids is not None:
            metadata["input_ids_shape"] = list(input_ids.shape)
            metadata["input_token_length"] = int(input_ids.shape[-1])
        for key in ("image_grid_thw", "video_grid_thw"):
            value = data.get(key)
            if value is not None:
                metadata[f"{key}_shape"] = list(value.shape)
                metadata[f"{key}_value"] = value.detach().cpu().tolist()
                metadata[f"{key}_rows"] = int(value.shape[0]) if hasattr(value, "shape") and value.ndim > 0 else None
        pixel_values = data.get("pixel_values")
        if pixel_values is not None:
            metadata["pixel_values_shape"] = list(pixel_values.shape)
        return metadata

    def infer(self, image_paths: list[str], prompt: str) -> str:
        images = [self.Image.open(path).convert("RGB") for path in image_paths]
        image_sizes = [img.size for img in images]
        content: list[dict[str, Any]] = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        inputs = self._processor_call(messages, images)
        self.last_processor_metadata = self._collect_processor_metadata(inputs, len(images), image_sizes)
        inputs = self._to_device(inputs, self.device)

        if str(self.device).startswith("cuda") and self.torch.cuda.is_available():
            self.torch.cuda.reset_peak_memory_stats(self.torch.device(self.device))

        with self.torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                do_sample=self.do_sample,
                max_new_tokens=self.max_new_tokens,
            )

        input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
        generated_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(input_ids, generated_ids)
        ]
        decoded = self.processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        self.last_debug_metadata = {
            "image_sizes": [list(size) for size in image_sizes],
            "processor_metadata": self.last_processor_metadata,
            "gpu_memory": self.gpu_memory_stats(),
        }
        return decoded.strip()
