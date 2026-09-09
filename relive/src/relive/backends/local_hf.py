"""Strict local Hugging Face backend for an already-reviewed checkpoint.

This adapter intentionally has no Qwen-specific class list, prompt template, or
processor fallback.  A metadata-only ``inspect-local-hf`` report must first be
used to populate the exact architecture, template hash, call mode, and input
placement in the configuration.  The adapter then fails closed if those
declarations no longer match the local checkpoint.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path
from typing import Any

from PIL import Image

from relive.config import validate_backend
from .base import Backend, BackendError
from .inspection import CheckpointInspectionError, checkpoint_metadata


ADAPTER_VERSION = "relive-local-hf-v1"


def _identity_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _torch_dtype(torch_module: Any, name: str) -> Any:
    return {"bfloat16": torch_module.bfloat16, "float16": torch_module.float16,
            "float32": torch_module.float32}[name]


def _max_memory(value: dict[str, str] | None) -> dict[int | str, str] | None:
    if value is None:
        return None
    parsed: dict[int | str, str] = {}
    for key, limit in value.items():
        if key.startswith("cuda:"):
            parsed[int(key.removeprefix("cuda:"))] = limit
        elif key.isdigit():
            parsed[int(key)] = limit
        else:
            parsed[key] = limit
    return parsed


def _processor_template(processor: Any) -> str | None:
    template = getattr(processor, "chat_template", None)
    if isinstance(template, str):
        return template
    tokenizer = getattr(processor, "tokenizer", None)
    template = getattr(tokenizer, "chat_template", None)
    return template if isinstance(template, str) else None


class LocalHFBackend(Backend):
    """One exact Hugging Face model and one exact checkpoint chat contract."""

    synthetic = False

    def __init__(self, config: dict[str, Any]):
        validated = validate_backend(config)
        if validated["kind"] != "local_hf":
            raise ValueError("LocalHFBackend requires kind=local_hf")
        super().__init__(validated)
        try:
            self.checkpoint = checkpoint_metadata(self.config["model_path"])
        except CheckpointInspectionError as exc:
            raise RuntimeError("LOCAL_HF_CHECKPOINT_METADATA_FAILURE") from exc
        self._validate_declared_checkpoint()
        try:
            import torch
            import transformers
        except ImportError as exc:
            raise RuntimeError("LOCAL_HF_DEPENDENCY_UNAVAILABLE") from exc
        self.torch = torch
        self.transformers = transformers
        self._load_exact_checkpoint()

    def _validate_declared_checkpoint(self) -> None:
        if self.checkpoint["checkpoint_metadata_sha256"] != self.config["checkpoint_metadata_sha256"]:
            raise RuntimeError("LOCAL_HF_CHECKPOINT_METADATA_MISMATCH")
        architectures = self.checkpoint["architectures"]
        if self.config["model_class"] not in architectures:
            raise RuntimeError("LOCAL_HF_MODEL_CLASS_NOT_DECLARED")
        matched = [candidate for candidate in self.checkpoint["chat_template_candidates"]
                   if candidate["source"] == self.config["chat_template_source"]
                   and candidate["sha256"] == self.config["chat_template_sha256"]]
        if len(matched) != 1:
            raise RuntimeError("LOCAL_HF_CHAT_TEMPLATE_MISMATCH")

    def _load_exact_checkpoint(self) -> None:
        model_class = getattr(self.transformers, self.config["model_class"], None)
        if model_class is None or not hasattr(model_class, "from_pretrained"):
            raise RuntimeError("LOCAL_HF_UNSUPPORTED_DECLARED_ARCHITECTURE")
        auto_processor = getattr(self.transformers, "AutoProcessor", None)
        if auto_processor is None or not hasattr(auto_processor, "from_pretrained"):
            raise RuntimeError("LOCAL_HF_AUTOPROCESSOR_UNAVAILABLE")
        processor_kwargs = {
            "trust_remote_code": self.config["trust_remote_code"],
            "local_files_only": True,
        }
        for config_key, processor_key in (("processor_min_pixels", "min_pixels"),
                                          ("processor_max_pixels", "max_pixels")):
            if self.config[config_key] is not None:
                processor_kwargs[processor_key] = self.config[config_key]
        try:
            self.processor = auto_processor.from_pretrained(self.config["model_path"], **processor_kwargs)
        except Exception as exc:
            raise RuntimeError("LOCAL_HF_PROCESSOR_LOAD_FAILURE") from exc
        if self.processor.__class__.__name__ != self.config["processor_class"]:
            raise RuntimeError("LOCAL_HF_PROCESSOR_CLASS_MISMATCH")
        actual_template = _processor_template(self.processor)
        if actual_template is None or hashlib.sha256(actual_template.encode("utf-8")).hexdigest() != self.config["chat_template_sha256"]:
            raise RuntimeError("LOCAL_HF_LOADED_TEMPLATE_MISMATCH")
        model_kwargs = {
            "torch_dtype": _torch_dtype(self.torch, self.config["dtype"]),
            "trust_remote_code": self.config["trust_remote_code"],
            "local_files_only": True,
        }
        if self.config["device_map"] is not None:
            model_kwargs["device_map"] = self.config["device_map"]
            max_memory = _max_memory(self.config["max_memory"])
            if max_memory is not None:
                model_kwargs["max_memory"] = max_memory
        try:
            self.model = model_class.from_pretrained(self.config["model_path"], **model_kwargs)
        except Exception as exc:
            raise RuntimeError("LOCAL_HF_MODEL_LOAD_FAILURE") from exc
        if self.model.__class__.__name__ != self.config["model_class"]:
            raise RuntimeError("LOCAL_HF_LOADED_MODEL_CLASS_MISMATCH")
        if self.config["device_map"] is None:
            try:
                self.model.to(self.config["device"])
            except Exception as exc:
                raise RuntimeError("LOCAL_HF_MODEL_PLACEMENT_FAILURE") from exc
        self.model.eval()
        self._fingerprint = self._make_fingerprint()

    def _make_fingerprint(self) -> dict[str, Any]:
        placement = {
            "dtype": self.config["dtype"], "device": self.config["device"],
            "device_map": self.config["device_map"], "input_device": self.config["input_device"],
            "max_memory": self.config["max_memory"],
        }
        scientific_identity = {
            "adapter_version": ADAPTER_VERSION,
            "model": self.config["model"], "model_path": self.config["model_path"],
            "revision": self.config["revision"],
            "checkpoint_metadata_sha256": self.checkpoint["checkpoint_metadata_sha256"],
            "model_class": self.config["model_class"], "processor_class": self.config["processor_class"],
            "chat_template_source": self.config["chat_template_source"],
            "chat_template_sha256": self.config["chat_template_sha256"],
            "chat_message_layout": self.config["chat_message_layout"],
            "processor_call_mode": self.config["processor_call_mode"],
            "chat_template_kwargs": self.config["chat_template_kwargs"],
            "trust_remote_code": self.config["trust_remote_code"], "local_files_only": True,
            "processor_min_pixels": self.config["processor_min_pixels"],
            "processor_max_pixels": self.config["processor_max_pixels"],
            "generation": self.config["generation"], "image_order": self.config["image_order"],
            "frame_encoding": self.config["frame_encoding"],
            "jpeg_quality": self.config.get("jpeg_quality"),
            "transformers_version": getattr(self.transformers, "__version__", None),
            "torch_version": getattr(self.torch, "__version__", None),
            "actual_model_class": self.model.__class__.__name__,
            "actual_processor_class": self.processor.__class__.__name__,
        }
        return {
            "adapter_version": ADAPTER_VERSION, "synthetic": False, "scientific_identity": scientific_identity,
            "model_identity_hash": _identity_hash(scientific_identity), "runtime_placement": placement,
            "checkpoint_metadata": deepcopy(self.checkpoint),
            "timeout_seconds": self.config["timeout_seconds"],
            "timeout_enforcement": "NOT_SUPPORTED_FOR_IN_PROCESS_GENERATE",
            "max_retries": self.config["max_retries"],
            "preprocessing": {"decode": "Pillow_RGB", "frame_encoding": self.config["frame_encoding"],
                              "jpeg_quality": self.config.get("jpeg_quality"), "resize": False},
        }

    def fingerprint(self) -> dict[str, Any]:
        return deepcopy(self._fingerprint)

    def _prepared_image(self, path: str) -> Image.Image:
        try:
            with Image.open(path) as source:
                source.load()
                image = source.convert("RGB").copy()
        except (OSError, ValueError):
            raise BackendError("LOCAL_HF_IMAGE_DECODE_FAILURE") from None
        encoding = self.config["frame_encoding"]
        if encoding == "source":
            return image
        buffer = BytesIO()
        try:
            options = {"quality": self.config["jpeg_quality"]} if encoding == "jpeg" else {}
            image.save(buffer, format=encoding.upper(), **options)
            with Image.open(BytesIO(buffer.getvalue())) as reloaded:
                return reloaded.convert("RGB").copy()
        except (OSError, ValueError):
            raise BackendError("LOCAL_HF_IMAGE_ENCODING_FAILURE") from None

    def _messages(self, request: dict[str, Any], images: list[Image.Image]) -> list[dict[str, Any]]:
        paths = request.get("image_paths", [])
        frame_ids = request.get("frame_ids", [])
        if not isinstance(paths, list) or not isinstance(frame_ids, list) or len(paths) != len(frame_ids):
            raise BackendError("FRAME_IDENTITY_LENGTH_MISMATCH")
        if self.config["chat_message_layout"] == "images_then_text":
            content = [{"type": "image", "image": image} for image in images]
            content.append({"type": "text", "text": request["prompt"]})
        else:
            content = [{"type": "text", "text": request["prompt"]}]
            content.extend({"type": "image", "image": image} for image in images)
        return [{"role": "user", "content": content}]

    def _model_inputs(self, messages: list[dict[str, Any]], images: list[Image.Image]) -> Any:
        kwargs = dict(self.config["chat_template_kwargs"])
        try:
            if self.config["processor_call_mode"] == "tokenized_chat_template":
                inputs = self.processor.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True, return_dict=True,
                    return_tensors="pt", **kwargs
                )
            else:
                rendered = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, **kwargs
                )
                if not isinstance(rendered, str):
                    raise TypeError("checkpoint template did not render text")
                inputs = self.processor(text=[rendered], images=images, return_tensors="pt")
        except Exception:
            raise BackendError("LOCAL_HF_TEMPLATE_APPLY_FAILURE") from None
        if not isinstance(inputs, dict) and not hasattr(inputs, "items"):
            raise BackendError("LOCAL_HF_PROCESSOR_FAILURE")
        if "input_ids" not in inputs:
            raise BackendError("LOCAL_HF_PROCESSOR_FAILURE")
        try:
            return inputs.to(self.config["input_device"]) if hasattr(inputs, "to") else {
                key: value.to(self.config["input_device"]) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
        except Exception:
            raise BackendError("LOCAL_HF_INPUT_PLACEMENT_FAILURE") from None

    def infer(self, request: dict[str, Any]) -> str:
        self.calls += 1
        paths = request.get("image_paths", [])
        if not isinstance(paths, list):
            raise BackendError("FRAME_IDENTITY_LENGTH_MISMATCH")
        images = [self._prepared_image(path) for path in paths]
        messages = self._messages(request, images)
        inputs = self._model_inputs(messages, images)
        try:
            with self.torch.inference_mode():
                generated_ids = self.model.generate(**inputs, **self.config["generation"])
            input_ids = inputs["input_ids"]
            trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)]
            decoded = self.processor.batch_decode(trimmed, skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False)
            if not isinstance(decoded, list) or len(decoded) != 1 or not isinstance(decoded[0], str):
                raise ValueError("invalid decode")
            return decoded[0].strip()
        except BackendError:
            raise
        except Exception:
            raise BackendError("LOCAL_HF_GENERATION_FAILURE") from None
