"""Read-only inspection and explicit processor-only probes for local checkpoints.

The default inspector reads only small checkpoint metadata files. Optional
processor probes instantiate ``AutoProcessor`` from local files but never a
model, weight tensor, or checkpoint-provided code unless the caller explicitly
sets ``trust_remote_code``. This keeps configuration decisions evidence-based
without making assumptions about a checkpoint's architecture or chat contract.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image


INSPECTION_VERSION = "relive-local-hf-inspection-v1"
_METADATA_NAMES = (
    "config.json",
    "generation_config.json",
    "processor_config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
)


class CheckpointInspectionError(ValueError):
    """A checkpoint cannot be described without making an adapter assumption."""


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointInspectionError(f"invalid checkpoint metadata JSON: {path.name}") from exc
    if not isinstance(payload, dict):
        raise CheckpointInspectionError(f"checkpoint metadata must be an object: {path.name}")
    return payload


def _template_values(value: Any, location: str) -> list[tuple[str, str]]:
    """Find template strings while retaining their declared source path."""
    if isinstance(value, str):
        return [(location, value)]
    if isinstance(value, dict):
        found: list[tuple[str, str]] = []
        for key in sorted(value):
            if isinstance(key, str):
                found.extend(_template_values(value[key], f"{location}.{key}"))
        return found
    return []


def _chat_template_candidates(metadata: dict[str, dict[str, Any]], root: Path) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for filename, payload in metadata.items():
        if "chat_template" in payload:
            for source, template in _template_values(payload["chat_template"], f"{filename}:chat_template"):
                candidates.append({"source": source,
                                   "sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
                                   "length": len(template)})
    # Some Hugging Face repositories place the template in a separate file.
    for path in sorted(root.rglob("*.jinja")):
        if not path.is_file():
            continue
        try:
            template = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise CheckpointInspectionError(f"cannot read chat template: {path.name}") from exc
        candidates.append({"source": f"file:{path.relative_to(root).as_posix()}",
                           "sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
                           "length": len(template)})
    return candidates


def checkpoint_metadata(model_path: str | Path) -> dict[str, Any]:
    """Return a stable, metadata-only identity for a local checkpoint.

    The result is intentionally independent of absolute path and hardware state
    so a reviewed identity can be compared when the same files are mounted on a
    GPU host.  Weight files are listed only by path and size; they are never
    opened or hashed by this preflight step.
    """
    root = Path(model_path).expanduser().resolve()
    if not root.is_dir():
        raise CheckpointInspectionError("model_path is not a readable directory")
    config_path = root / "config.json"
    if not config_path.is_file():
        raise CheckpointInspectionError("checkpoint config.json is required")
    parsed: dict[str, dict[str, Any]] = {}
    files: dict[str, dict[str, Any]] = {}
    for name in _METADATA_NAMES:
        path = root / name
        if path.is_file():
            parsed[name] = _read_json(path)
            files[name] = {"sha256": _sha256_file(path), "size_bytes": path.stat().st_size}
    config = parsed["config.json"]
    architectures = config.get("architectures", [])
    if not isinstance(architectures, list) or any(not isinstance(item, str) or not item for item in architectures):
        architectures = []
    # A file inventory gives reviewers a way to spot a changed checkpoint
    # without reading potentially very large tensor data during a read-only
    # inspection.
    weight_inventory = []
    for path in sorted(root.glob("*.safetensors")) + sorted(root.glob("*.bin")):
        if path.is_file():
            weight_inventory.append({"path": path.name, "size_bytes": path.stat().st_size})
    for path in sorted(root.glob("*.index.json")):
        if path.is_file() and path.name not in files:
            files[path.name] = {"sha256": _sha256_file(path), "size_bytes": path.stat().st_size}
    processor_declarations = []
    for filename, payload in parsed.items():
        for field in ("processor_class", "tokenizer_class", "image_processor_type"):
            value = payload.get(field)
            if isinstance(value, str) and value:
                processor_declarations.append({"source": f"{filename}:{field}", "value": value})
    identity = {
        "inspection_version": INSPECTION_VERSION,
        "metadata_files": files,
        "model_type": config.get("model_type") if isinstance(config.get("model_type"), str) else None,
        "architectures": architectures,
        "auto_map": config.get("auto_map") if isinstance(config.get("auto_map"), dict) else None,
        "processor_declarations": processor_declarations,
        "chat_template_candidates": _chat_template_candidates(parsed, root),
        "weight_file_inventory": weight_inventory,
    }
    return {**identity, "checkpoint_metadata_sha256": _canonical_hash(identity)}


def _transformers_report(architectures: list[str]) -> dict[str, Any]:
    try:
        import transformers
    except Exception:
        return {"available": False, "version": None,
                "declared_architecture_availability": {name: False for name in architectures}}
    availability = {}
    for name in architectures:
        target = getattr(transformers, name, None)
        availability[name] = bool(target is not None and hasattr(target, "from_pretrained"))
    return {"available": True, "version": getattr(transformers, "__version__", None),
            "declared_architecture_availability": availability,
            "auto_processor_available": bool(getattr(transformers, "AutoProcessor", None))}


def _gpu_report() -> dict[str, Any]:
    try:
        import torch
    except Exception:
        return {"torch_available": False, "torch_version": None, "cuda_available": False, "devices": []}
    cuda = getattr(torch, "cuda", None)
    available = bool(cuda and cuda.is_available())
    devices = []
    if available:
        try:
            count = int(cuda.device_count())
            for index in range(count):
                properties = cuda.get_device_properties(index)
                devices.append({"index": index, "name": getattr(properties, "name", None),
                                "total_memory_bytes": getattr(properties, "total_memory", None)})
        except Exception:
            devices = [{"status": "CUDA_ENUMERATION_FAILED"}]
    return {"torch_available": True, "torch_version": getattr(torch, "__version__", None),
            "cuda_available": available, "devices": devices}


def _processor_probe(
    model_path: str | Path,
    candidates: list[dict[str, Any]],
    expected_processor_classes: list[str],
    trust_remote_code: bool,
) -> dict[str, Any]:
    """Instantiate only AutoProcessor, never a model or checkpoint weight."""
    try:
        import transformers
        auto_processor = getattr(transformers, "AutoProcessor", None)
        if auto_processor is None or not hasattr(auto_processor, "from_pretrained"):
            return {"status": "AUTOPROCESSOR_UNAVAILABLE", "trust_remote_code": trust_remote_code}
        processor = auto_processor.from_pretrained(
            str(Path(model_path).expanduser().resolve()), trust_remote_code=trust_remote_code,
            local_files_only=True,
        )
        template = getattr(processor, "chat_template", None)
        if not isinstance(template, str):
            template = getattr(getattr(processor, "tokenizer", None), "chat_template", None)
        template_sha256 = hashlib.sha256(template.encode("utf-8")).hexdigest() if isinstance(template, str) else None
        matching_sources = [candidate["source"] for candidate in candidates if candidate["sha256"] == template_sha256]
        actual_class = processor.__class__.__name__
        class_matches = actual_class in expected_processor_classes
        return {
            "status": "PASS" if class_matches and bool(matching_sources) else "PROCESSOR_STATIC_CONTRACT_MISMATCH",
            "trust_remote_code": trust_remote_code,
            "actual_processor_class": actual_class,
            "declared_processor_classes": expected_processor_classes,
            "processor_class_matches_static_declaration": class_matches,
            "actual_chat_template_sha256": template_sha256,
            "matching_static_template_sources": matching_sources,
            "model_weights_loaded": False,
        }
    except Exception:
        return {"status": "PROCESSOR_PROBE_FAILED", "trust_remote_code": trust_remote_code,
                "model_weights_loaded": False}


def _probe_messages(prompt: str, images: list[Image.Image], layout: str) -> list[dict[str, Any]]:
    """Build the exact image-message representation used by ``LocalHFBackend``.

    This intentionally probes two explicit content orders rather than deriving
    a Qwen-specific layout from a class name or template filename.
    """
    image_content = [{"type": "image", "image": image} for image in images]
    text_content = {"type": "text", "text": prompt}
    content = image_content + [text_content] if layout == "images_then_text" else [text_content] + image_content
    return [{"role": "user", "content": content}]


def _probe_images() -> list[Image.Image]:
    """Create two non-medical images with distinct Qwen-compatible grids.

    They exist only in memory.  Their dimensions are multiples of 28, avoiding
    an accidental conclusion caused by a vision processor's patch-size check.
    """
    first = Image.new("RGB", (56, 56), color=(223, 51, 47))
    second = Image.new("RGB", (84, 56), color=(35, 99, 208))
    return [first, second]


def _image_identity(image: Image.Image) -> dict[str, Any]:
    payload = image.tobytes()
    return {
        "mode": image.mode,
        "width": image.width,
        "height": image.height,
        "rgb_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _value_summary(value: Any) -> dict[str, Any]:
    """Describe processor output without serializing tokens or pixel values."""
    result: dict[str, Any] = {"type": type(value).__name__}
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            result["shape"] = [int(item) for item in shape]
        except (TypeError, ValueError):
            result["shape"] = None
    dtype = getattr(value, "dtype", None)
    if dtype is not None:
        result["dtype"] = str(dtype)
    device = getattr(value, "device", None)
    if device is not None:
        result["device"] = str(device)
    try:
        result["numel"] = int(value.numel())
    except (AttributeError, TypeError, ValueError):
        try:
            result["numel"] = len(value)
        except TypeError:
            result["numel"] = None
    # Processor outputs are expected to be CPU tensors.  Do not call .to(),
    # .cpu(), or materialize arbitrary values merely to report a digest.
    try:
        raw = value.detach().contiguous().numpy().tobytes()
    except (AttributeError, RuntimeError, TypeError, ValueError):
        try:
            raw = _canonical_json_bytes(value.tolist())
        except (AttributeError, TypeError, ValueError):
            raw = None
    result["value_sha256"] = hashlib.sha256(raw).hexdigest() if raw is not None else None
    return result


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _grid_rows(value: Any) -> list[list[int]] | None:
    try:
        raw = value.tolist()
    except AttributeError:
        raw = value
    if not isinstance(raw, list) or any(not isinstance(row, list) for row in raw):
        return None
    rows: list[list[int]] = []
    for row in raw:
        if any(type(item) is not int for item in row):
            return None
        rows.append(list(row))
    return rows


def _nonempty(value: Any) -> bool:
    try:
        return int(value.numel()) > 0
    except (AttributeError, TypeError, ValueError):
        try:
            return len(value) > 0
        except TypeError:
            return False


def _safe_probe_error(exc: Exception) -> dict[str, Any]:
    """Keep diagnostics reviewable without leaking template or image contents."""
    message = str(exc)
    return {
        "error_type": type(exc).__name__,
        "error_message_sha256": hashlib.sha256(message.encode("utf-8")).hexdigest(),
    }


def _processor_inputs_for_contract(
    processor: Any,
    *,
    prompt: str,
    images: list[Image.Image],
    layout: str,
    call_mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    messages = _probe_messages(prompt, images, layout)
    details: dict[str, Any] = {
        "chat_message_layout": layout,
        "processor_call_mode": call_mode,
        "content_types": [item["type"] for item in messages[0]["content"]],
        "add_generation_prompt": True,
        "return_tensors": "pt",
    }
    if call_mode == "tokenized_chat_template":
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        details.update({"tokenize": True, "return_dict": True})
    else:
        rendered = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if not isinstance(rendered, str):
            raise TypeError("processor chat template did not render text")
        details.update({
            "tokenize": False,
            "rendered_prompt_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
            "rendered_prompt_length": len(rendered),
        })
        inputs = processor(text=[rendered], images=images, return_tensors="pt")
    if not isinstance(inputs, dict) and not hasattr(inputs, "items"):
        raise TypeError("processor did not return a mapping")
    return dict(inputs.items()), details


def _candidate_contract(
    processor: Any,
    *,
    prompt: str,
    images: list[Image.Image],
    layout: str,
    call_mode: str,
) -> dict[str, Any]:
    try:
        base, details = _processor_inputs_for_contract(
            processor, prompt=prompt, images=images, layout=layout, call_mode=call_mode,
        )
        swapped, swapped_details = _processor_inputs_for_contract(
            processor, prompt=prompt, images=list(reversed(images)), layout=layout, call_mode=call_mode,
        )
    except Exception as exc:
        return {
            "status": "PROCESSOR_IMAGE_CONTRACT_FAILED",
            "chat_message_layout": layout,
            "processor_call_mode": call_mode,
            **_safe_probe_error(exc),
        }
    base_grid = _grid_rows(base.get("image_grid_thw"))
    swapped_grid = _grid_rows(swapped.get("image_grid_thw"))
    base_pixels = base.get("pixel_values")
    swapped_pixels = swapped.get("pixel_values")
    conditions = {
        "input_ids_present": "input_ids" in base and _nonempty(base["input_ids"]),
        "pixel_values_present": base_pixels is not None and _nonempty(base_pixels),
        "image_grid_thw_present": base_grid is not None,
        "two_image_grid_rows": base_grid is not None and len(base_grid) == 2,
        "no_video_fallback": "pixel_values_videos" not in base and "video_grid_thw" not in base,
        "input_keys_stable_after_swap": set(base) == set(swapped),
        "grid_order_preserved_after_swap": (
            base_grid is not None and swapped_grid is not None and swapped_grid == list(reversed(base_grid))
        ),
        "pixel_values_change_after_swap": (
            base_pixels is not None and swapped_pixels is not None
            and _value_summary(base_pixels)["value_sha256"] is not None
            and _value_summary(base_pixels)["value_sha256"] != _value_summary(swapped_pixels)["value_sha256"]
        ),
    }
    base_summary = {key: _value_summary(value) for key, value in sorted(base.items())}
    swapped_summary = {key: _value_summary(value) for key, value in sorted(swapped.items())}
    return {
        "status": "PASS" if all(conditions.values()) else "PROCESSOR_IMAGE_CONTRACT_FAILED",
        **details,
        "swapped_rendered_prompt_sha256": swapped_details.get("rendered_prompt_sha256"),
        "base_inputs": {"keys": sorted(base), "values": base_summary, "image_grid_thw_rows": base_grid},
        "swapped_inputs": {"keys": sorted(swapped), "values": swapped_summary, "image_grid_thw_rows": swapped_grid},
        "conditions": conditions,
    }


def _processor_image_contract_probe(
    model_path: str | Path,
    candidates: list[dict[str, Any]],
    expected_processor_classes: list[str],
    trust_remote_code: bool,
) -> dict[str, Any]:
    """Exercise processor-only visual input contracts with synthetic images.

    No checkpoint weight is opened and no model class is imported or called.
    The report presents four explicit candidates; it never chooses one for a
    future model configuration.
    """
    try:
        import transformers
        auto_processor = getattr(transformers, "AutoProcessor", None)
        if auto_processor is None or not hasattr(auto_processor, "from_pretrained"):
            return {"status": "AUTOPROCESSOR_UNAVAILABLE", "model_weights_loaded": False}
        processor = auto_processor.from_pretrained(
            str(Path(model_path).expanduser().resolve()), trust_remote_code=trust_remote_code,
            local_files_only=True,
        )
        template = _processor_template(processor)
        template_sha256 = hashlib.sha256(template.encode("utf-8")).hexdigest() if isinstance(template, str) else None
        matching_sources = [candidate["source"] for candidate in candidates if candidate["sha256"] == template_sha256]
        actual_class = processor.__class__.__name__
        class_matches = actual_class in expected_processor_classes
        images = _probe_images()
        prompt = "ReliVE processor image-contract probe."
        candidate_reports = [
            _candidate_contract(processor, prompt=prompt, images=images, layout=layout, call_mode=call_mode)
            for layout in ("images_then_text", "text_then_images")
            for call_mode in ("tokenized_chat_template", "render_then_process")
        ]
        return {
            "status": (
                "PASS" if class_matches and bool(matching_sources)
                and any(item["status"] == "PASS" for item in candidate_reports) else "FAIL"
            ),
            "model_weights_loaded": False,
            "model_from_pretrained_calls": 0,
            "generate_calls": 0,
            "tensor_device_moves": 0,
            "synthetic_nonmedical_images": True,
            "trust_remote_code": trust_remote_code,
            "local_files_only": True,
            "actual_processor_class": actual_class,
            "declared_processor_classes": expected_processor_classes,
            "processor_class_matches_static_declaration": class_matches,
            "actual_chat_template_sha256": template_sha256,
            "matching_static_template_sources": matching_sources,
            "probe_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "probe_images": [_image_identity(image) for image in images],
            "candidates": candidate_reports,
            "selection_policy": "No candidate is selected automatically; copy one reviewed PASS contract into local_hf config.",
        }
    except Exception as exc:
        return {
            "status": "PROCESSOR_IMAGE_CONTRACT_PROBE_FAILED",
            "model_weights_loaded": False,
            "model_from_pretrained_calls": 0,
            "generate_calls": 0,
            "tensor_device_moves": 0,
            "trust_remote_code": trust_remote_code,
            **_safe_probe_error(exc),
        }


def _processor_template(processor: Any) -> str | None:
    template = getattr(processor, "chat_template", None)
    if isinstance(template, str):
        return template
    tokenizer = getattr(processor, "tokenizer", None)
    template = getattr(tokenizer, "chat_template", None)
    return template if isinstance(template, str) else None


def inspect_local_hf(model_path: str | Path, *, probe_processor: bool = False,
                     probe_processor_images: bool = False, trust_remote_code: bool = False) -> dict[str, Any]:
    """Inspect checkpoint metadata; optional probes load only ``AutoProcessor``.

    With no probe flags this never calls ``from_pretrained``. Both optional
    probes use local processor files only and never instantiate a model or load
    its weights, making them safe gates before a local backend configuration is
    written.
    """
    metadata = checkpoint_metadata(model_path)
    candidates = metadata["chat_template_candidates"]
    declared_processor_classes = [
        declaration["value"] for declaration in metadata["processor_declarations"]
        if declaration["source"].endswith(":processor_class")
    ]
    distinct = {candidate["sha256"] for candidate in candidates}
    template_status = "MISSING" if not candidates else ("UNIQUE_CONTENT" if len(distinct) == 1 else "AMBIGUOUS_CONTENT")
    report = {
        "inspection_version": INSPECTION_VERSION,
        "inspection_mode": (
            "metadata_plus_processor_image_contract_no_model_load" if probe_processor_images
            else ("metadata_plus_processor_no_model_load" if probe_processor
                  else "metadata_only_no_processor_or_model_load")
        ),
        "model_path": str(Path(model_path).expanduser().resolve()),
        "checkpoint": metadata,
        "chat_template_status": template_status,
        "transformers": _transformers_report(metadata["architectures"]),
        "gpu": _gpu_report(),
        "next_step": (
            "Select exact model_class, processor_class, chat_template_source, chat_template_sha256, "
            "message layout, processor call mode, input device, and generation settings in local_hf config."
        ),
    }
    if probe_processor or probe_processor_images:
        report["processor_probe"] = _processor_probe(
            model_path, candidates, declared_processor_classes, trust_remote_code,
        )
    if probe_processor_images:
        report["processor_image_contract_probe"] = _processor_image_contract_probe(
            model_path, candidates, declared_processor_classes, trust_remote_code,
        )
    return report
