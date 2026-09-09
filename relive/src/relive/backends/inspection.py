"""Read-only inspection for local Hugging Face checkpoints.

The inspector deliberately reads only small checkpoint metadata files.  It does
not instantiate a processor or model, load weights, execute remote code, or
construct a prompt.  That keeps the inspection useful for deciding an explicit
ReliVE adapter configuration without making assumptions about a checkpoint
whose architecture or chat template has not been reviewed yet.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


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


def _processor_probe(model_path: str | Path, candidates: list[dict[str, Any]], trust_remote_code: bool) -> dict[str, Any]:
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
        return {
            "status": "PASS", "trust_remote_code": trust_remote_code,
            "actual_processor_class": processor.__class__.__name__,
            "actual_chat_template_sha256": template_sha256,
            "matching_static_template_sources": [candidate["source"] for candidate in candidates
                                                 if candidate["sha256"] == template_sha256],
            "model_weights_loaded": False,
        }
    except Exception:
        return {"status": "PROCESSOR_PROBE_FAILED", "trust_remote_code": trust_remote_code,
                "model_weights_loaded": False}


def inspect_local_hf(model_path: str | Path, *, probe_processor: bool = False,
                     trust_remote_code: bool = False) -> dict[str, Any]:
    """Inspect static checkpoint metadata and installed runtime compatibility.

    This function never calls ``from_pretrained``.  It is safe to use as the
    first GPU-host command before the local backend configuration is written.
    """
    metadata = checkpoint_metadata(model_path)
    candidates = metadata["chat_template_candidates"]
    distinct = {candidate["sha256"] for candidate in candidates}
    template_status = "MISSING" if not candidates else ("UNIQUE_CONTENT" if len(distinct) == 1 else "AMBIGUOUS_CONTENT")
    report = {
        "inspection_version": INSPECTION_VERSION,
        "inspection_mode": ("metadata_plus_processor_no_model_load" if probe_processor
                            else "metadata_only_no_processor_or_model_load"),
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
    if probe_processor:
        report["processor_probe"] = _processor_probe(model_path, candidates, trust_remote_code)
    return report
