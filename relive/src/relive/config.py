"""Validated, explicit run configuration. Defaults are engineering presets only."""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlsplit


DEFAULTS = {
    "framework_version": "ReliVE-v1",
    "backend": {"kind": "mock", "model": "relive-mock", "revision": "v1", "rules": [],
                "generation": {}, "image_order": "chronological", "frame_encoding": "png",
                "timeout_seconds": 30, "max_retries": 0, "extra": {}},
    "acquisition": {"method": "sliding_windows", "window_size": 3, "stride": 2, "max_candidates": 4},
    "claims": {"max_claims": 3, "max_alternatives": 2, "contrast_fixtures": []},
    "spatial": {"blur_radius": 4.0, "control_count": 1, "control_aggregation": "all",
                "control_version": "relive-matched-corners-edges-v1", "alternate_regions": [[0.1, 0.1, 0.7, 0.7]],
                "max_area_warning": 0.8},
    "policy": {"name": "semantic_contrast_spatial", "version": "relive-v1-policy-1", "strict_alternatives": True},
    "adaptation": {"enabled": True, "actions": ["EXPAND_TEMPORAL_CONTEXT", "TRY_ALTERNATE_SUPPORT_REGION",
                    "NEXT_CANDIDATE", "ACQUIRE_MISSING_CLAIM"], "expand_frames": 2},
    "budget": {"max_calls": 60, "max_candidates": 6, "max_spatial_proposals": 2, "max_rounds": 4},
    "reasoning": {"mode": "strict_reliability", "max_answer_tokens": 128},
}

REAL_REQUIRED = {"kind", "base_url", "endpoint_path", "model", "revision", "credential_env",
                 "generation", "image_order", "frame_encoding", "timeout_seconds", "max_retries", "extra"}
LOCAL_HF_REQUIRED = {
    "kind", "model", "revision", "model_path", "checkpoint_metadata_sha256", "model_class",
    "processor_class", "chat_template_source", "chat_template_sha256", "chat_message_layout",
    "processor_call_mode", "chat_template_kwargs", "trust_remote_code", "local_files_only", "dtype",
    "device", "device_map", "input_device", "max_memory", "processor_min_pixels",
    "processor_max_pixels", "generation", "image_order", "frame_encoding", "timeout_seconds",
    "max_retries",
}
MOCK_FIELDS = set(DEFAULTS["backend"]) | {"jpeg_quality", "image_detail", "retry_backoff_seconds"}
OPENAI_FIELDS = REAL_REQUIRED | {"jpeg_quality", "image_detail", "retry_backoff_seconds"}
LOCAL_HF_FIELDS = LOCAL_HF_REQUIRED | {"jpeg_quality", "retry_backoff_seconds"}
SECRET_FIELDS = {"api_key", "apikey", "authorization", "password", "access_token", "secret", "secret_key", "token"}
_LOCAL_GENERATION_FIELDS = {
    "do_sample", "max_new_tokens", "min_new_tokens", "temperature", "top_p", "top_k",
    "typical_p", "repetition_penalty", "length_penalty", "num_beams", "use_cache",
}
_LOCAL_DEVICE_MAPS = {"auto", "balanced", "balanced_low_0", "sequential"}
_LOCAL_DTYPES = {"bfloat16", "float16", "float32"}


def _plain(value: Any, where: str = "config") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{where}: keys must be strings")
            if key.lower() in SECRET_FIELDS:
                raise ValueError(f"{where}: credentials must use credential_env, never literal secret fields")
            _plain(item, f"{where}.{key}")
    elif isinstance(value, list):
        for item in value:
            _plain(item, where)
    elif value is not None and not isinstance(value, (str, bool, int, float)):
        raise ValueError(f"{where}: only plain JSON values are allowed")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{where}: nonfinite numbers are forbidden")


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _number(value, name, minimum=0, maximum=None):
    if type(value) not in {int, float} or not math.isfinite(value) or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"Invalid numeric value for {name}")


def _sha256(value: Any, name: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


def _local_device(value: Any, name: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"(?:cuda(?::[0-9]+)?|cpu|mps)", value):
        raise ValueError(f"{name} must be an explicit cuda:N, cpu, or mps device")


def _local_max_memory(value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, dict) or not value:
        raise ValueError("backend.max_memory must be null or a nonempty object")
    for device, limit in value.items():
        if not isinstance(device, str) or not re.fullmatch(r"(?:cuda:[0-9]+|[0-9]+|cpu|disk)", device):
            raise ValueError("backend.max_memory has an unsupported device key")
        if not isinstance(limit, str) or not re.fullmatch(r"[1-9][0-9]*(?:MiB|GiB|MB|GB)", limit):
            raise ValueError("backend.max_memory limits must be explicit positive memory strings")


def _validate_local_generation(value: Any) -> None:
    if not isinstance(value, dict) or not value:
        raise ValueError("backend.generation must be a nonempty object for local_hf")
    unknown = set(value) - _LOCAL_GENERATION_FIELDS
    if unknown:
        raise ValueError(f"Unsupported local_hf generation fields: {sorted(unknown)}")
    if set(value) & {"model", "input_ids", "pixel_values", "stream", "messages", "return_dict"}:
        raise ValueError("backend.generation cannot override model inputs")
    if type(value.get("do_sample")) is not bool:
        raise ValueError("local_hf generation.do_sample must be explicit boolean")
    _integer(value.get("max_new_tokens"), "local_hf generation.max_new_tokens")
    if "min_new_tokens" in value:
        _integer(value["min_new_tokens"], "local_hf generation.min_new_tokens", minimum=0)
        if value["min_new_tokens"] > value["max_new_tokens"]:
            raise ValueError("local_hf generation.min_new_tokens cannot exceed max_new_tokens")
    for key in ("temperature", "top_p", "typical_p", "repetition_penalty", "length_penalty"):
        if key in value:
            _number(value[key], f"local_hf generation.{key}", minimum=0)
    for key in ("top_k", "num_beams"):
        if key in value:
            _integer(value[key], f"local_hf generation.{key}", minimum=1)
    if "use_cache" in value and type(value["use_cache"]) is not bool:
        raise ValueError("local_hf generation.use_cache must be boolean")


def _validate_local_hf(raw: dict[str, Any]) -> dict[str, Any]:
    missing = LOCAL_HF_REQUIRED - set(raw)
    if missing:
        raise ValueError(f"local_hf backend requires explicit settings: {sorted(missing)}")
    prohibited = set(raw) & {"rules", "base_url", "endpoint_path", "credential_env", "extra", "image_detail"}
    if prohibited:
        raise ValueError(f"local_hf backend cannot contain transport, fixture, or opaque extra fields: {sorted(prohibited)}")
    result = deepcopy(raw)
    for name in ("model", "revision", "model_class", "processor_class", "chat_template_source"):
        if not isinstance(result[name], str) or not result[name].strip():
            raise ValueError(f"backend.{name} must be explicit and nonempty")
    if not isinstance(result["model_path"], str) or not Path(result["model_path"]).is_absolute():
        raise ValueError("backend.model_path must be an explicit absolute checkpoint directory")
    _sha256(result["checkpoint_metadata_sha256"], "backend.checkpoint_metadata_sha256")
    _sha256(result["chat_template_sha256"], "backend.chat_template_sha256")
    if result["chat_message_layout"] not in {"images_then_text", "text_then_images"}:
        raise ValueError("backend.chat_message_layout must be images_then_text or text_then_images")
    if result["processor_call_mode"] not in {"render_then_process", "tokenized_chat_template"}:
        raise ValueError("backend.processor_call_mode is unsupported")
    if not isinstance(result["chat_template_kwargs"], dict):
        raise ValueError("backend.chat_template_kwargs must be an explicit object")
    reserved_template = {"messages", "tokenize", "add_generation_prompt", "return_dict", "return_tensors", "images", "text"}
    if set(result["chat_template_kwargs"]) & reserved_template:
        raise ValueError("backend.chat_template_kwargs cannot override the fixed adapter call contract")
    if type(result["trust_remote_code"]) is not bool:
        raise ValueError("backend.trust_remote_code must be explicit boolean")
    if result["local_files_only"] is not True:
        raise ValueError("backend.local_files_only must be true")
    if result["dtype"] not in _LOCAL_DTYPES:
        raise ValueError("backend.dtype must be bfloat16, float16, or float32")
    _local_device(result["device"], "backend.device")
    _local_device(result["input_device"], "backend.input_device")
    device_map = result["device_map"]
    if device_map is not None and not isinstance(device_map, dict) and device_map not in _LOCAL_DEVICE_MAPS:
        raise ValueError("backend.device_map must be null, an explicit supported placement policy, or a placement object")
    if isinstance(device_map, dict):
        if not device_map:
            raise ValueError("backend.device_map object cannot be empty")
        for module, placement in device_map.items():
            if not isinstance(module, str) or not module or not re.fullmatch(r"[A-Za-z0-9_.-]+", module):
                raise ValueError("backend.device_map module keys must be explicit model paths")
            if type(placement) is int:
                if placement < 0:
                    raise ValueError("backend.device_map numeric placements must be nonnegative")
            elif placement not in {"cpu", "disk"} and not (isinstance(placement, str) and re.fullmatch(r"cuda:[0-9]+", placement)):
                raise ValueError("backend.device_map has an unsupported placement")
    _local_max_memory(result["max_memory"])
    if result["device_map"] is None and result["max_memory"] is not None:
        raise ValueError("backend.max_memory requires an explicit device_map")
    for name in ("processor_min_pixels", "processor_max_pixels"):
        if result[name] is not None:
            _integer(result[name], f"backend.{name}")
    if (result["processor_min_pixels"] is not None and result["processor_max_pixels"] is not None
            and result["processor_min_pixels"] > result["processor_max_pixels"]):
        raise ValueError("processor_min_pixels cannot exceed processor_max_pixels")
    _validate_local_generation(result["generation"])
    if result["frame_encoding"] != "jpeg" and "jpeg_quality" in result:
        raise ValueError("backend.jpeg_quality is valid only when frame_encoding is jpeg")
    return result


def validate_backend(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("backend must be an object")
    _plain(raw, "backend")
    kind = raw.get("kind", "mock")
    if kind not in {"mock", "openai_compatible", "local_hf"}:
        raise ValueError("Unsupported backend kind")
    allowed = {"mock": MOCK_FIELDS, "openai_compatible": OPENAI_FIELDS,
               "local_hf": LOCAL_HF_FIELDS}[kind]
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown or inapplicable backend fields for {kind}: {sorted(unknown)}")
    if kind == "openai_compatible":
        missing = REAL_REQUIRED - set(raw)
        if missing:
            raise ValueError(f"Real backend requires explicit settings: {sorted(missing)}")
        result = deepcopy(raw)
        if "rules" in raw:
            raise ValueError("Synthetic fixture rules cannot be configured for real inference")
        parsed = urlsplit(raw["base_url"]) if isinstance(raw["base_url"], str) else None
        if not parsed or parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("base_url must be explicit HTTP(S), without credentials, query or fragment")
        if not isinstance(raw["endpoint_path"], str) or not re.fullmatch(r"/[A-Za-z0-9_/-]+", raw["endpoint_path"]) or ".." in raw["endpoint_path"]:
            raise ValueError("endpoint_path must be an explicit relative API route, e.g. /chat/completions")
        if not isinstance(raw["credential_env"], str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", raw["credential_env"]):
            raise ValueError("credential_env must name an environment variable")
    elif kind == "local_hf":
        result = _validate_local_hf(raw)
    else:
        if set(raw) & {"base_url", "endpoint_path", "credential_env"}:
            raise ValueError("Mock backend must not contain endpoint or credential configuration")
        result = deepcopy(DEFAULTS["backend"])
        result.update(deepcopy(raw))
    for name in ("model", "revision"):
        if not isinstance(result[name], str) or not result[name].strip():
            raise ValueError(f"backend.{name} must be explicit and nonempty")
    if result["image_order"] != "chronological":
        raise ValueError("Only chronological image ordering is implemented; caller preserves original frame order")
    if result["frame_encoding"] not in {"png", "jpeg", "source"}:
        raise ValueError("frame_encoding must be png, jpeg or source")
    if result["frame_encoding"] == "jpeg":
        if "jpeg_quality" not in result:
            raise ValueError("JPEG encoding requires explicit jpeg_quality")
        _integer(result["jpeg_quality"], "jpeg_quality")
        if result["jpeg_quality"] > 100:
            raise ValueError("jpeg_quality must be <= 100")
    if "image_detail" in result and result["image_detail"] not in {"auto", "low", "high"}:
        raise ValueError("image_detail must be auto, low or high")
    _number(result["timeout_seconds"], "timeout_seconds", minimum=0.001)
    _integer(result["max_retries"], "max_retries", minimum=0)
    if result["max_retries"] > 3:
        raise ValueError("At most three technical retries are supported")
    if "retry_backoff_seconds" in result:
        _number(result["retry_backoff_seconds"], "retry_backoff_seconds", maximum=10)
    generic_objects = ("generation", "extra") if kind != "local_hf" else ("generation",)
    for name in generic_objects:
        if not isinstance(result[name], dict):
            raise ValueError(f"backend.{name} must be an object")
        if set(result[name]) & {"model", "messages", "stream", "headers", "extra_headers", "base_url"}:
            raise ValueError(f"backend.{name} cannot override transport identity or messages")
    if kind != "local_hf" and set(result["generation"]) & set(result["extra"]):
        raise ValueError("generation and extra cannot override one another")
    if kind == "mock":
        if not isinstance(result["rules"], list):
            raise ValueError("backend.rules must be a list")
        for rule in result["rules"]:
            if not isinstance(rule, dict) or set(rule) - {"match", "response", "error", "retryable"} or not isinstance(rule.get("match"), dict):
                raise ValueError("Mock rules require match and response or error")
            if ("response" in rule) == ("error" in rule):
                raise ValueError("Each mock rule needs exactly one response or error")
            if "retryable" in rule and type(rule["retryable"]) is not bool:
                raise ValueError("Mock retryable must be boolean")
    return result


def validate_config(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("Config must be an object")
    _plain(raw)
    if set(raw) - set(DEFAULTS):
        raise ValueError(f"Unknown config sections: {sorted(set(raw) - set(DEFAULTS))}")
    result = deepcopy(DEFAULTS)
    for section, supplied in raw.items():
        if section == "backend":
            continue
        if isinstance(DEFAULTS[section], dict):
            if not isinstance(supplied, dict) or set(supplied) - set(DEFAULTS[section]):
                raise ValueError(f"Invalid or unknown fields in {section}")
            result[section].update(deepcopy(supplied))
        else:
            result[section] = supplied
    result["backend"] = validate_backend(raw.get("backend", {}))
    if result["framework_version"] != "ReliVE-v1":
        raise ValueError("Only ReliVE-v1 is implemented")
    if result["acquisition"]["method"] not in {"uniform", "sliding_windows"}:
        raise ValueError("Unsupported acquisition method")
    for section, names in {
        "acquisition": ("window_size", "stride", "max_candidates"),
        "claims": ("max_claims",),
        "budget": ("max_calls", "max_candidates", "max_spatial_proposals", "max_rounds"),
        "reasoning": ("max_answer_tokens",),
    }.items():
        for name in names:
            _integer(result[section][name], f"{section}.{name}")
    _integer(result["claims"]["max_alternatives"], "max_alternatives", minimum=0)
    fixtures = result["claims"]["contrast_fixtures"]
    if not isinstance(fixtures, list):
        raise ValueError("claims.contrast_fixtures must be a list")
    if fixtures and result["backend"]["kind"] != "mock":
        raise ValueError("Human-authored synthetic contrast fixtures require the mock backend")
    seen_fixtures = set()
    for fixture in fixtures:
        required = {"sample_id", "target_text", "alternatives", "comparison_dimension",
                    "exclusivity_status", "source"}
        if not isinstance(fixture, dict) or set(fixture) != required:
            raise ValueError("Synthetic contrast fixtures require exactly the declared fixture fields")
        for name in ("sample_id", "target_text", "comparison_dimension"):
            if not isinstance(fixture[name], str) or not fixture[name].strip():
                raise ValueError(f"contrast fixture {name} must be nonempty text")
        if fixture["source"] != "human_authored_synthetic":
            raise ValueError("Contrast fixture source must be human_authored_synthetic")
        if fixture["exclusivity_status"] not in {"DECLARED_EXCLUSIVE", "NONEXCLUSIVE", "UNRESOLVED"}:
            raise ValueError("Invalid fixture exclusivity_status")
        alternatives = fixture["alternatives"]
        if not isinstance(alternatives, list) or not alternatives or len(alternatives) > result["claims"]["max_alternatives"]:
            raise ValueError("Contrast fixtures require alternatives within max_alternatives")
        if any(not isinstance(text, str) or not text.strip() for text in alternatives):
            raise ValueError("Contrast fixture alternatives must be nonempty text")
        normalized = [fixture["target_text"].strip(), *(text.strip() for text in alternatives)]
        if len(set(normalized)) != len(normalized):
            raise ValueError("Contrast fixture alternatives must be distinct from the target and one another")
        binding = (fixture["sample_id"], fixture["target_text"])
        if binding in seen_fixtures:
            raise ValueError("Duplicate sample and target contrast fixture")
        seen_fixtures.add(binding)
    _integer(result["adaptation"]["expand_frames"], "expand_frames")
    _integer(result["spatial"]["control_count"], "control_count")
    _number(result["spatial"]["blur_radius"], "blur_radius", minimum=0.001)
    _number(result["spatial"]["max_area_warning"], "max_area_warning", minimum=0, maximum=1)
    if result["spatial"]["control_aggregation"] != "all":
        raise ValueError("Only the predeclared all-controls aggregation is supported")
    for name in ("control_version",):
        if not isinstance(result["spatial"][name], str) or not result["spatial"][name]:
            raise ValueError(f"spatial.{name} must be nonempty")
    regions = result["spatial"]["alternate_regions"]
    if not isinstance(regions, list):
        raise ValueError("alternate_regions must be a list")
    for box in regions:
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError("alternate_regions require normalized xyxy rectangles")
        for coordinate in box:
            _number(coordinate, "alternate_regions coordinate", minimum=0, maximum=1)
        if box[0] >= box[2] or box[1] >= box[3]:
            raise ValueError("alternate_regions rectangles must have positive area")
    if result["policy"]["name"] not in {"acquisition_only", "semantic_only", "semantic_keep_drop", "semantic_spatial", "semantic_contrast_spatial"}:
        raise ValueError("Unsupported certificate policy")
    if (result["backend"]["kind"] != "mock"
            and result["policy"]["name"] == "semantic_contrast_spatial"):
        raise ValueError(
            "semantic_contrast_spatial on a real backend requires a task-declared, "
            "public-option, or ontology-declared exclusivity source with provenance; "
            "ReliVE-v1 has no such real-runtime adapter. Use semantic_spatial."
        )
    if result["policy"]["version"] != "relive-v1-policy-1":
        raise ValueError("Only policy version relive-v1-policy-1 is implemented")
    for section, key in (("adaptation", "enabled"), ("policy", "strict_alternatives")):
        if type(result[section][key]) is not bool:
            raise ValueError(f"{section}.{key} must be boolean")
    actions = result["adaptation"]["actions"]
    if not isinstance(actions, list) or any(not isinstance(action, str) for action in actions) or len(set(actions)) != len(actions):
        raise ValueError("adaptation.actions must contain unique action names")
    supported_actions = {"NEXT_CANDIDATE", "EXPAND_TEMPORAL_CONTEXT", "TRY_ALTERNATE_SUPPORT_REGION",
                         "ACQUIRE_MISSING_CLAIM"}
    unknown_actions = sorted(set(actions) - supported_actions)
    if unknown_actions:
        raise ValueError(f"UNSUPPORTED_ACTION in adaptation.actions: {unknown_actions}")
    if result["reasoning"]["mode"] not in {"strict_reliability", "benchmark_forced"}:
        raise ValueError("Unsupported answer mode")
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    elif path.suffix.lower() in {".yaml", ".yml"}:
        import yaml
        data = yaml.safe_load(text)
    else:
        raise ValueError("Configuration must be .json, .yaml or .yml")
    return validate_config(data)
