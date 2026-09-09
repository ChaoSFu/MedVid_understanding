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
BACKEND_FIELDS = set(DEFAULTS["backend"]) | REAL_REQUIRED | {"jpeg_quality", "image_detail", "retry_backoff_seconds"}
SECRET_FIELDS = {"api_key", "apikey", "authorization", "password", "access_token", "secret", "secret_key", "token"}


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


def validate_backend(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("backend must be an object")
    _plain(raw, "backend")
    unknown = set(raw) - BACKEND_FIELDS
    if unknown:
        raise ValueError(f"Unknown backend fields: {sorted(unknown)}")
    kind = raw.get("kind", "mock")
    if kind not in {"mock", "openai_compatible"}:
        raise ValueError("Unsupported backend kind")
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
    for name in ("generation", "extra"):
        if not isinstance(result[name], dict):
            raise ValueError(f"backend.{name} must be an object")
        if set(result[name]) & {"model", "messages", "stream", "headers", "extra_headers", "base_url"}:
            raise ValueError(f"backend.{name} cannot override transport identity or messages")
    if set(result["generation"]) & set(result["extra"]):
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
