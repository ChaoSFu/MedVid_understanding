"""Shared strict JSON parsing for model-facing protocol messages.

The parser deliberately accepts only a JSON document, optionally enclosed by
one complete JSON Markdown fence.  This keeps rendering conveniences separate
from the closed schemas used by the semantic, claim, contrast, and spatial
protocols.
"""
from __future__ import annotations

import json
from typing import Any


def strict_json(raw: str) -> Any:
    """Parse one closed JSON response without ambiguous or non-finite values."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"DUPLICATE_JSON_KEY: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"NONFINITE_JSON_CONSTANT: {value}")

    if not isinstance(raw, str):
        raise ValueError("RESPONSE_MUST_BE_TEXT")
    payload = raw.strip()
    if payload.startswith("```"):
        opening, separator, remainder = payload.partition("\n")
        if opening not in {"```", "```json"} or not separator or not remainder.endswith("\n```"):
            raise ValueError("INCOMPLETE_OR_UNSUPPORTED_JSON_FENCE")
        payload = remainder[:-4].strip()
    return json.loads(payload, object_pairs_hook=pairs, parse_constant=constant)
