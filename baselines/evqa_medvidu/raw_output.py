from __future__ import annotations

import ast
import re
from typing import Any


def parse_temporal_segments(text: str | None) -> list[list[float]]:
    if not text:
        return []
    try:
        match = re.search(r"\[\s*\[.*?\]\s*\]", text, flags=re.DOTALL)
        if match:
            value = ast.literal_eval(match.group(0))
            return _coerce_segments(value)
    except Exception:
        pass
    pairs = re.findall(r"\[\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\]", text)
    return [[float(a), float(b)] for a, b in pairs]


def _coerce_segments(value: Any) -> list[list[float]]:
    out: list[list[float]] = []
    if isinstance(value, list):
        if len(value) == 2 and all(isinstance(x, (int, float)) for x in value):
            return [[float(value[0]), float(value[1])]]
        for item in value:
            if isinstance(item, dict) and "start" in item and "end" in item:
                out.append([float(item["start"]), float(item["end"])])
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                out.append([float(item[0]), float(item[1])])
    return out


def extract_answer_letter(text: str | None) -> str | None:
    if not text:
        return None
    match = re.search(r"(?:^|\b)(?:answer\s*:\s*)?([A-E])(?:\b|[.)])", text, flags=re.IGNORECASE)
    return match.group(1).upper() if match else None


def parse_cvs_components(text: str | None) -> dict[str, int]:
    if not text:
        return {}
    lower = text.lower()
    patterns = {
        "two_structures": r"two\s+structures?\s*:\s*([012])",
        "cystic_plate": r"cystic\s+plate\s*:\s*([012])",
        "hepatocystic_triangle": r"hepatocystic\s+triangle\s*:\s*([012])",
    }
    out: dict[str, int] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, lower)
        if match:
            out[key] = int(match.group(1))
    return out


def format_cvs_components(components: dict[str, int]) -> str:
    return (
        f"Two structures: {components['two_structures']}, "
        f"Cystic plate: {components['cystic_plate']}, "
        f"Hepatocystic triangle: {components['hepatocystic_triangle']}"
    )

