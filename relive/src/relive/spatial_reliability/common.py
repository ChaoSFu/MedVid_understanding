from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable


class SpatialReliabilityError(ValueError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json_load(path: str | Path) -> Any:
    def reject_constant(value: str) -> None:
        raise SpatialReliabilityError(f"NONFINITE_JSON_CONSTANT:{value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SpatialReliabilityError(f"DUPLICATE_JSON_KEY:{key}")
            result[key] = value
        return result

    try:
        text = Path(path).read_bytes().decode("utf-8")
        return json.loads(text, parse_constant=reject_constant, object_pairs_hook=reject_duplicates)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SpatialReliabilityError(f"INVALID_JSON:{path}") from exc


def write_once(path: str | Path, payload: Any, *, jsonl: bool = False) -> Path:
    target = Path(path)
    if jsonl:
        body = b"".join(canonical_bytes(row) + b"\n" for row in payload)
    else:
        body = canonical_bytes(payload) + b"\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() != body:
            raise SpatialReliabilityError(f"IMMUTABLE_ARTIFACT_EXISTS:{target}")
        return target
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
    finally:
        os.unlink(temporary)
    return target


def runtime_gt_audit(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    forbidden = ("gt_bbox", "iou", "ground_truth", "oracle", "gt_alignment", "rc_info", "bbox")
    issues: list[dict[str, Any]] = []

    def walk(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                folded = key.casefold()
                if any(token in folded for token in forbidden):
                    issues.append({"path": f"{path}.{key}", "reason": "FORBIDDEN_SPATIAL_GT_FIELD"})
                walk(item, f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")

    rows = list(rows)
    for index, row in enumerate(rows):
        walk(row, f"rows[{index}]")
    return {
        "format": "relive-spatial-runtime-gt-isolation-v1",
        "status": "PASS" if not issues else "FAIL",
        "rows_checked": len(rows),
        "issues": issues,
        "gt_used": False,
        "oracle_manifest_opened": False,
    }
