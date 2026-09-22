"""Strict data-split manifest validation for protocol-first ReliVE-v2."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from relive.storage.artifacts import canonical_json, stable_hash
from .protocol import PROTOCOL_VERSION, ProtocolContractError, ProtocolManifest
from .task_selection import TALSelectionError, strict_jsonl

MANIFEST_FILENAMES = {
    "protocol_dev": "protocol_dev_manifest.jsonl",
    "calibration": "calibration_manifest.jsonl",
    "blind_test": "blind_test_manifest.jsonl",
}


class ProtocolManifestError(ValueError):
    pass


def sha256_path(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load(path: str | Path, split: str) -> list[ProtocolManifest]:
    try:
        rows = strict_jsonl(Path(path), error_code="PROTOCOL_SPLIT_MANIFEST")
    except (OSError, TALSelectionError) as exc:
        raise ProtocolManifestError("PROTOCOL_SPLIT_MANIFEST_INVALID") from exc
    manifests: list[ProtocolManifest] = []
    required = {"protocol_version", "program_sha256", "split", "source_video_sha256", "frame_sha256s", "mode"}
    for row in rows:
        if set(row) != required or row.get("split") != split:
            raise ProtocolManifestError("PROTOCOL_SPLIT_MANIFEST_SCHEMA_INVALID")
        try:
            manifests.append(ProtocolManifest(**row))
        except (TypeError, ProtocolContractError) as exc:
            raise ProtocolManifestError("PROTOCOL_SPLIT_MANIFEST_VALUE_INVALID") from exc
    return manifests


def validate_split_manifests(*, protocol_dev: str | Path, calibration: str | Path,
                             blind_test: str | Path) -> dict[str, Any]:
    paths = {"protocol_dev": Path(protocol_dev), "calibration": Path(calibration), "blind_test": Path(blind_test)}
    rows = {split: _load(path, split) for split, path in paths.items()}
    video_owner: dict[str, str] = {}
    frame_owner: dict[str, str] = {}
    for split, manifests in rows.items():
        for manifest in manifests:
            prior_video = video_owner.setdefault(manifest.source_video_sha256, split)
            if prior_video != split:
                raise ProtocolManifestError("CROSS_SPLIT_SOURCE_VIDEO_LEAKAGE")
            for frame_hash in manifest.frame_sha256s:
                prior_frame = frame_owner.setdefault(frame_hash, split)
                if prior_frame != split:
                    raise ProtocolManifestError("CROSS_SPLIT_FRAME_LEAKAGE")
    payload = {"format": "relive-v2-protocol-split-validation-v1", "status": "PASS", "protocol_version": PROTOCOL_VERSION,
               "manifest_sha256": {split: sha256_path(path) for split, path in sorted(paths.items())},
               "records_by_split": {split: len(items) for split, items in sorted(rows.items())},
               "source_video_count": len(video_owner), "frame_count": len(frame_owner),
               "gt_used": False, "model_calls_made": 0, "backend_loaded": False, "cache_opened": False,
               "certificate_created": False, "new_verified_count": 0}
    payload["validation_sha256"] = stable_hash(payload)
    return payload


def write_manifest(path: str | Path, rows: list[ProtocolManifest]) -> None:
    output = Path(path)
    if output.exists():
        raise ProtocolManifestError("PROTOCOL_MANIFEST_IMMUTABLE_OUTPUT_EXISTS")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"".join((canonical_json({key: value for key, value in row.to_dict().items() if key != "manifest_sha256"}) + "\n").encode("utf-8") for row in rows))
