from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

from evidence_stability.phase_d2 import assert_no_forbidden_model_fields


EXPECTED_SMOKE_INTERVENTIONS = 12
EXPECTED_FULL_GENERATION_VALID_INTERVENTIONS = 777

FORBIDDEN_MANIFEST_FIELDS = {
    "target_action",
    "candidate_label",
    "original_candidate_label",
    "strict_valid",
    "any_gt_valid",
    "validity_reason",
    "gt_duration_total",
    "gt_duration_bin",
    "original_gt_alignment_class",
    "intervention_gt_alignment_class",
    "original_evidence_density",
    "intervention_evidence_density",
    "original_gt_evidence_recall",
    "intervention_gt_evidence_recall",
    "original_n_gt_visible",
    "intervention_n_gt_visible",
    "intervention_n_unique_gt_visible",
}


def normalize_root(path: str | Path) -> Path:
    return Path(path).expanduser()


def normalize_manifest_relative_path(path: str | PurePosixPath) -> str:
    text = str(path).replace("\\", "/")
    pure = PurePosixPath(text)
    if pure.is_absolute():
        raise ValueError(f"Relative manifest path must not be absolute: {path}")
    parts = pure.parts
    if not parts or any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"Unsafe relative manifest path: {path}")
    return pure.as_posix()


def relative_frame_path(frame_path: str | Path, source_frame_root: str | Path) -> str:
    root = normalize_root(source_frame_root)
    frame = Path(frame_path).expanduser()
    try:
        rel = frame.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"PATH_OUTSIDE_SOURCE_ROOT: {frame} is not below {root}") from exc
    return normalize_manifest_relative_path(rel.as_posix())


def absolute_from_relative(root: str | Path, relative_path: str) -> Path:
    rel = normalize_manifest_relative_path(relative_path)
    candidate = normalize_root(root) / Path(rel)
    try:
        candidate.relative_to(normalize_root(root))
    except ValueError as exc:
        raise ValueError(f"Manifest path escapes frame root: {relative_path}") from exc
    return candidate


def load_smoke_ids_artifact(path: str | Path) -> list[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        ids = [str(item) for item in data]
    else:
        ids = [str(item["intervention_id"]) for item in data.get("interventions", [])]
    if not ids:
        raise RuntimeError(f"Smoke selection file contains no intervention IDs: {path}")
    return ids


def validate_unique_intervention_ids(rows: list[dict[str, Any]]) -> None:
    counts = Counter(str(row.get("intervention_id")) for row in rows)
    duplicates = sorted(intervention_id for intervention_id, count in counts.items() if count > 1)
    if duplicates:
        raise RuntimeError(f"Duplicate intervention_id values in D.1 manifest: {duplicates[:5]}")


def selected_interventions(
    rows: list[dict[str, Any]],
    *,
    scope: str,
    smoke_ids: list[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validate_unique_intervention_ids(rows)
    by_id = {str(row["intervention_id"]): row for row in rows}
    expectations: dict[str, Any] = {}
    if scope == "smoke":
        if smoke_ids is None:
            raise RuntimeError("Smoke scope requires frozen smoke intervention IDs.")
        missing = [intervention_id for intervention_id in smoke_ids if intervention_id not in by_id]
        if missing:
            raise RuntimeError(f"Frozen smoke intervention IDs missing from interventions.jsonl: {missing[:5]}")
        selected = [by_id[intervention_id] for intervention_id in smoke_ids]
        invalid = [str(row["intervention_id"]) for row in selected if not row.get("generation_valid")]
        if invalid:
            raise RuntimeError(f"Frozen smoke selection contains generation-invalid interventions: {invalid[:5]}")
        expectations = {
            "expected_smoke_interventions": EXPECTED_SMOKE_INTERVENTIONS,
            "smoke_count_matches_current_expectation": len(selected) == EXPECTED_SMOKE_INTERVENTIONS,
        }
        return selected, expectations
    if scope == "full":
        selected = [row for row in rows if row.get("generation_valid")]
        invalid = [str(row["intervention_id"]) for row in selected if not row.get("generation_valid")]
        if invalid:
            raise RuntimeError(f"Full selection includes generation-invalid interventions: {invalid[:5]}")
        expectations = {
            "expected_generation_valid_interventions": EXPECTED_FULL_GENERATION_VALID_INTERVENTIONS,
            "full_count_matches_current_expectation": len(selected) == EXPECTED_FULL_GENERATION_VALID_INTERVENTIONS,
        }
        return selected, expectations
    raise ValueError(f"Unsupported scope: {scope}")


def intervention_frame_record(row: dict[str, Any], source_frame_root: str | Path) -> dict[str, Any]:
    frame_paths = row.get("intervened_frame_paths")
    if not frame_paths:
        raise RuntimeError(f"Intervention has empty intervened_frame_paths: {row.get('intervention_id')}")
    declared = row.get("intervened_n_frames")
    if declared is not None and int(declared) != len(frame_paths):
        raise RuntimeError(
            f"Frame count mismatch for {row.get('intervention_id')}: "
            f"declared {declared}, actual {len(frame_paths)}"
        )
    relative_paths = [relative_frame_path(path, source_frame_root) for path in frame_paths]
    record = {
        "intervention_id": str(row["intervention_id"]),
        "dataset_name": str(row.get("dataset_name", "")),
        "target_field": row.get("target_field"),
        "intervention_family": row.get("intervention_family"),
        "intervention_type": str(row.get("intervention_type", "")),
        "n_frames": len(relative_paths),
        "relative_frame_paths": relative_paths,
        "all_frames_present_at_destination": None,
        "n_missing_at_destination": None,
    }
    assert_manifest_record_gt_free(record)
    return record


def assert_manifest_record_gt_free(record: dict[str, Any]) -> None:
    present_forbidden = sorted(set(record) & FORBIDDEN_MANIFEST_FIELDS)
    if present_forbidden:
        raise RuntimeError(f"Forbidden GT/label-derived fields in frame manifest: {present_forbidden}")
    assert_no_forbidden_model_fields(record, "frame manifest")


def stat_frame(path: Path, *, verify_hash: bool) -> dict[str, Any]:
    exists = path.exists()
    info: dict[str, Any] = {"exists": exists, "is_symlink": path.is_symlink()}
    if exists:
        stat = path.stat()
        info["size_bytes"] = int(stat.st_size)
        if verify_hash and path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    digest.update(chunk)
            info["sha256"] = digest.hexdigest()
    if path.is_symlink():
        try:
            resolved = path.resolve(strict=True)
            info["symlink_target"] = str(resolved)
        except FileNotFoundError:
            info["symlink_target"] = None
    return info


def symlink_escapes_root(path: Path, root: str | Path) -> bool:
    if not path.is_symlink():
        return False
    try:
        path.resolve(strict=True).relative_to(normalize_root(root).resolve(strict=False))
    except (FileNotFoundError, ValueError):
        return True
    return False


def human_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def coverage(existing: int, required: int) -> float:
    if required == 0:
        return 1.0
    return existing / required


def build_frame_manifest(
    rows: list[dict[str, Any]],
    *,
    scope: str,
    source_frame_root: str | Path,
    destination_frame_root: str | Path,
    smoke_ids: list[str] | None = None,
    verify_hash: bool = False,
) -> dict[str, Any]:
    selected, expectations = selected_interventions(rows, scope=scope, smoke_ids=smoke_ids)
    interventions = [intervention_frame_record(row, source_frame_root) for row in selected]
    all_refs = [path for item in interventions for path in item["relative_frame_paths"]]
    unique_paths = sorted(set(all_refs))
    duplicate_unique_path_count = len(unique_paths) - len(set(unique_paths))
    if duplicate_unique_path_count:
        raise RuntimeError("Unique required frame list contains duplicate paths.")

    source_root = str(normalize_root(source_frame_root))
    destination_root = str(normalize_root(destination_frame_root))
    by_frame: dict[str, dict[str, Any]] = {}
    source_missing_paths: list[str] = []
    destination_missing_paths: list[str] = []
    destination_existing_paths: list[str] = []
    missing_source_bytes_total = 0
    n_size_mismatch = 0
    n_hash_mismatch = 0
    n_destination_symlink_outside_root = 0

    for rel in unique_paths:
        src_path = absolute_from_relative(source_root, rel)
        dst_path = absolute_from_relative(destination_root, rel)
        src = stat_frame(src_path, verify_hash=verify_hash)
        dst = stat_frame(dst_path, verify_hash=verify_hash)
        if not src["exists"]:
            source_missing_paths.append(rel)
        if dst["exists"]:
            destination_existing_paths.append(rel)
        else:
            destination_missing_paths.append(rel)
        if src["exists"] and not dst["exists"]:
            missing_source_bytes_total += int(src.get("size_bytes", 0))
        size_mismatch = (
            src["exists"]
            and dst["exists"]
            and src.get("size_bytes") is not None
            and dst.get("size_bytes") is not None
            and src.get("size_bytes") != dst.get("size_bytes")
        )
        hash_mismatch = (
            verify_hash
            and src["exists"]
            and dst["exists"]
            and src.get("sha256") is not None
            and dst.get("sha256") is not None
            and src.get("sha256") != dst.get("sha256")
        )
        n_size_mismatch += int(bool(size_mismatch))
        n_hash_mismatch += int(bool(hash_mismatch))
        destination_symlink_outside_root = symlink_escapes_root(dst_path, destination_root)
        n_destination_symlink_outside_root += int(destination_symlink_outside_root)
        by_frame[rel] = {
            "relative_path": rel,
            "source_absolute_path": str(src_path),
            "destination_absolute_path": str(dst_path),
            "exists_at_source": bool(src["exists"]),
            "exists_at_destination": bool(dst["exists"]),
            "source_size_bytes": src.get("size_bytes"),
            "destination_size_bytes": dst.get("size_bytes"),
            "source_is_symlink": bool(src["is_symlink"]),
            "destination_is_symlink": bool(dst["is_symlink"]),
            "destination_symlink_outside_root": destination_symlink_outside_root,
            "size_mismatch": bool(size_mismatch),
        }
        if verify_hash:
            by_frame[rel]["source_sha256"] = src.get("sha256")
            by_frame[rel]["destination_sha256"] = dst.get("sha256")
            by_frame[rel]["hash_mismatch"] = bool(hash_mismatch)

    for item in interventions:
        missing = [path for path in item["relative_frame_paths"] if not by_frame[path]["exists_at_destination"]]
        item["all_frames_present_at_destination"] = not missing
        item["n_missing_at_destination"] = len(missing)

    by_dataset = aggregate_by(
        interventions,
        by_frame,
        key_name="dataset_name",
        key_func=lambda item: str(item.get("dataset_name") or "UNKNOWN"),
    )
    by_intervention_type = aggregate_by(
        interventions,
        by_frame,
        key_name="intervention_type",
        key_func=lambda item: str(item.get("intervention_type") or "UNKNOWN"),
    )

    n_existing_destination = len(destination_existing_paths)
    n_required = len(unique_paths)
    n_fully_available = sum(1 for item in interventions if item["all_frames_present_at_destination"])
    frame_count_dist = dict(sorted(Counter(int(item["n_frames"]) for item in interventions).items()))
    target_field_dist = dict(sorted(Counter(str(item.get("target_field") or "UNKNOWN") for item in interventions).items()))
    intervention_type_dist = dict(sorted(Counter(str(item.get("intervention_type") or "UNKNOWN") for item in interventions).items()))
    dataset_dist = dict(sorted(Counter(str(item.get("dataset_name") or "UNKNOWN") for item in interventions).items()))

    manifest = {
        "scope": scope,
        "n_interventions": len(interventions),
        "n_generation_valid_interventions": len(interventions) if scope == "full" else None,
        "n_smoke_interventions": len(interventions) if scope == "smoke" else None,
        "n_total_frame_references": len(all_refs),
        "n_unique_required_frames": n_required,
        "source_frame_root": source_root,
        "destination_frame_root": destination_root,
        "verify_hash": verify_hash,
        "n_existing_at_source": n_required - len(source_missing_paths),
        "n_missing_at_source": len(source_missing_paths),
        "n_existing_at_destination": n_existing_destination,
        "n_missing_at_destination": len(destination_missing_paths),
        "destination_coverage_rate": coverage(n_existing_destination, n_required),
        "missing_source_bytes_total": missing_source_bytes_total,
        "missing_source_bytes_human": human_bytes(missing_source_bytes_total),
        "n_size_mismatch": n_size_mismatch,
        "n_hash_mismatch": n_hash_mismatch if verify_hash else None,
        "n_destination_symlink_outside_root": n_destination_symlink_outside_root,
        "n_interventions_fully_available": n_fully_available,
        "n_interventions_blocked_by_missing_frames": len(interventions) - n_fully_available,
        "smoke_ready_for_inference": (len(destination_missing_paths) == 0) if scope == "smoke" else None,
        "duplicate_unique_path_count": duplicate_unique_path_count,
        "frame_count_distribution": frame_count_dist,
        "intervention_type_distribution": intervention_type_dist,
        "dataset_distribution": dataset_dist,
        "target_field_distribution": target_field_dist,
        "by_dataset": by_dataset,
        "by_intervention_type": by_intervention_type,
        "expectations": expectations,
        "source_missing_frames": source_missing_paths,
        "destination_missing_frames": destination_missing_paths,
        "destination_existing_frames": destination_existing_paths,
        "required_frames": [by_frame[rel] for rel in unique_paths],
        "interventions": interventions,
    }
    assert_manifest_record_gt_free({k: v for k, v in manifest.items() if k not in {"required_frames", "interventions"}})
    for item in manifest["required_frames"]:
        assert_manifest_record_gt_free(item)
    for item in manifest["interventions"]:
        assert_manifest_record_gt_free(item)
    return manifest


def aggregate_by(
    interventions: list[dict[str, Any]],
    by_frame: dict[str, dict[str, Any]],
    *,
    key_name: str,
    key_func,
) -> list[dict[str, Any]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for item in interventions:
        group = key_func(item)
        grouped[group].update(item["relative_frame_paths"])
    rows = []
    for group, paths in sorted(grouped.items()):
        required = len(paths)
        existing = sum(1 for path in paths if by_frame[path]["exists_at_destination"])
        missing = required - existing
        rows.append(
            {
                key_name: group,
                "required_unique_frames": required,
                "existing_at_destination": existing,
                "missing_at_destination": missing,
                "coverage_rate": coverage(existing, required),
            }
        )
    return rows


def summary_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "scope",
        "n_interventions",
        "n_generation_valid_interventions",
        "n_smoke_interventions",
        "n_total_frame_references",
        "n_unique_required_frames",
        "n_existing_at_source",
        "n_missing_at_source",
        "n_existing_at_destination",
        "n_missing_at_destination",
        "destination_coverage_rate",
        "missing_source_bytes_total",
        "missing_source_bytes_human",
        "n_size_mismatch",
        "n_hash_mismatch",
        "n_interventions_fully_available",
        "n_interventions_blocked_by_missing_frames",
        "smoke_ready_for_inference",
        "duplicate_unique_path_count",
        "frame_count_distribution",
        "intervention_type_distribution",
        "dataset_distribution",
        "target_field_distribution",
        "by_dataset",
        "by_intervention_type",
        "expectations",
        "verify_hash",
    ]
    return {key: manifest.get(key) for key in keys if manifest.get(key) is not None}
