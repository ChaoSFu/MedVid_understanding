"""Zero-model export of fresh public records for human visual selection.

The exporter intentionally has no claim, runner, verifier, certificate, or
model backend dependency. It creates only a deterministic public-frame review
set; a human must later author claims and freeze a small public window.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .audits import audit_runtime_imports
from .data.medvidu import audit_frame_mapping, load_public_records
from .fresh_public_runtime import EXCLUDED_SOURCE_RECORD_INDICES
from .storage.artifacts import canonical_json


PUBLIC_RECORD_CANDIDATE_FORMAT = "relive-phase35-public-record-candidates-v1"


class PublicRecordCandidateError(ValueError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _selector_rows(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicRecordCandidateError(f"unreadable public selector: {path}") from exc
    required = {"source_record_index", "sample_id", "public_record_sha256", "question_sha256", "qa_type",
                "question", "frame_count", "first_verified_frame_path", "last_verified_frame_path", "dataset_name"}
    if not rows or any(not isinstance(row, dict) or set(row) != required for row in rows):
        raise PublicRecordCandidateError("public selector does not match the closed public selector schema")
    if [row["source_record_index"] for row in rows] != sorted(row["source_record_index"] for row in rows):
        raise PublicRecordCandidateError("public selector must be ordered by source_record_index")
    if len({row["source_record_index"] for row in rows}) != len(rows):
        raise PublicRecordCandidateError("public selector has duplicate source_record_index")
    return rows


def _git_state(repo_root: Path) -> dict[str, Any]:
    try:
        branch = subprocess.check_output(["git", "-C", str(repo_root), "branch", "--show-current"], text=True).strip()
        commit = subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(["git", "-C", str(repo_root), "status", "--short"], text=True).splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PublicRecordCandidateError("unable to read git state") from exc
    if not branch or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise PublicRecordCandidateError("invalid git state")
    return {"branch": branch, "commit": commit, "worktree_status": status}


def _page_path(record_index: int, page_index: int) -> str:
    return f"contact_sheets/source_{record_index:06d}_page_{page_index:03d}.png"


def _thumbnail(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    copy = image.convert("RGB")
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "black")
    left = (size[0] - copy.width) // 2
    top = (size[1] - copy.height) // 2
    canvas.paste(copy, (left, top))
    return canvas


def _caption_text(value: str) -> str:
    """Keep labels renderable by Pillow's minimal ASCII/Latin-1 fallback font."""
    return value.encode("latin-1", "replace").decode("latin-1")


def _contact_sheet(*, destination: Path, record_index: int, sample_id: str, dataset_name: str | None,
                   qa_type: str, paths: list[str], orders: list[int], columns: int, thumb_size: tuple[int, int]) -> None:
    if len(paths) != len(orders) or not paths:
        raise PublicRecordCandidateError("invalid contact sheet page")
    title_height, caption_height = 56, 34
    rows = (len(paths) + columns - 1) // columns
    width = columns * thumb_size[0]
    height = title_height + rows * (thumb_size[1] + caption_height)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title = f"source_record_index={record_index} | sample_id={sample_id} | dataset={dataset_name} | qa_type={qa_type}"
    subtitle = f"Public frames only; continuous original frame orders {orders[0]}-{orders[-1]}; no claim or model output."
    draw.text((6, 4), _caption_text(title), fill="black", font=ImageFont.load_default())
    draw.text((6, 24), _caption_text(subtitle), fill="black", font=ImageFont.load_default())
    for index, (path, order) in enumerate(zip(paths, orders)):
        col, row = index % columns, index // columns
        x, y = col * thumb_size[0], title_height + row * (thumb_size[1] + caption_height)
        try:
            with Image.open(path) as image:
                canvas.paste(_thumbnail(image, thumb_size), (x, y))
        except (OSError, ValueError) as exc:
            raise PublicRecordCandidateError(f"unable to render verified public frame: {path}") from exc
        draw.text((x + 3, y + thumb_size[1] + 3), f"order={order}", fill="black", font=ImageFont.load_default())
        draw.text((x + 3, y + thumb_size[1] + 16), _caption_text(Path(path).name[:36]), fill="black", font=ImageFont.load_default())
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG")


def _write_new(path: Path, payload: bytes) -> None:
    if path.exists():
        raise PublicRecordCandidateError(f"refusing to overwrite immutable artifact: {path}")
    path.write_bytes(payload)


def export_public_record_candidates(*, public_selector: Path, source_json: Path, frame_root: Path,
                                    source_prefix: str, output_dir: Path, repo_root: Path,
                                    batch_size: int = 10, page_size: int = 30, columns: int = 5,
                                    thumbnail_width: int = 256, thumbnail_height: int = 192) -> dict[str, Any]:
    """Export a deterministic, non-overlapping human-review set from public data."""
    if output_dir.exists():
        raise PublicRecordCandidateError("candidate export output directory must not exist")
    if type(batch_size) is not int or not 1 <= batch_size <= 50:
        raise PublicRecordCandidateError("batch_size must be an integer from 1 to 50")
    if type(page_size) is not int or not 1 <= page_size <= 60 or type(columns) is not int or not 1 <= columns <= 10:
        raise PublicRecordCandidateError("page_size and columns are outside safe contact-sheet bounds")
    if type(thumbnail_width) is not int or type(thumbnail_height) is not int or thumbnail_width < 32 or thumbnail_height < 32:
        raise PublicRecordCandidateError("thumbnail dimensions are invalid")
    selector = _selector_rows(public_selector)
    records, source_sha = load_public_records(source_json)
    by_index = {record.source_record_index: record for record in records}
    selected, seen_hashes, seen_samples = [], set(), set()
    duplicates_skipped = Counter()
    for row in selector:
        index = row["source_record_index"]
        if index in EXCLUDED_SOURCE_RECORD_INDICES:
            duplicates_skipped["excluded_development_source_index"] += 1
            continue
        record = by_index.get(index)
        if record is None or record.sample_id != row["sample_id"] or record.public_record_sha256 != row["public_record_sha256"]:
            raise PublicRecordCandidateError("public selector cannot be bound to the current public source projection")
        if record.public_record_sha256 in seen_hashes:
            duplicates_skipped["duplicate_public_record_sha256"] += 1
            continue
        if record.sample_id in seen_samples:
            duplicates_skipped["duplicate_sample_id"] += 1
            continue
        if len(record.video_paths) < 3:
            duplicates_skipped["fewer_than_three_public_frames"] += 1
            continue
        selected.append(record)
        seen_hashes.add(record.public_record_sha256)
        seen_samples.add(record.sample_id)
        if len(selected) == batch_size:
            break
    if len(selected) != batch_size:
        raise PublicRecordCandidateError(f"only {len(selected)} fresh deduplicated public records found; batch_size={batch_size}")
    path_audit, mapper = audit_frame_mapping(selected, source_prefix, frame_root)
    if path_audit["status"] != "PASS":
        raise PublicRecordCandidateError("public frame mapping audit failed; no contact sheets were written")
    source_audit = audit_runtime_imports()
    if source_audit["status"] != "PASS":
        raise PublicRecordCandidateError("runtime import audit failed")
    git = _git_state(repo_root)
    output_dir.mkdir(parents=True)
    manifest_rows = []
    for record in selected:
        mapped_paths = []
        for source_path in record.video_paths:
            mapped = mapper.map(source_path)
            if mapped.status != "PASS" or mapped.resolved_path is None:
                raise PublicRecordCandidateError("verified public frame became unavailable")
            mapped_paths.append(mapped.resolved_path)
        pages = []
        for page_index, start in enumerate(range(0, len(mapped_paths), page_size)):
            end = min(start + page_size, len(mapped_paths))
            relative = _page_path(record.source_record_index, page_index)
            _contact_sheet(destination=output_dir / relative, record_index=record.source_record_index,
                           sample_id=record.sample_id, dataset_name=record.dataset_name, qa_type=record.qa_type,
                           paths=mapped_paths[start:end], orders=list(range(start, end)), columns=columns,
                           thumb_size=(thumbnail_width, thumbnail_height))
            pages.append({"frame_order_start": start, "frame_order_end": end - 1,
                          "frame_orders": list(range(start, end)), "path": relative})
        manifest_rows.append({"format": PUBLIC_RECORD_CANDIDATE_FORMAT, "source_record_index": record.source_record_index,
                              "sample_id": record.sample_id, "public_record_sha256": record.public_record_sha256,
                              "dataset_name": record.dataset_name, "source_qa_type": record.qa_type,
                              "public_frame_count": len(mapped_paths), "public_frame_orders": list(range(len(mapped_paths))),
                              "first_public_frame_path": mapped_paths[0], "last_public_frame_path": mapped_paths[-1],
                              "contact_sheet_pages": pages, "claim_generated": False,
                              "human_visual_confirmation": None, "model_calls_made": 0, "gt_used": False})
    body = b"".join((canonical_json(row) + "\n").encode("utf-8") for row in manifest_rows)
    manifest_sha = hashlib.sha256(body).hexdigest()
    _write_new(output_dir / "fresh_source_record_candidates.jsonl", body)
    report = {"status": "PASS", "phase": "ReliVE Phase 3.5 fresh public-record candidate export",
              "model_calls_made": 0, "relive_runner_called": False, "spatial_proposer_called": False,
              "refinement_called": False, "certificate_called": False, "claim_generated": False,
              "public_selector": str(public_selector), "public_selector_sha256": _sha(public_selector),
              "source_json": str(source_json), "source_json_sha256": source_sha,
              "fresh_source_manifest": str(output_dir / "fresh_source_record_candidates.jsonl"),
              "fresh_source_manifest_sha256": manifest_sha, "batch_size": batch_size,
              "selection_rule": "ascending public-selector source_record_index after fixed development exclusions and duplicate checks",
              "exclusion_and_duplicate_audit": {"status": "PASS", "excluded_source_record_indices": sorted(EXCLUDED_SOURCE_RECORD_INDICES),
                                                "skipped": dict(sorted(duplicates_skipped.items())),
                                                "selected_source_record_indices": [record.source_record_index for record in selected],
                                                "selected_public_record_sha256_unique": len(seen_hashes) == len(selected),
                                                "selected_sample_id_unique": len(seen_samples) == len(selected)},
              "contact_sheet": {"page_size": page_size, "columns": columns, "thumbnail_size": [thumbnail_width, thumbnail_height],
                                "frame_policy": "all public frames are rendered in original continuous order across pages"},
              "public_frame_mapping_audit": path_audit, "runtime_import_audit": source_audit,
              "git": git, "prohibited_inputs_not_opened": ["reference_answer", "assistant_answer", "temporal_gt",
                                                               "bbox_mask_gt", "struc_info", "RC_info", "evaluation_artifacts"]}
    _write_new(output_dir / "public_record_candidate_export_report.json", (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    return report
