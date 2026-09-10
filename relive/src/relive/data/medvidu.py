"""GT-isolated projection of MedVidU records into a ReliVE public runtime.

The original MedVidU file contains assistant turns and structured annotations.
This module intentionally never imports the historical dataset loaders and
never accesses values from assistant turns, ``struc_info``, ``RC_info``, source
``metadata``, or annotation-shaped fields.  Native MedVidU task types are
reported, not silently converted to ReliVE ``action_qa``.  The only
runtime-producing adapter emits ``claim_verification`` through an explicit
user-authored public-claim manifest; generic ``action_qa`` has no MedVidU path.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any

from PIL import Image

from relive.data.schemas import FIELD_SOURCES, SCHEMA_VERSION, load_runtime


PREPARATION_VERSION = "relive-medvidu-public-runtime-v1"
PUBLIC_ADAPTER = "user_claim_verification_v1"
REPORT_ONLY_ADAPTER = "report_only"
ALLOWED_SOURCE_FIELDS = ("id", "conversations[from=human].value", "video", "sampled_video_frames", "qa_type", "dataset_name")
FORBIDDEN_SOURCE_CATEGORIES = (
    "conversations[from!=human].value", "struc_info", "RC_info", "is_RC", "train", "metadata",
    "answer", "ground_truth", "bbox", "mask", "region", "temporal_span", "timestamp_gt",
)
_CLAIM_FIELDS = {"claim_id", "text", "entity", "action", "target"}
_MANIFEST_FIELDS = {"source_record_index", "public_record_sha256", "target_claim"}

TASK_COMPATIBILITY = {
    "tal": "Temporal action localization requires answer time spans; ReliVE-v1 does not infer or emit them.",
    "stg": "Spatio-temporal grounding requires bounding boxes; they are outside the public runtime.",
    "region_caption_gemini": "Region captioning requires a bounding-box query; it is outside the public runtime.",
    "region_caption_gpt": "Region captioning requires a bounding-box query; it is outside the public runtime.",
    "next_action": "Next-action prediction concerns an unobserved future action, not current visual evidence.",
    "cvs_assessment": "CVS assessment requires a multi-dimensional score, not a single visible action claim.",
    "skill_assessment": "Skill assessment requires a multi-dimensional score, not a single visible action claim.",
    "dense_captioning_gemini": "Dense captioning requires multiple time-bounded events.",
    "dense_captioning_gpt": "Dense captioning requires multiple time-bounded events.",
    "video_summary_gemini": "Video summary requires a multi-claim narrative beyond the single-action MVP.",
    "video_summary_gpt": "Video summary requires a multi-claim narrative beyond the single-action MVP.",
}


class MedVidUPreparationError(ValueError):
    """A public runtime cannot be formed under the declared isolation contract."""


@dataclass(frozen=True)
class PublicMedVidURecord:
    source_record_index: int
    source_id: str
    sample_id: str
    public_record_sha256: str
    question_sha256: str
    question: str
    qa_type: str
    video_paths: tuple[str, ...]
    sampled_frame_count: int
    dataset_name: str | None


@dataclass(frozen=True)
class _MappedFrame:
    source_path: str
    resolved_path: str | None
    status: str


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nonempty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MedVidUPreparationError(f"{label} must be nonempty text")
    return value.strip()


def _public_question(conversations: Any) -> str:
    if not isinstance(conversations, list):
        raise MedVidUPreparationError("conversations must be a list")
    humans: list[str] = []
    for turn in conversations:
        if not isinstance(turn, dict):
            raise MedVidUPreparationError("conversation turn must be an object")
        # Read role first.  For every non-human role, do not access its value.
        role = turn.get("from")
        if role == "human":
            humans.append(_nonempty_text(turn.get("value"), "human question"))
    if len(humans) != 1:
        raise MedVidUPreparationError("each source record requires exactly one human question")
    question = humans[0]
    if question.startswith("<video>"):
        question = question.removeprefix("<video>").lstrip("\r\n ")
    return _nonempty_text(question, "human question after video marker")


def _project_record(row: Any, index: int) -> PublicMedVidURecord:
    if not isinstance(row, dict):
        raise MedVidUPreparationError("source record must be an object")
    source_id = _nonempty_text(row.get("id"), "source id")
    qa_type = _nonempty_text(row.get("qa_type"), "qa_type")
    question = _public_question(row.get("conversations"))
    video = row.get("video")
    sampled = row.get("sampled_video_frames")
    if not isinstance(video, list) or not video:
        raise MedVidUPreparationError("video must be a nonempty ordered list of public frame paths")
    if not isinstance(sampled, list) or len(sampled) != len(video):
        raise MedVidUPreparationError("sampled_video_frames must pair one-for-one with video paths")
    video_paths = []
    for path in video:
        video_paths.append(_nonempty_text(path, "video frame path"))
    # This accesses only the explicitly allowed public pairing vector.  Its
    # values never enter the runtime and never become timestamps or ordering.
    for value in sampled:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise MedVidUPreparationError("sampled_video_frames must contain finite public frame references")
    dataset_name_raw = row.get("dataset_name")
    dataset_name = _nonempty_text(dataset_name_raw, "dataset_name") if dataset_name_raw is not None else None
    public_identity = {
        "source_record_index": index, "source_id": source_id, "qa_type": qa_type, "question": question,
        "video": video_paths, "sampled_frame_count": len(sampled),
        "dataset_name": dataset_name,
    }
    digest = _sha256(public_identity)
    return PublicMedVidURecord(
        source_record_index=index, source_id=source_id, sample_id=f"medvidu:{source_id}:{digest[:16]}",
        public_record_sha256=digest, question_sha256=hashlib.sha256(question.encode("utf-8")).hexdigest(),
        question=question, qa_type=qa_type, video_paths=tuple(video_paths),
        sampled_frame_count=len(sampled), dataset_name=dataset_name,
    )


def load_public_records(source_json: str | Path) -> tuple[list[PublicMedVidURecord], str]:
    """Project only the allowed source fields; never return original rows."""
    source = Path(source_json).expanduser().resolve()
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MedVidUPreparationError("source_json must be a readable JSON array") from exc
    if not isinstance(raw, list) or not raw:
        raise MedVidUPreparationError("source_json must contain a nonempty record array")
    records = [_project_record(row, index) for index, row in enumerate(raw)]
    if len({record.sample_id for record in records}) != len(records):
        raise MedVidUPreparationError("public record projection produced ambiguous sample ids")
    return records, _sha256_file(source)


def _normalize_schema(question: str) -> str:
    result = re.sub(r"\d+(?:\.\d+)?", "<NUMBER>", question)
    return re.sub(r"\s+", " ", result).strip()


def medvidu_schema_report(records: list[PublicMedVidURecord], source_json_sha256: str) -> dict[str, Any]:
    groups: dict[str, list[PublicMedVidURecord]] = defaultdict(list)
    for record in records:
        groups[record.qa_type].append(record)
    qa_types = {}
    for qa_type in sorted(groups):
        rows = groups[qa_type]
        patterns = Counter(_normalize_schema(row.question) for row in rows)
        qa_types[qa_type] = {
            "record_count": len(rows), "unique_human_question_schemas": len(patterns),
            "representative_human_question_schemas": [pattern for pattern, _ in patterns.most_common(3)],
            "runtime_status": "UNSUPPORTED_NO_EXPLICIT_ADAPTER",
            "reason": TASK_COMPATIBILITY.get(qa_type, "No explicit ReliVE public-runtime adapter is implemented for this qa_type."),
        }
    return {
        "preparation_version": PREPARATION_VERSION, "source_json_sha256": source_json_sha256,
        "record_count": len(records), "qa_types": qa_types,
        "native_action_qa_adapter": "NOT_IMPLEMENTED_NO_NATIVE_QA_TYPE_CONFIRMED",
        "allowed_source_fields": list(ALLOWED_SOURCE_FIELDS),
        "forbidden_source_categories": list(FORBIDDEN_SOURCE_CATEGORIES),
    }


def _public_record_index_payload(records: list[PublicMedVidURecord]) -> bytes:
    """Selectors for a user-authored claim manifest, with no answers or GT."""
    return b"".join(_canonical_bytes({
        "source_record_index": record.source_record_index, "sample_id": record.sample_id,
        "public_record_sha256": record.public_record_sha256, "question_sha256": record.question_sha256,
        "qa_type": record.qa_type,
    }) + b"\n" for record in records)


def _public_question_selector_payload(records: list[PublicMedVidURecord], mapper: "_FrameMapper") -> bytes:
    """Write only reviewed human questions and public, decoded frame references.

    This is deliberately narrower than a source-row export.  Every listed
    record has already passed the requested frame mapping audit, and the
    selector contains the two values required to bind a later user-authored
    claim manifest: ``source_record_index`` and ``public_record_sha256``.
    It never emits source annotation fields or values from non-human turns.
    """
    rows = []
    for record in records:
        mapped_paths = [mapper.map(path) for path in record.video_paths]
        if any(item.status != "PASS" or item.resolved_path is None for item in mapped_paths):
            raise MedVidUPreparationError("PUBLIC_SELECTOR_REQUIRES_VERIFIED_FRAME_PATHS")
        row = {
            "source_record_index": record.source_record_index,
            "sample_id": record.sample_id,
            "public_record_sha256": record.public_record_sha256,
            "question_sha256": record.question_sha256,
            "qa_type": record.qa_type,
            "question": record.question,
            "frame_count": record.sampled_frame_count,
            "first_verified_frame_path": mapped_paths[0].resolved_path,
            "last_verified_frame_path": mapped_paths[-1].resolved_path,
        }
        if record.dataset_name is not None:
            row["dataset_name"] = record.dataset_name
        rows.append(row)
    return b"".join(_canonical_bytes(row) + b"\n" for row in rows)


class _FrameMapper:
    def __init__(self, source_prefix: str | Path, frame_root: str | Path):
        self.source_prefix = PurePosixPath(str(source_prefix))
        if not self.source_prefix.is_absolute() or self.source_prefix == PurePosixPath("/"):
            raise MedVidUPreparationError("source_prefix must be a non-root absolute POSIX path")
        self.frame_root = Path(frame_root).expanduser()
        try:
            self.resolved_root = self.frame_root.resolve(strict=True)
        except OSError as exc:
            raise MedVidUPreparationError("frame_root is not a readable directory") from exc
        if not self.resolved_root.is_dir():
            raise MedVidUPreparationError("frame_root must be a directory")
        self.cache: dict[str, _MappedFrame] = {}

    def map(self, source_path: str) -> _MappedFrame:
        prior = self.cache.get(source_path)
        if prior is not None:
            return prior
        try:
            source = PurePosixPath(source_path)
            if not source.is_absolute() or source == self.source_prefix or any(part in {".", ".."} for part in source.parts):
                raise MedVidUPreparationError("INVALID_SOURCE_FRAME_PATH")
            relative = source.relative_to(self.source_prefix)
            if not relative.parts or any(part in {".", ".."} for part in relative.parts):
                raise MedVidUPreparationError("INVALID_SOURCE_FRAME_PATH")
            candidate = (self.resolved_root.joinpath(*relative.parts)).resolve(strict=True)
            candidate.relative_to(self.resolved_root)
            if not candidate.is_file():
                raise MedVidUPreparationError("MISSING_FRAME")
            with Image.open(candidate) as image:
                image.verify()
            result = _MappedFrame(source_path, str(candidate), "PASS")
        except (OSError, ValueError, MedVidUPreparationError) as exc:
            reason = str(exc) if str(exc) in {"INVALID_SOURCE_FRAME_PATH", "MISSING_FRAME"} else "UNREADABLE_OR_INVALID_FRAME"
            result = _MappedFrame(source_path, None, reason)
        self.cache[source_path] = result
        return result


def audit_frame_mapping(records: list[PublicMedVidURecord], source_prefix: str | Path, frame_root: str | Path) -> tuple[dict[str, Any], _FrameMapper]:
    mapper = _FrameMapper(source_prefix, frame_root)
    total, duplicate_references = 0, 0
    seen: set[str] = set()
    failures = []
    for record in records:
        for path in record.video_paths:
            total += 1
            if path in seen:
                duplicate_references += 1
            seen.add(path)
            result = mapper.map(path)
            if result.status != "PASS" and len(failures) < 20:
                failures.append({"source_path": path, "status": result.status})
    status_counts = Counter(item.status for item in mapper.cache.values())
    audit = {
        "status": "PASS" if not failures and status_counts.get("PASS", 0) == len(mapper.cache) else "FAIL",
        "source_prefix": str(source_prefix), "frame_root": str(mapper.resolved_root),
        "records_checked": len(records), "frame_references_checked": total,
        "unique_frame_paths_checked": len(mapper.cache), "duplicate_frame_references_preserved": duplicate_references,
        "image_decode_verified": status_counts.get("PASS", 0), "status_counts": dict(sorted(status_counts.items())),
        "failure_examples": failures,
        "timestamp_policy": "sampled_video_frames were paired only; no timestamps, FPS, or time relation were emitted.",
    }
    return audit, mapper


def _manifest_rows(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise MedVidUPreparationError("public claim manifest must be readable JSONL") from exc
    rows = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MedVidUPreparationError(f"invalid public claim manifest line {number}") from exc
        if not isinstance(row, dict) or set(row) != _MANIFEST_FIELDS:
            raise MedVidUPreparationError("public claim manifest rows need exactly source_record_index, public_record_sha256, target_claim")
        if type(row["source_record_index"]) is not int or row["source_record_index"] < 0:
            raise MedVidUPreparationError("manifest source_record_index must be a nonnegative integer")
        if not isinstance(row["public_record_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", row["public_record_sha256"]):
            raise MedVidUPreparationError("manifest public_record_sha256 must be a SHA-256 digest")
        claim = row["target_claim"]
        if not isinstance(claim, dict) or set(claim) - _CLAIM_FIELDS:
            raise MedVidUPreparationError("manifest target_claim contains forbidden or unsupported fields")
        if not isinstance(claim.get("claim_id"), str) or not claim["claim_id"].strip() or not isinstance(claim.get("text"), str) or not claim["text"].strip():
            raise MedVidUPreparationError("manifest target_claim needs nonempty claim_id and text")
        for key in ("entity", "action", "target"):
            if key in claim and (not isinstance(claim[key], str) or not claim[key].strip()):
                raise MedVidUPreparationError(f"manifest target_claim.{key} must be nonempty text")
        rows.append({"source_record_index": row["source_record_index"], "public_record_sha256": row["public_record_sha256"],
                     "target_claim": dict(claim)})
    if not rows or len({row["source_record_index"] for row in rows}) != len(rows):
        raise MedVidUPreparationError("public claim manifest must have unique nonempty source selectors")
    return rows


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise MedVidUPreparationError(f"refusing to replace existing output with different content: {path.name}")
        return
    descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise MedVidUPreparationError(f"refusing to replace existing output with different content: {path.name}")
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _validate_runtime_bytes(runtime_payload: bytes, sidecar_payload: bytes) -> None:
    with tempfile.TemporaryDirectory(prefix="relive-medvidu-runtime-") as directory:
        path = Path(directory) / "runtime.jsonl"
        path.write_bytes(runtime_payload)
        Path(str(path) + ".provenance.json").write_bytes(sidecar_payload)
        try:
            load_runtime(path)
        except Exception as exc:
            raise MedVidUPreparationError("generated runtime failed closed-schema validation") from exc


def _runtime_rows(selected: list[tuple[PublicMedVidURecord, dict[str, Any]]], mapper: _FrameMapper) -> list[dict[str, Any]]:
    rows = []
    for record, manifest in selected:
        frames = []
        for order, source_path in enumerate(record.video_paths):
            mapped = mapper.map(source_path)
            if mapped.status != "PASS" or mapped.resolved_path is None:
                raise MedVidUPreparationError("FRAME_MAPPING_AUDIT_FAILED; no runtime was written")
            frames.append({"frame_id": f"{record.sample_id}:frame:{order:06d}", "path": mapped.resolved_path, "order": order})
        metadata: dict[str, Any] = {
            "runtime_adapter": PUBLIC_ADAPTER, "source_qa_type": record.qa_type,
            "source_record_sha256": record.public_record_sha256, "nonofficial_protocol": True,
        }
        if record.dataset_name is not None:
            metadata["dataset_name"] = record.dataset_name
        rows.append({"sample_id": record.sample_id, "task": "claim_verification", "question": record.question,
                     "frames": frames, "target_claim": manifest["target_claim"], "metadata": metadata})
    return rows


def _isolation_audit(*, source_json: str | Path, source_sha256: str, adapter: str,
                     records: list[PublicMedVidURecord], path_audit: dict[str, Any],
                     runtime_sha256: str | None, manifest_path: str | Path | None) -> dict[str, Any]:
    return {
        "audit_version": PREPARATION_VERSION, "status": "PASS" if path_audit["status"] == "PASS" else "FAIL",
        "adapter": adapter,
        "classification": ("custom_public_claim_protocol_smoke_not_official_medvidu_qa" if adapter == PUBLIC_ADAPTER
                           else "schema_and_public_path_audit_only_no_runtime_generated"),
        "source_json_path": str(Path(source_json).expanduser().resolve()), "source_json_sha256": source_sha256,
        "selected_public_record_count": len(records), "selected_source_record_indices": [r.source_record_index for r in records],
        "allowed_source_fields": list(ALLOWED_SOURCE_FIELDS),
        "forbidden_source_categories_not_selected": list(FORBIDDEN_SOURCE_CATEGORIES),
        "assistant_value_policy": "roles inspected; non-human conversation values were not accessed",
        "source_metadata_policy": "source metadata object was not accessed or propagated",
        "sampled_frame_policy": "paired length/type validation only; no value emitted as timestamp or ground truth",
        "path_mapping": path_audit, "claim_manifest_path": (str(Path(manifest_path).expanduser().resolve()) if manifest_path else None),
        "runtime_sha256": runtime_sha256,
    }


def prepare_medvidu(
    source_json: str | Path,
    frame_root: str | Path,
    output_dir: str | Path,
    *,
    source_prefix: str | Path = "/root/data",
    adapter: str = REPORT_ONLY_ADAPTER,
    path_audit_scope: str = "selected",
    max_samples: int = 5,
    public_claim_manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Write a report-only audit or explicit public-claim runtime.

    ``report_only`` never writes runtime JSONL and never calls a model.  The
    only runtime-producing adapter is intentionally user-authored: it binds a
    claim manifest to a hash of a public source projection and labels results as
    non-official MedVidU protocol smoke outputs.
    """
    if adapter not in {REPORT_ONLY_ADAPTER, PUBLIC_ADAPTER}:
        raise MedVidUPreparationError("UNSUPPORTED_NO_EXPLICIT_ADAPTER")
    if path_audit_scope not in {"selected", "all"}:
        raise MedVidUPreparationError("path_audit_scope must be selected or all")
    if type(max_samples) is not int or not 1 <= max_samples <= 5:
        raise MedVidUPreparationError("max_samples must be between 1 and 5 for bounded smoke preparation")
    destination = Path(output_dir).expanduser().resolve()
    if "mock" in destination.parts:
        raise MedVidUPreparationError("MedVidU public-runtime preparation cannot write under a mock output directory")
    records, source_sha256 = load_public_records(source_json)
    schema = medvidu_schema_report(records, source_sha256)
    if adapter == REPORT_ONLY_ADAPTER:
        audit_records = records if path_audit_scope == "all" else records[:max_samples]
        path_audit, mapper = audit_frame_mapping(audit_records, source_prefix, frame_root)
        isolation = _isolation_audit(source_json=source_json, source_sha256=source_sha256, adapter=adapter,
                                     records=audit_records, path_audit=path_audit, runtime_sha256=None,
                                     manifest_path=None)
        schema_path = destination / "medvidu_human_question_schema.json"
        index_path = destination / "medvidu_public_record_index.jsonl"
        selector_path = destination / "medvidu_public_question_selector.jsonl"
        audit_path = destination / "medvidu_gt_isolation_audit.json"
        _atomic_write(schema_path, _canonical_bytes(schema) + b"\n")
        _atomic_write(index_path, _public_record_index_payload(records))
        _atomic_write(audit_path, _canonical_bytes(isolation) + b"\n")
        if path_audit["status"] == "PASS":
            _atomic_write(selector_path, _public_question_selector_payload(audit_records, mapper))
        return {"status": path_audit["status"], "adapter": adapter, "runtime_generated": False,
                "schema_report": str(schema_path), "public_record_index": str(index_path),
                "public_question_selector": (str(selector_path) if path_audit["status"] == "PASS" else None),
                "gt_isolation_audit": str(audit_path), "path_mapping": path_audit}
    if public_claim_manifest is None:
        raise MedVidUPreparationError("user_claim_verification_v1 requires --public-claim-manifest")
    by_index = {record.source_record_index: record for record in records}
    selected = []
    for manifest in _manifest_rows(public_claim_manifest):
        record = by_index.get(manifest["source_record_index"])
        if record is None or record.public_record_sha256 != manifest["public_record_sha256"]:
            raise MedVidUPreparationError("public claim manifest selector does not match the current public source projection")
        selected.append((record, manifest))
    selected.sort(key=lambda pair: pair[0].source_record_index)
    selected = selected[:max_samples]
    selected_records = [record for record, _ in selected]
    path_audit, mapper = audit_frame_mapping(selected_records, source_prefix, frame_root)
    if path_audit["status"] != "PASS":
        raise MedVidUPreparationError("FRAME_MAPPING_AUDIT_FAILED; no runtime was written")
    rows = _runtime_rows(selected, mapper)
    runtime_payload = b"".join(_canonical_bytes(row) + b"\n" for row in rows)
    runtime_sha256 = hashlib.sha256(runtime_payload).hexdigest()
    sidecar = {"schema_version": SCHEMA_VERSION, "source_kind": "public_runtime", "runtime_sha256": runtime_sha256,
               "field_sources": FIELD_SOURCES}
    sidecar_payload = _canonical_bytes(sidecar) + b"\n"
    _validate_runtime_bytes(runtime_payload, sidecar_payload)
    isolation = _isolation_audit(source_json=source_json, source_sha256=source_sha256, adapter=adapter,
                                 records=selected_records, path_audit=path_audit, runtime_sha256=runtime_sha256,
                                 manifest_path=public_claim_manifest)
    runtime_path = destination / "medvidu_user_claim_verification.runtime.jsonl"
    sidecar_path = Path(str(runtime_path) + ".provenance.json")
    audit_path = Path(str(runtime_path) + ".gt_isolation_audit.json")
    _atomic_write(runtime_path, runtime_payload)
    _atomic_write(sidecar_path, sidecar_payload)
    _atomic_write(audit_path, _canonical_bytes(isolation) + b"\n")
    return {"status": "PASS", "adapter": adapter, "runtime_generated": True, "runtime": str(runtime_path),
            "provenance_sidecar": str(sidecar_path), "gt_isolation_audit": str(audit_path),
            "runtime_sha256": runtime_sha256, "path_mapping": path_audit,
            "classification": isolation["classification"]}
