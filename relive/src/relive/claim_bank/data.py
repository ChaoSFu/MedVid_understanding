"""Allowlist-only source projection, audit, and deterministic public sampling."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image

from relive.storage.artifacts import stable_hash
from relive.v2.task_selection import strict_json_loads
from .contracts import ClaimBankError, QA_TYPES, SAMPLER_VERSION, VERSION

METADATA_KEYS = {"video_id", "fps", "start_frame", "end_frame", "input_video_start_frame",
                 "input_video_end_frame", "input_video_start_time", "input_video_end_time"}
RUNTIME_KEYS = {"format", "sample_key", "dataset_sha256", "source_record_index", "original_sample_id",
                "qa_type", "data_source", "dataset_name", "question", "video", "sampled_video_frames",
                "metadata", "public_record_sha256", "input_issues"}
ALLOWED_SOURCE_FIELDS = ["id (identity only)", "qa_type", "data_source", "dataset_name", "video",
                         "sampled_video_frames", "conversations[from=human].value",
                         "metadata." + ",".join(sorted(METADATA_KEYS))]


def digest(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: str | Path) -> Any:
    return strict_json_loads(Path(path).read_text(encoding="utf-8"), error_code="CLAIMGEN_JSON_INVALID")


def rows(path: str | Path) -> list[dict]:
    value = [strict_json_loads(line, error_code="CLAIMGEN_JSONL_INVALID")
             for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(x, dict) for x in value):
        raise ClaimBankError("JSONL_OBJECTS_REQUIRED")
    return value


def write_once(path: Path, value: Any, *, jsonl: bool = False) -> None:
    from relive.storage.artifacts import canonical_json
    body = (("".join(canonical_json(v) + "\n" for v in value)) if jsonl else canonical_json(value) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Use the core atomic create-if-absent store, including immutable JSONL exports.
    import os
    import tempfile
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != body:
                raise ClaimBankError(f"IMMUTABLE_OUTPUT_EXISTS:{path}") from None
    finally:
        os.unlink(temporary)


def project(record: dict, index: int, dataset_hash: str) -> dict:
    """Never access non-human values, annotations, assessment labels or GT metadata.

    The JSON container is deserialized to project it; forbidden values are not
    inspected or copied. Dataset bytes are hashed only as an identity binding.
    """
    issues = []
    humans = []
    conversations = record.get("conversations")
    for turn in conversations if isinstance(conversations, list) else []:
        if isinstance(turn, dict) and turn.get("from") == "human":
            value = turn.get("value")
            if isinstance(value, str) and value.strip():
                humans.append(value)
    if len(humans) != 1:
        issues.append("EXACTLY_ONE_HUMAN_QUESTION_REQUIRED")
    question = humans[0] if len(humans) == 1 else ""
    video = record.get("video")
    if not isinstance(video, list) or not video or any(not isinstance(p, str) or not p for p in video):
        issues.append("VIDEO_PATH_LIST_INVALID")
        video = []
    sampled = record.get("sampled_video_frames")
    if not isinstance(sampled, list) or len(sampled) != len(video) or any(type(v) not in {int, float} or not math.isfinite(v) for v in sampled):
        issues.append("SAMPLED_FRAME_INDICES_UNAVAILABLE")
        sampled = []
    metadata = record.get("metadata")
    public_metadata = {}
    if isinstance(metadata, dict):
        for key in sorted(METADATA_KEYS):
            v = metadata.get(key)
            if type(v) in {str, int, float} and (not isinstance(v, float) or math.isfinite(v)):
                public_metadata[key] = v
    def public_string(key):
        v = record.get(key)
        return v if isinstance(v, str) else None
    value = {"format": "relive-claimgen-runtime-v1", "sample_key": f"medvidu-{dataset_hash}-{index:06d}",
             "dataset_sha256": dataset_hash, "source_record_index": index,
             "original_sample_id": public_string("id"), "qa_type": public_string("qa_type"),
             "data_source": public_string("data_source"), "dataset_name": public_string("dataset_name"),
             "question": question, "video": video, "sampled_video_frames": sampled,
             "metadata": public_metadata, "input_issues": issues}
    value["public_record_sha256"] = stable_hash(value)
    return value


def validate_runtime(row: dict) -> None:
    if set(row) != RUNTIME_KEYS or set(row.get("metadata", {})) - METADATA_KEYS:
        raise ClaimBankError("RUNTIME_ALLOWLIST_VIOLATION")
    if row.get("format") != "relive-claimgen-runtime-v1":
        raise ClaimBankError("RUNTIME_FORMAT_INVALID")
    if type(row.get("source_record_index")) is not int or row["source_record_index"] < 0:
        raise ClaimBankError("RUNTIME_INDEX_INVALID")
    h = row.get("dataset_sha256")
    if not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h):
        raise ClaimBankError("RUNTIME_DATASET_HASH_INVALID")
    if row["sample_key"] != f"medvidu-{h}-{row['source_record_index']:06d}":
        raise ClaimBankError("RUNTIME_IDENTITY_INVALID")
    if stable_hash({k: v for k, v in row.items() if k != "public_record_sha256"}) != row["public_record_sha256"]:
        raise ClaimBankError("RUNTIME_HASH_MISMATCH")
    if not isinstance(row["question"], str) or not isinstance(row["video"], list) or any(not isinstance(p, str) for p in row["video"]):
        raise ClaimBankError("RUNTIME_FIELDS_INVALID")


def resolve_path(path: str, data_root: str | None = None, path_map: dict | None = None) -> Path:
    mappings = dict(path_map or {})
    if data_root:
        mappings.setdefault("/root/data", data_root)
    for prefix in sorted(mappings, key=len, reverse=True):
        if path == prefix or path.startswith(prefix.rstrip("/") + "/"):
            relative = PurePosixPath(path[len(prefix):].lstrip("/"))
            if ".." in relative.parts:
                raise ClaimBankError("FRAME_PATH_TRAVERSAL")
            root = Path(mappings[prefix]).expanduser().resolve()
            target = root.joinpath(*relative.parts).resolve()
            if not target.is_relative_to(root):
                raise ClaimBankError("FRAME_PATH_ESCAPES_ROOT")
            return target
    return Path(path).expanduser().resolve()


def audit_dataset(dataset: str, output: str, *, data_root: str | None = None,
                  path_map: dict | None = None, expected_count: int | None = 6270,
                  prepare: bool = False) -> dict:
    source_hash = digest(dataset)
    source = load_json(dataset)
    if not isinstance(source, list) or any(not isinstance(r, dict) for r in source):
        raise ClaimBankError("DATASET_MUST_BE_OBJECT_ARRAY")
    projected = [project(r, i, source_hash) for i, r in enumerate(source)]
    qa = Counter(r["qa_type"] for r in projected)
    groups = defaultdict(list)
    path_status = {}
    lengths = defaultdict(list)
    task_paths = defaultdict(set)
    for row in projected:
        if row["original_sample_id"] is not None:
            groups[row["original_sample_id"]].append(row["source_record_index"])
        lengths[row["qa_type"]].append(len(row["video"]))
        task_paths[row["qa_type"]].update(row["video"])
        for p in row["video"]:
            if p not in path_status:
                try:
                    path_status[p] = resolve_path(p, data_root, path_map).is_file()
                except (OSError, ValueError):
                    path_status[p] = False
    def distribution(xs):
        ordered = sorted(xs)
        return {"count": len(xs), "min": min(xs), "median": ordered[len(xs) // 2],
                "p95": ordered[min(len(xs) - 1, int(len(xs) * .95))], "max": max(xs)}
    report = {"format": "relive-claimgen-dataset-audit-v1", "protocol_version": VERSION,
              "source_path": str(Path(dataset).resolve()), "dataset_sha256": source_hash,
              "record_count": len(projected), "expected_count": expected_count,
              "count_matches": expected_count is None or len(projected) == expected_count,
              "qa_type_counts": dict(sorted(qa.items(), key=lambda p: str(p[0]))),
              "unsupported_qa_types": sorted(str(t) for t in qa if t not in QA_TYPES),
              "duplicate_id_group_count": sum(len(v) > 1 for v in groups.values()),
              "duplicate_original_ids": [{"original_sample_id": k, "source_record_indexes": v} for k, v in sorted(groups.items()) if len(v) > 1],
              "frame_count_distributions": {str(k): distribution(v) for k, v in sorted(lengths.items(), key=lambda p: str(p[0]))},
              "unique_frame_paths": len(path_status), "existing_unique_frame_paths": sum(path_status.values()),
              "missing_unique_frame_paths": sum(not v for v in path_status.values()),
              "frame_path_status_by_qa_type": {str(k): {"unique_paths": len(v), "existing": sum(path_status[p] for p in v), "missing": sum(not path_status[p] for p in v)} for k, v in task_paths.items()},
              "missing_path_examples": sorted(p for p, ok in path_status.items() if not ok)[:20],
              "sampled_indices_missing_count": sum("SAMPLED_FRAME_INDICES_UNAVAILABLE" in r["input_issues"] for r in projected),
              "timebase": "UNRESOLVED_ORDER_ONLY_NO_FILENAME_OR_FPS_INFERENCE",
              "model_calls_made": 0, "cache_writes": 0, "certificate_writes": 0, "new_verified_count": 0}
    out = Path(output)
    write_once(out / "dataset_audit.json", report)
    write_once(out / "gt_isolation_audit.json", isolation_audit(projected))
    if prepare:
        if not report["count_matches"]:
            raise ClaimBankError("EXPECTED_DATASET_COUNT_MISMATCH")
        write_once(out / "runtime_manifest.jsonl", projected, jsonl=True)
        write_once(out / "runtime_manifest.provenance.json", {
            "format": "relive-claimgen-projection-v1", "dataset_sha256": source_hash,
            "runtime_sha256": digest(out / "runtime_manifest.jsonl"), "row_count": len(projected),
            "allowed_source_fields": ALLOWED_SOURCE_FIELDS,
            "forbidden_values_accessed": False, "source_container_deserialized": True,
            "evaluation_sidecar_created": False})
    return report


def isolation_audit(runtime: list[dict]) -> dict:
    for row in runtime:
        validate_runtime(row)
    return {"format": "relive-claimgen-gt-isolation-v1", "status": "PASS", "rows_checked": len(runtime),
            "gt_used": False, "allowed_source_fields": ALLOWED_SOURCE_FIELDS,
            "forbidden_source_values_accessed": False,
            "limitation": "Whitelist/dataflow audit; cannot prove public question text itself was never contaminated upstream."}


def load_runtime(path: str) -> list[dict]:
    runtime = rows(path)
    isolation_audit(runtime)
    if len({r["sample_key"] for r in runtime}) != len(runtime):
        raise ClaimBankError("DUPLICATE_RUNTIME_SAMPLE_KEY")
    provenance = load_json(str(Path(path).with_suffix(".provenance.json")))
    if provenance.get("runtime_sha256") != digest(path) or provenance.get("row_count") != len(runtime) or any(r["dataset_sha256"] != provenance.get("dataset_sha256") for r in runtime):
        raise ClaimBankError("RUNTIME_PROVENANCE_MISMATCH")
    return runtime


def sample_frames(row: dict, requirements: list[dict], max_frames: int,
                  data_root: str | None = None, path_map: dict | None = None) -> dict:
    if not 1 <= max_frames <= 16:
        raise ClaimBankError("MAX_FRAMES_MUST_BE_1_TO_16")
    unique = {}
    for i, p in enumerate(row["video"]):
        unique.setdefault(p, []).append(i)
    paths = list(unique)
    n = len(paths)
    temporal = any(r["evidence_type"] in {"TEMPORAL", "COMPOSITIONAL"} for r in requirements)
    def uniform(total, count):
        if count <= 1:
            return [0] if total else []
        return [round(i * (total - 1) / (count - 1)) for i in range(count)]
    count = min(max_frames, n)
    if temporal and count >= 4 and n > count:
        # Groups refer to adjacent supplied entries, NOT necessarily consecutive native frames.
        anchors = uniform(n, count // 2)
        indexes = {j for a in anchors for j in (a, a + 1 if a < n - 1 else a - 1)}
        for j in uniform(n, count):
            if len(indexes) >= count:
                break
            indexes.add(j)
        selected = sorted(indexes)[:count]
    else:
        selected = uniform(n, count)
    frames = []
    for j in selected:
        source_path = paths[j]
        positions = unique[source_path]
        p = resolve_path(source_path, data_root, path_map)
        entry = {"original_frame_path": source_path, "resolved_path": str(p),
                 "frame_order": positions[0], "source_list_indices": positions,
                 "sampled_video_frames": [row["sampled_video_frames"][i] for i in positions] if row["sampled_video_frames"] else [],
                 "timestamp_seconds": None, "timestamp_status": "UNRESOLVED_TIMEBASE",
                 "sha256": None, "image_status": "MISSING", "image_size": None}
        if p.is_file():
            entry["sha256"] = digest(p)
            try:
                with Image.open(p) as image:
                    image.verify()
                with Image.open(p) as image:
                    entry["image_size"] = list(image.size)
                entry["image_status"] = "AVAILABLE"
            except (OSError, ValueError):
                entry["image_status"] = "DECODE_FAILURE"
        frames.append(entry)
    return {"sample_key": row["sample_key"], "sampling_version": SAMPLER_VERSION,
            "strategy": "UNIFORM_ADJACENT_SUPPLIED_GROUPS" if temporal else "UNIFORM_COVERAGE",
            "original_count": len(row["video"]), "unique_path_count": n,
            "max_frames": max_frames, "selected_frames": frames,
            "coverage": {"selected_unique_count": len(frames), "unique_fraction": len(frames) / n if n else 0,
                         "source_order_range": [frames[0]["frame_order"], frames[-1]["frame_order"]] if frames else None,
                         "complete_observation": len(frames) == n and bool(n),
                         "seconds_range": None},
            "limitations": ["QUERY_TIME_MAPPING_UNRESOLVED", "SPARSE_FRAMES_DO_NOT_ESTABLISH_CONTINUOUS_ACTION_OR_EXHAUSTIVE_COVERAGE"],
            "supplemental_observation": {"enabled": False, "rounds_used": 0, "remaining_round_budget": 0}}
