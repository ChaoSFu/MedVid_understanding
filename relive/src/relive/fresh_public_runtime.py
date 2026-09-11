"""Frozen public-window runtime preparation for Phase 3.5/3-v3 development.

This module has no model, spatial-proposer, verifier, or certificate imports.
It binds a human-reviewed public frame window to a closed runtime only after
checking public-record identity, duplicates, claim scope, and frame paths.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from .audits import audit_runtime_imports
from .claim_scope import (CLAIM_SCOPE_ROUTER_VERSION, LOCAL_ATOMIC, claim_text_sha256,
                          route_claim_scope, stable_scope_hash)
from .data.medvidu import (ALLOWED_SOURCE_FIELDS, FORBIDDEN_SOURCE_CATEGORIES,
                           audit_frame_mapping, load_public_records)
from .data.schemas import FIELD_SOURCES, SCHEMA_VERSION, load_runtime
from .storage.artifacts import canonical_json


FRESH_SOURCE_FORMAT = "relive-phase35-fresh-source-manifest-v1"
FRESH_WINDOW_FORMAT = "relive-phase35-public-window-confirmation-v1"
FRESH_RUNTIME_FORMAT = "relive-phase35-frozen-public-window-runtime-v1"
EXCLUDED_SOURCE_RECORD_INDICES = frozenset({0, 1700, 2480, 2790, 5520})


class FreshRuntimeError(ValueError):
    pass


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise FreshRuntimeError(f"unreadable JSONL: {path}") from exc
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise FreshRuntimeError("expected a nonempty JSONL object sequence")
    return rows


def _validate_claim_rows(path: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    expected = {"source_record_index", "public_record_sha256", "target_claim"}
    result = []
    for row in rows:
        if set(row) != expected:
            raise FreshRuntimeError("fresh claims rows must contain exactly source_record_index, public_record_sha256, target_claim")
        index, digest, claim = row["source_record_index"], row["public_record_sha256"], row["target_claim"]
        if type(index) is not int or index < 0:
            raise FreshRuntimeError("source_record_index must be a nonnegative integer")
        if index in EXCLUDED_SOURCE_RECORD_INDICES:
            raise FreshRuntimeError(f"fresh cohort includes excluded development source_record_index: {index}")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise FreshRuntimeError("public_record_sha256 must be a SHA-256 digest")
        if not isinstance(claim, dict) or set(claim) != {"claim_id", "text"}:
            raise FreshRuntimeError("fresh target_claim must contain exactly claim_id and text")
        if not all(isinstance(claim[key], str) and claim[key].strip() for key in ("claim_id", "text")):
            raise FreshRuntimeError("fresh target_claim claim_id and text must be nonempty")
        result.append({"source_record_index": index, "public_record_sha256": digest,
                       "target_claim": {"claim_id": claim["claim_id"].strip(), "text": claim["text"].strip()}})
    if len({row["source_record_index"] for row in result}) != len(result):
        raise FreshRuntimeError("duplicate source_record_index in fresh claims")
    if len({row["public_record_sha256"] for row in result}) != len(result):
        raise FreshRuntimeError("duplicate public_record_sha256 in fresh claims")
    if len({row["target_claim"]["claim_id"] for row in result}) != len(result):
        raise FreshRuntimeError("duplicate claim_id in fresh claims")
    if len({row["target_claim"]["text"] for row in result}) != len(result):
        raise FreshRuntimeError("duplicate claim text in fresh claims")
    return result


def _validate_windows(path: Path, claim_ids: set[str]) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(path)
    expected = {"claim_id", "frozen_frame_orders", "human_public_visual_confirmation",
                "human_public_visual_confirmation_note"}
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if set(row) != expected:
            raise FreshRuntimeError("window confirmations require exactly claim_id, frozen_frame_orders, human_public_visual_confirmation, human_public_visual_confirmation_note")
        claim_id, orders = row["claim_id"], row["frozen_frame_orders"]
        if not isinstance(claim_id, str) or claim_id not in claim_ids or claim_id in result:
            raise FreshRuntimeError("window confirmation claim_id must bind exactly one fresh claim")
        if (not isinstance(orders, list) or not 1 <= len(orders) <= 3 or any(type(value) is not int or value < 0 for value in orders)
                or orders != sorted(orders) or len(set(orders)) != len(orders)
                or orders != list(range(orders[0], orders[0] + len(orders)))):
            raise FreshRuntimeError("frozen_frame_orders must be 1-3 unique consecutive nonnegative public frame orders")
        if row["human_public_visual_confirmation"] is not True:
            raise FreshRuntimeError("human_public_visual_confirmation must be true")
        note = row["human_public_visual_confirmation_note"]
        if not isinstance(note, str) or not note.strip() or len(note) > 1000:
            raise FreshRuntimeError("human_public_visual_confirmation_note must be nonempty and bounded")
        lower = note.casefold()
        if any(token in lower for token in ("bbox", "mask", "pixel", "roi", "ground truth", "gt")):
            raise FreshRuntimeError("human confirmation note must not contain ROI or GT material")
        result[claim_id] = {"frozen_frame_orders": orders,
                            "human_public_visual_confirmation": True,
                            "human_public_visual_confirmation_note": note.strip()}
    if set(result) != claim_ids:
        raise FreshRuntimeError("window confirmations must bind every fresh claim exactly once")
    return result


def _git_state(repo_root: Path) -> dict[str, Any]:
    import subprocess
    try:
        branch = subprocess.check_output(["git", "-C", str(repo_root), "branch", "--show-current"], text=True).strip()
        commit = subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(["git", "-C", str(repo_root), "status", "--short"], text=True).splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise FreshRuntimeError("unable to read git branch/commit/status") from exc
    if not branch or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise FreshRuntimeError("invalid git branch or commit")
    return {"branch": branch, "commit": commit, "worktree_status": status}


def _atomic_create(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FreshRuntimeError(f"refusing to overwrite frozen artifact: {path}")
    path.write_bytes(payload)


def _runtime_validation(runtime_payload: bytes, sidecar_payload: bytes) -> None:
    import tempfile
    with tempfile.TemporaryDirectory(prefix="relive-phase35-runtime-") as directory:
        runtime = Path(directory) / "runtime.jsonl"
        runtime.write_bytes(runtime_payload)
        Path(str(runtime) + ".provenance.json").write_bytes(sidecar_payload)
        try:
            load_runtime(runtime)
        except Exception as exc:
            raise FreshRuntimeError("generated frozen-window runtime failed closed-schema validation") from exc


def prepare_fresh_public_runtime(*, source_json: Path, frame_root: Path, source_prefix: str,
                                 fresh_claims: Path, window_confirmations: Path, config: Path,
                                 output_dir: Path, repo_root: Path) -> dict[str, Any]:
    """Freeze human-confirmed public windows before every model or spatial call."""
    if output_dir.exists():
        raise FreshRuntimeError("fresh runtime output directory must not exist")
    claims = _validate_claim_rows(fresh_claims)
    confirmations = _validate_windows(window_confirmations, {row["target_claim"]["claim_id"] for row in claims})
    records, source_sha = load_public_records(source_json)
    by_index = {record.source_record_index: record for record in records}
    selected = []
    for claim in claims:
        record = by_index.get(claim["source_record_index"])
        if record is None or record.public_record_sha256 != claim["public_record_sha256"]:
            raise FreshRuntimeError("fresh claim does not match the current public source projection")
        decision = route_claim_scope(claim["target_claim"]["text"], qa_type=record.qa_type,
                                     task_metadata={"dataset_name": record.dataset_name, "source_qa_type": record.qa_type})
        if decision.claim_scope != LOCAL_ATOMIC:
            raise FreshRuntimeError(f"fresh claim is not LOCAL_ATOMIC: {claim['target_claim']['claim_id']} ({decision.claim_scope})")
        orders = confirmations[claim["target_claim"]["claim_id"]]["frozen_frame_orders"]
        if orders[-1] >= len(record.video_paths):
            raise FreshRuntimeError("frozen frame order exceeds public record frame sequence")
        selected.append((record, claim, confirmations[claim["target_claim"]["claim_id"]], decision))
    if len({record.sample_id for record, *_ in selected}) != len(selected):
        raise FreshRuntimeError("duplicate public sample mapping in fresh cohort")
    path_audit, mapper = audit_frame_mapping([record for record, *_ in selected], source_prefix, frame_root)
    if path_audit["status"] != "PASS":
        raise FreshRuntimeError("public frame mapping audit failed; no fresh runtime was written")
    frozen_rows, runtime_rows = [], []
    for record, claim, confirmation, decision in sorted(selected, key=lambda item: item[0].source_record_index):
        orders = confirmation["frozen_frame_orders"]
        frames = []
        for order in orders:
            mapped = mapper.map(record.video_paths[order])
            if mapped.status != "PASS" or mapped.resolved_path is None:
                raise FreshRuntimeError("frozen public frame cannot be mapped")
            frame_id = f"{record.sample_id}:frame:{order:06d}"
            frames.append({"frame_id": frame_id, "path": mapped.resolved_path, "order": order})
        claim_value = {**claim["target_claim"], "time_scope": {"frame_ids": [frame["frame_id"] for frame in frames]}}
        metadata = {"runtime_adapter": "phase35_frozen_public_window_v1", "source_qa_type": record.qa_type,
                    "source_record_sha256": record.public_record_sha256, "nonofficial_protocol": True}
        if record.dataset_name is not None:
            metadata["dataset_name"] = record.dataset_name
        runtime_rows.append({"sample_id": record.sample_id, "task": "claim_verification", "question": record.question,
                             "frames": frames, "target_claim": claim_value, "metadata": metadata})
        frozen_rows.append({"format": FRESH_SOURCE_FORMAT, "source_record_index": record.source_record_index,
                            "sample_id": record.sample_id, "public_record_sha256": record.public_record_sha256,
                            "target_claim": claim["target_claim"], "claim_text_sha256": claim_text_sha256(claim["target_claim"]["text"]),
                            "frozen_frame_orders": orders, "frozen_frame_ids": [frame["frame_id"] for frame in frames],
                            "claim_scope": decision.claim_scope,
                            "single_roi_certificate_applicable": decision.single_roi_certificate_applicable,
                            "reason_code": decision.reason_code, "router_version": CLAIM_SCOPE_ROUTER_VERSION,
                            "gt_used": False, **confirmation})
    runtime_payload = b"".join((canonical_json(row) + "\n").encode("utf-8") for row in runtime_rows)
    runtime_sha = hashlib.sha256(runtime_payload).hexdigest()
    provenance = {"schema_version": SCHEMA_VERSION, "source_kind": "public_runtime", "runtime_sha256": runtime_sha,
                  "field_sources": FIELD_SOURCES}
    sidecar_payload = (canonical_json(provenance) + "\n").encode("utf-8")
    _runtime_validation(runtime_payload, sidecar_payload)
    source_manifest_payload = b"".join((canonical_json(row) + "\n").encode("utf-8") for row in frozen_rows)
    source_manifest_sha = hashlib.sha256(source_manifest_payload).hexdigest()
    git = _git_state(repo_root)
    source_audit = audit_runtime_imports()
    if source_audit["status"] != "PASS":
        raise FreshRuntimeError("runtime import audit failed")
    duplicate_audit = {"status": "PASS", "excluded_source_record_indices": sorted(EXCLUDED_SOURCE_RECORD_INDICES),
                       "selected_source_record_indices": [row["source_record_index"] for row in frozen_rows],
                       "duplicate_source_record_index_count": len(frozen_rows) - len({row["source_record_index"] for row in frozen_rows}),
                       "duplicate_public_record_sha256_count": len(frozen_rows) - len({row["public_record_sha256"] for row in frozen_rows}),
                       "duplicate_sample_id_count": len(frozen_rows) - len({row["sample_id"] for row in frozen_rows}),
                       "duplicate_claim_id_count": len(frozen_rows) - len({row["target_claim"]["claim_id"] for row in frozen_rows}),
                       "duplicate_claim_text_count": len(frozen_rows) - len({row["target_claim"]["text"] for row in frozen_rows})}
    isolation = {"audit_version": FRESH_RUNTIME_FORMAT, "status": "PASS", "classification": "phase35_fresh_local_atomic_spatial_preflight_not_official_benchmark",
                 "source_json_path": str(source_json), "source_json_sha256": source_sha,
                 "fresh_source_manifest_sha256": source_manifest_sha, "runtime_sha256": runtime_sha,
                 "allowed_source_fields": list(ALLOWED_SOURCE_FIELDS), "forbidden_source_categories_not_selected": list(FORBIDDEN_SOURCE_CATEGORIES),
                 "assistant_value_policy": "roles inspected; non-human conversation values were not accessed",
                 "source_metadata_policy": "source metadata object was not accessed or propagated",
                 "sampled_frame_policy": "paired length/type validation only; no value emitted as timestamp or ground truth",
                 "human_confirmation_policy": "stored only in frozen source manifest; absent from runtime and model inputs",
                 "prohibited_inputs_not_opened": ["reference_answer", "assistant_answer", "temporal_gt", "bbox_mask_gt", "struc_info", "RC_info", "evaluation_artifacts"]}
    report = {"status": "PASS", "phase": "ReliVE Phase 3.5 fresh LOCAL_ATOMIC public runtime preparation",
              "model_calls_made": 0, "cache_mutated": False, "spatial_proposer_called": False,
              "refinement_called": False, "certificate_called": False, "router_version": CLAIM_SCOPE_ROUTER_VERSION,
              "fresh_claims_input": str(fresh_claims), "fresh_claims_input_sha256": sha256_path(fresh_claims),
              "window_confirmations_input": str(window_confirmations), "window_confirmations_input_sha256": sha256_path(window_confirmations),
              "fresh_source_manifest": "fresh_source_manifest.jsonl", "fresh_source_manifest_sha256": source_manifest_sha,
              "runtime": "frozen_public_window.runtime.jsonl", "runtime_sha256": runtime_sha,
              "config": str(config), "config_sha256": sha256_path(config), "git": git,
              "duplicate_and_old_sample_exclusion_audit": duplicate_audit, "gt_isolation_audit": isolation,
              "runtime_import_audit": source_audit, "selected_count": len(frozen_rows),
              "claim_scope_distribution": dict(Counter(row["claim_scope"] for row in frozen_rows)),
              "frozen_before_model_inference": True, "push_status": "code must be committed and pushed by caller before selecting this external manifest"}
    output_dir.mkdir(parents=True)
    _atomic_create(output_dir / "fresh_source_manifest.jsonl", source_manifest_payload)
    _atomic_create(output_dir / "fresh_claims.jsonl", b"".join((canonical_json(row) + "\n").encode("utf-8") for row in claims))
    runtime_path = output_dir / "frozen_public_window.runtime.jsonl"
    _atomic_create(runtime_path, runtime_payload)
    _atomic_create(Path(str(runtime_path) + ".provenance.json"), sidecar_payload)
    _atomic_create(Path(str(runtime_path) + ".gt_isolation_audit.json"), (canonical_json(isolation) + "\n").encode("utf-8"))
    _atomic_create(output_dir / "fresh_runtime_preparation_report.json", (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    return report
