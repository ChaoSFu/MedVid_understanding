"""Static, portable contracts for protocol-development case drafts.

The module intentionally has no backend, cache, certificate, or media-model
imports.  It can recover a concatenated human draft only to preserve it in a
strict normalized form; recovery facts remain visible in provenance and block
readiness until humans complete the contract.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

from relive.storage.artifacts import canonical_json, stable_hash
from .task_selection import TALSelectionError, strict_json_loads


CASE_FORMAT = "relive-v2-protocol-dev-case-v1"
SCHEMA_VERSION = "1.0.0"
NORMALIZER_VERSION = "relive-v2-protocol-dev-normalizer-v1"
AUDIT_FORMAT = "relive-v2-protocol-dev-readiness-audit-v1"
HUMAN_QUEUE_FORMAT = "relive-v2-protocol-dev-human-completion-queue-v1"
HUMAN_COMPLETION_APPLICATION_FORMAT = "relive-v2-protocol-dev-human-completion-application-v1"
ORACLE_TEMPLATE_FORMAT = "relive-v2-protocol-dev-oracle-annotation-v1"
SOURCE_MATERIALIZATION_FORMAT = "relive-v2-protocol-dev-source-materialization-v1"
CLAIM_TYPES = frozenset({"SPATIAL_RELATION", "CONTACT_ACTION", "POSTCONDITION_PERSISTENCE"})
CASE_STATUSES = frozenset({"READY", "PENDING", "INVALID"})
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_MACHINE_ROOT = re.compile(r"^/(?:root/data|mnt/hdd3?|mnt/data)/[^/]+/")
_FULLWIDTH = {"，": ",", "：": ":", "；": ";", "（": "(", "）": ")"}
_ORACLE_TOKENS = ("oracle", "coordinate", "bbox", "mask", "tube", "annotation")
_SCHEMA_PATH = Path(__file__).resolve().parents[3] / "schemas" / "v2" / "relive_v2_protocol_dev_case.schema.json"


class ProtocolDevCaseError(ValueError):
    pass


def sha256_path(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _json_object(path: str | Path, code: str) -> dict[str, Any]:
    try:
        value = strict_json_loads(Path(path).read_bytes().decode("utf-8"), error_code=code)
    except (OSError, UnicodeDecodeError, TALSelectionError) as exc:
        raise ProtocolDevCaseError(f"{code}_INVALID") from exc
    if not isinstance(value, dict):
        raise ProtocolDevCaseError(f"{code}_OBJECT_REQUIRED")
    return value


def _write(path: Path, value: Any, *, jsonl: bool = False) -> None:
    if path.exists():
        raise ProtocolDevCaseError("IMMUTABLE_OUTPUT_EXISTS")
    path.parent.mkdir(parents=True, exist_ok=True)
    if jsonl:
        content = b"".join((canonical_json(item) + "\n").encode("utf-8") for item in value)
    else:
        content = (canonical_json(value) + "\n").encode("utf-8")
    path.write_bytes(content)


def _draft_objects(path: str | Path) -> tuple[list[tuple[dict[str, Any], str, list[str]]], list[str]]:
    """Recover only delimited draft objects, recording every non-JSON fact."""
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ProtocolDevCaseError("DRAFT_UNREADABLE") from exc
    issues: list[str] = []
    try:
        strict_json_loads(raw, error_code="DRAFT")
    except TALSelectionError:
        issues.append("DRAFT_NOT_SINGLE_STRICT_JSON_OBJECT")
    normalized = raw
    fullwidth = []
    for source, replacement in _FULLWIDTH.items():
        if source in normalized:
            fullwidth.append(source); normalized = normalized.replace(source, replacement)
    if fullwidth:
        issues.append("FULLWIDTH_PUNCTUATION:" + ",".join(fullwidth))
    decoder = json.JSONDecoder()
    offset = 0; objects: list[tuple[dict[str, Any], str, list[str]]] = []
    while offset < len(normalized):
        while offset < len(normalized) and normalized[offset].isspace():
            offset += 1
        if offset == len(normalized):
            break
        try:
            value, end = decoder.raw_decode(normalized, offset)
        except json.JSONDecodeError as exc:
            raise ProtocolDevCaseError("DRAFT_OBJECT_RECOVERY_FAILED") from exc
        if not isinstance(value, dict):
            raise ProtocolDevCaseError("DRAFT_OBJECT_REQUIRED")
        item_issues = list(issues)
        if len(objects) > 0:
            item_issues.append("CONCATENATED_TOP_LEVEL_OBJECT")
        objects.append((value, hashlib.sha256(raw[offset:end].encode("utf-8")).hexdigest(), item_issues))
        offset = end
    if len(objects) != 9:
        issues.append(f"DRAFT_CASE_COUNT:{len(objects)}")
    return objects, issues


def _case_id(raw: dict[str, Any], index: int) -> tuple[str, bool]:
    for key in ("case_id", "protocol_dev_case_id", "protocol_dev_id", "protocol_dev_prefilter_id"):
        if _text(raw.get(key)):
            return raw[key].strip(), key != "case_id"
    return f"protocol-dev-draft-{index:02d}", True


def _dataset_root_key(dataset: Any) -> str | None:
    if not _text(dataset):
        return None
    return re.sub(r"[^A-Za-z0-9]+", "_", dataset.strip()).upper().strip("_") + "_ROOT"


def _relative_pattern(value: Any, dataset: Any) -> tuple[str | None, str | None]:
    if not _text(value):
        return None, None
    path = value.strip()
    prefix = _MACHINE_ROOT.match(path)
    if prefix:
        return path[prefix.end():], "MACHINE_ROOT_STRIPPED"
    if path.startswith("/"):
        return None, "ABSOLUTE_PATH_NOT_PORTABLE"
    return path, None


def _list_of_ints(value: Any) -> list[int]:
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        return []
    return value


def _range(value: Any) -> list[int] | None:
    if isinstance(value, list) and len(value) == 2 and all(type(item) is int for item in value) and value[0] <= value[1]:
        return list(value)
    return None


def _window(value: Any) -> dict[str, Any] | None:
    if isinstance(value, list) and len(value) == 2 and all(isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(float(item)) for item in value) and value[0] <= value[1]:
        return {"start_seconds": float(value[0]), "end_seconds": float(value[1])}
    return None


def _first(raw: dict[str, Any], *paths: tuple[str, ...]) -> Any:
    for path in paths:
        value: Any = raw
        for key in path:
            if not isinstance(value, dict) or key not in value:
                value = None; break
            value = value[key]
        if value is not None:
            return value
    return None


def _entities(raw: dict[str, Any], claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    value = raw.get("entities")
    entities: list[dict[str, Any]] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                identifier = item.get("entity_id") or item.get("canonical_name")
                if _text(identifier):
                    entities.append({"entity_id": identifier, "definition": item.get("definition"), "source_object": item.get("source_object")})
    elif isinstance(value, dict):
        for role, item in sorted(value.items()):
            if isinstance(item, dict):
                identifier = item.get("canonical_name") or item.get("source_name") or role
                entities.append({"entity_id": identifier, "definition": item.get("definition"), "source_object": item.get("source_name")})
    if not entities:
        for claim in claims:
            for field in ("subject", "object"):
                value = claim.get(field)
                if _text(value) and not any(item["entity_id"] == value for item in entities):
                    entities.append({"entity_id": value, "definition": None, "source_object": None})
    return entities


def _claim(raw_value: Any, *, polarity: str, case_id: str, raw: dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw_value, dict):
        text = raw_value.get("text") or raw_value.get("claim_text")
        claim_id = raw_value.get("claim_id") if _text(raw_value.get("claim_id")) else f"{case_id}-{'T' if polarity == 'TRUE' else 'F'}"
        subject, predicate, object_ = raw_value.get("subject"), raw_value.get("predicate"), raw_value.get("object")
        source_match = raw_value.get("matched_to_claim_id")
        change = raw_value.get("changed_predicate") or raw_value.get("changed_field") or raw_value.get("counterfactual_change")
    else:
        text = raw_value if _text(raw_value) else None
        atomic = raw.get("atomic_predicate") if isinstance(raw.get("atomic_predicate"), dict) else {}
        claim_id = f"{case_id}-{'T' if polarity == 'TRUE' else 'F'}"
        subject = atomic.get("subject")
        object_ = atomic.get("object") or atomic.get("reference_object")
        predicate = atomic.get("true_state") if polarity == "TRUE" else atomic.get("false_state")
        source_match = None
        change = raw.get("changed_predicate") if polarity == "FALSE" else None
    return {"claim_id": claim_id, "polarity": polarity, "text": text, "subject": subject,
            "predicate": predicate, "object": object_, "temporal_scope_ref": "evidence_window",
            "matched_to_claim_id": source_match if polarity == "FALSE" else None,
            "changed_predicate": change if polarity == "FALSE" else None,
            "source_claim_id": raw_value.get("claim_id") if isinstance(raw_value, dict) else None}


def _role_names(raw: dict[str, Any]) -> list[str]:
    source = _first(raw, ("required_evidence_components",), ("required_evidence_roles",), ("required_roi_roles",), ("required_postcondition_components",))
    if not isinstance(source, list):
        return []
    result = []
    for item in source:
        value = item.get("role") if isinstance(item, dict) else item
        if _text(value): result.append(value)
    return sorted(set(result))


def normalize_draft(*, draft_path: str | Path, cases_dir: str | Path, manifest_path: str | Path) -> dict[str, Any]:
    """Write one strict, portable case file per draft object; never edit input."""
    cases_path = Path(cases_dir); manifest = Path(manifest_path)
    if cases_path.exists() or manifest.exists():
        raise ProtocolDevCaseError("IMMUTABLE_OUTPUT_EXISTS")
    objects, batch_issues = _draft_objects(draft_path)
    cases = []
    for index, (raw, raw_hash, parse_issues) in enumerate(objects, start=1):
        case_id, id_derived = _case_id(raw, index)
        dataset, video_id = raw.get("dataset_name"), raw.get("source_video_id")
        claim_type = raw.get("claim_type")
        pattern = _first(raw, ("evidence_window", "image_path_pattern"))
        relative_pattern, path_issue = _relative_pattern(pattern, dataset)
        sampled_ids = _first(raw, ("evidence_window", "sampled_frame_ids"), ("primary_postcondition_window", "sampled_frame_ids"), ("interaction_window", "sampled_frame_ids"))
        keyframes = _first(raw, ("evidence_window", "stg_anchor_frames"), ("oracle_annotation", "suggested_keyframes"), ("poststate_keyframes",))
        evidence_range = _first(raw, ("evidence_frame_range",), ("evidence_window", "native_frame_envelope_inclusive"), ("evidence_window", "native_frame_range_inclusive"), ("primary_postcondition_window", "native_frame_envelope_inclusive"), ("poststate_evidence_window_frames",))
        interaction_range = _first(raw, ("structured_track_frame_range",), ("interaction_window_frames",), ("interaction_window", "sampled_frame_range"))
        after_range = _first(raw, ("primary_postcondition_window", "native_frame_envelope_inclusive"), ("poststate_window_frames",), ("poststate_evidence_window_frames",))
        true_claim = _claim(raw.get("true_claim"), polarity="TRUE", case_id=case_id, raw=raw)
        false_claim = _claim(raw.get("false_claim"), polarity="FALSE", case_id=case_id, raw=raw)
        if false_claim["matched_to_claim_id"] is None:
            false_claim["matched_to_claim_id"] = true_claim["claim_id"]
        source_records = raw.get("source_records") if isinstance(raw.get("source_records"), list) else []
        record_hints = []
        for item in source_records:
            if isinstance(item, dict): record_hints.append({"role": item.get("role"), "source_record_uid_hint": item.get("source_record_uid"), "source_sample_id_hint": item.get("source_sample_id")})
        if not record_hints:
            uids = raw.get("source_record_uids") if isinstance(raw.get("source_record_uids"), list) else [raw.get("source_record_uid")]
            samples = raw.get("source_sample_ids") if isinstance(raw.get("source_sample_ids"), list) else [raw.get("source_sample_id")]
            record_hints = [{"role": None, "source_record_uid_hint": uid, "source_sample_id_hint": samples[pos] if pos < len(samples) else None} for pos, uid in enumerate(uids) if uid or (pos < len(samples) and samples[pos])]
        raw_decision = raw.get("human_review_decision") or _first(raw, ("human_review", "decision"))
        raw_rationale = raw.get("human_review_rationale") or _first(raw, ("human_review", "rationale"))
        normalized_decision = raw_decision if raw_decision in {"ADMIT", "REJECT", "RESERVE"} else "PENDING"
        oracle = raw.get("oracle_annotation") if isinstance(raw.get("oracle_annotation"), dict) else {}
        oracle_status = raw.get("oracle_evidence_status") or oracle.get("coordinates_status") or oracle.get("status") or "PENDING"
        coverage = _first(raw, ("evidence_window", "coverage_semantics"), ("relation_semantics", "frame_quantifier"), ("operational_semantics", "temporal_quantifier"))
        case = {"format": CASE_FORMAT, "schema_version": SCHEMA_VERSION, "case_id": case_id,
                "case_revision": "draft-normalized-r1", "split": "protocol_dev", "case_status": "PENDING",
                "source": {"dataset": dataset, "video_id": video_id, "source_task": raw.get("source_qa_type") or "PENDING",
                           "source_record_uid": None, "source_record_canonical_hashes": [], "source_record_hints": record_hints,
                           "source_sample_ids": [item.get("source_sample_id_hint") for item in record_hints if item.get("source_sample_id_hint")]},
                "task": {"claim_type": claim_type, "subtype": raw.get("contact_subtype") or raw.get("postcondition_subtype") or raw.get("geometry_type")},
                "frame_locator": {"dataset_root_key": _dataset_root_key(dataset), "relative_path_pattern": relative_pattern,
                                  "path_status": "DECLARED_UNVERIFIED" if relative_pattern else "PENDING",
                                  "frame_id_scheme": _first(raw, ("evidence_window", "authoritative_locator")) or "PENDING",
                                  "evidence_frame_range": _range(evidence_range), "sampled_frame_ids": _list_of_ints(sampled_ids),
                                  "keyframe_ids": _list_of_ints(keyframes), "coverage_semantics": coverage or "PENDING"},
                "temporal_spec": {"source_timebase": _first(raw, ("evidence_window", "canonical_timebase")) or raw.get("canonical_timebase") or "PENDING",
                                  "evidence_window": _window(_first(raw, ("evidence_window_absolute_seconds",), ("evidence_window", "absolute_seconds_inclusive"), ("evidence_window", "absolute_video_seconds_derived_at_15fps"), ("evidence_window", "absolute_video_seconds_derived"), ("primary_postcondition_window", "absolute_video_seconds_derived"))),
                                  "interaction_window": _window(_first(raw, ("interaction_window", "absolute_video_seconds"))),
                                  "after_window": _window(_first(raw, ("primary_postcondition_window", "absolute_video_seconds_derived"))),
                                  "interaction_frame_range": _range(interaction_range), "after_frame_range": _range(after_range)},
                "entities": _entities(raw, [true_claim, false_claim]), "claims": [true_claim, false_claim],
                "evidence_contract": {"required_evidence_roles": _role_names(raw), "geometry_type": raw.get("geometry_type"),
                                      "coverage_semantics": coverage or "PENDING", "automatic_oracle_access": "FORBIDDEN"},
                "human_review": {"decision": normalized_decision, "reviewer_id": _first(raw, ("human_review", "reviewer_id")),
                                 "rationale": raw_rationale, "source_decision_literal": raw_decision},
                "oracle_annotation": {"status": "PENDING" if oracle_status != "READY" else "READY", "artifact_ref": None,
                                      "coordinates_present": False, "automatic_certificate_access": "FORBIDDEN"},
                "provenance": {"normalizer_version": NORMALIZER_VERSION, "source_draft_sha256": sha256_path(draft_path),
                               "source_object_sha256": raw_hash, "source_object_index": index, "source_parse_issues": parse_issues,
                               "case_id_structurally_derived": id_derived, "machine_path_issue": path_issue,
                               "raw_oracle_status": oracle_status}}
        # A malformed explicit cross-reference is structural, not a missing
        # annotation.  Other incompleteness stays PENDING for human review.
        if "MATCHED_TO_CLAIM_ID_INVALID" in _schema_issues(case):
            case["case_status"] = "INVALID"
        cases.append(case)
    cases_path.mkdir(parents=True)
    for case in cases:
        _write(cases_path / f"{case['case_id']}.json", case)
    rows = [{"case_id": case["case_id"], "case_sha256": sha256_path(cases_path / f"{case['case_id']}.json"),
             "split": case["split"], "case_status": case["case_status"], "dataset": case["source"]["dataset"],
             "video_id": case["source"]["video_id"], "claim_type": case["task"]["claim_type"],
             "case_path": f"cases/{case['case_id']}.json"} for case in sorted(cases, key=lambda item: item["case_id"])]
    _write(manifest, rows, jsonl=True)
    return {"format": CASE_FORMAT, "status": "NORMALIZED_PENDING_STATIC_AUDIT", "case_count": len(cases),
            "cases_dir": str(cases_path), "manifest": str(manifest), "manifest_sha256": sha256_path(manifest),
            "source_draft_sha256": sha256_path(draft_path), "source_draft_issues": batch_issues,
            "model_calls_made": 0, "cache_writes": 0, "certificate_writes": 0, "new_verified_count": 0}


_TOP_LEVEL = frozenset({"format", "schema_version", "case_id", "case_revision", "split", "case_status", "source", "task", "frame_locator", "temporal_spec", "entities", "claims", "evidence_contract", "human_review", "oracle_annotation", "provenance"})


def _schema_issues(case: dict[str, Any]) -> list[str]:
    issues = []
    if set(case) != _TOP_LEVEL: issues.append("CASE_SCHEMA_KEYS_INVALID")
    if case.get("format") != CASE_FORMAT or case.get("schema_version") != SCHEMA_VERSION: issues.append("CASE_FORMAT_OR_VERSION_INVALID")
    if not _text(case.get("case_id")) or case.get("split") != "protocol_dev": issues.append("CASE_ID_OR_SPLIT_INVALID")
    if case.get("case_status") not in CASE_STATUSES: issues.append("CASE_STATUS_INVALID")
    task = case.get("task")
    try:
        schema = _json_object(_SCHEMA_PATH, "CASE_SCHEMA")
        one_of = {item["properties"]["task"]["properties"]["claim_type"]["const"] for item in schema["oneOf"]}
    except (KeyError, TypeError, ProtocolDevCaseError):
        one_of = set()
    if one_of != CLAIM_TYPES or not isinstance(task, dict) or task.get("claim_type") not in one_of:
        issues.append("CLAIM_TYPE_ONEOF_INVALID")
    source = case.get("source")
    if not isinstance(source, dict) or not _text(source.get("dataset")) or not _text(source.get("video_id")): issues.append("SOURCE_DATASET_OR_VIDEO_REQUIRED")
    locator = case.get("frame_locator")
    if not isinstance(locator, dict) or not _text(locator.get("dataset_root_key")): issues.append("FRAME_LOCATOR_SCHEMA_INVALID")
    if isinstance(locator, dict) and isinstance(locator.get("relative_path_pattern"), str) and locator["relative_path_pattern"].startswith("/"): issues.append("MACHINE_PATH_IN_CASE")
    claims = case.get("claims")
    if not isinstance(claims, list) or len(claims) != 2 or {item.get("polarity") for item in claims if isinstance(item, dict)} != {"TRUE", "FALSE"}: issues.append("TRUE_FALSE_CLAIM_PAIR_REQUIRED")
    else:
        true = next(item for item in claims if item["polarity"] == "TRUE"); false = next(item for item in claims if item["polarity"] == "FALSE")
        if false.get("matched_to_claim_id") != true.get("claim_id"): issues.append("MATCHED_TO_CLAIM_ID_INVALID")
        values = (true.get("subject"), false.get("subject"), true.get("object"), false.get("object"), true.get("predicate"), false.get("predicate"))
        if any(value is None for value in values): issues.append("CLAIM_COMPONENTS_PENDING")
        elif true["subject"] != false["subject"] or true["object"] != false["object"] or true["predicate"] == false["predicate"]: issues.append("FALSE_CLAIM_NOT_PREDICATE_ONLY")
    oracle = case.get("oracle_annotation")
    if not isinstance(oracle, dict) or oracle.get("automatic_certificate_access") != "FORBIDDEN" or oracle.get("coordinates_present") is not False: issues.append("ORACLE_ISOLATION_INVALID")
    return issues


def _read_cases(cases_dir: str | Path) -> list[tuple[Path, dict[str, Any]]]:
    directory = Path(cases_dir)
    paths = sorted(directory.glob("*.json"))
    if not paths: raise ProtocolDevCaseError("CASES_DIR_EMPTY")
    result = []
    for path in paths:
        result.append((path, _json_object(path, "CASE")))
    return result


def _data_roots(path: str | Path | None) -> dict[str, Path]:
    if path is None:
        raw = os.environ.get("RELIVE_V2_DATA_ROOTS_JSON")
        if not raw: return {}
        try: value = strict_json_loads(raw, error_code="DATA_ROOTS")
        except TALSelectionError as exc: raise ProtocolDevCaseError("DATA_ROOTS_INVALID") from exc
    else:
        value = _json_object(path, "DATA_ROOTS")
    if not isinstance(value, dict) or any(not _text(key) or not _text(item) for key, item in value.items()):
        raise ProtocolDevCaseError("DATA_ROOTS_SCHEMA_INVALID")
    return {key: Path(item) for key, item in value.items()}


def _oracle_files(directory: str | Path | None) -> set[str]:
    if directory is None: return set()
    root = Path(directory)
    if not root.exists(): return set()
    return {path.name for path in root.iterdir() if path.is_file()}


def _frame_issues(case: dict[str, Any], roots: dict[str, Path]) -> list[str]:
    locator = case["frame_locator"]
    evidence = locator.get("evidence_frame_range"); sampled = locator.get("sampled_frame_ids", []); keyframes = locator.get("keyframe_ids", [])
    issues = []
    if evidence:
        if any(frame < evidence[0] or frame > evidence[1] for frame in sampled): issues.append("SAMPLED_FRAME_OUTSIDE_EVIDENCE_WINDOW")
        if any(frame < evidence[0] or frame > evidence[1] for frame in keyframes): issues.append("KEYFRAME_OUTSIDE_EVIDENCE_WINDOW")
    temporal = case.get("temporal_spec", {})
    interaction, after = temporal.get("interaction_frame_range"), temporal.get("after_frame_range")
    if interaction and after and after[0] <= interaction[1]:
        issues.append("INTERACTION_POSTCONDITION_FRAME_RANGE_CONFLICT")
    coverage = locator.get("coverage_semantics")
    claim_text = " ".join(
        value for claim in case.get("claims", []) if isinstance(claim, dict)
        for value in (claim.get("claim_text"),) if isinstance(value, str)
    ).casefold()
    if coverage == "DISCRETE_SAMPLED_FRAMES" and re.search(r"\b(?:all|every)\s+(?:continuous\s+|native\s+)?frames?\b", claim_text):
        issues.append("SAMPLED_FRAME_CLAIM_CONTINUITY_CONFLICT")
    if roots and locator.get("relative_path_pattern") and locator.get("dataset_root_key") in roots:
        try:
            for frame in sampled or keyframes:
                path = roots[locator["dataset_root_key"]] / locator["relative_path_pattern"].format(frame=frame)
                if not path.is_file(): issues.append("FRAME_PATH_MISSING"); break
        except (ValueError, KeyError): issues.append("RELATIVE_PATH_PATTERN_INVALID")
    return issues


def _source_binding_issues(case: dict[str, Any]) -> list[str]:
    """Catch only explicit source/case inconsistencies; never infer source semantics."""
    source = case.get("source", {})
    video_id = source.get("video_id")
    hints = source.get("source_sample_ids", [])
    if not isinstance(video_id, str) or not isinstance(hints, list):
        return []
    for sample_id in hints:
        if not isinstance(sample_id, str):
            continue
        # Public sample IDs encode the video immediately before ``&&``.  A
        # discrepancy is a binding error; opaque IDs remain pending elsewhere.
        public_video = sample_id.split("&&", 1)[0]
        if public_video and public_video != sample_id and public_video != video_id:
            return ["SOURCE_RECORD_VIDEO_MISMATCH"]
    return []


def _runtime_view(case: dict[str, Any]) -> dict[str, Any]:
    return {"format": CASE_FORMAT, "schema_version": SCHEMA_VERSION, "case_id": case["case_id"], "case_revision": case["case_revision"],
            "split": case["split"], "case_status": case["case_status"], "source": case["source"], "task": case["task"],
            "frame_locator": case["frame_locator"], "temporal_spec": case["temporal_spec"], "entities": case["entities"],
            "claims": case["claims"], "evidence_contract": {key: value for key, value in case["evidence_contract"].items() if key != "automatic_oracle_access"}, "provenance": {"normalizer_version": case["provenance"]["normalizer_version"], "source_object_sha256": case["provenance"]["source_object_sha256"]}}


def assert_automatic_certificate_input_safe(value: Any) -> None:
    """The automatic path accepts runtime views only, never manual oracle data."""
    if isinstance(value, dict):
        for key, nested in value.items():
            if any(token in key.casefold() for token in _ORACLE_TOKENS):
                raise ProtocolDevCaseError("HUMAN_ORACLE_AUTOMATIC_CERTIFICATE_FORBIDDEN")
            assert_automatic_certificate_input_safe(nested)
    elif isinstance(value, list):
        for nested in value: assert_automatic_certificate_input_safe(nested)


def rebuild_manifest(*, cases_dir: str | Path, manifest_path: str | Path) -> dict[str, Any]:
    """Build a canonical manifest from case bytes without changing any case."""
    manifest = Path(manifest_path)
    if manifest.exists():
        raise ProtocolDevCaseError("IMMUTABLE_OUTPUT_EXISTS")
    rows = []
    for path, case in _read_cases(cases_dir):
        rows.append({"case_id": case["case_id"], "case_sha256": sha256_path(path), "split": case["split"],
                     "case_status": case["case_status"], "dataset": case["source"]["dataset"],
                     "video_id": case["source"]["video_id"], "claim_type": case["task"]["claim_type"],
                     "case_path": f"cases/{path.name}"})
    _write(manifest, sorted(rows, key=lambda item: item["case_id"]), jsonl=True)
    return {"format": CASE_FORMAT, "case_count": len(rows), "manifest": str(manifest),
            "manifest_sha256": sha256_path(manifest), "model_calls_made": 0, "cache_writes": 0,
            "certificate_writes": 0, "new_verified_count": 0}


def _public_qa_records(path: str | Path) -> list[dict[str, Any]]:
    """Read candidate records while projecting out non-human conversation values."""
    try:
        raw = Path(path).read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ProtocolDevCaseError("SOURCE_QA_UNREADABLE") from exc
    try:
        parsed = strict_json_loads(raw, error_code="SOURCE_QA")
        records = parsed.get("records") if isinstance(parsed, dict) and isinstance(parsed.get("records"), list) else parsed
        if not isinstance(records, list):
            raise ProtocolDevCaseError("SOURCE_QA_RECORD_LIST_REQUIRED")
    except TALSelectionError:
        records = []
        for line in raw.splitlines():
            if line.strip():
                try: records.append(strict_json_loads(line, error_code="SOURCE_QA_JSONL"))
                except TALSelectionError as exc: raise ProtocolDevCaseError("SOURCE_QA_INVALID") from exc
    if any(not isinstance(row, dict) for row in records):
        raise ProtocolDevCaseError("SOURCE_QA_RECORD_OBJECT_REQUIRED")
    return records


def _public_source_projection(row: dict[str, Any]) -> dict[str, Any]:
    """Hash only public binding fields, never assistant/answer/annotation values."""
    conversations = row.get("conversations")
    human_questions = []
    if isinstance(conversations, list):
        for turn in conversations:
            if isinstance(turn, dict) and turn.get("from") == "human" and isinstance(turn.get("value"), str):
                human_questions.append(turn["value"])
    projection = {key: row.get(key) for key in ("id", "sample_id", "source_record_uid", "qa_type", "dataset_name", "video", "sampled_video_frames")}
    projection["human_questions"] = human_questions
    return projection


def materialize_source_records(*, cases_dir: str | Path, qa_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Make an immutable public-field source binding map; it never edits cases."""
    output = Path(output_dir)
    if output.exists(): raise ProtocolDevCaseError("IMMUTABLE_OUTPUT_EXISTS")
    records = _public_qa_records(qa_path)
    candidates = []
    for index, row in enumerate(records):
        projection = _public_source_projection(row)
        public_hash = hashlib.sha256(canonical_json(projection).encode("utf-8")).hexdigest()
        identifiers = {value for value in (row.get("id"), row.get("sample_id"), row.get("source_record_uid")) if _text(value)}
        candidates.append({"source_record_index": index, "source_record_uid": "relive-v2-source-v1:" + public_hash,
                           "source_record_canonical_sha256": public_hash, "identifiers": sorted(identifiers)})
    rows = []
    for _, case in _read_cases(cases_dir):
        source = case["source"]; selectors = set(source.get("source_sample_ids", []))
        selectors.update(item.get("source_record_uid_hint") for item in source.get("source_record_hints", []) if isinstance(item, dict) and _text(item.get("source_record_uid_hint")))
        selectors.update(item.get("source_sample_id_hint") for item in source.get("source_record_hints", []) if isinstance(item, dict) and _text(item.get("source_sample_id_hint")))
        if _text(source.get("source_record_uid")): selectors.add(source["source_record_uid"])
        matches_by_selector = {selector: [item for item in candidates if item["source_record_uid"] == selector or selector in item["identifiers"]] for selector in selectors}
        matched_by_index = {item["source_record_index"]: item for matches in matches_by_selector.values() for item in matches}
        matched = [matched_by_index[index] for index in sorted(matched_by_index)]
        if not selectors: status, reason = "PENDING", "SOURCE_SELECTOR_MISSING"
        elif any(len(matches) > 1 for matches in matches_by_selector.values()): status, reason = "INVALID", "SOURCE_RECORD_AMBIGUOUS"
        elif any(len(matches) == 0 for matches in matches_by_selector.values()): status, reason = "PENDING", "SOURCE_RECORD_NOT_FOUND"
        else: status, reason = "MATERIALIZED", None
        rows.append({"case_id": case["case_id"], "status": status, "reason_code": reason,
                     "selectors": sorted(selectors), "source_records": [{key: item[key] for key in ("source_record_index", "source_record_uid", "source_record_canonical_sha256")} for item in matched]})
    output.mkdir(parents=True)
    _write(output / "protocol_dev_source_record_materialization.jsonl", sorted(rows, key=lambda item: item["case_id"]), jsonl=True)
    report = {"format": SOURCE_MATERIALIZATION_FORMAT, "status": "PASS", "qa_file_sha256": sha256_path(qa_path),
              "qa_record_count": len(records), "case_count": len(rows), "materialized_case_count": sum(row["status"] == "MATERIALIZED" for row in rows),
              "output_sha256": sha256_path(output / "protocol_dev_source_record_materialization.jsonl"),
              "assistant_or_gt_values_accessed": False, "model_calls_made": 0, "cache_writes": 0,
              "certificate_writes": 0, "new_verified_count": 0}
    _write(output / "protocol_dev_source_record_materialization_report.json", report)
    return report


def _human_queue_row(case: dict[str, Any]) -> dict[str, Any]:
    human = case["human_review"]
    true_claim = next(item for item in case["claims"] if item["polarity"] == "TRUE")
    false_claim = next(item for item in case["claims"] if item["polarity"] == "FALSE")
    components_missing = any(value is None for value in (true_claim.get("subject"), true_claim.get("predicate"), true_claim.get("object"), false_claim.get("subject"), false_claim.get("predicate"), false_claim.get("object"))) or not case["entities"] or not case["evidence_contract"]["required_evidence_roles"]
    suggested = case["frame_locator"].get("suggested_keyframes", [])
    return {"format": HUMAN_QUEUE_FORMAT, "case_id": case["case_id"], "claim_type": case["task"]["claim_type"],
            "true_claim": true_claim.get("text"), "matched_false_claim": false_claim.get("text"),
            "evidence_window": case["temporal_spec"]["evidence_window"], "evidence_frame_range": case["frame_locator"]["evidence_frame_range"],
            "allowed_decision_values": ["ADMIT", "REJECT", "RESERVE"], "current_decision": human.get("decision"),
            "current_rationale": human.get("rationale"), "human_required": {
                "human_review.decision": human.get("decision") == "PENDING",
                "human_review.reviewer_id": not _text(human.get("reviewer_id")),
                "human_review.rationale": not _text(human.get("rationale")),
                "entity_and_claim_component_confirmation": components_missing,
                "exact_keyframe_selection": not bool(case["frame_locator"]["keyframe_ids"] or suggested),
                "oracle_coordinates": case["oracle_annotation"]["status"] != "READY"},
            "current_keyframe_ids": case["frame_locator"]["keyframe_ids"], "suggested_keyframes": suggested,
            "keyframe_selection_constraint": case.get("provenance", {}).get("author_confirmed_keyframe_constraint"), "current_entities": case["entities"],
            "current_required_evidence_roles": case["evidence_contract"]["required_evidence_roles"],
            "machine_derived_not_human_input": ["source_record_canonical_hashes", "derived_timestamps", "frame_counts", "normalized_paths", "manifest_hashes"]}


def prepare_human_completion(*, cases_dir: str | Path, reviews_dir: str | Path, oracle_dir: str | Path) -> dict[str, Any]:
    """Write separate human-only queue and coordinate-free oracle templates."""
    reviews, oracle = Path(reviews_dir), Path(oracle_dir)
    queue_path, markdown_path = reviews / "protocol_dev_human_completion_queue.jsonl", reviews / "protocol_dev_human_completion_queue.md"
    if queue_path.exists() or markdown_path.exists() or oracle.exists(): raise ProtocolDevCaseError("IMMUTABLE_OUTPUT_EXISTS")
    cases = [case for _, case in _read_cases(cases_dir)]
    rows = [_human_queue_row(case) for case in sorted(cases, key=lambda item: item["case_id"])]
    _write(queue_path, rows, jsonl=True)
    oracle.mkdir(parents=True)
    for case in sorted(cases, key=lambda item: item["case_id"]):
        _write(oracle / f"{case['case_id']}.oracle.json", {"format": ORACLE_TEMPLATE_FORMAT, "case_id": case["case_id"],
               "status": "PENDING", "runtime_exposed": False, "automatic_certificate_access": "FORBIDDEN", "artifact_type": "PENDING", "coordinates": None})
    lines = ["# Protocol-dev human completion queue", "", "Only fields marked `true` under `human_required` need a human answer. Source hashes, paths, timestamps, counts, and manifest hashes are machine/environment work.", ""]
    for row in rows:
        required = [key for key, value in row["human_required"].items() if value]
        lines.extend([f"## {row['case_id']}", "", f"- True claim: {row['true_claim']}", f"- Matched false claim: {row['matched_false_claim']}", f"- Evidence window: {row['evidence_window']}", f"- Allowed decision: {', '.join(row['allowed_decision_values'])}", f"- Current rationale: {row['current_rationale'] or 'PENDING'}", f"- Needed: {', '.join(required) or 'none'}", ""])
    _write(markdown_path, "\n".join(lines))
    return {"format": HUMAN_QUEUE_FORMAT, "status": "PASS", "case_count": len(rows), "queue_sha256": sha256_path(queue_path),
            "oracle_template_count": len(rows), "model_calls_made": 0, "cache_writes": 0, "certificate_writes": 0, "new_verified_count": 0}


def _completion_rows(path: str | Path) -> dict[str, dict[str, Any]]:
    """Read a human-completion queue without accepting oracle payloads."""
    try:
        raw = Path(path).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ProtocolDevCaseError("HUMAN_COMPLETION_QUEUE_INVALID") from exc
    rows: dict[str, dict[str, Any]] = {}
    forbidden = {"coordinates", "mask", "bbox", "tube", "oracle_annotation"}
    for line in raw:
        if not line.strip():
            continue
        try:
            row = strict_json_loads(line, error_code="HUMAN_COMPLETION_QUEUE")
        except TALSelectionError as exc:
            raise ProtocolDevCaseError("HUMAN_COMPLETION_QUEUE_INVALID") from exc
        if not isinstance(row, dict) or row.get("format") != HUMAN_QUEUE_FORMAT or not _text(row.get("case_id")):
            raise ProtocolDevCaseError("HUMAN_COMPLETION_QUEUE_SCHEMA_INVALID")
        if forbidden.intersection(row):
            raise ProtocolDevCaseError("HUMAN_ORACLE_COMPLETION_QUEUE_FORBIDDEN")
        case_id = row["case_id"]
        if case_id in rows:
            raise ProtocolDevCaseError("HUMAN_COMPLETION_QUEUE_DUPLICATE_CASE")
        rows[case_id] = row
    if not rows:
        raise ProtocolDevCaseError("HUMAN_COMPLETION_QUEUE_EMPTY")
    return rows


def _applyable_keyframes(case: dict[str, Any], row: dict[str, Any]) -> tuple[list[int] | None, str | None]:
    """Accept human frame IDs only when their coordinate system is already bound.

    A queue can preserve a human's proposed IDs while this function keeps them
    out of a case if no native frame envelope or existing locator binds them.
    This prevents seconds, source IDs, and file ordinals from being conflated.
    """
    frames = row.get("current_keyframe_ids")
    if not isinstance(frames, list) or any(type(frame) is not int for frame in frames) or len(set(frames)) != len(frames):
        return None, "KEYFRAME_SELECTION_INVALID"
    if not frames:
        return None, "KEYFRAME_SELECTION_MISSING"
    locator, temporal = case["frame_locator"], case["temporal_spec"]
    existing = locator.get("keyframe_ids", [])
    if existing:
        return frames, None
    ranges = [locator.get("evidence_frame_range"), temporal.get("interaction_frame_range"), temporal.get("after_frame_range")]
    if any(isinstance(bounds, list) and len(bounds) == 2 and all(isinstance(value, int) for value in bounds)
           and all(bounds[0] <= frame <= bounds[1] for frame in frames) for bounds in ranges):
        return frames, None
    return None, "KEYFRAME_REFERENCE_UNBOUND"


def apply_human_completion(*, cases_dir: str | Path, completion_queue: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Create immutable reviewed case copies from a completed human queue.

    The source cases and queue remain unchanged.  Oracle coordinates are not a
    legal input here; timestamps and frame references retain their existing
    fail-closed status until separately bound.
    """
    output = Path(output_dir)
    if output.exists():
        raise ProtocolDevCaseError("IMMUTABLE_OUTPUT_EXISTS")
    source = _read_cases(cases_dir); by_id = {case["case_id"]: (path, case) for path, case in source}
    rows = _completion_rows(completion_queue)
    if set(rows) != set(by_id):
        raise ProtocolDevCaseError("HUMAN_COMPLETION_QUEUE_CASE_SET_MISMATCH")
    completed = output / "cases"; completed.mkdir(parents=True)
    application_rows = []
    source_hashes = {}
    for case_id in sorted(by_id):
        path, original = by_id[case_id]; row = rows[case_id]
        decision, reviewer, rationale = row.get("current_decision"), row.get("reviewer_id"), row.get("current_rationale")
        if decision not in {"ADMIT", "REJECT", "RESERVE"} or not _text(reviewer) or not _text(rationale):
            raise ProtocolDevCaseError("HUMAN_COMPLETION_REQUIRED_FIELD_MISSING")
        value = json.loads(canonical_json(original))
        keyframes, keyframe_reason = _applyable_keyframes(value, row)
        value["human_review"] = {**value["human_review"], "decision": decision, "reviewer_id": reviewer.strip(), "rationale": rationale.strip(),
                                 "source_decision_literal": "HUMAN_COMPLETION_QUEUE_APPLIED"}
        if keyframes is not None:
            value["frame_locator"]["keyframe_ids"] = keyframes
        previous = value.get("case_revision", "draft-normalized-r1")
        value["case_revision"] = previous + "+human-completion-r1"
        value["case_status"] = "PENDING"
        value["provenance"] = {**value["provenance"], "human_completion_queue_sha256": sha256_path(completion_queue),
                               "human_completion_application": {"decision": decision, "reviewer_id": reviewer.strip(),
                                   "keyframes_applied": keyframes is not None, "keyframe_status": "APPLIED" if keyframes is not None else "PENDING",
                                   "keyframe_reason": keyframe_reason}}
        destination = completed / path.name
        _write(destination, value)
        source_hashes[case_id] = sha256_path(path)
        application_rows.append({"case_id": case_id, "status": "APPLIED" if keyframes is not None else "APPLIED_WITH_PENDING_FRAME_BINDING",
                                 "source_case_sha256": source_hashes[case_id], "reviewed_case_sha256": sha256_path(destination),
                                 "keyframes_applied": keyframes is not None, "keyframe_reason": keyframe_reason,
                                 "oracle_coordinates_accepted": False})
    manifest_result = rebuild_manifest(cases_dir=completed, manifest_path=output / "protocol_dev_manifest.jsonl")
    _write(output / "human_completion_application.jsonl", application_rows, jsonl=True)
    report = {"format": HUMAN_COMPLETION_APPLICATION_FORMAT, "status": "PASS", "case_count": len(application_rows),
              "completion_queue_sha256": sha256_path(completion_queue), "source_case_sha256": source_hashes,
              "application_rows_sha256": sha256_path(output / "human_completion_application.jsonl"),
              "reviewed_manifest_sha256": manifest_result["manifest_sha256"],
              "keyframe_binding_pending_count": sum(not row["keyframes_applied"] for row in application_rows),
              "model_calls_made": 0, "cache_writes": 0, "certificate_writes": 0, "new_verified_count": 0}
    _write(output / "human_completion_application_report.json", report)
    return {**report, "reviewed_cases_dir": str(completed)}


def resolve_frame_patterns(*, cases_dir: str | Path, data_roots: str | Path, output_dir: str | Path,
                           strict: bool = False) -> dict[str, Any]:
    """Derive a pattern only from an unambiguous set of existing frame files.

    The resolver produces a reviewable proposal and never mutates a frozen case.
    """
    output = Path(output_dir)
    if output.exists(): raise ProtocolDevCaseError("IMMUTABLE_OUTPUT_EXISTS")
    roots = _data_roots(data_roots); rows = []
    for _, case in _read_cases(cases_dir):
        locator, source = case["frame_locator"], case["source"]
        root = roots.get(locator.get("dataset_root_key")); frames = locator.get("keyframe_ids") or locator.get("sampled_frame_ids") or locator.get("suggested_keyframes")
        if root is None: status, reason, pattern = "PENDING", "DATASET_ROOT_MISSING", None
        elif not frames: status, reason, pattern = "PENDING", "FRAME_REFERENCE_MISSING", None
        elif not root.is_dir(): status, reason, pattern = "PENDING", "DATASET_ROOT_UNREADABLE", None
        else:
            candidates: dict[int, list[Path]] = {frame: [] for frame in frames}
            for path in root.rglob("*"):
                if not path.is_file() or source["video_id"] not in path.parts: continue
                if path.stem.isdigit() and int(path.stem) in candidates: candidates[int(path.stem)].append(path)
            if any(len(paths) != 1 for paths in candidates.values()): status, reason, pattern = "PENDING", "FRAME_PATH_NOT_UNIQUE", None
            else:
                selected = [paths[0] for _, paths in sorted(candidates.items())]
                parents, suffixes, widths = {path.parent for path in selected}, {path.suffix for path in selected}, {len(path.stem) for path in selected}
                if len(parents) != 1 or len(suffixes) != 1 or len(widths) != 1:
                    status, reason, pattern = "PENDING", "FRAME_PATTERN_NOT_UNIFORM", None
                else:
                    parent, suffix, width = selected[0].parent, selected[0].suffix, next(iter(widths))
                    pattern = (parent.relative_to(root).as_posix() + "/" if parent != root else "") + "{frame:0" + str(width) + "d}" + suffix
                    if all((root / pattern.format(frame=frame)).is_file() for frame in frames): status, reason = "RESOLVED", None
                    else: status, reason, pattern = "PENDING", "FRAME_PATTERN_RECHECK_FAILED", None
        rows.append({"case_id": case["case_id"], "status": status, "reason_code": reason,
                     "dataset_root_key": locator.get("dataset_root_key"), "derived_relative_path_pattern": pattern,
                     "frame_ids_checked": frames or [], "frame_reference_kind": "KEYFRAME" if locator.get("keyframe_ids") else ("SAMPLED" if locator.get("sampled_frame_ids") else "SUGGESTED")})
    output.mkdir(parents=True)
    path = output / "protocol_dev_frame_pattern_resolution.jsonl"
    _write(path, sorted(rows, key=lambda item: item["case_id"]), jsonl=True)
    report = {"format": "relive-v2-protocol-dev-frame-pattern-resolution-v1", "status": "PASS" if all(row["status"] == "RESOLVED" for row in rows) else "PENDING", "case_count": len(rows),
              "resolved_count": sum(row["status"] == "RESOLVED" for row in rows), "output_sha256": sha256_path(path),
              "model_calls_made": 0, "cache_writes": 0, "certificate_writes": 0, "new_verified_count": 0}
    _write(output / "protocol_dev_frame_pattern_resolution_report.json", report)
    if strict and report["status"] != "PASS":
        raise ProtocolDevCaseError("FRAME_PATH_BINDING_INCOMPLETE")
    return report


def ego_candidate_path_resolution(*, cases_dir: str | Path, data_roots: str | Path, output_dir: str | Path,
                                  case_id: str = "PD-C-EGO-01", render_previews: bool = False) -> dict[str, Any]:
    """Enumerate every session-frame candidate and leave directory choice to a human."""
    output = Path(output_dir)
    if output.exists(): raise ProtocolDevCaseError("IMMUTABLE_OUTPUT_EXISTS")
    cases = {case["case_id"]: case for _, case in _read_cases(cases_dir)}
    if case_id not in cases: raise ProtocolDevCaseError("EGO_CASE_NOT_FOUND")
    case, roots = cases[case_id], _data_roots(data_roots); locator, source = case["frame_locator"], case["source"]
    root, frames = roots.get(locator["dataset_root_key"]), locator.get("keyframe_ids")
    rows = []; parents: dict[str, set[int]] = {}
    if root is not None and root.is_dir():
        for frame in frames:
            # The session string and frame number must both occur in the file
            # stem; no ordinal or first-match selection is used.
            pattern = re.compile(r"(?:^|_)" + re.escape(source["video_id"]) + r"_0*" + str(frame) + r"$")
            candidates = sorted(path for path in root.rglob("*") if path.is_file() and pattern.search(path.stem))
            for path in candidates:
                relative = path.relative_to(root).as_posix(); parent = path.parent.relative_to(root).as_posix()
                parents.setdefault(parent, set()).add(frame)
                rows.append({"format": "relive-v2-protocol-dev-candidate-path-resolution-v1", "case_id": case_id,
                             "video_id": source["video_id"], "frame_id": frame, "candidate_relative_path": relative,
                             "candidate_parent_directory": parent, "file_sha256": sha256_path(path)})
    common = sorted(parent for parent, seen in parents.items() if set(frames).issubset(seen))
    output.mkdir(parents=True)
    _write(output / "candidate_path_resolution_queue.jsonl", rows, jsonl=True)
    preview_status = "NOT_REQUESTED"
    if render_previews and common:
        try:
            from PIL import Image, ImageDraw
            preview_dir = output / "previews"; preview_dir.mkdir()
            for parent in common:
                selected = []
                for frame in (frames[0], frames[-1]):
                    pattern = re.compile(r"(?:^|_)" + re.escape(source["video_id"]) + r"_0*" + str(frame) + r"$")
                    choices = [path for path in Path(root).rglob("*") if path.is_file() and path.parent.relative_to(root).as_posix() == parent and pattern.search(path.stem)]
                    if len(choices) != 1: raise StopIteration
                    selected.append(choices[0])
                images = [Image.open(path).convert("RGB") for path in selected]
                width = max(image.width for image in images); height = max(image.height for image in images)
                sheet = Image.new("RGB", (width * 2, height + 24), "white")
                for index, image in enumerate(images): sheet.paste(image, (index * width, 24))
                draw = ImageDraw.Draw(sheet); draw.text((0, 0), f"{parent}: {frames[0]} | {frames[-1]}", fill="black")
                digest = hashlib.sha256(parent.encode("utf-8")).hexdigest()[:12]
                sheet.save(preview_dir / f"{case_id}_{digest}_first_last.png")
            preview_status = "RENDERED"
        except (ImportError, OSError, StopIteration): preview_status = "PREVIEW_RENDERER_OR_SOURCE_UNAVAILABLE"
    report = {"format": "relive-v2-protocol-dev-candidate-path-resolution-v1", "status": "PENDING_HUMAN_DIRECTORY_CONFIRMATION",
              "case_id": case_id, "frame_ids": frames, "candidate_count": len(rows), "common_candidate_directories": common,
              "previews": preview_status, "queue_sha256": sha256_path(output / "candidate_path_resolution_queue.jsonl"),
              "model_calls_made": 0, "cache_writes": 0, "certificate_writes": 0, "new_verified_count": 0}
    _write(output / "candidate_path_resolution_report.json", report)
    return report


def copesd_timebase_audit(*, cases_dir: str | Path, data_roots: str | Path, output_dir: str | Path,
                         timebase_manifest: str | Path | None, public_frame_registry: str | Path | None = None,
                         case_id: str = "PD-S-08") -> dict[str, Any]:
    """Map CoPESD images only through an externally documented timebase manifest."""
    output = Path(output_dir)
    if output.exists(): raise ProtocolDevCaseError("IMMUTABLE_OUTPUT_EXISTS")
    cases = {case["case_id"]: case for _, case in _read_cases(cases_dir)}
    if case_id not in cases: raise ProtocolDevCaseError("COPESD_CASE_NOT_FOUND")
    case, roots = cases[case_id], _data_roots(data_roots); root = roots.get(case["frame_locator"]["dataset_root_key"])
    registry = _json_object(public_frame_registry, "COPESD_PUBLIC_FRAME_REGISTRY") if public_frame_registry else None
    if registry is not None and (registry.get("dataset") != case["source"]["dataset"] or registry.get("video_id") != case["source"]["video_id"] or registry.get("dataset_root_key") != case["frame_locator"]["dataset_root_key"]):
        raise ProtocolDevCaseError("COPESD_PUBLIC_FRAME_REGISTRY_BINDING_INVALID")
    public_ids = registry.get("public_frame_file_ids", []) if isinstance(registry, dict) else []
    pattern = registry.get("relative_path_pattern") if isinstance(registry, dict) else None
    if registry is not None and (not isinstance(public_ids, list) or any(type(item) is not int for item in public_ids) or not isinstance(pattern, str)):
        raise ProtocolDevCaseError("COPESD_PUBLIC_FRAME_REGISTRY_SCHEMA_INVALID")
    if timebase_manifest is None:
        mappings = [{"case_id": case_id, "video_id": case["source"]["video_id"], "public_file_number": frame,
                     "timestamp_seconds": None, "relative_path": pattern.format(frame=frame) if pattern else None,
                     "file_exists": bool(root and pattern and (root / pattern.format(frame=frame)).is_file()),
                     "timebase_status": "UNRESOLVED_DOCUMENTED_TIMEBASE_REQUIRED"} for frame in public_ids]
        status, reason = "PENDING", "TIMEBASE_PROVENANCE_REQUIRED"
    else:
        try:
            raw_rows = [strict_json_loads(line, error_code="COPESD_TIMEBASE") for line in Path(timebase_manifest).read_text(encoding="utf-8").splitlines() if line.strip()]
        except (OSError, UnicodeDecodeError, TALSelectionError) as exc: raise ProtocolDevCaseError("COPESD_TIMEBASE_INVALID") from exc
        interval = case["temporal_spec"]["evidence_window"]; mappings = []
        for row in raw_rows:
            if not isinstance(row, dict) or row.get("video_id") != case["source"]["video_id"]: continue
            timestamp, frame = row.get("timestamp_seconds"), row.get("frame_id")
            relative = row.get("relative_path")
            if not isinstance(timestamp, (int, float)) or not isinstance(frame, int) or not isinstance(relative, str) or not _text(row.get("timebase_source")) or not _HEX.fullmatch(str(row.get("source_reference_sha256", ""))): continue
            if interval and interval["start_seconds"] <= float(timestamp) <= interval["end_seconds"]:
                path = root / relative if root is not None else None
                mappings.append({"case_id": case_id, "video_id": case["source"]["video_id"], "timestamp_seconds": float(timestamp), "frame_id": frame,
                                 "relative_path": relative, "file_exists": bool(path and path.is_file()), "timebase_status": "VALIDATED"})
        status, reason = ("READY_FOR_HUMAN_KEYFRAME_SELECTION", None) if mappings and all(item["file_exists"] for item in mappings) else ("PENDING", "NO_VALIDATED_IMAGE_TIME_MAPPING")
    output.mkdir(parents=True)
    _write(output / "copesd_image_time_mapping.jsonl", mappings, jsonl=True)
    queue = {"format": "relive-v2-protocol-dev-copesd-keyframe-selection-v1", "case_id": case_id, "status": status,
             "reason_code": reason, "evidence_interval": case["temporal_spec"]["evidence_window"],
             "instruction": "Select exact keyframes only from the validated mapping; source sample-ID numbers are not image-frame identifiers.", "mapping_sha256": sha256_path(output / "copesd_image_time_mapping.jsonl"), "public_frame_registry_sha256": sha256_path(public_frame_registry) if public_frame_registry else None}
    _write(output / "copesd_human_keyframe_selection_queue.json", queue)
    return {**queue, "model_calls_made": 0, "cache_writes": 0, "certificate_writes": 0, "new_verified_count": 0}


def prepare_frame_binding_completion_queue(*, cases_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Write only the human/environment decisions still needed for four blocked cases."""
    output = Path(output_dir)
    if output.exists(): raise ProtocolDevCaseError("IMMUTABLE_OUTPUT_EXISTS")
    cases = {case["case_id"]: case for _, case in _read_cases(cases_dir)}
    expected = {"PD-C-EGO-01", "PD-P-CholecT50-VID68-GBPACK-01", "PD-S-05", "PD-S-08"}
    if not expected.issubset(cases): raise ProtocolDevCaseError("FRAME_BINDING_CASES_MISSING")
    s05 = cases["PD-S-05"]
    s05_suggested = s05["frame_locator"].get("suggested_keyframes", [])
    s05_row = {"format": "relive-v2-protocol-dev-frame-binding-completion-v1", "case_id": "PD-S-05",
               "status": "PENDING_MACHINE_FRAME_PATH_DERIVATION" if s05_suggested else "PENDING_AUTHOR_CONFIRMATION_OF_STABLE_WINDOW",
               "human_required": [] if s05_suggested else ["Confirm whether the entire frozen evidence range 18001-18501 is an approved stable evidence window before suggested keyframes can enter the canonical case."],
               "environment_required": ["CHOLECTRACK20_ROOT"],
               "machine_required": ["Derive a relative path using the author-confirmed suggested keyframes." if s05_suggested else "Only then derive path using [18001,18101,18201,18301,18401,18501]."],
               "suggested_keyframes": s05_suggested, "proposed_selection_rule": "UNIFORM_KEYFRAMES_WITHIN_HUMAN_APPROVED_EVIDENCE_WINDOW"}
    s08 = cases["PD-S-08"]
    s08_locator, s08_temporal = s08["frame_locator"], s08["temporal_spec"]
    s08_timebase_bound = (s08_temporal.get("source_timebase") != "PENDING" and
                           isinstance(s08_locator.get("evidence_frame_range"), list) and
                           bool(s08_locator.get("sampled_frame_ids")))
    s08_row = {"format": "relive-v2-protocol-dev-frame-binding-completion-v1", "case_id": "PD-S-08",
               "status": "PENDING_HUMAN_KEYFRAME_SELECTION_FROM_BOUND_MAPPING" if s08_timebase_bound else "PENDING_VALIDATED_COPESD_TIMEBASE",
               "human_required": ["Choose exact keyframes only from the listed frame/time mapping."] if s08_timebase_bound else ["Choose keyframes only after reviewing the validated image/time/file mapping."],
               "environment_required": ["COPESD_ROOT"],
               "machine_required": ["Verify selected frame paths and their SHA-256 values before updating a canonical case."],
               "evidence_interval_seconds": s08_temporal["evidence_window"],
               "evidence_frame_range": s08_locator.get("evidence_frame_range"),
               "allowed_frame_ids": s08_locator.get("sampled_frame_ids", [])}
    rows = [
        {"format": "relive-v2-protocol-dev-frame-binding-completion-v1", "case_id": "PD-C-EGO-01", "status": "PENDING_HUMAN_DIRECTORY_CONFIRMATION",
         "human_required": ["Choose one common 06_1 candidate directory after reviewing every enumerated frame path and preview."], "environment_required": ["EGOSURGERY_ROOT"], "machine_required": ["Run candidate path inspection; do not write the derived queue."], "frame_ids": cases["PD-C-EGO-01"]["frame_locator"]["keyframe_ids"]},
        {"format": "relive-v2-protocol-dev-frame-binding-completion-v1", "case_id": "PD-P-CholecT50-VID68-GBPACK-01", "status": "PENDING_HUMAN_POSTSTATE_KEYFRAMES",
         "human_required": ["Select at least three post-state keyframes in inclusive range 1683-1691.", "Each selected frame must clearly show gallbladder, specimen bag, and containment after insertion and bag closing."], "environment_required": ["CHOLECT50_ROOT"], "machine_required": ["Derive a relative path only after human keyframes are in the canonical case."], "strongest_poststate_frame": 1687, "forbidden_frame_range": [1692, None]},
        s05_row,
        s08_row,
    ]
    output.mkdir(parents=True)
    _write(output / "protocol_dev_frame_binding_completion_queue.jsonl", rows, jsonl=True)
    lines = ["# Protocol-dev frame-path binding completion queue", ""]
    for row in rows:
        lines.extend([f"## {row['case_id']}", "", f"Status: `{row['status']}`", "", "Human required:"] + [f"- {item}" for item in row["human_required"]] + [""])
    _write(output / "protocol_dev_frame_binding_completion_queue.md", "\n".join(lines))
    return {"format": "relive-v2-protocol-dev-frame-binding-completion-v1", "status": "PASS", "queue_count": len(rows),
            "queue_sha256": sha256_path(output / "protocol_dev_frame_binding_completion_queue.jsonl"),
            "model_calls_made": 0, "cache_writes": 0, "certificate_writes": 0, "new_verified_count": 0}


def audit_cases(*, cases_dir: str | Path, oracle_dir: str | Path | None, data_roots: str | Path | None,
                output_dir: str | Path, strict: bool = False) -> dict[str, Any]:
    """Read-only readiness audit.  Optional roots enable file existence checks."""
    output = Path(output_dir)
    if output.exists(): raise ProtocolDevCaseError("IMMUTABLE_OUTPUT_EXISTS")
    cases = _read_cases(cases_dir); roots = _data_roots(data_roots); oracle_files = _oracle_files(oracle_dir)
    rows = []; global_issues = []; seen_ids: set[str] = set(); seen_videos: set[tuple[str, str]] = set()
    types: dict[str, int] = {item: 0 for item in CLAIM_TYPES}
    for path, case in cases:
        issues = _schema_issues(case) + _frame_issues(case, roots) + _source_binding_issues(case)
        if case.get("case_id") in seen_ids: issues.append("DUPLICATE_CASE_ID")
        seen_ids.add(case.get("case_id")); source = case.get("source", {}); video = (source.get("dataset"), source.get("video_id"))
        if video in seen_videos: issues.append("DUPLICATE_SOURCE_VIDEO")
        seen_videos.add(video)
        if video == ("NurViD", "vR0_BaXYcE4"): issues.append("PILOT_VIDEO_LEAKAGE")
        claim_type = case.get("task", {}).get("claim_type");
        if claim_type in types: types[claim_type] += 1
        oracle = case.get("oracle_annotation", {})
        artifact_ref = oracle.get("artifact_ref") if isinstance(oracle, dict) else None
        if artifact_ref and artifact_ref not in oracle_files: issues.append("ORACLE_ARTIFACT_REF_MISSING")
        if oracle.get("status") == "READY" and not artifact_ref: issues.append("ORACLE_STATUS_REF_INCONSISTENT")
        if case.get("source", {}).get("source_record_canonical_hashes") != []: pass
        else: issues.append("SOURCE_RECORD_CANONICAL_HASH_PENDING")
        if case.get("human_review", {}).get("reviewer_id") is None: issues.append("HUMAN_REVIEWER_PENDING")
        if case.get("human_review", {}).get("decision") == "PENDING": issues.append("HUMAN_REVIEW_DECISION_PENDING")
        if not _text(case.get("frame_locator", {}).get("relative_path_pattern")): issues.append("PORTABLE_FRAME_PATTERN_PENDING")
        if case.get("oracle_annotation", {}).get("status") != "READY": issues.append("ORACLE_ANNOTATION_PENDING")
        source_parse = case.get("provenance", {}).get("source_parse_issues", [])
        if source_parse: issues.extend("SOURCE_DRAFT_" + issue for issue in source_parse)
        try: assert_automatic_certificate_input_safe(_runtime_view(case))
        except ProtocolDevCaseError as exc: issues.append(str(exc))
        invalid = any(item in {"CASE_SCHEMA_KEYS_INVALID", "CASE_FORMAT_OR_VERSION_INVALID", "CASE_ID_OR_SPLIT_INVALID", "CLAIM_TYPE_ONEOF_INVALID", "SOURCE_DATASET_OR_VIDEO_REQUIRED", "MACHINE_PATH_IN_CASE", "TRUE_FALSE_CLAIM_PAIR_REQUIRED", "MATCHED_TO_CLAIM_ID_INVALID", "FALSE_CLAIM_NOT_PREDICATE_ONLY", "ORACLE_ISOLATION_INVALID", "DUPLICATE_CASE_ID", "DUPLICATE_SOURCE_VIDEO", "PILOT_VIDEO_LEAKAGE", "SAMPLED_FRAME_OUTSIDE_EVIDENCE_WINDOW", "KEYFRAME_OUTSIDE_EVIDENCE_WINDOW", "INTERACTION_POSTCONDITION_FRAME_RANGE_CONFLICT", "SAMPLED_FRAME_CLAIM_CONTINUITY_CONFLICT", "SOURCE_RECORD_VIDEO_MISMATCH"} for item in issues)
        pending = bool(issues)
        status = "INVALID" if invalid else ("PENDING" if pending else "READY")
        rows.append({"case_id": case.get("case_id"), "case_path": path.name, "case_sha256": sha256_path(path),
                     "claim_type": claim_type, "dataset": source.get("dataset"), "video_id": source.get("video_id"),
                     "readiness": status, "missing_or_issue_fields": sorted(set(issues))})
    if len(cases) != 9: global_issues.append("CASE_COUNT_NOT_NINE")
    if any(types[item] != 3 for item in CLAIM_TYPES): global_issues.append("CLAIM_TYPE_BALANCE_NOT_THREE_EACH")
    summary = {"format": AUDIT_FORMAT, "status": "PASS" if not global_issues and all(row["readiness"] == "READY" for row in rows) else "PENDING", "case_count": len(rows),
               "claim_type_counts": dict(sorted(types.items())), "case_readiness_counts": {status: sum(row["readiness"] == status for row in rows) for status in ("READY", "PENDING", "INVALID")},
               "global_issues": sorted(global_issues), "rows": rows, "cases_dir_sha256": stable_hash([{key: row[key] for key in ("case_id", "case_sha256")} for row in rows]),
               "oracle_dir_provided": oracle_dir is not None, "data_roots_provided": bool(roots),
               "model_calls_made": 0, "cache_writes": 0, "certificate_writes": 0, "new_verified_count": 0, "gt_used": False}
    summary["audit_content_sha256"] = stable_hash(summary)
    _write(output / "protocol_dev_readiness_report.json", summary)
    markdown = ["# Protocol-dev readiness report", "", f"Status: **{summary['status']}**", "", "| Case | Type | Readiness | Issues |", "|---|---|---|---|"]
    for row in rows:
        markdown.append(f"| {row['case_id']} | {row['claim_type']} | {row['readiness']} | {', '.join(row['missing_or_issue_fields']) or 'none'} |")
    markdown.extend(["", "No model, cache, certificate, GT, or VERIFIED operation was performed.", ""])
    _write(output / "protocol_dev_readiness_report.md", "\n".join(markdown))
    if strict and summary["status"] != "PASS":
        raise ProtocolDevCaseError("STRICT_READINESS_NOT_READY")
    return summary
