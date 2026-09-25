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
