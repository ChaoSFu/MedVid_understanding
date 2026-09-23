"""Protocol-development cohort contracts with human/oracle isolation.

This module is deliberately data- and model-free.  It turns a two-stage human
review into three disjoint manifests only after every admission condition has
been met.  Incomplete review data is useful, but has only ``PREVIEW_ONLY``
authority and cannot be mistaken for a frozen runtime cohort.
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable

from relive.storage.artifacts import canonical_json, stable_hash
from .differential_evidence import load_policy
from .task_selection import TALSelectionError, strict_json_loads


FORMAT = "relive-v2-protocol-dev-manifest-v1"
PRECONDITION_FORMAT = "relive-v2-protocol-dev-precondition-freeze-v1"
PILOT_VIDEO = ("NurViD", "vR0_BaXYcE4")
CLAIM_TYPES = frozenset({"SPATIAL_RELATION", "CONTACT_ACTION", "POSTCONDITION_PERSISTENCE"})
DECISIONS = frozenset({"ADMIT", "REJECT", "RESERVE"})
TRUTH_LABELS = frozenset({"TRUE", "FALSE"})
HEX = frozenset("0123456789abcdef")
RUNTIME_FORBIDDEN_TOKENS = frozenset({
    "truth", "label", "changed_predicate", "oracle", "reviewer", "rationale",
    "decision", "human", "ground_truth", "answer",
})


class ProtocolDevManifestError(ValueError):
    pass


def sha256_path(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _hex(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= HEX


def _git_commit(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 40 and set(value) <= HEX


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _finite_span(value: Any) -> bool:
    return (isinstance(value, dict) and set(value) == {"start_seconds", "end_seconds"}
            and all(isinstance(value[key], (int, float)) and not isinstance(value[key], bool)
                    and float("-inf") < float(value[key]) < float("inf")
                    for key in value) and value["start_seconds"] <= value["end_seconds"])


def _strict_jsonl(path: str | Path, *, code: str, allow_empty: bool = False) -> list[dict[str, Any]]:
    try:
        text = Path(path).read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ProtocolDevManifestError(f"{code}_UNREADABLE") from exc
    if not text:
        if allow_empty:
            return []
        raise ProtocolDevManifestError(f"{code}_EMPTY")
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            raise ProtocolDevManifestError(f"{code}_BLANK_LINE")
        try:
            item = strict_json_loads(line, error_code=code)
        except TALSelectionError as exc:
            raise ProtocolDevManifestError(str(exc)) from exc
        if not isinstance(item, dict):
            raise ProtocolDevManifestError(f"{code}_OBJECT_REQUIRED")
        rows.append(item)
    return rows


def _strict_object(path: str | Path, *, code: str) -> dict[str, Any]:
    try:
        value = strict_json_loads(Path(path).read_bytes().decode("utf-8"), error_code=code)
    except (OSError, UnicodeDecodeError, TALSelectionError) as exc:
        raise ProtocolDevManifestError(f"{code}_INVALID") from exc
    if not isinstance(value, dict):
        raise ProtocolDevManifestError(f"{code}_OBJECT_REQUIRED")
    return value


def source_record_uid(*, dataset: str, video_id: str, source_record_canonical_hashes: Iterable[str], source_task: str) -> str:
    """Stable identity for a source binding, never a question/id shortcut."""
    hashes = tuple(sorted(source_record_canonical_hashes))
    if not _text(dataset) or not _text(video_id) or not _text(source_task) or not hashes or any(not _hex(item) for item in hashes):
        raise ProtocolDevManifestError("SOURCE_RECORD_UID_BINDING_INVALID")
    return "source_record_" + stable_hash({"dataset": dataset, "video_id": video_id,
                                             "source_record_canonical_hashes": hashes,
                                             "source_task": source_task})[:24]


STAGE_A_FIELDS = frozenset({
    "candidate_id", "dataset", "video_id", "source_record_uid", "source_record_canonical_hashes",
    "source_task", "proposed_claim_type", "source_temporal_span", "media_path", "decision",
    "reviewer_id", "rationale",
})
STAGE_B_FIELDS = frozenset({
    "case_id", "pair_id", "dataset", "video_id", "source_record_uid", "claim_type", "truth_label",
    "claim", "changed_predicate", "temporal_window", "interaction_window", "after_window",
    "involved_entities", "required_evidence_roles", "oracle_evidence_tube_or_mask", "observability",
    "media_path", "reviewer_id", "rationale",
})


def _stage_a_problems(row: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    if set(row) != STAGE_A_FIELDS:
        return ["STAGE_A_SCHEMA_INVALID"]
    for key in ("candidate_id", "dataset", "video_id", "source_record_uid", "source_task", "media_path", "decision", "reviewer_id", "rationale"):
        if not _text(row[key]):
            problems.append(f"STAGE_A_{key.upper()}_REQUIRED")
    hashes = row["source_record_canonical_hashes"]
    if not isinstance(hashes, list) or not hashes or any(not _hex(value) for value in hashes):
        problems.append("SOURCE_RECORD_CANONICAL_HASH_REQUIRED")
    elif len(set(hashes)) != len(hashes):
        problems.append("SOURCE_RECORD_CANONICAL_HASH_DUPLICATE")
    else:
        try:
            expected = source_record_uid(dataset=row["dataset"], video_id=row["video_id"],
                                         source_record_canonical_hashes=hashes, source_task=row["source_task"])
            if row["source_record_uid"] != expected:
                problems.append("SOURCE_RECORD_UID_NOT_CANONICALLY_BOUND")
        except ProtocolDevManifestError:
            problems.append("SOURCE_RECORD_UID_BINDING_INVALID")
    if row["proposed_claim_type"] not in CLAIM_TYPES:
        problems.append("PROPOSED_CLAIM_TYPE_INVALID")
    if not _finite_span(row["source_temporal_span"]):
        problems.append("SOURCE_TEMPORAL_SPAN_INVALID")
    if row["decision"] not in DECISIONS:
        problems.append("HUMAN_DECISION_INCOMPLETE")
    if ("STAGE_A_REVIEWER_ID_REQUIRED" in problems or "STAGE_A_RATIONALE_REQUIRED" in problems):
        problems.append("HUMAN_REVIEW_INCOMPLETE")
    return sorted(set(problems))


def _stage_b_problems(row: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    if set(row) != STAGE_B_FIELDS:
        return ["STAGE_B_SCHEMA_INVALID"]
    for key in ("case_id", "pair_id", "dataset", "video_id", "source_record_uid", "claim_type", "truth_label", "changed_predicate", "observability", "media_path", "reviewer_id", "rationale"):
        if not _text(row[key]):
            problems.append(f"STAGE_B_{key.upper()}_REQUIRED")
    if row["claim_type"] not in CLAIM_TYPES:
        problems.append("CASE_CLAIM_TYPE_INVALID")
    if row["truth_label"] not in TRUTH_LABELS:
        problems.append("CASE_TRUTH_LABEL_INVALID")
    if not isinstance(row["claim"], dict) or set(row["claim"]) != {"subject", "predicate", "object", "qualifiers"} or not all(_text(row["claim"].get(key)) for key in ("subject", "predicate", "object")) or not isinstance(row["claim"].get("qualifiers"), dict):
        problems.append("CASE_CLAIM_SCHEMA_INVALID")
    for key in ("temporal_window", "interaction_window", "after_window"):
        if row[key] is not None and not _finite_span(row[key]):
            problems.append(f"CASE_{key.upper()}_INVALID")
    if row["temporal_window"] is None:
        problems.append("CASE_TEMPORAL_WINDOW_REQUIRED")
    if row.get("claim_type") == "CONTACT_ACTION" and row["interaction_window"] is None:
        problems.append("CONTACT_ACTION_INTERACTION_WINDOW_REQUIRED")
    if row.get("claim_type") == "POSTCONDITION_PERSISTENCE" and row["after_window"] is None:
        problems.append("POSTCONDITION_AFTER_WINDOW_REQUIRED")
    if not isinstance(row["involved_entities"], list) or not row["involved_entities"] or any(not _text(value) for value in row["involved_entities"]):
        problems.append("CASE_ENTITIES_INVALID")
    if not isinstance(row["required_evidence_roles"], list) or not row["required_evidence_roles"] or any(not _text(value) for value in row["required_evidence_roles"]):
        problems.append("CASE_EVIDENCE_ROLES_INVALID")
    if not isinstance(row["oracle_evidence_tube_or_mask"], dict) or not row["oracle_evidence_tube_or_mask"]:
        problems.append("CASE_ORACLE_EVIDENCE_REQUIRED")
    return sorted(set(problems))


def _changed_only_predicate(true_case: dict[str, Any], false_case: dict[str, Any]) -> bool:
    true_claim, false_claim = true_case["claim"], false_case["claim"]
    return (true_claim["subject"] == false_claim["subject"] and true_claim["object"] == false_claim["object"]
            and true_claim["qualifiers"] == false_claim["qualifiers"]
            and true_claim["predicate"] != false_claim["predicate"]
            and true_case["changed_predicate"] == false_case["changed_predicate"] == "predicate"
            and all(true_case[key] == false_case[key] for key in ("dataset", "video_id", "source_record_uid", "claim_type", "temporal_window", "interaction_window", "after_window", "involved_entities", "required_evidence_roles", "media_path")))


def _contains_runtime_forbidden(value: Any) -> bool:
    if isinstance(value, dict):
        return any(any(token in key.casefold() for token in RUNTIME_FORBIDDEN_TOKENS) or _contains_runtime_forbidden(item)
                   for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_runtime_forbidden(item) for item in value)
    return False


def assert_automatic_runtime_safe(value: Any) -> None:
    """Reject manual/oracle material at the automatic runtime boundary."""
    if _contains_runtime_forbidden(value):
        raise ProtocolDevManifestError("HUMAN_ORACLE_AUTOMATIC_RUNTIME_FORBIDDEN")


def _write(path: Path, value: Any, *, jsonl: bool = False) -> None:
    if path.exists():
        raise ProtocolDevManifestError("IMMUTABLE_OUTPUT_EXISTS")
    path.parent.mkdir(parents=True, exist_ok=True)
    if jsonl:
        content = b"".join((canonical_json(row) + "\n").encode("utf-8") for row in value)
    else:
        content = (canonical_json(value) + "\n").encode("utf-8")
    path.write_bytes(content)


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNAVAILABLE"


def freeze_preconditions(*, policy_path: str | Path, acceptance_dir: str | Path,
                         pilot_closure: str | Path, output_dir: str | Path,
                         accepted_code_commit: str) -> dict[str, Any]:
    """Freeze the acceptance/pilot references before any protocol-dev cohort work."""
    if not _git_commit(accepted_code_commit):
        raise ProtocolDevManifestError("ACCEPTED_CODE_COMMIT_INVALID")
    policy, policy_sha = load_policy(policy_path)
    acceptance = Path(acceptance_dir)
    report_path = acceptance / "differential_policy_acceptance_report.json"
    regression_path = acceptance / "pilot_regression_report.json"
    report = _strict_object(report_path, code="ACCEPTANCE_REPORT")
    regression = _strict_object(regression_path, code="PILOT_REGRESSION")
    closure = _strict_object(pilot_closure, code="PILOT_CLOSURE")
    if (report.get("status") != "PASS" or report.get("policy_sha256") != policy_sha
            or report.get("model_calls_made") != 0 or report.get("certificate_created") is not False
            or regression.get("status") != "PASS" or regression.get("new_model_calls") != 0
            or regression.get("cache_writes") != 0 or regression.get("certificate_writes") != 0
            or closure.get("status") != "PILOT_CLOSED" or closure.get("closure_manifest_sha256") != stable_hash({key: value for key, value in closure.items() if key != "closure_manifest_sha256"})):
        raise ProtocolDevManifestError("ACCEPTANCE_OR_PILOT_CONTRACT_INVALID")
    output = Path(output_dir)
    if output.exists():
        raise ProtocolDevManifestError("IMMUTABLE_OUTPUT_EXISTS")
    payload = {"format": PRECONDITION_FORMAT, "status": "PASS", "policy_sha256": policy_sha,
               "policy_version": policy["policy_version"], "acceptance_artifact_path": str(acceptance),
               "acceptance_report_sha256": sha256_path(report_path),
               "pilot_closure_path": str(Path(pilot_closure)),
               "pilot_closure_sha256": sha256_path(pilot_closure),
               "pilot_closure_manifest_sha256": closure["closure_manifest_sha256"],
               "accepted_code_commit": accepted_code_commit, "current_code_commit": _git_head(),
               "model_calls_made": 0, "cache_writes": 0, "certificate_writes": 0,
               "certificate_created": False, "new_verified_count": 0, "gt_used": False}
    payload["freeze_report_sha256"] = stable_hash(payload)
    _write(output / "protocol_dev_precondition_freeze_report.json", payload)
    return payload


def _load_precondition(path: str | Path, policy_sha: str) -> dict[str, Any]:
    value = _strict_object(path, code="PRECONDITION_FREEZE")
    payload = {key: item for key, item in value.items() if key != "freeze_report_sha256"}
    if (value.get("format") != PRECONDITION_FORMAT or value.get("status") != "PASS"
            or value.get("policy_sha256") != policy_sha or value.get("freeze_report_sha256") != stable_hash(payload)):
        raise ProtocolDevManifestError("PRECONDITION_FREEZE_INVALID")
    return value


def prefilter_csv_to_stage_a_preview(*, prefilter_csv: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Convert a non-authoritative CSV into a deliberately incomplete Stage-A form."""
    source = Path(prefilter_csv)
    try:
        with source.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            headers = set(reader.fieldnames or [])
            required = {"protocol_dev_prefilter_id", "claim_type", "dataset_name", "source_video_id",
                        "source_qa_types", "human_review_decision", "human_review_rationale"}
            if not required.issubset(headers):
                raise ProtocolDevManifestError("PREFILTER_CSV_SCHEMA_INVALID")
            rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise ProtocolDevManifestError("PREFILTER_CSV_UNREADABLE") from exc
    output = Path(output_dir)
    if output.exists():
        raise ProtocolDevManifestError("IMMUTABLE_OUTPUT_EXISTS")
    preview = []
    for row in rows:
        decision = (row.get("human_review_decision") or "").strip()
        preview.append({"candidate_id": (row.get("protocol_dev_prefilter_id") or "").strip(),
                        "dataset": (row.get("dataset_name") or "").strip(),
                        "video_id": (row.get("source_video_id") or "").strip(),
                        "source_record_uid": "", "source_record_canonical_hashes": [],
                        "source_task": (row.get("source_qa_types") or "").strip(),
                        "proposed_claim_type": (row.get("claim_type") or "").strip(),
                        "source_temporal_span": None, "media_path": "",
                        "decision": decision if decision in DECISIONS else "", "reviewer_id": "",
                        "rationale": (row.get("human_review_rationale") or "").strip()})
    _write(output / "protocol_dev_stage_a_candidate_review.jsonl", preview, jsonl=True)
    _write(output / "protocol_dev_stage_b_case_definition.jsonl", [], jsonl=True)
    report = {"format": FORMAT, "status": "PREVIEW_ONLY", "prefilter_csv_sha256": sha256_path(source),
              "candidate_count": len(preview), "stage_a_template": str(output / "protocol_dev_stage_a_candidate_review.jsonl"),
              "stage_b_template": str(output / "protocol_dev_stage_b_case_definition.jsonl"),
              "required_missing_fields": ["source_record_canonical_hashes", "source_record_uid", "source_temporal_span", "media_path", "decision", "reviewer_id", "rationale"],
              "model_calls_made": 0, "cache_writes": 0, "certificate_writes": 0,
              "new_verified_count": 0, "gt_used": False}
    report["preview_report_sha256"] = stable_hash(report)
    _write(output / "protocol_dev_prefilter_preview_report.json", report)
    return report


def stage_b_template(*, stage_a_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    stage_a = _strict_jsonl(stage_a_path, code="STAGE_A", allow_empty=False)
    if Path(output_path).exists():
        raise ProtocolDevManifestError("IMMUTABLE_OUTPUT_EXISTS")
    rows = []
    for item in stage_a:
        if item.get("decision") == "ADMIT" and not _stage_a_problems(item):
            pair_id = "pair_" + stable_hash({"candidate_id": item["candidate_id"], "source_record_uid": item["source_record_uid"]})[:24]
            for label in ("TRUE", "FALSE"):
                rows.append({"case_id": "", "pair_id": pair_id, "dataset": item["dataset"], "video_id": item["video_id"],
                             "source_record_uid": item["source_record_uid"], "claim_type": item["proposed_claim_type"],
                             "truth_label": label, "claim": {"subject": "", "predicate": "", "object": "", "qualifiers": {}},
                             "changed_predicate": "predicate", "temporal_window": None, "interaction_window": None,
                             "after_window": None, "involved_entities": [], "required_evidence_roles": [],
                             "oracle_evidence_tube_or_mask": {}, "observability": "", "media_path": item["media_path"],
                             "reviewer_id": "", "rationale": ""})
    _write(Path(output_path), rows, jsonl=True)
    return {"status": "PREVIEW_ONLY", "pair_count": len(rows) // 2, "case_count": len(rows),
            "stage_b_template_sha256": sha256_path(output_path), "model_calls_made": 0,
            "cache_writes": 0, "certificate_writes": 0, "new_verified_count": 0}


def derive_stage_a_source_uids(*, stage_a_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Derive source UIDs from supplied canonical hashes without filling reviews."""
    rows = _strict_jsonl(stage_a_path, code="STAGE_A", allow_empty=False)
    derived = []
    for row in rows:
        if set(row) != STAGE_A_FIELDS:
            raise ProtocolDevManifestError("STAGE_A_SCHEMA_INVALID")
        hashes = row["source_record_canonical_hashes"]
        if not isinstance(hashes, list) or not hashes or any(not _hex(value) for value in hashes):
            raise ProtocolDevManifestError("SOURCE_RECORD_CANONICAL_HASH_REQUIRED")
        replacement = dict(row)
        replacement["source_record_uid"] = source_record_uid(dataset=row["dataset"], video_id=row["video_id"],
            source_record_canonical_hashes=hashes, source_task=row["source_task"])
        derived.append(replacement)
    _write(Path(output_path), derived, jsonl=True)
    return {"status": "PREVIEW_ONLY", "source_uid_count": len(derived),
            "derived_stage_a_sha256": sha256_path(output_path), "model_calls_made": 0,
            "cache_writes": 0, "certificate_writes": 0, "new_verified_count": 0}


def _audit(stage_a: list[dict[str, Any]], stage_b: list[dict[str, Any]], external_splits: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    issues = [problem for row in stage_a for problem in _stage_a_problems(row)]
    issues.extend(problem for row in stage_b for problem in _stage_b_problems(row))
    ids = [row.get("candidate_id") for row in stage_a]
    if len(ids) != len(set(ids)):
        issues.append("DUPLICATE_CANDIDATE_ID")
    bindings = [row.get("source_record_uid") for row in stage_a]
    if len(bindings) != len(set(bindings)):
        issues.append("DUPLICATE_SOURCE_RECORD_UID")
    videos = [(row.get("dataset"), row.get("video_id")) for row in stage_a]
    if len(videos) != len(set(videos)):
        issues.append("DUPLICATE_PROTOCOL_DEV_VIDEO")
    if PILOT_VIDEO in videos:
        issues.append("PILOT_VIDEO_LEAKAGE")
    registry: dict[tuple[str, str], str] = {}
    for row in external_splits:
        if set(row) != {"dataset", "video_id", "split"} or not all(_text(row.get(key)) for key in row):
            issues.append("EXTERNAL_SPLIT_REGISTRY_SCHEMA_INVALID"); continue
        video = (row["dataset"], row["video_id"]); existing = registry.setdefault(video, row["split"])
        if existing != row["split"]:
            issues.append("EXTERNAL_SPLIT_REGISTRY_CONFLICT")
    if any(video in registry and registry[video] != "protocol_dev" for video in videos):
        issues.append("CROSS_SPLIT_VIDEO_LEAKAGE")
    for row in stage_a:
        if _text(row.get("media_path")) and not Path(row["media_path"]).is_file():
            issues.append("MEDIA_PATH_UNRESOLVED")
    admitted = [row for row in stage_a if row.get("decision") == "ADMIT"]
    admitted_by_uid = {row.get("source_record_uid"): row for row in admitted}
    pairs: dict[str, list[dict[str, Any]]] = {}
    for row in stage_b:
        pairs.setdefault(str(row.get("pair_id")), []).append(row)
    if len(stage_b) != len(admitted) * 2:
        issues.append("CASE_COUNT_DOES_NOT_MATCH_ADMITTED_VIDEOS")
    claim_keys = []
    for pair_id, cases in pairs.items():
        if len(cases) != 2 or {case.get("truth_label") for case in cases} != TRUTH_LABELS:
            issues.append("TRUE_FALSE_PAIR_REQUIRED"); continue
        true_case = next(case for case in cases if case["truth_label"] == "TRUE")
        false_case = next(case for case in cases if case["truth_label"] == "FALSE")
        if true_case.get("source_record_uid") not in admitted_by_uid:
            issues.append("CASE_NOT_BOUND_TO_ADMITTED_VIDEO")
        if not _changed_only_predicate(true_case, false_case):
            issues.append("FALSE_CLAIM_MUST_CHANGE_EXACTLY_ONE_PREDICATE")
        claim_keys.extend(canonical_json(case.get("claim")) for case in cases)
    if len(claim_keys) != len(set(claim_keys)):
        issues.append("DUPLICATE_CLAIM")
    type_videos = {claim_type: {(row.get("dataset"), row.get("video_id")) for row in admitted if row.get("proposed_claim_type") == claim_type} for claim_type in CLAIM_TYPES}
    if any(len(type_videos[claim_type]) != 3 for claim_type in CLAIM_TYPES) or len(set(videos)) != 9 or len(admitted) != 9 or len(stage_b) != 18:
        issues.append("FROZEN_COHORT_CARDINALITY_INVALID")
    report = {"candidate_count": len(stage_a), "admitted_video_count": len(admitted), "case_count": len(stage_b),
              "videos_by_claim_type": {key: len(value) for key, value in sorted(type_videos.items())},
              "external_split_video_count": len(registry), "pilot_video_excluded": PILOT_VIDEO not in videos,
              "source_record_uid_duplicate_count": len(bindings) - len(set(bindings)),
              "video_duplicate_count": len(videos) - len(set(videos)), "issues": sorted(set(issues))}
    return sorted(set(issues)), report


def build_protocol_dev_manifests(*, policy_path: str | Path, precondition_freeze_report: str | Path,
                                 stage_a_path: str | Path, stage_b_path: str | Path,
                                 output_dir: str | Path, external_split_registry: str | Path | None = None) -> dict[str, Any]:
    """Build a frozen cohort iff all reviewed-source conditions are satisfied."""
    _, policy_sha = load_policy(policy_path)
    precondition = _load_precondition(precondition_freeze_report, policy_sha)
    output = Path(output_dir)
    if output.exists():
        raise ProtocolDevManifestError("IMMUTABLE_OUTPUT_EXISTS")
    stage_a = _strict_jsonl(stage_a_path, code="STAGE_A", allow_empty=False)
    stage_b = _strict_jsonl(stage_b_path, code="STAGE_B", allow_empty=True)
    external = _strict_jsonl(external_split_registry, code="EXTERNAL_SPLIT", allow_empty=True) if external_split_registry else []
    issues, audit = _audit(stage_a, stage_b, external)
    if external_split_registry is None:
        issues = sorted(set(issues + ["EXTERNAL_SPLIT_REGISTRY_REQUIRED_FOR_FREEZE"]))
        audit["issues"] = issues
    frozen = not issues
    runtime: list[dict[str, Any]] = []
    ground_truth: list[dict[str, Any]] = []
    oracle: list[dict[str, Any]] = []
    if frozen:
        for row in sorted(stage_b, key=lambda item: item["case_id"]):
            runtime_row = {key: row[key] for key in ("case_id", "dataset", "video_id", "source_record_uid", "claim_type", "claim", "temporal_window", "interaction_window", "after_window", "involved_entities", "required_evidence_roles", "observability", "media_path")}
            assert_automatic_runtime_safe(runtime_row)
            runtime.append(runtime_row)
            ground_truth.append({key: row[key] for key in ("case_id", "pair_id", "truth_label", "changed_predicate", "reviewer_id", "rationale")})
            oracle.append({"case_id": row["case_id"], "source_record_uid": row["source_record_uid"],
                           "oracle_evidence_tube_or_mask": row["oracle_evidence_tube_or_mask"],
                           "provenance": "HUMAN_ORACLE", "automatic_certificate_eligible": False})
    _write(output / "protocol_dev_runtime_manifest.jsonl", runtime, jsonl=True)
    _write(output / "protocol_dev_ground_truth_manifest.jsonl", ground_truth, jsonl=True)
    _write(output / "protocol_dev_oracle_evidence_manifest.jsonl", oracle, jsonl=True)
    file_hashes = {name: sha256_path(output / name) for name in ("protocol_dev_runtime_manifest.jsonl", "protocol_dev_ground_truth_manifest.jsonl", "protocol_dev_oracle_evidence_manifest.jsonl")}
    audit_payload = {"format": FORMAT, "status": "PASS" if frozen else "PREVIEW_ONLY", **audit,
                     "runtime_oracle_contamination": any(_contains_runtime_forbidden(row) for row in runtime),
                     "automatic_certificate_oracle_blocked": True, "media_paths_checked": len(stage_a),
                     "model_calls_made": 0, "cache_writes": 0, "certificate_writes": 0,
                     "certificate_created": False, "new_verified_count": 0, "gt_used": False}
    audit_payload["audit_report_sha256"] = stable_hash(audit_payload)
    _write(output / "protocol_dev_audit_report.json", audit_payload)
    manifest = {"format": FORMAT, "freeze_status": "FROZEN_PROTOCOL_DEV" if frozen else "PREVIEW_ONLY",
                "policy_sha256": policy_sha, "accepted_code_commit": precondition["accepted_code_commit"],
                "precondition_freeze_report_sha256": sha256_path(precondition_freeze_report),
                "stage_a_sha256": sha256_path(stage_a_path), "stage_b_sha256": sha256_path(stage_b_path),
                "external_split_registry_sha256": sha256_path(external_split_registry) if external_split_registry else None,
                "artifact_sha256": file_hashes, "audit_report_sha256": sha256_path(output / "protocol_dev_audit_report.json"),
                "video_split": [{"dataset": row["dataset"], "video_id": row["video_id"], "split": "protocol_dev"} for row in sorted(stage_a, key=lambda item: (item.get("dataset", ""), item.get("video_id", "")))],
                "case_counts": {"runtime": len(runtime), "ground_truth": len(ground_truth), "oracle": len(oracle)},
                "human_review_complete": frozen, "model_calls_made": 0, "cache_writes": 0,
                "certificate_writes": 0, "certificate_created": False, "new_verified_count": 0, "gt_used": False}
    manifest["manifest_content_sha256"] = stable_hash(manifest)
    _write(output / "protocol_dev_freeze_manifest.json", manifest)
    return manifest


def validate_protocol_dev_freeze(*, freeze_manifest: str | Path, policy_path: str | Path) -> dict[str, Any]:
    """Read-only validation of a protocol-dev freeze and its three outputs."""
    _, policy_sha = load_policy(policy_path)
    manifest_path = Path(freeze_manifest)
    manifest = _strict_object(manifest_path, code="PROTOCOL_DEV_FREEZE")
    payload = {key: value for key, value in manifest.items() if key != "manifest_content_sha256"}
    required = {"format", "freeze_status", "policy_sha256", "accepted_code_commit", "precondition_freeze_report_sha256",
                "stage_a_sha256", "stage_b_sha256", "external_split_registry_sha256", "artifact_sha256",
                "audit_report_sha256", "video_split", "case_counts", "human_review_complete", "model_calls_made",
                "cache_writes", "certificate_writes", "certificate_created", "new_verified_count", "gt_used",
                "manifest_content_sha256"}
    if (set(manifest) != required or manifest.get("format") != FORMAT or manifest.get("policy_sha256") != policy_sha
            or manifest.get("manifest_content_sha256") != stable_hash(payload)):
        raise ProtocolDevManifestError("PROTOCOL_DEV_FREEZE_MANIFEST_INVALID")
    root = manifest_path.parent
    for name, expected in manifest["artifact_sha256"].items():
        if not _hex(expected) or sha256_path(root / name) != expected:
            raise ProtocolDevManifestError("PROTOCOL_DEV_ARTIFACT_TAMPERED")
    audit_path = root / "protocol_dev_audit_report.json"
    audit = _strict_object(audit_path, code="PROTOCOL_DEV_AUDIT")
    if sha256_path(audit_path) != manifest["audit_report_sha256"] or audit.get("audit_report_sha256") != stable_hash({key: value for key, value in audit.items() if key != "audit_report_sha256"}):
        raise ProtocolDevManifestError("PROTOCOL_DEV_AUDIT_TAMPERED")
    for name in ("protocol_dev_runtime_manifest.jsonl", "protocol_dev_ground_truth_manifest.jsonl", "protocol_dev_oracle_evidence_manifest.jsonl"):
        _strict_jsonl(root / name, code="PROTOCOL_DEV_OUTPUT", allow_empty=True)
    runtime_rows = _strict_jsonl(root / "protocol_dev_runtime_manifest.jsonl", code="PROTOCOL_DEV_RUNTIME", allow_empty=True)
    if any(_contains_runtime_forbidden(row) for row in runtime_rows):
        raise ProtocolDevManifestError("RUNTIME_ORACLE_CONTAMINATION")
    if manifest["freeze_status"] == "FROZEN_PROTOCOL_DEV" and (not manifest["human_review_complete"] or manifest["case_counts"] != {"runtime": 18, "ground_truth": 18, "oracle": 18}):
        raise ProtocolDevManifestError("FROZEN_PROTOCOL_DEV_CARDINALITY_INVALID")
    result = {"format": FORMAT, "status": "PASS", "freeze_status": manifest["freeze_status"],
              "freeze_manifest_sha256": sha256_path(manifest_path), "policy_sha256": policy_sha,
              "model_calls_made": 0, "cache_writes": 0, "certificate_writes": 0,
              "certificate_created": False, "new_verified_count": 0, "gt_used": False}
    result["validation_sha256"] = stable_hash(result)
    return result
