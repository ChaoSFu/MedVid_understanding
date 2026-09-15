"""Immutable, pre-video TAL RequirementSpec freeze artifacts."""
from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path
from typing import Any

from relive.storage.artifacts import canonical_json, stable_hash
from .task_selection import TALRequirementSelection, TALSelectionError, strict_json_loads, strict_jsonl
from .temporal_localization import EventOntology, PARSER_VERSION, TemporalLocalizationTaskAdapter

FREEZE_FORMAT = "relive-v2-tal-requirement-freeze-v1"
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_ARTIFACTS = ("v2_requirement_specs.jsonl", "v2_requirement_build_report.jsonl", "v2_requirement_unresolved.jsonl")


class RequirementFreezeError(ValueError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: bytes) -> None:
    if path.exists():
        raise RequirementFreezeError("IMMUTABLE_OUTPUT_EXISTS")
    path.write_bytes(payload)


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8")


def _audit_imports() -> dict[str, Any]:
    root = Path(__file__).parent
    sources = [root / "task_selection.py", root / "temporal_localization.py", root / "requirement_freeze.py", root / "task_adapters" / "base.py"]
    forbidden = ("relive.backends", "relive.runner", "relive.verification", "relive.certificate", "relive.interventions",
                 "relive.evaluation", "relive.data.medvidu", "relive.video", "relive.retrieval", "relive.evidence")
    issues = []
    for source in sources:
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        issues.extend(f"{source.name}:{item}" for item in forbidden if any(name == item or name.startswith(item + ".") for name in imported))
    return {"status": "PASS" if not issues else "FAIL", "files_checked": len(sources), "issues": issues,
            "limitation": "Static import audit verifies code boundaries, not independent curator provenance."}


def _close_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    payload["manifest_content_sha256"] = stable_hash(payload)
    return payload


def freeze_tal_requirements(*, selection: TALRequirementSelection, selector_sha256: str,
                            selection_manifest_sha256: str, ontology: EventOntology, output_dir: Path) -> dict[str, Any]:
    """Freeze only requirement metadata; no video, model, cache, or certificate is opened."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RequirementFreezeError("OUTPUT_DIRECTORY_MUST_BE_EMPTY")
    if selection.source_selector_sha256 != selector_sha256:
        raise RequirementFreezeError("SELECTION_SELECTOR_BINDING_MISMATCH")
    if not _HEX.fullmatch(selection_manifest_sha256) or not _HEX.fullmatch(ontology.ontology_sha256):
        raise RequirementFreezeError("FREEZE_INPUT_HASH_INVALID")
    imports = _audit_imports()
    if imports["status"] != "PASS":
        raise RequirementFreezeError("STATIC_ISOLATION_AUDIT_FAILED")
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter = TemporalLocalizationTaskAdapter()
    specs: list[dict[str, Any]] = []
    report: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for public_question in selection.items:
        result = adapter.build_requirement(public_question, ontology)
        row = {"sample_id": public_question.sample_id, "question_sha256": public_question.question_sha256,
               "source_record_index": public_question.source_record_index, "build_status": result.status.value,
               "requirement_id": result.requirement.requirement_id if result.requirement else None,
               "reason_code": result.reason_code, "adapter_version": result.adapter_version,
               "ontology_sha256": result.ontology_sha256, "terminal_query_parser_version": result.parser_version,
               "template_id": result.template_id, "normalized_event_phrase": result.normalized_event_phrase}
        report.append(row)
        if result.requirement is not None:
            specs.append(result.requirement.to_canonical_dict())
        else:
            unresolved.append(row)
    paths = {name: output_dir / name for name in _ARTIFACTS}
    _write(paths["v2_requirement_specs.jsonl"], _jsonl(specs))
    _write(paths["v2_requirement_build_report.jsonl"], _jsonl(report))
    _write(paths["v2_requirement_unresolved.jsonl"], _jsonl(unresolved))
    status = "PASS" if specs and not unresolved else ("PASS_WITH_UNRESOLVED" if specs else "NO_REQUIREMENTS_FROZEN")
    manifest = _close_manifest({
        "format": FREEZE_FORMAT, "status": status, "selection_status": "FROZEN_PRE_VIDEO_ACCESS",
        "public_question_selector_sha256": selector_sha256, "selection_manifest_sha256": selection_manifest_sha256,
        "ontology_sha256": ontology.ontology_sha256, "adapter_version": adapter.adapter_version,
        "terminal_query_parser_version": PARSER_VERSION,
        "requirement_count": len(specs), "build_report_count": len(report), "unresolved_count": len(unresolved),
        "requirement_specs_sha256": _sha(paths["v2_requirement_specs.jsonl"]),
        "build_report_sha256": _sha(paths["v2_requirement_build_report.jsonl"]),
        "unresolved_sha256": _sha(paths["v2_requirement_unresolved.jsonl"]),
        "gt_used": False, "frames_read": 0, "videos_read": 0, "model_calls_made": 0, "backend_loaded": False,
        "cache_opened": False, "claim_graph_created": False, "hypothesis_count": 0, "certificate_created": False,
        "new_verified_count": 0, "certificate_status": "NOT_APPLICABLE",
    })
    manifest_path = output_dir / "v2_requirement_manifest.json"
    _write(manifest_path, (canonical_json(manifest) + "\n").encode("utf-8"))
    audit = {"format": FREEZE_FORMAT, "status": status,
             "allowed_inputs_opened": ["GT-isolated public question selector", "frozen TAL selection manifest", "versioned TAL event ontology"],
             "forbidden_inputs_not_opened": ["frames", "videos", "reference_answer", "assistant_answer", "temporal_gt", "bbox", "mask", "ROI", "certificate", "model_result", "evaluation"],
             "imports_audited": imports, "frames_read": 0, "videos_read": 0, "gt_used": False, "model_calls_made": 0,
             "backend_loaded": False, "cache_opened": False, "claim_graph_created": False, "hypothesis_count": 0,
             "certificate_created": False, "new_verified_count": 0, "certificate_status": "NOT_APPLICABLE",
             "manifest_sha256": _sha(manifest_path)}
    audit["audit_content_sha256"] = stable_hash(audit)
    _write(output_dir / "v2_requirement_freeze_audit.json", (canonical_json(audit) + "\n").encode("utf-8"))
    return audit


def _strict_artifact_object(path: Path, *, code: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes().decode("utf-8")
        row = strict_json_loads(raw, error_code=code)
    except (OSError, UnicodeDecodeError, TALSelectionError) as exc:
        raise RequirementFreezeError(f"{code}_INVALID") from exc
    if not isinstance(row, dict):
        raise RequirementFreezeError(f"{code}_OBJECT_REQUIRED")
    if raw.encode("utf-8") != (canonical_json(row) + "\n").encode("utf-8"):
        raise RequirementFreezeError(f"{code}_NONCANONICAL_BYTES")
    return row


def _strict_rows(path: Path, *, code: str) -> tuple[dict[str, Any], ...]:
    try:
        raw = path.read_bytes()
        if not raw:
            return ()
        rows = strict_jsonl(path, error_code=code)
    except (OSError, TALSelectionError) as exc:
        raise RequirementFreezeError(f"{code}_INVALID") from exc
    if raw != _jsonl(list(rows)):
        raise RequirementFreezeError(f"{code}_NONCANONICAL_BYTES")
    return rows


def validate_requirement_freeze_artifacts(output_dir: Path) -> dict[str, Any]:
    """Validate frozen bytes only; never reopen selector, ontology, video, or backend."""
    paths = {name: output_dir / name for name in (*_ARTIFACTS, "v2_requirement_manifest.json", "v2_requirement_freeze_audit.json")}
    if not all(path.is_file() for path in paths.values()):
        raise RequirementFreezeError("FREEZE_ARTIFACT_MISSING")
    manifest = _strict_artifact_object(paths["v2_requirement_manifest.json"], code="MANIFEST")
    audit = _strict_artifact_object(paths["v2_requirement_freeze_audit.json"], code="AUDIT")
    required_manifest = {"format", "status", "selection_status", "public_question_selector_sha256", "selection_manifest_sha256",
                         "ontology_sha256", "adapter_version", "terminal_query_parser_version", "requirement_count",
                         "build_report_count", "unresolved_count", "requirement_specs_sha256", "build_report_sha256",
                         "unresolved_sha256", "gt_used", "frames_read", "videos_read", "model_calls_made", "backend_loaded",
                         "cache_opened", "claim_graph_created", "hypothesis_count", "certificate_created", "new_verified_count",
                         "certificate_status", "manifest_content_sha256"}
    if set(manifest) != required_manifest or manifest["format"] != FREEZE_FORMAT:
        raise RequirementFreezeError("MANIFEST_SCHEMA_INVALID")
    content = {key: value for key, value in manifest.items() if key != "manifest_content_sha256"}
    if manifest["manifest_content_sha256"] != stable_hash(content):
        raise RequirementFreezeError("MANIFEST_CONTENT_HASH_MISMATCH")
    hash_fields = {"requirement_specs_sha256": "v2_requirement_specs.jsonl", "build_report_sha256": "v2_requirement_build_report.jsonl", "unresolved_sha256": "v2_requirement_unresolved.jsonl"}
    if any(not isinstance(manifest[key], str) or not _HEX.fullmatch(manifest[key]) or manifest[key] != _sha(paths[name]) for key, name in hash_fields.items()):
        raise RequirementFreezeError("FREEZE_ARTIFACT_BYTES_CHANGED")
    if not all(isinstance(manifest[key], str) and _HEX.fullmatch(manifest[key]) for key in ("public_question_selector_sha256", "selection_manifest_sha256", "ontology_sha256")):
        raise RequirementFreezeError("MANIFEST_INPUT_BINDING_INVALID")
    specs = _strict_rows(paths["v2_requirement_specs.jsonl"], code="SPECS") if manifest["requirement_count"] else ()
    reports = _strict_rows(paths["v2_requirement_build_report.jsonl"], code="REPORT")
    unresolved = _strict_rows(paths["v2_requirement_unresolved.jsonl"], code="UNRESOLVED") if manifest["unresolved_count"] else ()
    if any(type(manifest[key]) is not int or manifest[key] < 0 for key in ("requirement_count", "build_report_count", "unresolved_count")):
        raise RequirementFreezeError("MANIFEST_COUNT_INVALID")
    if len(specs) != manifest["requirement_count"] or len(reports) != manifest["build_report_count"] or len(unresolved) != manifest["unresolved_count"] or len(reports) != len(specs) + len(unresolved):
        raise RequirementFreezeError("FREEZE_ARTIFACT_COUNT_MISMATCH")
    audit_content = {key: value for key, value in audit.items() if key != "audit_content_sha256"}
    if audit.get("audit_content_sha256") != stable_hash(audit_content):
        raise RequirementFreezeError("AUDIT_CONTENT_HASH_MISMATCH")
    if audit.get("format") != FREEZE_FORMAT or audit.get("manifest_sha256") != _sha(paths["v2_requirement_manifest.json"]):
        raise RequirementFreezeError("FREEZE_AUDIT_BINDING_MISMATCH")
    if audit.get("imports_audited", {}).get("status") != "PASS" or audit.get("status") not in {"PASS", "PASS_WITH_UNRESOLVED", "NO_REQUIREMENTS_FROZEN"}:
        raise RequirementFreezeError("STATIC_ISOLATION_AUDIT_FAILED")
    return {"status": "PASS", "manifest_content_sha256": manifest["manifest_content_sha256"], "audit_content_sha256": audit["audit_content_sha256"],
            "gt_used": False, "frames_read": 0, "videos_read": 0, "model_calls_made": 0, "backend_loaded": False,
            "cache_opened": False, "certificate_created": False, "new_verified_count": 0,
            "certificate_status": "NOT_APPLICABLE"}
