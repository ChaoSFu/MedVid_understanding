"""Immutable, pre-video TAL RequirementSpec freeze artifacts."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
from typing import Any
from relive.storage.artifacts import canonical_json, stable_hash
from .task_selection import TALRequirementSelection
from .temporal_localization import EventOntology, RequirementBuildStatus, TemporalLocalizationTaskAdapter

FREEZE_FORMAT="relive-v2-tal-requirement-freeze-v1"
class RequirementFreezeError(ValueError): pass

def _sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def _write(path:Path,payload:bytes)->None:
    if path.exists():raise RequirementFreezeError("IMMUTABLE_OUTPUT_EXISTS")
    path.write_bytes(payload)
def _jsonl(rows:list[dict[str,Any]])->bytes:return "".join(canonical_json(row)+"\n" for row in rows).encode("utf-8")
def _audit_imports()->dict[str,Any]:
    root=Path(__file__).parent
    sources=[root/"task_selection.py",root/"temporal_localization.py",root/"requirement_freeze.py",root/"task_adapters"/"base.py"]
    forbidden=("relive.backends","relive.runner","relive.verification","relive.certificate","relive.interventions","relive.evaluation","relive.data.medvidu")
    issues=[]
    for source in sources:
        tree=ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        imported=[]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import): imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module: imported.append(node.module)
        issues.extend(f"{source.name}:{item}" for item in forbidden if any(name == item or name.startswith(item + ".") for name in imported))
    return {"status":"PASS" if not issues else "FAIL","files_checked":len(sources),"issues":issues,"limitation":"Static import audit verifies code boundaries, not independent curator provenance."}

def freeze_tal_requirements(*,selection:TALRequirementSelection,selector_sha256:str,selection_manifest_sha256:str,ontology:EventOntology,output_dir:Path)->dict[str,Any]:
    if output_dir.exists() and any(output_dir.iterdir()):raise RequirementFreezeError("OUTPUT_DIRECTORY_MUST_BE_EMPTY")
    if selection.source_selector_sha256!=selector_sha256:raise RequirementFreezeError("SELECTION_SELECTOR_BINDING_MISMATCH")
    output_dir.mkdir(parents=True,exist_ok=True)
    adapter=TemporalLocalizationTaskAdapter(); specs=[]; report=[]; unresolved=[]
    for public_question in selection.items:
        result=adapter.build_requirement(public_question,ontology)
        row={"sample_id":public_question.sample_id,"question_sha256":public_question.question_sha256,"source_record_index":public_question.source_record_index,"build_status":result.status.value,"requirement_id":result.requirement.requirement_id if result.requirement else None,"reason_code":result.reason_code,"adapter_version":result.adapter_version,"ontology_sha256":result.ontology_sha256}
        report.append(row)
        if result.requirement is not None:specs.append(result.requirement.to_canonical_dict())
        else:unresolved.append(row)
    specs_path=output_dir/"v2_requirement_specs.jsonl"; report_path=output_dir/"v2_requirement_build_report.jsonl"; unresolved_path=output_dir/"v2_requirement_unresolved.jsonl"
    _write(specs_path,_jsonl(specs)); _write(report_path,_jsonl(report)); _write(unresolved_path,_jsonl(unresolved))
    status="PASS" if specs and not unresolved else ("PASS_WITH_UNRESOLVED" if specs else "NO_REQUIREMENTS_FROZEN")
    manifest={"format":FREEZE_FORMAT,"status":status,"selection_status":"FROZEN_PRE_VIDEO_ACCESS","public_question_selector_sha256":selector_sha256,"selection_manifest_sha256":selection_manifest_sha256,"ontology_sha256":ontology.ontology_sha256,"adapter_version":adapter.adapter_version,"requirement_count":len(specs),"unresolved_count":len(unresolved),"requirement_specs_sha256":_sha(specs_path),"build_report_sha256":_sha(report_path),"gt_used":False,"frames_read":0,"videos_read":0,"model_calls_made":0,"backend_loaded":False,"cache_opened":False,"claim_graph_created":False,"hypothesis_count":0,"certificate_created":False,"new_verified_count":0,"certificate_status":"NOT_APPLICABLE"}
    manifest["manifest_content_sha256"]=stable_hash(manifest)
    manifest_path=output_dir/"v2_requirement_manifest.json"; _write(manifest_path,(canonical_json(manifest)+"\n").encode())
    imports=_audit_imports()
    audit={"format":FREEZE_FORMAT,"status":status,"allowed_inputs_opened":["GT-isolated public question selector","frozen TAL selection manifest","versioned TAL event ontology"],"forbidden_inputs_not_opened":["frames","videos","reference_answer","assistant_answer","temporal_gt","bbox","mask","ROI","certificate","model_result","evaluation"],"imports_audited":imports,"frames_read":0,"videos_read":0,"gt_used":False,"model_calls_made":0,"backend_loaded":False,"cache_opened":False,"claim_graph_created":False,"hypothesis_count":0,"certificate_created":False,"new_verified_count":0,"certificate_status":"NOT_APPLICABLE","manifest_sha256":_sha(manifest_path)}
    audit["audit_content_sha256"]=stable_hash(audit)
    _write(output_dir/"v2_requirement_freeze_audit.json",(canonical_json(audit)+"\n").encode())
    return audit

def validate_requirement_freeze_artifacts(output_dir: Path) -> dict[str, Any]:
    """Validate immutable requirement-freeze bytes without reopening source data."""
    paths={name: output_dir/name for name in ("v2_requirement_specs.jsonl","v2_requirement_build_report.jsonl","v2_requirement_unresolved.jsonl","v2_requirement_manifest.json","v2_requirement_freeze_audit.json")}
    if not all(path.is_file() for path in paths.values()): raise RequirementFreezeError("FREEZE_ARTIFACT_MISSING")
    try: manifest=json.loads(paths["v2_requirement_manifest.json"].read_text(encoding="utf-8")); audit=json.loads(paths["v2_requirement_freeze_audit.json"].read_text(encoding="utf-8"))
    except Exception as exc: raise RequirementFreezeError("FREEZE_ARTIFACT_JSON_INVALID") from exc
    if manifest.get("manifest_content_sha256") != stable_hash({key:value for key,value in manifest.items() if key!="manifest_content_sha256"}): raise RequirementFreezeError("MANIFEST_CONTENT_HASH_MISMATCH")
    if manifest.get("requirement_specs_sha256") != _sha(paths["v2_requirement_specs.jsonl"]) or manifest.get("build_report_sha256") != _sha(paths["v2_requirement_build_report.jsonl"]): raise RequirementFreezeError("FREEZE_ARTIFACT_BYTES_CHANGED")
    if audit.get("manifest_sha256") != _sha(paths["v2_requirement_manifest.json"]): raise RequirementFreezeError("FREEZE_AUDIT_BINDING_MISMATCH")
    return {"status":"PASS","manifest_content_sha256":manifest["manifest_content_sha256"],"gt_used":False,"frames_read":0,"model_calls_made":0,"certificate_created":False}
