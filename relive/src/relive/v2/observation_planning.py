"""Stage 3C immutable ObservationClaim decomposition and acquisition planning."""
from __future__ import annotations
from decimal import Decimal
import hashlib
from pathlib import Path
from typing import Any

from relive.storage.artifacts import canonical_json, stable_hash
from .task_selection import TALSelectionError, strict_json_loads, strict_jsonl

FORMAT = "relive-v2-tal-observation-planning-v1"
ROLES = ("PRECONDITION_ALIGNMENT", "ACTION_CORE_PRESSING", "POSTCONDITION_ATTACHMENT")

class ObservationPlanningError(ValueError): pass
def _sha(p: Path) -> str: return hashlib.sha256(p.read_bytes()).hexdigest()
def _obj(p: Path, c: str) -> dict[str, Any]:
    try: raw=p.read_bytes(); x=strict_json_loads(raw.decode(), error_code=c)
    except (OSError, UnicodeDecodeError, TALSelectionError) as e: raise ObservationPlanningError(c+"_INVALID") from e
    if not isinstance(x,dict) or raw != (canonical_json(x)+"\n").encode(): raise ObservationPlanningError(c+"_NONCANONICAL")
    return x
def _rows(p: Path,c: str) -> list[dict[str,Any]]:
    try: x=list(strict_jsonl(p,error_code=c)); raw=p.read_bytes()
    except (OSError,TALSelectionError) as e: raise ObservationPlanningError(c+"_INVALID") from e
    if raw != b"".join((canonical_json(i)+"\n").encode() for i in x): raise ObservationPlanningError(c+"_NONCANONICAL")
    return x
def _write(p: Path,x: Any):
    if p.exists(): raise ObservationPlanningError("IMMUTABLE_OUTPUT_EXISTS")
    p.write_bytes((canonical_json(x)+"\n").encode() if isinstance(x,dict) else b"".join((canonical_json(i)+"\n").encode() for i in x))
def _d(x: Any)->Decimal: return Decimal(str(x))
def _num(x: Decimal)->float: return float(x)

TEMPLATES={
"PRECONDITION_ALIGNMENT":{"observation_role":"PRECONDITION","subject":"POUCHING_SYSTEM_BASE","predicate":"VISUALLY_ALIGNED_WITH","object":"TARGET_SKIN_AREA","temporal_quantifier":"PRE_OR_EARLY_EVENT","observability":"APPARENT_2D","evidence_geometry":"RELATIONAL_COMPOSITE","required_components":["BASE","TARGET_SKIN_AREA","RELATIVE_ALIGNMENT"],"surface":"The pouching-system base is visibly aligned with the target skin area."},
"ACTION_CORE_PRESSING":{"observation_role":"ACTION_CORE","subject":"OPERATOR_HAND","predicate":"VISIBLE_FIXING_OR_PRESSING_MOTION","object":"POUCHING_SYSTEM_BASE","temporal_quantifier":"DURING_EVENT","observability":"APPARENT_2D_TEMPORAL","evidence_geometry":"DYNAMIC_RELATIONAL_COMPOSITE","required_components":["OPERATOR_HAND","POUCHING_SYSTEM_BASE","HAND_BASE_INTERFACE","TEMPORAL_MOTION_PATTERN"],"surface":"The operator's hand visibly performs a fixing or pressing motion on the base."},
"POSTCONDITION_ATTACHMENT":{"observation_role":"POSTCONDITION","subject":"POUCHING_SYSTEM_BASE","predicate":"REMAINS_VISIBLY_ATTACHED_TO","object":"TARGET_SKIN_AREA","temporal_quantifier":"AFTER_INTERACTION","observability":"APPARENT_2D_TEMPORAL","evidence_geometry":"DYNAMIC_RELATIONAL_COMPOSITE","required_components":["POUCHING_SYSTEM_BASE","TARGET_SKIN_AREA","ATTACHMENT_INTERFACE","POST_INTERACTION_STATE"],"surface":"The base remains visibly attached to the target skin area after the interaction."}}

def _domain(role: str,start: Decimal,end: Decimal,total: Decimal,ctx: Decimal,zone: Decimal):
    if role=="PRECONDITION_ALIGNMENT": a,b=max(Decimal(0),start-ctx),min(end,start+zone)
    elif role=="ACTION_CORE_PRESSING": a,b=start,end
    else: a,b=max(start,end-zone),min(total,end+ctx)
    if b<a: raise ObservationPlanningError("OBSERVATION_DOMAIN_INVALID")
    return a,b
def _windows(a: Decimal,b: Decimal,length: Decimal,stride: Decimal,frames: list[dict[str,Any]],limit:int):
    pairs=[]
    if b-a<=length: pairs=[(a,b)]
    else:
        s=a
        while s+length<=b: pairs.append((s,s+length)); s+=stride
        if pairs[-1][1]!=b: pairs.append((b-length,b))
    out=[]
    for s,e in pairs:
        ids=[]; orders=[]; seen=set()
        for f in frames:
            t=_d(f["timestamp_seconds"])
            if s<=t<=e:
                key=stable_hash({"sha":f["frame_sha256"],"ref":f["source_frame_reference"],"time":f["timestamp_seconds"]})
                if key not in seen: seen.add(key); ids.append("unique_visual_frame_"+key[:24])
                orders.append(f["frame_order"])
        if not ids: raise ObservationPlanningError("ACQUISITION_WINDOW_HAS_NO_UNIQUE_FRAME")
        out.append((s,e,ids,orders))
    uniq=[]
    for x in out:
        if (x[0],x[1],tuple(x[2]),tuple(x[3])) not in [(y[0],y[1],tuple(y[2]),tuple(y[3])) for y in uniq]: uniq.append(x)
    if len(uniq)>limit: raise ObservationPlanningError("OBSERVATION_WINDOW_BUDGET_EXCEEDED")
    if not uniq or uniq[0][0]!=a or uniq[-1][1]!=b: raise ObservationPlanningError("OBSERVATION_DOMAIN_COVERAGE_FAILED")
    return uniq

def prepare(*,requirement_dir:Path,selection_manifest:Path,video_index_dir:Path,stage3b_dir:Path,comparison_path:Path,policy_path:Path,output_dir:Path)->dict[str,Any]:
    if output_dir.exists(): raise ObservationPlanningError("OUTPUT_DIRECTORY_MUST_BE_NEW")
    comparison=_obj(comparison_path,"INDEPENDENT_REPEAT")
    if comparison.get("ready_for_stage3c") is not True: raise ObservationPlanningError("STAGE3B_REPRODUCIBILITY_GATE_NOT_MET")
    parent=_obj(stage3b_dir/"run"/"v2_stage3b_claim_graph.json","PARENT_GRAPH"); hyps=_rows(stage3b_dir/"run"/"v2_stage3b_hypothesis_claims.jsonl","HYPOTHESES")
    if parent.get("nodes")!=hyps or any(h.get("status")!="CANDIDATE_UNVERIFIED" for h in hyps): raise ObservationPlanningError("PARENT_CLAIM_GRAPH_MUTATED")
    req=_rows(requirement_dir/"v2_requirement_specs.jsonl","REQUIREMENT"); index=_rows(video_index_dir/"v2_video_index.jsonl","VIDEO_INDEX")
    if len(req)!=1 or len(index)!=1: raise ObservationPlanningError("FROZEN_INPUT_CARDINALITY_INVALID")
    policy=_obj(policy_path,"POLICY");
    if policy.get("policy_version")!="relive-v2-tal-observation-temporal-planning-v1": raise ObservationPlanningError("POLICY_VERSION_INVALID")
    frames=index[0]["frames"]; total=max(_d(f["timestamp_seconds"]) for f in frames); length,stride,ctx,zone=map(_d,(policy["fine_window_seconds"],policy["fine_window_stride_seconds"],policy["boundary_context_seconds"],policy["boundary_zone_seconds"])); limit=policy["max_fine_windows_per_claim"]
    if type(limit)is not int or limit<1: raise ObservationPlanningError("POLICY_WINDOW_BUDGET_INVALID")
    output_dir.mkdir(parents=True); policy_sha=_sha(policy_path); req_sha=_sha(requirement_dir/"v2_requirement_specs.jsonl"); claims=[]; domains=[]; physical={}; bindings=[]; obligations=[]; edges=[]
    positives=[h for h in hyps if h.get("polarity")=="POSITIVE"]; negatives=[h for h in hyps if h.get("polarity")=="NO_VISIBLE_EVENT"]
    for hyp in positives:
        interval=hyp.get("candidate_interval");
        if not isinstance(interval,dict): raise ObservationPlanningError("POSITIVE_HYPOTHESIS_INTERVAL_INVALID")
        start,end=_d(interval["start_seconds"]),_d(interval["end_seconds"])
        for role in ROLES:
            template=TEMPLATES[role]; claim_id="observation_"+stable_hash({"parent":hyp["hypothesis_id"],"template":role,"policy":policy_sha,"requirement":req_sha})[:24]
            claim={"claim_id":claim_id,"parent_claim_id":hyp["hypothesis_id"],"requirement_id":hyp["requirement_id"],"claim_role":"OBSERVATION","template_id":role,"generation_reason":"REQUIRED_EVIDENCE_DECOMPOSITION","generation_stage":"STAGE_3C","polarity":"POSITIVE","status":"CANDIDATE_UNVERIFIED",**template,"provenance":{"observation_planning_policy_sha256":policy_sha,"requirement_specs_sha256":req_sha}}
            claims.append(claim); edges.append({"edge_type":"DECOMPOSED_INTO_REQUIRED_OBSERVATION","parent_claim_id":hyp["hypothesis_id"],"child_claim_id":claim_id})
            a,b=_domain(role,start,end,total,ctx,zone); truncated= role=="POSTCONDITION_ATTACHMENT" and end+ctx>total
            domain={"claim_id":claim_id,"parent_hypothesis_id":hyp["hypothesis_id"],"observation_role":role,"start_seconds":_num(a),"end_seconds":_num(b),"timestamp_domain":"CLIP_LOCAL","right_context_truncated_by_clip_boundary":truncated,"planning_policy_sha256":policy_sha,"domain_coverage":1.0,"uncovered_duration_seconds":0.0}
            domains.append(domain)
            for s,e,ids,orders in _windows(a,b,length,stride,frames,limit):
                base={"start_seconds":_num(s),"end_seconds":_num(e),"window_length_seconds":_num(e-s),"included_unique_frame_ids":ids,"included_logical_frame_orders":orders,"video_index_manifest_sha256":_sha(video_index_dir/"v2_video_index_manifest.json"),"planning_policy_sha256":policy_sha,"status":"PLANNED_UNSCORED"}
                key=stable_hash(base); physical.setdefault(key,{"acquisition_window_id":"acquisition_"+key[:24],**base})
                bindings.append({"claim_id":claim_id,"parent_hypothesis_id":hyp["hypothesis_id"],"observation_role":role,"acquisition_window_id":physical[key]["acquisition_window_id"],"status":"PLANNED_UNSCORED"})
    for hyp in negatives:
        obligations.append({"obligation_type":"GLOBAL_NEGATIVE_COVERAGE","parent_hypothesis_id":hyp["hypothesis_id"],"required_coverage":"FULL_CLIP","status":"UNTESTED","admission_rule":"ALL_POSITIVE_HYPOTHESES_MUST_FAIL_VERIFICATION","factual_claim":False})
        edges.append({"edge_type":"REQUIRES_GLOBAL_NEGATIVE_COVERAGE","parent_claim_id":hyp["hypothesis_id"],"obligation_type":"GLOBAL_NEGATIVE_COVERAGE"})
    if len(claims)!=len(positives)*len(ROLES) or any(c["status"]!="CANDIDATE_UNVERIFIED" for c in claims): raise ObservationPlanningError("OBSERVATION_DECOMPOSITION_INVALID")
    graph={"format":FORMAT,"parent_claim_graph_sha256":_sha(stage3b_dir/"run"/"v2_stage3b_claim_graph.json"),"requirement_binding":parent.get("requirement_binding"),"nodes":hyps+claims,"edges":edges,"additive_update_only":True,"existing_node_mutation_count":0,"existing_edge_mutation_count":0,"new_observation_node_count":len(claims),"new_negative_coverage_obligation_count":len(obligations)}
    artifacts={"v2_stage3c_observation_templates.json":TEMPLATES,"v2_stage3c_temporal_planning_policy.json":policy,"v2_stage3c_observation_claims.jsonl":claims,"v2_stage3c_observation_domains.jsonl":domains,"v2_stage3c_physical_acquisition_windows.jsonl":sorted(physical.values(),key=lambda x:x["acquisition_window_id"]),"v2_stage3c_claim_window_bindings.jsonl":sorted(bindings,key=lambda x:(x["claim_id"],x["acquisition_window_id"])),"v2_stage3c_negative_coverage_obligation.json":{"obligations":obligations},"v2_stage3c_claim_graph.json":graph}
    for name,x in artifacts.items(): _write(output_dir/name,x)
    hashes={name:_sha(output_dir/name) for name in artifacts}; manifest={"format":FORMAT,"status":"PASS","stage_status":"OBSERVATION_CLAIMS_AND_ACQUISITION_PLAN_FROZEN","artifact_sha256":hashes,"positive_hypothesis_count":len(positives),"negative_hypothesis_count":len(negatives),"observation_claim_count":len(claims),"observation_claims_per_positive_hypothesis":len(ROLES),"physical_acquisition_window_count":len(physical),"claim_window_binding_count":len(bindings),"deduplicated_binding_savings":len(bindings)-len(physical),"all_domain_coverage":1.0,"existing_node_mutation_count":0,"existing_edge_mutation_count":0,"model_calls_made":0,"backend_loaded":False,"cache_opened":False,"assistant_or_gt_values_accessed":False,"gt_used":False,"certificate_created":False,"new_verified_count":0,"certificate_status":"NOT_APPLICABLE","ready_for_observation_retrieval":True}
    manifest["manifest_content_sha256"]=stable_hash(manifest); _write(output_dir/"v2_stage3c_manifest.json",manifest)
    audit={"format":FORMAT,"status":"PASS","allowed_inputs_opened":["frozen RequirementSpec artifacts","frozen selection manifest","Immutable VideoIndex","Stage 3B frozen artifacts","independent repeat comparison","versioned Stage 3C policy"],"forbidden_inputs_not_opened":["raw trainval JSON","assistant answer","reference answer","temporal GT","bbox","mask","ROI","model","certificate"],"parent_claim_graph_sha256":graph["parent_claim_graph_sha256"],"model_calls_made":0,"backend_loaded":False,"cache_opened":False,"gt_used":False,"certificate_created":False,"new_verified_count":0}; audit["audit_content_sha256"]=stable_hash(audit); _write(output_dir/"v2_stage3c_freeze_audit.json",audit)
    return manifest

def validate(output_dir:Path)->dict[str,Any]:
    manifest=_obj(output_dir/"v2_stage3c_manifest.json","MANIFEST")
    if manifest.get("manifest_content_sha256")!=stable_hash({k:v for k,v in manifest.items() if k!="manifest_content_sha256"}): raise ObservationPlanningError("MANIFEST_TAMPERED")
    for name,expected in manifest.get("artifact_sha256",{}).items():
        if _sha(output_dir/name)!=expected: raise ObservationPlanningError("ARTIFACT_TAMPERED")
    claims=_rows(output_dir/"v2_stage3c_observation_claims.jsonl","CLAIMS"); domains=_rows(output_dir/"v2_stage3c_observation_domains.jsonl","DOMAINS"); binds=_rows(output_dir/"v2_stage3c_claim_window_bindings.jsonl","BINDINGS")
    if len(claims)!=manifest.get("observation_claim_count") or len(domains)!=len(claims) or not binds or any(c.get("status")!="CANDIDATE_UNVERIFIED" for c in claims): raise ObservationPlanningError("OBSERVATION_ARTIFACT_SCHEMA_INVALID")
    return {"status":"PASS","stage_status":manifest["stage_status"],"observation_claim_count":len(claims),"model_calls_made":0,"backend_loaded":False,"cache_opened":False,"gt_used":False,"certificate_created":False,"new_verified_count":0,"certificate_status":"NOT_APPLICABLE","ready_for_observation_retrieval":True}
