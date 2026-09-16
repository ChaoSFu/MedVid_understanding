"""Stage 3E: metadata-only, rank-based temporal evidence-chain composition."""
from __future__ import annotations
from decimal import Decimal
import hashlib
from pathlib import Path
from typing import Any
from relive.storage.artifacts import canonical_json, stable_hash
from .task_selection import strict_json_loads, strict_jsonl, TALSelectionError
from .observation_retrieval import ObservationRetrievalError, validate as validate_stage3d

FORMAT="relive-v2-tal-temporal-evidence-composition-v1"; ROLES=("PRECONDITION_ALIGNMENT","ACTION_CORE_PRESSING","POSTCONDITION_ATTACHMENT")
class TemporalCompositionError(ValueError): pass
def _sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def _rows(p:Path,c:str)->list[dict[str,Any]]:
 try: x=list(strict_jsonl(p,error_code=c)); raw=p.read_bytes()
 except (OSError,TALSelectionError) as e: raise TemporalCompositionError(c+"_INVALID") from e
 if raw!=b''.join((canonical_json(i)+'\n').encode() for i in x):raise TemporalCompositionError(c+"_NONCANONICAL")
 return x
def _obj(p:Path,c:str)->dict[str,Any]:
 try: raw=p.read_bytes();x=strict_json_loads(raw.decode(),error_code=c)
 except (OSError,UnicodeDecodeError,TALSelectionError) as e:raise TemporalCompositionError(c+"_INVALID") from e
 if not isinstance(x,dict) or raw!=(canonical_json(x)+'\n').encode():raise TemporalCompositionError(c+"_NONCANONICAL")
 return x
def _write(p:Path,x:Any):
 if p.exists():raise TemporalCompositionError("IMMUTABLE_OUTPUT_EXISTS")
 p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((canonical_json(x)+'\n').encode() if isinstance(x,dict) else b''.join((canonical_json(i)+'\n').encode() for i in x))
def _d(x:Any)->Decimal:return Decimal(str(x))
def _role_key(x:dict[str,Any])->str:return x["observation_role"]

def prepare(*,stage3c_dir:Path,stage3d_dir:Path,video_index_dir:Path,policy_path:Path,output_dir:Path)->dict[str,Any]:
 if output_dir.exists() and any(output_dir.iterdir()):raise TemporalCompositionError("OUTPUT_DIRECTORY_MUST_BE_EMPTY")
 try: dvalid=validate_stage3d(stage3d_dir)
 except ObservationRetrievalError as e:raise TemporalCompositionError("FROZEN_STAGE3D_ARTIFACT_INVALID") from e
 if dvalid.get("status")!="PASS" or not dvalid.get("ready_for_temporal_evidence_composition"):raise TemporalCompositionError("STAGE3D_NOT_READY")
 policy=_obj(policy_path,"POLICY")
 if policy.get("format")!="relive-v2-tal-temporal-evidence-composition-policy-v1" or policy.get("max_candidate_chains_per_hypothesis")!=3:raise TemporalCompositionError("POLICY_INVALID")
 pm=_obj(stage3c_dir/"v2_stage3c_manifest.json","STAGE3C_MANIFEST");claims=_rows(stage3c_dir/"v2_stage3c_observation_claims.jsonl","CLAIMS");domains={x["claim_id"]:x for x in _rows(stage3c_dir/"v2_stage3c_observation_domains.jsonl","DOMAINS")};windows={x["acquisition_window_id"]:x for x in _rows(stage3c_dir/"v2_stage3c_physical_acquisition_windows.jsonl","WINDOWS")}; binds=_rows(stage3c_dir/"v2_stage3c_claim_window_bindings.jsonl","BINDINGS");graph=_obj(stage3c_dir/"v2_stage3c_claim_graph.json","GRAPH")
 if any(c.get("status")!="CANDIDATE_UNVERIFIED" for c in claims):raise TemporalCompositionError("OBSERVATION_STATUS_MUTATED")
 candidates=_rows(stage3d_dir/"run"/"v2_stage3d_candidate_evidence_sets.jsonl","CANDIDATES"); rankings=_rows(stage3d_dir/"run"/"v2_stage3d_observation_rankings.jsonl","RANKINGS")
 if any(c.get("status")!="CANDIDATE_EVIDENCE_UNVERIFIED" for c in candidates):raise TemporalCompositionError("CANDIDATE_STATUS_INVALID")
 rank={(x["observation_claim_id"],x["binding_id"]):x for x in rankings}; claim={x["claim_id"]:x for x in claims}; allowed={(x["claim_id"],x["acquisition_window_id"]):x for x in binds}; positive=[x for x in graph.get("nodes",[]) if x.get("polarity")=="POSITIVE" and x.get("claim_role")=="TARGET_HYPOTHESIS"]
 psha=_sha(policy_path); enumerated=[]; chains=[]; summaries=[]; physical={}; hbindings=[]
 byparent:dict[str,dict[str,list[dict[str,Any]]]]={h["hypothesis_id"]:{r:[] for r in ROLES} for h in positive}
 for item in candidates:
  c=claim.get(item["observation_claim_id"])
  if not c or c["parent_claim_id"] not in byparent or c["template_id"] not in ROLES or (c["claim_id"],item["physical_window_id"]) not in allowed:raise TemporalCompositionError("CANDIDATE_PROVENANCE_INVALID")
  byparent[c["parent_claim_id"]][c["template_id"]].append({**item,"claim":c})
 for h in positive:
  hid=h["hypothesis_id"]; role=byparent[hid]; missing=any(not role[r] for r in ROLES); total=compatible=rejected=0; selected=[]
  if missing: summaries.append({"parent_hypothesis_id":hid,"composition_status":"UNRESOLVED","failure_reason":"OBSERVATION_ROLE_CANDIDATE_MISSING","allowed_next_action":"TEMPORAL_REACQUIRE","combination_count":0,"compatible_count":0,"rejected_order_count":0,"selected_count":0});continue
  interval=h.get("candidate_interval",{});start,end=_d(interval["start_seconds"]),_d(interval["end_seconds"]); context=_d(_obj(stage3c_dir/"v2_stage3c_temporal_planning_policy.json","PLAN_POLICY")["boundary_context_seconds"]); index=_rows(video_index_dir/"v2_video_index.jsonl","VIDEO_INDEX"); clip=max(_d(f["timestamp_seconds"]) for f in index[0]["frames"]); es=max(Decimal(0),start-context);ee=min(clip,end+context)
  raw=[]
  for pre in role[ROLES[0]]:
   for act in role[ROLES[1]]:
    for post in role[ROLES[2]]:
     total+=1; ws=[windows[x["physical_window_id"]] for x in (pre,act,post)]; mids=[(_d(w["start_seconds"])+_d(w["end_seconds"]))/2 for w in ws]; ok=mids[0]<=mids[1]<=mids[2] and all(es<=_d(w["start_seconds"]) and _d(w["end_seconds"])<=ee for w in ws)
     enum={"parent_hypothesis_id":hid,"pre_binding_id":pre["binding_id"],"action_binding_id":act["binding_id"],"post_binding_id":post["binding_id"],"role_order_valid":mids[0]<=mids[1]<=mids[2],"envelope_valid":all(es<=_d(w["start_seconds"]) and _d(w["end_seconds"])<=ee for w in ws),"status":"COMPATIBLE" if ok else "REJECTED_TEMPORAL_ORDER_OR_ENVELOPE"};enumerated.append(enum)
     if not ok:rejected+=1;continue
     compatible+=1; ids=[pre["physical_window_id"],act["physical_window_id"],post["physical_window_id"]]; cid="temporal_chain_"+stable_hash({"parent":hid,"claims":[pre["observation_claim_id"],act["observation_claim_id"],post["observation_claim_id"]],"windows":ids,"policy":psha})[:24]
     row={"chain_id":cid,"parent_hypothesis_id":hid,"pre_observation_claim_id":pre["observation_claim_id"],"action_observation_claim_id":act["observation_claim_id"],"post_observation_claim_id":post["observation_claim_id"],"pre_evidence_window_id":ids[0],"action_evidence_window_id":ids[1],"post_evidence_window_id":ids[2],"pre_retrieval_task_id":pre["retrieval_task_id"],"action_retrieval_task_id":act["retrieval_task_id"],"post_retrieval_task_id":post["retrieval_task_id"],"pre_rank":pre["rank"],"action_rank":act["rank"],"post_rank":post["rank"],"role_internal_margins":{"pre":pre["support_margin"],"action":act["support_margin"],"post":post["support_margin"]},"representative_times":{"pre":float(mids[0]),"action":float(mids[1]),"post":float(mids[2])},"role_order_valid":True,"pre_action_gap_seconds":float(max(Decimal(0),_d(ws[1]["start_seconds"])-_d(ws[0]["end_seconds"]))),"action_post_gap_seconds":float(max(Decimal(0),_d(ws[2]["start_seconds"])-_d(ws[1]["end_seconds"]))),"chain_span_seconds":float(max(_d(w["end_seconds"]) for w in ws)-min(_d(w["start_seconds"]) for w in ws)),"shared_physical_window_across_roles":len(set(ids))<3,"distinct_physical_window_count":len(set(ids)),"status":"CANDIDATE_TEMPORAL_EVIDENCE_UNVERIFIED","provenance":{"stage3e_policy_sha256":psha,"stage3d_candidate_evidence_sets_sha256":_sha(stage3d_dir/"run"/"v2_stage3d_candidate_evidence_sets.jsonl")}}
     raw.append(row)
  raw.sort(key=lambda x:(max(x["pre_rank"],x["action_rank"],x["post_rank"]),x["pre_rank"]+x["action_rank"]+x["post_rank"],x["action_rank"],x["pre_rank"],x["post_rank"],x["chain_span_seconds"],-x["role_internal_margins"]["action"],-x["role_internal_margins"]["pre"],-x["role_internal_margins"]["post"],x["pre_evidence_window_id"],x["action_evidence_window_id"],x["post_evidence_window_id"]))
  selected=raw[:3]
  for n,row in enumerate(selected,1):row["selection_rank_within_hypothesis"]=n;chains.append(row);pk=tuple([row["pre_evidence_window_id"],row["action_evidence_window_id"],row["post_evidence_window_id"]]);physical.setdefault(pk,{"physical_chain_id":"physical_chain_"+stable_hash(pk)[:24],"pre_evidence_window_id":pk[0],"action_evidence_window_id":pk[1],"post_evidence_window_id":pk[2],"status":"CANDIDATE_TEMPORAL_EVIDENCE_UNVERIFIED"});hbindings.append({"parent_hypothesis_id":hid,"chain_id":row["chain_id"],"physical_chain_id":physical[pk]["physical_chain_id"],"status":"CANDIDATE_TEMPORAL_EVIDENCE_UNVERIFIED"})
  summaries.append({"parent_hypothesis_id":hid,"composition_status":"CANDIDATE_CHAINS_COMPOSED" if selected else "UNRESOLVED","failure_reason":None if selected else "TEMPORAL_ORDER_UNRESOLVED","allowed_next_action":None if selected else "TEMPORAL_REACQUIRE","combination_count":total,"compatible_count":compatible,"rejected_order_count":rejected,"selected_count":len(selected)})
 out={"v2_stage3e_composition_policy.json":policy,"v2_stage3e_enumerated_combinations.jsonl":enumerated,"v2_stage3e_compatible_chains.jsonl":chains,"v2_stage3e_physical_chains.jsonl":sorted(physical.values(),key=lambda x:x["physical_chain_id"]),"v2_stage3e_hypothesis_chain_bindings.jsonl":hbindings,"v2_stage3e_hypothesis_composition_summary.jsonl":summaries,"v2_stage3e_temporal_evidence_graph.json":{"format":FORMAT,"nodes":chains,"bindings":hbindings,"claim_graph_mutated":False}}
 output_dir.mkdir(parents=True)
 for n,x in out.items():_write(output_dir/n,x)
 hashes={n:_sha(output_dir/n) for n in out}; anychains=bool(chains); manifest={"format":FORMAT,"status":"PASS" if anychains else "UNRESOLVED","stage_status":"TEMPORAL_EVIDENCE_CHAINS_COMPOSED_UNVERIFIED" if anychains else "TEMPORAL_EVIDENCE_COMPOSITION_UNRESOLVED","artifact_sha256":hashes,"positive_hypothesis_count":len(positive),"physical_chain_count":len(physical),"hypothesis_chain_binding_count":len(hbindings),"cross_hypothesis_dedup_savings":len(hbindings)-len(physical),"hypothesis_mutation_count":0,"observation_claim_mutation_count":0,"parent_claim_graph_mutation_count":0,"supported_hypothesis_count":0,"verified_hypothesis_count":0,"supported_observation_count":0,"verified_observation_count":0,"no_visible_event_status":"CANDIDATE_UNVERIFIED","negative_coverage_obligation_status":"UNTESTED","frames_opened":0,"model_calls_made":0,"backend_loaded":False,"cache_opened":False,"assistant_or_gt_values_accessed":False,"gt_used":False,"certificate_created":False,"new_verified_count":0,"certificate_status":"NOT_APPLICABLE","ready_for_typed_spatial_evidence_planning":anychains};manifest["manifest_content_sha256"]=stable_hash(manifest);_write(output_dir/"v2_stage3e_manifest.json",manifest)
 audit={"format":FORMAT,"status":"PASS","allowed_inputs_opened":["frozen RequirementSpec","selection manifest","Immutable VideoIndex","Stage 3B ClaimGraph","Stage 3C artifacts","Stage 3D scientific artifacts","versioned Stage 3E policy"],"forbidden_inputs_not_opened":["raw trainval JSON","assistant answer","reference answer","conversations[1]","temporal GT","frame pixels","bbox","mask","ROI","certificate"],"artifact_sha256":{**hashes,"v2_stage3e_manifest.json":_sha(output_dir/"v2_stage3e_manifest.json")},"frames_opened":0,"model_calls_made":0,"backend_loaded":False,"cache_opened":False,"gt_used":False,"assistant_or_gt_values_accessed":False,"certificate_created":False,"new_verified_count":0,"certificate_status":"NOT_APPLICABLE"};audit["audit_content_sha256"]=stable_hash(audit);_write(output_dir/"v2_stage3e_freeze_audit.json",audit)
 return manifest
def validate(output_dir:Path)->dict[str,Any]:
 m=_obj(output_dir/"v2_stage3e_manifest.json","MANIFEST")
 if m.get("manifest_content_sha256")!=stable_hash({k:v for k,v in m.items() if k!="manifest_content_sha256"}):raise TemporalCompositionError("MANIFEST_TAMPERED")
 for n,h in m.get("artifact_sha256",{}).items():
  if _sha(output_dir/n)!=h:raise TemporalCompositionError("ARTIFACT_TAMPERED")
 a=_obj(output_dir/"v2_stage3e_freeze_audit.json","AUDIT")
 if a.get("audit_content_sha256")!=stable_hash({k:v for k,v in a.items() if k!="audit_content_sha256"}):raise TemporalCompositionError("AUDIT_TAMPERED")
 return {"status":m["status"],"stage_status":m["stage_status"],"physical_chain_count":m["physical_chain_count"],"hypothesis_chain_binding_count":m["hypothesis_chain_binding_count"],"frames_opened":0,"model_calls_made":0,"backend_loaded":False,"cache_opened":False,"gt_used":False,"certificate_created":False,"new_verified_count":0,"certificate_status":"NOT_APPLICABLE","ready_for_typed_spatial_evidence_planning":m["ready_for_typed_spatial_evidence_planning"]}
