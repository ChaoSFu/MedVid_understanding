"""Deterministic, outcome-bound engineering smoke selection for Stage 3G v3."""
from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Any
from relive.storage.artifacts import canonical_json, stable_hash
from .spatial_anchor_grounding import rows, sha

FORMAT='relive-v2-spatial-anchor-grounding-v3-smoke-selection-v1'
ROLES=('PRECONDITION_ALIGNMENT','ACTION_CORE_PRESSING','POSTCONDITION_ATTACHMENT')

def _write(path:Path, value:Any)->None:
 if path.exists():raise ValueError('IMMUTABLE_OUTPUT_EXISTS')
 path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b''.join((canonical_json(x)+'\n').encode() for x in value))

def freeze(*,v2_output_dir:Path,output_dir:Path)->dict[str,Any]:
 if output_dir.exists() and any(output_dir.iterdir()):raise ValueError('OUTPUT_DIRECTORY_MUST_BE_EMPTY')
 source=rows(v2_output_dir/'run'/'v2_tal_spatial_anchor_groundings.jsonl','V2_GROUNDINGS')
 # This is deliberately historical-output-conditioned, bounded engineering selection only.
 selected=[];used=set()
 def choose(candidates:list[dict[str,Any]],reason:str)->None:
  candidates=[row for row in candidates if row['anchor_candidate_id'] not in used]
  if not candidates:return
  row=min(candidates,key=lambda item:stable_hash({'anchor_candidate_id':item['anchor_candidate_id'],'selection_reason':reason}))
  used.add(row['anchor_candidate_id']);selected.append({'anchor_candidate_id':row['anchor_candidate_id'],'selection_reason':reason,'source_v2_result_sha256':row['canonical_result_sha256']})
 for role in ROLES:
  failures=[row for row in source if row.get('observation_role')==role and row.get('failure_reason') in {'MODEL_OUTPUT_PARSE_FAILURE','MODEL_OUTPUT_SCHEMA_VIOLATION'}]
  choose(failures or [row for row in source if row.get('observation_role')==role],f'ROLE_COVERAGE_{role}')
 choose([row for row in source if row.get('failure_reason')=='MODEL_OUTPUT_SCHEMA_VIOLATION'],'V2_SCHEMA_VIOLATION_COVERAGE')
 if len(selected)<3:raise ValueError('SMOKE_ROLE_COVERAGE_UNAVAILABLE')
 _write(output_dir/'v2_tal_stage3g_v3_smoke_selection.jsonl',selected)
 report={'format':FORMAT,'status':'PASS','selection_kind':'OUTPUT_CONTRACT_ENGINEERING_SMOKE_NOT_SCIENTIFIC','source_v2_groundings_sha256':sha(v2_output_dir/'run'/'v2_tal_spatial_anchor_groundings.jsonl'),'selection_manifest_sha256':sha(output_dir/'v2_tal_stage3g_v3_smoke_selection.jsonl'),'anchor_count':len(selected),'roles_requested':list(ROLES),'selected_anchor_ids_frozen_before_v3_model_calls':True,'v3_output_used_for_selection':False,'model_calls_made':0,'backend_loaded':False,'cache_opened':False,'gt_used':False,'certificate_created':False,'new_verified_count':0,'certificate_status':'NOT_APPLICABLE'}
 report['report_content_sha256']=stable_hash(report);(output_dir/'v2_tal_stage3g_v3_smoke_selection_report.json').write_bytes((canonical_json(report)+'\n').encode());return report
