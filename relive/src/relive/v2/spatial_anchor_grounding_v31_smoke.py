"""Freeze the exact v3 output-contract smoke IDs for a single v3.1 comparison."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from relive.storage.artifacts import canonical_json, stable_hash
from .spatial_anchor_grounding import rows, sha

FORMAT='relive-v2-spatial-anchor-grounding-v3.1-smoke-selection-v1'

def _write(path:Path,value:Any)->None:
 if path.exists():raise ValueError('IMMUTABLE_OUTPUT_EXISTS')
 path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b''.join((canonical_json(x)+'\n').encode() for x in value))

def freeze(*,v3_selection_manifest:Path,output_dir:Path)->dict[str,Any]:
 """Copy ID order only; no v3.1 model output is read or consulted."""
 if output_dir.exists() and any(output_dir.iterdir()):raise ValueError('OUTPUT_DIRECTORY_MUST_BE_EMPTY')
 source=rows(v3_selection_manifest,'V3_SMOKE_SELECTION')
 expected={'anchor_candidate_id','selection_reason','source_v2_result_sha256'}
 if len(source)!=4 or any(set(row)!=expected for row in source):raise ValueError('V3_SMOKE_SELECTION_INVALID')
 ids=[row['anchor_candidate_id'] for row in source]
 if len(set(ids))!=4:raise ValueError('V3_SMOKE_SELECTION_DUPLICATE')
 frozen=[{'anchor_candidate_id':row['anchor_candidate_id'],'selection_reason':'SAME_AS_V3_OUTPUT_CONTRACT_SMOKE_'+row['selection_reason'],'source_v2_result_sha256':row['source_v2_result_sha256']} for row in source]
 _write(output_dir/'v2_tal_stage3g_v31_smoke_selection.jsonl',frozen)
 report={'format':FORMAT,'status':'PASS','selection_kind':'SAME_AS_V3_OUTPUT_CONTRACT_ENGINEERING_SMOKE_NOT_SCIENTIFIC','v3_selection_manifest_sha256':sha(v3_selection_manifest),'selection_manifest_sha256':sha(output_dir/'v2_tal_stage3g_v31_smoke_selection.jsonl'),'anchor_count':4,'anchor_order_preserved':True,'v3_1_output_used_for_selection':False,'model_calls_made':0,'backend_loaded':False,'cache_opened':False,'gt_used':False,'certificate_created':False,'new_verified_count':0,'certificate_status':'NOT_APPLICABLE'}
 report['report_content_sha256']=stable_hash(report);(output_dir/'v2_tal_stage3g_v31_smoke_selection_report.json').write_bytes((canonical_json(report)+'\n').encode());return report
