"""Zero-model comparison of two independent Stage 3G-A v3.2 fresh runs.

The run artifact's historic ``canonical_grounding_result_sha256`` intentionally
binds every result field, including measured constraint-initialization latency.
That is useful for an immutable run audit, but wall-clock measurements are not
a model output.  This comparator therefore reports that legacy identity while
gating reproducibility on a separately versioned scientific-output projection.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Any
from relive.storage.artifacts import canonical_json,stable_hash
from .task_selection import TALSelectionError,strict_json_loads,strict_jsonl
class SpatialAnchorRepeatError(ValueError):pass
SCIENTIFIC_OUTPUT_PROJECTION_VERSION='relive-v2-stage3g-v3.2-scientific-output-projection-v1'
_DYNAMIC_EXECUTION_FIELDS=('initialization_overhead_seconds',)
def _sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def _obj(p:Path)->dict[str,Any]:
 try:raw=p.read_bytes();v=strict_json_loads(raw.decode(),error_code='REPEAT')
 except (OSError,UnicodeDecodeError,TALSelectionError) as e:raise SpatialAnchorRepeatError('RUN_INVALID') from e
 if not isinstance(v,dict) or raw!=(canonical_json(v)+'\n').encode():raise SpatialAnchorRepeatError('RUN_INVALID')
 return v
def _rows(p:Path)->list[dict[str,Any]]:
 try:v=list(strict_jsonl(p,error_code='REPEAT'));raw=p.read_bytes()
 except (OSError,TALSelectionError) as e:raise SpatialAnchorRepeatError('RUN_INVALID') from e
 if raw!=b''.join((canonical_json(x)+'\n').encode() for x in v):raise SpatialAnchorRepeatError('RUN_INVALID')
 return v
def _scientific_output_projection(row:dict[str,Any])->dict[str,Any]:
 """Return output identity without per-run measurement values.

 Raw constrained text, generated-token metadata, parsed components, frozen
 bindings, and all static constraint fields remain bound.  Only the explicitly
 versioned wall-clock measurement is excluded.
 """
 value={k:v for k,v in row.items() if k not in {'cache_key','cache_hit','image_path','image_path_sha256','canonical_result_sha256'}}
 metadata=value.get('constraint_metadata')
 if isinstance(metadata,dict) and isinstance(metadata.get('execution'),dict):
  metadata={**metadata,'execution':{k:v for k,v in metadata['execution'].items() if k not in _DYNAMIC_EXECUTION_FIELDS}}
  value={**value,'constraint_metadata':metadata}
 return value
def _dynamic_execution_projection(row:dict[str,Any])->dict[str,Any]:
 execution=row.get('constraint_metadata',{}).get('execution',{}) if isinstance(row.get('constraint_metadata'),dict) else {}
 return {k:execution.get(k) for k in _DYNAMIC_EXECUTION_FIELDS if k in execution}
def compare(*,run_a:Path,run_b:Path,output_dir:Path)->dict[str,Any]:
 if output_dir.exists() and any(output_dir.iterdir()):raise SpatialAnchorRepeatError('OUTPUT_DIRECTORY_MUST_BE_EMPTY')
 ma,mb=_obj(run_a/'run'/'v2_tal_stage3g_v3_2_manifest.json'),_obj(run_b/'run'/'v2_tal_stage3g_v3_2_manifest.json')
 a,b=_rows(run_a/'run'/'v2_tal_spatial_anchor_groundings_v3_2.jsonl'),_rows(run_b/'run'/'v2_tal_spatial_anchor_groundings_v3_2.jsonl')
 key=lambda x:(x['spatial_grounding_task_id'],x['anchor_candidate_id'])
 aa,bb={key(x):x for x in a},{key(x):x for x in b}
 if len(aa)!=len(a) or len(bb)!=len(b):raise SpatialAnchorRepeatError('DUPLICATE_TASK_ANCHOR_KEY')
 keys=sorted(set(aa)|set(bb));paired=[k for k in keys if k in aa and k in bb]
 def artifact_projection(x):return {k:v for k,v in x.items() if k not in {'raw_response','cache_key','cache_hit','image_path','image_path_sha256'}}
 raw=sum(aa[k]['raw_response']==bb[k]['raw_response'] for k in paired);artifact_exact=sum(artifact_projection(aa[k])==artifact_projection(bb[k]) for k in paired);scientific_exact=sum(_scientific_output_projection(aa[k])==_scientific_output_projection(bb[k]) for k in paired);status=sum((aa[k]['status'],aa[k]['failure_reason'])==(bb[k]['status'],bb[k]['failure_reason']) for k in paired)
 roles=[];bbox=[]
 for k in paired:
  ca={x['role']:x for x in aa[k]['components']};cb={x['role']:x for x in bb[k]['components']}
  for r in set(ca)|set(cb):
   roles.append(ca.get(r,{}).get('visibility')==cb.get(r,{}).get('visibility'));bbox.append(ca.get(r,{}).get('bbox_2d_raw')==cb.get(r,{}).get('bbox_2d_raw'))
 coords=[abs(x-y) for k in paired for ca,cb in [(aa[k]['components'],bb[k]['components'])] for x,y in zip([z for c in ca for z in (c['bbox_2d_raw'] or [])],[z for c in cb for z in (c['bbox_2d_raw'] or [])])]
 scientific_a=[_scientific_output_projection(aa[k]) for k in paired]
 scientific_b=[_scientific_output_projection(bb[k]) for k in paired]
 dynamic_differences=sum(_dynamic_execution_projection(aa[k])!=_dynamic_execution_projection(bb[k]) for k in paired)
 bindings={k:ma.get(k)==mb.get(k) for k in ('stage3f_manifest_sha256','upstream_sha256','policy_sha256','config_sha256')}
 passed=(len(a)==len(b)==len(paired) and raw==len(paired) and scientific_exact==len(paired)
         and status==len(paired) and all(roles) and all(bbox) and all(bindings.values()))
 result={'format':'relive-v2-spatial-anchor-grounding-v3.2-independent-repeat-v2','status':'PASS' if passed else 'FAIL','comparison_contract_version':'relive-v2-stage3g-v3.2-repeat-comparison-v2','scientific_output_projection_version':SCIENTIFIC_OUTPUT_PROJECTION_VERSION,'excluded_dynamic_execution_fields':list(_DYNAMIC_EXECUTION_FIELDS),'task_anchor_count_a':len(a),'task_anchor_count_b':len(b),'missing_or_duplicate_key_count':len(keys)-len(paired),'exact_raw_response_match_count':raw,'exact_artifact_result_match_count':artifact_exact,'exact_parsed_result_match_count':artifact_exact,'scientific_output_exact_match_count':scientific_exact,'scientific_output_hash_a':stable_hash(scientific_a),'scientific_output_hash_b':stable_hash(scientific_b),'scientific_output_hash_equal':stable_hash(scientific_a)==stable_hash(scientific_b),'dynamic_execution_metadata_difference_count':dynamic_differences,'exact_status_failure_match_count':status,'role_visibility_agreement_count':sum(roles),'bbox_exact_match_count':sum(bbox),'max_coordinate_absolute_difference':max(coords,default=0),'legacy_canonical_grounding_result_hash_equal':ma.get('canonical_grounding_result_sha256')==mb.get('canonical_grounding_result_sha256'),'canonical_grounding_result_hash_equal':ma.get('canonical_grounding_result_sha256')==mb.get('canonical_grounding_result_sha256'),'scientific_manifest_bindings_equal':bindings,'model_calls_made':0,'backend_loaded':False,'cache_opened':False,'certificate_created':False,'new_verified_count':0,'gt_used':False}
 result['comparison_content_sha256']=stable_hash(result);output_dir.mkdir(parents=True);(output_dir/'v2_tal_stage3g_v3_2_independent_repeat_comparison.json').write_bytes((canonical_json(result)+'\n').encode());return result
