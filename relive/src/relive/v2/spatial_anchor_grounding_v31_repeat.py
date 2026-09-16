"""Zero-model comparison of two independent Stage 3G-A v3.1 fresh runs."""
from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Any
from relive.storage.artifacts import canonical_json,stable_hash
from .task_selection import TALSelectionError,strict_json_loads,strict_jsonl
class SpatialAnchorRepeatError(ValueError):pass
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
def compare(*,run_a:Path,run_b:Path,output_dir:Path)->dict[str,Any]:
 if output_dir.exists() and any(output_dir.iterdir()):raise SpatialAnchorRepeatError('OUTPUT_DIRECTORY_MUST_BE_EMPTY')
 ma,mb=_obj(run_a/'run'/'v2_tal_stage3g_v3_1_manifest.json'),_obj(run_b/'run'/'v2_tal_stage3g_v3_1_manifest.json')
 a,b=_rows(run_a/'run'/'v2_tal_spatial_anchor_groundings_v3_1.jsonl'),_rows(run_b/'run'/'v2_tal_spatial_anchor_groundings_v3_1.jsonl')
 key=lambda x:(x['spatial_grounding_task_id'],x['anchor_candidate_id'])
 aa,bb={key(x):x for x in a},{key(x):x for x in b}
 if len(aa)!=len(a) or len(bb)!=len(b):raise SpatialAnchorRepeatError('DUPLICATE_TASK_ANCHOR_KEY')
 keys=sorted(set(aa)|set(bb));paired=[k for k in keys if k in aa and k in bb]
 def parsed(x):return {k:v for k,v in x.items() if k not in {'raw_response','cache_key','cache_hit','image_path','image_path_sha256'}}
 raw=sum(aa[k]['raw_response']==bb[k]['raw_response'] for k in paired);parsed_exact=sum(parsed(aa[k])==parsed(bb[k]) for k in paired);status=sum((aa[k]['status'],aa[k]['failure_reason'])==(bb[k]['status'],bb[k]['failure_reason']) for k in paired)
 roles=[];bbox=[]
 for k in paired:
  ca={x['role']:x for x in aa[k]['components']};cb={x['role']:x for x in bb[k]['components']}
  for r in set(ca)|set(cb):
   roles.append(ca.get(r,{}).get('visibility')==cb.get(r,{}).get('visibility'));bbox.append(ca.get(r,{}).get('bbox_2d_raw')==cb.get(r,{}).get('bbox_2d_raw'))
 coords=[abs(x-y) for k in paired for ca,cb in [(aa[k]['components'],bb[k]['components'])] for x,y in zip([z for c in ca for z in (c['bbox_2d_raw'] or [])],[z for c in cb for z in (c['bbox_2d_raw'] or [])])]
 result={'format':'relive-v2-spatial-anchor-grounding-v3.1-independent-repeat-v1','status':'PASS','task_anchor_count_a':len(a),'task_anchor_count_b':len(b),'missing_or_duplicate_key_count':len(keys)-len(paired),'exact_raw_response_match_count':raw,'exact_parsed_result_match_count':parsed_exact,'exact_status_failure_match_count':status,'role_visibility_agreement_count':sum(roles),'bbox_exact_match_count':sum(bbox),'max_coordinate_absolute_difference':max(coords,default=0),'canonical_grounding_result_hash_equal':ma.get('canonical_grounding_result_sha256')==mb.get('canonical_grounding_result_sha256'),'scientific_manifest_bindings_equal':{k:ma.get(k)==mb.get(k) for k in ('stage3f_manifest_sha256','upstream_sha256','policy_sha256','config_sha256')},'model_calls_made':0,'backend_loaded':False,'cache_opened':False,'certificate_created':False,'new_verified_count':0,'gt_used':False}
 result['comparison_content_sha256']=stable_hash(result);output_dir.mkdir(parents=True);(output_dir/'v2_tal_stage3g_v3_1_independent_repeat_comparison.json').write_bytes((canonical_json(result)+'\n').encode());return result
