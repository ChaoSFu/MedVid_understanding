"""Read-only output-contract diagnostics for immutable Stage 3G-A v2 runs."""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
from typing import Any
from relive.storage.artifacts import canonical_json, stable_hash
from .spatial_anchor_grounding import SpatialAnchorGroundingError, rows, obj, sha

FORMAT='relive-v2-spatial-anchor-grounding-v2-output-contract-diagnostic-v1'

def _write(path:Path, value:Any)->None:
 if path.exists():raise ValueError('IMMUTABLE_OUTPUT_EXISTS')
 path.parent.mkdir(parents=True,exist_ok=True)
 raw=(canonical_json(value)+'\n').encode() if isinstance(value,dict) else b''.join((canonical_json(x)+'\n').encode() for x in value)
 path.write_bytes(raw)

def _closure(raw:str)->str:
 text=raw.strip()
 if text.startswith('{') and text.endswith('}'):return 'CLOSED_OBJECT'
 if text.startswith('[') and text.endswith(']'):return 'CLOSED_ARRAY'
 return 'UNCLOSED_OR_OTHER'

def diagnose(*,v2_output_dir:Path,output_dir:Path,max_examples_per_failure:int=3)->dict[str,Any]:
 """Inspect copied values only; this never rewrites, reparses, or repairs v2."""
 if max_examples_per_failure < 1 or max_examples_per_failure > 5:raise ValueError('EXAMPLE_LIMIT_INVALID')
 if output_dir.exists() and any(output_dir.iterdir()):raise ValueError('OUTPUT_DIRECTORY_MUST_BE_EMPTY')
 try:
  manifest=obj(v2_output_dir/'run'/'v2_tal_stage3g_manifest.json','V2_MANIFEST')
  source=rows(v2_output_dir/'run'/'v2_tal_spatial_anchor_groundings.jsonl','V2_GROUNDINGS')
  pre=obj(v2_output_dir/'stage3g_preflight.json','V2_PREFLIGHT')
 except SpatialAnchorGroundingError as exc:raise ValueError('V2_ARTIFACT_INVALID') from exc
 if manifest.get('grounding_result_count')!=len(source):raise ValueError('V2_COUNT_MISMATCH')
 if manifest.get('format')!='relive-v2-spatial-anchor-grounding-v1':raise ValueError('V2_FORMAT_INVALID')
 records=[]
 for row in source:
  raw=row.get('raw_response')
  if not isinstance(raw,str):raise ValueError('V2_RAW_RESPONSE_INVALID')
  records.append({'failure_reason':row.get('failure_reason') or 'NONE','raw_response_sha256':hashlib.sha256(raw.encode()).hexdigest(),'raw_response_closure':_closure(raw),'finish_reason':'NOT_RECORDED_IN_V2','generated_token_count':'NOT_RECORDED_IN_V2','reached_max_new_tokens':'NOT_RECORDED_IN_V2'})
 counts={
  'failure_reason':dict(sorted(Counter(r['failure_reason'] for r in records).items())),
  'raw_response_closure':dict(sorted(Counter(r['raw_response_closure'] for r in records).items())),
  'finish_reason':dict(sorted(Counter(r['finish_reason'] for r in records).items())),
  'generated_token_count':dict(sorted(Counter(str(r['generated_token_count']) for r in records).items())),
  'reached_max_new_tokens':dict(sorted(Counter(str(r['reached_max_new_tokens']) for r in records).items())),
 }
 grouped=defaultdict(list)
 for record in records:grouped[record['failure_reason']].append(record)
 examples=[]
 for reason in sorted(grouped):
  # Predeclared, anonymous rule: ascending raw SHA; never expose task, anchor, image, or raw text.
  for item in sorted(grouped[reason],key=lambda x:x['raw_response_sha256'])[:max_examples_per_failure]:
   examples.append({'failure_reason':reason,'raw_response_sha256':item['raw_response_sha256'],'raw_response_closure':item['raw_response_closure'],'finish_reason':'NOT_RECORDED_IN_V2','generated_token_count':'NOT_RECORDED_IN_V2','reached_max_new_tokens':'NOT_RECORDED_IN_V2','selection_rule':'PER_FAILURE_ASCENDING_RAW_RESPONSE_SHA256_FIRST_N_ANONYMOUS'})
 report={'format':FORMAT,'status':'PASS_WITH_HISTORICAL_GENERATION_METADATA_UNAVAILABLE','v2_output_dir_sha256_binding':stable_hash({'preflight_sha256':sha(v2_output_dir/'stage3g_preflight.json'),'manifest_sha256':sha(v2_output_dir/'run'/'v2_tal_stage3g_manifest.json'),'groundings_sha256':sha(v2_output_dir/'run'/'v2_tal_spatial_anchor_groundings.jsonl')}),'v2_result_count':len(source),'v2_max_new_tokens':pre.get('generation_parameters',{}).get('max_new_tokens'),'v2_generation_metadata_available':False,'read_only_diagnostic':True,'v2_scientific_artifacts_modified':False,'counts':counts,'example_selection_rule':'PER_FAILURE_ASCENDING_RAW_RESPONSE_SHA256_FIRST_N_ANONYMOUS','example_limit_per_failure':max_examples_per_failure,'examples_sha256':stable_hash(examples),'model_calls_made':0,'backend_loaded':False,'cache_opened':False,'gt_used':False,'certificate_created':False,'new_verified_count':0,'certificate_status':'NOT_APPLICABLE'}
 report['report_content_sha256']=stable_hash(report)
 _write(output_dir/'v2_stage3g_output_contract_diagnostic.json',report);_write(output_dir/'v2_stage3g_output_contract_anonymous_examples.jsonl',examples)
 return report
