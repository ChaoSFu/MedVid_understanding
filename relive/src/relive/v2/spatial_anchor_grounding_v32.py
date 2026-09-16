"""ReliVE-v2 Stage 3G-A v3.2 strict JSON-object anchor grounding."""
from __future__ import annotations
import hashlib, math, re, shutil
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Protocol
from PIL import Image, ImageDraw
from relive.backends import make_backend
from relive.config import load_config
from relive.storage.artifacts import ArtifactStore, canonical_json, stable_hash
from .spatial_evidence_planning import SpatialPlanError, validate as validate_stage3f
from .task_selection import TALSelectionError, strict_json_loads, strict_jsonl
from .token_json_constraint import GroundingJsonGrammar, TokenLevelGroundingConstraint, GRAMMAR_VERSION, IMPLEMENTATION_VERSION

FORMAT="relive-v2-spatial-anchor-grounding-v3.2"
CANDIDATE="SPATIAL_ANCHOR_GROUNDING_CANDIDATE_UNVERIFIED"
VISIBILITY=("VISIBLE","NOT_VISIBLE","AMBIGUOUS")
ROLES={
 "BASE_PLATE":"the visible base plate or adhesive base component",
 "TARGET_SKIN_AREA":"the skin area on which the base plate is placed",
 "ALIGNMENT_INTERFACE":"the visible contact or alignment interface between base plate and target skin",
 "OPERATOR_HAND":"the operator hand interacting with the base plate",
 "HAND_BASE_INTERFACE":"the visible contact interface between operator hand and base plate",
 "BASE_SKIN_INTERFACE":"the visible contact interface between base plate and skin",
 "ATTACHMENT_INTERFACE":"the visible attached interface between base plate and target skin",
 "BASE_REMAINS_ALIGNED_OR_ATTACHED_AFTER_INTERACTION":"the visible post-interaction base-to-skin state",
 "OPERATOR_HAND_NOT_REQUIRED_INSIDE_TARGET_REGION":"a contextual absence condition; localize it only if visibly distinguishable",
}

class SpatialAnchorGrounder(Protocol):
 """Backend-neutral one-image, one-task anchor-grounding contract."""
 def fingerprint(self) -> dict[str, Any]: ...
 def infer(self, request: dict[str, Any]) -> str: ...
 def infer_with_generation_metadata(self, request: dict[str, Any]) -> dict[str, Any]: ...
 def infer_with_token_constraint(self, request: dict[str, Any], constraint: Any) -> dict[str, Any]: ...
 def audit_token_constraint(self, constraint: Any) -> dict[str, Any]: ...

class SpatialAnchorGroundingError(ValueError): pass

def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def jsonl(rows:list[dict[str,Any]])->bytes:return b''.join((canonical_json(x)+'\n').encode() for x in rows)
def write(path:Path,value:Any)->None:
 if path.exists():raise SpatialAnchorGroundingError('IMMUTABLE_OUTPUT_EXISTS')
 path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((canonical_json(value)+'\n').encode() if isinstance(value,dict) else jsonl(value))
def obj(path:Path,code:str)->dict[str,Any]:
 try: raw=path.read_bytes();value=strict_json_loads(raw.decode(),error_code=code)
 except (OSError,UnicodeDecodeError,TALSelectionError) as exc:raise SpatialAnchorGroundingError(code+'_INVALID') from exc
 if not isinstance(value,dict) or raw!=(canonical_json(value)+'\n').encode():raise SpatialAnchorGroundingError(code+'_NONCANONICAL')
 return value
def rows(path:Path,code:str)->list[dict[str,Any]]:
 try: value=list(strict_jsonl(path,error_code=code));raw=path.read_bytes()
 except (OSError,TALSelectionError) as exc:raise SpatialAnchorGroundingError(code+'_INVALID') from exc
 if raw!=jsonl(value) or any(not isinstance(x,dict) for x in value):raise SpatialAnchorGroundingError(code+'_NONCANONICAL')
 return value

def _frame(anchor:dict[str,Any],frames:list[dict[str,Any]])->dict[str,Any]:
 hits=[x for x in frames if x.get('source_frame_reference')==anchor.get('source_frame_reference') and x.get('timestamp_seconds')==anchor.get('timestamp_seconds')]
 identities={(x.get('frame_sha256'),x.get('resolved_frame_path')) for x in hits}
 if not hits or len(identities)!=1:raise SpatialAnchorGroundingError('ANCHOR_FRAME_REFERENCE_INVALID')
 f=hits[0]; key='unique_visual_frame_'+stable_hash({'frame_sha256':f['frame_sha256'],'source_frame_reference':f['source_frame_reference'],'timestamp_seconds':f['timestamp_seconds']})[:24]
 if key!=anchor.get('unique_visual_frame_id'):raise SpatialAnchorGroundingError('ANCHOR_UNIQUE_VISUAL_ID_INVALID')
 if sorted(anchor.get('logical_aliases',[]))!=sorted(x['frame_order'] for x in frames if x['frame_sha256']==f['frame_sha256'] and x['source_frame_reference']==f['source_frame_reference'] and x['timestamp_seconds']==f['timestamp_seconds']):raise SpatialAnchorGroundingError('ANCHOR_ALIAS_BINDING_INVALID')
 return f

def _image(frame:dict[str,Any])->tuple[str,int,int]:
 path=Path(frame['resolved_frame_path'])
 if sha(path)!=frame['frame_sha256']:raise SpatialAnchorGroundingError('FRAME_SHA256_MISMATCH')
 try:
  with Image.open(path) as im: im.load();width,height=im.size
 except (OSError,ValueError) as exc:raise SpatialAnchorGroundingError('FRAME_IMAGE_INVALID') from exc
 if not(width>0 and height>0):raise SpatialAnchorGroundingError('FRAME_DIMENSIONS_INVALID')
 return str(path),width,height

def _upstream(stage3f:Path,stage3c:Path,stage3d:Path,stage3e:Path,index:Path)->dict[str,str]:
 try: valid=validate_stage3f(stage3f)
 except SpatialPlanError as exc:raise SpatialAnchorGroundingError('FROZEN_STAGE3F_ARTIFACT_INVALID') from exc
 if valid.get('status')!='PASS' or not valid.get('ready_for_stage3g_spatial_grounding'):raise SpatialAnchorGroundingError('STAGE3F_NOT_READY')
 m=obj(stage3f/'v2_tal_spatial_plan_manifest.json','STAGE3F_MANIFEST'); expected=m.get('upstream_sha256')
 actual={'stage3c_manifest_sha256':sha(stage3c/'v2_stage3c_manifest.json'),'stage3d_call_plan_sha256':sha(stage3d/'v2_stage3d_call_plan.json'),'stage3d_run_summary_sha256':sha(stage3d/'v2_stage3d_run_summary.json'),'stage3e_manifest_sha256':sha(stage3e/'v2_stage3e_manifest.json'),'video_index_manifest_sha256':sha(index/'v2_video_index_manifest.json')}
 if expected!=actual:raise SpatialAnchorGroundingError('STAGE3F_UPSTREAM_BINDING_MISMATCH')
 return {'stage3f_manifest_sha256':sha(stage3f/'v2_tal_spatial_plan_manifest.json'),**actual}

def _effective_config(config:dict[str,Any],policy:dict[str,Any])->dict[str,Any]:
 if type(policy.get('max_new_tokens')) is not int or not 128 < policy['max_new_tokens'] <= 1024:
  raise SpatialAnchorGroundingError('GENERATION_POLICY_INVALID')
 out=deepcopy(config);generation=out.get('backend',{}).get('generation')
 if not isinstance(generation,dict) or generation.get('do_sample') is not False:
  raise SpatialAnchorGroundingError('GENERATION_POLICY_INVALID')
 generation['max_new_tokens']=policy['max_new_tokens']
 return out

def _closure(raw:str)->str:
 """Describe raw structure only; this function never repairs or accepts it."""
 text=raw.strip()
 if text.startswith('{') and text.endswith('}'):return 'CLOSED_OBJECT'
 if text.startswith('[') and text.endswith(']'):return 'CLOSED_ARRAY'
 return 'UNCLOSED_OR_OTHER'

def _smoke_selection(path:Path|None, entries:list[dict[str,Any]])->tuple[list[dict[str,Any]],dict[str,Any]]:
 if path is None:return entries,{'selection_mode':'FULL_FROZEN_COHORT','smoke_selection_manifest_sha256':None}
 selected=rows(path,'SMOKE_SELECTION');ids=[row.get('anchor_candidate_id') for row in selected]
 if (not ids or any(not isinstance(value,str) for value in ids) or len(ids)!=len(set(ids))
     or any(set(row)!={'anchor_candidate_id','selection_reason','source_v2_result_sha256'} for row in selected)):raise SpatialAnchorGroundingError('SMOKE_SELECTION_INVALID')
 by_id={entry['anchor_candidate_id']:entry for entry in entries}
 if any(value not in by_id for value in ids):raise SpatialAnchorGroundingError('SMOKE_SELECTION_UNKNOWN_ANCHOR')
 return [by_id[value] for value in ids],{'selection_mode':'OUTPUT_CONTRACT_ENGINEERING_SMOKE_NOT_SCIENTIFIC','smoke_selection_manifest_sha256':sha(path)}

def _prompt(entry:dict[str,Any])->str:
 required=entry['required_component_roles'];context=entry['contextual_requirements'];all_roles=required+context
 if len(all_roles)!=len(set(all_roles)) or any(role not in ROLES for role in all_roles):raise SpatialAnchorGroundingError('ROLE_CONTRACT_INVALID')
 from importlib.resources import files
 text=files('relive').joinpath('prompts','v2_tal_spatial_anchor_grounding_v3_2.txt').read_text(encoding='utf-8').format(observation_claim=entry['observation_claim'],observation_role=entry['observation_role'],required_roles=', '.join(required),contextual_roles=', '.join(context) or 'none',role_descriptions='\n'.join(f'- {r}: {ROLES[r]}' for r in all_roles)).rstrip()
 return text

def _parse(raw:str,expected:list[str],width:int,height:int)->tuple[list[dict[str,Any]],str|None]:
 text=raw.strip()
 # v3_2's generated object contract does not permit Markdown fences or arrays.
 if not (text.startswith('{') and text.endswith('}')):return [],'MODEL_OUTPUT_PARSE_FAILURE'
 try:value=strict_json_loads(text,error_code='GROUNDING_JSON')
 except TALSelectionError:return [],'MODEL_OUTPUT_PARSE_FAILURE'
 if not isinstance(value,dict) or set(value)!={'components'} or not isinstance(value['components'],list):return [],'MODEL_OUTPUT_SCHEMA_VIOLATION'
 got=[];seen=set()
 for c in value['components']:
  if not isinstance(c,dict) or set(c)!={'role','visibility','bbox_2d'}:return [],'MODEL_OUTPUT_SCHEMA_VIOLATION'
  role,vis,box=c['role'],c['visibility'],c['bbox_2d']
  if role not in expected:return [],'UNKNOWN_COMPONENT_ROLE'
  if role in seen:return [],'DUPLICATE_COMPONENT_ROLE'
  seen.add(role)
  if vis not in VISIBILITY:return [],'MODEL_OUTPUT_SCHEMA_VIOLATION'
  if vis=='VISIBLE':
   if not isinstance(box,list) or len(box)!=4 or any(type(x) is not int for x in box):return [],'INVALID_BOUNDING_BOX'
   x1,y1,x2,y2=box
   if not(0<=x1<x2<=1000 and 0<=y1<y2<=1000):return [],'INVALID_BOUNDING_BOX'
   norm=[x1/1000,y1/1000,x2/1000,y2/1000];pix=[math.floor(norm[0]*width),math.floor(norm[1]*height),math.ceil(norm[2]*width),math.ceil(norm[3]*height)]
  else:
   if box is not None:return [],'MODEL_OUTPUT_SCHEMA_VIOLATION'
   norm=pix=None
  got.append({'role':role,'visibility':vis,'bbox_2d_raw':box,'bbox_normalized_xyxy':norm,'bbox_pixel_xyxy':pix})
 if set(seen)!=set(expected):return [],'REQUIRED_COMPONENT_MISSING'
 return sorted(got,key=lambda x:expected.index(x['role'])),None

def _status(components:list[dict[str,Any]],required:list[str],failure:str|None)->str:
 if failure:return CANDIDATE
 v={x['role']:x['visibility'] for x in components}
 if any(v[r]=='NOT_VISIBLE' for r in required):return 'REQUIRED_COMPONENT_NOT_VISIBLE'
 if any(v[r]=='AMBIGUOUS' for r in required):return 'REQUIRED_COMPONENT_AMBIGUOUS'
 return 'COMPONENTS_LOCALIZED_UNVERIFIED'

def _plan(stage3f:Path,index:Path)->list[dict[str,Any]]:
 plans=rows(stage3f/'v2_tal_spatial_evidence_plans.jsonl','PLANS'); tasks={x['spatial_grounding_task_id']:x for x in rows(stage3f/'v2_tal_spatial_grounding_tasks.jsonl','TASKS')}; frames=rows(index/'v2_video_index.jsonl','VIDEO_INDEX')
 if len(frames)!=1 or not isinstance(frames[0].get('frames'),list):raise SpatialAnchorGroundingError('VIDEO_INDEX_SCHEMA_INVALID')
 task_claim={}
 for p in plans:
  tid=p.get('spatial_grounding_task_id'); cid=p.get('observation_claim_id')
  if tid not in tasks:raise SpatialAnchorGroundingError('PLAN_TASK_REFERENCE_INVALID')
  task_claim.setdefault(tid,cid)
  if task_claim[tid]!=cid:raise SpatialAnchorGroundingError('TASK_SEMANTICS_CONFLICT')
 claims={x['claim_id']:x for x in rows(stage3f.parent/'__never__','NO')} if False else None
 # Claims are reached from Stage 3C by caller below; task-level semantics are immutable here.
 return []

def _entries(stage3f:Path,stage3c:Path,index:Path)->list[dict[str,Any]]:
 tasks=rows(stage3f/'v2_tal_spatial_grounding_tasks.jsonl','TASKS'); plans=rows(stage3f/'v2_tal_spatial_evidence_plans.jsonl','PLANS'); claims={x['claim_id']:x for x in rows(stage3c/'v2_stage3c_observation_claims.jsonl','CLAIMS')}; iv=rows(index/'v2_video_index.jsonl','VIDEO_INDEX')
 if len(iv)!=1:raise SpatialAnchorGroundingError('VIDEO_INDEX_SCHEMA_INVALID')
 frames=iv[0].get('frames',[]); claim_by_task={}
 for plan in plans:
  tid,cid=plan.get('spatial_grounding_task_id'),plan.get('observation_claim_id')
  if tid in claim_by_task and claim_by_task[tid]!=cid:raise SpatialAnchorGroundingError('TASK_SEMANTICS_CONFLICT')
  claim_by_task[tid]=cid
 out=[]
 for task in sorted(tasks,key=lambda x:x['spatial_grounding_task_id']):
  cid=claim_by_task.get(task['spatial_grounding_task_id']);claim=claims.get(cid)
  if not claim:raise SpatialAnchorGroundingError('TASK_CLAIM_BINDING_INVALID')
  expected=task['required_component_roles']+task['contextual_requirements']
  for anchor in task.get('anchor_candidates',[]):
   frame=_frame(anchor,frames);path,width,height=_image(frame)
   base={'spatial_grounding_task_id':task['spatial_grounding_task_id'],'anchor_candidate_id':'anchor_'+stable_hash({'task':task['spatial_grounding_task_id'],'anchor':anchor['unique_visual_frame_id']})[:24],'unique_visual_frame_id':anchor['unique_visual_frame_id'],'frame_sha256':frame['frame_sha256'],'source_frame_reference':frame['source_frame_reference'],'timestamp_seconds':frame['timestamp_seconds'],'logical_aliases':anchor['logical_aliases'],'image_path':path,'image_width':width,'image_height':height,'observation_claim_id':cid,'observation_claim':claim.get('surface'),'observation_role':task['observation_role'],'required_component_roles':task['required_component_roles'],'contextual_requirements':task['contextual_requirements'],'geometry_type':task['geometry_type'],'stage3f_task_anchor_sha256':stable_hash({'task':task,'anchor':anchor})}
   base['prompt']=_prompt(base);base['prompt_sha256']=hashlib.sha256(base['prompt'].encode()).hexdigest();out.append(base)
 if len({(x['spatial_grounding_task_id'],x['unique_visual_frame_id']) for x in out})!=len(out):raise SpatialAnchorGroundingError('DUPLICATE_TASK_ANCHOR_ENUMERATION')
 return out

def preflight(*,stage3f_dir:Path,stage3c_dir:Path,stage3d_dir:Path,stage3e_dir:Path,video_index_dir:Path,config_path:Path,policy_path:Path,output_dir:Path,smoke_selection_manifest:Path|None=None,backend_factory:Callable[[dict[str,Any]],Any]=make_backend)->dict[str,Any]:
 if output_dir.exists() and any(output_dir.iterdir()):raise SpatialAnchorGroundingError('OUTPUT_DIRECTORY_MUST_BE_EMPTY')
 upstream=_upstream(stage3f_dir,stage3c_dir,stage3d_dir,stage3e_dir,video_index_dir);policy=obj(policy_path,'POLICY')
 if (policy.get('format')!='relive-v2-spatial-anchor-grounding-v3.2-policy-v1'
     or policy.get('json_schema_constrained_decoding_supported') is not False
     or policy.get('grammar_version')!=GRAMMAR_VERSION
     or policy.get('grammar_implementation_version')!=IMPLEMENTATION_VERSION
     or policy.get('structured_generation_enforcement')!='TOKEN_LEVEL_PREFIX_ALLOWED_LOGITS_FAIL_CLOSED'):raise SpatialAnchorGroundingError('POLICY_INVALID')
 config=load_config(config_path)
 if config.get('backend',{}).get('kind')!='local_hf':raise SpatialAnchorGroundingError('NATIVE_LOCAL_HF_REQUIRED')
 effective=_effective_config(config,policy)
 entries,selection=_smoke_selection(smoke_selection_manifest,_entries(stage3f_dir,stage3c_dir,video_index_dir));backend=backend_factory(effective['backend']);fingerprint=backend.fingerprint()
 if not callable(getattr(backend,'infer_with_token_constraint',None)) or not callable(getattr(backend,'audit_token_constraint',None)):raise SpatialAnchorGroundingError('TOKEN_LEVEL_CONSTRAINT_INTERFACE_UNAVAILABLE')
 frozen=[];grammar_audits=[]
 for e in entries:
  grammar=GroundingJsonGrammar(e['required_component_roles']+e['contextual_requirements'])
  audit=backend.audit_token_constraint(TokenLevelGroundingConstraint(grammar))
  if (not isinstance(audit,dict) or not isinstance(audit.get('binding'),dict) or not isinstance(audit.get('tokenizer_binding'),dict)
      or audit['binding'].get('grammar_spec_sha256')!=grammar.spec_sha256 or type(audit.get('initial_allowed_token_count')) is not int
      or audit['initial_allowed_token_count'] <= 0):raise SpatialAnchorGroundingError('TOKEN_CONSTRAINT_AUDIT_INVALID')
  grammar_audits.append({'grammar_spec_sha256':grammar.spec_sha256,'tokenizer_binding':audit['tokenizer_binding'],'initial_allowed_token_count':audit['initial_allowed_token_count']})
  frozen.append(e|{'image_path_sha256':hashlib.sha256(e['image_path'].encode()).hexdigest(),'grammar_spec':grammar.spec,'grammar_spec_sha256':grammar.spec_sha256})
 out={'format':FORMAT,'status':'PASS','mode':'preflight','provider_type':policy['provider_type'],'provider_name':policy['provider_name'],'stage3f_manifest_sha256':upstream['stage3f_manifest_sha256'],'upstream_sha256':upstream,'policy_sha256':sha(policy_path),'config_sha256':sha(config_path),'generation_parameters':effective['backend']['generation'],'generation_contract':{'generation_contract_version':policy['generation_contract_version'],'native_generation_metadata_supported':True,'json_schema_constrained_decoding_supported':False,'structured_generation_enforcement':policy['structured_generation_enforcement'],'enforcement':'ACTUAL_GENERATE_LOGITS_MASK_WITH_FAIL_CLOSED_PREFIX_GRAMMAR','grammar_version':GRAMMAR_VERSION,'grammar_implementation_version':IMPLEMENTATION_VERSION},'model_fingerprint':fingerprint,'token_constraint_preflight_audits':grammar_audits,'token_constraint_preflight_audit_sha256':stable_hash(grammar_audits),'planned_model_calls':len(entries),'unique_task_anchor_count':len(entries),'frozen_calls':frozen,**selection,'model_calls_made':0,'new_model_calls':0,'cache_hits':0,'backend_loaded':True,'cache_opened':False,'frames_read':len(entries),'unique_frame_bytes_opened':len({e['frame_sha256'] for e in entries}),'videos_read':0,'gt_used':False,'assistant_or_gt_values_accessed':False,'boxes_created':0,'points_created':0,'masks_created':0,'masklets_created':0,'support_tubes_created':0,'certificate_created':False,'new_verified_count':0,'certificate_status':'NOT_APPLICABLE'}
 out['preflight_content_sha256']=stable_hash(out);output_dir.mkdir(parents=True);write(output_dir/'stage3g_v3_2_preflight.json',out);return out

def _preflight(root:Path)->dict[str,Any]:
 p=obj(root/'stage3g_v3_2_preflight.json','PREFLIGHT')
 if p.get('preflight_content_sha256')!=stable_hash({k:v for k,v in p.items() if k!='preflight_content_sha256'}):raise SpatialAnchorGroundingError('PREFLIGHT_TAMPERED')
 return p

def _reverify(entry:dict[str,Any])->None:
 path=Path(entry['image_path'])
 if hashlib.sha256(str(path).encode()).hexdigest()!=entry['image_path_sha256']:raise SpatialAnchorGroundingError('IMAGE_PATH_BINDING_INVALID')
 actual,w,h=_image({'resolved_frame_path':str(path),'frame_sha256':entry['frame_sha256']})
 if actual!=str(path) or (w,h)!=(entry['image_width'],entry['image_height']):raise SpatialAnchorGroundingError('FRAME_DIMENSION_DRIFT')

def _cache_key(entry:dict[str,Any],pre:dict[str,Any],policy:dict[str,Any])->str:
 return stable_hash({'format':FORMAT,'provider':{'type':policy['provider_type'],'name':policy['provider_name']},'model_fingerprint':pre['model_fingerprint'],'config_sha256':pre['config_sha256'],'stage3f_manifest_sha256':pre['stage3f_manifest_sha256'],'grounding_task_id':entry['spatial_grounding_task_id'],'anchor_candidate_id':entry['anchor_candidate_id'],'unique_visual_frame_id':entry['unique_visual_frame_id'],'frame_sha256':entry['frame_sha256'],'prompt_sha256':entry['prompt_sha256'],'parser_version':policy['parser_version'],'generation_contract_version':policy['generation_contract_version'],'generation':pre['generation_parameters'],'coordinate_policy':policy['coordinate_conversion_policy_version'],'grammar_spec_sha256':entry['grammar_spec_sha256'],'grammar_version':policy['grammar_version'],'grammar_implementation_version':policy['grammar_implementation_version']})

def _review(run_dir:Path,row:dict[str,Any])->dict[str,Any]:
 directory=run_dir/'review_packet'/row['anchor_candidate_id'];directory.mkdir(parents=True,exist_ok=True)
 source=Path(row['image_path']);copy=directory/'frame.png'
 with Image.open(source) as image:
  image.load();base=image.convert('RGB');base.save(copy)
  canvas=base.copy();draw=ImageDraw.Draw(canvas)
  for component in row.get('components',[]):
   box=component.get('bbox_pixel_xyxy')
   if box:
    draw.rectangle(box,outline=(255,0,0),width=max(1,base.width//200));draw.text((box[0],max(0,box[1]-12)),component['role'],fill=(255,0,0))
  overlay=directory/'overlay.png';canvas.save(overlay)
 metadata={k:v for k,v in row.items() if k not in {'image_path','raw_response'}}
 metadata['raw_response_sha256']=row.get('raw_response_sha256');write(directory/'metadata.json',metadata)
 return {'anchor_candidate_id':row['anchor_candidate_id'],'spatial_grounding_task_id':row['spatial_grounding_task_id'],'frame_copy_sha256':sha(copy),'overlay_sha256':sha(overlay),'metadata_sha256':sha(directory/'metadata.json'),'status':row['status'],'failure_reason':row['failure_reason']}

def execute(*,output_dir:Path,config_path:Path,policy_path:Path,mode:str,backend_factory:Callable[[dict[str,Any]],Any]=make_backend)->dict[str,Any]:
 if mode not in ('run','replay'):raise SpatialAnchorGroundingError('MODE_INVALID')
 pre=_preflight(output_dir);policy=obj(policy_path,'POLICY')
 if sha(config_path)!=pre['config_sha256'] or sha(policy_path)!=pre['policy_sha256']:raise SpatialAnchorGroundingError('CONFIG_OR_POLICY_DRIFT')
 entries=pre['frozen_calls'];expected=pre['planned_model_calls']
 if len(entries)!=expected or len({(x['spatial_grounding_task_id'],x['anchor_candidate_id']) for x in entries})!=expected:raise SpatialAnchorGroundingError('FROZEN_CALL_PLAN_INVALID')
 run_dir=output_dir/mode
 if run_dir.exists() and any(run_dir.iterdir()):raise SpatialAnchorGroundingError('IMMUTABLE_RUN_OUTPUT_EXISTS')
 cache=ArtifactStore(output_dir/'cache');backend=backend_factory(_effective_config(load_config(config_path),policy)['backend']);results=[];traces=[];new=hits=0
 for entry in entries:
  _reverify(entry)
  key=_cache_key(entry,pre,policy);stored=None
  try: stored=cache.get_json('stage3g_v3_2',key)
  except Exception: stored=None
  if stored is not None:
   result=stored.get('result')
   if not isinstance(result,dict) or stored.get('cache_key')!=key:raise SpatialAnchorGroundingError('CACHE_RECORD_INVALID')
   hits+=1;cache_hit=True
  else:
   if mode=='replay':raise SpatialAnchorGroundingError('REPLAY_CACHE_MISS')
   grammar=GroundingJsonGrammar(entry['required_component_roles']+entry['contextual_requirements'])
   if grammar.spec_sha256!=entry.get('grammar_spec_sha256') or grammar.spec!=entry.get('grammar_spec'):raise SpatialAnchorGroundingError('GRAMMAR_BINDING_DRIFT')
   generated=backend.infer_with_token_constraint({'prompt':entry['prompt'],'image_paths':[entry['image_path']],'frame_ids':[entry['unique_visual_frame_id']]},TokenLevelGroundingConstraint(grammar))
   if (not isinstance(generated,dict) or not isinstance(generated.get('raw_response'),str)
       or not isinstance(generated.get('generation_metadata'),dict) or not isinstance(generated.get('constraint_metadata'),dict)):
    raise SpatialAnchorGroundingError('GENERATION_METADATA_INVALID')
   raw=generated['raw_response'];metadata=generated['generation_metadata'];constraint_metadata=generated['constraint_metadata']
   execution=constraint_metadata.get('execution');tokenizer_binding=constraint_metadata.get('tokenizer_binding')
   if (not isinstance(execution,dict) or not isinstance(tokenizer_binding,dict) or execution.get('grammar_spec_sha256')!=entry['grammar_spec_sha256']
       or execution.get('grammar_version')!=GRAMMAR_VERSION or execution.get('implementation_version')!=IMPLEMENTATION_VERSION):
    raise SpatialAnchorGroundingError('CONSTRAINT_METADATA_INVALID')
   if (metadata.get('finish_reason') not in {'EOS_TOKEN','MAX_NEW_TOKENS','OTHER_STOP'} or type(metadata.get('generated_token_count')) is not int or metadata.get('max_new_tokens')!=pre['generation_parameters']['max_new_tokens'] or type(metadata.get('reached_max_new_tokens')) is not bool):raise SpatialAnchorGroundingError('GENERATION_METADATA_INVALID')
   components,failure=_parse(raw,entry['required_component_roles']+entry['contextual_requirements'],entry['image_width'],entry['image_height'])
   if execution.get('constraint_failure') is not None:
    failure='TOKEN_CONSTRAINT_FAILURE:'+str(execution['constraint_failure'])
   result={'raw_response':raw,'raw_response_sha256':hashlib.sha256(raw.encode()).hexdigest(),'raw_response_closure':_closure(raw),'generation_metadata':metadata,'constraint_metadata':constraint_metadata,'components':components,'failure_reason':failure,'status':_status(components,entry['required_component_roles'],failure)}
   cache.put_json('stage3g_v3_2',key,{'cache_key':key,'result':result});new+=1;cache_hit=False
  row={k:v for k,v in entry.items() if k not in {'prompt','image_path_sha256'}}|result|{'provider_type':policy['provider_type'],'provider_name':policy['provider_name'],'parser_version':policy['parser_version'],'generation_contract_version':policy['generation_contract_version'],'coordinate_conversion_policy_version':policy['coordinate_conversion_policy_version'],'stage3f_manifest_sha256':pre['stage3f_manifest_sha256'],'cache_key':key,'cache_hit':cache_hit}
  row['canonical_result_sha256']=stable_hash({k:v for k,v in row.items() if k not in {'cache_key','cache_hit','raw_response'}})
  results.append(row);traces.append({'spatial_grounding_task_id':row['spatial_grounding_task_id'],'anchor_candidate_id':row['anchor_candidate_id'],'cache_key':key,'cache_hit':cache_hit,'status':row['status'],'failure_reason':row['failure_reason'],'raw_response_sha256':row['raw_response_sha256'],'raw_response_closure':row['raw_response_closure'],'generation_metadata':row['generation_metadata'],'constraint_metadata':row['constraint_metadata']})
 run_dir.mkdir(parents=True)
 bindings=[];unresolved=[]
 for row in results:
  bindings.append({'spatial_grounding_task_id':row['spatial_grounding_task_id'],'anchor_candidate_id':row['anchor_candidate_id'],'unique_visual_frame_id':row['unique_visual_frame_id'],'status':row['status'],'failure_reason':row['failure_reason'],'canonical_result_sha256':row['canonical_result_sha256']})
  if row['failure_reason'] or row['status']!='COMPONENTS_LOCALIZED_UNVERIFIED':unresolved.append({'spatial_grounding_task_id':row['spatial_grounding_task_id'],'anchor_candidate_id':row['anchor_candidate_id'],'status':row['status'],'failure_reason':row['failure_reason']})
 for name,value in {'v2_tal_spatial_anchor_groundings_v3_2.jsonl':results,'v2_tal_spatial_anchor_grounding_v3_2_unresolved.jsonl':unresolved,'v2_tal_spatial_anchor_grounding_v3_2_bindings.jsonl':bindings,'v2_tal_spatial_anchor_grounding_v3_2_trace.jsonl':traces}.items():write(run_dir/name,value)
 review=[_review(run_dir,row) for row in results];write(run_dir/'v2_tal_spatial_anchor_v3_2_review_packet_index.jsonl',review)
 hashes={path.name:sha(path) for path in run_dir.glob('v2_tal_spatial_anchor_grounding*.jsonl')}|{'v2_tal_spatial_anchor_v3_2_review_packet_index.jsonl':sha(run_dir/'v2_tal_spatial_anchor_v3_2_review_packet_index.jsonl')}
 canonical=[{k:v for k,v in row.items() if k not in {'cache_key','cache_hit','raw_response','image_path','image_path_sha256'}} for row in results]
 parse_failures=sum(r['failure_reason']=='MODEL_OUTPUT_PARSE_FAILURE' for r in results);schema_failures=sum(r['failure_reason']=='MODEL_OUTPUT_SCHEMA_VIOLATION' for r in results);strict_contract=sum(r['failure_reason'] is None for r in results)
 tokenizer_bindings=sorted({canonical_json(r['constraint_metadata']['tokenizer_binding']) for r in results})
 constraint_executions=[r['constraint_metadata']['execution'] for r in results]
 manifest={'format':FORMAT,'status':'PASS','mode':mode,'stage_status':'SPATIAL_ANCHOR_GROUNDINGS_GENERATED_UNVERIFIED','candidate_status':CANDIDATE,'stage3f_manifest_sha256':pre['stage3f_manifest_sha256'],'upstream_sha256':pre['upstream_sha256'],'preflight_sha256':sha(output_dir/'stage3g_v3_2_preflight.json'),'provider_type':policy['provider_type'],'provider_name':policy['provider_name'],'model_fingerprint':pre['model_fingerprint'],'policy_sha256':pre['policy_sha256'],'config_sha256':pre['config_sha256'],'generation_contract':pre['generation_contract'],'grammar_version':policy['grammar_version'],'grammar_implementation_version':policy['grammar_implementation_version'],'grammar_spec_sha256':stable_hash([x['grammar_spec_sha256'] for x in entries]),'tokenizer_binding_sha256':stable_hash(tokenizer_bindings),'tokenizer_binding_count':len(tokenizer_bindings),'constraint_execution':{'constraint_failure_count':sum(x.get('constraint_failure') is not None for x in constraint_executions),'callback_count':sum(x.get('callback_count',0) for x in constraint_executions),'candidate_evaluations':sum(x.get('candidate_evaluations',0) for x in constraint_executions),'initialization_overhead_seconds':sum(float(x.get('initialization_overhead_seconds') or 0) for x in constraint_executions)},'selection_mode':pre['selection_mode'],'smoke_selection_manifest_sha256':pre['smoke_selection_manifest_sha256'],'planned_model_calls':expected,'new_model_calls':new,'cache_hits':hits,'logical_model_calls':expected,'canonical_grounding_result_sha256':stable_hash(canonical),'artifact_sha256':hashes,'grounding_result_count':len(results),'unresolved_count':len(unresolved),'review_item_count':len(review),'parse_failure_count':parse_failures,'schema_violation_count':schema_failures,'constraint_failure_count':sum(str(r['failure_reason']).startswith('TOKEN_CONSTRAINT_FAILURE:') for r in results),'parse_or_schema_compliant_count':expected-parse_failures-schema_failures,'parse_or_schema_compliance_rate':(expected-parse_failures-schema_failures)/expected if expected else 0.0,'strict_required_role_contract_compliant_count':strict_contract,'strict_required_role_contract_compliance_rate':strict_contract/expected if expected else 0.0,'visible_required_role_count':sum(1 for r in results for c in r['components'] if c['role'] in r['required_component_roles'] and c['visibility']=='VISIBLE'),'required_role_localized_count':sum(1 for r in results for c in r['components'] if c['role'] in r['required_component_roles'] and c['visibility']=='VISIBLE'),'not_visible_role_count':sum(1 for r in results for c in r['components'] if c['visibility']=='NOT_VISIBLE'),'ambiguous_role_count':sum(1 for r in results for c in r['components'] if c['visibility']=='AMBIGUOUS'),'legal_bbox_count':sum(1 for r in results for c in r['components'] if c['bbox_2d_raw'] is not None),'actual_unlocalized_role_count':sum(1 for r in results for c in r['components'] if c['visibility']!='VISIBLE'),'frames_read':expected,'unique_frame_bytes_opened':len({r['frame_sha256'] for r in results}),'videos_read':0,'model_calls_made':new,'backend_loaded':True,'cache_opened':True,'gt_used':False,'assistant_or_gt_values_accessed':False,'boxes_created':sum(1 for r in results for c in r['components'] if c['bbox_2d_raw'] is not None),'points_created':0,'masks_created':0,'masklets_created':0,'support_tubes_created':0,'certificate_created':False,'new_verified_count':0,'certificate_status':'NOT_APPLICABLE','ready_for_spatial_grounding_audit':bool(results),'ready_for_dense_box_tube_generation':False,'ready_for_sam2_propagation':False,'ready_for_evidence_verification':False}
 manifest['manifest_content_sha256']=stable_hash(manifest);write(run_dir/'v2_tal_stage3g_v3_2_manifest.json',manifest)
 audit={k:manifest[k] for k in ('format','status','mode','stage3f_manifest_sha256','upstream_sha256','generation_contract','grammar_version','grammar_implementation_version','grammar_spec_sha256','tokenizer_binding_sha256','tokenizer_binding_count','constraint_execution','planned_model_calls','new_model_calls','cache_hits','logical_model_calls','frames_read','unique_frame_bytes_opened','videos_read','model_calls_made','backend_loaded','cache_opened','gt_used','assistant_or_gt_values_accessed','boxes_created','points_created','masks_created','masklets_created','support_tubes_created','certificate_created','new_verified_count','certificate_status')}|{'artifact_sha256':{**hashes,'v2_tal_stage3g_v3_2_manifest.json':sha(run_dir/'v2_tal_stage3g_v3_2_manifest.json')}}
 audit['audit_content_sha256']=stable_hash(audit);write(run_dir/'v2_tal_stage3g_v3_2_audit.json',audit)
 summary={k:manifest[k] for k in ('format','status','mode','stage_status','planned_model_calls','new_model_calls','cache_hits','logical_model_calls','canonical_grounding_result_sha256','grounding_result_count','unresolved_count','review_item_count','model_calls_made','new_verified_count','certificate_status')};write(output_dir/f'v2_tal_stage3g_v3_2_{mode}_summary.json',summary)
 return summary

def validate(*,output_dir:Path,mode:str='run')->dict[str,Any]:
 pre=_preflight(output_dir);run=output_dir/mode;m=obj(run/'v2_tal_stage3g_v3_2_manifest.json','MANIFEST');audit=obj(run/'v2_tal_stage3g_v3_2_audit.json','AUDIT')
 if m.get('manifest_content_sha256')!=stable_hash({k:v for k,v in m.items() if k!='manifest_content_sha256'}):raise SpatialAnchorGroundingError('MANIFEST_TAMPERED')
 if audit.get('audit_content_sha256')!=stable_hash({k:v for k,v in audit.items() if k!='audit_content_sha256'}):raise SpatialAnchorGroundingError('AUDIT_TAMPERED')
 for name,digest in m['artifact_sha256'].items():
  if sha(run/name)!=digest:raise SpatialAnchorGroundingError('ARTIFACT_TAMPERED')
 result=rows(run/'v2_tal_spatial_anchor_groundings_v3_2.jsonl','GROUNDINGS')
 if len(result)!=pre['planned_model_calls'] or len({(x['spatial_grounding_task_id'],x['anchor_candidate_id']) for x in result})!=len(result):raise SpatialAnchorGroundingError('GROUNDING_ENUMERATION_INVALID')
 for row in result:
  _reverify({**row,'image_path_sha256':hashlib.sha256(row['image_path'].encode()).hexdigest()}) if 'image_path' in row else None
  parsed,failure=_parse(row['raw_response'],row['required_component_roles']+row['contextual_requirements'],row['image_width'],row['image_height'])
  projection={k:v for k,v in row.items() if k not in {'cache_key','cache_hit','raw_response','canonical_result_sha256'}}
  if (parsed!=row['components'] or failure!=row['failure_reason'] or row['status']!=_status(parsed,row['required_component_roles'],failure)
      or row.get('raw_response_sha256')!=hashlib.sha256(row['raw_response'].encode()).hexdigest()
      or row.get('raw_response_closure')!=_closure(row['raw_response'])
      or not isinstance(row.get('generation_metadata'),dict)
      or not isinstance(row.get('constraint_metadata'),dict)
      or not isinstance(row['constraint_metadata'].get('execution'),dict)
      or row['constraint_metadata']['execution'].get('grammar_spec_sha256')!=row.get('grammar_spec_sha256')
      or row['generation_metadata'].get('max_new_tokens')!=pre['generation_parameters']['max_new_tokens']
      or row.get('canonical_result_sha256')!=stable_hash(projection)):
   raise SpatialAnchorGroundingError('GROUNDING_RESULT_TAMPERED')
 canonical=[{k:v for k,v in row.items() if k not in {'cache_key','cache_hit','raw_response','image_path','image_path_sha256'}} for row in result]
 if m.get('canonical_grounding_result_sha256')!=stable_hash(canonical):raise SpatialAnchorGroundingError('CANONICAL_RESULT_TAMPERED')
 review=rows(run/'v2_tal_spatial_anchor_v3_2_review_packet_index.jsonl','REVIEW_PACKET_INDEX')
 if len(review)!=len(result):raise SpatialAnchorGroundingError('REVIEW_PACKET_INDEX_INVALID')
 for item in review:
  directory=run/'review_packet'/item['anchor_candidate_id']
  if (sha(directory/'frame.png')!=item.get('frame_copy_sha256') or sha(directory/'overlay.png')!=item.get('overlay_sha256')
      or sha(directory/'metadata.json')!=item.get('metadata_sha256')):raise SpatialAnchorGroundingError('REVIEW_PACKET_TAMPERED')
 if any(m.get(k)!=v for k,v in {'certificate_created':False,'new_verified_count':0,'certificate_status':'NOT_APPLICABLE','points_created':0,'masks_created':0,'masklets_created':0,'support_tubes_created':0,'gt_used':False}.items()):raise SpatialAnchorGroundingError('BOUNDARY_AUDIT_INVALID')
 return {'status':'PASS','mode':mode,'stage_status':m['stage_status'],'planned_model_calls':m['planned_model_calls'],'new_model_calls':m['new_model_calls'],'cache_hits':m['cache_hits'],'canonical_grounding_result_sha256':m['canonical_grounding_result_sha256'],'parse_failure_count':m['parse_failure_count'],'schema_violation_count':m['schema_violation_count'],'constraint_failure_count':m['constraint_failure_count'],'visible_required_role_count':m['visible_required_role_count'],'required_role_localized_count':m['required_role_localized_count'],'not_visible_role_count':m['not_visible_role_count'],'ambiguous_role_count':m['ambiguous_role_count'],'legal_bbox_count':m['legal_bbox_count'],'actual_unlocalized_role_count':m['actual_unlocalized_role_count'],'ready_for_spatial_grounding_audit':True,'new_verified_count':0,'certificate_status':'NOT_APPLICABLE'}
