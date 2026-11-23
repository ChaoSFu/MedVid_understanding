"""V3 inference on the shared frozen adapter/cache. No images or semantic retries."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import json

from relive.backends import make_backend
from relive.backends.base import Backend
from relive.claim_bank.pipeline import ReplayBackend
from relive.config import load_config
from relive.storage.artifacts import ArtifactStore,stable_hash
from relive.storage.cache import Budget,CachedInference
from relive.claim_revalidation.offline import digest,read_json,write
from .contract import VERSION,protocol,prompt_for,PROMPT_PATH
from .validation import validate
from .sources import new_dir,seal,load_references,load_run

class GenerationEnvelope(Backend):
    """Content-cache the exact raw completion together with stopping metadata.

    This wrapper changes the transport envelope only, never the prompt or raw
    text. LocalHF's existing metadata method is one generate() invocation.
    """
    def __init__(self,backend):
        super().__init__(backend.config);self.inner=backend;self.synthetic=backend.synthetic
    def fingerprint(self):return {'adapter_version':'v3-generation-envelope-v1','synthetic':self.synthetic,'inner':self.inner.fingerprint()}
    def infer(self,request):
        if request.get('image_paths') or request.get('frame_ids'):raise ValueError('V3_IMAGE_INPUT_FORBIDDEN')
        self.calls+=1
        if callable(getattr(self.inner,'infer_with_generation_metadata',None)):
            v=self.inner.infer_with_generation_metadata(request)
            if not isinstance(v,dict) or not isinstance(v.get('raw_response'),str) or not isinstance(v.get('generation_metadata'),dict):raise ValueError('INVALID_GENERATION_METADATA_ENVELOPE')
            raw,meta=v['raw_response'],v['generation_metadata']
        else:
            raw=self.inner.infer(request);meta={'status':'SYNTHETIC_NOT_MEASURED' if self.synthetic else 'UNAVAILABLE_FROM_ADAPTER','generated_token_count':None}
        return json.dumps({'envelope_version':'v3-generation-envelope-v1','raw_response':raw,'generation_metadata':meta},ensure_ascii=False,sort_keys=True)

def implementation_hash():
    here=Path(__file__).parent;core=here.parent
    files=[here/n for n in ('contract.py','validation.py','runner.py','sources.py')]+[PROMPT_PATH,core/'storage/cache.py',core/'backends/local_hf.py']
    return stable_hash({str(p.relative_to(core)):digest(p) for p in files})

def metrics(results,usage,synthetic,dry_run):
    audits=[a for r in results for a in r['claim_audit']]
    counts=Counter(a['state'] for a in audits)
    valid=[c for r in results for c in r['claims']]
    required={(r['sample_key'],q['requirement_id']) for r in results for q in r['requirements'] if q['importance']=='REQUIRED'}
    covered={(r['sample_key'],rid) for r in results for c in r['claims'] for rid in c['requirement_ids']}
    meta=[(r.get('generation_metadata') or {}) for r in results if r.get('call')]
    tokens=[v.get('generated_token_count') for v in meta]
    return {'format':VERSION,'status':'DRY_RUN' if dry_run else 'SYNTHETIC_TEST_ONLY' if synthetic else 'COMPLETE',
        'sample_count':len(results),'requirement_count':sum(len(r['requirements']) for r in results),'retained_hypothesis_count':len(valid),
        'structural_shape_pass_count':sum(a['structural_status']=='PASS' for a in audits),'claim_state_counts':dict(sorted(counts.items())),
        'generation_coverage':{'numerator':sum(bool(r['claims']) for r in results),'denominator':22,'meaning':'contains retained unverified hypotheses, including review-required templates'},
        'plan_status_counts':dict(sorted(Counter(r['plan_status'] for r in results).items())),
        'required_requirement_coverage':{'numerator':len(required&covered),'denominator':len(required),'semantic_coverage':'REVIEW_REQUIRED'},
        'atomicity':{'detected_hard_chains':sum('EXPLICIT_MULTI_PREDICATE_CHAIN' in a['errors'] for a in audits),'true_violation_rate':None,'status':'REVIEW_REQUIRED'},
        'redundancy':{'exact_duplicate_rejections':sum('DUPLICATE_PROPOSITION' in a['errors'] for a in audits),'paraphrase_redundancy':'REVIEW_REQUIRED'},
        'question_relevance':{'value':None,'status':'REVIEW_REQUIRED'},'unsupported_specificity':{'value':None,'status':'REVIEW_REQUIRED'},
        'dependency_validity':{'structural_errors':sum(bool(a['dependency_errors']) for a in audits),'logical_validity':'REVIEW_REQUIRED'},
        'unresolved_target_cases':sum(r['plan_status']=='UNRESOLVED_TARGET_REFERENCE' for r in results),
        'spatial_eligible_claims':None,'spatial_eligibility_status':'NOT_COMPILED: requires deterministic compiler and review gate; never inferred from a type label',
        'token_usage':{'generated_tokens':sum(tokens) if tokens and all(type(t) is int for t in tokens) else None,
                       'metadata_available_calls':sum(type(t) is int for t in tokens),'status':'MEASURED' if tokens and all(type(t) is int for t in tokens) else 'UNAVAILABLE_OR_SYNTHETIC'},
        'usage':usage,'new_model_calls':usage['new_calls'],'latency_seconds':usage['latency_seconds'],
        'synthetic_test':synthetic,'gt_used':False,'image_reads':0,'certificate_writes':0,'new_verified_count':0,
        'scientific_status':'UNVERIFIED_HYPOTHESIS_PLANNING_ONLY'}

def run(*,references_dir,config_path,cache_dir,output_dir,resume=False,replay_from=None,dry_run=False,_backend=None):
    public,refm=load_references(references_dir);cfg=load_config(config_path)
    if cfg['backend']['kind']!='mock' and refm['synthetic_source']:raise ValueError('SYNTHETIC_SOURCE_FOR_REAL_MODEL_FORBIDDEN')
    if cfg['backend']['kind']!='mock' and not dry_run and not refm.get('stage1_independently_checked',False):raise ValueError('INDEPENDENT_STAGE1_BINDING_REQUIRED')
    if cfg['backend'].get('max_retries',0)!=0 or (cfg['backend']['kind']!='mock' and cfg['backend'].get('generation',{}).get('do_sample') is not False):raise ValueError('FROZEN_GREEDY_ZERO_RETRY_REQUIRED')
    if _backend is not None and not _backend.synthetic:raise ValueError('INJECTED_BACKEND_MUST_BE_SYNTHETIC')
    identity={'references_sha256':digest(Path(references_dir)/'manifest.json'),'protocol_sha256':protocol()['protocol_sha256'],
              'implementation_sha256':implementation_hash(),'config_sha256':digest(config_path),'dry_run':dry_run}
    if replay_from and (resume or dry_run or _backend):raise ValueError('REPLAY_ARGUMENTS_INVALID')
    prior=None
    if replay_from:
        _,pm=load_run(replay_from,references_dir);prior=read_json(Path(replay_from)/'run_binding.json')
        if pm['identity']!=identity:raise ValueError('REPLAY_IDENTITY_MISMATCH')
    out=Path(output_dir).resolve()
    if out.exists():
        if not resume or not (out/'run_binding.json').is_file():raise ValueError('NEW_OUTPUT_DIRECTORY_OR_VALID_RESUME_REQUIRED')
        if read_json(out/'run_binding.json')['identity']!=identity:raise ValueError('RESUME_IDENTITY_MISMATCH')
    else:new_dir(out)
    with ArtifactStore(out).run_lock():
        if (out/'manifest.json').exists():load_run(out,references_dir);return read_json(out/'stage2_v3_metrics.json')
        backend=None if dry_run else ReplayBackend(prior['backend_fingerprint'],cfg['backend']) if prior else GenerationEnvelope(_backend or make_backend(cfg['backend']))
        binding={'identity':identity,'backend_fingerprint':backend.fingerprint() if backend else None,
                 'synthetic_test':bool(backend and backend.synthetic),'image_reads':0}
        if (out/'run_binding.json').exists():
            if read_json(out/'run_binding.json')!=binding:raise ValueError('RESUME_BACKEND_MISMATCH')
        else:write(out/'run_binding.json',binding)
        budget=Budget(22);inference=CachedInference(backend,ArtifactStore(Path(cache_dir)/VERSION)) if backend else None
        results=[];resumed=0
        for row in public:
            path=out/'samples'/(row['sample_key']+'.json')
            if path.exists():
                cp=read_json(path)
                if stable_hash(cp['result'])!=cp['result_sha256'] or any(cp['result'][k]!=row[k] for k in row):raise ValueError('CHECKPOINT_INVALID')
                results.append(cp['result']);resumed+=1;continue
            raw=None;call=None;metadata=None
            check={'plan_status':'NOT_EXECUTED' if dry_run else 'FROZEN_INPUT_MISMATCH','claims':[],'claim_audit':[],'root_errors':[],
                   'unresolved_references':[],'requirement_issues':[],'uncovered_requirements':[],'semantic_quality':'REVIEW_REQUIRED'}
            prompt=prompt_for(row)
            if inference and row['source_binding_status']=='PASS':
                request={'stage':'hypothesis_graph_v3','prompt_version':VERSION,'prompt':prompt,'image_paths':[],'frame_ids':[],
                         'context':{'qa_type':row['qa_type']},'hypothesis_binding':{'sample_key':row['sample_key'],'question_sha256':row['question_sha256'],
                         'requirements_sha256':row['requirements_sha256'],'frozen_identity':identity}}
                if prior and not inference.has_terminal_result(request):raise ValueError('REPLAY_CACHE_MISS')
                c=inference.call(request,budget);call={k:c[k] for k in ('execution_status','failure_reason','cache_key','raw_response_ref')}
                if c['execution_status']=='OK':
                    envelope=json.loads(c['raw_text']);raw=envelope['raw_response'];metadata=envelope['generation_metadata'];check=validate(raw,row)
                else:check.update(plan_status='MODEL_OR_SCHEMA_FAILURE',root_errors=['MODEL_EXECUTION_FAILURE'])
            result={**row,**check,'format':VERSION,'raw_response':raw,'call':call,'generation_metadata':metadata,
                    'prompt_sha256':stable_hash(prompt),'synthetic_test':binding['synthetic_test'],'input_modality':'TEXT_ONLY','image_reads':0}
            if prior and result!=read_json(Path(replay_from)/'samples'/(row['sample_key']+'.json'))['result']:raise ValueError('REPLAY_RESULT_MISMATCH')
            path.parent.mkdir(exist_ok=True);write(path,{'result':result,'result_sha256':stable_hash(result)});results.append(result)
        write(out/'stage2_v3_smoke22.jsonl',results,jsonl=True)
        write(out/'protocol.json',protocol())
        summary=metrics(results,budget.snapshot(),binding['synthetic_test'],dry_run);write(out/'stage2_v3_metrics.json',summary)
        write(out/'stage2_v3_replay_audit.json',{'status':'PASS','new_model_calls':budget.new_calls,'cache_hits':budget.cache_hits,
              'exact_replay':'PASS' if prior else 'NOT_RUN','resumed_checkpoint_count':resumed,'usage_scope':'this invocation only'})
        write(out/'stage2_v3_gt_isolation_audit.json',{'status':'PASS','generator_fields':['question','qa_type','requirements'],
              'old_claims_in_prompt':False,'gt_used':False,'image_reads':0,'certificate_writes':0,'new_verified_count':0})
        names=['stage2_v3_smoke22.jsonl','protocol.json','run_binding.json','stage2_v3_metrics.json','stage2_v3_replay_audit.json','stage2_v3_gt_isolation_audit.json']
        seal(out,names,'relive-v3-run-v1',identity=identity,synthetic_test=binding['synthetic_test'],
             checkpoint_sha256={r['sample_key']:digest(out/'samples'/(r['sample_key']+'.json')) for r in results})
        load_run(out,references_dir)
        return summary
