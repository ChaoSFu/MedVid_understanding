"""V3.1 inference on shared adapters/cache. Frozen V3 is never modified."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import json

from relive.backends import make_backend
from relive.claim_bank.pipeline import ReplayBackend
from relive.config import load_config
from relive.storage.artifacts import ArtifactStore,stable_hash
from relive.storage.cache import Budget,CachedInference
from relive.claim_revalidation.offline import digest,read_json,write
from .contract import VERSION,protocol,prompt_for,PROMPT_PATH
from .validation import validate
from .sources import new_dir,seal,load_references,load_run,check_result

from relive.hypothesis_graph.runner import GenerationEnvelope, metrics as legacy_metrics

def implementation_hash():
    here=Path(__file__).parent;core=here.parent
    files = [here / n for n in ('contract.py', 'validation.py', 'predicates.py', 'runner.py', 'sources.py')]
    files += [PROMPT_PATH, core / 'storage/cache.py', core / 'storage/artifacts.py',
              core / 'backends/local_hf.py', core / 'hypothesis_graph/sources.py',
              core / 'hypothesis_graph/runner.py', core / 'hypothesis_graph/validation.py',
              core / 'hypothesis_graph/contract.py', core / 'claim_planner/contract.py',
              core / 'v2/task_selection.py']
    return stable_hash({str(p.relative_to(core)):digest(p) for p in files})

def metrics(results, usage, synthetic, dry_run):
    summary = legacy_metrics(results, usage, synthetic, dry_run)
    summary['format'] = VERSION
    summary['root_error_counts'] = dict(sorted(Counter(
        error for r in results for error in r['root_errors']
    ).items()))
    summary['candidate_error_counts'] = dict(sorted(Counter(
        error for r in results for a in r['claim_audit'] for error in a['errors']
    ).items()))
    summary['semantic_accuracy'] = 'NOT_ESTABLISHED'
    return summary

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
        if (out/'manifest.json').exists():load_run(out,references_dir);return read_json(out/'stage2_v3_1_metrics.json')
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
                check_result(cp['result'],row,identity,binding['synthetic_test'])
                results.append(cp['result']);resumed+=1;continue
            raw=None;call=None;metadata=None
            check={'plan_status':'NOT_EXECUTED' if dry_run else 'FROZEN_INPUT_MISMATCH','claims':[],'claim_audit':[],'root_errors':[],'root_error_details':[],
                   'unresolved_references':[],'requirement_issues':[],'uncovered_requirements':[],'semantic_quality':'REVIEW_REQUIRED'}
            prompt=prompt_for(row)
            if inference and row['source_binding_status']=='PASS':
                request={'stage':'hypothesis_graph_v31','prompt_version':VERSION,'prompt':prompt,'image_paths':[],'frame_ids':[],
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
        write(out/'stage2_v3_1_smoke22.jsonl',results,jsonl=True)
        write(out/'protocol.json',protocol())
        summary=metrics(results,budget.snapshot(),binding['synthetic_test'],dry_run);write(out/'stage2_v3_1_metrics.json',summary)
        write(out/'stage2_v3_1_replay_audit.json',{'status':'PASS','new_model_calls':budget.new_calls,'cache_hits':budget.cache_hits,
              'exact_replay':'PASS' if prior else 'NOT_RUN','resumed_checkpoint_count':resumed,'usage_scope':'this invocation only'})
        write(out/'stage2_v3_1_gt_isolation_audit.json',{'status':'PASS','generator_fields':['question','qa_type','requirements'],
              'old_claims_in_prompt':False,'gt_used':False,'image_reads':0,'certificate_writes':0,'new_verified_count':0})
        names=['stage2_v3_1_smoke22.jsonl','protocol.json','run_binding.json','stage2_v3_1_metrics.json','stage2_v3_1_replay_audit.json','stage2_v3_1_gt_isolation_audit.json']
        seal(out,names,'relive-v3.1-run-v1',identity=identity,synthetic_test=binding['synthetic_test'],
             checkpoint_sha256={r['sample_key']:digest(out/'samples'/(r['sample_key']+'.json')) for r in results})
        load_run(out,references_dir)
        return summary
