"""Frozen public V3 references; historical claims quarantined for comparison only."""
from __future__ import annotations

from pathlib import Path

from relive.claim_revalidation.offline import digest,read_json,write,source_rows,manifest_check
from relive.claim_revalidation.normalization import public_time_queries
from relive.storage.artifacts import stable_hash
from relive.v2.task_selection import strict_json_loads
from .contract import SMOKE22,VERSION

PUBLIC_KEYS={'sample_key','source_record_index','qa_type','question','requirements','question_sha256','requirements_sha256',
             'question_metadata','source_binding_status','source_binding_issues'}

def rows(path):return [strict_json_loads(s,error_code='V3_JSONL_INVALID') for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]

def new_dir(path):
    p=Path(path).resolve()
    if p.exists():raise ValueError('NEW_OUTPUT_DIRECTORY_REQUIRED')
    p.mkdir(parents=True);return p

def seal(out,names,format_name,**extra):
    m={'format':format_name,**extra,'artifact_sha256':{n:digest(out/n) for n in sorted(names)}}
    m['manifest_content_sha256']=stable_hash(m);write(out/'manifest.json',m);return m

def prepare(*,v2_input,v21_dir,output_dir,stage1_dir=None,legacy_diagnostic=None):
    data,_,_,cache_watch=source_rows(v2_input)
    m21=manifest_check(v21_dir)
    if m21['identity']['input_sha256']!=digest(v2_input):raise ValueError('V21_SOURCE_BINDING_MISMATCH')
    r21=rows(Path(v21_dir)/'stage2_v2_1_revalidated.jsonl')
    if [r['source_record_index'] for r in r21]!=SMOKE22:raise ValueError('V21_SELECTION_INVALID')
    watch={str(Path(v2_input).resolve()):digest(v2_input),str((Path(v21_dir)/'manifest.json').resolve()):digest(Path(v21_dir)/'manifest.json'),**cache_watch}
    s1={};source_kind='FROZEN_V2_V21_COPIES_STAGE1_BANK_UNAVAILABLE'
    if stage1_dir:
        from relive.claim_bank.pipeline import validate_bank
        validate_bank(stage1_dir)
        s1m=read_json(Path(stage1_dir)/'bank_manifest.json')
        if s1m['stage']!=1 or s1m['dry_run']:raise ValueError('INVALID_STAGE1_BANK')
        s1={r['sample_key']:r for r in rows(Path(stage1_dir)/'claim_bank.jsonl')};source_kind='VALIDATED_STAGE1_V2_V21'
        watch[str((Path(stage1_dir)/'bank_manifest.json').resolve())]=digest(Path(stage1_dir)/'bank_manifest.json')
    old={}
    if legacy_diagnostic:
        d=read_json(legacy_diagnostic);old={r['sample_key']:r for r in d['stage2']['samples']}
        if len(old)!=22:raise ValueError('V1_SELECTION_INVALID')
        watch[str(Path(legacy_diagnostic).resolve())]=digest(legacy_diagnostic)
    public=[];quarantine=[]
    for a,b in zip(data,r21):
        issues=[]
        for label,c in [('V2.1',b),*([('Stage1',s1.get(a['sample_key']))] if stage1_dir else []),*([('V1',old.get(a['sample_key']))] if legacy_diagnostic else [])]:
            if c is None or any(c.get(k)!=a[k] for k in ('sample_key','question','qa_type','requirements')):issues.append(label+'_FROZEN_INPUT_MISMATCH')
        public.append({**{k:a[k] for k in ('sample_key','source_record_index','qa_type','question','requirements','question_sha256','requirements_sha256')},
                       'question_metadata':{'explicit_time_queries':public_time_queries(a['question']),'source':'PUBLIC_QUESTION_ONLY','frame_time_mapping':'NOT_BOUND'},
                       'source_binding_status':'FAIL' if issues else 'PASS','source_binding_issues':issues})
        v1=old.get(a['sample_key'])
        quarantine.append({'sample_key':a['sample_key'],'v1':None if v1 is None else {k:v1[k] for k in ('claims','generation_status','failure_reasons','raw_responses')},
                           'v2':{'raw_response':a['raw_response'],'generation_status':a['generation_status'],'rejected_candidates':a['rejected_candidates']},
                           'v21':{'candidate_records':b['candidate_records'],'generation_status':b['generation_status']}})
    out=new_dir(output_dir)
    write(out/'public_references.jsonl',public,jsonl=True);write(out/'comparison_quarantine.jsonl',quarantine,jsonl=True)
    audit={'sample_count':22,'requirement_count':100,'source_kind':source_kind,'stage1_bank_independently_checked':bool(stage1_dir),
           'mismatched_samples':[r['sample_key'] for r in public if r['source_binding_status']=='FAIL'],
           'new_model_calls':0,'image_reads':0,'cache_writes':0,'gt_used':False,'certificate_writes':0,'new_verified_count':0}
    write(out/'reference_audit.json',audit)
    if any(digest(p)!=h for p,h in watch.items()):raise ValueError('SOURCE_CHANGED')
    seal(out,['public_references.jsonl','comparison_quarantine.jsonl','reference_audit.json'],'relive-v3-references-v1',
         source_sha256=watch,synthetic_source=data[0]['synthetic_test'],stage1_independently_checked=bool(stage1_dir))
    load_references(out)
    return audit

def load_references(root):
    m=manifest_check(root)
    if m['format']!='relive-v3-references-v1':raise ValueError('INVALID_V3_REFERENCES')
    rr=rows(Path(root)/'public_references.jsonl')
    if [r['source_record_index'] for r in rr]!=SMOKE22 or sum(len(r['requirements']) for r in rr)!=100:raise ValueError('FROZEN_SELECTION_CHANGED')
    for r in rr:
        if set(r)!=PUBLIC_KEYS or r['question_sha256']!=stable_hash(r['question']) or r['requirements_sha256']!=stable_hash(r['requirements']):raise ValueError('PUBLIC_REFERENCE_HASH_OR_ALLOWLIST_INVALID')
        for q in r['requirements']:
            if set(q)!={'requirement_id','description','type','evidence_type','importance'}:raise ValueError('REQUIREMENT_ALLOWLIST_INVALID')
        if r['question_metadata']['explicit_time_queries']!=public_time_queries(r['question']):raise ValueError('QUESTION_METADATA_BINDING_INVALID')
    return rr,m

def load_run(root,refs):
    from .validation import validate
    from .contract import protocol
    m=manifest_check(root);public,_=load_references(refs)
    if m['format']!='relive-v3-run-v1' or m['identity']['references_sha256']!=digest(Path(refs)/'manifest.json') or m['identity']['protocol_sha256']!=protocol()['protocol_sha256']:raise ValueError('V3_RUN_BINDING_INVALID')
    result=rows(Path(root)/'stage2_v3_smoke22.jsonl')
    if len(result)!=22:raise ValueError('V3_RESULT_COVERAGE_INVALID')
    for r,p in zip(result,public):
        if any(r[k]!=p[k] for k in PUBLIC_KEYS):raise ValueError('V3_FROZEN_INPUT_CHANGED')
        cp=Path(root)/'samples'/(p['sample_key']+'.json')
        checkpoint=read_json(cp)
        if digest(cp)!=m['checkpoint_sha256'].get(p['sample_key']) or checkpoint!={'result':r,'result_sha256':stable_hash(r)}:
            raise ValueError('V3_CHECKPOINT_BINDING_INVALID')
        if not m['identity']['dry_run'] and p['source_binding_status']=='PASS' and (r['call'] or {}).get('execution_status')=='OK':
            check=validate(r['raw_response'],p)
            if any(r[k]!=v for k,v in check.items()):raise ValueError('V3_GRAPH_VALIDATION_MISMATCH')
        elif r['claims'] or r['claim_audit'] or r['raw_response'] is not None:
            raise ValueError('V3_UNEXECUTED_GRAPH_FORBIDDEN')
        for c in r['claims']:
            if c['status']!='UNVERIFIED' or c['evidence_status']!='NOT_SEARCHED':raise ValueError('V3_TRUTH_STATE_INVALID')
    return result,m
