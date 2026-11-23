"""V2.2 lossy diagnostic projections only. Never a V3 generation input or bank."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import html
import json

from relive.claim_planner import contract as v2
from relive.claim_revalidation.offline import digest, source_rows, write
from relive.storage.artifacts import stable_hash
from .validation import text_audit

def project_a(candidate):
    view=deepcopy(candidate);removed=[]
    def walk(value,path):
        if isinstance(value,dict):
            for key in list(value):
                if key in {'query_time','bindings','frame','timestamp','bbox','stride_seconds','start_seconds','end_seconds','fps','candidate_frames','spatial_regions'}:
                    removed.append({'field':path+'.'+key,'value':value.pop(key)})
                else:walk(value[key],path+'.'+key)
        elif isinstance(value,list):
            for i,x in enumerate(value):walk(x,f'{path}[{i}]')
    walk(view,'$')
    schema=deepcopy(v2.CLAIM_SCHEMA)
    schema['properties'].pop('bindings');schema['required'].remove('bindings')
    s=schema['properties']['proposition']['properties']['scope'];s['properties'].pop('query_time');s['required'].remove('query_time')
    return view,removed,v2.shape_errors(view,schema)

def project_b(candidate):
    fields=('text','type','requirement_ids','dependencies','status')
    view={k:deepcopy(candidate[k]) for k in fields if k in candidate}
    schema=v2.obj({k:deepcopy(v2.CLAIM_SCHEMA['properties'][k]) for k in fields})
    return view,v2.shape_errors(view,schema)

def run_diagnostic(*,input_path,output_dir):
    from pathlib import Path
    from .sources import new_dir,seal
    before=digest(input_path);rows,raw,_,watched=source_rows(input_path)
    records=[]
    for row,completion in zip(rows,raw):
        ids={c['claim_id'] for c in completion['claims']}
        graph={c['claim_id']:[d['claim_id'] for d in c['dependencies'] if d['relation'] in ('REQUIRES','SUPPORTS')] for c in completion['claims']}
        def cycle(node,target,seen):return node==target or (node not in seen and any(cycle(n,target,seen|{node}) for n in graph.get(node,[]) if n in graph))
        seen=set()
        for i,c in enumerate(completion['claims']):
            a,removed,ae=project_a(c);b,be=project_b(c)
            hard,risk=text_audit(c['text'],row,strict_numbers=False)
            if any(d['claim_id'] not in ids for d in c['dependencies']):hard.append('DANGLING_DEPENDENCY')
            if any(cycle(n,c['claim_id'],set()) for n in graph[c['claim_id']]):hard.append('REQUIRES_OR_SUPPORTS_CYCLE')
            norm=' '.join(c['text'].casefold().split())
            if norm in seen:hard.append('EXACT_REDUNDANCY')
            seen.add(norm)
            records.append({'sample_key':row['sample_key'],'source_record_index':row['source_record_index'],'candidate_index':i,
                'claim_id':c['claim_id'],'original_text':c['text'],'text_sha256':stable_hash(c['text']),'raw_candidate':c,
                'diagnostic_a':{'view':a,'removed_fields':removed,'shape_errors':ae,'shape_pass':not ae},
                'diagnostic_b':{'view':b,'shape_errors':be,'shape_pass':not be},
                'diagnostic_c':{'detected_contract_violations':sorted(set(hard)),'detected_semantic_risks':risk,
                    'state':'REJECTED' if hard or be else 'REVIEW_REQUIRED',
                    'atomicity':'REVIEW_REQUIRED','question_relevance':'REVIEW_REQUIRED','independent_testability':'REVIEW_REQUIRED',
                    'predicate_consistency':'REVIEW_REQUIRED','unsupported_specificity':'REVIEW_REQUIRED','logical_dependency_validity':'REVIEW_REQUIRED'},
                'diagnostic_only':True,'publication_allowed':False})
    assert len(records)==84
    summary={'format':'relive-stage2-v2.2-diagnostic-v1','source_sha256':before,'sample_count':22,'raw_candidates':84,
             'original_v2_admitted':0,'diagnostic_a_shape_pass':sum(r['diagnostic_a']['shape_pass'] for r in records),
             'diagnostic_b_shape_pass':sum(r['diagnostic_b']['shape_pass'] for r in records),
             'diagnostic_c_state_counts':dict(sorted(Counter(r['diagnostic_c']['state'] for r in records).items())),
             'diagnostic_c_semantically_proven':0,'new_model_calls':0,'image_reads':0,'gt_used':False,'cache_writes':0,
             'certificate_writes':0,'new_verified_count':0,'diagnostic_only':True,'publication_allowed':False,
             'limit':'A/B are lossy structural projections, not repaired historical claims or formal V3 candidates. C cannot prove semantic quality.'}
    out=new_dir(output_dir)
    write(out/'v2_2_candidate_audit.jsonl',records,jsonl=True);write(out/'v2_2_diagnostic.json',summary)
    esc=lambda x:html.escape(str(x))
    parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>V2.2 offline diagnostic</title>',
           '<style>body{font:15px/1.6 system-ui;max-width:1100px;margin:30px auto;padding:0 16px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f4f8;padding:12px}article{border:1px solid #ccd6e4;padding:16px;margin:20px 0}</style>',
           '<h1>V2.2 · 零调用诊断（禁止作为正式 Claim Bank）</h1>',f'<pre>{esc(json.dumps(summary,ensure_ascii=False,indent=2))}</pre>']
    for r in records:
        parts += [f'<article><h2>index={r["source_record_index"]} / {esc(r["claim_id"])}</h2><p>{esc(r["original_text"])}</p>',
                  f'<details><summary>A / B 投影视图及 C 审查</summary><pre>{esc(json.dumps(r,ensure_ascii=False,indent=2))}</pre></details></article>']
    (out/'v2_2_diagnostic.html').write_text('\n'.join(parts)+'</html>',encoding='utf-8')
    (out/'v2_2_summary.md').write_text('# V2.2 diagnostic only\n\n```json\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n```\n\nNo candidate was published. Human semantic review remains required.\n',encoding='utf-8')
    if digest(input_path)!=before or any(digest(p)!=h for p,h in watched.items()):raise ValueError('HISTORICAL_ARTIFACT_CHANGED')
    seal(out,[p.name for p in out.iterdir()],'relive-v2.2-diagnostic-manifest-v1',source_sha256=before)
    return summary
