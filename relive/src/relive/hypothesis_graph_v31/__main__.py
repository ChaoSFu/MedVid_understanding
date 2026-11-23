"""python -m relive.hypothesis_graph_v31 --help"""
import argparse
import json


def main(argv=None):
    p=argparse.ArgumentParser(description='Stage 2 V3.1 instance-only minimal graphs; no evidence search or certificates.')
    p.add_argument('--mode',required=True,choices=['diagnostic','prepare','run','replay','validate','compare','review-template','compile','mock-smoke','diagnose-v3'])
    for name in ('v2-input','v21-dir','stage1-dir','legacy-diagnostic','references-dir','config','cache-dir','output-dir','run-dir','review-jsonl','diagnosis-json'):
        p.add_argument('--'+name)
    for name in ('resume','dry-run','publish'):
        p.add_argument('--'+name,action='store_true')
    a=p.parse_args(argv)
    def need(*names):
        for n in names:
            if not getattr(a,n):p.error('--'+n.replace('_','-')+' is required for '+a.mode)
    try:
        if a.mode=='diagnose-v3':
            need('diagnosis_json','output_dir');from .report import diagnose_v3
            result=diagnose_v3(diagnosis_json=a.diagnosis_json,output_dir=a.output_dir)
        elif a.mode=='diagnostic':
            need('v2_input','output_dir');from relive.hypothesis_graph.diagnostic import run_diagnostic
            result=run_diagnostic(input_path=a.v2_input,output_dir=a.output_dir)
        elif a.mode=='prepare':
            need('v2_input','v21_dir','output_dir');from .sources import prepare
            result=prepare(v2_input=a.v2_input,v21_dir=a.v21_dir,stage1_dir=a.stage1_dir,
                           legacy_diagnostic=a.legacy_diagnostic,output_dir=a.output_dir)
        elif a.mode in ('run','replay','mock-smoke'):
            need('references_dir','config','cache_dir','output_dir');from .runner import run
            if a.mode=='replay':need('run_dir')
            backend=None
            if a.mode=='mock-smoke':
                from relive.hypothesis_graph.fixtures import FixtureBackend
                from relive.config import load_config
                cfg=load_config(a.config)
                if cfg['backend']['kind']!='mock':p.error('mock-smoke requires a mock configuration')
                backend=FixtureBackend(cfg['backend'])
            result=run(references_dir=a.references_dir,config_path=a.config,cache_dir=a.cache_dir,
                       output_dir=a.output_dir,resume=a.resume,dry_run=a.dry_run,
                       replay_from=a.run_dir if a.mode=='replay' else None,_backend=backend)
        elif a.mode=='validate':
            need('run_dir','references_dir');from .sources import load_run
            samples,m=load_run(a.run_dir,a.references_dir)
            result={'status':'PASS','samples':len(samples),'requirements':sum(len(r['requirements']) for r in samples),
                    'synthetic_test':m['synthetic_test'],'new_model_calls':0,'image_reads':0,'certificate_writes':0,'new_verified_count':0}
        elif a.mode=='compare':
            need('run_dir','references_dir','output_dir');from .report import compare
            result=compare(run_dir=a.run_dir,references_dir=a.references_dir,output_dir=a.output_dir,baseline_diagnosis=a.diagnosis_json)
        elif a.mode=='review-template':
            need('run_dir','references_dir','output_dir');from .compiler import make_review_template
            result=make_review_template(run_dir=a.run_dir,references_dir=a.references_dir,output_dir=a.output_dir)
        else:
            need('run_dir','references_dir','output_dir');from .compiler import compile_bank
            if a.dry_run==a.publish:p.error('compile requires exactly one of --dry-run / --publish')
            result=compile_bank(run_dir=a.run_dir,references_dir=a.references_dir,output_dir=a.output_dir,
                                review_jsonl=a.review_jsonl,publish=a.publish)
        print(json.dumps(result,ensure_ascii=False,sort_keys=True))
        return 2 if result.get('status')=='FORMAL_PUBLICATION_BLOCKED' else 0
    except (ValueError,OSError,KeyError,TypeError) as exc:
        print(json.dumps({'status':'FAIL','error':str(exc),'new_verified_count':0},ensure_ascii=False));return 2


if __name__=='__main__':raise SystemExit(main())
