"""python -m RAG.knowledge_builder.build --help"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
import argparse
import hashlib
import logging
import os
import sys

from .common import stable_hash,file_hash,read_json,write_json,run_lock,canonical
from .discover import discover
from .pdf_parser import parse_pdf,source_metadata,render_page,VERSION as PARSER_VERSION
from .content_selector import select,VERSION as SELECTOR_VERSION
from .source_policy import source_findings,review_template,apply_review,summarize_source_scope
from .schemas import LIBRARY,RESPONSE,TYPES
from .provider import Provider,Config,ProviderError,BudgetStop
from .extractor import extract,MockProvider,PROMPT,CHECK_PROMPT,VERSION as EXTRACTOR_VERSION
from .validator import validate_card,validate_library
from .normalizer import normalize

LOG=logging.getLogger('surgical_knowledge_builder')


def build(*,input_dir,output,cache_dir='RAG/cache',mode='inspect',resume=False,max_files=None,max_pages=None,
          include_files=(),source_review=None,provider_config=None,provider_factory=None,
          enable_vision=False,max_vision_pages=3,max_api_calls=100,max_cost_usd=None,created_at=None,tables=True,
          max_unit_chars=4000,max_unit_blocks=6):
    if mode not in ('inspect','mock','build'):raise ValueError('MODE_INVALID')
    output=Path(output).resolve();cache_dir=Path(cache_dir).resolve();out=output.parent
    if mode=='mock' and output.name=='relive_surgical_visual_knowledge_v0.1.json':raise ValueError('MOCK_CANNOT_WRITE_FORMAL_FILENAME')
    base,sources,discovery_errors=discover(input_dir)
    if out.is_relative_to(base) or cache_dir.is_relative_to(base):raise ValueError('OUTPUT_OR_CACHE_INSIDE_REFERENCES_FORBIDDEN')
    if any(v is not None and v<1 for v in (max_files,max_pages)) or max_api_calls<0 or max_vision_pages<0:raise ValueError('LIMITS_INVALID')
    if any(type(v) is not int or v<1 for v in (max_unit_chars,max_unit_blocks)):raise ValueError('SEMANTIC_UNIT_LIMIT_INVALID')
    requested=set(include_files)
    if requested-set(s['file_path'] for s in sources):raise ValueError('REQUESTED_SMOKE_FILE_MISSING')
    rr={}
    if source_review:
        for line in Path(source_review).read_text(encoding='utf-8').splitlines():
            if not line.strip():continue
            from .common import strict_loads
            r=strict_loads(line)
            if r['source_id'] in rr:raise ValueError('DUPLICATE_SOURCE_REVIEW')
            rr[r['source_id']]=r
        if set(rr)-{s['source_id'] for s in sources}:raise ValueError('SOURCE_REVIEW_UNKNOWN_SOURCE')
    cfg=provider_config or Config.environment()
    code_hash=stable_hash({p.name:file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))})
    identity={'input_files':{s['file_path']:s['sha256'] for s in sources},'mode':mode,'max_files':max_files,'max_pages':max_pages,
              'include_files':sorted(include_files),'source_review_sha256':file_hash(source_review) if source_review else None,
              'parser':PARSER_VERSION,'selector':SELECTOR_VERSION,'extractor':EXTRACTOR_VERSION,'code_sha256':code_hash,
              'prompt_sha256':stable_hash([PROMPT,CHECK_PROMPT]),'schema_sha256':stable_hash(RESPONSE),
              'provider':cfg.public() if mode=='build' else None,'enable_vision':enable_vision,'max_vision_pages':max_vision_pages,
              'tables':tables,'max_unit_chars':max_unit_chars,'max_unit_blocks':max_unit_blocks}
    binding_path=out/'build_binding.json'
    if binding_path.exists():
        binding=read_json(binding_path)
        if not resume:raise ValueError('OUTPUT_EXISTS_USE_RESUME_OR_NEW_DIRECTORY')
        if binding['identity']!=identity:raise ValueError('RESUME_IDENTITY_MISMATCH')
    else:
        if output.exists():raise ValueError('OUTPUT_EXISTS_WITHOUT_BINDING')
        binding={'identity':identity,'created_at':created_at or datetime.now(timezone.utc).isoformat()}
    with run_lock(out):
        write_json(binding_path,binding)
        if (out/'build_summary.json').exists():
            summary=read_json(out/'build_summary.json')
            for name,sha in summary['artifact_sha256'].items():
                if file_hash(out/name)!=sha:raise ValueError('COMPLETED_BUILD_ARTIFACT_CHANGED')
            return {**summary,'invocation':{'new_api_calls':0,'completed_result_reused':True}}
        provider=None;provider_error=None
        if mode=='mock':provider=provider_factory() if provider_factory else MockProvider()
        elif mode=='build':
            try:provider=provider_factory() if provider_factory else Provider(cfg,cache_dir,max_calls=max_api_calls,max_cost_usd=max_cost_usd)
            except ProviderError as exc:provider_error=str(exc)
        parsed={};units={};selection=[];seen_files={};seen_content={};duplicates=[];parse_hits=0;selected_count=0
        LOG.info('Discovered %s PDFs; subdirectories=%s',len(sources),dict(Counter(str(Path(s['file_path']).parent) for s in sources)))
        for i,s in enumerate(sources):
            path=base/s['file_path']
            chosen=(not requested or s['file_path'] in requested) and (max_files is None or selected_count<max_files)
            s={**s,'selected_for_processing':chosen}
            if s['processing_status']=='FAILED':
                sources[i]={**s,'pdf_pages':0,'page_count':0,'text_extractable':False,'title':s['file_name'],
                            'probable_source':None,'document_category':'unknown','source_audit':source_findings(s['file_name'],''),
                            'benchmark_independent':None,'chunk_count':0,'review_status':'pending','disposition':'failed'};continue
            if not chosen:
                import pymupdf
                try:
                    with pymupdf.open(path) as d:count=len(d)
                    sources[i]={**s,'pdf_pages':count,'page_count':count,'text_extractable':None,'title':s['file_name'],
                                'probable_source':None,'document_category':'not_parsed','processing_status':'DEFERRED_BY_LIMIT',
                                'source_audit':source_findings(s['file_name'],''),'benchmark_independent':None,
                                'chunk_count':0,'review_status':'pending','disposition':'indexed_only'}
                except Exception as exc:
                    sources[i]={**s,'pdf_pages':0,'page_count':0,'text_extractable':False,'title':s['file_name'],
                                'probable_source':None,'document_category':'unknown','processing_status':'FAILED',
                                'failure_reason':type(exc).__name__,'source_audit':source_findings(s['file_name'],''),
                                'benchmark_independent':None,'chunk_count':0,'review_status':'pending','disposition':'failed'}
                continue
            selected_count+=1
            try:
                p,hit=parse_pdf(path,s['sha256'],cache_dir,max_pages=max_pages,tables=tables);parse_hits+=int(hit)
                s=source_metadata(s,p);parsed[s['source_id']]=p
                alltext='\n'.join(page['raw_text'] for page in p['pages'])
                s['source_audit']=source_findings(s['file_name'],alltext)
                if s['source_id'] in rr:s=apply_review(s,rr[s['source_id']])
                duplicate=seen_files.get(s['sha256'])
                text_match=seen_content.get(s['content_hash']) if not p['unprocessed_page_numbers'] else None
                if not duplicate and text_match:
                    previous=next(old for old in sources[:i] if old['source_id']==text_match)
                    previous_parsed=parsed[text_match]
                    # Equal extracted text does not imply equal illustrations.
                    # Render only candidate duplicate pairs, not the whole library.
                    same_visual=len(previous_parsed['pages'])==len(p['pages']) and all(
                        hashlib.sha256(render_page(base/previous['file_path'],page['page'])).digest()==hashlib.sha256(render_page(path,page['page'])).digest()
                        for page in p['pages'])
                    if same_visual:duplicate=text_match
                    else:duplicates.append({'source_id':s['source_id'],'duplicate_of':text_match,
                                            'basis':'TEXT_MATCH_BUT_VISUAL_CONTENT_DIFFERS','auto_skip':False})
                if duplicate:
                    s['duplicate_of']=duplicate;s['disposition']='duplicate';duplicates.append({'source_id':s['source_id'],'duplicate_of':duplicate,'basis':'FILE_OR_FULL_TEXT_AND_RENDER_MATCH','auto_skip':True})
                else:
                    seen_files[s['sha256']]=s['source_id']
                    if not p['unprocessed_page_numbers']:seen_content.setdefault(s['content_hash'],s['source_id'])
                uu,audit=select(s,p,max_input_chars=max_unit_chars,max_blocks=max_unit_blocks);units[s['source_id']]=uu
                selection.extend({'source_id':s['source_id'],**r} for r in audit)
                s['candidate_unit_count']=len(uu)
                LOG.info('%s: %s pages, %s units, %s',s['file_path'],s['pdf_pages'],len(uu),s['source_audit']['status'])
            except Exception as exc:
                if isinstance(exc,ValueError) and str(exc).startswith('SOURCE_REVIEW'):raise
                s={**s,'pdf_pages':s.get('pdf_pages',0),'page_count':s.get('pdf_pages',0),'text_extractable':False,
                   'title':s['file_name'],'probable_source':None,'document_category':'unknown','processing_status':'FAILED',
                   'failure_reason':type(exc).__name__,'source_audit':source_findings(s['file_name'],''),
                   'benchmark_independent':None,'chunk_count':0,'review_status':'pending','disposition':'failed'}
            sources[i]=s
        records=[];excluded=[];failures=[];jobs=[];vision_selected=0;unit_results=[]
        for s in sources:
            if s['processing_status'] not in ('PARSED','PARSED_PARTIAL') or s.get('duplicate_of'):continue
            allowed=s['source_audit']['extraction_allowed'] or mode=='mock' and not s['source_audit']['risk_terms']
            if not allowed:continue
            for unit in units.get(s['source_id'],[]):jobs.append((s,unit,None))
            p=parsed[s['source_id']]
            for page in p['pages']:
                if not page['vision_candidate']:continue
                page['vision_status']='PENDING_VISION_DISABLED'
                if not enable_vision:continue
                if vision_selected>=max_vision_pages:page['vision_status']='DEFERRED_VISION_BUDGET';continue
                candidates=[u for u in units.get(s['source_id'],[]) if page['page'] in u['pages']]
                if not candidates:page['vision_status']='PENDING_LABELED_CONTEXT_REQUIRED';continue
                # Labels and captions must come from admissible clean blocks.
                page_blocks=[b for u in candidates for b in u['blocks'] if b['page']==page['page']]
                unit={'source_id':s['source_id'],'source_sha256':s['sha256'],'title':s['title'],'section':None,
                      'pages':[page['page']],'blocks':page_blocks}
                unit['unit_id']='vision_'+stable_hash(unit)[:24]
                jobs.append((s,unit,page['page']));vision_selected+=1;page['vision_status']='SELECTED_REQUIRES_LABELED_EXTRACTION'
        def process(job):
            s,u,page=job
            audit={'source_id':s['source_id'],'unit_id':u['unit_id'],'pages':u['pages'],
                   'method':'vision_llm' if page else 'text_llm','input_sha256':stable_hash(u),
                   'candidate_count':None,'accepted_count':0,'rejected_count':0,'omitted_content_reasons':[]}
            if provider is None:return [],[],{'source_id':s['source_id'],'unit_id':u['unit_id'],'reason':provider_error or 'INSPECT_ONLY_NO_SEMANTIC_EXTRACTION'}, {**audit,'status':'NOT_EXECUTED'}
            try:
                image=render_page(base/s['file_path'],page) if page else None
                accepted,rejected,result=extract(u,provider,image=image);passed=[]
                for rec in accepted:
                    errors=validate_card(rec['card'],s,u,method='vision_llm' if page else 'text_llm',synthetic=mode=='mock')
                    if errors:rejected.append({**rec,'rejection_reasons':errors})
                    else:passed.append(rec)
                audit.update(status='COMPLETED',candidate_count=len(result['value']['cards']),accepted_count=len(passed),
                             rejected_count=len(rejected),omitted_content_reasons=result['value']['omitted_content_reasons'],
                             extraction_request_key=result['request_key'])
                return passed,rejected,None,audit
            except (ProviderError,ValueError) as exc:return [],[],{'source_id':s['source_id'],'unit_id':u['unit_id'],'pages':u['pages'],'reason':str(exc),'budget_stop':isinstance(exc,BudgetStop)}, {**audit,'status':'FAILED','failure_reason':str(exc)}
            except Exception as exc:return [],[],{'source_id':s['source_id'],'unit_id':u['unit_id'],'pages':u['pages'],'reason':'UNIT_FAILURE_'+type(exc).__name__,'budget_stop':False}, {**audit,'status':'FAILED','failure_reason':'UNIT_FAILURE_'+type(exc).__name__}
        if mode!='inspect' and provider:
            # Bounded dispatch: only concurrency-sized batches are submitted.
            workers=cfg.max_concurrency if mode=='build' else 1
            stop=False
            for start in range(0,len(jobs),workers):
                if stop:
                    failures.extend({'source_id':s['source_id'],'unit_id':u['unit_id'],'reason':'DEFERRED_AFTER_BUDGET_STOP'} for s,u,_ in jobs[start:]);break
                batch=jobs[start:start+workers]
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    for accepted,rejected,failure,audit in pool.map(process,batch):
                        unit_results.append(audit)
                        records.extend(accepted);excluded.extend(rejected)
                        if failure:failures.append(failure);stop=stop or failure.get('budget_stop',False)
        chunks,dups,near,conflicts=normalize(records);excluded.extend(conflicts)
        for s in sources:
            for k,v in {'publisher':None,'source_url':None,'publication_year':None,'license':'unknown',
                        'document_type':s['document_category']}.items():s.setdefault(k,v)
            s['chunk_count']=sum(any(r['source_id']==s['source_id'] for r in c['source_refs']) for c in chunks)
            if s['chunk_count']:s['disposition']='selected_chunks'
            elif s['source_audit']['risk_terms'] and not s['source_audit']['extraction_allowed']:s['disposition']='excluded_benchmark_risk'
            elif s['source_audit']['status']=='REVIEWED_EXCLUDE':s['disposition']='excluded_by_review'
        scope_audit=summarize_source_scope(sources,rr)
        status='INSPECT_ONLY' if mode=='inspect' else 'MOCK_ONLY' if mode=='mock' else 'BLOCKED_API_CONFIGURATION' if provider_error else 'PARTIAL_BUDGET_STOP' if any(f.get('budget_stop') for f in failures) else 'SOURCE_REVIEW_REQUIRED' if scope_audit['selected_review_pending'] else 'NO_ADMISSIBLE_SOURCE_UNITS' if not jobs else 'PARTIAL_EXTRACTION_FAILURE' if failures else 'BUILT_PENDING_MEDICAL_REVIEW'
        library={'schema_version':1,'library_id':'relive.surgical_visual_knowledge.mock' if mode=='mock' else 'relive.surgical_visual_knowledge',
                 'library_version':'0.1.0','created_at':binding['created_at'],'knowledge_language':'en','source_policy':'benchmark_independent',
                 'build_status':status,'synthetic_test':mode=='mock','source_index':sources,'chunks':chunks}
        checks=validate_library(library)
        changed=[s['file_path'] for s in sources if s['sha256'] and file_hash(base/s['file_path'])!=s['sha256']]
        if changed:raise ValueError('SOURCE_FILES_CHANGED')
        parsed_count=sum(s['processing_status'] in ('PARSED','PARSED_PARTIAL') for s in sources)
        coverage={'pdf_count':len(sources),'subdirectory_counts':dict(sorted(Counter(str(Path(s['file_path']).parent) for s in sources).items())),
                  'successfully_parsed_documents':parsed_count,'failed_files':[s for s in sources if s['processing_status']=='FAILED'],
                  'deferred_files':[s['file_path'] for s in sources if s['processing_status']=='DEFERRED_BY_LIMIT'],
                  'total_pdf_pages':sum(s['pdf_pages'] for s in sources),'parsed_pages':sum(len(p['pages']) for p in parsed.values()),
                  'discovery_errors':discovery_errors,'page_failures':[{'source_id':s['source_id'],**f} for s in sources for f in s.get('page_failures',[])],
                  'page_parse_status_counts':dict(Counter(page['parse_status'] for p in parsed.values() for page in p['pages']))}
        unit_manifest=[{'source_id':sid,'unit_id':u['unit_id'],'pages':u['pages'],'section':u['section'],
                        'input_sha256':stable_hash(u),'input_characters':sum(len(b['text']) for b in u['blocks']),
                        'block_count':len(u['blocks']),'block_ids':[b['block_id'] for b in u['blocks']]}
                       for sid,uu in sorted(units.items()) for u in uu]
        unit_counts={'candidate_units':len(unit_manifest),'planned_model_units':len(jobs),
                     'completed_units':sum(r['status']=='COMPLETED' for r in unit_results),
                     'failed_units':sum(r['status']=='FAILED' for r in unit_results),
                     'not_executed_units':len(jobs)-len(unit_results),
                     'oversized_blocks_pending':sum(r['reason']=='OVERSIZED_SEMANTIC_BLOCK_PENDING' for r in selection)}
        report={**coverage,'status':status,'synthetic_test':mode=='mock','knowledge_counts':{k:sum(c['knowledge_type']==k for c in chunks) for k in TYPES},
                'document_contributions':{s['file_path']:s['chunk_count'] for s in sources},'duplicate_documents':duplicates,'duplicate_knowledge':dups,
                'similar_or_conflicting_knowledge':near,'rejected_knowledge_count':len(excluded),'validation':checks,
                'rejection_reason_counts':dict(sorted(Counter(reason for r in excluded for reason in r.get('rejection_reasons',[])).items())),
                'benchmark_independence':scope_audit,
                'unit_counts':unit_counts,'extraction_coverage':'NOT_EXHAUSTIVE_SEE_UNIT_OMISSIONS_AND_PENDING_BLOCKS',
                'semantic_quality':'PENDING_MEDICAL_REVIEW' if chunks and mode=='build' else 'SYNTHETIC_NOT_MEDICAL_KNOWLEDGE' if mode=='mock' else 'NOT_EXTRACTED',
                'provider_error':provider_error,'extraction_failures':failures,'usage':provider.snapshot() if provider else {'new_api_calls':0},
                'parse_cache_hits':parse_hits,'source_files_unchanged':True,'certificate_writes':0,'new_verified_count':0}
        manifest={'format':'surgical-reference-source-manifest-v0.1','references_root':'CLI_INPUT_DIRECTORY',**coverage,'sources':sources}
        page_audit=[{'source_id':sid,**{k:v for k,v in page.items() if k not in ('raw_text','blocks','clean_text','clean_blocks')}} for sid,p in parsed.items() for page in p['pages']]
        write_json(output,library);write_json(out/'source_manifest.json',manifest);write_json(out/'quality_report.json',report)
        write_json(out/'source_review_template.jsonl',[review_template(s) for s in sources],jsonl=True)
        write_json(out/'source_audit_report.json',{'sources':[{'source_id':s['source_id'],'file_path':s['file_path'],'audit':s['source_audit']} for s in sources]})
        write_json(out/'excluded_knowledge.jsonl',excluded,jsonl=True);write_json(out/'extraction_failures.jsonl',failures,jsonl=True)
        write_json(out/'accepted_candidate_audit.jsonl',records,jsonl=True)
        write_json(out/'extraction_unit_manifest.jsonl',unit_manifest,jsonl=True)
        write_json(out/'unit_extraction_audit.jsonl',unit_results,jsonl=True)
        write_json(out/'content_selection_audit.jsonl',selection,jsonl=True);write_json(out/'page_parsing_audit.jsonl',page_audit,jsonl=True)
        # Verbatim source excerpts are inspection aids, explicitly not generated
        # knowledge cards or approved evidence sufficiency recommendations.
        examples=[]
        for s in sources:
            if s['source_audit']['risk_terms']:continue
            for u in units.get(s['source_id'],[])[:1]:
                b=u['blocks'][0];examples.append({'source_id':s['source_id'],'file_path':s['file_path'],'page':b['page'],
                    'block_id':b['block_id'],'verbatim_source_excerpt':b['text'][:500],'status':'SOURCE_EXCERPT_ONLY_NOT_A_KNOWLEDGE_CARD'})
        write_json(out/'source_excerpt_examples.json',examples)
        lines=['# Source inventory','', '| Relative path | Pages | Extractable | Probable source | Category | Status |', '|---|---:|---|---|---|---|']
        lines += [f'| {s["file_path"]} | {s["pdf_pages"]} | {s["text_extractable"]} | {s["probable_source"]} | {s["document_category"]} | {s["processing_status"]} |' for s in sources]
        inventory='\n'.join(lines)+'\n'
        inventory_path=out/'source_inventory.md'
        if not inventory_path.exists():inventory_path.write_text(inventory,encoding='utf-8')
        names=[p for p in out.iterdir() if p.is_file() and not p.name.startswith('.') and p.name!='build_summary.json']
        summary={'format':'surgical-knowledge-build-summary-v0.1','status':status,'pdf_count':len(sources),'parsed_documents':parsed_count,
                 'chunk_count':len(chunks),'knowledge_counts':report['knowledge_counts'],'usage':report['usage'],
                 'unit_counts':unit_counts,
                 'synthetic_test':mode=='mock','provider_error':provider_error,'certificate_writes':0,'new_verified_count':0,
                 'artifact_sha256':{p.name:file_hash(p) for p in sorted(names)}}
        write_json(out/'build_summary.json',summary);return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['inspect','mock','build','validate'],default='inspect')
    p.add_argument('--input-dir',default='RAG/references');p.add_argument('--output',required=True);p.add_argument('--cache-dir',default='RAG/cache')
    p.add_argument('--provider',choices=['openai-compatible'],default='openai-compatible')
    p.add_argument('--model');p.add_argument('--vision-model');p.add_argument('--base-url');p.add_argument('--source-review')
    for n in ['max-files','max-pages']:p.add_argument('--'+n,type=int)
    p.add_argument('--max-unit-chars',type=int,default=4000)
    p.add_argument('--max-unit-blocks',type=int,default=6)
    p.add_argument('--include-file',action='append',default=[])
    p.add_argument('--max-api-calls',type=int,default=100);p.add_argument('--max-cost-usd',type=float)
    p.add_argument('--max-vision-pages',type=int,default=3)
    p.add_argument('--enable-vision',action='store_true',default=os.environ.get('ENABLE_VISION_EXTRACTION','false').lower()=='true')
    p.add_argument('--dry-run',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--verbose',action='store_true')
    p.add_argument('--no-tables',action='store_true',help='Report vector-table extraction as NOT_REQUESTED, not absent.')
    a=p.parse_args(argv);logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,format='%(levelname)s %(message)s')
    try:
        if a.mode=='validate':
            from .validator import validate_output
            result=validate_output(a.output,a.input_dir,a.cache_dir)
        else:
            result=build(input_dir=a.input_dir,output=a.output,cache_dir=a.cache_dir,mode='inspect' if a.dry_run else a.mode,
                         resume=a.resume,max_files=a.max_files,max_pages=a.max_pages,include_files=a.include_file,source_review=a.source_review,
                         provider_config=Config.environment(base_url=a.base_url,text_model=a.model,vision_model=a.vision_model),
                         enable_vision=a.enable_vision,max_vision_pages=a.max_vision_pages,max_api_calls=a.max_api_calls,
                         max_cost_usd=a.max_cost_usd,tables=not a.no_tables,
                         max_unit_chars=a.max_unit_chars,max_unit_blocks=a.max_unit_blocks)
        print(canonical(result));return 2 if result.get('status') in ('BLOCKED_API_CONFIGURATION','PARTIAL_BUDGET_STOP','PARTIAL_EXTRACTION_FAILURE','SOURCE_REVIEW_REQUIRED','NO_ADMISSIBLE_SOURCE_UNITS') else 0
    except (ValueError,OSError,KeyError) as exc:
        LOG.error('%s',str(exc));return 2


if __name__=='__main__':raise SystemExit(main())
