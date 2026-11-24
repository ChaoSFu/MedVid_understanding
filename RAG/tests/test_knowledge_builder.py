from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import pymupdf
from jsonschema import Draft202012Validator

from RAG.knowledge_builder.common import read_json,write_json,stable_hash,file_hash
from RAG.knowledge_builder.discover import discover
from RAG.knowledge_builder.pdf_parser import parse_pdf,source_metadata,reading_order
from RAG.knowledge_builder.content_selector import select
from RAG.knowledge_builder.source_policy import source_findings,review_template,apply_review
from RAG.knowledge_builder.extractor import extract,MockProvider
from RAG.knowledge_builder.validator import validate_card,validate_library,validate_output
from RAG.knowledge_builder.normalizer import normalize
from RAG.knowledge_builder.provider import Provider,Config,BudgetStop,ProviderError
from RAG.knowledge_builder.schemas import RESPONSE,LIBRARY,TYPES
from RAG.knowledge_builder.build import build

SENTENCE='The surgical forceps has curved jaws and a long narrow shaft visible in the illustration.'


def pdf(path,text=SENTENCE,pages=1):
    path.parent.mkdir(parents=True,exist_ok=True)
    with pymupdf.open() as d:
        for _ in range(pages):
            p=d.new_page()
            if text:p.insert_textbox(pymupdf.Rect(40,40,550,600),text,fontsize=11)
        d.save(path)


class Fixtures(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.refs=self.root/'references';self.refs.mkdir();self.cache=self.root/'cache'
        self.path=self.refs/'器械 with spaces.pdf';pdf(self.path)
    def source(self):
        _,ss,_=discover(self.refs);s=next(s for s in ss if s['file_path']==self.path.name)
        p,_=parse_pdf(self.path,s['sha256'],self.cache,tables=False);s=source_metadata(s,p)
        s['source_audit']=source_findings(s['file_name'],p['pages'][0]['raw_text'])
        return s,p
    def records(self):
        s,p=self.source();u=select(s,p)[0][0];passed,_,_=extract(u,MockProvider())
        return s,u,passed


class DiscoveryParsingTests(Fixtures):
    def test_recursive_29_nested_files_unicode_spaces_uppercase(self):
        for i in range(29):pdf(self.refs/'未知 子目录'/f'{i} name.PDF')
        _,ss,err=discover(self.refs);self.assertEqual(len(ss),30);self.assertFalse(err)
        self.assertEqual(len({s['source_id'] for s in ss}),30)

    def test_source_id_stable_and_content_addressed(self):
        a=discover(self.refs)[1];b=discover(self.refs)[1];self.assertEqual(a,b)
        pdf(self.refs/'second.pdf',text='A different surgical instrument has straight jaws.')
        found=next(s for s in discover(self.refs)[1] if s['file_path']==a[0]['file_path'])
        self.assertEqual(found['source_id'],a[0]['source_id'])

    def test_page_numbers_and_original_bytes_preserved(self):
        pdf(self.refs/'three.pdf',pages=3);h=file_hash(self.refs/'three.pdf')
        p,_=parse_pdf(self.refs/'three.pdf',h,self.cache,tables=False)
        self.assertEqual([p['page'] for p in p['pages']],[1,2,3]);self.assertEqual(file_hash(self.refs/'three.pdf'),h)

    def test_parse_cache_and_hash_validation(self):
        h=file_hash(self.path);a,hit=parse_pdf(self.path,h,self.cache,tables=False)
        b,hit2=parse_pdf(self.path,h,self.cache,tables=False)
        self.assertFalse(hit);self.assertTrue(hit2);self.assertEqual(a,b)
        path=next((self.cache/'parsed').glob('*.json'));d=read_json(path);d['pages'][0]['raw_text']='TAMPER';path.write_text(json.dumps(d))
        with self.assertRaises(ValueError):parse_pdf(self.path,h,self.cache,tables=False)

    def test_corrupt_pdf_recorded_and_other_document_continues(self):
        (self.refs/'broken.pdf').write_bytes(b'not a pdf')
        z=build(input_dir=self.refs,output=self.root/'inspect/k.json',cache_dir=self.cache,tables=False)
        self.assertEqual(z['pdf_count'],2);self.assertEqual(z['parsed_documents'],1)
        self.assertEqual(len(read_json(self.root/'inspect/quality_report.json')['failed_files']),1)

    def test_blank_document_pending_not_fake_knowledge(self):
        pdf(self.refs/'blank.pdf',text='')
        z=build(input_dir=self.refs,output=self.root/'mock/k.mock.json',mode='mock',cache_dir=self.cache,tables=False)
        lib=read_json(self.root/'mock/k.mock.json')
        blank=next(s for s in lib['source_index'] if s['file_name']=='blank.pdf')
        self.assertEqual(blank['chunk_count'],0);self.assertFalse(blank['text_extractable'])
        self.assertTrue(blank['pending_visual_pages'])

    def test_limits_are_reported_not_silent_omissions(self):
        pdf(self.refs/'two.pdf',pages=4)
        z=build(input_dir=self.refs,output=self.root/'inspect/k.json',cache_dir=self.cache,max_files=1,max_pages=1,tables=False)
        self.assertEqual(z['pdf_count'],2)
        ss=read_json(self.root/'inspect/source_manifest.json')['sources'];self.assertTrue(any(s['processing_status']=='DEFERRED_BY_LIMIT' for s in ss))

    def test_noise_removed_with_trace_raw_kept(self):
        pdf(self.refs/'web.pdf',text='Home Instruments & Sutures\n'+SENTENCE+'\nhttps://example.com\nAccept all cookies')
        h=file_hash(self.refs/'web.pdf');p,_=parse_pdf(self.refs/'web.pdf',h,self.cache,tables=False)
        self.assertIn('Accept all cookies',p['pages'][0]['raw_text'])
        self.assertNotIn('Accept all cookies',p['pages'][0]['clean_text']);self.assertTrue(p['pages'][0]['removed_noise'])

    def test_two_column_order_preserves_block_provenance(self):
        blocks=[{'block_id':str(i),'bbox':[x,y,x+180,y+20],'text':'paragraph'} for i,(x,y) in enumerate([(20,20),(20,60),(20,100),(330,20),(330,60),(330,100)])]
        ordered,mode=reading_order(blocks,600)
        self.assertIn('TWO_COLUMN',mode);self.assertEqual([b['block_id'] for b in ordered],['0','1','2','3','4','5'])

    def test_duplicates_by_content_reported(self):
        pdf(self.refs/'copy.pdf')
        z=build(input_dir=self.refs,output=self.root/'inspect/k.json',cache_dir=self.cache,tables=False)
        self.assertEqual(len(read_json(self.root/'inspect/quality_report.json')['duplicate_documents']),1)

    def test_equal_text_different_diagram_not_skipped(self):
        path=self.refs/'different_diagram.pdf'
        with pymupdf.open(self.path) as d:
            d[0].draw_rect(pymupdf.Rect(30,650,80,700),color=(1,0,0),fill=(1,0,0));d.save(path)
        build(input_dir=self.refs,output=self.root/'inspect/k.json',cache_dir=self.cache,tables=False)
        sources=read_json(self.root/'inspect/source_manifest.json')['sources']
        self.assertTrue(all(not s.get('duplicate_of') for s in sources))
        dup=read_json(self.root/'inspect/quality_report.json')['duplicate_documents'][0]
        self.assertFalse(dup['auto_skip']);self.assertEqual(dup['basis'],'TEXT_MATCH_BUT_VISUAL_CONTENT_DIFFERS')


class SourceAndCardTests(Fixtures):
    def test_benchmark_filter_does_not_ban_medical_terms(self):
        self.assertFalse(source_findings('CVS clipping suture.pdf','Critical view of safety and surgical knots.')['risk_terms'])
        for term in ['MedVidU','CholecT50','CVAT mask annotation','OpenSurgicalVision','CoPESD']:
            self.assertTrue(source_findings('source.pdf',term)['risk_terms'])

    def test_source_approval_is_hash_bound_not_automatic(self):
        s,_=self.source();self.assertFalse(s['source_audit']['extraction_allowed'])
        r=review_template(s);r.update(decision='INCLUDE',reviewer_id='human-test',rationale='Test only.',independence_basis='Independent synthetic reference.')
        self.assertTrue(apply_review(s,r)['source_audit']['extraction_allowed'])
        r['sha256']='0'*64
        with self.assertRaises(ValueError):apply_review(s,r)

    def test_mixed_document_needs_page_allowlist(self):
        s,_=self.source();s['source_audit']=source_findings(s['file_name'],'CVAT')
        r=review_template(s);r.update(decision='INCLUDE',reviewer_id='test',rationale='test',independence_basis='test')
        with self.assertRaises(ValueError):apply_review(s,r)
        r['allowed_pages']=[1];self.assertTrue(apply_review(s,r)['source_audit']['extraction_allowed'])

    def test_grounded_card_schema_and_six_types(self):
        s,u,rr=self.records();card=rr[0]['card'];self.assertFalse(list(Draft202012Validator(RESPONSE).iter_errors({'cards':[card],'omitted_content_reasons':[]})))
        self.assertEqual(validate_card(card,s,u,method='text_llm',synthetic=True),[])
        for kind in TYPES:
            c=deepcopy(card);c['knowledge_type']=kind
            if kind=='procedure':c['knowledge_basis']='standard_procedure_prior'
            self.assertEqual(validate_card(c,s,u,method='text_llm',synthetic=True),[])

    def test_missing_or_invented_quote_page_section_rejected(self):
        s,u,rr=self.records()
        for field,value in [('page',999),('quote','This instrument is a unicorn with invented medical properties.'),('block_id','unknown')]:
            c=deepcopy(rr[0]['card']);c['source_facts'][0][field]=value
            self.assertTrue(validate_card(c,s,u,method='text_llm',synthetic=True))
        c=deepcopy(rr[0]['card']);c['source_refs'][0]['section']='Imagined section'
        self.assertIn('SECTION_NOT_IN_SUPPLIED_LOCATOR',validate_card(c,s,u,method='text_llm',synthetic=True))

    def test_ungrounded_guidance_and_features_rejected(self):
        s,u,rr=self.records();c=deepcopy(rr[0]['card'])
        c['visual_features']=[{'text':'Purple jaws.','supporting_fact_ids':['F999']}]
        c['evidence_guidance']['supporting_observations']=[{'text':'The function proves the action.','supporting_fact_ids':[],'derived_guidance_ids':[]}]
        e=validate_card(c,s,u,method='text_llm',synthetic=True)
        self.assertIn('UNSUPPORTED_FACT_REFERENCE',e);self.assertIn('UNGROUNDED_EVIDENCE_GUIDANCE',e)

    def test_benchmark_in_derived_field_rejected(self):
        s,u,rr=self.records();c=deepcopy(rr[0]['card']);c['aliases']=['MedVidU label']
        self.assertIn('BENCHMARK_SPECIFIC_CONTENT',validate_card(c,s,u,method='text_llm',synthetic=True))

    def test_vision_cannot_name_unlabeled_figure(self):
        s,u,rr=self.records();c=deepcopy(rr[0]['card']);f=c['source_facts'][0]
        f.update(extraction_method='vision_llm',basis='figure_interpretation',figure_label=None)
        self.assertIn('FIGURE_LABEL_NOT_SOURCE_GROUNDED',validate_card(c,s,u,method='vision_llm',synthetic=True))

    def test_dedup_and_pending_review_never_medical_approval(self):
        s,u,rr=self.records();before=deepcopy(rr)
        chunks,dups,_,_=normalize(rr+deepcopy(rr));self.assertEqual(len(chunks),1);self.assertEqual(len(dups),1)
        self.assertEqual(rr,before);self.assertEqual(chunks[0]['review_status'],'pending');self.assertFalse(chunks[0]['reviewer'])

    def test_model_variants_not_automatically_merged(self):
        s,u,rr=self.records();other=deepcopy(rr[0]);other['card']['entity_name']='Different forceps model'
        chunks,dups,_,_=normalize(rr+[other]);self.assertEqual(len(chunks),2)


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.env=patch.dict(os.environ,{'LLM_API_KEY':'SYNTHETIC_SECRET_NOT_A_REAL_KEY'});self.env.start();self.addCleanup(self.env.stop)
        self.cfg=Config(base_url='https://provider.invalid/v1',text_model='test-model',max_retries=2,requests_per_minute=1000000)
        self.schema={'type':'object','properties':{'value':{'type':'string'}},'required':['value'],'additionalProperties':False}
    def provider(self,**kw):return Provider(self.cfg,self.root,**kw)
    def complete(self,p):return p.complete(prompt='Extract independently sourced surgical facts.',schema=self.schema,identity={'file_sha':'a','pages':[1],'prompt':'v1'})
    def envelope(self,text):return {'choices':[{'message':{'content':text}}],'usage':{'prompt_tokens':10,'completion_tokens':5}}

    def test_missing_key_stops_no_network(self):
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaisesRegex(ProviderError,'KEY_MISSING'):self.provider()

    def test_custom_base_and_secret_not_in_cache(self):
        p=self.provider()
        with patch.object(p,'_send',return_value=(self.envelope('{"value":"ok"}'),None)):
            r=self.complete(p)
        for path in self.root.rglob('*.json'):self.assertNotIn(os.environ['LLM_API_KEY'],path.read_text())
        self.assertEqual(r['value'],{'value':'ok'});self.assertEqual(p.calls,1)

    def test_thinking_environment_is_explicit_and_validated(self):
        for value,expected in [('',None),('false',False),('true',True),(' FALSE ',False)]:
            with self.subTest(value=value),patch.dict(os.environ,{'LLM_ENABLE_THINKING':value}):
                self.assertIs(Config.environment().enable_thinking,expected)
        with patch.dict(os.environ,{},clear=True):self.assertIsNone(Config.environment().enable_thinking)
        with patch.dict(os.environ,{'LLM_ENABLE_THINKING':'off'}):
            with self.assertRaisesRegex(ProviderError,'LLM_ENABLE_THINKING_INVALID'):Config.environment()
        self.cfg.enable_thinking='false'
        with self.assertRaisesRegex(ProviderError,'LLM_ENABLE_THINKING_INVALID'):self.provider()

    def test_thinking_payload_cache_binding_and_replay(self):
        keys=[]
        for setting in (None,False,True):
            with self.subTest(setting=setting):
                self.cfg.enable_thinking=setting;p=self.provider()
                with patch.object(p,'_send',return_value=(self.envelope('{"value":"ok"}'),None)) as send:
                    result=self.complete(p)
                payload=send.call_args.args[0]
                if setting is None:self.assertNotIn('enable_thinking',payload)
                else:self.assertIs(payload['enable_thinking'],setting)
                keys.append(result['request_key'])
                replay=self.provider()
                with patch.object(replay,'_send',side_effect=AssertionError('NETWORK')):
                    self.assertEqual(self.complete(replay),result)
                self.assertEqual(replay.calls,0);self.assertEqual(replay.hits,1)
        self.assertEqual(len(set(keys)),3)

    def test_server_credential_echo_is_redacted_and_rejected(self):
        p=self.provider();secret=os.environ['LLM_API_KEY']
        with patch.object(p,'_send',return_value=(self.envelope(json.dumps({'value':secret})),None)):
            with self.assertRaisesRegex(ProviderError,'CREDENTIAL_ECHO_REDACTED'):self.complete(p)
        for path in self.root.rglob('*.json'):self.assertNotIn(secret,path.read_text())

    def test_structured_output_capability_fallback_audited(self):
        p=self.provider()
        with patch.object(p,'_send',side_effect=[(None,'STRUCTURED_OUTPUT_UNSUPPORTED'),(self.envelope('{"value":"ok"}'),None)]),patch('time.sleep'):
            self.assertEqual(self.complete(p)['response_mode'],'json_object')
        self.assertEqual(p.calls,2);self.assertEqual(p.capabilities['test-model:json_schema'],'REJECTED_BY_SERVER')

    def test_malformed_json_bounded_retry_and_raw_preserved(self):
        p=self.provider()
        with patch.object(p,'_send',side_effect=[(self.envelope('not-json'),None),(self.envelope('{"value":"ok"}'),None)]),patch('time.sleep'):
            self.complete(p)
        self.assertEqual(p.calls,2);self.assertTrue(any('not-json' in x.read_text() for x in self.root.rglob('*.json')))

    def test_output_limit_environment_payload_and_cache_binding(self):
        with patch.dict(os.environ,{},clear=True):self.assertEqual(Config.environment().max_output_tokens,6000)
        with patch.dict(os.environ,{'LLM_MAX_OUTPUT_TOKENS':'24000'}):self.assertEqual(Config.environment().max_output_tokens,24000)
        with patch.dict(os.environ,{'LLM_MAX_OUTPUT_TOKENS':'invalid'}):
            with self.assertRaisesRegex(ProviderError,'LLM_MAX_OUTPUT_TOKENS_INVALID'):Config.environment()
        keys=[]
        for limit in (6000,24000):
            self.cfg.max_output_tokens=limit;p=self.provider()
            with patch.object(p,'_send',return_value=(self.envelope('{"value":"ok"}'),None)) as send:
                result=self.complete(p)
            keys.append(result['request_key']);self.assertEqual(send.call_args.args[0]['max_tokens'],limit)
        self.assertEqual(len(set(keys)),2)
        for invalid in (0,-1,False,1.5,'24000'):
            self.cfg.max_output_tokens=invalid
            with self.assertRaisesRegex(ProviderError,'LLM_MAX_OUTPUT_TOKENS_INVALID'):self.provider()

    def test_length_finish_is_terminal_even_if_json_looks_complete(self):
        for raw in ('{"value":"ok"}','{"value":"partial'):
            with self.subTest(raw=raw),tempfile.TemporaryDirectory() as d:
                p=Provider(self.cfg,d);env=self.envelope(raw);env['choices'][0]['finish_reason']='length'
                with patch.object(p,'_send',return_value=(env,None)) as send,patch('time.sleep') as sleep:
                    with self.assertRaisesRegex(ProviderError,'MODEL_OUTPUT_TRUNCATED'):self.complete(p)
                self.assertEqual(send.call_count,1);sleep.assert_not_called()
                self.assertFalse(list(Path(d).rglob('success.json')))
                record=read_json(next(Path(d).rglob('*.json')))
                self.assertEqual(record['finish_reason'],'length')
                self.assertEqual(record['truncation_evidence'],'PROVIDER_FINISH_REASON_LENGTH')

    def test_invalid_json_at_limit_without_finish_reason_stops_once(self):
        p=self.provider();env=self.envelope('{"value":"partial')
        env['usage']['completion_tokens']=self.cfg.max_output_tokens
        with patch.object(p,'_send',return_value=(env,None)) as send:
            with self.assertRaisesRegex(ProviderError,'MODEL_OUTPUT_TRUNCATED'):self.complete(p)
        self.assertEqual(send.call_count,1)
        record=read_json(next(self.root.rglob('*.json')))
        self.assertEqual(record['truncation_evidence'],'INVALID_JSON_AT_TOKEN_LIMIT')
        self.assertEqual(record['validation_error']['kind'],'JSON_SYNTAX')
        self.assertEqual(record['request_identity']['pages'],[1])
        self.assertEqual(p.snapshot()['attempt_failure_reason_counts'],{'MODEL_OUTPUT_TRUNCATED':1})

    def test_complete_json_at_limit_stop_is_valid_and_replay_is_zero_calls(self):
        p=self.provider();env=self.envelope('{"value":"ok"}')
        env['choices'][0]['finish_reason']='stop';env['usage']['completion_tokens']=6000
        with patch.object(p,'_send',return_value=(env,None)):result=self.complete(p)
        self.assertEqual(result['finish_reason'],'stop')
        replay=self.provider()
        with patch.object(replay,'_send',side_effect=AssertionError('NETWORK')):self.assertEqual(self.complete(replay),result)
        self.assertEqual(replay.calls,0);self.assertEqual(replay.hits,1)

    def test_schema_error_has_path_and_bounded_retry(self):
        p=self.provider()
        with patch.object(p,'_send',side_effect=[(self.envelope('{"value":123}'),None),
                                              (self.envelope('{"value":"ok"}'),None)]),patch('time.sleep'):
            self.complete(p)
        attempts=[read_json(x) for x in self.root.rglob('*.json') if x.name!='success.json']
        invalid=next(r for r in attempts if r['failure_reason'])
        self.assertEqual(invalid['failure_reason'],'MODEL_SCHEMA_INVALID')
        self.assertEqual(invalid['validation_error'],{'kind':'JSON_SCHEMA','path':['value'],'validator':'type'})
        self.assertEqual(p.calls,2)

    def test_filtered_response_is_terminal_and_never_cached_as_success(self):
        p=self.provider();env=self.envelope('{"value":"ok"}');env['choices'][0]['finish_reason']='content_filter'
        with patch.object(p,'_send',return_value=(env,None)) as send:
            with self.assertRaisesRegex(ProviderError,'MODEL_RESPONSE_NOT_COMPLETED'):self.complete(p)
        self.assertEqual(send.call_count,1);self.assertFalse(list(self.root.rglob('success.json')))

    def test_output_limit_is_in_cost_reservation(self):
        self.cfg.max_output_tokens=24000;self.cfg.input_usd_per_million=0;self.cfg.output_usd_per_million=1
        p=self.provider(max_cost_usd=0.01)
        with patch.object(p,'_send',side_effect=AssertionError('NETWORK')):
            with self.assertRaises(BudgetStop):self.complete(p)
        self.assertEqual(p.calls,0)

    def test_cache_replay_zero_network_and_identity_changes(self):
        p=self.provider()
        with patch.object(p,'_send',return_value=(self.envelope('{"value":"ok"}'),None)):a=self.complete(p)
        q=self.provider()
        with patch.object(q,'_send',side_effect=AssertionError('NETWORK')):b=self.complete(q)
        self.assertEqual(a,b);self.assertEqual(q.calls,0);self.assertEqual(q.hits,1)
        q.config.text_model='different-model'
        with patch.object(q,'_send',return_value=(self.envelope('{"value":"new"}'),None)):c=self.complete(q)
        self.assertNotEqual(c['request_key'],a['request_key'])

    def test_max_calls_and_token_usage(self):
        p=self.provider(max_calls=1)
        with patch.object(p,'_send',return_value=(self.envelope('malformed'),None)),patch('time.sleep'):
            with self.assertRaises(BudgetStop):self.complete(p)
        self.assertEqual(p.calls,1);self.assertEqual(p.snapshot()['input_tokens'],10)

    def test_cost_budget_requires_rates_and_reserves_before_send(self):
        with self.assertRaises(ProviderError):self.provider(max_cost_usd=1)
        self.cfg.input_usd_per_million=10;self.cfg.output_usd_per_million=10
        p=self.provider(max_cost_usd=0)
        with patch.object(p,'_send',side_effect=AssertionError('NETWORK')):
            with self.assertRaises(BudgetStop):self.complete(p)
        self.assertEqual(p.calls,0)

    def test_vision_unsupported_is_explicit(self):
        self.cfg.vision_model='vision-test';p=self.provider()
        with patch.object(p,'_send',return_value=(None,'VISION_UNSUPPORTED')):
            with self.assertRaisesRegex(ProviderError,'VISION_UNSUPPORTED'):p.complete(prompt='labeled figure',schema=self.schema,identity={},image=b'png')
        self.assertEqual(p.calls,1)

    def test_no_redirect_or_embedded_credentials(self):
        self.cfg.base_url='https://secret:password@provider.invalid/v1'
        with self.assertRaises(ProviderError):self.provider()


class PipelineTests(Fixtures):
    def test_inspect_zero_api_formal_json_empty_and_schema_valid(self):
        with patch('RAG.knowledge_builder.provider.Provider._send',side_effect=AssertionError('NETWORK')):
            z=build(input_dir=self.refs,output=self.root/'inspect/library.json',cache_dir=self.cache,tables=False)
        self.assertEqual(z['usage']['new_api_calls'],0);lib=read_json(self.root/'inspect/library.json')
        self.assertFalse(lib['chunks']);self.assertEqual(validate_library(lib)['schema'],'PASS')
        self.assertEqual(validate_output(self.root/'inspect/library.json',self.refs,self.cache)['source_file_hashes'],'PASS')

    def test_mock_forbidden_in_formal_filename_and_schema(self):
        with self.assertRaises(ValueError):build(input_dir=self.refs,output=self.root/'relive_surgical_visual_knowledge_v0.1.json',cache_dir=self.cache,mode='mock')
        z=build(input_dir=self.refs,output=self.root/'mock/k.mock.json',cache_dir=self.cache,mode='mock',tables=False)
        lib=read_json(self.root/'mock/k.mock.json');self.assertTrue(lib['synthetic_test']);self.assertGreater(z['chunk_count'],0)
        lib['library_id']='relive.surgical_visual_knowledge'
        with self.assertRaises(ValueError):validate_library(lib)

    def test_resume_byte_identical_zero_new_invocations(self):
        output=self.root/'mock/k.mock.json';args=dict(input_dir=self.refs,output=output,cache_dir=self.cache,mode='mock',tables=False)
        first=build(**args);before=output.read_bytes();second=build(**args,resume=True)
        self.assertEqual(output.read_bytes(),before);self.assertEqual(second['invocation']['new_api_calls'],0)
        with self.assertRaises(ValueError):build(**args)

    def test_changed_pdf_fails_resume_binding(self):
        args=dict(input_dir=self.refs,output=self.root/'mock/k.mock.json',cache_dir=self.cache,mode='mock',tables=False)
        build(**args);self.path.unlink();pdf(self.path,text='The surgical scissors have straight sharp blades.')
        with self.assertRaisesRegex(ValueError,'RESUME_IDENTITY_MISMATCH'):build(**args,resume=True)

    def test_no_credentials_build_reports_block_not_fake_cards(self):
        with patch.dict(os.environ,{},clear=True):
            z=build(input_dir=self.refs,output=self.root/'build/library.json',cache_dir=self.cache,mode='build',tables=False)
        self.assertEqual(z['status'],'BLOCKED_API_CONFIGURATION');self.assertEqual(z['chunk_count'],0)

    def test_mock_artifacts_deterministic_when_created_at_fixed(self):
        libs=[]
        for n in ('a','b'):
            build(input_dir=self.refs,output=self.root/n/'k.mock.json',cache_dir=self.cache,mode='mock',tables=False,created_at='2026-01-01T00:00:00Z')
            libs.append((self.root/n/'k.mock.json').read_bytes())
        self.assertEqual(*libs)

    def test_page_source_review_pending_has_no_real_requests(self):
        class Forbidden:
            synthetic=False
            def snapshot(self):return {'new_api_calls':0}
            def complete(self,**kw):raise AssertionError('UNREVIEWED_SOURCE_SENT_TO_MODEL')
        z=build(input_dir=self.refs,output=self.root/'build/library.json',cache_dir=self.cache,mode='build',provider_factory=Forbidden,tables=False)
        self.assertEqual(z['status'],'SOURCE_REVIEW_REQUIRED');self.assertEqual(z['usage']['new_api_calls'],0)

    def test_offline_validation_checks_real_pdf_hashes(self):
        build(input_dir=self.refs,output=self.root/'inspect/library.json',cache_dir=self.cache,tables=False)
        self.path.unlink();pdf(self.path,text='This changed document describes a different surgical instrument.')
        with self.assertRaisesRegex(ValueError,'SOURCE_FILE_HASH_MISMATCH'):validate_output(self.root/'inspect/library.json',self.refs,self.cache)


if __name__=='__main__':unittest.main()
