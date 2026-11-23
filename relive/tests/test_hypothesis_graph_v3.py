"""Engineering fixtures only. No inference, clinical truth or certificates."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from relive.backends.base import Backend
from relive.storage.artifacts import stable_hash
from relive.hypothesis_graph.contract import SCHEMA, SMOKE22, VERSION, protocol, prompt_for
from relive.hypothesis_graph.validation import validate
from relive.hypothesis_graph.compiler import compile_node, compile_bank, review_rows, make_review_template
from relive.hypothesis_graph.sources import write, seal, rows, digest, load_run, load_references, prepare
from relive.hypothesis_graph.runner import run, GenerationEnvelope
from relive.hypothesis_graph.fixtures import FixtureBackend
from relive.hypothesis_graph.report import compare
from relive.hypothesis_graph.diagnostic import project_a, project_b, run_diagnostic
from relive.claim_planner import contract as v2
from relive.claim_planner.fixtures import claim as old_claim
from relive.claim_revalidation.offline import revalidate

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/hypothesis_graph_stage2_v3_mock.yaml'
REQ={'requirement_id':'R1','description':'Inspect the forceps motion.','type':'ACTION','evidence_type':'TEMPORAL','importance':'REQUIRED'}
ROW={'question':'Is the forceps visible? Does it move tissue?','qa_type':'tal','requirements':[REQ]}


def claim(cid='H1',text='A forceps is visible.',kind='OBJECT',**kw):
    return {'claim_id':cid,'text':text,'type':kind,'role':'TARGET','requirement_ids':['R1'],
            'requires':[],'alternative_to':[],**kw}


def graph(claims=None,**kw):
    return {'plan_status':'PLANNED','claims':[claim()] if claims is None else claims,
            'unresolved_references':[],'requirement_issues':[],'uncovered_requirements':[],**kw}


def evaluate(value,row=None):return validate(json.dumps(value),ROW if row is None else row)


def make_refs(path, *, mismatch=False):
    path.mkdir();rr=[]
    qa=['tal','stg','region_caption_gemini','region_caption_gpt','dense_captioning_gpt',
        'dense_captioning_gemini','next_action','video_summary_gemini','video_summary_gpt','cvs_assessment','skill_assessment']
    for i,index in enumerate(SMOKE22):
        reqs=[{**REQ,'requirement_id':f'R{n+1}'} for n in range(5 if i<12 else 4)]
        q='Is the forceps visible?'
        rr.append({'sample_key':f'synthetic-{index}','source_record_index':index,'qa_type':qa[i//2],
            'question':q,'requirements':reqs,'question_sha256':stable_hash(q),'requirements_sha256':stable_hash(reqs),
            'question_metadata':{'explicit_time_queries':[],'source':'PUBLIC_QUESTION_ONLY','frame_time_mapping':'NOT_BOUND'},
            'source_binding_status':'FAIL' if mismatch and i==0 else 'PASS','source_binding_issues':['TEST_MISMATCH'] if mismatch and i==0 else []})
    write(path/'public_references.jsonl',rr,jsonl=True)
    quarantine=[{'sample_key':r['sample_key'],'v1':None,'v2':{'raw_response':'HISTORICAL_CLAIM_SENTINEL'},'v21':{'candidate_records':[]}} for r in rr]
    write(path/'comparison_quarantine.jsonl',quarantine,jsonl=True)
    seal(path,['public_references.jsonl','comparison_quarantine.jsonl'],'relive-v3-references-v1',synthetic_source=True,stage1_independently_checked=False)
    return rr


class ValidatorTests(unittest.TestCase):
    def test_closed_schema_roundtrip_and_protocol_hash(self):
        self.assertEqual(json.loads(json.dumps(SCHEMA)),SCHEMA)
        self.assertEqual(json.loads((ROOT/'schemas/hypothesis_graph_stage2_v3.schema.json').read_text()),SCHEMA)
        p=protocol();h=p.pop('protocol_sha256');self.assertEqual(h,stable_hash(p))

    def test_program_only_status_constants_and_no_locations(self):
        r=evaluate(graph());self.assertEqual(r['claim_audit'][0]['state'],'STRUCTURALLY_ACCEPTED')
        self.assertEqual(r['claims'][0],{**claim(),'status':'UNVERIFIED','evidence_status':'NOT_SEARCHED'})
        for key,value in [('status','VERIFIED'),('bbox',[1,2,3,4]),('time',0),('evidence_status','FOUND')]:
            c=claim();c[key]=value
            self.assertFalse(evaluate(graph([c]))['claims'])

    def test_invalid_json_fence_and_duplicate_keys(self):
        for raw in ['```json\n{}\n```','{"claims":[],"claims":[]}','[]','null']:
            r=validate(raw,ROW);self.assertTrue(r['root_errors']);self.assertFalse(r['claims'])

    def test_unknown_type_role_and_extra_root_fields(self):
        for c in [claim(kind='ACTION'),claim(role='FACT')]:self.assertFalse(evaluate(graph([c]))['claims'])
        self.assertFalse(evaluate(graph(bbox=[]))['claims'])

    def test_empty_or_unable_plan_not_silently_admitted(self):
        self.assertTrue(evaluate(graph([]))['root_errors'])
        self.assertTrue(evaluate(graph(plan_status='UNABLE_TO_PLAN'))['root_errors'])

    def test_numeric_search_bindings_rejected_not_fixed(self):
        for text in ['The forceps is visible at 12 seconds.','Frame 8 contains a forceps.','The fps is 30.','The ROI is [1,2,3,4].']:
            r=evaluate(graph([claim(text=text)]));self.assertFalse(r['claims']);self.assertEqual(r['claim_audit'][0]['raw_candidate']['text'],text)

    def test_case04_parameterized_visibility_and_association(self):
        cs=[claim(text='The queried pen is visible at time t.'),claim('H2','The detected pen is the same tracked instance across t_i and t_j.','TEMPORAL_EVENT',requires=['H1'])]
        r=evaluate(graph(cs));self.assertEqual(len(r['claims']),2);self.assertFalse(r['root_errors'])

    def test_parameterized_duplicate_rejected(self):
        r=evaluate(graph([claim(text='A forceps is visible at time t_i.'),claim('H2','A forceps is visible at time t_j.')]))
        self.assertEqual(len(r['claims']),1);self.assertIn('DUPLICATE_PROPOSITION',r['claim_audit'][1]['errors'])

    def test_unique_ids_and_valid_requirement_references(self):
        self.assertFalse(evaluate(graph([claim(),claim()]))['claims'])
        self.assertFalse(evaluate(graph([claim(requirement_ids=['R404'])]))['claims'])
        self.assertIn('QUESTION_ONLY_RELEVANCE_REVIEW',evaluate(graph([claim(requirement_ids=[])]))['claim_audit'][0]['risks'])

    def test_requires_cycle_and_dangling(self):
        for cs in [[claim(requires=['H9'])],[claim(requires=['H2']),claim('H2',requires=['H1'])]]:
            self.assertFalse(evaluate(graph(cs))['claims'])

    def test_partial_acceptance_propagates_prerequisite_block(self):
        cs=[claim(text='A forceps is visible at 12 seconds.'),claim('H2','The forceps grasps tissue.','STATIC_RELATION',requires=['H1']),claim('H3','A needle is visible.')]
        r=evaluate(graph(cs));self.assertEqual([c['claim_id'] for c in r['claims']],['H3'])
        self.assertEqual(r['claim_audit'][1]['state'],'BLOCKED_BY_DEPENDENCY')

    def test_rejected_alternative_cannot_leave_dangling_edge(self):
        r=evaluate(graph([claim(text='A forceps is visible at 12 seconds.'),claim('H2',alternative_to=['H1'])]))
        self.assertFalse(r['claims'])

    def test_alternatives_are_reviewed_not_assumed_exclusive(self):
        r=evaluate(graph([claim(alternative_to=['H2']),claim('H2','A needle is visible.')]))
        self.assertIn('ALTERNATIVE_EXCLUSIVITY_REVIEW',r['claim_audit'][0]['risks'])

    def test_requires_does_not_imply_truth(self):
        r=evaluate(graph([claim(),claim('H2','The forceps grasps tissue.','STATIC_RELATION',requires=['H1'])]))
        self.assertEqual(r['claims'][1]['status'],'UNVERIFIED')
        self.assertIn('LOGICAL_DEPENDENCY_REVIEW',r['claim_audit'][1]['risks'])

    def test_case02_motion_not_exclusive_and_bad_necessity_reviewed(self):
        self.assertFalse(evaluate(graph([claim(text='The camera is always stable.')]))['claims'])
        r=evaluate(graph([claim(text='Instrument manipulation must be absent.')]))
        self.assertIn('UNSUPPORTED_NECESSITY_REVIEW',r['claim_audit'][0]['risks'])
        good=graph([claim(text='The camera view shifts.',kind='TEMPORAL_EVENT')])
        good['requirement_issues']=[{'requirement_id':'R1','issue':'Camera and tissue motion can co-occur; absence is not necessary.'}]
        self.assertEqual(evaluate(good)['requirement_issues'],good['requirement_issues'])

    def test_case05to08_rc_unknown_target_not_guessed(self):
        row={**ROW,'qa_type':'region_caption_gpt'}
        self.assertFalse(evaluate(graph(),row)['claims'])
        r=evaluate(graph([],plan_status='UNRESOLVED_TARGET_REFERENCE',unresolved_references=['Public box absent.']),row)
        self.assertFalse(r['root_errors']);self.assertFalse(r['claims'])
        guessed=evaluate(graph(plan_status='UNRESOLVED_TARGET_REFERENCE',unresolved_references=['Box absent']),row)
        self.assertIn('RC_TARGET_GUESS_FORBIDDEN',guessed['claim_audit'][0]['errors'])

    def test_case10to12_chained_events_rejected(self):
        r=evaluate(graph([claim(text='The tool cuts tissue then retracts it.',kind='TEMPORAL_EVENT')]))
        self.assertIn('EXPLICIT_MULTI_PREDICATE_CHAIN',r['claim_audit'][0]['errors'])

    def test_case14_future_does_not_become_observed_fact(self):
        r=evaluate(graph([claim(text='The surgeon will cut tissue.',kind='TEMPORAL_EVENT')]))
        self.assertIn('PROSPECTIVE_FACT_REVIEW',r['claim_audit'][0]['risks']);self.assertEqual(r['claims'][0]['status'],'UNVERIFIED')

    def test_case17to18_abstract_summary_template_remains_review(self):
        r=evaluate(graph([claim(text='A candidate event occurs.',kind='TEMPORAL_EVENT')]))
        self.assertEqual(r['claim_audit'][0]['state'],'REVIEW_REQUIRED')

    def test_case19to20_independent_clinical_criteria_not_forced_chain(self):
        cs=[claim(text='The criterion holds.',kind='HIGH_LEVEL',role='INTERPRETATION'),claim('H2','The other criterion holds.','HIGH_LEVEL',role='INTERPRETATION')]
        r=evaluate(graph(cs));self.assertTrue(all(not c['requires'] for c in r['claims']))
        self.assertTrue(all('HIGH_LEVEL_PREREQUISITES_MISSING' in a['risks'] for a in r['claim_audit']))

    def test_case21to22_presence_does_not_establish_skill(self):
        r=evaluate(graph([claim(),claim('H2','The handling is efficient.','HIGH_LEVEL',role='INTERPRETATION',requires=['H1'])]))
        high=r['claims'][1];self.assertEqual(high['status'],'UNVERIFIED')
        self.assertEqual(compile_node(high,{c['claim_id']:c for c in r['claims']},{'inference_requirement':'Review tissue handling.'})['compilation_status'],'UNCOMPILABLE')

    def test_question_priority_prompt_no_images_claims_gt(self):
        row={**ROW,'claims':[{'text':'OLD_CLAIM_SENTINEL'}],'struc_info':{'answer':'GT_SENTINEL'},'RC_info':'GT_SENTINEL','conversations':[{'from':'gpt','value':'GT_SENTINEL'}]}
        text=prompt_for(row);self.assertNotIn('GT_SENTINEL',text);self.assertNotIn('OLD_CLAIM_SENTINEL',text)
        self.assertIn('Question semantics take priority',text)

    def test_coverage_id_count_not_semantic_coverage(self):
        r=evaluate(graph(uncovered_requirements=['R1']));self.assertTrue(r['coverage_report_disagrees'])
        self.assertEqual(r['uncovered_requirements'],['R1'])


class CompilerTests(unittest.TestCase):
    def compiled(self,c,others=None,review=None):
        c={**c,'status':'UNVERIFIED','evidence_status':'NOT_SEARCHED'}
        return compile_node(c,{c['claim_id']:c,**(others or {})},review)

    def test_object_single_frame_and_parameter(self):
        for text in ['A forceps is visible.','The queried pen is visible at time t.']:
            z=self.compiled(claim(text=text));self.assertEqual(z['evidence_modality'],'SINGLE_FRAME');self.assertEqual(z['candidate_frames'],[])

    def test_attribute_object_and_discriminator(self):
        z=self.compiled(claim(text='The forceps jaws are open.',kind='ATTRIBUTE'))
        self.assertEqual(z['spatial_targets'],['forceps jaws']);self.assertEqual(z['discriminative_attribute'],'open')

    def test_static_relation_two_regions_interface(self):
        z=self.compiled(claim(text='The forceps grasps tissue.',kind='STATIC_RELATION'))
        self.assertEqual(z['evidence_modality'],'MULTI_REGION_SINGLE_FRAME');self.assertEqual(z['spatial_targets'],['forceps','tissue','contact_interface'])

    def test_temporal_event_multiple_observations_not_fake_time(self):
        z=self.compiled(claim(text='The forceps moves tissue.',kind='TEMPORAL_EVENT'))
        self.assertEqual(z['minimum_observations'],2);self.assertEqual(z['time_bindings'],{});self.assertEqual(z['spatial_regions'],[])

    def test_high_level_explicit_inference_required(self):
        c=claim('H2','The handling is efficient.','HIGH_LEVEL',requires=['H1'])
        other={'H1':claim(text='The forceps moves tissue.',kind='TEMPORAL_EVENT')}
        self.assertEqual(self.compiled(c,other)['compilation_status'],'UNCOMPILABLE')
        z=self.compiled(c,other,{'inference_requirement':'Compare motion features with the question-defined criterion.'})
        self.assertEqual(z['evidence_modality'],'COMPOSITIONAL');self.assertFalse(z['prerequisites_imply_claim'])

    def test_unknown_sentence_and_target_uncompilable(self):
        for c in [claim(text='Maybe something matters.'),claim(text='The unresolved target is visible.'),claim(text='A forceps is visible at 12 seconds.')]:
            self.assertEqual(self.compiled(c)['compilation_status'],'UNCOMPILABLE')

    def test_compiler_does_not_mutate_claim_or_invent_bindings(self):
        c=claim();before=deepcopy(c);z=self.compiled(c)
        self.assertEqual(c,before);self.assertEqual(z['claim_text'],c['text']);self.assertEqual(z['evidence_status'],'NOT_SEARCHED')


class FlowTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.refs=self.root/'refs';make_refs(self.refs)
        self.cache=self.root/'cache';self.out=self.root/'run'

    def execute(self,**kw):
        return run(references_dir=self.refs,config_path=CONFIG,cache_dir=self.cache,output_dir=kw.pop('output_dir',self.out),**kw)

    def test_mock22_exact_replay_without_loading_backend_or_images(self):
        b=FixtureBackend({'kind':'mock','max_retries':0})
        before={p:digest(p) for p in self.refs.iterdir()}
        with patch('PIL.Image.open',side_effect=AssertionError('IMAGE_READ')):
            fresh=self.execute(_backend=b)
            with patch('relive.hypothesis_graph.runner.make_backend',side_effect=AssertionError('MODEL_LOAD')):
                replay=self.execute(output_dir=self.root/'replay',replay_from=self.out)
        self.assertEqual(b.calls,22);self.assertEqual(fresh['new_model_calls'],22);self.assertEqual(replay['new_model_calls'],0)
        self.assertEqual(replay['usage']['cache_hits'],22);self.assertTrue(replay['synthetic_test'])
        self.assertEqual((self.out/'stage2_v3_smoke22.jsonl').read_bytes(),(self.root/'replay/stage2_v3_smoke22.jsonl').read_bytes())
        self.assertEqual(fresh['claim_state_counts'],replay['claim_state_counts']);self.assertIsNone(fresh['token_usage']['generated_tokens'])
        self.assertTrue(all(digest(p)==h for p,h in before.items()))
        self.assertEqual(fresh['requirement_count'],100)

    def test_comparison_and_dry_compiler_full22(self):
        self.execute(_backend=FixtureBackend({'kind':'mock'}))
        with patch('PIL.Image.open',side_effect=AssertionError('IMAGE_READ')),patch('relive.hypothesis_graph.runner.make_backend',side_effect=AssertionError('MODEL_LOAD')):
            c=compare(run_dir=self.out,references_dir=self.refs,output_dir=self.root/'compare')
            z=compile_bank(run_dir=self.out,references_dir=self.refs,output_dir=self.root/'compiler')
        self.assertEqual(c['sample_count'],22);self.assertEqual(z['new_model_calls'],0);self.assertFalse(z['publication_allowed'])
        self.assertEqual((self.root/'compiler/evidence_contracts.jsonl').read_bytes(),b'')
        html=(self.root/'compare/stage2_v1_v2_v3_comparison.html').read_text()
        self.assertEqual(html.count('<article id='),22);self.assertIn('SYNTHETIC TEST ONLY',html)

    def test_synthetic_and_pending_review_cannot_publish(self):
        self.execute(_backend=FixtureBackend({'kind':'mock'}))
        make_review_template(run_dir=self.out,references_dir=self.refs,output_dir=self.root/'review')
        rr=rows(self.root/'review/semantic_review.jsonl')
        for r in rr:r.update(decision='APPROVE',reviewer_id='TEST_REVIEWER',rationale='Synthetic test only.')
        write(self.root/'approved.jsonl',rr,jsonl=True)
        z=compile_bank(run_dir=self.out,references_dir=self.refs,review_jsonl=self.root/'approved.jsonl',output_dir=self.root/'compiled',publish=True)
        self.assertFalse(z['publication_allowed']);self.assertIn('SYNTHETIC_OR_UNEXECUTED_PUBLICATION_FORBIDDEN',z['gate_reasons'])

    def test_review_hash_mismatch_and_duplicate_fail_closed(self):
        self.execute(_backend=FixtureBackend({'kind':'mock'}))
        rr=review_rows(self.out,self.refs);rr[0]['claim_sha256']='0'*64;rr.append(rr[0]);write(self.root/'bad.jsonl',rr,jsonl=True)
        z=compile_bank(run_dir=self.out,references_dir=self.refs,review_jsonl=self.root/'bad.jsonl',output_dir=self.root/'compiled',publish=True)
        self.assertIn('REVIEW_BINDING_MISMATCH',z['gate_reasons']);self.assertIn('UNEXPECTED_OR_DUPLICATE_REVIEW',z['gate_reasons'])

    def test_source_mismatch_skips_sample_without_request(self):
        refs=self.root/'mismatch_refs';make_refs(refs,mismatch=True)
        b=FixtureBackend({'kind':'mock'})
        z=run(references_dir=refs,config_path=CONFIG,cache_dir=self.cache,output_dir=self.out,_backend=b)
        self.assertEqual(b.calls,21);self.assertEqual(z['plan_status_counts']['FROZEN_INPUT_MISMATCH'],1)

    def test_dry_run_zero_calls_and_no_cache_directory(self):
        z=self.execute(dry_run=True)
        self.assertEqual(z['new_model_calls'],0);self.assertFalse(self.cache.exists());self.assertEqual(z['retained_hypothesis_count'],0)

    def test_output_immutable_and_finished_resume_readonly(self):
        self.execute(_backend=FixtureBackend({'kind':'mock'}));h=digest(self.out/'manifest.json')
        with self.assertRaises(ValueError):self.execute()
        self.execute(resume=True);self.assertEqual(digest(self.out/'manifest.json'),h)

    def test_checkpoints_and_manifest_tampering_detected(self):
        self.execute(_backend=FixtureBackend({'kind':'mock'}))
        cp=next((self.out/'samples').iterdir());cp.write_text('{}')
        with self.assertRaises((ValueError,KeyError)):load_run(self.out,self.refs)

    def test_resume_after_interrupt_counts_only_remaining_calls(self):
        class Interrupt(FixtureBackend):
            def infer(self,request):
                if self.calls==3:raise KeyboardInterrupt()
                return super().infer(request)
        with self.assertRaises(KeyboardInterrupt):self.execute(_backend=Interrupt({'kind':'mock'}))
        b=FixtureBackend({'kind':'mock'})
        z=self.execute(_backend=b,resume=True)
        self.assertEqual(b.calls,19);self.assertEqual(z['new_model_calls'],19)
        self.assertEqual(len(rows(self.out/'stage2_v3_smoke22.jsonl')),22)

    def test_replay_cache_miss_never_falls_back_to_inference(self):
        self.execute(_backend=FixtureBackend({'kind':'mock'}))
        with self.assertRaises(ValueError):
            run(references_dir=self.refs,config_path=CONFIG,cache_dir=self.root/'missing-cache',output_dir=self.root/'replay',replay_from=self.out)

    def test_generation_metadata_uses_one_adapter_call(self):
        class Meta(Backend):
            synthetic=True
            def infer(self,request):raise AssertionError('SECOND_CALL')
            def infer_with_generation_metadata(self,request):
                self.calls+=1
                return {'raw_response':'{}','generation_metadata':{'generated_token_count':7,'finish_reason':'EOS_TOKEN'}}
        b=Meta({});z=json.loads(GenerationEnvelope(b).infer({'image_paths':[],'frame_ids':[]}))
        self.assertEqual(b.calls,1);self.assertEqual(z['generation_metadata']['generated_token_count'],7)

    def test_generation_envelope_rejects_images(self):
        with self.assertRaises(ValueError):GenerationEnvelope(FixtureBackend({})).infer({'image_paths':['a.png']})

    def test_failed_response_preserved_without_semantic_retry(self):
        class Malformed(FixtureBackend):
            def infer(self,request):self.calls+=1;return '```json\nnot a graph\n```'
        b=Malformed({'kind':'mock'});z=self.execute(_backend=b)
        self.assertEqual(b.calls,22);self.assertEqual(z['retained_hypothesis_count'],0)
        self.assertTrue(all(r['raw_response']=='```json\nnot a graph\n```' for r in rows(self.out/'stage2_v3_smoke22.jsonl')))

    def test_hash_binding_changes_block_replay(self):
        self.execute(_backend=FixtureBackend({'kind':'mock'}))
        cfg=self.root/'changed.yaml';cfg.write_text(CONFIG.read_text()+'\n# different frozen config bytes\n')
        with self.assertRaises(ValueError):
            run(references_dir=self.refs,config_path=cfg,cache_dir=self.cache,output_dir=self.root/'replay',replay_from=self.out)

    def test_reference_extra_gt_fields_and_corruption_fail_closed(self):
        path=self.refs/'public_references.jsonl';data=rows(path)
        data[0]['struc_info']={'answer':'GT_SENTINEL'}
        path.write_text(''.join(json.dumps(r)+'\n' for r in data))
        with self.assertRaises(ValueError):self.execute(dry_run=True)

    def test_real_run_requires_independent_stage1_before_model_loading(self):
        public,_=load_references(self.refs)
        with patch('relive.hypothesis_graph.runner.load_references',return_value=(public,{'synthetic_source':False,'stage1_independently_checked':False})),patch('relive.hypothesis_graph.runner.make_backend',side_effect=AssertionError('MODEL_LOAD')):
            with self.assertRaisesRegex(ValueError,'INDEPENDENT_STAGE1_BINDING_REQUIRED'):
                run(references_dir=self.refs,config_path=ROOT/'configs/qwen35_9b_hypothesis_graph_stage2_v3.yaml',
                    cache_dir=self.cache,output_dir=self.out)

    def test_gt_and_old_claim_sentinels_never_in_requests_or_run_artifacts(self):
        class Capture(FixtureBackend):
            def infer(self,request):
                self.assertions.append(json.dumps(request));return super().infer(request)
        b=Capture({'kind':'mock'});b.assertions=[];self.execute(_backend=b)
        for s in b.assertions+[p.read_text() for p in self.out.rglob('*.json*')]:
            self.assertNotIn('HISTORICAL_CLAIM_SENTINEL',s);self.assertNotIn('GT_SENTINEL',s)


class DiagnosticTests(unittest.TestCase):
    def candidate(self):
        c=old_claim();c['text']='The marker is visible.'
        c['proposition']['scope']['query_time']={'start_seconds':None,'end_seconds':None,'stride_seconds':None,'question_quote':'Is the marker visible?'}
        return c

    def test_projections_preserve_text_and_do_not_modify_raw(self):
        c=self.candidate();before=deepcopy(c)
        a,removed,errors=project_a(c);b,be=project_b(c)
        self.assertEqual(c,before);self.assertEqual(a['text'],c['text']);self.assertEqual(b['text'],c['text'])
        self.assertNotIn('bindings',a);self.assertTrue(removed);self.assertEqual(be,[])

    def test_full84_diagnostic_and_source_prepare_zero_calls(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);refs=root/'refs';public=make_refs(refs);data=[];remaining=12
            for r in public:
                rc=r['qa_type'].startswith('region_caption');count=0 if rc else 5 if remaining else 4
                if count==5:remaining-=1
                cs=[]
                for i in range(count):
                    c=self.candidate();c.update(claim_id=f'H{i+1}');cs.append(c)
                value={'planning_status':'UNRESOLVED_TARGET_REFERENCE' if rc else 'GENERATED','entities':[],
                       'claims':cs,'information_gaps':[{'code':'MISSING_PUBLIC_TARGET','description':'Public target absent.','requirement_ids':[]}] if rc else []}
                raw=json.dumps(value);parsed=v2.validate(raw,r)
                data.append({**{k:r[k] for k in ('sample_key','source_record_index','question','qa_type','requirements','question_sha256','requirements_sha256')},
                             **parsed,'format':v2.VERSION,'raw_response':raw,'image_reads':0,'input_modality':'TEXT_ONLY','synthetic_test':True,'call':None})
            source=root/'v2.jsonl';write(source,data,jsonl=True);before=digest(source)
            with patch('relive.hypothesis_graph.runner.make_backend',side_effect=AssertionError('MODEL_LOAD')),patch('PIL.Image.open',side_effect=AssertionError('IMAGE_READ')):
                z=run_diagnostic(input_path=source,output_dir=root/'diag')
                revalidate(input_path=source,output_dir=root/'v21',offline=True,no_model_calls=True)
                prepare(v2_input=source,v21_dir=root/'v21',output_dir=root/'public')
            self.assertEqual(z['raw_candidates'],84);self.assertEqual(z['new_model_calls'],0);self.assertFalse(z['publication_allowed'])
            self.assertEqual(digest(source),before);self.assertEqual(len(rows(root/'diag/v2_2_candidate_audit.jsonl')),84)
            self.assertEqual(sum(len(r['requirements']) for r in load_references(root/'public')[0]),100)


if __name__=='__main__':unittest.main()
