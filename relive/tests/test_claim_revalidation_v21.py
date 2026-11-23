from __future__ import annotations

import ast
from copy import deepcopy
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

from relive.claim_planner import contract as v2
from relive.claim_revalidation.normalization import normalize, public_time_queries
from relive.claim_revalidation.validation import SCHEMA, policy, validate_sample
from relive.claim_revalidation.offline import RevalidationError, digest, revalidate, spatial_category
from relive.storage.artifacts import canonical_json, stable_hash

ROOT = Path(__file__).resolve().parents[1]
REQ = {'requirement_id':'R1','description':'Determine whether the marker is visible.','type':'OBJECT','evidence_type':'SPATIAL','importance':'REQUIRED'}

def entity():
    return {'entity_id':'E1','label':'marker','source':'QUESTION','source_quote':'marker','requirement_id':'R1'}

def claim(cid='H1', qualifier=None):
    return {'claim_id':cid,'text':'The marker is visible.','type':'OBJECT','status':'UNVERIFIED','requirement_ids':['R1'],
            'answer_relevance':'REQUIRED','verification_purpose':'Test the question-named object presence.',
            'proposition':{'predicate':'IS_VISIBLE','subject':'E1','object':None,'qualifier':qualifier,'polarity':'POSITIVE',
                           'scope':{'kind':'BOUNDED_EVENT_SEARCH','description':'Search the bounded input clip.',
                                    'query_time':{'start_seconds':None,'end_seconds':None,'stride_seconds':None,'question_quote':'Is the marker visible?'}}},
            'evidence_requirement':{'modality':'SINGLE_FRAME','spatial_targets':['E1'],'temporal_requirement':None},
            'dependencies':[],'interpretation_rule':None,'bindings':{'frame':None,'timestamp':None,'bbox':None},'evidence_status':'NOT_SEARCHED'}

def response(claims=None, entities=None, status='GENERATED'):
    return {'planning_status':status,'entities':[entity()] if entities is None else entities,
            'claims':[claim()] if claims is None else claims,'information_gaps':[]}

class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.row={'question':'Is the marker visible?','qa_type':'stg','requirements':[deepcopy(REQ)]}

    def evaluate(self, value=None):
        return validate_sample(response() if value is None else value,self.row)

    def record(self, value=None):
        return self.evaluate(value)['candidate_records'][0]

    def test_A_null_time_object_equivalent_unbound_search(self):
        original=response(); result=self.evaluate(original)
        r=result['candidate_records'][0]
        self.assertEqual(r['state'],'ACCEPTED')
        self.assertEqual(r['normalized_candidate']['proposition']['scope']['kind'],'SEARCH_INPUT_CLIP')
        self.assertIsNone(r['normalized_candidate']['proposition']['scope']['query_time'])
        self.assertIsNone(original['claims'][0]['proposition']['scope']['query_time']['start_seconds'])
        self.assertTrue(all(n['semantic_preserving'] for n in result['normalizations']))

    def test_A_invented_zero_or_stride_not_erased(self):
        for start,stride in [(0,None),(None,1),(0,1),(None,0.5)]:
            c=claim(); q=c['proposition']['scope']['query_time']; q.update(start_seconds=start,stride_seconds=stride)
            r=self.record(response([c]))
            self.assertEqual(r['state'],'REJECTED')
            self.assertEqual(r['normalized_candidate']['proposition']['scope']['query_time'],q)
            self.assertIn('QUERY_TIME_NOT_PUBLIC_OR_CONFLICTING',r['contract_errors'])

    def test_B_explicit_public_time_is_preserved(self):
        self.row['question']='Track the marker from 90.0 to 114.0 seconds, predicting its location every 8 seconds.'
        c=claim(); c['proposition']['scope'].update(kind='QUERY_TIME_SET',query_time={'start_seconds':90.0,'end_seconds':114.0,'stride_seconds':8,'question_quote':self.row['question']})
        r=self.record(response([c]))
        self.assertEqual(r['state'],'ACCEPTED')
        self.assertEqual(r['normalized_candidate']['proposition']['scope']['query_time'],c['proposition']['scope']['query_time'])

    def test_B_wrong_citation_only_corrected_when_numbers_match_question(self):
        self.row['question']='Track the marker from 90.0 to 114.0 seconds, sampled every 8 seconds.'
        c=claim(); c['proposition']['scope'].update(kind='QUERY_TIME_SET',query_time={'start_seconds':90,'end_seconds':114,'stride_seconds':8,'question_quote':'A requirement sentence that is not in the question.'})
        r=self.evaluate(response([c]))
        self.assertEqual(r['candidate_records'][0]['state'],'ACCEPTED')
        changes=[n for n in r['normalizations'] if n['rule_id']=='PUBLIC_TIME_QUOTE_CORRECTION']
        self.assertEqual(len(changes),1)
        self.assertEqual(changes[0]['after'],self.row['question'])

    def test_C_wrong_time_or_stride_stays_rejected(self):
        self.row['question']='Track the marker from 90 to 114 seconds every 8 seconds.'
        for start,end,stride in [(91,114,8),(90,115,8),(90,114,4),(114,90,8)]:
            c=claim(); c['proposition']['scope'].update(kind='QUERY_TIME_SET',query_time={'start_seconds':start,'end_seconds':end,'stride_seconds':stride,'question_quote':self.row['question']})
            self.assertIn('QUERY_TIME_NOT_PUBLIC_OR_CONFLICTING',self.record(response([c]))['contract_errors'])

    def test_requirement_times_do_not_authorize_question_time(self):
        self.row['requirements'][0]['description']='Inspect the marker from 90 to 114 seconds every 8 seconds.'
        c=claim(); c['proposition']['scope']['query_time']={'start_seconds':90,'end_seconds':114,'stride_seconds':8,'question_quote':self.row['requirements'][0]['description']}
        self.assertIn('QUERY_TIME_NOT_PUBLIC_OR_CONFLICTING',self.record(response([c]))['contract_errors'])

    def test_scores_and_frame_numbers_are_not_time_ranges(self):
        for q in ('Score the marker 0, 1, or 2.','Frames 90 through 114; fps 8.','Video id 90 and annotation 114.'):
            self.assertEqual(public_time_queries(q),[])

    def test_multiple_time_queries_not_guessed(self):
        self.row['question']='Inspect from 1 to 2 seconds and from 3 to 4 seconds.'
        c=claim(); c['proposition']['scope']['query_time']={'start_seconds':1,'end_seconds':2,'stride_seconds':None,'question_quote':self.row['question']}
        self.assertIn('QUERY_TIME_NOT_PUBLIC_OR_CONFLICTING',self.record(response([c]))['contract_errors'])

    def test_provable_requirement_source_correction(self):
        e=entity(); e.update(source_quote='Determine whether the marker is visible.')
        result=self.evaluate(response(entities=[e]))
        self.assertEqual(result['normalized_response']['entities'][0]['source'],'REQUIREMENT')
        self.assertEqual(result['candidate_records'][0]['state'],'ACCEPTED')
        self.assertTrue(any(c['rule_id']=='PROVENANCE_SOURCE_CORRECTION' for c in result['normalizations']))

    def test_question_citation_keeps_requirement_context(self):
        e=entity(); result=self.evaluate(response(entities=[e]))
        self.assertEqual(result['normalized_response']['entities'][0],e)
        self.assertEqual(result['candidate_records'][0]['state'],'ACCEPTED')

    def test_untraceable_provenance_is_not_repaired(self):
        e=entity(); e['source_quote']='GT invented marker description'
        result=self.record(response(entities=[e]))
        self.assertIn('ENTITY_PROVENANCE_INVALID',result['contract_errors'])

    def test_ambiguous_requirement_citation_not_selected_by_first_match(self):
        self.row['requirements']=[deepcopy(REQ),{**REQ,'requirement_id':'R2'}]
        e=entity(); e.update(source_quote=REQ['description'],requirement_id=None)
        result=self.evaluate(response(entities=[e]))
        self.assertEqual(result['normalized_response']['entities'][0]['source'],'QUESTION')
        self.assertEqual(result['candidate_records'][0]['state'],'REJECTED')

    def test_no_medical_alias_or_extra_attribute_guess(self):
        for label in ('purple marker','surgical marker','marker view'):
            e=entity(); e['label']=label
            r=self.record(response(entities=[e]))
            self.assertEqual(r['state'],'REVIEW_REQUIRED')
            self.assertIn('ENTITY_PROVENANCE_REVIEW_REQUIRED',r['blocking_semantic_reviews'])

    def test_separators_are_orthographic_only(self):
        self.row['question']='Is the test marker visible?'
        e=entity();e.update(label='test_marker',source_quote='test marker')
        c=claim();c['text']='The test marker is visible.';c['proposition']['scope']['query_time']['question_quote']=self.row['question']
        result=self.evaluate(response([c],[e]))
        self.assertEqual(result['normalized_response']['entities'][0]['label'],'test marker')
        self.assertEqual(result['candidate_records'][0]['state'],'ACCEPTED')

    def test_domain_knowledge_without_real_source_never_admitted(self):
        e=entity();e['source']='DOMAIN_KNOWLEDGE'
        self.assertEqual(self.record(response(entities=[e]))['state'],'REVIEW_REQUIRED')

    def test_unresolved_variable_remains_unresolved(self):
        e=entity();e.update(label='target_object',source='UNRESOLVED',source_quote=None)
        result=self.evaluate(response(entities=[e]))
        self.assertEqual(result['normalized_response']['entities'][0]['label'],'$target_object')
        self.assertEqual(result['normalized_response']['entities'][0]['source'],'UNRESOLVED')

    def test_static_min_one_is_retained_not_changed_to_two(self):
        c=claim();c['evidence_requirement']['temporal_requirement']={'description':'Inspect one frame.','min_observations':1}
        r=self.record(response([c]))
        self.assertEqual(r['state'],'ACCEPTED')
        self.assertEqual(r['normalized_candidate']['evidence_requirement'],c['evidence_requirement'])

    def test_zero_observations_never_usable(self):
        c=claim();c['evidence_requirement']['temporal_requirement']={'description':'No observations.','min_observations':0}
        r=self.record(response([c]))
        self.assertIn('NONPOSITIVE_OBSERVATION_REQUIREMENT',r['contract_errors'])
        self.assertEqual(r['state'],'REJECTED')

    def test_dynamic_action_not_downgraded_to_single_frame(self):
        c=claim();c.update(type='ACTION',text='The marker moves upward.');c['proposition']['predicate']='MOTION'
        r=self.record(response([c]))
        self.assertIn('ACTION_REQUIRES_TEMPORAL_CONTRACT',r['contract_errors'])
        self.assertEqual(r['normalized_candidate']['evidence_requirement']['modality'],'SINGLE_FRAME')

    def test_multi_frame_action_can_be_admitted_as_unverified(self):
        c=claim();c.update(type='ACTION',text='The marker moves upward.');c['proposition']['predicate']='MOTION'
        c['evidence_requirement'].update(modality='MULTI_FRAME',temporal_requirement={'description':'Observe motion.','min_observations':2})
        r=self.record(response([c]));self.assertEqual(r['state'],'ACCEPTED');self.assertEqual(r['normalized_candidate']['status'],'UNVERIFIED')
        self.assertIsNone(spatial_category(r))

    def test_static_modality_alias_and_unique_label_reference(self):
        e2={**entity(),'entity_id':'E2','label':'target','source_quote':'target'}
        self.row['question']='Is the marker next to the target?'
        c=claim();c.update(type='RELATION',text='The marker is next to the target.');c['proposition'].update(predicate='STATIC_RELATION',object='E2')
        c['proposition']['scope']['query_time']['question_quote']=self.row['question']
        c['evidence_requirement'].update(modality='SPATIAL_MULTI_REGION',spatial_targets=['marker','target'])
        result=self.evaluate(response([c],[entity(),e2]));r=result['candidate_records'][0]
        self.assertEqual(r['state'],'ACCEPTED');self.assertEqual(spatial_category(r),'STATIC_RELATION')
        self.assertEqual(r['normalized_candidate']['evidence_requirement']['spatial_targets'],['E1','E2'])
        self.assertEqual(r['raw_candidate']['evidence_requirement']['modality'],'SPATIAL_MULTI_REGION')

    def test_partial_acceptance_and_dependency_blocking(self):
        a=claim();b=claim('H2','B');b['text']='The marker is purple.';b['proposition']['scope']['query_time']['stride_seconds']=1
        c=claim('H3','C');c['dependencies']=[{'claim_id':'H2','relation':'REQUIRES','rationale':'Need H2 before H3.'}]
        records=self.evaluate(response([a,b,c]))['candidate_records']
        self.assertEqual([r['state'] for r in records],['ACCEPTED','REJECTED','BLOCKED_BY_DEPENDENCY'])
        self.assertEqual(records[2]['normalized_candidate']['dependencies'],c['dependencies'])

    def test_malformed_candidate_does_not_discard_independent_legal_claim(self):
        for bad in (None, {'claim_id':'H2','text':'Malformed.'}, {**claim('H2'),'proposition':{**claim()['proposition'],'predicate':['IS_VISIBLE']}}):
            with self.subTest(bad=bad):
                result=self.evaluate(response([claim(),bad]))
                self.assertEqual(len(result['candidate_records']),2)
                self.assertEqual(result['candidate_records'][0]['state'],'ACCEPTED')
                self.assertEqual(result['candidate_records'][1]['state'],'REJECTED')

    def test_cycles_do_not_remove_independent_valid_node(self):
        a,b,c=claim(),claim('H2','B'),claim('H3','C')
        a['dependencies']=[{'claim_id':'H2','relation':'REQUIRES','rationale':'Cycle.'}]
        b['dependencies']=[{'claim_id':'H1','relation':'SUPPORTS','rationale':'Cycle.'}]
        records=self.evaluate(response([a,b,c]))['candidate_records']
        self.assertEqual([r['state'] for r in records],['REJECTED','REJECTED','ACCEPTED'])
        self.assertIn('DEPENDENCY_CYCLE',records[0]['dependency_errors'])

    def test_invalid_dependency_not_deleted_or_renumbered(self):
        c=claim();c['dependencies']=[{'claim_id':'H6','relation':'REQUIRES','rationale':'Missing node.'}]
        r=self.record(response([c]));self.assertIn('DEPENDENCY_REFERENCE_INVALID',r['dependency_errors'])
        self.assertEqual(r['normalized_candidate']['dependencies'],c['dependencies'])

    def test_compound_predicate_is_not_rewritten_for_admission(self):
        c=claim();c.update(type='ACTION',text='The marker moves upward followed by a downward motion.');c['proposition']['predicate']='MOTION'
        r=self.record(response([c]));self.assertIn('MULTIPLE_INDEPENDENT_PREDICATES',r['contract_errors']);self.assertEqual(r['normalized_candidate']['text'],c['text'])

    def test_independent_layers_run_even_after_schema_failure(self):
        c=claim();c['proposition']['scope']['query_time']['start_seconds']=0
        c['text']='The marker is visible in every frame.'
        r=self.record(response([c]));self.assertTrue(r['schema_errors']);self.assertIn('GLOBAL_SCOPE_ASSERTION_FORBIDDEN',r['contract_errors'])

    def test_spatial_eligibility_depends_on_contract_not_type_name(self):
        a=self.record();self.assertEqual(spatial_category(a),'OBJECT')
        c=claim();c['evidence_requirement'].update(modality='MULTI_FRAME',temporal_requirement={'description':'Several queried frames.','min_observations':2})
        b=self.record(response([c]));self.assertEqual(b['state'],'ACCEPTED');self.assertIsNone(spatial_category(b))

    def test_v2_contract_and_prompt_hash_frozen(self):
        self.assertEqual(v2.protocol()['protocol_sha256'],'9303524f3010b466877be2f06d47d5ccebbdad4e3672b9e4837b68b3a1527ae0')
        self.assertEqual(policy(),policy());self.assertEqual(json.loads(json.dumps(SCHEMA)),SCHEMA)
        self.assertEqual(json.loads((ROOT/'schemas/claim_revalidation_stage2_v2_1.schema.json').read_text()),SCHEMA)

class OfflineIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.path=self.root/'input.jsonl'
        counts=[3,3,3,3,0,0,0,0,6,5,6,4,4,4,6,6,6,6,4,5,6,4]
        data=[]
        for i,(index,n) in enumerate(zip(v2.SMOKE22,counts)):
            reqs=[{**REQ,'requirement_id':f'R{j+1}','description':f'Determine marker condition {j+1}.'} for j in range(5 if i<12 else 4)]
            source={'sample_key':f'medvidu-{"a"*64}-{index:06d}','source_record_index':index,'qa_type':'region_caption_gpt' if n==0 else 'stg',
                    'question':'Is the marker visible?','requirements':reqs,'question_sha256':stable_hash('Is the marker visible?'),
                    'requirements_sha256':stable_hash(reqs),'format':v2.VERSION,'image_reads':0,'input_modality':'TEXT_ONLY','synthetic_test':True,'call':None}
            claims=[claim(f'H{j+1}',f'condition_{j+1}') for j in range(n)]
            for j,c in enumerate(claims): c['text']=f'The marker is visible under candidate condition {j+1}.'
            raw=response(claims,[],status='UNRESOLVED_TARGET_REFERENCE') if not n else response(claims)
            if not n: raw['information_gaps']=[{'code':'MISSING_PUBLIC_BOX','description':'Target unresolved.','requirement_ids':['R1']}]
            source['raw_response']=json.dumps(raw)
            old=v2.validate(source['raw_response'],source)
            source.update(old)
            data.append(source)
        self.data=data;self.save(data)

    def save(self,data):
        self.path.write_text(''.join(canonical_json(r)+'\n' for r in data))

    def run_offline(self,name='out',**kw):
        return revalidate(input_path=str(self.path),output_dir=str(self.root/name),offline=True,no_model_calls=True,**kw)

    def test_complete22_84_100_zero_call_and_hash_invariants(self):
        before=digest(self.path)
        with patch.object(socket,'socket',side_effect=AssertionError('network forbidden')):
            result=self.run_offline()
        self.assertEqual(result['sample_count'],22);self.assertEqual(result['raw_candidate_count'],84)
        self.assertEqual(result['accepted_claim_count'],84);self.assertEqual(result['new_model_calls'],0)
        self.assertEqual(result['image_reads'],0);self.assertFalse(result['gt_used']);self.assertEqual(result['certificate_writes'],0)
        self.assertEqual(digest(self.path),before)
        out=[json.loads(s) for s in (self.root/'out/stage2_v2_1_revalidated.jsonl').read_text().splitlines()]
        self.assertEqual(sum(len(s['requirements']) for s in out),100)
        self.assertTrue(all(s['requirements_sha256']==r['requirements_sha256'] and s['raw_response']==r['raw_response'] for s,r in zip(out,self.data)))
        self.assertTrue(all(r['raw_candidate']['text']==r['normalized_candidate']['text'] for s in out for r in s['candidate_records']))
        self.assertTrue(all(r['raw_candidate']['dependencies']==r['normalized_candidate']['dependencies'] for s in out for r in s['candidate_records']))

    def test_four_rc_stay_unresolved_and_empty(self):
        result=self.run_offline();self.assertEqual(result['unresolved_rc_sample_count'],4)

    def test_offline_replay_has_identical_artifact_bytes(self):
        self.run_offline('first');r=self.run_offline('second',replay_from=str(self.root/'first'))
        self.assertEqual(r['new_model_calls'],0)
        for name in ('stage2_v2_1_revalidated.jsonl','stage2_v2_1_claims.jsonl','stage2_v2_1_metrics.json','stage2_v2_vs_v2_1.html'):
            self.assertEqual((self.root/'first'/name).read_bytes(),(self.root/'second'/name).read_bytes())
        receipt=json.loads((self.root/'second/offline_revalidation_audit.json').read_text())
        self.assertEqual(receipt['exact_byte_replay'],'PASS')

    def test_missing_zero_call_flags_fail_before_work(self):
        with self.assertRaisesRegex(RevalidationError,'EXPLICIT_OFFLINE_ZERO_CALL_FLAGS_REQUIRED'):
            revalidate(input_path=str(self.path),output_dir=str(self.root/'out'),offline=False,no_model_calls=True)
        self.assertFalse((self.root/'out').exists())

    def test_explicit_source_hash_cannot_be_bypassed(self):
        with self.assertRaisesRegex(RevalidationError,'FROZEN_SOURCE_SHA_MISMATCH'):
            self.run_offline(expected_input_sha256='0'*64)
        self.assertFalse((self.root/'out').exists())

    def test_online_or_generation_flags_absent(self):
        import subprocess, sys
        r=subprocess.run([sys.executable,'-m','relive.claim_revalidation','revalidate','--input',str(self.path),
                          '--output-dir',str(self.root/'out'),'--offline','--no-model-calls','--model','Qwen'],capture_output=True,text=True)
        self.assertEqual(r.returncode,2);self.assertFalse((self.root/'out').exists())

    def test_nongraph_cache_reference_cannot_open_an_image_or_qa_file(self):
        data=deepcopy(self.data);data[0]['call']={'raw_response_ref':str(self.root/'image.png'),'cache_key':'a'*64};self.save(data)
        with self.assertRaisesRegex(RevalidationError,'RAW_CACHE_REFERENCE_INVALID'):self.run_offline()

    def test_hash_or_selection_or_count_mismatch_fail_closed(self):
        for kind in ('hash','selection','count'):
            data=deepcopy(self.data)
            if kind=='hash':data[0]['requirements'][0]['description']='Changed requirement.'
            elif kind=='selection':data.pop()
            else:
                raw=json.loads(data[0]['raw_response']);raw['claims'].pop();data[0]['raw_response']=json.dumps(raw);data[0].update(v2.validate(data[0]['raw_response'],data[0]))
            self.save(data)
            with self.assertRaises(RevalidationError):self.run_offline(kind)

    def test_gt_fields_are_rejected_not_consumed(self):
        data=deepcopy(self.data);data[0]['struc_info']='FORBIDDEN_GT_SENTINEL';self.save(data)
        with self.assertRaisesRegex(RevalidationError,'GT_FIELDS_FORBIDDEN'):self.run_offline()

    def test_existing_output_not_overwritten(self):
        self.run_offline();before=digest(self.root/'out/manifest.json')
        with self.assertRaisesRegex(RevalidationError,'NEW_OUTPUT_DIRECTORY_REQUIRED'):self.run_offline()
        self.assertEqual(digest(self.root/'out/manifest.json'),before)

    def test_explicit_cache_miss_no_fallback_inference(self):
        with self.assertRaises(RevalidationError):self.run_offline(cache_root=str(self.root/'missing_cache'))

    def test_source_cache_read_only_and_binding_match(self):
        data=deepcopy(self.data);row=data[0]
        identity={'request':{'planner_binding':{k:row[k] for k in ('question_sha256','requirements_sha256')}}}
        record={'identity':identity,'cache_key':stable_hash(identity),'raw_text':row['raw_response']}
        cache=self.root/'source_cache'/f'{stable_hash(record)}.json';cache.parent.mkdir();cache.write_text(canonical_json(record))
        row['call']={'raw_response_ref':str(cache),'cache_key':record['cache_key']};self.save(data)
        before=digest(cache);self.run_offline();self.assertEqual(digest(cache),before)
        receipt=json.loads((self.root/'out/offline_revalidation_audit.json').read_text());self.assertEqual(receipt['cache_matches'],1)

    def test_mismatched_accessible_cache_fail_closed(self):
        data=deepcopy(self.data);row=data[0]
        record={'identity':{},'cache_key':'a'*64,'raw_text':'different completion'}
        cache=self.root/f'{stable_hash(record)}.json';cache.write_text(canonical_json(record))
        row['call']={'raw_response_ref':str(cache),'cache_key':'a'*64};self.save(data)
        with self.assertRaisesRegex(RevalidationError,'RAW_CACHE_BINDING_MISMATCH'):self.run_offline()

    def test_import_graph_has_no_runtime_or_image_dependency(self):
        forbidden={'backends','inference','cache','torch','PIL','openai','verification','certificates','adaptation'}
        for p in (ROOT/'src/relive/claim_revalidation').glob('*.py'):
            tree=ast.parse(p.read_text(),feature_version=(3,10))
            for n in ast.walk(tree):
                names=[a.name for a in n.names] if isinstance(n,ast.Import) else [n.module or ''] if isinstance(n,ast.ImportFrom) else []
                self.assertFalse(any(set(name.split('.'))&forbidden for name in names),(p,names))

    def test_reports_cover_all_84_and_distinguish_status_layers(self):
        self.run_offline()
        body=(self.root/'out/stage2_v2_vs_v2_1.html').read_text()
        self.assertEqual(body.count('<section id="case'),22);self.assertEqual(body.count('<article>'),84)
        self.assertIn('Structural admission',body);self.assertIn('Contract validity',body);self.assertIn('Semantic review',body)
        self.assertEqual(json.loads((self.root/'out/stage2_v2_1_rejection_audit.json').read_text())['candidate_count'],84)

if __name__=='__main__':unittest.main()
