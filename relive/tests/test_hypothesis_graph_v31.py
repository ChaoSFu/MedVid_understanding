"""V3.1 regressions: preserved real failures plus synthetic transport only."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from relive.hypothesis_graph import contract as v3
from relive.hypothesis_graph.validation import validate as validate_v3
from relive.hypothesis_graph.compiler import policy as v3_compiler_policy
from relive.hypothesis_graph.fixtures import FixtureBackend
from relive.hypothesis_graph_v31.contract import SCHEMA, VERSION, prompt_for, protocol
from relive.hypothesis_graph_v31.predicates import analyze, numeric_audit
from relive.hypothesis_graph_v31.validation import validate
from relive.hypothesis_graph_v31.runner import run
from relive.hypothesis_graph_v31.sources import rows, load_run, digest, write
from relive.hypothesis_graph_v31.compiler import compile_node, compile_bank, review_rows
from relive.hypothesis_graph_v31.report import diagnose_v3, compare
from relive.storage.artifacts import stable_hash
from test_hypothesis_graph_v3 import claim, graph, make_refs, ROW

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/hypothesis_graph_stage2_v3_1_mock.yaml'
FIXTURE = ROOT / 'tests/fixtures/hypothesis_graph_v3_failure_responses.json'


def evaluate(payload, row=None):
    return validate(json.dumps(payload), row or ROW)


def compiled(c, others=None, review=None):
    c = {**c, 'status': 'UNVERIFIED', 'evidence_status': 'NOT_SEARCHED'}
    return compile_node(c, {c['claim_id']: c, **(others or {})}, review)


class InstanceContractTests(unittest.TestCase):
    def test_schema_and_protocol_stable(self):
        self.assertEqual(SCHEMA, json.loads((ROOT / 'schemas/hypothesis_graph_stage2_v3_1.schema.json').read_text()))
        p = protocol()
        self.assertEqual(p.pop('protocol_sha256'), stable_hash(p))
        self.assertEqual(SCHEMA['properties'], v3.SCHEMA['properties'])

    def test_model_template_and_generation_config_unchanged(self):
        from relive.config import load_config
        old = load_config(ROOT / 'configs/qwen35_9b_hypothesis_graph_stage2_v3.yaml')
        new = load_config(ROOT / 'configs/qwen35_9b_hypothesis_graph_stage2_v3_1.yaml')
        self.assertEqual(old['backend'], new['backend'])
        self.assertEqual(old['policy'], new['policy'])

    def test_prompt_uses_instance_not_schema_definition(self):
        prompt = prompt_for(ROW)
        self.assertNotIn('"properties":', prompt)
        self.assertNotIn('"$schema":', prompt)
        self.assertNotIn('Return exactly this closed minimal schema', prompt)
        self.assertIn('RESULT INSTANCE', prompt)
        self.assertIn('"claims": [', prompt)
        self.assertIn('do not reduce an action to mere contact', prompt)

    def test_prompt_input_allowlist_and_gt_sentinels(self):
        polluted = {**ROW, 'struc_info': 'GT_SENTINEL', 'RC_info': 'GT_SENTINEL',
                    'claims': ['OLD_CLAIM_SENTINEL'], 'oracle': 'GT_SENTINEL'}
        p = prompt_for(polluted)
        self.assertNotIn('GT_SENTINEL', p)
        self.assertNotIn('OLD_CLAIM_SENTINEL', p)

    def test_schema_echo_and_wrapper_rejected_not_unwrapped(self):
        for value, code in [
            (SCHEMA, 'ROOT_SCHEMA_DEFINITION_INSTEAD_OF_INSTANCE'),
            ({'properties': graph()}, 'ROOT_INSTANCE_WRAPPED_IN_SCHEMA'),
        ]:
            r = evaluate(value)
            self.assertEqual(r['root_errors'], [code])
            self.assertEqual(r['claims'], [])
            self.assertEqual(r['claim_audit'], [])

    def test_duplicate_keys_and_fence_not_repaired(self):
        self.assertEqual(validate('{"claims":[],"claims":[]}', ROW)['root_errors'], ['DUPLICATE_JSON_KEY'])
        raw = chr(96) * 3 + 'json\n{}\n' + chr(96) * 3
        self.assertEqual(validate(raw, ROW)['root_errors'], ['STRICT_JSON_PARSE_FAILURE'])

    def test_extra_fields_and_program_truth_constants(self):
        for field, value in [('bbox', [0, 0, 1, 1]), ('status', 'VERIFIED'), ('time', 1)]:
            c = claim()
            c[field] = value
            self.assertFalse(evaluate(graph([c]))['claims'])
        c = evaluate(graph())['claims'][0]
        self.assertEqual(c['status'], 'UNVERIFIED')
        self.assertEqual(c['evidence_status'], 'NOT_SEARCHED')

    def test_numeric_ids_not_timestamps(self):
        a = numeric_audit('The object identified in H1 moves.', ['H1'], ['R1'])
        self.assertEqual(a['numeric_literals'], [])
        self.assertEqual(a['errors'], [])
        cs = [claim(), claim('H2', 'The object identified in H1 moves.', 'TEMPORAL_EVENT', requires=['H1'])]
        a = evaluate(graph(cs))['claim_audit'][1]
        self.assertNotIn('NUMERIC_EXECUTION_BINDING_FORBIDDEN', a['errors'])
        self.assertIn('TEXT_REFERENCE_BINDING_REVIEW', a['risks'])

    def test_real_numeric_bindings_still_rejected(self):
        for text in [
            'The object is visible at 84.0 seconds.', 'The forceps is at frame 12.',
            'The object moves at time -1.5.', 'The interval ends at 1e3 seconds.',
            'The ROI is [0, 20, 200, 400].',
        ]:
            with self.subTest(text=text):
                self.assertIn('NUMERIC_EXECUTION_BINDING_FORBIDDEN', numeric_audit(text)['errors'])
                self.assertFalse(evaluate(graph([claim(text=text)]))['claims'])

    def test_unknown_text_id_not_silently_allowed(self):
        a = evaluate(graph([claim(text='The object identified in H9 moves.', kind='TEMPORAL_EVENT')]))['claim_audit'][0]
        self.assertIn('TEXT_CLAIM_REFERENCE_UNKNOWN', a['errors'])

    def test_rc_abstract_anaphor_not_identity_guess(self):
        row = {**ROW, 'qa_type': 'region_caption_gpt'}
        cs = [
            claim(text='The queried target object is visible.'),
            claim('H2', 'The object identified in H1 moves.', 'TEMPORAL_EVENT', requires=['H1']),
        ]
        r = evaluate(graph(cs, plan_status='UNRESOLVED_TARGET_REFERENCE', unresolved_references=['Public box absent.']), row)
        self.assertEqual(len(r['claims']), 2)
        self.assertTrue(all(a['state'] == 'REVIEW_REQUIRED' for a in r['claim_audit']))
        self.assertTrue(all('RC_CONCRETE_TARGET_BINDING_FORBIDDEN' not in a['errors'] for a in r['claim_audit']))
        self.assertEqual(compile_node(r['claims'][0], {c['claim_id']: c for c in r['claims']})['compilation_status'], 'UNCOMPILABLE')

    def test_rc_named_identity_and_resolved_status_blocked(self):
        row = {**ROW, 'qa_type': 'region_caption_gpt'}
        r = evaluate(graph(plan_status='UNRESOLVED_TARGET_REFERENCE', unresolved_references=['Box absent']), row)
        self.assertIn('RC_CONCRETE_TARGET_BINDING_FORBIDDEN', r['claim_audit'][0]['errors'])
        self.assertFalse(evaluate(graph(), row)['claims'])

    def test_static_contact_not_object_and_not_cutting(self):
        text = 'Surgical instruments are in direct contact with biological tissue.'
        a = evaluate(graph([claim(text=text)]))['claim_audit'][0]
        self.assertIn('TYPE_PREDICATE_CONFLICT', a['errors'])
        r = evaluate(graph([claim(text=text, kind='STATIC_RELATION')]))
        self.assertEqual(r['claims'][0]['type'], 'STATIC_RELATION')
        self.assertIn('CONTACT_ALONE_DOES_NOT_ESTABLISH_ACTION', r['claim_audit'][0]['risks'])

    def test_tautology_and_future_subset_flagged(self):
        for text, flag in [
            ('The current phase is either complete or ongoing.', 'LOW_DISCRIMINATION_OR_TAUTOLOGY_REVIEW'),
            ("The next phase following the current phase is 'closure'.", 'PROSPECTIVE_HYPOTHESIS_REVIEW'),
        ]:
            r = evaluate(graph([claim(text=text, kind='HIGH_LEVEL', role='INTERPRETATION')]))
            self.assertIn(flag, r['claim_audit'][0]['risks'])
            self.assertEqual(r['claims'][0]['status'], 'UNVERIFIED')

    def test_bad_dependency_dag_and_partial_pruning(self):
        self.assertFalse(evaluate(graph([claim(requires=['H2']), claim('H2', requires=['H1'])]))['claims'])
        cs = [
            claim(text='The object is visible at time 10 seconds.'),
            claim('H2', 'The forceps moves tissue.', 'TEMPORAL_EVENT', requires=['H1']),
            claim('H3', 'The needle is visible.'),
        ]
        r = evaluate(graph(cs))
        self.assertEqual([c['claim_id'] for c in r['claims']], ['H3'])
        self.assertEqual(r['claim_audit'][1]['state'], 'BLOCKED_BY_DEPENDENCY')

    def test_alternative_pruning_and_mutual_exclusivity_review(self):
        r = evaluate(graph([claim(alternative_to=['H2']), claim('H2', 'The needle is visible.')]))
        self.assertIn('ALTERNATIVE_EXCLUSIVITY_REVIEW', r['claim_audit'][0]['risks'])
        cs = [claim(alternative_to=['H2']), claim('H2', 'The needle is visible at 8 seconds.')]
        self.assertFalse(evaluate(graph(cs))['claims'])

    def test_malformed_id_and_empty_planned_graph_fail_closed(self):
        c = claim()
        c['claim_id'] = []
        self.assertFalse(evaluate(graph([c]))['claims'])
        self.assertTrue(evaluate(graph([]))['root_errors'])

    def test_parameter_duplicates_do_not_change_text(self):
        cs = [claim(text='A forceps is visible at time t_i.'), claim('H2', 'A forceps is visible at time t_j.')]
        before = deepcopy(cs)
        r = evaluate(graph(cs))
        self.assertEqual(cs, before)
        self.assertEqual(len(r['claims']), 1)
        self.assertIn('DUPLICATE_PROPOSITION', r['claim_audit'][1]['errors'])


class PredicateCompilerTests(unittest.TestCase):
    def test_progressive_and_plural_motion_forms(self):
        for text in [
            'The forceps moves tissue.', 'The forceps is moving tissue.',
            'The instruments are currently actively retracting tissue.',
            'The knife is dissecting tissue.',
        ]:
            z = compiled(claim(text=text, kind='TEMPORAL_EVENT'))
            self.assertEqual(z['compilation_status'], 'COMPILED', z)
            self.assertEqual(z['evidence_modality'], 'MULTI_FRAME')
            self.assertEqual(z['minimum_observations'], 2)
            self.assertEqual(z['claim_text'], text)

    def test_static_contact_and_state_variations(self):
        z = compiled(claim(text='The forceps are in direct contact with tissue.', kind='STATIC_RELATION'))
        self.assertEqual(z['evidence_modality'], 'MULTI_REGION_SINGLE_FRAME')
        self.assertEqual(z['spatial_targets'], ['The forceps', 'tissue', 'contact_interface'])
        self.assertEqual(compiled(claim(text='The needle can be seen.'))['evidence_modality'], 'SINGLE_FRAME')
        self.assertEqual(compiled(claim(text='The needle is visible within the image.'))['evidence_modality'], 'SINGLE_FRAME')
        self.assertEqual(compiled(claim(text='The forceps jaws are closed.', kind='ATTRIBUTE'))['discriminative_attribute'], 'closed')

    def test_exact_source_spans_and_no_invented_bindings(self):
        text = '  The forceps is moving tissue. '
        parsed = analyze(text)
        for a in parsed['arguments']:
            self.assertEqual(text[slice(*a['span'])], a['text'])
        self.assertEqual(text[slice(*parsed['predicate_span'])], parsed['predicate'])
        c = claim(text=text, kind='TEMPORAL_EVENT')
        before = deepcopy(c)
        z = compiled(c)
        self.assertEqual(c, before)
        self.assertEqual(z['candidate_frames'], [])
        self.assertEqual(z['time_bindings'], {})
        self.assertEqual(z['spatial_regions'], [])

    def test_interpretations_not_converted_to_motion(self):
        for text in [
            'The movements are efficient.', 'The steps proceed with logical continuity.',
            'The forceps moves tissue then cuts it.',
            'The holder is manipulating a thread passing through tissue.',
            'The needle is visible. It moves tissue.',
            'It moves tissue.',
        ]:
            self.assertEqual(compiled(claim(text=text, kind='TEMPORAL_EVENT'))['compilation_status'], 'UNCOMPILABLE')

    def test_compiler_type_conflict_and_unknown_stay_uncompilable(self):
        for c in [claim(text='The forceps are in contact with tissue.'), claim(text='Something is appropriately positioned.')]:
            self.assertEqual(compiled(c)['compilation_status'], 'UNCOMPILABLE')

    def test_negation_and_modality_never_silently_compile_positive(self):
        for text in ['The knife does not cut tissue.', 'The knife can cut tissue.',
                     'The forceps moves tissue causing bleeding.']:
            z = compiled(claim(text=text, kind='TEMPORAL_EVENT'))
            self.assertEqual(z['compilation_status'], 'UNCOMPILABLE')
            self.assertEqual(z['claim_text'], text)

    def test_symbolic_parameter_and_unknown_target(self):
        z = compiled(claim(text='The needle is visible at time t.'))
        self.assertEqual(z['source_phrase_analysis']['time_parameter'], 't')
        self.assertEqual(z['time_bindings'], {})
        self.assertEqual(compiled(claim(text='The target object is visible.'))['compilation_status'], 'UNCOMPILABLE')

    def test_high_level_rules_and_human_inference_unchanged(self):
        c = claim('H2', 'The handling meets the criterion.', 'HIGH_LEVEL', role='INTERPRETATION', requires=['H1'])
        prereq = {'H1': claim(text='The forceps moves tissue.', kind='TEMPORAL_EVENT')}
        self.assertEqual(compiled(c, prereq)['reason'], 'EXPLICIT_INFERENCE_REQUIREMENT_REQUIRED')
        z = compiled(c, prereq, {'inference_requirement': 'Check motion against the explicit question criterion.'})
        self.assertEqual(z['evidence_modality'], 'COMPOSITIONAL')
        self.assertFalse(z['prerequisites_imply_claim'])
        self.assertEqual(compiled(c, {'H1': claim()})['reason'], 'PRESENCE_ALONE_CANNOT_ESTABLISH_INTERPRETATION')


class RealFailureRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(FIXTURE.read_text())
        cls.samples = {r['source_record_index']: r for r in cls.data['samples']}

    def test_original_v3_protocol_and_compiler_frozen(self):
        self.assertEqual(v3.protocol()['protocol_sha256'], 'e05155db77ffa49630130e8fc5d05b6ce854df953faad4af3c978d229df4c1a0')
        self.assertEqual(v3_compiler_policy()['policy_sha256'], 'e0903c920c515e52cfcdeb7ff2f9c41b80c9ccc64cabc43342031abb6e449e1c')

    def test_actual_schema_echo_wrapper_and_duplicate_failures(self):
        expected = {
            0: 'ROOT_SCHEMA_DEFINITION_INSTEAD_OF_INSTANCE',
            1: 'DUPLICATE_JSON_KEY', 1700: 'ROOT_INSTANCE_WRAPPED_IN_SCHEMA',
            2651: 'ROOT_INSTANCE_WRAPPED_IN_SCHEMA',
        }
        for index, code in expected.items():
            row = self.samples[index]
            r = validate(row['raw_response'], row)
            self.assertEqual(r['root_errors'], [code])
            self.assertFalse(r['claims'])

    def test_actual_rc_causal_claim_id_is_not_time_or_identity_guess(self):
        row = self.samples[2481]
        r = validate(row['raw_response'], row)
        audit = r['claim_audit'][3]
        self.assertNotIn('NUMERIC_EXECUTION_BINDING_FORBIDDEN', audit['errors'])
        self.assertNotIn('RC_CONCRETE_TARGET_BINDING_FORBIDDEN', audit['errors'])
        self.assertEqual(audit['state'], 'BLOCKED_BY_DEPENDENCY')
        self.assertIn('NUMERIC_EXECUTION_BINDING_FORBIDDEN', r['claim_audit'][0]['errors'])

    def test_actual_skill_contact_type_and_downstream_blocking(self):
        row = self.samples[6121]
        r = validate(row['raw_response'], row)
        self.assertIn('TYPE_PREDICATE_CONFLICT', r['claim_audit'][0]['errors'])
        self.assertEqual(r['claim_audit'][1]['state'], 'BLOCKED_BY_DEPENDENCY')

    def test_actual_next_action_tautology_and_prospection(self):
        row = self.samples[4271]
        r = validate(row['raw_response'], row)
        self.assertIn('LOW_DISCRIMINATION_OR_TAUTOLOGY_REVIEW', r['claim_audit'][1]['risks'])
        self.assertIn('PROSPECTIVE_HYPOTHESIS_REVIEW', r['claim_audit'][3]['risks'])

    def test_fixture_and_legacy_results_not_modified(self):
        before = digest(FIXTURE)
        for row in self.samples.values():
            old = validate_v3(row['raw_response'], row)
            self.assertEqual(old['root_errors'], row['root_errors'])
            validate(row['raw_response'], row)
        self.assertEqual(digest(FIXTURE), before)


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.refs = self.root / 'refs'
        make_refs(self.refs)
        self.cache = self.root / 'cache'
        self.out = self.root / 'run'

    def execute(self, **kw):
        return run(references_dir=self.refs, config_path=CONFIG, cache_dir=self.cache,
                   output_dir=kw.pop('output_dir', self.out), **kw)

    def test_mock22_replay_exact_separate_cache_and_no_images(self):
        old = self.cache / v3.VERSION
        old.mkdir(parents=True)
        sentinel = old / 'unchanged.json'
        sentinel.write_text('{"history":true}')
        before = digest(sentinel)
        with patch('PIL.Image.open', side_effect=AssertionError('IMAGE_READ')):
            fresh = self.execute(_backend=FixtureBackend({'kind': 'mock'}))
            with patch('relive.hypothesis_graph_v31.runner.make_backend', side_effect=AssertionError('MODEL_LOAD')):
                replay = self.execute(output_dir=self.root / 'replay', replay_from=self.out)
        self.assertEqual(fresh['new_model_calls'], 22)
        self.assertEqual(replay['new_model_calls'], 0)
        self.assertEqual(replay['usage']['cache_hits'], 22)
        self.assertEqual((self.out / 'stage2_v3_1_smoke22.jsonl').read_bytes(),
                         (self.root / 'replay/stage2_v3_1_smoke22.jsonl').read_bytes())
        self.assertEqual(digest(sentinel), before)
        self.assertTrue((self.cache / VERSION).is_dir())
        self.assertTrue(fresh['synthetic_test'])
        self.assertEqual(fresh['requirement_count'], 100)

    def test_malformed_responses_preserved_without_retry(self):
        class Echo(FixtureBackend):
            def infer(self, request):
                self.calls += 1
                return json.dumps(SCHEMA)
        b = Echo({'kind': 'mock'})
        r = self.execute(_backend=b)
        self.assertEqual(b.calls, 22)
        self.assertEqual(r['root_error_counts'], {'ROOT_SCHEMA_DEFINITION_INSTEAD_OF_INSTANCE': 22})
        self.assertTrue(all(r['raw_response'] == json.dumps(SCHEMA) for r in rows(self.out / 'stage2_v3_1_smoke22.jsonl')))

    def test_resume_reuses_completed_checkpoints(self):
        class Interrupt(FixtureBackend):
            def infer(self, request):
                if self.calls == 3:
                    raise KeyboardInterrupt()
                return super().infer(request)
        with self.assertRaises(KeyboardInterrupt):
            self.execute(_backend=Interrupt({'kind': 'mock'}))
        r = self.execute(_backend=FixtureBackend({'kind': 'mock'}), resume=True)
        self.assertEqual(r['new_model_calls'], 19)
        self.assertEqual(len(rows(self.out / 'stage2_v3_1_smoke22.jsonl')), 22)

    def test_dry_run_zero_calls_and_no_cache(self):
        r = self.execute(dry_run=True)
        self.assertEqual(r['new_model_calls'], 0)
        self.assertFalse(self.cache.exists())
        self.assertFalse(r['certificate_writes'])

    def test_compiler_publication_and_reviews_remain_fail_closed(self):
        self.execute(_backend=FixtureBackend({'kind': 'mock'}))
        rr = review_rows(self.out, self.refs)
        for r in rr:
            r.update(decision='APPROVE', reviewer_id='TEST_ONLY', rationale='Engineering test, not medical approval.')
        write(self.root / 'review.jsonl', rr, jsonl=True)
        with patch('PIL.Image.open', side_effect=AssertionError('IMAGE_READ')):
            audit = compile_bank(run_dir=self.out, references_dir=self.refs, output_dir=self.root / 'compiler',
                                 review_jsonl=self.root / 'review.jsonl', publish=True)
        self.assertFalse(audit['publication_allowed'])
        self.assertIn('SYNTHETIC_OR_UNEXECUTED_PUBLICATION_FORBIDDEN', audit['gate_reasons'])
        self.assertEqual((self.root / 'compiler/evidence_contracts.jsonl').read_bytes(), b'')
        self.assertEqual(audit['new_model_calls'], 0)

    def test_checkpoint_tampering_and_replay_cache_miss(self):
        self.execute(_backend=FixtureBackend({'kind': 'mock'}))
        with self.assertRaises(ValueError):
            run(references_dir=self.refs, config_path=CONFIG, cache_dir=self.root / 'missing',
                output_dir=self.root / 'replay', replay_from=self.out)
        next((self.out / 'samples').iterdir()).write_text('{}')
        with self.assertRaises((ValueError, KeyError)):
            load_run(self.out, self.refs)

    def test_comparison_outputs_mark_mock_and_all22(self):
        self.execute(_backend=FixtureBackend({'kind': 'mock'}))
        compare(run_dir=self.out, references_dir=self.refs, output_dir=self.root / 'comparison')
        text = (self.root / 'comparison/stage2_v3_1_comparison.html').read_text()
        self.assertIn('SYNTHETIC TEST ONLY', text)
        self.assertEqual(text.count('<article>'), 22)

    def test_diagnostic_preserves_history_and_does_not_publish(self):
        source = self.root / 'diagnosis.json'
        write(source, json.loads(FIXTURE.read_text()))
        before = digest(source)
        r = diagnose_v3(diagnosis_json=source, output_dir=self.root / 'diagnostic')
        self.assertEqual(r['root_shape_counts'], {
            'SCHEMA_DEFINITION': 1, 'DUPLICATE_JSON_KEYS': 1,
            'WRAPPED_INSTANCE': 2, 'RESULT_INSTANCE': 3,
        })
        self.assertEqual(digest(source), before)
        self.assertEqual(r['new_model_calls'], 0)
        self.assertTrue(all(not v['publication_allowed'] for v in rows(self.root / 'diagnostic/v3_failure_diagnostic.jsonl')))


if __name__ == '__main__':
    unittest.main()
