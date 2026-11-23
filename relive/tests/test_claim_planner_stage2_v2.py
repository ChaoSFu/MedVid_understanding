from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from relive.claim_bank import contracts as old, contracts_v11, contracts_v12
from relive.claim_bank.contracts import ClaimBankError, QA_TYPES
from relive.claim_bank.data import digest, load_json, project, rows, write_once
from relive.claim_planner import contract as v2
from relive.claim_planner.fixtures import claim, entity, payload, FixtureBackend
from relive.claim_planner.report import compare, metrics
from relive.claim_planner.runner import generate, validate_run
from relive.claim_planner.sources import freeze, load_references
from relive.config import load_config
from relive.storage.artifacts import stable_hash

ROOT = Path(__file__).resolve().parents[1]
CONFIG = str(ROOT / 'configs/claim_planner_stage2_v2_mock.yaml')
REQ = {"requirement_id": "R1", "description": "Determine whether a forceps is visible.", "type": "OBJECT", "evidence_type": "SPATIAL", "importance": "REQUIRED"}

class ContractTests(unittest.TestCase):
    def setUp(self):
        self.row = {'qa_type': 'stg', 'question': 'Track the forceps from 90.0 to 114.0 seconds, predicting its location every 8 seconds.', 'requirements': [REQ]}

    def evaluate(self, claims=None, **changes):
        value = {'planning_status': 'GENERATED', 'entities': [entity()], 'claims': [claim()] if claims is None else claims, 'information_gaps': []}
        value.update(changes)
        return v2.validate(json.dumps(value), self.row)

    def errors(self, result):
        return {e for r in result['rejected_candidates'] for e in r['errors']}

    def test_schema_roundtrip_and_stable_protocol(self):
        self.assertEqual(json.loads(json.dumps(v2.SCHEMA)), v2.SCHEMA)
        self.assertEqual(load_json(ROOT / 'schemas/claim_planner_stage2_v2.schema.json'), v2.SCHEMA)
        self.assertEqual(v2.protocol(), v2.protocol())
        p = v2.protocol()
        self.assertEqual(p.pop('protocol_sha256'), stable_hash(p))
        self.assertEqual(v2.shape_errors(json.loads(json.dumps({'planning_status': 'GENERATED', 'entities': [entity()], 'claims': [claim()], 'information_gaps': []})), v2.SCHEMA), [])

    def test_baseline_claim_is_unbound_and_unverified(self):
        r = self.evaluate()
        self.assertEqual(r['generation_status'], 'GENERATED')
        self.assertEqual(r['claims'][0]['status'], 'UNVERIFIED')
        self.assertEqual(r['claims'][0]['evidence_status'], 'NOT_SEARCHED')
        self.assertEqual(r['semantic_review_status'], 'REVIEW_REQUIRED')

    def test_partial_success_keeps_exact_valid_claim(self):
        good, bad = claim(), claim('H2')
        bad['status'] = 'SUPPORTED'
        r = self.evaluate([good, bad])
        self.assertEqual(r['generation_status'], 'PARTIAL_GENERATED')
        self.assertEqual(r['claims'], [good])
        self.assertEqual(r['rejected_candidates'][0]['raw_candidate'], bad)

    def test_malformed_individual_claim_does_not_discard_others(self):
        for bad in (None, 'broken', {}, {'claim_id': []}):
            with self.subTest(bad=bad):
                r = self.evaluate([claim(), bad])
                self.assertEqual(r['claims'], [claim()])
                self.assertEqual(r['generation_status'], 'PARTIAL_GENERATED')

    def test_invalid_unused_entity_quarantined_without_losing_valid_claim(self):
        r = self.evaluate(entities=[entity(), None])
        self.assertEqual(r['claims'], [claim()])
        self.assertEqual(r['entities'], [entity()])
        self.assertEqual(r['raw_entities'], [entity(), None])
        self.assertTrue(r['entity_audit'])

    def test_freeze_reference_bytes_deterministic(self):
        # Schema/protocol source is reproducibly serializable, independent of process hash order.
        self.assertEqual(stable_hash(v2.protocol()), stable_hash(json.loads(json.dumps(v2.protocol()))))

    def test_root_schema_and_json_do_not_salvage(self):
        for raw in ('```json\n{}\n```', '[{}]', '{}', '{', '{"claims":[],"claims":[]}'):
            with self.subTest(raw=raw):
                r = v2.validate(raw, self.row)
                self.assertEqual(r['claims'], [])
                self.assertTrue(r['root_errors'])

    def test_all_bindings_must_remain_null(self):
        for key in ('frame', 'timestamp', 'bbox'):
            c = claim()
            c['bindings'][key] = 5
            self.assertEqual(self.evaluate([c])['claims'], [])

    def test_requires_reference_does_not_propagate_truth(self):
        a, b = claim(), claim('H2', kind='ACTION', predicate='MOTION')
        b['text'] = 'The unresolved target moves upward within a bounded event.'
        b['evidence_requirement'].update(modality='MULTI_FRAME', temporal_requirement={'description': 'Compare movement in a bounded interval.', 'min_observations': 2})
        b['dependencies'] = [{'claim_id': 'H1', 'relation': 'REQUIRES', 'rationale': 'Identify target before assessing motion.'}]
        r = self.evaluate([a, b])
        self.assertEqual(len(r['claims']), 2)
        self.assertTrue(all(c['status'] == 'UNVERIFIED' for c in r['claims']))

    def test_dangling_rejected_dependency_pruned_recursively(self):
        a, b, c = claim(), claim('H2', kind='ATTRIBUTE', predicate='HAS_ATTRIBUTE'), claim('H3', kind='ACTION', predicate='MOTION')
        a['status'] = 'VERIFIED'
        b['text'] = 'The target has an unresolved attribute.'
        b['dependencies'] = [{'claim_id': 'H1', 'relation': 'REQUIRES', 'rationale': 'Target identity.'}]
        c['text'] = 'The target moves.'
        c['evidence_requirement'].update(modality='MULTI_FRAME', temporal_requirement={'description': 'Motion evidence.', 'min_observations': 2})
        c['dependencies'] = [{'claim_id': 'H2', 'relation': 'REQUIRES', 'rationale': 'Identify state.'}]
        r = self.evaluate([a, b, c])
        self.assertEqual(r['claims'], [])
        self.assertIn('DEPENDENCY_REJECTED', self.errors(r))

    def test_cycles_rejected_but_independent_claim_retained(self):
        a, b, c = claim(), claim('H2', kind='ATTRIBUTE', predicate='HAS_ATTRIBUTE'), claim('H3', kind='ACTION', predicate='MOTION')
        b['text'] = 'The target has an unresolved attribute.'
        a['dependencies'] = [{'claim_id': 'H2', 'relation': 'REQUIRES', 'rationale': 'Synthetic cycle.'}]
        b['dependencies'] = [{'claim_id': 'H1', 'relation': 'SUPPORTS', 'rationale': 'Synthetic cycle.'}]
        c['text'] = 'The target moves.'
        c['evidence_requirement'].update(modality='MULTI_FRAME', temporal_requirement={'description': 'Motion evidence.', 'min_observations': 2})
        r = self.evaluate([a, b, c])
        self.assertEqual([x['claim_id'] for x in r['claims']], ['H3'])
        self.assertIn('DEPENDENCY_CYCLE', self.errors(r))

    def test_duplicate_ids_rejected(self):
        self.assertIn('DUPLICATE_CLAIM_ID', self.errors(self.evaluate([claim(), claim()])))

    def test_invalid_first_duplicate_does_not_suppress_valid_independent_claim(self):
        a, b = claim(), claim('H2')
        a['dependencies'] = [{'claim_id': 'H1', 'relation': 'REQUIRES', 'rationale': 'Invalid self cycle.'}]
        r = self.evaluate([a, b])
        self.assertEqual(r['claims'], [b])
        self.assertEqual(r['generation_status'], 'PARTIAL_GENERATED')

    def test_identical_entity_aliases_do_not_bypass_duplicate_check(self):
        a, b = claim(), claim('H2')
        b['text'] = 'The same unresolved target is present in the input clip.'
        b['proposition']['subject'] = 'E2'
        b['evidence_requirement']['spatial_targets'] = ['E2']
        self.assertIn('DUPLICATE_SEMANTIC_PROPOSITION', self.errors(self.evaluate([a, b], entities=[entity(), entity('E2')])))

    def test_typed_alternatives_require_same_predicate_opposite_polarity(self):
        a, b = claim(), claim('H2')
        b['text'] = 'The target is not visible within the input clip.'
        b['proposition']['polarity'] = 'NEGATIVE'
        b['dependencies'] = [{'claim_id': 'H1', 'relation': 'ALTERNATIVE_TO', 'rationale': 'Opposite polarity of the same bounded proposition.'}]
        self.assertEqual(len(self.evaluate([a, b])['claims']), 2)
        b['proposition'].update(predicate='HAS_ATTRIBUTE', qualifier='$attribute')
        b['type'] = 'ATTRIBUTE'
        self.assertIn('ALTERNATIVE_NOT_ESTABLISHED_EXCLUSIVE', self.errors(self.evaluate([a, b])))

    def test_case02_global_stability_and_double_negation_rejected(self):
        self.row['qa_type'] = 'tal'
        for text, code in [('The camera view remains stable across all provided frames.', 'GLOBAL_SCOPE_ASSERTION_FORBIDDEN'),
                           ('Forceps engages tissue in every frame.', 'GLOBAL_SCOPE_ASSERTION_FORBIDDEN'),
                           ('No tissue actions are absent.', 'DOUBLE_NEGATION_FORBIDDEN')]:
            c = claim(); c['text'] = text
            self.assertIn(code, self.errors(self.evaluate([c])))
        self.assertIn('can co-occur', v2.prompt_for(self.row))

    def test_case04_parameterized_time_template_not_per_frame_duplicates(self):
        a = claim()
        a['proposition']['scope'].update(kind='QUERY_TIME_SET', query_time={'question_quote': self.row['question'], 'start_seconds': 90.0, 'end_seconds': 114.0, 'stride_seconds': 8})
        self.assertEqual(self.evaluate([a])['generation_status'], 'GENERATED')
        b = deepcopy(a); b['claim_id'] = 'H2'; b['text'] = 'The target is visible at another queried time.'
        b['proposition']['scope']['query_time']['start_seconds'] = 114.0
        self.assertIn('DUPLICATE_SEMANTIC_PROPOSITION', self.errors(self.evaluate([a, b])))
        a['text'] = 'A pen is visible in frame 8.'
        self.assertIn('FRAME_SPECIFIC_PLANNING_FORBIDDEN', self.errors(self.evaluate([a])))

    def test_query_time_must_use_public_seconds_not_frame_filenames(self):
        for start, end, stride, quote in [(529, 557, 4, '529 to 557'), (114, 90, 8, self.row['question']), (90, 114, 7, self.row['question'])]:
            c = claim()
            c['proposition']['scope'].update(kind='QUERY_TIME_SET', query_time={'question_quote': quote, 'start_seconds': start, 'end_seconds': end, 'stride_seconds': stride})
            self.assertIn('QUERY_TIME_NOT_PUBLIC_OR_CONFLICTING', self.errors(self.evaluate([c])))

    def test_case05_to08_missing_rc_target_no_guessed_scene_claim(self):
        for qa in ('region_caption_gpt', 'region_caption_gemini'):
            self.row['qa_type'] = qa
            result = v2.validate(json.dumps(payload(self.row)), self.row)
            self.assertEqual(result['generation_status'], 'UNRESOLVED_TARGET_REFERENCE')
            self.assertEqual(result['claims'], [])
            e = {'entity_id': 'E1', 'label': 'forceps', 'source': 'QUESTION', 'source_quote': 'forceps', 'requirement_id': None}
            result = self.evaluate(entities=[e], planning_status='UNRESOLVED_TARGET_REFERENCE', information_gaps=[{'code': 'MISSING', 'description': 'Query box absent.', 'requirement_ids': ['R1']}])
            self.assertIn('RC_TARGET_IDENTITY_UNAVAILABLE', self.errors(result))

    def test_case10_to12_explicit_predicate_chains_not_accepted(self):
        for text in ('The instrument grasps tissue then retracts it.', 'Cutting is followed by suturing.', 'The action transitions to suture placement.'):
            c = claim(); c['text'] = text
            self.assertIn('MULTIPLE_INDEPENDENT_PREDICATES', self.errors(self.evaluate([c])))

    def test_case14_prompt_preserves_hypothetical_future(self):
        self.row['qa_type'] = 'next_action'
        self.assertIn('Never assert a future step has happened', v2.prompt_for(self.row))
        self.assertNotIn('GT_SENTINEL', v2.prompt_for(self.row))

    def test_case17_to18_unsupported_concrete_entity_not_invented(self):
        self.row.update(qa_type='video_summary_gpt', question='Give an overview of the video.')
        bad = {'entity_id': 'E1', 'label': 'purple scalpel', 'source': 'QUESTION', 'source_quote': 'purple scalpel', 'requirement_id': None}
        self.assertIn('ENTITY_REFERENCE_OR_SPECIFICITY_INVALID', self.errors(self.evaluate(entities=[bad])))
        self.assertEqual(self.evaluate()['claims'][0]['proposition']['subject'], 'E1')

    def test_case19_to20_clinical_interpretation_needs_rule_and_visual_premise(self):
        self.row.update(qa_type='cvs_assessment', question='Assess whether the cystic plate is exposed.')
        c = claim('H2', kind='HIGH_LEVEL', predicate='INTERPRETATION')
        c['text'] = 'The candidate criterion is satisfied within the bounded scope.'
        c['evidence_requirement']['modality'] = 'COMPOSITIONAL'
        self.assertIn('INTERPRETATION_PREREQUISITES_REQUIRED', self.errors(self.evaluate([claim(), c])))
        c['interpretation_rule'] = {'description': 'Evaluate the criterion using acquired anatomical evidence.', 'question_quote': 'cystic plate is exposed'}
        c['dependencies'] = [{'claim_id': 'H1', 'relation': 'SUPPORTS', 'rationale': 'Partial evidence only; never an entailment.'}]
        self.assertIn('INTERPRETATION_NEEDS_MORE_THAN_PRESENCE', self.errors(self.evaluate([claim(), c])))
        base = claim(kind='RELATION', predicate='STATIC_RELATION')
        base['text'] = 'The candidate anatomy has the specified relation.'
        base['proposition']['object'] = 'E2'
        base['evidence_requirement'].update(modality='MULTI_REGION_SINGLE_FRAME', spatial_targets=['E1', 'E2'])
        r = self.evaluate([base, c], entities=[entity(), entity('E2', '$second_structure')])
        self.assertEqual(len(r['claims']), 2)
        self.assertEqual(r['claims'][1]['status'], 'UNVERIFIED')
        a = deepcopy(base); a['dependencies'] = [{'claim_id': 'H2', 'relation': 'REQUIRES', 'rationale': 'Invalid interpretation-as-fact.'}]
        self.assertIn('UNVERIFIED_INTERPRETATION_AS_PREMISE', self.errors(self.evaluate([a, c], entities=[entity(), entity('E2', '$second_structure')])))

    def test_case21_to22_skill_interpretation_requires_temporal_evidence(self):
        self.row.update(qa_type='skill_assessment', question='Assess instrument handling quality.')
        c = claim('H2', kind='HIGH_LEVEL', predicate='INTERPRETATION')
        c.update(text='The handling meets the candidate quality criterion.', interpretation_rule={'description': 'Assess the question criterion using acquired movement evidence.', 'question_quote': 'instrument handling quality'})
        c['dependencies'] = [{'claim_id': 'H1', 'relation': 'SUPPORTS', 'rationale': 'Conditional evidence only.'}]
        c['evidence_requirement']['modality'] = 'COMPOSITIONAL'
        self.assertIn('SKILL_INTERPRETATION_NEEDS_TEMPORAL_PREMISE', self.errors(self.evaluate([claim(), c])))

    def test_action_single_frame_forbidden(self):
        c = claim(kind='ACTION', predicate='MOTION')
        self.assertIn('ACTION_REQUIRES_TEMPORAL_CONTRACT', self.errors(self.evaluate([c])))

    def test_multiple_structured_predicates_rejected(self):
        c = claim(); c['proposition']['predicate'] = ['IS_VISIBLE', 'CONTACT']
        self.assertEqual(self.evaluate([c])['claims'], [])

    def test_object_label_does_not_hide_held_and_positioned(self):
        c = claim(); c['text'] = 'A pen is held by a hand and positioned near an incision.'
        errors = self.errors(self.evaluate([c]))
        self.assertIn('MULTIPLE_INDEPENDENT_PREDICATES', errors)
        self.assertIn('TEXT_TYPE_COMPOUND_MISMATCH', errors)

    def test_language_semantics_not_claimed_as_objective_quality(self):
        c = claim(); c['text'] = 'An instrument is visible, suggesting a possible procedure.'
        r = self.evaluate([c])
        self.assertEqual(r['generation_status'], 'GENERATED')
        self.assertIn('ATOMICITY_OR_INFERENCE_REVIEW', r['claim_audit'][0]['semantic_risk_warnings'])
        self.assertEqual(r['semantic_review_status'], 'REVIEW_REQUIRED')

class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runtime = self.root / 'runtime_manifest.jsonl'
        public, legacy = [], []
        for i, index in enumerate(v2.SMOKE22):
            qa = QA_TYPES[i // 2]
            record = {'id': 'duplicated', 'qa_type': qa, 'dataset_name': 'FIXTURE', 'data_source': 'FIXTURE',
                      'video': ['/unavailable/GT_PATH_SENTINEL.jpg'], 'sampled_video_frames': [0],
                      'metadata': {'cvs_scores': 'GT_SENTINEL'}, 'struc_info': 'GT_SENTINEL', 'RC_info': 'GT_SENTINEL',
                      'conversations': [{'from': 'human', 'value': 'Inspect the question-named candidate event.'}, {'from': 'gpt', 'value': 'GT_SENTINEL'}]}
            r = project(record, index, 'a' * 64)
            public.append(r)
            reqs = [{**REQ, 'requirement_id': f'R{n + 1}', 'description': f'Determine whether candidate condition {n + 1} holds.'} for n in range(5 if i < 12 else 4)]
            legacy.append({**{k: r[k] for k in ('sample_key', 'source_record_index', 'question', 'qa_type')}, 'requirements': reqs,
                           'claims': [], 'generation_status': 'MODEL_OR_SCHEMA_FAILURE', 'failure_reasons': ['OLD_FAILURE'],
                           'raw_responses': [{'raw_text': 'OLD_CLAIM_SENTINEL GT_SENTINEL', 'execution_status': 'OK'}]})
        write_once(self.runtime, public, jsonl=True)
        write_once(self.runtime.with_suffix('.provenance.json'), {'runtime_sha256': digest(self.runtime), 'row_count': 22, 'dataset_sha256': 'a' * 64})
        self.diag = self.root / 'diagnostic.json'
        write_once(self.diag, {'stage2': {'samples': legacy}})
        self.refs = self.root / 'references'
        freeze(runtime_manifest=str(self.runtime), legacy_diagnostic=str(self.diag), output_dir=str(self.refs), allow_stage2_requirements=True)

    def run_fixture(self, name, **extra):
        backend = FixtureBackend(load_config(CONFIG)['backend'])
        summary = generate(references_dir=str(self.refs), config_path=CONFIG, cache_dir=str(self.root / 'cache'), output_dir=str(self.root / name), _backend=backend, **extra)
        return summary, backend

    def test_fixed_22_and_exact_100_frozen_requirements(self):
        data, old_rows, manifest = load_references(self.refs, include_legacy=True)
        self.assertEqual([r['source_record_index'] for r in data], v2.SMOKE22)
        self.assertEqual(sum(len(r['requirements']) for r in data), 100)
        self.assertEqual([r['requirements'] for r in data], [r['requirements'] for r in old_rows])
        self.assertFalse(manifest['source']['independent_stage1_bank_checked'])

    def test_fallback_must_be_explicit(self):
        with self.assertRaisesRegex(ClaimBankError, 'STAGE1_REQUIRED_OR_EXPLICIT_FALLBACK'):
            freeze(runtime_manifest=str(self.runtime), legacy_diagnostic=str(self.diag), output_dir=str(self.root / 'new'))

    def test_frozen_references_byte_identical_in_new_directory(self):
        other = self.root / 'references_repeat'
        freeze(runtime_manifest=str(self.runtime), legacy_diagnostic=str(self.diag), output_dir=str(other), allow_stage2_requirements=True)
        for name in ('frozen_requirements.jsonl', 'legacy_comparison.jsonl', 'reference_audit.json', 'manifest.json'):
            self.assertEqual((self.refs / name).read_bytes(), (other / name).read_bytes())

    def make_stage1_bank(self):
        from relive.backends.mock import MockBackend
        from relive.claim_bank.pipeline import generate as legacy_generate
        legacy_rows = load_json(self.diag)['stage2']['samples']
        rules = [{'match': {'stage': 'claimgen_stage1', 'qa_type': r['qa_type']},
                  'response': {'planning_status': 'PLANNED', 'planning_reason': None, 'requirements': r['requirements']}}
                 for r in legacy_rows[::2]]
        legacy_generate(runtime_manifest=str(self.runtime), stage=1, output_dir=str(self.root / 'stage1_fixture'),
                        config_path=str(ROOT / 'configs/claim_bank_mock_v11.yaml'), cache_dir=str(self.root / 'stage1_fixture_cache'),
                        protocol_version='v1.1', _backend=MockBackend({'kind': 'mock', 'rules': rules, 'max_retries': 0}))
        return self.root / 'stage1_fixture'

    def test_accessible_stage1_bank_exactly_compared_and_bound(self):
        bank = self.make_stage1_bank()
        before = {p: digest(p) for p in bank.rglob('*.json*')}
        freeze(runtime_manifest=str(self.runtime), legacy_diagnostic=str(self.diag), stage1_dir=str(bank), output_dir=str(self.root / 'independent_refs'))
        _, _, m = load_references(self.root / 'independent_refs')
        self.assertTrue(m['source']['independent_stage1_bank_checked'])
        self.assertEqual(m['source']['source_sha256'], digest(bank / 'bank_manifest.json'))
        self.assertEqual(before, {p: digest(p) for p in before})

    def test_accessible_stage1_mismatch_cannot_use_fallback_to_bypass(self):
        bank = self.make_stage1_bank()
        diag = load_json(self.diag)
        diag['stage2']['samples'][0]['requirements'][0]['description'] = 'A changed planning suggestion.'
        altered = self.root / 'changed_legacy.json'; write_once(altered, diag)
        with self.assertRaisesRegex(ClaimBankError, 'STAGE1_STAGE2_REQUIREMENTS_MISMATCH'):
            freeze(runtime_manifest=str(self.runtime), legacy_diagnostic=str(altered), stage1_dir=str(bank),
                   output_dir=str(self.root / 'mismatch_refs'), allow_stage2_requirements=True)

    def test_changed_selection_question_or_count_fail_closed(self):
        for kind in ('index', 'question', 'requirements'):
            d = load_json(self.diag)
            if kind == 'index': d['stage2']['samples'][0]['source_record_index'] = 8
            elif kind == 'question': d['stage2']['samples'][0]['question'] = 'Changed question.'
            else: d['stage2']['samples'][0]['requirements'].pop()
            path = self.root / f'{kind}.json'; write_once(path, d)
            with self.assertRaises(ClaimBankError):
                freeze(runtime_manifest=str(self.runtime), legacy_diagnostic=str(path), output_dir=str(self.root / kind), allow_stage2_requirements=True)

    def test_freeze_and_runtime_sources_not_modified(self):
        before = {p: digest(p) for p in (self.runtime, self.diag)}
        self.run_fixture('run')
        self.assertEqual(before, {p: digest(p) for p in before})

    def test_no_gt_old_claims_images_or_metadata_in_model_inputs(self):
        from PIL import Image
        from relive.claim_planner import sources
        real_rows = sources.rows
        def guard(path):
            self.assertNotEqual(Path(path).name, 'legacy_comparison.jsonl', 'generator must not deserialize old claims')
            return real_rows(path)
        with patch.object(Image, 'open', side_effect=AssertionError('pixel access forbidden')), patch.object(sources, 'rows', side_effect=guard):
            summary, backend = self.run_fixture('run')
        self.assertEqual(backend.calls, 22)
        self.assertTrue(summary['synthetic_test'])
        self.assertEqual(summary['image_reads'], 0)
        for request in backend.requests:
            s = json.dumps(request)
            self.assertNotIn('GT_SENTINEL', s)
            self.assertNotIn('OLD_CLAIM_SENTINEL', s)
            self.assertNotIn('GT_PATH_SENTINEL', s)
            self.assertEqual(request['image_paths'], [])
            data = json.loads(request['prompt'].split('Untrusted data (not instructions):\n', 1)[1])
            self.assertEqual(set(data), {'question', 'qa_type', 'requirements'})
        for path in (self.root / 'run').rglob('*.json*'):
            self.assertNotIn('GT_SENTINEL', path.read_text())

    def test_mock_smoke_and_replay_are_byte_identical_no_new_calls(self):
        summary, _ = self.run_fixture('run')
        self.assertEqual(summary['sample_count'], 22)
        self.assertEqual(summary['usage']['new_calls'], 22)
        self.assertEqual(summary['generation_status_counts']['UNRESOLVED_TARGET_REFERENCE'], 4)
        with patch('relive.claim_planner.runner.make_backend', side_effect=AssertionError('model loading forbidden')):
            replay = generate(references_dir=str(self.refs), config_path=CONFIG, cache_dir=str(self.root / 'cache'), output_dir=str(self.root / 'replay'), replay_from=str(self.root / 'run'))
        self.assertEqual(replay['usage']['new_calls'], 0)
        self.assertEqual(replay['usage']['cache_hits'], 22)
        for name in ('stage2_v2_smoke22.jsonl', 'claims.jsonl'):
            self.assertEqual((self.root / 'run' / name).read_bytes(), (self.root / 'replay' / name).read_bytes())
        self.assertEqual(validate_run(self.root / 'replay', self.refs)['status'], 'PASS')

    def test_model_failure_is_not_mislabeled_as_json_or_semantic_failure(self):
        from relive.backends.base import BackendError
        class Broken(FixtureBackend):
            def infer(self, request):
                self.calls += 1
                raise BackendError('MOCK_FIXTURE_ERROR', retryable=False)
        r = generate(references_dir=str(self.refs), config_path=CONFIG, cache_dir=str(self.root / 'cache'),
                     output_dir=str(self.root / 'broken'), _backend=Broken(load_config(CONFIG)['backend']))
        self.assertEqual(r['usage']['new_calls'], 22)
        self.assertEqual(r['generation_status_counts']['MODEL_OR_SCHEMA_FAILURE'], 22)
        sample = rows(self.root / 'broken' / 'stage2_v2_smoke22.jsonl')[0]
        self.assertEqual(sample['root_error_category'], 'MODEL')
        self.assertEqual(sample['root_errors'], ['MODEL_EXECUTION_FAILURE'])
        self.assertEqual(sample['claims'], [])

    def test_replay_miss_stops_without_backend_call(self):
        self.run_fixture('run')
        with self.assertRaisesRegex(ClaimBankError, 'REPLAY_CACHE_MISS'):
            generate(references_dir=str(self.refs), config_path=CONFIG, cache_dir=str(self.root / 'empty_cache'), output_dir=str(self.root / 'replay'), replay_from=str(self.root / 'run'))

    def test_dry_run_no_backend_no_cache(self):
        with patch('relive.claim_planner.runner.make_backend', side_effect=AssertionError('no model')):
            r = generate(references_dir=str(self.refs), config_path=CONFIG, cache_dir=str(self.root / 'no_cache'), output_dir=str(self.root / 'dry'), dry_run=True)
        self.assertEqual(r['usage']['new_calls'], 0)
        self.assertFalse((self.root / 'no_cache').exists())
        self.assertEqual(validate_run(self.root / 'dry', self.refs)['status'], 'PASS')

    def test_synthetic_or_unknown_source_cannot_be_presented_as_real(self):
        with patch('relive.claim_planner.runner.make_backend', side_effect=AssertionError('no model loading')):
            with self.assertRaisesRegex(ClaimBankError, 'SYNTHETIC_OR_UNKNOWN_REQUIREMENTS_FOR_REAL_RUN'):
                generate(references_dir=str(self.refs), config_path=str(ROOT / 'configs/qwen35_9b_claim_planner_stage2_v2.yaml'),
                         cache_dir=str(self.root / 'cache'), output_dir=str(self.root / 'real'), dry_run=True)

    def test_resume_complete_no_new_calls_or_overwrite(self):
        first, _ = self.run_fixture('run')
        before = digest(self.root / 'run' / 'manifest.json')
        second = generate(references_dir=str(self.refs), config_path=CONFIG, cache_dir=str(self.root / 'cache'), output_dir=str(self.root / 'run'), resume=True)
        self.assertEqual(first, second)
        self.assertEqual(digest(self.root / 'run' / 'manifest.json'), before)
        with self.assertRaisesRegex(ClaimBankError, 'NEW_OUTPUT_DIRECTORY_REQUIRED'):
            self.run_fixture('run')

    def test_resume_cannot_touch_existing_unrelated_directory(self):
        unrelated = self.root / 'historical_output'
        unrelated.mkdir()
        (unrelated / 'frozen.json').write_text('{"historical":true}')
        before = {p.name: digest(p) for p in unrelated.iterdir()}
        with self.assertRaisesRegex(ClaimBankError, 'RESUME_BINDING_REQUIRED'):
            generate(references_dir=str(self.refs), config_path=CONFIG, cache_dir=str(self.root / 'cache'),
                     output_dir=str(unrelated), resume=True)
        self.assertEqual(before, {p.name: digest(p) for p in unrelated.iterdir()})

    def test_validation_detects_hash_tamper(self):
        self.run_fixture('run')
        (self.root / 'run' / 'claims.jsonl').write_text('[]\n')
        with self.assertRaisesRegex(ClaimBankError, 'ARTIFACT_HASH_MISMATCH'):
            validate_run(self.root / 'run', self.refs)

    def test_resume_from_interrupted_checkpoint_counts_remaining_calls(self):
        class Interrupted(FixtureBackend):
            def infer(self, request):
                if self.calls == 2:
                    raise KeyboardInterrupt('Synthetic interruption')
                return super().infer(request)
        with self.assertRaises(KeyboardInterrupt):
            generate(references_dir=str(self.refs), config_path=CONFIG, cache_dir=str(self.root / 'cache'),
                     output_dir=str(self.root / 'run'), _backend=Interrupted(load_config(CONFIG)['backend']))
        summary, _ = self.run_fixture('run', resume=True)
        self.assertEqual(summary['usage']['new_calls'], 20)
        audit = load_json(self.root / 'run' / 'cache_replay_audit.json')
        self.assertEqual(audit['resumed_checkpoint_count'], 2)
        self.assertEqual(validate_run(self.root / 'run', self.refs)['status'], 'PASS')

    def test_report_all22_and_synthetic_label_and_raw_quarantine(self):
        self.run_fixture('run')
        result = compare(references_dir=str(self.refs), run_dir=str(self.root / 'run'), output_dir=str(self.root / 'report'))
        self.assertEqual(result['status'], 'SYNTHETIC_COMPARISON_ONLY')
        body = (self.root / 'report' / 'stage2_v1_vs_v2_smoke22.html').read_text()
        self.assertEqual(body.count('<section id="case'), 22)
        self.assertIn('Rejected Raw Candidates', body)
        self.assertIn('SYNTHETIC FIXTURE ONLY', body)
        self.assertIn('REVIEW_REQUIRED', body)
        measured = load_json(self.root / 'report' / 'stage2_v2_metrics.json')
        self.assertIsNone(measured['question_relevance']['value'])
        self.assertIsNone(measured['unsupported_specificity']['value'])
        self.assertEqual(len(rows(self.root / 'report' / 'semantic_review_queue.jsonl')), 22)

    def test_original_protocol_hashes_unchanged(self):
        self.assertEqual(old.protocol()['protocol_sha256'], 'f0dbde883dbd5811ab44aabb05986e3c5dd98332e703bdfa8cc15dc3279e4e79')
        self.assertEqual(contracts_v11.protocol()['protocol_sha256'], '139d890cf6586d848467926984a4ce84ff8d91908965eab76aca80b64815f2d5')
        self.assertEqual(contracts_v12.protocol()['protocol_sha256'], 'b7570f36b95dc8c519912ede28a08605bf63a4d73669809be1ce1153583e74a5')

if __name__ == '__main__':
    unittest.main()
