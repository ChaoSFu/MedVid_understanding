"""Synthetic contract/transport tests. No benchmark expected answers or real model."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from relive.backends.base import BackendError
from relive.claim_revalidation.offline import write, read_json, digest
from relive.hypothesis_graph.sources import load_references
from relive.stage2_lite_ablation.gate import validate as old_gate
from relive.stage2_single_pass_v41.contract import (MODEL_SCHEMA, NORMALIZED_SCHEMA, protocol, prompt_for,
                                                  event_labels, max_claims, SMOKE22)
from relive.stage2_single_pass_v41.gate import validate
from relive.stage2_single_pass_v41.normalization import normalize
from relive.stage2_single_pass_v41.runner import run, implementation_hash
from relive.stage2_single_pass_v41.storage import verify
from relive.stage2_single_pass_v41.sources import load_inputs
from relive.stage2_single_pass_v41.fixtures import MockAdapter
from relive.storage.artifacts import stable_hash
from test_stage2_lite_ablation import sources as a_sources

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/stage2_v41_mock.yaml'
REQ = {'requirement_id': 'R1', 'description': 'Inspect the queried object.', 'type': 'OBJECT', 'evidence_type': 'SPATIAL', 'importance': 'REQUIRED'}
ROW = {'question': 'Is the queried object visible?', 'qa_type': 'stg', 'requirements': [REQ]}


def claim(cid='H1', text='The queried object is visible at time t.', kind='OBJECT', **kw):
    return {'claim_id': cid, 'text': text, 'type': kind, 'role': 'TARGET', 'requirement_ids': ['R1'],
            'requires': [], 'alternative_to': [], **kw}


def graph(cs=None, **kw):
    return {'plan_status': 'PLANNED', 'claims': [claim()] if cs is None else cs,
            'unresolved_slots': [], 'requirement_issues': [], **kw}


def slot(): return {'slot_id': 'S1', 'purpose': 'Identify the main observed task-relevant action.', 'status': 'UNBOUND'}


def make_inputs(root):
    refs, a = a_sources(root)
    public, _ = load_references(refs)
    baseline_a = read_json(a)['samples']; records = []
    for row, old in zip(public, baseline_a):
        rc = row['qa_type'].startswith('region_caption')
        value = graph([] if rc else [claim()], plan_status='UNRESOLVED_TARGET_REFERENCE' if rc else 'PLANNED',
                      unresolved_slots=[slot()] if rc else [])
        raw = json.dumps(value)
        records.append({**row, 'A': old, 'B': {'raw_response': raw, 'gate': old_gate(raw, row),
                       'generation_metadata': {'status': 'SYNTHETIC_NOT_MEASURED'}, 'execution_status': 'OK',
                       'recorded_latency_seconds': 0.0}, 'C': None})
    b = root / 'paired_B0.jsonl'; write(b, records, jsonl=True)
    return refs, a, b


class ContractTests(unittest.TestCase):
    def test_schema_and_version_hash(self):
        saved = read_json(ROOT / 'schemas/stage2_single_pass_v41.schema.json')
        self.assertEqual(saved, {'model': MODEL_SCHEMA, 'canonical': NORMALIZED_SCHEMA})
        p = protocol(); h = p.pop('protocol_sha256'); self.assertEqual(h, stable_hash(p))
        self.assertEqual(p['max_calls_per_sample'], 1); self.assertFalse(p['repair_enabled'])

    def test_prompt_instance_has_every_fixed_field(self):
        p = prompt_for(ROW)
        for text in ('"requires": []', '"alternative_to": []', '"purpose":', '"status": "UNBOUND"'):
            self.assertIn(text, p)
        self.assertNotIn('"properties":', p); self.assertNotIn('"$schema":', p)
        self.assertIn('not an assertion of verified truth', p)

    def test_allowlist_sentinels_and_no_old_claims(self):
        row = {**ROW, 'RC_info': 'GT_SENTINEL', 'struc_info': 'GT_SENTINEL', 'video': 'IMAGE_SENTINEL',
               'conversations': [{'from': 'gpt', 'value': 'GT_SENTINEL'}], 'claims': ['OLD_SENTINEL']}
        p = prompt_for(row)
        for s in ('GT_SENTINEL', 'IMAGE_SENTINEL', 'OLD_SENTINEL'): self.assertNotIn(s, p)

    def test_missing_arrays_audited_without_text_change(self):
        c = claim(); c.pop('requires'); c.pop('alternative_to'); before = deepcopy(c)
        r = validate(graph([c]), ROW)
        self.assertEqual(c, before)
        self.assertEqual(r['claims'][0]['text'], c['text'])
        self.assertEqual(r['claims'][0]['requires'], [])
        self.assertEqual(len(r['normalizations']), 2)
        self.assertIn('DEPENDENCY_NOT_DECLARED_REVIEW', r['claim_audit'][0]['risks'])
        self.assertTrue(r['raw_schema_errors']); self.assertFalse(r['claim_audit'][0]['errors'])

    def test_existing_edges_never_deleted_or_defaulted(self):
        c = claim(requires=['H9']); r = validate(graph([c]), ROW)
        self.assertEqual(r['normalized_graph']['claims'][0]['requires'], ['H9'])
        self.assertFalse(r['normalizations']); self.assertFalse(r['claims'])

    def test_null_arrays_not_silently_defaulted(self):
        self.assertFalse(validate(graph([claim(requires=None)]), ROW)['claims'])

    def test_unknown_text_prerequisite_not_logically_completed(self):
        c = claim(text='The object identified in H2 is visible.'); c.pop('requires')
        r = validate(graph([c]), ROW)
        self.assertIn('TEXT_CLAIM_REFERENCE_UNKNOWN', r['claim_audit'][0]['errors'])
        cs = [claim(), claim('H2', 'The object identified in H1 is visible.')]; cs[1].pop('requires')
        r = validate(graph(cs), ROW)
        self.assertIn('TEXT_REFERENCE_WITHOUT_DECLARED_DEPENDENCY', r['claim_audit'][1]['risks'])

    def test_aliases_audited_but_unknown_types_still_rejected(self):
        r = validate(graph([claim(text='The object moves.', kind='ACTION')]), ROW)
        self.assertEqual(r['claims'][0]['type'], 'TEMPORAL_EVENT')
        self.assertEqual(r['normalizations'][0]['code'], 'DOCUMENTED_ENUM_ALIAS')
        self.assertFalse(validate(graph([claim(kind='COMPOSITIONAL')]), ROW)['claims'])

    def test_question_only_claim_does_not_invent_requirement(self):
        r = validate(graph([claim(requirement_ids=[])]), ROW)
        self.assertTrue(r['claims']); self.assertEqual(r['covered_required_ids'], [])
        self.assertIn('QUESTION_ONLY_RELEVANCE_REVIEW', r['claim_audit'][0]['risks'])

    def test_json_fence_schema_echo_and_duplicate_keys(self):
        for raw in ('```json\n{}\n```', '{"claims":[],"claims":[]}', MODEL_SCHEMA, {'properties': graph()}):
            self.assertTrue(validate(raw, ROW)['root_errors'])

    def test_no_automatic_escape_or_trailing_backtick_recovery(self):
        for raw in (json.dumps(graph()) + '`', json.dumps(graph(), indent=2).replace('\n', '\\n')):
            self.assertFalse(validate(raw, ROW)['claims'])

    def test_no_model_truth_status_or_bbox_fields(self):
        for k, v in [('status', 'VERIFIED'), ('bbox', [1, 2, 3, 4]), ('evidence_status', 'SUPPORTED')]:
            c = claim(); c[k] = v
            self.assertFalse(validate(graph([c]), ROW)['claims'])
        c = validate(graph(), ROW)['claims'][0]
        self.assertEqual((c['status'], c['evidence_status']), ('UNVERIFIED', 'NOT_SEARCHED'))

    def test_partial_acceptance_does_not_delete_valid_sibling(self):
        r = validate(graph([claim(), claim('H2', 'Another object is visible.', kind='COMPOSITIONAL')]), ROW)
        self.assertEqual([c['claim_id'] for c in r['claims']], ['H1'])
        self.assertEqual(r['claim_audit'][1]['state'], 'REJECTED')

    def test_slots_do_not_count_as_claims_or_required_id_coverage(self):
        r = validate(graph([], plan_status='INCOMPLETE_PLAN', unresolved_slots=[slot()]), ROW)
        self.assertEqual(r['claims'], []); self.assertEqual(r['covered_required_ids'], [])
        self.assertFalse(r['downstream_publication_allowed'])

    def test_duplicate_ids_and_requirement_reference(self):
        self.assertFalse(validate(graph([claim(), claim(text='Another object is visible.')]), ROW)['claims'])
        self.assertFalse(validate(graph([claim(requirement_ids=['R7'])]), ROW)['claims'])

    def test_cycles_dangling_edges_and_self_alternative(self):
        cs = [claim(requires=['H2']), claim('H2', 'Another object is visible.', requires=['H1'])]
        self.assertFalse(validate(graph(cs), ROW)['claims'])
        self.assertFalse(validate(graph([claim(alternative_to=['H1'])]), ROW)['claims'])
        cs = [claim(requirement_ids=['R8']), claim('H2', 'Another object is visible.', requires=['H1'])]
        self.assertEqual(validate(graph(cs), ROW)['claim_audit'][1]['state'], 'BLOCKED_BY_DEPENDENCY')

    def test_numeric_bindings_rejected_symbolic_time_allowed(self):
        for text in ('The object is visible at 12 seconds.', 'The object appears at frame 3.', 'The ROI is [1,2,3,4].'):
            self.assertFalse(validate(graph([claim(text=text)]), ROW)['claims'])
        self.assertTrue(validate(graph(), ROW)['claims'])

    def test_numeric_slot_not_a_backdoor_for_execution_binding(self):
        s = {**slot(), 'purpose': 'Locate the object at frame 12.'}
        r = validate(graph([], plan_status='INCOMPLETE_PLAN', unresolved_slots=[s]), ROW)
        self.assertIn('SLOT_NUMERIC_EXECUTION_BINDING_FORBIDDEN', r['root_errors'])

    def test_exact_duplicate_time_parameters(self):
        r = validate(graph([claim(text='The object is visible at time t_i.'), claim('H2', 'The object is visible at time t_j.')]), ROW)
        self.assertEqual(len(r['claims']), 1)

    def test_rc_unresolved_no_guessed_identity(self):
        row = {**ROW, 'qa_type': 'region_caption_gpt'}
        self.assertFalse(validate(graph(), row)['claims'])
        r = validate(graph([], plan_status='UNRESOLVED_TARGET_REFERENCE', unresolved_slots=[slot()]), row)
        self.assertEqual(r['plan_status'], 'UNRESOLVED_TARGET_REFERENCE'); self.assertFalse(r['root_errors'])

    def test_dense_question_targets_survive_incomplete_requirements(self):
        labels = [f'event_{i}' for i in range(7)]
        row = {**ROW, 'qa_type': 'dense_captioning_gpt', 'question': 'Locate ' + repr(labels)}
        self.assertEqual(event_labels(row), labels); self.assertEqual(max_claims(row), 7)
        self.assertIn('event_6', prompt_for(row))
        cs = [claim(f'H{i+1}', label + ' occurs.', 'TEMPORAL_EVENT', requirement_ids=[]) for i, label in enumerate(labels)]
        self.assertEqual(len(validate(graph(cs), row)['claims']), 7)
        self.assertEqual(validate(graph(cs), row)['question_target_label_gaps'], [])

    def test_neighbor_order_and_first_occurrence_not_required(self):
        row = {**ROW, 'qa_type': 'tal', 'question': 'When does the target event happen?'}
        r = validate(graph([claim(text='The target event has not occurred previously.', kind='TEMPORAL_EVENT')]), row)
        self.assertIn('FIRST_OCCURRENCE_ASSUMPTION_REVIEW', r['claim_audit'][0]['risks'])
        self.assertIn('not its neighbors', prompt_for(row))

    def test_camera_and_instrument_can_cooccur(self):
        row = {**ROW, 'qa_type': 'tal', 'question': 'When does camera adjustment occur?'}
        p = prompt_for(row); self.assertIn('absence of concurrent instrument activity', p)
        r = validate(graph([claim(text='The camera shifts.', kind='TEMPORAL_EVENT')]), row)
        self.assertFalse(r['claim_audit'][0]['errors'])

    def test_stg_does_not_require_bbox_and_flags_3d_template(self):
        r = validate(graph(), ROW); self.assertTrue(r['claims'])
        r = validate(graph([claim(text='The object is at coordinates (x, y, z).')]), ROW)
        self.assertIn('UNREQUESTED_3D_COORDINATE_TEMPLATE', r['claim_audit'][0]['risks'])

    def test_compound_predicates_review_not_truth_decision(self):
        r = validate(graph([claim(text='One tool grasps tissue while another tool cuts it.', kind='TEMPORAL_EVENT')]), ROW)
        self.assertIn('MULTI_PREDICATE_REVIEW', r['claim_audit'][0]['risks'])
        self.assertEqual(r['semantic_quality'], 'NOT_REVIEWED')

    def test_two_entity_relation_not_mechanically_split(self):
        r = validate(graph([claim(text='The interface between object A and object B is visible.')]), ROW)
        self.assertFalse(any('MULTI' in x for x in r['claim_audit'][0]['risks']))

    def test_next_phase_not_candidate_list_order(self):
        r = validate(graph([claim(text='The next phase is the one following the current phase in the provided list.', kind='HIGH_LEVEL')]), ROW)
        self.assertIn('LIST_ORDER_ASSUMPTION_REVIEW', r['claim_audit'][0]['risks'])
        self.assertIn('Candidate ordering is not chronology', prompt_for({**ROW, 'qa_type': 'next_action'}))

    def test_summary_slots_and_placeholder_rejection_warning(self):
        row = {**ROW, 'qa_type': 'video_summary_gpt'}
        r = validate(graph([], plan_status='INCOMPLETE_PLAN', unresolved_slots=[slot()]), row)
        self.assertFalse(r['claims']); self.assertTrue(r['unresolved_slots'])
        r = validate(graph([claim(text='A specific instrument performs a specific action.', kind='TEMPORAL_EVENT')]), row)
        self.assertIn('PLACEHOLDER_RISK', r['claim_audit'][0]['risks'])

    def test_cvs_skill_observations_are_not_scores(self):
        for qa in ('cvs_assessment', 'skill_assessment'):
            r = validate(graph([claim(text='The result is adequate.', kind='HIGH_LEVEL')]), {**ROW, 'qa_type': qa})
            self.assertIn('INTERPRETATION_REQUIRES_CRITERIA_REVIEW', r['claim_audit'][0]['risks'])
            self.assertFalse(r['downstream_publication_allowed'])

    def test_v31_layout_projection_not_old_admission_rewrite(self):
        old = {'plan_status': 'PLANNED', 'claims': [claim()], 'unresolved_references': [],
               'requirement_issues': [], 'uncovered_requirements': ['R1 (metadata)']}
        before = deepcopy(old); r = validate(json.dumps(old), ROW, layout='V31')
        self.assertEqual(old, before); self.assertTrue(r['claims'])
        self.assertEqual(r['source_layout_warnings'][0]['value'], ['R1 (metadata)'])


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.refs, self.a, self.b = make_inputs(self.root)
        self.kw = dict(stage1_bank=self.refs, baseline_v31=self.a, baseline_b=self.b,
                       config_path=CONFIG, cache_dir=self.root / 'cache', output_dir=self.root / 'run')

    def tearDown(self): self.temp.cleanup()

    def test_fixed_22_and_100_freeze_historical_sources(self):
        hashes = [digest(p) for p in (self.a, self.b)]; public, _, _, freeze = load_inputs(self.refs, self.a, self.b)
        r = run(**self.kw)
        self.assertEqual([x['source_record_index'] for x in public], SMOKE22)
        self.assertEqual(r['requirements_sha256'], freeze['requirements_sha256'])
        self.assertEqual(hashes, [digest(p) for p in (self.a, self.b)])

    def test_max_one_call_per_sample_no_repair_path(self):
        class Recording(MockAdapter):
            def infer(self, request):
                self.requests.append(request); return super().infer(request)
        adapter = Recording({'kind': 'mock', 'model': 'stage2-v41-test-only', 'max_retries': 0}); adapter.requests = []
        with patch('relive.stage2_lite_ablation.repair.apply_patch_once', side_effect=AssertionError('REPAIR_CALLED')):
            r = run(**self.kw, _backend=adapter)
        self.assertEqual(len(adapter.requests), 22)
        self.assertEqual(len({q['planning_binding']['sample_key'] for q in adapter.requests}), 22)
        self.assertEqual({q['stage'] for q in adapter.requests}, {'stage2_v41_generation'})
        self.assertEqual((r['repair_calls'], r['critic_calls']), (0, 0))

    def test_arms_C_and_extra_call_budget_rejected(self):
        with self.assertRaisesRegex(ValueError, 'NO_REPAIR'): run(**self.kw, arms=('C',))
        with self.assertRaisesRegex(ValueError, 'MAX_22'): run(**self.kw, max_new_calls=23)

    def test_cli_repair_flag_is_not_available(self):
        from relive.stage2_single_pass_v41.__main__ import main
        import contextlib, io
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                main(['--stage1-bank', str(self.refs), '--baseline-v31', str(self.a), '--baseline-b', str(self.b),
                      '--output-dir', str(self.root / 'bad'), '--repair'])
        self.assertEqual(error.exception.code, 2)

    def test_real_frozen_backend_unchanged(self):
        from relive.config import load_config
        cfg = load_config(ROOT / 'configs/qwen35_9b_stage2_v41.yaml')
        self.assertEqual(cfg['backend'],
                         load_config(ROOT / 'configs/qwen35_9b_stage2_v4_lite.yaml')['backend'])
        self.assertFalse(cfg['adaptation']['enabled']); self.assertEqual(cfg['adaptation']['actions'], [])

    def test_dry_run_does_not_load_model_or_open_cache(self):
        with patch('relive.stage2_single_pass_v41.runner.make_backend', side_effect=AssertionError('MODEL')):
            r = run(**self.kw, dry_run=True)
        self.assertEqual(r['B1_execution'], 'NOT_RUN'); self.assertFalse((self.root / 'cache').exists())

    def test_exact_replay_zero_calls_and_bytes(self):
        run(**self.kw)
        with patch('relive.stage2_single_pass_v41.runner.make_backend', side_effect=AssertionError('MODEL')):
            r = run(**{**self.kw, 'output_dir': self.root / 'replay'}, replay=True, reuse_run=self.root / 'run', max_new_calls=0)
        self.assertEqual(r['usage_this_invocation']['new_calls'], 0)
        self.assertEqual((self.root / 'run/smoke22_results.jsonl').read_bytes(), (self.root / 'replay/smoke22_results.jsonl').read_bytes())

    def test_replay_cache_miss_fails_closed(self):
        run(**self.kw)
        with patch('relive.stage2_single_pass_v41.runner.CachedInference.has_terminal_result', return_value=False):
            with self.assertRaisesRegex(ValueError, 'REPLAY_CACHE_MISS'):
                run(**{**self.kw, 'output_dir': self.root / 'miss'}, replay=True, reuse_run=self.root / 'run')

    def test_offline_reads_saved_outputs_only(self):
        run(**self.kw)
        with patch('relive.stage2_single_pass_v41.runner.CachedInference', side_effect=AssertionError('CACHE_OPEN')):
            r = run(**{**self.kw, 'output_dir': self.root / 'offline'}, offline=True, reuse_run=self.root / 'run')
        self.assertEqual(r['usage_this_invocation']['calls'], 0)

    def test_offline_historical_arms_need_no_generated_B1(self):
        r = run(**self.kw, arms=('A', 'B0'), offline=True)
        self.assertEqual(set(r['arms']), {'A', 'B0'})
        self.assertEqual(r['B1_execution'], 'NOT_RUN'); self.assertEqual(r['usage_this_invocation']['calls'], 0)

    def test_transport_failure_not_regenerated(self):
        class Failure(MockAdapter):
            def infer(self, request): self.calls += 1; raise BackendError('TRANSPORT_FAILURE', retryable=True)
        backend = Failure({'kind': 'mock', 'model': 'stage2-v41-test-only', 'max_retries': 0})
        r = run(**self.kw, _backend=backend)
        self.assertEqual(backend.calls, 22); self.assertEqual(r['usage_this_invocation']['new_calls'], 22)
        self.assertEqual(r['arms']['B1']['root_failure_samples'], 22)

    def test_budget_partial_resume_in_new_directory(self):
        first = run(**self.kw, max_new_calls=1); original = digest(self.root / 'run/manifest.json')
        self.assertEqual(first['status'], 'PARTIAL')
        resumed = run(**self.kw, resume=True)
        self.assertEqual(resumed['status'], 'SYNTHETIC_TEST_ONLY')
        self.assertEqual(resumed['usage_this_invocation']['new_calls'], 21)
        self.assertEqual(original, digest(self.root / 'run/manifest.json'))

    def test_completed_resume_returns_original_usage(self):
        first = run(**self.kw); resumed = run(**self.kw, resume=True)
        self.assertEqual(first['usage_this_invocation'], resumed['usage_this_invocation'])

    def test_report_interruption_resume_preserves_partial_report(self):
        def fail(out, rows, *a, **kw): write(out / 'smoke22_results.jsonl', [], jsonl=True); raise RuntimeError('REPORT_CRASH')
        with patch('relive.stage2_single_pass_v41.reports.publish', side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, 'REPORT_CRASH'): run(**self.kw)
        before = digest(self.root / 'run/smoke22_results.jsonl')
        r = run(**self.kw, resume=True)
        self.assertEqual(r['usage_this_invocation']['new_calls'], 0)
        self.assertEqual(before, digest(self.root / 'run/smoke22_results.jsonl'))

    def test_no_overwrite_and_manifest_tamper(self):
        run(**self.kw)
        with self.assertRaisesRegex(ValueError, 'NEW_OUTPUT'): run(**self.kw)
        (self.root / 'run/normalized_claims.jsonl').write_text('tamper')
        with self.assertRaisesRegex(ValueError, 'HASH_INVALID'): verify(self.root / 'run')

    def test_input_change_stops_before_model(self):
        data = [json.loads(s) for s in self.b.read_text().splitlines()]; data[0]['question'] = 'Changed'
        self.b.write_text('\n'.join(json.dumps(x) for x in data))
        with self.assertRaisesRegex(ValueError, 'FROZEN_PUBLIC_INPUT'): run(**self.kw)

    def test_uniform_gate_raw_retention_and_blank_human_metrics(self):
        r = run(**self.kw); self.assertEqual(set(r['arms']), {'A', 'B0', 'B1'})
        for arm in r['arms'].values():
            self.assertIsNone(arm['human_approved_claim_rate']); self.assertIsNone(arm['graph_level_sufficiency_rate'])
        page = (self.root / 'run/ablation_A_B0_B1.html').read_text()
        for i in range(1, 23): self.assertIn(f'Case {i:02d}', page)
        self.assertIn('The queried tool is visible at time t.', page)
        self.assertEqual(len(list((self.root / 'run/raw_completions').glob('*.json'))), 66)
        with (self.root / 'run/blind_claim_review.csv').open() as stream:
            records = list(csv.DictReader(stream))
        self.assertTrue(all(not x['overall_decision'] and x['review_status'] == 'NOT_REVIEWED' for x in records))
        self.assertTrue(all('arm' not in x for x in records))
        for k in ('image_reads', 'certificate_writes', 'new_verified_count'): self.assertEqual(r[k], 0)

    def test_historical_outputs_never_enter_generator_requests(self):
        class Recording(MockAdapter):
            def infer(self, request):
                if 'HISTORICAL_ONLY_SENTINEL' in request['prompt']: raise AssertionError('HISTORY_LEAK')
                return super().infer(request)
        data = [json.loads(s) for s in self.b.read_text().splitlines()]
        for row in data:
            value = json.loads(row['B']['raw_response'])
            for c in value['claims']: c['text'] = 'HISTORICAL_ONLY_SENTINEL is visible.'
            row['B']['raw_response'] = json.dumps(value); row['B']['gate'] = old_gate(row['B']['raw_response'], row)
        self.b.write_text('\n'.join(json.dumps(x) for x in data))
        run(**self.kw, _backend=Recording({'kind': 'mock', 'model': 'stage2-v41-test-only', 'max_retries': 0}))


if __name__ == '__main__': unittest.main()
