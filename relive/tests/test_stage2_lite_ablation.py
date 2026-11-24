"""Synthetic engineering tests, not medical truth or benchmark answer fixtures."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from relive.hypothesis_graph.sources import load_references
from relive.hypothesis_graph_v31.validation import validate as old_validate
from relive.storage.artifacts import stable_hash
from relive.claim_revalidation.offline import write, read_json, digest
from relive.stage2_lite_ablation.contract import (SCHEMA, protocol, prompt_for, repair_prompt,
                                                max_claims, event_labels, SMOKE22)
from relive.stage2_lite_ablation.gate import validate
from relive.stage2_lite_ablation.repair import apply_patch_once
from relive.stage2_lite_ablation.runner import run, verify_output, read_rows
from relive.stage2_lite_ablation.sources import load_inputs
from relive.stage2_lite_ablation.fixtures import MockAdapter
from test_hypothesis_graph_v3 import make_refs, graph as historical_graph

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/stage2_v4_lite_mock.yaml'
REQ = {'requirement_id': 'R1', 'description': 'Inspect the queried object.', 'type': 'OBJECT',
       'evidence_type': 'SPATIAL', 'importance': 'REQUIRED'}
ROW = {'question': 'Is the forceps visible?', 'qa_type': 'stg', 'requirements': [REQ]}


def claim(cid='H1', text='The forceps is visible.', kind='OBJECT', **kw):
    return {'claim_id': cid, 'text': text, 'type': kind, 'role': 'TARGET',
            'requirement_ids': ['R1'], 'requires': [], 'alternative_to': [], **kw}


def graph(cs=None, **kw):
    return {'plan_status': 'PLANNED', 'claims': [claim()] if cs is None else cs,
            'unresolved_slots': [], 'requirement_issues': [], **kw}


def slot(sid='S1'):
    return {'slot_id': sid, 'purpose': 'Bind the observed event from future evidence.', 'status': 'UNBOUND'}


def patch_text(ops):
    return json.dumps({'operations': ops, 'rationale': 'Test-only patch rationale.'})


def op(kind, ids, cs=None, slots=None, **kw):
    return {'op': kind, 'ids': ids, 'claims': cs or [], 'slots': slots or [], 'reason': 'Test-only operation.', **kw}


def sources(root):
    refs = root / 'references'; make_refs(refs)
    public, _ = load_references(refs)
    baseline = []
    for r in public:
        raw = json.dumps(historical_graph())
        baseline.append({**r, **old_validate(raw, r), 'format': 'relive-claimgen-stage2-v3.1',
                         'raw_response': raw, 'synthetic_test': True,
                         'generation_metadata': {'generated_token_count': None}, 'call': None})
    p = root / 'baseline.json'
    write(p, {'scope': 'ALL_22_SAMPLES_WITH_COMPLETE_RAW_RESPONSES', 'samples': baseline})
    return refs, p


class GateTests(unittest.TestCase):
    def check(self, g=None, row=None): return validate(g or graph(), row or ROW)

    def test_schema_roundtrip_protocol_hash(self):
        self.assertEqual(SCHEMA, json.loads((ROOT / 'schemas/stage2_v4_lite.schema.json').read_text()))
        p = protocol(); h = p.pop('protocol_sha256'); self.assertEqual(h, stable_hash(p))

    def test_program_injects_only_unverified(self):
        r = self.check(); self.assertEqual(r['claims'][0]['status'], 'UNVERIFIED')
        self.assertEqual(r['claims'][0]['evidence_status'], 'NOT_SEARCHED')
        for k in ('status', 'bbox', 'evidence_status'):
            c = claim(); c[k] = 'VERIFIED'
            self.assertFalse(self.check(graph([c]))['claims'])

    def test_schema_echo_wrappers_json_duplicate_keys(self):
        for raw in (SCHEMA, {'properties': graph()}, '```json\n{}\n```', '{"claims":[],"claims":[]}', 'null', '[]'):
            self.assertTrue(validate(raw, ROW)['root_errors'])

    def test_duplicate_ids_both_rejected(self):
        self.assertFalse(self.check(graph([claim(), claim(text='A knife is visible.')]))['claims'])

    def test_bad_type_and_documented_alias_only(self):
        for kind in ('COMPOSITIONAL', 'INTERPRETATION', 'invalid'):
            self.assertFalse(self.check(graph([claim(kind=kind)]))['claims'])
        r = self.check(graph([claim(text='The forceps moves.', kind='ACTION')]))
        self.assertEqual(r['claims'][0]['type'], 'TEMPORAL_EVENT')
        self.assertEqual(r['normalizations'][0]['before'], 'ACTION')

    def test_invalid_requirement_and_metadata_refs(self):
        self.assertFalse(self.check(graph([claim(requirement_ids=['R8'])]))['claims'])
        self.assertTrue(self.check(graph(requirement_issues=[{'requirement_id': 'R1 (Synthesis)', 'issue': 'Test'}]))['root_errors'])

    def test_numeric_bindings_ids_and_symbolic_time(self):
        for text in ('The forceps is visible at 12 seconds.', 'The forceps is visible at frame 4.', 'The bbox is [1,2,3,4].'):
            self.assertFalse(self.check(graph([claim(text=text)]))['claims'])
        self.assertTrue(self.check(graph([claim(text='The forceps is visible at time t.')]))['claims'])

    def test_dependency_cycles_and_self_reference(self):
        cs = [claim(requires=['H2']), claim('H2', 'A knife is visible.', requires=['H1'])]
        self.assertFalse(self.check(graph(cs))['claims'])
        self.assertFalse(self.check(graph([claim(alternative_to=['H1'])]))['claims'])

    def test_blocked_dependency_after_pruning(self):
        r = self.check(graph([claim(requirement_ids=['R9']), claim('H2', 'A knife is visible.', requires=['H1'])]))
        self.assertEqual(r['claim_audit'][1]['state'], 'BLOCKED_BY_DEPENDENCY')
        self.assertFalse(r['claims'])

    def test_alternative_not_dangling_after_pruning(self):
        r = self.check(graph([claim(requirement_ids=['R9']), claim('H2', 'A knife is visible.', alternative_to=['H1'])]))
        self.assertFalse(r['claims'])

    def test_empty_plan_requires_slot(self):
        self.assertTrue(self.check(graph([]))['repair_needed'])
        r = self.check(graph([], plan_status='INCOMPLETE_PLAN', unresolved_slots=[slot()]))
        self.assertFalse(r['repair_needed']); self.assertEqual(r['plan_status'], 'INCOMPLETE_PLAN')

    def test_rc_no_guessed_target(self):
        row = {**ROW, 'qa_type': 'region_caption_gpt'}
        self.assertFalse(self.check(row=row)['claims'])
        r = self.check(graph([], plan_status='UNRESOLVED_TARGET_REFERENCE', unresolved_slots=[slot()]), row)
        self.assertFalse(r['repair_needed']); self.assertEqual(r['plan_status'], 'UNRESOLVED_TARGET_REFERENCE')

    def test_partial_acceptance(self):
        r = self.check(graph([claim(), claim('H2', 'The knife is visible.', kind='COMPOSITIONAL')]))
        self.assertEqual([c['claim_id'] for c in r['claims']], ['H1'])

    def test_empty_text_and_exact_duplicate(self):
        self.assertFalse(self.check(graph([claim(text=' ')]))['claims'])
        r = self.check(graph([claim(), claim('H2', 'THE FORCEPS IS VISIBLE!')]))
        self.assertEqual(len(r['claims']), 1)

    def test_parameterized_duplicate_not_per_timestamp(self):
        r = self.check(graph([claim(text='The forceps is visible at time t_i.'),
                              claim('H2', 'The forceps is visible at time t_j.')]))
        self.assertEqual(len(r['claims']), 1)

    def test_unable_plan_with_claims_rejected(self):
        self.assertFalse(self.check(graph(plan_status='UNABLE_TO_PLAN'))['claims'])

    def test_type_predicate_conflict(self):
        self.assertFalse(self.check(graph([claim(text='The forceps touches tissue.')]))['claims'])

    def test_connective_is_review_not_medical_proof(self):
        r = self.check(graph([claim(text='The forceps grasps tissue while the knife cuts tissue.', kind='STATIC_RELATION')]))
        self.assertIn('MULTI_PREDICATE_RISK', r['claim_audit'][0]['risks'])
        self.assertEqual(r['claim_audit'][0]['state'], 'REVIEW_REQUIRED')
        self.assertTrue(r['repair_needed'])

    def test_no_repair_for_unverified_or_unknown_dependency_semantics(self):
        cs = [claim(), claim('H2', 'The forceps touches tissue.', kind='STATIC_RELATION', requires=['H1'])]
        r = self.check(graph(cs)); self.assertFalse(r['repair_needed'])
        self.assertIn('DEPENDENCY_NECESSITY_REVIEW', r['claim_audit'][1]['risks'])

    def test_high_level_not_established_premise(self):
        cs = [claim(text='The procedure meets a clinical criterion.', kind='HIGH_LEVEL'),
              claim('H2', 'The forceps is visible.', requires=['H1'])]
        r = self.check(graph(cs))
        self.assertIn('UNVERIFIED_HIGH_LEVEL_PREREQUISITE_REVIEW', r['claim_audit'][1]['risks'])
        self.assertEqual(r['semantic_quality'], 'NOT_REVIEWED')

    def test_dynamic_dense_limit_covers_seven_events(self):
        row = {**ROW, 'qa_type': 'dense_captioning_gpt', 'question': "Locate ['a','b','c','d','e','f','g']."}
        self.assertEqual(max_claims(row), 7)
        cs = [claim('H' + str(i + 1), f'Event_{i} occurs.', 'TEMPORAL_EVENT') for i in range(7)]
        self.assertEqual(len(self.check(graph(cs), row)['claims']), 7)
        self.assertLessEqual(max_claims({**row, 'question': 'Locate ' + repr([str(i) for i in range(20)])}), 12)

    def test_secure_base_different_events_not_auto_alternatives(self):
        cs = [claim(text='Securing the base plate concludes.', kind='TEMPORAL_EVENT'),
              claim('H2', 'Securing the bag begins.', 'TEMPORAL_EVENT')]
        r = self.check(graph(cs)); self.assertTrue(all(not c['alternative_to'] for c in r['claims']))

    def test_adjust_camera_allows_instrument_activity(self):
        row = {**ROW, 'qa_type': 'tal', 'question': 'When does camera adjustment occur?'}
        r = self.check(graph([claim(text='The camera shifts.', kind='TEMPORAL_EVENT')]), row)
        self.assertTrue(r['claims']); self.assertFalse(r['repair_needed'])
        self.assertIn('REQUIREMENT_ASSUMPTION_REVIEW', self.check(graph([claim(text='No knife is engaging tissue.', kind='TEMPORAL_EVENT')]), row)['claim_audit'][0]['risks'])

    def test_forceps_unknown_bbox_not_unable_to_plan(self):
        self.assertTrue(self.check(graph([claim(text='The forceps is visible at time t.')]))['claims'])

    def test_pen_placeholder_and_continuity_review(self):
        for text, code in [('The Pen occupies a specific spatial region at time t.', 'PLACEHOLDER_RISK'),
                           ('The Pen maintains continuous visibility.', 'CONTINUOUS_VISIBILITY_ASSUMPTION')]:
            self.assertIn(code, self.check(graph([claim(text=text)]))['claim_audit'][0]['risks'])

    def test_next_action_list_not_order(self):
        r = self.check(graph([claim(text='The next phase is the one immediately following the current phase in the provided list.', kind='HIGH_LEVEL')]))
        self.assertIn('CANDIDATE_LIST_ORDER_ASSUMPTION', r['claim_audit'][0]['risks'])

    def test_summary_slots_not_vague_claims(self):
        row = {**ROW, 'qa_type': 'video_summary_gpt'}
        r = self.check(graph([claim(text='A specific instrument performs a specific action.', kind='TEMPORAL_EVENT')]), row)
        self.assertTrue(r['repair_needed'])
        self.assertFalse(self.check(graph([], plan_status='INCOMPLETE_PLAN', unresolved_slots=[slot()]), row)['repair_needed'])

    def test_cvs_independent_clinical_criteria_stay_unverified(self):
        cs = [claim('H' + str(i), f'Criterion_{i} is adequate.', 'HIGH_LEVEL') for i in range(1, 4)]
        r = self.check(graph(cs)); self.assertEqual(len(r['claims']), 3)
        self.assertTrue(all(not c['requires'] and c['status'] == 'UNVERIFIED' for c in r['claims']))

    def test_skill_interpretation_review_not_metadata_error(self):
        r = self.check(graph([claim(text='The tissue handling is gentle.', kind='HIGH_LEVEL')]))
        self.assertFalse(r['root_errors']); self.assertEqual(r['claim_audit'][0]['state'], 'REVIEW_REQUIRED')

    def test_generator_and_repair_allowlist(self):
        polluted = {**ROW, 'struc_info': 'GT_SENTINEL', 'RC_info': 'GT_SENTINEL', 'claims': 'OLD_CLAIM_SENTINEL',
                    'conversations': [{'from': 'gpt', 'value': 'GT_SENTINEL'}], 'video': 'IMAGE_SENTINEL'}
        p = prompt_for(polluted); g = validate(graph(), ROW)
        repair = repair_prompt(polluted, g['normalized_graph'], g)
        for text in (p, repair):
            for sentinel in ('GT_SENTINEL', 'OLD_CLAIM_SENTINEL', 'IMAGE_SENTINEL'): self.assertNotIn(sentinel, text)
            self.assertNotIn('"$schema":', text); self.assertNotIn('"properties":', text)


class RepairTests(unittest.TestCase):
    def before(self): return validate(graph([claim(text='A specific instrument performs a specific action.', kind='TEMPORAL_EVENT')]), ROW)

    def test_demote_preserves_original_and_audit(self):
        before = self.before(); saved = deepcopy(before)
        after, a = apply_patch_once(patch_text([op('DEMOTE_TO_SLOT', ['H1'], slots=[slot()])]), before, ROW)
        self.assertEqual(before, saved); self.assertEqual(a['status'], 'APPLIED')
        self.assertFalse(after['claims']); self.assertEqual(after['unresolved_slots'][0]['status'], 'UNBOUND')
        self.assertEqual(a['semantic_improvement'], 'NOT_REVIEWED')

    def test_invalid_patch_rollback(self):
        for raw in ('bad', '{}', patch_text([op('FIX_TYPE', ['H9'], [claim()])])):
            before = self.before(); after, a = apply_patch_once(raw, before, ROW)
            self.assertEqual(after, before); self.assertEqual(a['status'], 'ROLLED_BACK')

    def test_fix_type_and_edge_only_authorized_fields(self):
        before = validate(graph([claim(kind='TEMPORAL_EVENT')]), ROW)
        after, a = apply_patch_once(patch_text([op('FIX_TYPE', ['H1'], [claim()])]), before, ROW)
        self.assertEqual(a['status'], 'APPLIED'); self.assertTrue(after['claims'])
        bad = claim(text='The knife is visible.')
        self.assertEqual(apply_patch_once(patch_text([op('FIX_TYPE', ['H1'], [bad])]), before, ROW)[1]['status'], 'ROLLED_BACK')

    def test_keep_and_drop(self):
        before = self.before()
        self.assertEqual(apply_patch_once(patch_text([op('KEEP', ['H1'])]), before, ROW)[1]['status'], 'APPLIED')
        self.assertEqual(apply_patch_once(patch_text([op('DROP', ['H1'])]), before, ROW)[1]['status'], 'ROLLED_BACK')

    def test_split_and_merge(self):
        before = validate(graph([claim(text='The forceps is visible and the knife is visible.')]), ROW)
        cs = [claim(), claim('H2', 'The knife is visible.')]
        self.assertEqual(apply_patch_once(patch_text([op('SPLIT', ['H1'], cs)]), before, ROW)[1]['status'], 'APPLIED')
        before = validate(graph([claim(), claim('H2')]), ROW)
        self.assertEqual(apply_patch_once(patch_text([op('MERGE', ['H1', 'H2'], [claim()])]), before, ROW)[1]['status'], 'APPLIED')

    def test_fix_edge_and_drop_duplicate(self):
        before = validate(graph([claim(), claim('H2', 'The knife is visible.', requires=['H9'])]), ROW)
        fixed = claim('H2', 'The knife is visible.')
        self.assertEqual(apply_patch_once(patch_text([op('FIX_EDGE', ['H2'], [fixed])]), before, ROW)[1]['status'], 'APPLIED')
        before = validate(graph([claim(), claim('H2')]), ROW)
        self.assertEqual(apply_patch_once(patch_text([op('DROP', ['H2'])]), before, ROW)[1]['status'], 'APPLIED')

    def test_repair_cannot_leave_dangling_dependencies(self):
        before = validate(graph([claim(text='A specific instrument performs a specific action.', kind='TEMPORAL_EVENT'),
                                 claim('H2', 'The forceps is visible.', requires=['H1'])]), ROW)
        after, a = apply_patch_once(patch_text([op('DEMOTE_TO_SLOT', ['H1'], slots=[slot()])]), before, ROW)
        self.assertEqual(a['status'], 'ROLLED_BACK'); self.assertEqual(after, before)

    def test_no_unflagged_rewrite_or_full_regeneration(self):
        before = validate(graph(), ROW)
        self.assertEqual(apply_patch_once(patch_text([op('DROP', ['H1'])]), before, ROW)[1]['status'], 'ROLLED_BACK')
        self.assertEqual(apply_patch_once(patch_text([op('REPLACE_GRAPH', [], graph=graph())]), before, ROW)[1]['status'], 'ROLLED_BACK')

    def test_malformed_root_allows_explained_replacement(self):
        before = validate('invalid', ROW)
        after, a = apply_patch_once(patch_text([op('REPLACE_GRAPH', [], graph=graph())]), before, ROW)
        self.assertEqual(a['status'], 'APPLIED'); self.assertTrue(after['claims'])


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.refs, self.baseline = sources(self.root)
        self.kw = dict(stage1_bank=self.refs, v31_results=self.baseline, config_path=CONFIG,
                       cache_dir=self.root / 'cache', output_dir=self.root / 'run')

    def tearDown(self): self.temp.cleanup()

    def test_frozen_inputs_22_100_and_no_history_write(self):
        before = digest(self.baseline); public, _, freeze = load_inputs(self.refs, self.baseline)
        result = run(**self.kw)
        self.assertEqual([r['source_record_index'] for r in public], SMOKE22)
        self.assertEqual(sum(len(r['requirements']) for r in public), 100)
        self.assertEqual(before, digest(self.baseline)); self.assertEqual(result['requirements_sha256'], freeze['requirements_sha256'])

    def test_paired_B_C_and_max_two_calls(self):
        result = run(**self.kw)
        self.assertTrue(result['B_C_shared_generation'])
        data = read_rows(self.root / 'run/paired_results.jsonl')
        self.assertEqual(len(data), 22)
        for r in data:
            self.assertEqual(r['C']['parent_B_sha256'], stable_hash(r['B']))
            self.assertEqual(bool(r['C']['repair_audit'].get('call')), r['B']['gate']['repair_needed'])
        self.assertEqual(result['cost']['B']['logical_calls_in_result'], 22)
        self.assertLessEqual(result['usage_this_invocation']['calls'], 44)

    def test_replay_zero_calls_exact_rows_and_no_model_load(self):
        run(**self.kw)
        with patch('relive.stage2_lite_ablation.runner.make_backend', side_effect=AssertionError('MODEL_LOAD')):
            r = run(**{**self.kw, 'output_dir': self.root / 'replay'}, replay=True, reuse_run=self.root / 'run')
        self.assertEqual(r['usage_this_invocation']['new_calls'], 0)
        self.assertEqual((self.root / 'run/paired_results.jsonl').read_bytes(), (self.root / 'replay/paired_results.jsonl').read_bytes())
        self.assertEqual(read_json(self.root / 'replay/replay_audit.json')['status'], 'PASS')

    def test_offline_no_cache_or_model(self):
        run(**self.kw)
        with patch('relive.stage2_lite_ablation.runner.CachedInference', side_effect=AssertionError('CACHE_OPENED')):
            r = run(**{**self.kw, 'output_dir': self.root / 'offline'}, offline=True, reuse_run=self.root / 'run')
        self.assertEqual(r['usage_this_invocation']['calls'], 0)

    def test_B_then_C_reuses_generation(self):
        b = run(**self.kw, arms=('A', 'B'))
        c = run(**{**self.kw, 'output_dir': self.root / 'with_c'}, reuse_run=self.root / 'run')
        self.assertEqual(c['cost']['B']['new_calls_this_invocation'], 0)
        self.assertEqual(b['arms']['B'], c['arms']['B'])

    def test_budget_partial_and_immutable_resume(self):
        p = run(**self.kw, max_new_calls=1)
        self.assertEqual(p['status'], 'PARTIAL'); self.assertEqual(p['usage_this_invocation']['new_calls'], 1)
        before = digest(self.root / 'run/manifest.json')
        completed = run(**self.kw, resume=True)
        self.assertEqual(completed['status'], 'SYNTHETIC_TEST_ONLY')
        self.assertEqual(before, digest(self.root / 'run/manifest.json'))
        self.assertIn('continuations', completed['output_dir'])

    def test_output_no_overwrite_and_hash_tamper_detected(self):
        run(**self.kw)
        with self.assertRaisesRegex(ValueError, 'NEW_OUTPUT'): run(**self.kw)
        (self.root / 'run/ablation_B_raw.jsonl').write_text('tampered')
        with self.assertRaisesRegex(ValueError, 'HASH_INVALID'): verify_output(self.root / 'run')

    def test_stage1_mismatch_fail_closed(self):
        d = read_json(self.baseline); d['samples'][0]['requirements'][0]['description'] = 'changed'
        self.baseline.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError, 'FROZEN_INPUT'): run(**self.kw)

    def test_no_truth_writes_and_html_contains_all_cases(self):
        r = run(**self.kw)
        for key in ('image_reads', 'certificate_writes', 'new_verified_count'): self.assertEqual(r[key], 0)
        page = (self.root / 'run/ablation_comparison.html').read_text()
        for i in range(1, 23): self.assertIn(f'Case {i:02d}', page)
        self.assertIn('The queried instrument is visible.', page)
        self.assertIn('NOT_REVIEWED', page)
        self.assertEqual(read_json(self.root / 'run/gt_isolation_audit.json')['status'], 'PASS')
        with (self.root / 'run/claim_quality_review_template.csv').open() as stream:
            templates = list(csv.DictReader(stream))
        self.assertTrue(all(not x['overall_decision'] and x['review_status'] == 'NOT_REVIEWED' for x in templates))

    def test_dry_run_no_model_calls_or_cache(self):
        r = run(**self.kw, dry_run=True)
        self.assertEqual(r['status'], 'DRY_RUN'); self.assertEqual(r['usage_this_invocation']['calls'], 0)
        self.assertFalse((self.root / 'cache').exists())

    def test_replay_cache_miss_forbids_inference(self):
        run(**self.kw)
        with patch('relive.stage2_lite_ablation.runner.CachedInference.has_terminal_result', return_value=False):
            with self.assertRaisesRegex(ValueError, 'CACHE_MISS'):
                run(**{**self.kw, 'output_dir': self.root / 'miss'}, replay=True, reuse_run=self.root / 'run')

    def test_completed_resume_returns_saved_accounting(self):
        first = run(**self.kw); resumed = run(**self.kw, resume=True)
        self.assertEqual(first['usage_this_invocation'], resumed['usage_this_invocation'])

    def test_resume_after_report_publication_crash(self):
        from relive.stage2_lite_ablation import reports
        original = reports.generate_reports
        def interrupted(out, *args, **kwargs):
            write(out / 'paired_results.jsonl', [], jsonl=True)
            raise RuntimeError('SIMULATED_REPORT_CRASH')
        with patch.object(reports, 'generate_reports', side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, 'SIMULATED_REPORT_CRASH'): run(**self.kw)
        self.assertFalse((self.root / 'run/manifest.json').exists())
        before = digest(self.root / 'run/paired_results.jsonl')
        r = run(**self.kw, resume=True)
        self.assertEqual(r['usage_this_invocation']['new_calls'], 0)
        self.assertIn('continuations', r['output_dir'])
        self.assertEqual(before, digest(self.root / 'run/paired_results.jsonl'))

    def test_skip_repair_and_force_versioned_run(self):
        r = run(**self.kw, skip_repair=True, force_new_run=True)
        self.assertEqual(r['repair']['executed_calls'], 0)
        self.assertNotIn('C', r['arms'])
        self.assertNotEqual(r['output_dir'], str(self.root / 'run'))

    def test_real_backend_configuration_unchanged(self):
        from relive.config import load_config
        self.assertEqual(load_config(ROOT / 'configs/qwen35_9b_stage2_v4_lite.yaml')['backend'],
                         load_config(ROOT / 'configs/qwen35_9b_hypothesis_graph_stage2_v3_1.yaml')['backend'])

    def test_missing_historical_costs_not_fabricated(self):
        r = run(**self.kw)
        self.assertIsNone(r['cost']['A_historical']['original_latency_seconds'])
        self.assertIsNone(r['cost']['A_historical']['new_calls_in_original_run'])
        self.assertIsNone(r['cost']['B']['cost_estimate'])
        self.assertEqual(r['real_new_model_calls'], 0)

    def test_historical_claims_do_not_enter_requests(self):
        d = read_json(self.baseline)
        for s in d['samples']:
            old = historical_graph(); old['claims'][0]['text'] = 'HISTORICAL_CLAIM_SENTINEL is visible.'
            s['raw_response'] = json.dumps(old); s.update(old_validate(s['raw_response'], s))
        self.baseline.write_text(json.dumps(d))
        class Recording(MockAdapter):
            def infer(self, request):
                if 'HISTORICAL_CLAIM_SENTINEL' in request['prompt']: raise AssertionError('OLD_CLAIM_LEAK')
                return super().infer(request)
        r = run(**self.kw, _backend=Recording({'kind': 'mock', 'model': 'stage2-v4-lite-test-only', 'max_retries': 0}))
        self.assertEqual(r['status'], 'SYNTHETIC_TEST_ONLY')


if __name__ == '__main__': unittest.main()
