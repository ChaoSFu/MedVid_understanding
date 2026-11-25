"""Original-index binding, upstream reuse, one-call generation and exact replay."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

from relive.storage.artifacts import stable_hash
from relive.claim_revalidation.offline import write, read_json, digest
from relive.claim_bank.pipeline import generate
from relive.backends.mock import MockBackend
from relive.stg_claim_bank_v41.data import prepare, load_data, bind_stage1, find_stage1
from relive.stg_claim_bank_v41.runner import run
from relive.stage2_single_pass_v41.contract import prompt_for
from relive.hypothesis_graph.sources import rows

ROOT = Path(__file__).resolve().parents[1]
S1_CONFIG = ROOT / 'configs/claim_bank_mock_v11.yaml'
CONFIG = ROOT / 'configs/stage2_v41_mock.yaml'


def make_data(root):
    source = [{'id': 'DUPLICATE_ID', 'qa_type': 'stg', 'dataset_name': 'TEST_ONLY', 'data_source': 'TEST_ONLY',
               'video': ['/not-present/frame.jpg'], 'sampled_video_frames': [0], 'metadata': {'video_id': 'fixture', 'hidden_answer': 'GT_SENTINEL'},
               'conversations': [{'from': 'human', 'value': f'Track the queried tool in query {i}.'},
                                 {'from': 'gpt', 'value': 'GT_SENTINEL'}], 'struc_info': 'GT_SENTINEL', 'RC_info': 'GT_SENTINEL'} for i in range(2)]
    file = root / 'stg.json'; save = root / 'source_index.jsonl'; write(file, source)
    mapping = [{'subset': 'stg', 'subset_file': file.name, 'subset_record_index': i, 'source_record_index': 100+i,
                'source_dataset_sha256': 'a'*64, 'source_sample_key': f'medvidu-{"a"*64}-{100+i:06d}',
                'source_record_canonical_sha256': stable_hash(r)} for i, r in enumerate(source)]
    write(save, mapping, jsonl=True)
    prepare(file, save, root / 'data', expected_count=2)
    return file, save


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.source, self.mapping = make_data(self.root)
        self.data = self.root / 'data'; self.stage1 = self.root / 's1'
        self.make_stage1()
        self.kw = dict(data_dir=self.data, stage1_bank=self.stage1, config_path=CONFIG,
                       cache_dir=self.root / 'cache', output_dir=self.root / 'run', progress_every=0)

    def tearDown(self): self.temp.cleanup()

    def make_stage1(self, backend=None, destination=None):
        generate(runtime_manifest=str(self.data / 'runtime_manifest.jsonl'), output_dir=str(destination or self.stage1),
                 stage=1, config_path=str(S1_CONFIG), cache_dir=str(self.root / ('s1cache' if backend is None else 's1_unable_cache')),
                 protocol_version='v1.1', _backend=backend)

    def test_original_source_domain_and_duplicate_ids(self):
        rr, m = load_data(self.data)
        self.assertEqual([r['source_record_index'] for r in rr], [100, 101])
        self.assertEqual(len({r['sample_key'] for r in rr}), 2)
        self.assertTrue(all(r['dataset_sha256'] == 'a'*64 for r in rr))
        self.assertTrue(all(r['original_sample_id'] == 'DUPLICATE_ID' for r in rr))

    def test_runtime_and_prompt_gt_isolation(self):
        text = (self.data / 'runtime_manifest.jsonl').read_text(); self.assertNotIn('GT_SENTINEL', text)
        runtime, _ = load_data(self.data); upstream, _ = bind_stage1(runtime, self.stage1)
        p = prompt_for({**runtime[0], 'requirements': upstream[0]['requirements']})
        self.assertNotIn('GT_SENTINEL', p); self.assertNotIn('/not-present/frame.jpg', p)

    def test_canonical_binding_tamper_detected(self):
        source = read_json(self.source); source[0]['conversations'][0]['value'] = 'Changed'
        self.source.write_text(json.dumps(source))
        with self.assertRaisesRegex(ValueError, 'SOURCE_HASH_MISMATCH'): load_data(self.data)

    def test_prepare_does_not_overwrite(self):
        with self.assertRaisesRegex(ValueError, 'NEW_DATA_OUTPUT'): prepare(self.source, self.mapping, self.data, 2)

    def test_reused_stage1_no_new_planning_calls_or_images(self):
        old = digest(self.stage1 / 'claim_bank.jsonl')
        r = run(**self.kw)
        self.assertEqual(r['stage1_new_calls'], 0); self.assertEqual(r['usage_this_invocation']['new_calls'], 2)
        self.assertEqual(r['image_reads'], 0); self.assertEqual(r['repair_calls'], 0)
        self.assertEqual(old, digest(self.stage1 / 'claim_bank.jsonl'))
        self.assertEqual(len(rows(self.root / 'run/claim_bank.jsonl')), 2)

    def test_smoke_then_full_reuses_same_request_cache(self):
        run(**self.kw, limit=1)
        full = run(**{**self.kw, 'output_dir': self.root / 'full'})
        self.assertEqual(full['usage_this_invocation']['cache_hits'], 1)
        self.assertEqual(full['usage_this_invocation']['new_calls'], 1)
        self.assertTrue(full['complete_source_coverage'])

    def test_exact_replay_does_not_load_model(self):
        run(**self.kw)
        with patch('relive.stg_claim_bank_v41.runner.make_backend', side_effect=AssertionError('MODEL_LOAD')):
            r = run(**{**self.kw, 'output_dir': self.root / 'replay'}, replay_from=self.root / 'run', max_new_calls=0)
        self.assertEqual(r['usage_this_invocation']['new_calls'], 0)
        self.assertEqual((self.root / 'run/claim_bank.jsonl').read_bytes(), (self.root / 'replay/claim_bank.jsonl').read_bytes())

    def test_replay_miss_fails_closed(self):
        run(**self.kw)
        with patch('relive.stg_claim_bank_v41.runner.CachedInference.has_terminal_result', return_value=False):
            with self.assertRaisesRegex(ValueError, 'CACHE_MISS'):
                run(**{**self.kw, 'output_dir': self.root / 'miss'}, replay_from=self.root / 'run')

    def test_unplanned_upstream_question_only_is_explicit(self):
        class Unable(MockBackend):
            def infer(self, request):
                self.calls += 1
                return json.dumps({'planning_status': 'UNABLE_TO_PLAN', 'planning_reason': 'TEST_ONLY upstream abstention.', 'requirements': []})
        cfg = {'kind': 'mock', 'model': 'fixture-unable', 'max_retries': 0}
        other = self.root / 's1_unable'; self.make_stage1(Unable(cfg), other)
        r = run(**{**self.kw, 'stage1_bank': other})
        self.assertEqual(r['conditioning_counts']['QUESTION_ONLY_UPSTREAM_UNPLANNED'], 2)
        self.assertEqual(r['stage1_new_calls'], 0)
        data = rows(self.root / 'run/claim_bank.jsonl')
        self.assertTrue(all(not x['requirements'] and x['gate']['upstream_review_required'] for x in data))
        replay = run(**{**self.kw, 'stage1_bank': other, 'output_dir': self.root / 'replay'}, replay_from=self.root / 'run')
        self.assertEqual(replay['usage_this_invocation']['new_calls'], 0)

    def test_upstream_skip_is_explicit_no_calls(self):
        class Unable(MockBackend):
            def infer(self, request): return json.dumps({'planning_status': 'UNABLE_TO_PLAN', 'planning_reason': 'TEST_ONLY.', 'requirements': []})
        other = self.root / 'unable'; self.make_stage1(Unable({'kind': 'mock', 'model': 'fixture-unable', 'max_retries': 0}), other)
        r = run(**{**self.kw, 'stage1_bank': other}, on_unplanned='skip')
        self.assertEqual(r['usage_this_invocation']['calls'], 0)
        self.assertEqual(r['generation_status_counts']['UPSTREAM_NOT_PLANNED'], 2)

    def test_preflight_no_backend_or_cache(self):
        r = run(**self.kw, dry_run=True)
        self.assertEqual(r['usage_this_invocation']['calls'], 0)
        self.assertFalse((self.root / 'cache').exists())

    def test_generation_constants_and_no_certificate(self):
        r = run(**self.kw)
        for key in ('repair_calls', 'critic_calls', 'certificate_writes', 'new_verified_count'): self.assertEqual(r[key], 0)
        self.assertTrue(all(c['status'] == 'UNVERIFIED' and c['evidence_status'] == 'NOT_SEARCHED' for c in rows(self.root / 'run/claims.jsonl')))

    def test_data_discovery_does_not_choose_synthetic_banks(self):
        with self.assertRaisesRegex(ValueError, 'SELECTION_REQUIRED'): find_stage1(self.data, self.root)

    def test_completed_resume_no_mutation(self):
        first = run(**self.kw); h = digest(self.root / 'run/manifest.json')
        r = run(**self.kw, resume=True)
        self.assertEqual(r['usage_this_invocation'], first['usage_this_invocation'])
        self.assertEqual(h, digest(self.root / 'run/manifest.json'))

    def test_one_call_budget_and_output_overwrite_prohibited(self):
        with self.assertRaisesRegex(ValueError, 'ONE_CALL'): run(**self.kw, max_new_calls=3)
        run(**self.kw)
        with self.assertRaisesRegex(ValueError, 'NEW_OUTPUT'): run(**self.kw)

    def test_same_generator_gate_and_backend_as_v41(self):
        from relive.stg_claim_bank_v41 import runner as batch
        from relive.stage2_single_pass_v41 import contract, gate
        from relive.config import load_config
        self.assertIs(batch.prompt_for, contract.prompt_for); self.assertIs(batch.validate, gate.validate)
        self.assertEqual(load_config(ROOT/'configs/qwen35_9b_stg_claim_bank_v41.yaml')['backend'],
                         load_config(ROOT/'configs/qwen35_9b_stage2_v41.yaml')['backend'])

    def test_partial_budget_can_continue_using_cache_in_new_output(self):
        r = run(**self.kw, max_new_calls=1)
        self.assertEqual(r['status'], 'PARTIAL')
        old = digest(self.root/'run/manifest.json')
        full = run(**{**self.kw, 'output_dir': self.root/'full'})
        self.assertEqual(full['usage_this_invocation']['new_calls'], 1)
        self.assertEqual(old, digest(self.root/'run/manifest.json'))

    def test_frozen_synthetic_upstream_cannot_enter_real_model(self):
        with patch('relive.stg_claim_bank_v41.runner.make_backend', side_effect=AssertionError('MODEL_LOAD')):
            with self.assertRaisesRegex(ValueError, 'SYNTHETIC_STAGE1'):
                run(**{**self.kw, 'config_path': ROOT/'configs/qwen35_9b_stg_claim_bank_v41.yaml'})


if __name__ == '__main__': unittest.main()
