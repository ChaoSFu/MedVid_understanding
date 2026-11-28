from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from relive.config import load_config
from relive.storage.artifacts import ArtifactStore, ArtifactError
from relive.storage.cache import Budget, BudgetExceeded, CachedInference
from relive.backends.mock import MockBackend
from RAG.stage1_rag.common import read_json, write_json, jsonl, file_hash, canonical, verify, seal
from RAG.stage1_rag import corpus, protocol, runner
from RAG.stage1_rag.smoke import fixture_library, mock_config, mock_dataset, mock_smoke


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.lib = self.root/'lib.json'; write_json(self.lib, fixture_library())
        self.config = self.root/'config.json'; write_json(self.config, mock_config())
        self.data = self.root/'data.json'; write_json(self.data, mock_dataset())

    def inputs(self):
        corpus.freeze(self.lib, self.root/'corpus', allow_pending=True, synthetic_fixture=True)
        corpus.index(self.root/'corpus', self.root/'index')
        protocol.prepare(self.data, self.root/'questions', 22)
        self.runtime = self.root/'questions/question_runtime_manifest.jsonl'

    def experiment(self, config=None, max_calls=None):
        self.inputs()
        runner.preflight(self.runtime, self.root/'index', config or self.config, self.root/'preflight', max_calls=max_calls)

    def execute(self, **kwargs):
        return runner.run(self.root/'preflight', self.root/'index', self.root/'cache', self.root/'run', **kwargs)


class IsolationTests(Fixture):
    def test_forbidden_fields_not_even_accessed(self):
        class Guard(dict):
            def get(self, key, *args):
                if key in ('value', 'struc_info', 'RC_info', 'video', 'metadata', 'id'): raise AssertionError('FORBIDDEN_READ')
                return super().get(key, *args)
        rec = Guard({'qa_type': 'tal', 'conversations': [{'from': 'human', 'value': 'What forceps action should be inspected?'},
                                                       Guard({'from': 'gpt', 'value': 'FORBIDDEN_GT_SENTINEL'})]})
        row = protocol.project(rec, 0, 'a'*64)
        self.assertNotIn('FORBIDDEN_GT_SENTINEL', protocol.prompt(row, []))

    def test_source_sentinel_absent_from_all_runtime_requests_and_artifacts(self):
        self.experiment(); self.execute()
        roots = [self.root/x for x in ('questions', 'preflight', 'run', 'cache')]
        for root in roots:
            for p in root.rglob('*'):
                if p.is_file() and p.suffix in ('.json', '.jsonl'):
                    self.assertNotIn('FORBIDDEN_GT_SENTINEL', p.read_text())
                    self.assertNotIn('MUST_NOT_OPEN.jpg', p.read_text())

    def test_duplicate_source_ids_have_unique_dataset_index_keys(self):
        protocol.prepare(self.data, self.root/'questions', 22)
        rows = protocol.load_questions(self.root/'questions/question_runtime_manifest.jsonl')
        self.assertEqual(len({r['sample_key'] for r in rows}), 22)

    def test_missing_multiple_questions_rejected(self):
        for turns in ([], [{'from': 'human', 'value': 'a'}, {'from': 'human', 'value': 'b'}]):
            with self.assertRaises(ValueError): protocol.project({'conversations': turns, 'qa_type': 'tal'}, 0, 'a'*64)

    def test_unsupported_task_rejected(self):
        row = mock_dataset()[0]; row['qa_type'] = 'unsupported'
        with self.assertRaisesRegex(ValueError, 'UNSUPPORTED'): protocol.project(row, 0, 'a'*64)

    def test_runtime_extra_oracle_field_rejected(self):
        row = protocol.project(mock_dataset()[0], 0, 'a'*64); row['oracle'] = 'sentinel'
        with self.assertRaisesRegex(ValueError, 'ALLOWLIST'): protocol.validate_question(row)

    def test_question_hash_mismatch(self):
        row = protocol.project(mock_dataset()[0], 0, 'a'*64); row['question'] += ' changed'
        with self.assertRaisesRegex(ValueError, 'HASH'): protocol.validate_question(row)

    def test_changing_gt_and_media_cannot_change_prompt(self):
        raw = mock_dataset()[0]; row = protocol.project(raw, 0, 'a'*64)
        raw['conversations'][1]['value'] = 'A DIFFERENT GT ANSWER'
        raw['video'] = ['different.jpg']; raw['struc_info'] = {'span': [100, 200]}
        other = protocol.project(raw, 0, 'b'*64)
        self.assertEqual(protocol.prompt(row, []), protocol.prompt(other, []))

    def test_cohort_requires_two_per_all_eleven_tasks(self):
        rows = [protocol.project(r, i, 'a'*64) for i, r in enumerate(mock_dataset())]
        self.assertEqual(len(protocol.select(rows)), 22)
        with self.assertRaisesRegex(ValueError, 'COVERAGE'): protocol.select(rows[:-1])

    def test_never_opens_images(self):
        with patch('PIL.Image.open', side_effect=AssertionError('IMAGE_READ')):
            self.experiment(); self.execute()


class FreezeRetrievalTests(Fixture):
    def test_freeze_requires_explicit_pending_and_synthetic_consent(self):
        with self.assertRaisesRegex(ValueError, 'MOCK_CORPUS'): corpus.freeze(self.lib, self.root/'out', allow_pending=True)
        with self.assertRaisesRegex(ValueError, 'PENDING_KNOWLEDGE'): corpus.freeze(self.lib, self.root/'out', synthetic_fixture=True)

    def test_expected_library_hash_and_count_checked(self):
        for kwargs in ({'expected_sha': 'a'*64}, {'expected_count': 2}):
            with self.assertRaises(ValueError): corpus.freeze(self.lib, self.root/'out', allow_pending=True, synthetic_fixture=True, **kwargs)

    def test_freeze_keeps_source_hash_and_review_labels(self):
        original = file_hash(self.lib); self.inputs()
        self.assertEqual(original, file_hash(self.lib))
        self.assertEqual(jsonl(self.root/'corpus/cards.jsonl')[0]['review_status'], 'pending')
        self.assertFalse(read_json(self.root/'corpus/manifest.json')['review_status_changed'])
        with self.assertRaisesRegex(ValueError, 'NEW_OUTPUT'): corpus.freeze(self.lib, self.root/'corpus', allow_pending=True, synthetic_fixture=True)

    def test_benchmark_specific_knowledge_excluded(self):
        lib = fixture_library(); lib['chunks'][0]['text'] = 'CholecT50 dataset specific action annotation'
        self.lib.write_text(canonical(lib))
        with self.assertRaisesRegex(ValueError, 'BENCHMARK'): corpus.freeze(self.lib, self.root/'out', allow_pending=True, synthetic_fixture=True)

    def test_unadmitted_real_source_rejected(self):
        lib = fixture_library(); lib.update(library_id='relive.surgical_visual_knowledge', synthetic_test=False, build_status='BUILT_PENDING_MEDICAL_REVIEW')
        lib['chunks'][0]['extraction_provenance'][0]['synthetic'] = False
        self.lib.write_text(canonical(lib))
        with self.assertRaisesRegex(ValueError, 'SOURCE_NOT_ADMITTED'): corpus.freeze(self.lib, self.root/'out', allow_pending=True)

    def test_duplicate_card_ids_rejected(self):
        lib = fixture_library(); lib['chunks'] *= 2; self.lib.write_text(canonical(lib))
        with self.assertRaisesRegex(ValueError, 'DUPLICATE_CHUNK'): corpus.freeze(self.lib, self.root/'out', allow_pending=True, synthetic_fixture=True)

    def test_index_and_ranking_deterministic(self):
        self.inputs(); corpus.index(self.root/'corpus', self.root/'index2')
        self.assertEqual(file_hash(self.root/'index/index.sqlite'), file_hash(self.root/'index2/index.sqlite'))
        a = corpus.Retriever(self.root/'index').retrieve('Inspect forceps jaws.', 'tal')
        b = corpus.Retriever(self.root/'index2').retrieve('Inspect forceps jaws.', 'tal')
        self.assertEqual(a, b); self.assertEqual(a['status'], 'RETRIEVED')
        self.assertLessEqual(a['context_bytes'], 6000)

    def test_nohit_and_unresolved_box_do_not_guess(self):
        self.inputs(); retriever = corpus.Retriever(self.root/'index')
        self.assertEqual(retriever.retrieve('xylophonexyz', 'stg')['status'], 'NO_MATCH_BASELINE_FALLBACK')
        row = retriever.retrieve('Describe the forceps inside the bounding box.', 'region_caption_gpt')
        self.assertEqual(row['status'], 'UNRESOLVED_TARGET_REFERENCE'); self.assertEqual(row['provided_cards'], [])

    def test_ties_diversity_and_top_k_are_deterministic(self):
        lib = fixture_library(); base = lib['chunks'][0]; source = lib['source_index'][0]
        lib['source_index'] = []; lib['chunks'] = []
        for s in range(3):
            sid = f'src{s}'; doc = deepcopy(source); doc['source_id'] = sid; doc['file_name'] = doc['file_path'] = sid+'.pdf'
            doc['chunk_count'] = 3; lib['source_index'].append(doc)
            for n in range(3):
                c = deepcopy(base); c['chunk_id'] = f'card{s}{n}'
                c['source_facts'][0]['source_id'] = sid
                c['source_refs'][0]['source_id'] = sid; c['locator'][0]['source_id'] = sid
                lib['chunks'].append(c)
        self.lib.write_text(canonical(lib)); self.inputs()
        result = corpus.Retriever(self.root/'index').retrieve('forceps', 'tal')
        self.assertEqual([c['chunk_id'] for c in result['provided_cards']], ['card00', 'card01', 'card10', 'card11'])
        self.assertEqual(len(result['selected']), 4)

    def test_retrieval_does_not_drop_query_target_at_tail(self):
        terms = corpus.query(' '.join('word'+str(i) for i in range(200)) + ' forceps')
        self.assertIn('forceps', terms); self.assertEqual(len(terms), 128)

    def test_oversize_card_not_cut_mid_sentence(self):
        lib = fixture_library(); lib['chunks'][0]['text'] = 'forceps ' + 'complete medical sentence. '*2000
        self.lib.write_text(canonical(lib)); self.inputs()
        self.assertEqual(corpus.Retriever(self.root/'index').retrieve('forceps', 'tal')['provided_cards'], [])

    def test_index_tamper_fails_before_model(self):
        self.experiment(); path = self.root/'index/index.sqlite'; path.write_bytes(path.read_bytes()+b'bad')
        with patch('relive.backends.make_backend', side_effect=AssertionError('MODEL_LOAD')):
            with self.assertRaisesRegex(ValueError, 'HASH'): self.execute()

    def test_real_backend_cannot_consume_mock_corpus(self):
        self.inputs()
        config = Path(__file__).resolve().parents[3]/'relive/configs/qwen35_9b_claim_bank_v1.yaml'
        with self.assertRaisesRegex(ValueError, 'MOCK_KNOWLEDGE'): runner.preflight(self.runtime, self.root/'index', config, self.root/'preflight')


class ContractTests(Fixture):
    def setUp(self):
        super().setUp(); self.value = mock_config()['backend']['rules'][0]['response']

    def test_wrapper_preserves_legacy_five_field_requirements(self):
        obj, errors = protocol.parse(canonical(self.value), [])
        self.assertFalse(errors); self.assertEqual(set(obj['plan']['requirements'][0]), {'requirement_id', 'description', 'type', 'evidence_type', 'importance'})

    def test_citations_must_be_supplied_and_unique(self):
        self.value['knowledge_citations'] = [{'requirement_id': 'R1', 'chunk_ids': ['x']}]
        self.assertEqual(protocol.parse(canonical(self.value), [dict(chunk_id='x')])[1], [])
        self.assertIn('CITATION_NOT_PROVIDED', protocol.parse(canonical(self.value), [])[1])
        self.value['knowledge_citations'] *= 2
        self.assertIn('CITATION_REQUIREMENT_INVALID', protocol.parse(canonical(self.value), [dict(chunk_id='x')])[1])

    def test_citations_cannot_reference_missing_requirement(self):
        self.value['knowledge_citations'] = [{'requirement_id': 'R6', 'chunk_ids': ['x']}]
        self.assertIn('CITATION_REQUIREMENT_INVALID', protocol.parse(canonical(self.value), [dict(chunk_id='x')])[1])

    def test_duplicate_json_keys_and_answer_fields_rejected(self):
        self.assertIsNone(protocol.parse('{"plan":{},"plan":{}}', [])[0])
        self.value['answer'] = 'yes'
        self.assertIsNone(protocol.parse(canonical(self.value), [])[0])

    def test_action_and_clinical_evidence_types_enforced(self):
        for typ in ('ACTION', 'HIGH_LEVEL'):
            self.value['plan']['requirements'][0]['type'] = typ
            self.assertIsNone(protocol.parse(canonical(self.value), [])[0])

    def test_malformed_field_types_fail_closed(self):
        for v in (None, [], {}, True, 2):
            obj = deepcopy(self.value); obj['plan']['planning_status'] = v
            self.assertIsNone(protocol.parse(canonical(obj), [])[0])

    def test_preserve_original_requirement_list(self):
        self.value['plan']['planning_status'] = 'GENERATED'
        self.assertEqual(protocol.preserved(canonical(self.value)), self.value['plan']['requirements'])

    def test_both_arms_have_same_prompt_when_no_hit(self):
        row = protocol.project(mock_dataset()[0], 0, 'a'*64)
        a = runner.request_for(row, [], 'baseline', 'experiment')
        b = runner.request_for(row, [], 'rag', 'experiment')
        self.assertEqual(a['prompt'], b['prompt']); self.assertEqual(a['image_paths'], [])


class ExecutionTests(Fixture):
    def test_22_pair_smoke_and_exact_readonly_replay(self):
        z = mock_smoke(self.root/'smoke')
        self.assertEqual(z['pair_count'], 22); self.assertEqual(z['fresh_synthetic_calls'], 44)
        self.assertEqual(z['real_model_calls'], 0); self.assertEqual(z['replay_cache_writes'], 0)
        self.assertEqual(z['replay_synthetic_calls'], 0)

    def test_fresh_and_replay_do_not_load_model_in_replay(self):
        self.experiment(); self.execute(); before = runner._cache_snapshot(self.root/'cache')
        with patch('relive.backends.make_backend', side_effect=AssertionError('MODEL_LOAD')):
            z = runner.run(self.root/'preflight', self.root/'index', self.root/'cache', self.root/'replay', replay_source=self.root/'run')
        self.assertEqual(z['usage']['cache_hits'], 44); self.assertEqual(before, runner._cache_snapshot(self.root/'cache'))

    def test_replay_cache_miss_no_new_cache_files_or_calls(self):
        self.experiment(); self.execute(); empty = self.root/'empty-cache'; empty.mkdir()
        with patch('relive.backends.make_backend', side_effect=AssertionError('MODEL_LOAD')):
            with self.assertRaisesRegex(ArtifactError, 'REPLAY_CACHE_MISS'):
                runner.run(self.root/'preflight', self.root/'index', empty, self.root/'replay', replay_source=self.root/'run')
        self.assertEqual(list(empty.iterdir()), [])

    def test_schema_repair_once_preserves_valid_requirements(self):
        cfg = mock_config(); good = deepcopy(cfg['backend']['rules'][0]['response']); bad = deepcopy(good)
        bad['plan']['planning_status'] = 'GENERATED'
        cfg['backend']['rules'] = [{'match': {'format_attempt': 0}, 'response': bad}, {'match': {'format_attempt': 1}, 'response': good}]
        path = self.root/'repair.json'; write_json(path, cfg); self.experiment(path); z = self.execute()
        self.assertEqual(z['synthetic_backend_calls'], 88)
        self.assertEqual(z['planning_status_counts']['rag'], {'PLANNED': 22})

    def test_repair_cannot_change_valid_requirements(self):
        cfg = mock_config(); good = deepcopy(cfg['backend']['rules'][0]['response']); bad = deepcopy(good)
        bad['plan']['planning_status'] = 'GENERATED'; good['plan']['requirements'][0]['description'] = 'Changed medical proposition.'
        cfg['backend']['rules'] = [{'match': {'format_attempt': 0}, 'response': bad}, {'match': {'format_attempt': 1}, 'response': good}]
        path = self.root/'repair.json'; write_json(path, cfg); self.experiment(path); z = self.execute()
        self.assertEqual(z['planning_status_counts']['rag'], {'MODEL_OR_SCHEMA_FAILURE': 22})
        self.assertEqual(z['synthetic_backend_calls'], 88)

    def test_valid_abstention_not_semantically_retried(self):
        cfg = mock_config(); cfg['backend']['rules'][0]['response']['plan'] = {
            'planning_status': 'UNABLE_TO_PLAN', 'planning_reason': 'The question cannot define a visual evidence need.', 'requirements': []}
        path = self.root/'abstain.json'; write_json(path, cfg); self.experiment(path); z = self.execute()
        self.assertEqual(z['synthetic_backend_calls'], 44); self.assertEqual(z['requirement_count'], 0)

    def test_budget_exhaustion_keeps_completed_checkpoints(self):
        self.experiment(max_calls=2)
        with self.assertRaisesRegex(ValueError, 'BUDGET_EXHAUSTED'): self.execute()
        self.assertEqual(len(list((self.root/'run/checkpoints').glob('*.json'))), 2)

    def test_interrupted_resume_matches_continuous_run(self):
        self.experiment(); original = runner.plan_one; n = 0
        def interrupt(*args, **kwargs):
            nonlocal n
            n += 1
            if n == 5: raise KeyboardInterrupt()
            return original(*args, **kwargs)
        with patch.object(runner, 'plan_one', side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt): self.execute()
        z = self.execute(resume=True)
        self.assertEqual(z['resumed_checkpoint_count'], 4); self.assertEqual(z['synthetic_backend_calls'], 40)
        runner.run(self.root/'preflight', self.root/'index', self.root/'cache', self.root/'continuous')
        self.assertEqual(file_hash(self.root/'run/plans.jsonl'), file_hash(self.root/'continuous/plans.jsonl'))

    def test_no_overwrite_and_completed_resume_no_model_load(self):
        self.experiment(); z = self.execute(); h = file_hash(self.root/'run/manifest.json')
        with patch('relive.backends.make_backend', side_effect=AssertionError('MODEL_LOAD')):
            with self.assertRaisesRegex(ValueError, 'NEW_OUTPUT'): self.execute()
            self.assertEqual(self.execute(resume=True), z)
        self.assertEqual(h, file_hash(self.root/'run/manifest.json'))

    def test_protocol_change_blocks_run_before_backend(self):
        self.experiment()
        with patch('RAG.stage1_rag.runner.protocol', return_value={'protocol_sha256': 'changed'}):
            with self.assertRaisesRegex(ValueError, 'PROTOCOL_OR_CODE_CHANGED'): self.execute()

    def test_nested_output_or_cache_cannot_modify_sealed_inputs(self):
        self.experiment()
        with patch('relive.backends.make_backend', side_effect=AssertionError('MODEL_LOAD')):
            with self.assertRaisesRegex(ValueError, 'OVERLAPS_IMMUTABLE'):
                runner.run(self.root/'preflight', self.root/'index', self.root/'cache', self.root/'preflight/run')
            with self.assertRaisesRegex(ValueError, 'OVERLAPS_IMMUTABLE'):
                runner.run(self.root/'preflight', self.root/'index', self.root/'index/cache', self.root/'run')

    def test_preflight_files_and_selection_are_reproducible(self):
        self.experiment()
        runner.preflight(self.runtime, self.root/'index', self.config, self.root/'preflight2')
        self.assertEqual(file_hash(self.root/'preflight/manifest.json'), file_hash(self.root/'preflight2/manifest.json'))

    def test_api_determinism_cannot_inherit_service_default_or_extra_temperature(self):
        self.inputs()
        # Real-knowledge flag is a fixture for this preflight validation only; no backend is created.
        path = self.root/'index/manifest.json'; manifest = read_json(path); manifest['synthetic_test'] = False
        path.write_text(canonical(manifest))
        base = {'kind': 'openai_compatible', 'base_url': 'https://example.invalid/v1', 'endpoint_path': '/chat/completions',
                'model': 'fixed-model', 'revision': 'fixed', 'credential_env': 'SYNTHETIC_TEST_KEY',
                'generation': {}, 'image_order': 'chronological', 'frame_encoding': 'png', 'timeout_seconds': 30, 'max_retries': 0, 'extra': {}}
        for i, extra in enumerate(({}, {'temperature': 1})):
            cfg = deepcopy(base); cfg['extra'] = extra; p = self.root/f'api{i}.json'
            write_json(p, {'backend': cfg, 'policy': {'name': 'semantic_spatial'}})
            with self.assertRaisesRegex(ValueError, 'TEMPERATURE_ZERO'):
                runner.preflight(self.runtime, self.root/'index', p, self.root/f'preflight{i}')

    def test_result_tampering_rejected(self):
        self.experiment(); self.execute()
        path = self.root/'run/plans.jsonl'; path.write_text(path.read_text().replace('PLANNED', 'TAMPERED'))
        with self.assertRaisesRegex(ValueError, 'HASH'):
            runner.validate(self.root/'run', self.root/'preflight', self.root/'index')

    def test_valid_rag_citations_keep_page_provenance_separate(self):
        cfg = mock_config(); response = deepcopy(cfg['backend']['rules'][0]['response'])
        response['knowledge_citations'] = [{'requirement_id': 'R1', 'chunk_ids': ['synthetic.forceps.001']}]
        cfg['backend']['rules'].insert(0, {'match': {'arm': 'rag'}, 'response': response})
        path = self.root/'citations.json'; write_json(path, cfg); self.experiment(path); result = self.execute()
        self.assertEqual(result['citation_requirement_count'], 22)
        citations = jsonl(self.root/'run/requirement_provenance.jsonl')
        self.assertTrue(all(c['arm'] == 'rag' and c['source_refs']['synthetic.forceps.001'][0]['page_start'] == 1 for c in citations))
        requirements = jsonl(self.root/'run/requirements.jsonl')
        self.assertTrue(all('source_refs' not in r and 'chunk_ids' not in r for r in requirements))

    def test_always_invalid_output_stops_after_one_repair_and_fails_gate(self):
        cfg = mock_config(); cfg['backend']['rules'][0]['response'] = '{bad json'
        path = self.root/'invalid.json'; write_json(path, cfg); self.experiment(path); result = self.execute()
        self.assertEqual(result['engineering_gate'], 'FAIL'); self.assertEqual(result['synthetic_backend_calls'], 88)
        self.assertEqual(result['requirement_count'], 0)
        self.assertEqual(len(jsonl(self.root/'run/plans.jsonl')), 44)

    def test_shared_cache_readonly_reader_preserves_attempt_cost(self):
        config = load_config(self.config)['backend']; backend = MockBackend(config)
        inference = CachedInference(backend, ArtifactStore(self.root/'cache'))
        req = {'stage': 'stage1_rag', 'prompt': 'synthetic', 'prompt_version': 'test', 'image_paths': [], 'frame_ids': []}
        inference.call(req, Budget(2)); before = runner._cache_snapshot(self.root/'cache')
        n = backend.calls; budget = Budget(2); result = inference.replay_readonly(req, budget)
        self.assertEqual(backend.calls, n); self.assertEqual(budget.new_calls, 0); self.assertEqual(budget.calls, 1)
        self.assertEqual(result['latency_seconds'], 0); self.assertEqual(before, runner._cache_snapshot(self.root/'cache'))

    def test_comparison_has_pending_review_no_fabricated_semantic_scores(self):
        self.experiment(); self.execute(); runner.compare(self.root/'run', self.root/'preflight', self.root/'index', self.root/'comparison')
        q = jsonl(self.root/'comparison/human_review_queue.jsonl')
        self.assertEqual(len(q), 22); self.assertTrue(all(r['decision'] == 'PENDING' for r in q))
        self.assertFalse(read_json(self.root/'comparison/comparison_report.json')['automatic_semantic_improvement_claimed'])
