"""Immutable paired experiment with bounded repair and read-only exact replay."""
from collections import Counter
import html
import json
from pathlib import Path

from relive.config import load_config
from relive.storage.artifacts import ArtifactStore
from relive.storage.cache import Budget, CachedInference
from relive.claim_bank.pipeline import ReplayBackend
from .common import VERSION, canonical, stable_hash, file_hash, read_json, write_json, jsonl, new_output, separate_output, seal, verify, run_lock
from .corpus import Retriever, POLICY
from .protocol import ARMS, legacy, load_questions, select, prompt, parse, preserved, repair_prompt, protocol


def preflight(runtime_manifest, index_dir, config_path, output_dir, per_qa_type=2, max_calls=None):
    separate_output(output_dir,index_dir)
    rows = select(load_questions(runtime_manifest), per_qa_type)
    retriever = Retriever(index_dir); config = load_config(config_path)
    synthetic = config['backend']['kind'] == 'mock'
    if retriever.manifest['synthetic_test'] and not synthetic: raise ValueError('MOCK_KNOWLEDGE_REAL_MODEL_FORBIDDEN')
    generation = config['backend'].get('generation', {})
    if generation.get('do_sample', False) or generation.get('temperature', 0) not in (0, 0.0, None):
        raise ValueError('DETERMINISTIC_GENERATION_REQUIRED')
    if config['backend']['kind'] == 'openai_compatible':
        effective = {**generation, **config['backend']['extra']}
        if type(effective.get('temperature')) not in (int,float) or effective['temperature'] != 0:
            raise ValueError('API_TEMPERATURE_ZERO_MUST_BE_EXPLICIT')
    # Hard attempt budget includes the shared backend's technical retries.
    bound = len(rows) * len(ARMS) * 2 * (config['backend'].get('max_retries', 0) + 1)
    if max_calls is not None and (type(max_calls) is not int or max_calls < 1): raise ValueError('MAX_CALLS_INVALID')
    root = new_output(output_dir)
    write_json(root/'selected_questions.jsonl', rows, jsonl=True)
    retrieval = [{'sample_key': r['sample_key'], **retriever.retrieve(r['question'], r['qa_type'])} for r in rows]
    write_json(root/'retrieval.jsonl', retrieval, jsonl=True)
    write_json(root/'retrieval_review_queue.jsonl', [
        {'sample_key': row['sample_key'], 'source_record_index': row['source_record_index'], 'qa_type': row['qa_type'],
         'question': row['question'], 'retrieval_status': found['status'], 'provided_cards': found['provided_cards'],
         'matched_terms': found['selected'], 'decision': 'PENDING', 'reviewer_id': None, 'rationale': None,
         'allowed_decisions': ['RELEVANT', 'PARTIALLY_RELEVANT', 'IRRELEVANT', 'UNCERTAIN', 'NO_CONTEXT'],
         'automatic_semantic_relevance_verified': False}
        for row, found in zip(rows, retrieval)], jsonl=True)
    write_json(root/'protocol.json', protocol()); write_json(root/'frozen_config.json', config)
    write_json(root/'gt_isolation_audit.json', {'status': 'PASS', 'runtime_allowlist_checked': True,
               'retrieval_query_fields': ['question', 'qa_type'], 'images_opened': 0, 'gt_used': False,
               'model_calls_made': 0, 'certificate_writes': 0, 'new_verified_count': 0})
    return seal(root, 'stage1-rag-preflight-v1', protocol_sha256=protocol()['protocol_sha256'],
                source_runtime_sha256=file_hash(runtime_manifest), index_manifest_sha256=file_hash(Path(index_dir)/'manifest.json'),
                source_library_sha256=retriever.manifest['source_library_sha256'], retrieval_policy=POLICY,
                sample_count=len(rows), arms=list(ARMS), per_qa_type=per_qa_type,
                sample_indices=[r['source_record_index'] for r in rows], synthetic_test=synthetic,
                max_calls=max_calls if max_calls is not None else bound, theoretical_attempt_bound=bound,
                model_calls_made=0, knowledge_policy=retriever.manifest['knowledge_policy'])


def load_experiment(preflight_dir, index_dir):
    root = Path(preflight_dir); m = verify(root, 'stage1-rag-preflight-v1')
    if m['protocol_sha256'] != protocol()['protocol_sha256']: raise ValueError('PROTOCOL_OR_CODE_CHANGED')
    retriever = Retriever(index_dir)
    if file_hash(Path(index_dir)/'manifest.json') != m['index_manifest_sha256']: raise ValueError('INDEX_BINDING_CHANGED')
    rows = load_questions(root/'selected_questions.jsonl'); retrieval = jsonl(root/'retrieval.jsonl')
    if len(rows) != m['sample_count'] or len(retrieval) != len(rows): raise ValueError('COHORT_COUNT_MISMATCH')
    if rows != select(rows, m['per_qa_type']): raise ValueError('COHORT_ORDER_OR_COVERAGE_INVALID')
    for row, recorded in zip(rows, retrieval):
        actual = {'sample_key': row['sample_key'], **retriever.retrieve(row['question'], row['qa_type'])}
        if actual != recorded: raise ValueError('RETRIEVAL_NOT_REPRODUCIBLE')
    return m, rows, retrieval, read_json(root/'frozen_config.json')


def request_for(row, cards, arm, experiment_sha, text=None, attempt=0):
    return {'stage': 'stage1_rag', 'prompt': prompt(row, cards) if text is None else text,
            'prompt_version': VERSION + ('-format-repair' if attempt else '-plan'), 'image_paths': [], 'frame_ids': [],
            'context': {'arm': arm, 'sample_key': row['sample_key'], 'experiment_sha256': experiment_sha, 'format_attempt': attempt}}


def plan_one(row, cards, arm, experiment_sha, inference, budget, readonly=False):
    traces = []; result = None; errors = []
    original = prompt(row, cards); previous_raw = None; protected = None
    for attempt in range(2):
        text = original if attempt == 0 else repair_prompt(original, previous_raw, errors)
        req = request_for(row, cards, arm, experiment_sha, text, attempt)
        call = inference.replay_readonly(req, budget) if readonly else inference.call(req, budget)
        parsed, errors = parse(call['raw_text'], cards) if call['execution_status'] == 'OK' else (None, ['MODEL_EXECUTION_FAILED'])
        if parsed is not None and attempt and protected is not None and parsed['plan']['requirements'] != protected:
            parsed = None; errors = ['FORMAT_REPAIR_CHANGED_VALID_REQUIREMENTS']
        traces.append({'format_attempt': attempt, 'request': req, 'call': call, 'validation_errors': errors})
        if parsed is not None:
            result = parsed; break
        if call['execution_status'] != 'OK': break
        previous_raw = call['raw_text']; protected = preserved(previous_raw)
    status = 'MODEL_OR_SCHEMA_FAILURE' if result is None else result['plan']['planning_status']
    flags = legacy.planning_flags(result['plan']) if result else []
    if result and cards and not result['knowledge_citations']:
        flags.append('RAG_CONTEXT_WITHOUT_CITATION_REVIEW')
    return {'arm': arm, 'sample_key': row['sample_key'], 'source_record_index': row['source_record_index'], 'qa_type': row['qa_type'],
            'question': row['question'], 'status': status, 'result': result, 'failure_reasons': errors,
            'knowledge_chunk_ids': [c['chunk_id'] for c in cards], 'review_flags': flags, 'traces': traces}


def _cache_snapshot(root):
    return {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(root.rglob('*')) if p.is_file()}


def _logical_calls(row):
    return sum(len(t['call']['attempts']) for t in row['traces'])


def run(preflight_dir, index_dir, cache_dir, output_dir, *, resume=False, replay_source=None):
    immutable_inputs=[preflight_dir,index_dir]+([replay_source] if replay_source else [])
    separate_output(output_dir,*immutable_inputs)
    separate_output(cache_dir,*immutable_inputs)
    m, rows, retrieval, config = load_experiment(preflight_dir, index_dir)
    experiment_sha = file_hash(Path(preflight_dir)/'manifest.json'); replay = replay_source is not None
    if replay and resume: raise ValueError('REPLAY_RESUME_NOT_ALLOWED')
    root = Path(output_dir); cache = Path(cache_dir).resolve()
    if root.resolve() == cache or root.resolve().is_relative_to(cache) or cache.is_relative_to(root.resolve()):
        raise ValueError('CACHE_AND_OUTPUT_MUST_BE_SEPARATE')
    before = None
    if root.exists():
        if not resume: raise ValueError('NEW_OUTPUT_DIRECTORY_REQUIRED')
        existing = read_json(root/'binding.json')
        if existing['experiment_sha256'] != experiment_sha: raise ValueError('RESUME_BINDING_MISMATCH')
        if (root/'manifest.json').exists():
            validate(root, preflight_dir, index_dir)
            return read_json(root/'generation_summary.json')
    if replay:
        verify(replay_source, 'stage1-rag-run-v1')
        binding = read_json(Path(replay_source)/'binding.json')
        if binding['experiment_sha256'] != experiment_sha: raise ValueError('REPLAY_EXPERIMENT_MISMATCH')
        if not cache.is_dir(): raise ValueError('REPLAY_CACHE_MISSING')
        backend = ReplayBackend(binding['backend_fingerprint'], config['backend'])
        before = _cache_snapshot(cache)
    else:
        from relive.backends import make_backend
        backend = make_backend(config['backend'])
        binding = {'experiment_sha256': experiment_sha, 'backend_fingerprint': backend.fingerprint(),
                   'synthetic_test': bool(backend.synthetic)}
    if binding['synthetic_test'] != m['synthetic_test']: raise ValueError('BACKEND_SYNTHETIC_MISMATCH')
    if root.exists():
        if not resume: raise ValueError('NEW_OUTPUT_DIRECTORY_REQUIRED')
        if read_json(root/'binding.json') != binding: raise ValueError('RESUME_BINDING_MISMATCH')
        if (root/'manifest.json').exists():
            verify(root, 'stage1-rag-run-v1'); return read_json(root/'generation_summary.json')
    else: new_output(root)
    write_json(root/'binding.json', binding)
    inference = CachedInference(backend, ArtifactStore(cache))
    completed = []; resumed = 0; limit = m['max_calls']
    with run_lock(root):
        expected_paths = set()
        for row, found in zip(rows, retrieval):
            for arm in ARMS:
                p = root/'checkpoints'/f"{row['source_record_index']:06d}-{arm}.json"
                expected_paths.add(p)
        if set((root/'checkpoints').glob('*.json')) - expected_paths: raise ValueError('UNEXPECTED_CHECKPOINT')
        for p in sorted(expected_paths):
            if p.exists():
                checkpoint = read_json(p)
                if checkpoint['binding_sha256'] != stable_hash(binding): raise ValueError('CHECKPOINT_BINDING_MISMATCH')
                # Revalidate bytes against recorded cache on resume below, not just the checkpoint label.
                completed.append(checkpoint['row'])
        consumed = sum(_logical_calls(r) for r in completed)
        if consumed > limit: raise ValueError('CHECKPOINT_BUDGET_EXCEEDED')
        budget = Budget(max(1, limit - consumed))
        output_rows = []
        for row, found in zip(rows, retrieval):
            for arm in ARMS:
                cards = found['provided_cards'] if arm == 'rag' else []
                p = root/'checkpoints'/f"{row['source_record_index']:06d}-{arm}.json"
                if p.exists():
                    saved = read_json(p)['row']
                    check_budget = Budget(max(1, _logical_calls(saved)))
                    actual = plan_one(row, cards, arm, experiment_sha, inference, check_budget, readonly=True)
                    if _semantic(actual) != _semantic(saved) or _trace_identity(actual) != _trace_identity(saved):
                        raise ValueError('CHECKPOINT_CONTENT_MISMATCH')
                    current = saved; resumed += 1
                else:
                    if consumed + budget.calls >= limit: raise ValueError('EXPERIMENT_CALL_BUDGET_EXHAUSTED')
                    current = plan_one(row, cards, arm, experiment_sha, inference, budget, readonly=replay)
                    write_json(p, {'binding_sha256': stable_hash(binding), 'row': current})
                output_rows.append(current)
        plans = [_semantic(r) for r in output_rows]
        requirements = []; citations = []
        for r in output_rows:
            if r['result'] is None: continue
            for req in r['result']['plan']['requirements']:
                requirements.append({'arm': r['arm'], 'sample_key': r['sample_key'], **req})
            for cite in r['result']['knowledge_citations']:
                cards_by_id = {c['chunk_id']: c for f in retrieval if f['sample_key'] == r['sample_key'] for c in f['provided_cards']}
                citations.append({'arm': r['arm'], 'sample_key': r['sample_key'], **cite,
                                  'source_refs': {cid: cards_by_id[cid]['source_refs'] for cid in cite['chunk_ids']},
                                  'attribution': 'MODEL_DECLARED_NOT_ENTAILMENT_VERIFIED'})
        write_json(root/'plans.jsonl', plans, jsonl=True)
        write_json(root/'requirements.jsonl', requirements, jsonl=True)
        write_json(root/'requirement_provenance.jsonl', citations, jsonl=True)
        write_json(root/'execution_traces.jsonl', output_rows, jsonl=True)
        equal = None
        if replay:
            equal = all(file_hash(root/name) == file_hash(Path(replay_source)/name)
                        for name in ('plans.jsonl', 'requirements.jsonl', 'requirement_provenance.jsonl'))
            if not equal: raise ValueError('REPLAY_SEMANTIC_RESULT_MISMATCH')
            if before != _cache_snapshot(cache): raise ValueError('REPLAY_CACHE_MUTATION')
        summary = {'format': VERSION, 'status': 'COMPLETE', 'semantic_quality': 'HUMAN_REVIEW_REQUIRED',
                   'engineering_gate': 'FAIL' if any(r['status'] == 'MODEL_OR_SCHEMA_FAILURE' for r in output_rows) else 'PASS',
                   'execution_mode': 'replay' if replay else 'run', 'synthetic_test': binding['synthetic_test'],
                   'sample_count': len(rows), 'plan_count': len(plans), 'requirement_count': len(requirements),
                   'planning_status_counts': {arm: dict(sorted(Counter(r['status'] for r in output_rows if r['arm'] == arm).items())) for arm in ARMS},
                   'model_calls_made': 0 if binding['synthetic_test'] else budget.new_calls,
                   'synthetic_backend_calls': budget.new_calls if binding['synthetic_test'] else 0,
                   'usage': budget.snapshot(), 'usage_scope': 'THIS_INVOCATION_EXCLUDING_RESUMED_CHECKPOINTS',
                   'total_logical_attempts': sum(_logical_calls(r) for r in output_rows), 'resumed_checkpoint_count': resumed,
                   'images_opened': 0, 'gt_used': False, 'certificate_writes': 0, 'new_verified_count': 0,
                   'knowledge_policy': m['knowledge_policy'], 'experiment_sha256': experiment_sha}
        summary['citation_requirement_count'] = len(citations)
        summary['format_repair_count'] = sum(max(0,len(r['traces'])-1) for r in output_rows)
        summary['review_flag_counts'] = dict(sorted(Counter(f for r in output_rows for f in r['review_flags']).items()))
        summary['retrieval_status_counts'] = dict(sorted(Counter(f['status'] for f in retrieval).items()))
        write_json(root/'generation_summary.json', summary)
        write_json(root/'gt_isolation_audit.json', {'status': 'PASS', 'question_only_allowlist_validated': True,
                   'images_opened': 0, 'gt_used': False, 'certificate_writes': 0, 'new_verified_count': 0})
        write_json(root/'cache_replay_audit.json', {'status': 'PASS' if replay else 'NOT_EXECUTED',
                   'exact_result_comparison': equal, 'new_model_calls': summary['model_calls_made'],
                   'cache_writes': 0 if replay else None, 'cache_bytes_unchanged': True if replay else None,
                   'resumed_checkpoint_count': resumed})
        seal(root, 'stage1-rag-run-v1', experiment_sha256=experiment_sha, synthetic_test=binding['synthetic_test'],
             replay_source_manifest_sha256=file_hash(Path(replay_source)/'manifest.json') if replay else None)
    return summary


def _semantic(r):
    return {k: r[k] for k in ('arm', 'sample_key', 'source_record_index', 'qa_type', 'question', 'status', 'result', 'failure_reasons', 'knowledge_chunk_ids', 'review_flags')}


def _trace_identity(row):
    return [{'request': t['request'], 'validation_errors': t['validation_errors'],
             'call': {k: t['call'][k] for k in ('execution_status', 'raw_text', 'raw_response_ref', 'failure_reason', 'cache_key')},
             'attempts': [{k: a[k] for k in ('raw_response_ref', 'attempt_index', 'execution_status', 'failure_reason', 'retryable')}
                          for a in t['call']['attempts']]} for t in row['traces']]


def validate(run_dir, preflight_dir, index_dir):
    m, rows, retrieval, _ = load_experiment(preflight_dir, index_dir)
    verify(run_dir, 'stage1-rag-run-v1'); root = Path(run_dir)
    binding = read_json(root/'binding.json')
    if binding['experiment_sha256'] != file_hash(Path(preflight_dir)/'manifest.json'): raise ValueError('EXPERIMENT_BINDING_MISMATCH')
    plans = jsonl(root/'plans.jsonl')
    expected = {(r['sample_key'], arm) for r in rows for arm in ARMS}
    if len(plans) != len(expected) or {(p['sample_key'], p['arm']) for p in plans} != expected: raise ValueError('PAIR_COVERAGE_INVALID')
    found = {r['sample_key']: r for r in retrieval}
    for p in plans:
        cards = found[p['sample_key']]['provided_cards'] if p['arm'] == 'rag' else []
        if p['result'] is not None:
            result, errors = parse(canonical(p['result']), cards)
            if errors or result['plan']['planning_status'] != p['status']: raise ValueError('SAVED_PLAN_INVALID')
        elif p['status'] != 'MODEL_OR_SCHEMA_FAILURE': raise ValueError('PLAN_STATUS_INVALID')
    summary = read_json(root/'generation_summary.json')
    if summary['gt_used'] or summary['images_opened'] or summary['certificate_writes'] or summary['new_verified_count']:
        raise ValueError('FORBIDDEN_ACTIVITY')
    return {'status': 'PASS', 'pair_count': len(rows), 'plan_count': len(plans), 'semantic_quality': 'HUMAN_REVIEW_REQUIRED',
            'engineering_gate': summary['engineering_gate'],
            'synthetic_test': m['synthetic_test'], 'model_calls_made': 0, 'images_opened': 0, 'certificate_writes': 0, 'new_verified_count': 0}


def compare(run_dir, preflight_dir, index_dir, output_dir):
    separate_output(output_dir,run_dir,preflight_dir,index_dir)
    report = validate(run_dir, preflight_dir, index_dir); root = new_output(output_dir)
    plans = jsonl(Path(run_dir)/'plans.jsonl'); found = {r['sample_key']: r for r in jsonl(Path(preflight_dir)/'retrieval.jsonl')}
    by_sample = {}; queue = []; sections = []
    for p in plans: by_sample.setdefault(p['sample_key'], {})[p['arm']] = p
    criteria = ['question_relevance', 'essential_evidence_coverage', 'action_and_clinical_semantics_preserved',
                'no_event_occurrence_assumption', 'type_compatibility', 'nonredundancy', 'knowledge_source_applicability']
    for key, pair in by_sample.items():
        queue.append({'sample_key': key, 'source_record_index': pair['baseline']['source_record_index'], 'qa_type': pair['baseline']['qa_type'],
                      'decision': 'PENDING', 'reviewer_id': None, 'rationale': None,
                      'rubric': {arm: {c: None for c in criteria} for arm in ARMS},
                      'allowed_ratings': ['PASS', 'FAIL', 'UNCERTAIN'], 'preferred_arm': None})
        cards = found[key]['provided_cards']
        sources = {s['source_id']: s for s in read_json(Path(index_dir)/'sources.json')}
        display_cards = [{**c, 'source_documents': [{k: sources[r['source_id']].get(k) for k in
                         ('source_id', 'title', 'file_path', 'publisher', 'source_url')} for r in c['source_refs']]} for c in cards]
        esc = lambda v: html.escape(json.dumps(v, ensure_ascii=False, sort_keys=True, indent=2))
        def article(arm):
            p = pair[arm]; result = p['result']
            reqs = result['plan']['requirements'] if result else []
            table = ('<table><thead><tr><th>ID</th><th>Observation to plan</th><th>Type / evidence</th><th>Importance</th></tr></thead><tbody>'
                     + ''.join('<tr>' + ''.join('<td>' + html.escape(str(v)) + '</td>' for v in
                         (r['requirement_id'], r['description'], r['type'] + ' / ' + r['evidence_type'], r['importance'])) + '</tr>' for r in reqs)
                     + '</tbody></table>')
            return '<article><h3>' + arm + ' — ' + html.escape(p['status']) + '</h3>' + table + '<details><summary>Exact saved plan and citations</summary><pre>' + esc(p) + '</pre></details></article>'
        sections.append('<section><h2>' + html.escape(pair['baseline']['qa_type']) + ' / ' + str(pair['baseline']['source_record_index'])
                        + '</h2><p>' + html.escape(pair['baseline']['question']) + '</p><div class="pair">'
                        + ''.join(article(arm) for arm in ARMS)
                        + '</div><details><summary>Retrieved pending-review cards / PDF pages</summary><pre>' + esc(display_cards) + '</pre></details></section>')
    write_json(root/'human_review_queue.jsonl', queue, jsonl=True)
    write_json(root/'comparison_report.json', {**report, 'automatic_semantic_improvement_claimed': False,
               'run_manifest_sha256': file_hash(Path(run_dir)/'manifest.json'),
               'retrieval_status_counts': dict(Counter(r['status'] for r in found.values())),
               'planning_status_counts': read_json(Path(run_dir)/'generation_summary.json')['planning_status_counts']})
    page = ('<!doctype html><meta charset="utf-8"><title>Stage 1 matched baseline vs RAG</title>'
            '<style>body{font:16px system-ui;margin:28px;background:#f7f9fc}section{background:white;padding:20px;margin:24px 0}'
            '.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}'
            'table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccd5df;padding:8px;text-align:left;font-size:14px;vertical-align:top}'
            '@media(max-width:900px){.pair{grid-template-columns:1fr}}</style><h1>Stage 1: matched baseline / RAG</h1>'
            '<p>Question-only planning. No observed events or medical approval. Semantic assessment requires human review. Synthetic: '
            + str(report['synthetic_test']) + '</p>' + ''.join(sections))
    (root/'comparison.html').write_text(page, encoding='utf-8')
    (root/'README.md').write_text('Review both arms using human_review_queue.jsonl. All decisions are PENDING.\n'
                                'This is a matched new baseline, not a byte-identical legacy Stage 1 rerun.\n'
                                'No model calls, images, GT, certificates or verified outputs are produced by comparison.\n')
    return seal(root, 'stage1-rag-comparison-v1', **report)
