"""Scalable STG binding over the unchanged V4.1 single-pass implementation."""
from collections import Counter
import json
from pathlib import Path

from relive.backends import make_backend
from relive.config import load_config
from relive.claim_bank.pipeline import ReplayBackend
from relive.claim_revalidation.offline import write, read_json, digest
from relive.hypothesis_graph.sources import rows
from relive.hypothesis_graph.runner import GenerationEnvelope
from relive.storage.artifacts import ArtifactStore, stable_hash
from relive.storage.cache import Budget, CachedInference
from relive.stage2_single_pass_v41.contract import protocol, prompt_for, VERSION as GENERATOR_VERSION
from relive.stage2_single_pass_v41.gate import validate
from relive.stage2_single_pass_v41.runner import implementation_hash as core_hash
from relive.stage2_single_pass_v41.storage import verify, seal, save, checkpoint
from .data import load_data, bind_stage1

VERSION = 'relive-stg-claim-bank-v41-v1'


def implementation_hash():
    return stable_hash({'core_v41': core_hash(), 'batch': {p.name: digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))}})


def run(*, data_dir, stage1_bank, config_path, cache_dir, output_dir, limit=None, max_new_calls=None,
        on_unplanned='question-only', resume=False, replay_from=None, dry_run=False, progress_every=10, _backend=None):
    if on_unplanned not in ('question-only', 'skip'): raise ValueError('UPSTREAM_POLICY_INVALID')
    if limit is not None and (type(limit) is not int or limit < 1): raise ValueError('LIMIT_INVALID')
    if replay_from and (resume or dry_run or _backend): raise ValueError('REPLAY_FLAGS_INVALID')
    runtime, data_manifest = load_data(data_dir); upstream, upstream_binding = bind_stage1(runtime, stage1_bank)
    cfg = load_config(config_path)
    if cfg['backend'].get('max_retries', 0) != 0 or (cfg['backend']['kind'] != 'mock' and cfg['backend'].get('generation', {}).get('do_sample') is not False):
        raise ValueError('GREEDY_ZERO_RETRY_REQUIRED')
    if cfg['backend']['kind'] != 'mock' and upstream_binding['synthetic_stage1']: raise ValueError('SYNTHETIC_STAGE1_FOR_REAL_MODEL_FORBIDDEN')
    if _backend is not None and not _backend.synthetic: raise ValueError('INJECTED_BACKEND_MUST_BE_SYNTHETIC')
    public = []
    for r, u in zip(runtime, upstream):
        planned = u['planning_status'] == 'PLANNED' and bool(u['requirements'])
        if not planned and u['requirements']: raise ValueError('UNPLANNED_UPSTREAM_WITH_REQUIREMENTS')
        public.append({'sample_key': r['sample_key'], 'source_record_index': r['source_record_index'],
                       'original_sample_id': r['original_sample_id'], 'dataset_name': r['dataset_name'],
                       'qa_type': 'stg', 'question': r['question'], 'requirements': u['requirements'],
                       'question_sha256': stable_hash(r['question']), 'requirements_sha256': u['requirements_sha256'],
                       'upstream_status': u['planning_status'],
                       'conditioning': 'FROZEN_REQUIREMENTS' if planned else 'QUESTION_ONLY_UPSTREAM_UNPLANNED'})
    selected = public[:limit] if limit else public
    budget_limit = len(selected); max_new_calls = budget_limit if max_new_calls is None else max_new_calls
    if type(max_new_calls) is not int or not 0 <= max_new_calls <= budget_limit: raise ValueError('ONE_CALL_PER_SAMPLE_BUDGET_REQUIRED')
    identity = {'generator_protocol_sha256': protocol()['protocol_sha256'], 'implementation_sha256': implementation_hash(),
                'data_manifest_sha256': digest(Path(data_dir) / 'manifest.json'), 'stage1_binding': upstream_binding,
                'config_sha256': stable_hash(cfg), 'on_unplanned': on_unplanned}
    prior = None; previous = {}
    if replay_from:
        prior = verify(replay_from)
        if prior['identity'] != identity or prior['selected_keys'] != [r['sample_key'] for r in selected] or prior['dry_run']:
            raise ValueError('REPLAY_IDENTITY_OR_SELECTION_MISMATCH')
        previous = {r['sample_key']: r for r in rows(Path(replay_from) / 'claim_bank.jsonl')}
    out = Path(output_dir).resolve()
    if out.exists() and not resume: raise ValueError('NEW_OUTPUT_DIRECTORY_REQUIRED')
    if resume and (out / 'manifest.json').exists():
        old = verify(out)
        if old['identity'] != identity or old['selected_keys'] != [r['sample_key'] for r in selected]: raise ValueError('RESUME_IDENTITY_MISMATCH')
        if old['complete']: return {**read_json(out / 'generation_summary.json'), 'output_dir': str(out)}
        raise ValueError('SEALED_PARTIAL_RUN_USE_NEW_OUTPUT_SAME_CACHE')
    if resume and (out / 'claim_bank.jsonl').exists() and not (out / 'manifest.json').exists():
        old_binding = read_json(out / 'run_binding.json')
        if old_binding['identity'] != identity: raise ValueError('RESUME_IDENTITY_MISMATCH')
        from datetime import datetime, timezone
        new = out / 'continuations' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        new.mkdir(parents=True); write(new / 'run_binding.json', old_binding)
        for p in sorted((out / 'checkpoints').glob('*.json')): save(new / 'checkpoints' / p.name, checkpoint(p, identity), identity)
        out = new
    out.mkdir(parents=True, exist_ok=True)
    with ArtifactStore(out).run_lock():
        backend = None
        if not dry_run:
            if prior: backend = ReplayBackend(prior['backend_fingerprint'], cfg['backend'])
            else:
                if cfg['backend']['kind'] == 'mock':
                    from .fixtures import MockAdapter
                    inner = _backend or MockAdapter(cfg['backend'])
                else: inner = make_backend(cfg['backend'])
                backend = GenerationEnvelope(inner)
        fingerprint = backend.fingerprint() if backend else None
        binding = {'identity': identity, 'backend_fingerprint': fingerprint, 'dry_run': dry_run,
                   'selected_keys': [r['sample_key'] for r in selected], 'generator_protocol': GENERATOR_VERSION,
                   'repair_calls': 0, 'critic_calls': 0}
        if (out / 'run_binding.json').exists():
            if read_json(out / 'run_binding.json') != binding: raise ValueError('RESUME_BINDING_MISMATCH')
        else: write(out / 'run_binding.json', binding)
        budget = Budget(budget_limit); inference = CachedInference(backend, ArtifactStore(Path(cache_dir) / VERSION)) if backend else None
        results = []; resumed = 0
        for i, row in enumerate(selected):
            path = out / 'checkpoints' / (row['sample_key'] + '.json')
            if path.exists():
                result = checkpoint(path, identity); resumed += 1
                if any(result[k] != row[k] for k in row): raise ValueError('CHECKPOINT_PUBLIC_INPUT_CHANGED')
            else:
                result = {**row, 'raw_response': None, 'generation_metadata': {}, 'call': None, 'gate': None,
                          'generation_status': 'NOT_EXECUTED', 'synthetic_test': bool(backend and backend.synthetic)}
                skip = on_unplanned == 'skip' and row['conditioning'] == 'QUESTION_ONLY_UPSTREAM_UNPLANNED'
                if skip: result['generation_status'] = 'UPSTREAM_NOT_PLANNED'
                elif inference:
                    prompt = prompt_for(row)
                    request = {'stage': 'stage2_v41_generation', 'prompt_version': GENERATOR_VERSION,
                               'prompt': prompt, 'image_paths': [], 'frame_ids': [], 'context': {'qa_type': 'stg'},
                               'stg_batch_binding': {'identity': identity, 'sample_key': row['sample_key']}}
                    cached = inference.has_terminal_result(request)
                    if prior and not cached: raise ValueError('EXACT_REPLAY_CACHE_MISS')
                    if cached or budget.new_calls < max_new_calls:
                        c = inference.call(request, budget); record = read_json(c['raw_response_ref'])
                        envelope = json.loads(c['raw_text']) if c['execution_status'] == 'OK' else {}
                        raw = envelope.get('raw_response'); gate = validate(raw, row)
                        if row['conditioning'] == 'QUESTION_ONLY_UPSTREAM_UNPLANNED':
                            # No fabricated requirements or implication of upstream success.
                            gate['upstream_review_required'] = True
                        hard_failure = bool(gate['root_errors']) or (not gate['claims'] and any(a['errors'] for a in gate['claim_audit']))
                        result.update(raw_response=raw, generation_metadata=envelope.get('generation_metadata', {}), gate=gate,
                                      call={k: c[k] for k in ('execution_status', 'failure_reason', 'cache_key', 'raw_response_ref')},
                                      recorded_latency_seconds=record['latency_seconds'], prompt_sha256=stable_hash(prompt),
                                      generation_status='MODEL_OR_SCHEMA_FAILURE' if hard_failure else 'GENERATED' if gate['claims'] else 'NO_VALID_CLAIMS')
                if result['generation_status'] != 'NOT_EXECUTED': save(path, result, identity)
            if result['gate'] is not None:
                checked = validate(result['raw_response'], row)
                if row['conditioning'] == 'QUESTION_ONLY_UPSTREAM_UNPLANNED': checked['upstream_review_required'] = True
                if result['gate'] != checked: raise ValueError('RESULT_GATE_MISMATCH')
            if prior and result != previous[row['sample_key']]: raise ValueError('EXACT_REPLAY_RESULT_MISMATCH')
            results.append(result)
            if progress_every and ((i+1) % progress_every == 0 or i+1 == len(selected)):
                print(json.dumps({'processed': i+1, 'selected': len(selected), 'new_calls': budget.new_calls,
                                  'cache_hits': budget.cache_hits, 'sample_status': result['generation_status']}), flush=True)
        if digest(Path(stage1_bank) / 'bank_manifest.json') != upstream_binding['stage1_manifest_sha256'] or digest(Path(stage1_bank) / 'claim_bank.jsonl') != upstream_binding['stage1_bank_sha256']:
            raise ValueError('FROZEN_STAGE1_CHANGED')
        load_data(data_dir)
        claims = [{'sample_key': r['sample_key'], 'source_record_index': r['source_record_index'], **c}
                  for r in results for c in (r['gate']['claims'] if r['gate'] else [])]
        complete = all(r['generation_status'] != 'NOT_EXECUTED' for r in results) and not dry_run
        synthetic = bool(backend and backend.synthetic)
        summary = {'format': VERSION, 'generator_protocol': GENERATOR_VERSION,
                   'status': 'DRY_RUN' if dry_run else 'PARTIAL' if not complete else 'SYNTHETIC_TEST_ONLY' if synthetic else 'COMPLETE',
                   'sample_count': len(results), 'source_stg_count': len(public), 'complete_source_coverage': complete and len(results) == len(public),
                   'claim_count': len(claims), 'synthetic_test': synthetic, 'conditioning_counts': dict(Counter(r['conditioning'] for r in results)),
                   'generation_status_counts': dict(Counter(r['generation_status'] for r in results)),
                   'gate_state_counts': dict(Counter(a['state'] for r in results for a in (r['gate']['claim_audit'] if r['gate'] else []))),
                   'usage_this_invocation': budget.snapshot(), 'resumed_checkpoints': resumed,
                   'repair_calls': 0, 'critic_calls': 0, 'stage1_new_calls': 0, 'image_reads': 0, 'gt_used': False,
                   'certificate_writes': 0, 'new_verified_count': 0, 'semantic_quality': 'NOT_REVIEWED'}
        write(out / 'claim_bank.jsonl', results, jsonl=True); write(out / 'claims.jsonl', claims, jsonl=True)
        write(out / 'generation_failures.jsonl', [r for r in results if r['generation_status'] != 'GENERATED'], jsonl=True)
        write(out / 'generation_summary.json', summary)
        write(out / 'cache_replay_audit.json', {'status': 'PASS' if prior and complete else 'NOT_RUN', 'exact_result_comparison': 'PASS' if prior and complete else 'NOT_RUN',
              'new_model_calls': budget.new_calls, 'cache_hits': budget.cache_hits, 'repair_calls': 0})
        write(out / 'gt_isolation_audit.json', {'status': 'PASS', 'generator_fields': ['question', 'qa_type', 'requirements'],
              'gt_used': False, 'image_reads': 0, 'certificate_writes': 0, 'new_verified_count': 0})
        write(out / 'protocol.json', {'batch_version': VERSION, 'core_v41': protocol(), 'on_unplanned_stage1': on_unplanned,
                                     'limit_affects_cache_identity': False, 'max_calls_per_sample': 1})
        seal(out, format=VERSION, identity=identity, dry_run=dry_run, complete=complete, synthetic_test=synthetic,
             selected_keys=[r['sample_key'] for r in selected], backend_fingerprint=fingerprint, repair_calls=0)
        verify(out)
        return {**summary, 'output_dir': str(out)}
