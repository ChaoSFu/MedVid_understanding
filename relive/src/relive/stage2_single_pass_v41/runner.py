"""Independent V4.1 cache and a single generation per sample. No repair imports."""
import json
from pathlib import Path

from relive.backends import make_backend
from relive.config import load_config
from relive.claim_bank.pipeline import ReplayBackend
from relive.hypothesis_graph.runner import GenerationEnvelope
from relive.hypothesis_graph.sources import rows
from relive.storage.artifacts import ArtifactStore, stable_hash
from relive.storage.cache import Budget, CachedInference
from relive.claim_revalidation.offline import read_json, write, digest
from .contract import VERSION, PROMPT_PATH, protocol, prompt_for
from .gate import validate
from .sources import load_inputs, check_unchanged
from .storage import verify, checkpoint, save, seal


def implementation_hash():
    root = Path(__file__).parent; core = root.parent
    paths = list(root.glob('*.py')) + [PROMPT_PATH]
    paths += [core / x for x in ('storage/cache.py', 'storage/artifacts.py', 'backends/local_hf.py',
              'hypothesis_graph/runner.py', 'hypothesis_graph/sources.py', 'hypothesis_graph_v31/validation.py',
              'hypothesis_graph_v31/predicates.py', 'claim_planner/contract.py', 'v2/task_selection.py',
              'stage2_lite_ablation/contract.py', 'stage2_lite_ablation/gate.py', 'stage2_lite_ablation/sources.py')]
    return stable_hash({str(p.relative_to(core)): digest(p) for p in sorted(paths)})


def run(*, stage1_bank, baseline_v31, baseline_b, output_dir, config_path=None, cache_dir=None,
        model=None, arms=('A', 'B0', 'B1'), dry_run=False, offline=False, replay=False, reuse_run=None,
        resume=False, max_new_calls=22, _backend=None):
    if not arms or not set(arms) <= {'A', 'B0', 'B1'}: raise ValueError('V41_ACCEPTS_A_B0_B1_ONLY_NO_REPAIR')
    if type(max_new_calls) is not int or not 0 <= max_new_calls <= 22: raise ValueError('MAX_22_GENERATION_CALLS')
    if (replay or offline and 'B1' in arms) and not reuse_run: raise ValueError('REUSE_RUN_REQUIRED')
    if replay and offline or dry_run and (replay or offline): raise ValueError('MODE_CONFLICT')
    public, baseline_a, b0, freeze = load_inputs(stage1_bank, baseline_v31, baseline_b)
    cfg = load_config(config_path) if config_path else None
    if cfg and (cfg['backend'].get('max_retries', 0) != 0 or (cfg['backend']['kind'] != 'mock' and cfg['backend'].get('generation', {}).get('do_sample') is not False)):
        raise ValueError('FROZEN_GREEDY_ZERO_RETRIES_REQUIRED')
    if model and (not cfg or model != cfg['backend']['model']): raise ValueError('MODEL_CONFIG_IDENTITY_MISMATCH')
    execute = 'B1' in arms and not dry_run and not offline
    if execute and (not cfg or not cache_dir): raise ValueError('CONFIG_AND_CACHE_REQUIRED')
    if execute and cfg['backend']['kind'] != 'mock' and not replay:
        if freeze['synthetic_source'] or not freeze['stage1_independently_checked']:
            raise ValueError('INDEPENDENT_NONSYNTHETIC_STAGE1_REQUIRED')
    if _backend is not None and not _backend.synthetic: raise ValueError('SYNTHETIC_INJECTION_ONLY')
    identity = {'protocol_sha256': protocol()['protocol_sha256'], 'implementation_sha256': implementation_hash(),
                'public_inputs_sha256': freeze['public_inputs_sha256'], 'sources_sha256': stable_hash(freeze['source_sha256']),
                'config_sha256': stable_hash(cfg) if cfg else None}
    prior = None; prior_rows = {}
    if reuse_run:
        prior = verify(reuse_run)
        if prior['identity'] != identity or prior['dry_run']: raise ValueError('REUSE_IDENTITY_MISMATCH')
        prior_rows = {r['sample_key']: r for r in rows(Path(reuse_run) / 'smoke22_results.jsonl')}
    out = Path(output_dir).resolve()
    if out.exists() and not resume: raise ValueError('NEW_OUTPUT_DIRECTORY_REQUIRED')
    if resume and (out / 'manifest.json').exists():
        previous = verify(out)
        if previous['identity'] != identity: raise ValueError('RESUME_IDENTITY_MISMATCH')
        if previous['complete']: return {**read_json(out / 'ablation_A_B0_B1_metrics.json'), 'output_dir': str(out)}
        from datetime import datetime, timezone
        continuation = out / 'continuations' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        return run(stage1_bank=stage1_bank, baseline_v31=baseline_v31, baseline_b=baseline_b,
                   output_dir=continuation, config_path=config_path, cache_dir=cache_dir, model=model, arms=arms,
                   reuse_run=out, max_new_calls=max_new_calls, _backend=_backend)
    if resume and (out / 'smoke22_results.jsonl').exists():
        binding = read_json(out / 'run_binding.json')
        if binding['identity'] != identity: raise ValueError('RESUME_IDENTITY_MISMATCH')
        from datetime import datetime, timezone
        target = out / 'continuations' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        target.mkdir(parents=True); write(target / 'run_binding.json', binding)
        for path in sorted((out / 'checkpoints').glob('*.json')): save(target / 'checkpoints' / path.name, checkpoint(path, identity), identity)
        out = target
    out.mkdir(parents=True, exist_ok=True)
    with ArtifactStore(out).run_lock():
        backend = None
        if execute:
            if replay: backend = ReplayBackend(prior['backend_fingerprint'], cfg['backend'])
            else:
                if cfg['backend']['kind'] == 'mock':
                    from .fixtures import MockAdapter
                    inner = _backend or MockAdapter(cfg['backend'])
                else: inner = make_backend(cfg['backend'])
                backend = GenerationEnvelope(inner)
        fingerprint = backend.fingerprint() if backend else prior['backend_fingerprint'] if prior else None
        binding = {'identity': identity, 'backend_fingerprint': fingerprint, 'dry_run': dry_run,
                   'offline': offline, 'replay': replay, 'arms': list(arms), 'repair_enabled': False}
        if (out / 'run_binding.json').exists():
            if read_json(out / 'run_binding.json') != binding: raise ValueError('RESUME_BINDING_MISMATCH')
        else: write(out / 'run_binding.json', binding)
        synthetic = bool(fingerprint and fingerprint.get('synthetic')) or freeze['synthetic_source']
        budget = Budget(22); inference = CachedInference(backend, ArtifactStore(Path(cache_dir) / VERSION)) if backend else None
        result_rows = []; invocations = []; resumed = 0
        for row, a, old_b in zip(public, baseline_a, b0):
            path = out / 'checkpoints' / (row['sample_key'] + '.json'); b1 = None
            previous = prior_rows.get(row['sample_key'])
            if 'B1' in arms:
                if path.exists(): b1 = checkpoint(path, identity); resumed += 1
                elif previous and previous.get('B1') and not replay: b1 = previous['B1']
                elif inference:
                    prompt = prompt_for(row)
                    request = {'stage': 'stage2_v41_generation', 'prompt_version': VERSION, 'prompt': prompt,
                               'image_paths': [], 'frame_ids': [], 'context': {'qa_type': row['qa_type']},
                               'planning_binding': {'identity': identity, 'sample_key': row['sample_key']}}
                    cached = inference.has_terminal_result(request)
                    if replay and not cached: raise ValueError('REPLAY_CACHE_MISS')
                    if cached or budget.new_calls < max_new_calls:
                        call = inference.call(request, budget); raw_record = read_json(call['raw_response_ref'])
                        envelope = json.loads(call['raw_text']) if call['execution_status'] == 'OK' else {}
                        b1 = {'raw_response': envelope.get('raw_response'), 'generation_metadata': envelope.get('generation_metadata', {}),
                              'execution_status': call['execution_status'], 'failure_reason': call['failure_reason'],
                              'cache_key': call['cache_key'], 'raw_response_ref': call['raw_response_ref'],
                              'recorded_latency_seconds': raw_record['latency_seconds'], 'prompt_sha256': stable_hash(prompt),
                              'gate': validate(envelope.get('raw_response'), row)}
                        invocations.append({'sample_key': row['sample_key'], 'cache_hit': call['cache_hit'], 'stage': request['stage']})
                if b1:
                    if b1['gate'] != validate(b1['raw_response'], row): raise ValueError('B1_REVALIDATION_MISMATCH')
                    if previous and replay and previous['B1'] != b1: raise ValueError('EXACT_REPLAY_MISMATCH')
                    if not path.exists(): save(path, b1, identity)
            result_rows.append({**row, 'A': {'historical': a, 'gate': validate(a['raw_response'], row, layout='V31')},
                                'B0': {'historical': old_b, 'gate': validate(old_b['raw_response'], row)}, 'B1': b1})
        complete = 'B1' not in arms or all(r['B1'] is not None for r in result_rows)
        check_unchanged(freeze)
        from .reports import publish
        metrics = publish(out, result_rows, freeze, arms, budget.snapshot(), synthetic=synthetic, dry_run=dry_run,
                          complete=complete, replay=replay, resumed=resumed, config=cfg, backend_fingerprint=fingerprint)
        write(out / 'protocol.json', protocol())
        seal(out, format=VERSION, identity=identity, freeze=freeze, backend_fingerprint=fingerprint,
             dry_run=dry_run, complete=complete, synthetic_test=synthetic, repair_enabled=False,
             normal_generation_calls=len(invocations), repair_calls=0, critic_calls=0)
        verify(out)
        return {**metrics, 'output_dir': str(out)}
