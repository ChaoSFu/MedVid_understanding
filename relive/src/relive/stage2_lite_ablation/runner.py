"""Paired B/C runner. Generation once, optional repair once, write-once artifacts."""
from copy import deepcopy
import json
from pathlib import Path

from relive.backends import make_backend
from relive.config import load_config
from relive.claim_bank.pipeline import ReplayBackend
from relive.hypothesis_graph.runner import GenerationEnvelope
from relive.storage.artifacts import ArtifactStore, stable_hash
from relive.storage.cache import Budget, CachedInference
from relive.claim_revalidation.offline import read_json, digest, write
from .contract import VERSION, protocol, prompt_for, repair_prompt, PROMPT, REPAIR_PROMPT
from .gate import validate
from .repair import apply_patch_once
from .sources import load_inputs, check_unchanged


def implementation_hash():
    root = Path(__file__).parent; core = root.parent
    files = list(root.glob('*.py')) + [PROMPT, REPAIR_PROMPT]
    files += [core / x for x in ('storage/cache.py', 'storage/artifacts.py', 'backends/local_hf.py',
                                'hypothesis_graph/runner.py', 'hypothesis_graph/sources.py',
                                'hypothesis_graph_v31/validation.py', 'hypothesis_graph_v31/predicates.py',
                                'claim_planner/contract.py', 'v2/task_selection.py')]
    return stable_hash({str(p.relative_to(core)): digest(p) for p in sorted(files)})


def verify_output(root):
    root = Path(root); m = read_json(root / 'manifest.json')
    if stable_hash({k: v for k, v in m.items() if k != 'manifest_sha256'}) != m['manifest_sha256']:
        raise ValueError('ABLATION_MANIFEST_INVALID')
    for name, h in m['files'].items():
        p = root / name
        if not p.resolve().is_relative_to(root.resolve()) or digest(p) != h:
            raise ValueError('ABLATION_OUTPUT_HASH_INVALID')
    return m


def _checkpoint(path, identity):
    cp = read_json(path)
    if cp['identity'] != identity or cp['sha256'] != stable_hash(cp['value']): raise ValueError('CHECKPOINT_INVALID')
    return cp['value']


def _save(path, value, identity):
    path.parent.mkdir(exist_ok=True)
    write(path, {'identity': identity, 'value': value, 'sha256': stable_hash(value)})


def run(*, stage1_bank, v31_results, output_dir, config_path=None, cache_dir=None, model=None,
        arms=('A', 'B', 'C'), dry_run=False, offline=False, replay=False, reuse_run=None,
        resume=False, skip_repair=False, max_new_calls=44, force_new_run=False, _backend=None):
    arms = tuple(x for x in ('A', 'B', 'C') if x in arms)
    if skip_repair: arms = tuple(x for x in arms if x != 'C')
    if not arms or max_new_calls < 0 or max_new_calls > 44: raise ValueError('ARMS_OR_BUDGET_INVALID')
    if (offline or replay) and not reuse_run: raise ValueError('REUSE_RUN_REQUIRED')
    if offline and replay or dry_run and (offline or replay or reuse_run): raise ValueError('MODE_CONFLICT')
    public, baseline, freeze = load_inputs(stage1_bank, v31_results)
    cfg = load_config(config_path) if config_path else None
    if model:
        if not cfg: raise ValueError('MODEL_NEEDS_CONFIG')
        if model != cfg['backend']['model']: raise ValueError('MODEL_OVERRIDE_REQUIRES_NEW_FROZEN_CONFIG')
    execution = bool({'B', 'C'} & set(arms)) and not dry_run and not offline
    if execution and (not cfg or not cache_dir): raise ValueError('CONFIG_AND_CACHE_REQUIRED')
    if cfg and (cfg['backend'].get('max_retries', 0) != 0 or (cfg['backend']['kind'] != 'mock' and cfg['backend'].get('generation', {}).get('do_sample') is not False)):
        raise ValueError('GREEDY_ZERO_TECHNICAL_RETRIES_REQUIRED')
    if execution and cfg['backend']['kind'] != 'mock' and not replay:
        if freeze['synthetic_source'] or not freeze['stage1_independently_checked']:
            raise ValueError('INDEPENDENT_NONSYNTHETIC_STAGE1_REQUIRED_FOR_REAL_CALLS')
    if _backend is not None and not _backend.synthetic: raise ValueError('INJECTED_BACKEND_MUST_BE_SYNTHETIC')
    identity = {'protocol_sha256': protocol()['protocol_sha256'], 'implementation_sha256': implementation_hash(),
                'public_inputs_sha256': freeze['public_inputs_sha256'],
                'historical_sources_sha256': stable_hash(freeze['source_sha256']),
                'config_sha256': stable_hash(cfg) if cfg else None}
    reused = None; prior = None
    if reuse_run:
        prior = verify_output(reuse_run)
        if prior['identity'] != identity: raise ValueError('REUSE_IDENTITY_MISMATCH')
        if prior['dry_run']: raise ValueError('CANNOT_REUSE_DRY_RUN')
        reused = {r['sample_key']: r for r in read_rows(Path(reuse_run) / 'paired_results.jsonl')}
    out = Path(output_dir).resolve()
    if force_new_run:
        from datetime import datetime, timezone
        if resume or out.exists() and not out.is_dir(): raise ValueError('FORCE_NEW_RUN_CONFLICT')
        out = out / ('run_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    if out.exists() and not resume: raise ValueError('NEW_OUTPUT_DIRECTORY_REQUIRED')
    if resume and (out / 'paired_results.jsonl').exists() and not (out / 'manifest.json').exists():
        # A crash during report publication must not force overwriting the
        # partially published report. Copy checked checkpoints into a new run.
        previous_binding = read_json(out / 'run_binding.json')
        if previous_binding['identity'] != identity: raise ValueError('RESUME_IDENTITY_MISMATCH')
        from datetime import datetime, timezone
        fresh = out / 'continuations' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        fresh.mkdir(parents=True)
        write(fresh / 'run_binding.json', previous_binding)
        for path in sorted((out / 'checkpoints').glob('*.json')):
            value = _checkpoint(path, identity)
            _save(fresh / 'checkpoints' / path.name, value, identity)
        out = fresh
    if resume and (out / 'manifest.json').exists():
        prior_resume = verify_output(out)
        if prior_resume['identity'] != identity: raise ValueError('RESUME_IDENTITY_MISMATCH')
        if not prior_resume['complete']:
            # A budget-stopped run is already sealed. Continue into a new child
            # run, reusing its B artifacts, without overwriting any old report.
            from datetime import datetime, timezone
            continuation = out / 'continuations' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            return run(stage1_bank=stage1_bank, v31_results=v31_results, output_dir=continuation,
                       config_path=config_path, cache_dir=cache_dir, model=model, arms=arms,
                       offline=offline, replay=replay, reuse_run=out, max_new_calls=max_new_calls,
                       _backend=_backend)
    out.mkdir(parents=True, exist_ok=True)
    with ArtifactStore(out).run_lock():
        if (out / 'manifest.json').exists():
            m = verify_output(out)
            if m['identity'] != identity: raise ValueError('RESUME_IDENTITY_MISMATCH')
            return {**read_json(out / 'ablation_metrics.json'), 'output_dir': str(out)}
        backend = None
        if execution:
            if replay:
                backend = ReplayBackend(prior['backend_fingerprint'], cfg['backend'])
            else:
                if cfg['backend']['kind'] == 'mock':
                    from .fixtures import MockAdapter
                    inner = _backend or MockAdapter(cfg['backend'])
                else:
                    inner = _backend or make_backend(cfg['backend'])
                backend = GenerationEnvelope(inner)
        fingerprint = backend.fingerprint() if backend else prior['backend_fingerprint'] if prior else None
        binding = {'identity': identity, 'backend_fingerprint': fingerprint, 'arms': arms,
                   'dry_run': dry_run, 'offline': offline, 'replay': replay}
        # JSON serializes tuples as arrays; compare after canonicalization.
        binding = json.loads(json.dumps(binding))
        if (out / 'run_binding.json').exists():
            if read_json(out / 'run_binding.json') != binding: raise ValueError('RESUME_BINDING_MISMATCH')
        else: write(out / 'run_binding.json', binding)
        synthetic = bool(fingerprint and fingerprint.get('synthetic')) or freeze['synthetic_source']
        budget = Budget(44); inference = CachedInference(backend, ArtifactStore(Path(cache_dir) / VERSION)) if backend else None
        invocation = []; paired = []; resumed_count = 0

        def call(stage, prompt, row, parent_sha=None):
            request = {'stage': 'stage2_lite_' + stage, 'prompt_version': VERSION + ':' + stage,
                       'prompt': prompt, 'image_paths': [], 'frame_ids': [],
                       'context': {'qa_type': row['qa_type']}, 'planning_binding': {
                           'identity': identity, 'sample_key': row['sample_key'], 'parent_generation_sha256': parent_sha}}
            cached = inference.has_terminal_result(request)
            if replay and not cached: raise ValueError('EXACT_REPLAY_CACHE_MISS')
            if not cached and budget.new_calls >= max_new_calls: return None
            result = inference.call(request, budget)
            record = read_json(result['raw_response_ref'])
            if result['execution_status'] == 'OK':
                envelope = json.loads(result['raw_text']); raw = envelope['raw_response']; metadata = envelope['generation_metadata']
            else: raw = None; metadata = {'status': 'EXECUTION_FAILED'}
            value = {'raw_response': raw, 'generation_metadata': metadata,
                     'execution_status': result['execution_status'], 'failure_reason': result['failure_reason'],
                     'cache_key': result['cache_key'], 'raw_response_ref': result['raw_response_ref'],
                     'prompt_sha256': stable_hash(prompt), 'recorded_latency_seconds': record['latency_seconds']}
            invocation.append({'sample_key': row['sample_key'], 'stage': stage, 'cache_hit': result['cache_hit'],
                               'cache_key': result['cache_key'], 'new_calls': 0 if result['cache_hit'] else 1})
            return value

        for row, old in zip(public, baseline):
            key = row['sample_key']; bp = out / 'checkpoints' / (key + '.B.json'); cp = out / 'checkpoints' / (key + '.C.json')
            b = c = None; repair = {'status': 'NOT_REQUESTED'}
            cached_pair = reused.get(key) if reused else None
            if {'B', 'C'} & set(arms):
                if bp.exists():
                    b = _checkpoint(bp, identity); resumed_count += 1
                elif cached_pair and cached_pair.get('B') and not replay:
                    b = cached_pair['B']
                elif not dry_run and not offline:
                    b = call('generation', prompt_for(row), row)
                if b is not None:
                    b = deepcopy(b)
                    checked = validate(b['raw_response'], row)
                    if 'gate' in b and b['gate'] != checked: raise ValueError('B_REVALIDATION_MISMATCH')
                    b['gate'] = checked
                    if cached_pair and replay and b != cached_pair['B']: raise ValueError('B_EXACT_REPLAY_MISMATCH')
                    if not bp.exists(): _save(bp, b, identity)
                if 'C' in arms and b:
                    if cp.exists():
                        c = _checkpoint(cp, identity); repair = c['repair_audit']; resumed_count += 1
                    elif cached_pair and cached_pair.get('C') and not replay:
                        c = deepcopy(cached_pair['C']); repair = c['repair_audit']
                    elif offline:
                        if not cached_pair or not cached_pair['C']: raise ValueError('OFFLINE_C_NOT_AVAILABLE')
                        c = cached_pair['C']; repair = c['repair_audit']
                    elif b['gate']['repair_needed']:
                        repair_call = call('repair', repair_prompt(row, b['gate']['normalized_graph'], b['gate']), row, stable_hash(b))
                        if repair_call is None:
                            repair = {'status': 'BUDGET_EXHAUSTED', 'trigger_reasons': b['gate']['repair_reasons']}
                        else:
                            after, audit = apply_patch_once(repair_call['raw_response'], b['gate'], row)
                            repair = {**audit, 'trigger_reasons': b['gate']['repair_reasons'], 'call': repair_call}
                            c = {'parent_B_sha256': stable_hash(b), 'gate': after, 'repair_audit': repair}
                    else:
                        repair = {'status': 'NO_REPAIR_NEEDED', 'semantic_quality': 'NOT_REVIEWED'}
                        c = {'parent_B_sha256': stable_hash(b), 'gate': b['gate'], 'repair_audit': repair}
                    if c:
                        if c['parent_B_sha256'] != stable_hash(b): raise ValueError('C_DID_NOT_REUSE_B')
                        if c['repair_audit'].get('call'):
                            after, audit = apply_patch_once(c['repair_audit']['call']['raw_response'], b['gate'], row)
                            if c['gate'] != after: raise ValueError('C_REVALIDATION_MISMATCH')
                        elif c['gate'] != b['gate']: raise ValueError('C_UNRECORDED_CHANGE')
                        if cached_pair and replay and c != cached_pair['C']: raise ValueError('C_EXACT_REPLAY_MISMATCH')
                        if not cp.exists(): _save(cp, c, identity)
            paired.append({**row, 'A': old, 'B': b, 'C': c, 'repair': repair})
        check_unchanged(freeze)
        complete = all((r['B'] is not None if {'B', 'C'} & set(arms) else True) and
                       (r['C'] is not None if 'C' in arms else True) for r in paired)
        from .reports import generate_reports
        summary = generate_reports(out, paired, arms, freeze, budget.snapshot(), invocation,
                                   synthetic=synthetic, dry_run=dry_run, complete=complete,
                                   replay=replay, offline=offline, resumed=resumed_count, config=cfg)
        write(out / 'protocol.json', protocol())
        files = {str(p.relative_to(out)): digest(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name != '.run.lock'}
        manifest = {'format': VERSION, 'identity': identity, 'freeze': freeze, 'backend_fingerprint': fingerprint,
                    'arms': list(arms), 'dry_run': dry_run, 'synthetic_test': synthetic,
                    'complete': complete, 'files': files, 'historical_outputs_modified': False}
        manifest['manifest_sha256'] = stable_hash(manifest); write(out / 'manifest.json', manifest)
        verify_output(out)
        return {**summary, 'output_dir': str(out)}


def read_rows(path):
    from relive.hypothesis_graph.sources import rows
    return rows(path)
