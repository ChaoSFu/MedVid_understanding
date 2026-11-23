"""Zero-call V2.1 runner. No backend, inference, cache writer, PIL or GT loader."""
from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
import re

from relive.claim_planner import contract as v2
from relive.storage.artifacts import canonical_json, stable_hash
from relive.v2.task_selection import strict_json_loads
from .validation import policy, validate_sample
from .normalization import VERSION

class RevalidationError(ValueError):
    pass

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_json(path):
    return strict_json_loads(Path(path).read_text(encoding='utf-8'), error_code='OFFLINE_JSON_INVALID')

def write(path, value, *, jsonl=False):
    body = ''.join(canonical_json(v) + '\n' for v in value) if jsonl else canonical_json(value) + '\n'
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(body)

def cache_check(row, cache_root=None):
    call = row.get('call') or {}
    ref = call.get('raw_response_ref')
    path = Path(ref) if isinstance(ref, str) else None
    if path and not re.fullmatch(r'[a-f0-9]{64}\.json', path.name):
        raise RevalidationError('RAW_CACHE_REFERENCE_INVALID')
    if cache_root:
        if not path or not re.fullmatch(r'[a-f0-9]{64}\.json', path.name) or not re.fullmatch(r'[a-f0-9]{64}', str(call.get('cache_key', ''))):
            raise RevalidationError('RAW_CACHE_REFERENCE_INVALID')
        path = Path(cache_root) / 'raw' / call['cache_key'] / path.name
    if not path or not path.is_file():
        if cache_root:
            raise RevalidationError('EXPLICIT_RAW_CACHE_MISSING')
        return {'status': 'NOT_AVAILABLE', 'source': 'embedded raw_response used', 'cache_writes': 0}, None
    record = read_json(path)
    if path.stem != stable_hash(record) or record.get('raw_text') != row['raw_response'] or record.get('cache_key') != call.get('cache_key') or stable_hash(record.get('identity')) != call.get('cache_key'):
        raise RevalidationError('RAW_CACHE_BINDING_MISMATCH')
    binding = record.get('identity', {}).get('request', {}).get('planner_binding', {})
    if any(binding.get(k) != row[k] for k in ('question_sha256', 'requirements_sha256')):
        raise RevalidationError('RAW_CACHE_INPUT_MISMATCH')
    return {'status': 'MATCH', 'path': str(path.resolve()), 'sha256': digest(path), 'cache_writes': 0}, path

def source_rows(path, cache_root=None):
    data = [strict_json_loads(s, error_code='OFFLINE_JSONL_INVALID') for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]
    if [r.get('source_record_index') for r in data] != v2.SMOKE22 or len({r.get('sample_key') for r in data}) != 22:
        raise RevalidationError('SMOKE_SELECTION_INVALID')
    if sum(len(r['requirements']) for r in data) != 100:
        raise RevalidationError('REQUIREMENT_COUNT_INVALID')
    if len({r.get('synthetic_test') for r in data}) != 1 or any(type(r.get('synthetic_test')) is not bool for r in data):
        raise RevalidationError('SYNTHETIC_BINDING_INVALID')
    parsed, caches, watched = [], [], {}
    forbidden = {'struc_info', 'RC_info', 'answer', 'gt_answer', 'gt_bbox', 'gt_spans', 'gt_score', 'oracle'}
    for r in data:
        if set(r) & forbidden:
            raise RevalidationError('GT_FIELDS_FORBIDDEN')
        if r.get('format') != v2.VERSION or not isinstance(r.get('question'), str) or r['question_sha256'] != stable_hash(r['question']) or r['requirements_sha256'] != stable_hash(r['requirements']):
            raise RevalidationError('FROZEN_INPUT_HASH_INVALID')
        if not isinstance(r.get('raw_response'), str) or r.get('image_reads') != 0 or r.get('input_modality') != 'TEXT_ONLY':
            raise RevalidationError('FROZEN_RAW_RESPONSE_INVALID')
        # Reproduce the historical rejection, without changing its interpretation.
        old = v2.validate(r['raw_response'], r)
        if any(old[k] != r[k] for k in ('generation_status', 'claims', 'rejected_candidates')):
            raise RevalidationError('V2_RESULT_REPRODUCTION_MISMATCH')
        raw = strict_json_loads(r['raw_response'], error_code='RAW_COMPLETION_PARSE_FAILURE')
        if not isinstance(raw, dict) or not isinstance(raw.get('claims'), list):
            raise RevalidationError('RAW_COMPLETION_SCHEMA_INVALID')
        if any(not isinstance(c, dict) or not isinstance(c.get('text'), str) for c in raw['claims']):
            raise RevalidationError('RAW_CANDIDATE_IDENTITY_INVALID')
        parsed.append(raw)
        audit, cache_path = cache_check(r, cache_root)
        caches.append(audit)
        if cache_path: watched[str(cache_path.resolve())] = digest(cache_path)
    if sum(len(o['claims']) for o in parsed) != 84 or sum(len(r['rejected_candidates']) for r in data) != 84 or any(r['claims'] for r in data):
        raise RevalidationError('FROZEN_CANDIDATE_COUNT_OR_BASELINE_INVALID')
    expected = Counter({'MODEL_OR_SCHEMA_FAILURE': 18, 'UNRESOLVED_TARGET_REFERENCE': 4})
    if Counter(r['generation_status'] for r in data) != expected:
        raise RevalidationError('BASELINE_STATUS_COUNTS_INVALID')
    return data, parsed, caches, watched

def spatial_category(record):
    if record['state'] != 'ACCEPTED': return None
    c = record['normalized_candidate']; ev = c['evidence_requirement']
    if ev['modality'] not in ('SINGLE_FRAME', 'MULTI_REGION_SINGLE_FRAME'): return None
    t = ev['temporal_requirement']
    if t is not None and t['min_observations'] != 1: return None
    if c['type'] in ('OBJECT', 'ATTRIBUTE'): return c['type']
    if c['type'] == 'RELATION' and c['proposition']['predicate'] in ('STATIC_RELATION', 'CONTACT'): return 'STATIC_RELATION'
    return None

def aggregate(samples, source):
    records = [r for s in samples for r in s['candidate_records']]
    states = Counter(r['state'] for r in records)
    required = {(s['sample_key'], r['requirement_id']) for s in samples for r in s['requirements'] if r['importance'] == 'REQUIRED'}
    covered = {(s['sample_key'], rid) for s in samples for r in s['candidate_records'] if r['state'] == 'ACCEPTED' for rid in r['normalized_candidate']['requirement_ids']}
    by_type = {k: {'raw_candidates': 0, **{state: 0 for state in ('ACCEPTED', 'REJECTED', 'BLOCKED_BY_DEPENDENCY', 'REVIEW_REQUIRED')}} for k in ('OBJECT', 'ATTRIBUTE', 'RELATION', 'ACTION', 'HIGH_LEVEL')}
    reasons, unique = Counter(), Counter()
    for r in records:
        kind = r['raw_candidate']['type']
        if kind in by_type:
            by_type[kind]['raw_candidates'] += 1; by_type[kind][r['state']] += 1
        codes = sorted(set(r['schema_errors'] + r['contract_errors'] + r['blocking_semantic_reviews'] + r['dependency_errors'] + r['dependency_semantic_reviews']))
        reasons.update(codes)
        if r['blocked_by']: reasons['DEPENDENCY_NOT_ADMITTED'] += 1
        unique.update([r['state']])
    eligible = Counter(filter(None, (spatial_category(r) for r in records)))
    old_errors = Counter(e for s in source for c in s['rejected_candidates'] for e in c['errors'])
    return {'protocol_version': VERSION, 'sample_count': len(samples), 'raw_candidate_count': len(records),
            'accepted_claim_count': states['ACCEPTED'], 'not_exported_claim_count': len(records) - states['ACCEPTED'],
            'candidate_state_counts': {k: states[k] for k in ('ACCEPTED', 'REJECTED', 'BLOCKED_BY_DEPENDENCY', 'REVIEW_REQUIRED')},
            'structural_admission_count': sum(r['structural_admission'] == 'PASS' for r in records),
            'local_contract_pass_count': sum(r['contract_validity'] == 'PASS' for r in records),
            'partial_sample_count': sum(s['generation_status'] == 'PARTIAL_GENERATED' for s in samples),
            'sample_status_counts': dict(sorted(Counter(s['generation_status'] for s in samples).items())),
            'unresolved_rc_sample_count': sum(s['generation_status'] == 'UNRESOLVED_TARGET_REFERENCE' for s in samples),
            'required_requirement_coverage': {'numerator': len(required & covered), 'denominator': len(required), 'value': len(required & covered) / len(required) if required else None, 'meaning': 'admitted hypotheses referencing required IDs; semantic coverage not proven'},
            'accepted_by_type': {k: v['ACCEPTED'] for k, v in by_type.items()}, 'candidate_states_by_type': by_type,
            'spatial_experiment_eligibility': {'total': sum(eligible.values()), 'by_category': {k: eligible[k] for k in ('OBJECT', 'ATTRIBUTE', 'STATIC_RELATION')},
                                              'meaning': 'unverified, unbound candidates with admissible static contracts only; no grounding or experimental admission performed'},
            'unadmitted_reason_counts': dict(sorted(reasons.items())), 'old_v2_rejection_reason_counts': dict(sorted(old_errors.items())),
            'new_model_calls': 0, 'image_reads': 0, 'gt_used': False, 'cache_writes': 0,
            'certificate_writes': 0, 'new_verified_count': 0, 'requirements_unchanged': True,
            'claim_texts_unchanged': True, 'raw_responses_unchanged': True,
            'semantic_review_status': 'REVIEW_REQUIRED', 'synthetic_test': source[0]['synthetic_test'],
            'v2_historical_model_calls': None if source[0]['synthetic_test'] else 22,
            'historical_call_count_source': 'task-declared frozen V2 smoke summary; not inferred as fresh calls from an embedded cache reference'}

def manifest_check(root):
    root = Path(root); m = read_json(root / 'manifest.json')
    if stable_hash({k: v for k, v in m.items() if k != 'manifest_content_sha256'}) != m['manifest_content_sha256']:
        raise RevalidationError('DERIVED_MANIFEST_INVALID')
    for name, h in m['artifact_sha256'].items():
        if Path(name).name != name or digest(root / name) != h: raise RevalidationError('DERIVED_ARTIFACT_INVALID')
    return m

def implementation_hash():
    root = Path(__file__).parent
    files = sorted(root.glob('*.py'))
    files += [Path(v2.__file__)]
    return stable_hash({p.name if p.parent == root else 'frozen_v2/' + p.name: digest(p) for p in files})

def revalidate(*, input_path, output_dir, offline, no_model_calls, v2_metrics=None, v2_html=None, cache_root=None, replay_from=None, expected_input_sha256=None):
    if offline is not True or no_model_calls is not True:
        raise RevalidationError('EXPLICIT_OFFLINE_ZERO_CALL_FLAGS_REQUIRED')
    source_path = Path(input_path).resolve()
    watch = {str(source_path): digest(source_path)}
    if expected_input_sha256 and watch[str(source_path)] != expected_input_sha256:
        raise RevalidationError('FROZEN_SOURCE_SHA_MISMATCH')
    data, raw, caches, cache_files = source_rows(source_path, cache_root)
    watch.update(cache_files)
    optional = {}
    for name, supplied, filename in (('v2_metrics', v2_metrics, 'stage2_v2_metrics.json'), ('v2_html', v2_html, 'stage2_v1_vs_v2_smoke22.html')):
        p = Path(supplied).resolve() if supplied else source_path.parent / filename
        if supplied and not p.is_file(): raise RevalidationError('EXPLICIT_BASELINE_ARTIFACT_MISSING')
        if p.is_file():
            watch[str(p)] = digest(p); optional[name] = {'path': str(p), 'sha256': digest(p)}
            if name == 'v2_metrics':
                old_metrics = read_json(p)
                if old_metrics.get('generation_coverage', {}).get('numerator') != 0 or old_metrics.get('generation_coverage', {}).get('denominator') != 22:
                    raise RevalidationError('BASELINE_METRICS_MISMATCH')
    samples = []
    for row, completion, cache in zip(data, raw, caches):
        result = validate_sample(completion, row)
        count = sum(r['state'] == 'ACCEPTED' for r in result['candidate_records'])
        rc = row['qa_type'].startswith('region_caption')
        if rc and (completion['planning_status'] != 'UNRESOLVED_TARGET_REFERENCE' or completion['claims']):
            raise RevalidationError('FROZEN_RC_TARGET_STATUS_INVALID')
        for record in result['candidate_records']:
            c = record['normalized_candidate']; original = record['raw_candidate']
            if c['text'] != original['text'] or c['dependencies'] != original['dependencies'] or c['type'] != original['type']:
                raise RevalidationError('PROTECTED_CANDIDATE_CHANGED')
            record['source_ref'] = {'input_sha256': watch[str(source_path)], 'sample_key': row['sample_key'], 'candidate_index': record['candidate_index'],
                                    'raw_response_sha256': stable_hash(row['raw_response'])}
            record['old_v2_rejection'] = next((c['errors'] for c in row['rejected_candidates'] if c.get('candidate_index') == record['candidate_index']), [])
            record['spatial_eligibility_category'] = spatial_category(record)
        samples.append({'format': VERSION, **{k: row[k] for k in ('sample_key', 'source_record_index', 'qa_type', 'question', 'question_sha256', 'requirements', 'requirements_sha256')},
                        'raw_response': row['raw_response'], 'raw_response_sha256': stable_hash(row['raw_response']),
                        'old_v2_status': row['generation_status'], 'old_v2_candidates': completion['claims'], 'cache_binding_audit': cache,
                        **result, 'generation_status': 'UNRESOLVED_TARGET_REFERENCE' if rc else 'GENERATED' if count == len(completion['claims']) and count else 'PARTIAL_GENERATED' if count else 'NO_ADMITTED_CLAIMS',
                        'requirements_unchanged': True, 'new_model_calls': 0, 'image_reads': 0, 'gt_used': False, 'certificate_writes': 0, 'new_verified_count': 0})
    records = [r for s in samples for r in s['candidate_records']]
    if len(records) != 84 or len({(r['source_ref']['sample_key'], r['candidate_index']) for r in records}) != 84:
        raise RevalidationError('CANDIDATE_COVERAGE_INVALID')
    stats = aggregate(samples, data)
    identity = {'input_sha256': watch[str(source_path)], 'policy_sha256': policy()['policy_sha256'],
                'implementation_sha256': implementation_hash(), 'baseline_artifacts': optional}
    if replay_from:
        m = manifest_check(replay_from)
        if m['identity'] != identity: raise RevalidationError('OFFLINE_REPLAY_IDENTITY_MISMATCH')
    out = Path(output_dir).resolve()
    if out.exists(): raise RevalidationError('NEW_OUTPUT_DIRECTORY_REQUIRED')
    if cache_root and out.is_relative_to(Path(cache_root).resolve()): raise RevalidationError('OUTPUT_INSIDE_FROZEN_CACHE_FORBIDDEN')
    for p in cache_files:
        if out.is_relative_to(Path(p).parent): raise RevalidationError('OUTPUT_INSIDE_FROZEN_CACHE_FORBIDDEN')
        for ancestor in Path(p).parents:
            if ancestor.name == 'raw' and out.is_relative_to(ancestor.parent):
                raise RevalidationError('OUTPUT_INSIDE_FROZEN_CACHE_FORBIDDEN')
    out.mkdir(parents=True)
    write(out / 'stage2_v2_1_revalidated.jsonl', samples, jsonl=True)
    admitted = [{'sample_key': s['sample_key'], 'source_record_index': s['source_record_index'], 'admission_status': r['state'],
                 'structural_admission': r['structural_admission'], 'contract_validity': r['contract_validity'],
                 'semantic_review_status': r['semantic_review_status'], 'source_ref': r['source_ref'], 'claim': r['normalized_candidate']}
                for s in samples for r in s['candidate_records'] if r['state'] == 'ACCEPTED']
    write(out / 'stage2_v2_1_claims.jsonl', admitted, jsonl=True)
    write(out / 'stage2_v2_1_metrics.json', stats)
    write(out / 'stage2_v2_1_rejection_audit.json', {'candidate_count': 84, 'not_exported': [r for r in records if r['state'] != 'ACCEPTED']})
    write(out / 'stage2_v2_1_normalization_audit.json', {'policy_sha256': identity['policy_sha256'], 'rule_definitions': policy()['normalization_rules'],
        'samples': [{'sample_key': s['sample_key'], 'changes': s['normalizations']} for s in samples]})
    write(out / 'policy.json', policy())
    from .report import render
    render(out, samples, stats)
    main_names = ['stage2_v2_1_revalidated.jsonl', 'stage2_v2_1_claims.jsonl', 'stage2_v2_1_metrics.json', 'stage2_v2_1_rejection_audit.json', 'stage2_v2_1_normalization_audit.json', 'policy.json', 'stage2_v2_vs_v2_1.html', 'stage2_v2_1_summary.md']
    if replay_from and any((out / n).read_bytes() != (Path(replay_from) / n).read_bytes() for n in main_names):
        raise RevalidationError('OFFLINE_REPLAY_BYTES_MISMATCH')
    if any(digest(p) != h for p, h in watch.items()): raise RevalidationError('SOURCE_ARTIFACT_CHANGED')
    receipt = {'status': 'PASS', 'mode': 'offline-replay' if replay_from else 'offline-revalidation',
               'candidate_records': 84, 'new_model_calls': 0, 'image_reads': 0, 'gt_used': False, 'cache_writes': 0,
               'certificate_writes': 0, 'new_verified_count': 0, 'requirements_unchanged': True,
               'raw_responses_unchanged': True, 'claim_texts_unchanged': True, 'dependencies_unchanged': True,
               'watched_source_sha256': watch, 'exact_byte_replay': 'PASS' if replay_from else 'NOT_RUN',
               'cache_matches': sum(c['status'] == 'MATCH' for c in caches), 'embedded_raw_only': sum(c['status'] == 'NOT_AVAILABLE' for c in caches)}
    write(out / 'offline_revalidation_audit.json', receipt)
    names = main_names + ['offline_revalidation_audit.json']
    m = {'format': 'relive-stage2-v2.1-offline-manifest-v1', 'identity': identity,
         'artifact_sha256': {n: digest(out / n) for n in names}}
    m['manifest_content_sha256'] = stable_hash(m); write(out / 'manifest.json', m)
    return {'status': 'OFFLINE_REVALIDATION_COMPLETE', 'output_dir': str(out), **stats}
