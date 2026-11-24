"""Uniform A/B0/B1 audit, natural-language comparison and unfilled blind reviews."""
from collections import Counter
import html
import json

from relive.claim_revalidation.offline import write
from relive.stage2_lite_ablation.reports import write_csv
from relive.storage.artifacts import stable_hash
from relive.claim_planner.contract import shape_errors
from .contract import CLAIM

REVIEW_FIELDS = ['atomicity', 'relevance', 'independent_verifiability', 'non_redundancy',
                 'dependency_validity', 'unsupported_specificity', 'placeholder', 'overall_decision', 'reviewer_notes']
STATES = ['ACCEPTED_STRUCTURAL', 'REVIEW_REQUIRED', 'REJECTED', 'BLOCKED_BY_DEPENDENCY']


def metric(rows, arm):
    gs = [r[arm]['gate'] for r in rows if r[arm]]
    audits = [a for g in gs for a in g['claim_audit']]
    kept = [c for g in gs for c in g['claims']]
    errors = Counter(e for a in audits for e in a['errors'])
    warnings = Counter(e for a in audits for e in a['risks'])
    states = Counter(a['state'] for a in audits)
    normalized = [a for g in gs for a in g['normalizations'] if a['scope'] == 'claim_field']
    result = {'samples_audited': len(gs), 'raw_candidates': len(audits),
              'retained_claims': len(kept), 'gate_state_counts': {s: states[s] for s in STATES},
              'generation_coverage': {'numerator': sum(bool(g['claims']) for g in gs), 'denominator': 22},
              'slot_only_samples': sum(not g['claims'] and bool(g['unresolved_slots']) for g in gs),
              'unresolved_slot_count': sum(len(g['unresolved_slots']) for g in gs),
              'required_id_coverage': {'numerator': sum(len(g['covered_required_ids']) for g in gs),
                                       'denominator': sum(q['importance'] == 'REQUIRED' for r in rows for q in r['requirements']),
                                       'semantic_coverage': 'NOT_REVIEWED'},
              'candidate_error_counts': dict(sorted(errors.items())),
              'heuristic_warning_counts': dict(sorted(warnings.items())),
              'raw_canonical_schema_error_count': sum(len(g['raw_schema_errors']) for g in gs),
              'raw_canonical_schema_failure_candidates': sum(bool(shape_errors(a['raw_claim'], CLAIM)) for a in audits),
              'normalized_candidate_schema_failures': sum(any(e.startswith('CLAIM_SCHEMA:') for e in a['errors']) for a in audits),
              'invalid_type_candidates': errors['CLAIM_SCHEMA:$.type:ENUM'],
              'root_failure_samples': sum(bool(g['root_errors']) for g in gs),
              'normalization_counts': dict(sorted(Counter(a['code'] + ':' + a['field'] for a in normalized).items())),
              'exact_duplicate_candidates': errors['EXACT_DUPLICATE_TEXT'],
              'cycle_node_count': errors['DEPENDENCY_CYCLE'], 'unsupported_numeric_binding_count': errors['NUMERIC_EXECUTION_BINDING_FORBIDDEN'],
              'retained_DAG_and_references': 'PASS_BY_CONSTRUCTION',
              'unresolved_target_samples': sum(g['plan_status'] == 'UNRESOLVED_TARGET_REFERENCE' for g in gs),
              'potential_spatial_types': dict(sorted(Counter(c['type'] for c in kept if c['type'] in ('OBJECT', 'ATTRIBUTE', 'STATIC_RELATION')).items())),
              'spatial_admission': 'NOT_PERFORMED', 'semantic_quality': 'NOT_REVIEWED',
              'human_approved_claim_rate': None, 'claims_actually_reviewed': 0,
              'graph_level_sufficiency_rate': None, 'graphs_actually_reviewed': 0,
              'human_invalid_dependency_edges': None}
    if arm in ('A', 'B0'):
        old = [r[arm]['historical'] for r in rows]
        gates = [o if arm == 'A' else o['gate'] for o in old]
        result['historical_retained_unchanged'] = sum(len(g['claims']) for g in gates)
        result['historical_states_unchanged'] = dict(sorted(Counter(a['state'] for g in gates for a in g['claim_audit']).items()))
        result['audit_scope'] = 'V4.1 audit on copies; historical admission never rewritten'
    return result


def measured_cost(records):
    tokens = [(r.get('generation_metadata') or {}).get('generated_token_count') for r in records]
    times = [r.get('recorded_latency_seconds') for r in records]
    known = bool(tokens) and all(type(x) is int for x in tokens)
    return {'recorded_calls': len(records), 'output_tokens': sum(tokens) if known else None,
            'mean_output_tokens': sum(tokens) / len(tokens) if known else None,
            'latency_seconds': sum(times) if times and all(type(t) in (int, float) for t in times) else None,
            'input_tokens': None, 'cost_estimate': None, 'human_approved_claim_cost': None}


def publish(out, rows, freeze, arms, usage, *, synthetic, dry_run, complete, replay, resumed, config, backend_fingerprint):
    write(out / 'smoke22_results.jsonl', rows, jsonl=True)
    normalized = []; audits = []
    for r in rows:
        for arm in arms:
            g = r[arm]['gate'] if r[arm] else None
            audits.append({'sample_key': r['sample_key'], 'arm': arm, 'gate': g})
            for c in g['claims'] if g else []: normalized.append({'sample_key': r['sample_key'], 'arm': arm, **c})
    write(out / 'normalized_claims.jsonl', normalized, jsonl=True)
    write(out / 'quality_gate_audit.jsonl', audits, jsonl=True)
    rawdir = out / 'raw_completions'; rawdir.mkdir()
    for r in rows:
        for arm in arms:
            if not r[arm]: continue
            source = r[arm]['historical'] if arm != 'B1' else r['B1']
            write(rawdir / (r['sample_key'] + '.' + arm + '.json'),
                  {'sample_key': r['sample_key'], 'arm': arm, 'raw_response': source['raw_response'],
                   'raw_response_sha256': stable_hash(source['raw_response'])})
    b1 = [r['B1'] for r in rows if r['B1']]
    a_cost = measured_cost([r['A']['historical'] for r in rows]); b_cost = measured_cost([r['B0']['historical'] for r in rows])
    old_metrics = freeze['baseline_metadata'].get('historical_metrics') or {}
    a_cost['latency_seconds'] = old_metrics.get('latency_seconds')
    a_cost['historical_new_calls'] = (old_metrics.get('usage') or {}).get('new_calls')
    a_cost['new_calls_this_invocation'] = b_cost['new_calls_this_invocation'] = 0
    b1_cost = {**measured_cost(b1), 'new_calls_this_invocation': usage['new_calls']}
    def scientific(fp):
        while isinstance(fp, dict) and 'inner' in fp: fp = fp['inner']
        return fp.get('scientific_identity') if isinstance(fp, dict) else None
    current_identity = scientific(backend_fingerprint)
    old_a = scientific((freeze['baseline_metadata'].get('historical_binding') or {}).get('backend_fingerprint'))
    old_b = scientific(freeze['B0_metadata'].get('backend_fingerprint'))
    comparability = {name: 'UNKNOWN' if old is None or current_identity is None else 'MATCH' if old == current_identity else 'DIFFERENT'
                     for name, old in [('A_vs_B1', old_a), ('B0_vs_B1', old_b)]}
    metrics = {'format': 'relive-stage2-single-pass-v4.1',
               'status': 'DRY_RUN' if dry_run else 'PARTIAL' if not complete else 'SYNTHETIC_TEST_ONLY' if synthetic else 'COMPLETE',
               'B1_execution': 'NOT_RUN' if dry_run or 'B1' not in arms else 'SYNTHETIC_TEST_ONLY' if synthetic else 'EXECUTED',
               'sample_count': 22, 'requirement_count': 100, 'arms': {arm: metric(rows, arm) for arm in arms},
               'cost': {'A': a_cost, 'B0': b_cost, 'B1': b1_cost}, 'usage_this_invocation': usage,
               'normal_generation_calls_this_invocation': usage['new_calls'], 'repair_calls': 0, 'critic_calls': 0,
               'real_new_model_calls': 0 if synthetic else usage['new_calls'],
               'synthetic_adapter_new_calls': usage['new_calls'] if synthetic else 0, 'synthetic_test': synthetic,
               'image_reads': 0, 'gt_used': False, 'certificate_writes': 0, 'new_verified_count': 0,
               'requirements_sha256': freeze['requirements_sha256'], 'stage1_requirements_unchanged': True,
               'historical_inputs_unchanged': True, 'semantic_quality': 'NOT_REVIEWED',
               'B1_replaces_B0': 'NOT_ESTABLISHED_WITHOUT_REAL_RUN_AND_HUMAN_REVIEW',
               'resumed_checkpoint_count': resumed, 'model_config': config,
               'decoding_parameters': {'explicit': (config or {}).get('backend', {}).get('generation'),
                                       'temperature': (config or {}).get('backend', {}).get('generation', {}).get('temperature'),
                                       'top_p': (config or {}).get('backend', {}).get('generation', {}).get('top_p'),
                                       'random_seed': None, 'seed_control': 'not exposed by frozen adapter'},
               'A_model_metadata': freeze['baseline_metadata'], 'B0_model_metadata': freeze['B0_metadata'],
               'environment_comparability': comparability,
               'cost_notes': 'A/B0 historical completions only; null means unavailable; no billing estimates or semantic success inferred'}
    write(out / 'ablation_A_B0_B1_metrics.json', metrics)
    write(out / 'gt_isolation_audit.json', {'status': 'PASS', 'basis': 'public allowlist plus sentinel regression tests',
          'generator_fields': ['question', 'qa_type', 'requirements', 'question-derived lexical decision targets'],
          'historical_claims_in_prompt': False, 'gt_used': False, 'image_reads': 0, 'repair_calls': 0,
          'critic_calls': 0, 'certificate_writes': 0, 'new_verified_count': 0})
    write(out / 'cache_replay_audit.json', {'status': 'PASS' if replay and complete and usage['new_calls'] == 0 else 'NOT_RUN',
          'exact_result_comparison': 'PASS' if replay and complete else 'NOT_RUN', 'new_model_calls': usage['new_calls'],
          'cache_hits': usage['cache_hits'], 'repair_calls': 0, 'critic_calls': 0})
    _reviews(out, rows, arms, freeze)
    _comparison(out, rows, arms, metrics)
    return metrics


def _reviews(out, rows, arms, freeze):
    claims = []; graphs = []; mapping = []
    for r in rows:
        for arm in arms:
            g = r[arm]['gate'] if r[arm] else None
            graph_id = 'G' + stable_hash([freeze['public_inputs_sha256'], r['sample_key'], arm])[:20]
            graphs.append({'blind_id': graph_id, 'question': r['question'], 'requirements': json.dumps(r['requirements']),
                           'graph': json.dumps(g['normalized_graph'], ensure_ascii=False) if g else 'NOT_EXECUTED',
                           'answer_sufficiency': '', 'dependency_validity': '', 'invalid_dependency_edges': '',
                           'reviewer_notes': '', 'review_status': 'NOT_REVIEWED'})
            mapping.append({'blind_id': graph_id, 'kind': 'graph', 'sample_key': r['sample_key'], 'arm': arm})
            for a in g['claim_audit'] if g else []:
                c = a['raw_claim'] if isinstance(a['raw_claim'], dict) else {'text': str(a['raw_claim'])}
                bid = 'Q' + stable_hash([graph_id, a['candidate_index']])[:20]
                claims.append({'blind_id': bid, 'question': r['question'], 'claim_id': c.get('claim_id'),
                               'claim_text': c.get('text'), 'claim_type': c.get('type'),
                               'requires': json.dumps(c.get('requires', None)), 'alternative_to': json.dumps(c.get('alternative_to', None)),
                               **{k: '' for k in REVIEW_FIELDS}, 'review_status': 'NOT_REVIEWED'})
                mapping.append({'blind_id': bid, 'kind': 'claim', 'sample_key': r['sample_key'], 'arm': arm,
                                'candidate_index': a['candidate_index'], 'gate_state': a['state']})
    claims.sort(key=lambda x: x['blind_id']); graphs.sort(key=lambda x: x['blind_id'])
    write_csv(out / 'blind_claim_review.csv', claims, ['blind_id', 'question', 'claim_id', 'claim_text', 'claim_type', 'requires',
                                                     'alternative_to', *REVIEW_FIELDS, 'review_status'])
    write_csv(out / 'blind_graph_review.csv', graphs, ['blind_id', 'question', 'requirements', 'graph', 'answer_sufficiency',
                                                     'dependency_validity', 'invalid_dependency_edges', 'reviewer_notes', 'review_status'])
    write(out / 'blind_review_mapping.json', mapping)
    write(out / 'human_review_policy.json', {'values': ['PASS', 'FAIL', 'UNCERTAIN'], 'empty': 'NOT_REVIEWED',
          'human_approved_claim_rate': 'overall_decision PASS / actually reviewed claims',
          'graph_level_sufficiency_rate': 'answer_sufficiency PASS / actually reviewed graphs',
          'review_count': 0, 'quality_inference_from_code_warnings': 'FORBIDDEN'})


def _comparison(out, rows, arms, metrics):
    esc = lambda x: html.escape(str(x))
    page = ['<!doctype html><meta charset="utf-8"><title>ReliVE V4.1 A/B0/B1</title>',
            '<style>body{font:16px/1.5 system-ui;max-width:1550px;margin:30px auto;padding:20px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f8;padding:12px}section{border-top:2px solid #aaa;margin:30px 0}.arms{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px}li{margin:12px 0}</style>',
            '<h1>V4.1 single-pass hypotheses</h1><p>UNVERIFIED / NOT_SEARCHED. Semantic quality: NOT_REVIEWED. Repair and critic calls: 0.</p>',
            '<p>' + ('B1 SYNTHETIC TEST ONLY' if metrics['synthetic_test'] else 'B1 ' + metrics['B1_execution']) + '</p>',
            '<details><summary>Metrics and inference costs</summary><pre>' + esc(json.dumps(metrics, ensure_ascii=False, indent=2)) + '</pre></details>']
    md = ['# V4.1 A/B0/B1', '', f"Execution: {metrics['B1_execution']}; semantic quality: NOT_REVIEWED.",
          'All claims remain UNVERIFIED / NOT_SEARCHED. Historical admission is unchanged; common audit acts on copies.', '']
    for i, r in enumerate(rows, 1):
        page += [f'<section><h2>Case {i:02d} · source {r["source_record_index"]} · {esc(r["qa_type"])}</h2>',
                 '<h3>Original question</h3><pre>' + esc(r['question']) + '</pre>',
                 '<h3>Frozen requirements</h3><pre>' + esc(json.dumps(r['requirements'], ensure_ascii=False, indent=2)) + '</pre><div class="arms">']
        md += [f'## Case {i:02d} / source {r["source_record_index"]}', '', r['question'], '']
        for arm in arms:
            g = r[arm]['gate'] if r[arm] else None
            page.append('<article><h3>' + arm + '</h3>'); md += [f'### {arm}', '']
            if g is None:
                page.append('<p>NOT_EXECUTED</p></article>'); md.append('NOT_EXECUTED'); continue
            retained = [c['claim_id'] for c in g['claims']]
            page.append('<p>Status: ' + esc(g['plan_status']) + '; common-audit retained IDs: ' + esc(retained) + '</p>')
            md.append('Common-audit retained IDs: ' + str(retained))
            source = r[arm]['historical'] if arm != 'B1' else r['B1']
            cost_line = ('Output tokens: ' + str((source.get('generation_metadata') or {}).get('generated_token_count'))
                         + '; recorded latency seconds: ' + str(source.get('recorded_latency_seconds'))
                         + '; missing values mean unavailable, not zero.')
            page.append('<p>' + esc(cost_line) + '</p>'); md += [cost_line, '']
            if arm != 'B1':
                old = r[arm]['historical']; old_gate = old if arm == 'A' else old['gate']
                page.append('<p>Historical retained IDs (unchanged): ' + esc([c['claim_id'] for c in old_gate['claims']]) + '</p>')
            page.append('<ol>')
            for a in g['claim_audit']:
                c = a['raw_claim'] if isinstance(a['raw_claim'], dict) else {'text': str(a['raw_claim'])}
                line = f"{c.get('claim_id')}: {c.get('text')} | type={c.get('type')} | requires={c.get('requires', 'UNDECLARED')} | alternatives={c.get('alternative_to', 'UNDECLARED')} | {a['state']}"
                page.append('<li>' + esc(line) + '<pre>' + esc(json.dumps({'errors': a['errors'], 'warnings': a['risks'],
                              'field_diagnostic': a['field_diagnostic']}, ensure_ascii=False)) + '</pre></li>')
                md.append('- ' + line)
            page += ['</ol><details><summary>Normalized graph, slots, coverage and audit</summary><pre>'
                     + esc(json.dumps(g, ensure_ascii=False, indent=2)) + '</pre></details>']
            raw = r[arm]['historical']['raw_response'] if arm != 'B1' else r['B1']['raw_response']
            page.append('<details><summary>Exact raw completion</summary><pre>' + esc(raw) + '</pre></details></article>')
            md += ['', 'Slots: ' + json.dumps(g['unresolved_slots'], ensure_ascii=False), '']
        page.append('</div></section>')
    with (out / 'ablation_A_B0_B1.html').open('x', encoding='utf-8') as stream: stream.write('\n'.join(page))
    with (out / 'ablation_A_B0_B1.md').open('x', encoding='utf-8') as stream: stream.write('\n'.join(md))
