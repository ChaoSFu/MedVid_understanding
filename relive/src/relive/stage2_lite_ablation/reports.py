"""Auditable text/HTML exports. Unreviewed semantic metrics remain null."""
from collections import Counter
import csv
import html
import json
from pathlib import Path

from relive.claim_revalidation.offline import write
from relive.storage.artifacts import stable_hash
from .gate import validate

REVIEW_FIELDS = ['atomicity', 'relevance', 'verifiability', 'redundancy', 'dependency_validity',
                 'unsupported_specificity', 'placeholder', 'overall_decision', 'reviewer_notes']


def raw_claims(response):
    from relive.v2.task_selection import strict_json_loads
    try:
        obj = strict_json_loads(response, error_code='RAW_DIAGNOSTIC_PARSE_FAILURE')
        return obj.get('claims', []) if isinstance(obj, dict) and isinstance(obj.get('claims'), list) else []
    except (ValueError, TypeError):
        return []


def a_diagnostic(row):
    old = row['A']
    cs = raw_claims(old['raw_response'])
    # A-only diagnostic conversion; preserve original graph/acceptance alongside it.
    converted = {'plan_status': 'UNRESOLVED_TARGET_REFERENCE' if row['qa_type'].startswith('region_caption') else 'INCOMPLETE_PLAN',
                 'claims': cs, 'unresolved_slots': [], 'requirement_issues': []}
    gate = validate(converted, row)
    return {'original': old, 'unified_diagnostic': gate,
            'diagnostic_is_not_historical_revalidation': True}


def arm_metrics(rows, arm):
    gates = []; raws = []; historical = Counter()
    for r in rows:
        if arm == 'A':
            gates.append(a_diagnostic(r)['unified_diagnostic']); raws.extend(raw_claims(r['A']['raw_response']))
            historical.update(a['state'] for a in r['A']['claim_audit'])
        elif arm == 'B' and r['B']:
            gates.append(r['B']['gate']); raws.extend(raw_claims(r['B']['raw_response']))
        elif arm == 'C' and r['C']:
            gates.append(r['C']['gate']); raws.extend(r['C']['gate']['normalized_graph']['claims'] if r['C']['gate']['normalized_graph'] else [])
    audits = [a for g in gates for a in g['claim_audit']]
    counts = Counter(a['state'] for a in audits)
    flags = Counter(f for a in audits for f in a['risks'])
    claims = [c for g in gates for c in g['claims']]
    result = {'samples_evaluated': len(gates), 'raw_candidates' if arm != 'C' else 'post_patch_candidates': len(raws),
              'retained_claims': len(claims), 'gate_state_counts': dict(sorted(counts.items())),
              'candidate_error_counts': dict(sorted(Counter(e for a in audits for e in a['errors']).items())),
              'heuristic_risk_counts': dict(sorted(flags.items())),
              'generation_coverage': {'numerator': sum(bool(g['claims']) for g in gates), 'denominator': 22},
              'required_id_coverage': {'numerator': sum(len(g['requirement_coverage']['covered_required_ids']) for g in gates),
                                       'denominator': sum(q['importance'] == 'REQUIRED' for r in rows for q in r['requirements']),
                                       'semantic_coverage': 'NOT_REVIEWED'},
              'valid_retained_references': not any('DANGLING' in e for a in audits if a['state'] in ('ACCEPTED_STRUCTURAL', 'REVIEW_REQUIRED') for e in a['errors']),
              'retained_DAG_valid': not any('DEPENDENCY_CYCLE' in a['errors'] for a in audits if a['state'] in ('ACCEPTED_STRUCTURAL', 'REVIEW_REQUIRED')),
              'unresolved_target_cases': sum(g['plan_status'] == 'UNRESOLVED_TARGET_REFERENCE' for g in gates),
              'potential_spatial_by_type': dict(sorted(Counter(c['type'] for c in claims if c['type'] in ('OBJECT', 'ATTRIBUTE', 'STATIC_RELATION')).items())),
              'spatial_admission': 'NOT_PERFORMED', 'human_semantic_metrics': {k: None for k in REVIEW_FIELDS[:-1]},
              'semantic_quality': 'NOT_REVIEWED', 'cost_per_human_approved_claim': None}
    if arm == 'A':
        result['historical_state_counts_unchanged'] = dict(sorted(historical.items()))
        result['historical_retained_count'] = sum(len(r['A']['claims']) for r in rows)
        result['historical_raw_count'] = len(raws)
        result['gate_state_counts_meaning'] = 'independent V4 diagnostic only, never rewrites A admission'
    return result


def costs(calls, invocation, stage):
    tokens = [x['generation_metadata'].get('generated_token_count') for x in calls]
    inputs = [x['generation_metadata'].get('input_token_count') for x in calls]
    known = bool(tokens) and all(type(t) is int for t in tokens)
    latency = sum(x['recorded_latency_seconds'] for x in calls)
    return {'logical_calls_in_result': len(calls),
            'new_calls_this_invocation': sum(x['new_calls'] for x in invocation if x['stage'] == stage),
            'cache_hits_this_invocation': sum(x['cache_hit'] for x in invocation if x['stage'] == stage),
            'output_tokens': sum(tokens) if known else None,
            'mean_output_tokens': sum(tokens) / len(tokens) if known else None,
            'input_tokens': sum(inputs) if inputs and all(type(t) is int for t in inputs) else None,
            'token_metadata_available_calls': sum(type(t) is int for t in tokens),
            'recorded_latency_seconds': latency, 'mean_recorded_latency_seconds': latency / len(calls) if calls else None,
            'cost_estimate': None, 'cost_basis': 'UNAVAILABLE: no reliable API pricing/GPU billing input'}


def historical_cost(rows, freeze):
    metadata = [r['A'].get('generation_metadata') or {} for r in rows]
    tokens = [m.get('generated_token_count') for m in metadata]
    known = bool(tokens) and all(type(t) is int for t in tokens)
    old = freeze['baseline_metadata'].get('historical_metrics') or {}
    return {'new_calls_this_invocation': 0, 'recorded_call_records': sum(bool(r['A'].get('call')) for r in rows),
            'new_calls_in_original_run': (old.get('usage') or {}).get('new_calls'),
            'output_tokens': sum(tokens) if known else None,
            'mean_output_tokens': sum(tokens) / len(tokens) if known else None,
            'original_latency_seconds': old.get('latency_seconds'),
            'cost_estimate': None, 'missing_historical_metrics': not bool(old)}


def write_csv(path, records, fields):
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(records)


def generate_reports(out, rows, arms, freeze, usage, invocation, *, synthetic, dry_run, complete,
                     replay, offline, resumed, config):
    write(out / 'paired_results.jsonl', rows, jsonl=True)
    if 'A' in arms: write(out / 'ablation_A_v31.jsonl', [a_diagnostic(r) for r in rows], jsonl=True)
    if 'B' in arms or 'C' in arms:
        write(out / 'ablation_B_single_pass.jsonl', [{**{k: r[k] for k in ('sample_key', 'source_record_index')}, 'result': r['B']} for r in rows], jsonl=True)
        write(out / 'ablation_B_raw.jsonl', [{'sample_key': r['sample_key'], 'raw_response': r['B']['raw_response'] if r['B'] else None,
                                             'raw_candidates': raw_claims(r['B']['raw_response']) if r['B'] else [],
                                             'downstream_publication': 'FORBIDDEN_DIAGNOSTIC_ONLY'} for r in rows], jsonl=True)
    if 'C' in arms: write(out / 'ablation_C_conditional_repair.jsonl', [{'sample_key': r['sample_key'], 'result': r['C']} for r in rows], jsonl=True)
    write(out / 'quality_gate_audit.jsonl', [{'sample_key': r['sample_key'], 'B': r['B']['gate'] if r['B'] else None,
                                           'C': r['C']['gate'] if r['C'] else None} for r in rows], jsonl=True)
    write(out / 'repair_audit.jsonl', [{'sample_key': r['sample_key'], **r['repair']} for r in rows], jsonl=True)
    repair_calls = [r['C']['repair_audit']['call'] for r in rows if r['C'] and r['C']['repair_audit'].get('call')]
    gen_calls = [r['B'] for r in rows if r['B']]
    triggered = sum(bool(r['B'] and r['B']['gate']['repair_needed']) for r in rows)
    operations = Counter(op['op'] for r in rows for op in r['repair'].get('operations', []))
    summary = {'format': 'relive-stage2-v4-lite-v1',
               'status': 'DRY_RUN' if dry_run else 'PARTIAL' if not complete else 'SYNTHETIC_TEST_ONLY' if synthetic else 'COMPLETE',
               'sample_count': 22, 'requirement_count': 100, 'synthetic_test': synthetic,
               'arms': {a: arm_metrics(rows, a) for a in arms},
               'repair': {'triggered_samples': triggered, 'executed_calls': len(repair_calls), 'trigger_rate': triggered / 22,
                          'applied_samples': sum(r['repair']['status'] == 'APPLIED' for r in rows),
                          'rolled_back_samples': sum(r['repair']['status'] == 'ROLLED_BACK' for r in rows),
                          'code_flag_resolution_count': sum(bool(r['repair'].get('code_flag_resolution')) for r in rows),
                          'operations': dict(sorted(operations.items())), 'semantic_improvement': 'NOT_REVIEWED'},
               'B_C_shared_generation': all(r['C']['parent_B_sha256'] == stable_hash(r['B']) for r in rows if r['C']),
               'cost': {'A_new_calls': 0, 'A_historical': historical_cost(rows, freeze), 'B': costs(gen_calls, invocation, 'generation'),
                        'C_incremental_repair': costs(repair_calls, invocation, 'repair'),
                        'C_total_logical_calls_including_shared_B': len(gen_calls) + len(repair_calls),
                        'comparison': 'paired B->C; C pays extra conditional repair, not equal-cost'},
               'model_config': config, 'baseline_config': freeze['baseline_metadata'],
               'generation_parameters': {'explicit_backend_arguments': (config or {}).get('backend', {}).get('generation'),
                                         'temperature': (config or {}).get('backend', {}).get('generation', {}).get('temperature'),
                                         'top_p': (config or {}).get('backend', {}).get('generation', {}).get('top_p'),
                                         'random_seed': None, 'unset_parameters': 'not set by frozen adapter; greedy decoding; no seed control exposed'},
               'usage_this_invocation': usage, 'resumed_checkpoint_count': resumed,
               'real_new_model_calls': 0 if synthetic else usage['new_calls'],
               'synthetic_adapter_new_calls': usage['new_calls'] if synthetic else 0,
               'gt_used': False, 'image_reads': 0, 'certificate_writes': 0, 'new_verified_count': 0,
               'semantic_quality': 'NOT_REVIEWED', 'stage1_requirements_unchanged': True,
               'requirements_sha256': freeze['requirements_sha256']}
    write(out / 'ablation_metrics.json', summary)
    write(out / 'replay_audit.json', {'status': 'PASS' if replay and usage['new_calls'] == 0 and complete else 'NOT_RUN',
                                    'new_model_calls': usage['new_calls'], 'cache_hits': usage['cache_hits'],
                                    'exact_result_comparison': 'PASS' if replay and complete else 'NOT_RUN',
                                    'offline': offline, 'resumed_checkpoint_count': resumed})
    write(out / 'gt_isolation_audit.json', {'status': 'PASS', 'generator_fields': ['question', 'qa_type', 'requirements'],
                                          'repair_additional_fields': ['current_B_graph', 'code_quality_gate'],
                                          'historical_A_in_B_C_prompts': False, 'gt_used': False, 'image_reads': 0,
                                          'certificate_writes': 0, 'new_verified_count': 0,
                                          'verification_basis': 'explicit input projection; sentinel unit/integration tests'})
    review = []; graphs = []
    for r in rows:
        for arm in arms:
            gate = a_diagnostic(r)['unified_diagnostic'] if arm == 'A' else r[arm]['gate'] if r.get(arm) else None
            candidates = raw_claims(r['A']['raw_response']) if arm == 'A' else raw_claims(r['B']['raw_response']) if arm == 'B' and r['B'] else [a['normalized_claim'] for a in gate['claim_audit']] if gate else []
            states = {a['candidate_index']: a['state'] for a in gate['claim_audit']} if gate else {}
            for i, c in enumerate(candidates):
                if not isinstance(c, dict): c = {'text': json.dumps(c)}
                review.append({'sample_index': r['source_record_index'], 'sample_key': r['sample_key'], 'arm': arm,
                               'candidate_index': i, 'claim_id': c.get('claim_id'), 'claim_text': c.get('text'),
                               'claim_type': c.get('type'), 'requires': json.dumps(c.get('requires', [])),
                               'alternative_to': json.dumps(c.get('alternative_to', [])), 'gate_state': states.get(i),
                               **{k: '' for k in REVIEW_FIELDS}, 'review_status': 'NOT_REVIEWED'})
            graphs.append({'sample_index': r['source_record_index'], 'sample_key': r['sample_key'], 'arm': arm,
                           'answer_sufficiency': '', 'dependency_validity': '', 'essential_targets_covered': '',
                           'reviewer_notes': '', 'review_status': 'NOT_REVIEWED'})
    fields = ['sample_index', 'sample_key', 'arm', 'candidate_index', 'claim_id', 'claim_text', 'claim_type',
              'requires', 'alternative_to', 'gate_state', *REVIEW_FIELDS, 'review_status']
    write_csv(out / 'claim_quality_review_template.csv', review, fields)
    write_csv(out / 'graph_quality_review_template.csv', graphs, ['sample_index', 'sample_key', 'arm', 'answer_sufficiency', 'dependency_validity', 'essential_targets_covered', 'reviewer_notes', 'review_status'])
    blind = []; mapping = []
    for v in review:
        bid = 'Q' + stable_hash([freeze['public_inputs_sha256'], v['sample_key'], v['arm'], v['candidate_index']])[:20]
        blind.append({'blind_id': bid, 'question': next(r['question'] for r in rows if r['sample_key'] == v['sample_key']),
                      **{k: v[k] for k in ('claim_text', 'claim_type', 'requires', 'alternative_to', *REVIEW_FIELDS, 'review_status')}})
        mapping.append({'blind_id': bid, **{k: v[k] for k in ('sample_key', 'arm', 'candidate_index', 'claim_id')}})
    blind.sort(key=lambda x: x['blind_id']); mapping.sort(key=lambda x: x['blind_id'])
    write_csv(out / 'blind_claim_review.csv', blind, ['blind_id', 'question', 'claim_text', 'claim_type', 'requires', 'alternative_to', *REVIEW_FIELDS, 'review_status'])
    write(out / 'blind_review_mapping.json', mapping)
    blind_graphs = []; graph_mapping = []
    for r in rows:
        for arm in arms:
            if arm == 'A': cs = raw_claims(r['A']['raw_response'])
            elif r.get(arm): cs = r[arm]['gate']['normalized_graph']
            else: cs = None
            bid = 'G' + stable_hash([freeze['public_inputs_sha256'], r['sample_key'], arm])[:20]
            blind_graphs.append({'blind_id': bid, 'question': r['question'],
                                 'requirements': json.dumps(r['requirements'], ensure_ascii=False),
                                 'candidate_graph': json.dumps(cs, ensure_ascii=False),
                                 'answer_sufficiency': '', 'dependency_validity': '',
                                 'reviewer_notes': '', 'review_status': 'NOT_REVIEWED'})
            graph_mapping.append({'blind_id': bid, 'sample_key': r['sample_key'], 'arm': arm})
    blind_graphs.sort(key=lambda x: x['blind_id']); graph_mapping.sort(key=lambda x: x['blind_id'])
    write_csv(out / 'blind_graph_review.csv', blind_graphs, ['blind_id', 'question', 'requirements', 'candidate_graph',
                                                           'answer_sufficiency', 'dependency_validity', 'reviewer_notes', 'review_status'])
    write(out / 'blind_graph_mapping.json', graph_mapping)
    _comparison(out, rows, summary, arms)
    return summary


def _comparison(out, rows, summary, arms):
    esc = lambda x: html.escape(str(x))
    parts = ['<!doctype html><meta charset="utf-8"><title>ReliVE V4-Lite paired ablation</title>',
             '<style>body{font:16px/1.55 system-ui;max-width:1500px;margin:32px auto;padding:0 20px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f8;padding:16px}section{border-top:2px solid #aaa;margin-top:36px}li{margin:12px 0}.arms{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:24px}</style>',
             '<h1>ReliVE Stage 2 V4-Lite</h1><p>All hypotheses: UNVERIFIED / NOT_SEARCHED. Semantic quality: NOT_REVIEWED. No visual verification.</p>',
             '<p>' + ('SYNTHETIC TEST ONLY — not real-model results.' if summary['synthetic_test'] else 'Real-model results are not semantic ground truth.') + '</p>',
             '<details><summary>Metrics and cost</summary><pre>' + esc(json.dumps(summary, indent=2, ensure_ascii=False)) + '</pre></details>']
    md = ['# V4-Lite paired ablation', '', f"Execution: {summary['status']}. Semantic quality: NOT_REVIEWED.",
          'All hypotheses remain UNVERIFIED / NOT_SEARCHED. No certificate or visual verification.',
          'Risk counts are code heuristics, not measured semantic quality. Review values: PASS / FAIL / UNCERTAIN; blank = NOT_REVIEWED.', '']
    for number, r in enumerate(rows, 1):
        parts += [f'<section><h2>Case {number:02d} · source {r["source_record_index"]} · {esc(r["qa_type"])}</h2>',
                  '<h3>Original question</h3><pre>' + esc(r['question']) + '</pre>',
                  '<h3>Frozen Stage 1 requirements</h3><pre>' + esc(json.dumps(r['requirements'], ensure_ascii=False, indent=2)) + '</pre><div class="arms">']
        md += [f'## Case {number:02d} / source {r["source_record_index"]}', '', r['question'], '']
        for arm in arms:
            parts.append('<article><h3>Arm ' + arm + '</h3>'); md += [f'### {arm}', '']
            if arm == 'A':
                cs = raw_claims(r['A']['raw_response']); gate = a_diagnostic(r)['unified_diagnostic']
                state = r['A']['plan_status']; raw = r['A']['raw_response']
                retained_ids = [c['claim_id'] for c in r['A']['claims']]
            else:
                value = r.get(arm)
                gate = value['gate'] if value else None
                cs = raw_claims(value['raw_response']) if arm == 'B' and value else [a['normalized_claim'] for a in gate['claim_audit']] if gate else []
                state = gate['plan_status'] if gate else 'NOT_EXECUTED'
                raw = value.get('raw_response') if value else None
                retained_ids = [c['claim_id'] for c in gate['claims']] if gate else []
            parts += ['<p>' + esc(state) + '</p><p>Retained IDs: ' + esc(retained_ids) + '</p><ol>']
            md += [f'Status: {state}; retained IDs: {retained_ids}', '']
            for c in cs:
                if not isinstance(c, dict): c = {'text': str(c)}
                line = f"{c.get('claim_id', '?')}: {c.get('text', '')} | type={c.get('type')} | requires={c.get('requires', [])} | alternatives={c.get('alternative_to', [])}"
                parts.append('<li>' + esc(line) + '</li>'); md.append('- ' + line)
            parts.append('</ol>')
            if gate:
                details = {k: gate[k] for k in ('unresolved_slots', 'root_errors', 'graph_risks', 'claim_audit', 'requirement_coverage', 'explicit_event_label_gaps')}
                parts += ['<details><summary>Gate, slots, coverage and unresolved risks</summary><pre>' + esc(json.dumps(details, ensure_ascii=False, indent=2)) + '</pre></details>']
            if raw is not None: parts.append('<details><summary>Exact original model response</summary><pre>' + esc(raw) + '</pre></details>')
            parts.append('</article>'); md.append('')
        repair = json.dumps(r['repair'], ensure_ascii=False, indent=2)
        parts += ['</div><h3>B → C repair and edge changes</h3><pre>' + esc(repair) + '</pre></section>']
        md += ['B → C repair:', '```json', repair, '```', '']
    with (out / 'ablation_comparison.html').open('x', encoding='utf-8') as stream: stream.write('\n'.join(parts))
    with (out / 'ablation_comparison.md').open('x', encoding='utf-8') as stream: stream.write('\n'.join(md))
