"""Immutable, zero-call comparison. Historical responses are never repaired."""
from collections import Counter
import html
import json
from pathlib import Path

from relive.storage.artifacts import stable_hash
from relive.v2.task_selection import strict_json_loads
from .sources import load_run, load_references, read_json, write, digest, seal, new_dir


def diagnose_v3(*, diagnosis_json, output_dir):
    source = read_json(diagnosis_json)
    records = []
    for r in source['samples']:
        raw = r['raw_response']
        try:
            value = strict_json_loads(raw, error_code='V3_DIAGNOSTIC')
        except (ValueError, TypeError) as exc:
            category = 'DUPLICATE_JSON_KEYS' if 'DUPLICATE_KEY' in str(exc) else 'INVALID_JSON'
        else:
            if isinstance(value, dict) and isinstance(value.get('properties'), dict):
                category = 'WRAPPED_INSTANCE' if isinstance(value['properties'].get('claims'), list) else 'SCHEMA_DEFINITION'
            else:
                category = 'RESULT_INSTANCE' if isinstance(value, dict) and isinstance(value.get('claims'), list) else 'OTHER_ROOT'
        records.append({
            'source_record_index': r['source_record_index'], 'qa_type': r['qa_type'],
            'historical_status': r['plan_status'], 'historical_root_errors': r['root_errors'],
            'raw_response': raw, 'raw_response_sha256': stable_hash(raw),
            'observed_root_shape': category, 'diagnostic_only': True,
            'historical_result_changed': False, 'publication_allowed': False,
        })
    out = new_dir(output_dir)
    write(out / 'v3_failure_diagnostic.jsonl', records, jsonl=True)
    summary = {
        'scope': 'OFFLINE_SHAPE_DIAGNOSTIC_NOT_REVALIDATION_OR_RECOVERY',
        'sample_count': len(records), 'root_shape_counts': dict(sorted(Counter(r['observed_root_shape'] for r in records).items())),
        'source_sha256': digest(diagnosis_json), 'new_model_calls': 0, 'image_reads': 0,
        'cache_writes': 0, 'certificate_writes': 0, 'new_verified_count': 0,
    }
    write(out / 'v3_failure_summary.json', summary)
    seal(out, ['v3_failure_diagnostic.jsonl', 'v3_failure_summary.json'], 'relive-v3.1-offline-diagnostic-v1')
    return summary


def compare(*, run_dir, references_dir, output_dir, baseline_diagnosis=None):
    samples, manifest = load_run(run_dir, references_dir)
    baseline = None
    if baseline_diagnosis:
        old = read_json(baseline_diagnosis)['samples']
        if len(old) != len(samples) or any(
            any(a.get(k) != b.get(k) for k in ('source_record_index', 'qa_type', 'question', 'requirements'))
            for a, b in zip(old, samples)
        ):
            raise ValueError('BASELINE_FROZEN_INPUT_MISMATCH')
        baseline = old
    out = new_dir(output_dir)
    esc = lambda x: html.escape(str(x))
    dump = lambda x: esc(json.dumps(x, ensure_ascii=False, indent=2))
    parts = [
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Stage 2 V3.1 review</title>',
        '<style>body{font:16px/1.65 system-ui;max-width:1150px;margin:30px auto;padding:0 16px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f4f8;padding:16px}article{border-top:2px solid #28648a;margin-top:30px}</style>',
        '<h1>Stage 2 V3.1 — unverified planning review</h1>',
        '<p>SYNTHETIC TEST ONLY</p>' if manifest['synthetic_test'] else '<p>Real text-only generation; semantic quality and visual truth remain unverified.</p>',
    ]
    review = ['# Stage 2 V3.1 quality review', '',
              'SYNTHETIC TEST ONLY' if manifest['synthetic_test'] else 'REAL GENERATION; semantic review required', '',
              'Structure does not prove atomicity, relevance or clinical correctness.',
              '', '| Index | Plan | Retained | States |', '|---:|---|---:|---|']
    for i, r in enumerate(samples):
        states = dict(Counter(a['state'] for a in r['claim_audit']))
        review.append(f'| {r["source_record_index"]} | {r["plan_status"]} | {len(r["claims"])} | {states} |')
        parts += [
            f'<article><h2>Case {i+1:02}: {r["source_record_index"]} / {esc(r["qa_type"])}</h2>',
            f'<h3>Original question</h3><p>{esc(r["question"])}</p>',
            f'<h3>Unchanged requirements</h3><pre>{dump(r["requirements"])}</pre>',
        ]
        if baseline:
            parts.append(f'<h3>Frozen V3 raw response — not repaired</h3><pre>{dump({k:baseline[i][k] for k in ("plan_status","root_errors","raw_response")})}</pre>')
        parts.append(f'<h3>V3.1 result and per-node audit</h3><pre>{dump({k:r[k] for k in ("plan_status","claims","requirement_issues","root_errors","root_error_details","claim_audit","raw_response")})}</pre></article>')
    parts.append('</html>')
    (out / 'stage2_v3_1_comparison.html').write_text('\n'.join(parts), encoding='utf-8')
    (out / 'stage2_v3_1_quality_review.md').write_text('\n'.join(review) + '\n', encoding='utf-8')
    seal(out, ['stage2_v3_1_comparison.html', 'stage2_v3_1_quality_review.md'],
         'relive-v3.1-comparison-v1', source_run_sha256=digest(Path(run_dir) / 'manifest.json'),
         baseline_sha256=digest(baseline_diagnosis) if baseline_diagnosis else None)
    return {'status': 'REVIEW_REQUIRED', 'sample_count': len(samples), 'synthetic_test': manifest['synthetic_test'],
            'new_model_calls': 0, 'image_reads': 0, 'certificate_writes': 0, 'new_verified_count': 0}
