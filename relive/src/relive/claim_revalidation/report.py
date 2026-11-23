"""Static escaped report; no model-generated judgements or invented review scores."""
from __future__ import annotations

import html
import json

def render(out, samples, stats):
    esc = lambda v: html.escape(str(v))
    old_calls = stats['v2_historical_model_calls']
    table = [('Samples', 22, stats['sample_count']), ('Raw candidates', 84, stats['raw_candidate_count']),
             ('Accepted claims', 0, stats['accepted_claim_count']), ('Not exported (all states)', 84, stats['not_exported_claim_count']),
             ('Partial samples', 0, stats['partial_sample_count']), ('Unresolved RC samples', 4, stats['unresolved_rc_sample_count']),
             ('Required requirement ID coverage', 0, f'{stats["required_requirement_coverage"]["numerator"]}/{stats["required_requirement_coverage"]["denominator"]}'),
             ('New model calls', old_calls if old_calls is not None else 'synthetic source', 0)]
    disclaimer = ('V2 原结果保持冻结；V2.1 是对同一批原始候选的独立离线派生校验。ACCEPTED 只表示可消费合同通过，'
                  '所有命题仍是 UNVERIFIED / NOT_SEARCHED，未检查视频真假。REVIEW_REQUIRED 和 BLOCKED 节点不导出到 Claim Bank。'
                  '没有移除模型自行填入的数字时间约束；它们仍须有原问题来源。')
    parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>Stage 2 V2 → V2.1 offline revalidation</title><style>body{font:15px/1.65 system-ui;max-width:1320px;margin:30px auto;padding:0 18px;background:#f6f8fb;color:#182334}section,article{background:white;padding:20px;margin:20px 0;border:1px solid #d5deea}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f0f3f7;padding:12px}table{border-collapse:collapse;width:100%}th,td{border:1px solid #d5deea;padding:9px;text-align:left;vertical-align:top}summary{cursor:pointer}.columns{display:grid;grid-template-columns:1fr 1fr;gap:16px}.note{background:#fff0d2;padding:14px}@media(max-width:900px){.columns{display:block}}</style>',
             '<h1>Stage 2 V2 → V2.1 · 零调用复验</h1>', f'<p class="note">{esc(disclaimer)}</p>',
             '<table><tr><th>Metric</th><th>Frozen V2</th><th>Offline V2.1</th></tr>']
    parts += [f'<tr><td>{esc(a)}</td><td>{esc(b)}</td><td>{esc(c)}</td></tr>' for a,b,c in table]
    parts += ['</table>', '<p>' + ' · '.join(f'<a href="#case{i:02d}">{i:02d}</a>' for i in range(1,23)) + '</p>']
    md = ['# Stage 2 V2.1 offline revalidation', '', disclaimer, '', '| Metric | Frozen V2 | Offline V2.1 |', '|---|---|---|']
    md += [f'| {a} | {b} | {c} |' for a,b,c in table]
    for i, s in enumerate(samples,1):
        title = f'{i:02d} · {s["qa_type"]} · index={s["source_record_index"]}'
        parts += [f'<section id="case{i:02d}"><h2>{esc(title)}</h2>', f'<p>{esc(s["old_v2_status"])} → {esc(s["generation_status"])}</p>',
                  f'<h3>原始 Question</h3><pre>{esc(s["question"])}</pre>',
                  f'<details><summary>100 条冻结 requirements 中本例的原文 / hash</summary><pre>{esc(json.dumps(s["requirements"],ensure_ascii=False,indent=2))}</pre><p>{esc(s["requirements_sha256"])}</p></details>',
                  f'<details><summary>实体 provenance / 原始 raw response</summary><pre>{esc(json.dumps(s["entity_audit"],ensure_ascii=False,indent=2))}</pre><pre>{esc(s["raw_response"])}</pre></details>']
        md += ['', f'## {title}', '', f'{s["old_v2_status"]} → {s["generation_status"]}', '', '```text', s['question'], '```', '', 'Frozen requirements:', '```json', json.dumps(s['requirements'],ensure_ascii=False,indent=2), '```']
        if not s['candidate_records']: parts.append('<p>没有原始候选；RC 目标引用保持 UNRESOLVED_TARGET_REFERENCE。</p>')
        for r in s['candidate_records']:
            c = r['normalized_candidate']; j = r['candidate_index']
            changes = [n for n in s['normalizations'] if n['field'].startswith(f'$.claims[{j}]') or n['field'].startswith('$.entities[')]
            parts += [f'<article><h3>{esc(r["claim_id"])} · {esc(r["state"])} · {esc(c["type"])}</h3>',
                      f'<p><strong>{esc(c["text"])}</strong></p>',
                      f'<p>Structural admission: {esc(r["structural_admission"])} · Contract validity: {esc(r["contract_validity"])} · Semantic review: {esc(r["semantic_review_status"])}</p>',
                      f'<p>V2 拒绝原因：{esc(r["old_v2_rejection"])}</p>',
                      '<div class="columns"><div><h4>原始 candidate</h4>', f'<pre>{esc(json.dumps(r["raw_candidate"],ensure_ascii=False,indent=2))}</pre></div>',
                      '<div><h4>V2.1 candidate（文本/类型/依赖不变）</h4>', f'<pre>{esc(json.dumps(c,ensure_ascii=False,indent=2))}</pre></div></div>',
                      '<h4>校验层 / 拒绝、审查、阻塞原因</h4>', f'<pre>{esc(json.dumps({k:r[k] for k in ("layers","schema_errors","contract_errors","blocking_semantic_reviews","dependency_errors","dependency_semantic_reviews","blocked_by","general_semantic_warnings","spatial_eligibility_category")},ensure_ascii=False,indent=2))}</pre>',
                      f'<details><summary>确定性 normalization / 来源与文本 hash</summary><pre>{esc(json.dumps(changes,ensure_ascii=False,indent=2))}</pre><pre>{esc(json.dumps(r["source_ref"],ensure_ascii=False,indent=2))}</pre><p>text hash: {esc(r["text_sha256"])}</p></details></article>']
            md += ['', f'### {r["claim_id"]}: {r["state"]}', '', c['text'], '', '```json', json.dumps({k:r[k] for k in ('schema_errors','contract_errors','blocking_semantic_reviews','dependency_errors','dependency_semantic_reviews','blocked_by')},ensure_ascii=False,indent=2), '```']
        parts.append('<a href="#">返回目录</a></section>')
    parts += ['<h2>实际统计与类型分布</h2>', f'<pre>{esc(json.dumps(stats,ensure_ascii=False,indent=2))}</pre></html>']
    md += ['', '## Actual metrics', '', '```json', json.dumps(stats,ensure_ascii=False,indent=2), '```', '']
    with (out / 'stage2_v2_vs_v2_1.html').open('x',encoding='utf-8') as f: f.write('\n'.join(parts))
    with (out / 'stage2_v2_1_summary.md').open('x',encoding='utf-8') as f: f.write('\n'.join(md))
