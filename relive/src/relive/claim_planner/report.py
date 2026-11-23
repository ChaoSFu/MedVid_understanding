"""Read-only old/new comparison. Semantic changes require explicit human review."""
from __future__ import annotations

from collections import Counter
import html
import json
from pathlib import Path

from relive.claim_bank.data import digest, load_json, rows, write_once
from .runner import validate_run
from .sources import load_references, new_directory, seal

def metrics(results, references):
    accepted = [c for r in results for c in r["claims"]]
    rejections = [c for r in results for c in r["rejected_candidates"]]
    errors = Counter(e for c in rejections for e in c["errors"])
    required = [(r["sample_key"], q["requirement_id"]) for r in references for q in r["requirements"] if q["importance"] == "REQUIRED"]
    covered = {(r["sample_key"], rid) for r in results for c in r["claims"] for rid in c["requirement_ids"]}
    covered_count = sum(pair in covered for pair in required)
    return {
        "generation_coverage": {"numerator": sum(bool(r["claims"]) for r in results), "denominator": len(results),
                                "value": sum(bool(r["claims"]) for r in results) / len(results), "meaning": "samples with any structurally admitted UNVERIFIED hypothesis"},
        "atomicity_violation": {"status": "REVIEW_REQUIRED", "value": None, "detected_contract_rejections": sum(errors[c] for c in ("MULTIPLE_INDEPENDENT_PREDICATES", "TEXT_TYPE_COMPOUND_MISMATCH")), "limit": "Structured predicate arity plus explicit chain patterns; not complete linguistic atomicity measurement."},
        "redundancy": {"status": "REVIEW_REQUIRED", "value": None, "detected_rejections": errors["DUPLICATE_SEMANTIC_PROPOSITION"], "limit": "Exact structured/text equivalence; paraphrases or aliased entity IDs can evade detection."},
        "question_relevance": {"status": "REVIEW_REQUIRED", "value": None, "limit": "Valid requirement links and a purpose are necessary, not proof of answer discrimination."},
        "verification_contract_coverage": {"numerator": len(accepted), "denominator": len(accepted), "value": 1.0 if accepted else None, "meaning": "complete static contract shape only; operator executability and acquired evidence not tested"},
        "dependency_validity": {"accepted_graphs_checked": len(results), "invalid_retained_edges": 0, "status": "STRUCTURAL_PASS", "logical_validity": "REVIEW_REQUIRED"},
        "requirement_coverage": {"numerator": covered_count, "denominator": len(required), "value": covered_count / len(required) if required else None, "meaning": "required IDs referenced by admitted hypotheses; semantic coverage remains REVIEW_REQUIRED"},
        "unsupported_specificity": {"status": "REVIEW_REQUIRED", "value": None, "detected_reference_rejections": errors["ENTITY_REFERENCE_OR_SPECIFICITY_INVALID"], "limit": "Quotes constrain entities; arbitrary detail in prose/qualifier still needs review."},
        "rejection_reason_counts": dict(sorted(errors.items())),
    }

def compare(*, references_dir, run_dir, output_dir):
    validate_run(run_dir, references_dir)
    frozen, legacy, _ = load_references(references_dir, include_legacy=True)
    current = rows(Path(run_dir) / "stage2_v2_smoke22.jsonl")
    summary = load_json(Path(run_dir) / "stage2_v2_smoke22_summary.json")
    synthetic = summary["synthetic_test"]
    heading = "SYNTHETIC FIXTURE ONLY — no real V2 smoke executed" if synthetic else "Real model text planning — semantic quality not yet adjudicated"
    disclaimer = ("V1 为基于采样视频帧的旧 observation-informed 候选；V2 为只读问题和冻结 requirements 的文本假设规划。"
                  "输入和计算预算不匹配，本报告不是性能优劣实验。新输出不继承旧 claim，也没有执行证据搜索或验证。"
                  "结构合法不等于视觉或临床正确。mock 输出不得作为真实 Claim Bank。")
    e = lambda v: html.escape(str(v))
    parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>Stage 2 V1 vs V2 · smoke22</title><style>body{font:15px/1.6 system-ui;max-width:1320px;margin:30px auto;padding:0 18px;background:#f5f7fa;color:#192433}section{padding:22px;background:white;border:1px solid #dce2ea;margin:24px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f0f3f7;padding:12px}table{width:100%;border-collapse:collapse}td,th{padding:9px;border:1px solid #dce2ea;vertical-align:top;text-align:left}.columns{display:grid;grid-template-columns:1fr 1fr;gap:20px}.warning{padding:16px;background:#fff0cf}summary{cursor:pointer}@media(max-width:900px){.columns{display:block}}</style>',
             f'<h1>Stage 2 V1 vs V2 · 22 样本</h1><p class="warning">{e(heading)}</p><p>{e(disclaimer)}</p>',
             '<nav>' + ' · '.join(f'<a href="#case{i:02d}">{i:02d}</a>' for i in range(1, 23)) + '</nav>']
    md = ['# Stage 2 V1 vs V2 · 22 样本', '', heading, '', disclaimer, '']
    analysis = []
    reviews = []
    for i, (source, old, new) in enumerate(zip(frozen, legacy, current), 1):
        required = {r["requirement_id"] for r in source["requirements"] if r["importance"] == "REQUIRED"}
        mapped = {rid for c in new["claims"] for rid in c["requirement_ids"]}
        old_text = {c["text"] for c in old["claims"]}
        new_text = {c["text"] for c in new["claims"]}
        delta = {"source_record_index": source["source_record_index"], "sample_key": source["sample_key"],
                 "old_status": old["generation_status"], "new_status": new["generation_status"],
                 "old_claim_count": len(old["claims"]), "new_claim_count": len(new["claims"]),
                 "requirements_unchanged": source["requirements"] == old["requirements"] == new["requirements"],
                 "uncovered_required_requirement_ids": sorted(required - mapped),
                 "old_text_not_repeated_verbatim": sorted(old_text - new_text),
                 "new_text_not_in_old_verbatim": sorted(new_text - old_text),
                 "deletion_split_merge_mapping": "REVIEW_REQUIRED: independent generations; text differences do not establish a deliberate deletion/split/merge or its reason.",
                 "answer_discrimination": "REVIEW_REQUIRED", "verifiability": "New claims have explicit unbound contracts; no acquired evidence or clinical validity tested.",
                 "synthetic_test": synthetic}
        analysis.append(delta)
        reviews.append({"sample_key": source["sample_key"], "case_number": i, "source_record_index": source["source_record_index"],
                        "review_status": "PENDING", "reviewer_id": "", "rationale": "",
                        "atomicity": None, "question_relevance": None, "unsupported_specificity": None,
                        "deletions_splits_merges": [], "answer_discrimination": None})
        title = f'{i:02d} · {source["qa_type"]} · index={source["source_record_index"]}'
        parts += [f'<section id="case{i:02d}"><h2>{e(title)}</h2><h3>A. 原始问题</h3>', f'<pre>{e(source["question"])}</pre>',
                  '<h3>B. Stage 1 Requirements（原样）</h3><table><tr><th>ID</th><th>描述</th><th>类型 / evidence / importance</th></tr>']
        for r in source["requirements"]:
            parts.append(f'<tr><td>{e(r["requirement_id"])}</td><td>{e(r["description"])}</td><td>{e(r["type"])} / {e(r["evidence_type"])} / {e(r["importance"])}</td></tr>')
        parts += ['</table><div class="columns"><div><h3>C. Stage 2 V1（旧 observation contract）</h3>',
                  f'<p>{e(old["generation_status"])} — {e(old["failure_reasons"])}</p>']
        for c in old["claims"]:
            parts += [f'<p><strong>{e(c["claim_id"])}</strong> {e(c["text"])}</p>', f'<small>{e(c["type"])} / {e(c.get("evidence_type"))} · dependencies={e(c.get("depends_on", []))}</small>']
        if not old["claims"]:
            parts.append('<p>无已保存合法 claims。</p><details><summary>Rejected Raw Candidates（未接收，不重新解析）</summary>')
            for a in old["raw_responses"]:
                parts.append(f'<pre>{e(a.get("raw_text"))}</pre>')
            parts.append('</details>')
        parts.append('</div><div><h3>D. Stage 2 V2（文本假设规划）</h3>')
        parts.append(f'<p>{e(new["generation_status"])} · {e(heading)}</p>')
        for c in new["claims"]:
            parts += [f'<p><strong>{e(c["claim_id"])}</strong> {e(c["text"])}</p>',
                      f'<pre>{e(json.dumps({k: c[k] for k in ("type", "status", "proposition", "verification_purpose", "requirement_ids", "evidence_requirement", "dependencies", "interpretation_rule", "bindings")}, ensure_ascii=False, indent=2))}</pre>']
        parts += [f'<p>信息缺口：</p><pre>{e(json.dumps(new["information_gaps"], ensure_ascii=False, indent=2))}</pre>',
                  f'<details><summary>V2 rejected candidates / audit / raw completion</summary><pre>{e(json.dumps({k: new[k] for k in ("rejected_candidates", "root_errors", "claim_audit", "raw_response")}, ensure_ascii=False, indent=2))}</pre></details>',
                  '</div></div><h3>E. 逐 Case 差异与待审事项</h3>', f'<pre>{e(json.dumps(delta, ensure_ascii=False, indent=2))}</pre>', '<a href="#">返回目录</a></section>']
        md += [f'## {title}', '', '### A. Question', '', '```text', source['question'], '```', '', '### B. Frozen Stage 1 requirements', '', '```json', json.dumps(source['requirements'], ensure_ascii=False, indent=2), '```', '',
               '### C. V1 saved claims / status', '', '```json', json.dumps({k: old[k] for k in ('generation_status', 'failure_reasons', 'claims')}, ensure_ascii=False, indent=2), '```', '']
        if not old['claims']:
            md += ['Rejected Raw Candidates (not accepted):', '', *[json.dumps(a.get('raw_text'), ensure_ascii=False) + '\n' for a in old['raw_responses']]]
        md += ['### D. V2', '', '```json', json.dumps({k: new[k] for k in ('generation_status', 'claims', 'entities', 'information_gaps', 'rejected_candidates', 'root_errors')}, ensure_ascii=False, indent=2), '```', '',
               '### E. Difference analysis — REVIEW_REQUIRED', '', '```json', json.dumps(delta, ensure_ascii=False, indent=2), '```', '']
    measured = metrics(current, frozen)
    parts.append('<h2>Diagnostic metrics（不是客观语义正确率）</h2><pre>' + e(json.dumps(measured, ensure_ascii=False, indent=2)) + '</pre></html>')
    md += ['## Diagnostic metrics', '', '```json', json.dumps(measured, ensure_ascii=False, indent=2), '```', '']
    out = new_directory(output_dir)
    # Plain report artifacts are written once into a newly created directory.
    (out / 'stage2_v1_vs_v2_smoke22.html').write_text('\n'.join(parts), encoding='utf-8')
    (out / 'stage2_v1_vs_v2_smoke22.md').write_text('\n'.join(md), encoding='utf-8')
    write_once(out / 'stage2_v2_metrics.json', measured)
    write_once(out / 'case_difference_audit.jsonl', analysis, jsonl=True)
    write_once(out / 'semantic_review_queue.jsonl', reviews, jsonl=True)
    seal(out, ['stage2_v1_vs_v2_smoke22.html', 'stage2_v1_vs_v2_smoke22.md', 'stage2_v2_metrics.json', 'case_difference_audit.jsonl', 'semantic_review_queue.jsonl'],
         'relive-stage2-v2-comparison-v1', references_sha256=digest(Path(references_dir) / 'manifest.json'),
         run_sha256=digest(Path(run_dir) / 'manifest.json'), synthetic_test=synthetic,
         model_calls_made=0, cache_writes=0, certificate_writes=0, new_verified_count=0)
    return {"status": "SYNTHETIC_COMPARISON_ONLY" if synthetic else "REPORT_READY_REVIEW_REQUIRED", "case_count": len(current), "output_dir": str(out)}
