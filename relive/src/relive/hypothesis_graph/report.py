"""Full-text historical diagnostic comparison; never supplies planner inputs."""
from collections import Counter
from pathlib import Path
import html
import json
from relive.storage.artifacts import stable_hash

from .sources import load_run, load_references, rows, new_dir, seal, write, digest
from .compiler import compile_node


def compare(*,run_dir,references_dir,output_dir):
    samples,m=load_run(run_dir,references_dir);_,rm=load_references(references_dir)
    old=rows(Path(references_dir)/'comparison_quarantine.jsonl')
    if [r['sample_key'] for r in old]!=[r['sample_key'] for r in samples]:raise ValueError('COMPARISON_SELECTION_MISMATCH')
    out=new_dir(output_dir)
    esc=lambda x:html.escape(str(x))
    dump=lambda x:esc(json.dumps(x,ensure_ascii=False,indent=2))
    parts=['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>ReliVE V1 / V2 / V3</title>',
       '<style>body{font:16px/1.65 system-ui;max-width:1280px;margin:30px auto;padding:0 20px;color:#152235}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eff3f8;padding:16px}article{border-top:3px solid #245d87;margin-top:40px}h3{margin-top:32px}.warning{background:#fff1c7;padding:20px}li{margin:12px 0}</style>',
       '<h1>ReliVE · V1 / V2 / V3 逐例诊断</h1>',
       '<p class="warning">V1 使用视觉输入；V2/V3 是文本规划。输入不同，本报告不能证明公平的性能提升。所有命题均未验证。'
       +(' 本次 V3 为 SYNTHETIC TEST ONLY，不能用来评价生成质量。' if m['synthetic_test'] else ' V3 为真实模型输出，语义质量仍需人工审查。')+'</p>']
    quality=['# Stage 2 V3 semantic quality review','',
             'SYNTHETIC TEST ONLY — real generation quality is unknown.' if m['synthetic_test'] else 'Real text-only generation; no visual truth or semantic accuracy is established.',
             '', 'V1 observed images; V2/V3 planned from text. This is a diagnostic comparison, not a fair performance ranking.',
             '', '| Case | index | retained | state counts | unlinked requirement IDs | compiler potential |',
             '|---|---:|---:|---|---|---|']
    spatial=Counter()
    for i,(s,o) in enumerate(zip(samples,old),1):
        counts=dict(Counter(a['state'] for a in s['claim_audit']))
        graph={c['claim_id']:c for c in s['claims']}
        compiled=[compile_node(c,graph) for c in s['claims']]
        potential=[c['claim_id'] for c,z in zip(s['claims'],compiled) if z['compilation_status']=='COMPILED' and c['type'] in ('OBJECT','ATTRIBUTE','STATIC_RELATION')]
        spatial.update(c['type'] for c,z in zip(s['claims'],compiled) if z['compilation_status']=='COMPILED' and c['type'] in ('OBJECT','ATTRIBUTE','STATIC_RELATION'))
        quality.append(f'| {i:02} | {s["source_record_index"]} | {len(s["claims"])} | {counts} | {s.get("program_uncovered_requirements",[])} | {potential} |')
        parts += [f'<article id="case-{i:02}"><h2>Case {i:02} · index {s["source_record_index"]} · {esc(s["qa_type"])}</h2>',
                  f'<h3>A · Original Question</h3><p>{esc(s["question"])}</p>',
                  f'<h3>B · Frozen Stage 1 Requirements</h3><pre>{dump(s["requirements"])}</pre><p>SHA-256: {s["requirements_sha256"]}</p>',
                  '<h3>C · V1 saved claims</h3>']
        if o['v1'] is None:parts.append('<p>V1 artifact unavailable. No reconstruction.</p>')
        else:
            parts.append(f'<pre>{dump(o["v1"]["claims"])}</pre>')
            parts.append(f'<details><summary>V1 raw attempts (may include failed candidates, never silently admitted)</summary><pre>{dump(o["v1"]["raw_responses"])}</pre></details>')
        parts += [f'<h3>D · V2 raw hypotheses / V2.1 decisions</h3><pre>{esc(o["v2"]["raw_response"])}</pre>',
                  f'<details><summary>V2.1 per-candidate acceptance / rejection / review reasons</summary><pre>{dump(o["v21"])}</pre></details>',
                  '<h3>E · V3 minimal graph</h3><ul>']
        for c in s['claims']:
            a=next(a for a in s['claim_audit'] if a.get('claim_sha256')==stable_hash(c))
            parts.append(f'<li><b>{esc(c["claim_id"])}</b> — {esc(c["text"])}<br>{esc(c["type"])} / {esc(c["role"])} / {esc(a["state"])} / UNVERIFIED / NOT_SEARCHED<br>'
                         f'requires={esc(c["requires"])}; alternative_to={esc(c["alternative_to"])}; requirements={esc(c["requirement_ids"])}; flags={esc(a["risks"])}</li>')
        parts += [f'</ul><pre>{dump({k:s[k] for k in ("plan_status","requirement_issues","uncovered_requirements","root_errors","claim_audit","raw_response")})}</pre>',
                  '<h3>F · Comparison / open review</h3>',
                  '<p>自动检查仅证明结构、引用、保守文本约束。问题相关性、同义重复、单一谓词、逻辑依赖和关键条件覆盖仍需人工审查。</p>',
                  f'<pre>{dump({"state_counts":counts,"deterministic_compiler_dry_run":compiled,"semantic_improvement":"REVIEW_REQUIRED; no automatic accuracy claim"})}</pre></article>']
    quality += ['', 'All requires edges mean “inspect prerequisite first”, never implication. Review every logical edge and same-target/scope mutual exclusivity of alternatives.',
                '', 'Read requirement_issues against the original question; do not rewrite frozen requirements or treat REQUIRED as truth.',
                '', 'A compiled spatial contract is only a possible evidence interface, not experimental admission or a certificate.',
                '', 'No Evidence Acquisition / Grounding / KEEP / DROP / Verification / Adaptation was executed.']
    (out/'stage2_v1_v2_v3_comparison.html').write_text('\n'.join(parts)+'</html>',encoding='utf-8')
    (out/'stage2_v3_quality_review.md').write_text('\n'.join(quality)+'\n',encoding='utf-8')
    audit={'status':'DIAGNOSTIC_ONLY','synthetic_test':m['synthetic_test'],'sample_count':22,
           'potential_spatial_candidates_by_type':dict(sorted(spatial.items())),
           'semantic_improvement':'REVIEW_REQUIRED','stage1_independently_checked':rm.get('stage1_independently_checked',False),
           'new_model_calls':0,'image_reads':0,'cache_writes':0,'certificate_writes':0,'new_verified_count':0}
    write(out/'comparison_audit.json',audit)
    seal(out,['stage2_v1_v2_v3_comparison.html','stage2_v3_quality_review.md','comparison_audit.json'],
         'relive-v3-comparison-v1',source_run_sha256=digest(Path(run_dir)/'manifest.json'),references_sha256=digest(Path(references_dir)/'manifest.json'))
    return audit
