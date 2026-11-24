"""Read-only input binding. Historical candidates are quarantined to arm A."""
from pathlib import Path

from relive.claim_revalidation.offline import read_json, digest
from relive.hypothesis_graph.sources import load_references, rows
from relive.hypothesis_graph_v31.sources import load_run
from relive.hypothesis_graph_v31.validation import validate as historical_validate
from relive.storage.artifacts import stable_hash
from .contract import SMOKE22, public_input


def load_inputs(stage1_bank, v31_results):
    root = Path(stage1_bank); baseline_path = Path(v31_results)
    watch = {}; independent = False; synthetic = False
    if (root / 'bank_manifest.json').is_file():
        from relive.claim_bank.pipeline import validate_bank
        validate_bank(root)
        m = read_json(root / 'bank_manifest.json')
        if m['stage'] != 1 or m['dry_run']: raise ValueError('FROZEN_STAGE1_EXECUTED_BANK_REQUIRED')
        selected = {r['source_record_index']: r for r in rows(root / 'claim_bank.jsonl')}
        public = [selected[i] for i in SMOKE22]
        independent = True; synthetic = bool(m.get('synthetic_test', False))
        for n in ('bank_manifest.json', 'claim_bank.jsonl'): watch[str((root / n).resolve())] = digest(root / n)
    else:
        public, m = load_references(root)
        independent = bool(m.get('stage1_independently_checked', False)); synthetic = m['synthetic_source']
        for n in ('manifest.json', 'public_references.jsonl'): watch[str((root / n).resolve())] = digest(root / n)
    if any(r.get('source_binding_status', 'PASS') != 'PASS' for r in public):
        raise ValueError('FROZEN_SOURCE_BINDING_FAILED')
    public = [{**public_input(r), 'sample_key': r['sample_key'], 'source_record_index': r['source_record_index'],
               'question_sha256': stable_hash(r['question']), 'requirements_sha256': stable_hash(r['requirements'])} for r in public]
    if [r['source_record_index'] for r in public] != SMOKE22 or sum(len(r['requirements']) for r in public) != 100:
        raise ValueError('SMOKE22_OR_100_REQUIREMENTS_CHANGED')
    if len({r['sample_key'] for r in public}) != 22: raise ValueError('SAMPLE_KEY_NOT_UNIQUE')
    for r in public:
        if any(set(q) != {'requirement_id', 'description', 'type', 'evidence_type', 'importance'} for q in r['requirements']):
            raise ValueError('REQUIREMENT_ALLOWLIST_INVALID')
    baseline_metadata = {'configuration_comparability': 'UNKNOWN: diagnostic lacks frozen model/config fingerprint'}
    if baseline_path.is_dir():
        # Original sealed run verification requires the frozen references directory.
        if (root / 'bank_manifest.json').is_file(): raise ValueError('USE_V3_REFERENCES_DIR_WITH_SEALED_V31_RUN')
        baseline, bm = load_run(baseline_path, root)
        binding = read_json(baseline_path / 'run_binding.json')
        baseline_metadata = {'configuration_comparability': 'CHECK_BACKEND_FINGERPRINT', 'historical_binding': binding,
                             'historical_metrics': read_json(baseline_path / 'stage2_v3_1_metrics.json')}
        for n in ('manifest.json', 'stage2_v3_1_smoke22.jsonl', 'run_binding.json'):
            watch[str((baseline_path / n).resolve())] = digest(baseline_path / n)
    else:
        d = read_json(baseline_path); baseline = d['samples']
        watch[str(baseline_path.resolve())] = digest(baseline_path)
        if d.get('scope') != 'ALL_22_SAMPLES_WITH_COMPLETE_RAW_RESPONSES': raise ValueError('FULL_V31_DIAGNOSIS_REQUIRED')
    if len(baseline) != 22: raise ValueError('BASELINE_SELECTION_INVALID')
    for p, b in zip(public, baseline):
        if any(b[k] != p[k] for k in ('question', 'requirements', 'sample_key', 'source_record_index', 'qa_type')):
            raise ValueError('BASELINE_FROZEN_INPUT_MISMATCH')
        if b['format'] != 'relive-claimgen-stage2-v3.1': raise ValueError('V31_BASELINE_REQUIRED')
        check = historical_validate(b['raw_response'], p)
        if any(b[k] != check[k] for k in ('claims', 'claim_audit', 'plan_status', 'root_errors')):
            raise ValueError('HISTORICAL_VALIDATION_MISMATCH')
        if any(c.get('status') != 'UNVERIFIED' or c.get('evidence_status') != 'NOT_SEARCHED' for c in b['claims']):
            raise ValueError('HISTORICAL_TRUTH_STATE_INVALID')
    freeze = {'sample_count': 22, 'requirement_count': 100,
              'requirements_sha256': stable_hash([r['requirements'] for r in public]),
              'public_inputs_sha256': stable_hash(public), 'stage1_independently_checked': independent,
              'synthetic_source': synthetic, 'source_sha256': watch,
              'baseline_metadata': baseline_metadata}
    return public, baseline, freeze


def check_unchanged(freeze):
    if any(digest(p) != h for p, h in freeze['source_sha256'].items()): raise ValueError('HISTORICAL_SOURCE_CHANGED')
