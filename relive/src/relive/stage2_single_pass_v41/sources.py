"""Historical A and B0 are read-only comparison data, never generator inputs."""
from copy import deepcopy
from pathlib import Path
from relive.stage2_lite_ablation.sources import load_inputs as load_a, check_unchanged
from relive.stage2_lite_ablation.gate import validate as validate_b0
from relive.hypothesis_graph.sources import rows
from relive.claim_revalidation.offline import read_json, digest
from relive.storage.artifacts import stable_hash
from .storage import verify


def load_inputs(stage1_bank, baseline_v31, baseline_b):
    public, a, freeze = load_a(stage1_bank, baseline_v31)
    freeze = deepcopy(freeze); root = Path(baseline_b)
    if root.is_dir():
        m = verify(root)
        if m.get('format') != 'relive-stage2-v4-lite-v1' or m.get('dry_run'): raise ValueError('EXECUTED_V4_BASELINE_REQUIRED')
        if m['identity']['public_inputs_sha256'] != freeze['public_inputs_sha256']: raise ValueError('B0_PUBLIC_INPUT_HASH_MISMATCH')
        records = rows(root / 'paired_results.jsonl')
        freeze['source_sha256'][str((root / 'manifest.json').resolve())] = digest(root / 'manifest.json')
        freeze['source_sha256'][str((root / 'paired_results.jsonl').resolve())] = digest(root / 'paired_results.jsonl')
        metadata = {'backend_fingerprint': m['backend_fingerprint'], 'source_kind': 'SEALED_V4_RUN'}
    else:
        records = rows(root)
        freeze['source_sha256'][str(root.resolve())] = digest(root)
        metadata = {'backend_fingerprint': None, 'source_kind': 'EMBEDDED_PAIRED_RESULTS',
                    'configuration_comparability': 'UNKNOWN_WITHOUT_ORIGINAL_RUN_MANIFEST'}
    if len(records) != 22: raise ValueError('B0_FULL_SMOKE22_REQUIRED')
    result = []
    for row, record in zip(public, records):
        if any(record.get(k) != row[k] for k in ('sample_key', 'source_record_index', 'qa_type', 'question', 'requirements',
                                                'question_sha256', 'requirements_sha256')):
            raise ValueError('B0_FROZEN_PUBLIC_INPUT_CHANGED')
        b = record.get('B')
        if not isinstance(b, dict) or not isinstance(b.get('raw_response'), (str, type(None))): raise ValueError('B0_EXECUTED_RESULT_REQUIRED')
        old = validate_b0(b['raw_response'], row)
        if any(b['gate'][k] != old[k] for k in ('claims', 'claim_audit', 'root_errors', 'plan_status', 'unresolved_slots')):
            raise ValueError('B0_HISTORICAL_GATE_MISMATCH')
        result.append(deepcopy(b))
    freeze['B0_metadata'] = metadata
    freeze['A_raw_sha256'] = stable_hash([r['raw_response'] for r in a])
    freeze['B0_raw_sha256'] = stable_hash([r['raw_response'] for r in result])
    return public, a, result, freeze
