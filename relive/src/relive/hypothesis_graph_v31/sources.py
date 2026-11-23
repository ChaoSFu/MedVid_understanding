"""Frozen reference reuse and V3.1 result verification; no old-result rewriting."""
from pathlib import Path
from relive.hypothesis_graph.sources import (
    PUBLIC_KEYS, rows, new_dir, seal, write, digest, read_json,
    manifest_check, prepare, load_references,
)
from relive.storage.artifacts import stable_hash
from .contract import VERSION, prompt_for


def check_result(result, public, identity, synthetic):
    from .validation import validate
    if any(result[k] != public[k] for k in PUBLIC_KEYS):
        raise ValueError('V31_FROZEN_INPUT_CHANGED')
    if (result['format'] != VERSION or result['synthetic_test'] != synthetic
            or result['image_reads'] != 0 or result['input_modality'] != 'TEXT_ONLY'
            or result['prompt_sha256'] != stable_hash(prompt_for(public))):
        raise ValueError('V31_RESULT_PROVENANCE_INVALID')
    if not identity['dry_run'] and public['source_binding_status'] == 'PASS' and (result['call'] or {}).get('execution_status') == 'OK':
        checked = validate(result['raw_response'], public)
        if any(result[k] != v for k, v in checked.items()):
            raise ValueError('V31_GRAPH_VALIDATION_MISMATCH')
    elif result['claims'] or result['claim_audit'] or result['raw_response'] is not None:
        raise ValueError('V31_UNEXECUTED_GRAPH_FORBIDDEN')
    if any(c['status'] != 'UNVERIFIED' or c['evidence_status'] != 'NOT_SEARCHED' for c in result['claims']):
        raise ValueError('V31_TRUTH_STATE_INVALID')

def load_run(root,refs):
    from .contract import protocol
    m=manifest_check(root);public,_=load_references(refs)
    if m['format']!='relive-v3.1-run-v1' or m['identity']['references_sha256']!=digest(Path(refs)/'manifest.json') or m['identity']['protocol_sha256']!=protocol()['protocol_sha256']:raise ValueError('V3_RUN_BINDING_INVALID')
    binding=read_json(Path(root)/'run_binding.json')
    if binding['identity']!=m['identity'] or binding['synthetic_test']!=m['synthetic_test']:
        raise ValueError('V31_RUN_BINDING_INVALID')
    result=rows(Path(root)/'stage2_v3_1_smoke22.jsonl')
    if len(result)!=22:raise ValueError('V3_RESULT_COVERAGE_INVALID')
    for r,p in zip(result,public):
        check_result(r,p,m['identity'],m['synthetic_test'])
        cp=Path(root)/'samples'/(p['sample_key']+'.json')
        checkpoint=read_json(cp)
        if digest(cp)!=m['checkpoint_sha256'].get(p['sample_key']) or checkpoint!={'result':r,'result_sha256':stable_hash(r)}:
            raise ValueError('V3_CHECKPOINT_BINDING_INVALID')
    return result,m
