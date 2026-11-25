"""Portable STG projection with original full-dataset identities."""
from collections import Counter
from pathlib import Path
from relive.claim_bank.data import project, load_runtime, isolation_audit
from relive.claim_bank.pipeline import validate_bank
from relive.claim_revalidation.offline import read_json, write, digest
from relive.hypothesis_graph.sources import rows
from relive.storage.artifacts import stable_hash
from relive.stage2_single_pass_v41.storage import verify, seal

DATA_VERSION = 'relive-stg-v41-inputs-v1'


def prepare(dataset_json, source_index, output_dir, expected_count=780):
    source = Path(dataset_json).resolve(); index_path = Path(source_index).resolve(); out = Path(output_dir).resolve()
    if out.exists(): raise ValueError('NEW_DATA_OUTPUT_DIRECTORY_REQUIRED')
    if source.parent != index_path.parent or source.parent != out.parent:
        raise ValueError('PORTABLE_INPUTS_REQUIRE_SIBLING_SOURCE_FILES')
    data = read_json(source); mapping = rows(index_path)
    if not isinstance(data, list) or len(data) != expected_count or len(mapping) != len(data):
        raise ValueError('STG_COUNT_OR_SOURCE_MAPPING_INVALID')
    source_hashes = {source.name: digest(source), index_path.name: digest(index_path)}
    original_hashes = {m['source_dataset_sha256'] for m in mapping}
    if len(original_hashes) != 1: raise ValueError('ORIGINAL_DATASET_HASH_NOT_UNIQUE')
    original_hash = next(iter(original_hashes)); runtime = []
    for position, (record, m) in enumerate(zip(data, mapping)):
        if record.get('qa_type') != 'stg' or m.get('subset') != 'stg' or m.get('subset_record_index') != position:
            raise ValueError('STG_ONLY_ORDERED_MAPPING_REQUIRED')
        if m.get('subset_file') != source.name or stable_hash(record) != m['source_record_canonical_sha256']:
            raise ValueError('SOURCE_RECORD_CANONICAL_HASH_MISMATCH')
        if type(m.get('source_record_index')) is not int or m['source_record_index'] < 0:
            raise ValueError('ORIGINAL_SOURCE_INDEX_INVALID')
        public = project(record, m['source_record_index'], original_hash)
        if public['sample_key'] != m['source_sample_key'] or not public['question'] or 'EXACTLY_ONE_HUMAN_QUESTION_REQUIRED' in public['input_issues']:
            raise ValueError('ORIGINAL_SOURCE_IDENTITY_OR_PUBLIC_INPUT_INVALID')
        runtime.append(public)
    if len({r['sample_key'] for r in runtime}) != len(runtime): raise ValueError('DUPLICATE_ORIGINAL_SOURCE_INDEX')
    audit = isolation_audit(runtime)
    out.mkdir()
    write(out / 'runtime_manifest.jsonl', runtime, jsonl=True)
    write(out / 'runtime_manifest.provenance.json', {'format': DATA_VERSION, 'dataset_sha256': original_hash,
          'runtime_sha256': digest(out / 'runtime_manifest.jsonl'), 'row_count': len(runtime),
          'subset_sha256': source_hashes[source.name], 'source_index_sha256': source_hashes[index_path.name],
          'source_container_deserialized': True, 'identity_domain': 'ORIGINAL_FULL_TRAINVAL_NOT_SUBSET_INDEX'})
    write(out / 'gt_isolation_audit.json', {**audit, 'model_calls': 0, 'image_reads': 0})
    seal(out, format=DATA_VERSION, source_files_sha256=source_hashes, original_dataset_sha256=original_hash,
         sample_count=len(runtime), dataset_counts=dict(sorted(Counter(r['dataset_name'] for r in runtime).items())),
         model_calls=0, image_reads=0, cache_writes=0, certificate_writes=0, new_verified_count=0)
    load_data(out)
    return {'status': 'PASS', 'sample_count': len(runtime), 'output_dir': str(out), 'original_dataset_sha256': original_hash}


def load_data(data_dir):
    root = Path(data_dir); m = verify(root)
    if m['format'] != DATA_VERSION: raise ValueError('STG_INPUT_MANIFEST_REQUIRED')
    for name, h in m['source_files_sha256'].items():
        if Path(name).name != name or digest(root.parent / name) != h: raise ValueError('UPLOADED_SOURCE_HASH_MISMATCH')
    runtime = load_runtime(str(root / 'runtime_manifest.jsonl'))
    if len(runtime) != m['sample_count'] or any(r['qa_type'] != 'stg' for r in runtime): raise ValueError('STG_DATA_COVERAGE_INVALID')
    return runtime, m


def bind_stage1(runtime, stage1_dir):
    root = Path(stage1_dir); report = validate_bank(root); m = read_json(root / 'bank_manifest.json')
    if m['stage'] != 1 or m['dry_run']: raise ValueError('EXECUTED_STAGE1_BANK_REQUIRED')
    bank = {r['sample_key']: r for r in rows(root / 'claim_bank.jsonl')}
    selected = []
    for r in runtime:
        old = bank.get(r['sample_key'])
        fields = ('sample_key', 'dataset_sha256', 'source_record_index', 'qa_type', 'question', 'public_record_sha256')
        if old is None or any(old[k] != r[k] for k in fields): raise ValueError('STAGE1_ORIGINAL_SOURCE_BINDING_MISMATCH')
        selected.append({'requirements': old['requirements'], 'planning_status': old.get('planning_status'),
                         'generation_status': old['generation_status'], 'sample_key': old['sample_key'],
                         'requirements_sha256': stable_hash(old['requirements'])})
    binding = {'stage1_manifest_sha256': digest(root / 'bank_manifest.json'),
               'stage1_bank_sha256': digest(root / 'claim_bank.jsonl'), 'stage1_protocol_sha256': m['protocol_sha256'],
               'synthetic_stage1': report['synthetic_test'], 'selected_requirements_sha256': stable_hash(selected),
               'planning_status_counts': dict(sorted(Counter(r['planning_status'] for r in selected).items()))}
    return selected, binding


def find_stage1(data_dir, search_root):
    runtime, _ = load_data(data_dir); target = {r['sample_key']: r for r in runtime}; matches = []
    for path in sorted(Path(search_root).rglob('bank_manifest.json')):
        try:
            m = read_json(path)
            if m.get('stage') != 1 or m.get('dry_run') or m.get('synthetic_test') or m.get('sample_count', 0) < len(runtime): continue
            selected = {r['sample_key']: r for r in rows(path.parent / 'claim_bank.jsonl') if r['sample_key'] in target}
            if set(selected) != set(target): continue
            if any(any(selected[k][f] != target[k][f] for f in ('question', 'qa_type', 'public_record_sha256')) for k in target): continue
            matches.append(path.parent)
        except (ValueError, KeyError, OSError, TypeError): continue
    if len(matches) != 1:
        raise ValueError('STAGE1_BANK_SELECTION_REQUIRED:' + str([str(p) for p in matches]))
    bind_stage1(runtime, matches[0])
    return matches[0]
