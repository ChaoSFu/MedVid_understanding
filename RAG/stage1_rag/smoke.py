"""Synthetic-only 22-pair plumbing smoke. Not a medical experiment."""
from pathlib import Path
from .common import new_output, write_json, read_json, file_hash
from .corpus import freeze, index
from .protocol import prepare, legacy
from .runner import preflight, run, validate, compare


def fixture_library():
    sentence = 'Synthetic fixture: a forceps illustration has two curved jaws and a narrow shaft.'
    fact = {'fact_id': 'f1', 'text': sentence, 'quote': sentence, 'source_id': 'synthetic-source', 'page': 1,
            'block_id': 'p1-b1', 'extraction_method': 'text_llm', 'basis': 'body_text', 'figure_label': None}
    refs = [{'source_id': 'synthetic-source', 'page_start': 1, 'page_end': 1, 'section': None}]
    card = {'chunk_id': 'synthetic.forceps.001', 'knowledge_type': 'instrument', 'entity_name': 'Forceps',
            'aliases': [], 'procedure_scope': ['synthetic fixture'], 'text': sentence, 'text_supporting_fact_ids': ['f1'],
            **{k: [] for k in ['visual_features', 'motion_features', 'spatial_relations', 'discriminative_features', 'confusable_with']},
            'evidence_guidance': {k: [] for k in ['search_targets', 'spatial_cues', 'temporal_cues', 'supporting_observations', 'insufficient_observations', 'uncertainty_conditions']},
            'source_facts': [fact], 'derived_guidance': [], 'source_refs': refs, 'locator': refs,
            'scope': 'Synthetic test only', 'uncertainty_notes': [], 'knowledge_basis': 'general_medical_knowledge',
            'review_status': 'pending', 'reviewer': '', 'reviewed_at': '',
            'source_grounding_status': 'EXACT_QUOTES_CHECKED_SEMANTIC_REVIEW_PENDING',
            'extraction_provenance': [{'request_key': 'synthetic', 'method': 'mock', 'model': 'mock', 'synthetic': True}]}
    source = {'source_id': 'synthetic-source', 'file_path': 'synthetic.pdf', 'file_name': 'synthetic.pdf',
              'sha256': '0'*64, 'title': 'Synthetic test fixture', 'publisher': None, 'source_url': None,
              'publication_year': None, 'pdf_pages': 1, 'document_type': 'synthetic', 'license': 'synthetic',
              'benchmark_independent': True, 'disposition': 'selected_chunks', 'chunk_count': 1, 'review_status': 'pending'}
    return {'schema_version': 1, 'library_id': 'relive.surgical_visual_knowledge.mock', 'library_version': '0.1.0',
            'created_at': 'SYNTHETIC', 'knowledge_language': 'en', 'source_policy': 'benchmark_independent',
            'build_status': 'MOCK_ONLY', 'synthetic_test': True, 'source_index': [source], 'chunks': [card]}


def mock_config():
    plan = {'planning_status': 'PLANNED', 'planning_reason': None, 'requirements': [
        {'requirement_id': 'R1', 'description': 'Determine the visible forceps configuration relevant to the question.',
         'type': 'OBJECT', 'evidence_type': 'SPATIAL', 'importance': 'REQUIRED'}]}
    return {'framework_version': 'ReliVE-v1', 'backend': {'kind': 'mock', 'model': 'stage1-rag-synthetic',
            'max_retries': 0, 'rules': [{'match': {'stage': 'stage1_rag'},
                                      'response': {'plan': plan, 'knowledge_citations': []}}]}}


def mock_dataset():
    return [{'qa_type': qa, 'id': 'deliberately-duplicated', 'video': ['MUST_NOT_OPEN.jpg'],
             'conversations': [{'from': 'human', 'value': f'What forceps evidence is needed? Synthetic {n}.'},
                               {'from': 'gpt', 'value': 'FORBIDDEN_GT_SENTINEL'}],
             'struc_info': {'answer': 'FORBIDDEN_GT_SENTINEL'}, 'RC_info': {'box': 'FORBIDDEN_GT_SENTINEL'}}
            for qa in sorted(legacy.QA_TYPES) for n in range(2)]


def mock_smoke(output_dir):
    root = new_output(output_dir)
    write_json(root/'synthetic_library.json', fixture_library()); write_json(root/'synthetic_data.json', mock_dataset())
    write_json(root/'synthetic_config.json', mock_config())
    freeze(root/'synthetic_library.json', root/'corpus', allow_pending=True, synthetic_fixture=True)
    index(root/'corpus', root/'index'); prepare(root/'synthetic_data.json', root/'questions', 22)
    preflight(root/'questions/question_runtime_manifest.jsonl', root/'index', root/'synthetic_config.json', root/'preflight')
    fresh = run(root/'preflight', root/'index', root/'cache', root/'run')
    replay = run(root/'preflight', root/'index', root/'cache', root/'replay', replay_source=root/'run')
    result = validate(root/'run', root/'preflight', root/'index')
    compare(root/'run', root/'preflight', root/'index', root/'comparison')
    if 'FORBIDDEN_GT_SENTINEL' in (root/'run/plans.jsonl').read_text(): raise ValueError('GT_LEAK')
    if not read_json(root/'replay/cache_replay_audit.json')['exact_result_comparison']: raise ValueError('REPLAY_FAILED')
    result.update({'fresh_synthetic_calls': fresh['synthetic_backend_calls'], 'replay_synthetic_calls': replay['synthetic_backend_calls'],
                   'replay_cache_writes': 0, 'real_model_calls': 0, 'medical_experiment_executed': False})
    write_json(root/'mock_smoke_report.json', result)
    return result
