"""Exact provenance checks plus conservative exclusions, not medical truth proof."""
from collections import Counter
from jsonschema import Draft202012Validator
from .schemas import CARD,LIBRARY
from .source_policy import BENCHMARK
from .common import norm,canonical
import re


def validate_card(card,source,unit,*,method,synthetic=False):
    errors=['SCHEMA:'+e.json_path for e in Draft202012Validator(CARD).iter_errors(card)]
    if errors:return sorted(set(errors))
    if not synthetic and source['source_audit']['extraction_allowed'] is not True:errors.append('SOURCE_NOT_REVIEWED_INDEPENDENT')
    # Scan the entire card, not only summary text; IDs/derivations can leak too.
    if BENCHMARK.search(canonical(card)):errors.append('BENCHMARK_SPECIFIC_CONTENT')
    if len(card['text'].split())<6:errors.append('NON_RETRIEVABLE_TEXT_FRAGMENT')
    if re.search(r'\b(?:always|never fails|in every patient|guarantees)\b',card['text'],re.I):errors.append('UNSUPPORTED_UNIVERSAL_ASSERTION')
    if card['knowledge_type']=='procedure' and card['knowledge_basis']!='standard_procedure_prior':errors.append('PROCEDURE_PRIOR_NOT_DECLARED')
    if re.search(r'\b(?:this|the current|test) video\b|\b(?:sample_id|anchor_id|frame_id)\b',card['text'],re.I):errors.append('VIDEO_SPECIFIC_ASSERTION')
    blocks={(b['page'],b['block_id']):b['text'] for b in unit['blocks']}
    facts={f['fact_id']:f for f in card['source_facts']};derived={g['guidance_id']:g for g in card['derived_guidance']}
    if len(facts)!=len(card['source_facts']) or len(derived)!=len(card['derived_guidance']):errors.append('DUPLICATE_LOCAL_FACT_OR_GUIDANCE_ID')
    for f in facts.values():
        if f['source_id']!=source['source_id'] or not 1<=f['page']<=source['pdf_pages']:errors.append('FACT_SOURCE_OR_PAGE_INVALID')
        if (f['page'],f['block_id']) not in blocks or norm(f['quote']) not in norm(blocks.get((f['page'],f['block_id']),'')):errors.append('FACT_QUOTE_NOT_IN_SUPPLIED_BLOCK')
        if len(norm(f['quote']))<20:errors.append('FACT_QUOTE_TOO_SHORT')
        if f['extraction_method']!=method:errors.append('EXTRACTION_METHOD_MISMATCH')
        if method=='vision_llm':
            if f['basis']!='figure_interpretation' or not f['figure_label'] or norm(f['figure_label']) not in norm('\n'.join(blocks.values())):errors.append('FIGURE_LABEL_NOT_SOURCE_GROUNDED')
            elif re.fullmatch(r'(?:fig(?:ure)?\.?\s*)?\d+[a-z]?',f['figure_label'].strip(),re.I):errors.append('FIGURE_NUMBER_ALONE_IS_NOT_ENTITY_LABEL')
            context=norm('\n'.join(blocks.values())).casefold()
            if not any(norm(name).casefold() in context for name in [card['entity_name'],*card['aliases']]):errors.append('VISION_ENTITY_NAME_NOT_IN_LABELED_CONTEXT')
        elif f['basis']!='body_text' or f['figure_label'] is not None:errors.append('TEXT_CANNOT_CLAIM_FIGURE_INTERPRETATION')
    def check_links(ids):
        if not ids or not set(ids)<=set(facts):errors.append('UNSUPPORTED_FACT_REFERENCE')
    check_links(card['text_supporting_fact_ids'])
    for group in ['visual_features','motion_features','spatial_relations','discriminative_features','confusable_with']:
        for item in card[group]:check_links(item['supporting_fact_ids'])
    for g in derived.values():check_links(g['supporting_fact_ids'])
    for group in card['evidence_guidance'].values():
        for item in group:
            if not item['supporting_fact_ids'] and not item['derived_guidance_ids']:errors.append('UNGROUNDED_EVIDENCE_GUIDANCE')
            if item['supporting_fact_ids']:check_links(item['supporting_fact_ids'])
            if not set(item['derived_guidance_ids'])<=set(derived):errors.append('DERIVED_GUIDANCE_REFERENCE_INVALID')
    fact_pages={f['page'] for f in facts.values()};ref_pages=set()
    for ref in card['source_refs']:
        if ref['source_id']!=source['source_id'] or not 1<=ref['page_start']<=ref['page_end']<=source['pdf_pages']:errors.append('SOURCE_REF_RANGE_INVALID');continue
        ref_pages.update(range(ref['page_start'],ref['page_end']+1))
        if ref['section'] is not None and ref['section']!=unit['section']:errors.append('SECTION_NOT_IN_SUPPLIED_LOCATOR')
    if ref_pages!=fact_pages:errors.append('SOURCE_REFS_NOT_EXACT_FACT_PAGE_COVERAGE')
    return sorted(set(errors))


def validate_library(library):
    errors=['LIBRARY_SCHEMA:'+e.json_path for e in Draft202012Validator(LIBRARY).iter_errors(library)]
    sources=library['source_index'];chunks=library['chunks'];by_id={s['source_id']:s for s in sources}
    if len(by_id)!=len(sources):errors.append('DUPLICATE_SOURCE_ID')
    if len({c['chunk_id'] for c in chunks})!=len(chunks):errors.append('DUPLICATE_CHUNK_ID')
    if library['synthetic_test']!=(library['library_id']=='relive.surgical_visual_knowledge.mock'):errors.append('SYNTHETIC_LIBRARY_ID_MISMATCH')
    for c in chunks:
        for ref in c['source_refs']:
            s=by_id.get(ref['source_id'])
            if s is None or not 1<=ref['page_start']<=ref['page_end']<=s['pdf_pages']:errors.append('LIBRARY_SOURCE_REF_INVALID')
            elif not library['synthetic_test'] and s['benchmark_independent'] is not True:errors.append('LIBRARY_SOURCE_NOT_APPROVED')
        if not library['synthetic_test'] and any(p['synthetic'] for p in c['extraction_provenance']):errors.append('MOCK_IN_FORMAL_LIBRARY')
    if errors:raise ValueError(';'.join(sorted(set(errors))))
    return {'schema':'PASS','unique_ids':'PASS','source_references':'PASS','medical_approval':'PENDING_NOT_PROVEN'}


def validate_output(output,references_root,cache_dir):
    """Offline verification reopens actual PDFs and rechecks literal fact quotes."""
    from pathlib import Path
    from .common import read_json,file_hash,stable_hash
    from .pdf_parser import parse_pdf
    output=Path(output);root=Path(references_root).resolve();library=read_json(output)
    checks=validate_library(library)
    summary=read_json(output.parent/'build_summary.json');binding=read_json(output.parent/'build_binding.json')
    for name,h in summary['artifact_sha256'].items():
        if Path(name).name!=name or file_hash(output.parent/name)!=h:raise ValueError('OUTPUT_ARTIFACT_HASH_MISMATCH')
    parsed={};failures=[]
    for s in library['source_index']:
        path=root/s['file_path']
        if not path.resolve().is_relative_to(root) or not path.is_file():raise ValueError('SOURCE_FILE_MISSING_OR_ESCAPES_ROOT')
        if s['sha256'] is None:
            failures.append({'source_id':s['source_id'],'reason':'ORIGINAL_SOURCE_UNREADABLE'});continue
        if file_hash(path)!=s['sha256']:raise ValueError('SOURCE_FILE_HASH_MISMATCH')
        if s['processing_status'] not in ('PARSED','PARSED_PARTIAL'):continue
        p,_=parse_pdf(path,s['sha256'],cache_dir,max_pages=binding['identity']['max_pages'],tables=binding['identity']['tables'])
        if p['pdf_pages']!=s['pdf_pages']:raise ValueError('SOURCE_PDF_PAGE_COUNT_MISMATCH')
        parsed[s['source_id']]={(page['page'],b['block_id']):b['clean_text'] for page in p['pages'] for b in page['clean_blocks']}
    for chunk in library['chunks']:
        fids={f['fact_id'] for f in chunk['source_facts']}
        if not set(chunk['text_supporting_fact_ids'])<=fids:raise ValueError('CHUNK_FACT_REFERENCE_INVALID')
        for fact in chunk['source_facts']:
            block=parsed.get(fact['source_id'],{}).get((fact['page'],fact['block_id']))
            if not block or norm(fact['quote']) not in norm(block):raise ValueError('OFFLINE_FACT_QUOTE_MISMATCH')
    return {**checks,'source_file_hashes':'PASS','literal_quote_grounding':'PASS',
            'known_unreadable_sources':failures,'api_calls':0,'medical_approval':'PENDING_NOT_PROVEN'}
