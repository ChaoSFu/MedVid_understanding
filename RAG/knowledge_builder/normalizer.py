"""Exact semantic-field deduplication; near matches/conflicts require review."""
from copy import deepcopy
from difflib import SequenceMatcher
import re
from .common import stable_hash,norm,canonical


def normalize(records):
    chunks=[];duplicates=[];near=[]
    for rec in records:
        card=deepcopy(rec['card'])
        fact_map={f['fact_id']:'fact_'+stable_hash({k:v for k,v in f.items() if k!='fact_id'})[:24] for f in card['source_facts']}
        for f in card['source_facts']:f['fact_id']=fact_map[f['fact_id']]
        card['text_supporting_fact_ids']=[fact_map[x] for x in card['text_supporting_fact_ids']]
        for group in ['visual_features','motion_features','spatial_relations','discriminative_features','confusable_with']:
            for x in card[group]:x['supporting_fact_ids']=[fact_map[i] for i in x['supporting_fact_ids']]
        guidance_map={g['guidance_id']:'guide_'+stable_hash({'text':g['text'],'facts':[fact_map[i] for i in g['supporting_fact_ids']]})[:24] for g in card['derived_guidance']}
        for g in card['derived_guidance']:
            g['guidance_id']=guidance_map[g['guidance_id']];g['supporting_fact_ids']=[fact_map[i] for i in g['supporting_fact_ids']]
        for group in card['evidence_guidance'].values():
            for x in group:
                x['supporting_fact_ids']=[fact_map[i] for i in x['supporting_fact_ids']]
                x['derived_guidance_ids']=[guidance_map[i] for i in x['derived_guidance_ids']]
        def semantics(c):
            def strip(value):
                if isinstance(value,dict):return {k:strip(v) for k,v in value.items() if k not in ('supporting_fact_ids','derived_guidance_ids','guidance_id')}
                if isinstance(value,list):return [strip(v) for v in value]
                return norm(value).casefold() if isinstance(value,str) else value
            return strip({k:c[k] for k in ('knowledge_type','entity_name','aliases','text','procedure_scope','visual_features','motion_features','spatial_relations','discriminative_features','confusable_with','evidence_guidance','derived_guidance','scope','knowledge_basis','uncertainty_notes')})
        signature=stable_hash(semantics(card))
        card.update(chunk_id=card['knowledge_type']+'.'+signature[:24],locator=deepcopy(card['source_refs']),
                    review_status='pending',reviewer='',reviewed_at='',
                    source_grounding_status='EXACT_QUOTES_CHECKED_SEMANTIC_REVIEW_PENDING',
                    extraction_provenance=[rec['extraction_provenance']])
        same=next((c for c in chunks if c['chunk_id']==card['chunk_id']),None)
        if same:
            for field in ('aliases','text_supporting_fact_ids','source_facts','source_refs','locator','extraction_provenance','derived_guidance'):
                same[field]=sorted({canonical(x):x for x in same[field]+card[field]}.values(),key=canonical)
            for field in ('visual_features','motion_features','spatial_relations','discriminative_features','confusable_with'):
                for existing,incoming in zip(same[field],card[field]):
                    existing['supporting_fact_ids']=sorted(set(existing['supporting_fact_ids']+incoming['supporting_fact_ids']))
            for category in same['evidence_guidance']:
                for existing,incoming in zip(same['evidence_guidance'][category],card['evidence_guidance'][category]):
                    for refs in ('supporting_fact_ids','derived_guidance_ids'):existing[refs]=sorted(set(existing[refs]+incoming[refs]))
            duplicates.append({'merged_chunk_id':same['chunk_id'],'source_refs':card['source_refs']});continue
        for other in chunks:
            if norm(other['entity_name']).casefold()!=norm(card['entity_name']).casefold() or other['knowledge_type']!=card['knowledge_type']:continue
            similarity=SequenceMatcher(None,norm(other['text']).casefold(),norm(card['text']).casefold()).ratio()
            neg=lambda s:bool(re.search(r'\b(?:not|no|never|without)\b',s,re.I))
            near.append({'chunk_ids':[other['chunk_id'],card['chunk_id']],
                         'reason':'POSSIBLE_POLARITY_CONFLICT' if neg(other['text'])!=neg(card['text']) else 'SAME_ENTITY_VARIANT_OR_PARAPHRASE_REVIEW',
                         'lexical_similarity_not_semantic_score':similarity,'auto_merged':False})
        chunks.append(card)
    # Conflicting cards are withheld rather than silently coexisting as facts.
    conflicted={cid for n in near if n['reason']=='POSSIBLE_POLARITY_CONFLICT' for cid in n['chunk_ids']}
    excluded=[{'card':c,'rejection_reasons':['POSSIBLE_SOURCE_CONFLICT_REVIEW_REQUIRED']} for c in chunks if c['chunk_id'] in conflicted]
    return sorted([c for c in chunks if c['chunk_id'] not in conflicted],key=lambda c:c['chunk_id']),duplicates,near,excluded
