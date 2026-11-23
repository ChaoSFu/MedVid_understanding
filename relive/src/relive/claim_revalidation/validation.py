"""Independent layered V2.1 validation. Schema admission never establishes truth."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import re

from relive.claim_planner import contract as v2
from relive.storage.artifacts import stable_hash
from .normalization import VERSION, RULES, lexical, normalize, public_time_queries

SCHEMA = deepcopy(v2.SCHEMA)
SCHEMA['$id'] = VERSION
CLAIM_SCHEMA = SCHEMA['properties']['claims']['items']
CLAIM_SCHEMA['properties']['proposition']['properties']['scope']['properties']['kind']['enum'].append('SEARCH_INPUT_CLIP')
# One observation is a legitimate *static* contract. Dynamic requirements are
# independently checked below; zero is never a usable observation requirement.
CLAIM_SCHEMA['properties']['evidence_requirement']['properties']['temporal_requirement']['anyOf'][0]['properties']['min_observations']['minimum'] = 1
SCHEMA['properties']['entities']['items']['properties']['source']['enum'].append('DOMAIN_KNOWLEDGE')
SCHEMA['description'] = 'V2.1 shape schema. Provenance, time, modality/type, dependency and semantic-review gates are mandatory separate layers.'

def policy():
    p = {'protocol_version': VERSION, 'schema': SCHEMA, 'normalization_rules': RULES,
         'parent_protocol_sha256': v2.protocol()['protocol_sha256'], 'parent_prompt_frozen': True,
         'time_policy': 'Never remove a non-null model-created numeric constraint. Numeric queries must exactly match one independently parsed public-question seconds range/stride.',
         'entity_policy': 'QUESTION/UNRESOLVED requirement_id can retain a valid contextual link. REQUIREMENT source needs exact cited provenance. Only separator/case equivalence, no medical synonym inference.',
         'knowledge_sources': [], 'domain_knowledge_admission': 'UNAVAILABLE_WITHOUT_A_DECLARED_SOURCE',
         'admission_policy': 'Only schema/contract PASS, no blocking semantic ambiguity, and all declared dependency nodes admitted. No edge deletion or candidate rewrite.',
         'validator_version': 'offline-layered-recovery-v1', 'new_model_calls': 0,
         'semantic_limit': 'An admitted unverified hypothesis still requires later human/visual scrutiny. Accepted is not SUPPORTED or VERIFIED.'}
    p['policy_sha256'] = stable_hash(p)
    return p

def phrase_in(label, quote):
    return bool(lexical(label) and re.search(r'(?<!\w)' + re.escape(lexical(label)) + r'(?!\w)', lexical(quote)))

def entity_validation(value, row):
    reqs = {r['requirement_id']: r for r in row['requirements']}
    items = value.get('entities', [])
    items = items if isinstance(items, list) else []
    counts = Counter(e.get('entity_id') for e in items if isinstance(e, dict) and isinstance(e.get('entity_id'), str))
    result = []
    for i, e in enumerate(items):
        errors = v2.shape_errors(e, SCHEMA['properties']['entities']['items'])
        reviews = []
        if not errors:
            source, quote, rid = e['source'], e['source_quote'], e['requirement_id']
            if counts[e['entity_id']] > 1:
                errors.append('DUPLICATE_ENTITY_ID')
            if rid is not None and rid not in reqs:
                errors.append('ENTITY_REQUIREMENT_REFERENCE_INVALID')
            if source == 'QUESTION' and (not quote or quote not in row['question']):
                errors.append('ENTITY_QUOTE_NOT_IN_QUESTION')
            if source == 'REQUIREMENT' and (rid not in reqs or not quote or quote not in reqs[rid]['description']):
                errors.append('ENTITY_QUOTE_NOT_IN_CITED_REQUIREMENT')
            if source in ('QUESTION', 'REQUIREMENT') and not phrase_in(e['label'], quote):
                reviews.append('ENTITY_ALIAS_EQUIVALENCE_UNPROVEN')
            if source == 'UNRESOLVED' and (quote is not None or not re.fullmatch(r'\$[A-Za-z][A-Za-z0-9_]*', e['label'])):
                errors.append('UNRESOLVED_ENTITY_REPRESENTATION_INVALID')
            if source == 'DOMAIN_KNOWLEDGE':
                reviews.append('DOMAIN_KNOWLEDGE_SOURCE_UNAVAILABLE')
        result.append({'entity_index': i, 'entity': e, 'errors': sorted(set(errors)), 'blocking_reviews': sorted(set(reviews)),
                       'state': 'REJECTED' if errors else 'REVIEW_REQUIRED' if reviews else 'ACCEPTED'})
    return result

def time_validation(c, row):
    p = c.get('proposition') if isinstance(c, dict) else None
    scope = p.get('scope') if isinstance(p, dict) else None
    if not isinstance(scope, dict):
        return ['TIME_LAYER_NOT_EVALUABLE']
    q, queries = scope.get('query_time'), public_time_queries(row['question'])
    if q is None:
        return ['QUERY_TIME_PARAMETERS_REQUIRED'] if scope.get('kind') == 'QUERY_TIME_SET' else []
    if not isinstance(q, dict):
        return ['QUERY_TIME_NOT_PUBLIC_OR_CONFLICTING']
    keys = ('start_seconds', 'end_seconds', 'stride_seconds')
    if len(queries) != 1 or not all(type(q.get(k)) in (int, float) or q.get(k) is None for k in keys):
        return ['QUERY_TIME_NOT_PUBLIC_OR_CONFLICTING']
    if any(q.get(k) != queries[0][k] for k in keys) or not isinstance(q.get('question_quote'), str) or q['question_quote'] not in row['question'] or scope.get('kind') == 'SEARCH_INPUT_CLIP':
        return ['QUERY_TIME_NOT_PUBLIC_OR_CONFLICTING']
    if q['start_seconds'] is None or q['end_seconds'] is None or q['start_seconds'] > q['end_seconds']:
        return ['QUERY_TIME_NOT_PUBLIC_OR_CONFLICTING']
    return []

def semantic_audit(c, entities):
    """Conservative flags, not a general semantic classifier or a truth oracle."""
    if not isinstance(c, dict):
        return [], ['SEMANTIC_LAYER_NOT_EVALUABLE']
    text = c.get('text', '') if isinstance(c, dict) else ''
    if not isinstance(text, str):
        return [], ['SEMANTIC_LAYER_NOT_EVALUABLE']
    errors, reviews = [], []
    if re.search(r'\b(?:always|never|every frame|all (?:provided )?frames|throughout (?:the )?(?:entire|whole))\b', text, re.I):
        errors.append('GLOBAL_SCOPE_ASSERTION_FORBIDDEN')
    if re.search(r'\bno\b.{0,110}\babsent\b|\bnot\b.{0,40}\bnot\b', text, re.I):
        errors.append('DOUBLE_NEGATION_FORBIDDEN')
    if re.search(r'\b(?:then|followed (?:immediately )?by|in order to|so that)\b|\bheld\b.{0,60}\band (?:is )?positioned\b', text, re.I):
        errors.append('MULTIPLE_INDEPENDENT_PREDICATES')
    if re.search(r'\b(?:score|rating)\s*(?:is|:|=|of)\s*\d|\b(?:is verified|is confirmed|proven to)\b', text, re.I):
        errors.append('FINAL_CONCLUSION_FORBIDDEN')
    if re.search(r'\bframe\s*\d+\b', text, re.I):
        errors.append('FRAME_SPECIFIC_PLANNING_FORBIDDEN')
    p = c.get('proposition')
    if not isinstance(p, dict):
        return errors, ['SEMANTIC_LAYER_NOT_EVALUABLE']
    qualifier = p.get('qualifier') or ''
    if isinstance(qualifier, str) and (re.search(r'[_\s]and[_\s]|with_identification', qualifier, re.I) or re.search(r'\band\b.{0,35}\b(?:is|are|being)\b', text, re.I)):
        reviews.append('POSSIBLE_COMPOSITE_PREDICATES')
    if re.search(r'\bor\b|;|\b(?:indicat\w*|suggest\w*|confirms?|efficient|logical|adequate|continuous|without occlusion)\b', text, re.I):
        reviews.append('ALTERNATIVES_INFERENCE_OR_CONTINUITY_REVIEW')
    if re.search(r'\b(?:boundaries|sequence)\s+(?:starts|ends|can be determined)|specific timestamp', text, re.I):
        reviews.append('META_DESCRIPTION_OR_UNBOUND_EVENT_REVIEW')
    if p.get('polarity') == 'NEGATIVE':
        reviews.append('ABSENCE_COVERAGE_REVIEW')
    ev = c.get('evidence_requirement')
    if isinstance(ev, dict) and ev.get('modality') in ('SINGLE_FRAME', 'MULTI_REGION_SINGLE_FRAME') and re.search(r'\b(?:motion|moves|transition|next phase|continuous|throughout)\b', text, re.I):
        reviews.append('STATIC_TEMPORAL_CONTRACT_CONFLICT')
    if c.get('type') in ('OBJECT', 'ATTRIBUTE') and re.search(r'\b(?:is held|grasps|retracts|moves|is positioned near)\b', text, re.I):
        errors.append('TEXT_TYPE_COMPOUND_MISMATCH')
    # A declared detailed subject and a coarser English subject may differ in
    # granularity. Do not "repair" either; keep an explicit review boundary.
    e = entities.get(p.get('subject')) if isinstance(p.get('subject'), str) else None
    if e and e.get('source') != 'UNRESOLVED' and not phrase_in(e.get('label'), text) and not re.search(r'\b' + re.escape(str(e.get('entity_id'))) + r'\b', text):
        reviews.append('ENTITY_TEXT_ALIGNMENT_REVIEW')
    return sorted(set(errors)), sorted(set(reviews))

def validate_sample(raw, row):
    normalized, changes = normalize(raw, row)
    root_schema = deepcopy(SCHEMA)
    root_schema['properties']['claims']['items'] = {}
    root_schema['properties']['entities']['items'] = {}
    root_errors = v2.shape_errors(normalized, root_schema)
    if not isinstance(normalized, dict):
        return {'normalized_response': normalized, 'normalizations': changes, 'root_errors': root_errors,
                'entity_audit': [], 'candidate_records': []}
    entity_audit = entity_validation(normalized, row)
    entities = {e['entity']['entity_id']: e['entity'] for e in entity_audit if isinstance(e['entity'], dict) and isinstance(e['entity'].get('entity_id'), str)}
    es = {e['entity']['entity_id']: e for e in entity_audit if isinstance(e['entity'], dict) and isinstance(e['entity'].get('entity_id'), str)}
    reqs = {r['requirement_id']: r for r in row['requirements']}
    claims = normalized.get('claims', [])
    claims = claims if isinstance(claims, list) else []
    old_claims = raw.get('claims', []) if isinstance(raw, dict) else []
    ids = Counter(c.get('claim_id') for c in claims if isinstance(c, dict) and isinstance(c.get('claim_id'), str))
    records = []
    compatibility = {'OBJECT': {'IS_VISIBLE'}, 'ATTRIBUTE': {'HAS_ATTRIBUTE'}, 'RELATION': {'CONTACT', 'STATIC_RELATION'}, 'ACTION': {'MOTION', 'CHANGE', 'EVENT'}, 'HIGH_LEVEL': {'INTERPRETATION'}}
    for i, c in enumerate(claims):
        original = old_claims[i]
        # Hard guarantees against changing scientific hypotheses/dependencies.
        for field in ('claim_id', 'text', 'type', 'dependencies', 'verification_purpose', 'interpretation_rule', 'status', 'evidence_status', 'requirement_ids'):
            if isinstance(c, dict) and isinstance(original, dict) and c.get(field) != original.get(field):
                raise ValueError('PROTECTED_CANDIDATE_FIELD_CHANGED')
        schema = v2.shape_errors(c, CLAIM_SCHEMA)
        errors = list(root_errors)
        reviews = []
        time_errors = time_validation(c, row)
        errors.extend(time_errors)
        sem_errors, sem_reviews = semantic_audit(c, entities)
        errors.extend(sem_errors); reviews.extend(sem_reviews)
        # Continue independently evaluable layers even if shape validation failed.
        if isinstance(c, dict):
            refs = c.get('requirement_ids')
            if isinstance(refs, list) and all(isinstance(v, str) for v in refs):
                if not refs or not set(refs) <= set(reqs) or len(refs) != len(set(refs)):
                    errors.append('REQUIREMENT_REFERENCE_INVALID')
            p, ev = c.get('proposition'), c.get('evidence_requirement')
            if isinstance(p, dict) and isinstance(ev, dict):
                targets = [p.get('subject'), *([p['object']] if p.get('object') is not None else [])]
                if isinstance(ev.get('spatial_targets'), list): targets += ev['spatial_targets']
                for target in targets:
                    if not isinstance(target, str) or target not in es:
                        errors.append('ENTITY_REFERENCE_INVALID')
                    elif es[target]['errors']:
                        errors.append('ENTITY_PROVENANCE_INVALID')
                    elif es[target]['blocking_reviews']:
                        reviews.append('ENTITY_PROVENANCE_REVIEW_REQUIRED')
                if not isinstance(c.get('type'), str) or not isinstance(p.get('predicate'), str) or p['predicate'] not in compatibility.get(c.get('type'), set()):
                    errors.append('TYPE_PREDICATE_MISMATCH')
                temporal = ev.get('temporal_requirement')
                n = temporal.get('min_observations') if isinstance(temporal, dict) else None
                modality = ev.get('modality')
                if c.get('type') == 'ACTION' and (modality not in ('MULTI_FRAME', 'COMPOSITIONAL') or type(n) is not int or n < 2):
                    errors.append('ACTION_REQUIRES_TEMPORAL_CONTRACT')
                if modality == 'MULTI_FRAME' and (type(n) is not int or n < 2):
                    errors.append('MULTI_FRAME_OBSERVATION_CONTRACT_INVALID')
                if temporal is not None and modality in ('SINGLE_FRAME', 'MULTI_REGION_SINGLE_FRAME') and (c.get('type') not in ('OBJECT', 'ATTRIBUTE', 'RELATION') or n != 1):
                    errors.append('STATIC_OBSERVATION_CONTRACT_CONFLICT')
                if isinstance(temporal, dict) and (type(n) is not int or n < 1):
                    errors.append('NONPOSITIVE_OBSERVATION_REQUIREMENT')
                if c.get('type') == 'HIGH_LEVEL' and temporal is not None and n == 1:
                    reviews.append('SINGLE_OBSERVATION_INTERPRETATION_REVIEW')
                if modality == 'MULTI_REGION_SINGLE_FRAME' and isinstance(ev.get('spatial_targets'), list) and (not all(isinstance(t, str) for t in ev['spatial_targets']) or len(set(t for t in ev['spatial_targets'] if isinstance(t, str))) < 2):
                    errors.append('MULTI_REGION_TARGETS_REQUIRED')
                if c.get('type') == 'RELATION' and p.get('object') is None:
                    errors.append('RELATION_OBJECT_REQUIRED')
                if row['qa_type'].startswith('region_caption') and any(entities.get(t, {}).get('source') != 'UNRESOLVED' for t in targets if isinstance(t, str)):
                    errors.append('RC_TARGET_IDENTITY_UNAVAILABLE')
            if c.get('type') == 'HIGH_LEVEL':
                rule = c.get('interpretation_rule')
                if not isinstance(rule, dict) or not c.get('dependencies') or not isinstance(c.get('evidence_requirement'), dict) or c['evidence_requirement'].get('modality') != 'COMPOSITIONAL':
                    errors.append('INTERPRETATION_PREREQUISITES_REQUIRED')
                elif not isinstance(rule.get('question_quote'), str) or rule['question_quote'] not in row['question']:
                    errors.append('INTERPRETATION_RULE_NOT_QUESTION_GROUNDED')
            if isinstance(c.get('claim_id'), str) and ids[c['claim_id']] > 1:
                errors.append('DUPLICATE_CLAIM_ID')
        records.append({'candidate_index': i, 'claim_id': c.get('claim_id') if isinstance(c, dict) else None,
                        'raw_candidate': original, 'normalized_candidate': c, 'raw_candidate_sha256': stable_hash(original),
                        'text_sha256': stable_hash(original.get('text')) if isinstance(original, dict) else stable_hash(None),
                        'dependencies_sha256': stable_hash(original.get('dependencies')) if isinstance(original, dict) else stable_hash(None),
                        'structural_admission': 'FAIL' if schema else 'PASS', 'schema_errors': schema,
                        'contract_errors': sorted(set(errors)), 'blocking_semantic_reviews': sorted(set(reviews)),
                        'semantic_review_status': 'REVIEW_REQUIRED',
                        'general_semantic_warnings': ['ENGLISH_PREDICATE_ALIGNMENT_NOT_PROVEN', 'QUESTION_RELEVANCE_NOT_PROVEN', 'VISUAL_TRUTH_NOT_ASSESSED'],
                        'dependency_errors': [], 'dependency_semantic_reviews': [], 'blocked_by': [],
                        'layers': {'parsing': 'PASS', 'normalization': 'NORMALIZED' if any(x['field'].startswith(f'$.claims[{i}]') for x in changes) else 'UNCHANGED',
                                   'schema': 'FAIL' if schema else 'PASS', 'time': 'FAIL' if time_errors else 'PASS', 'semantic_risk': 'REVIEW_REQUIRED'}})
    by_id = {r['claim_id']: r for r in records if isinstance(r['claim_id'], str) and ids[r['claim_id']] == 1}
    graph = {}
    for r in records:
        c = r['normalized_candidate']
        deps = c.get('dependencies', []) if isinstance(c, dict) else []
        deps = deps if isinstance(deps, list) else []
        graph[r['candidate_index']] = []
        for d in deps:
            if not isinstance(d, dict) or not isinstance(d.get('claim_id'), str) or d['claim_id'] not in by_id:
                r['dependency_errors'].append('DEPENDENCY_REFERENCE_INVALID'); continue
            other = by_id[d['claim_id']]
            graph[r['candidate_index']].append(other['candidate_index'])
            other_c = other['normalized_candidate']
            if d.get('relation') in ('REQUIRES', 'SUPPORTS') and other_c.get('type') == 'HIGH_LEVEL':
                r['dependency_semantic_reviews'].append('UNVERIFIED_INTERPRETATION_PREMISE_REVIEW')
            if d.get('relation') == 'ALTERNATIVE_TO':
                try:
                    ok = v2.signature(c, polarity=False, entities=entities) == v2.signature(other_c, polarity=False, entities=entities) and c['proposition']['polarity'] != other_c['proposition']['polarity'] and c['proposition']['scope'] == other_c['proposition']['scope']
                except (KeyError, TypeError): ok = False
                if not ok: r['dependency_semantic_reviews'].append('ALTERNATIVE_EXCLUSIVITY_UNPROVEN')
        if isinstance(c, dict) and c.get('type') == 'HIGH_LEVEL':
            lower = [by_id[d['claim_id']]['normalized_candidate'] for d in deps if isinstance(d, dict) and d.get('relation') in ('REQUIRES', 'SUPPORTS') and isinstance(d.get('claim_id'), str) and d['claim_id'] in by_id and by_id[d['claim_id']]['normalized_candidate'].get('type') != 'HIGH_LEVEL']
            if not lower: r['contract_errors'].append('LOW_LEVEL_EVIDENCE_REQUIRED')
            elif all(x.get('type') == 'OBJECT' for x in lower): r['contract_errors'].append('INTERPRETATION_NEEDS_MORE_THAN_PRESENCE')
            if row['qa_type'] == 'skill_assessment' and not any(isinstance(x.get('evidence_requirement'), dict) and x['evidence_requirement'].get('temporal_requirement') for x in lower):
                r['contract_errors'].append('SKILL_INTERPRETATION_NEEDS_TEMPORAL_PREMISE')
    def reaches(node, target, seen):
        return node == target or (node not in seen and any(reaches(n, target, seen | {node}) for n in graph.get(node, [])))
    for r in records:
        i = r['candidate_index']
        if any(reaches(n, i, set()) for n in graph[i]): r['dependency_errors'].append('DEPENDENCY_CYCLE')
        r['contract_errors'] = sorted(set(r['contract_errors']))
        r['state'] = 'REJECTED' if r['schema_errors'] or r['contract_errors'] or r['dependency_errors'] else 'REVIEW_REQUIRED' if r['blocking_semantic_reviews'] or r['dependency_semantic_reviews'] else 'ACCEPTED'
        r['contract_validity'] = 'FAIL' if r['contract_errors'] or r['dependency_errors'] else 'NOT_EVALUABLE' if r['schema_errors'] else 'REVIEW_REQUIRED' if r['blocking_semantic_reviews'] or r['dependency_semantic_reviews'] else 'PASS'
        r['layers']['provenance'] = 'FAIL' if any(x.startswith(('ENTITY_', 'RC_')) for x in r['contract_errors']) else 'REVIEW_REQUIRED' if 'ENTITY_PROVENANCE_REVIEW_REQUIRED' in r['blocking_semantic_reviews'] else 'PASS'
        r['layers']['dependencies'] = 'FAIL' if r['dependency_errors'] else 'REVIEW_REQUIRED' if r['dependency_semantic_reviews'] else 'PASS'
    def propagate():
        changed = True
        while changed:
            changed = False
            for r in records:
                blocked = [records[n]['claim_id'] for n in graph[r['candidate_index']] if records[n]['state'] != 'ACCEPTED']
                r['blocked_by'] = blocked
                if blocked and r['state'] == 'ACCEPTED':
                    r['state'] = 'BLOCKED_BY_DEPENDENCY'; r['contract_validity'] = 'BLOCKED'; changed = True
    propagate()
    signatures = set()
    for r in records:
        if r['state'] != 'ACCEPTED': continue
        c = r['normalized_candidate']; sig = v2.signature(c, entities=entities)
        if sig in signatures:
            r['state'] = 'REJECTED'; r['contract_errors'].append('DUPLICATE_SEMANTIC_PROPOSITION'); r['contract_validity'] = 'FAIL'
        else: signatures.add(sig)
    propagate()
    return {'normalized_response': normalized, 'normalizations': changes, 'root_errors': root_errors,
            'entity_audit': entity_audit, 'candidate_records': records}
