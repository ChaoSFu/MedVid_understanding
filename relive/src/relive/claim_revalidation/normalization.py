"""Narrow, auditable representation changes. Never edit hypotheses or evidence budgets."""
from __future__ import annotations

from copy import deepcopy
import re

VERSION = 'relive-claimgen-stage2-v2.1'
RULES = {
    'NULL_QUERY_TO_UNBOUND_SEARCH': 'An all-null time object with a traceable public quote, and no explicit question range, becomes an unbound input-clip search.',
    'PUBLIC_TIME_QUOTE_CORRECTION': 'Retain all numeric values; replace an erroneous citation only when those values exactly match the unique time query independently parsed from the original question.',
    'PROVENANCE_SOURCE_CORRECTION': 'An exact quote found in the explicitly referenced frozen requirement, but not in the question, changes QUESTION to REQUIREMENT.',
    'UNIQUE_REQUIREMENT_CITATION': 'An exact quote found in exactly one frozen requirement can supply its missing source pointer.',
    'LABEL_SEPARATOR_CANONICALIZATION': 'Replace underscores with spaces and collapse whitespace in entity labels; no stemming, spelling, medical aliases or synonym inference.',
    'UNRESOLVED_VARIABLE_SERIALIZATION': 'Prefix an already explicitly UNRESOLVED symbolic label with $; never resolve its identity.',
    'STATIC_MODALITY_ALIAS': 'SPATIAL_MULTI_REGION maps to MULTI_REGION_SINGLE_FRAME only for static OBJECT/ATTRIBUTE/RELATION without a temporal requirement.',
    'UNIQUE_ENTITY_REFERENCE': 'A reference equal to exactly one declared entity label under separator/case normalization can use that entity ID.',
}

def lexical(value):
    return ' '.join(value.replace('_', ' ').casefold().split()) if isinstance(value, str) else ''

def public_time_queries(question):
    """Seconds only, explicit range + optional explicit stride. No filename/FPS inference.

    Unsupported or multiple ranges are not guessed. Quotes include the full
    public sentence containing the range and stride, never a requirement.
    """
    number = r'\d+(?:\.\d+)?'
    pattern = rf'\b(?:from\s+({number})\s*(?:seconds?|s)?\s+to\s+({number})|between\s+({number})\s*(?:seconds?|s)?\s+and\s+({number}))\s*(?:seconds?\b|s\b)'
    result = []
    for match in re.finditer(pattern, question, re.I):
        start, end = (match.group(1), match.group(2)) if match.group(1) else (match.group(3), match.group(4))
        # Sentence punctuation excludes decimal points.
        before = list(re.finditer(r'[.!?](?:\s|$)|\n', question[:match.start()]))
        left = before[-1].end() if before else 0
        tail = re.search(r'[.!?](?:\s|$)|\n', question[match.end():])
        right = match.end() + tail.end() if tail else len(question)
        quote = question[left:right].strip()
        strides = re.findall(rf'\bevery\s+({number})\s*(?:seconds?\b|s\b)', quote, re.I)
        if len(strides) > 1:
            continue
        result.append({'start_seconds': float(start), 'end_seconds': float(end),
                       'stride_seconds': float(strides[0]) if strides else None, 'question_quote': quote})
    return result

def quoted_sources(quote, row):
    if not isinstance(quote, str) or not quote.strip():
        return []
    sources = []
    if quote in row['question']:
        sources.append('QUESTION')
    sources.extend(r['requirement_id'] for r in row['requirements'] if quote in r['description'])
    return sources

def normalize(raw, row):
    value = deepcopy(raw)
    changes = []
    def change(container, key, after, field, rule, **proof):
        before = deepcopy(container[key])
        if before != after:
            changes.append({'field': field, 'before': before, 'after': deepcopy(after),
                            'rule_id': rule, 'semantic_preserving': True, 'proof': proof})
            container[key] = after
    if not isinstance(value, dict):
        return value, changes
    entities = value.get('entities', [])
    reqs = {r['requirement_id']: r for r in row['requirements']}
    for i, e in enumerate(entities if isinstance(entities, list) else []):
        if not isinstance(e, dict):
            continue
        path = f'$.entities[{i}]'
        label = e.get('label')
        if isinstance(label, str):
            label = ' '.join(label.replace('_', ' ').split()) if e.get('source') != 'UNRESOLVED' else label
            change(e, 'label', label, path + '.label', 'LABEL_SEPARATOR_CANONICALIZATION')
        quote, rid, source = e.get('source_quote'), e.get('requirement_id'), e.get('source')
        if source == 'UNRESOLVED' and isinstance(label, str) and re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', label) and quote is None:
            change(e, 'label', '$' + label, path + '.label', 'UNRESOLVED_VARIABLE_SERIALIZATION')
        if source == 'QUESTION' and isinstance(quote, str) and quote not in row['question']:
            if isinstance(rid, str) and rid in reqs and quote and quote in reqs[rid]['description']:
                change(e, 'source', 'REQUIREMENT', path + '.source', 'PROVENANCE_SOURCE_CORRECTION', requirement_id=rid, exact_quote=quote)
            elif rid is None:
                matches = [r['requirement_id'] for r in row['requirements'] if quote and quote in r['description']]
                if len(matches) == 1:
                    change(e, 'source', 'REQUIREMENT', path + '.source', 'PROVENANCE_SOURCE_CORRECTION', exact_quote=quote)
                    change(e, 'requirement_id', matches[0], path + '.requirement_id', 'UNIQUE_REQUIREMENT_CITATION', exact_quote=quote)
    ids = {e.get('entity_id') for e in entities if isinstance(e, dict) and isinstance(e.get('entity_id'), str)} if isinstance(entities, list) else set()
    aliases = {}
    for e in entities if isinstance(entities, list) else []:
        if isinstance(e, dict) and isinstance(e.get('entity_id'), str) and isinstance(e.get('label'), str):
            aliases.setdefault(lexical(e['label']), []).append(e['entity_id'])
    queries = public_time_queries(row['question'])
    claims = value.get('claims', [])
    for i, c in enumerate(claims if isinstance(claims, list) else []):
        if not isinstance(c, dict):
            continue
        path = f'$.claims[{i}]'
        p = c.get('proposition')
        scope = p.get('scope') if isinstance(p, dict) else None
        q = scope.get('query_time') if isinstance(scope, dict) else None
        if isinstance(q, dict) and set(q) == {'question_quote', 'start_seconds', 'end_seconds', 'stride_seconds'}:
            numbers = [q['start_seconds'], q['end_seconds'], q['stride_seconds']]
            if all(v is None for v in numbers) and not queries and quoted_sources(q['question_quote'], row):
                change(scope, 'query_time', None, path + '.proposition.scope.query_time', 'NULL_QUERY_TO_UNBOUND_SEARCH', quote_sources=quoted_sources(q['question_quote'], row))
                if scope.get('kind') in ('INPUT_CLIP', 'QUERY_TIME_SET', 'BOUNDED_EVENT_SEARCH'):
                    change(scope, 'kind', 'SEARCH_INPUT_CLIP', path + '.proposition.scope.kind', 'NULL_QUERY_TO_UNBOUND_SEARCH')
            elif len(queries) == 1 and isinstance(q['question_quote'], str) and q['question_quote'] not in row['question'] and all(type(q[k]) in (int, float) or q[k] is None for k in ('start_seconds', 'end_seconds', 'stride_seconds')) and all(q[k] == queries[0][k] for k in ('start_seconds', 'end_seconds', 'stride_seconds')):
                change(q, 'question_quote', queries[0]['question_quote'], path + '.proposition.scope.query_time.question_quote', 'PUBLIC_TIME_QUOTE_CORRECTION', original_question_range=queries[0])
        ev = c.get('evidence_requirement')
        if isinstance(ev, dict) and ev.get('modality') == 'SPATIAL_MULTI_REGION' and c.get('type') in ('OBJECT', 'ATTRIBUTE', 'RELATION') and ev.get('temporal_requirement') is None:
            change(ev, 'modality', 'MULTI_REGION_SINGLE_FRAME', path + '.evidence_requirement.modality', 'STATIC_MODALITY_ALIAS')
        # Reference-only changes are allowed only for a unique declared symbol.
        for target, field in ((p, 'subject'), (p, 'object')):
            if isinstance(target, dict) and isinstance(target.get(field), str) and target[field] not in ids:
                options = aliases.get(lexical(target[field]), [])
                if len(options) == 1:
                    change(target, field, options[0], path + '.proposition.' + field, 'UNIQUE_ENTITY_REFERENCE')
        if isinstance(ev, dict) and isinstance(ev.get('spatial_targets'), list):
            for j, item in enumerate(ev['spatial_targets']):
                if isinstance(item, str) and item not in ids:
                    options = aliases.get(lexical(item), [])
                    if len(options) == 1:
                        change(ev['spatial_targets'], j, options[0], f'{path}.evidence_requirement.spatial_targets[{j}]', 'UNIQUE_ENTITY_REFERENCE')
    return value, changes
