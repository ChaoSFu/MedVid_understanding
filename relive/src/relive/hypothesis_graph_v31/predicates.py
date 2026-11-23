"""Conservative phrase analysis shared by validator and compiler.

No entity inference, rewriting, clinical rules or synonym substitution. Extracted
arguments retain exact source substrings and spans. Unknown phrases abstain.
"""
import re


def numeric_audit(text, claim_ids=(), requirement_ids=()):
    masked = text
    for identifier in sorted(set(claim_ids) | set(requirement_ids), key=len, reverse=True):
        masked = re.sub(r'\b' + re.escape(identifier) + r'\b', 'REFERENCE', masked)
    numbers = re.findall(r'(?<![\w])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?(?![\w])', masked)
    execution = bool(numbers and (
        re.search(r'\b(?:seconds?|minutes?|hours?|milliseconds?|frames?|fps|stride|timestamp|time|interval)\b|\bt\s*=|\broi\b|\bbbox\b', masked, re.I)
        or re.search(r'\[\s*\d+(?:\.\d+)?\s*,', masked)
    ))
    return {
        'errors': ['NUMERIC_EXECUTION_BINDING_FORBIDDEN'] if execution else [],
        'risks': ['NUMERIC_CONTENT_REVIEW'] if numbers and not execution else [],
        'numeric_literals': numbers,
    }


PRESENCE = [
    r'(?P<subject>.+?)\s+(?:is|are)\s+(?P<predicate>visible|present)',
    r'(?P<subject>.+?)\s+(?P<predicate>can be seen)',
]
VISIBLE_SCOPE = r'(?:\s+(?P<scope>(?:in|within) (?:the )?(?:(?:endoscopic|operative|surgical) )?(?:field of view|scene|image|frame)))?'
RELATIONS = [
    r'(?P<subject>.+?)\s+(?P<predicate>(?:is|are) (?:in (?:direct )?contact with|inside|above|below|separated from|to the right of|to the left of))\s+(?P<object>.+)',
    r'(?P<subject>.+?)\s+(?P<predicate>grasps?|encloses?|contacts?|touches?|overlaps?)\s+(?P<object>.+)',
]
MOTION = r'(?:moves?|retracts?|lifts?|rotates?|opens?|closes?|shifts?|zooms?|pans?|cuts?|separates?|changes?|dissects?|handles?|manipulates?)'
PROGRESSIVE = r'(?:moving|retracting|lifting|rotating|opening|closing|shifting|zooming|panning|cutting|separating|changing|dissecting|handling|manipulating)'


def analyze(text):
    offset = len(text) - len(text.lstrip())
    if re.search(r'\b(?:not|never|without|no longer)\b', text, re.I):
        return {'status': 'UNPARSEABLE', 'reason': 'NEGATION_REQUIRES_SEMANTIC_REVIEW'}
    stem = text.strip().rstrip('.')
    # Strip only an explicit symbolic parameter suffix; preserve it separately.
    param = re.search(r'\s+at time (t(?:_[ij])?)$', stem)
    parameter = param[1] if param else None
    if param:
        stem = stem[:param.start()]
    patterns = [
        *[('OBJECT', p + VISIBLE_SCOPE) for p in PRESENCE],
        ('ATTRIBUTE', r'(?P<subject>.+?)\s+(?:is|are)\s+(?P<predicate>open|closed|bent|straight|intact|occluded|partially occluded)'),
        *[('STATIC_RELATION', p) for p in RELATIONS],
        ('TEMPORAL_EVENT', rf'(?P<subject>.+?)\s+(?P<predicate>{MOTION})(?:\s+(?P<object>.+))?'),
        ('TEMPORAL_EVENT', rf'(?P<subject>.+?)\s+(?P<predicate>(?:is|are) (?:currently )?(?:actively )?{PROGRESSIVE})(?:\s+(?P<object>.+))?'),
    ]
    for kind, pattern in patterns:
        m = re.fullmatch(pattern, stem, flags=re.I)
        if not m:
            continue
        args = []
        for name in ('subject', 'object'):
            if m.groupdict().get(name):
                value = m[name]
                if re.search(r'\b(?:and|or|then|because|confirms?|indicates?|demonstrates?|passing|forming)\b|[.,;!?]|\b(?:to|in order to)\s+\w+', value, re.I):
                    return {'status': 'UNPARSEABLE', 'reason': 'AMBIGUOUS_OR_COMPOSITE_ARGUMENT'}
                if name == 'subject' and re.fullmatch(r'it|they|this|that|these|those', value, re.I):
                    return {'status': 'UNPARSEABLE', 'reason': 'UNRESOLVED_ENTITY_REFERENCE'}
                if re.search(r'\b(?:does|did|can|could|should|will|was|were|is|are|have|has|causing|resulting|leading)\b', value, re.I):
                    return {'status': 'UNPARSEABLE', 'reason': 'UNRESOLVED_CLAUSE_OR_MODALITY'}
                args.append({'role': name, 'text': value,
                             'span': [n + offset for n in m.span(name)]})
        return {
            'status': 'PARSED', 'type': kind, 'predicate': m['predicate'],
            'predicate_span': [n + offset for n in m.span('predicate')], 'arguments': args,
            'time_parameter': parameter,
            'scope_phrase': m.groupdict().get('scope'),
        }
    return {'status': 'UNPARSEABLE', 'reason': 'PREDICATE_NOT_IN_CONTROLLED_GRAMMAR'}


def abstract_reference(text):
    return bool(re.search(
        r'\b(?:unresolved|queried|task.relevant|target) (?:object|entity|reference|region)|'
        r'\bobject (?:identified|referenced|described) in H[1-6]\b|'
        r'\b(?:object|structure|entity)\b.{0,90}\b(?:bounding box|queried region|unresolved target)\b|\{target\}|\$target',
        text, re.I,
    ))
