"""One common offline gate for A/B0/B1. Structure is not semantic truth."""
from collections import Counter
from copy import deepcopy
import re

from relive.claim_planner.contract import shape_errors
from relive.hypothesis_graph_v31.predicates import analyze, numeric_audit
from relive.stage2_lite_ablation.gate import canonical_text
from .contract import NORMALIZED_SCHEMA, CLAIM, max_claims, event_labels
from .normalization import normalize


def warnings_for(c, row):
    text = c['text']; low = text.lower(); risks = []
    # A conjunction inside a two-argument relation is not a second fact. Mask
    # only that diagnostic span; original claim text is never changed.
    syntax = re.sub(r'\bbetween\b([^.;]*?)\band\b([^.;]*?)(?=\s+(?:is|are|was|were)\b)',
                    'between RELATED_ENTITIES', low)
    if re.search(r'\bspecific (?:instrument|action|type|anatomical|mechanical)|\bsome (?:region|action|steps?)|'
                 r'occup(?:y|ies) (?:a |some |a specific )?(?:spatial )?region|'
                 r'(?:has|have) (?:a defined )?(?:start and end|onset and cessation)|'
                 r'(?:begins|starts) at time t_start.*ends at time t_end|'
                 r'phase (?:is one of|belongs to)', low):
        risks.append('PLACEHOLDER_RISK')
    if re.search(r'\bwhile\b|\bthen\b|;|\band\b.{0,40}\b(?:is|are|begins|ends|moves|cuts|grasps)\b', syntax):
        risks.append('MULTI_PREDICATE_REVIEW')
    elif re.search(r'\b(?:and|or)\b.{0,80}\b(?:is|are)\s+(?:visible|present)\b', syntax):
        risks.append('MULTIPLE_PRESENCE_FACTS_REVIEW')
    elif re.search(r'\b(?:and|or)\b', low) and not re.search(r'\bbetween\b.+\band\b', low):
        # Advisory only: alternate names/arguments can still be one relation.
        risks.append('CONNECTIVE_SCOPE_REVIEW')
    if re.search(r'\b(?:no|never|without)\b', low): risks.append('NEGATION_SCOPE_REVIEW')
    if re.search(r'all (?:provided |supplied )?frames|every frame|throughout|continuous visibility|continuously visible', low):
        risks.append('CONTINUITY_SCOPE_REVIEW')
    if re.search(r'\(\s*x\s*,\s*y\s*,\s*z\s*\)', low) and not re.search(r'3d|three.dimensional', row['question'], re.I):
        risks.append('UNREQUESTED_3D_COORDINATE_TEMPLATE')
    if re.search(r'coordinates?|bounding box|occup(?:y|ies).{0,30}region', low):
        risks.append('LOCALIZATION_OUTPUT_AS_HYPOTHESIS_REVIEW')
    if re.search(r'next.*(?:following|provided list|list order)', low): risks.append('LIST_ORDER_ASSUMPTION_REVIEW')
    if c['type'] == 'HIGH_LEVEL' or re.search(r'adequate|sufficient extent|efficient|gentle|quality|confirms|secure knots?|logical (?:sequence|progression)', low):
        risks.append('INTERPRETATION_REQUIRES_CRITERIA_REVIEW')
    if row['qa_type'].startswith('video_summary') and re.search(r'procedure is|clinical purpose|confirms.*procedure', low):
        risks.append('QUESTION_BACKGROUND_REPETITION_REVIEW')
    if row['qa_type'] == 'tal' and re.search(r'previously|first occurrence|first time', low) and not re.search(r'first|previous', row['question'], re.I):
        risks.append('FIRST_OCCURRENCE_ASSUMPTION_REVIEW')
    if re.search(r'\b(?:it|they|this object|that structure)\b', low): risks.append('ENTITY_BINDING_REVIEW')
    if not c['requirement_ids']: risks.append('QUESTION_ONLY_RELEVANCE_REVIEW')
    if c['requires']: risks.append('DEPENDENCY_NECESSITY_REVIEW')
    if c['alternative_to']: risks.append('ALTERNATIVE_EXCLUSIVITY_REVIEW')
    return risks


def validate(raw, row, *, layout='V41'):
    normalized = normalize(raw, layout=layout); value = normalized['graph']
    root_errors = list(normalized['parse_errors']); audits = []
    reqids = {q['requirement_id'] for q in row['requirements']}
    rc = row['qa_type'].startswith('region_caption')
    if not isinstance(value, dict): root_errors.append('ROOT_OBJECT_REQUIRED')
    elif any(k in value for k in ('$schema', 'properties', '$defs')): root_errors.append('MODEL_SCHEMA_ECHO')
    else:
        root_schema = deepcopy(NORMALIZED_SCHEMA)
        root_schema['properties']['claims']['items'] = {}
        root_errors.extend('ROOT_SCHEMA:' + e for e in shape_errors(value, root_schema))
        if value.get('plan_status') == 'UNABLE_TO_PLAN' and value.get('claims'): root_errors.append('UNABLE_PLAN_WITH_CLAIMS')
        if rc and value.get('plan_status') != 'UNRESOLVED_TARGET_REFERENCE': root_errors.append('RC_STATUS_INVALID')
        for issue in value.get('requirement_issues', []) if isinstance(value.get('requirement_issues'), list) else []:
            if not isinstance(issue, dict) or not isinstance(issue.get('requirement_id'), str) or issue['requirement_id'] not in reqids:
                root_errors.append('REQUIREMENT_ISSUE_REFERENCE_INVALID')
    source = value.get('claims', []) if isinstance(value, dict) else []
    if not isinstance(source, list): source = []
    ids = [c.get('claim_id') for c in source if isinstance(c, dict) and isinstance(c.get('claim_id'), str)]
    counts = Counter(ids); seen = set()
    # The input raw text is preserved separately; per-node pre-default values come
    # from the actual unmodified JSON instance, including historical layout nodes.
    try:
        from relive.v2.task_selection import strict_json_loads
        raw_obj = strict_json_loads(raw, error_code='JSON') if isinstance(raw, str) else deepcopy(raw)
        originals = raw_obj.get('claims', []) if isinstance(raw_obj, dict) else []
    except (ValueError, TypeError, RecursionError): originals = []
    for i, c in enumerate(source):
        errors = ['CLAIM_SCHEMA:' + e for e in shape_errors(c, CLAIM)]; risks = []
        if not errors:
            cid = c['claim_id']
            if counts[cid] != 1: errors.append('DUPLICATE_CLAIM_ID')
            if i >= max_claims(row): errors.append('DYNAMIC_CLAIM_LIMIT_EXCEEDED')
            if any(q not in reqids for q in c['requirement_ids']): errors.append('REQUIREMENT_REFERENCE_INVALID')
            for field in ('requirement_ids', 'requires', 'alternative_to'):
                if len(c[field]) != len(set(c[field])): errors.append(field.upper() + '_DUPLICATE_REFERENCE')
            for field in ('requires', 'alternative_to'):
                if cid in c[field]: errors.append(field.upper() + '_SELF_REFERENCE')
                if any(q not in counts for q in c[field]): errors.append(field.upper() + '_REFERENCE_INVALID')
            key = canonical_text(c['text'])
            if key in seen: errors.append('EXACT_DUPLICATE_TEXT')
            seen.add(key)
            n = numeric_audit(c['text'], ids, reqids); errors.extend(n['errors']); risks.extend(n['risks'])
            text_refs = set(re.findall(r'\bH\d+\b', c['text']))
            if text_refs - set(ids): errors.append('TEXT_CLAIM_REFERENCE_UNKNOWN')
            if text_refs - set(c['requires']): risks.append('TEXT_REFERENCE_WITHOUT_DECLARED_DEPENDENCY')
            if rc: errors.append('UNRESOLVED_TARGET_REFERENCE')
            phrase = analyze(c['text'])
            if phrase['status'] == 'PARSED' and phrase['type'] != c['type']: errors.append('TYPE_PREDICATE_CONFLICT')
            risks.extend(warnings_for(c, row))
            if any(a.get('candidate_index') == i and a.get('field') == 'requires' and a['code'] == 'DEFAULT_NO_DECLARED_EDGE'
                   for a in normalized['normalizations']):
                risks.append('DEPENDENCY_NOT_DECLARED_REVIEW')
        before = originals[i] if isinstance(originals, list) and i < len(originals) else c
        audits.append({'candidate_index': i, 'claim_id': c.get('claim_id') if isinstance(c, dict) else None,
                       'raw_claim': before, 'normalized_claim': c, 'errors': sorted(set(errors)),
                       'risks': sorted(set(risks)), 'state': None,
                       'field_diagnostic': {'missing': sorted(set(CLAIM['required']) - set(before)) if isinstance(before, dict) else [],
                                            'extra': sorted(set(before) - set(CLAIM['properties'])) if isinstance(before, dict) else []}})
    nodes = {a['claim_id']: a for a in audits if a['claim_id'] and not a['errors']}
    edges = {k: a['normalized_claim']['requires'] for k, a in nodes.items()}

    def reaches(start, current, visited):
        return any(dst == start or dst not in visited and reaches(start, dst, visited | {dst}) for dst in edges.get(current, []))

    for cid, a in nodes.items():
        if reaches(cid, cid, {cid}): a['errors'].append('DEPENDENCY_CYCLE')
        if any(nodes[q]['normalized_claim']['type'] == 'HIGH_LEVEL' for q in edges[cid] if q in nodes):
            a['risks'].append('UNVERIFIED_HIGH_LEVEL_PREREQUISITE_REVIEW')
    slots = value.get('unresolved_slots', []) if isinstance(value, dict) and not root_errors else []
    if len({s['slot_id'] for s in slots}) != len(slots): root_errors.append('DUPLICATE_SLOT_ID')
    if any(numeric_audit(s['purpose'], ids, reqids)['errors'] for s in slots):
        root_errors.append('SLOT_NUMERIC_EXECUTION_BINDING_FORBIDDEN')
    if root_errors: slots = []
    for a in audits:
        if root_errors: a['errors'].append('ROOT_CONTRACT_INVALID')
        a['state'] = 'REJECTED' if a['errors'] else 'REVIEW_REQUIRED' if a['risks'] else 'ACCEPTED_STRUCTURAL'
    while True:
        kept = {a['claim_id'] for a in audits if a['state'] in ('ACCEPTED_STRUCTURAL', 'REVIEW_REQUIRED')}
        blocked = [a for a in audits if a['claim_id'] in kept and any(
            q not in kept for q in a['normalized_claim']['requires'] + a['normalized_claim']['alternative_to'])]
        if not blocked: break
        for a in blocked:
            a['state'] = 'BLOCKED_BY_DEPENDENCY'; a['errors'].append('DANGLING_RETAINED_EDGE')
    claims = [{**a['normalized_claim'], 'status': 'UNVERIFIED', 'evidence_status': 'NOT_SEARCHED'}
              for a in audits if a['state'] in ('ACCEPTED_STRUCTURAL', 'REVIEW_REQUIRED')]
    required = {q['requirement_id'] for q in row['requirements'] if q['importance'] == 'REQUIRED'}
    covered = {q for c in claims for q in c['requirement_ids']}
    graph_warnings = []
    if not claims and not slots and not rc: graph_warnings.append('EMPTY_PLAN_WITHOUT_SLOT')
    text = ' '.join(c['text'] for c in claims) + ' ' + ' '.join(s['purpose'] for s in slots)
    key = canonical_text(text).replace('_', ' ')
    lexical_gaps = [label for label in event_labels(row) if canonical_text(label).replace('_', ' ') not in key]
    status = ('UNRESOLVED_TARGET_REFERENCE' if rc else 'MODEL_OR_SCHEMA_FAILURE' if root_errors
              else 'INCOMPLETE_PLAN' if required - covered or graph_warnings or any(a['errors'] for a in audits)
              else 'PLANNED' if claims else 'UNBOUND_EVIDENCE_SLOTS')
    return {'normalized_graph': value, 'claims': claims, 'unresolved_slots': slots, 'claim_audit': audits,
            'normalizations': normalized['normalizations'], 'raw_schema_errors': normalized['raw_schema_errors'],
            'source_layout_warnings': normalized['source_layout_warnings'], 'root_errors': sorted(set(root_errors)),
            'graph_warnings': graph_warnings, 'plan_status': status,
            'required_ids': sorted(required), 'covered_required_ids': sorted(required & covered),
            'uncovered_required_ids': sorted(required - covered), 'question_target_label_gaps': lexical_gaps,
            'coverage_meaning': 'retained ID links and lexical labels only; semantic sufficiency NOT_REVIEWED',
            'semantic_quality': 'NOT_REVIEWED', 'downstream_publication_allowed': False,
            'spatial_admission': 'NOT_PERFORMED', 'repair_calls': 0, 'critic_calls': 0}
