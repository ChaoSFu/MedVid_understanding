"""Deterministic structure checks and explicitly heuristic semantic flags.

ACCEPTED_STRUCTURAL never means medically correct, relevant, or visually true.
Flags are conservative review/repair candidates, not measured semantic errors.
"""
from collections import Counter
from copy import deepcopy
import re

from relive.claim_planner.contract import shape_errors
from relive.hypothesis_graph_v31.predicates import numeric_audit, analyze
from relive.v2.task_selection import strict_json_loads
from .contract import SCHEMA, CLAIM, ALIASES, max_claims, event_labels

REPAIR_RISKS = {
    'PLACEHOLDER_RISK', 'MULTI_PREDICATE_RISK', 'COMPOSITE_FACT_REVIEW',
    'CANDIDATE_LIST_ORDER_ASSUMPTION', 'QUESTION_CONTEXT_REPETITION',
    'CONTINUOUS_VISIBILITY_ASSUMPTION',
}


def canonical_text(text):
    text = re.sub(r'\bat time t(?:_[ij])?\b', 'at time PARAMETER', text, flags=re.I)
    return re.sub(r'[^\w]+', ' ', text.casefold()).strip()


def semantic_flags(c, row):
    text = c['text']; low = text.lower(); flags = []
    if re.search(r'\bspecific (?:instrument|action|type|anatomical|mechanical)|\bsome (?:region|action|step)|'
                 r'occupies (?:a|some|a specific) (?:spatial )?region|'
                 r'(?:has|have) (?:a defined )?(?:start and end|onset and cessation)|'
                 r'phase is one of|phase belongs to', low):
        flags.append('PLACEHOLDER_RISK')
    # Connectives alone do not prove multiple predicates. These are review flags.
    if re.search(r'\b(?:while|then|because|in order to)\b|;|\band\b.{0,45}\b(?:is|are|begins|ends|moves|cuts|grasps)\b', low):
        flags.append('MULTI_PREDICATE_RISK')
    elif re.search(r'\b(?:and|or)\b', low):
        flags.append('COMPOSITE_FACT_REVIEW')
    if re.search(r'\b(?:no|never|absent|without)\b', low):
        flags.append('NEGATION_SCOPE_REVIEW')
    if re.search(r'all (?:provided |supplied )?frames|every frame|throughout|continuous visibility|continuously visible', low):
        flags.append('CONTINUOUS_VISIBILITY_ASSUMPTION')
    if 'next' in low and re.search(r'following.*list|next.*(?:provided|listed).*phase|immediately following', low):
        flags.append('CANDIDATE_LIST_ORDER_ASSUMPTION')
    if row['qa_type'] == 'tal' and re.search(r'camera|view', row['question'], re.I) and re.search(r'\bno\b.*(?:forceps|knife)|without.*instrument', low):
        flags.append('REQUIREMENT_ASSUMPTION_REVIEW')
    if c['type'] == 'HIGH_LEVEL' or re.search(r'adequate|secure knot|efficient|gentle|confirms|logical sequence|quality', low):
        flags.append('CLINICAL_OR_QUALITY_INTERPRETATION_REVIEW')
    if row['qa_type'].startswith('video_summary') and re.search(r'procedure is|clinical purpose|characterize the procedure', low):
        flags.append('QUESTION_CONTEXT_REPETITION')
    if re.search(r'\b(?:it|they|this object|that structure)\b', low):
        flags.append('ENTITY_BINDING_REVIEW')
    if c['requires']:
        flags.append('DEPENDENCY_NECESSITY_REVIEW')
    if c['alternative_to']:
        flags.append('ALTERNATIVE_EXCLUSIVITY_REVIEW')
    return sorted(set(flags))


def validate(raw, row):
    root_errors = []; normalizations = []; parsed = None
    try:
        parsed = strict_json_loads(raw, error_code='STRICT_JSON_PARSE_FAILURE') if isinstance(raw, str) else deepcopy(raw)
    except (ValueError, TypeError):
        root_errors.append('STRICT_JSON_PARSE_FAILURE')
    if not isinstance(parsed, dict):
        root_errors.append('ROOT_OBJECT_REQUIRED')
        parsed = None
    elif any(k in parsed for k in ('$schema', 'properties', '$defs')):
        root_errors.append('MODEL_SCHEMA_ECHO')
    graph = deepcopy(parsed)
    # Check root fields and slots without rejecting all valid siblings for one bad node.
    if graph is not None:
        skeleton = {**graph, 'claims': []}
        root_errors.extend('ROOT_SCHEMA:' + e for e in shape_errors(skeleton, SCHEMA))
        if not isinstance(graph.get('claims'), list):
            root_errors.append('CLAIMS_ARRAY_REQUIRED')
        if graph.get('plan_status') == 'UNABLE_TO_PLAN' and graph.get('claims'):
            root_errors.append('UNABLE_PLAN_WITH_CLAIMS')
        rid = {r['requirement_id'] for r in row['requirements']}
        issues = graph.get('requirement_issues', [])
        if isinstance(issues, list) and any(isinstance(x, dict) and (
                not isinstance(x.get('requirement_id'), str) or x['requirement_id'] not in rid) for x in issues):
            root_errors.append('REQUIREMENT_ISSUE_REFERENCE_INVALID')
    source = graph.get('claims', []) if graph else []
    if not isinstance(source, list):
        source = []
    audits = []; seen_text = set()
    ids = [c.get('claim_id') for c in source if isinstance(c, dict) and isinstance(c.get('claim_id'), str)]
    counts = Counter(ids)
    reqids = {r['requirement_id'] for r in row['requirements']}
    rc = row['qa_type'].startswith('region_caption')
    for i, original in enumerate(source):
        c = deepcopy(original); errors = []; risks = []
        if isinstance(c, dict) and isinstance(c.get('type'), str) and c['type'] in ALIASES:
            old = c['type']; c['type'] = ALIASES[old]
            normalizations.append({'candidate_index': i, 'field': 'type', 'before': old, 'after': c['type'],
                                   'reason': 'DOCUMENTED_ENUM_ALIAS_ONLY'})
        errors.extend('CLAIM_SCHEMA:' + e for e in shape_errors(c, CLAIM))
        if not errors:
            cid = c['claim_id']
            if counts[cid] != 1: errors.append('DUPLICATE_CLAIM_ID')
            if i >= max_claims(row): errors.append('DYNAMIC_CLAIM_LIMIT_EXCEEDED')
            if any(r not in reqids for r in c['requirement_ids']): errors.append('REQUIREMENT_REFERENCE_INVALID')
            if len(c['requirement_ids']) != len(set(c['requirement_ids'])): errors.append('DUPLICATE_REQUIREMENT_REFERENCE')
            for name in ('requires', 'alternative_to'):
                if cid in c[name]: errors.append(name.upper() + '_SELF_REFERENCE')
                if any(x not in counts for x in c[name]): errors.append(name.upper() + '_REFERENCE_INVALID')
                if len(c[name]) != len(set(c[name])): errors.append(name.upper() + '_DUPLICATE_REFERENCE')
            key = canonical_text(c['text'])
            if key in seen_text: errors.append('EXACT_DUPLICATE_TEXT')
            seen_text.add(key)
            num = numeric_audit(c['text'], ids, reqids)
            errors.extend(num['errors']); risks.extend(num['risks'])
            if re.search(r'\bbbox\b|bounding box\s*[:=]|\bframe(?:_index|_id)\s*[:=]', c['text'], re.I):
                errors.append('EXECUTION_BINDING_FORBIDDEN')
            if rc: errors.append('UNRESOLVED_TARGET_REFERENCE')
            phrase = analyze(c['text'])
            if phrase['status'] == 'PARSED' and phrase['type'] != c['type']:
                errors.append('TYPE_PREDICATE_CONFLICT')
            risks.extend(semantic_flags(c, row))
        audits.append({'candidate_index': i, 'claim_id': c.get('claim_id') if isinstance(c, dict) else None,
                       'raw_claim': original, 'normalized_claim': c, 'errors': sorted(set(errors)),
                       'risks': sorted(set(risks)), 'state': None})
    nodes = {a['claim_id']: a for a in audits if a['claim_id'] and not a['errors']}
    edges = {cid: a['normalized_claim']['requires'] for cid, a in nodes.items()}

    def reaches(start, current, visited):
        for dst in edges.get(current, []):
            if dst == start: return True
            if dst not in visited and reaches(start, dst, visited | {dst}): return True
        return False

    for cid, a in nodes.items():
        if reaches(cid, cid, {cid}): a['errors'].append('DEPENDENCY_CYCLE')
        if any(nodes[x]['normalized_claim']['type'] == 'HIGH_LEVEL' for x in edges[cid] if x in nodes):
            a['risks'].append('UNVERIFIED_HIGH_LEVEL_PREREQUISITE_REVIEW')
    if root_errors:
        for a in audits: a['errors'].append('ROOT_CONTRACT_INVALID')
    for a in audits:
        a['state'] = 'REJECTED' if a['errors'] else 'REVIEW_REQUIRED' if a['risks'] else 'ACCEPTED_STRUCTURAL'
    changed = True
    while changed:
        changed = False
        valid = {a['claim_id'] for a in audits if a['state'] in ('REVIEW_REQUIRED', 'ACCEPTED_STRUCTURAL')}
        for a in audits:
            if a['state'] in ('REVIEW_REQUIRED', 'ACCEPTED_STRUCTURAL') and any(x not in valid for x in a['normalized_claim']['requires']):
                a['state'] = 'BLOCKED_BY_DEPENDENCY'; a['errors'].append('DANGLING_RETAINED_DEPENDENCY'); changed = True
    claims = [{**a['normalized_claim'], 'status': 'UNVERIFIED', 'evidence_status': 'NOT_SEARCHED'}
              for a in audits if a['state'] in ('ACCEPTED_STRUCTURAL', 'REVIEW_REQUIRED')]
    valid = {c['claim_id'] for c in claims}
    # Alternatives are also checked after pruning: no stale edge is silently removed.
    stale = {c['claim_id'] for c in claims if any(x not in valid for x in c['alternative_to'])}
    if stale:
        for a in audits:
            if a['claim_id'] in stale:
                a['state'] = 'BLOCKED_BY_DEPENDENCY'; a['errors'].append('DANGLING_RETAINED_ALTERNATIVE')
        # Re-evaluate full graph with newly blocked edges via a fixed point.
        while True:
            valid = {a['claim_id'] for a in audits if a['state'] in ('ACCEPTED_STRUCTURAL', 'REVIEW_REQUIRED')}
            blocked = [a for a in audits if a['claim_id'] in valid and any(
                x not in valid for x in a['normalized_claim']['requires'] + a['normalized_claim']['alternative_to'])]
            if not blocked: break
            for a in blocked:
                a['state'] = 'BLOCKED_BY_DEPENDENCY'; a['errors'].append('DANGLING_RETAINED_EDGE')
        claims = [{**a['normalized_claim'], 'status': 'UNVERIFIED', 'evidence_status': 'NOT_SEARCHED'}
                  for a in audits if a['state'] in ('ACCEPTED_STRUCTURAL', 'REVIEW_REQUIRED')]
    covered = {r for c in claims for r in c['requirement_ids']}
    required = {r['requirement_id'] for r in row['requirements'] if r['importance'] == 'REQUIRED'}
    slots = graph.get('unresolved_slots', []) if graph and not root_errors else []
    graph_risks = []
    if not claims and not slots and not rc: graph_risks.append('EMPTY_PLAN_WITHOUT_SLOT')
    if graph and graph.get('plan_status') == 'PLANNED' and not claims: graph_risks.append('PLANNED_WITHOUT_CLAIMS')
    if len({s['slot_id'] for s in slots}) != len(slots): root_errors.append('DUPLICATE_SLOT_ID')
    if rc and graph and graph.get('plan_status') != 'UNRESOLVED_TARGET_REFERENCE': root_errors.append('RC_STATUS_INVALID')
    if root_errors:
        claims = []
        for a in audits:
            a['state'] = 'REJECTED'
            if 'ROOT_CONTRACT_INVALID' not in a['errors']: a['errors'].append('ROOT_CONTRACT_INVALID')
        covered = set()
    missing = sorted(required - covered)
    # Explicit-name coverage is only lexical diagnostic, never semantic proof.
    texts = ' '.join(c['text'].casefold().replace('-', ' ') for c in claims)
    missing_events = [x for x in event_labels(row) if x.casefold().replace('-', ' ') not in texts]
    if missing_events and not slots: graph_risks.append('EXPLICIT_EVENT_LABEL_COVERAGE_REVIEW')
    # Lexical name gaps and unknown dependency semantics need review, not an
    # automatic claim that repair is warranted. Known schema/edge defects and
    # specific high-risk text shapes can justify the one repair attempt.
    repairable_graph = [x for x in graph_risks if x != 'EXPLICIT_EVENT_LABEL_COVERAGE_REVIEW']
    reasons = sorted(set(root_errors + repairable_graph + [e for a in audits for e in a['errors']]
                         + [r for a in audits for r in a['risks'] if r in REPAIR_RISKS]))
    status = ('UNRESOLVED_TARGET_REFERENCE' if rc else 'MODEL_OR_SCHEMA_FAILURE' if root_errors
              else 'INCOMPLETE_PLAN' if missing or graph_risks or any(a['errors'] for a in audits)
              else 'PLANNED' if claims else 'UNBOUND_EVIDENCE_SLOTS')
    return {'parsed_graph': parsed, 'normalized_graph': graph, 'claims': claims,
            'unresolved_slots': slots, 'claim_audit': audits, 'normalizations': normalizations,
            'root_errors': sorted(set(root_errors)), 'graph_risks': sorted(set(graph_risks)),
            'plan_status': status, 'uncovered_required_ids': missing,
            'explicit_event_label_gaps': missing_events,
            'requirement_coverage': {'covered_required_ids': sorted(required & covered), 'required_ids': sorted(required),
                                     'meaning': 'retained ID links only; not semantic sufficiency'},
            'repair_reasons': reasons, 'repair_needed': bool(reasons) and not (rc and not source and not root_errors),
            'semantic_quality': 'NOT_REVIEWED', 'spatial_admission': 'NOT_PERFORMED',
            'downstream_publication_allowed': False}
