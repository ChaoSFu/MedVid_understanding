"""Strict instance validation and conservative, explicit semantic risk checks."""
from collections import Counter
from copy import deepcopy
import re

from relive.claim_planner.contract import shape_errors
from relive.hypothesis_graph.validation import text_audit as legacy_text_audit
from relive.storage.artifacts import stable_hash
from relive.v2.task_selection import strict_json_loads
from .contract import SCHEMA, CLAIM_SCHEMA
from .predicates import numeric_audit, analyze, abstract_reference


def text_audit(text, row, claim_ids=()):
    if not isinstance(text, str):
        return ['CLAIM_TEXT_INVALID'], ['SEMANTICS_NOT_EVALUABLE'], {}
    errors, risks = legacy_text_audit(text, row, strict_numbers=False)
    numeric = numeric_audit(text, claim_ids, [r['requirement_id'] for r in row['requirements']])
    errors += numeric['errors']
    risks += numeric['risks']
    parsed = analyze(text)
    if parsed['status'] != 'PARSED':
        risks.append('PREDICATE_OR_ARGUMENTS_REQUIRE_REVIEW')
    if abstract_reference(text):
        risks.append('UNRESOLVED_ENTITY_OR_TEMPLATE_REVIEW')
    if re.search(r'\beither\b.+\b(?:complete|ongoing)\b.+\bor\b|\bone of (?:the )?(?:listed|provided)\b', text, re.I):
        risks.append('LOW_DISCRIMINATION_OR_TAUTOLOGY_REVIEW')
    if re.search(r'\bnext (?:phase|step|action)\b.*\bis\b|\bwill\b|\bgoing to\b', text, re.I):
        risks.append('PROSPECTIVE_HYPOTHESIS_REVIEW')
    if re.search(r'\b(?:efficient|gentleness|logical continuity|smooth progression|controlled tension|structural integrity)\b', text, re.I):
        risks.append('INTERPRETIVE_LANGUAGE_REVIEW')
    if re.search(r'\b(?:is|are) (?:making |in (?:direct )?)contact with\b', text, re.I):
        risks.append('CONTACT_ALONE_DOES_NOT_ESTABLISH_ACTION')
    text_refs = set(re.findall(r'\bH\d+\b', text))
    if not text_refs <= set(claim_ids):
        errors.append('TEXT_CLAIM_REFERENCE_UNKNOWN')
    if text_refs:
        risks.append('TEXT_REFERENCE_BINDING_REVIEW')
    return sorted(set(errors)), sorted(set(risks)), {**parsed, 'numeric_audit': numeric}


def validate(raw, row):
    result = {
        'plan_status': 'MODEL_OR_SCHEMA_FAILURE', 'claims': [], 'claim_audit': [],
        'root_errors': [], 'root_error_details': [], 'unresolved_references': [],
        'requirement_issues': [], 'uncovered_requirements': [],
        'semantic_quality': 'REVIEW_REQUIRED',
    }
    try:
        value = strict_json_loads(raw or '', error_code='V31_JSON')
    except (ValueError, TypeError, RecursionError) as exc:
        result['root_errors'] = [
            'DUPLICATE_JSON_KEY' if 'DUPLICATE_KEY' in str(exc) else 'STRICT_JSON_PARSE_FAILURE'
        ]
        return result
    if isinstance(value, dict) and isinstance(value.get('properties'), dict):
        result['root_errors'] = [
            'ROOT_INSTANCE_WRAPPED_IN_SCHEMA' if isinstance(value['properties'].get('claims'), list)
            else 'ROOT_SCHEMA_DEFINITION_INSTEAD_OF_INSTANCE'
        ]
        result['root_error_details'] = [{
            'expected': sorted(SCHEMA['properties']), 'actual': sorted(value),
            'automatic_unwrap': False,
        }]
        return result
    root_schema = deepcopy(SCHEMA)
    root_schema['properties']['claims']['items'] = {}
    result['root_errors'] = shape_errors(value, root_schema)
    if result['root_errors']:
        result['root_error_details'] = [{
            'missing_fields': sorted(set(SCHEMA['properties']) - set(value)) if isinstance(value, dict) else [],
            'extra_fields': sorted(set(value) - set(SCHEMA['properties'])) if isinstance(value, dict) else [],
        }]
        return result
    for key in ('unresolved_references', 'requirement_issues', 'uncovered_requirements'):
        result[key] = value[key]
    reqs = {r['requirement_id'] for r in row['requirements']}
    if any(r['requirement_id'] not in reqs for r in value['requirement_issues']) or not set(value['uncovered_requirements']) <= reqs:
        result['root_errors'].append('REQUIREMENT_METADATA_REFERENCE_INVALID')
    if value['plan_status'] == 'PLANNED' and not value['claims']:
        result['root_errors'].append('EMPTY_PLANNED_GRAPH')
    if value['plan_status'] == 'UNABLE_TO_PLAN' and value['claims']:
        result['root_errors'].append('UNABLE_PLAN_WITH_CLAIMS')
    rc = row['qa_type'].startswith('region_caption')
    if rc and (value['plan_status'] != 'UNRESOLVED_TARGET_REFERENCE' or not value['unresolved_references']):
        result['root_errors'].append('RC_UNRESOLVED_REFERENCE_REQUIRED')
    ids = Counter(c.get('claim_id') for c in value['claims'] if isinstance(c, dict) and isinstance(c.get('claim_id'), str))
    records = []
    for index, c in enumerate(value['claims']):
        schema_errors = shape_errors(c, CLAIM_SCHEMA)
        hard, risks, parsed = list(result['root_errors']), [], {}
        if isinstance(c, dict):
            e, risks, parsed = text_audit(c.get('text'), row, ids)
            hard += e
            if isinstance(c.get('claim_id'), str) and ids.get(c['claim_id'], 0) > 1:
                hard.append('DUPLICATE_CLAIM_ID')
            links = c.get('requirement_ids')
            if isinstance(links, list) and all(isinstance(v, str) for v in links):
                if not set(links) <= reqs or len(set(links)) != len(links):
                    hard.append('REQUIREMENT_REFERENCE_INVALID')
                if not links:
                    risks.append('QUESTION_ONLY_RELEVANCE_REVIEW')
            if parsed.get('status') == 'PARSED' and c.get('type') != parsed['type']:
                hard.append('TYPE_PREDICATE_CONFLICT')
            if c.get('type') == 'HIGH_LEVEL':
                risks.append('HIGH_LEVEL_INFERENCE_RULE_REVIEW')
                if c.get('role') != 'INTERPRETATION':
                    risks.append('INTERPRETATION_ROLE_REVIEW')
                if not c.get('requires'):
                    risks.append('HIGH_LEVEL_PREREQUISITES_MISSING')
            if c.get('role') == 'INTERPRETATION' and c.get('type') != 'HIGH_LEVEL':
                risks.append('TYPE_ROLE_ALIGNMENT_REVIEW')
            if rc:
                risks.append('UNRESOLVED_TARGET_TEMPLATE')
                # Never infer identity from failure to use a preferred phrase.
                # Only explicit named presence/attribute targets are a hard binding.
                if parsed.get('status') == 'PARSED' and parsed['type'] in ('OBJECT', 'ATTRIBUTE') and not abstract_reference(c.get('text', '')):
                    hard.append('RC_CONCRETE_TARGET_BINDING_FORBIDDEN')
                else:
                    risks.append('RC_TARGET_IDENTITY_REQUIRES_REVIEW')
        records.append({
            'candidate_index': index, 'raw_candidate': c, 'schema_errors': schema_errors,
            'errors': sorted(set(hard)), 'risks': sorted(set(risks)),
            'predicate_analysis': parsed, 'dependency_errors': [], 'blocked_by': [],
            'structural_status': 'FAIL' if schema_errors else 'PASS', 'state': 'PENDING',
        })
    by_id = {r['raw_candidate']['claim_id']: r for r in records
             if not r['schema_errors'] and ids[r['raw_candidate']['claim_id']] == 1}
    requires = {cid: r['raw_candidate']['requires'] for cid, r in by_id.items()}
    incoming = Counter(ref for refs in requires.values() for ref in refs)

    def reaches(node, target, seen):
        return node == target or (node not in seen and any(
            reaches(n, target, seen | {node}) for n in requires.get(node, []) if n in by_id
        ))

    for cid, audit in by_id.items():
        c = audit['raw_candidate']
        for field in ('requires', 'alternative_to'):
            if len(set(c[field])) != len(c[field]):
                audit['dependency_errors'].append('DUPLICATE_EDGE')
            if any(d not in by_id or d == cid for d in c[field]):
                audit['dependency_errors'].append('DANGLING_OR_SELF_REFERENCE')
        if any(reaches(n, cid, set()) for n in requires[cid] if n in by_id):
            audit['dependency_errors'].append('REQUIRES_CYCLE')
        if c['alternative_to']:
            audit['risks'].append('ALTERNATIVE_EXCLUSIVITY_REVIEW')
        if c['requires']:
            audit['risks'].append('LOGICAL_DEPENDENCY_REVIEW')
        if c['role'] == 'PREREQUISITE' and not incoming[cid]:
            audit['risks'].append('UNUSED_PREREQUISITE_REVIEW')
        if any(by_id[d]['raw_candidate']['type'] == 'HIGH_LEVEL' for d in c['requires'] if d in by_id):
            audit['risks'].append('INTERPRETATION_AS_PREMISE_REVIEW')
    for audit in records:
        audit['risks'] = sorted(set(audit['risks']))
        audit['dependency_errors'] = sorted(set(audit['dependency_errors']))
        if audit['dependency_errors']:
            audit['structural_status'] = 'FAIL'
        audit['state'] = (
            'REJECTED' if audit['schema_errors'] or audit['errors'] or audit['dependency_errors']
            else 'REVIEW_REQUIRED' if audit['risks'] else 'STRUCTURALLY_ACCEPTED'
        )

    def prune():
        changed = True
        while changed:
            changed = False
            for audit in records:
                if audit['schema_errors']:
                    continue
                c = audit['raw_candidate']
                audit['blocked_by'] = [d for d in c['requires'] + c['alternative_to']
                                       if d not in by_id or by_id[d]['state'] in ('REJECTED', 'BLOCKED_BY_DEPENDENCY')]
                if audit['blocked_by'] and audit['state'] in ('STRUCTURALLY_ACCEPTED', 'REVIEW_REQUIRED'):
                    audit['state'] = 'BLOCKED_BY_DEPENDENCY'
                    changed = True
    prune()
    seen = set()
    for audit in records:
        if audit['state'] not in ('STRUCTURALLY_ACCEPTED', 'REVIEW_REQUIRED'):
            continue
        c = audit['raw_candidate']
        norm = ' '.join(re.findall(r'\w+', re.sub(r'\bt(?:_[ij])?\b', 'TIME_PARAMETER', c['text'].casefold())))
        if norm in seen:
            audit['errors'].append('DUPLICATE_PROPOSITION')
            audit['state'] = 'REJECTED'
        seen.add(norm)
    prune()
    for audit in records:
        if audit['schema_errors']:
            continue
        c = {**audit['raw_candidate'], 'status': 'UNVERIFIED', 'evidence_status': 'NOT_SEARCHED'}
        audit['claim_sha256'] = stable_hash(c)
        if audit['state'] in ('STRUCTURALLY_ACCEPTED', 'REVIEW_REQUIRED'):
            result['claims'].append(c)
    result['claim_audit'] = records
    used = {rid for c in result['claims'] for rid in c['requirement_ids']}
    result['program_uncovered_requirements'] = sorted(reqs - used)
    result['coverage_report_disagrees'] = set(value['uncovered_requirements']) != reqs - used
    result['plan_status'] = value['plan_status'] if not result['root_errors'] else 'MODEL_OR_SCHEMA_FAILURE'
    if not rc and value['claims'] and not result['claims'] and not result['root_errors']:
        result['plan_status'] = 'NO_ADMITTED_HYPOTHESES'
    result['graph_sha256'] = stable_hash({k: result[k] for k in (
        'claims', 'unresolved_references', 'requirement_issues', 'uncovered_requirements',
    )})
    return result
