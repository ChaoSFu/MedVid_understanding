"""Lossless defaults only. Never infer edges, rewrite text, unwrap or repair JSON."""
from copy import deepcopy
from relive.claim_planner.contract import shape_errors
from relive.v2.task_selection import strict_json_loads
from .contract import NORMALIZED_SCHEMA, ALIASES


def normalize(raw, *, layout='V41'):
    audit = []; errors = []; source_warnings = []
    try:
        value = strict_json_loads(raw, error_code='STRICT_JSON_PARSE_FAILURE') if isinstance(raw, str) else deepcopy(raw)
    except (ValueError, TypeError, RecursionError):
        return {'graph': None, 'normalizations': [], 'raw_schema_errors': ['STRICT_JSON_PARSE_FAILURE'],
                'source_layout_warnings': [], 'parse_errors': ['STRICT_JSON_PARSE_FAILURE']}
    if layout == 'V31' and isinstance(value, dict) and not any(k in value for k in ('properties', '$schema', '$defs')):
        allowed = {'plan_status', 'claims', 'unresolved_references', 'requirement_issues', 'uncovered_requirements'}
        if set(value) != allowed:
            errors.append('HISTORICAL_ROOT_LAYOUT_INVALID')
        else:
            old = deepcopy(value)
            refs = value['unresolved_references']
            if not isinstance(refs, list) or any(not isinstance(x, str) for x in refs):
                errors.append('HISTORICAL_SLOT_SOURCE_INVALID')
            else:
                value = {k: value[k] for k in ('plan_status', 'claims', 'requirement_issues')}
                value['unresolved_slots'] = [{'slot_id': f'S{i+1}', 'purpose': text, 'status': 'UNBOUND'} for i, text in enumerate(refs)]
                audit.append({'scope': 'root_layout', 'code': 'V31_READ_ONLY_LAYOUT_PROJECTION',
                              'before': old, 'after': deepcopy(value), 'claim_nodes_unchanged': True})
                source_warnings.append({'code': 'HISTORICAL_UNCOVERED_IDS_NOT_USED_AS_COVERAGE',
                                        'value': old['uncovered_requirements']})
    raw_errors = shape_errors(value, NORMALIZED_SCHEMA)
    if isinstance(value, dict) and isinstance(value.get('claims'), list):
        for index, c in enumerate(value['claims']):
            if not isinstance(c, dict): continue
            for field in ('requires', 'alternative_to'):
                if field not in c:
                    c[field] = []
                    audit.append({'scope': 'claim_field', 'candidate_index': index, 'claim_id': c.get('claim_id'),
                                  'field': field, 'code': 'DEFAULT_NO_DECLARED_EDGE',
                                  'before': {'present': False}, 'after': [],
                                  'semantic_independence_established': False})
            if isinstance(c.get('type'), str) and c['type'] in ALIASES:
                before = c['type']; c['type'] = ALIASES[before]
                audit.append({'scope': 'claim_field', 'candidate_index': index, 'claim_id': c.get('claim_id'),
                              'field': 'type', 'code': 'DOCUMENTED_ENUM_ALIAS', 'before': before, 'after': c['type']})
    return {'graph': value, 'normalizations': audit, 'raw_schema_errors': raw_errors,
            'source_layout_warnings': source_warnings, 'parse_errors': errors}
