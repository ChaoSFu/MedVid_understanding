"""One transactional patch; no silent rewriting or repeated repair."""
from copy import deepcopy
from relive.v2.task_selection import strict_json_loads
from relive.claim_planner.contract import shape_errors
from .contract import CLAIM, SLOT
from .gate import validate, canonical_text

OPS = {'KEEP', 'SPLIT', 'MERGE', 'DROP', 'DEMOTE_TO_SLOT', 'FIX_EDGE', 'FIX_TYPE', 'REPLACE_GRAPH'}


def flag_set(gate):
    return set(gate['root_errors'] + gate['graph_risks'] + [
        f"{a['claim_id']}:{e}" for a in gate['claim_audit'] for e in a['errors'] + a['risks']])


def apply_patch_once(raw_patch, before, row):
    """Rollback invalid patches. Code-flag clearance is not semantic improvement."""
    audit = {'status': 'ROLLED_BACK', 'operations': [], 'reason': None,
             'semantic_improvement': 'NOT_REVIEWED', 'code_flags_cleared': [], 'code_flags_remaining': []}
    try:
        patch = strict_json_loads(raw_patch, error_code='PATCH_PARSE_INVALID')
        if (not isinstance(patch, dict) or set(patch) != {'operations', 'rationale'}
                or not isinstance(patch['rationale'], str) or not patch['rationale'].strip()
                or not isinstance(patch['operations'], list) or not 1 <= len(patch['operations']) <= 48):
            raise ValueError('PATCH_ROOT_INVALID')
        graph = deepcopy(before['normalized_graph'])
        original = deepcopy(graph)
        touched = set()
        flagged = {a['claim_id'] for a in before['claim_audit'] if a['errors'] or a['risks']}
        if graph:
            # Edges into changed nodes may also need explicit updates.
            flagged |= {c.get('claim_id') for c in graph.get('claims', []) if isinstance(c, dict)
                        and any(x in flagged for x in c.get('requires', []) + c.get('alternative_to', []))}
        for op in patch['operations']:
            if not isinstance(op, dict) or op.get('op') not in OPS:
                raise ValueError('PATCH_OPERATION_INVALID')
            root_replace = op['op'] == 'REPLACE_GRAPH'
            if set(op) != {'op', 'ids', 'claims', 'slots', 'reason'} | ({'graph'} if root_replace else set()):
                raise ValueError('PATCH_OPERATION_FIELDS_INVALID')
            if not isinstance(op['reason'], str) or not op['reason'].strip(): raise ValueError('PATCH_REASON_REQUIRED')
            if any(not isinstance(op[k], list) for k in ('ids', 'claims', 'slots')): raise ValueError('PATCH_ARRAY_REQUIRED')
            if any(not isinstance(x, str) for x in op['ids']): raise ValueError('PATCH_ID_INVALID')
            if any(shape_errors(c, CLAIM) for c in op['claims']): raise ValueError('PATCH_CLAIM_SCHEMA_INVALID')
            if any(shape_errors(s, SLOT) for s in op['slots']): raise ValueError('PATCH_SLOT_SCHEMA_INVALID')
            if root_replace:
                no_local_nodes = not (before['normalized_graph'] or {}).get('claims') and before['repair_needed']
                if len(patch['operations']) != 1 or not (before['root_errors'] or no_local_nodes) or op['ids'] or op['claims'] or op['slots']:
                    raise ValueError('WHOLE_GRAPH_REPAIR_NOT_JUSTIFIED')
                graph = deepcopy(op['graph']); continue
            if not isinstance(graph, dict) or not isinstance(graph.get('claims'), list): raise ValueError('LOCAL_PATCH_REQUIRES_GRAPH')
            ids = op['ids']; kind = op['op']
            if not ids or len(set(ids)) != len(ids) or touched & set(ids): raise ValueError('PATCH_TARGET_DUPLICATE')
            selected = [c for c in graph['claims'] if isinstance(c, dict) and c.get('claim_id') in ids]
            if len(selected) != len(ids): raise ValueError('PATCH_TARGET_MISSING_OR_AMBIGUOUS')
            duplicate_merge = kind == 'MERGE' and bool(set(ids) & flagged) and len({
                canonical_text(c.get('text', '')) for c in selected}) == 1
            if kind != 'KEEP' and not (set(ids) <= flagged or duplicate_merge or before['graph_risks']):
                raise ValueError('UNFLAGGED_NODE_MODIFICATION_FORBIDDEN')
            touched.update(ids)
            if kind in ('KEEP', 'DROP') and (len(ids) != 1 or op['claims'] or op['slots']): raise ValueError('PATCH_CARDINALITY')
            if kind == 'SPLIT' and (len(ids) != 1 or len(op['claims']) < 2 or op['slots']): raise ValueError('PATCH_CARDINALITY')
            if kind == 'MERGE' and (len(ids) < 2 or len(op['claims']) != 1 or op['slots']): raise ValueError('PATCH_CARDINALITY')
            if kind == 'DEMOTE_TO_SLOT' and (len(ids) != 1 or op['claims'] or not op['slots']): raise ValueError('PATCH_CARDINALITY')
            if kind in ('FIX_EDGE', 'FIX_TYPE'):
                if len(ids) != 1 or len(op['claims']) != 1 or op['slots']: raise ValueError('PATCH_CARDINALITY')
                allowed = {'requires', 'alternative_to'} if kind == 'FIX_EDGE' else {'type'}
                if any(selected[0].get(k) != op['claims'][0].get(k) for k in set(selected[0]) | set(op['claims'][0]) if k not in allowed):
                    raise ValueError('PATCH_CHANGED_UNAUTHORIZED_FIELDS')
            if kind == 'KEEP': continue
            # Keep relative order: replacement occupies the first selected position.
            insertion = next(i for i, c in enumerate(graph['claims']) if isinstance(c, dict) and c.get('claim_id') in ids)
            graph['claims'] = [c for c in graph['claims'] if not isinstance(c, dict) or c.get('claim_id') not in ids]
            graph['claims'][insertion:insertion] = deepcopy(op['claims'])
            graph['unresolved_slots'].extend(deepcopy(op['slots']))
        if not graph.get('claims') and graph.get('plan_status') == 'PLANNED': graph['plan_status'] = 'INCOMPLETE_PLAN'
        after = validate(graph, row)
        before_flags, after_flags = flag_set(before), flag_set(after)
        if after['root_errors'] or any(a['errors'] for a in after['claim_audit']):
            raise ValueError('PATCH_REVALIDATION_FAILED')
        # New review flags are not silently traded for cleared flags.
        if after_flags - before_flags:
            raise ValueError('PATCH_INTRODUCED_NEW_FLAGS')
        audit.update(status='APPLIED', operations=patch['operations'], reason=patch['rationale'],
                     original_graph=original, patched_graph=graph,
                     code_flags_cleared=sorted(before_flags - after_flags), code_flags_remaining=sorted(after_flags),
                     code_flag_resolution=not after['repair_needed'])
        return after, audit
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        audit['reason'] = str(exc) if isinstance(exc, ValueError) else 'PATCH_STRUCTURE_INVALID'
        return before, audit
