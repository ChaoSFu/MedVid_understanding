"""Minimal-graph structural checks and explicitly limited semantic-risk audit."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import re

from relive.claim_planner.contract import shape_errors
from relive.storage.artifacts import stable_hash
from relive.v2.task_selection import strict_json_loads
from .contract import SCHEMA, CLAIM_SCHEMA

def text_audit(text, row, *, strict_numbers=True):
    errors, risks = [], []
    if not isinstance(text,str): return ['CLAIM_TEXT_INVALID'],['SEMANTICS_NOT_EVALUABLE']
    if strict_numbers and re.search(r'\d',text): errors.append('MODEL_NUMERIC_INSTANTIATION_FORBIDDEN')
    if re.search(r'\b(?:fps|stride|frame(?: index)?)\s*[:=]?\s*\d|\[\s*\d+\s*,\s*\d+',text,re.I): errors.append('EXECUTION_BINDING_FORBIDDEN')
    if re.search(r'\b(?:always|never|every frame|all (?:sampled |provided )?frames|throughout (?:the )?(?:entire|whole))\b',text,re.I): errors.append('UNBOUNDED_SCOPE_FORBIDDEN')
    if re.search(r'\bno\b.{0,100}\babsent\b|\bnot\b.{0,40}\bnot\b',text,re.I): errors.append('DOUBLE_NEGATION_FORBIDDEN')
    if re.search(r'\b(?:followed (?:immediately )?by|then|in order to|so that)\b|\bheld\b.{0,60}\band (?:is )?positioned\b',text,re.I): errors.append('EXPLICIT_MULTI_PREDICATE_CHAIN')
    if re.search(r'\b(?:score|rating)\s*(?:is|:|=|of)\s*\d|\b(?:is verified|has been verified|is confirmed|proven to)\b',text,re.I): errors.append('FINAL_VERDICT_FORBIDDEN')
    if re.search(r'\b(?:and|or)\b|;|\b(?:suggest\w*|indicat\w*|confirms?|efficient|logical|adequate|continuous|without occlusion)\b',text,re.I): risks.append('ATOMICITY_INFERENCE_OR_CONTINUITY_REVIEW')
    if re.search(r'\b(?:must|necessary|requires)\b.{0,55}\b(?:absent|absence|no )|\b(?:absent|absence)\b.{0,55}\b(?:must|required|necessary)\b',text,re.I): risks.append('UNSUPPORTED_NECESSITY_REVIEW')
    if re.search(r'\b(?:starts?|ends?) at (?:a )?specific timestamp|bounding box.*(?:valid|coordinates)|can be determined',text,re.I): risks.append('META_DESCRIPTION_TESTABILITY_REVIEW')
    if re.search(r'\b(?:abstract|candidate|question.relevant|task.relevant) (?:event|target|object|condition)\b|\{[^}]+\}|\$\w+',text,re.I): risks.append('ABSTRACT_TEMPLATE_REVIEW')
    public=(row['question']+' '+' '.join(r['description'] for r in row['requirements'])).casefold()
    for word in re.findall(r'\b(?:purple|red|blue|green|yellow|left|right|upper|lower|proximal|distal)\b',text,re.I):
        if not re.search(r'\b'+re.escape(word.casefold())+r'\b',public): risks.append('UNSUPPORTED_VISUAL_SPECIFICITY_REVIEW')
    if re.search(r'\b(?:will|going to|next action is|next phase is)\b',text,re.I): risks.append('PROSPECTIVE_FACT_REVIEW')
    return sorted(set(errors)),sorted(set(risks))

def validate(raw,row):
    result={'plan_status':'MODEL_OR_SCHEMA_FAILURE','claims':[],'claim_audit':[], 'root_errors':[],
            'unresolved_references':[],'requirement_issues':[],'uncovered_requirements':[],
            'semantic_quality':'REVIEW_REQUIRED'}
    try: value=strict_json_loads(raw or '',error_code='V3_JSON_INVALID')
    except (ValueError,TypeError,RecursionError):
        result['root_errors']=['STRICT_JSON_PARSE_FAILURE'];return result
    root_schema=deepcopy(SCHEMA);root_schema['properties']['claims']['items']={}
    errors=shape_errors(value,root_schema)
    if errors:result['root_errors']=errors;return result
    for k in ('unresolved_references','requirement_issues','uncovered_requirements'):result[k]=value[k]
    reqs={r['requirement_id'] for r in row['requirements']}
    if any(r['requirement_id'] not in reqs for r in value['requirement_issues']) or not set(value['uncovered_requirements'])<=reqs:
        result['root_errors'].append('REQUIREMENT_METADATA_REFERENCE_INVALID')
    if value['plan_status']=='PLANNED' and not value['claims']:result['root_errors'].append('EMPTY_PLANNED_GRAPH')
    if value['plan_status']=='UNABLE_TO_PLAN' and value['claims']:result['root_errors'].append('UNABLE_PLAN_WITH_CLAIMS')
    rc=row['qa_type'].startswith('region_caption')
    if rc and (value['plan_status']!='UNRESOLVED_TARGET_REFERENCE' or not value['unresolved_references']):
        result['root_errors'].append('RC_UNRESOLVED_REFERENCE_REQUIRED')
    ids=Counter(c.get('claim_id') for c in value['claims'] if isinstance(c,dict) and isinstance(c.get('claim_id'),str))
    records=[]
    for i,c in enumerate(value['claims']):
        schema=shape_errors(c,CLAIM_SCHEMA);hard=list(result['root_errors']);risks=[]
        if isinstance(c,dict):
            e,w=text_audit(c.get('text'),row);hard+=e;risks+=w
            if isinstance(c.get('claim_id'),str) and ids[c['claim_id']]>1:hard.append('DUPLICATE_CLAIM_ID')
            links=c.get('requirement_ids')
            if isinstance(links,list) and all(isinstance(r,str) for r in links):
                if not set(links)<=reqs or len(set(links))!=len(links):hard.append('REQUIREMENT_REFERENCE_INVALID')
                if not links:risks.append('QUESTION_ONLY_RELEVANCE_REVIEW')
            if c.get('type')=='HIGH_LEVEL':
                risks.append('HIGH_LEVEL_INFERENCE_RULE_REVIEW')
                if c.get('role')!='INTERPRETATION':risks.append('INTERPRETATION_ROLE_REVIEW')
                if not c.get('requires'):risks.append('HIGH_LEVEL_PREREQUISITES_MISSING')
            if c.get('role')=='INTERPRETATION' and c.get('type')!='HIGH_LEVEL':risks.append('TYPE_ROLE_ALIGNMENT_REVIEW')
            if c.get('type') in ('OBJECT','ATTRIBUTE') and re.search(r'\b(?:grasps|retracts|moves|is held|is positioned near)\b',str(c.get('text','')),re.I):hard.append('TYPE_PREDICATE_CONFLICT')
            if rc:
                risks.append('UNRESOLVED_TARGET_TEMPLATE')
                if not re.search(r'\b(?:queried target|unresolved target|target object)\b|\{target\}|\$target',str(c.get('text','')),re.I):hard.append('RC_TARGET_GUESS_FORBIDDEN')
        records.append({'candidate_index':i,'raw_candidate':c,'schema_errors':schema,'errors':sorted(set(hard)),
                        'risks':sorted(set(risks)),'dependency_errors':[],'blocked_by':[],
                        'structural_status':'FAIL' if schema else 'PASS','state':'PENDING'})
    by_id={r['raw_candidate']['claim_id']:r for r in records if not r['schema_errors'] and ids[r['raw_candidate']['claim_id']]==1}
    requires={k:list(r['raw_candidate']['requires']) for k,r in by_id.items()}
    incoming=Counter(ref for refs in requires.values() for ref in refs)
    def reaches(node,target,seen):return node==target or (node not in seen and any(reaches(n,target,seen|{node}) for n in requires.get(node,[]) if n in by_id))
    for cid,r in by_id.items():
        c=r['raw_candidate']
        for field in ('requires','alternative_to'):
            if len(set(c[field]))!=len(c[field]):r['dependency_errors'].append('DUPLICATE_EDGE')
            if any(d not in by_id or d==cid for d in c[field]):r['dependency_errors'].append('DANGLING_OR_SELF_REFERENCE')
        if any(reaches(n,cid,set()) for n in requires[cid] if n in by_id):r['dependency_errors'].append('REQUIRES_CYCLE')
        if c['alternative_to']:r['risks'].append('ALTERNATIVE_EXCLUSIVITY_REVIEW')
        if c['requires']:r['risks'].append('LOGICAL_DEPENDENCY_REVIEW')
        if c['role']=='PREREQUISITE' and not incoming[cid]:r['risks'].append('UNUSED_PREREQUISITE_REVIEW')
        if any(by_id[d]['raw_candidate']['type']=='HIGH_LEVEL' for d in c['requires'] if d in by_id):r['risks'].append('INTERPRETATION_AS_PREMISE_REVIEW')
    for r in records:
        r['risks']=sorted(set(r['risks']));r['dependency_errors']=sorted(set(r['dependency_errors']))
        if r['dependency_errors']:r['structural_status']='FAIL'
        r['state']='REJECTED' if r['schema_errors'] or r['errors'] or r['dependency_errors'] else 'REVIEW_REQUIRED' if r['risks'] else 'STRUCTURALLY_ACCEPTED'
    def prune():
        changed=True
        while changed:
            changed=False
            for r in records:
                if r['schema_errors']:continue
                c=r['raw_candidate'];r['blocked_by']=[d for d in c['requires']+c['alternative_to'] if d not in by_id or by_id[d]['state'] in ('REJECTED','BLOCKED_BY_DEPENDENCY')]
                if r['blocked_by'] and r['state'] in ('STRUCTURALLY_ACCEPTED','REVIEW_REQUIRED'):
                    r['state']='BLOCKED_BY_DEPENDENCY';changed=True
    prune()
    seen=set()
    for r in records:
        if r['state'] not in ('STRUCTURALLY_ACCEPTED','REVIEW_REQUIRED'):continue
        c=r['raw_candidate'];norm=' '.join(re.findall(r'\w+',re.sub(r'\bt(?:_[ij])?\b','TIME_PARAMETER',c['text'].casefold())))
        if norm in seen:r['errors'].append('DUPLICATE_PROPOSITION');r['state']='REJECTED'
        else:seen.add(norm)
    prune()
    for r in records:
        if not r['schema_errors']:
            c={**r['raw_candidate'],'status':'UNVERIFIED','evidence_status':'NOT_SEARCHED'}
            r['claim_sha256']=stable_hash(c)
            if r['state'] in ('STRUCTURALLY_ACCEPTED','REVIEW_REQUIRED'):result['claims'].append(c)
    result['claim_audit']=records
    used={rid for c in result['claims'] for rid in c['requirement_ids']}
    result['program_uncovered_requirements']=sorted(reqs-used)
    result['coverage_report_disagrees']=set(value['uncovered_requirements'])!=reqs-used
    result['plan_status']=value['plan_status'] if not result['root_errors'] else 'MODEL_OR_SCHEMA_FAILURE'
    if not rc and value['claims'] and not result['claims']:result['plan_status']='NO_ADMITTED_HYPOTHESES'
    result['graph_sha256']=stable_hash({k:result[k] for k in ('claims','unresolved_references','requirement_issues','uncovered_requirements')})
    return result
