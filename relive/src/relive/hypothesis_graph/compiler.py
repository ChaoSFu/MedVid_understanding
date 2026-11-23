"""Pure deterministic compiler. No backend, image loader or certificate API.

Controlled-English extraction is intentionally conservative. Human review
authorizes a planning interpretation, never certifies visual truth.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import re

from relive.storage.artifacts import stable_hash
from .sources import digest, rows, write, new_dir, seal, load_run

VERSION = 'relive-hypothesis-contract-compiler-v1'


def policy():
    p = {'version': VERSION, 'implementation_sha256': digest(__file__),
         'modalities': {'OBJECT': 'SINGLE_FRAME', 'ATTRIBUTE': 'SINGLE_FRAME',
                        'STATIC_RELATION': 'MULTI_REGION_SINGLE_FRAME',
                        'TEMPORAL_EVENT': 'MULTI_FRAME', 'HIGH_LEVEL': 'COMPOSITIONAL'},
         'all_retained_nodes_require_human_approval': True,
         'synthetic_publication_forbidden': True, 'unknown_parse': 'UNCOMPILABLE',
         'truth_propagation': 'FORBIDDEN', 'model_calls': 0, 'image_reads': 0}
    p['policy_sha256'] = stable_hash(p)
    return p


def _entity(text):
    return re.sub(r'^(?:the|a|an)\s+', '', text.strip(), flags=re.I)


def compile_node(claim, graph, review=None):
    """No invented entities: every extracted target is a substring of the claim.

    This recognizes a limited English predicate vocabulary, not general NLP.
    Unknown sentences stay UNCOMPILABLE even when the type label is legal.
    """
    text = claim['text']; kind = claim['type']; review = review or {}
    base = {'claim_id': claim['claim_id'], 'claim_type':kind, 'claim_text': text, 'claim_sha256': stable_hash(claim),
            'status': 'UNVERIFIED', 'evidence_status': 'NOT_SEARCHED',
            'candidate_frames': [], 'spatial_regions': [], 'time_bindings': {},
            'requires': claim['requires'], 'alternative_to': claim['alternative_to']}
    def fail(reason):
        return {**base, 'compilation_status': 'UNCOMPILABLE', 'reason': reason}
    if any(re.search(pattern, text, re.I) for pattern in
           (r'\d', r'\b(?:queried target|unresolved target|target object|question.relevant|candidate event)\b', r'\{[^}]+\}', r'\b(?:then|so that|in order to)\b')):
        return fail('UNRESOLVED_OR_UNSUPPORTED_PROPOSITION')
    targets=[]; details={}
    stem = re.sub(r'[.]$', '', text.strip())
    stem = re.sub(r'\s+at time t$', '', stem, flags=re.I)
    if kind == 'OBJECT':
        m=re.fullmatch(r'(.+?) (?:is|are) (?:visible|present)', stem, re.I)
        if not m:return fail('OBJECT_TARGET_NOT_PARSEABLE')
        targets=[_entity(m[1])]
    elif kind == 'ATTRIBUTE':
        m=re.fullmatch(r'(.+?) (?:is|are) (open|closed|bent|straight|intact|occluded|partially occluded)', stem, re.I)
        if not m:return fail('ATTRIBUTE_NOT_PARSEABLE')
        targets=[_entity(m[1])];details={'discriminative_attribute':m[2].casefold()}
    elif kind == 'STATIC_RELATION':
        m=re.fullmatch(r'(.+?) (grasps|encloses|contacts|touches|overlaps|is inside|is above|is below|is to the right of|is to the left of|is separated from) (.+)', stem, re.I)
        if not m:return fail('RELATION_ARGUMENTS_NOT_PARSEABLE')
        targets=[_entity(m[1]),_entity(m[3]),
                 'contact_interface' if m[2].casefold() in ('grasps','contacts','touches') else 'relation_interface']
        details={'relation_predicate':m[2].casefold()}
    elif kind == 'TEMPORAL_EVENT':
        m=re.fullmatch(r'(.+?) (moves|retracts|lifts|rotates|opens|closes|changes|shifts|zooms|pans|separates|cuts)(?: (.+))?', stem, re.I)
        if not m:return fail('TEMPORAL_CHANGE_NOT_PARSEABLE')
        targets=[_entity(m[1])]
        if m[3]:targets.append(_entity(m[3]))
        details={'temporal_change_requirement':text, 'minimum_observations':2,
                 'continuity': 'NOT_ESTABLISHED_BY_SPARSE_SAMPLING'}
    elif kind == 'HIGH_LEVEL':
        prereqs=[graph.get(cid) for cid in claim['requires']]
        inference=review.get('inference_requirement')
        if not prereqs or any(c is None or c['type']=='HIGH_LEVEL' for c in prereqs):return fail('LOWER_LEVEL_PREREQUISITES_REQUIRED')
        if all(c['type']=='OBJECT' for c in prereqs):return fail('PRESENCE_ALONE_CANNOT_ESTABLISH_INTERPRETATION')
        if not isinstance(inference,str) or not inference.strip():return fail('EXPLICIT_INFERENCE_REQUIREMENT_REQUIRED')
        details={'prerequisite_facts':claim['requires'], 'inference_requirement':inference,
                 'inference_status':'NOT_PERFORMED', 'prerequisites_imply_claim':False}
    else:return fail('UNSUPPORTED_CLAIM_TYPE')
    if any(not t.strip() or re.search(r'\b(?:and|or|will|must)\b', t, re.I) for t in targets):
        return fail('AMBIGUOUS_ENTITY_ARGUMENTS')
    return {**base, 'compilation_status':'COMPILED', 'evidence_modality':policy()['modalities'][kind],
            'spatial_targets':targets, **details}


def review_rows(run_dir, references_dir):
    samples,_=load_run(run_dir,references_dir);binding=digest(Path(run_dir)/'manifest.json')
    result=[]
    for s in samples:
        for c in s['claims']:
            audit=next(a for a in s['claim_audit'] if a.get('claim_sha256')==stable_hash(c))
            result.append({'format':'relive-v3-semantic-review-v1','sample_key':s['sample_key'],
                'claim_id':c['claim_id'],'run_manifest_sha256':binding,'graph_sha256':s['graph_sha256'],
                'claim_sha256':stable_hash(c),'claim_text':c['text'],'validator_state':audit['state'],
                'review_flags':audit['risks'],'requirement_issues':s['requirement_issues'],
                'decision':'PENDING','reviewer_id':'','rationale':'','inference_requirement':None,
                'requirement_issues_resolution':''})
    return result


def make_review_template(*,run_dir,references_dir,output_dir):
    out=new_dir(output_dir);rr=review_rows(run_dir,references_dir)
    write(out/'semantic_review.jsonl',rr,jsonl=True)
    seal(out,['semantic_review.jsonl'],'relive-v3-review-template-v1',source_run_sha256=digest(Path(run_dir)/'manifest.json'))
    return {'status':'REVIEW_REQUIRED','rows':len(rr),'new_model_calls':0,'image_reads':0}


def compile_bank(*,run_dir,references_dir,output_dir,review_jsonl=None,publish=False):
    samples,m=load_run(run_dir,references_dir)
    expected={(r['sample_key'],r['claim_id']):r for r in review_rows(run_dir,references_dir)}
    supplied={};errors=[];watch={str(Path(run_dir)/'manifest.json'):digest(Path(run_dir)/'manifest.json')}
    if review_jsonl:
        watch[str(review_jsonl)]=digest(review_jsonl)
        for r in rows(review_jsonl):
            key=(r.get('sample_key'),r.get('claim_id'))
            if key not in expected or key in supplied:
                errors.append('UNEXPECTED_OR_DUPLICATE_REVIEW');continue
            template=expected[key]
            protected=set(template)-{'decision','reviewer_id','rationale','inference_requirement','requirement_issues_resolution'}
            if any(r.get(k)!=template[k] for k in protected):errors.append('REVIEW_BINDING_MISMATCH')
            supplied[key]=r
    if publish:
        if m['synthetic_test'] or m['identity']['dry_run']:errors.append('SYNTHETIC_OR_UNEXECUTED_PUBLICATION_FORBIDDEN')
        from .sources import load_references
        if not load_references(references_dir)[1].get('stage1_independently_checked',False):
            errors.append('INDEPENDENT_STAGE1_BINDING_REQUIRED')
    diagnostics=[];potential=Counter()
    for s in samples:
        graph={c['claim_id']:c for c in s['claims']}
        if s['source_binding_status']!='PASS' or s['root_errors']:errors.append('SOURCE_OR_ROOT_VALIDATION_FAILED')
        if s['plan_status'] not in ('PLANNED','INCOMPLETE_PLAN','UNRESOLVED_TARGET_REFERENCE'):errors.append('PLAN_NOT_USABLE')
        if s['unresolved_references'] and s['claims']:errors.append('UNRESOLVED_REFERENCES_BLOCK_PUBLICATION')
        for c in s['claims']:
            key=(s['sample_key'],c['claim_id']);r=supplied.get(key,{})
            reviewed=r.get('decision')=='APPROVE' and all(isinstance(r.get(k),str) and r[k].strip() for k in ('reviewer_id','rationale'))
            if s['requirement_issues'] and not r.get('requirement_issues_resolution','').strip():reviewed=False
            if not reviewed:errors.append('SEMANTIC_REVIEW_REQUIRED')
            compiled=compile_node(c,graph,r)
            if compiled['compilation_status']!='COMPILED':errors.append('UNCOMPILABLE_NODE')
            elif c['type'] in ('OBJECT','ATTRIBUTE','STATIC_RELATION'):
                potential[c['type']]+=1
            diagnostics.append({'sample_key':s['sample_key'],'graph_sha256':s['graph_sha256'],
                                'provenance':{'run_manifest_sha256':digest(Path(run_dir)/'manifest.json'),
                                              'references_manifest_sha256':digest(Path(references_dir)/'manifest.json'),
                                              'review_sha256':digest(review_jsonl) if review_jsonl else None,
                                              'compiler_policy_sha256':policy()['policy_sha256']},
                                'semantic_review_complete':reviewed,'diagnostic_only':True,'contract':compiled})
    if not diagnostics:errors.append('NO_COMPILABLE_CANDIDATES')
    # A compiled prerequisite is necessary; never publish a dangling contract.
    compiled_ids={(d['sample_key'],d['contract']['claim_id']) for d in diagnostics if d['contract']['compilation_status']=='COMPILED'}
    for d in diagnostics:
        if any((d['sample_key'],cid) not in compiled_ids for cid in d['contract']['requires']):errors.append('COMPILED_DEPENDENCY_MISSING')
    allowed=publish and not errors
    out=new_dir(output_dir)
    write(out/'contract_diagnostics.jsonl',diagnostics,jsonl=True)
    write(out/'evidence_contracts.jsonl',[{**d,'diagnostic_only':False} for d in diagnostics] if allowed else [],jsonl=True)
    audit={'format':VERSION,'status':'PUBLISHED' if allowed else 'FORMAL_PUBLICATION_BLOCKED' if publish else 'DRY_RUN',
           'publication_allowed':allowed,'formal_contract_count':len(diagnostics) if allowed else 0,
           'diagnostic_node_count':len(diagnostics),'gate_reasons':sorted(set(errors)),
           'potential_spatial_candidates_by_type':dict(sorted(potential.items())),
           'spatial_experiment_admission':'NOT_PERFORMED', 'policy':policy(),
           'source_run_sha256':digest(Path(run_dir)/'manifest.json'),'review_sha256':digest(review_jsonl) if review_jsonl else None,
           'new_model_calls':0,'image_reads':0,'gt_used':False,'cache_writes':0,'certificate_writes':0,'new_verified_count':0}
    write(out/'contract_compiler_audit.json',audit)
    if any(digest(p)!=h for p,h in watch.items()):raise ValueError('FROZEN_INPUT_CHANGED')
    seal(out,['contract_diagnostics.jsonl','evidence_contracts.jsonl','contract_compiler_audit.json'],'relive-v3-compiler-manifest-v1')
    return audit
