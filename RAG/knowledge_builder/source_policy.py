"""Fail-closed source approval. Generic medical terms are deliberately allowed."""
import re
from .common import stable_hash

VERSION='surgical-independent-source-policy-v0.1'
BENCHMARK=re.compile(r'\b(?:MedVidU|CholecT50|CholecTrack20|Endoscapes|CoPESD|JIGSAWS|NurViD|EgoSurgery|VideoQA|benchmark|ground[ -]?truth|CVAT|OpenSurgicalVision)\b|dataset[- ]specific|mask annotation|annotation guidelines',re.I)
SOURCE_REVIEW_KEYS={'source_id','file_path','title','sha256','decision','reviewer_id','rationale','independence_basis','allowed_pages'}


def source_findings(name,text):
    matches=sorted({m.group(0) for m in BENCHMARK.finditer(name+'\n'+text)})
    return {'policy_version':VERSION,'status':'QUARANTINED_BENCHMARK_RISK' if matches else 'INDEPENDENCE_REVIEW_REQUIRED',
            'risk_terms':matches,'benchmark_independent':None,
            'extraction_allowed':False,'reason':'Explicit benchmark/project material requires exclusion; independent source approval is never inferred from a filename.' if matches else 'No automatic leak signature found; this is not proof of independence.'}


def review_template(source):
    return {'source_id':source['source_id'],'file_path':source['file_path'],'title':source['title'],
            'sha256':source['sha256'],'decision':'PENDING','reviewer_id':'',
            'rationale':'','independence_basis':'','allowed_pages':[]}


def apply_review(source,review):
    decision=review.get('decision')
    if set(review)!=SOURCE_REVIEW_KEYS or any(review[k]!=source[k] for k in ('source_id','file_path','title','sha256')):
        raise ValueError('SOURCE_REVIEW_BINDING_INVALID')
    if decision not in ('INCLUDE','EXCLUDE','PENDING'):raise ValueError('SOURCE_REVIEW_DECISION_INVALID')
    if any(type(p) is not int or not 1<=p<=source['pdf_pages'] for p in review['allowed_pages']):raise ValueError('SOURCE_REVIEW_PAGE_INVALID')
    if len(set(review['allowed_pages']))!=len(review['allowed_pages']):raise ValueError('SOURCE_REVIEW_DUPLICATE_PAGE')
    if decision=='INCLUDE':
        if not all(isinstance(review[k],str) and review[k].strip() for k in ('reviewer_id','rationale','independence_basis')):raise ValueError('SOURCE_REVIEW_INCOMPLETE')
        # A flagged mixed document cannot be admitted wholesale. Even explicitly
        # approved pages are filtered per block and again per output card.
        if source['source_audit']['risk_terms'] and not review['allowed_pages']:raise ValueError('MIXED_SOURCE_REQUIRES_PAGE_ALLOWLIST')
    return {**source,'benchmark_independent':True if decision=='INCLUDE' else None,
            'source_review':review,'source_review_sha256':stable_hash(review),
            'source_audit':{**source['source_audit'],'extraction_allowed':decision=='INCLUDE',
                            'status':'REVIEWED_INCLUDE' if decision=='INCLUDE' else 'REVIEWED_EXCLUDE' if decision=='EXCLUDE' else 'INDEPENDENCE_REVIEW_REQUIRED'}}
