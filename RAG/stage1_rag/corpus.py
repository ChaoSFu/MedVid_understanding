"""CPU-only frozen lexical index. Never reads benchmark questions or answers."""
from collections import Counter
import re
import sqlite3
from pathlib import Path
from RAG.knowledge_builder.validator import validate_library,validate_output
from RAG.knowledge_builder.source_policy import BENCHMARK
from .common import new_output,separate_output,seal,verify,read_json,write_json,file_hash,stable_hash,canonical,jsonl

POLICY={'version':'stage1-rag-retrieval-v1','top_k':4,'candidate_k':20,'max_per_source':2,
        'max_context_bytes':6000,'budget_unit':'UTF8_BYTES_NOT_MODEL_TOKENS',
        'query_fields':['question','qa_type'],'ranker':'SQLite_FTS5_BM25',
        'weights':[5.0,2.0,1.0,2.0],'tokenizer':'porter unicode61',
        'tie_breaker':'chunk_id','minimum_literal_query_overlap':1}
STOP=set('a an the is are was were be been being this that these those there their it its to of and or in on at for from with by as which what when where how why you your we our i have has had can could should would will must do does did not no video videos frame frames question answer describe identify list following provided given expert analyst medical surgical surgery procedure procedures footage shows show please within between seconds second generate all each'.split())


def tokens(text):
    return re.findall(r'[a-z][a-z0-9]*',text.casefold())


def query(question):
    # Preserve the tail/query target on long questions, not just the preamble.
    ts=list(dict.fromkeys(t for t in tokens(question) if t not in STOP and len(t)>2))
    return ts if len(ts)<=128 else ts[:64]+ts[-64:]


def freeze(knowledge_json,output_dir,*,allow_pending=False,expected_sha=None,expected_count=None,
           references_root=None,parse_cache=None,synthetic_fixture=False):
    path=Path(knowledge_json);lib=read_json(path);sha=file_hash(path)
    if expected_sha and sha!=expected_sha:raise ValueError('KNOWLEDGE_HASH_MISMATCH')
    if expected_count is not None and len(lib['chunks'])!=expected_count:raise ValueError('KNOWLEDGE_COUNT_MISMATCH')
    validate_library(lib)
    if lib['synthetic_test'] and not synthetic_fixture:raise ValueError('MOCK_CORPUS_FORBIDDEN')
    if lib['build_status'] not in ('BUILT_PENDING_MEDICAL_REVIEW','MOCK_ONLY'):raise ValueError('KNOWLEDGE_BUILD_INCOMPLETE')
    if not allow_pending:raise ValueError('PENDING_KNOWLEDGE_REQUIRES_EXPLICIT_RESEARCH_MODE')
    sources={s['source_id']:s for s in lib['source_index']}
    for c in lib['chunks']:
        if BENCHMARK.search(canonical(c)):raise ValueError('BENCHMARK_CONTENT_IN_KNOWLEDGE')
        for ref in c['source_refs']:
            s=sources[ref['source_id']]
            if not lib['synthetic_test'] and (s['benchmark_independent'] is not True or s.get('source_audit',{}).get('extraction_allowed') is not True):
                raise ValueError('KNOWLEDGE_SOURCE_NOT_ADMITTED')
    if not lib['chunks']:raise ValueError('EMPTY_KNOWLEDGE')
    source_audit=None
    if (path.parent/'build_summary.json').exists():
        summary=read_json(path.parent/'build_summary.json')
        if summary['status']!='BUILT_PENDING_MEDICAL_REVIEW' and not lib['synthetic_test']:raise ValueError('SOURCE_BUILD_INCOMPLETE')
        for name,h in summary['artifact_sha256'].items():
            if Path(name).name!=name or file_hash(path.parent/name)!=h:raise ValueError('SOURCE_BUILD_ARTIFACT_HASH_MISMATCH')
        source_audit=file_hash(path.parent/'build_summary.json')
    if references_root:
        if not parse_cache:raise ValueError('PARSE_CACHE_REQUIRED')
        validate_output(path,references_root,parse_cache)
    keys=('chunk_id','knowledge_type','entity_name','aliases','procedure_scope','text','text_supporting_fact_ids',
          'visual_features','motion_features','spatial_relations','discriminative_features','confusable_with',
          'evidence_guidance','source_facts','derived_guidance','source_refs','review_status','knowledge_basis')
    cards=[{k:c[k] for k in keys} for c in sorted(lib['chunks'],key=lambda x:x['chunk_id'])]
    root=new_output(output_dir);write_json(root/'cards.jsonl',cards,jsonl=True)
    write_json(root/'sources.json',lib['source_index'])
    m=seal(root,'stage1-rag-corpus-v1',source_library_sha256=sha,source_build_summary_sha256=source_audit,
           card_count=len(cards),synthetic_test=lib['synthetic_test'],knowledge_policy='RESEARCH_PENDING_NOT_MEDICALLY_APPROVED',
           review_status_changed=False,model_calls_made=0,certificate_writes=0,new_verified_count=0,
           literal_pdf_quote_recheck='PASS' if references_root else 'NOT_RUN_USE_SOURCE_BUILD_PROVENANCE')
    if file_hash(path)!=sha:raise ValueError('SOURCE_KNOWLEDGE_CHANGED')
    return m


def index(corpus_dir,output_dir):
    separate_output(output_dir,corpus_dir)
    m=verify(corpus_dir,'stage1-rag-corpus-v1');cards=jsonl(Path(corpus_dir)/'cards.jsonl')
    root=new_output(output_dir)
    db=sqlite3.connect(root/'index.sqlite')
    try:
        db.execute('PRAGMA journal_mode=OFF')
        db.execute("CREATE VIRTUAL TABLE search USING fts5(names,body,cues,scope,chunk_id UNINDEXED, tokenize='porter unicode61')")
        for c in cards:
            cues=' '.join(x['text'] for k in ['visual_features','motion_features','spatial_relations','discriminative_features'] for x in c[k])
            cues+=' '+' '.join(x['text'] for v in c['evidence_guidance'].values() for x in v)
            db.execute('INSERT INTO search VALUES (?,?,?,?,?)',(' '.join([c['entity_name'],*c['aliases']]),c['text'],cues,' '.join(c['procedure_scope']),c['chunk_id']))
        db.commit()
    finally:db.close()
    write_json(root/'cards.jsonl',cards,jsonl=True);write_json(root/'retrieval_policy.json',POLICY)
    write_json(root/'sources.json',read_json(Path(corpus_dir)/'sources.json'))
    return seal(root,'stage1-rag-index-v1',corpus_manifest_sha256=file_hash(Path(corpus_dir)/'manifest.json'),
                source_library_sha256=m['source_library_sha256'],synthetic_test=m['synthetic_test'],
                knowledge_policy=m['knowledge_policy'],sqlite_version=sqlite3.sqlite_version,card_count=len(cards),model_calls_made=0)


def snippet(c):
    # Whole fields/items only; no truncation of a medical sentence or negation.
    return {k:c[k] for k in ('chunk_id','knowledge_type','entity_name','text','knowledge_basis','review_status',
                            'visual_features','discriminative_features','evidence_guidance','derived_guidance','source_facts','source_refs')}


class Retriever:
    def __init__(self,index_dir):
        self.root=Path(index_dir).resolve();self.manifest=verify(self.root,'stage1-rag-index-v1')
        self.policy=read_json(self.root/'retrieval_policy.json')
        if self.policy!=POLICY:raise ValueError('RETRIEVAL_POLICY_MISMATCH')
        self.cards={c['chunk_id']:c for c in jsonl(self.root/'cards.jsonl')}
        if len(self.cards)!=self.manifest['card_count']:raise ValueError('INDEX_CARD_COUNT_MISMATCH')

    def retrieve(self,question,qa_type):
        ts=query(question);ranked=[]
        # RC box-only references cannot be resolved by a medical prior.
        unresolved=qa_type.startswith('region_caption') and bool(re.search(r'\b(?:box|bbox|bounding)\b',question,re.I))
        if ts and not unresolved:
            match=' OR '.join('"'+t+'"' for t in ts)
            db=sqlite3.connect(self.root.joinpath('index.sqlite').as_uri()+'?mode=ro',uri=True)
            try:
                hits=db.execute('SELECT chunk_id,bm25(search,5,2,1,2) FROM search WHERE search MATCH ? ORDER BY 2,1 LIMIT ?',
                                (match,POLICY['candidate_k'])).fetchall()
            finally:db.close()
            for cid,score in hits:
                c=self.cards[cid];overlap=sorted(set(ts)&set(tokens(' '.join([c['entity_name'],*c['aliases'],c['text'],*c['procedure_scope']]))))
                if overlap:ranked.append({'chunk_id':cid,'bm25_score':score,'matched_terms':overlap})
        provided=[];selected=[];source_counts=Counter()
        for hit in ranked:
            c=self.cards[hit['chunk_id']];sources=sorted({r['source_id'] for r in c['source_refs']})
            if any(source_counts[s]>=POLICY['max_per_source'] for s in sources):continue
            full=snippet(c)
            variants=[full,{k:v for k,v in full.items() if k not in ('visual_features','discriminative_features','derived_guidance','evidence_guidance')},
                      {k:full[k] for k in ('chunk_id','knowledge_type','entity_name','text','knowledge_basis','review_status','source_refs')}]
            chosen=next((v for v in variants if len(canonical(provided+[v]).encode())<=POLICY['max_context_bytes']),None)
            if chosen is None:continue
            provided.append(chosen);selected.append(hit)
            for s in sources:source_counts[s]+=1
            if len(provided)==POLICY['top_k']:break
        return {'query_terms':ts,'qa_type':qa_type,'ranked_candidates':ranked,'provided_cards':provided,
                'selected':selected,'context_sha256':stable_hash(provided),'context_bytes':len(canonical(provided).encode()),
                'status':'UNRESOLVED_TARGET_REFERENCE' if unresolved else 'RETRIEVED' if provided else 'NO_MATCH_BASELINE_FALLBACK',
                'knowledge_policy':self.manifest['knowledge_policy']}
