"""Closed extraction schema and persisted-library schema (Draft 2020-12)."""
TYPES=['instrument','anatomy','action','interaction','procedure','evidence_criteria']
MAX_CARDS_PER_UNIT=4
TEXT={'type':'string','minLength':1}
STRINGS={'type':'array','items':TEXT}


def obj(props):return {'type':'object','properties':props,'required':list(props),'additionalProperties':False}
def array(items):return {'type':'array','items':items}
def nullable(value):return {'anyOf':[value,{'type':'null'}]}


FACT=obj({'fact_id':TEXT,'text':TEXT,'source_id':TEXT,'page':{'type':'integer','minimum':1},
          'block_id':TEXT,'quote':TEXT,'extraction_method':{'enum':['text_llm','vision_llm']},
          'basis':{'enum':['body_text','figure_interpretation']},'figure_label':nullable(TEXT)})
DERIVED=obj({'guidance_id':TEXT,'text':TEXT,'supporting_fact_ids':array(TEXT),'derivation_rationale':TEXT})
GUIDANCE=obj({k:array(obj({'text':TEXT,'supporting_fact_ids':array(TEXT),'derived_guidance_ids':array(TEXT)}))
              for k in ['search_targets','spatial_cues','temporal_cues','supporting_observations','insufficient_observations','uncertainty_conditions']})
FEATURE=array(obj({'text':TEXT,'supporting_fact_ids':array(TEXT)}))
REF=obj({'source_id':TEXT,'page_start':{'type':'integer','minimum':1},'page_end':{'type':'integer','minimum':1},'section':nullable(TEXT)})
CARD=obj({'knowledge_type':{'enum':TYPES},'entity_name':TEXT,'aliases':STRINGS,'procedure_scope':STRINGS,
          'text':TEXT,'text_supporting_fact_ids':array(TEXT),
          **{k:FEATURE for k in ['visual_features','motion_features','spatial_relations','discriminative_features','confusable_with']},
          'evidence_guidance':GUIDANCE,'source_facts':array(FACT),'derived_guidance':array(DERIVED),
          'source_refs':array(REF),'scope':TEXT,'uncertainty_notes':STRINGS,
          'knowledge_basis':{'enum':['general_medical_knowledge','standard_procedure_prior']}})
CARD['properties']['source_facts']['minItems']=1
CARD['properties']['source_refs']['minItems']=1
CARD['properties']['text_supporting_fact_ids']['minItems']=1
RESPONSE=obj({'cards':array(CARD),'omitted_content_reasons':STRINGS})
RESPONSE['properties']['cards']['maxItems']=MAX_CARDS_PER_UNIT
CHUNK=obj({**CARD['properties'],'chunk_id':TEXT,'locator':array(REF),'review_status':{'const':'pending'},
           'reviewer':{'const':''},'reviewed_at':{'const':''},'source_grounding_status':{'const':'EXACT_QUOTES_CHECKED_SEMANTIC_REVIEW_PENDING'},
           'extraction_provenance':array(obj({'request_key':TEXT,'method':{'enum':['text_llm','vision_llm','mock']},'model':TEXT,'synthetic':{'type':'boolean'}}))})
SOURCE={'type':'object','properties':{'source_id':TEXT,'file_path':TEXT,'file_name':TEXT,
    'sha256':nullable({'type':'string','pattern':'^[a-f0-9]{64}$'}),'title':TEXT,'publisher':nullable(TEXT),
    'source_url':nullable(TEXT),'publication_year':nullable({'type':'integer'}),'pdf_pages':{'type':'integer','minimum':0},
    'document_type':TEXT,'license':TEXT,'benchmark_independent':nullable({'type':'boolean'}),
    'disposition':TEXT,'chunk_count':{'type':'integer','minimum':0},'review_status':{'const':'pending'}},
    'required':['source_id','file_path','file_name','sha256','title','publisher','source_url','publication_year','pdf_pages','document_type','license','benchmark_independent','disposition','chunk_count','review_status'],
    'additionalProperties':True}
LIBRARY=obj({'schema_version':{'const':1},'library_id':{'enum':['relive.surgical_visual_knowledge','relive.surgical_visual_knowledge.mock']},
             'library_version':{'const':'0.1.0'},'created_at':TEXT,'knowledge_language':{'const':'en'},
             'source_policy':{'const':'benchmark_independent'},'build_status':TEXT,'synthetic_test':{'type':'boolean'},
             'source_index':array(SOURCE),'chunks':array(CHUNK)})
