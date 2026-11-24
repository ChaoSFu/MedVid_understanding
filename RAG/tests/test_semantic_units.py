"""Offline fixtures for bounded source units and conservative admission."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from jsonschema import Draft202012Validator

from RAG.knowledge_builder.common import read_json,write_json
from RAG.knowledge_builder.content_selector import select
from RAG.knowledge_builder.extractor import extract,MockProvider,CHECK_SCHEMA
from RAG.knowledge_builder.schemas import RESPONSE,MAX_CARDS_PER_UNIT
from RAG.knowledge_builder.source_policy import source_findings,review_template,apply_review,summarize_source_scope
from RAG.knowledge_builder.build import build
from RAG.tests.test_knowledge_builder import Fixtures,pdf,SENTENCE


class SemanticUnitTests(unittest.TestCase):
    def source(self):
        return {'source_id':'src_fixture','sha256':'a'*64,'title':'Synthetic independent source'}

    def page(self,number,texts,headings=()):
        return {'page':number,'headings':list(headings),'clean_blocks':[
            {'block_id':f'p{number}_b{i}','clean_text':text,'bbox':[0,i,100,i+1]}
            for i,text in enumerate(texts)]}

    def test_page_block_and_character_limits_preserve_whole_paragraphs(self):
        pages=[self.page(n,[f'{SENTENCE} Synthetic paragraph {i}.' for i in range(9)]) for n in (1,2)]
        units,audit=select(self.source(),{'pages':pages},max_input_chars=250,max_blocks=2)
        expected={b['block_id']:b['clean_text'] for p in pages for b in p['clean_blocks']}
        actual=[b for u in units for b in u['blocks']]
        self.assertEqual(Counter(b['block_id'] for b in actual),Counter(expected.keys()))
        self.assertTrue(all(b['text']==expected[b['block_id']] for b in actual))
        self.assertTrue(all(len(u['pages'])==1 and len(u['blocks'])<=2 and sum(len(b['text']) for b in u['blocks'])<=250 for u in units))
        self.assertEqual(sum(r['reason']=='EXTRACTION_UNIT_FORMED' for r in audit),len(units))

    def test_short_heading_is_context_not_lost_by_topic_filter(self):
        page=self.page(1,['Forceps',SENTENCE],headings=['Forceps'])
        units,audit=select(self.source(),{'pages':[page]})
        self.assertEqual(units[0]['section'],'Forceps')
        self.assertEqual(units[0]['blocks'][0]['text'],SENTENCE)
        self.assertTrue(any(r['reason']=='SOURCE_HEADING_CONTEXT' for r in audit))

    def test_benchmark_heading_never_enters_unit_locator_or_prompt_context(self):
        heading='CVAT benchmark annotation guidelines'
        page=self.page(1,[heading,SENTENCE],headings=[heading])
        units,audit=select(self.source(),{'pages':[page]})
        self.assertEqual(len(units),1);self.assertIsNone(units[0]['section'])
        self.assertNotIn('CVAT',json.dumps(units))
        self.assertTrue(any(r['reason']=='BENCHMARK_SPECIFIC_BLOCK' for r in audit))

    def test_bibliography_continues_across_pages_before_size_or_topic_filter(self):
        refs='Surgical instrument paper. DOI: fixture. '*150
        pages=[self.page(1,[SENTENCE,'REFERENCES'],headings=['REFERENCES']),self.page(2,[refs])]
        units,audit=select(self.source(),{'pages':pages},max_input_chars=200)
        self.assertEqual(len(units),1);self.assertEqual(units[0]['blocks'][0]['text'],SENTENCE)
        self.assertTrue(any(r['reason']=='BIBLIOGRAPHY_CONTINUATION' and r['page']==2 for r in audit))
        self.assertFalse(any(r['reason']=='OVERSIZED_SEMANTIC_BLOCK_PENDING' for r in audit))

    def test_next_explicit_section_resumes_medical_content_after_bibliography(self):
        pages=[self.page(1,['Bibliography'],headings=['Bibliography']),
               self.page(2,['Surgical instrument reference. DOI: fixture.']),
               self.page(3,['Surgical Instruments',SENTENCE],headings=['Surgical Instruments'])]
        units,_=select(self.source(),{'pages':pages})
        self.assertEqual(len(units),1);self.assertEqual(units[0]['pages'],[3])
        self.assertEqual(units[0]['section'],'Surgical Instruments')

    def test_reference_word_in_medical_paragraph_does_not_start_bibliography(self):
        text='The surgical instrument has references engraved on its shaft for visual alignment.'
        units,audit=select(self.source(),{'pages':[self.page(1,[text])]})
        self.assertEqual(units[0]['blocks'][0]['text'],text)
        self.assertFalse(any(r['reason']=='BIBLIOGRAPHY_SECTION_START' for r in audit))

    def test_bibliography_publisher_metadata_does_not_suppress_next_page(self):
        metadata='Bibliography\nSurgical Journal\nDOI 10.fixture/article\nISSN 1234-5678'
        pages=[self.page(1,[metadata]),self.page(2,[SENTENCE])]
        units,_=select(self.source(),{'pages':pages})
        self.assertEqual(len(units),1);self.assertEqual(units[0]['pages'],[2])

    def test_undetected_small_font_medical_section_resumes_after_references(self):
        pages=[self.page(1,['REFERENCES','Surgical instrument journal. DOI: fixture.'],headings=['REFERENCES']),
               self.page(2,[SENTENCE])]
        units,_=select(self.source(),{'pages':pages})
        self.assertEqual(len(units),1);self.assertEqual(units[0]['blocks'][0]['text'],SENTENCE)

    def test_split_reference_titles_do_not_reopen_a_citation_dense_page(self):
        pages=[self.page(1,['REFERENCES'],headings=['REFERENCES']),self.page(2,[
            'Surgical instrument title that looks relevant but is only a cited article.',
            'Example Journal 2020; 10: 20-30 DOI fixture PMID 12345'])]
        units,audit=select(self.source(),{'pages':pages})
        self.assertEqual(units,[])
        self.assertTrue(any(r.get('page_citation_signal_count',0)>=3 for r in audit))

    def test_explicit_medical_heading_resumes_even_with_later_citations(self):
        pages=[self.page(1,['REFERENCES'],headings=['REFERENCES']),self.page(2,[
            'Surgical Instruments',SENTENCE,'Example Journal 2020; 10: 20-30 DOI fixture PMID 12345'],headings=['Surgical Instruments'])]
        units,_=select(self.source(),{'pages':pages})
        self.assertTrue(any(b['text']==SENTENCE for u in units for b in u['blocks']))

    def test_oversized_paragraph_is_pending_never_sliced(self):
        text=SENTENCE*10
        units,audit=select(self.source(),{'pages':[self.page(1,[text])]},max_input_chars=150)
        self.assertEqual(units,[])
        self.assertTrue(any(r['reason']=='OVERSIZED_SEMANTIC_BLOCK_PENDING' and r['input_characters']==len(text) for r in audit))

    def test_page_allowlist_and_unit_identity_deterministic(self):
        source=self.source();source['source_review']={'allowed_pages':[2]}
        parsed={'pages':[self.page(n,[SENTENCE]*3) for n in (1,2)]}
        a=select(source,parsed,max_blocks=2);b=select(source,parsed,max_blocks=2)
        self.assertEqual(a,b);self.assertEqual({p for u in a[0] for p in u['pages']},{2})
        self.assertNotEqual(a[0],select(source,parsed,max_blocks=1)[0])
        for invalid in (0,-1,False):
            with self.assertRaisesRegex(ValueError,'SEMANTIC_UNIT_LIMIT_INVALID'):
                select(source,parsed,max_blocks=invalid)


class ConservativeScreenTests(Fixtures):
    def test_nonvisual_or_uncertain_relevance_withheld_even_if_consistent(self):
        s,u,_=self.records()
        class Screen(MockProvider):
            relevance='NONVISUAL'
            def complete(self,**kwargs):
                result=super().complete(**kwargs)
                if kwargs['identity'].get('stage')=='source_consistency':
                    for assessment in result['value']['assessments']:
                        assessment['visual_relevance']=self.relevance
                return result
        for relevance in ('NONVISUAL','UNCERTAIN'):
            p=Screen();p.relevance=relevance
            accepted,rejected,_=extract(u,p)
            self.assertEqual(accepted,[]);self.assertEqual(len(rejected),1)
            self.assertIn('LLM_VISUAL_RELEVANCE_'+relevance,rejected[0]['rejection_reasons'])

    def test_unsupported_guidance_does_not_salvage_core_or_relax_check(self):
        s,u,_=self.records()
        class Screen(MockProvider):
            def complete(self,**kwargs):
                result=super().complete(**kwargs)
                if kwargs['identity'].get('stage')=='source_consistency':
                    for a in result['value']['assessments']:
                        a.update(decision='UNSUPPORTED',unsupported_fields=['evidence_guidance.search_targets'])
                return result
        accepted,rejected,_=extract(u,Screen())
        self.assertFalse(accepted)
        self.assertEqual(rejected[0]['rejection_reasons'],['LLM_SOURCE_CONSISTENCY_UNSUPPORTED'])

    def test_screen_requires_relevance_and_extraction_card_ceiling(self):
        s,u,records=self.records();card=records[0]['card']
        self.assertTrue(list(Draft202012Validator(RESPONSE).iter_errors({'cards':[card]*(MAX_CARDS_PER_UNIT+1),'omitted_content_reasons':[]})))
        assessment={'candidate_index':0,'decision':'CONSISTENT','unsupported_fields':[],'rationale':'Synthetic fixture.'}
        self.assertTrue(list(Draft202012Validator(CHECK_SCHEMA).iter_errors({'assessments':[assessment]})))

    def test_checked_in_extraction_schema_matches_runtime(self):
        self.assertEqual(read_json(Path(__file__).parents[1]/'schemas/extraction.schema.json'),
                         {'$schema':'https://json-schema.org/draft/2020-12/schema',**RESPONSE})


class SourceScopeTests(Fixtures):
    def test_selected_exclusion_is_not_a_pending_review(self):
        source,_=self.source();review=review_template(source);review['decision']='EXCLUDE'
        source=apply_review(source,review);source['selected_for_processing']=True
        audit=summarize_source_scope([source],{source['source_id']:review})
        self.assertEqual(audit['status'],'REVIEWED_SOURCES_ONLY')
        self.assertEqual(audit['selected_review_pending'],[])
        self.assertEqual(audit['selected_reviewed_exclude'],[source['source_id']])

    def test_deferred_benchmark_source_not_admitted_and_does_not_block_selected(self):
        source,_=self.source();r=review_template(source)
        r.update(decision='INCLUDE',reviewer_id='fixture',rationale='Fixture only.',independence_basis='Synthetic source.')
        selected=apply_review(source,r);selected['selected_for_processing']=True
        other=deepcopy(source);other.update(source_id='src_deferred',file_path='mixed.pdf',sha256='b'*64,selected_for_processing=False)
        other['source_audit']=source_findings('mixed.pdf','CVAT')
        excluded=review_template(other);excluded['decision']='EXCLUDE'
        audit=summarize_source_scope([selected,other],{selected['source_id']:r,other['source_id']:excluded})
        self.assertEqual(audit['status'],'REVIEWED_SOURCES_ONLY')
        self.assertFalse(audit['deferred_sources'][0]['extraction_allowed'])
        self.assertEqual(audit['deferred_sources'][0]['declared_review_decision'],'EXCLUDE')
        self.assertFalse(audit['flagged_sources'][0]['selected_for_processing'])
        other['selected_for_processing']=True
        self.assertEqual(summarize_source_scope([selected,other],{})['status'],'SOURCE_REVIEW_REQUIRED')

    def test_unit_manifest_and_omission_audit_are_separate_and_immutable(self):
        output=self.root/'mock/knowledge.mock.json'
        args=dict(input_dir=self.refs,output=output,mode='mock',cache_dir=self.cache,tables=False,max_unit_blocks=1)
        with patch('RAG.knowledge_builder.provider.Provider._send',side_effect=AssertionError('NETWORK')):
            result=build(**args)
        manifest=[json.loads(l) for l in (output.parent/'extraction_unit_manifest.jsonl').read_text().splitlines()]
        audit=[json.loads(l) for l in (output.parent/'unit_extraction_audit.jsonl').read_text().splitlines()]
        self.assertEqual(len(manifest),result['unit_counts']['candidate_units'])
        self.assertTrue(all(r['block_count']<=1 for r in manifest))
        self.assertTrue(all(r['status']=='COMPLETED' and 'omitted_content_reasons' in r for r in audit))
        before=output.read_bytes();build(**args,resume=True);self.assertEqual(output.read_bytes(),before)
        with self.assertRaisesRegex(ValueError,'RESUME_IDENTITY_MISMATCH'):
            build(**{**args,'max_unit_blocks':2},resume=True)

    def test_all_excluded_is_not_misreported_as_missing_review(self):
        source,_=self.source();review=review_template(source);review['decision']='EXCLUDE'
        path=self.root/'review.jsonl';write_json(path,[review],jsonl=True)
        class NoCalls:
            def snapshot(self):return {'new_api_calls':0}
            def complete(self,**kwargs):raise AssertionError('EXCLUDED_SOURCE_SENT_TO_MODEL')
        result=build(input_dir=self.refs,output=self.root/'all_excluded/k.json',mode='build',cache_dir=self.cache,
                     source_review=path,provider_factory=NoCalls,tables=False)
        self.assertEqual(result['status'],'NO_ADMISSIBLE_SOURCE_UNITS')
        self.assertEqual(read_json(self.root/'all_excluded/quality_report.json')['benchmark_independence']['selected_review_pending'],[])


if __name__=='__main__':unittest.main()
