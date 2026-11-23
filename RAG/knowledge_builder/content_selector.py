"""Semantic paragraph/section candidates, never fixed token windows."""
import re
from .common import stable_hash,norm
from .source_policy import BENCHMARK

VERSION='visual-section-selection-v0.1'
RELEVANT=re.compile(r'instrument|forceps|scissor|scalpel|grasper|knife|needle|sutur|knot|dissect|retract|grasp|mucosa|duct|artery|gallbladder|tissue|anatom|hemosta|haemosta|clip|cauter|critical view|surg|wound|组织|器械|解剖|缝合|切开|剥离',re.I)
LOW_VALUE=re.compile(r'^(?:references|bibliography|author contributions|conflicts? of interest|funding|correspondence|table of contents)\b|\b(?:buy now|request a quote|market leader|sales contact)\b',re.I)


def select(source,parsed,*,max_input_chars=18000):
    units=[];audit=[];current=[];current_size=0;section=None
    def flush():
        nonlocal current,current_size
        if current:
            payload={'source_id':source['source_id'],'source_sha256':source['sha256'],'title':source['title'],
                     'section':section,'blocks':current,'pages':sorted({b['page'] for b in current})}
            payload['unit_id']='unit_'+stable_hash({'version':VERSION,**payload})[:24];units.append(payload)
        current=[];current_size=0
    allowed=source.get('source_review',{}).get('allowed_pages',[])
    for page in parsed['pages']:
        if allowed and page['page'] not in allowed:
            audit.append({'page':page['page'],'reason':'OUTSIDE_SOURCE_REVIEW_PAGE_ALLOWLIST'});continue
        for b in page['clean_blocks']:
            text=b['clean_text'];reason=None
            if BENCHMARK.search(text):reason='BENCHMARK_SPECIFIC_BLOCK'
            elif LOW_VALUE.search(text):reason='NONVISUAL_REFERENCE_OR_PROMOTION'
            elif len(norm(text))<35:reason='SHORT_HEADING_OR_FRAGMENT'
            elif not RELEVANT.search(text):reason='NO_VISUAL_MEDICAL_TOPIC_SIGNAL'
            if reason:audit.append({'page':page['page'],'block_id':b['block_id'],'reason':reason});continue
            if len(text)>max_input_chars:
                audit.append({'page':page['page'],'block_id':b['block_id'],'reason':'OVERSIZED_SEMANTIC_BLOCK_PENDING'});continue
            is_heading=text in page['headings']
            if is_heading or (current and (current_size+len(text)>max_input_chars or page['page']>current[-1]['page']+1)):flush()
            if is_heading:section=text
            current.append({'page':page['page'],'block_id':b['block_id'],'text':text,'bbox':b['bbox']})
            current_size+=len(text)
        # Keep complete paragraphs. The budget may divide adjacent paragraphs,
        # never cut a sentence into an arbitrary fixed-token chunk.
    flush()
    return units,audit
