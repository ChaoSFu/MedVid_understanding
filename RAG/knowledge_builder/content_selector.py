"""Semantic paragraph/section candidates, never fixed token windows."""
import re
from .common import stable_hash,norm
from .source_policy import BENCHMARK

VERSION='visual-section-selection-v0.2'
RELEVANT=re.compile(r'instrument|forceps|scissor|scalpel|grasper|knife|needle|sutur|knot|dissect|retract|grasp|mucosa|duct|artery|gallbladder|tissue|anatom|hemosta|haemosta|clip|cauter|critical view|surg|wound|组织|器械|解剖|缝合|切开|剥离',re.I)
LOW_VALUE=re.compile(r'^(?:references|bibliography|author contributions|conflicts? of interest|funding|correspondence|table of contents)\b|\b(?:buy now|request a quote|market leader|sales contact)\b',re.I)


def select(source,parsed,*,max_input_chars=4000,max_blocks=6):
    if any(type(v) is not int or v<1 for v in (max_input_chars,max_blocks)):
        raise ValueError('SEMANTIC_UNIT_LIMIT_INVALID')
    units=[];audit=[];current=[];current_size=0;section=None
    def flush():
        nonlocal current,current_size
        if current:
            payload={'source_id':source['source_id'],'source_sha256':source['sha256'],'title':source['title'],
                     'section':section,'blocks':current,'pages':sorted({b['page'] for b in current})}
            payload['unit_id']='unit_'+stable_hash({'version':VERSION,**payload})[:24];units.append(payload)
            audit.append({'reason':'EXTRACTION_UNIT_FORMED','unit_id':payload['unit_id'],
                          'pages':payload['pages'],'section':section,'input_characters':current_size,
                          'block_ids':[b['block_id'] for b in current],'block_count':len(current)})
        current=[];current_size=0
    allowed=source.get('source_review',{}).get('allowed_pages',[])
    for page in parsed['pages']:
        # Page and paragraph boundaries are real source boundaries. Never cut
        # a paragraph to fit a token estimate or append a later page silently.
        if current and page['page']!=current[-1]['page']:flush()
        if allowed and page['page'] not in allowed:
            audit.append({'page':page['page'],'reason':'OUTSIDE_SOURCE_REVIEW_PAGE_ALLOWLIST'});continue
        for b in page['clean_blocks']:
            text=b['clean_text'];reason=None
            is_heading=text in page['headings']
            if is_heading:
                flush();section=None
                if not BENCHMARK.search(text) and not LOW_VALUE.search(text):
                    section=text
                    audit.append({'page':page['page'],'block_id':b['block_id'],
                                  'reason':'SOURCE_HEADING_CONTEXT','section':section})
            if BENCHMARK.search(text):reason='BENCHMARK_SPECIFIC_BLOCK'
            elif LOW_VALUE.search(text):reason='NONVISUAL_REFERENCE_OR_PROMOTION'
            elif len(norm(text))<35:reason='SHORT_HEADING_OR_FRAGMENT'
            elif not RELEVANT.search(text):reason='NO_VISUAL_MEDICAL_TOPIC_SIGNAL'
            if reason:audit.append({'page':page['page'],'block_id':b['block_id'],'reason':reason});continue
            if len(text)>max_input_chars:
                audit.append({'page':page['page'],'block_id':b['block_id'],
                              'reason':'OVERSIZED_SEMANTIC_BLOCK_PENDING','input_characters':len(text)});continue
            if current and (current_size+len(text)>max_input_chars or len(current)>=max_blocks):flush()
            current.append({'page':page['page'],'block_id':b['block_id'],'text':text,'bbox':b['bbox']})
            current_size+=len(text)
        # Keep complete paragraphs. The budget may divide adjacent paragraphs,
        # never cut a sentence into an arbitrary fixed-token chunk.
    flush()
    return units,audit
