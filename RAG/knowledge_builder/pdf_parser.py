"""Page/block provenance, conservative layout handling and auditable cleaning."""
from collections import Counter
from pathlib import Path
import re
import pymupdf
from .common import stable_hash,read_json,write_json,norm,file_hash

VERSION='pymupdf-block-parser-v0.1'
NOISE=re.compile(r'^(?:accept (?:all )?cookies|cookie (?:policy|settings)|privacy policy|all rights reserved|©\s*\d|home instruments & sutures|https?://\S+\s*$|\d{1,2}/\d{1,2}/\d{2,4}.*(?:AM|PM))',re.I)


def reading_order(blocks,width):
    text=[b for b in blocks if b['text'].strip()]
    left=[b for b in text if b['bbox'][2]<width*.59]
    right=[b for b in text if b['bbox'][0]>width*.43]
    two=len(left)>=3 and len(right)>=3 and len(left)+len(right)>=len(text)*.65
    if not two:return sorted(text,key=lambda b:(b['bbox'][1],b['bbox'][0])),'TOP_DOWN'
    spanning=sorted([b for b in text if b not in left and b not in right],key=lambda b:b['bbox'][1])
    out=[];remaining=[b for b in text if b not in spanning]
    for separator in spanning:
        band=[b for b in remaining if b['bbox'][1]<separator['bbox'][1]]
        out.extend(sorted(band,key=lambda b:(b['bbox'][0]>width*.43,b['bbox'][1],b['bbox'][0])))
        remaining=[b for b in remaining if b not in band];out.append(separator)
    out.extend(sorted(remaining,key=lambda b:(b['bbox'][0]>width*.43,b['bbox'][1],b['bbox'][0])))
    return out,'TWO_COLUMN_HEURISTIC_REVIEW_REQUIRED'


def parse_pdf(path,sha,cache_dir,*,max_pages=None,tables=True):
    key=stable_hash({'sha':sha,'version':VERSION,'engine':pymupdf.VersionBind,'max_pages':max_pages,'tables':tables})
    cached=Path(cache_dir)/'parsed'/f'{key}.json'
    if cached.is_file():
        v=read_json(cached)
        if v['sha256']!=sha or v['payload_sha256']!=stable_hash({k:x for k,x in v.items() if k!='payload_sha256'}):raise ValueError('PARSE_CACHE_CORRUPTED')
        return v,True
    pages=[]
    with pymupdf.open(path) as doc:
        if doc.needs_pass:raise ValueError('PDF_PASSWORD_REQUIRED')
        count=len(doc);metadata=doc.metadata;toc=doc.get_toc()
        for i in range(min(count,max_pages) if max_pages else count):
            page=doc[i]
            try:
                dd=page.get_text('dict',flags=pymupdf.TEXTFLAGS_TEXT)
                blocks=[]
                for j,b in enumerate(dd['blocks']):
                    if b.get('type')!=0:continue
                    lines=[''.join(span['text'] for span in line['spans']) for line in b['lines']]
                    sizes=[span['size'] for line in b['lines'] for span in line['spans']]
                    blocks.append({'block_id':f'p{i+1}_b{j}','bbox':list(b['bbox']),'text':'\n'.join(lines),
                                   'max_font_size':max(sizes,default=0)})
                ordered,layout=reading_order(blocks,page.rect.width)
                table_data=[];table_error=None;table_status='NOT_REQUESTED' if not tables else 'NO_VECTOR_TABLE_CANDIDATE'
                if tables:
                    drawings=page.get_drawings()
                    if sum(len(d.get('items',[])) for d in drawings)>=12:
                        try:
                            for table in page.find_tables(strategy='lines').tables:
                                table_data.append({'bbox':list(table.bbox),'cells':table.extract()})
                            table_status='CHECKED_VECTOR_LINES'
                        except Exception as exc:table_error=type(exc).__name__;table_status='TABLE_PARSE_FAILED'
                raw='\n\n'.join(b['text'] for b in ordered)
                image_count=len(page.get_images());raster=bool(image_count and len(norm(raw))<150)
                pages.append({'page':i+1,'width':page.rect.width,'height':page.rect.height,
                              'raw_text':raw,'blocks':ordered,'layout':layout,'images':image_count,
                              'tables':table_data,'table_status':table_status,'table_error':table_error,
                              'links':[l['uri'] for l in page.get_links() if 'uri' in l],
                              'parse_status':'IMAGE_OR_SPARSE_TEXT_PENDING' if raster or len(norm(raw))<40 else 'TEXT_PARSED',
                              'page_error':None})
            except Exception as exc:
                pages.append({'page':i+1,'width':page.rect.width,'height':page.rect.height,'raw_text':'','blocks':[],
                              'layout':'UNKNOWN','images':None,'tables':[],'table_status':'NOT_PARSED','table_error':None,
                              'links':[],'parse_status':'PAGE_PARSE_FAILED','page_error':type(exc).__name__})
    margins=Counter()
    for p in pages:
        for b in p['blocks']:
            if b['bbox'][1]<p['height']*.12 or b['bbox'][3]>p['height']*.88:
                margins[norm(b['text'])]+=1
    for p in pages:
        clean=[];removed=[];seen=set()
        for b in p['blocks']:
            lines=[]
            for line in b['text'].splitlines():
                n=norm(line)
                reason='WEB_NAVIGATION_OR_FOOTER' if NOISE.search(n) else None
                if n and n in seen and len(n)<160:reason='REPEATED_SHORT_LINE'
                if reason:removed.append({'block_id':b['block_id'],'text':line,'reason':reason})
                else:lines.append(line)
                seen.add(n)
            if margins[norm(b['text'])]>=max(3,len(pages)*.4) and (b['bbox'][1]<p['height']*.12 or b['bbox'][3]>p['height']*.88):
                removed.append({'block_id':b['block_id'],'text':b['text'],'reason':'REPEATED_MARGIN'});lines=[]
            if lines:clean.append({**b,'clean_text':'\n'.join(lines)})
        p['clean_blocks']=clean;p['clean_text']='\n\n'.join(b['clean_text'] for b in clean);p['removed_noise']=removed
        p['headings']=[b['clean_text'] for b in clean if len(b['clean_text'].split())<16 and b['max_font_size']>=12]
        p['content_sha256']=stable_hash(norm(p['clean_text']))
        p['vision_candidate']=bool(p['images'] or p['tables'] or p['parse_status']!='TEXT_PARSED')
        p['vision_status']='NOT_REQUESTED_CANDIDATE' if p['vision_candidate'] else 'NOT_SELECTED'
    if file_hash(path)!=sha:raise ValueError('PDF_CHANGED_DURING_PARSING')
    v={'version':VERSION,'engine_version':pymupdf.VersionBind,'sha256':sha,'pdf_pages':count,
       'parsed_page_count':len(pages),'unprocessed_page_numbers':list(range(len(pages)+1,count+1)),
       'metadata':metadata,'toc':toc,'pages':pages,'content_hash':stable_hash([norm(p['clean_text']) for p in pages])}
    v['payload_sha256']=stable_hash(v);write_json(cached,v);return v,False


def source_metadata(source,parsed):
    pages=parsed['pages'];start='\n'.join(p['raw_text'] for p in pages[:3]);meta=parsed['metadata']
    known=[('Open Surgery',r'Open Surgery|Tobias Todsen'),('American College of Surgeons',r'American College of Surgeons'),
           ('APGO',r'Association of Professors of Gynecology'),('WHO',r'World Health Organization|WHO best practices'),
           ('SAGES',r'SAGES'),('Olympus',r'Olympus'),('ESGE',r'European Society of Gastrointestinal Endoscopy')]
    probable=next((name for name,pattern in known if re.search(pattern,start+' '+str(meta.get('title','')),re.I)),None)
    category='instrument_education' if re.search(r'instrument|knife',source['file_name'],re.I) else 'surgical_education'
    if re.search(r'Nursing',start,re.I):category='nursing_reference'
    if re.search(r'REVIEW|doi:',start,re.I):category='clinical_review_or_technical_article'
    title=meta.get('title') or next((b['clean_text'] for p in pages[:2] for b in p['clean_blocks'] if b['clean_text'].strip()),source['file_name'])
    return {**source,'title':title[:500],'title_basis':'PDF_METADATA' if meta.get('title') else 'EXTRACTED_TEXT_CANDIDATE',
            'publisher':None,'probable_source':probable,'source_url':None,'publication_year':None,'license':'unknown',
            'pdf_pages':parsed['pdf_pages'],'page_count':parsed['pdf_pages'],'parsed_pages':len(pages),
            'text_extractable':any(p['raw_text'].strip() for p in pages),'content_hash':parsed['content_hash'],
            'document_category':category,'document_type':category,'benchmark_independent':None,'review_status':'pending',
            'disposition':'indexed_only','chunk_count':0,'processing_status':'PARSED_PARTIAL' if parsed['unprocessed_page_numbers'] or any(p['page_error'] for p in pages) else 'PARSED',
            'pending_visual_pages':[p['page'] for p in pages if p['vision_candidate']],
            'page_failures':[{'page':p['page'],'reason':p['page_error']} for p in pages if p['page_error']]}


def render_page(path,page_number):
    with pymupdf.open(path) as doc:
        return doc[page_number-1].get_pixmap(matrix=pymupdf.Matrix(1.3,1.3),alpha=False).tobytes('png')
