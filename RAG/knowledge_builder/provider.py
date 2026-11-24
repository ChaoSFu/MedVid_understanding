"""Bounded OpenAI-compatible transport; capability fallback is explicit/audited.

The ReliVE transport returns only completion text and requires experimental
policy settings. This standalone corpus adapter also needs usage, structured
output negotiation and independent cost budgets, so it has no runtime imports.
"""
from dataclasses import dataclass
import base64
import json
import os
from pathlib import Path
import threading
import time
from urllib import request,error
from urllib.parse import urlsplit
from jsonschema import Draft202012Validator
from .common import stable_hash,read_json,write_json,strict_loads


class ProviderError(ValueError):pass
class BudgetStop(ProviderError):pass
class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None


@dataclass
class Config:
    base_url:str=''
    text_model:str=''
    vision_model:str=''
    credential_env:str='LLM_API_KEY'
    timeout:float=120
    max_retries:int=3
    max_concurrency:int=3
    requests_per_minute:float=20
    response_format:str='auto'
    max_output_tokens:int=6000
    enable_thinking:bool|None=None
    input_usd_per_million:float|None=None
    output_usd_per_million:float|None=None

    @classmethod
    def environment(cls,**overrides):
        e=os.environ
        values={'base_url':e.get('LLM_BASE_URL',''),'text_model':e.get('LLM_TEXT_MODEL',''),
                'vision_model':e.get('LLM_VISION_MODEL',''),'timeout':float(e.get('LLM_TIMEOUT','120')),
                'max_retries':int(e.get('LLM_MAX_RETRIES','3')),'max_concurrency':int(e.get('LLM_MAX_CONCURRENCY','3')),
                'requests_per_minute':float(e.get('LLM_REQUESTS_PER_MINUTE','20')),'response_format':e.get('LLM_RESPONSE_FORMAT','auto')}
        thinking=e.get('LLM_ENABLE_THINKING','').strip().lower()
        if thinking not in ('','true','false'):raise ProviderError('LLM_ENABLE_THINKING_INVALID')
        values['enable_thinking']=None if not thinking else thinking=='true'
        for k,env in [('input_usd_per_million','LLM_INPUT_USD_PER_MILLION'),('output_usd_per_million','LLM_OUTPUT_USD_PER_MILLION')]:
            values[k]=float(e[env]) if e.get(env) else None
        values.update({k:v for k,v in overrides.items() if v is not None});return cls(**values)

    def public(self):return dict(self.__dict__)


class Provider:
    synthetic=False
    def __init__(self,config,cache_dir,*,max_calls=100,max_cost_usd=None):
        self.config=config;self.cache=Path(cache_dir);self.max_calls=max_calls;self.max_cost=max_cost_usd
        url=urlsplit(config.base_url)
        if url.scheme not in ('https','http') or not url.netloc or url.username or url.password or url.query or url.fragment:raise ProviderError('LLM_BASE_URL_INVALID')
        if url.scheme=='http' and url.hostname not in ('localhost','127.0.0.1','::1'):raise ProviderError('REMOTE_API_REQUIRES_HTTPS')
        if not os.environ.get(config.credential_env):raise ProviderError('LLM_API_KEY_MISSING')
        if not config.text_model:raise ProviderError('LLM_TEXT_MODEL_MISSING')
        if config.max_concurrency<1 or config.max_retries<0 or config.requests_per_minute<=0 or config.timeout<=0:raise ProviderError('LLM_LIMITS_INVALID')
        if config.response_format not in ('auto','json_schema','json_object','prompt'):raise ProviderError('LLM_RESPONSE_FORMAT_INVALID')
        if config.enable_thinking is not None and type(config.enable_thinking) is not bool:raise ProviderError('LLM_ENABLE_THINKING_INVALID')
        if max_cost_usd is not None and (config.input_usd_per_million is None or config.output_usd_per_million is None):raise ProviderError('COST_BUDGET_REQUIRES_EXPLICIT_PRICE_RATES')
        if any(v is not None and v<0 for v in (max_cost_usd,config.input_usd_per_million,config.output_usd_per_million)):raise ProviderError('NEGATIVE_COST_INVALID')
        self.lock=threading.Lock();self.semaphore=threading.Semaphore(config.max_concurrency);self.next_start=0
        self.calls=0;self.hits=0;self.input_tokens=0;self.output_tokens=0;self.unknown_usage_calls=0;self.reserved_cost=0.;self.cost=0.;self.latency=0.
        self.capabilities={};self.events=[]

    def snapshot(self):
        return {'new_api_calls':self.calls,'cache_hits':self.hits,'input_tokens':self.input_tokens,'output_tokens':self.output_tokens,
                'usage_missing_calls':self.unknown_usage_calls,'latency_seconds':self.latency,
                'estimated_cost_usd':self.cost if self.unknown_usage_calls==0 and self.config.input_usd_per_million is not None and self.config.output_usd_per_million is not None else None,
                'known_usage_estimated_cost_usd':self.cost if self.config.input_usd_per_million is not None and self.config.output_usd_per_million is not None else None,
                'pricing_source':'USER_CONFIGURED_RATES_NOT_PROVIDER_BILLING','reserved_cost_upper_bound_usd':self.reserved_cost,
                'max_api_calls':self.max_calls,'max_cost_usd':self.max_cost,'capabilities':self.capabilities}

    def _admit(self,prompt,image):
        if image and self.max_cost is not None:raise BudgetStop('VISION_COST_UPPER_BOUND_UNAVAILABLE')
        cfg=self.config
        upper=((len(prompt.encode())+10000)*cfg.input_usd_per_million+cfg.max_output_tokens*cfg.output_usd_per_million)/1e6 if self.max_cost is not None else 0
        with self.lock:
            if self.calls>=self.max_calls:raise BudgetStop('API_CALL_BUDGET_REACHED')
            if self.max_cost is not None and self.reserved_cost+upper>self.max_cost:raise BudgetStop('API_COST_BUDGET_REACHED')
            self.calls+=1;self.reserved_cost+=upper
            delay=max(0,self.next_start-time.monotonic());self.next_start=max(time.monotonic(),self.next_start)+60/cfg.requests_per_minute
        if delay:time.sleep(delay)

    def _send(self,payload):
        # No redirects, response headers, credential values or server error bodies
        # are persisted or emitted. Capability evidence is a tiny allowlisted code.
        req=request.Request(self.config.base_url.rstrip('/')+'/chat/completions',
                            data=json.dumps(payload,allow_nan=False).encode(),method='POST',
                            headers={'Authorization':'Bearer '+os.environ[self.config.credential_env],'Content-Type':'application/json'})
        try:
            with request.build_opener(NoRedirect()).open(req,timeout=self.config.timeout) as response:
                return strict_loads(response.read().decode()),None
        except error.HTTPError as exc:
            body=exc.read(65536).decode(errors='replace').casefold()
            unsupported=exc.code in (400,422) and any(t in body for t in ('response_format','json_schema','json_object')) and any(t in body for t in ('unsupported','not support','invalid parameter','unknown parameter','not available'))
            vision_bad=exc.code in (400,422) and any(t in body for t in ('image_url','vision','image input')) and any(t in body for t in ('not support','unsupported'))
            return None,'VISION_UNSUPPORTED' if vision_bad else 'STRUCTURED_OUTPUT_UNSUPPORTED' if unsupported else f'HTTP_{exc.code}'
        except (error.URLError,TimeoutError,OSError):return None,'TRANSPORT_FAILURE'
        except (ValueError,UnicodeError):return None,'RESPONSE_ENVELOPE_INVALID'

    def complete(self,*,prompt,schema,identity,image=None):
        model=self.config.vision_model if image else self.config.text_model
        if not model:raise ProviderError('VISION_MODEL_UNCONFIGURED')
        key=stable_hash({'provider_config':self.config.public(),'model':model,'identity':identity,'prompt':prompt,
                         'schema':schema,'image_sha256':stable_hash(base64.b64encode(image).decode()) if image else None,'adapter':'corpus-api-v0.1'})
        cached=self.cache/'llm'/key/'success.json'
        if cached.exists():
            result=read_json(cached)
            if result['request_key']!=key or result['payload_hash']!=stable_hash({k:v for k,v in result.items() if k!='payload_hash'}):raise ProviderError('LLM_CACHE_CORRUPT')
            Draft202012Validator(schema).validate(result['value'])
            with self.lock:self.hits+=1
            return result
        levels=['json_schema','json_object','prompt'] if self.config.response_format=='auto' else [self.config.response_format]
        response_mode=levels[0];last='NOT_ATTEMPTED'
        for attempt in range(self.config.max_retries+1):
            content=[{'type':'text','text':prompt+'\nReturn JSON matching this schema:\n'+json.dumps(schema)}]
            if image:content.append({'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(image).decode()}})
            payload={'model':model,'messages':[{'role':'user','content':content}],'temperature':0,'max_tokens':self.config.max_output_tokens}
            if self.config.enable_thinking is not None:payload['enable_thinking']=self.config.enable_thinking
            if response_mode=='json_schema':payload['response_format']={'type':'json_schema','json_schema':{'name':'surgical_knowledge','strict':True,'schema':schema}}
            elif response_mode=='json_object':payload['response_format']={'type':'json_object'}
            with self.semaphore:
                self._admit(prompt+json.dumps(schema),image);start=time.monotonic();envelope,err=self._send(payload);elapsed=time.monotonic()-start
            raw=None;value=None;usage=envelope.get('usage',{}) if isinstance(envelope,dict) else {}
            usage={k:v for k,v in usage.items() if k in ('prompt_tokens','completion_tokens','total_tokens') and type(v) is int and v>=0} if isinstance(usage,dict) else {}
            try:
                if err:raise ValueError(err)
                raw=envelope['choices'][0]['message']['content']
                if not isinstance(raw,str):raise ValueError('NON_TEXT_COMPLETION')
                secret=os.environ.get(self.config.credential_env)
                if secret and secret in raw:
                    raw=raw.replace(secret,'[REDACTED_CREDENTIAL]');err='CREDENTIAL_ECHO_REDACTED'
                    raise ValueError(err)
                value=strict_loads(raw);Draft202012Validator(schema).validate(value)
            except Exception:
                last=err or 'MODEL_JSON_OR_SCHEMA_INVALID'
            else:last=None
            with self.lock:
                self.latency+=elapsed
                if type(usage.get('prompt_tokens')) is int and type(usage.get('completion_tokens')) is int:
                    self.input_tokens+=usage['prompt_tokens'];self.output_tokens+=usage['completion_tokens']
                    if self.config.input_usd_per_million is not None and self.config.output_usd_per_million is not None:
                        self.cost+=(usage['prompt_tokens']*self.config.input_usd_per_million+usage['completion_tokens']*self.config.output_usd_per_million)/1e6
                else:self.unknown_usage_calls+=1
            record={'request_key':key,'attempt':attempt,'response_mode':response_mode,'raw_response':raw,
                    'failure_reason':last,'usage':usage,'model':model,'latency_seconds':elapsed}
            write_json(self.cache/'llm'/key/(stable_hash(record)+'.json'),record)
            if last is None:
                result={'request_key':key,'value':value,'raw_response':raw,'model':model,'usage':usage,'response_mode':response_mode,'synthetic':False}
                result['payload_hash']=stable_hash(result);write_json(cached,result)
                with self.lock:self.capabilities[model+':'+response_mode]='ACCEPTED_AND_SCHEMA_VALIDATED'
                return result
            if last=='VISION_UNSUPPORTED':
                with self.lock:self.capabilities[model+':vision']='UNSUPPORTED'
                break
            if last=='STRUCTURED_OUTPUT_UNSUPPORTED' and len(levels)>1:
                with self.lock:self.capabilities[model+':'+response_mode]='REJECTED_BY_SERVER'
                levels.pop(0);response_mode=levels[0]
            elif last not in ('HTTP_408','HTTP_429','HTTP_500','HTTP_502','HTTP_503','HTTP_504','TRANSPORT_FAILURE','MODEL_JSON_OR_SCHEMA_INVALID','RESPONSE_ENVELOPE_INVALID'):break
            if attempt<self.config.max_retries:time.sleep(min(2**attempt,8))
        raise ProviderError(last or 'EXTRACTION_FAILED')
