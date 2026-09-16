from __future__ import annotations
import json, re
from pathlib import Path
import unittest
from relive.storage.artifacts import canonical_json
from relive.v2.token_json_constraint import GroundingJsonGrammar, TokenLevelGroundingConstraint
from relive.v2.observation_retrieval import execute as d_run, preflight as d_preflight
from relive.v2.temporal_evidence_composition import prepare as e_prepare
from relive.v2.spatial_evidence_planning import prepare as f_prepare
from relive.v2.spatial_anchor_grounding_v32 import preflight, execute, validate
from relive.v2.spatial_anchor_grounding_v32_repeat import compare
from relive.v2.spatial_anchor_grounding_v32_smoke import freeze
from tests.test_v2_tal_observation_retrieval import ObservationRetrievalTests, _FakeObservationBackend

class _Tokenizer:
    # Contains multi-character legal pieces and deliberately illegal backtick.
    pieces = ['{', '}', '[', ']', ':', ',', '"', 'components', 'role', 'visibility', 'bbox_2d',
              'VISIBLE', 'NOT_VISIBLE', 'AMBIGUOUS', 'null', 'BASE_PLATE', 'TARGET_SKIN_AREA',
              'ALIGNMENT_INTERFACE', '0', '1', '2', '3', '4', '1000', '`']
    vocab_size = len(pieces) + 1
    def decode(self, ids, **_):
        return ''.join('' if i == self.vocab_size - 1 else self.pieces[i] for i in ids)

class _Grounder:
    synthetic = True
    def fingerprint(self): return {'adapter_version':'fake-stage3g-v32','scientific_identity':{'model':'fake','generation':{'do_sample':False}}}
    def audit_token_constraint(self, grammar):
        return {'binding':{'grammar_spec_sha256':grammar.spec_sha256},'tokenizer_binding':{'tokenizer_class':'fake','tokenizer_vocabulary_sha256':'fake'},'initial_allowed_token_count':1}
    def infer_with_token_constraint(self, request, grammar):
        roles = re.search(r'Required roles: (.*)\n',request['prompt']).group(1).split(', ')
        context = re.search(r'Contextual roles: (.*)\n',request['prompt']).group(1)
        roles += [] if context == 'none' else context.split(', ')
        raw = json.dumps({'components':[{'role':x,'visibility':'NOT_VISIBLE','bbox_2d':None} for x in roles]},separators=(',',':'))
        return {'raw_response':raw,'generation_metadata':{'finish_reason':'EOS_TOKEN','generated_token_count':10,'max_new_tokens':512,'reached_max_new_tokens':False},
                'constraint_metadata':{'binding':{'fake':True},'tokenizer_binding':{'tokenizer_class':'fake'},'execution':{'grammar_version':'relive-v2-grounding-token-json-grammar-v3.2','grammar_spec_sha256':grammar.spec_sha256,'implementation_version':'relive-v2-token-prefix-constraint-v1','constraint_failure':None,'constraint_failure_step':None,'final_prefix_status':'COMPLETE','final_prefix_reason':None}}}

class Stage3GV32Tests(unittest.TestCase):
 def setUp(self):
  self.d=ObservationRetrievalTests(methodName='test_packets_dedup_fanout_run_replay_and_no_admission');self.d.setUp();self.d.prepare();d_preflight(output_dir=self.d.out,config_path=self.d.base.config,stage3c_dir=self.d.stage3c,backend_factory=lambda _:_FakeObservationBackend());d_run(output_dir=self.d.out,config_path=self.d.base.config,mode='run',backend_factory=lambda _:_FakeObservationBackend())
  root=self.d.base.root;self.e=root/'e';self.f=root/'f';self.g=root/'g32';self.g2=root/'g322';base=Path(__file__).resolve().parents[1]
  e_prepare(stage3c_dir=self.d.stage3c,stage3d_dir=self.d.out,video_index_dir=self.d.base.index,policy_path=base/'configs/v2/tal_temporal_evidence_composition_policy.json',output_dir=self.e)
  f_prepare(stage3c_dir=self.d.stage3c,stage3d_dir=self.d.out,stage3e_dir=self.e,video_index_dir=self.d.base.index,policy_path=base/'configs/v2/tal_spatial_evidence_planning_policy.json',output_dir=self.f)
  self.policy=base/'configs/v2/tal_spatial_anchor_grounding_v32_policy.json'
 def tearDown(self): self.d.tearDown()
 def pf(self,out=None,smoke=None): return preflight(stage3f_dir=self.f,stage3c_dir=self.d.stage3c,stage3d_dir=self.d.out,stage3e_dir=self.e,video_index_dir=self.d.base.index,config_path=self.d.base.config,policy_path=self.policy,output_dir=out or self.g,smoke_selection_manifest=smoke,backend_factory=lambda _:_Grounder())
 def test_prefix_grammar_rejects_historical_and_terminal_failures(self):
  g=GroundingJsonGrammar(['BASE_PLATE'])
  good='{"components":[{"role":"BASE_PLATE","visibility":"NOT_VISIBLE","bbox_2d":null}]}'
  self.assertEqual(g.prefix_state(good).status,'COMPLETE')
  for bad in ('[{"role":"BASE_PLATE"}]', good+'`', '{"components":[{"role":"BASE_PLATE","visibility":"NOT_VISIBLE","bbox_2d":[null,null,null,null]}]}', '{"components":[]}'):
   self.assertEqual(g.prefix_state(bad).status,'INVALID')
  self.assertEqual(g.prefix_state('{"components":[{"role":"BASE_PLATE","visibility":"VISIBLE","bbox_2d":[1001').status,'INVALID')
 def test_fake_tokenizer_multichar_and_eos_contract(self):
  g=GroundingJsonGrammar(['BASE_PLATE']);c=TokenLevelGroundingConstraint(g);t=_Tokenizer();c.prepare(t,0,{t.vocab_size-1})
  ids=[]
  for piece in ['{','"','components','"',':','[','{','"','role','"',':','"','BASE_PLATE','"',',','"','visibility','"',':','"','NOT_VISIBLE','"',',','"','bbox_2d','"',':','null','}',']','}']:
   ids.append(t.pieces.index(piece));self.assertNotEqual(g.prefix_state(t.decode(ids)).status,'INVALID')
  self.assertEqual(g.prefix_state(t.decode(ids)).status,'COMPLETE')
  self.assertEqual(c.allowed_token_ids(ids),[t.vocab_size-1])
  self.assertNotIn(t.pieces.index('`'), c.allowed_token_ids(ids[:-1]))
  early=TokenLevelGroundingConstraint(g);early.prepare(t,0,{t.vocab_size-1})
  self.assertEqual(early.final_metadata([t.vocab_size-1])['constraint_failure'],'PREMATURE_EOS')
  class NoLegalTokenizer(_Tokenizer):
   pieces=['`'];vocab_size=2
  blocked=TokenLevelGroundingConstraint(g);bt=NoLegalTokenizer();blocked.prepare(bt,0,{1})
  self.assertEqual(blocked.allowed_token_ids([]),[1])
  self.assertEqual(blocked.final_metadata([1])['constraint_failure'],'NO_LEGAL_SUCCESSOR_TOKEN')
 def test_v32_run_replay_validate_repeat_and_same_v31_smoke_order(self):
  source=self.d.base.root/'v31.jsonl';records=[]
  all_entries=self.pf(self.d.base.root/'pre')['frozen_calls']
  for entry in all_entries[:4]: records.append({'anchor_candidate_id':entry['anchor_candidate_id'],'selection_reason':'FIXED','source_v2_result_sha256':'a'*64})
  source.write_text(''.join(canonical_json(x)+'\n' for x in records));smoke=self.d.base.root/'smoke';freeze(v31_selection_manifest=source,output_dir=smoke)
  selected=[json.loads(x)['anchor_candidate_id'] for x in (smoke/'v2_tal_stage3g_v32_smoke_selection.jsonl').read_text().splitlines()]
  self.assertEqual(selected,[x['anchor_candidate_id'] for x in records])
  plan=self.pf(smoke=smoke/'v2_tal_stage3g_v32_smoke_selection.jsonl');self.assertEqual(plan['generation_contract']['enforcement'],'ACTUAL_GENERATE_LOGITS_MASK_WITH_FAIL_CLOSED_PREFIX_GRAMMAR')
  first=execute(output_dir=self.g,config_path=self.d.base.config,policy_path=self.policy,mode='run',backend_factory=lambda _:_Grounder());second=execute(output_dir=self.g,config_path=self.d.base.config,policy_path=self.policy,mode='replay',backend_factory=lambda _:_Grounder())
  self.assertEqual(first['new_model_calls'],plan['planned_model_calls']);self.assertEqual(second['cache_hits'],plan['planned_model_calls']);self.assertEqual(validate(output_dir=self.g)['status'],'PASS')
  self.pf(self.g2,smoke=smoke/'v2_tal_stage3g_v32_smoke_selection.jsonl');execute(output_dir=self.g2,config_path=self.d.base.config,policy_path=self.policy,mode='run',backend_factory=lambda _:_Grounder())
  self.assertTrue(compare(run_a=self.g,run_b=self.g2,output_dir=self.d.base.root/'cmp')['canonical_grounding_result_hash_equal'])
if __name__=='__main__': unittest.main()
