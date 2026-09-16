from __future__ import annotations
import json,re
from pathlib import Path
import unittest
from relive.v2.observation_retrieval import execute as d_run, preflight as d_preflight
from relive.v2.temporal_evidence_composition import prepare as e_prepare
from relive.v2.spatial_evidence_planning import prepare as f_prepare
from relive.v2.spatial_anchor_grounding_v31 import preflight,execute,validate,SpatialAnchorGroundingError,_parse
from relive.v2.spatial_anchor_grounding_v31_repeat import compare
from relive.v2.spatial_anchor_grounding_v31_smoke import freeze
from relive.storage.artifacts import canonical_json
from tests.test_v2_tal_observation_retrieval import ObservationRetrievalTests,_FakeObservationBackend

class _Grounder:
 synthetic=True
 def __init__(self,bad=False):self.bad=bad
 def fingerprint(self):return {'adapter_version':'fake-stage3g-v31','scientific_identity':{'model':'fake','generation':{'do_sample':False}}}
 def infer_with_generation_metadata(self,request):
  required=re.search(r'Required roles: (.*)\n',request['prompt']).group(1).split(', ')
  context=re.search(r'Contextual roles: (.*)\n',request['prompt']).group(1);roles=required+([] if context=='none' else context.split(', '))
  if self.bad:
   raw=json.dumps({'components':[{'role':role,'visibility':'NOT_VISIBLE','bbox_2d':[None,None,None,None]} for role in roles]})
  else:
   raw=json.dumps({'components':[{'role':role,'visibility':'NOT_VISIBLE','bbox_2d':None} for role in roles]})
  return {'raw_response':raw,'generation_metadata':{'finish_reason':'EOS_TOKEN','generated_token_count':22,'max_new_tokens':512,'reached_max_new_tokens':False}}

class Stage3GV31Tests(unittest.TestCase):
 def setUp(self):
  self.d=ObservationRetrievalTests(methodName='test_packets_dedup_fanout_run_replay_and_no_admission');self.d.setUp();self.d.prepare();d_preflight(output_dir=self.d.out,config_path=self.d.base.config,stage3c_dir=self.d.stage3c,backend_factory=lambda _:_FakeObservationBackend());d_run(output_dir=self.d.out,config_path=self.d.base.config,mode='run',backend_factory=lambda _:_FakeObservationBackend())
  root=self.d.base.root;self.e=root/'e';self.f=root/'f';self.g=root/'g31';self.g2=root/'g312';base=Path(__file__).resolve().parents[1]
  e_prepare(stage3c_dir=self.d.stage3c,stage3d_dir=self.d.out,video_index_dir=self.d.base.index,policy_path=base/'configs/v2/tal_temporal_evidence_composition_policy.json',output_dir=self.e)
  f_prepare(stage3c_dir=self.d.stage3c,stage3d_dir=self.d.out,stage3e_dir=self.e,video_index_dir=self.d.base.index,policy_path=base/'configs/v2/tal_spatial_evidence_planning_policy.json',output_dir=self.f)
  self.policy=base/'configs/v2/tal_spatial_anchor_grounding_v31_policy.json'
 def tearDown(self):self.d.tearDown()
 def pf(self,out=None,smoke=None):return preflight(stage3f_dir=self.f,stage3c_dir=self.d.stage3c,stage3d_dir=self.d.out,stage3e_dir=self.e,video_index_dir=self.d.base.index,config_path=self.d.base.config,policy_path=self.policy,output_dir=out or self.g,smoke_selection_manifest=smoke,backend_factory=lambda _:_Grounder())
 def test_same_v3_smoke_ids_are_frozen_in_order(self):
  source=self.d.base.root/'v3.jsonl';records=[]
  all_entries=self.pf(self.d.base.root/'pre') ['frozen_calls']
  for entry in all_entries[:4]:records.append({'anchor_candidate_id':entry['anchor_candidate_id'],'selection_reason':'FIXED','source_v2_result_sha256':'a'*64})
  source.write_text(''.join(canonical_json(row)+'\n' for row in records))
  smoke=self.d.base.root/'smoke';report=freeze(v3_selection_manifest=source,output_dir=smoke)
  frozen=[json.loads(line) for line in (smoke/'v2_tal_stage3g_v31_smoke_selection.jsonl').read_text().splitlines()]
  self.assertEqual([x['anchor_candidate_id'] for x in frozen],[x['anchor_candidate_id'] for x in records]);self.assertTrue(report['anchor_order_preserved'])
 def test_v31_prompt_examples_and_run_replay_validate_repeat(self):
  text=(Path(__file__).resolve().parents[1]/'src/relive/prompts/v2_tal_spatial_anchor_grounding_v3_1.txt').read_text()
  self.assertIn('NOT_VISIBLE: {{"role":"ROLE","visibility":"NOT_VISIBLE","bbox_2d":null}}',text);self.assertIn('[null,null,null,null]',text)
  plan=self.pf();self.assertEqual(plan['generation_contract']['structured_generation_enforcement'],'PROMPT_ONLY_FAIL_CLOSED')
  first=execute(output_dir=self.g,config_path=self.d.base.config,policy_path=self.policy,mode='run',backend_factory=lambda _:_Grounder());second=execute(output_dir=self.g,config_path=self.d.base.config,policy_path=self.policy,mode='replay',backend_factory=lambda _:_Grounder())
  self.assertEqual(first['new_model_calls'],plan['planned_model_calls']);self.assertEqual(second['cache_hits'],plan['planned_model_calls']);self.assertEqual(validate(output_dir=self.g)['status'],'PASS')
  self.pf(self.g2);execute(output_dir=self.g2,config_path=self.d.base.config,policy_path=self.policy,mode='run',backend_factory=lambda _:_Grounder())
  self.assertTrue(compare(run_a=self.g,run_b=self.g2,output_dir=self.d.base.root/'cmp')['canonical_grounding_result_hash_equal'])
 def test_null_array_remains_schema_failure(self):
  self.pf();execute(output_dir=self.g,config_path=self.d.base.config,policy_path=self.policy,mode='run',backend_factory=lambda _:_Grounder(bad=True))
  rows=[json.loads(line) for line in (self.g/'run/v2_tal_spatial_anchor_groundings_v3_1.jsonl').read_text().splitlines()]
  self.assertTrue(all(row['failure_reason']=='MODEL_OUTPUT_SCHEMA_VIOLATION' for row in rows));self.assertEqual(_parse('{"components":[{"role":"BASE_PLATE","visibility":"NOT_VISIBLE","bbox_2d":[null,null,null,null]}]}',['BASE_PLATE'],10,10)[1],'MODEL_OUTPUT_SCHEMA_VIOLATION')
if __name__=='__main__':unittest.main()
