from __future__ import annotations
import json,re
from pathlib import Path
import unittest
from relive.v2.observation_retrieval import execute as d_run, preflight as d_preflight
from relive.v2.temporal_evidence_composition import prepare as e_prepare
from relive.v2.spatial_evidence_planning import prepare as f_prepare
from relive.v2.spatial_anchor_grounding_v3 import preflight,execute,validate,SpatialAnchorGroundingError,_parse
from relive.v2.spatial_anchor_grounding_v3_repeat import compare
from relive.v2.spatial_anchor_grounding_v2_diagnostic import diagnose
from relive.v2.spatial_anchor_grounding_v3_smoke import freeze
from tests.test_v2_tal_observation_retrieval import ObservationRetrievalTests,_FakeObservationBackend

class _Grounder:
 synthetic=True
 def __init__(self,bad=False):self.bad=bad;self.calls=0
 def fingerprint(self):return {'adapter_version':'fake-stage3g-v3','scientific_identity':{'model':'fake-grounder','generation':{'do_sample':False}}}
 def infer_with_generation_metadata(self,request):
  self.calls+=1
  if self.bad:raw='[{"role":"BASE_PLATE","visibility":"VISIBLE","bbox_2d":[1,2,3,4]}]'
  else:
   required=re.search(r'Required roles: (.*)\n',request['prompt']).group(1).split(', ')
   context=re.search(r'Contextual roles: (.*)\n',request['prompt']).group(1)
   roles=required+([] if context=='none' else context.split(', '))
   raw=json.dumps({'components':[{'role':x,'visibility':'VISIBLE','bbox_2d':[10,20,900,950]} for x in roles]})
  return {'raw_response':raw,'generation_metadata':{'finish_reason':'EOS_TOKEN','generated_token_count':31,'max_new_tokens':512,'reached_max_new_tokens':False}}

class Stage3GV3Tests(unittest.TestCase):
 def setUp(self):
  self.d=ObservationRetrievalTests(methodName='test_packets_dedup_fanout_run_replay_and_no_admission');self.d.setUp();self.d.prepare();d_preflight(output_dir=self.d.out,config_path=self.d.base.config,stage3c_dir=self.d.stage3c,backend_factory=lambda _:_FakeObservationBackend());d_run(output_dir=self.d.out,config_path=self.d.base.config,mode='run',backend_factory=lambda _:_FakeObservationBackend())
  root=self.d.base.root;self.e=root/'e';self.f=root/'f';self.g=root/'g3';self.g2=root/'g32';self.v2=root/'v2';base=Path(__file__).resolve().parents[1]
  e_prepare(stage3c_dir=self.d.stage3c,stage3d_dir=self.d.out,video_index_dir=self.d.base.index,policy_path=base/'configs/v2/tal_temporal_evidence_composition_policy.json',output_dir=self.e)
  f_prepare(stage3c_dir=self.d.stage3c,stage3d_dir=self.d.out,stage3e_dir=self.e,video_index_dir=self.d.base.index,policy_path=base/'configs/v2/tal_spatial_evidence_planning_policy.json',output_dir=self.f)
  self.policy=base/'configs/v2/tal_spatial_anchor_grounding_v3_policy.json'
 def tearDown(self):self.d.tearDown()
 def pf(self,out=None,smoke=None):return preflight(stage3f_dir=self.f,stage3c_dir=self.d.stage3c,stage3d_dir=self.d.out,stage3e_dir=self.e,video_index_dir=self.d.base.index,config_path=self.d.base.config,policy_path=self.policy,output_dir=out or self.g,smoke_selection_manifest=smoke,backend_factory=lambda _:_Grounder())
 def test_v3_full_preflight_run_replay_validate_and_compare(self):
  plan=self.pf();self.assertEqual(plan['generation_contract']['json_schema_constrained_decoding_supported'],False);self.assertEqual(plan['selection_mode'],'FULL_FROZEN_COHORT')
  first=execute(output_dir=self.g,config_path=self.d.base.config,policy_path=self.policy,mode='run',backend_factory=lambda _:_Grounder());second=execute(output_dir=self.g,config_path=self.d.base.config,policy_path=self.policy,mode='replay',backend_factory=lambda _:_Grounder())
  self.assertEqual((first['new_model_calls'],second['cache_hits']),(plan['planned_model_calls'],plan['planned_model_calls']))
  self.assertEqual(validate(output_dir=self.g)['status'],'PASS')
  self.pf(self.g2);execute(output_dir=self.g2,config_path=self.d.base.config,policy_path=self.policy,mode='run',backend_factory=lambda _:_Grounder())
  self.assertTrue(compare(run_a=self.g,run_b=self.g2,output_dir=self.d.base.root/'comparison')['canonical_grounding_result_hash_equal'])
 def test_v3_rejects_top_level_array_without_repair(self):
  self.pf();execute(output_dir=self.g,config_path=self.d.base.config,policy_path=self.policy,mode='run',backend_factory=lambda _:_Grounder(bad=True))
  results=[json.loads(line) for line in (self.g/'run/v2_tal_spatial_anchor_groundings_v3.jsonl').read_text().splitlines()]
  self.assertTrue(all(row['failure_reason']=='MODEL_OUTPUT_PARSE_FAILURE' for row in results));self.assertTrue(all(row['raw_response_closure']=='CLOSED_ARRAY' for row in results))
  self.assertEqual(_parse('[{"x":1}]',['BASE_PLATE'],10,10)[1],'MODEL_OUTPUT_PARSE_FAILURE')
 def test_smoke_selection_freezes_v2_failure_categories_before_v3(self):
  # A compact synthetic immutable v2 run: one parse failure for each semantic role and one schema failure.
  (self.v2/'run').mkdir(parents=True)
  role=['PRECONDITION_ALIGNMENT','ACTION_CORE_PRESSING','POSTCONDITION_ATTACHMENT']
  source=[]
  for index,name in enumerate(role):source.append({'anchor_candidate_id':f'anchor-{index}','observation_role':name,'failure_reason':'MODEL_OUTPUT_PARSE_FAILURE','canonical_result_sha256':str(index),'raw_response':'['})
  source.append({'anchor_candidate_id':'anchor-schema','observation_role':role[0],'failure_reason':'MODEL_OUTPUT_SCHEMA_VIOLATION','canonical_result_sha256':'schema','raw_response':'{}'})
  from relive.storage.artifacts import canonical_json
  (self.v2/'run/v2_tal_spatial_anchor_groundings.jsonl').write_text(''.join(canonical_json(row)+'\n' for row in source))
  report=freeze(v2_output_dir=self.v2,output_dir=self.d.base.root/'smoke');self.assertEqual(report['anchor_count'],4)
  selected=[json.loads(x) for x in (self.d.base.root/'smoke/v2_tal_stage3g_v3_smoke_selection.jsonl').read_text().splitlines()]
  self.assertEqual(len(selected),4);self.assertEqual(len({x['anchor_candidate_id'] for x in selected}),4)
 def test_v2_diagnostic_is_read_only_and_marks_old_metadata_unavailable(self):
  (self.v2/'run').mkdir(parents=True);from relive.storage.artifacts import canonical_json
  source=[{'failure_reason':'MODEL_OUTPUT_PARSE_FAILURE','raw_response':'[', 'anchor_candidate_id':'x'}]
  (self.v2/'run/v2_tal_spatial_anchor_groundings.jsonl').write_text(canonical_json(source[0])+'\n')
  manifest={'format':'relive-v2-spatial-anchor-grounding-v1','grounding_result_count':1};(self.v2/'run/v2_tal_stage3g_manifest.json').write_text(canonical_json(manifest)+'\n')
  pre={'generation_parameters':{'max_new_tokens':512}};(self.v2/'stage3g_preflight.json').write_text(canonical_json(pre)+'\n')
  report=diagnose(v2_output_dir=self.v2,output_dir=self.d.base.root/'diag');self.assertFalse(report['v2_generation_metadata_available']);self.assertEqual(report['counts']['raw_response_closure'],{'UNCLOSED_OR_OTHER':1})
if __name__=='__main__':unittest.main()
