from __future__ import annotations
import json, re
from pathlib import Path
import unittest
from relive.v2.observation_retrieval import execute as d_run, preflight as d_preflight
from relive.v2.temporal_evidence_composition import prepare as e_prepare
from relive.v2.spatial_evidence_planning import prepare as f_prepare
from relive.v2.spatial_anchor_grounding import preflight,execute,validate,SpatialAnchorGroundingError,_parse
from relive.v2.spatial_anchor_grounding_repeat import compare
from tests.test_v2_tal_observation_retrieval import ObservationRetrievalTests,_FakeObservationBackend

class _Grounder:
 synthetic=True
 def __init__(self, bad=False):self.bad=bad;self.calls=0
 def fingerprint(self):return {'adapter_version':'fake-stage3g-v1','scientific_identity':{'model':'fake-grounder','processor':'fake','generation':{'do_sample':False}}}
 def infer(self,request):
  self.calls+=1
  if self.bad:return '{"components":[]}'
  required=re.search(r'Required roles: (.*)\n',request['prompt']).group(1).split(', ')
  context=re.search(r'Contextual roles: (.*)\n',request['prompt']).group(1)
  roles=required+([] if context=='none' else context.split(', '))
  return json.dumps({'components':[{'role':x,'visibility':'VISIBLE','bbox_2d':[10,20,900,950]} for x in roles]})

class Stage3GTests(unittest.TestCase):
 def setUp(self):
  self.d=ObservationRetrievalTests(methodName='test_packets_dedup_fanout_run_replay_and_no_admission');self.d.setUp();self.d.prepare();d_preflight(output_dir=self.d.out,config_path=self.d.base.config,stage3c_dir=self.d.stage3c,backend_factory=lambda _:_FakeObservationBackend());d_run(output_dir=self.d.out,config_path=self.d.base.config,mode='run',backend_factory=lambda _:_FakeObservationBackend())
  root=self.d.base.root;self.e=root/'e';self.f=root/'f';self.g=root/'g';base=Path(__file__).resolve().parents[1]
  e_prepare(stage3c_dir=self.d.stage3c,stage3d_dir=self.d.out,video_index_dir=self.d.base.index,policy_path=base/'configs/v2/tal_temporal_evidence_composition_policy.json',output_dir=self.e)
  f_prepare(stage3c_dir=self.d.stage3c,stage3d_dir=self.d.out,stage3e_dir=self.e,video_index_dir=self.d.base.index,policy_path=base/'configs/v2/tal_spatial_evidence_planning_policy.json',output_dir=self.f)
  self.policy=base/'configs/v2/tal_spatial_anchor_grounding_policy.json'
 def tearDown(self):self.d.tearDown()
 def pf(self,out=None):return preflight(stage3f_dir=self.f,stage3c_dir=self.d.stage3c,stage3d_dir=self.d.out,stage3e_dir=self.e,video_index_dir=self.d.base.index,config_path=self.d.base.config,policy_path=self.policy,output_dir=out or self.g,backend_factory=lambda _:_Grounder())
 def test_full_anchor_enumeration_run_replay_validation_and_review(self):
  plan=self.pf();self.assertGreater(plan['planned_model_calls'],0);self.assertEqual(plan['frames_read'],plan['planned_model_calls'])
  first=execute(output_dir=self.g,config_path=self.d.base.config,policy_path=self.policy,mode='run',backend_factory=lambda _:_Grounder());second=execute(output_dir=self.g,config_path=self.d.base.config,policy_path=self.policy,mode='replay',backend_factory=lambda _:_Grounder())
  self.assertEqual((first['new_model_calls'],first['cache_hits']),(plan['planned_model_calls'],0));self.assertEqual((second['new_model_calls'],second['cache_hits']),(0,plan['planned_model_calls']));self.assertEqual(first['canonical_grounding_result_sha256'],second['canonical_grounding_result_sha256'])
  rows=[json.loads(x) for x in (self.g/'run/v2_tal_spatial_anchor_groundings.jsonl').read_text().splitlines()]
  self.assertEqual(len(rows),plan['planned_model_calls']);self.assertTrue(all(x['status']=='COMPONENTS_LOCALIZED_UNVERIFIED' for x in rows));self.assertTrue(all(x['geometry_type']!='DYNAMIC_SUPPORT_TUBE_COMPLETED' for x in rows));self.assertTrue((self.g/'run/review_packet').is_dir());self.assertEqual(validate(output_dir=self.g)['status'],'PASS')
 def test_image_tampering_fails_before_model_call(self):
  plan=self.pf();path=Path(plan['frozen_calls'][0]['image_path']);original=path.read_bytes()
  try:
   path.write_bytes(original+b'changed')
   with self.assertRaisesRegex(SpatialAnchorGroundingError,'FRAME_SHA256_MISMATCH'):
    execute(output_dir=self.g,config_path=self.d.base.config,policy_path=self.policy,mode='run',backend_factory=lambda _:_Grounder())
  finally:path.write_bytes(original)
 def test_bad_output_is_retained_not_promoted_and_tamper_fails(self):
  self.pf();execute(output_dir=self.g,config_path=self.d.base.config,policy_path=self.policy,mode='run',backend_factory=lambda _:_Grounder(bad=True))
  rows=[json.loads(x) for x in (self.g/'run/v2_tal_spatial_anchor_groundings.jsonl').read_text().splitlines()];self.assertTrue(all(x['failure_reason']=='REQUIRED_COMPONENT_MISSING' for x in rows));self.assertTrue(all(x['status']=='SPATIAL_ANCHOR_GROUNDING_CANDIDATE_UNVERIFIED' for x in rows))
  p=self.g/'run/v2_tal_spatial_anchor_groundings.jsonl';p.write_bytes(p.read_bytes()+b' ')
  with self.assertRaisesRegex(SpatialAnchorGroundingError,'ARTIFACT_TAMPERED'):validate(output_dir=self.g)
 def test_strict_parser_rejects_bad_visibility_boxes_and_accepts_fenced_json(self):
  expected=['BASE_PLATE']
  parsed,error=_parse('```json\n{"components":[{"role":"BASE_PLATE","visibility":"VISIBLE","bbox_2d":[0,0,1000,1000]}]}\n```',expected,200,100)
  self.assertIsNone(error);self.assertEqual(parsed[0]['bbox_pixel_xyxy'],[0,0,200,100])
  for payload in ({'components':[{'role':'BASE_PLATE','visibility':'VISIBLE','bbox_2d':None}]},{'components':[{'role':'BASE_PLATE','visibility':'NOT_VISIBLE','bbox_2d':[1,1,2,2]}]},{'components':[{'role':'BASE_PLATE','visibility':'VISIBLE','bbox_2d':[1,1,1,2]}]}):
   _,error=_parse(json.dumps(payload),expected,20,20);self.assertIn(error,{'INVALID_BOUNDING_BOX','MODEL_OUTPUT_SCHEMA_VIOLATION'})
 def test_independent_repeat_comparator(self):
  self.pf();execute(output_dir=self.g,config_path=self.d.base.config,policy_path=self.policy,mode='run',backend_factory=lambda _:_Grounder())
  other=self.d.base.root/'g2';self.pf(other);execute(output_dir=other,config_path=self.d.base.config,policy_path=self.policy,mode='run',backend_factory=lambda _:_Grounder())
  result=compare(run_a=self.g,run_b=other,output_dir=self.d.base.root/'cmp');self.assertTrue(result['canonical_grounding_result_hash_equal']);self.assertEqual(result['missing_or_duplicate_key_count'],0)
if __name__=='__main__':unittest.main()
