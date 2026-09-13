"""GPU-free Phase 3.6 route, contract, formal-path, and replay tests."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import sys, tempfile, unittest
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from relive.config import load_config
from relive.data.schemas import FIELD_SOURCES
from relive.phase35_v3 import preflight as v3_preflight, execute as v3_execute
from relive.phase36 import Phase36Error, ROUTES, _parent, _route, execute, preflight
from relive.spatial_adaptation_v2 import parse_refinement_v2

ROOT = Path(__file__).resolve().parents[1]

class Phase36Tests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
  frames=[]
  for i in range(3):
   p=self.root/f'f{i}.png'; Image.new('RGB',(40,30),(40+i*30,80,120)).save(p); frames.append(p)
  claims=[('phase35-local-001','The tip of the forceps is touching tissue.'),('phase35-local-002','The jaws of the forceps are touching tissue.'),('phase35-local-003','A hand is holding a white pad against skin.')]
  rows=[]
  for i,(cid,text) in enumerate(claims):
   rows.append({'sample_id':f's{i}','task':'claim_verification','question':'q','frames':[{'frame_id':f'{cid}:f','path':str(frames[i]),'order':0}], 'target_claim':{'claim_id':cid,'text':text,'time_scope':{'frame_ids':[f'{cid}:f']}},'metadata':{'dataset_name':'fixture','source_qa_type':'tal','runtime_adapter':'phase35_frozen_public_window_v1','nonofficial_protocol':True}})
  self.runtime=self.root/'runtime.jsonl'; payload=''.join(json.dumps(r,sort_keys=True,separators=(',',':'))+'\n' for r in rows).encode(); self.runtime.write_bytes(payload); sha=hashlib.sha256(payload).hexdigest()
  Path(str(self.runtime)+'.provenance.json').write_text(json.dumps({'schema_version':'relive-runtime-v1','source_kind':'public_runtime','runtime_sha256':sha,'field_sources':FIELD_SOURCES}))
  Path(str(self.runtime)+'.gt_isolation_audit.json').write_text(json.dumps({'status':'PASS','runtime_sha256':sha,'classification':'fixture'}))
  cfg=load_config(ROOT/'configs'/'mock_smoke.yaml'); cfg['claims']['contrast_fixtures']=[]; cfg['policy']={'name':'semantic_spatial','version':'relive-v1-policy-1','strict_alternatives':True}; cfg['adaptation']={'enabled':False,'actions':[],'expand_frames':1}; cfg['spatial']['intervention']={'operator':'opaque_gray','operator_version':'relive-opaque-gray-hard-mask-v1','parameters':{'fill_rgb':[127,127,127]}}; cfg['budget'].update(max_calls=6,max_candidates=1,max_spatial_proposals=1,max_rounds=1)
  rules=[]
  def rule(match,response): rules.append({'match':match,'response':response})
  # Frozen Phase 3-v3 R0 outcomes: two spatial failures and one ORIGINAL insufficiency.
  rule({'stage':'semantic','claim_text':claims[2][1],'variant':'ORIGINAL'},{'status':'INSUFFICIENT'})
  for text in (claims[0][1],claims[1][1]): rule({'stage':'semantic','claim_text':text,'variant':'DROP_TARGET'},{'status':'SUPPORTED'})
  rule({'stage':'semantic','claim_text':claims[1][1],'variant':'KEEP_TARGET'},{'status':'INSUFFICIENT'})
  rule({'stage':'spatial','claim_text':claims[0][1]},{'support_region':[0.386,0.438,0.896,0.999],'coordinate_system':'normalized_0_1_xyxy'})
  rule({'stage':'spatial','claim_text':claims[1][1]},{'support_region':[0.396,0.549,0.522,0.654],'coordinate_system':'normalized_0_1_xyxy'})
  rule({'stage':'reground_local_interaction_overbroad'},{'status':'PROPOSED','bbox_normalized_0_1000':[80,120,360,420],'reason':'interaction'})
  rule({'stage':'reground_local_interaction_incomplete'},{'status':'PROPOSED','bbox_normalized_0_1000':[180,250,620,720],'reason':'complete interaction'})
  cfg['backend']['rules']=rules; self.config=self.root/'cfg.json'; self.config.write_text(json.dumps(cfg))
  manifest={'format':'relive-phase35-prospective-local-atomic-manifest-v1','router_version':'relive-claim-scope-router-v1','selection_status':'FROZEN_PRE_SPATIAL_CERTIFICATE_OUTCOMES','selection_rule':{'input_order':'public runtime JSONL order','include_scope':'LOCAL_ATOMIC','max_samples':5,'exclude_sample_ids_from':'old'},'prospective_runtime':str(self.runtime),'prospective_runtime_sha256':sha,'excluded_historical_sample_ids_sha256':'0'*64,'selected_sample_count':3,'selected':[{'format':'relive-claim-scope-audit-v1','sample_id':f's{i}','dataset_name':'fixture','source_qa_type':'tal','claim_id':cid,'claim_text_sha256':hashlib.sha256(text.encode()).hexdigest(),'development_diagnostic_only':False,'claim_scope':'LOCAL_ATOMIC','single_roi_certificate_applicable':True,'reason_code':'LOCAL_OBJECT_STATE_OR_RELATION','router_version':'relive-claim-scope-router-v1','gt_used':False,'diagnostic_rationale':'local'} for i,(cid,text) in enumerate(claims)],'not_a_temporal_candidate_pool':True,'phase3_v3_executed':False,'gt_used':False,'prohibited_inputs_not_opened':['reference_answer','assistant_answer','temporal_gt','bbox_mask_gt','struc_info','RC_info','evaluation_artifacts']}
  content={k:v for k,v in manifest.items() if k!='manifest_sha256'}; manifest['manifest_sha256']=hashlib.sha256(json.dumps([content],sort_keys=True,separators=(',',':')).encode()).hexdigest(); self.manifest=self.root/'manifest.json';self.manifest.write_text(json.dumps(manifest))
  self.v3=self.root/'v3'; v3_preflight(config_path=self.config,runtime_path=self.runtime,prospective_manifest_path=self.manifest,output_dir=self.v3,require_real=False); v3_execute(config_path=self.config,runtime_path=self.runtime,prospective_manifest_path=self.manifest,output_dir=self.v3,mode='run',require_real=False)
 def tearDown(self): self.temp.cleanup()
 def test_routes_contract_and_replay(self):
  out=self.root/'p36'; plan=preflight(config_path=self.config,runtime_path=self.runtime,prospective_manifest_path=self.manifest,phase35_v3_run_dir=self.v3/'run',output_dir=out,require_real=False)
  self.assertEqual(plan['cohort_claim_ids'],list(ROUTES)); self.assertEqual(plan['excluded']['claim_id'],'phase35-local-003'); self.assertEqual(plan['model_calls_made'],0)
  first=execute(config_path=self.config,runtime_path=self.runtime,prospective_manifest_path=self.manifest,phase35_v3_run_dir=self.v3/'run',output_dir=out,mode='run',require_real=False)
  replay=execute(config_path=self.config,runtime_path=self.runtime,prospective_manifest_path=self.manifest,phase35_v3_run_dir=self.v3/'run',output_dir=out,mode='replay',require_real=False)
  self.assertEqual(first['status'],'PASS'); self.assertTrue(first['summary']['one_round_only']); self.assertGreater(first['summary']['new_model_calls'],0); self.assertEqual(replay['summary']['new_model_calls'],0); self.assertGreater(replay['summary']['cache_hits'],0); self.assertTrue((out/'phase36_summary.md').is_file()); self.assertEqual(len((out/'run'/'phase36_formal_verification.jsonl').read_text().splitlines()),2)
 def test_rejects_noop_and_no_second_round(self):
  # Parser-level no-op is already strict; Phase 3.6 freezes its single route table.
  self.assertEqual(set(ROUTES),{'phase35-local-001','phase35-local-002'})
  # Exact R0 copies are terminal REFINEMENT_NO_OP and cannot reach formal intervention.
  samples = __import__('relive.data.schemas', fromlist=['load_runtime']).load_runtime(self.runtime)
  sample = next(item for item in samples if item.target_claim.claim_id == 'phase35-local-001')
  parent = __import__('relive.types', fromlist=['SpatialProposal']).SpatialProposal('r0', 'candidate', sample.target_claim.claim_id, (0.1, 0.2, 0.3, 0.4))
  route = _route('phase35-local-001')
  no_op = parse_refinement_v2('{\"status\":\"PROPOSED\",\"bbox_normalized_0_1000\":[100,200,300,400],\"reason\":\"same\"}', __import__('relive.phase35_v3', fromlist=['candidate_for']).candidate_for(sample), sample.target_claim, parent=parent, route=route, raw_response_ref='raw', protocol_format='phase36')
  self.assertEqual(no_op['outcome'], 'REFINEMENT_NO_OP')
  with self.assertRaisesRegex(Phase36Error,'output directory'):
   preflight(config_path=self.config,runtime_path=self.runtime,prospective_manifest_path=self.manifest,phase35_v3_run_dir=self.v3/'run',output_dir=self.v3,require_real=False)
if __name__=='__main__': unittest.main()
