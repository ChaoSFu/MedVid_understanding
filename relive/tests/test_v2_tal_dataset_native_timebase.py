from __future__ import annotations
import hashlib, tempfile, unittest
from pathlib import Path
from PIL import Image
from relive.storage.artifacts import canonical_json, stable_hash
from relive.v2.dataset_native_timebase import DatasetNativeTimebaseError, _audit_record, audit_dataset_native_timebase, export_dataset_native_timebase

class DatasetNativeTimebaseTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.frames=self.root/'NurViD/frames_2fps/vR0_BaXYcE4';self.frames.mkdir(parents=True)
  self.refs=[40,42,53,53,151];self.paths=[f'/root/data/NurViD/frames_2fps/vR0_BaXYcE4/{x:06d}.jpg' for x in self.refs]
  for ref in [1,*set(self.refs)]:Image.new('RGB',(4,3),(ref%255,0,0)).save(self.frames/f'{ref:06d}.jpg')
  self.row={'id':'vR0_BaXYcE4&&19.50&&75.20&&1.0','sampled_video_frames':self.refs,'video':self.paths,'train':False,'data_source':'NurViD','dataset_name':'NurViD','qa_type':'tal','metadata':{'video_id':'vR0_BaXYcE4','fps':1.0,'input_video_start_time':19.5,'input_video_end_time':75.2},'conversations':[{'from':'gpt','value':'DO_NOT_READ_GT_SENTINEL'}],'struc_info':'DO_NOT_READ_GT_SENTINEL'}
  self.data=self.root/'data.json';self.data.write_text(canonical_json([self.row])+'\n')
 def tearDown(self):self.tmp.cleanup()
 def test_formula_duplicates_and_full_audit(self):
  from relive.v2.video_index import FrameRootMapper
  proof=_audit_record(self.row,FrameRootMapper.create('/root/data',self.root),require_files=True)
  self.assertEqual(proof['source_timestamp_seconds'],['19.5','20.5','26','26','75'])
  self.assertEqual(proof['clip_timestamp_seconds'],['0.0','1.0','6.5','6.5','55.5'])
  self.assertEqual(proof['duplicate_logical_frame_count'],1)
  out=self.root/'audit';report=audit_dataset_native_timebase(dataset_json=self.data,frame_root=self.root,source_prefix='/root/data',output_dir=out)
  self.assertEqual(report['status'],'PASS');self.assertNotIn('DO_NOT_READ_GT_SENTINEL',(out/'v2_nurvid_dataset_native_timebase_records.jsonl').read_text())
 def test_bad_layout_index_and_boundaries_fail_closed(self):
  bad=dict(self.row);bad['sampled_video_frames']=[40,39,151,151,151]
  with self.assertRaisesRegex(DatasetNativeTimebaseError,'NON_MONOTONIC'):_audit_record(bad,None,require_files=False)
  bad=dict(self.row);bad['video']=list(self.paths);bad['video'][0]=bad['video'][0].replace('000040','000041')
  with self.assertRaisesRegex(DatasetNativeTimebaseError,'PATH'):_audit_record(bad,None,require_files=False)
  (self.frames/'000001.jpg').unlink()
  from relive.v2.video_index import FrameRootMapper
  with self.assertRaisesRegex(DatasetNativeTimebaseError,'INDEX_BASE'):_audit_record(self.row,FrameRootMapper.create('/root/data',self.root),require_files=True)
 def test_unsupported_layout_fails(self):
  with self.assertRaisesRegex(DatasetNativeTimebaseError,'UNSUPPORTED'):
   audit_dataset_native_timebase(dataset_json=self.data,frame_root=self.root,source_prefix='/root/data',output_dir=self.root/'x',frame_bank_layout='frames_1fps')
if __name__=='__main__':unittest.main()

class DatasetNativeIntegrationTests(DatasetNativeTimebaseTests):
 def test_stage2_consumes_dataset_native_manifest_without_deduplication(self):
  import shutil
  from relive.v2.video_index import _source_identity, freeze_video_index, validate_video_index_artifacts
  from relive.v2.task_selection import freeze_selector_order, write_selection
  from relive.v2.requirement_freeze import freeze_tal_requirements
  from relive.v2.temporal_localization import load_event_ontology
  question='When does Secure the base happen?'; row=dict(self.row);row['conversations']=[{'from':'human','value':'<video>'+question},{'from':'gpt','value':'DO_NOT_READ_GT_SENTINEL'}]
  identity,_,_=_source_identity(row,0)
  selector=self.root/'selector.jsonl';selector.write_text(canonical_json({**{key:identity[key] for key in ('source_record_index','sample_id','public_record_sha256','question_sha256')},'qa_type':'tal','question':question})+'\n')
  selection=freeze_selector_order(selector_path=selector,max_samples=1); selected=self.root/'selection.json';write_selection(selection,selected)
  ontology=load_event_ontology(Path(__file__).resolve().parents[1]/'configs/v2/tal_event_ontology.yaml'); req=self.root/'req';freeze_tal_requirements(selection=selection,selector_sha256=hashlib.sha256(selector.read_bytes()).hexdigest(),selection_manifest_sha256=hashlib.sha256(selected.read_bytes()).hexdigest(),ontology=ontology,output_dir=req)
  source=self.root/'source.json';source.write_text(canonical_json([row])+'\n'); policy=Path(__file__).resolve().parents[1]/'configs/v2/tal_timebase_sources.yaml'
  media=self.root/'media';freeze_video_index(requirement_dir=req,selection_manifest_path=selected,source_json=source,frame_root=self.root,source_prefix='/root/data',timebase_policy_path=policy,output_dir=media)
  timestamps=self.root/'timestamps';result=export_dataset_native_timebase(dataset_json=source,selection_manifest=selected,requirement_freeze_dir=req,media_audit_dir=media,frame_root=self.root,source_prefix='/root/data',output_dir=timestamps)
  self.assertEqual(result['timebase_status'],'RESOLVED_DATASET_NATIVE')
  resolved=self.root/'resolved';freeze_video_index(requirement_dir=req,selection_manifest_path=selected,source_json=source,frame_root=self.root,source_prefix='/root/data',timebase_policy_path=policy,output_dir=resolved,public_timestamp_manifest_path=timestamps/'public_per_frame_timestamps.jsonl',public_timestamp_provenance_path=timestamps/'public_per_frame_timestamps.provenance.json')
  output=validate_video_index_artifacts(resolved);self.assertEqual(output['index_status'],'RESOLVED_DATASET_NATIVE');self.assertTrue(output['ready_for_hypothesis_generation']);self.assertEqual(len(__import__('json').loads((resolved/'v2_video_index.jsonl').read_text())['frames']),5)

class DatasetNativeCliGateTests(DatasetNativeTimebaseTests):
 def test_cli_stops_before_export_when_global_audit_is_unresolved(self):
  import os, subprocess, sys
  bad=dict(self.row);bad['video']=list(self.paths);bad['video'][0]=bad['video'][0].replace('000040','000041')
  dataset=self.root/'bad.json';dataset.write_text(canonical_json([bad])+'\n')
  script=Path(__file__).resolve().parents[1]/'scripts/export_v2_tal_dataset_native_timebase.py'
  result=subprocess.run([sys.executable,str(script),'--dataset-json',str(dataset),'--selection-manifest',str(self.root/'unused'), '--requirement-freeze-dir',str(self.root/'unused'), '--media-audit-dir',str(self.root/'unused'),'--frame-root',str(self.root),'--source-prefix','/root/data','--frame-bank-layout','frames_2fps','--audit-output-dir',str(self.root/'audit'),'--output-dir',str(self.root/'export')],cwd=script.parents[1],env={**os.environ,'PYTHONPATH':str(script.parents[1]/'src')},capture_output=True,text=True)
  self.assertEqual(result.returncode,2);self.assertFalse((self.root/'export').exists());self.assertIn('UNRESOLVED_TIMEBASE_SOURCE',result.stdout)
