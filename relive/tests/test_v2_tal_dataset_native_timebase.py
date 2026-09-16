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
  self.assertEqual(proof['clip_timestamp_seconds'],['0','1','6.5','6.5','55.5'])
  self.assertEqual(proof['anchor_source_frame_reference'],40)
  self.assertEqual(proof['duplicate_logical_frame_count'],1)
  out=self.root/'audit';report=audit_dataset_native_timebase(dataset_json=self.data,frame_root=self.root,source_prefix='/root/data',output_dir=out)
  self.assertEqual(report['status'],'PASS');self.assertNotIn('DO_NOT_READ_GT_SENTINEL',(out/'v2_nurvid_dataset_native_timebase_records.jsonl').read_text())
 def test_bad_layout_index_and_boundaries_fail_closed(self):
  bad=dict(self.row);bad['sampled_video_frames']=[40,39,151,151,151]
  with self.assertRaisesRegex(DatasetNativeTimebaseError,'NON_MONOTONIC'):_audit_record(bad,None,require_files=False)
  bad=dict(self.row);bad['video']=list(self.paths);bad['video'][0]=bad['video'][0].replace('000040','000041')
  with self.assertRaisesRegex(DatasetNativeTimebaseError,'PATH'):_audit_record(bad,None,require_files=False)
  # Index base is intentionally irrelevant to first-presented-frame time.
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
  self.assertEqual(result['timebase_status'],'RESOLVED_DATASET_NATIVE_CLIP_LOCAL')
  resolved=self.root/'resolved';freeze_video_index(requirement_dir=req,selection_manifest_path=selected,source_json=source,frame_root=self.root,source_prefix='/root/data',timebase_policy_path=policy,output_dir=resolved,public_timestamp_manifest_path=timestamps/'public_per_frame_timestamps.jsonl',public_timestamp_provenance_path=timestamps/'public_per_frame_timestamps.provenance.json')
  output=validate_video_index_artifacts(resolved);self.assertEqual(output['index_status'],'RESOLVED_DATASET_NATIVE_CLIP_LOCAL');self.assertTrue(output['ready_for_hypothesis_generation']);self.assertEqual(len(__import__('json').loads((resolved/'v2_video_index.jsonl').read_text())['frames']),5)

class ClipLocalContractTests(DatasetNativeTimebaseTests):
 def test_record6_and_record67_style_are_clip_local_without_bank_base(self):
  for refs, start, end, expected in [([209,327],0.0,60.0,['0','59']),([131,472],65.0,235.8,['0','170.5'])]:
   row=dict(self.row);row['sampled_video_frames']=refs;row['video']=[f'/root/data/NurViD/frames_2fps/{row["metadata"]["video_id"]}/{x:06d}.jpg' for x in refs];row['metadata']=dict(row['metadata'],input_video_start_time=start,input_video_end_time=end)
   proof=_audit_record(row,None,require_files=False);self.assertEqual(proof['clip_timestamp_seconds'],expected)
 def test_span_and_tail_bound_fail_closed(self):
  row=dict(self.row);row['sampled_video_frames']=[40,200];row['video']=[self.paths[0],self.paths[-1].replace('000151','000200')]
  with self.assertRaisesRegex(DatasetNativeTimebaseError,'SPAN_EXCEEDS'):_audit_record(row,None,require_files=False)
  row=dict(self.row);row['sampled_video_frames']=[40,42];row['video']=self.paths[:2]
  with self.assertRaisesRegex(DatasetNativeTimebaseError,'TAIL_GAP'):_audit_record(row,None,require_files=False)
