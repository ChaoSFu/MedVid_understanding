"""GT-isolated NurViD ``frames_2fps`` clip-local timebase adapter."""
from __future__ import annotations
from decimal import Decimal
from pathlib import Path, PurePosixPath
from typing import Any, Mapping
from relive.storage.artifacts import canonical_json, stable_hash
from .task_selection import IDENTITY_FIELDS, TALSelectionError, strict_json_loads
from .video_index import FrameRootMapper, _load_requirements, _load_selection, _sha, _write, _jsonl
from .timestamp_provenance import TIMESTAMP_AUDIT_FORMAT, TIMESTAMP_PROVENANCE_FORMAT

ADAPTER_VERSION="relive-v2-nurvid-frames-2fps-clip-local-v2"; FRAME_BANK_LAYOUT="frames_2fps"; RATE=Decimal("2")
ALLOWED=frozenset({"sampled_video_frames","video","id","train","data_source","dataset_name","qa_type","metadata"}); META=frozenset({"video_id","fps","input_video_start_time","input_video_end_time"})
class DatasetNativeTimebaseError(ValueError): pass

def _dec(v: Any, code: str) -> Decimal:
 if isinstance(v,bool) or not isinstance(v,(str,int,float)): raise DatasetNativeTimebaseError(code)
 try: x=Decimal(str(v))
 except Exception as e: raise DatasetNativeTimebaseError(code) from e
 if not x.is_finite(): raise DatasetNativeTimebaseError(code)
 return x

def _safe(row: Mapping[str,Any]) -> dict[str,Any]:
 if not isinstance(row,Mapping): raise DatasetNativeTimebaseError("SOURCE_RECORD_INVALID")
 x={k:row.get(k) for k in ALLOWED}; md={k:x["metadata"].get(k) for k in META} if isinstance(x["metadata"],Mapping) else {}
 if x["dataset_name"]!="NurViD" or x["data_source"]!="NurViD": raise DatasetNativeTimebaseError("UNSUPPORTED_FRAME_BANK_LAYOUT")
 if not isinstance(x["id"],str) or not x["id"] or set(md)!=META or not isinstance(md["video_id"],str) or not md["video_id"]: raise DatasetNativeTimebaseError("SOURCE_PUBLIC_METADATA_INVALID")
 refs,paths=x["sampled_video_frames"],x["video"]
 if not isinstance(refs,list) or not isinstance(paths,list) or not refs or len(refs)!=len(paths): raise DatasetNativeTimebaseError("FRAME_REFERENCE_COUNT_MISMATCH")
 if any(type(i) is not int or i<0 for i in refs): raise DatasetNativeTimebaseError("FRAME_REFERENCE_INVALID")
 if any(b<a for a,b in zip(refs,refs[1:])): raise DatasetNativeTimebaseError("NON_MONOTONIC_SOURCE_REFERENCE")
 if any(not isinstance(p,str) or not p for p in paths): raise DatasetNativeTimebaseError("FRAME_REFERENCE_PATH_MISMATCH")
 start,end,fps=_dec(md["input_video_start_time"],"SOURCE_TIME_METADATA_INVALID"),_dec(md["input_video_end_time"],"SOURCE_TIME_METADATA_INVALID"),_dec(md["fps"],"SOURCE_TIME_METADATA_INVALID")
 if end<=start or fps<=0 or fps>RATE: raise DatasetNativeTimebaseError("SOURCE_TIME_METADATA_INVALID")
 return {"id":x["id"],"refs":refs,"paths":paths,"video_id":md["video_id"],"start":start,"end":end,"fps":fps}

def _path_ref(path:str,video_id:str)->int:
 p=PurePosixPath(path); name=p.name
 if len(p.parts)<3 or p.parent.name!=video_id or "frames_2fps" not in p.parts or len(name)!=10 or not name.endswith(".jpg") or not name[:-4].isdigit(): raise DatasetNativeTimebaseError("FRAME_REFERENCE_PATH_MISMATCH")
 return int(name[:-4])

def _audit_record(record:Mapping[str,Any], mapper:FrameRootMapper|None, *, require_files:bool)->dict[str,Any]:
 r=_safe(record); refs,paths=r["refs"],r["paths"]
 if [_path_ref(p,r["video_id"]) for p in paths]!=refs: raise DatasetNativeTimebaseError("FRAME_REFERENCE_PATH_MISMATCH")
 if require_files:
  if mapper is None: raise DatasetNativeTimebaseError("FRAME_ROOT_REQUIRED")
  for p in paths: mapper.materialize(p)
 anchor=refs[0]; times=[(Decimal(ref)-Decimal(anchor))/RATE for ref in refs]
 span=times[-1]; duration=r["end"]-r["start"]; input_period=Decimal(1)/r["fps"]; bank_period=Decimal(1)/RATE; tail=duration-span; allowed=input_period+bank_period
 if span<0 or span>duration: raise DatasetNativeTimebaseError("CLIP_LOCAL_SPAN_EXCEEDS_DECLARED_DURATION")
 if tail<0 or tail>allowed: raise DatasetNativeTimebaseError("CLIP_LOCAL_TAIL_GAP_EXCEEDS_SAMPLING_BOUND")
 return {"record_id":r["id"],"video_id":r["video_id"],"refs":refs,"paths":paths,"anchor_source_frame_reference":anchor,"clip_timestamp_seconds":[str(v) for v in times],"frame_count":len(refs),"duplicate_logical_frame_count":len(refs)-len(set(refs)),"input_video_start_time":str(r["start"]),"input_video_end_time":str(r["end"]),"clip_duration_seconds":str(duration),"observed_timestamp_span_seconds":str(span),"tail_gap_seconds":str(tail),"input_sample_period_seconds":str(input_period),"frame_bank_period_seconds":"0.5","max_allowed_tail_gap_seconds":str(allowed),"legacy_absolute_clock_diagnostic":{"first_reference":refs[0],"last_reference":refs[-1],"bank_000001_required":False}}

def _load_dataset(path:Path)->list[Any]:
 try:v=strict_json_loads(path.read_bytes().decode(),error_code="SOURCE_DATASET")
 except (OSError,UnicodeDecodeError,TALSelectionError) as e: raise DatasetNativeTimebaseError("SOURCE_DATASET_UNREADABLE") from e
 if not isinstance(v,list): raise DatasetNativeTimebaseError("SOURCE_DATASET_INVALID")
 return v

def audit_dataset_native_timebase(*,dataset_json:Path,frame_root:Path,source_prefix:str,output_dir:Path,dataset_name:str="NurViD",data_source:str="NurViD",frame_bank_layout:str=FRAME_BANK_LAYOUT)->dict[str,Any]:
 if dataset_name!="NurViD" or data_source!="NurViD" or frame_bank_layout!=FRAME_BANK_LAYOUT: raise DatasetNativeTimebaseError("UNSUPPORTED_FRAME_BANK_LAYOUT")
 if output_dir.exists() and any(output_dir.iterdir()): raise DatasetNativeTimebaseError("OUTPUT_DIRECTORY_MUST_BE_EMPTY")
 mapper=FrameRootMapper.create(source_prefix,frame_root); rows=[]
 for i,row in enumerate(_load_dataset(dataset_json)):
  if isinstance(row,Mapping) and row.get("dataset_name")==dataset_name and row.get("data_source")==data_source:
   try: rows.append({"source_record_index":i,"status":"PASS",**_audit_record(row,mapper,require_files=True)})
   except DatasetNativeTimebaseError as e: rows.append({"source_record_index":i,"status":"UNRESOLVED","reason_code":str(e)})
 passed=[r for r in rows if r["status"]=="PASS"]; unresolved=[r for r in rows if r["status"]!="PASS"]; counts={k:sum(x.get("reason_code")==k for x in unresolved) for k in sorted({x["reason_code"] for x in unresolved})}
 output_dir.mkdir(parents=True,exist_ok=True); record_path=output_dir/"v2_nurvid_dataset_native_timebase_records.jsonl";_write(record_path,_jsonl(rows))
 report={"format":"relive-v2-nurvid-dataset-native-timebase-audit-v2","status":"PASS","timebase_status":"RESOLVED_DATASET_NATIVE_CLIP_LOCAL","timebase_provenance_tier":"DATASET_INTERNAL_DERIVED","timestamp_domain":"CLIP_LOCAL","timestamp_origin":"FIRST_PRESENTED_FRAME","frame_bank_layout":FRAME_BANK_LAYOUT,"frame_bank_rate_hz":2,"original_video_pts_verified":False,"external_original_video_opened":False,"cohort_records_checked":len(rows),"cohort_records_passed":len(passed),"cohort_records_unresolved":len(unresolved),"cohort_failure_reason_counts":counts,"public_metadata_projection_sha256":_sha(record_path),"gt_used":False,"assistant_or_gt_values_accessed":False,"model_calls_made":0,"backend_loaded":False,"cache_opened":False,"certificate_created":False,"new_verified_count":0,"certificate_status":"NOT_APPLICABLE"};report["audit_content_sha256"]=stable_hash(report);_write(output_dir/"v2_nurvid_dataset_native_timebase_audit.json",(canonical_json(report)+"\n").encode());return report

def export_dataset_native_timebase(*,dataset_json:Path,selection_manifest:Path,requirement_freeze_dir:Path,media_audit_dir:Path,frame_root:Path,source_prefix:str,output_dir:Path,frame_bank_layout:str=FRAME_BANK_LAYOUT,cohort_audit:dict[str,Any]|None=None)->dict[str,Any]:
 if frame_bank_layout!=FRAME_BANK_LAYOUT or output_dir.exists() and any(output_dir.iterdir()): raise DatasetNativeTimebaseError("UNSUPPORTED_FRAME_BANK_LAYOUT" if frame_bank_layout!=FRAME_BANK_LAYOUT else "OUTPUT_DIRECTORY_MUST_BE_EMPTY")
 req,_=_load_requirements(requirement_freeze_dir);sel,sel_sha=_load_selection(selection_manifest,req)
 from .video_index import _strict_rows
 projections=_strict_rows(media_audit_dir/"v2_public_media_projection.jsonl",code="MEDIA_PROJECTION")
 if len(sel["items"])!=1 or len(projections)!=1: raise DatasetNativeTimebaseError("TIMESTAMP_EXPORT_REQUIRES_SINGLE_PUBLIC_RECORD")
 chosen,projection=sel["items"][0],projections[0]
 if any(chosen[k]!=projection[k] for k in IDENTITY_FIELDS): raise DatasetNativeTimebaseError("SELECTED_IDENTITY_BINDING_MISMATCH")
 data=_load_dataset(dataset_json);idx=chosen["source_record_index"]
 if type(idx)is not int or idx<0 or idx>=len(data):raise DatasetNativeTimebaseError("SELECTED_IDENTITY_BINDING_MISMATCH")
 proof=_audit_record(data[idx],FrameRootMapper.create(source_prefix,frame_root),require_files=True)
 if not chosen["sample_id"].startswith("medvidu:"+proof["record_id"]+":") or [x["source_frame_reference"] for x in projection["frames"]]!=proof["paths"]:raise DatasetNativeTimebaseError("SELECTED_IDENTITY_BINDING_MISMATCH")
 frames=projection["frames"]; timestamps=[float(Decimal(v)) for v in proof["clip_timestamp_seconds"]]; rows=[{**{k:chosen[k] for k in IDENTITY_FIELDS},"frame_order":i,"source_frame_reference":frame["source_frame_reference"],"timestamp_seconds":ts,"timestamp_unit":"seconds"} for i,(frame,ts) in enumerate(zip(frames,timestamps))]
 output_dir.mkdir(parents=True); public=output_dir/"v2_nurvid_public_metadata_projection.jsonl";_write(public,_jsonl([{**{k:chosen[k] for k in IDENTITY_FIELDS},"record_id":proof["record_id"],"video_id":proof["video_id"],"source_frame_references":proof["refs"],"frame_bank_layout":FRAME_BANK_LAYOUT,"frame_bank_rate_hz":2,"anchor_source_frame_reference":proof["anchor_source_frame_reference"],"timestamp_domain":"CLIP_LOCAL"}]))
 manifest=output_dir/"public_per_frame_timestamps.jsonl";_write(manifest,_jsonl(rows));msha=_sha(manifest)
 prov={"format":TIMESTAMP_PROVENANCE_FORMAT,"timestamp_manifest_sha256":msha,"source_type":"VERSIONED_DATASET_TIMEBASE_ADAPTER","dataset_name":"NurViD",**{k:chosen[k] for k in IDENTITY_FIELDS},"time_origin":"CLIP_LOCAL_ZERO","timestamp_origin":"FIRST_PRESENTED_FRAME","timestamp_domain":"CLIP_LOCAL","timestamp_unit":"seconds","mapping_method":"(source_frame_reference-anchor_source_frame_reference)/2","adapter_version":ADAPTER_VERSION,"source_reference":str(public),"source_reference_sha256":_sha(public),"frame_count":len(rows),"gt_used":False,"assistant_or_gt_values_accessed":False,"model_calls_made":0,"backend_loaded":False,"cache_opened":False,"certificate_created":False,"new_verified_count":0,"certificate_status":"NOT_APPLICABLE","public_media_projection_sha256":projection["public_media_projection_sha256"],"frame_sha256":[x["frame_sha256"] for x in frames],"anchor_source_frame_reference":proof["anchor_source_frame_reference"],"frame_bank_layout":FRAME_BANK_LAYOUT,"frame_bank_rate_hz":2,"derivation_formula":"(reference-anchor_reference)/2","timestamp_resolution_seconds":0.5,"absolute_source_timebase_status":"UNRESOLVED_NOT_REQUIRED","original_video_pts_verified":False,"external_original_video_opened":False,"selected_source_record_index":idx,"selection_manifest_sha256":sel_sha,**{k:v for k,v in proof.items() if k in {"clip_duration_seconds","observed_timestamp_span_seconds","tail_gap_seconds","input_sample_period_seconds","frame_bank_period_seconds","max_allowed_tail_gap_seconds","duplicate_logical_frame_count"}}};side=output_dir/"public_per_frame_timestamps.provenance.json";_write(side,(canonical_json(prov)+"\n").encode())
 audit={"format":TIMESTAMP_AUDIT_FORMAT,"status":"PASS","timestamp_manifest_created":True,"timebase_status":"RESOLVED_DATASET_NATIVE_CLIP_LOCAL","timestamp_domain":"CLIP_LOCAL","timestamp_origin":"FIRST_PRESENTED_FRAME","selected_source_record_index":idx,"selected_record_status":"PASS","selected_record_reason_codes":[],"selected_record_manifest_created":True,"cohort_records_checked":cohort_audit.get("cohort_records_checked") if cohort_audit else None,"cohort_records_passed":cohort_audit.get("cohort_records_passed") if cohort_audit else None,"cohort_records_unresolved":cohort_audit.get("cohort_records_unresolved") if cohort_audit else None,"cohort_failure_reason_counts":cohort_audit.get("cohort_failure_reason_counts") if cohort_audit else None,"timestamp_manifest_sha256":msha,"timestamp_provenance_sha256":_sha(side),"frame_count":len(rows),"gt_used":False,"assistant_or_gt_values_accessed":False,"model_calls_made":0,"backend_loaded":False,"cache_opened":False,"certificate_created":False,"new_verified_count":0,"certificate_status":"NOT_APPLICABLE"};audit["audit_content_sha256"]=stable_hash(audit);_write(output_dir/"v2_tal_timestamp_source_audit.json",(canonical_json(audit)+"\n").encode());return audit
