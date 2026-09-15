#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from relive.v2.requirement_freeze import RequirementFreezeError,freeze_tal_requirements
from relive.v2.task_selection import TALSelectionError,load_selection,read_public_question_selector
from relive.v2.temporal_localization import TALAdapterError,load_event_ontology

def main()->int:
    parser=argparse.ArgumentParser();parser.add_argument("--public-question-selector",required=True,type=Path);parser.add_argument("--selection-manifest",required=True,type=Path);parser.add_argument("--event-ontology",required=True,type=Path);parser.add_argument("--output-dir",required=True,type=Path);args=parser.parse_args()
    try:
        _,digest=read_public_question_selector(args.public_question_selector);selection=load_selection(path=args.selection_manifest,selector_path=args.public_question_selector);ontology=load_event_ontology(args.event_ontology);selection_digest=hashlib.sha256(args.selection_manifest.read_bytes()).hexdigest();audit=freeze_tal_requirements(selection=selection,selector_sha256=digest,selection_manifest_sha256=selection_digest,ontology=ontology,output_dir=args.output_dir)
    except (TALSelectionError,TALAdapterError,RequirementFreezeError) as exc:print(f"ReliVE-v2 TAL requirement freeze error: {exc}");return 2
    print(json.dumps(audit,sort_keys=True));return 0 if audit["status"]!="NO_REQUIREMENTS_FROZEN" else 2
if __name__=="__main__":raise SystemExit(main())
