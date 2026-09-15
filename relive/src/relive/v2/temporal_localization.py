"""Deterministic, pre-video TAL requirement construction."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import hashlib,json,re
from pathlib import Path
from typing import Any
from .contracts import (AnswerCardinality, AnswerSchema, AnswerSchemaSpec, AnswerTimebase, RequirementSpec,
                        SpatialRequirement, TaskType, TemporalRequirement, make_requirement_spec)
from .task_selection import PublicTaskQuestion

ONTOLOGY_FORMAT="relive-v2-tal-event-ontology-v1"
ADAPTER_VERSION="relive-v2-tal-task-adapter-v1"
_HEX=re.compile(r"[0-9a-f]{64}\Z")
_FORBIDDEN=("frame","roi","bbox","mask","model_result","certificate","reference_answer","assistant_answer","temporal_gt","temporal_span","start_time","end_time","timestamp_gt","region","likelihood","support_margin","delta_drop")

class TALAdapterError(ValueError): pass
class RequirementBuildStatus(str,Enum): FROZEN="FROZEN"; UNRESOLVED="UNRESOLVED"; UNSUPPORTED="UNSUPPORTED"; INVALID_INPUT="INVALID_INPUT"

@dataclass(frozen=True)
class EventDefinition:
    event_id:str
    aliases:tuple[str,...]
    answer_schema:AnswerSchemaSpec
    required_evidence:tuple[str,...]
    temporal_requirement:TemporalRequirement
    spatial_requirement:SpatialRequirement
    ontology_source:str
    def __post_init__(self)->None:
        if not re.fullmatch(r"[a-z][a-z0-9_]*",self.event_id): raise TALAdapterError("EVENT_ID_INVALID")
        if not self.aliases or any(not isinstance(alias,str) or not alias.strip() for alias in self.aliases): raise TALAdapterError("EVENT_ALIASES_REQUIRED")
        if len({_normalize(alias) for alias in self.aliases})!=len(self.aliases): raise TALAdapterError("EVENT_ALIAS_DUPLICATE")
        if not self.required_evidence or len(set(self.required_evidence))!=len(self.required_evidence) or any(not re.fullmatch(r"[A-Z][A-Z0-9_]*",item) for item in self.required_evidence): raise TALAdapterError("REQUIRED_EVIDENCE_INVALID")
        if not isinstance(self.temporal_requirement,TemporalRequirement) or not isinstance(self.spatial_requirement,SpatialRequirement): raise TALAdapterError("EVENT_REQUIREMENT_INVALID")
        if not isinstance(self.ontology_source,str) or not self.ontology_source: raise TALAdapterError("ONTOLOGY_SOURCE_REQUIRED")
    def to_canonical_dict(self)->dict[str,Any]: return {"aliases":list(self.aliases),"answer_schema":self.answer_schema.to_canonical_dict(),"required_evidence":list(self.required_evidence),"temporal_requirement":self.temporal_requirement.value,"spatial_requirement":self.spatial_requirement.value,"ontology_source":self.ontology_source}

@dataclass(frozen=True)
class EventOntology:
    ontology_version:str
    ontology_sha256:str
    events:tuple[EventDefinition,...]
    def __post_init__(self)->None:
        if not isinstance(self.ontology_version,str) or not self.ontology_version: raise TALAdapterError("ONTOLOGY_VERSION_REQUIRED")
        if not _HEX.fullmatch(self.ontology_sha256): raise TALAdapterError("ONTOLOGY_SHA256_INVALID")
        if not self.events or len({event.event_id for event in self.events})!=len(self.events): raise TALAdapterError("EVENTS_INVALID")
        aliases=[_normalize(alias) for event in self.events for alias in event.aliases]
        if len(set(aliases))!=len(aliases): raise TALAdapterError("ONTOLOGY_ALIAS_NOT_UNIQUE")

@dataclass(frozen=True)
class RequirementBuildResult:
    status:RequirementBuildStatus
    requirement:RequirementSpec|None
    reason_code:str|None
    adapter_version:str
    question_sha256:str
    ontology_sha256:str
    def __post_init__(self)->None:
        if not isinstance(self.status,RequirementBuildStatus): raise TALAdapterError("BUILD_STATUS_INVALID")
        if self.status is RequirementBuildStatus.FROZEN and self.requirement is None: raise TALAdapterError("FROZEN_REQUIREMENT_REQUIRED")
        if self.status is not RequirementBuildStatus.FROZEN and self.requirement is not None: raise TALAdapterError("NONFROZEN_REQUIREMENT_FORBIDDEN")
        if not _HEX.fullmatch(self.question_sha256) or not _HEX.fullmatch(self.ontology_sha256): raise TALAdapterError("BUILD_HASH_INVALID")

def _normalize(value:str)->str:
    return re.sub(r"\s+"," ",value.strip().casefold())
def _sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def _strict_json_compatible_yaml(path:Path)->dict[str,Any]:
    try: value=json.loads(path.read_text(encoding="utf-8"))
    except (OSError,UnicodeDecodeError,json.JSONDecodeError) as exc: raise TALAdapterError("ONTOLOGY_MUST_BE_JSON_COMPATIBLE_SAFE_YAML") from exc
    if not isinstance(value,dict): raise TALAdapterError("ONTOLOGY_OBJECT_REQUIRED")
    return value
def _reject_forbidden(value:Any)->bool:
    if isinstance(value,dict): return any(any(token in str(key).casefold() for token in _FORBIDDEN) or _reject_forbidden(nested) for key,nested in value.items())
    if isinstance(value,list): return any(_reject_forbidden(item) for item in value)
    return False

def load_event_ontology(path:Path)->EventOntology:
    raw=_strict_json_compatible_yaml(path)
    if _reject_forbidden(raw): raise TALAdapterError("ONTOLOGY_PROHIBITED_FIELD")
    if set(raw)!={"format","ontology_version","events"} or raw["format"]!=ONTOLOGY_FORMAT or not isinstance(raw["events"],dict): raise TALAdapterError("ONTOLOGY_SCHEMA_INVALID")
    events=[]
    for event_id,row in raw["events"].items():
        allowed={"aliases","answer_schema","required_evidence","temporal_requirement","spatial_requirement","ontology_source"}
        if not isinstance(row,dict) or set(row)!=allowed: raise TALAdapterError("EVENT_SCHEMA_INVALID")
        answer=row["answer_schema"]
        if not isinstance(answer,dict) or set(answer)!={"type","timebase","cardinality"}: raise TALAdapterError("ANSWER_SCHEMA_INVALID")
        try: schema=AnswerSchemaSpec(AnswerSchema("EVENT_INTERVAL") if answer["type"]=="INTERVAL" else AnswerSchema(answer["type"]),AnswerTimebase(answer["timebase"]),AnswerCardinality(answer["cardinality"])); temporal=TemporalRequirement(row["temporal_requirement"]); spatial=SpatialRequirement(row["spatial_requirement"])
        except (ValueError,TypeError) as exc: raise TALAdapterError("ONTOLOGY_ENUM_INVALID") from exc
        events.append(EventDefinition(event_id,tuple(row["aliases"]) if isinstance(row["aliases"],list) else (),schema,tuple(row["required_evidence"]) if isinstance(row["required_evidence"],list) else (),temporal,spatial,row["ontology_source"]))
    return EventOntology(raw["ontology_version"],_sha(path),tuple(events))

def _event_phrase(question:str)->str|None:
    normalized=_normalize(question).rstrip("?.!")
    templates=(r"when does (?P<event>.+) happen",r"when is (?P<event>.+) performed",r"during what interval does (?P<event>.+) occur")
    for template in templates:
        match=re.fullmatch(template,normalized)
        if match:return match.group("event")
    return None

class TemporalLocalizationTaskAdapter:
    adapter_version=ADAPTER_VERSION
    def build_requirement(self, public_question:PublicTaskQuestion, ontology:EventOntology)->RequirementBuildResult:
        if public_question.source_qa_type.casefold()!="tal": return RequirementBuildResult(RequirementBuildStatus.UNSUPPORTED,None,"UNSUPPORTED_TASK_TYPE",self.adapter_version,public_question.question_sha256,ontology.ontology_sha256)
        phrase=_event_phrase(public_question.question_text)
        if phrase is None:return RequirementBuildResult(RequirementBuildStatus.UNRESOLVED,None,"UNSUPPORTED_TAL_TEMPLATE",self.adapter_version,public_question.question_sha256,ontology.ontology_sha256)
        matches=[]
        for event in ontology.events:
            if any(re.search(r"(?<!\w)"+re.escape(_normalize(alias))+r"(?!\w)",phrase) for alias in event.aliases): matches.append(event)
        if not matches:return RequirementBuildResult(RequirementBuildStatus.UNRESOLVED,None,"UNRESOLVED_EVENT",self.adapter_version,public_question.question_sha256,ontology.ontology_sha256)
        if len(matches)!=1:return RequirementBuildResult(RequirementBuildStatus.UNRESOLVED,None,"AMBIGUOUS_EVENT_MATCH",self.adapter_version,public_question.question_sha256,ontology.ontology_sha256)
        event=matches[0]
        provenance={"public_source_identity":public_question.identity_dict(),"ontology_version":ontology.ontology_version,"ontology_sha256":ontology.ontology_sha256,"adapter_version":self.adapter_version}
        requirement=make_requirement_spec(task=TaskType.TEMPORAL_LOCALIZATION,question_text=public_question.question_text,target_event=event.event_id,answer_schema=event.answer_schema,required_evidence=event.required_evidence,temporal_requirement=event.temporal_requirement,spatial_requirement=event.spatial_requirement,planner_version=self.adapter_version,provenance=provenance)
        return RequirementBuildResult(RequirementBuildStatus.FROZEN,requirement,None,self.adapter_version,public_question.question_sha256,ontology.ontology_sha256)
