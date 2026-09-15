"""Closed, pre-video TAL selection over a GT-isolated public selector."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

from relive.storage.artifacts import canonical_json, stable_hash

SELECTION_FORMAT = "relive-v2-tal-requirement-selection-v1"
SELECTION_STATUS = "FROZEN_PRE_VIDEO_ACCESS"
SELECTOR_BASE = frozenset({"source_record_index", "sample_id", "public_record_sha256", "question_sha256", "qa_type", "question"})
SELECTOR_OPTIONAL = frozenset({"dataset_name", "frame_count", "first_verified_frame_path", "last_verified_frame_path"})
_FORBIDDEN = ("answer", "reference_answer", "assistant_answer", "temporal_gt", "temporal_span", "start_time", "end_time", "timestamp_gt", "bbox", "mask", "region", "roi", "struc_info", "rc_info", "certificate", "semantic_status", "model_result", "likelihood", "support_margin", "delta_drop")
_HEX = re.compile(r"[0-9a-f]{64}\Z")

class TALSelectionError(ValueError):
    pass

@dataclass(frozen=True)
class PublicTaskQuestion:
    source_record_index: int
    sample_id: str
    public_record_sha256: str
    question_text: str
    question_sha256: str
    source_qa_type: str
    dataset_name: str | None

    def __post_init__(self) -> None:
        if type(self.source_record_index) is not int or self.source_record_index < 0:
            raise TALSelectionError("SOURCE_RECORD_INDEX_INVALID")
        for value, code in ((self.sample_id, "SAMPLE_ID_REQUIRED"), (self.question_text, "QUESTION_REQUIRED"), (self.source_qa_type, "QA_TYPE_REQUIRED")):
            if not isinstance(value, str) or not value.strip(): raise TALSelectionError(code)
        if not _HEX.fullmatch(self.public_record_sha256): raise TALSelectionError("PUBLIC_RECORD_SHA256_INVALID")
        if self.question_sha256 != hashlib.sha256(self.question_text.encode("utf-8")).hexdigest(): raise TALSelectionError("QUESTION_SHA256_MISMATCH")
        if self.dataset_name is not None and (not isinstance(self.dataset_name, str) or not self.dataset_name.strip()): raise TALSelectionError("DATASET_NAME_INVALID")

    def identity_dict(self) -> dict[str, Any]:
        return {"source_record_index": self.source_record_index, "sample_id": self.sample_id,
                "public_record_sha256": self.public_record_sha256, "question_sha256": self.question_sha256}

@dataclass(frozen=True)
class TALRequirementSelection:
    source_selector_sha256: str
    selection_basis: str
    items: tuple[PublicTaskQuestion, ...]

    def __post_init__(self) -> None:
        if not _HEX.fullmatch(self.source_selector_sha256): raise TALSelectionError("SOURCE_SELECTOR_SHA256_INVALID")
        if self.selection_basis not in {"PUBLIC_SELECTOR_ORDER", "EXPLICIT_PUBLIC_IDENTITY"}: raise TALSelectionError("SELECTION_BASIS_INVALID")
        if not self.items or len({item.source_record_index for item in self.items}) != len(self.items): raise TALSelectionError("SELECTION_ITEMS_INVALID")

    def to_canonical_dict(self) -> dict[str, Any]:
        payload={"format": SELECTION_FORMAT,"selection_status": SELECTION_STATUS,"source_selector_sha256": self.source_selector_sha256,
                 "selection_basis": self.selection_basis,"gt_used": False,"items": [item.identity_dict() for item in self.items]}
        payload["selection_content_sha256"]=stable_hash(payload)
        return payload


def _strict_json(line: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        row={}
        for key,value in items:
            if key in row: raise TALSelectionError("SELECTOR_DUPLICATE_KEY")
            row[key]=value
        return row
    try: row=json.loads(line, object_pairs_hook=pairs, parse_constant=lambda _: (_ for _ in ()).throw(TALSelectionError("NONFINITE_JSON_FORBIDDEN")))
    except (json.JSONDecodeError, TypeError) as exc: raise TALSelectionError("INVALID_SELECTOR_JSON") from exc
    if not isinstance(row,dict): raise TALSelectionError("SELECTOR_OBJECT_REQUIRED")
    return row

def _contains_forbidden(value: Any) -> bool:
    if isinstance(value,dict):
        return any(any(token in str(key).casefold() for token in _FORBIDDEN) or _contains_forbidden(nested) for key,nested in value.items())
    if isinstance(value,list): return any(_contains_forbidden(item) for item in value)
    return False

def read_public_question_selector(path: Path) -> tuple[tuple[PublicTaskQuestion,...],str]:
    try: raw=path.read_bytes()
    except OSError as exc: raise TALSelectionError("PUBLIC_SELECTOR_UNREADABLE") from exc
    rows=tuple(_strict_json(line) for line in raw.decode("utf-8").splitlines() if line.strip())
    if not rows: raise TALSelectionError("PUBLIC_SELECTOR_EMPTY")
    result=[]
    for row in rows:
        if _contains_forbidden(row): raise TALSelectionError("SELECTOR_PROHIBITED_FIELD")
        if not SELECTOR_BASE.issubset(row) or not set(row).issubset(SELECTOR_BASE | SELECTOR_OPTIONAL): raise TALSelectionError("SELECTOR_CLOSED_SCHEMA_VIOLATION")
        for key in ("frame_count","first_verified_frame_path","last_verified_frame_path"):
            if key in row and row[key] is None: raise TALSelectionError("SELECTOR_OPTIONAL_FIELD_INVALID")
        if "frame_count" in row and (type(row["frame_count"]) is not int or row["frame_count"] < 0): raise TALSelectionError("FRAME_COUNT_INVALID")
        if any(key in row and not isinstance(row[key],str) for key in ("first_verified_frame_path","last_verified_frame_path","dataset_name")): raise TALSelectionError("SELECTOR_OPTIONAL_FIELD_INVALID")
        result.append(PublicTaskQuestion(row["source_record_index"],row["sample_id"],row["public_record_sha256"],row["question"],row["question_sha256"],row["qa_type"],row.get("dataset_name")))
    if [item.source_record_index for item in result] != sorted(item.source_record_index for item in result) or len({item.source_record_index for item in result}) != len(result): raise TALSelectionError("SELECTOR_ORDER_OR_UNIQUENESS_INVALID")
    return tuple(result),hashlib.sha256(raw).hexdigest()

def freeze_selector_order(*, selector_path: Path, max_samples: int) -> TALRequirementSelection:
    if type(max_samples) is not int or max_samples < 1: raise TALSelectionError("MAX_SAMPLES_POSITIVE_INTEGER_REQUIRED")
    rows,digest=read_public_question_selector(selector_path)
    selected=tuple(row for row in rows if row.source_qa_type.casefold()=="tal")[:max_samples]
    if len(selected)<max_samples: raise TALSelectionError("INSUFFICIENT_TAL_ROWS")
    return TALRequirementSelection(digest,"PUBLIC_SELECTOR_ORDER",selected)

def write_selection(selection: TALRequirementSelection,path: Path) -> None:
    if path.exists(): raise TALSelectionError("SELECTION_OUTPUT_MUST_NOT_EXIST")
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(canonical_json(selection.to_canonical_dict())+"\n",encoding="utf-8")

def load_selection(*, path: Path, selector_path: Path) -> TALRequirementSelection:
    try: row=_strict_json(path.read_text(encoding="utf-8"))
    except OSError as exc: raise TALSelectionError("SELECTION_MANIFEST_UNREADABLE") from exc
    allowed={"format","selection_status","source_selector_sha256","selection_basis","gt_used","items","selection_content_sha256"}
    if set(row)!=allowed or row["format"]!=SELECTION_FORMAT or row["selection_status"]!=SELECTION_STATUS or row["gt_used"] is not False: raise TALSelectionError("SELECTION_MANIFEST_SCHEMA_INVALID")
    content={key:value for key,value in row.items() if key!="selection_content_sha256"}
    if row["selection_content_sha256"]!=stable_hash(content): raise TALSelectionError("SELECTION_MANIFEST_HASH_MISMATCH")
    source,digest=read_public_question_selector(selector_path)
    if row["source_selector_sha256"]!=digest: raise TALSelectionError("SELECTION_SOURCE_HASH_MISMATCH")
    by_identity={(item.source_record_index,item.sample_id,item.public_record_sha256,item.question_sha256):item for item in source}
    items=[]
    if not isinstance(row["items"],list) or not row["items"]: raise TALSelectionError("SELECTION_ITEMS_INVALID")
    for identity in row["items"]:
        if not isinstance(identity,dict) or set(identity)!={"source_record_index","sample_id","public_record_sha256","question_sha256"}: raise TALSelectionError("SELECTION_ITEM_SCHEMA_INVALID")
        key=(identity["source_record_index"],identity["sample_id"],identity["public_record_sha256"],identity["question_sha256"])
        if key not in by_identity: raise TALSelectionError("SELECTION_IDENTITY_NOT_IN_SELECTOR")
        items.append(by_identity[key])
    return TALRequirementSelection(row["source_selector_sha256"],row["selection_basis"],tuple(items))
