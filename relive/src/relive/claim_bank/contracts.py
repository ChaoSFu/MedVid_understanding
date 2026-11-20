"""Frozen generation surfaces and conservative, GT-free quality checks.

Checks establish structural validity and flag lexical risks; they cannot prove
clinical correctness, atomic semantics, or visual grounding.
"""
from __future__ import annotations

import json
import re
from typing import Any

from relive.storage.artifacts import stable_hash
from relive.v2.task_selection import strict_json_loads

VERSION = "relive-claimgen-v1"
PARSER_VERSION = "relive-claimgen-parser-v1"
SAMPLER_VERSION = "uniform-coverage-adjacent-groups-v1"
TYPES = {"OBJECT", "ATTRIBUTE", "RELATION", "ACTION", "HIGH_LEVEL"}
EVIDENCE = {"SPATIAL", "SPATIAL_MULTI_REGION", "TEMPORAL", "COMPOSITIONAL"}
STATUSES = {"GENERATED", "NO_VALID_CLAIMS", "INSUFFICIENT_VISUAL_CONTEXT",
            "UNRESOLVED_TARGET_REFERENCE", "MISSING_FRAMES", "MODEL_OR_SCHEMA_FAILURE"}
QA_TYPES = ("tal", "stg", "next_action", "region_caption_gpt", "region_caption_gemini",
            "dense_captioning_gpt", "dense_captioning_gemini", "cvs_assessment",
            "skill_assessment", "video_summary_gpt", "video_summary_gemini")

STAGE1_PROMPT = (
    "Analyze the medical video question and identify the minimal visual evidence requirements needed to answer it. "
    "Do not answer the question or assume any event occurred. Decompose the task into independently checkable "
    "requirements about objects, attributes, relations, actions, or higher-level clinical conditions. Classify "
    "each requirement by the visual evidence it needs. Focus on information that could change the final answer. "
    "Avoid redundant requirements and unsupported assumptions. Return structured JSON only."
)
STAGE2_PROMPT = (
    "Given the medical video question, its evidence requirements, and the supplied video frames, generate up to "
    "six question-relevant, independently verifiable candidate claims. Each claim must express one clear proposition "
    "about a visible object, attribute, relation, action, or clinical condition. Use the frames to propose plausible "
    "hypotheses, but do not assume any claim is true. Classify the claim type, required evidence type, and dependencies "
    "on other claims. Preserve information essential for answering the question; do not reduce actions or interactions "
    "to object presence alone. Do not invent unobserved details, use ground-truth information, or produce the final "
    "answer. Prefer claims whose verification could distinguish competing answers. Return structured JSON with all "
    "claim statuses set to UNVERIFIED."
)
TASK_POLICIES = {
    "stg": "Preserve question-specified object/instance and query times. Propose location hypotheses without coordinates. Query seconds must not be mapped to frames without a proven timebase.",
    "tal": "Preserve the queried action and relevant object/relation conditions. Listed actions are candidates, not observed occurrences. Never emit occurrence spans.",
    "next_action": "Use the present state and question-listed alternatives. Preserve distinctions between candidate next steps. A future action may only be a QUESTION_HYPOTHESIS explicitly conditional/prospective, never a reported observation.",
    "region_caption": "The queried box is unavailable. Never infer its target from dataset labels. Return UNRESOLVED_TARGET_REFERENCE. Optional scene claims must have target_alignment=UNRESOLVED, never claim correspondence to the boxed object.",
    "dense_captioning": "Retain discriminative events, relations and temporal changes. Candidate action names are alternatives. Sparse frames do not establish complete event coverage.",
    "cvs_assessment": "Plan question-stated anatomical criteria and spatial relations separately. Do not reduce criteria to object presence. Do not emit scores or a final assessment.",
    "skill_assessment": "Retain question-stated behavior and operation-quality dimensions as temporal/compositional requirements. Do not emit ratings or a final assessment.",
    "video_summary": "Propose a few question-relevant observations with sampled coverage only. Never claim exhaustive coverage or produce a final summary answer.",
}


class ClaimBankError(ValueError):
    pass


def task_policy(qa_type: str) -> str:
    key = next((k for k in TASK_POLICIES if qa_type.startswith(k)), qa_type)
    return TASK_POLICIES.get(key, "Unsupported task; return no claims.")


def protocol() -> dict[str, Any]:
    value = {
        "protocol_version": VERSION, "parser_version": PARSER_VERSION,
        "sampler_version": SAMPLER_VERSION, "stage1_prompt": STAGE1_PROMPT,
        "stage2_prompt": STAGE2_PROMPT, "task_policies": TASK_POLICIES,
        "max_requirements": 6, "max_claims": 6, "max_format_repairs": 1,
        "supplemental_observation": {"enabled": False, "max_rounds": 0,
                                     "interface": "extra frame-group observation requires a new protocol version"},
        "quality_check_scope": "CLOSED_SCHEMA_PLUS_CONSERVATIVE_LEXICAL_CHECKS_NOT_SEMANTIC_PROOF",
        "claim_truth_status": "UNVERIFIED", "question_candidates_are_facts": False,
        "timebase_policy": "ORDER_ONLY_UNLESS_EXPLICIT_PUBLIC_TIMESTAMPS_AVAILABLE",
        "output_contracts": {"stage1": output_contract(1), "stage2": output_contract(2)},
    }
    value["protocol_sha256"] = stable_hash(value)
    return value


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip()) and len(value) <= 3000


def _closed(obj: Any, keys: set[str]) -> bool:
    return isinstance(obj, dict) and set(obj) == keys


def _typed(obj: dict) -> bool:
    return obj.get("type") in TYPES and obj.get("evidence_type") in EVIDENCE


def _compatible(obj: dict) -> bool:
    return not ((obj["type"] == "ACTION" and obj["evidence_type"] not in {"TEMPORAL", "COMPOSITIONAL"})
                or (obj["type"] == "HIGH_LEVEL" and obj["evidence_type"] != "COMPOSITIONAL")
                or (obj["type"] == "RELATION" and obj["evidence_type"] not in {"SPATIAL_MULTI_REGION", "TEMPORAL", "COMPOSITIONAL"}))


def _parse(raw: str | None, stage: int, *, requirements: list[dict] | None = None,
          question: str = "", frame_orders: list[int] | None = None) -> tuple[dict | None, list[str]]:
    try:
        obj = strict_json_loads(raw or "", error_code="CLAIMGEN_JSON_INVALID")
    except (ValueError, TypeError):
        return None, ["STRICT_JSON_PARSE_FAILURE"]
    errors = []
    if stage == 1:
        if not _closed(obj, {"planning_status", "requirements"}) or obj.get("planning_status") not in {"PLANNED", "UNABLE_TO_PLAN"}:
            return None, ["STAGE1_SCHEMA"]
        items = obj["requirements"]
        if not isinstance(items, list) or len(items) > 6:
            return None, ["REQUIREMENT_COUNT"]
        if bool(items) != (obj["planning_status"] == "PLANNED"):
            errors.append("PLANNING_STATUS_COUNT_MISMATCH")
        keys, id_key, text_key = {"requirement_id", "description", "type", "evidence_type", "importance"}, "requirement_id", "description"
    else:
        if not _closed(obj, {"generation_status", "claims", "coverage_notes"}) or obj.get("generation_status") not in {"GENERATED", "NO_VALID_CLAIMS", "INSUFFICIENT_VISUAL_CONTEXT", "UNRESOLVED_TARGET_REFERENCE"}:
            return None, ["STAGE2_SCHEMA"]
        items = obj["claims"]
        if not isinstance(items, list) or len(items) > 6 or not isinstance(obj["coverage_notes"], str):
            return None, ["CLAIM_COUNT_OR_COVERAGE"]
        if (not items and obj["generation_status"] == "GENERATED") or (items and obj["generation_status"] in {"NO_VALID_CLAIMS", "INSUFFICIENT_VISUAL_CONTEXT"}):
            errors.append("CLAIM_STATUS_COUNT_MISMATCH")
        keys = {"claim_id", "text", "type", "evidence_type", "requirement_ids", "depends_on", "status", "generation_basis", "question_quote", "frame_orders", "target_alignment"}
        id_key, text_key = "claim_id", "text"
    ids, texts = set(), set()
    for item in items:
        if not _closed(item, keys) or not _typed(item) or not _text(item.get(text_key)):
            return None, ["ITEM_SCHEMA"]
        item_id = item[id_key]
        if not isinstance(item_id, str) or not re.fullmatch(r"R[1-6]" if stage == 1 else r"H[1-6]", item_id) or item_id in ids:
            errors.append("DUPLICATE_OR_INVALID_ID")
        ids.add(item_id)
        normalized = " ".join(re.findall(r"\w+", item[text_key].casefold()))
        if normalized in texts:
            errors.append("DUPLICATE_PROPOSITION")
        texts.add(normalized)
        if not _compatible(item):
            errors.append("TYPE_EVIDENCE_MISMATCH")
        if stage == 1:
            if item["importance"] not in {"REQUIRED", "OPTIONAL"}:
                errors.append("IMPORTANCE_INVALID")
            continue
        reqs = {r["requirement_id"]: r for r in requirements or []}
        for field in ("requirement_ids", "depends_on", "frame_orders"):
            if not isinstance(item[field], list) or any(type(v) is not (int if field == "frame_orders" else str) for v in item[field]):
                return None, ["REFERENCE_SCHEMA"]
            if len(set(item[field])) != len(item[field]):
                errors.append("DUPLICATE_REFERENCE")
        if not item["requirement_ids"] or not set(item["requirement_ids"]) <= set(reqs):
            errors.append("REQUIREMENT_REFERENCE_INVALID")
        if item["status"] != "UNVERIFIED":
            errors.append("CLAIM_MUST_BE_UNVERIFIED")
        if item["generation_basis"] not in {"QUESTION_HYPOTHESIS", "VISUAL_HYPOTHESIS"}:
            errors.append("GENERATION_BASIS_INVALID")
        if item["target_alignment"] not in {"QUESTION_ALIGNED", "SCENE_ONLY", "UNRESOLVED"}:
            errors.append("TARGET_ALIGNMENT_INVALID")
        if not set(item["frame_orders"]) <= set(frame_orders or []):
            errors.append("FRAME_REFERENCE_INVALID")
        if item["generation_basis"] == "VISUAL_HYPOTHESIS" and not item["frame_orders"]:
            errors.append("VISUAL_HYPOTHESIS_REQUIRES_SUPPLIED_FRAME")
        quote = item["question_quote"]
        if quote is not None and (not _text(quote) or quote not in question):
            errors.append("QUESTION_QUOTE_INVALID")
        if item["generation_basis"] == "QUESTION_HYPOTHESIS" and quote is None:
            errors.append("QUESTION_HYPOTHESIS_REQUIRES_VERBATIM_QUOTE")
        text = item["text"]
        if re.search(r"\b(?:then|in order to|so that|because|therefore)\b|[;\n]", text, re.I):
            errors.append("MULTI_ACTION_OR_PURPOSE_RISK")
        if len(re.findall(r"[.!?](?:\s|$)", text)) > 1:
            errors.append("MULTIPLE_PROPOSITIONS_RISK")
        if item["type"] in {"OBJECT", "ATTRIBUTE"} and re.search(r"\b(?:dissects?|inserts?|cuts?|retracts?|removes?|moves?)\b", text, re.I):
            errors.append("ACTION_MISCLASSIFIED")
    if stage == 2:
        graph = {x["claim_id"]: x["depends_on"] for x in items}
        def visit(node: str, stack: set[str]) -> bool:
            return node in stack or any(d not in graph or visit(d, stack | {node}) for d in graph[node])
        if any(visit(n, set()) for n in graph):
            errors.append("DEPENDENCY_INVALID_OR_CYCLIC")
        # Dependencies always reference hypotheses, never established premises.
    return (None if errors else obj), sorted(set(errors))


def parse(raw: str | None, stage: int, **kwargs) -> tuple[dict | None, list[str]]:
    try:
        return _parse(raw, stage, **kwargs)
    except (KeyError, TypeError, ValueError, RecursionError):
        return None, ["ITEM_SCHEMA"]


def output_contract(stage: int) -> str:
    requirement = {"requirement_id": "R1", "description": "What visible evidence is needed?", "type": "OBJECT", "evidence_type": "SPATIAL", "importance": "REQUIRED"}
    claim = {"claim_id": "H1", "text": "One independently checkable proposition.", "type": "OBJECT", "evidence_type": "SPATIAL", "requirement_ids": ["R1"], "depends_on": [], "status": "UNVERIFIED", "generation_basis": "VISUAL_HYPOTHESIS", "question_quote": None, "frame_orders": [0], "target_alignment": "QUESTION_ALIGNED"}
    example = ({"planning_status": "PLANNED", "requirements": [requirement]} if stage == 1 else {"generation_status": "GENERATED", "claims": [claim], "coverage_notes": "Only supplied samples observed."})
    return ("\nClosed output contract (no extra keys, no Markdown): " + json.dumps(example)
            + "\nTypes: OBJECT/ATTRIBUTE/RELATION/ACTION/HIGH_LEVEL. Evidence: SPATIAL/SPATIAL_MULTI_REGION/TEMPORAL/COMPOSITIONAL. "
            "ACTION needs TEMPORAL or COMPOSITIONAL; HIGH_LEVEL needs COMPOSITIONAL; RELATION needs SPATIAL_MULTI_REGION, TEMPORAL or COMPOSITIONAL. "
            "Return at most six items. Requirement importance REQUIRED/OPTIONAL. Stage1 with zero requirements: UNABLE_TO_PLAN. "
            "Stage2 statuses GENERATED/NO_VALID_CLAIMS/INSUFFICIENT_VISUAL_CONTEXT/UNRESOLVED_TARGET_REFERENCE. "
            "QUESTION_HYPOTHESIS needs question_quote copied verbatim from the question; VISUAL_HYPOTHESIS needs supplied frame_orders. "
            "target_alignment: QUESTION_ALIGNED/SCENE_ONLY/UNRESOLVED. Dependencies refer only to UNVERIFIED hypotheses, including HIGH_LEVEL. "
            "Use one primary predicate; do not chain actions, emit scores, coordinates, answers, or assume candidate alternatives occurred. "
            "If needed action evidence is unavailable, return INSUFFICIENT_VISUAL_CONTEXT; do not replace it by object presence.")
