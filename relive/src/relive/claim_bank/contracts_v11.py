"""v1.1: isolated planning/observation contracts; v1 stays in contracts.py."""
from __future__ import annotations

import json
import re

from relive.storage.artifacts import stable_hash
from relive.v2.task_selection import strict_json_loads
from . import contracts as v1

VERSION = "relive-claimgen-v1.1"
PARSER_VERSION = "relive-claimgen-parser-v1.1"
SAMPLER_VERSION = v1.SAMPLER_VERSION
QA_TYPES = v1.QA_TYPES
STATUSES = v1.STATUSES
STAGE1_PROMPT = v1.STAGE1_PROMPT + (
    " This stage is question-only evidence planning. Images are deliberately NOT supplied. "
    "Plan what a later observer must inspect, not what has already been observed. "
    "Missing images, unknown occurrence times, a missing queried box, and not knowing the answer "
    "are NOT by themselves reasons to be unable to plan. Write requirements as questions to resolve "
    "or observations to make, without assuming any candidate event occurred. "
    "For a question listing more than six alternatives, group shared evidence needs into at most six "
    "requirements rather than emitting one requirement per alternative. "
    "UNABLE_TO_PLAN is allowed only when the question itself does not permit a meaningful visual "
    "evidence plan; explain that question-level blocker in planning_reason. Do not invent requirements "
    "merely to avoid abstention."
)
STAGE2_PROMPT = v1.STAGE2_PROMPT

STAGE1_TASK_POLICIES = {
    "stg": "Plan identification of the question-named object/instance and its visible location at the question-specified query times. Time mapping is a later acquisition prerequisite; do not calculate coordinates, timestamps or assume object presence.",
    "tal": "Plan which visible object, interaction and event-boundary evidence would distinguish occurrence versus nonoccurrence of the queried action. Candidate actions are alternatives, never established events. Do not emit occurrence spans.",
    "next_action": "Plan present-state, completion and precondition evidence that could distinguish the question-listed possible next steps. No observation is available at planning time. Do not select or assert the next action.",
    "region_caption": "Plan identification of the publicly queried target, its visible properties and its activity/state over the query interval. The query-box reference remains an unresolved prerequisite for later acquisition, but target-independent planning can proceed. Do not guess the target or request hidden annotations.",
    "dense_captioning": "Plan discrimination among question-listed candidate event categories, temporal onset/offset evidence and visible object/interaction cues for captions. Group shared needs to respect the six-requirement limit. Do not report occurrences, times or captions.",
    "cvs_assessment": "Plan separate evidence requirements for the clinical criteria explicitly stated in the question and their anatomical/spatial prerequisites. Do not assume any criterion is met or emit scores. Lack of images is expected at planning time.",
    "skill_assessment": "Plan behavior and operation-quality evidence for the question-stated rating dimensions, grouping overlapping needs. Do not rate performance or assume a behavior occurred. Lack of images is expected at planning time.",
    "video_summary": "Plan major visible entities, interactions, events and changes needed to summarize the input. Describe information to inspect, not a summary or a claim of exhaustive observation.",
}


def task_policy(qa_type: str, stage: int) -> str:
    if stage == 2:
        return v1.task_policy(qa_type)
    key = next((k for k in STAGE1_TASK_POLICIES if qa_type.startswith(k)), qa_type)
    return STAGE1_TASK_POLICIES.get(key, "The question has an unsupported task; do not invent a plan.")


def output_contract(stage: int) -> str:
    common = (
        "\nReturn one complete JSON object only, without Markdown or extra keys. "
        "Types: OBJECT/ATTRIBUTE/RELATION/ACTION/HIGH_LEVEL. "
        "Evidence: SPATIAL/SPATIAL_MULTI_REGION/TEMPORAL/COMPOSITIONAL. "
        "ACTION requires TEMPORAL or COMPOSITIONAL; HIGH_LEVEL requires COMPOSITIONAL; "
        "RELATION requires SPATIAL_MULTI_REGION, TEMPORAL or COMPOSITIONAL."
    )
    if stage == 1:
        planned = {"planning_status": "PLANNED", "planning_reason": None, "requirements": [
            {"requirement_id": "R1", "description": "Determine which visible interaction would distinguish the question's alternatives.",
             "type": "RELATION", "evidence_type": "SPATIAL_MULTI_REGION", "importance": "REQUIRED"}]}
        unable = {"planning_status": "UNABLE_TO_PLAN", "planning_reason": "Explain why the question itself cannot define a visual evidence need.", "requirements": []}
        return (common + "\nRequired top-level keys: planning_status, planning_reason, requirements. "
                "The ONLY allowed planning_status values are PLANNED and UNABLE_TO_PLAN. "
                "With 1–6 requirements, use PLANNED and planning_reason=null. "
                "With zero requirements, use UNABLE_TO_PLAN and a nonempty question-level planning_reason. "
                "Each requirement has exactly requirement_id (unique R1–R6), description, type, evidence_type, "
                "importance (REQUIRED or OPTIONAL). These requirements are observation needs, not facts. "
                "Do not emit claims or a final answer.\nPlanned example: " + json.dumps(planned)
                + "\nUnplannable-question example: " + json.dumps(unable))
    if stage != 2:
        raise v1.ClaimBankError("STAGE_INVALID")
    claim = {"claim_id": "H1", "text": "One independently checkable proposition.", "type": "OBJECT",
             "evidence_type": "SPATIAL", "requirement_ids": ["R1"], "depends_on": [],
             "status": "UNVERIFIED", "generation_basis": "VISUAL_HYPOTHESIS", "question_quote": None,
             "frame_orders": [0], "target_alignment": "QUESTION_ALIGNED"}
    example = {"generation_status": "GENERATED", "claims": [claim], "coverage_notes": "Only supplied samples observed."}
    return (common + "\nRequired top-level keys: generation_status, claims, coverage_notes. "
            "generation_status: GENERATED/NO_VALID_CLAIMS/INSUFFICIENT_VISUAL_CONTEXT/UNRESOLVED_TARGET_REFERENCE. "
            "Return 0–6 claims with unique H1–H6 IDs and all statuses UNVERIFIED. "
            "QUESTION_HYPOTHESIS requires question_quote copied verbatim; VISUAL_HYPOTHESIS requires supplied frame_orders. "
            "target_alignment: QUESTION_ALIGNED/SCENE_ONLY/UNRESOLVED. Dependencies reference only unverified hypotheses. "
            "Use one main predicate, not chained actions, purposes, scores, coordinates or answers. "
            "If action evidence is missing, return INSUFFICIENT_VISUAL_CONTEXT, not merely object presence. "
            "No usable claims: an empty claims array and an appropriate non-GENERATED status.\nExample: " + json.dumps(example))


def protocol() -> dict:
    p = {**v1.protocol(), "protocol_version": VERSION, "parser_version": PARSER_VERSION,
         "stage1_prompt": STAGE1_PROMPT, "stage2_prompt": STAGE2_PROMPT,
         "task_policies": {"stage1": STAGE1_TASK_POLICIES, "stage2": v1.TASK_POLICIES},
         "output_contracts": {"stage1": output_contract(1), "stage2": output_contract(2)},
         "format_repair_version": "relive-claimgen-format-repair-v1.1",
         "format_repair_contract": "One schema-only repair; preserve any originally valid requirements list exactly; no semantic retry on abstention.",
         "planning_report": "Explicit planning_status, planning_reason and review flags; generation_status remains a separate batch field."}
    p.pop("protocol_sha256")
    p["protocol_sha256"] = stable_hash(p)
    return p


def parse(raw: str | None, stage: int, **kwargs):
    if stage == 2:
        return v1.parse(raw, stage, **kwargs)
    if stage != 1:
        return None, ["STAGE_INVALID"]
    try:
        value = strict_json_loads(raw or "", error_code="CLAIMGEN_JSON_INVALID")
    except (ValueError, TypeError):
        return None, ["STRICT_JSON_PARSE_FAILURE"]
    if not isinstance(value, dict):
        return None, ["STAGE1_TOP_LEVEL_FIELDS_INVALID"]
    errors = []
    if set(value) != {"planning_status", "planning_reason", "requirements"}:
        errors.append("STAGE1_TOP_LEVEL_FIELDS_INVALID")
    status = value.get("planning_status")
    if status not in ("PLANNED", "UNABLE_TO_PLAN"):
        errors.append("STAGE1_PLANNING_STATUS_INVALID")
    reason = value.get("planning_reason")
    if (status == "PLANNED" and reason is not None) or (status == "UNABLE_TO_PLAN" and not v1._text(reason)):
        errors.append("STAGE1_PLANNING_REASON_INVALID")
    if errors:
        return None, sorted(errors)
    parsed, errors = v1.parse(json.dumps({"planning_status": status, "requirements": value["requirements"]}), 1)
    return (value if parsed else None), errors


def diagnostics(raw: str | None, errors: list[str]) -> list[dict]:
    try:
        value = strict_json_loads(raw or "", error_code="CLAIMGEN_JSON_INVALID")
    except (ValueError, TypeError):
        return [{"path": "$", "expected": "one complete, strict JSON object", "errors": errors}]
    details = []
    if "STAGE1_PLANNING_STATUS_INVALID" in errors:
        details.append({"path": "$.planning_status", "expected": ["PLANNED", "UNABLE_TO_PLAN"], "actual": value.get("planning_status")})
    if "STAGE1_TOP_LEVEL_FIELDS_INVALID" in errors:
        details.append({"path": "$", "expected_keys": ["planning_status", "planning_reason", "requirements"],
                        "actual_keys": sorted(value) if isinstance(value, dict) else None, "actual_type": type(value).__name__})
    if "STAGE1_PLANNING_REASON_INVALID" in errors:
        details.append({"path": "$.planning_reason", "expected": "null for PLANNED; nonempty question-level reason for UNABLE_TO_PLAN", "actual": value.get("planning_reason")})
    return details or [{"path": "$", "errors": errors}]


def preserved_requirements(raw: str | None) -> list | None:
    """Protect valid propositions even when the surrounding status/keys are wrong."""
    try:
        value = strict_json_loads(raw or "", error_code="CLAIMGEN_JSON_INVALID")
    except (ValueError, TypeError):
        return None
    reqs = value.get("requirements") if isinstance(value, dict) else None
    if isinstance(reqs, list):
        parsed, _ = v1.parse(json.dumps({"planning_status": "PLANNED" if reqs else "UNABLE_TO_PLAN", "requirements": reqs}), 1)
        if parsed:
            return reqs
    return None


def planning_flags(parsed: dict) -> list[str]:
    if parsed["planning_status"] != "UNABLE_TO_PLAN":
        return []
    flags = ["PLANNING_ABSTENTION_REVIEW_REQUIRED"]
    if re.search(r"\b(?:no images?|no frames?|without (?:images?|frames?|video)|cannot (?:see|view)|(?:images?|frames?|video).{0,30}(?:not (?:supplied|provided|available)|unavailable|missing))\b", parsed["planning_reason"], re.I):
        flags.append("IMAGE_ABSENCE_NOT_VALID_PLANNING_BLOCKER")
    return flags
