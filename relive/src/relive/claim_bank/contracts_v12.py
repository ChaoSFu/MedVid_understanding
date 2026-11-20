"""Stage 2 v1.2: model image indices, explicit temporal role, layered audit.

Consumes frozen Stage-1 v1.1 requirements. No old completions are normalized,
repaired or imported. Binding happens only after a NEW completion passes.
"""
from __future__ import annotations

from copy import deepcopy
import json
import re

from relive.storage.artifacts import stable_hash
from relive.v2.task_selection import strict_json_loads
from . import contracts as v1
from . import contracts_v11 as planning

VERSION = "relive-claimgen-stage2-v1.2"
PARSER_VERSION = "relive-claimgen-stage2-parser-v1.2"
BINDING_VERSION = "claimgen-image-index-binding-v1"
QA_TYPES, STATUSES, SAMPLER_VERSION = v1.QA_TYPES, v1.STATUSES, v1.SAMPLER_VERSION
ROLES = {"OBSERVED_CONTEXT", "FUTURE_HYPOTHESIS", "QUESTION_TIME_QUERY"}
CLAIM_KEYS = {"claim_id", "text", "type", "evidence_type", "requirement_ids", "depends_on",
              "status", "generation_basis", "question_quote", "image_indices", "temporal_role", "target_alignment"}
STAGE2_PROMPT = v1.STAGE2_PROMPT + (
    " This is candidate generation, not verification. Return fewer claims when appropriate; six is a maximum, "
    "not a target. Every claim needs at least one genuinely relevant supplied requirement_id. Do not add "
    "unrelated scene descriptions to fill the list or invent events to satisfy a requirement. Requirements "
    "are observation needs, not established facts; do not inherit unsupported assumptions from their wording. "
    "OBSERVED_CONTEXT covers proposed observations in supplied images, including current actions in next-action tasks. "
    "FUTURE_HYPOTHESIS is an explicitly prospective question-derived alternative, never a visually observed future event. "
    "QUESTION_TIME_QUERY asserts correspondence to a question-specified time and requires a verified time mapping. "
    "If that mapping is unavailable, propose only sampled observations without claiming query-time alignment, "
    "or return INSUFFICIENT_VISUAL_CONTEXT. Each text should have one main observable predicate; avoid "
    "chains such as cutting followed by suturing, statements about how boundaries are defined, verification "
    "language such as 'is confirmed', and purpose/causal inference. Never infer a boxed target without a supplied "
    "public target reference. Distinguish candidate alternatives without treating them as established premises."
)


def task_policy(qa_type: str, stage: int) -> str:
    if stage != 2:
        raise v1.ClaimBankError("V12_IS_STAGE2_ONLY_USE_FROZEN_STAGE1_V11")
    if qa_type == "next_action":
        return ("Current visible state, relations and actions may be VISUAL_HYPOTHESIS with OBSERVED_CONTEXT. "
                "Any prediction, regardless of claim type, must use FUTURE_HYPOTHESIS and QUESTION_HYPOTHESIS "
                "with a verbatim question quote and explicit tentative wording. Do not output a final next-step answer.")
    if qa_type.startswith("region_caption"):
        return ("No public query box is supplied. The generation_status MUST be UNRESOLVED_TARGET_REFERENCE. "
                "Zero claims is valid. Optional requirement-relevant scene hypotheses must have target_alignment=UNRESOLVED; "
                "do not say 'the object inside the box' or assert correspondence to the queried box/ROI.")
    return v1.task_policy(qa_type)


def output_contract(stage: int) -> str:
    if stage != 2:
        raise v1.ClaimBankError("V12_IS_STAGE2_ONLY_USE_FROZEN_STAGE1_V11")
    example = {"generation_status": "GENERATED", "coverage_notes": "Only supplied images observed.", "claims": [
        {"claim_id": "H1", "text": "One question-relevant observable proposition.", "type": "OBJECT",
         "evidence_type": "SPATIAL", "requirement_ids": ["R1"], "depends_on": [], "status": "UNVERIFIED",
         "generation_basis": "VISUAL_HYPOTHESIS", "question_quote": None, "image_indices": [0],
         "temporal_role": "OBSERVED_CONTEXT", "target_alignment": "QUESTION_ALIGNED"}]}
    return ("\nClosed JSON contract; no extra keys, Markdown, coordinates, scores or answers. Example:\n"
            + json.dumps(example) + "\nExactly 0–6 claims with unique H1–H6 IDs; all status=UNVERIFIED. "
            "Top-level generation_status: GENERATED/NO_VALID_CLAIMS/INSUFFICIENT_VISUAL_CONTEXT/UNRESOLVED_TARGET_REFERENCE. "
            "Use image_indices only: zero-based indices into the actual supplied image list, from the provided "
            "available_image_indices. Never output frame_orders, filename numbers or source-frame identifiers. "
            "VISUAL_HYPOTHESIS: question_quote MUST be JSON null and image_indices nonempty. "
            "QUESTION_HYPOTHESIS: question_quote MUST be an exact substring of the ORIGINAL QUESTION, never a requirement description. "
            "requirement_ids MUST be nonempty and use supplied IDs; SCENE_ONLY does not waive this rule. "
            "depends_on may reference only other UNVERIFIED claim IDs without cycles; it does not establish their truth. "
            "temporal_role: OBSERVED_CONTEXT/FUTURE_HYPOTHESIS/QUESTION_TIME_QUERY. "
            "target_alignment: QUESTION_ALIGNED/SCENE_ONLY/UNRESOLVED. "
            "Types: OBJECT/ATTRIBUTE/RELATION/ACTION/HIGH_LEVEL. Evidence: SPATIAL/SPATIAL_MULTI_REGION/TEMPORAL/COMPOSITIONAL. "
            "ACTION requires TEMPORAL or COMPOSITIONAL; HIGH_LEVEL requires COMPOSITIONAL; "
            "RELATION requires SPATIAL_MULTI_REGION, TEMPORAL or COMPOSITIONAL. "
            "Empty claims require a non-GENERATED status. Region-caption with no public query box always requires "
            "UNRESOLVED_TARGET_REFERENCE, even when scene hypotheses are returned.")


def protocol() -> dict:
    p = {"protocol_version": VERSION, "stage": 2, "parser_version": PARSER_VERSION,
         "binding_version": BINDING_VERSION, "sampler_version": SAMPLER_VERSION,
         "accepted_stage1_protocol_sha256": [planning.protocol()["protocol_sha256"]],
         "stage2_prompt": STAGE2_PROMPT, "output_contract": output_contract(2),
         "task_policies": {qa: task_policy(qa, 2) for qa in QA_TYPES},
         "max_claims": 6, "max_format_repairs": 1,
         "format_repair": "Syntax/shape only; preserve identifiable claim propositions. No binding/task/quality retries.",
         "diagnostic_layers": ["schema", "binding", "task", "semantic_risk"],
         "truth_status": "UNVERIFIED", "supplemental_observation_enabled": False,
         "semantic_limit": "Lexical risk flags are not proof of atomicity, relevance or clinical correctness."}
    p["protocol_sha256"] = stable_hash(p)
    return p


def _sequence(value, scalar):
    return isinstance(value, list) and all(type(v) is scalar for v in value)


def evaluate(raw: str | None, row: dict, requirements: list[dict], sampling: dict):
    """Collect all evaluable errors; an earlier binding error never hides task checks."""
    audit = {layer: {"status": "PASS", "issues": []} for layer in ("schema", "binding", "task", "semantic_risk")}
    def issue(layer, code, path, expected=None, actual=None):
        audit[layer]["issues"].append({"code": code, "path": path, "expected": expected, "actual": actual})
    try:
        obj = strict_json_loads(raw or "", error_code="CLAIMGEN_JSON_INVALID")
    except (ValueError, TypeError, RecursionError):
        obj = None
        issue("schema", "STRICT_JSON_PARSE_FAILURE", "$", "one strict JSON object")
    if not isinstance(obj, dict):
        if not audit["schema"]["issues"]:
            issue("schema", "STAGE2_ROOT_SCHEMA", "$", "object", type(obj).__name__)
        for layer in ("binding", "task", "semantic_risk"):
            audit[layer]["status"] = "NOT_EVALUABLE"
        return None, _finish(audit), audit
    if set(obj) != {"generation_status", "claims", "coverage_notes"}:
        issue("schema", "STAGE2_ROOT_SCHEMA", "$", ["generation_status", "claims", "coverage_notes"], sorted(obj))
    if obj.get("generation_status") not in ("GENERATED", "NO_VALID_CLAIMS", "INSUFFICIENT_VISUAL_CONTEXT", "UNRESOLVED_TARGET_REFERENCE"):
        issue("schema", "GENERATION_STATUS_INVALID", "$.generation_status")
    if not isinstance(obj.get("coverage_notes"), str):
        issue("schema", "COVERAGE_NOTES_INVALID", "$.coverage_notes")
    claims = obj.get("claims")
    if not isinstance(claims, list):
        issue("schema", "CLAIMS_ARRAY_REQUIRED", "$.claims")
        claims = []
    if len(claims) > 6:
        issue("schema", "CLAIM_COUNT_EXCEEDED", "$.claims", "0..6", len(claims))
    if (not claims and obj.get("generation_status") == "GENERATED") or (claims and obj.get("generation_status") in ("NO_VALID_CLAIMS", "INSUFFICIENT_VISUAL_CONTEXT")):
        issue("schema", "CLAIM_STATUS_COUNT_MISMATCH", "$.claims")
    reqs = {r["requirement_id"]: r for r in requirements}
    frames = sampling["selected_frames"]
    ids, texts, graph = set(), set(), {}
    for i, c in enumerate(claims):
        path = f"$.claims[{i}]"
        if not isinstance(c, dict):
            issue("schema", "CLAIM_OBJECT_REQUIRED", path)
            continue
        if set(c) != CLAIM_KEYS:
            issue("schema", "CLAIM_FIELDS_INVALID", path, sorted(CLAIM_KEYS), sorted(c))
        cid = c.get("claim_id")
        if not isinstance(cid, str) or not re.fullmatch(r"H[1-6]", cid) or cid in ids:
            issue("schema", "CLAIM_ID_INVALID", path + ".claim_id")
        else:
            ids.add(cid)
        text = c.get("text")
        if not v1._text(text):
            issue("schema", "CLAIM_TEXT_INVALID", path + ".text")
            text = ""
        normalized = " ".join(re.findall(r"\w+", text.casefold()))
        if normalized in texts:
            issue("task", "DUPLICATE_PROPOSITION", path + ".text")
        texts.add(normalized)
        typed = isinstance(c.get("type"), str) and c["type"] in v1.TYPES and isinstance(c.get("evidence_type"), str) and c["evidence_type"] in v1.EVIDENCE
        if not typed:
            issue("schema", "TYPE_OR_EVIDENCE_ENUM_INVALID", path)
        elif not v1._compatible(c):
            issue("task", "TYPE_EVIDENCE_MISMATCH", path, "compatible type/evidence pair", [c["type"], c["evidence_type"]])
        if c.get("status") != "UNVERIFIED":
            issue("schema", "CLAIM_MUST_BE_UNVERIFIED", path + ".status", "UNVERIFIED", c.get("status"))
        if c.get("generation_basis") not in ("QUESTION_HYPOTHESIS", "VISUAL_HYPOTHESIS"):
            issue("schema", "GENERATION_BASIS_INVALID", path + ".generation_basis")
        if c.get("temporal_role") not in tuple(ROLES):
            issue("schema", "TEMPORAL_ROLE_INVALID", path + ".temporal_role")
        if c.get("target_alignment") not in ("QUESTION_ALIGNED", "SCENE_ONLY", "UNRESOLVED"):
            issue("schema", "TARGET_ALIGNMENT_INVALID", path + ".target_alignment")
        valid_lists = {}
        for field, scalar in (("requirement_ids", str), ("depends_on", str), ("image_indices", int)):
            value = c.get(field)
            valid_lists[field] = _sequence(value, scalar)
            if not valid_lists[field]:
                issue("schema", "REFERENCE_LIST_SCHEMA", path + "." + field)
            elif len(set(value)) != len(value):
                issue("binding", "DUPLICATE_REFERENCE", path + "." + field)
        if valid_lists["requirement_ids"] and (not c["requirement_ids"] or not set(c["requirement_ids"]) <= set(reqs)):
            issue("binding", "REQUIREMENT_REFERENCE_INVALID", path + ".requirement_ids", sorted(reqs), c["requirement_ids"])
        image_indices = c.get("image_indices", []) if valid_lists["image_indices"] else []
        if valid_lists["image_indices"] and not all(0 <= j < len(frames) for j in image_indices):
            issue("binding", "IMAGE_INDEX_INVALID", path + ".image_indices", list(range(len(frames))), image_indices)
        quote = c.get("question_quote")
        if c.get("generation_basis") == "VISUAL_HYPOTHESIS":
            if quote is not None:
                issue("binding", "VISUAL_QUESTION_QUOTE_MUST_BE_NULL", path + ".question_quote", None, quote)
            if not image_indices:
                issue("binding", "VISUAL_HYPOTHESIS_NEEDS_IMAGE", path + ".image_indices")
        elif c.get("generation_basis") == "QUESTION_HYPOTHESIS":
            if not v1._text(quote) or quote not in row["question"]:
                issue("binding", "QUESTION_QUOTE_INVALID", path + ".question_quote", "verbatim original-question substring", quote)
        if isinstance(cid, str) and cid in ids and valid_lists["depends_on"]:
            graph[cid] = c["depends_on"]
        role = c.get("temporal_role")
        future_text = bool(re.search(r"\b(?:next (?:phase|step|action)|will (?:be|occur|perform|follow)|going to)\b", text, re.I))
        if future_text and role != "FUTURE_HYPOTHESIS":
            issue("task", "FUTURE_TEXT_ROLE_MISMATCH", path + ".temporal_role")
        if role == "FUTURE_HYPOTHESIS":
            if c.get("generation_basis") != "QUESTION_HYPOTHESIS" or not re.search(r"\b(?:may|might|could|candidate|possible|would|if)\b", text, re.I):
                issue("task", "FUTURE_ACTION_MUST_REMAIN_HYPOTHETICAL", path)
        if role == "QUESTION_TIME_QUERY" and (not image_indices or any(j < 0 or j >= len(frames) or frames[j]["timestamp_seconds"] is None for j in image_indices)):
            issue("binding", "QUERY_TIME_MAPPING_UNRESOLVED", path + ".temporal_role")
        if c.get("type") == "ACTION" and c.get("generation_basis") == "VISUAL_HYPOTHESIS" and len(set(image_indices)) < 2:
            issue("task", "ACTION_REQUIRES_MULTIPLE_OBSERVATIONS", path + ".image_indices")
        if re.search(r"\b(?:score|rating)\s*(?:is|:|=|of)\s*\d", text, re.I):
            issue("task", "FINAL_ASSESSMENT_SCORE_FORBIDDEN", path + ".text")
        if re.search(r"\b(?:is confirmed|are confirmed|has been verified|is verified|proven to)\b", text, re.I):
            issue("task", "VERIFICATION_LANGUAGE_FORBIDDEN", path + ".text")
        if re.search(r"\b(?:then|followed (?:immediately )?by|in order to|so that)\b|[;\n]", text, re.I):
            issue("task", "MULTI_ACTION_OR_PURPOSE_RISK", path + ".text")
        if re.search(r"\b(?:boundaries?.{0,40}(?:are|is) defined|how to (?:identify|define)|requirement)\b", text, re.I):
            issue("task", "META_DESCRIPTION_NOT_VISUAL_PROPOSITION", path + ".text")
        if len(re.findall(r"[.!?](?:\s|$)", text)) > 1:
            issue("task", "MULTIPLE_PROPOSITIONS_RISK", path + ".text")
        for pattern, code in ((r"\b(?:suggesting|indicating|because|therefore|to ensure)\b", "INFERENCE_OR_PURPOSE_REVIEW"),
                              (r"\b(?:all frames|throughout|every occurrence|smooth|without slippage)\b", "SPARSE_OBSERVATION_SCOPE_REVIEW")):
            if re.search(pattern, text, re.I):
                issue("semantic_risk", code, path + ".text")
        if row["qa_type"].startswith("region_caption"):
            if c.get("target_alignment") != "UNRESOLVED" or re.search(r"\b(?:inside the (?:bounding )?box|object (?:in|within) the box|within the (?:defined )?region of interest)\b", text, re.I):
                issue("task", "REGION_CAPTION_TARGET_REFERENCE_UNAVAILABLE", path)
    if row["qa_type"].startswith("region_caption") and obj.get("generation_status") != "UNRESOLVED_TARGET_REFERENCE":
        issue("task", "REGION_CAPTION_TARGET_REFERENCE_UNAVAILABLE", "$.generation_status", "UNRESOLVED_TARGET_REFERENCE", obj.get("generation_status"))
    def visit(node, stack):
        return node in stack or any(d not in graph or visit(d, stack | {node}) for d in graph[node])
    if any(visit(n, set()) for n in graph):
        issue("binding", "DEPENDENCY_INVALID_OR_CYCLIC", "$.claims[*].depends_on")
    for rid, req in reqs.items():
        linked = [c for c in claims if isinstance(c, dict) and _sequence(c.get("requirement_ids"), str) and rid in c["requirement_ids"]]
        if req["type"] in {"ACTION", "HIGH_LEVEL"} and linked and all(c.get("type") == "OBJECT" for c in linked):
            issue("task", "ACTION_OR_CRITERION_REDUCED_TO_OBJECT", "$.requirements." + rid)
    return obj, _finish(audit), audit


def _finish(audit):
    errors = set()
    for layer, info in audit.items():
        if info["issues"]:
            info["status"] = "REVIEW_REQUIRED" if layer == "semantic_risk" else "FAIL"
            if layer != "semantic_risk":
                errors.update(i["code"] for i in info["issues"])
    return sorted(errors)


def bind(obj: dict, sampling: dict, audit: dict) -> dict:
    """Only a validated NEW model object may be bound. No coordinate repair."""
    if any(audit[layer]["status"] != "PASS" for layer in ("schema", "binding", "task")):
        raise v1.ClaimBankError("CANNOT_BIND_INVALID_CLAIMS")
    bound = deepcopy(obj)
    for i, claim in enumerate(bound["claims"]):
        frames = [sampling["selected_frames"][j] for j in claim["image_indices"]]
        claim["frame_orders"] = [f["frame_order"] for f in frames]
        claim["frame_bindings"] = [{"image_index": j, **deepcopy(f)} for j, f in zip(claim["image_indices"], frames)]
        claim["quality_review_flags"] = sorted({x["code"] for x in audit["semantic_risk"]["issues"] if x["path"].startswith(f"$.claims[{i}].")})
    return bound


def validate_bound(obj: dict, row: dict, requirements: list[dict], sampling: dict):
    unbound = deepcopy(obj)
    for c in unbound["claims"]:
        if set(c) != CLAIM_KEYS | {"frame_orders", "frame_bindings", "quality_review_flags"}:
            raise v1.ClaimBankError("BOUND_CLAIM_FIELDS_INVALID")
        for key in ("frame_orders", "frame_bindings", "quality_review_flags"):
            c.pop(key)
    value, errors, audit = evaluate(json.dumps(unbound), row, requirements, sampling)
    if errors or bind(value, sampling, audit) != obj:
        raise v1.ClaimBankError("BOUND_CLAIM_REFERENCE_MISMATCH")


def preservation_fields(raw: str | None):
    try:
        value = strict_json_loads(raw or "", error_code="CLAIMGEN_JSON_INVALID")
    except (ValueError, TypeError):
        return None
    claims = value.get("claims") if isinstance(value, dict) else None
    if not isinstance(claims, list) or any(not isinstance(c, dict) or not v1._text(c.get("text")) for c in claims):
        return None
    # References can be malformed in a schema repair. Protect every field that
    # was present, except that missing fields can be supplied; never edit text.
    return [{k: deepcopy(c[k]) for k in CLAIM_KEYS if k in c} for c in claims]


def preservation_matches(protected, candidate):
    if protected is None:
        return True
    if not isinstance(candidate.get("claims"), list) or any(not isinstance(c, dict) for c in candidate["claims"]):
        return False
    return len(protected) == len(candidate["claims"]) and all(
        all(c.get(k) == v for k, v in original.items()) for original, c in zip(protected, candidate["claims"]))
