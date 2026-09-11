"""Strict claim proposals without rationale leakage into the verifier."""
from __future__ import annotations

import hashlib
import json
from importlib.resources import files

from relive.json_protocol import strict_json
from relive.types import Claim, ContrastGroup, ExclusivityStatus, EvidenceCandidate

PROMPT_VERSIONS = {
    "semantic": "relive-semantic-v4",
    "claims": "relive-claims-v1",
    "contrasts": "relive-contrasts-v1",
    "spatial": "relive-spatial-v2",
    "spatial_refine_dependence": "relive-spatial-refine-dependence-v1",
    "spatial_refine_sufficiency": "relive-spatial-refine-sufficiency-v1",
    "spatial_refine_control_geometry": "relive-spatial-refine-control-geometry-v1",
    "spatial_refine_intervention_effect": "relive-spatial-refine-intervention-effect-v1",
    "spatial_refine_dependence_v2": "relive-spatial-refine-dependence-v2",
    "spatial_refine_sufficiency_v2": "relive-spatial-refine-sufficiency-v2",
    "spatial_refine_control_geometry_v2": "relive-spatial-refine-control-geometry-v2",
    "spatial_refine_intervention_effect_v2": "relive-spatial-refine-intervention-effect-v2",
}


def stable_id(prefix: str, value) -> str:
    digest = hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                       separators=(",", ":")).encode()).hexdigest()
    return f"{prefix}_{digest[:24]}"


def prompt(stage: str, data: dict) -> str:
    template = files("relive").joinpath("prompts", f"{stage}.txt").read_text(encoding="utf-8")
    return template + "\nInput data:\n" + json.dumps(data, ensure_ascii=False, sort_keys=True)


def claim_scope(candidate: EvidenceCandidate) -> dict:
    scope = {"frame_ids": list(candidate.frame_ids), "localization": "frame_ids"}
    if candidate.timestamps and all(t is not None for t in candidate.timestamps):
        scope.update(timestamps=list(candidate.timestamps), localization="supplied_timestamps")
    return scope


def parse_claims(raw: str, sample_id: str, candidate: EvidenceCandidate, maximum: int) -> tuple[Claim, ...]:
    obj = strict_json(raw)
    if not isinstance(obj, dict) or set(obj) != {"claims"} or not isinstance(obj["claims"], list):
        raise ValueError("Expected exactly a claims array")
    if len(obj["claims"]) > maximum:
        raise ValueError("Claim count exceeds frozen maximum")
    result, seen = [], set()
    for row in obj["claims"]:
        if not isinstance(row, dict) or set(row) != {"text"}:
            raise ValueError("Each claim must contain exactly text")
        text = row["text"]
        if not isinstance(text, str) or not text.strip() or len(text) > 2000:
            raise ValueError("Claim text must be nonempty and bounded")
        text = text.strip()
        if text in seen:
            raise ValueError("Duplicate claim proposal")
        seen.add(text)
        result.append(Claim(stable_id("claim", [sample_id, text]), text,
                            time_scope=claim_scope(candidate), source="frozen_model_proposal"))
    return tuple(result)


def make_contrast(target: Claim, alternatives: list[str], candidate: EvidenceCandidate,
                  dimension: str, exclusivity: ExclusivityStatus, source: str):
    if not dimension.strip() or not source.strip():
        raise ValueError("Contrast dimension and declaration source required")
    seen = {target.text}
    claims = []
    for text in alternatives:
        if not isinstance(text, str) or not text.strip() or text.strip() in seen:
            raise ValueError("Alternatives must be nonempty, unique and distinct from target")
        text = text.strip()
        seen.add(text)
        claims.append(Claim(stable_id("contrast", [target.claim_id, text]), text,
                            entity=target.entity, time_scope=target.time_scope,
                            source=source))
    group = ContrastGroup(stable_id("group", [candidate.candidate_id, target.claim_id, alternatives, source]),
                          (target.claim_id, *(c.claim_id for c in claims)), dimension, exclusivity,
                          {"source": source, "entity": target.entity,
                           "time_scope": target.time_scope, "automatically_established_exclusivity": False})
    return tuple(claims), group


def parse_contrasts(raw: str, target: Claim, candidate: EvidenceCandidate, maximum: int):
    obj = strict_json(raw)
    if not isinstance(obj, dict) or set(obj) != {"alternatives", "comparison_dimension"}:
        raise ValueError("Expected alternatives and comparison_dimension only")
    rows = obj["alternatives"]
    if not isinstance(rows, list) or len(rows) > maximum:
        raise ValueError("Invalid alternatives count")
    if not isinstance(obj["comparison_dimension"], str):
        raise ValueError("Invalid comparison dimension")
    if any(not isinstance(r, dict) or set(r) != {"text"} for r in rows):
        raise ValueError("Alternative schema requires only text")
    return make_contrast(target, [r["text"] for r in rows], candidate, obj["comparison_dimension"],
                         ExclusivityStatus.UNRESOLVED, "frozen_model_proposal")
