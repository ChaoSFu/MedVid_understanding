from __future__ import annotations

from enum import Enum
from typing import Any


PROTOCOL_VERSION = "relive-spatial-evidence-reliability-v1"
PROPOSAL_PROMPT_VERSION = "relive-spatial-sufficiency-proposal-v1"
REFINEMENT_PROMPT_VERSION = "relive-spatial-sufficiency-repair-v1"
RESIDUAL_PROMPT_VERSION = "relive-spatial-residual-search-v1"
VERIFIER_PROMPT_VERSION = "relive-spatial-semantic-verifier-v3"
MAX_REPAIR_ATTEMPTS = 1
MAX_RESIDUAL_ATTEMPTS = 1
MAX_EVIDENCE_REGIONS = 2


class ScientificStatus(str, Enum):
    BASE_UNSUPPORTED = "BASE_UNSUPPORTED"
    SPATIAL_INSUFFICIENT = "SPATIAL_INSUFFICIENT"
    SUPPORTED_EVIDENCE = "SUPPORTED_EVIDENCE"
    DEPENDENCY_VERIFIED = "DEPENDENCY_VERIFIED"
    DEPENDENCY_SUPPORTED_NO_CONTROL = "DEPENDENCY_SUPPORTED_NO_CONTROL"
    INTERVENTION_UNSTABLE = "INTERVENTION_UNSTABLE"
    SPATIAL_SUFFICIENCY_UNRESOLVED = "SPATIAL_SUFFICIENCY_UNRESOLVED"
    REFINEMENT_NO_OP = "REFINEMENT_NO_OP"
    RESIDUAL_REGION_NOT_SUFFICIENT = "RESIDUAL_REGION_NOT_SUFFICIENT"
    EVIDENCE_SET_VERIFIED = "EVIDENCE_SET_VERIFIED"
    DEPENDENCY_UNRESOLVED = "DEPENDENCY_UNRESOLVED"
    PROPOSAL_UNRESOLVED = "PROPOSAL_UNRESOLVED"
    TECHNICAL_FAILURE = "TECHNICAL_FAILURE"


SEMANTIC_STATUSES = frozenset({"SUPPORTED", "CONTRADICTED", "INSUFFICIENT"})


def _semantic(value: Any) -> str | None:
    if isinstance(value, str) and value in SEMANTIC_STATUSES:
        return value
    return None


def classify_formal_result(*, original: str, keep: str, drop: str,
                           control_constructible: bool, control: str | None) -> ScientificStatus:
    """Frozen decision order; technical/unparseable values fail closed."""
    original, keep, drop, control = map(_semantic, (original, keep, drop, control))
    if original is None or keep is None or drop is None or (control_constructible and control is None):
        return ScientificStatus.TECHNICAL_FAILURE
    if original != "SUPPORTED":
        return ScientificStatus.BASE_UNSUPPORTED
    if control_constructible and control != "SUPPORTED":
        return ScientificStatus.INTERVENTION_UNSTABLE
    if keep != "SUPPORTED":
        return ScientificStatus.SPATIAL_INSUFFICIENT
    if drop == "SUPPORTED":
        return ScientificStatus.SUPPORTED_EVIDENCE
    return (ScientificStatus.DEPENDENCY_VERIFIED if control_constructible
            else ScientificStatus.DEPENDENCY_SUPPORTED_NO_CONTROL)


def controller_action(status: ScientificStatus, *, repair_attempts: int = 0,
                      residual_attempts: int = 0) -> str:
    if status == ScientificStatus.SPATIAL_INSUFFICIENT and repair_attempts < MAX_REPAIR_ATTEMPTS:
        return "SUFFICIENCY_REPAIR"
    if status == ScientificStatus.SUPPORTED_EVIDENCE and residual_attempts < MAX_RESIDUAL_ATTEMPTS:
        return "RESIDUAL_SUPPORT_SEARCH"
    return "STOP"


PROPOSAL_PROMPT = (
    "Propose one compact visual region that contains all visible information needed to determine "
    "whether the claim is supported. The region should be sufficient for judging the claim, not "
    "merely the smallest possible object box. Return strict JSON with status, "
    "bbox_normalized_0_1000, and reason. Return UNRESOLVED with a null bbox if no concrete local "
    "region can be identified."
)
REFINEMENT_PROMPT = (
    "Revise the previous region so that it contains all visible information necessary to judge the "
    "claim. The region may shift or expand. Preserve compactness, but prioritize sufficiency over "
    "minimum area. Do not invent evidence."
)
RESIDUAL_PROMPT = (
    "Inspect only the supplied image where the current evidence region has already been masked. "
    "Determine whether there is another concrete local visual region outside the masked area that "
    "independently contains enough visible information to support the claim. If no such local region "
    "can be identified, return UNRESOLVED."
)
VERIFIER_MASK_NOTICE = (
    "Gray masked pixels represent unavailable visual information. They must not be interpreted as "
    "visual evidence either for or against the claim."
)


def verifier_prompt(*, claim: str, variant: str) -> str:
    """Return the frozen, single-image semantic-verifier v2 prompt."""
    return (
        "Judge whether the supplied image supports this claim: " + claim + "\n" +
        VERIFIER_MASK_NOTICE + "\n"
        "Return exactly one JSON object with no commentary or additional text. "
        "A single complete ```json Markdown wrapper is allowed, but a bare JSON object is preferred. "
        "The object must contain exactly these three fields: status, observation, frame_index. "
        "status must be SUPPORTED, CONTRADICTED, or INSUFFICIENT. "
        "observation must be one short sentence of at most 40 words. "
        "Because exactly one image is supplied, frame_index must be the integer 0. "
        "Do not return frame_references. "
        "Example: {\"status\":\"SUPPORTED\",\"observation\":\"Visible evidence supports the claim.\",\"frame_index\":0}. "
        f"The input variant is {variant}; use identical semantics for every variant."
    )


def protocol_manifest() -> dict[str, Any]:
    return {
        "format": PROTOCOL_VERSION,
        "frozen": True,
        "model_parameters_frozen": True,
        "max_sufficiency_repairs": MAX_REPAIR_ATTEMPTS,
        "max_residual_searches": MAX_RESIDUAL_ATTEMPTS,
        "max_evidence_regions": MAX_EVIDENCE_REGIONS,
        "semantic_statuses": sorted(SEMANTIC_STATUSES),
        "scientific_statuses": [item.value for item in ScientificStatus],
        "prompt_versions": {
            "proposal": PROPOSAL_PROMPT_VERSION,
            "refinement": REFINEMENT_PROMPT_VERSION,
            "residual": RESIDUAL_PROMPT_VERSION,
            "verifier": VERIFIER_PROMPT_VERSION,
        },
        "principle": "Sufficiency is not dependency.",
    }
