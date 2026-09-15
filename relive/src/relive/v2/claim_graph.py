"""Append-only ClaimSpec graph validation for ReliVE-v2."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
from relive.storage.artifacts import stable_hash
from .contracts import ClaimRole, ClaimSpec, RequirementSpec

GRAPH_VERSION = "relive-v2-claim-graph-v1"

class ClaimGraphError(ValueError):
    pass

@dataclass(frozen=True)
class ClaimGraph:
    requirement_id: str
    nodes: tuple[ClaimSpec, ...]
    graph_version: str = GRAPH_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.requirement_id, str) or not self.requirement_id:
            raise ClaimGraphError("REQUIREMENT_ID_REQUIRED")
        if not isinstance(self.nodes, tuple):
            raise ClaimGraphError("NODES_TUPLE_REQUIRED")
        _validate_nodes(self.requirement_id, self.nodes)

    def to_canonical_dict(self) -> dict:
        return {"requirement_id": self.requirement_id, "nodes": [node.to_canonical_dict() for node in self.nodes], "graph_version": self.graph_version}

    def content_sha256(self) -> str:
        return stable_hash(self.to_canonical_dict())


def _validate_nodes(requirement_id: str, nodes: tuple[ClaimSpec, ...]) -> None:
    seen: dict[str, ClaimSpec] = {}
    null_count = 0
    for node in nodes:
        if node.requirement_id != requirement_id:
            raise ClaimGraphError("CLAIM_REQUIREMENT_ID_MISMATCH")
        previous = seen.get(node.claim_id)
        if previous is not None:
            if previous.to_canonical_dict() != node.to_canonical_dict():
                raise ClaimGraphError("CLAIM_ID_CONTENT_COLLISION")
            raise ClaimGraphError("DUPLICATE_CLAIM_ID")
        if node.parent_claim_id == node.claim_id:
            raise ClaimGraphError("CLAIM_SELF_PARENT_FORBIDDEN")
        if node.claim_role in (ClaimRole.TARGET_HYPOTHESIS, ClaimRole.NULL_HYPOTHESIS):
            if node.parent_claim_id is not None:
                raise ClaimGraphError("ROOT_HYPOTHESIS_PARENT_FORBIDDEN")
            if node.claim_role is ClaimRole.NULL_HYPOTHESIS:
                null_count += 1
                if null_count > 1:
                    raise ClaimGraphError("MULTIPLE_NULL_HYPOTHESES")
        else:
            if node.parent_claim_id is None:
                raise ClaimGraphError("OBSERVATION_PARENT_REQUIRED")
            parent = seen.get(node.parent_claim_id)
            if parent is None:
                raise ClaimGraphError("CLAIM_PARENT_MUST_EXIST_EARLIER")
            if parent.claim_role not in (ClaimRole.TARGET_HYPOTHESIS, ClaimRole.NULL_HYPOTHESIS, ClaimRole.OBSERVATION):
                raise ClaimGraphError("OBSERVATION_PARENT_ROLE_INVALID")
        seen[node.claim_id] = node


def empty_claim_graph(requirement: RequirementSpec) -> ClaimGraph:
    return ClaimGraph(requirement.requirement_id, ())


def append_claim(graph: ClaimGraph, claim: ClaimSpec) -> ClaimGraph:
    return ClaimGraph(graph.requirement_id, (*graph.nodes, claim), graph.graph_version)


def append_claims(graph: ClaimGraph, claims: Sequence[ClaimSpec]) -> ClaimGraph:
    result = graph
    for claim in claims:
        result = append_claim(result, claim)
    return result


def validate_hypothesis_set(graph: ClaimGraph, *, min_non_null_hypotheses: int) -> None:
    if not isinstance(min_non_null_hypotheses, int) or isinstance(min_non_null_hypotheses, bool) or min_non_null_hypotheses < 1:
        raise ClaimGraphError("MIN_NON_NULL_HYPOTHESES_MUST_BE_POSITIVE")
    non_null = [node for node in graph.nodes if node.claim_role is ClaimRole.TARGET_HYPOTHESIS]
    null = [node for node in graph.nodes if node.claim_role is ClaimRole.NULL_HYPOTHESIS]
    if len(non_null) < min_non_null_hypotheses:
        raise ClaimGraphError("INSUFFICIENT_NON_NULL_HYPOTHESES")
    if len(null) != 1:
        raise ClaimGraphError("EXACTLY_ONE_NULL_HYPOTHESIS_REQUIRED")


def graph_sha256(graph: ClaimGraph) -> str:
    return graph.content_sha256()
