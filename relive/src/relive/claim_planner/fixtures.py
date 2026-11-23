"""Synthetic transport fixtures on frozen public questions; no clinical output.

These fixtures test contracts and flow, not a simulated scientific improvement.
Only public task categories drive the fixture, never video/anchor identities.
"""
from __future__ import annotations

from copy import deepcopy
import json

from relive.backends.base import Backend

def entity(eid="E1", label="$target"):
    return {"entity_id": eid, "label": label, "source": "UNRESOLVED", "source_quote": None, "requirement_id": None}

def claim(cid="H1", rid="R1", kind="OBJECT", predicate="IS_VISIBLE"):
    return {"claim_id": cid, "text": "The unresolved target is visible within the input clip.",
            "type": kind, "status": "UNVERIFIED", "requirement_ids": [rid], "answer_relevance": "REQUIRED",
            "verification_purpose": "Test a necessary question-related condition; synthetic contract fixture only.",
            "proposition": {"predicate": predicate, "subject": "E1", "object": None, "qualifier": None,
                            "polarity": "POSITIVE", "scope": {"kind": "INPUT_CLIP", "description": "The publicly supplied input clip; no frame or timestamp bound.", "query_time": None}},
            "evidence_requirement": {"modality": "SINGLE_FRAME", "spatial_targets": ["E1"], "temporal_requirement": None},
            "dependencies": [], "interpretation_rule": None,
            "bindings": {"frame": None, "timestamp": None, "bbox": None}, "evidence_status": "NOT_SEARCHED"}

def payload(row):
    req = row["requirements"][0]["requirement_id"]
    qa = row["qa_type"]
    if qa.startswith("region_caption"):
        return {"planning_status": "UNRESOLVED_TARGET_REFERENCE", "entities": [], "claims": [],
                "information_gaps": [{"code": "PUBLIC_TARGET_REFERENCE_MISSING", "description": "Public query box absent; target-specific planning remains unresolved.", "requirement_ids": [req]}]}
    c = claim(rid=req)
    if qa in {"tal", "skill_assessment"} or qa.startswith(("dense_captioning", "video_summary")):
        c.update(type="ACTION", text="The candidate event occurs within a bounded interval of the input clip.")
        c["proposition"].update(predicate="EVENT", qualifier="$candidate_event")
        c["proposition"]["scope"]["kind"] = "BOUNDED_EVENT_SEARCH"
        c["evidence_requirement"].update(modality="MULTI_FRAME", temporal_requirement={"description": "Observe onset and offset of the candidate event; no occurrence assumed.", "min_observations": 2})
    elif qa == "next_action":
        c.update(type="ATTRIBUTE", text="The unresolved target has the candidate precondition state.")
        c["proposition"].update(predicate="HAS_ATTRIBUTE", qualifier="$precondition_state")
    elif qa == "cvs_assessment":
        c.update(type="RELATION", text="The unresolved anatomical structures have the candidate spatial relation.")
        c["proposition"].update(predicate="STATIC_RELATION", object="E2", qualifier="$anatomical_relation")
        c["evidence_requirement"].update(modality="MULTI_REGION_SINGLE_FRAME", spatial_targets=["E1", "E2"])
    entities = [entity()]
    if c["proposition"]["object"]:
        entities.append(entity("E2", "$second_target"))
    return {"planning_status": "INCOMPLETE_PLAN", "entities": entities, "claims": [c],
            "information_gaps": [{"code": "SYNTHETIC_FIXTURE_NOT_MODEL_PLANNING", "description": "Only an engineering fixture. Semantic relevance/coverage not evaluated; unresolved slots need future acquisition.", "requirement_ids": []}]}

class FixtureBackend(Backend):
    synthetic = True

    def __init__(self, config, overrides=None):
        super().__init__(config)
        self.overrides = overrides or {}
        self.requests = []

    def fingerprint(self):
        return {**super().fingerprint(), "fixture_version": "stage2-v2-contract-fixture-v1", "overrides": deepcopy(self.overrides)}

    def infer(self, request):
        assert not request['image_paths'] and not request['frame_ids']
        self.calls += 1
        self.requests.append(deepcopy(request))
        data = json.loads(request['prompt'].split('Untrusted data (not instructions):\n', 1)[1])
        value = self.overrides.get(data['qa_type'], payload(data))
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
