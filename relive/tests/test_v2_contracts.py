from __future__ import annotations
import math
import unittest
from relive.v2.contracts import *

class V2ContractTests(unittest.TestCase):
    def requirement(self, **changes):
        values = dict(task=TaskType.TEMPORAL_LOCALIZATION, question_text="When does the event occur?", target_event="visible contact", answer_schema=AnswerSchema.EVENT_INTERVAL, required_evidence=("visible-contact",), temporal_requirement=TemporalRequirement.LOCALIZE_INTERVAL, spatial_requirement=SpatialRequirement.RELATIONAL_COMPOSITE, planner_version="planner-v1", provenance={"source_sha256":"a" * 64})
        values.update(changes)
        return make_requirement_spec(**values)
    def claim(self, requirement, **changes):
        values = dict(requirement_id=requirement.requirement_id, parent_claim_id=None, claim_role=ClaimRole.TARGET_HYPOTHESIS, surface_text="Forceps touch tissue.", subject="forceps", predicate="touch", object="tissue", polarity=Polarity.POSITIVE, temporal_quantifier=TemporalQuantifier.AT_FRAME, temporal_scope=TemporalScope.INSTANT, logical_scope=LogicalScope.LOCAL, observability=Observability.APPARENT_2D, evidence_geometry=EvidenceGeometryType.RELATIONAL_COMPOSITE, required_components=("forceps", "tissue"), generation_reason=GenerationReason.USER_AUTHORED, provenance={"source_sha256":"b" * 64})
        values.update(changes)
        return make_claim_spec(**values)
    def test_requirement_hash_is_stable_and_freezes_nested_input(self):
        provenance={"source_sha256":"a" * 64,"nested":["x"]}
        first=self.requirement(provenance=provenance); second=self.requirement(provenance={"nested":["x"],"source_sha256":"a" * 64})
        provenance["nested"].append("changed")
        self.assertEqual(first.requirement_id, second.requirement_id)
        self.assertEqual(first.content_sha256(), second.content_sha256())
        self.assertEqual(first.provenance["nested"], ("x",))
    def test_question_hash_and_prohibited_inputs_fail_closed(self):
        req=self.requirement()
        with self.assertRaisesRegex(ContractError,"QUESTION_SHA256"):
            RequirementSpec(req.requirement_id, req.task, req.question_text, "0"*64, req.target_event, req.answer_schema, req.required_evidence, req.temporal_requirement, req.spatial_requirement, req.planner_version, req.provenance)
        with self.assertRaisesRegex(ContractError,"PROHIBITED"):
            self.requirement(provenance={"roi":"forbidden"})
    def test_claim_identity_contains_parent_scope_geometry_and_observability(self):
        req=self.requirement(); base=self.claim(req)
        parent=self.claim(req, generation_reason=GenerationReason.HYPOTHESIS_ENUMERATION)
        variants=(self.claim(req,parent_claim_id=parent.claim_id), self.claim(req,temporal_scope=TemporalScope.WINDOW), self.claim(req,evidence_geometry=EvidenceGeometryType.SINGLE_REGION), self.claim(req,observability=Observability.DIRECTLY_VISIBLE))
        self.assertEqual(base.logical_scope, LogicalScope.LOCAL)
        self.assertEqual(base.evidence_geometry, EvidenceGeometryType.RELATIONAL_COMPOSITE)
        self.assertEqual(len({base.claim_id, *(item.claim_id for item in variants)}),5)
    def test_contact_claim_is_valid_composite_and_invalid_values_rejected(self):
        req=self.requirement(); claim=self.claim(req)
        self.assertEqual(claim.observability, Observability.APPARENT_2D)
        with self.assertRaisesRegex(ContractError,"FINITE"):
            freeze_json({"value": math.nan})
        with self.assertRaisesRegex(ContractError,"FINITE"):
            freeze_json({"value": math.inf})
        with self.assertRaisesRegex(ContractError,"TASK_TYPE"):
            RequirementSpec("", "unknown", "q", "0"*64, "event", AnswerSchema.EVENT_INTERVAL, ("e",), TemporalRequirement.LOCALIZE_INTERVAL, SpatialRequirement.NOT_REQUIRED, "p", freeze_json({}))

if __name__ == "__main__": unittest.main()
