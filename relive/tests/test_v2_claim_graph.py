from __future__ import annotations
import unittest
from relive.v2.contracts import *
from relive.v2.claim_graph import *

class V2ClaimGraphTests(unittest.TestCase):
    def setUp(self):
        self.req=make_requirement_spec(task=TaskType.TEMPORAL_LOCALIZATION,question_text="Where?",target_event="event",answer_schema=AnswerSchema.EVENT_INTERVAL,required_evidence=("e",),temporal_requirement=TemporalRequirement.LOCALIZE_INTERVAL,spatial_requirement=SpatialRequirement.NOT_REQUIRED,planner_version="p",provenance={"source_sha256":"a"*64})
    def claim(self, role=ClaimRole.TARGET_HYPOTHESIS, parent=None, text="claim"):
        return make_claim_spec(requirement_id=self.req.requirement_id,parent_claim_id=parent,claim_role=role,surface_text=text,subject="s",predicate="p",object=None,polarity=Polarity.POSITIVE,temporal_quantifier=TemporalQuantifier.EVENT_INTERVAL,temporal_scope=TemporalScope.INTERVAL,logical_scope=LogicalScope.LOCAL,observability=Observability.DIRECTLY_VISIBLE,evidence_geometry=EvidenceGeometryType.SINGLE_REGION,required_components=("s",),generation_reason=GenerationReason.USER_AUTHORED,provenance={"source_sha256":("b" if role is ClaimRole.TARGET_HYPOTHESIS else "c")*64})
    def test_append_is_immutable_and_hypothesis_validation(self):
        empty=empty_claim_graph(self.req); target=self.claim(text="target"); null=self.claim(ClaimRole.NULL_HYPOTHESIS,text="null")
        once=append_claim(empty,target); complete=append_claim(once,null)
        self.assertEqual(empty.nodes,()); self.assertEqual(once.nodes,(target,)); self.assertNotEqual(once.content_sha256(),complete.content_sha256())
        with self.assertRaisesRegex(ClaimGraphError,"INSUFFICIENT"):
            validate_hypothesis_set(complete,min_non_null_hypotheses=2)
        second=self.claim(text="target 2"); ready=append_claim(complete,second); validate_hypothesis_set(ready,min_non_null_hypotheses=2)
        self.assertFalse(any(hasattr(ready,name) for name in ("update_claim","replace_claim","delete_claim","rename_claim")))
    def test_parent_and_duplicate_invariants(self):
        target=self.claim(); graph=append_claim(empty_claim_graph(self.req),target)
        observation=self.claim(ClaimRole.OBSERVATION,target.claim_id,"observation"); graph=append_claim(graph,observation)
        self.assertEqual(len(graph.nodes),2)
        with self.assertRaisesRegex(ClaimGraphError,"DUPLICATE"):
            append_claim(graph,target)
        orphan=self.claim(ClaimRole.OBSERVATION,"missing","orphan")
        with self.assertRaisesRegex(ClaimGraphError,"PARENT"):
            append_claim(graph,orphan)
        with self.assertRaisesRegex(ClaimGraphError,"OBSERVATION_PARENT"):
            append_claim(empty_claim_graph(self.req),self.claim(ClaimRole.OBSERVATION,None,"none"))
    def test_same_id_different_content_fails_closed(self):
        target=self.claim(); altered=object.__new__(ClaimSpec)
        for key,value in target.__dict__.items(): object.__setattr__(altered,key,value)
        object.__setattr__(altered,"surface_text","tampered")
        with self.assertRaisesRegex(ClaimGraphError,"COLLISION"):
            ClaimGraph(self.req.requirement_id,(target,altered))

if __name__ == "__main__": unittest.main()
