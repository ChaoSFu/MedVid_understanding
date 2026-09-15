from __future__ import annotations
import unittest
from relive.v2.adaptation_controller import *
from relive.v2.failure_taxonomy import *
from relive.v2.contracts import freeze_json

class V2ControllerTests(unittest.TestCase):
    def setUp(self):
        self.budget=ClaimBudget(1,1,1,1,1)
        self.state=ReliVEState("a"*64,"b"*64,"claim",None,None,"verifier",self.budget,0)
        self.policy=default_policy()
    def diagnosis(self,failure=None, outcome=DiagnosisOutcome.UNRESOLVED):
        if failure is None: return diagnosis_complete()
        return VerificationDiagnosis(outcome,failure,freeze_json({"source":"test"}))
    def test_frozen_routes_and_no_admission(self):
        expected={None:AdaptationAction.STOP_DIAGNOSTIC,VerificationFailure.ORIGINAL_INSUFFICIENT:AdaptationAction.TEMPORAL_REACQUIRE,VerificationFailure.KEEP_SUPPORT_LOST:AdaptationAction.SPATIAL_RECOMPOSE,VerificationFailure.DROP_SUPPORT_PERSISTS:AdaptationAction.RESIDUAL_LOCALIZE,VerificationFailure.CONTROL_SUPPORT_LOST:AdaptationAction.INTERVENTION_DIAGNOSTIC,VerificationFailure.TECHNICAL_EXECUTION_FAILURE:AdaptationAction.TECHNICAL_DIAGNOSTIC}
        for failure,action in expected.items():
            diagnosis=self.diagnosis(failure, DiagnosisOutcome.TECHNICAL_FAILURE if failure is VerificationFailure.TECHNICAL_EXECUTION_FAILURE else DiagnosisOutcome.UNRESOLVED)
            decision=decide_transition(self.state,diagnosis,self.policy)
            self.assertEqual(decision.action,action); self.assertTrue(decision.diagnostic_only); self.assertFalse(decision.certificate_authority)
    def test_full_gray_requires_predeclared_switch_and_budget(self):
        diagnosis=self.diagnosis(VerificationFailure.DROP_SUPPORT_PERSISTS_WITH_FULL_GRAY_SUPPORT)
        denied=decide_transition(self.state,diagnosis,FrozenAdaptationPolicy("p",tuple(AdaptationAction),False))
        self.assertEqual(denied.action,AdaptationAction.STOP_ABSTAIN)
        accepted=decide_transition(self.state,diagnosis,self.policy); self.assertEqual(accepted.action,AdaptationAction.VERIFIER_SWITCH)
        empty=ReliVEState("a"*64,"b"*64,"claim",None,None,"verifier",ClaimBudget(1,1,1,1,0),0)
        stopped=decide_transition(empty,diagnosis,self.policy); self.assertEqual(stopped.action,AdaptationAction.STOP_ABSTAIN); self.assertIn("VERIFIER_SWITCHES",stopped.termination_reason)
    def test_budget_dimensions_policy_and_cycle(self):
        mapping={AdaptationAction.TEMPORAL_REACQUIRE:"temporal_jumps_remaining",AdaptationAction.TEMPORAL_EXPAND_CONTEXT:"boundary_refinements_remaining",AdaptationAction.TEMPORAL_DISCRIMINATIVE_RETRIEVAL:"temporal_jumps_remaining",AdaptationAction.SPATIAL_RECOMPOSE:"spatial_regrounds_remaining",AdaptationAction.SPATIAL_RECOMPOSE_CONTROL_GEOMETRY:"spatial_regrounds_remaining",AdaptationAction.RESIDUAL_LOCALIZE:"residual_components_remaining",AdaptationAction.VERIFIER_SWITCH:"verifier_switches_remaining"}
        for action,field in mapping.items():
            after=consume_budget(self.budget,action); self.assertEqual(getattr(after,field),0)
        only_stop=FrozenAdaptationPolicy("p",(AdaptationAction.STOP_DIAGNOSTIC,),True)
        denied=decide_transition(self.state,self.diagnosis(VerificationFailure.ORIGINAL_INSUFFICIENT),only_stop)
        self.assertEqual(denied.action,AdaptationAction.STOP_ABSTAIN)
        second=ReliVEState("a"*64,"b"*64,"claim",None,None,"verifier",ClaimBudget(0,0,0,0,0),5)
        self.assertEqual(content_state_sha256(self.state),content_state_sha256(second)); self.assertNotEqual(event_state_sha256(self.state),event_state_sha256(second))
        cycle=guard_proposed_state(second,{content_state_sha256(self.state)}); self.assertTrue(cycle.cycle_detected); self.assertEqual(cycle.action,AdaptationAction.STOP_ABSTAIN)

if __name__ == "__main__": unittest.main()

class V2IsolationTests(unittest.TestCase):
    def test_controller_sources_do_not_import_backend_cache_or_certificate_authority(self):
        from pathlib import Path
        root=Path(__file__).resolve().parents[1] / "src" / "relive" / "v2"
        text="\n".join((root / name).read_text() for name in ("adaptation_controller.py","fixture_replay.py","failure_taxonomy.py"))
        for forbidden in ("relive.backends", "certificate import", "ArtifactStore", "CachedInference", "torch", "transformers"):
            self.assertNotIn(forbidden,text)
