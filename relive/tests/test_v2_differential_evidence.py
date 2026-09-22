from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from relive.storage.artifacts import canonical_json, stable_hash
from relive.v2.differential_evidence import (
    CalibrationArtifact, DifferentialEvidenceError, InterventionVariant,
    MatchedControlSet, admit, choose_adapt_route, differential_cache_key,
    load_policy, metrics, verify_closed_pilot,
)
from relive.v2.oracle_feasibility import prepare_runtime


def h(value: str) -> str: return hashlib.sha256(value.encode()).hexdigest()
def variant(name, score, status="SUPPORTED"): return InterventionVariant(name, score, status, h("renderer"))


class DifferentialPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy_path = Path(__file__).resolve().parents[1] / "configs/v2/differential_evidence_policy.yaml"
        _, cls.policy_sha = load_policy(cls.policy_path)

    def controls(self, *, tier="TIER_1_EXACT", keep=(1.0,), drop=(1.0,), clean=True):
        return MatchedControlSet(tier, tuple(variant("KEEP_MATCHED_CONTROL", score) for score in keep),
            tuple(variant("DROP_MATCHED_CONTROL", score) for score in drop), clean, True, True, True, True,
            True, True, True, True, True, h("tolerance") if tier == "TIER_2_MATCHED" else None)

    def calibration(self, policy=None):
        return CalibrationArtifact(policy or self.policy_sha, {"tau_original": .5, "tau_keep_floor": .5,
            "tau_keep": .1, "tau_drop": .1, "tau_evidence": .5, "tau_operator": .5}, "synthetic-calibration")

    def certificate(self, value, calibration):
        return admit(policy_sha256=self.policy_sha, calibration=calibration, value=value, evidence_quality=1., operator_quality=1.,
            model_revision="revision", prompt_hash=h("prompt"), evidence_program_hash=h("program"), control_set_hash=h("control"), intervention_renderer_hash=h("renderer"))

    def test_differential_passes_when_strict_fails(self):
        value=metrics(original=variant("ORIGINAL", 2), keep_target=variant("KEEP_TARGET", 1.5), drop_target=variant("DROP_TARGET", .2, "SUPPORTED"), controls=self.controls(keep=(1,), drop=(1,)))
        self.assertFalse(value.strict_local_dependence)
        self.assertEqual(self.certificate(value, self.calibration()).status, "CALIBRATED_DIFFERENTIAL_DEPENDENCE")

    def test_strict_and_differential_pass(self):
        value=metrics(original=variant("ORIGINAL", 2), keep_target=variant("KEEP_TARGET", 1.5), drop_target=variant("DROP_TARGET", .2, "INSUFFICIENT"), controls=self.controls(keep=(1,), drop=(1,)))
        self.assertTrue(value.strict_local_dependence)
        self.assertIn("STRICT_LOCAL_DEPENDENCE", self.certificate(value, self.calibration()).diagnostic_labels)

    def test_keep_can_be_insufficient_but_score_differential(self):
        value=metrics(original=variant("ORIGINAL", 2), keep_target=variant("KEEP_TARGET", 1.2, "INSUFFICIENT"), drop_target=variant("DROP_TARGET", .2), controls=self.controls(keep=(.8,), drop=(1,)))
        self.assertEqual(self.certificate(value, self.calibration()).status, "CALIBRATED_DIFFERENTIAL_DEPENDENCE")

    def test_drop_can_remain_supported_but_have_large_differential(self):
        value=metrics(original=variant("ORIGINAL", 2), keep_target=variant("KEEP_TARGET", 1.2), drop_target=variant("DROP_TARGET", .2, "SUPPORTED"), controls=self.controls(keep=(.8,), drop=(1,)))
        self.assertGreater(value.g_drop, .1)
        self.assertEqual(self.certificate(value, self.calibration()).status, "CALIBRATED_DIFFERENTIAL_DEPENDENCE")

    def test_no_target_control_difference_fails(self):
        value=metrics(original=variant("ORIGINAL", 2), keep_target=variant("KEEP_TARGET", 1), drop_target=variant("DROP_TARGET", 1), controls=self.controls(keep=(1,), drop=(1,)))
        self.assertEqual(self.certificate(value, self.calibration()).failure_reason, "DIFFERENTIAL_THRESHOLDS_NOT_MET")

    def test_tier_one_one_control_is_allowed(self): self.assertEqual(self.controls().tier, "TIER_1_EXACT")
    def test_tier_two_requires_two_controls(self):
        with self.assertRaisesRegex(DifferentialEvidenceError, "CARDINALITY"):
            self.controls(tier="TIER_2_MATCHED")
    def test_tier_two_two_controls_and_bound_tolerance_are_allowed(self):
        self.assertEqual(self.controls(tier="TIER_2_MATCHED", keep=(1, .9), drop=(1, .8)).tier, "TIER_2_MATCHED")
    def test_tier_two_unmatched_quality_is_rejected(self):
        with self.assertRaisesRegex(DifferentialEvidenceError, "TIER_2_MATCHING"):
            MatchedControlSet("TIER_2_MATCHED", (variant("KEEP_MATCHED_CONTROL", 1), variant("KEEP_MATCHED_CONTROL", .9)),
                (variant("DROP_MATCHED_CONTROL", 1), variant("DROP_MATCHED_CONTROL", .9)), True, True, True, True, True,
                False, True, True, True, True, h("tolerance"))
    def test_control_contamination_rejected(self):
        with self.assertRaisesRegex(DifferentialEvidenceError, "QUALITY"):
            self.controls(clean=False)
    def test_missing_continuous_score_cannot_certify(self):
        with self.assertRaisesRegex(DifferentialEvidenceError, "CONTINUOUS_SUPPORT_SCORE_UNAVAILABLE"):
            metrics(original=variant("ORIGINAL", None), keep_target=variant("KEEP_TARGET", 1), drop_target=variant("DROP_TARGET", 1), controls=self.controls())
    def test_no_calibration_is_diagnostic_only(self):
        value=metrics(original=variant("ORIGINAL", 2), keep_target=variant("KEEP_TARGET", 1.5), drop_target=variant("DROP_TARGET", .1), controls=self.controls())
        self.assertEqual(self.certificate(value, None).status, "DIAGNOSTIC_ONLY")
    def test_calibration_policy_mismatch_rejected(self):
        value=metrics(original=variant("ORIGINAL", 2), keep_target=variant("KEEP_TARGET", 1.5), drop_target=variant("DROP_TARGET", .1), controls=self.controls())
        self.assertEqual(self.certificate(value, self.calibration(h("other"))).failure_reason, "CALIBRATION_POLICY_HASH_MISMATCH")
    def test_human_oracle_taint_propagates(self):
        value=metrics(original=variant("ORIGINAL", 2), keep_target=variant("KEEP_TARGET", 1.5), drop_target=variant("DROP_TARGET", .1), controls=self.controls())
        result=admit(policy_sha256=self.policy_sha, calibration=self.calibration(), value=value, evidence_quality=1, operator_quality=1, model_revision="r", prompt_hash=h("p"), evidence_program_hash=h("e"), control_set_hash=h("c"), intervention_renderer_hash=h("i"), provenance={"HUMAN_ORACLE"})
        self.assertEqual(result.failure_reason, "HUMAN_ORACLE_CONTAMINATION")
    def test_r1_forbids_r2_and_routes_cannot_be_best_of(self):
        with self.assertRaisesRegex(DifferentialEvidenceError, "R1_FORBIDS_R2"): choose_adapt_route(("STOP",), completed_rounds=2)
        with self.assertRaisesRegex(DifferentialEvidenceError, "ADAPT_ROUTE_AMBIGUOUS"): choose_adapt_route(("A", "B"), completed_rounds=0)
    def test_cache_identity_binds_policy_and_control_set(self):
        args=dict(policy_sha256=h("policy"), control_set_hash=h("control"), evidence_program_hash=h("program"), prompt_hash=h("prompt"), model_revision="r")
        self.assertNotEqual(differential_cache_key(**args), differential_cache_key(**{**args,"control_set_hash":h("different")}))

    def test_closed_pilot_check_is_read_only(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/"closure.json"; row={"status":"PILOT_CLOSED","source_artifacts_unchanged":True,"label_resolution_applied":True}
            row["closure_manifest_sha256"]=stable_hash(row); p.write_text(canonical_json(row))
            before=p.read_bytes(); self.assertEqual(verify_closed_pilot(p)["status"], "PILOT_CLOSED"); self.assertEqual(before,p.read_bytes())

    def test_oracle_dry_run_reads_runtime_not_truth(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); runtime=root/"runtime.jsonl"
            runtime.write_text(canonical_json({"case_id":"case","source_video_id":"video","claim":"claim","claim_type":"SPATIAL_RELATION","evidence_references":[],"intervention_plan":{},"provenance":[],"control_tier":"TIER_1_EXACT"})+"\n")
            result=prepare_runtime(runtime_manifest=runtime, policy_path=self.policy_path, calibration_artifact=None, output_root=root/"out", cache_root=root/"cache", model_path=None, dry_run=True)
            self.assertEqual(result["status"], "ORACLE_DIAGNOSTIC_ONLY"); self.assertFalse(result["gt_used"]); self.assertEqual(result["model_calls_made"], 0)

    def test_policy_declares_postcondition_prerequisites_and_no_backend(self):
        policy, _ = load_policy(self.policy_path)
        self.assertEqual(set(policy["postcondition_requirements"]), {"INTERACTION_END_LOCALIZED", "POST_WINDOW_AVAILABLE", "OPERATOR_SUPPORT_ABSENT"})
        source = (Path(__file__).resolve().parents[1] / "src/relive/v2/differential_evidence.py").read_text()
        self.assertNotIn("make_backend", source)


if __name__ == "__main__": unittest.main()
