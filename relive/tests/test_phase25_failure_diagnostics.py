"""GPU-free checks for the read-only Phase 2.5 diagnostics."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from relive.failure_diagnostics import (classify_control_count_mismatch,
                                        diagnose_phase2_failures,
                                        refinement_eligibility)
from relive.interventions import audit_control_geometry, generate_control_regions


class Phase25FailureDiagnosticsTests(unittest.TestCase):
    def test_control_count_mismatch_distinguishes_documented_unavailability(self):
        kind, reason = classify_control_count_mismatch(
            expected_count=1, generated_count=0, inference_result_count=0,
            controls_status="CONTROL_UNAVAILABLE")
        self.assertEqual(kind, "B_LEGITIMATE_PROTOCOL_UNAVAILABLE")
        self.assertEqual(reason, "GEOMETRY_GENERATED_FEWER_THAN_FROZEN_REQUIRED_CONTROLS")
        kind, _ = classify_control_count_mismatch(
            expected_count=1, generated_count=1, inference_result_count=0,
            controls_status="CONTROL_AVAILABLE")
        self.assertEqual(kind, "A_IMPLEMENTATION_OR_ARTIFACT_BUG")

    def test_control_geometry_audit_keeps_existing_control_selection(self):
        region = [0.1, 0.1, 0.4, 0.4]
        core = generate_control_regions(region, 1)
        audit = audit_control_geometry(region, 1)
        self.assertEqual(audit["regions"], core["regions"])
        self.assertEqual(audit["status"], core["status"])
        self.assertTrue(any(row["state"] == "SELECTED" for row in audit["candidate_placements"]))

    def test_original_insufficient_prohibits_spatial_refinement(self):
        result = refinement_eligibility(["ORIGINAL_INSUFFICIENT"])
        self.assertEqual(result["recommended_action"], "TEMPORAL_REACQUIRE")
        self.assertFalse(result["refinement_eligible"])

    def test_dependence_unresolved_allows_spatial_refinement(self):
        result = refinement_eligibility(["DEPENDENCE_UNRESOLVED"])
        self.assertEqual(result["recommended_action"], "SPATIAL_REFINE")
        self.assertTrue(result["refinement_eligible"])

    def test_technical_error_never_enters_semantic_refinement(self):
        result = refinement_eligibility(["TECHNICAL_FAILURE", "DEPENDENCE_UNRESOLVED"])
        self.assertEqual(result["root_cause_class"], "TECHNICAL_ERROR")
        self.assertFalse(result["refinement_eligible"])

    def test_read_only_artifact_audit_reports_protocol_unavailability_separately(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, cache, output = root / "run", root / "cache", root / "diagnostic"
            (run / "samples").mkdir(parents=True)
            (run / "events" / "controls").mkdir(parents=True)
            cache.mkdir()
            proposal = {"proposal_id": "proposal-1", "support_region": [0.1, 0.1, 0.8, 0.8],
                        "frame_mapping": {"frame_ids": ["f0"]}}
            cert = {
                "certificate_id": "certificate-1", "candidate_id": "candidate-1", "final_status": "UNCERTAIN",
                "failure_reasons": ["CONTROL_UNAVAILABLE", "CONTROL_RESULT_COUNT_MISMATCH"],
                "checks": {"semantic": {"references": {"input_references": {"frame_ids": ["f0"], "image_paths": []}}},
                           "spatial": {"proposal": proposal, "expected_control_count": 1,
                                       "support_area_fraction": 0.49,
                                       "references": {"controls": []}}},
            }
            sample = {"sample_id": "qa-1", "certificates": [cert],
                      "candidate_traversal": {"trace": [{"candidate_id": "candidate-1", "candidate_rank": 0}]}}
            (run / "samples" / "sample.json").write_text(json.dumps(sample), encoding="utf-8")
            controls = {"sample_id": "qa-1", "proposal_id": "proposal-1", "status": "CONTROL_UNAVAILABLE",
                        "requested_count": 1, "valid_count": 0, "regions": []}
            (run / "events" / "controls" / "controls.json").write_text(json.dumps(controls), encoding="utf-8")
            report = diagnose_phase2_failures(run, output, cache)
            row = json.loads((output / "phase2_failure_diagnostics.jsonl").read_text().splitlines()[0])
            self.assertEqual(report["model_calls_made"], 0)
            self.assertEqual(row["control_count_audit"]["classification"], "B_LEGITIMATE_PROTOCOL_UNAVAILABLE")
            self.assertEqual(row["root_cause_class"], "TECHNICAL_DIAGNOSTIC")
            self.assertFalse(row["refinement_eligible"])
            self.assertTrue((output / "phase2_failure_diagnostics.md").is_file())
            self.assertTrue((output / "phase2_failure_diagnostics.md").read_text().startswith("# ReliVE Phase 2.5"))


if __name__ == "__main__":
    unittest.main()
