from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relive.storage.artifacts import canonical_json
from relive.v2.protocol_dev_cases import (
    ProtocolDevCaseError, assert_automatic_certificate_input_safe, audit_cases,
    normalize_draft,
)


ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT.parent / "outputs" / "9个 case.json"


class ProtocolDevCaseTests(unittest.TestCase):
    def normalize(self, root: Path) -> tuple[Path, Path]:
        cases, manifest = root / "cases", root / "manifests" / "protocol_dev_manifest.jsonl"
        result = normalize_draft(draft_path=DRAFT, cases_dir=cases, manifest_path=manifest)
        self.assertEqual(result["case_count"], 9)
        return cases, manifest

    def audit(self, root: Path, cases: Path, **kwargs):
        return audit_cases(cases_dir=cases, oracle_dir=kwargs.get("oracle_dir"), data_roots=kwargs.get("data_roots"), output_dir=root / "audit", strict=kwargs.get("strict", False))

    def test_nine_draft_objects_normalize_to_portable_per_case_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); cases, manifest = self.normalize(root)
            self.assertEqual(len(list(cases.glob("*.json"))), 9)
            self.assertEqual(len(manifest.read_text().splitlines()), 9)
            first = json.loads((cases / "PD-C-EGO-01.json").read_text())
            self.assertNotIn("/root/data", canonical_json(first))
            self.assertNotIn("oracle_evidence", canonical_json(first))
            self.assertEqual(first["oracle_annotation"]["artifact_ref"], None)

    def test_schema_only_audit_reports_balanced_cases_pending_and_no_activity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); cases, _ = self.normalize(root); result = self.audit(root, cases)
            self.assertEqual(result["case_count"], 9)
            self.assertEqual(result["claim_type_counts"], {"CONTACT_ACTION": 3, "POSTCONDITION_PERSISTENCE": 3, "SPATIAL_RELATION": 3})
            self.assertEqual(result["case_readiness_counts"], {"READY": 0, "PENDING": 8, "INVALID": 1})
            self.assertEqual((result["model_calls_made"], result["cache_writes"], result["certificate_writes"], result["new_verified_count"]), (0, 0, 0, 0))

    def test_strict_readiness_fails_closed_for_pending_drafts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); cases, _ = self.normalize(root)
            with self.assertRaisesRegex(ProtocolDevCaseError, "STRICT_READINESS_NOT_READY"):
                self.audit(root, cases, strict=True)

    def test_duplicate_case_and_pilot_video_are_invalid(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); cases, _ = self.normalize(root)
            first = cases / "PD-S-05.json"; duplicate = cases / "duplicate.json"; duplicate.write_bytes(first.read_bytes())
            report = self.audit(root, cases)
            issues = {issue for row in report["rows"] for issue in row["missing_or_issue_fields"]}
            self.assertIn("DUPLICATE_CASE_ID", issues)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); cases, _ = self.normalize(root); case = cases / "PD-S-05.json"; value = json.loads(case.read_text()); value["source"]["dataset"] = "NurViD"; value["source"]["video_id"] = "vR0_BaXYcE4"; case.write_text(canonical_json(value) + "\n")
            report = self.audit(root, cases); issues = {issue for row in report["rows"] for issue in row["missing_or_issue_fields"]}
            self.assertIn("PILOT_VIDEO_LEAKAGE", issues)

    def test_matching_and_frame_window_errors_are_not_repaired(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); cases, _ = self.normalize(root); case = cases / "PD-S-05.json"; value = json.loads(case.read_text())
            value["claims"][1]["matched_to_claim_id"] = "missing"; value["frame_locator"]["evidence_frame_range"] = [10, 20]; value["frame_locator"]["keyframe_ids"] = [21]
            case.write_text(canonical_json(value) + "\n")
            report = self.audit(root, cases); row = next(item for item in report["rows"] if item["case_id"] == "PD-S-05")
            self.assertEqual(row["readiness"], "INVALID"); self.assertIn("MATCHED_TO_CLAIM_ID_INVALID", row["missing_or_issue_fields"]); self.assertIn("KEYFRAME_OUTSIDE_EVIDENCE_WINDOW", row["missing_or_issue_fields"])

    def test_interaction_and_postcondition_overlap_is_invalid(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); cases, _ = self.normalize(root); case = cases / "PD-P-02.json"; value = json.loads(case.read_text())
            value["temporal_spec"]["interaction_frame_range"] = [10, 20]; value["temporal_spec"]["after_frame_range"] = [20, 30]
            case.write_text(canonical_json(value) + "\n")
            report = self.audit(root, cases); row = next(item for item in report["rows"] if item["case_id"] == "PD-P-02")
            self.assertEqual(row["readiness"], "INVALID"); self.assertIn("INTERACTION_POSTCONDITION_FRAME_RANGE_CONFLICT", row["missing_or_issue_fields"])

    def test_sampled_frame_continuity_and_explicit_source_mismatch_are_invalid(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); cases, _ = self.normalize(root); case = cases / "PD-S-05.json"; value = json.loads(case.read_text())
            value["frame_locator"]["coverage_semantics"] = "DISCRETE_SAMPLED_FRAMES"
            value["claims"][0]["claim_text"] = "The relation is visible in all continuous frames."
            value["source"]["source_sample_ids"] = ["another-video&&0.0&&1.0"]
            case.write_text(canonical_json(value) + "\n")
            report = self.audit(root, cases); row = next(item for item in report["rows"] if item["case_id"] == "PD-S-05")
            self.assertEqual(row["readiness"], "INVALID")
            self.assertIn("SAMPLED_FRAME_CLAIM_CONTINUITY_CONFLICT", row["missing_or_issue_fields"])
            self.assertIn("SOURCE_RECORD_VIDEO_MISMATCH", row["missing_or_issue_fields"])

    def test_oracle_isolation_and_optional_frame_existence_audit(self):
        with self.assertRaisesRegex(ProtocolDevCaseError, "HUMAN_ORACLE"):
            assert_automatic_certificate_input_safe({"oracle_annotation": {"artifact_ref": "x"}})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); cases, _ = self.normalize(root); case = cases / "PD-C-EGO-01.json"; value = json.loads(case.read_text())
            value["frame_locator"]["relative_path_pattern"] = "frames/{frame:04d}.jpg"; value["frame_locator"]["dataset_root_key"] = "TEST_ROOT"; value["frame_locator"]["sampled_frame_ids"] = [1]; value["frame_locator"]["evidence_frame_range"] = [1, 1]
            case.write_text(canonical_json(value) + "\n"); roots = root / "roots.json"; roots.write_text(canonical_json({"TEST_ROOT": str(root / "missing")}) + "\n")
            report = self.audit(root, cases, data_roots=roots); row = next(item for item in report["rows"] if item["case_id"] == "PD-C-EGO-01")
            self.assertIn("FRAME_PATH_MISSING", row["missing_or_issue_fields"])

    def test_schema_declares_exactly_three_oneof_claim_types(self):
        schema = json.loads((ROOT / "schemas/v2/relive_v2_protocol_dev_case.schema.json").read_text())
        one_of = {item["properties"]["task"]["properties"]["claim_type"]["const"] for item in schema["oneOf"]}
        self.assertEqual(one_of, {"SPATIAL_RELATION", "CONTACT_ACTION", "POSTCONDITION_PERSISTENCE"})


if __name__ == "__main__":
    unittest.main()
