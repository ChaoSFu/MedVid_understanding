from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from relive.storage.artifacts import canonical_json
from relive.v2.protocol_dev_cases import (
    ProtocolDevCaseError, apply_human_completion, assert_automatic_certificate_input_safe, audit_cases,
    copesd_timebase_audit, ego_candidate_path_resolution, materialize_source_records,
    normalize_draft, prepare_frame_binding_completion_queue, prepare_human_completion,
    resolve_frame_patterns,
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

    def test_source_materialization_hashes_only_public_fields_and_fails_closed_when_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); cases, _ = self.normalize(root)
            case = cases / "PD-S-05.json"; value = json.loads(case.read_text())
            value["source"]["source_sample_ids"] = ["source-1"]; value["source"]["source_record_hints"] = []
            case.write_text(canonical_json(value) + "\n")
            qa = root / "qa.json"; qa.write_text(canonical_json([{
                "id": "source-1", "qa_type": "public", "dataset_name": "X", "video": ["f.jpg"], "sampled_video_frames": [0],
                "conversations": [{"from": "human", "value": "public question"}, {"from": "gpt", "value": "must not enter hash"}],
            }]) + "\n")
            result = materialize_source_records(cases_dir=cases, qa_path=qa, output_dir=root / "materialized")
            self.assertEqual(result["materialized_case_count"], 1)
            row = next(json.loads(line) for line in (root / "materialized/protocol_dev_source_record_materialization.jsonl").read_text().splitlines() if json.loads(line)["case_id"] == "PD-S-05")
            self.assertEqual(row["status"], "MATERIALIZED")
            self.assertNotIn("must not enter hash", canonical_json(row))

    def test_source_materialization_rejects_ambiguous_selector(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); cases, _ = self.normalize(root); case = cases / "PD-S-05.json"; value = json.loads(case.read_text())
            value["source"]["source_sample_ids"] = ["shared"]; value["source"]["source_record_hints"] = []
            case.write_text(canonical_json(value) + "\n")
            qa = root / "qa.json"; qa.write_text(canonical_json([
                {"id": "shared", "qa_type": "public", "dataset_name": "X", "video": ["a.jpg"], "sampled_video_frames": [0], "conversations": []},
                {"id": "shared", "qa_type": "public", "dataset_name": "X", "video": ["b.jpg"], "sampled_video_frames": [0], "conversations": []},
            ]) + "\n")
            materialize_source_records(cases_dir=cases, qa_path=qa, output_dir=root / "materialized")
            row = next(json.loads(line) for line in (root / "materialized/protocol_dev_source_record_materialization.jsonl").read_text().splitlines() if json.loads(line)["case_id"] == "PD-S-05")
            self.assertEqual((row["status"], row["reason_code"]), ("INVALID", "SOURCE_RECORD_AMBIGUOUS"))

    def test_human_queue_excludes_machine_hashes_and_oracle_templates_have_no_coordinates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); cases, _ = self.normalize(root)
            result = prepare_human_completion(cases_dir=cases, reviews_dir=root / "reviews", oracle_dir=root / "oracle")
            self.assertEqual(result["oracle_template_count"], 9)
            row = json.loads((root / "reviews/protocol_dev_human_completion_queue.jsonl").read_text().splitlines()[0])
            self.assertNotIn("source_record_canonical_hashes", row["human_required"])
            oracle = json.loads((root / "oracle/PD-S-05.oracle.json").read_text())
            self.assertEqual(oracle["coordinates"], None); self.assertFalse(oracle["runtime_exposed"])

    def test_completed_human_queue_applies_review_fields_without_oracle_or_unbound_timebase(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); cases, _ = self.normalize(root)
            queue_dir, oracle = root / "reviews", root / "oracle"
            prepare_human_completion(cases_dir=cases, reviews_dir=queue_dir, oracle_dir=oracle)
            queue = queue_dir / "protocol_dev_human_completion_queue.jsonl"
            rows = [json.loads(line) for line in queue.read_text().splitlines()]
            for row in rows:
                row["current_decision"] = "ADMIT"; row["current_rationale"] = "human reviewed"; row["reviewer_id"] = "reviewer-1"
                if not row["current_keyframe_ids"]:
                    row["current_keyframe_ids"] = [1683, 1687, 1691] if row["case_id"] == "PD-P-CholecT50-VID68-GBPACK-01" else [1434, 1437, 1441]
            queue.write_text("".join(canonical_json(row) + "\n" for row in rows))
            applied = apply_human_completion(cases_dir=cases, completion_queue=queue, output_dir=root / "applied")
            self.assertEqual(applied["case_count"], 9); self.assertEqual(applied["keyframe_binding_pending_count"], 5)
            reviewed = json.loads((root / "applied/cases/PD-S-08.json").read_text())
            self.assertEqual(reviewed["human_review"]["reviewer_id"], "reviewer-1")
            self.assertEqual(reviewed["frame_locator"]["keyframe_ids"], [])
            self.assertEqual(reviewed["oracle_annotation"], {"artifact_ref": None, "automatic_certificate_access": "FORBIDDEN", "coordinates_present": False, "status": "PENDING"})

    def test_completed_human_queue_rejects_oracle_payload_and_case_set_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); cases, _ = self.normalize(root)
            queue_dir, oracle = root / "reviews", root / "oracle"
            prepare_human_completion(cases_dir=cases, reviews_dir=queue_dir, oracle_dir=oracle)
            queue = queue_dir / "protocol_dev_human_completion_queue.jsonl"
            rows = [json.loads(line) for line in queue.read_text().splitlines()]
            rows[0]["coordinates"] = [[0, 0, 1, 1]]
            queue.write_text("".join(canonical_json(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ProtocolDevCaseError, "HUMAN_ORACLE"):
                apply_human_completion(cases_dir=cases, completion_queue=queue, output_dir=root / "blocked")

    def test_frame_resolver_requires_unambiguous_existing_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); cases, _ = self.normalize(root); case = cases / "PD-S-05.json"; value = json.loads(case.read_text())
            value["frame_locator"].update({"dataset_root_key": "TEST_ROOT", "keyframe_ids": [1, 2], "sampled_frame_ids": []})
            case.write_text(canonical_json(value) + "\n")
            frames = root / "frames/VID110"; frames.mkdir(parents=True); (frames / "0001.jpg").write_bytes(b"one"); (frames / "0002.jpg").write_bytes(b"two")
            roots = root / "roots.json"; roots.write_text(canonical_json({"TEST_ROOT": str(root / "frames")}) + "\n")
            result = resolve_frame_patterns(cases_dir=cases, data_roots=roots, output_dir=root / "resolved")
            self.assertEqual(result["resolved_count"], 1)
            row = next(json.loads(line) for line in (root / "resolved/protocol_dev_frame_pattern_resolution.jsonl").read_text().splitlines() if json.loads(line)["case_id"] == "PD-S-05")
            self.assertEqual(row["derived_relative_path_pattern"], "VID110/{frame:04d}.jpg")

    def test_frame_resolver_strictly_fails_until_every_case_is_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); cases, _ = self.normalize(root); roots = root / "roots.json"; roots.write_text(canonical_json({"CHOLECTRACK20_ROOT": str(root)}) + "\n")
            with self.assertRaisesRegex(ProtocolDevCaseError, "FRAME_PATH_BINDING_INCOMPLETE"):
                resolve_frame_patterns(cases_dir=cases, data_roots=roots, output_dir=root / "resolution", strict=True)
            self.assertTrue((root / "resolution/protocol_dev_frame_pattern_resolution.jsonl").is_file())

    def test_ego_candidates_and_copesd_timebase_remain_reviewable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); cases, _ = self.normalize(root); frames = root / "frames"; ego = frames / "Ego/06_1"; ego.mkdir(parents=True)
            for frame in [529, 533, 537, 541, 545, 549, 553, 557]: (ego / f"06_1_{frame:04d}.jpg").write_bytes(str(frame).encode())
            copesd = frames / "CoPESD/012626"; copesd.mkdir(parents=True); (copesd / "0001.jpg").write_bytes(b"image")
            roots = root / "roots.json"; roots.write_text(canonical_json({"EGOSURGERY_ROOT": str(frames), "COPESD_ROOT": str(frames)}) + "\n")
            ego_result = ego_candidate_path_resolution(cases_dir=cases, data_roots=roots, output_dir=root / "ego")
            self.assertEqual(ego_result["common_candidate_directories"], ["Ego/06_1"])
            registry = root / "registry.json"; registry.write_text(canonical_json({"dataset": "CoPESD", "video_id": "012626", "dataset_root_key": "COPESD_ROOT", "relative_path_pattern": "CoPESD/012626/{frame}.jpg", "public_frame_file_ids": [1]}) + "\n")
            pending = copesd_timebase_audit(cases_dir=cases, data_roots=roots, timebase_manifest=None, public_frame_registry=registry, output_dir=root / "copesd_pending")
            self.assertEqual((pending["status"], pending["reason_code"]), ("PENDING", "TIMEBASE_PROVENANCE_REQUIRED"))
            timebase = root / "timebase.jsonl"; timebase.write_text(canonical_json({"video_id": "012626", "frame_id": 1, "timestamp_seconds": 1440.0, "relative_path": "CoPESD/012626/0001.jpg", "timebase_source": "DOCUMENTED_SOURCE_FRAME_INDEX_AND_FPS", "source_reference_sha256": "a" * 64}) + "\n")
            copesd_result = copesd_timebase_audit(cases_dir=cases, data_roots=roots, timebase_manifest=timebase, public_frame_registry=registry, output_dir=root / "copesd")
            self.assertEqual(copesd_result["status"], "READY_FOR_HUMAN_KEYFRAME_SELECTION")

    def test_frame_binding_queue_does_not_select_poststate_or_copesd_keyframes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); cases, _ = self.normalize(root)
            result = prepare_frame_binding_completion_queue(cases_dir=cases, output_dir=root / "queue")
            self.assertEqual(result["queue_count"], 4)
            rows = [json.loads(line) for line in (root / "queue/protocol_dev_frame_binding_completion_queue.jsonl").read_text().splitlines()]
            post = next(row for row in rows if row["case_id"] == "PD-P-CholecT50-VID68-GBPACK-01")
            self.assertEqual(post["strongest_poststate_frame"], 1687)
            self.assertNotIn("keyframe_ids", post)

    def test_frame_binding_queue_exposes_only_author_bound_copesd_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); cases, _ = self.normalize(root)
            case = cases / "PD-S-08.json"; value = json.loads(case.read_text())
            value["frame_locator"].update({"evidence_frame_range": [10, 14], "sampled_frame_ids": [10, 12, 14]})
            value["temporal_spec"]["source_timebase"] = "AUTHOR_CONFIRMED_DOCUMENTED_SOURCE_FRAME_INDEX_AND_FPS"
            case.write_text(canonical_json(value) + "\n")
            prepare_frame_binding_completion_queue(cases_dir=cases, output_dir=root / "queue")
            rows = [json.loads(line) for line in (root / "queue/protocol_dev_frame_binding_completion_queue.jsonl").read_text().splitlines()]
            copesd = next(row for row in rows if row["case_id"] == "PD-S-08")
            self.assertEqual(copesd["status"], "PENDING_HUMAN_KEYFRAME_SELECTION_FROM_BOUND_MAPPING")
            self.assertEqual(copesd["allowed_frame_ids"], [10, 12, 14])


if __name__ == "__main__":
    unittest.main()
