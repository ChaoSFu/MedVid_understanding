"""CPU-only contract tests for the isolated Phase 4A-0 review packet."""
from __future__ import annotations
import hashlib, json, tempfile, unittest
from pathlib import Path
from PIL import Image

from relive.phase4a0 import (ELIGIBLE, Phase4A0Error, SPEC_FORMAT, prepare,
                             sha, claim_sha256, validate_reviews, export_historical_pilots)


class Phase4A0Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.frame = self.root / "public.png"; Image.new("RGB", (40, 30), (40, 80, 120)).save(self.frame)
        self.spec = {"format": SPEC_FORMAT, "audit_case_id": "pilot-001", "source_claim_id": "phase35-local-001", "claim_text": "A visible white object is centered.", "source_record_index": 2, "public_record_sha256": "publichash", "frame_orders": [11], "frame_paths": [str(self.frame)], "frame_sha256": [hashlib.sha256(self.frame.read_bytes()).hexdigest()], "frozen_support_region": [.2, .2, .7, .8], "coordinate_system": "normalized_0_1_xyxy", "intervention": {"operator": "opaque_gray", "operator_version": "relive-opaque-gray-hard-mask-v1", "parameters": {"fill_rgb": [127,127,127]}}, "matched_control_regions": [[.0,.0,.5,.6]], "candidate_manifest_sha256": "frozen", "historical_pilot": True, "development_control": False, "gt_used": False}
        self.spec["claim_sha256"] = claim_sha256(self.spec["claim_text"])
        self.manifest = self.root / "candidates.jsonl"; self.manifest.write_text(json.dumps(self.spec) + "\n")
        self.out = self.root / "out"

    def tearDown(self): self.temp.cleanup()

    def _review(self, reviewer: str, status="INSUFFICIENT"):
        template = [json.loads(x) for x in (self.out / "review_template.jsonl").read_text().splitlines()]
        row = template[0]; row["reviewer_id"] = reviewer
        letters = row["variant_assessments"]
        mapping = json.loads((self.out / "blind_mapping.json").read_text())["cases"][row["case_blind_id"]]["variants"]
        inverse = {value:key for key,value in mapping.items()}
        letters[inverse["ORIGINAL"]] = "SUPPORTED"; letters[inverse["KEEP_TARGET"]] = "SUPPORTED"; letters[inverse["DROP_TARGET"]] = status; letters[inverse["DROP_MATCHED_CONTROL"]] = "SUPPORTED"
        row.update({"claim_wording":"CLEAR", "temporal_scope":"VALID", "observability":"DIRECTLY_VISIBLE", "eligibility_label":"SINGLE_ROI_ELIGIBLE"})
        path = self.root / f"{reviewer}.jsonl"; path.write_text(json.dumps(row)+"\n"); return path

    def test_prepare_is_zero_model_blind_and_seed_stable(self):
        report = prepare(self.manifest, self.out, seed=9)
        self.assertEqual(report["model_calls_made"], 0); self.assertFalse(report["backend_loaded"])
        packet = (self.out / "review_packet" / "case-001.json").read_text()
        self.assertNotIn("pilot-001", packet); self.assertNotIn("ORIGINAL", packet)
        self.assertTrue((self.out / "new_calibration_candidate_manifest.template.json").is_file())
        with self.assertRaisesRegex(Phase4A0Error, "OUTPUT_DIRECTORY_MUST_BE_EMPTY"): prepare(self.manifest, self.out, seed=9)

    def test_each_required_variant_has_pixel_audit(self):
        prepare(self.manifest, self.out)
        mapping = json.loads((self.out / "blind_mapping.json").read_text())
        audits = next(iter(mapping["cases"].values()))["pixel_audits"]
        self.assertEqual(set(audits), {"ORIGINAL", "KEEP_TARGET", "DROP_TARGET", "DROP_MATCHED_CONTROL"})
        self.assertTrue(all(a["pixel_audit_pass"] for values in audits.values() for a in values))

    def test_packet_does_not_contain_source_or_outcome_terms(self):
        prepare(self.manifest, self.out)
        packet = "\n".join(path.read_text(errors="ignore") for path in (self.out / "review_packet").glob("*.json"))
        self.assertNotIn(self.spec["source_claim_id"], packet)
        self.assertNotIn("certificate", packet.casefold())
        self.assertNotIn("DROP_TARGET", packet)

    def test_mapping_is_separate_from_packet(self):
        prepare(self.manifest, self.out)
        self.assertTrue((self.out / "blind_mapping.json").is_file())
        self.assertFalse((self.out / "review_packet" / "blind_mapping.json").exists())

    def test_missing_public_frame_fails_closed(self):
        self.spec["frame_paths"] = [str(self.root / "gone.png")]
        bad = self.root / "missing.jsonl"; bad.write_text(json.dumps(self.spec) + "\n")
        with self.assertRaisesRegex(Phase4A0Error, "FROZEN_ARTIFACT_MISSING"):
            prepare(bad, self.root / "missingout")

    def test_tampered_frame_hash_fails_closed(self):
        self.spec["frame_sha256"] = ["0" * 64]
        bad = self.root / "hash.jsonl"; bad.write_text(json.dumps(self.spec) + "\n")
        with self.assertRaisesRegex(Phase4A0Error, "PUBLIC_FRAME_SHA256_MISMATCH"):
            prepare(bad, self.root / "hashout")

    def test_claim_hash_is_bound(self):
        self.spec["claim_sha256"] = "0" * 64
        bad = self.root / "claim.jsonl"; bad.write_text(json.dumps(self.spec) + "\n")
        with self.assertRaisesRegex(Phase4A0Error, "CLAIM_SHA256_MISMATCH"):
            prepare(bad, self.root / "claimout")

    def test_strict_review_awaiting_agreement_and_eligibility(self):
        prepare(self.manifest, self.out, seed=1)
        first = self._review("r1")
        self.assertEqual(validate_reviews(self.out, [first])["status"], "AWAITING_HUMAN_REVIEWS")
        second = self._review("r2")
        result = validate_reviews(self.out, [first, second])
        self.assertEqual(result["eligible_count"], 1)
        report = json.loads((self.out / "eligibility_report.jsonl").read_text())
        self.assertEqual(report["eligibility_status"], ELIGIBLE)

    def test_disagreement_and_fail_closed_input(self):
        prepare(self.manifest, self.out, seed=2)
        first, second = self._review("r1"), self._review("r2", "CONTRADICTED")
        self.assertEqual(validate_reviews(self.out, [first, second])["status"], "REQUIRES_ADJUDICATION")
        bad = dict(self.spec); bad["reference_answer"] = "no"; badpath = self.root / "bad.jsonl"; badpath.write_text(json.dumps(bad)+"\n")
        with self.assertRaisesRegex(Phase4A0Error, "GT_OR_HISTORICAL_OUTCOME_FIELD_FORBIDDEN"):
            prepare(badpath, self.root / "bad")

    def test_duplicate_reviewer_is_rejected(self):
        prepare(self.manifest, self.out)
        review = self._review("r1")
        with self.assertRaisesRegex(Phase4A0Error, "DUPLICATE_REVIEWER_ID"):
            validate_reviews(self.out, [review, review])

    def test_contradicted_drop_is_never_eligible(self):
        prepare(self.manifest, self.out)
        result = validate_reviews(self.out, [self._review("r1", "CONTRADICTED"), self._review("r2", "CONTRADICTED")])
        self.assertEqual(result["eligible_count"], 0)

    def test_unknown_reason_code_is_rejected(self):
        prepare(self.manifest, self.out); review = self._review("r1")
        row = json.loads(review.read_text()); row["reason_codes"] = ["GT_LEAK"] ; review.write_text(json.dumps(row)+"\n")
        with self.assertRaisesRegex(Phase4A0Error, "REVIEW_REASON_CODE_INVALID"):
            validate_reviews(self.out, [review])

    def test_wrong_packet_id_and_non_string_reviewer_are_rejected(self):
        prepare(self.manifest, self.out); review = self._review("r1")
        row = json.loads(review.read_text()); row["packet_id"] = "wrong"; review.write_text(json.dumps(row)+"\n")
        with self.assertRaisesRegex(Phase4A0Error, "REVIEW_PACKET_ID_MISMATCH"):
            validate_reviews(self.out, [review])
        row["packet_id"] = json.loads((self.out / "review_template.jsonl").read_text())["packet_id"]
        row["reviewer_id"] = 1; review.write_text(json.dumps(row)+"\n")
        with self.assertRaisesRegex(Phase4A0Error, "REVIEWER_ID_REQUIRED_AND_CONSISTENT"):
            validate_reviews(self.out, [review])

    def test_different_seed_only_changes_mapping_order(self):
        first, second = self.root / "one", self.root / "two"
        prepare(self.manifest, first, seed=1); prepare(self.manifest, second, seed=2)
        # Pixel artifacts are generated before blinding; their bytes are seed-independent.
        self.assertEqual(hashlib.sha256(next((first / "generated_images").glob("*.png")).read_bytes()).hexdigest(), hashlib.sha256(next((second / "generated_images").glob("*.png")).read_bytes()).hexdigest())

    def test_audit_never_reports_verified(self):
        report = prepare(self.manifest, self.out)
        self.assertEqual(report["new_verified_count"], 0)
        self.assertNotIn("VERIFIED", (self.out / "phase4a0_audit.json").read_text())

    def test_export_historical_uses_stored_r1_r0_and_never_reconstructs_local003(self):
        # The existing Phase 3.6 fixture gives local-001 a persisted R1 control;
        # its Phase 3.5 run gives local-002 a persisted R0 control.
        from test_phase36 import Phase36Tests
        fixture = Phase36Tests(methodName="test_routes_contract_and_replay"); fixture.setUp()
        try:
            from relive.phase36 import execute as phase36_execute, preflight as phase36_preflight
            p36 = fixture.root / "p36-export"
            phase36_preflight(config_path=fixture.config, runtime_path=fixture.runtime, prospective_manifest_path=fixture.manifest, phase35_v3_run_dir=fixture.v3 / "run", output_dir=p36, require_real=False)
            phase36_execute(config_path=fixture.config, runtime_path=fixture.runtime, prospective_manifest_path=fixture.manifest, phase35_v3_run_dir=fixture.v3 / "run", output_dir=p36, mode="run", require_real=False)
            rows = [json.loads(line) for line in fixture.runtime.read_text().splitlines()]
            source_rows = []
            for index, row in enumerate(rows):
                public = (chr(97 + index) * 64)
                row["metadata"]["source_record_sha256"] = public
                source_rows.append({"source_record_index": index + 2, "sample_id": row["sample_id"], "public_record_sha256": public,
                                    "target_claim": row["target_claim"], "claim_text_sha256": claim_sha256(row["target_claim"]["text"])})
            fixture.runtime.write_text("".join(json.dumps(row) + "\n" for row in rows))
            (fixture.runtime.parent / "fresh_source_manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in source_rows))
            out = fixture.root / "export"
            report = export_historical_pilots(runtime_path=fixture.runtime, prospective_manifest_path=fixture.manifest, phase35_v3_run_dir=fixture.v3 / "run", phase36_run_dir=p36 / "run", output_dir=out)
            self.assertEqual(report["candidate_count"], 2)
            by_claim = {row["source_claim_id"]: row for row in [json.loads(line) for line in (out / "phase4a0_historical_pilot_candidates.jsonl").read_text().splitlines()]}
            self.assertEqual(by_claim["phase35-local-001"]["frozen_support_region"], [0.08, 0.12, 0.36, 0.42])
            self.assertEqual(by_claim["phase35-local-002"]["frozen_support_region"], [0.396, 0.549, 0.522, 0.654])
            self.assertEqual(next(case for case in report["cases"] if case["source_claim_id"] == "phase35-local-003")["status"], "NO_FROZEN_AUDITABLE_ROI")
        finally: fixture.tearDown()

if __name__ == "__main__": unittest.main()
