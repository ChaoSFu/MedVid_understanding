from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from relive.backends.mock import MockBackend
from relive.claim_bank.contracts import ClaimBankError, QA_TYPES, parse, protocol
from relive.claim_bank.data import (audit_dataset, digest, load_json, load_runtime, project,
                                    resolve_path, rows, sample_frames, validate_runtime, write_once)
from relive.claim_bank.pipeline import _task_checks, generate, prompt_for, selection, validate_bank

ROOT = Path(__file__).resolve().parents[1]
CONFIG = str(ROOT / "configs/claim_bank_mock_v1.yaml")
REQ = {"requirement_id": "R1", "description": "Inspect the marker.", "type": "OBJECT", "evidence_type": "SPATIAL", "importance": "REQUIRED"}
CLAIM = {"claim_id": "H1", "text": "A marker is visible.", "type": "OBJECT", "evidence_type": "SPATIAL", "requirement_ids": ["R1"], "depends_on": [], "status": "UNVERIFIED", "generation_basis": "VISUAL_HYPOTHESIS", "question_quote": None, "frame_orders": [0], "target_alignment": "QUESTION_ALIGNED"}


class ClaimBankTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.image = self.root / "frame.png"
        Image.new("RGB", (20, 16), (20, 40, 60)).save(self.image)
        self.records = []
        for qa in QA_TYPES:
            for i in range(2):
                self.records.append({"id": "duplicate", "qa_type": qa, "dataset_name": "Fixture",
                                     "data_source": "Fixture", "video": [str(self.image)] * 3,
                                     "sampled_video_frames": [1, 2, 3],
                                     "metadata": {"video_id": "synthetic", "fps": "0.5", "cvs_scores": "GT_SENTINEL"},
                                     "conversations": [{"from": "human", "value": "Inspect the synthetic marker."},
                                                       {"from": "gpt", "value": "GT_SENTINEL"}],
                                     "struc_info": "GT_SENTINEL", "RC_info": {"bbox": "GT_SENTINEL"},
                                     "answer": "GT_SENTINEL"})
        self.source = self.root / "source.json"
        self.source.write_text(json.dumps(self.records))
        self.prepared = self.root / "prepared"
        audit_dataset(str(self.source), str(self.prepared), expected_count=22, prepare=True)
        self.runtime = str(self.prepared / "runtime_manifest.jsonl")

    def run_stage(self, stage, name, **kwargs):
        out = self.root / name
        summary = generate(runtime_manifest=self.runtime, output_dir=str(out), stage=stage,
                           config_path=CONFIG, cache_dir=str(self.root / "cache"), **kwargs)
        return out, summary

    def test_audit_count_duplicates_and_distribution(self):
        audit = load_json(self.prepared / "dataset_audit.json")
        self.assertEqual(audit["record_count"], 22)
        self.assertEqual(len(audit["qa_type_counts"]), 11)
        self.assertEqual(len(audit["duplicate_original_ids"][0]["source_record_indexes"]), 22)
        self.assertEqual(audit["model_calls_made"], 0)

    def test_expected_count_fail_closed(self):
        with self.assertRaisesRegex(ClaimBankError, "COUNT_MISMATCH"):
            audit_dataset(str(self.source), str(self.root / "wrong"), expected_count=6270, prepare=True)

    def test_projection_never_accesses_forbidden_values(self):
        class Guard(dict):
            def get(self, key, default=None):
                if key in {"struc_info", "RC_info", "answer", "cvs_scores"} or (key == "value" and super().get("from") == "gpt"):
                    raise AssertionError("forbidden access")
                return super().get(key, default)
        r = Guard(deepcopy(self.records[0]))
        r["metadata"] = Guard(r["metadata"])
        r["conversations"] = [Guard(t) for t in r["conversations"]]
        result = project(r, 0, "a" * 64)
        self.assertNotIn("GT_SENTINEL", json.dumps(result))
        self.assertEqual(result["original_sample_id"], "duplicate")

    def test_full_gt_sentinel_dataflow_both_stages(self):
        from relive.config import load_config
        captured = []
        class Capture(MockBackend):
            def infer(inner, request):
                captured.append(deepcopy(request))
                if request["stage"] == "claimgen_stage1":
                    self.assertEqual(request["image_paths"], [])
                else:
                    self.assertTrue(request["image_paths"])
                    with Image.open(request["image_paths"][0]) as image:
                        self.assertEqual(image.getpixel((0, 0)), (20, 40, 60))
                return super().infer(request)
        backend = Capture(load_config(CONFIG)["backend"])
        s1, _ = self.run_stage(1, "s1", _backend=backend)
        s2, summary = self.run_stage(2, "s2", stage1_dir=str(s1), _backend=backend)
        self.assertEqual(summary["sample_count"], 22)
        self.assertEqual(summary["generation_status_counts"], {"GENERATED": 18, "UNRESOLVED_TARGET_REFERENCE": 4})
        self.assertTrue(summary["synthetic_test"])
        for request in captured:
            self.assertNotIn("GT_SENTINEL", json.dumps(request))
        for p in [*self.prepared.rglob("*.json*"), *s1.rglob("*.json*"), *s2.rglob("*.json*")]:
            self.assertNotIn("GT_SENTINEL", p.read_text())
        self.assertEqual(validate_bank(s2)["status"], "PASS")
        self.assertTrue(all(c["status"] == "UNVERIFIED" for c in rows(s2 / "claims.jsonl")))

    def test_hash_identity_not_original_id(self):
        rs = load_runtime(self.runtime)
        self.assertEqual(len({r["sample_key"] for r in rs}), 22)
        self.assertNotEqual(project(self.records[0], 0, "a" * 64)["sample_key"], project(self.records[0], 0, "b" * 64)["sample_key"])

    def test_runtime_extra_field_and_hash_tamper(self):
        r = load_runtime(self.runtime)[0]
        r["RC_info"] = "forbidden"
        with self.assertRaisesRegex(ClaimBankError, "ALLOWLIST"):
            validate_runtime(r)
        r.pop("RC_info")
        r["question"] = "changed"
        with self.assertRaisesRegex(ClaimBankError, "HASH"):
            validate_runtime(r)

    def test_runtime_sidecar_binding(self):
        p = Path(self.runtime)
        p.write_text(p.read_text() + "\n")
        with self.assertRaisesRegex(ClaimBankError, "PROVENANCE"):
            load_runtime(self.runtime)

    def test_sampling_dedup_order_and_unknown_time(self):
        r = load_runtime(self.runtime)[0]
        s = sample_frames(r, [REQ], 12)
        self.assertEqual(len(s["selected_frames"]), 1)
        frame = s["selected_frames"][0]
        self.assertEqual(frame["source_list_indices"], [0, 1, 2])
        self.assertEqual(frame["sampled_video_frames"], [1, 2, 3])
        self.assertIsNone(frame["timestamp_seconds"])
        self.assertEqual(frame["sha256"], digest(self.image))
        self.assertEqual(s, sample_frames(r, [REQ], 12))

    def test_uniform_temporal_coverage_budget(self):
        r = load_runtime(self.runtime)[0]
        r["video"] = [str(self.root / f"{i}.png") for i in range(100)]
        r["sampled_video_frames"] = list(range(100))
        s = sample_frames(r, [{**REQ, "type": "ACTION", "evidence_type": "TEMPORAL"}], 12)
        selected = [f["frame_order"] for f in s["selected_frames"]]
        self.assertEqual(len(selected), 12)
        self.assertEqual(selected[0], 0)
        self.assertEqual(selected[-1], 99)
        self.assertTrue(any(b == a + 1 for a, b in zip(selected, selected[1:])))

    def test_path_mapping_no_replacement_or_escape(self):
        self.assertEqual(resolve_path("/root/data/a/b.jpg", str(self.root)), self.root / "a/b.jpg")
        with self.assertRaises(ClaimBankError):
            resolve_path("/root/data/../../secret", str(self.root))

    def test_missing_frames_no_stage2_inference(self):
        s1, _ = self.run_stage(1, "s1")
        self.image.unlink()
        s2, summary = self.run_stage(2, "s2", stage1_dir=str(s1))
        self.assertEqual(summary["model_calls_made"], 0)
        self.assertEqual(summary["generation_status_counts"], {"MISSING_FRAMES": 22})

    def test_stage1_no_image_access_and_dry_run_no_backend(self):
        with patch("relive.claim_bank.pipeline.sample_frames", side_effect=AssertionError("image access")), patch("relive.backends.make_backend", side_effect=AssertionError("model load")):
            _, summary = self.run_stage(1, "dry", dry_run=True)
        self.assertEqual(summary["status"], "DRY_RUN")
        self.assertEqual(summary["model_calls_made"], 0)
        self.assertFalse((self.root / "cache").exists())

    def test_real_config_contract(self):
        from relive.config import load_config
        b = load_config(ROOT / "configs/qwen35_9b_claim_bank_v1.yaml")["backend"]
        self.assertIs(b["generation"]["do_sample"], False)
        self.assertEqual(b["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(b["max_retries"], 0)

    def test_replay_byte_exact_no_model_load(self):
        s1, _ = self.run_stage(1, "s1")
        s2, _ = self.run_stage(2, "s2", stage1_dir=str(s1))
        raw_before = sorted((p, p.read_bytes()) for p in (self.root / "cache/raw").rglob("*.json"))
        with patch("relive.backends.make_backend", side_effect=AssertionError("model load")):
            replay, summary = self.run_stage(2, "replay", stage1_dir=str(s1), replay_only=True, replay_from=str(s2))
        self.assertEqual(summary["model_calls_made"], 0)
        for name in ("claim_bank.jsonl", "requirements.jsonl", "claims.jsonl", "frame_sampling_manifest.jsonl", "generation_failures.jsonl"):
            self.assertEqual((s2 / name).read_bytes(), (replay / name).read_bytes())
        self.assertEqual(raw_before, sorted((p, p.read_bytes()) for p in (self.root / "cache/raw").rglob("*.json")))

    def test_replay_miss_no_calls(self):
        s1, _ = self.run_stage(1, "s1")
        with patch("relive.storage.cache.CachedInference.has_terminal_result", return_value=False), patch("relive.claim_bank.pipeline.ReplayBackend.infer", side_effect=AssertionError("inference")):
            with self.assertRaisesRegex(ClaimBankError, "CACHE_MISS"):
                self.run_stage(1, "replay", replay_only=True, replay_from=str(s1))

    def test_completed_resume_no_calls_and_frame_tamper(self):
        s1, _ = self.run_stage(1, "s1")
        _, summary = self.run_stage(2, "s2", stage1_dir=str(s1))
        with patch("relive.backends.make_backend", side_effect=AssertionError("load")):
            _, resumed = self.run_stage(2, "s2", stage1_dir=str(s1), resume=True)
        self.assertEqual(summary, resumed)
        Image.new("RGB", (20, 16), (0, 0, 0)).save(self.image)
        with self.assertRaisesRegex(ClaimBankError, "FRAME_HASH_MISMATCH"):
            self.run_stage(2, "s2", stage1_dir=str(s1), resume=True)

    def test_interrupted_resume_recovers_cache_and_all_rows(self):
        from relive.config import load_config
        class Interrupt(MockBackend):
            def infer(inner, req):
                if inner.calls == 2:
                    raise KeyboardInterrupt()
                return super().infer(req)
        b = Interrupt(load_config(CONFIG)["backend"])
        with self.assertRaises(KeyboardInterrupt):
            self.run_stage(1, "partial", _backend=b)
        self.assertEqual(len(list((self.root / "partial/samples").glob("*.json"))), 2)
        _, summary = self.run_stage(1, "partial", resume=True)
        self.assertEqual(summary["sample_count"], 22)
        self.assertEqual(validate_bank(self.root / "partial")["status"], "PASS")

    def test_output_immutable_and_changed_resume_policy(self):
        self.run_stage(1, "s1")
        with self.assertRaisesRegex(ClaimBankError, "OUTPUT_EXISTS"):
            self.run_stage(1, "s1")
        with self.assertRaisesRegex(ClaimBankError, "PLAN_MISMATCH"):
            self.run_stage(1, "s1", resume=True, max_frames=16)

    def test_tamper_detected(self):
        s1, _ = self.run_stage(1, "s1")
        (s1 / "claims.jsonl").write_text('{}\n')
        with self.assertRaisesRegex(ClaimBankError, "HASH_MISMATCH"):
            validate_bank(s1)

    def test_fixed_smoke_source_order_and_bounds(self):
        runtime = load_runtime(self.runtime)
        selected = selection(runtime, None, 0, None, None, 2)
        self.assertEqual(len(selected), 22)
        self.assertEqual(selection(runtime, ["stg"], 0, None, 1, None)[0]["source_record_index"], 2)
        self.assertEqual(len(selection(runtime, None, 1, 3, None, None)), 2)

    def parse_claim(self, claim=None, reqs=None):
        return parse(json.dumps({"generation_status": "GENERATED", "claims": claim or [deepcopy(CLAIM)], "coverage_notes": "Sampled only."}), 2,
                     requirements=reqs or [REQ], question="A possible next action is cutting.", frame_orders=[0])

    def test_strict_json_invalid_duplicate_keys_and_fences(self):
        for raw in ('{} trailing', '```json\n{}\n```', '{"a":1,"a":2}', '{"x":NaN}', '[]'):
            self.assertIsNone(parse(raw, 1)[0])

    def test_forbidden_truth_status_rejected(self):
        for status in ("SUPPORTED", "VERIFIED", "CONTRADICTED"):
            c = {**CLAIM, "status": status}
            self.assertIn("CLAIM_MUST_BE_UNVERIFIED", self.parse_claim([c])[1])

    def test_cycles_missing_dependencies_and_requirements(self):
        for c in ({**CLAIM, "depends_on": ["H1"]}, {**CLAIM, "depends_on": ["H2"]}):
            self.assertIn("DEPENDENCY_INVALID_OR_CYCLIC", self.parse_claim([c])[1])
        self.assertIn("REQUIREMENT_REFERENCE_INVALID", self.parse_claim([{**CLAIM, "requirement_ids": ["R2"]}])[1])

    def test_duplicate_claims_and_single_predicate_risk(self):
        self.assertIn("DUPLICATE_PROPOSITION", self.parse_claim([CLAIM, {**CLAIM, "claim_id": "H2"}])[1])
        self.assertIn("MULTI_ACTION_OR_PURPOSE_RISK", self.parse_claim([{**CLAIM, "text": "The tool cuts then retracts tissue."}])[1])

    def test_action_evidence_mismatch_and_question_provenance(self):
        self.assertIn("TYPE_EVIDENCE_MISMATCH", self.parse_claim([{**CLAIM, "type": "ACTION"}])[1])
        self.assertIn("QUESTION_HYPOTHESIS_REQUIRES_VERBATIM_QUOTE", self.parse_claim([{**CLAIM, "generation_basis": "QUESTION_HYPOTHESIS"}])[1])
        c = {**CLAIM, "generation_basis": "QUESTION_HYPOTHESIS", "question_quote": "possible next action", "frame_orders": []}
        self.assertIsNotNone(self.parse_claim([c])[0])

    def test_unknown_frame_and_extra_bbox_rejected(self):
        self.assertIn("FRAME_REFERENCE_INVALID", self.parse_claim([{**CLAIM, "frame_orders": [123]}])[1])
        self.assertIsNone(self.parse_claim([{**CLAIM, "bbox": [0, 1, 2, 3]}])[0])

    def test_format_repair_bounded_and_schema_failure_retained(self):
        class Bad(MockBackend):
            def infer(inner, request):
                inner.calls += 1
                return "[]"
        b = Bad()
        out, summary = self.run_stage(1, "bad", limit=1, _backend=b)
        self.assertEqual(summary["usage"]["new_calls"], 2)
        self.assertEqual(rows(out / "claim_bank.jsonl")[0]["generation_status"], "MODEL_OR_SCHEMA_FAILURE")

    def test_semantic_failure_never_repaired(self):
        s1, _ = self.run_stage(1, "s1")
        class BadStatus(MockBackend):
            def infer(inner, request):
                inner.calls += 1
                return json.dumps({"generation_status": "GENERATED", "claims": [{**CLAIM, "status": "SUPPORTED"}], "coverage_notes": ""})
        _, summary = self.run_stage(2, "bad_status", limit=1, stage1_dir=str(s1), _backend=BadStatus())
        self.assertEqual(summary["model_calls_made"], 1)
        self.assertEqual(summary["claim_count"], 0)

    def test_stage1_only_question_and_qa_type_in_prompt(self):
        r = load_runtime(self.runtime)[0]
        prompt = prompt_for(r, 1, [], None)
        self.assertNotIn(str(self.image), prompt)
        self.assertNotIn('"dataset_name"', prompt)
        self.assertNotIn('"metadata"', prompt)

    def test_protocol_stability(self):
        self.assertEqual(protocol(), protocol())
        self.assertEqual(protocol()["claim_truth_status"], "UNVERIFIED")
        self.assertFalse(protocol()["supplemental_observation"]["enabled"])

    def test_runtime_import_audit_passes(self):
        from relive.audits import audit_runtime_imports
        self.assertEqual(audit_runtime_imports()["status"], "PASS")

    def test_action_not_reduced_to_presence(self):
        req = {**REQ, "type": "ACTION", "evidence_type": "TEMPORAL"}
        errors = _task_checks({"qa_type": "tal"}, {"generation_status": "GENERATED", "claims": [CLAIM]}, [req])
        self.assertIn("ACTION_OR_CRITERION_REDUCED_TO_OBJECT", errors)

    def test_question_candidate_not_reported_as_observed_next_action(self):
        req = {**REQ, "type": "ACTION", "evidence_type": "TEMPORAL"}
        c = {**CLAIM, "type": "ACTION", "evidence_type": "TEMPORAL", "text": "The instrument cuts tissue.", "frame_orders": [0, 1]}
        errors = _task_checks({"qa_type": "next_action"}, {"generation_status": "GENERATED", "claims": [c]}, [req])
        self.assertIn("FUTURE_ACTION_MUST_REMAIN_HYPOTHETICAL", errors)

    def test_region_caption_requires_unresolved_target_alignment(self):
        errors = _task_checks({"qa_type": "region_caption_gpt"}, {"generation_status": "GENERATED", "claims": [CLAIM]}, [REQ])
        self.assertIn("REGION_CAPTION_TARGET_REFERENCE_UNAVAILABLE", errors)

    def test_one_frame_insufficient_for_visual_action(self):
        req = {**REQ, "type": "ACTION", "evidence_type": "TEMPORAL"}
        c = {**CLAIM, "type": "ACTION", "evidence_type": "TEMPORAL"}
        errors = _task_checks({"qa_type": "tal"}, {"generation_status": "GENERATED", "claims": [c]}, [req])
        self.assertIn("ACTION_REQUIRES_MULTIPLE_OBSERVATIONS", errors)

    def test_unplannable_and_zero_claim_contracts(self):
        self.assertIsNotNone(parse('{"planning_status":"UNABLE_TO_PLAN","requirements":[]}', 1)[0])
        self.assertIsNotNone(parse('{"generation_status":"INSUFFICIENT_VISUAL_CONTEXT","claims":[],"coverage_notes":"Missing observations."}', 2)[0])

    def test_malformed_item_never_crashes_batch_parser(self):
        for bad in ([{**CLAIM, "type": []}], [{**CLAIM, "claim_id": {}}], [{**CLAIM, "frame_orders": [[0]]}]):
            self.assertIsNone(self.parse_claim(bad)[0])

    def test_reference_scores_cannot_be_generated_as_assessment(self):
        errors = _task_checks({"qa_type": "skill_assessment"}, {"generation_status": "GENERATED", "claims": [{**CLAIM, "text": "The rating is 5."}]}, [REQ])
        self.assertIn("FINAL_ASSESSMENT_SCORE_FORBIDDEN", errors)


if __name__ == "__main__":
    unittest.main()
