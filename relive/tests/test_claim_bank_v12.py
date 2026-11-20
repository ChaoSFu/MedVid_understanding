from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from relive.backends.mock import MockBackend
from relive.storage.artifacts import stable_hash
from relive.claim_bank import contracts as v1
from relive.claim_bank import contracts_v11 as v11
from relive.claim_bank import contracts_v12 as v12
from relive.claim_bank.data import audit_dataset, digest, load_json, load_runtime, rows, sample_frames
from relive.claim_bank.pipeline import generate, prompt_for, validate_bank

ROOT = Path(__file__).resolve().parents[1]
CONFIG = str(ROOT / "configs/claim_bank_mock_v12.yaml")
REQ = {"requirement_id": "R1", "description": "Determine whether a marker is visible.", "type": "OBJECT", "evidence_type": "SPATIAL", "importance": "REQUIRED"}
CLAIM = {"claim_id": "H1", "text": "A marker is visible.", "type": "OBJECT", "evidence_type": "SPATIAL", "requirement_ids": ["R1"], "depends_on": [], "status": "UNVERIFIED", "generation_basis": "VISUAL_HYPOTHESIS", "question_quote": None, "image_indices": [0], "temporal_role": "OBSERVED_CONTEXT", "target_alignment": "QUESTION_ALIGNED"}


class ClaimBankV12Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        paths = []
        for i in range(3):
            p = self.root / f"image-{i}.png"
            Image.new("RGB", (24, 24), (30 + i, 40, 50)).save(p)
            paths.append(str(p))
        video = [paths[0], paths[0], *([paths[1]] * 15), paths[2]]
        records = [{"id": "duplicate", "qa_type": qa, "data_source": "SYNTHETIC_FIXTURE", "dataset_name": "SYNTHETIC_FIXTURE",
                    "video": video, "sampled_video_frames": list(range(len(video))),
                    "metadata": {"fps": "0.5", "cvs_scores": "V12_GT_SENTINEL"},
                    "conversations": [{"from": "human", "value": "Which next action is possible: cutting or lifting?"},
                                      {"from": "gpt", "value": "V12_GT_SENTINEL"}],
                    "struc_info": "V12_GT_SENTINEL", "RC_info": "V12_GT_SENTINEL"}
                   for qa in v1.QA_TYPES for _ in range(2)]
        source = self.root / "source.json"
        source.write_text(json.dumps(records))
        audit_dataset(str(source), str(self.root / "data"), expected_count=22, prepare=True)
        self.runtime = str(self.root / "data/runtime_manifest.jsonl")
        self.row = load_runtime(self.runtime)[0]
        self.row["qa_type"] = "next_action"
        self.sampling = sample_frames(self.row, [REQ], 12)

    def run_stage(self, stage, name, **kwargs):
        out = self.root / name
        summary = generate(runtime_manifest=self.runtime, output_dir=str(out), stage=stage,
                           config_path=CONFIG, cache_dir=str(self.root / "cache_v12"),
                           protocol_version="v1.1" if stage == 1 else "v1.2", **kwargs)
        return out, summary

    def evaluate(self, claims=None, row=None, reqs=None, status="GENERATED"):
        value = {"generation_status": status, "coverage_notes": "Only supplied samples observed.", "claims": [deepcopy(CLAIM)] if claims is None else claims}
        return v12.evaluate(json.dumps(value), row or self.row, reqs or [REQ], self.sampling)

    def test_index_binding_handles_noncontiguous_source_orders(self):
        c = {**CLAIM, "image_indices": [0, 2]}
        value, errors, audit = self.evaluate([c])
        self.assertEqual(errors, [])
        bound = v12.bind(value, self.sampling, audit)
        self.assertEqual(bound["claims"][0]["frame_orders"], [0, 17])
        self.assertEqual(bound["claims"][0]["image_indices"], [0, 2])
        self.assertEqual(bound["claims"][0]["frame_bindings"][1]["sha256"], self.sampling["selected_frames"][2]["sha256"])
        self.assertEqual(bound["claims"][0]["status"], "UNVERIFIED")
        v12.validate_bound(bound, self.row, [REQ], self.sampling)

    def test_old_frame_order_field_is_not_auto_normalized(self):
        c = deepcopy(CLAIM)
        c["frame_orders"] = c.pop("image_indices")
        _, errors, _ = self.evaluate([c])
        self.assertIn("CLAIM_FIELDS_INVALID", errors)
        with self.assertRaises(v1.ClaimBankError):
            value, _, audit = self.evaluate([c])
            v12.bind(value, self.sampling, audit)

    def test_bad_negative_bool_and_out_of_range_indices(self):
        for indices in ([-1], [3], [True], ["0"]):
            _, errors, _ = self.evaluate([{**CLAIM, "image_indices": indices}])
            self.assertTrue(set(errors) & {"IMAGE_INDEX_INVALID", "REFERENCE_LIST_SCHEMA"})

    def test_visual_quote_must_be_null_even_for_real_question_quote(self):
        _, errors, audit = self.evaluate([{**CLAIM, "question_quote": "cutting"}])
        self.assertIn("VISUAL_QUESTION_QUOTE_MUST_BE_NULL", errors)
        self.assertEqual(audit["binding"]["issues"][0]["path"], "$.claims[0].question_quote")

    def test_question_quote_must_be_from_original_question_not_requirement(self):
        c = {**CLAIM, "generation_basis": "QUESTION_HYPOTHESIS", "question_quote": REQ["description"], "image_indices": []}
        self.assertIn("QUESTION_QUOTE_INVALID", self.evaluate([c])[1])
        c["question_quote"] = "cutting"
        self.assertEqual(self.evaluate([c])[1], [])

    def test_unlinked_scene_claim_rejected(self):
        c = {**CLAIM, "requirement_ids": [], "target_alignment": "SCENE_ONLY"}
        self.assertIn("REQUIREMENT_REFERENCE_INVALID", self.evaluate([c])[1])

    def test_all_evaluable_layers_run_despite_binding_failure(self):
        c = {**CLAIM, "type": "RELATION", "evidence_type": "SPATIAL", "image_indices": [99],
             "question_quote": REQ["description"], "requirement_ids": []}
        row = {**self.row, "qa_type": "region_caption_gpt"}
        _, errors, audit = self.evaluate([c], row=row)
        self.assertTrue({"IMAGE_INDEX_INVALID", "VISUAL_QUESTION_QUOTE_MUST_BE_NULL", "REQUIREMENT_REFERENCE_INVALID",
                         "TYPE_EVIDENCE_MISMATCH", "REGION_CAPTION_TARGET_REFERENCE_UNAVAILABLE"} <= set(errors))
        self.assertEqual(audit["schema"]["status"], "PASS")
        self.assertEqual(audit["binding"]["status"], "FAIL")
        self.assertEqual(audit["task"]["status"], "FAIL")

    def test_current_action_allowed_for_next_action_task(self):
        req = {**REQ, "type": "ACTION", "evidence_type": "TEMPORAL"}
        c = {**CLAIM, "text": "The instrument is cutting tissue.", "type": "ACTION", "evidence_type": "TEMPORAL", "image_indices": [0, 1]}
        self.assertEqual(self.evaluate([c], reqs=[req])[1], [])

    def test_future_role_checked_for_every_claim_type(self):
        for kind, evidence in (("OBJECT", "SPATIAL"), ("ATTRIBUTE", "SPATIAL"), ("RELATION", "SPATIAL_MULTI_REGION"), ("ACTION", "TEMPORAL"), ("HIGH_LEVEL", "COMPOSITIONAL")):
            c = {**CLAIM, "type": kind, "evidence_type": evidence, "text": "The next phase will be cutting.", "image_indices": [0, 1]}
            self.assertIn("FUTURE_TEXT_ROLE_MISMATCH", self.evaluate([c])[1])

    def test_prospective_question_hypothesis_allowed_not_visual_future(self):
        c = {**CLAIM, "text": "A possible next action could be cutting.", "type": "ACTION", "evidence_type": "TEMPORAL",
             "temporal_role": "FUTURE_HYPOTHESIS", "generation_basis": "QUESTION_HYPOTHESIS", "question_quote": "cutting", "image_indices": []}
        self.assertEqual(self.evaluate([c])[1], [])
        c.update(generation_basis="VISUAL_HYPOTHESIS", question_quote=None, image_indices=[0, 1])
        self.assertIn("FUTURE_ACTION_MUST_REMAIN_HYPOTHETICAL", self.evaluate([c])[1])

    def test_unknown_timebase_cannot_be_query_time_aligned(self):
        self.assertIn("QUERY_TIME_MAPPING_UNRESOLVED", self.evaluate([{**CLAIM, "temporal_role": "QUESTION_TIME_QUERY"}])[1])

    def test_region_caption_zero_claims_retains_unresolved(self):
        row = {**self.row, "qa_type": "region_caption_gpt"}
        self.assertEqual(self.evaluate([], row=row, status="UNRESOLVED_TARGET_REFERENCE")[1], [])
        c = {**CLAIM, "target_alignment": "UNRESOLVED"}
        self.assertEqual(self.evaluate([c], row=row, status="UNRESOLVED_TARGET_REFERENCE")[1], [])
        c["text"] = "The object inside the bounding box is a marker."
        self.assertIn("REGION_CAPTION_TARGET_REFERENCE_UNAVAILABLE", self.evaluate([c], row=row, status="UNRESOLVED_TARGET_REFERENCE")[1])

    def test_quality_chain_verification_and_meta_description_rejected(self):
        for text, code in (("Cutting is followed immediately by suturing.", "MULTI_ACTION_OR_PURPOSE_RISK"),
                           ("The absence of movement is confirmed.", "VERIFICATION_LANGUAGE_FORBIDDEN"),
                           ("Event boundaries are defined by instrument motion.", "META_DESCRIPTION_NOT_VISUAL_PROPOSITION")):
            self.assertIn(code, self.evaluate([{**CLAIM, "text": text}])[1])

    def test_lexical_risk_is_not_a_truth_verdict(self):
        value, errors, audit = self.evaluate([{**CLAIM, "text": "A marker is visible, suggesting a possible instrument."}])
        self.assertEqual(errors, [])
        self.assertEqual(audit["semantic_risk"]["status"], "REVIEW_REQUIRED")
        bound = v12.bind(value, self.sampling, audit)
        self.assertIn("INFERENCE_OR_PURPOSE_REVIEW", bound["claims"][0]["quality_review_flags"])
        self.assertEqual(bound["claims"][0]["status"], "UNVERIFIED")

    def test_object_prerequisite_allowed_when_action_is_preserved(self):
        req = {**REQ, "type": "ACTION", "evidence_type": "TEMPORAL"}
        self.assertIn("ACTION_OR_CRITERION_REDUCED_TO_OBJECT", self.evaluate(reqs=[req])[1])
        action = {**CLAIM, "claim_id": "H2", "text": "The instrument moves toward tissue.", "type": "ACTION", "evidence_type": "TEMPORAL", "image_indices": [0, 1]}
        self.assertEqual(self.evaluate([CLAIM, action], reqs=[req])[1], [])

    def test_cycle_duplicate_and_truth_status_fail_closed(self):
        for change, code in (({"depends_on": ["H1"]}, "DEPENDENCY_INVALID_OR_CYCLIC"),
                             ({"status": "SUPPORTED"}, "CLAIM_MUST_BE_UNVERIFIED")):
            self.assertIn(code, self.evaluate([{**CLAIM, **change}])[1])
        self.assertIn("DUPLICATE_PROPOSITION", self.evaluate([CLAIM, {**CLAIM, "claim_id": "H2"}])[1])

    def test_prompt_only_exposes_input_image_indices(self):
        prompt = prompt_for(self.row, 2, [REQ], self.sampling, "v1.2")
        data = json.loads(prompt.split("contract:\n", 1)[1])
        self.assertEqual(data["available_image_indices"], [0, 1, 2])
        self.assertNotIn("source_order_range", data["coverage"])
        self.assertNotIn("frame_order", json.dumps(data))
        self.assertNotIn("V12_GT_SENTINEL", prompt)

    def test_schema_failures_are_structured_not_exceptions(self):
        for raw in ("[]", "null", "{}", '{"claims":[null]}', '{"claims":[{"type":[],"image_indices":[true]}]}', "not json", '{"x":NaN}'):
            _, errors, audit = v12.evaluate(raw, self.row, [REQ], self.sampling)
            self.assertTrue(errors)
            self.assertEqual(audit["schema"]["status"], "FAIL")

    def test_upstream_v11_reused_read_only_and_exact_replay(self):
        s1, _ = self.run_stage(1, "s1")
        upstream_before = {str(p): digest(p) for p in s1.rglob("*.json*")}
        from relive.config import load_config
        requests = []
        class Capture(MockBackend):
            def infer(inner, request):
                requests.append(deepcopy(request))
                self.assertTrue(request["image_paths"])
                return super().infer(request)
        s2, summary = self.run_stage(2, "s2", stage1_dir=str(s1), _backend=Capture(load_config(CONFIG)["backend"]))
        self.assertEqual(summary["sample_count"], 22)
        self.assertEqual(summary["generation_status_counts"], {"GENERATED": 18, "UNRESOLVED_TARGET_REFERENCE": 4})
        self.assertEqual(summary["failure_reason_counts"], {})
        self.assertNotIn("V12_GT_SENTINEL", json.dumps(requests))
        for path in s2.rglob("*.json*"):
            self.assertNotIn("V12_GT_SENTINEL", path.read_text())
        with patch("relive.backends.make_backend", side_effect=AssertionError("weights loaded")):
            replay, result = self.run_stage(2, "replay", stage1_dir=str(s1), replay_only=True, replay_from=str(s2))
        self.assertEqual(result["model_calls_made"], 0)
        self.assertEqual(result["usage"]["cache_hits"], 22)
        for name in ("claim_bank.jsonl", "claims.jsonl", "requirements.jsonl", "frame_sampling_manifest.jsonl"):
            self.assertEqual((s2 / name).read_bytes(), (replay / name).read_bytes())
        self.assertEqual(upstream_before, {str(p): digest(p) for p in s1.rglob("*.json*")})
        self.assertEqual(validate_bank(s2)["status"], "PASS")
        plan = rows(s2 / "claim_bank.jsonl")[0]
        self.assertEqual(plan["generation_metadata"]["stage1_binding"]["requirements_sha256"], stable_hash(plan["requirements"]))

    def test_cross_protocol_rejection_and_stage1_not_reexecuted(self):
        with self.assertRaisesRegex(v1.ClaimBankError, "STAGE2_ONLY"):
            generate(runtime_manifest=self.runtime, output_dir=str(self.root / "bad"), stage=1,
                     config_path=CONFIG, cache_dir=str(self.root / "unused"), protocol_version="v1.2")

    def test_binding_tamper_fails_even_if_text_unchanged(self):
        value, _, audit = self.evaluate()
        bound = v12.bind(value, self.sampling, audit)
        bound["claims"][0]["frame_orders"] = [999]
        with self.assertRaisesRegex(v1.ClaimBankError, "REFERENCE_MISMATCH"):
            v12.validate_bound(bound, self.row, [REQ], self.sampling)

    def test_binding_errors_not_semantically_retried(self):
        s1, _ = self.run_stage(1, "s1")
        class WrongQuote(MockBackend):
            def infer(inner, request):
                inner.calls += 1
                return json.dumps({"generation_status": "GENERATED", "claims": [{**CLAIM, "question_quote": "not original"}], "coverage_notes": ""})
        out, summary = self.run_stage(2, "bad_quote", stage1_dir=str(s1), limit=1, _backend=WrongQuote())
        self.assertEqual(summary["model_calls_made"], 1)
        self.assertEqual(rows(out / "claim_bank.jsonl")[0]["claims"], [])

    def test_format_repair_cannot_change_existing_claim_text(self):
        s1, _ = self.run_stage(1, "s1")
        class Changed(MockBackend):
            def infer(inner, request):
                inner.calls += 1
                c = deepcopy(CLAIM)
                value = {"generation_status": "GENERATED", "claims": [c], "coverage_notes": ""}
                if "format-repair1" in request["prompt_version"]:
                    c["text"] = "A different object is visible."
                else:
                    value.pop("coverage_notes")
                return json.dumps(value)
        out, summary = self.run_stage(2, "changed", stage1_dir=str(s1), limit=1, _backend=Changed())
        self.assertEqual(summary["model_calls_made"], 2)
        self.assertEqual(summary["claim_count"], 0)
        self.assertIn("FORMAT_REPAIR_CHANGED_CLAIMS", rows(out / "claim_bank.jsonl")[0]["failure_reasons"])

    def test_legacy_contract_hashes_unchanged(self):
        self.assertEqual(v1.protocol()["protocol_sha256"], "f0dbde883dbd5811ab44aabb05986e3c5dd98332e703bdfa8cc15dc3279e4e79")
        self.assertEqual(v11.protocol()["protocol_sha256"], "139d890cf6586d848467926984a4ce84ff8d91908965eab76aca80b64815f2d5")

    def test_format_only_repair_and_replay_preserve_proposition(self):
        s1, _ = self.run_stage(1, "s1")
        class Repair(MockBackend):
            def infer(inner, request):
                inner.calls += 1
                value = {"generation_status": "GENERATED", "claims": [CLAIM], "coverage_notes": "Sampled only."}
                if "format-repair1" not in request["prompt_version"]:
                    value.pop("coverage_notes")
                return json.dumps(value)
        fresh, summary = self.run_stage(2, "repair", limit=1, stage1_dir=str(s1), _backend=Repair())
        self.assertEqual(summary["model_calls_made"], 2)
        self.assertEqual(summary["claim_count"], 1)
        replay, result = self.run_stage(2, "repair_replay", limit=1, stage1_dir=str(s1), replay_only=True, replay_from=str(fresh))
        self.assertEqual(result["model_calls_made"], 0)
        self.assertEqual(result["usage"]["cache_hits"], 2)
        self.assertEqual((fresh / "claim_bank.jsonl").read_bytes(), (replay / "claim_bank.jsonl").read_bytes())

    def test_stage1_v1_not_admitted_to_v12(self):
        old = self.root / "v1"
        generate(runtime_manifest=self.runtime, output_dir=str(old), stage=1,
                 config_path=str(ROOT / "configs/claim_bank_mock_v1.yaml"), cache_dir=str(self.root / "old_cache"), protocol_version="v1")
        with self.assertRaisesRegex(v1.ClaimBankError, "STAGE1_PROTOCOL_VERSION_MISMATCH"):
            self.run_stage(2, "mix", stage1_dir=str(old))

    def test_v12_completed_resume_validates_without_model(self):
        s1, _ = self.run_stage(1, "s1")
        _, summary = self.run_stage(2, "s2", stage1_dir=str(s1))
        with patch("relive.backends.make_backend", side_effect=AssertionError("weights loaded")):
            _, resumed = self.run_stage(2, "s2", stage1_dir=str(s1), resume=True)
        self.assertEqual(summary, resumed)

    def test_v12_replay_cache_miss_never_executes_model(self):
        s1, _ = self.run_stage(1, "s1")
        s2, _ = self.run_stage(2, "s2", stage1_dir=str(s1))
        with patch("relive.storage.cache.CachedInference.has_terminal_result", return_value=False), patch("relive.claim_bank.pipeline.ReplayBackend.infer", side_effect=AssertionError("inference")):
            with self.assertRaisesRegex(v1.ClaimBankError, "REPLAY_CACHE_MISS"):
                self.run_stage(2, "miss", stage1_dir=str(s1), replay_only=True, replay_from=str(s2))

    def test_malformed_repair_stays_structured_failure(self):
        s1, _ = self.run_stage(1, "s1")
        class MalformedRepair(MockBackend):
            def infer(inner, request):
                inner.calls += 1
                if "format-repair1" in request["prompt_version"]:
                    return '{"generation_status":"GENERATED","claims":null,"coverage_notes":""}'
                return json.dumps({"generation_status": "GENERATED", "claims": [CLAIM]})
        out, summary = self.run_stage(2, "malformed_repair", stage1_dir=str(s1), limit=1, _backend=MalformedRepair())
        self.assertEqual(summary["model_calls_made"], 2)
        self.assertEqual(rows(out / "claim_bank.jsonl")[0]["generation_status"], "MODEL_OR_SCHEMA_FAILURE")


if __name__ == "__main__":
    unittest.main()
