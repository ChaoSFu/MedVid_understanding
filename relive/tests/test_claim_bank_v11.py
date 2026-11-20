from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from relive.backends.mock import MockBackend
from relive.claim_bank import contracts as v1
from relive.claim_bank import contracts_v11 as v11
from relive.claim_bank.data import audit_dataset, load_json, load_runtime, rows
from relive.claim_bank.pipeline import generate, prompt_for, validate_bank

ROOT = Path(__file__).resolve().parents[1]
CONFIG = str(ROOT / "configs/claim_bank_mock_v11.yaml")
REQ = {"requirement_id": "R1", "description": "Determine whether the queried interaction occurs.",
       "type": "ACTION", "evidence_type": "TEMPORAL", "importance": "REQUIRED"}


class ClaimBankV11Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.frame = self.root / "frame.png"
        Image.new("RGB", (24, 24), (30, 40, 50)).save(self.frame)
        source = []
        for qa in v1.QA_TYPES:
            for _ in range(2):
                source.append({"id": "duplicate", "qa_type": qa, "video": [str(self.frame)],
                               "sampled_video_frames": [1], "metadata": {"answer": "V11_GT_SENTINEL"},
                               "conversations": [{"from": "human", "value": "Which of the listed interactions occurs?"},
                                                 {"from": "gpt", "value": "V11_GT_SENTINEL"}],
                               "RC_info": "V11_GT_SENTINEL", "struc_info": "V11_GT_SENTINEL"})
        self.source = self.root / "source.json"
        self.source.write_text(json.dumps(source))
        audit_dataset(str(self.source), str(self.root / "data"), expected_count=22, prepare=True)
        self.runtime = str(self.root / "data/runtime_manifest.jsonl")

    def run_stage(self, stage, name, **kwargs):
        out = self.root / name
        summary = generate(runtime_manifest=self.runtime, output_dir=str(out), stage=stage,
                           config_path=CONFIG, cache_dir=str(self.root / "cache"),
                           protocol_version="v1.1", **kwargs)
        return out, summary

    def test_no_observation_status_or_claim_contract_in_any_stage1_prompt(self):
        for row in load_runtime(self.runtime):
            prompt = prompt_for(row, 1, [], None, "v1.1")
            for forbidden in ("GENERATED", "INSUFFICIENT_VISUAL_CONTEXT", "UNRESOLVED_TARGET_REFERENCE",
                              "QUESTION_HYPOTHESIS", "VISUAL_HYPOTHESIS", "target_alignment", "frame_orders"):
                self.assertNotIn(forbidden, prompt)
            self.assertIn("Images are deliberately NOT supplied", prompt)
            self.assertIn("PLANNED", prompt)
            self.assertIn("UNABLE_TO_PLAN", prompt)

    def test_stage2_contract_has_no_planning_enums(self):
        text = v11.output_contract(2)
        self.assertNotIn("planning_status", text)
        self.assertNotIn("UNABLE_TO_PLAN", text)
        self.assertIn("UNVERIFIED", text)

    def test_region_policy_plans_missing_target_prerequisite(self):
        planning = v11.task_policy("region_caption_gpt", 1)
        self.assertIn("unresolved prerequisite", planning)
        self.assertNotIn("Return UNRESOLVED_TARGET_REFERENCE", planning)
        self.assertIn("Return UNRESOLVED_TARGET_REFERENCE", v11.task_policy("region_caption_gpt", 2))

    def test_task_planning_covers_dense_next_and_assessment(self):
        self.assertIn("onset/offset", v11.task_policy("dense_captioning_gpt", 1))
        self.assertIn("precondition", v11.task_policy("next_action", 1))
        self.assertIn("clinical criteria", v11.task_policy("cvs_assessment", 1))
        self.assertIn("Group shared", v11.task_policy("dense_captioning_gemini", 1))

    def test_wrong_stage_enum_remains_failure_not_alias(self):
        raw = json.dumps({"planning_status": "GENERATED", "planning_reason": None, "requirements": [REQ]})
        obj, errors = v11.parse(raw, 1)
        self.assertIsNone(obj)
        self.assertEqual(errors, ["STAGE1_PLANNING_STATUS_INVALID"])
        details = v11.diagnostics(raw, errors)
        self.assertEqual(details[0]["actual"], "GENERATED")
        self.assertEqual(details[0]["expected"], ["PLANNED", "UNABLE_TO_PLAN"])

    def test_original_failure_gets_both_missing_field_and_enum_diagnostics(self):
        raw = json.dumps({"planning_status": "GENERATED", "requirements": [REQ]})
        _, errors = v11.parse(raw, 1)
        self.assertIn("STAGE1_TOP_LEVEL_FIELDS_INVALID", errors)
        self.assertIn("STAGE1_PLANNING_STATUS_INVALID", errors)
        self.assertEqual(v11.preserved_requirements(raw), [REQ])

    def test_reason_required_only_for_abstention(self):
        for reason in (None, "", 123):
            raw = json.dumps({"planning_status": "UNABLE_TO_PLAN", "planning_reason": reason, "requirements": []})
            self.assertIsNone(v11.parse(raw, 1)[0])
        self.assertIsNotNone(v11.parse(json.dumps({"planning_status": "PLANNED", "planning_reason": None, "requirements": [REQ]}), 1)[0])

    def test_one_repair_corrects_status_without_changing_requirements(self):
        class StatusRepair(MockBackend):
            def infer(inner, request):
                inner.calls += 1
                self.assertEqual(request["image_paths"], [])
                self.assertNotIn("INSUFFICIENT_VISUAL_CONTEXT", request["prompt"])
                if "format-repair1" not in request["prompt_version"]:
                    return json.dumps({"planning_status": "GENERATED", "requirements": [REQ]})
                self.assertIn('"actual": "GENERATED"', request["prompt"])
                self.assertIn("Do not add, remove, reorder or rewrite", request["prompt"])
                return json.dumps({"planning_status": "PLANNED", "planning_reason": None, "requirements": [REQ]})
        out, summary = self.run_stage(1, "repaired", limit=1, _backend=StatusRepair())
        row = rows(out / "claim_bank.jsonl")[0]
        self.assertEqual(row["requirements"], [REQ])
        self.assertEqual(row["planning_status"], "PLANNED")
        self.assertEqual(summary["model_calls_made"], 2)
        self.assertTrue(row["generation_metadata"]["output_contract_audit"][1]["requirements_preservation_guard"])
        self.assertEqual(validate_bank(out)["status"], "PASS")

    def test_repair_cannot_remove_rewrite_or_reorder_valid_requirements(self):
        original = [REQ, {**REQ, "requirement_id": "R2", "description": "Determine visible prerequisites for the interaction."}]
        for index, changed in enumerate(([REQ], [{**REQ, "description": "Changed proposition."}, original[1]], list(reversed(original)))):
            class BadRepair(MockBackend):
                def infer(inner, request):
                    inner.calls += 1
                    if "format-repair1" not in request["prompt_version"]:
                        return json.dumps({"planning_status": "GENERATED", "requirements": original})
                    return json.dumps({"planning_status": "PLANNED", "planning_reason": None, "requirements": changed})
            out, summary = self.run_stage(1, "changed-" + str(index), limit=1,
                                          _backend=BadRepair({"kind": "mock", "revision": f"mutation-{index}"}))
            row = rows(out / "claim_bank.jsonl")[0]
            self.assertEqual(row["failure_reasons"], ["FORMAT_REPAIR_CHANGED_REQUIREMENTS"])
            self.assertEqual(row["planning_status"], "MODEL_OR_SCHEMA_FAILURE")
            self.assertEqual(summary["model_calls_made"], 2)
            self.assertEqual(row["requirements"], [])

    def test_abstention_is_explicit_and_not_retried(self):
        class Unable(MockBackend):
            def infer(inner, request):
                inner.calls += 1
                return json.dumps({"planning_status": "UNABLE_TO_PLAN", "planning_reason": "The question does not specify a visual task.", "requirements": []})
        out, summary = self.run_stage(1, "unable", limit=1, _backend=Unable())
        self.assertEqual(summary["model_calls_made"], 1)
        self.assertEqual(summary["requirement_count"], 0)
        self.assertEqual(summary["planning_status_counts"], {"UNABLE_TO_PLAN": 1})
        self.assertEqual(summary["planning_abstention_count"], 1)
        self.assertEqual(rows(out / "generation_failures.jsonl")[0]["planning_reason"], "The question does not specify a visual task.")
        self.assertEqual(validate_bank(out)["status"], "PASS")

    def test_missing_images_reason_flagged_without_fabricating_plan(self):
        value = {"planning_status": "UNABLE_TO_PLAN", "planning_reason": "No images were provided.", "requirements": []}
        parsed, _ = v11.parse(json.dumps(value), 1)
        self.assertEqual(parsed, value)
        self.assertIn("IMAGE_ABSENCE_NOT_VALID_PLANNING_BLOCKER", v11.planning_flags(parsed))

    def test_empty_requirements_protected_from_semantic_repair(self):
        class InventedRepair(MockBackend):
            def infer(inner, request):
                inner.calls += 1
                if "format-repair1" not in request["prompt_version"]:
                    return '{"planning_status":"UNABLE_TO_PLAN","requirements":[]}'
                return json.dumps({"planning_status": "PLANNED", "planning_reason": None, "requirements": [REQ]})
        out, summary = self.run_stage(1, "invented", limit=1, _backend=InventedRepair())
        self.assertEqual(summary["model_calls_made"], 2)
        self.assertEqual(rows(out / "claim_bank.jsonl")[0]["failure_reasons"], ["FORMAT_REPAIR_CHANGED_REQUIREMENTS"])

    def test_new_contract_cache_binding_and_cross_version_rejection(self):
        first, summary = self.run_stage(1, "v11", limit=1)
        self.assertEqual(summary["protocol_version"], "relive-claimgen-v1.1")
        v1_out = self.root / "v1"
        generate(runtime_manifest=self.runtime, output_dir=str(v1_out), stage=1,
                 config_path=str(ROOT / "configs/claim_bank_mock_v1.yaml"), cache_dir=str(self.root / "cache_v1"), limit=1)
        self.assertNotEqual(load_json(first / "run_binding.json")["identity"]["protocol_sha256"], load_json(v1_out / "run_binding.json")["identity"]["protocol_sha256"])
        with self.assertRaisesRegex(v1.ClaimBankError, "PROTOCOL_VERSION_MISMATCH"):
            self.run_stage(2, "mix", limit=1, stage1_dir=str(v1_out))

    def test_v1_contract_hash_and_strict_parser_unchanged(self):
        self.assertEqual(v1.protocol()["protocol_sha256"], "f0dbde883dbd5811ab44aabb05986e3c5dd98332e703bdfa8cc15dc3279e4e79")
        self.assertEqual(v1.parse(json.dumps({"planning_status": "GENERATED", "requirements": [REQ]}), 1)[1], ["STAGE1_SCHEMA"])

    def test_22_per_task_dataflow_sentinel_and_exact_replay(self):
        from relive.config import load_config
        calls = []
        class Capture(MockBackend):
            def infer(inner, request):
                calls.append(deepcopy(request))
                return super().infer(request)
        backend = Capture(load_config(CONFIG)["backend"])
        s1, first = self.run_stage(1, "s1", per_qa_type=2, _backend=backend)
        s2, second = self.run_stage(2, "s2", stage1_dir=str(s1), per_qa_type=2, _backend=backend)
        self.assertEqual(first["planning_status_counts"], {"PLANNED": 22})
        self.assertEqual(second["sample_count"], 22)
        self.assertEqual(second["generation_status_counts"]["UNRESOLVED_TARGET_REFERENCE"], 4)
        self.assertNotIn("V11_GT_SENTINEL", json.dumps(calls))
        with patch("relive.backends.make_backend", side_effect=AssertionError("model load")):
            replay, result = self.run_stage(2, "replay", stage1_dir=str(s1), per_qa_type=2, replay_only=True, replay_from=str(s2))
        self.assertEqual(result["model_calls_made"], 0)
        for name in ("claim_bank.jsonl", "claims.jsonl", "requirements.jsonl", "generation_failures.jsonl"):
            self.assertEqual((s2 / name).read_bytes(), (replay / name).read_bytes())
        for p in [*s1.rglob("*.json*"), *s2.rglob("*.json*")]:
            self.assertNotIn("V11_GT_SENTINEL", p.read_text())
        self.assertEqual(validate_bank(s2)["status"], "PASS")

    def test_stage1_repair_replay_has_zero_new_calls(self):
        class Repair(MockBackend):
            def infer(inner, req):
                inner.calls += 1
                return json.dumps({"planning_status": "PLANNED" if "format-repair1" in req["prompt_version"] else "GENERATED", "planning_reason": None, "requirements": [REQ]})
        first, _ = self.run_stage(1, "s1", limit=1, _backend=Repair())
        replay, summary = self.run_stage(1, "replay", limit=1, replay_only=True, replay_from=str(first))
        self.assertEqual(summary["usage"]["cache_hits"], 2)
        self.assertEqual(summary["model_calls_made"], 0)
        self.assertEqual((first / "claim_bank.jsonl").read_bytes(), (replay / "claim_bank.jsonl").read_bytes())

    def test_cli_defaults_to_v11(self):
        from relive.claim_bank.__main__ import main
        out = self.root / "cli"
        with patch("sys.argv", ["claim_bank", "--mode", "stage1", "--dry-run", "--runtime-manifest", self.runtime,
                               "--config", CONFIG, "--cache-dir", str(self.root / "unused"), "--output-dir", str(out)]):
            self.assertEqual(main(), 0)
        self.assertEqual(load_json(out / "generation_summary.json")["protocol_version"], "relive-claimgen-v1.1")
        self.assertFalse((self.root / "unused").exists())


if __name__ == "__main__":
    unittest.main()
