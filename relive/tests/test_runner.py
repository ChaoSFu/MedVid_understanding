"""End-to-end mock runner, resume, budget, and evaluation separation tests."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from relive.audits import audit_run, audit_strict_result
from relive.config import load_config, validate_config
from relive.evaluation.metrics import evaluate
from relive.runner import run


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "mock_smoke.yaml"
RUNTIME = ROOT / "examples" / "synthetic_runtime.jsonl"


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def config(self):
        return load_config(CONFIG)

    def test_full_mock_path_is_auditable_and_resume_makes_zero_new_calls(self):
        output = self.root / "run"
        summary = run(self.config(), RUNTIME, output, max_samples=2)
        self.assertTrue(summary["synthetic"])
        self.assertEqual(summary["strict_answered"], 2)
        self.assertEqual(summary["certificate_distribution"], {"VERIFIED": 2})
        self.assertGreater(summary["new_model_calls"], 0)
        self.assertEqual(audit_run(output)["status"], "PASS")
        replay = run(self.config(), RUNTIME, output, max_samples=2)
        self.assertEqual(replay["new_model_calls"], 0)
        self.assertEqual(replay["resumed_completed_samples"], 2)
        samples = [json.loads(path.read_text()) for path in (output / "samples").glob("*.json")]
        self.assertTrue(all(sample["strict"]["status"] == "ANSWERED" for sample in samples))
        self.assertTrue(all(sample["strict"]["input_references"][0]["frame_ids"] for sample in samples))
        self.assertTrue(all(sample["usage"]["calls"] <= sample["usage"]["max_calls"] for sample in samples))

    def test_policy_recalculation_reuses_raw_cache_without_model_calls(self):
        shared_cache = self.root / "shared-cache"
        first = run(self.config(), RUNTIME, self.root / "full", cache_dir=shared_cache, max_samples=1)
        self.assertGreater(first["new_model_calls"], 0)
        semantic = deepcopy(self.config())
        semantic["policy"]["name"] = "semantic_only"
        semantic = validate_config(semantic)
        recertified = run(semantic, RUNTIME, self.root / "semantic", cache_dir=shared_cache, max_samples=1)
        self.assertEqual(recertified["new_model_calls"], 0)
        record = next((self.root / "semantic" / "samples").glob("*.json"))
        self.assertEqual(json.loads(record.read_text())["certificates"][0]["policy_name"], "semantic_only")

    def test_call_budget_terminates_with_abstention_and_retains_attempts(self):
        config = deepcopy(self.config())
        config["budget"]["max_calls"] = 1
        config = validate_config(config)
        output = self.root / "budget"
        summary = run(config, RUNTIME, output, max_samples=1)
        self.assertEqual(summary["termination_distribution"], {"CALL_BUDGET_EXHAUSTED": 1})
        record = json.loads(next((output / "samples").glob("*.json")).read_text())
        self.assertEqual(record["strict"]["status"], "ABSTAIN")
        self.assertEqual(record["usage"]["calls"], 1)
        self.assertGreaterEqual(len(record["certificates"]), 1)
        self.assertEqual(audit_run(output)["status"], "PASS")

    def test_adaptation_event_preserves_deterministic_decision_provenance(self):
        config = deepcopy(self.config())
        config["backend"]["rules"].append({
            "match": {"stage": "semantic", "claim_text": "The instrument moves toward the visible tissue.",
                      "variant": "ORIGINAL"},
            "response": {"status": "INSUFFICIENT"},
        })
        config["budget"]["max_rounds"] = 2
        config = validate_config(config)
        output = self.root / "adaptation"
        run(config, RUNTIME, output, max_samples=1)
        record = json.loads(next((output / "samples").glob("*.json")).read_text())
        expansion = next(event for event in record["adaptation_events"]
                         if event["action"] == "EXPAND_TEMPORAL_CONTEXT")
        self.assertEqual(expansion["parameters"]["decision_reason"], "FIXED_NEIGHBOR_CONTEXT")
        self.assertEqual(expansion["parameters"]["extra_frames_each_side"], 1)
        self.assertTrue(expansion["previous_candidate_id"])
        self.assertTrue(expansion["next_candidate_id"])
        self.assertEqual(record["termination_reason"], "MAX_ROUNDS")

    def test_alternate_region_reuses_original_and_contrast_inputs(self):
        config = deepcopy(self.config())
        config["backend"]["rules"].append({
            "match": {"stage": "semantic", "claim_text": "The instrument moves toward the visible tissue.",
                      "variant": "KEEP_TARGET"},
            "response": {"status": "INSUFFICIENT"},
        })
        config["budget"]["max_rounds"] = 2
        config = validate_config(config)
        output = self.root / "alternate-region"
        run(config, RUNTIME, output, max_samples=1)
        record = json.loads(next((output / "samples").glob("*.json")).read_text())
        # The three alternate-region intervention inputs are distinct; identical
        # transformed images may still deduplicate in the content-addressed cache.
        self.assertLessEqual(record["usage"]["calls"], 9)
        self.assertTrue(any(event["action"] == "TRY_ALTERNATE_SUPPORT_REGION"
                            for event in record["adaptation_events"]))
        self.assertTrue(list((output / "events" / "verification_reuse").glob("*.json")))
        self.assertTrue(list((output / "events" / "contrast_reuse").glob("*.json")))

    def test_forced_fallback_is_kept_separate_from_strict_output(self):
        config = deepcopy(self.config())
        config["budget"]["max_calls"] = 1
        config["reasoning"]["mode"] = "benchmark_forced"
        config = validate_config(config)
        output = self.root / "forced"
        run(config, RUNTIME, output, max_samples=1)
        record = json.loads(next((output / "samples").glob("*.json")).read_text())
        self.assertEqual(record["strict"]["status"], "ABSTAIN")
        self.assertTrue(record["benchmark_forced"]["fallback_used"])
        self.assertEqual(record["benchmark_forced"]["certificate_ids"], [])

    def test_strict_source_audit_rejects_tampered_frame_reference(self):
        output = self.root / "audit-tamper"
        run(self.config(), RUNTIME, output, max_samples=1)
        record = json.loads(next((output / "samples").glob("*.json")).read_text())
        record["strict"]["input_references"][0]["frame_ids"] = ["unbound"]
        audit = audit_strict_result(record)
        self.assertEqual(audit["status"], "FAIL")
        self.assertTrue(any("frame IDs" in item for item in audit["issues"]))

    def test_phase_limit_prevents_full_benchmark_invocation(self):
        with self.assertRaisesRegex(ValueError, "1–5"):
            run(self.config(), RUNTIME, self.root / "too-many", max_samples=6, phase="smoke")
        with self.assertRaisesRegex(ValueError, "1–10"):
            run(self.config(), RUNTIME, self.root / "too-many-preflight", max_samples=11, phase="preflight")

    def test_independent_evaluation_reads_ground_truth_only_when_requested(self):
        output = self.root / "evaluation"
        run(self.config(), RUNTIME, output, max_samples=1)
        without_gt = evaluate(output)
        self.assertFalse(without_gt["gt_loaded"])
        self.assertEqual(without_gt["official_downstream_metrics"]["status"], "NOT_CONNECTED")
        gt = self.root / "held_out_gt.jsonl"
        gt.write_text(json.dumps({"sample_id": "synthetic-claim-verification", "task": "claim_verification", "answer": "SUPPORTED"}) + "\n")
        with_gt = evaluate(output, gt)
        self.assertTrue(with_gt["gt_loaded"])
        self.assertEqual(with_gt["strict_task_metrics"]["exact_match"]["value"], 1.0)


if __name__ == "__main__":
    unittest.main()
