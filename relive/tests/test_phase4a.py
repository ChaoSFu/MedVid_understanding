"""GPU-free Phase 4A frozen-choice audit tests."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import sys, unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from relive.phase36 import execute as phase36_execute, preflight as phase36_preflight
from relive.phase4a import (CHOICES, DEVELOPMENT_POSITIVE_CONTROL, Phase4AError, choice_prompt, execute, freeze_development_mismatched_public,
                            preflight, support_margin)
from relive.phase4a0 import ELIGIBLE_MANIFEST_FORMAT, ELIGIBLE_RECORD_FORMAT
from relive.storage.artifacts import canonical_json
from test_phase36 import Phase36Tests


class FakeChoiceBackend:
    synthetic = False
    prompts: list[str] = []
    duplicate = False
    def __init__(self, config): self.config = config
    def fingerprint(self): return {"adapter_version": "fixture-choice-v1", "model": "fixture"}
    def forced_choice_token_contract(self, request):
        self.__class__.prompts.append(request["prompt"])
        return {"choice_labels": ["A", "B", "C"], "choice_token_ids": {"A": 1, "B": 1 if self.duplicate else 2, "C": 3},
                "context_input_ids_sha256": hashlib.sha256((request["prompt"] + "|" + "|".join(request["image_paths"])).encode()).hexdigest(), "context_token_count": 10}
    def forced_choice_likelihood(self, request, *, choice_token_ids):
        self.__class__.prompts.append(request["prompt"])
        offsets = {"ORIGINAL": 3., "KEEP_TARGET": 2., "DROP_TARGET": 0., "DROP_MATCHED_CONTROL": 1., "FULL_GRAY": -1., "MISMATCHED_PUBLIC": -2.}
        a = offsets[request["context"]["variant"]]
        return {"choice_contract": self.forced_choice_token_contract(request), "choices": {
            "A": {"token_id": choice_token_ids["A"], "raw_logit": a, "log_probability": a},
            "B": {"token_id": choice_token_ids["B"], "raw_logit": 0., "log_probability": 0.},
            "C": {"token_id": choice_token_ids["C"], "raw_logit": -1., "log_probability": -1.}}}


class Phase4ATests(unittest.TestCase):
    def setUp(self):
        self.fixture = Phase36Tests(methodName="test_routes_contract_and_replay"); self.fixture.setUp()
        self.root, self.config, self.runtime, self.manifest, self.v3 = self.fixture.root, self.fixture.config, self.fixture.runtime, self.fixture.manifest, self.fixture.v3
        self.phase36 = self.root / "p36"
        phase36_preflight(config_path=self.config, runtime_path=self.runtime, prospective_manifest_path=self.manifest, phase35_v3_run_dir=self.v3 / "run", output_dir=self.phase36, require_real=False)
        phase36_execute(config_path=self.config, runtime_path=self.runtime, prospective_manifest_path=self.manifest, phase35_v3_run_dir=self.v3 / "run", output_dir=self.phase36, mode="run", require_real=False)
        rows = [json.loads(line) for line in self.runtime.read_text().splitlines()]
        by_claim = {row["target_claim"]["claim_id"]: row for row in rows}
        items = []
        # Freeze public alternatives before the fake scorer is constructed.
        for claim, other in (("phase35-local-001", "phase35-local-002"), ("phase35-local-002", "phase35-local-001")):
            row = by_claim[other]; frames = row["frames"]
            identity = [{"frame_id": f["frame_id"], "path": f["path"]} for f in frames]
            items.append({"source_id": f"phase35:{claim}", "frame_ids": [f["frame_id"] for f in frames], "frame_paths": [f["path"] for f in frames],
                          "selector": "fixture-fixed-other-public-source", "frame_manifest_sha256": hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()})
        mismatch = {"format": "relive-phase4a-mismatched-public-manifest-v1", "selection_status": "FROZEN_PRE_INFERENCE", "items": items}
        mismatch["mismatched_manifest_sha256"] = hashlib.sha256(json.dumps(mismatch, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.mismatch = self.root / "mismatch.json"; self.mismatch.write_text(json.dumps(mismatch))
        FakeChoiceBackend.prompts = []; FakeChoiceBackend.duplicate = False

    def tearDown(self): self.fixture.tearDown()
    def kwargs(self, out): return {"config_path": self.config, "runtime_path": self.runtime, "prospective_manifest_path": self.manifest, "phase35_v3_run_dir": self.v3 / "run", "phase36_run_dir": self.phase36 / "run", "mismatched_manifest_path": self.mismatch, "output_dir": out, "require_real": False, "backend_factory": FakeChoiceBackend}

    def test_preflight_run_replay_and_isolation(self):
        out = self.root / "phase4a"; plan = preflight(**self.kwargs(out))
        self.assertEqual(plan["status"], "PASS"); self.assertEqual(plan["model_calls_made"], 0)
        self.assertEqual(plan["calibration"]["status"], "MISSING_CALIBRATION_MANIFEST")
        frozen = json.loads((out / "phase4a_frozen_manifest.json").read_text())
        self.assertEqual(len(frozen["entries"]), 2)
        self.assertTrue(all(row["variants"]["FULL_GRAY"]["pixel_audits"] for row in frozen["entries"]))
        self.assertTrue(all(row["mismatched_selector"] == "fixture-fixed-other-public-source" for row in frozen["entries"]))
        first = execute(**self.kwargs(out), mode="run"); replay = execute(**self.kwargs(out), mode="replay")
        self.assertEqual(first["summary"]["new_model_calls"], 12); self.assertEqual(replay["summary"]["new_model_calls"], 0)
        self.assertEqual(replay["summary"]["cache_hits"], 12); self.assertTrue(first["audit"]["no_certificate_builder_called"])
        trace = [json.loads(line) for line in (out / "run" / "phase4a_trace.jsonl").read_text().splitlines()]
        self.assertEqual({row["claim_id"] for row in trace}, {"phase35-local-001", "phase35-local-002"})
        self.assertTrue(all(row["certificate_unchanged"] and row["new_verified_count"] == 0 and not row["gt_used"] for row in trace))
        self.assertTrue(all("model_fingerprint" in row for row in trace))
        self.assertTrue(all("DROP_TARGET" not in text and "MISMATCHED_PUBLIC" not in text for text in FakeChoiceBackend.prompts))
        self.assertTrue(all("FULL_GRAY" in row["variants"] for row in trace))

    def test_token_fail_closed_margin_and_prompt(self):
        self.assertTrue(choice_prompt("A visible instrument touches tissue.").endswith("\nAnswer:"))
        self.assertAlmostEqual(support_margin({"A": 0., "B": 0., "C": 0.}), -__import__("math").log(2))
        FakeChoiceBackend.duplicate = True
        with self.assertRaisesRegex(Phase4AError, "A/B/C next-token contract"):
            preflight(**self.kwargs(self.root / "duplicate"))

    def test_development_controls_can_run_without_phase35_inputs(self):
        runtime_rows = [json.loads(line) for line in self.runtime.read_text().splitlines()]
        frames = [row["frames"][0] for row in runtime_rows[:2]]
        specs = []
        for index, frame in enumerate(frames, 1):
            claim = f"A visible test object {index} is present."
            specs.append({"format": "relive-phase4a0-candidate-audit-spec-v1", "audit_case_id": f"development-{index}",
                          "source_claim_id": f"development-{index}", "claim_text": claim,
                          "claim_sha256": hashlib.sha256(claim.encode()).hexdigest(), "source_record_index": index,
                          "public_record_sha256": str(index) * 64, "frame_orders": [0], "frame_paths": [frame["path"]],
                          "frame_sha256": [hashlib.sha256(Path(frame["path"]).read_bytes()).hexdigest()],
                          "frozen_support_region": [.1,.1,.4,.4], "coordinate_system": "normalized_0_1_xyxy",
                          "intervention": {"operator":"opaque_gray", "operator_version":"relive-opaque-gray-hard-mask-v1", "parameters":{"fill_rgb":[127,127,127]}},
                          "matched_control_regions": [[.5,.1,.8,.4]], "candidate_manifest_sha256": "fixture",
                          "historical_pilot": False, "development_control": True, "gt_used": False})
        development = self.root / "development.jsonl"; development.write_text("".join(json.dumps(row)+"\n" for row in specs))
        review_a, review_b = self.root / "review-a.jsonl", self.root / "review-b.jsonl"
        review_a.write_text("fixture reviewer a\n"); review_b.write_text("fixture reviewer b\n")
        eligible_rows = []
        for spec in specs:
            eligible_rows.append({**spec, "format": ELIGIBLE_RECORD_FORMAT, "packet_id": "fixture-packet", "reviewer_ids": ["a", "b"],
                                  "review_file_sha256": {"a": hashlib.sha256(review_a.read_bytes()).hexdigest(), "b": hashlib.sha256(review_b.read_bytes()).hexdigest()},
                                  "review_file_paths": {"a": str(review_a), "b": str(review_b)}, "eligibility_status": "ELIGIBLE_FOR_PHASE4A_POSITIVE_CONTROL",
                                  "eligibility_policy_version": "fixture-v2", "cohort_kind": DEVELOPMENT_POSITIVE_CONTROL})
        eligible_path = self.root / "eligible.jsonl"; eligible_path.write_text("".join(canonical_json(row) + "\n" for row in eligible_rows))
        outer = {"format": ELIGIBLE_MANIFEST_FORMAT, "selection_status": "FROZEN_ELIGIBLE_CONTROLS", "cohort_kind": DEVELOPMENT_POSITIVE_CONTROL,
                 "gt_used": False, "eligible_controls_path": str(eligible_path), "eligible_controls_sha256": hashlib.sha256(eligible_path.read_bytes()).hexdigest(),
                 "eligible_count": 2, "inputs": {}}
        outer["manifest_content_sha256"] = hashlib.sha256(canonical_json(outer).encode()).hexdigest()
        eligibility = self.root / "eligible.manifest.json"; eligibility.write_text(json.dumps(outer))
        mismatch = self.root / "development-mismatch.json"
        self.assertEqual(freeze_development_mismatched_public(eligibility_manifest_path=eligibility, output_path=mismatch)["status"], "PASS")
        out = self.root / "development-audit"
        plan = preflight(config_path=self.config, mismatched_manifest_path=mismatch, output_dir=out,
                         eligibility_manifest_path=eligibility, source_mode=DEVELOPMENT_POSITIVE_CONTROL, require_real=False, backend_factory=FakeChoiceBackend)
        self.assertEqual(plan["development_controls"]["control_count"], 2)
        self.assertEqual(execute(config_path=self.config, mismatched_manifest_path=mismatch, output_dir=out,
                                 eligibility_manifest_path=eligibility, source_mode=DEVELOPMENT_POSITIVE_CONTROL, mode="run", require_real=False,
                                 backend_factory=FakeChoiceBackend)["summary"]["new_model_calls"], 12)
        review_a.write_text("tampered reviewer a\n")
        with self.assertRaisesRegex(Phase4AError, "FROZEN_INPUT_BYTES_CHANGED"):
            execute(config_path=self.config, mismatched_manifest_path=mismatch, output_dir=out,
                    eligibility_manifest_path=eligibility, source_mode=DEVELOPMENT_POSITIVE_CONTROL, mode="replay", require_real=False,
                    backend_factory=FakeChoiceBackend)

    def test_development_mode_requires_eligible_manifest(self):
        with self.assertRaisesRegex(Phase4AError, "REQUIRES_ELIGIBILITY"):
            preflight(config_path=self.config, mismatched_manifest_path=self.mismatch, output_dir=self.root / "no-eligibility",
                      source_mode=DEVELOPMENT_POSITIVE_CONTROL, require_real=False, backend_factory=FakeChoiceBackend)


if __name__ == "__main__": unittest.main()
