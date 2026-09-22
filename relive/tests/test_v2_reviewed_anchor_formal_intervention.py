"""GPU-free contracts for reviewed-anchor formal intervention orchestration."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from relive.config import load_config
from relive.v2.reviewed_anchor_formal_intervention import (
    ReviewedAnchorInterventionError, _trace_summary, execute, preflight, prepare_warning_adjudication_template,
)


ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")), encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


class ReviewedAnchorFormalInterventionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.image = self.root / "public.png"
        image = Image.new("RGB", (40, 30))
        image.putdata([(x * 6, y * 8, 90) for y in range(30) for x in range(40)])
        image.save(self.image)
        self.frame_sha = hashlib.sha256(self.image.read_bytes()).hexdigest()
        cfg = load_config(ROOT / "configs" / "mock_smoke.yaml")
        cfg["policy"] = {"name": "semantic_spatial", "version": "relive-v1-policy-1", "strict_alternatives": True}
        cfg["adaptation"] = {"enabled": False, "actions": [], "expand_frames": 1}
        cfg["spatial"]["intervention"] = {"operator": "opaque_gray", "operator_version": "relive-opaque-gray-hard-mask-v1", "parameters": {"fill_rgb": [127, 127, 127]}}
        cfg["budget"].update(max_calls=4, max_spatial_proposals=1, max_candidates=1, max_rounds=1)
        self.config = self.root / "formal.json"; write_json(self.config, cfg)
        self._inputs()

    def tearDown(self): self.tmp.cleanup()

    def _inputs(self):
        # Keep the production cardinalities in the fixture: this catches a
        # future relaxation of the public review contract without any model.
        anchors = [f"anchor-{i:02d}" for i in range(75)]
        raw, reviews = [], []
        for index, anchor in enumerate(anchors):
            count = 5 if index < 21 else 4  # 21*5 + 54*4 == 321
            components = []
            for role_index in range(count):
                role = ["OPERATOR_HAND", "BASE_PLATE", "HAND_BASE_INTERFACE", "TARGET_SKIN_AREA", "BASE_SKIN_INTERFACE"][role_index]
                components.append({"role": role, "visibility": "VISIBLE", "bbox_normalized_xyxy": [0.1, 0.1, 0.3, 0.3]})
                reviews.append({"anchor_candidate_id": anchor, "role": role, "model_visibility": "VISIBLE", "human_overlay_label": "ACCEPTED", "reason_code": ""})
            raw.append({"anchor_candidate_id": anchor, "components": components, "frame_sha256": self.frame_sha,
                        "image_path": str(self.image), "observation_claim_id": f"obs-{index % 15:02d}"})
        self.raw = self.root / "raw.jsonl"; self.review = self.root / "review.jsonl"
        write_jsonl(self.raw, raw); write_jsonl(self.review, reviews)
        eligible = []
        for index in range(5):
            anchor = anchors[index]
            components = [{"role": "OPERATOR_HAND", "bbox_usable_for_intervention": True, "adjudicated_bbox": [0.05, 0.05, 0.22, 0.25]},
                          {"role": "BASE_PLATE", "bbox_usable_for_intervention": True, "adjudicated_bbox": [0.3, 0.1, 0.5, 0.3]},
                          {"role": "HAND_BASE_INTERFACE", "bbox_usable_for_intervention": True, "adjudicated_bbox": [0.55, 0.2, 0.7, 0.4]}]
            eligible.append({"anchor_candidate_id": anchor, "claim_evidence_route": "ELIGIBLE_SPATIAL_ANCHOR",
                             "required_component_roles": [item["role"] for item in components], "adjudicated_components": components,
                             "frame_sha256": self.frame_sha, "image_path": str(self.image), "source_frame_reference": "public/000.jpg",
                             "timestamp_seconds": 2.0, "observation_claim_id": f"obs-{index:02d}", "observation_claim": f"A visible action {index}."})
        self.eligible = self.root / "eligible.jsonl"; write_jsonl(self.eligible, eligible)
        decisions = [{"observation_claim_id": f"obs-{index:02d}", "selected_anchor_candidate_id": anchors[index], "route": "ELIGIBLE_SPATIAL_ANCHOR"} for index in range(5)]
        decisions += [{"observation_claim_id": f"obs-{index + 5:02d}", "selected_anchor_candidate_id": anchors[index + 5], "route": "RELATIONAL_COMPOSITE_REQUIRED"} for index in range(5)]
        decisions += [{"observation_claim_id": f"obs-{index + 10:02d}", "selected_anchor_candidate_id": anchors[index + 10], "route": "TEMPORAL_REACQUIRE"} for index in range(5)]
        self.decisions = self.root / "decisions.jsonl"; write_jsonl(self.decisions, decisions)
        self.queue = self.root / "warnings.jsonl"
        warning_rows = []
        for index in range(5):
            warning_rows.append({"warning_code": "DUPLICATE_GROUNDING_INCONSISTENT_REVIEW", "grounding_signature": f"sig-{index}",
                                 "reviews": [{"anchor_candidate_id": anchors[index] if index == 0 else anchors[index + 20], "role": "HAND_BASE_INTERFACE"}]})
        write_jsonl(self.queue, warning_rows)
        self.report = self.root / "report.json"
        write_json(self.report, {"status": "PASS", "raw_anchor_manifest_sha256": hashlib.sha256(self.raw.read_bytes()).hexdigest(),
                                 "review_jsonl_sha256": hashlib.sha256(self.review.read_bytes()).hexdigest(), "raw_anchor_count": 75,
                                 "review_component_count": 321, "observation_route_counts": {"ELIGIBLE_SPATIAL_ANCHOR": 5, "RELATIONAL_COMPOSITE_REQUIRED": 5, "TEMPORAL_REACQUIRE": 5}})

    def _preflight(self, out: Path, adjudication=None):
        return preflight(eligible_manifest=self.eligible, observation_decisions=self.decisions, raw_grounding=self.raw,
                         human_review=self.review, validation_report=self.report, warning_queue=self.queue,
                         warning_adjudication=adjudication, config_path=self.config, output_dir=out,
                         raw_sha_prefix="", review_sha_prefix="")

    def _approved_adjudication(self):
        template = self.root / "template.jsonl"; prepare_warning_adjudication_template(self.queue, template)
        rows = [json.loads(line) for line in template.read_text().splitlines()]
        for row in rows:
            row.update(decision="CONTEXT_SPECIFIC_ALLOWED", reviewer_id="reviewer-1", rationale="Different public contexts were inspected.")
        ready = self.root / "ready.jsonl"; write_jsonl(ready, rows); return ready

    def test_unresolved_warning_on_required_eligible_role_blocks(self):
        out = self.root / "blocked"; result = self._preflight(out)
        self.assertEqual(result["status"], "FORMAL_RUN_BLOCKED")
        gate = json.loads((out / "warning_adjudication_gate.json").read_text())
        self.assertEqual(gate["blocking_warning_count"], 1)
        with self.assertRaisesRegex(ReviewedAnchorInterventionError, "FORMAL_RUN_BLOCKED"):
            execute(output_dir=out, config_path=self.config, mode="run")

    def test_unresolved_warning_outside_cohort_is_bound_but_does_not_block(self):
        rows = [json.loads(line) for line in self.queue.read_text().splitlines()]
        rows[0]["reviews"] = [{"anchor_candidate_id": "anchor-50", "role": "HAND_BASE_INTERFACE"}]
        write_jsonl(self.queue, rows)
        # The completed file cannot be reused because the source queue hash is
        # deliberately part of the reviewer decision binding.
        out = self.root / "unrelated"; result = self._preflight(out)
        self.assertEqual(result["status"], "PASS")
        gate = json.loads((out / "warning_adjudication_gate.json").read_text())
        self.assertEqual(gate["blocking_warning_count"], 0)
        self.assertEqual(gate["warning_count"], 5)

    def test_template_approved_preflight_run_replay_and_no_label_prompt_leakage(self):
        out = self.root / "ok"; self.assertEqual(self._preflight(out, self._approved_adjudication())["status"], "PASS")
        plan = json.loads((out / "reviewed_anchor_intervention_plan.json").read_text())
        self.assertEqual(len(plan["candidates"]), 5); self.assertEqual(plan["formal_variants"], ["ORIGINAL", "KEEP_TARGET", "DROP_TARGET", "DROP_MATCHED_CONTROL"])
        self.assertNotIn("ACCEPTED", json.dumps(plan["candidates"]))
        first = execute(output_dir=out, config_path=self.config, mode="run", smoke_candidate_id=plan["candidates"][0]["candidate_id"])
        replay = execute(output_dir=out, config_path=self.config, mode="replay", smoke_candidate_id=plan["candidates"][0]["candidate_id"])
        self.assertEqual(first["summary"]["new_model_calls"], 4)
        self.assertFalse(first["summary"]["full_formal_cohort_complete"])
        self.assertIsNone(first["summary"]["cohort_stop_recommendation"])
        self.assertIsNone(first["summary"]["smoke_observation"])
        self.assertEqual(replay["summary"]["new_model_calls"], 0)
        trace = json.loads((out / "run" / "reviewed_anchor_intervention_trace.jsonl").read_text())
        self.assertFalse(trace["human_labels_in_model_request"])
        self.assertEqual(trace["certificate"]["certificate_status"], "NOT_APPLICABLE_SYNTHETIC_TEST")

    def test_full_cohort_has_exactly_four_core_variants_and_reference_bound_pixel_audits(self):
        out = self.root / "full"; self._preflight(out, self._approved_adjudication())
        execute(output_dir=out, config_path=self.config, mode="run")
        rows = [json.loads(line) for line in (out / "run" / "reviewed_anchor_intervention_trace.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 5)
        for row in rows:
            self.assertEqual(row["formal_variants"], ["ORIGINAL", "KEEP_TARGET", "DROP_TARGET", "DROP_MATCHED_CONTROL"])
            self.assertTrue(row["spatial"]["pixel_audit_refs"])
            self.assertEqual(len(row["spatial"]["references"]["controls"]), 1)

    def test_hash_and_count_mismatch_fail_closed(self):
        self.report.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ReviewedAnchorInterventionError, "VALIDATION_REPORT"):
            self._preflight(self.root / "bad", self._approved_adjudication())

    def test_plan_and_synthetic_summary_are_byte_deterministic(self):
        ready = self._approved_adjudication()
        first, second = self.root / "first", self.root / "second"
        self._preflight(first, ready); self._preflight(second, ready)
        self.assertEqual((first / "reviewed_anchor_intervention_plan.json").read_bytes(),
                         (second / "reviewed_anchor_intervention_plan.json").read_bytes())
        execute(output_dir=first, config_path=self.config, mode="run")
        execute(output_dir=second, config_path=self.config, mode="run")
        self.assertEqual((first / "run" / "reviewed_anchor_summary.json").read_bytes(),
                         (second / "run" / "reviewed_anchor_summary.json").read_bytes())

    def test_drop_supported_smoke_never_triggers_five_candidate_stop_rule(self):
        row = {"certificate": {"final_status": "UNCERTAIN"},
               "spatial": {"references": {"drop": {"semantic_status": "SUPPORTED"}}},
               "usage": {"new_calls": 4, "cache_hits": 0}, "new_verified_count": 0}
        smoke = _trace_summary([row], mode="run", synthetic=False, formal_cohort_count=5)
        full = _trace_summary([row] * 5, mode="run", synthetic=False, formal_cohort_count=5)
        self.assertEqual(smoke["smoke_observation"], "DROP_TARGET_SUPPORTED_INTERVENTION_INSENSITIVE")
        self.assertIsNone(smoke["cohort_stop_recommendation"])
        self.assertEqual(full["cohort_stop_recommendation"], "PAUSE_FORMAL_EXPANSION_DIAGNOSTIC_REQUIRED")


if __name__ == "__main__": unittest.main()
