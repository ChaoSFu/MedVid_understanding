"""GPU-free tests for Phase 4A-0 development-control preparation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from relive.data.medvidu import load_public_records
from relive.phase4a0_development_controls import (DevelopmentControlError, ROI_TEMPLATE_FORMAT,
                                                   SELECTION_FORMAT, TARGET_OVERRIDE_FORMAT,
                                                   MATCHED_CONTROL_OVERRIDE_FORMAT, apply_matched_control_overrides,
                                                   apply_target_roi_overrides, freeze_development_controls,
                                                   prepare_development_controls)
from relive.phase4a0 import validate_candidate_specs


class DevelopmentControlPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.frames = self.root / "frames" / "dataset"
        self.frames.mkdir(parents=True)
        for index in range(3):
            Image.new("RGB", (24, 18), (index * 30, 20, 80)).save(self.frames / f"{index}.png")
        self.source = self.root / "source.json"
        self.source.write_text(json.dumps([{
            "id": "sample", "qa_type": "tal", "dataset_name": "fixture",
            "video": [f"/root/data/dataset/{index}.png" for index in range(3)],
            "sampled_video_frames": [0, 1, 2],
            "conversations": [{"from": "human", "value": "<video>What is shown?"},
                              {"from": "gpt", "value": "do not read"}],
        }]), encoding="utf-8")
        self.digest = load_public_records(self.source)[0][0].public_record_sha256

    def tearDown(self):
        self.temp.cleanup()

    def _row(self, case: str, claim: str, order: int) -> dict:
        return {"format": SELECTION_FORMAT, "development_case_id": case,
                "source_record_index": 0, "public_record_sha256": self.digest,
                "claim_id": f"claim-{case}", "claim_text": claim,
                "frozen_frame_orders": [order], "development_control": True, "gt_used": False}

    def test_allows_distinct_claims_from_same_source_and_binds_public_frame(self):
        selection = self.root / "selection.jsonl"
        rows = [self._row("dev-001", "A blue object is visible.", 0),
                self._row("dev-002", "A red object is visible.", 1)]
        selection.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        output = self.root / "out"
        report = prepare_development_controls(selection_path=selection, source_json=self.source,
                                              frame_root=self.root / "frames", source_prefix="/root/data",
                                              output_dir=output)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["model_calls_made"], 0)
        templates = [json.loads(line) for line in (output / "phase4a0_development_roi_freeze.template.jsonl").read_text().splitlines()]
        self.assertEqual(len(templates), 2)
        self.assertEqual(templates[0]["format"], ROI_TEMPLATE_FORMAT)
        self.assertIsNone(templates[0]["target_roi_normalized_0_1_xyxy"])
        self.assertTrue(Path(templates[0]["selected_public_frame_preview"]).is_file())
        expected = hashlib.sha256((self.frames / "0.png").read_bytes()).hexdigest()
        self.assertEqual(templates[0]["frame_sha256"], [expected])

    def test_rejects_duplicate_source_frame_selection(self):
        selection = self.root / "selection.jsonl"
        rows = [self._row("dev-001", "A blue object is visible.", 0),
                self._row("dev-002", "A red object is visible.", 0)]
        selection.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        with self.assertRaisesRegex(DevelopmentControlError, "DUPLICATE_SOURCE_FRAME_SELECTION"):
            prepare_development_controls(selection_path=selection, source_json=self.source,
                                         frame_root=self.root / "frames", source_prefix="/root/data",
                                         output_dir=self.root / "out")

    def test_applies_complete_user_target_overrides_without_selecting_controls(self):
        selection = self.root / "selection.jsonl"
        rows = [self._row("dev-001", "A blue object is visible.", 0),
                self._row("dev-002", "A red object is visible.", 1)]
        selection.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        prepared = self.root / "prepared"
        prepare_development_controls(selection_path=selection, source_json=self.source,
                                     frame_root=self.root / "frames", source_prefix="/root/data", output_dir=prepared)
        overrides = self.root / "overrides.jsonl"
        overrides.write_text("\n".join(json.dumps({"format": TARGET_OVERRIDE_FORMAT,
                                                       "development_case_id": case,
                                                       "target_roi_normalized_0_1_xyxy": region})
                                           for case, region in (("dev-001", [.1,.2,.5,.6]), ("dev-002", [.3,.2,.7,.6]))) + "\n")
        result = apply_target_roi_overrides(roi_template_path=prepared / "phase4a0_development_roi_freeze.template.jsonl",
                                            target_roi_overrides_path=overrides,
                                            output_path=self.root / "target-filled.jsonl")
        self.assertEqual(result["matched_control_status"], "AWAITING_HUMAN_SELECTION")
        rows = [json.loads(line) for line in (self.root / "target-filled.jsonl").read_text().splitlines()]
        self.assertEqual(rows[0]["selection_status"], "AWAITING_HUMAN_MATCHED_CONTROL")
        self.assertEqual(rows[0]["target_roi_normalized_0_1_xyxy"], [.1,.2,.5,.6])
        self.assertIsNone(rows[0]["matched_control_roi_normalized_0_1_xyxy"])

    def test_matched_controls_require_same_extent_and_no_overlap(self):
        selection = self.root / "selection.jsonl"
        rows = [self._row("dev-001", "A blue object is visible.", 0),
                self._row("dev-002", "A red object is visible.", 1)]
        selection.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        prepared = self.root / "prepared"
        prepare_development_controls(selection_path=selection, source_json=self.source,
                                     frame_root=self.root / "frames", source_prefix="/root/data", output_dir=prepared)
        targets = self.root / "targets.jsonl"
        targets.write_text("\n".join(json.dumps({"format": TARGET_OVERRIDE_FORMAT, "development_case_id": case,
                                                    "target_roi_normalized_0_1_xyxy": region})
                                        for case, region in (("dev-001", [.1,.2,.5,.6]), ("dev-002", [.3,.2,.6,.5]))) + "\n")
        target_sheet = self.root / "target-filled.jsonl"
        apply_target_roi_overrides(roi_template_path=prepared / "phase4a0_development_roi_freeze.template.jsonl",
                                   target_roi_overrides_path=targets, output_path=target_sheet)
        controls = self.root / "controls.jsonl"
        controls.write_text("\n".join(json.dumps({"format": MATCHED_CONTROL_OVERRIDE_FORMAT, "development_case_id": case,
                                                    "matched_control_roi_normalized_0_1_xyxy": region})
                                        for case, region in (("dev-001", [.55,.2,.95,.6]), ("dev-002", [.65,.2,.95,.5]))) + "\n")
        result = apply_matched_control_overrides(target_roi_worksheet_path=target_sheet,
                                                 matched_control_overrides_path=controls,
                                                 output_path=self.root / "ready.jsonl")
        self.assertEqual(result["geometry_audit"]["status"], "PASS")
        final = [json.loads(line) for line in (self.root / "ready.jsonl").read_text().splitlines()]
        self.assertEqual(final[0]["selection_status"], "READY_FOR_ZERO_MODEL_FREEZE_AUDIT")
        bad = self.root / "bad-controls.jsonl"
        bad.write_text(json.dumps({"format": MATCHED_CONTROL_OVERRIDE_FORMAT, "development_case_id": "dev-001",
                                   "matched_control_roi_normalized_0_1_xyxy": [.55,.2,.95,.6]}) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(DevelopmentControlError, "MUST_COVER"):
            apply_matched_control_overrides(target_roi_worksheet_path=target_sheet,
                                            matched_control_overrides_path=bad,
                                            output_path=self.root / "bad-ready.jsonl")

    def test_freeze_recomputes_bindings_and_emits_phase4a0_candidate_specs(self):
        selection = self.root / "selection.jsonl"
        selection.write_text(json.dumps(self._row("dev-001", "A blue object is visible.", 0)) + "\n", encoding="utf-8")
        prepared = self.root / "prepared"
        prepare_development_controls(selection_path=selection, source_json=self.source,
                                     frame_root=self.root / "frames", source_prefix="/root/data", output_dir=prepared)
        target = self.root / "target.jsonl"
        target.write_text(json.dumps({"format": TARGET_OVERRIDE_FORMAT, "development_case_id": "dev-001",
                                      "target_roi_normalized_0_1_xyxy": [.1,.2,.5,.6]}) + "\n")
        target_sheet = self.root / "target-filled.jsonl"
        apply_target_roi_overrides(roi_template_path=prepared / "phase4a0_development_roi_freeze.template.jsonl",
                                   target_roi_overrides_path=target, output_path=target_sheet)
        control = self.root / "control.jsonl"
        control.write_text(json.dumps({"format": MATCHED_CONTROL_OVERRIDE_FORMAT, "development_case_id": "dev-001",
                                       "matched_control_roi_normalized_0_1_xyxy": [.55,.2,.95,.6]}) + "\n")
        ready = self.root / "ready.jsonl"
        apply_matched_control_overrides(target_roi_worksheet_path=target_sheet,
                                        matched_control_overrides_path=control, output_path=ready)
        frozen = self.root / "frozen"
        report = freeze_development_controls(ready_worksheet_path=ready, output_dir=frozen)
        self.assertEqual(report["status"], "PASS")
        specs = validate_candidate_specs(frozen / "phase4a0_development_control_candidates.frozen.jsonl")
        self.assertTrue(specs[0]["development_control"])
        self.assertFalse(specs[0]["historical_pilot"])


if __name__ == "__main__":
    unittest.main()
