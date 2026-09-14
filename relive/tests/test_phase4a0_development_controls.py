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
                                                   apply_target_roi_overrides, prepare_development_controls)


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


if __name__ == "__main__":
    unittest.main()
