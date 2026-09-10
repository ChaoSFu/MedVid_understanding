"""Contract test for the no-model calibration intervention/operator visual sweep."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "calibration_intervention_sweep.py"


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


class CalibrationInterventionSweepTests(unittest.TestCase):
    def test_writes_operator_contact_sheets_without_model_or_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = root / "public.png"
            image = Image.new("RGB", (80, 60), (20, 40, 60))
            for x in range(15, 35):
                for y in range(15, 35):
                    image.putpixel((x, y), (220, 90, 30))
            image.save(frame)
            expected = {"ORIGINAL": "SUPPORTED", "KEEP_TARGET": "SUPPORTED",
                        "DROP_TARGET": "INSUFFICIENT", "DROP_MATCHED_CONTROL": "SUPPORTED"}
            controls = []
            for identifier, kind in (("positive", "positive_control_candidate"),
                                     ("broad", "broad_multiple_support_uncertainty_control"),
                                     ("negative", "visually_exclusive_negative_control")):
                frames = [{"frame_id": f"{identifier}:frame:0", "path": str(frame), "order": 0}]
                control = {"candidate_id": identifier, "kind": kind, "dataset_name": "fixture",
                           "source_qa_type": "fixture", "frame_ids": [frames[0]["frame_id"]],
                           "frame_paths": [str(frame)], "frame_orders": [0], "frame_sha256": digest(frames),
                           "atomic_claim_en": f"A visible {identifier} object.",
                           "diagnostic_only_bbox_normalized_xyxy": [0.1, 0.2, 0.4, 0.5],
                           "matched_control_bbox_normalized_xyxy": [0.6, 0.2, 0.9, 0.5],
                           "expected_outcomes_frozen_pre_inference": expected}
                control["claim_sha256"] = hashlib.sha256(control["atomic_claim_en"].encode()).hexdigest()
                controls.append(control)
            manifest = {"format": "relive-calibration-manifest-frozen-v1", "calibration_only": True,
                        "selection_status": "FROZEN_PRE_INFERENCE", "git_commit": "0" * 40,
                        "selected_positive_control": controls[0], "ancillary_controls": controls[1:]}
            manifest["frozen_manifest_sha256"] = digest(manifest)
            manifest_path = root / "frozen.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            output = root / "sweep"
            completed = subprocess.run([sys.executable, str(SCRIPT), "--manifest", str(manifest_path),
                                         "--output-dir", str(output), "--blur-radii", "4,8"], cwd=ROOT,
                                        env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")},
                                        text=True, capture_output=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads((output / "intervention_sweep_report.json").read_text())
            self.assertEqual(report["model_calls_made"], 0)
            self.assertFalse(report["cache_mutated"])
            self.assertEqual(len(report["controls"]), 3)
            self.assertEqual(len(report["controls"][0]["operator_sweeps"]), 4)
            self.assertTrue(all(Path(row["contact_sheet"]).is_file()
                                for control in report["controls"] for row in control["operator_sweeps"]))
            drop = report["controls"][0]["operator_sweeps"][0]["variants"][1]
            self.assertEqual(drop["variant"], "DROP_TARGET")
            self.assertEqual(drop["effect_class"], "EFFECTIVE_PIXEL_CHANGE")
            self.assertGreater(drop["pixel_metrics"]["roi_changed_pixel_count"], 0)
            self.assertEqual(drop["pixel_metrics"]["outside_changed_pixel_count"], 0)
            keep = report["controls"][0]["operator_sweeps"][0]["variants"][0]
            self.assertEqual(keep["effect_class"], "EFFECTIVE_PIXEL_CHANGE")
            self.assertGreater(keep["pixel_metrics"]["outside_changed_pixel_count"], 0)


if __name__ == "__main__":
    unittest.main()
