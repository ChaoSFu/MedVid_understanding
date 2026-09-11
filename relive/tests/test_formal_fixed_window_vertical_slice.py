"""The formal vertical slice must never inject a diagnostic calibration ROI."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "formal_fixed_window_vertical_slice.py"
CONFIG = ROOT / "configs" / "mock_smoke.yaml"


class FormalFixedWindowVerticalSliceTests(unittest.TestCase):
    def _manifest(self, image: Path) -> Path:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        control = {
            "candidate_id": "positive-public-window", "kind": "positive_control_candidate",
            "atomic_claim_en": "A synthetic local object is visibly present.",
            "frame_ids": ["public:f0", "public:f1", "public:f2"],
            "frame_paths": [str(image)] * 3, "frame_orders": [5, 6, 7],
            "dataset_name": "public-test", "source_qa_type": "diagnostic",
            # Must remain diagnostic-only and excluded from core inputs.
            "diagnostic_only_bbox_normalized_xyxy": [0.61, 0.11, 0.91, 0.71],
            "matched_control_bbox_normalized_xyxy": [0.01, 0.11, 0.31, 0.71],
        }
        manifest = {"format": "relive-calibration-manifest-frozen-v1", "calibration_only": True,
                    "selection_status": "FROZEN_PRE_INFERENCE", "git_commit": head,
                    "selected_positive_control": control, "ancillary_controls": []}
        manifest["frozen_manifest_sha256"] = hashlib.sha256(json.dumps(manifest, ensure_ascii=False,
            sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        path = image.parent / "manifest.json"; path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def test_formal_run_uses_automatic_proposal_and_replays_from_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); image = root / "public.png"
            Image.new("RGB", (60, 40), (20, 80, 140)).save(image)
            manifest = self._manifest(image); output = root / "vertical"
            env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
            base = [sys.executable, str(SCRIPT), "--manifest", str(manifest), "--config", str(CONFIG), "--output-dir", str(output)]
            subprocess.run([*base, "--mode", "preflight"], cwd=ROOT, env=env, check=True, capture_output=True, text=True)
            subprocess.run([*base, "--mode", "run"], cwd=ROOT, env=env, check=True, capture_output=True, text=True)
            replay = subprocess.run([*base, "--mode", "replay"], cwd=ROOT, env=env, check=True, capture_output=True, text=True)
            report = json.loads((output / "formal_vertical_slice_run.json").read_text())
            runtime = (output / "preflight" / "formal_fixed_window.runtime.jsonl").read_text()
            config = (output / "preflight" / "formal_fixed_window.config.json").read_text()
            self.assertTrue(report["formal_fixed_window_positive_verified"])
            self.assertEqual(report["automatic_spatial_proposal"]["support_region"], [0.1, 0.1, 0.4, 0.4])
            self.assertEqual(report["formal_certificate"]["provenance"]["intervention_protocol"]["operator"], "opaque_gray")
            self.assertEqual(json.loads((output / "run" / "manifest" / "run.json").read_text())["spatial_intervention"]["operator"], "opaque_gray")
            self.assertNotIn("0.61", runtime)
            self.assertNotIn("0.61", config)
            self.assertEqual(json.loads(replay.stdout)["summary"]["new_model_calls"], 0)


if __name__ == "__main__":
    unittest.main()
