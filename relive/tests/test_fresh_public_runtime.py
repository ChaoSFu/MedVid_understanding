"""GPU-free Phase 3.5 fresh public-window runtime preparation tests."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from relive.fresh_public_runtime import FreshRuntimeError, prepare_fresh_public_runtime
from relive.data.schemas import load_runtime


class FreshPublicRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.frame_root = self.root / "frames"
        self.frame_root.mkdir()
        for index in range(4):
            Image.new("RGB", (16, 12), (index * 20, 40, 80)).save(self.frame_root / f"f{index}.png")
        self.source = self.root / "public.json"
        base = {
            "qa_type": "stg", "dataset_name": "public-fixture",
            "video": [f"/root/data/f{index}.png" for index in range(4)], "sampled_video_frames": [0, 1, 2, 3],
            "conversations": [{"from": "human", "value": "<video>\nWhat is visible?"}, {"from": "gpt", "value": "hidden"}],
            "struc_info": {"hidden": True}, "RC_info": {"hidden": True}, "metadata": {"hidden": True},
        }
        self.source.write_text(json.dumps([{**base, "id": "old-video"}, {**base, "id": "fresh-video"}]), encoding="utf-8")
        self.config = self.root / "config.json"; self.config.write_text("{}\n", encoding="utf-8")
        self.repo = Path(__file__).resolve().parents[2]

    def tearDown(self):
        self.temp.cleanup()

    def _claims(self, index: int = 1) -> Path:
        path = self.root / "claims.jsonl"
        path.write_text(json.dumps({"source_record_index": index, "public_record_sha256": "0" * 64,
                                    "target_claim": {"claim_id": "fresh-local", "text": "The jaws of the forceps are contacting the tissue."}}) + "\n", encoding="utf-8")
        return path

    def _windows(self, orders=(1, 2, 3)) -> Path:
        path = self.root / "windows.jsonl"
        path.write_text(json.dumps({"claim_id": "fresh-local", "frozen_frame_orders": list(orders),
                                    "human_public_visual_confirmation": True,
                                    "human_public_visual_confirmation_note": "Forceps jaws visibly contact tissue in the frozen public frames."}) + "\n", encoding="utf-8")
        return path

    def _correct_claim_digest(self, claims: Path) -> None:
        # The public source projection derives index 1 and its public digest.
        from relive.data.medvidu import load_public_records
        record = load_public_records(self.source)[0][1]
        row = json.loads(claims.read_text())
        row["source_record_index"] = record.source_record_index
        row["public_record_sha256"] = record.public_record_sha256
        claims.write_text(json.dumps(row) + "\n", encoding="utf-8")

    def test_freezes_exact_window_without_human_text_in_runtime(self):
        claims, windows = self._claims(), self._windows()
        self._correct_claim_digest(claims)
        output = self.root / "fresh-output"
        report = prepare_fresh_public_runtime(source_json=self.source, frame_root=self.frame_root, source_prefix="/root/data",
                                              fresh_claims=claims, window_confirmations=windows, config=self.config,
                                              output_dir=output, repo_root=self.repo)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["model_calls_made"], 0)
        runtime = output / "frozen_public_window.runtime.jsonl"
        sample = load_runtime(runtime)[0]
        self.assertEqual([frame.order for frame in sample.frames], [1, 2, 3])
        self.assertEqual(sample.target_claim.time_scope["frame_ids"], [frame.frame_id for frame in sample.frames])
        runtime_body = runtime.read_text()
        self.assertNotIn("human_public_visual_confirmation", runtime_body)
        self.assertNotIn("bbox", runtime_body)
        manifest = json.loads((output / "fresh_source_manifest.jsonl").read_text())
        self.assertTrue(manifest["human_public_visual_confirmation"])
        self.assertEqual(manifest["claim_scope"], "LOCAL_ATOMIC")

    def test_rejects_old_source_index_and_nonlocal_claim(self):
        old_claims, windows = self._claims(index=0), self._windows()
        with self.assertRaisesRegex(FreshRuntimeError, "excluded development"):
            prepare_fresh_public_runtime(source_json=self.source, frame_root=self.frame_root, source_prefix="/root/data",
                                         fresh_claims=old_claims, window_confirmations=windows, config=self.config,
                                         output_dir=self.root / "old", repo_root=self.repo)
        claims = self._claims(); self._correct_claim_digest(claims)
        row = json.loads(claims.read_text())
        row["target_claim"]["text"] = "At least one laparoscopic surgical instrument is visibly present."
        claims.write_text(json.dumps(row) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(FreshRuntimeError, "not LOCAL_ATOMIC"):
            prepare_fresh_public_runtime(source_json=self.source, frame_root=self.frame_root, source_prefix="/root/data",
                                         fresh_claims=claims, window_confirmations=windows, config=self.config,
                                         output_dir=self.root / "nonlocal", repo_root=self.repo)

    def test_rejects_nonconsecutive_windows_and_roi_in_confirmation(self):
        claims = self._claims(); self._correct_claim_digest(claims)
        bad_window = self._windows(orders=(0, 2))
        with self.assertRaisesRegex(FreshRuntimeError, "consecutive"):
            prepare_fresh_public_runtime(source_json=self.source, frame_root=self.frame_root, source_prefix="/root/data",
                                         fresh_claims=claims, window_confirmations=bad_window, config=self.config,
                                         output_dir=self.root / "bad-window", repo_root=self.repo)
        row = json.loads(self._windows().read_text())
        row["human_public_visual_confirmation_note"] = "ROI covers the forceps."
        self._windows().write_text(json.dumps(row) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(FreshRuntimeError, "ROI or GT"):
            prepare_fresh_public_runtime(source_json=self.source, frame_root=self.frame_root, source_prefix="/root/data",
                                         fresh_claims=claims, window_confirmations=self.root / "windows.jsonl", config=self.config,
                                         output_dir=self.root / "bad-note", repo_root=self.repo)


if __name__ == "__main__":
    unittest.main()
