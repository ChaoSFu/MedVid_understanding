"""GPU-free public-record contact-sheet export tests."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from relive.data.medvidu import load_public_records
from relive.public_record_candidates import PublicRecordCandidateError, export_public_record_candidates


class PublicRecordCandidateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.frames = self.root / "frames"; self.frames.mkdir()
        for number in range(4):
            Image.new("RGB", (24, 16), (number * 30, 30, 90)).save(self.frames / f"{number}.png")
        rows = []
        for index in range(4):
            rows.append({"id": f"record-{index}", "qa_type": "stg", "dataset_name": "public-fixture",
                         "video": [f"/root/data/{number}.png" for number in range(4)], "sampled_video_frames": [0, 1, 2, 3],
                         "conversations": [{"from": "human", "value": "<video>\nPublic question"}, {"from": "gpt", "value": "hidden"}],
                         "struc_info": {"hidden": True}, "RC_info": {"hidden": True}, "metadata": {"hidden": True}})
        self.source = self.root / "source.json"; self.source.write_text(json.dumps(rows), encoding="utf-8")
        records, _ = load_public_records(self.source)
        selector_rows = []
        for record in records:
            selector_rows.append({"source_record_index": record.source_record_index, "sample_id": record.sample_id,
                                  "public_record_sha256": record.public_record_sha256, "question_sha256": "a" * 64,
                                  "qa_type": record.qa_type, "question": record.question, "frame_count": len(record.video_paths),
                                  "first_verified_frame_path": str(self.frames / "0.png"), "last_verified_frame_path": str(self.frames / "3.png"),
                                  "dataset_name": record.dataset_name})
        self.selector = self.root / "selector.jsonl"
        self.selector.write_text("".join(json.dumps(row) + "\n" for row in selector_rows), encoding="utf-8")
        self.repo = Path(__file__).resolve().parents[2]

    def tearDown(self):
        self.temp.cleanup()

    def test_exports_fresh_public_frames_without_claims(self):
        output = self.root / "out"
        report = export_public_record_candidates(public_selector=self.selector, source_json=self.source, frame_root=self.frames,
                                                 source_prefix="/root/data", output_dir=output, repo_root=self.repo,
                                                 batch_size=2, page_size=2, columns=2, thumbnail_width=64, thumbnail_height=48)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["model_calls_made"], 0)
        self.assertEqual(report["exclusion_and_duplicate_audit"]["selected_source_record_indices"], [1, 2])
        rows = [json.loads(line) for line in (output / "fresh_source_record_candidates.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["public_frame_orders"], [0, 1, 2, 3])
        self.assertNotIn("target_claim", rows[0])
        self.assertIsNone(rows[0]["human_visual_confirmation"])
        for page in rows[0]["contact_sheet_pages"]:
            self.assertTrue((output / page["path"]).is_file())

    def test_rejects_small_batch_when_only_excluded_records_exist(self):
        restricted = self.root / "restricted.jsonl"
        first = json.loads(self.selector.read_text().splitlines()[0])
        restricted.write_text(json.dumps(first) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(PublicRecordCandidateError, "only 0 fresh"):
            export_public_record_candidates(public_selector=restricted, source_json=self.source, frame_root=self.frames,
                                            source_prefix="/root/data", output_dir=self.root / "restricted-out", repo_root=self.repo,
                                            batch_size=1)


if __name__ == "__main__":
    unittest.main()
