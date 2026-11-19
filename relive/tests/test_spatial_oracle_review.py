from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from relive.spatial_reliability.common import SpatialReliabilityError, sha256_path
from relive.spatial_reliability.oracle_review import ORACLE_REVIEW_FORMAT, prepare_oracle_review


class SpatialOracleReviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.runtime_rows = []
        self.oracle_rows = []
        self.frame_hashes = {}
        for index in range(2):
            frame = self.root / f"frame-{index}.png"
            Image.new("RGB", (100, 80), (20 + index, 40, 60)).save(frame)
            digest = hashlib.sha256(frame.read_bytes()).hexdigest()
            self.frame_hashes[frame] = digest
            sample_id = f"sample-{index}"
            record_sha = f"{index + 1:064x}"
            self.runtime_rows.append({
                "sample_id": sample_id,
                "claim_id": f"claim-{index}",
                "claim_text": f"Instrument {index} is visible.",
                "dataset_name": "Fixture",
                "video_id": "video-a",
                "source_record_sha256": record_sha,
                "frame_path": str(frame),
                "frame_sha256": digest,
            })
            self.oracle_rows.append({
                "sample_id": sample_id,
                "source_record_sha256": record_sha,
                "gt_bbox": [20, 16, 80, 64],
            })
        self.runtime = self._jsonl("runtime.jsonl", self.runtime_rows)
        self.oracle = self._jsonl("oracle.jsonl", self.oracle_rows)
        self.result_rows = []
        classes = ("ORACLE_SINGLE_REGION_FEASIBLE", "ORACLE_RESIDUAL_SUPPORT")
        drops = ("INSUFFICIENT", "SUPPORTED")
        for index, (classification, drop) in enumerate(zip(classes, drops)):
            self.result_rows.append({
                "sample_id": f"sample-{index}",
                "claim_id": f"claim-{index}",
                "classification": classification,
                "semantic": {"ORIGINAL": "SUPPORTED", "KEEP_GT": "SUPPORTED", "DROP_GT": drop},
                "gt_region_normalized_0_1_xyxy": [0.2, 0.2, 0.8, 0.8],
                "bindings": {
                    "runtime_manifest_sha256": sha256_path(self.runtime),
                    "oracle_manifest_sha256": sha256_path(self.oracle),
                    "source_frame_sha256": self.runtime_rows[index]["frame_sha256"],
                },
            })
        self.results = self._jsonl("results.jsonl", self.result_rows)

    def tearDown(self):
        self.temporary.cleanup()

    def _jsonl(self, name, rows):
        path = self.root / name
        path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
                        encoding="utf-8")
        return path

    def test_creates_blank_review_and_visualizations_without_activity(self):
        output = self.root / "out"
        manifest = prepare_oracle_review(
            runtime_manifest=self.runtime,
            oracle_manifest=self.oracle,
            oracle_results=self.results,
            output_dir=output,
        )
        self.assertEqual(manifest["format"], ORACLE_REVIEW_FORMAT)
        self.assertEqual(manifest["status"], "PASS")
        self.assertEqual(manifest["cohort_count"], 2)
        self.assertEqual(manifest["human_decisions_filled"], 0)
        self.assertTrue(manifest["source_frames_unchanged"])
        self.assertEqual(manifest["model_calls_made"], 0)
        self.assertFalse(manifest["cache_opened"])
        self.assertEqual(manifest["cache_writes"], 0)
        self.assertFalse(manifest["certificate_created"])
        self.assertEqual(manifest["new_verified_count"], 0)
        rows = [json.loads(line) for line in
                (output / "oracle_human_review_template_v1.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(set(row["human_review"].values()) == {""} for row in rows))
        self.assertTrue((output / "oracle_review_contact_sheet_v1.png").is_file())
        self.assertEqual(len(list((output / "overlays").glob("*.png"))), 2)
        for path, digest in self.frame_hashes.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_outputs_are_byte_deterministic(self):
        first, second = self.root / "first", self.root / "second"
        for output in (first, second):
            prepare_oracle_review(runtime_manifest=self.runtime, oracle_manifest=self.oracle,
                                  oracle_results=self.results, output_dir=output)
        expected = {
            "oracle_review_items_v1.jsonl",
            "oracle_review_visualization_manifest_v1.jsonl",
            "oracle_human_review_template_v1.jsonl",
            "oracle_review_contact_sheet_v1.png",
            "oracle_review_manifest_v1.json",
            "README.md",
        }
        for relative in expected:
            self.assertEqual((first / relative).read_bytes(), (second / relative).read_bytes())
        for row in self.result_rows:
            relative = Path("overlays") / f"{row['sample_id']}.png"
            self.assertEqual((first / relative).read_bytes(), (second / relative).read_bytes())

    def test_tampered_binding_and_duplicate_json_fail_closed(self):
        tampered = [dict(row) for row in self.result_rows]
        tampered[0] = {**tampered[0], "claim_id": "wrong"}
        path = self._jsonl("tampered.jsonl", tampered)
        with self.assertRaisesRegex(SpatialReliabilityError, "BINDING_INVALID"):
            prepare_oracle_review(runtime_manifest=self.runtime, oracle_manifest=self.oracle,
                                  oracle_results=path, output_dir=self.root / "bad")

        duplicate = self.root / "duplicate.jsonl"
        duplicate.write_text('{"sample_id":"x","sample_id":"y"}\n', encoding="utf-8")
        with self.assertRaisesRegex(SpatialReliabilityError, "RESULTS_INVALID"):
            prepare_oracle_review(runtime_manifest=self.runtime, oracle_manifest=self.oracle,
                                  oracle_results=duplicate, output_dir=self.root / "duplicate-out")

    def test_review_module_has_no_runtime_backend_cache_or_certificate_import(self):
        source = (Path(__file__).parents[1] /
                  "src/relive/spatial_reliability/oracle_review.py").read_text()
        self.assertNotIn("relive.backends", source)
        self.assertNotIn("relive.storage.cache", source)
        self.assertNotIn("relive.certificate", source)
        self.assertNotIn("ModelSession", source)


if __name__ == "__main__":
    unittest.main()
