from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from relive.spatial_reliability.common import SpatialReliabilityError
from relive.spatial_reliability.posthoc_v2 import EVALUATION_V2_FORMAT, evaluate_v2


STATUSES = (
    "DEPENDENCY_VERIFIED",
    "DEPENDENCY_SUPPORTED_NO_CONTROL",
    "DEPENDENCY_UNRESOLVED",
    "BASE_UNSUPPORTED",
    "INTERVENTION_UNSTABLE",
    "REFINEMENT_NO_OP",
)


class SpatialPosthocV2Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.runtime_rows = []
        self.oracle_rows = []
        self.result_rows = []
        self.source_hashes = {}
        boxes = (
            [200, 200, 800, 800],
            [0, 0, 500, 500],
            [600, 600, 900, 900],
            [0, 0, 200, 200],
            [200, 200, 800, 800],
            [200, 200, 800, 800],
        )
        for index, (status, box) in enumerate(zip(STATUSES, boxes)):
            sample_id = f"sample-{index}"
            frame = self.root / f"frame-{index}.png"
            Image.new("RGB", (100, 100), (30 + index, 60, 90)).save(frame)
            frame_sha = hashlib.sha256(frame.read_bytes()).hexdigest()
            self.source_hashes[frame] = frame_sha
            record_sha = f"{index + 1:064x}"
            self.runtime_rows.append({
                "sample_id": sample_id,
                "source_record_sha256": record_sha,
                "frame_path": str(frame),
                "frame_sha256": frame_sha,
            })
            self.oracle_rows.append({
                "sample_id": sample_id,
                "source_record_sha256": record_sha,
                "gt_bbox": [20, 20, 80, 80],
            })
            trace = [{
                "stage": "R0_PROPOSAL",
                "result": {"status": "PROPOSED", "bbox_normalized_0_1000": box},
            }]
            if status == "REFINEMENT_NO_OP":
                trace.append({
                    "stage": "R1_PROPOSAL",
                    "result": {"status": "PROPOSED", "bbox_normalized_0_1000": box},
                })
            self.result_rows.append({
                "sample_id": sample_id,
                "status": status,
                "source_frame_sha256": frame_sha,
                "regions": [[value / 1000 for value in box]],
                "trace": trace,
                "gt_used": False,
                "certificate_created": False,
                "new_verified_count": 0,
            })
        self.runtime = self._jsonl("runtime.jsonl", self.runtime_rows)
        self.oracle = self._jsonl("oracle.jsonl", self.oracle_rows)
        self.results = self._jsonl("results.jsonl", self.result_rows)

    def tearDown(self):
        self.temporary.cleanup()

    def _jsonl(self, name, rows):
        path = self.root / name
        path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
                        encoding="utf-8")
        return path

    def test_status_stratification_has_no_certificate_metric_or_activity(self):
        report = evaluate_v2(
            runtime_results=self.results,
            runtime_manifest=self.runtime,
            oracle_manifest=self.oracle,
            output_dir=self.root / "out",
        )
        self.assertEqual(report["format"], EVALUATION_V2_FORMAT)
        self.assertEqual(report["cohort_count"], 6)
        self.assertEqual(report["scientific_status_counts"], {status: 1 for status in sorted(STATUSES)})
        self.assertEqual(report["strict_dependency_verified_recall_at_0_5"], 1.0)
        self.assertEqual(report["named_status_metrics"]["strict_dependency_verified"]["cohort_count"], 1)
        self.assertEqual(report["named_status_metrics"]["supported_without_control"]["cohort_count"], 1)
        self.assertTrue(report["dependency_admitted_is_descriptive_not_certificate"])
        self.assertFalse(report["certificate_metrics_computed"])
        self.assertNotIn("certificate_conditioned_recall_at_0_5", report)
        self.assertEqual(report["model_calls_made"], 0)
        self.assertEqual(report["cache_writes"], 0)
        self.assertFalse(report["certificate_created"])
        self.assertEqual(report["new_verified_count"], 0)
        self.assertTrue(report["source_frames_unchanged"])

    def test_visualizations_and_outputs_are_byte_deterministic(self):
        first = self.root / "first"
        second = self.root / "second"
        evaluate_v2(runtime_results=self.results, runtime_manifest=self.runtime,
                    oracle_manifest=self.oracle, output_dir=first)
        evaluate_v2(runtime_results=self.results, runtime_manifest=self.runtime,
                    oracle_manifest=self.oracle, output_dir=second)
        expected = {
            "gt_alignment_rows_v2.jsonl",
            "gt_alignment_summary_v2.json",
            "visualization_manifest_v2.jsonl",
            "contact_sheet_v2.png",
            "report_v2.md",
        }
        self.assertTrue(expected.issubset({path.name for path in first.iterdir()}))
        for relative in expected:
            self.assertEqual((first / relative).read_bytes(), (second / relative).read_bytes())
        for row in self.result_rows:
            relative = Path("overlays") / f"{row['sample_id']}.png"
            self.assertEqual((first / relative).read_bytes(), (second / relative).read_bytes())
        for path, digest in self.source_hashes.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_binding_certificate_and_duplicate_json_fail_closed(self):
        bad = [dict(row) for row in self.result_rows]
        bad[0]["certificate_created"] = True
        bad_path = self._jsonl("bad-results.jsonl", bad)
        with self.assertRaisesRegex(SpatialReliabilityError, "BINDING_INVALID"):
            evaluate_v2(runtime_results=bad_path, runtime_manifest=self.runtime,
                        oracle_manifest=self.oracle, output_dir=self.root / "bad-out")

        duplicate = self.root / "duplicate.jsonl"
        duplicate.write_text('{"sample_id":"x","sample_id":"y"}\n', encoding="utf-8")
        with self.assertRaisesRegex(SpatialReliabilityError, "RESULTS_INVALID"):
            evaluate_v2(runtime_results=duplicate, runtime_manifest=self.runtime,
                        oracle_manifest=self.oracle, output_dir=self.root / "duplicate-out")

    def test_posthoc_v2_does_not_import_runtime_backends_cache_or_certificate(self):
        source = (Path(__file__).parents[1] / "src/relive/spatial_reliability/posthoc_v2.py").read_text()
        self.assertNotIn("relive.backends", source)
        self.assertNotIn("relive.storage.cache", source)
        self.assertNotIn("relive.certificate", source)


if __name__ == "__main__":
    unittest.main()
