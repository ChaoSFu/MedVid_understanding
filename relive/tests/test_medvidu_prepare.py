"""Tests for MedVidU public projection and GT-isolated runtime preparation."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from relive.data.medvidu import (
    PUBLIC_ADAPTER,
    MedVidUPreparationError,
    _project_record,
    load_public_records,
    medvidu_schema_report,
    prepare_medvidu,
)
from relive.data.schemas import load_runtime


class _PoisonTurn(dict):
    def get(self, key, default=None):
        if key == "value" and super().get("from") != "human":
            raise AssertionError("assistant value was accessed")
        return super().get(key, default)


class _PoisonValue:
    def __str__(self):
        raise AssertionError("forbidden source value was accessed")


class MedVidUPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.frame_root = self.root / "valdata"
        (self.frame_root / "NurViD").mkdir(parents=True)
        Image.new("RGB", (7, 5), "green").save(self.frame_root / "NurViD" / "a.png")
        Image.new("RGB", (7, 5), "red").save(self.frame_root / "NurViD" / "b.jpg")
        self.raw = self.root / "source.json"
        rows = [
            self.row("duplicate-id", "/root/data/NurViD/a.png", "/root/data/NurViD/b.jpg"),
            self.row("duplicate-id", "/root/data/NurViD/b.jpg", "/root/data/NurViD/a.png"),
        ]
        self.raw.write_text(json.dumps(rows), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def row(identifier, first, second):
        return {
            "id": identifier, "qa_type": "tal", "dataset_name": "MedVidU-fixture",
            "video": [first, second, second], "sampled_video_frames": [8, 3, 3],
            "conversations": [
                {"from": "human", "value": "<video>\nWhen does suturing happen?"},
                {"from": "gpt", "value": "SECRET_ASSISTANT_ANSWER 12.0-14.0"},
            ],
            "struc_info": {"answer": "SECRET_STRUCTURE"}, "RC_info": {"bbox": [1, 2, 3, 4]},
            "metadata": {"fps": 30, "answer": "SECRET_METADATA"}, "is_RC": True, "train": True,
        }

    def test_projection_never_reads_assistant_or_annotation_values(self):
        row = self.row("id", "/root/data/NurViD/a.png", "/root/data/NurViD/b.jpg")
        row["conversations"][1] = _PoisonTurn(row["conversations"][1])
        row["struc_info"] = _PoisonValue()
        row["RC_info"] = _PoisonValue()
        row["metadata"] = _PoisonValue()
        projected = _project_record(row, 4)
        self.assertEqual(projected.question, "When does suturing happen?")
        self.assertEqual(projected.video_paths[-1], "/root/data/NurViD/b.jpg")
        self.assertEqual(projected.sampled_frame_count, 3)

    def test_schema_report_preserves_native_task_as_unsupported(self):
        records, source_hash = load_public_records(self.raw)
        self.assertEqual(len({record.sample_id for record in records}), 2)
        report = medvidu_schema_report(records, source_hash)
        self.assertEqual(report["qa_types"]["tal"]["runtime_status"], "UNSUPPORTED_NO_EXPLICIT_ADAPTER")
        self.assertIn("Temporal action localization", report["qa_types"]["tal"]["reason"])
        self.assertNotIn("SECRET_ASSISTANT_ANSWER", json.dumps(report))

    def test_report_only_audits_public_paths_without_generating_runtime(self):
        output = self.root / "real" / "preparation"
        result = prepare_medvidu(self.raw, self.frame_root, output, path_audit_scope="all", max_samples=1)
        self.assertEqual(result["status"], "PASS")
        self.assertFalse(result["runtime_generated"])
        self.assertTrue(Path(result["schema_report"]).is_file())
        self.assertTrue(Path(result["public_record_index"]).is_file())
        self.assertTrue(Path(result["gt_isolation_audit"]).is_file())
        body = Path(result["gt_isolation_audit"]).read_text()
        self.assertNotIn("SECRET_ASSISTANT_ANSWER", body)
        self.assertNotIn("SECRET_STRUCTURE", body)
        self.assertNotIn("SECRET_METADATA", body)
        self.assertNotIn("SECRET_ASSISTANT_ANSWER", Path(result["public_record_index"]).read_text())
        self.assertFalse(list(output.glob("*.runtime.jsonl")))

    def test_explicit_public_claim_manifest_creates_closed_runtime_without_time_or_gt(self):
        records, _ = load_public_records(self.raw)
        claim_manifest = self.root / "public_claims.jsonl"
        claim_manifest.write_text(json.dumps({
            "source_record_index": records[0].source_record_index,
            "public_record_sha256": records[0].public_record_sha256,
            "target_claim": {"claim_id": "public-claim-1", "text": "A suturing action is visible in the supplied frames."},
        }) + "\n", encoding="utf-8")
        output = self.root / "real" / "runtime"
        result = prepare_medvidu(self.raw, self.frame_root, output, adapter=PUBLIC_ADAPTER,
                                 max_samples=1, public_claim_manifest=claim_manifest)
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["runtime_generated"])
        sample = load_runtime(result["runtime"])[0]
        self.assertEqual(sample.task, "claim_verification")
        self.assertEqual([frame.order for frame in sample.frames], [0, 1, 2])
        self.assertEqual(sample.frames[1].path, sample.frames[2].path)
        self.assertTrue(all(frame.timestamp is None for frame in sample.frames))
        self.assertEqual(sample.metadata["runtime_adapter"], PUBLIC_ADAPTER)
        combined = "".join(Path(path).read_text() for path in (result["runtime"], result["provenance_sidecar"], result["gt_isolation_audit"]))
        for forbidden in ("SECRET_ASSISTANT_ANSWER", "SECRET_STRUCTURE", "SECRET_METADATA"):
            self.assertNotIn(forbidden, combined)
        self.assertNotIn("bbox", Path(result["runtime"]).read_text())
        self.assertIn("custom_public_claim_protocol_smoke_not_official_medvidu_qa", combined)

    def test_no_native_action_qa_substitution_or_forbidden_manifest_field(self):
        with self.assertRaisesRegex(MedVidUPreparationError, "UNSUPPORTED_NO_EXPLICIT_ADAPTER"):
            prepare_medvidu(self.raw, self.frame_root, self.root / "real" / "bad", adapter="action_qa")
        records, _ = load_public_records(self.raw)
        claim_manifest = self.root / "bad_claims.jsonl"
        claim_manifest.write_text(json.dumps({
            "source_record_index": 0, "public_record_sha256": records[0].public_record_sha256,
            "target_claim": {"claim_id": "c", "text": "Visible action.", "time_scope": {"start": 1}},
        }) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(MedVidUPreparationError, "forbidden"):
            prepare_medvidu(self.raw, self.frame_root, self.root / "real" / "bad-manifest", adapter=PUBLIC_ADAPTER,
                             public_claim_manifest=claim_manifest, max_samples=1)

    def test_missing_or_escape_frame_aborts_before_runtime_write(self):
        records, _ = load_public_records(self.raw)
        claim_manifest = self.root / "claims.jsonl"
        claim_manifest.write_text(json.dumps({
            "source_record_index": 0, "public_record_sha256": records[0].public_record_sha256,
            "target_claim": {"claim_id": "c", "text": "Visible action."},
        }) + "\n", encoding="utf-8")
        (self.frame_root / "NurViD" / "b.jpg").unlink()
        output = self.root / "real" / "missing"
        with self.assertRaisesRegex(MedVidUPreparationError, "FRAME_MAPPING_AUDIT_FAILED"):
            prepare_medvidu(self.raw, self.frame_root, output, adapter=PUBLIC_ADAPTER,
                             public_claim_manifest=claim_manifest, max_samples=1)
        self.assertFalse((output / "medvidu_user_claim_verification.runtime.jsonl").exists())
        escaped = self.row("id", "/root/data/../outside.png", "/root/data/NurViD/a.png")
        with self.assertRaisesRegex(MedVidUPreparationError, "FRAME_MAPPING_AUDIT_FAILED"):
            escaped_path = self.root / "escaped.json"
            escaped_path.write_text(json.dumps([escaped]), encoding="utf-8")
            escaped_records, _ = load_public_records(escaped_path)
            escape_manifest = self.root / "escape_claims.jsonl"
            escape_manifest.write_text(json.dumps({
                "source_record_index": 0, "public_record_sha256": escaped_records[0].public_record_sha256,
                "target_claim": {"claim_id": "c", "text": "Visible action."},
            }) + "\n", encoding="utf-8")
            prepare_medvidu(escaped_path, self.frame_root, self.root / "real" / "escape", adapter=PUBLIC_ADAPTER,
                             public_claim_manifest=escape_manifest, max_samples=1)


if __name__ == "__main__":
    unittest.main()
