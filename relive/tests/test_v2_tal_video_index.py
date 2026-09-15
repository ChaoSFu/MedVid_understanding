from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from relive.storage.artifacts import canonical_json
from relive.v2.requirement_freeze import freeze_tal_requirements
from relive.v2.task_selection import freeze_selector_order, write_selection
from relive.v2.temporal_localization import load_event_ontology
from relive.v2.video_index import (FrameRootMapper, VideoIndexError, _source_identity, freeze_video_index,
                                    validate_video_index_artifacts)


class _PoisonTurn(dict):
    def get(self, key, default=None):
        if key == "value":
            raise AssertionError("assistant value was accessed")
        return super().get(key, default)


class VideoIndexTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.frame_root = self.root / "frames"
        self.frame_root.mkdir()
        self.source_prefix = "/root/data"
        self.question = "When does Secure the base happen?"
        self.source_row = {"id": "case", "qa_type": "tal", "conversations": [{"from": "human", "value": "<video>" + self.question}, {"from": "gpt", "value": "FORBIDDEN"}],
                           "video": ["/root/data/a.jpg", "/root/data/b.jpg", "/root/data/b.jpg"], "sampled_video_frames": [9, 10, 11], "dataset_name": "public"}
        for name, color in (("a.jpg", (10, 20, 30)), ("b.jpg", (30, 20, 10))):
            Image.new("RGB", (8, 6), color).save(self.frame_root / name)
        identity, _, _ = _source_identity(self.source_row, 0)
        self.identity = identity
        self.selector = self.root / "selector.jsonl"
        selector_row = {"source_record_index": 0, "sample_id": identity["sample_id"], "public_record_sha256": identity["public_record_sha256"],
                        "question_sha256": identity["question_sha256"], "qa_type": "tal", "question": self.question}
        self.selector.write_text(canonical_json(selector_row) + "\n")
        self.selection = freeze_selector_order(selector_path=self.selector, max_samples=1)
        self.selection_path = self.root / "selection.json"
        write_selection(self.selection, self.selection_path)
        ontology = load_event_ontology(Path(__file__).resolve().parents[1] / "configs/v2/tal_event_ontology.yaml")
        self.requirements = self.root / "requirements"
        freeze_tal_requirements(selection=self.selection, selector_sha256=hashlib.sha256(self.selector.read_bytes()).hexdigest(),
                                selection_manifest_sha256=hashlib.sha256(self.selection_path.read_bytes()).hexdigest(),
                                ontology=ontology, output_dir=self.requirements)
        self.source_json = self.root / "source.json"
        self.source_json.write_text(canonical_json([self.source_row]) + "\n")
        self.policy = Path(__file__).resolve().parents[1] / "configs/v2/tal_timebase_sources.yaml"

    def tearDown(self):
        self.temp.cleanup()

    def timestamps(self, values=(0.0, 0.5, 1.0)):
        path = self.root / "timestamps.jsonl"
        rows = []
        for order, (reference, timestamp) in enumerate(zip(self.source_row["video"], values)):
            rows.append({"source_record_index": 0, "sample_id": self.identity["sample_id"], "public_record_sha256": self.identity["public_record_sha256"],
                         "question_sha256": self.identity["question_sha256"], "frame_order": order,
                         "source_frame_reference": reference, "timestamp_seconds": timestamp, "timestamp_unit": "seconds"})
        if any(isinstance(value, float) and value != value for value in values):
            path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=True) + "\n" for row in rows))
        else:
            path.write_text("".join(canonical_json(row) + "\n" for row in rows))
        return path

    def provenance(self, timestamps):
        source = self.root / "public-time-source.json"
        source.write_text(canonical_json({"public": "documented fixture"}) + "\n")
        path = self.root / "timestamps.provenance.json"
        path.write_text(canonical_json({"format": "relive-v2-tal-timestamp-provenance-v1", "timestamp_manifest_sha256": hashlib.sha256(timestamps.read_bytes()).hexdigest(), "source_type": "PUBLIC_PER_FRAME_TIMESTAMP", "dataset_name": "public", "sample_id": self.identity["sample_id"], "source_record_index": 0, "public_record_sha256": self.identity["public_record_sha256"], "question_sha256": self.identity["question_sha256"], "time_origin": "CLIP_LOCAL_ZERO", "timestamp_unit": "seconds", "mapping_method": "fixture documented per-frame timestamps", "adapter_version": "fixture-v1", "source_reference": str(source), "source_reference_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "frame_count": 3, "gt_used": False, "assistant_or_gt_values_accessed": False, "model_calls_made": 0, "backend_loaded": False, "cache_opened": False, "certificate_created": False, "new_verified_count": 0, "certificate_status": "NOT_APPLICABLE", "public_media_projection_sha256": __import__("relive.storage.artifacts", fromlist=["stable_hash"]).stable_hash({"sample_id": self.identity["sample_id"], "source_record_index": 0, "public_record_sha256": self.identity["public_record_sha256"], "question_sha256": self.identity["question_sha256"], "frame_count": 3, "logical_frame_order_preserved": True, "frames": [{"frame_order": order, "source_frame_reference": reference, "source_frame_path": reference, "resolved_frame_path": str((self.frame_root / Path(reference).name).resolve()), "frame_sha256": hashlib.sha256((self.frame_root / Path(reference).name).read_bytes()).hexdigest(), "width": 8, "height": 6, "image_format": "JPEG", "timestamp_seconds": None, "timestamp_provenance": None} for order, reference in enumerate(self.source_row["video"])]}), "frame_sha256": [hashlib.sha256((self.frame_root / Path(reference).name).read_bytes()).hexdigest() for reference in self.source_row["video"]]}) + "\n")
        return path

    def freeze(self, destination, timestamps=None, provenance=None):
        return freeze_video_index(requirement_dir=self.requirements, selection_manifest_path=self.selection_path,
                                  source_json=self.source_json, frame_root=self.frame_root, source_prefix=self.source_prefix,
                                  timebase_policy_path=self.policy, output_dir=destination,
                                  public_timestamp_manifest_path=timestamps, public_timestamp_provenance_path=provenance)

    def test_resolved_timebase_keeps_duplicate_logical_frames_and_revalidates_bytes(self):
        out = self.root / "resolved"
        timestamps = self.timestamps()
        audit = self.freeze(out, timestamps, self.provenance(timestamps))
        self.assertEqual(audit["index_status"], "RESOLVED")
        index = json.loads((out / "v2_video_index.jsonl").read_text())
        self.assertEqual(len(index["frames"]), 3)
        self.assertEqual(index["frames"][1]["source_frame_reference"], index["frames"][2]["source_frame_reference"])
        self.assertEqual(validate_video_index_artifacts(out, materialize_frames=True)["video_index_count"], 1)
        duplicate = self.root / "duplicate-timestamps"
        timestamps = self.timestamps((0.0, 0.5, 0.5))
        self.freeze(duplicate, timestamps, self.provenance(timestamps))
        self.assertEqual(json.loads((duplicate / "v2_video_index.jsonl").read_text())["duplicate_timestamp_frame_orders"], [2])
        Image.new("RGB", (8, 6), (255, 0, 0)).save(self.frame_root / "a.jpg")
        with self.assertRaisesRegex(VideoIndexError, "FRAME_BYTES_CHANGED"):
            validate_video_index_artifacts(out, materialize_frames=True)

    def test_missing_validated_timebase_is_auditable_unresolved_stop(self):
        out = self.root / "unresolved"
        audit = self.freeze(out)
        self.assertEqual(audit["index_status"], "UNRESOLVED_TIMEBASE")
        self.assertEqual((out / "v2_video_index.jsonl").read_bytes(), b"")
        result = validate_video_index_artifacts(out)
        self.assertEqual(result["index_status"], "UNRESOLVED_TIMEBASE")
        self.assertFalse(result["ready_for_hypothesis_generation"])

    def test_bad_timebase_is_unresolved_not_inferred(self):
        for values in ((0.0, float("nan"), 1.0), (1.0, 0.5, 2.0), (0.0, 0.0, 0.0)):
            with self.subTest(values=values):
                out = self.root / ("bad" + str(len(list(self.root.glob("bad*")))))
                timestamp = self.timestamps(values)
                self.freeze(out, timestamp, self.provenance(timestamp))
                self.assertEqual(validate_video_index_artifacts(out)["index_status"], "UNRESOLVED_TIMEBASE")

    def test_binding_and_artifact_tamper_fail_closed(self):
        out = self.root / "valid"
        timestamps = self.timestamps()
        self.freeze(out, timestamps, self.provenance(timestamps))
        (out / "v2_video_index.jsonl").write_bytes(b"tampered")
        with self.assertRaisesRegex(VideoIndexError, "ARTIFACT_BYTES_CHANGED|VIDEO_INDEX_INVALID"):
            validate_video_index_artifacts(out)
        bad_selection = self.root / "bad-selection.json"
        bad_selection.write_bytes(self.selection_path.read_bytes() + b" ")
        with self.assertRaisesRegex(VideoIndexError, "SELECTION_MANIFEST"):
            self.freeze(self.root / "bad-selection-output", self.timestamps()) if False else freeze_video_index(
                requirement_dir=self.requirements, selection_manifest_path=bad_selection, source_json=self.source_json,
                frame_root=self.frame_root, source_prefix=self.source_prefix, timebase_policy_path=self.policy,
                output_dir=self.root / "bad-selection-output", public_timestamp_manifest_path=self.timestamps())

    def test_source_identity_path_escape_and_poison_values_fail_closed(self):
        poisoned = dict(self.source_row)
        poisoned["conversations"] = [{"from": "human", "value": "<video>" + self.question}, _PoisonTurn(from_="ignored")]
        poisoned["conversations"][1]["from"] = "gpt"
        _source_identity(poisoned, 0)  # must not request non-human value
        escaped = dict(self.source_row)
        escaped["video"] = ["/root/data/../secret.jpg"] * 3
        source = self.root / "escaped.json"
        source.write_text(canonical_json([escaped]) + "\n")
        with self.assertRaisesRegex(VideoIndexError, "PUBLIC_RECORD_IDENTITY_MISMATCH|PATH"):
            freeze_video_index(requirement_dir=self.requirements, selection_manifest_path=self.selection_path, source_json=source,
                               frame_root=self.frame_root, source_prefix=self.source_prefix, timebase_policy_path=self.policy,
                               output_dir=self.root / "escaped-out")
        outside = self.root / "outside.jpg"; Image.new("RGB", (2, 2)).save(outside)
        os.symlink(outside, self.frame_root / "escape.jpg")
        with self.assertRaisesRegex(VideoIndexError, "ESCAPE"):
            FrameRootMapper.create(self.source_prefix, self.frame_root).materialize("/root/data/escape.jpg")
        symlinked = dict(self.source_row); symlinked["video"] = ["/root/data/escape.jpg"] * 3
        identity, _, _ = _source_identity(symlinked, 0)
        selector = self.root / "symlink-selector.jsonl"
        selector.write_text(canonical_json({"source_record_index":0,"sample_id":identity["sample_id"],"public_record_sha256":identity["public_record_sha256"],"question_sha256":identity["question_sha256"],"qa_type":"tal","question":self.question})+"\n")
        # The old requirements bind different public identity, so selector-source mismatch fails before any accepted index.
        source = self.root / "symlink-source.json";source.write_text(canonical_json([symlinked])+"\n")
        with self.assertRaises(VideoIndexError):
            freeze_video_index(requirement_dir=self.requirements, selection_manifest_path=self.selection_path, source_json=source,
                               frame_root=self.frame_root, source_prefix=self.source_prefix, timebase_policy_path=self.policy,
                               output_dir=self.root / "symlink-out")

    def test_two_output_directories_are_byte_identical(self):
        one, two = self.root / "one", self.root / "two"
        timestamps = self.timestamps()
        self.freeze(one, timestamps, self.provenance(timestamps))
        self.freeze(two, timestamps, self.provenance(timestamps))
        for name in ("v2_public_media_projection.jsonl", "v2_video_index.jsonl", "v2_video_index_unresolved.jsonl", "v2_video_index_manifest.json", "v2_video_index_audit.json"):
            self.assertEqual((one / name).read_bytes(), (two / name).read_bytes())


if __name__ == "__main__":
    unittest.main()

class VideoIndexCliTests(unittest.TestCase):
    setUp = VideoIndexTests.setUp
    tearDown = VideoIndexTests.tearDown
    freeze = VideoIndexTests.freeze
    timestamps = VideoIndexTests.timestamps
    provenance = VideoIndexTests.provenance

    def test_read_only_validator_cli(self):
        import subprocess
        import sys
        out = self.root / "cli"
        self.freeze(out)
        script = Path(__file__).resolve().parents[1] / "scripts/validate_v2_tal_video_index.py"
        result = subprocess.run([sys.executable, str(script), "--output-dir", str(out)], cwd=script.parents[1],
                                env={**os.environ, "PYTHONPATH": str(script.parents[1] / "src")}, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["index_status"], "UNRESOLVED_TIMEBASE")
        self.assertEqual(payload["new_verified_count"], 0)
