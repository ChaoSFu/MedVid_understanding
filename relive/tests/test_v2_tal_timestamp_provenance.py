from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

from relive.storage.artifacts import canonical_json
from relive.v2.timestamp_provenance import (
    DECODER_MAP_FORMAT, DOCUMENTED_SOURCE_FORMAT, TimestampProvenanceError,
    export_public_timestamps, validate_export, validate_timestamp_pair,
)


class TimestampProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.identity = {"source_record_index": 7, "sample_id": "medvidu:public:abc", "public_record_sha256": "a" * 64, "question_sha256": "b" * 64}
        self.refs = ["/root/data/a.jpg", "/root/data/b.jpg", "/root/data/c.jpg"]
        self.projection = self.root / "projection.jsonl"
        row = {**self.identity, "frame_count": 3, "logical_frame_order_preserved": True,
               "frames": [{"frame_order": i, "source_frame_reference": ref, "source_frame_path": ref, "resolved_frame_path": "/tmp/x", "frame_sha256": "c" * 64, "width": 4, "height": 3, "image_format": "JPEG", "timestamp_seconds": None, "timestamp_provenance": None} for i, ref in enumerate(self.refs)]}
        from relive.storage.artifacts import stable_hash
        row["public_media_projection_sha256"] = stable_hash(row)
        self.projection.write_text(canonical_json(row) + "\n")

    def tearDown(self): self.temp.cleanup()

    def documented(self, *, indices=(100, 102, 104), fps=2.0, origin="CLIP_LOCAL_ZERO"):
        path = self.root / "documented.json"
        source = {"format": DOCUMENTED_SOURCE_FORMAT, "adapter_version": "documented-fixture-v1", "dataset_name": "NurViD", "timestamp_unit": "seconds", "time_origin": origin, "mapping_method": "(source_frame_index - first_source_frame_index) / documented_fps", "records": [{**self.identity, "source_frame_references": self.refs, "frame_indices": list(indices), "fps": fps}]}
        path.write_text(canonical_json(source) + "\n"); return path

    def test_documented_frame_index_fps_export_is_deterministic_and_clip_local(self):
        source = self.documented(); one, two = self.root / "one", self.root / "two"
        result = export_public_timestamps(media_projection_path=self.projection, output_dir=one, documented_source_path=source)
        self.assertEqual(result["status"], "PASS"); self.assertEqual(validate_export(one)["frame_count"], 3)
        export_public_timestamps(media_projection_path=self.projection, output_dir=two, documented_source_path=source)
        for name in ("public_per_frame_timestamps.jsonl", "public_per_frame_timestamps.provenance.json", "v2_tal_timestamp_source_audit.json"):
            self.assertEqual((one / name).read_bytes(), (two / name).read_bytes())
        self.assertIn('"timestamp_seconds":1.0', (one / "public_per_frame_timestamps.jsonl").read_text())

    def test_decoder_pts_route_requires_unique_predeclared_mapping(self):
        video = self.root / "video.mp4"; video.write_bytes(b"public video fixture")
        mapping = self.root / "map.json"
        mapping.write_text(canonical_json({"format": DECODER_MAP_FORMAT, "time_origin": "CLIP_LOCAL_ZERO", "records": [{**self.identity, "source_frame_references": self.refs, "decoder_frame_indices": [1, 3, 5]}]}) + "\n")
        result = export_public_timestamps(media_projection_path=self.projection, output_dir=self.root / "pts", public_video_path=video, decoder_frame_map_path=mapping, pts_reader=lambda _: [0.0, .4, .8, 1.2, 1.6, 2.0])
        self.assertEqual(result["source_type"], "DECODER_PTS")
        bad = self.root / "bad-map.json"
        bad.write_text(canonical_json({"format": DECODER_MAP_FORMAT, "time_origin": "CLIP_LOCAL_ZERO", "records": [{**self.identity, "source_frame_references": self.refs, "decoder_frame_indices": [1, 1, 5]}]}) + "\n")
        with self.assertRaisesRegex(TimestampProvenanceError, "NOT_UNIQUE"):
            export_public_timestamps(media_projection_path=self.projection, output_dir=self.root / "badpts", public_video_path=video, decoder_frame_map_path=bad, pts_reader=lambda _: [0., .4, .8, 1.2, 1.6, 2.])

    def test_unresolved_and_sample_id_only_are_rejected(self):
        output = self.root / "unresolved"
        result = export_public_timestamps(media_projection_path=self.projection, output_dir=output)
        self.assertEqual(result["status"], "UNRESOLVED_TIMEBASE_SOURCE")
        self.assertFalse((output / "public_per_frame_timestamps.jsonl").exists())
        source = self.documented(indices=(0, 1, 2))
        raw = source.read_text().replace('"frame_indices"', '"sample_id_frame_indices"').replace('"source_frame_references"', '"sample_id_only_references"')
        source.write_text(raw)
        with self.assertRaises(TimestampProvenanceError):
            export_public_timestamps(media_projection_path=self.projection, output_dir=self.root / "id", documented_source_path=source)

    def test_sidecar_tamper_and_invalid_values_fail_closed(self):
        source = self.documented(); out = self.root / "out"
        export_public_timestamps(media_projection_path=self.projection, output_dir=out, documented_source_path=source)
        sidecar = out / "public_per_frame_timestamps.provenance.json"
        sidecar.write_bytes(sidecar.read_bytes().replace(b'"time_origin":"CLIP_LOCAL_ZERO"', b'"time_origin":"UNKNOWN"'))
        with self.assertRaises(TimestampProvenanceError): validate_export(out)
        # A timestamp JSONL alone can never establish resolved provenance.
        rows = out / "public_per_frame_timestamps.jsonl"
        identity = self.identity; frames = [{"source_frame_reference": ref} for ref in self.refs]
        with self.assertRaises(TimestampProvenanceError):
            validate_timestamp_pair(timestamp_manifest_path=rows, provenance_path=sidecar, identity=identity, frames=frames)

    def test_invalid_documented_timebase_is_rejected_before_export(self):
        for indices, fps in (((100, 99, 102), 2.0), ((100, 102, 104), 0.0)):
            with self.subTest(indices=indices, fps=fps):
                with self.assertRaises(TimestampProvenanceError):
                    export_public_timestamps(media_projection_path=self.projection, output_dir=self.root / ("x" + str(len(list(self.root.glob("x*"))))), documented_source_path=self.documented(indices=indices, fps=fps))


if __name__ == "__main__": unittest.main()

class TimestampTamperTests(unittest.TestCase):
    def test_manifest_and_source_reference_hash_tamper_fail_closed(self):
        fixture = TimestampProvenanceTests(); fixture.setUp()
        try:
            source = fixture.documented(); out = fixture.root / "out"
            export_public_timestamps(media_projection_path=fixture.projection, output_dir=out, documented_source_path=source)
            manifest = out / "public_per_frame_timestamps.jsonl"
            manifest.write_bytes(manifest.read_bytes().replace(b'"timestamp_seconds":1.0', b'"timestamp_seconds":1.5'))
            with self.assertRaises(TimestampProvenanceError): validate_export(out)
            # Re-export then alter the bound public source bytes.
            out2 = fixture.root / "out2"; export_public_timestamps(media_projection_path=fixture.projection, output_dir=out2, documented_source_path=source)
            source.write_text(source.read_text() + " ")
            with self.assertRaises(TimestampProvenanceError): validate_export(out2)
        finally:
            fixture.tearDown()
