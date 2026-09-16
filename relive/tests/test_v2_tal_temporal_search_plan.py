from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from relive.storage.artifacts import canonical_json, stable_hash
from relive.v2.requirement_freeze import freeze_tal_requirements
from relive.v2.task_selection import freeze_selector_order, write_selection
from relive.v2.temporal_localization import load_event_ontology
from relive.v2.temporal_search_plan import (TemporalSearchPlanError, freeze_temporal_search_plan,
                                             validate_hypothesis_claim, validate_observation_claim,
                                             validate_temporal_search_plan_artifacts, _windows)
from relive.v2.video_index import _source_identity, freeze_video_index


class TemporalSearchPlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.frames = self.root / "frames"; self.frames.mkdir()
        self.question = "When does Secure the base happen?"
        self.row = {"id": "case", "qa_type": "tal", "conversations": [{"from": "human", "value": "<video>" + self.question}, {"from": "gpt", "value": "forbidden"}],
                    "video": ["/root/data/a.jpg", "/root/data/b.jpg", "/root/data/b.jpg", "/root/data/c.jpg"],
                    "sampled_video_frames": [40, 41, 41, 55], "dataset_name": "NurViD"}
        for name, colour in (("a.jpg", (1, 2, 3)), ("b.jpg", (4, 5, 6)), ("c.jpg", (7, 8, 9))):
            Image.new("RGB", (8, 6), colour).save(self.frames / name)
        self.identity, _, _ = _source_identity(self.row, 0)
        self.selector = self.root / "selector.jsonl"
        self.selector.write_text(canonical_json({"source_record_index": 0, "sample_id": self.identity["sample_id"], "public_record_sha256": self.identity["public_record_sha256"], "question_sha256": self.identity["question_sha256"], "qa_type": "tal", "question": self.question}) + "\n")
        selection = freeze_selector_order(selector_path=self.selector, max_samples=1)
        self.selection = self.root / "selection.json"; write_selection(selection, self.selection)
        ontology = load_event_ontology(Path(__file__).resolve().parents[1] / "configs/v2/tal_event_ontology.yaml")
        self.requirements = self.root / "requirements"
        freeze_tal_requirements(selection=selection, selector_sha256=hashlib.sha256(self.selector.read_bytes()).hexdigest(), selection_manifest_sha256=hashlib.sha256(self.selection.read_bytes()).hexdigest(), ontology=ontology, output_dir=self.requirements)
        self.source = self.root / "source.json"; self.source.write_text(canonical_json([self.row]) + "\n")
        self.timestamp = self.root / "timestamps.jsonl"
        values = [0.0, 0.5, 0.5, 7.5]
        rows = [{"source_record_index": 0, "sample_id": self.identity["sample_id"], "public_record_sha256": self.identity["public_record_sha256"], "question_sha256": self.identity["question_sha256"], "frame_order": order, "source_frame_reference": reference, "timestamp_seconds": seconds, "timestamp_unit": "seconds"} for order, (reference, seconds) in enumerate(zip(self.row["video"], values))]
        self.timestamp.write_bytes(b"".join((canonical_json(item) + "\n").encode() for item in rows))
        self.projection_sha = stable_hash({"sample_id": self.identity["sample_id"], "source_record_index": 0, "public_record_sha256": self.identity["public_record_sha256"], "question_sha256": self.identity["question_sha256"], "frame_count": 4, "logical_frame_order_preserved": True, "frames": [{"frame_order": order, "source_frame_reference": reference, "source_frame_path": reference, "resolved_frame_path": str((self.frames / Path(reference).name).resolve()), "frame_sha256": hashlib.sha256((self.frames / Path(reference).name).read_bytes()).hexdigest(), "width": 8, "height": 6, "image_format": "JPEG", "timestamp_seconds": None, "timestamp_provenance": None} for order, reference in enumerate(self.row["video"])]})
        self.source_reference = self.root / "source-reference.json"; self.source_reference.write_text(canonical_json({"safe": "fixture"}) + "\n")
        self.provenance = self.root / "timestamps.provenance.json"
        self.provenance.write_text(canonical_json({"format": "relive-v2-tal-timestamp-provenance-v1", "timestamp_manifest_sha256": hashlib.sha256(self.timestamp.read_bytes()).hexdigest(), "source_type": "VERSIONED_DATASET_TIMEBASE_ADAPTER", "dataset_name": "NurViD", "sample_id": self.identity["sample_id"], "source_record_index": 0, "public_record_sha256": self.identity["public_record_sha256"], "question_sha256": self.identity["question_sha256"], "time_origin": "CLIP_LOCAL_ZERO", "timestamp_origin": "FIRST_PRESENTED_FRAME", "timestamp_unit": "seconds", "mapping_method": "fixture", "adapter_version": "fixture-clip-local-v1", "source_reference": str(self.source_reference), "source_reference_sha256": hashlib.sha256(self.source_reference.read_bytes()).hexdigest(), "frame_count": 4, "gt_used": False, "assistant_or_gt_values_accessed": False, "model_calls_made": 0, "backend_loaded": False, "cache_opened": False, "certificate_created": False, "new_verified_count": 0, "certificate_status": "NOT_APPLICABLE", "public_media_projection_sha256": self.projection_sha, "frame_sha256": [hashlib.sha256((self.frames / Path(reference).name).read_bytes()).hexdigest() for reference in self.row["video"]]}) + "\n")
        self.index = self.root / "index"
        freeze_video_index(requirement_dir=self.requirements, selection_manifest_path=self.selection, source_json=self.source, frame_root=self.frames, source_prefix="/root/data", timebase_policy_path=Path(__file__).resolve().parents[1] / "configs/v2/tal_timebase_sources.yaml", output_dir=self.index, public_timestamp_manifest_path=self.timestamp, public_timestamp_provenance_path=self.provenance)
        self.policy = Path(__file__).resolve().parents[1] / "configs/v2/tal_temporal_pyramid_policy.yaml"

    def tearDown(self):
        self.temp.cleanup()

    def freeze(self, out: Path):
        return freeze_temporal_search_plan(requirement_dir=self.requirements, selection_manifest_path=self.selection, video_index_dir=self.index, timestamp_manifest_path=self.timestamp, timestamp_provenance_path=self.provenance, policy_path=self.policy, output_dir=out)

    def test_schema_contracts_reject_unverified_bypass_and_parent_rewrite(self):
        valid = {"hypothesis_id": "hypothesis-x", "requirement_id": "requirement-x", "parent_claim_id": None, "claim_role": "TARGET_HYPOTHESIS", "target_event": "secure_the_base", "polarity": "POSITIVE", "candidate_interval": {"start_seconds": 0.0, "end_seconds": 1.0, "timestamp_domain": "CLIP_LOCAL"}, "generation_source": "CLAIM_CONDITIONED_COARSE_RETRIEVAL", "status": "CANDIDATE_UNVERIFIED", "supporting_window_ids": [], "provenance": {}}
        validate_hypothesis_claim(valid)
        for bad in ({key: value for key, value in valid.items() if key != "provenance"}, {**valid, "status": "VERIFIED"}, {**valid, "gt_overlap": 1.0}):
            with self.assertRaises(TemporalSearchPlanError): validate_hypothesis_claim(bad)
        child = {"claim_id": "child", "parent_claim_id": "parent", "claim_role": "OBSERVATION", "subject": "tool", "predicate": "touches", "object": "tissue", "polarity": "POSITIVE", "temporal_quantifier": "AT_FRAME", "frame_scope": "WINDOW", "observability": "DIRECTLY_VISIBLE", "evidence_geometry": "SINGLE_REGION", "required_components": [], "generation_reason": "OBSERVATION_DECOMPOSITION", "provenance": {}, "status": "CANDIDATE_UNVERIFIED"}
        validate_observation_claim(child, immutable_parent_ids=frozenset({"parent"}))
        with self.assertRaises(TemporalSearchPlanError): validate_observation_claim({**child, "claim_id": "parent"}, immutable_parent_ids=frozenset({"parent"}))

    def test_plan_is_deterministic_complete_and_tamper_evident(self):
        first, second = self.root / "one", self.root / "two"
        result = self.freeze(first)
        self.freeze(second)
        self.assertEqual(result["logical_frame_count"], 4)
        self.assertEqual(result["unique_visual_frame_count"], 3)
        self.assertEqual(result["duplicate_logical_frame_count"], 1)
        self.assertEqual(result["logical_frame_coverage"], 1.0)
        self.assertEqual(result["unique_visual_frame_coverage"], 1.0)
        plan = [json.loads(line) for line in (first / "v2_temporal_search_plan.jsonl").read_text().splitlines()]
        self.assertEqual([row["scale_seconds"] for row in plan if row["window_role"] == "PYRAMID"], [8, 16, 32])
        self.assertEqual(plan[-1]["window_role"], "FULL_CLIP")
        for name in ("v2_claim_graph_schema.json", "v2_hypothesis_contract.json", "v2_temporal_pyramid_policy.json", "v2_temporal_search_plan.jsonl", "v2_temporal_search_manifest.json", "v2_temporal_search_freeze_audit.json"):
            self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())
        self.assertTrue(validate_temporal_search_plan_artifacts(first)["ready_for_coarse_hypothesis_generation"])
        (first / "v2_temporal_search_plan.jsonl").write_bytes(b"tampered")
        with self.assertRaisesRegex(TemporalSearchPlanError, "BYTES_CHANGED|INVALID"):
            validate_temporal_search_plan_artifacts(first)

    def test_policy_and_video_index_binding_fail_closed(self):
        bad = self.root / "bad-policy.yaml"; bad.write_bytes(self.policy.read_bytes().replace(b"[8,16,32]", b"[8,16]"))
        with self.assertRaises(TemporalSearchPlanError):
            freeze_temporal_search_plan(requirement_dir=self.requirements, selection_manifest_path=self.selection, video_index_dir=self.index, timestamp_manifest_path=self.timestamp, timestamp_provenance_path=self.provenance, policy_path=bad, output_dir=self.root / "bad")
        (self.index / "v2_video_index_manifest.json").write_bytes(b"{}\n")
        with self.assertRaises(TemporalSearchPlanError): self.freeze(self.root / "bad-index")

    def test_tail_window_and_every_frozen_output_tamper_fail_closed(self):
        frames = [{"frame_order": order, "frame_sha256": f"{order:064x}", "source_frame_reference": str(order), "timestamp_seconds": order / 2.0} for order in range(112)]
        policy = json.loads(self.policy.read_text())
        windows = _windows(requirement_id="requirement-x", frames=frames, video_index_manifest_sha256="a" * 64, policy_sha256="b" * 64, policy=policy)
        eight = [row for row in windows if row["window_role"] == "PYRAMID" and row["scale_seconds"] == 8]
        self.assertEqual(eight[-1]["start_seconds"], 47.5)
        self.assertEqual(eight[-1]["end_seconds"], 55.5)
        for name in ("v2_claim_graph_schema.json", "v2_hypothesis_contract.json", "v2_temporal_pyramid_policy.json", "v2_temporal_search_plan.jsonl", "v2_temporal_search_manifest.json", "v2_temporal_search_freeze_audit.json"):
            with self.subTest(name=name):
                out = self.root / ("tamper-" + name.replace(".", "_"))
                self.freeze(out)
                (out / name).write_bytes(b"tampered")
                with self.assertRaises(TemporalSearchPlanError):
                    validate_temporal_search_plan_artifacts(out)


if __name__ == "__main__":
    unittest.main()
