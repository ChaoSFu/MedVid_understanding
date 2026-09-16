from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from relive.storage.artifacts import canonical_json
from relive.v2.coarse_hypothesis import (CHOICES, CoarseHypothesisError, _packet_indices, execute, prepare,
                                         preflight, support_margin, temporal_iou, validate)
from relive.v2.requirement_freeze import freeze_tal_requirements
from relive.v2.task_selection import freeze_selector_order, write_selection
from relive.v2.temporal_localization import load_event_ontology
from relive.v2.temporal_search_plan import freeze_temporal_search_plan
from relive.v2.video_index import _source_identity, freeze_video_index


class _FakeChoiceBackend:
    synthetic = True
    def fingerprint(self): return {"adapter_version": "fake-stage3b-v1", "model": "fake"}
    def forced_choice_token_contract(self, request, choices=CHOICES):
        return {"choice_labels": list(choices), "choice_token_ids": {label: index + 1 for index, label in enumerate(choices)}, "context_input_ids_sha256": hashlib.sha256(request["prompt"].encode()).hexdigest(), "context_token_count": 9}
    def forced_choice_likelihood(self, request, *, choice_token_ids=None, choices=CHOICES):
        contract = self.forced_choice_token_contract(request, choices=choices)
        assert choice_token_ids == contract["choice_token_ids"]
        base = int(hashlib.sha256(request["window_id"].encode()).hexdigest()[:4], 16) / 65535
        return {"choice_contract": contract, "choices": {label: {"token_id": contract["choice_token_ids"][label], "raw_logit": base - index, "log_probability": base - index} for index, label in enumerate(choices)}}


class CoarseHypothesisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name); self.frames = self.root / "frames"; self.frames.mkdir()
        refs = [f"/root/data/f{order:02d}.jpg" for order in range(56)]
        refs[7], refs[21] = refs[6], refs[20]
        for order in sorted({Path(item).name for item in refs}): Image.new("RGB", (8, 6), (len(order), 2, 3)).save(self.frames / order)
        self.question = "When does Secure the base happen?"
        self.row = {"id": "case", "qa_type": "tal", "conversations": [{"from": "human", "value": "<video>" + self.question}, {"from": "gpt", "value": "poison"}], "video": refs, "sampled_video_frames": list(range(40, 96)), "dataset_name": "NurViD"}
        self.identity, _, _ = _source_identity(self.row, 0)
        self.selector = self.root / "selector.jsonl"; self.selector.write_text(canonical_json({"source_record_index": 0, "sample_id": self.identity["sample_id"], "public_record_sha256": self.identity["public_record_sha256"], "question_sha256": self.identity["question_sha256"], "qa_type": "tal", "question": self.question}) + "\n")
        selection = freeze_selector_order(selector_path=self.selector, max_samples=1); self.selection = self.root / "selection.json"; write_selection(selection, self.selection)
        self.requirements = self.root / "requirements"; ontology = load_event_ontology(Path(__file__).resolve().parents[1] / "configs/v2/tal_event_ontology.yaml")
        freeze_tal_requirements(selection=selection, selector_sha256=hashlib.sha256(self.selector.read_bytes()).hexdigest(), selection_manifest_sha256=hashlib.sha256(self.selection.read_bytes()).hexdigest(), ontology=ontology, output_dir=self.requirements)
        self.source = self.root / "source.json"; self.source.write_text(canonical_json([self.row]) + "\n")
        policy = Path(__file__).resolve().parents[1] / "configs/v2/tal_timebase_sources.yaml"; unready = self.root / "unready"
        freeze_video_index(requirement_dir=self.requirements, selection_manifest_path=self.selection, source_json=self.source, frame_root=self.frames, source_prefix="/root/data", timebase_policy_path=policy, output_dir=unready)
        projection = json.loads((unready / "v2_public_media_projection.jsonl").read_text())
        self.timestamps = self.root / "timestamps.jsonl"; rows = []
        for order, reference in enumerate(refs): rows.append({"source_record_index": 0, "sample_id": self.identity["sample_id"], "public_record_sha256": self.identity["public_record_sha256"], "question_sha256": self.identity["question_sha256"], "frame_order": order, "source_frame_reference": reference, "timestamp_seconds": 6.0 if order == 7 else (20.0 if order == 21 else float(order)), "timestamp_unit": "seconds"})
        self.timestamps.write_bytes(b"".join((canonical_json(row) + "\n").encode() for row in rows))
        reference = self.root / "safe.json"; reference.write_text(canonical_json({"safe": True}) + "\n")
        self.provenance = self.root / "timestamps.provenance.json"; self.provenance.write_text(canonical_json({"format": "relive-v2-tal-timestamp-provenance-v1", "timestamp_manifest_sha256": hashlib.sha256(self.timestamps.read_bytes()).hexdigest(), "source_type": "VERSIONED_DATASET_TIMEBASE_ADAPTER", "dataset_name": "NurViD", "sample_id": self.identity["sample_id"], "source_record_index": 0, "public_record_sha256": self.identity["public_record_sha256"], "question_sha256": self.identity["question_sha256"], "time_origin": "CLIP_LOCAL_ZERO", "timestamp_origin": "FIRST_PRESENTED_FRAME", "timestamp_unit": "seconds", "mapping_method": "fixture", "adapter_version": "fixture-v1", "source_reference": str(reference), "source_reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(), "frame_count": 56, "gt_used": False, "assistant_or_gt_values_accessed": False, "model_calls_made": 0, "backend_loaded": False, "cache_opened": False, "certificate_created": False, "new_verified_count": 0, "certificate_status": "NOT_APPLICABLE", "public_media_projection_sha256": projection["public_media_projection_sha256"], "frame_sha256": [item["frame_sha256"] for item in projection["frames"]]}) + "\n")
        self.index = self.root / "index"; freeze_video_index(requirement_dir=self.requirements, selection_manifest_path=self.selection, source_json=self.source, frame_root=self.frames, source_prefix="/root/data", timebase_policy_path=policy, output_dir=self.index, public_timestamp_manifest_path=self.timestamps, public_timestamp_provenance_path=self.provenance)
        self.stage3a = self.root / "stage3a"; freeze_temporal_search_plan(requirement_dir=self.requirements, selection_manifest_path=self.selection, video_index_dir=self.index, timestamp_manifest_path=self.timestamps, timestamp_provenance_path=self.provenance, policy_path=Path(__file__).resolve().parents[1] / "configs/v2/tal_temporal_pyramid_policy.yaml", output_dir=self.stage3a)
        self.config = self.root / "config.json"; self.config.write_text(json.dumps({"policy": {"name": "semantic_spatial", "version": "relive-v1-policy-1", "strict_alternatives": True}, "backend": {"kind": "local_hf", "model": "fixture", "revision": "r1", "model_path": str(self.root / "model"), "checkpoint_metadata_sha256": "0" * 64, "model_class": "Model", "processor_class": "Processor", "chat_template_source": "fixture", "chat_template_sha256": "1" * 64, "chat_message_layout": "images_then_text", "processor_call_mode": "tokenized_chat_template", "chat_template_kwargs": {}, "trust_remote_code": False, "local_files_only": True, "dtype": "bfloat16", "device": "cuda:0", "device_map": None, "input_device": "cuda:0", "max_memory": None, "processor_min_pixels": None, "processor_max_pixels": None, "generation": {"do_sample": False, "max_new_tokens": 8}, "image_order": "chronological", "frame_encoding": "source", "timeout_seconds": 30, "max_retries": 0}}))
        self.out = self.root / "stage3b"

    def tearDown(self): self.temp.cleanup()
    def prepare(self): return prepare(requirement_dir=self.requirements, selection_manifest_path=self.selection, stage3a_dir=self.stage3a, video_index_dir=self.index, timestamp_manifest_path=self.timestamps, timestamp_provenance_path=self.provenance, config_path=self.config, output_dir=self.out)

    def test_packets_preflight_run_replay_and_no_admission(self):
        result = self.prepare(); self.assertEqual(result["packet_count"], 23)
        packets = [json.loads(line) for line in (self.out / "v2_stage3b_visual_packets.jsonl").read_text().splitlines()]
        self.assertTrue(all(len(row["frames"]) <= (16 if row["window_role"] == "GLOBAL_CONTEXT_DIAGNOSTIC" else 8) for row in packets))
        self.assertTrue(any(len(row["frames"][0]["logical_aliases"]) > 1 for row in packets))
        preflight(output_dir=self.out, config_path=self.config, backend_factory=lambda _: _FakeChoiceBackend())
        run = execute(output_dir=self.out, config_path=self.config, mode="run", backend_factory=lambda _: _FakeChoiceBackend())
        replay = execute(output_dir=self.out, config_path=self.config, mode="replay", backend_factory=lambda _: _FakeChoiceBackend())
        self.assertEqual((run["new_model_calls"], run["cache_hits"], replay["new_model_calls"], replay["cache_hits"]), (23, 0, 0, 23))
        self.assertEqual(run["numeric_results_sha256"], replay["numeric_results_sha256"])
        claims = [json.loads(line) for line in (self.out / "run" / "v2_stage3b_hypothesis_claims.jsonl").read_text().splitlines()]
        self.assertLessEqual(len(claims), 6); self.assertEqual(claims[-1]["polarity"], "NO_VISIBLE_EVENT"); self.assertTrue(all(row["status"] == "CANDIDATE_UNVERIFIED" for row in claims))
        self.assertTrue(validate(self.out)["status"] == "PASS")

    def test_deterministic_uniform_selection_margin_iou_and_drift_fail_closed(self):
        self.assertEqual(_packet_indices(17, 8), [0, 2, 5, 7, 9, 11, 14, 16])
        self.assertAlmostEqual(support_margin({"A": 0., "B": 0., "C": 0., "D": 0.}), -math.log(3))
        self.assertEqual(temporal_iou({"start_seconds": 0., "end_seconds": 8.}, {"start_seconds": 4., "end_seconds": 8.}), .5)
        self.prepare(); (self.stage3a / "v2_temporal_search_plan.jsonl").write_bytes(b"tampered")
        with self.assertRaises(CoarseHypothesisError): prepare(requirement_dir=self.requirements, selection_manifest_path=self.selection, stage3a_dir=self.stage3a, video_index_dir=self.index, timestamp_manifest_path=self.timestamps, timestamp_provenance_path=self.provenance, config_path=self.config, output_dir=self.root / "drift")


if __name__ == "__main__": unittest.main()
