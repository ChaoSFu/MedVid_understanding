"""GPU-free Phase 3-v3 frozen LOCAL_ATOMIC vertical-slice checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from relive.config import load_config
from relive.data.schemas import FIELD_SOURCES
from relive.phase35_v3 import (Phase35V3Error, candidate_rows, execute, preflight,
                               validate_prospective_manifest)
from relive.storage.artifacts import stable_hash

ROOT = Path(__file__).resolve().parents[1]


class Phase35V3Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.frames = []
        for index in range(2):
            path = self.root / f"f{index}.png"
            Image.new("RGB", (32, 24), (40 + index * 20, 80, 120)).save(path)
            self.frames.append(path)
        self.runtime = self.root / "fresh.runtime.jsonl"
        row = {
            "sample_id": "fresh-local", "task": "claim_verification", "question": "What is visible?",
            "frames": [{"frame_id": f"f{index}", "path": str(path), "order": index}
                       for index, path in enumerate(self.frames)],
            "target_claim": {"claim_id": "fresh-claim", "text": "The jaws of the forceps are touching tissue.",
                             "time_scope": {"frame_ids": ["f0", "f1"]}},
            "metadata": {"dataset_name": "fixture", "source_qa_type": "tal",
                         "runtime_adapter": "phase35_frozen_public_window_v1", "nonofficial_protocol": True},
        }
        payload = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        self.runtime.write_bytes(payload)
        runtime_sha = hashlib.sha256(payload).hexdigest()
        Path(str(self.runtime) + ".provenance.json").write_text(json.dumps({
            "schema_version": "relive-runtime-v1", "source_kind": "public_runtime",
            "runtime_sha256": runtime_sha, "field_sources": FIELD_SOURCES}), encoding="utf-8")
        Path(str(self.runtime) + ".gt_isolation_audit.json").write_text(json.dumps({
            "status": "PASS", "runtime_sha256": runtime_sha,
            "classification": "phase35_fresh_local_atomic_spatial_preflight_not_official_benchmark"}), encoding="utf-8")
        self.config = self.root / "mock.json"
        cfg = load_config(ROOT / "configs" / "mock_smoke.yaml")
        cfg["claims"]["contrast_fixtures"] = []
        cfg["policy"] = {"name": "semantic_spatial", "version": "relive-v1-policy-1", "strict_alternatives": True}
        cfg["adaptation"] = {"enabled": False, "actions": [], "expand_frames": 1}
        cfg["spatial"]["intervention"] = {"operator": "opaque_gray", "operator_version": "relive-opaque-gray-hard-mask-v1",
                                             "parameters": {"fill_rgb": [127, 127, 127]}}
        cfg["budget"].update(max_calls=5, max_candidates=1, max_spatial_proposals=1, max_rounds=1)
        self.config.write_text(json.dumps(cfg), encoding="utf-8")
        manifest = {
            "format": "relive-phase35-prospective-local-atomic-manifest-v1",
            "router_version": "relive-claim-scope-router-v1",
            "selection_status": "FROZEN_PRE_SPATIAL_CERTIFICATE_OUTCOMES",
            "selection_rule": {"input_order": "public runtime JSONL order", "include_scope": "LOCAL_ATOMIC",
                               "max_samples": 5, "exclude_sample_ids_from": "Phase 2 frozen candidate manifest"},
            "prospective_runtime": str(self.runtime), "prospective_runtime_sha256": runtime_sha,
            "excluded_historical_sample_ids_sha256": "0" * 64, "selected_sample_count": 1,
            "selected": [{"format": "relive-claim-scope-audit-v1", "sample_id": "fresh-local",
                          "dataset_name": "fixture", "source_qa_type": "tal", "claim_id": "fresh-claim",
                          "claim_text_sha256": hashlib.sha256(b"The jaws of the forceps are touching tissue.").hexdigest(),
                          "development_diagnostic_only": False, "claim_scope": "LOCAL_ATOMIC",
                          "single_roi_certificate_applicable": True, "reason_code": "LOCAL_OBJECT_STATE_OR_RELATION",
                          "router_version": "relive-claim-scope-router-v1", "gt_used": False,
                          "diagnostic_rationale": "local"}],
            "not_a_temporal_candidate_pool": True, "phase3_v3_executed": False, "gt_used": False,
            "prohibited_inputs_not_opened": ["reference_answer", "assistant_answer", "temporal_gt", "bbox_mask_gt",
                                               "struc_info", "RC_info", "evaluation_artifacts"],
        }
        content = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        manifest["manifest_sha256"] = hashlib.sha256(json.dumps([content], ensure_ascii=False, sort_keys=True,
                                                                  separators=(",", ":")).encode()).hexdigest()
        self.manifest = self.root / "prospective.json"
        self.manifest.write_text(json.dumps(manifest), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_preflight_run_and_replay_use_automatic_roi_only(self):
        out = self.root / "out"
        plan = preflight(config_path=self.config, runtime_path=self.runtime, prospective_manifest_path=self.manifest,
                         output_dir=out, require_real=False)
        self.assertEqual(plan["status"], "PASS")
        self.assertEqual(plan["model_calls_made"], 0)
        self.assertEqual(plan["planned_max_model_calls"], 5)
        self.assertTrue(plan["automatic_roi_only"])
        first = execute(config_path=self.config, runtime_path=self.runtime, prospective_manifest_path=self.manifest,
                        output_dir=out, mode="run", require_real=False)
        replay = execute(config_path=self.config, runtime_path=self.runtime, prospective_manifest_path=self.manifest,
                         output_dir=out, mode="replay", require_real=False)
        self.assertEqual(first["status"], "PASS")
        self.assertEqual(first["summary"]["new_model_calls"], 5)
        self.assertEqual(replay["summary"]["new_model_calls"], 0)
        self.assertEqual(replay["summary"]["cache_hits"], 5)
        self.assertFalse(first["reason_specific_refinement_called"])
        trace = json.loads((out / "run" / "phase35_v3_trace.jsonl").read_text())
        self.assertIsNotNone(trace["automatic_support_region"])
        self.assertEqual(trace["semantic_variants"]["drop"], "INSUFFICIENT")

    def test_rejects_nonlocal_manifest_and_gt_shaped_data(self):
        manifest = json.loads(self.manifest.read_text())
        manifest["selected"][0]["claim_scope"] = "MULTI_SUPPORT_POSSIBLE"
        content = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        manifest["manifest_sha256"] = hashlib.sha256(json.dumps([content], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.manifest.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(Phase35V3Error, "LOCAL_ATOMIC"):
            validate_prospective_manifest(self.manifest, self.runtime)
        manifest["selected"][0]["claim_scope"] = "LOCAL_ATOMIC"
        manifest["selected"][0]["bbox"] = [0, 0, 1, 1]
        content = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        manifest["manifest_sha256"] = hashlib.sha256(json.dumps([content], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.manifest.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(Phase35V3Error, "GT- or outcome-shaped"):
            validate_prospective_manifest(self.manifest, self.runtime)

    def test_candidate_is_exact_and_stable(self):
        _, samples = validate_prospective_manifest(self.manifest, self.runtime)
        first, digest = candidate_rows(samples)
        second, other = candidate_rows(samples)
        self.assertEqual(first, second)
        self.assertEqual(digest, other)
        self.assertEqual(first[0]["frame_ids"], ["f0", "f1"])
        self.assertEqual(first[0]["candidate_pool"], "exactly_one_frozen_public_window")


if __name__ == "__main__":
    unittest.main()
