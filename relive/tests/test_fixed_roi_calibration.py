"""Contract test for the isolated fixed-ROI calibration preflight."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from relive.data.schemas import load_runtime


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "fixed_roi_calibration.py"


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


class FixedRoiCalibrationTests(unittest.TestCase):
    def test_preflight_writes_closed_runtime_and_zero_call_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = root / "public.png"
            Image.new("RGB", (48, 32), (20, 80, 160)).save(frame)
            controls = []
            for identifier, kind, expected in (
                ("positive", "positive_control_candidate", {"ORIGINAL": "SUPPORTED", "KEEP_TARGET": "SUPPORTED", "DROP_TARGET": "INSUFFICIENT", "DROP_MATCHED_CONTROL": "SUPPORTED"}),
                ("broad", "broad_multiple_support_uncertainty_control", {"ORIGINAL": "SUPPORTED", "KEEP_TARGET": "SUPPORTED", "DROP_TARGET": "SUPPORTED", "DROP_MATCHED_CONTROL": "SUPPORTED"}),
                ("negative", "visually_exclusive_negative_control", {"ORIGINAL": "CONTRADICTED", "KEEP_TARGET": "CONTRADICTED", "DROP_TARGET": "INSUFFICIENT", "DROP_MATCHED_CONTROL": "CONTRADICTED"}),
            ):
                frames = [{"frame_id": f"{identifier}:frame:0", "path": str(frame), "order": 0}]
                controls.append({"candidate_id": identifier, "kind": kind, "sample_id": f"sample:{identifier}",
                                 "dataset_name": "fixture", "source_qa_type": "fixture", "frame_ids": [frames[0]["frame_id"]],
                                 "frame_paths": [str(frame)], "frame_orders": [0], "frame_sha256": digest(frames),
                                 "atomic_claim_en": f"A visible {identifier} object.",
                                 "diagnostic_only_bbox_normalized_xyxy": [0.1, 0.1, 0.4, 0.4],
                                 "matched_control_bbox_normalized_xyxy": [0.6, 0.1, 0.9, 0.4],
                                 "expected_outcomes_frozen_pre_inference": expected})
                controls[-1]["claim_sha256"] = hashlib.sha256(controls[-1]["atomic_claim_en"].encode()).hexdigest()
            manifest = {"format": "relive-calibration-manifest-frozen-v1", "calibration_only": True,
                        "selection_status": "FROZEN_PRE_INFERENCE", "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD~1"], cwd=ROOT, text=True).strip(),
                        "runtime_path": "/public/runtime.jsonl", "runtime_sha256": "0" * 64,
                        "phase16a_preparation_directory": "/public/phase16", "prohibited_inputs_not_opened": ["reference_answer"],
                        "selected_positive_control": controls[0], "ancillary_controls": controls[1:]}
            manifest["frozen_manifest_sha256"] = digest(manifest)
            manifest_path = root / "frozen.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            model = root / "reviewed-model"
            model.mkdir()
            config = root / "local.json"
            config.write_text(json.dumps({"framework_version": "ReliVE-v1", "policy": {
                "name": "semantic_spatial", "version": "relive-v1-policy-1", "strict_alternatives": True}, "backend": {
                "kind": "local_hf", "model": "fixture-model", "revision": "fixture-r1", "model_path": str(model),
                "checkpoint_metadata_sha256": "0" * 64, "model_class": "FixtureModel", "processor_class": "FixtureProcessor",
                "chat_template_source": "fixture", "chat_template_sha256": "1" * 64,
                "chat_message_layout": "images_then_text", "processor_call_mode": "tokenized_chat_template",
                "chat_template_kwargs": {}, "trust_remote_code": False, "local_files_only": True,
                "dtype": "bfloat16", "device": "cuda:0", "device_map": None, "input_device": "cuda:0",
                "max_memory": None, "processor_min_pixels": None, "processor_max_pixels": None,
                "generation": {"do_sample": False, "max_new_tokens": 8}, "image_order": "chronological",
                "frame_encoding": "source", "timeout_seconds": 30, "max_retries": 0,
            }}), encoding="utf-8")
            output, cache = root / "preflight", root / "cache"
            completed = subprocess.run([sys.executable, str(SCRIPT), "--mode", "preflight", "--manifest", str(manifest_path),
                                         "--config", str(config), "--output-dir", str(output), "--cache-dir", str(cache)],
                                        cwd=ROOT, env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")}, text=True, capture_output=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            plan = json.loads((output / "calibration_preflight.json").read_text())
            self.assertEqual(plan["model_calls_made"], 0)
            self.assertEqual(plan["planned_unique_semantic_calls"], 12)
            self.assertEqual(plan["fixed_roi_support"], "ISOLATED_CALIBRATION_ONLY_ADAPTER")
            self.assertEqual(plan["commit_relation"], "FROZEN_COMMIT_ANCESTOR_OF_EXECUTION_HEAD")
            runtime = load_runtime(output / "calibration.runtime.jsonl")
            self.assertEqual(len(runtime), 3)
            self.assertTrue(all(len(sample.frames) == 1 for sample in runtime))


if __name__ == "__main__":
    unittest.main()
