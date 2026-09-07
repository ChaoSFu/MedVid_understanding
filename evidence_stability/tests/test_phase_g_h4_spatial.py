import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from evidence_stability.spatial import (
    audit_stg_schema,
    bbox_area_fraction,
    box_iou,
    build_spatial_pointer_prompt,
    h4_gt_leakage_audit,
    h4_model_window_manifest_row,
    h4_generate_stg_windows,
    h4_window_temporal_alignment,
    normalize_stg_sample,
    normalized_to_pixel_bbox,
    parse_normalized_bbox_json,
)


def stg_sample(idx=0):
    return {
        "id": f"1.0$VID{idx}$0$100",
        "qa_type": "stg",
        "dataset_name": "CholecTrack20",
        "data_source": "CholecTrack20",
        "metadata": {
            "video_id": f"VID{idx}",
            "fps": "1.0",
            "input_video_start_frame": "0",
            "input_video_end_frame": "100",
        },
        "sampled_video_frames": [0, 4, 8, 12],
        "video": [f"/root/data/CholecTrack20/VID{idx}/{i:06d}.png" for i in [0, 4, 8, 12]],
        "conversations": [
            {
                "from": "human",
                "value": "<video>\nTrack the hook from 0.0 to 4.0 seconds, predicting its location every 4 seconds.",
            },
            {"from": "gpt", "value": "0.0 seconds: [10, 20, 30, 40] 4.0 seconds: [12, 22, 32, 42]"},
        ],
        "struc_info": [
            {
                "start": 0.0,
                "end": 4.0,
                "stride": 4,
                "object": "hook",
                "bbox_dict": {
                    "0.0": [10, 20, 30, 40],
                    "4.0": [12, 22, 32, 42],
                },
            }
        ],
    }


class PhaseGH4SpatialTests(unittest.TestCase):
    def test_strict_bbox_parser_accepts_exact_json(self):
        parsed = parse_normalized_bbox_json('{"bbox": [10, 20, 900, 950]}')
        self.assertTrue(parsed["bbox_valid"])
        self.assertEqual(parsed["bbox"], [10, 20, 900, 950])

    def test_strict_bbox_parser_rejects_extra_text_or_keys(self):
        self.assertFalse(parse_normalized_bbox_json('Here: {"bbox": [0, 0, 10, 10]}')["bbox_valid"])
        self.assertFalse(parse_normalized_bbox_json('{"bbox": [0, 0, 10, 10], "text": "x"}')["bbox_valid"])
        self.assertFalse(parse_normalized_bbox_json('{"bbox": [10, 0, 10, 20]}')["bbox_valid"])
        self.assertFalse(parse_normalized_bbox_json('{"bbox": [10.5, 0, 20, 20]}')["bbox_valid"])
        self.assertTrue(parse_normalized_bbox_json('{"bbox": [10.0, 0.0, 20.0, 20.0]}')["bbox_valid"])

    def test_bbox_area_pixel_mapping_and_iou(self):
        self.assertEqual(bbox_area_fraction([0, 0, 500, 500]), 0.25)
        self.assertEqual(normalized_to_pixel_bbox([0, 0, 500, 500], 1920, 1080), [0, 0, 960, 540])
        self.assertEqual(box_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertEqual(box_iou([0, 0, 10, 10], [10, 10, 20, 20]), 0.0)

    def test_spatial_pointer_prompt_is_json_only_and_gt_free(self):
        prompt = build_spatial_pointer_prompt("Track the hook.")
        self.assertIn('"bbox": [x1, y1, x2, y2]', prompt)
        lowered = prompt.lower()
        for forbidden in ["ground truth", "iou", "true_support", "spurious_support"]:
            self.assertNotIn(forbidden, lowered)

    def test_stg_schema_audit_records_official_metric_and_unfrozen_threshold(self):
        audit = audit_stg_schema([stg_sample()])
        self.assertEqual(audit["n_stg"], 1)
        self.assertEqual(audit["spatial_gt"]["type"], "per-requested-timestamp xyxy bbox_dict")
        self.assertIn("iou@0.5", audit["official_stg_metric"]["reported_metrics"])
        self.assertIsNone(audit["official_stg_metric"]["single_canonical_positive_threshold"])
        self.assertEqual(
            audit["official_stg_metric"]["threshold_freeze_status"],
            "NEEDS_HUMAN_FREEZE_BEFORE_TRUE_SPATIAL_SUPPORT_LABELS",
        )

    def test_h4_schema_audit_cli_writes_stop_files(self):
        from importlib.util import module_from_spec, spec_from_file_location

        script = Path(__file__).resolve().parents[1] / "scripts" / "13_audit_h4_stg_schema.py"
        spec = spec_from_file_location("h4_schema_audit_script", script)
        module = module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "stg.json"
            out = Path(tmp) / "out"
            data.write_text(__import__("json").dumps([stg_sample()]), encoding="utf-8")
            old_argv = module.sys.argv
            try:
                module.sys.argv = ["13_audit_h4_stg_schema.py", "--data_json", str(data), "--output_dir", str(out)]
                with redirect_stdout(StringIO()):
                    module.main()
            finally:
                module.sys.argv = old_argv
            self.assertTrue((out / "audit" / "medvidu_stg_schema.md").exists())
            self.assertTrue((out / "audit" / "temporal_eligibility_protocol_review.md").exists())
            self.assertTrue((out / "audit" / "spatial_label_threshold_review.md").exists())
            status = (out / "summary" / "h4_protocol_audit_status.md").read_text(encoding="utf-8")
            self.assertIn("READY_FOR_H4_PREFLIGHT", status)

    def test_h4_preflight_model_manifest_is_gt_free(self):
        normalized, error = normalize_stg_sample(stg_sample(), original_index=7, verify_paths=False)
        self.assertIsNone(error)
        assert normalized is not None
        windows = h4_generate_stg_windows(normalized, window_size=2, stride=2, drop_last=True)
        self.assertTrue(windows)
        model_row = h4_model_window_manifest_row(normalized, windows[0])
        leakage = h4_gt_leakage_audit([model_row])
        self.assertEqual(leakage["gt_leakage"], 0)
        self.assertNotIn("stg_bbox_dict", model_row)
        self.assertNotIn("processed_gt_spans", model_row)

    def test_h4_temporal_alignment_uses_h2_rule(self):
        normalized, error = normalize_stg_sample(stg_sample(), original_index=8, verify_paths=False)
        self.assertIsNone(error)
        assert normalized is not None
        windows = h4_generate_stg_windows(normalized, window_size=2, stride=2, drop_last=True)
        aligned = h4_window_temporal_alignment(normalized, windows[0])
        self.assertIn("h4_temporally_eligible", aligned)
        self.assertEqual(aligned["temporal_eligibility_rule"], "h2_strong_temporal_alignment")

    def test_h4_preflight_probe_cli_dry_run(self):
        from importlib.util import module_from_spec, spec_from_file_location

        prepare_script = Path(__file__).resolve().parents[1] / "scripts" / "14_prepare_h4_preflight.py"
        probe_script = Path(__file__).resolve().parents[1] / "scripts" / "15_probe_h4_spatial_preflight.py"
        prepare_spec = spec_from_file_location("h4_prepare_script", prepare_script)
        probe_spec = spec_from_file_location("h4_probe_script", probe_script)
        prepare_module = module_from_spec(prepare_spec)
        probe_module = module_from_spec(probe_spec)
        assert prepare_spec and prepare_spec.loader and probe_spec and probe_spec.loader
        prepare_spec.loader.exec_module(prepare_module)
        probe_spec.loader.exec_module(probe_module)

        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "stg.json"
            out = Path(tmp) / "out"
            data.write_text(__import__("json").dumps([stg_sample()]), encoding="utf-8")

            old_prepare_argv = prepare_module.sys.argv
            old_probe_argv = probe_module.sys.argv
            try:
                prepare_module.sys.argv = [
                    "14_prepare_h4_preflight.py",
                    "--data_json",
                    str(data),
                    "--output_dir",
                    str(out),
                    "--window_size",
                    "2",
                    "--stride",
                    "2",
                    "--preflight_qa",
                    "1",
                ]
                with redirect_stdout(StringIO()):
                    prepare_module.main()

                probe_module.sys.argv = [
                    "15_probe_h4_spatial_preflight.py",
                    "--manifest",
                    str(out / "manifest" / "h4_model_manifest_gt_free.jsonl"),
                    "--output_dir",
                    str(out),
                    "--model_backend",
                    "dummy",
                    "--dry_run",
                ]
                with redirect_stdout(StringIO()):
                    probe_module.main()
            finally:
                prepare_module.sys.argv = old_prepare_argv
                probe_module.sys.argv = old_probe_argv

            summary = __import__("json").loads((out / "summary" / "h4_spatial_preflight_probe_summary.json").read_text(encoding="utf-8"))
            self.assertFalse(summary["model_inference_executed"])
            self.assertEqual(summary["gt_leakage_audit"]["gt_leakage"], 0)

    def test_h4_spatial_intervention_generation_cli(self):
        from importlib.util import module_from_spec, spec_from_file_location

        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow unavailable")

        script = Path(__file__).resolve().parents[1] / "scripts" / "16_generate_h4_spatial_interventions.py"
        spec = spec_from_file_location("h4_intervention_script", script)
        module = module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame_paths = []
            for idx in range(2):
                path = root / f"frame_{idx}.png"
                image = Image.new("RGB", (64, 48), (20 + idx, 80, 140))
                for x in range(16, 32):
                    for y in range(12, 24):
                        image.putpixel((x, y), (220, 20, 20))
                image.save(path)
                frame_paths.append(str(path))

            pointer_path = root / "spatial_pointer_predictions.jsonl"
            supporting_path = root / "supporting.jsonl"
            out = root / "out"
            row = {
                "qa_id": "0001::clip",
                "clip_id": "clip",
                "candidate_id": "0001::clip::pos0000-0001",
                "window_id": "0001::clip::pos0000-0001",
                "dataset_name": "EgoSurgery",
                "human_question": "Track the tool.",
                "support_prediction": "YES",
                "bbox_valid": True,
                "predicted_bbox_norm": [100, 100, 400, 400],
                "bbox_area_fraction": 0.09,
                "frame_paths": frame_paths,
                "logical_frame_paths": ["EgoSurgery/a.png", "EgoSurgery/b.png"],
                "prompt_version": "spatial_pointer_v1",
                "prompt_hash": "abc",
            }
            pointer_path.write_text(__import__("json").dumps(row) + "\n", encoding="utf-8")
            supporting_path.write_text(__import__("json").dumps(row) + "\n", encoding="utf-8")
            old_argv = module.sys.argv
            try:
                module.sys.argv = [
                    "16_generate_h4_spatial_interventions.py",
                    "--spatial_pointer_predictions",
                    str(pointer_path),
                    "--supporting_candidates",
                    str(supporting_path),
                    "--output_dir",
                    str(out),
                    "--visual_limit",
                    "1",
                ]
                with redirect_stdout(StringIO()):
                    module.main()
            finally:
                module.sys.argv = old_argv

            summary = __import__("json").loads((out / "summary" / "h4_spatial_intervention_generation_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["input"]["n_valid_bbox_candidates"], 1)
            self.assertEqual(summary["generation"]["generation_valid"], 4)
            self.assertEqual(summary["gt_leakage_audit"]["gt_leakage"], 0)
            self.assertEqual(summary["pixel_audit"]["n_failures"], 0)

    def test_h4_spatial_intervention_probe_cli_dry_run(self):
        from importlib.util import module_from_spec, spec_from_file_location

        script = Path(__file__).resolve().parents[1] / "scripts" / "17_probe_h4_spatial_interventions.py"
        spec = spec_from_file_location("h4_intervention_probe_script", script)
        module = module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.jsonl"
            out = root / "out"
            row = {
                "intervention_id": "0001::clip::pos0000-0001::KEEP_ROI_V1",
                "candidate_id": "0001::clip::pos0000-0001",
                "qa_id": "0001::clip",
                "clip_id": "clip",
                "window_id": "0001::clip::pos0000-0001",
                "dataset_name": "EgoSurgery",
                "human_question": "Track the tool.",
                "support_prediction": "YES",
                "predicted_bbox_norm": [100, 100, 400, 400],
                "bbox_valid": True,
                "bbox_area_fraction": 0.09,
                "control_bbox_norm": [600, 600, 900, 900],
                "control_valid": True,
                "intervention_type": "KEEP_ROI_V1",
                "intervention_family": "H4_SPATIAL_ROI",
                "intervention_frame_paths": ["a.png", "b.png"],
                "logical_frame_paths": ["EgoSurgery/a.png", "EgoSurgery/b.png"],
                "n_frames": 2,
                "n_unique_frames": 2,
                "blur_version": "gaussian_blur_sigma_0.05_min_hw_v1",
                "blur_sigma_rule": "sigma = 0.05 * min(height, width)",
                "prompt_version": "spatial_pointer_v1",
                "prompt_hash": "abc",
            }
            manifest.write_text(__import__("json").dumps(row) + "\n", encoding="utf-8")
            old_argv = module.sys.argv
            try:
                module.sys.argv = [
                    "17_probe_h4_spatial_interventions.py",
                    "--intervention_manifest",
                    str(manifest),
                    "--output_dir",
                    str(out),
                    "--model_backend",
                    "dummy",
                    "--dry_run",
                ]
                with redirect_stdout(StringIO()):
                    module.main()
            finally:
                module.sys.argv = old_argv
            summary = __import__("json").loads((out / "summary" / "h4_spatial_intervention_preflight_probe_summary.json").read_text(encoding="utf-8"))
            self.assertFalse(summary["model_inference_executed"])
            self.assertEqual(summary["gt_leakage_audit"]["gt_leakage"], 0)

    def test_h4_preflight_join_cli_computes_spatial_delta(self):
        from importlib.util import module_from_spec, spec_from_file_location

        script = Path(__file__).resolve().parents[1] / "scripts" / "18_join_analyze_h4_spatial_preflight.py"
        spec = spec_from_file_location("h4_join_script", script)
        module = module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            window_gt = root / "window_gt.jsonl"
            pointer = root / "pointer.jsonl"
            interventions = root / "interventions.jsonl"
            out = root / "out"
            qa_id = "0001::clip"
            gt_rows = []
            pointer_rows = []
            intervention_rows = []
            specs = [
                ("true", [0, 0, 500, 500], "TRUE_SPATIAL_SUPPORT", {"KEEP_ROI_V1": "YES", "DROP_ROI_V1": "NO"}),
                ("spur", [500, 500, 900, 900], "SPURIOUS_SPATIAL_SUPPORT", {"KEEP_ROI_V1": "NO", "DROP_ROI_V1": "YES"}),
            ]
            for name, bbox, _label, preds in specs:
                candidate_id = f"{qa_id}::pos{name}"
                gt_rows.append(
                    {
                        "qa_id": qa_id,
                        "clip_id": "clip",
                        "candidate_id": candidate_id,
                        "window_id": candidate_id,
                        "dataset_name": "EgoSurgery",
                        "human_question": "Track the tool.",
                        "local_timestamps": [0.0, 1.0],
                        "stg_bbox_dict": {"0.0": [0, 0, 50, 50]},
                        "h4_temporally_eligible": True,
                        "temporal_eligibility_rule": "h2_strong_temporal_alignment",
                        "n_gt_visible_in_window": 1,
                        "n_gt_visible_total": 1,
                        "evidence_density": 0.5,
                        "gt_evidence_recall": 1.0,
                    }
                )
                pointer_rows.append(
                    {
                        "qa_id": qa_id,
                        "clip_id": "clip",
                        "candidate_id": candidate_id,
                        "window_id": candidate_id,
                        "dataset_name": "EgoSurgery",
                        "support_prediction": "YES",
                        "bbox_valid": True,
                        "predicted_bbox_norm": bbox,
                        "bbox_area_fraction": 0.25,
                        "processor_metadata": {"image_sizes": [[100, 100]]},
                    }
                )
                for intervention_type, pred in preds.items():
                    intervention_rows.append(
                        {
                            "qa_id": qa_id,
                            "clip_id": "clip",
                            "candidate_id": candidate_id,
                            "window_id": candidate_id,
                            "intervention_id": f"{candidate_id}::{intervention_type}",
                            "intervention_type": intervention_type,
                            "parsed_prediction": pred,
                        }
                    )
            window_gt.write_text("\n".join(__import__("json").dumps(row) for row in gt_rows) + "\n", encoding="utf-8")
            pointer.write_text("\n".join(__import__("json").dumps(row) for row in pointer_rows) + "\n", encoding="utf-8")
            interventions.write_text("\n".join(__import__("json").dumps(row) for row in intervention_rows) + "\n", encoding="utf-8")
            old_argv = module.sys.argv
            try:
                module.sys.argv = [
                    "18_join_analyze_h4_spatial_preflight.py",
                    "--window_gt",
                    str(window_gt),
                    "--spatial_pointer_predictions",
                    str(pointer),
                    "--intervention_predictions",
                    str(interventions),
                    "--output_dir",
                    str(out),
                ]
                with redirect_stdout(StringIO()):
                    module.main()
            finally:
                module.sys.argv = old_argv
            summary = __import__("json").loads((out / "summary" / "h4_preflight_join_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["candidate_join_audit"]["candidate_spatial_type_counts"]["TRUE_SPATIAL_SUPPORT"], 1)
            self.assertEqual(summary["candidate_join_audit"]["candidate_spatial_type_counts"]["SPURIOUS_SPATIAL_SUPPORT"], 1)
            self.assertEqual(summary["primary_preflight_qa"]["n_paired_qa"], 1)
            self.assertEqual(summary["primary_preflight_qa"]["mean_delta_C"], 1.0)


if __name__ == "__main__":
    unittest.main()
