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


if __name__ == "__main__":
    unittest.main()
