from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

from baselines.timelens_medvidu.audit_artifacts import frame_path_audit, gt_leakage_audit
from baselines.timelens_medvidu.cache import build_cache_key
from baselines.timelens_medvidu.config import GROUNDER_PROMPT, MIN_TOKENS, PROMPT_VERSION, TOTAL_TOKENS
from baselines.timelens_medvidu.dataset import MedVidUTALTimeLensDataset, build_official_prompt, build_qwen3_frame_list_messages
from baselines.timelens_medvidu.dataset import build_messages_for_adapter
from baselines.timelens_medvidu.evaluate_medvidu_tal import evaluate_predictions
from baselines.timelens_medvidu.io_utils import append_jsonl, assert_no_gt_leak, completed_cache, read_json, write_json, write_jsonl
from baselines.timelens_medvidu.manifest import build_gt_free_row, build_manifest, choose_smoke_rows, extract_question, remap_path
from baselines.timelens_medvidu.parser import parse_timelens_answer
from baselines.timelens_medvidu.timestamp_adapters import (
    STRICT_SINGLE_FPS_ADAPTER,
    TEXTUAL_TIMESTAMP_IMAGE_SEQUENCE_ADAPTER,
    audit_adapter_for_row,
    build_strict_single_fps_messages,
    build_textual_timestamp_image_sequence_messages,
)
from baselines.timelens_medvidu.timelens_runner import prediction_path_for
from baselines.timelens_medvidu.temporal_mapper import TemporalMapper, audit_uniform_spacing


def sample(qa_type: str = "tal", n: int = 5, dataset_name: str = "AVOS") -> dict:
    frames = list(range(10, 10 + n))
    video = ["/root/data/AVOS/frames_15fps/v/dup.jpg" if i in (2, 3) else f"/root/data/AVOS/frames_15fps/v/{i:04d}.jpg" for i in range(n)]
    if dataset_name == "NurViD":
        frames = [40, 42, 44, 46, 48][:n]
        video = [f"/root/data/NurViD/frames_2fps/v/{x:06d}.jpg" for x in frames]
    return {
        "id": f"id-{dataset_name}-{qa_type}",
        "qa_type": qa_type,
        "dataset_name": dataset_name,
        "data_source": dataset_name,
        "video": video,
        "sampled_video_frames": frames,
        "metadata": {"fps": "1.0", "video_id": "v", "input_video_start_time": "0.0", "input_video_end_time": "4.0"},
        "conversations": [
            {"from": "human", "value": "<video>\nWhen does the action happen?"},
            {"from": "gpt", "value": "1-2 seconds."},
        ],
        "struc_info": [{"spans": [{"start": 1.0, "end": 2.0}], "answer": "leaky"}],
    }


class TimeLensMedVidUTests(unittest.TestCase):
    def test_tal_filtering(self):
        with tempfile.TemporaryDirectory() as td:
            data = [sample("tal"), sample("stg")]
            data_path = Path(td) / "data.json"
            out = Path(td) / "manifest.jsonl"
            write_json(data_path, data)
            report = build_manifest(data_path, out, new_data_root=None)
            self.assertEqual(report["n_total_samples"], 2)
            self.assertEqual(report["n_tal_samples"], 1)
            self.assertEqual(report["manifest_rows"], 1)

    def test_gt_stripping_and_question_extraction(self):
        row = build_gt_free_row(sample(), 0, new_data_root=None)
        self.assertEqual(extract_question(sample()), "<video>\nWhen does the action happen?")
        self.assertEqual(row["human_question"], "<video>\nWhen does the action happen?")
        self.assertNotIn("struc_info", row)
        self.assertNotIn("conversations", row)
        self.assertFalse(row["gt_information_available_to_model"])
        assert_no_gt_leak(row)

    def test_logical_frame_order_and_duplicate_preservation(self):
        row = build_gt_free_row(sample(n=5), 0, new_data_root=None)
        self.assertEqual(row["video"][2], row["video"][3])
        self.assertEqual([obs["frame_position"] for obs in row["frame_observations"]], [0, 1, 2, 3, 4])

    def test_source_frame_to_local_time_mapping(self):
        result = TemporalMapper.map("AVOS", [10, 25], ["/a", "/b"], {})
        self.assertEqual([obs.local_time for obs in result.observations], [0.0, 1.0])
        self.assertEqual(result.clip_duration, 1.0)

    def test_monotonic_timestamp_validation(self):
        with self.assertRaises(ValueError):
            audit_uniform_spacing([0.0, 1.0, 0.5])

    def test_uniform_spacing_and_effective_fps(self):
        audit = audit_uniform_spacing([5.0, 5.5, 6.0])
        self.assertTrue(audit.uniform_positive_spacing)
        self.assertEqual(audit.effective_fps, 2.0)
        self.assertEqual(audit.max_timing_error, 0.0)

    def test_clip_local_normalization(self):
        audit = audit_uniform_spacing([10.0, 12.0, 14.0])
        self.assertEqual(audit.effective_fps, 0.5)
        self.assertEqual(audit.median_timing_error, 0.0)

    def test_duplicate_timestamp_marks_single_fps_incompatible(self):
        audit = audit_uniform_spacing([0.0, 1.0, 1.0, 2.0])
        self.assertTrue(audit.duplicate_timestamps_present)
        self.assertFalse(audit.qwen_frame_list_single_fps_compatible)

    def test_official_prompt_exactness(self):
        question = "Find action."
        self.assertEqual(build_official_prompt(question), GROUNDER_PROMPT.format(question))

    def test_qwen3_frame_list_message_construction(self):
        row = build_gt_free_row(sample(n=3), 0, new_data_root=None)
        messages = build_qwen3_frame_list_messages(row, MIN_TOKENS, TOTAL_TOKENS)
        content = messages[0]["content"]
        self.assertEqual(content[0]["type"], "video")
        self.assertEqual(content[0]["video"], row["video"])
        self.assertIn("fps", content[0])
        self.assertEqual(content[1]["text"], GROUNDER_PROMPT.format(row["human_question"]))

    def test_parser_reuses_official_extract_time(self):
        status, parsed = parse_timelens_answer("The event happens in 1.5 - 3 seconds.")
        self.assertEqual(status, "OK")
        self.assertEqual(parsed, [[1.5, 3.0]])

    def test_no_timestamp_parse_case(self):
        status, parsed = parse_timelens_answer("No event is visible.")
        self.assertEqual(status, "NO_TIMESTAMP")
        self.assertEqual(parsed, [])

    def test_multiple_interval_preservation(self):
        status, parsed = parse_timelens_answer("1-2 seconds and 4-5 seconds")
        self.assertEqual(status, "OK")
        self.assertEqual(parsed, [[1.0, 2.0], [4.0, 5.0]])

    def test_cache_key_stability_and_change(self):
        row = build_gt_free_row(sample(n=3), 0, new_data_root=None)
        fp = {"model_path": "/m", "config": "h"}
        prompt = build_official_prompt(row["human_question"])
        pixel = {"min_tokens": 64, "total_tokens": 14336}
        decoding = {"do_sample": False}
        key1 = build_cache_key(row, fp, prompt, pixel, decoding)
        key2 = build_cache_key(row, fp, prompt, pixel, decoding)
        self.assertEqual(key1, key2)
        row["frame_observations"][0]["local_time"] = 99.0
        key3 = build_cache_key(row, fp, prompt, pixel, decoding)
        self.assertNotEqual(key1, key3)

    def test_cache_restart(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "pred.jsonl"
            append_jsonl(path, {"sample_id": "s", "cache_key": "k", "inference_error": None})
            self.assertEqual(set(completed_cache(path)), {"k"})

    def test_raw_result_gt_leakage(self):
        row = {
            "sample_id": "s",
            "gt_information_available_to_model": False,
            "parsed_timestamps": [[1, 2]],
        }
        assert_no_gt_leak(row)
        with self.assertRaises(AssertionError):
            assert_no_gt_leak({"sample_id": "s", "gt_answer": "bad"})

    def test_evaluation_join(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data_path = root / "data.json"
            pred_path = root / "pred.jsonl"
            write_json(data_path, [sample()])
            write_jsonl(
                pred_path,
                [
                    {
                        "sample_id": "000000::id-AVOS-tal::tal",
                        "parsed_timestamps": [[1.0, 2.0]],
                        "parse_valid": True,
                        "parse_status": "OK",
                    }
                ],
            )
            result = evaluate_predictions(pred_path, data_path, root / "eval", "smoke")
            self.assertEqual(result["audit"]["matched"], 1)
            self.assertEqual(result["overall"]["Recall@0.50"], 1.0)

    def test_inference_dataset_cannot_access_gt(self):
        processor = SimpleNamespace()
        ds = MedVidUTALTimeLensDataset([{"struc_info": []}], processor, MIN_TOKENS, TOTAL_TOKENS)
        with self.assertRaises(AssertionError):
            _ = ds[0]

    def test_path_remap(self):
        self.assertEqual(remap_path("/root/data/AVOS/x.jpg", "/root/data", "/real"), "/real/AVOS/x.jpg")
        self.assertEqual(remap_path("/other/x.jpg", "/root/data", "/real"), "/other/x.jpg")

    def test_smoke_selection_is_gt_blind_size(self):
        rows = [build_gt_free_row(sample(n=3 + i, dataset_name="AVOS"), i, new_data_root=None) for i in range(5)]
        selected = choose_smoke_rows(rows, size=3, seed=42)
        self.assertEqual(len(selected), 3)
        self.assertTrue(all("struc_info" not in row for row in selected))

    def test_frame_and_leakage_audits(self):
        with tempfile.TemporaryDirectory() as td:
            frame = Path(td) / "f.jpg"
            frame.write_bytes(b"x")
            row = build_gt_free_row(sample(n=1), 0, new_data_root=None)
            row["video"] = [str(frame)]
            row["frame_observations"][0]["frame_path"] = str(frame)
            self.assertTrue(frame_path_audit([row])["all_frame_paths_exist"])
            self.assertEqual(gt_leakage_audit([row])["leak_count"], 0)

    def test_strict_single_fps_adapter_accepts_only_compatible_rows(self):
        row = build_gt_free_row(sample(n=3, dataset_name="EgoSurgery"), 0, new_data_root=None)
        row["sampled_video_frames"] = [10, 11, 12]
        row["frame_observations"][0]["local_time"] = 0.0
        row["frame_observations"][1]["local_time"] = 1.0
        row["frame_observations"][2]["local_time"] = 2.0
        row["timestamp_spacing_audit"] = audit_uniform_spacing([0.0, 1.0, 2.0]).to_dict()
        row["effective_fps"] = 1.0
        messages = build_strict_single_fps_messages(row)
        self.assertEqual(messages[0]["content"][0]["type"], "video")
        self.assertEqual(messages[0]["content"][0]["fps"], 1.0)

        row["timestamp_spacing_audit"] = audit_uniform_spacing([0.0, 1.0, 1.0]).to_dict()
        with self.assertRaises(ValueError):
            build_strict_single_fps_messages(row)

    def test_textual_timestamp_image_sequence_uses_original_local_times(self):
        row = build_gt_free_row(sample(n=3, dataset_name="NurViD"), 0, new_data_root=None)
        row["frame_observations"][1]["local_time"] = 1.25
        row["frame_observations"][2]["local_time"] = 1.25
        messages = build_textual_timestamp_image_sequence_messages(row)
        content = messages[0]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "Frame 1 timestamp: 0.000000 seconds"})
        self.assertEqual(content[1], {"type": "image", "image": row["video"][0]})
        self.assertEqual(content[2], {"type": "text", "text": "Frame 2 timestamp: 1.250000 seconds"})
        self.assertEqual(content[4], {"type": "text", "text": "Frame 3 timestamp: 1.250000 seconds"})
        self.assertEqual(content[-1]["text"], GROUNDER_PROMPT.format(row["human_question"]))

    def test_adapter_preflight_records_scientific_distinction(self):
        row = build_gt_free_row(sample(n=3), 0, new_data_root=None)
        row["timestamp_spacing_audit"] = audit_uniform_spacing([0.0, 1.0, 1.0]).to_dict()
        strict = audit_adapter_for_row(row, STRICT_SINGLE_FPS_ADAPTER, process_qwen=False).to_dict()
        textual = audit_adapter_for_row(row, TEXTUAL_TIMESTAMP_IMAGE_SEQUENCE_ADAPTER, process_qwen=False).to_dict()
        self.assertFalse(strict["applicable"])
        self.assertTrue(textual["applicable"])
        self.assertTrue(textual["official_prompt_included_verbatim"])
        self.assertFalse(textual["official_prompt_exact_match"])
        self.assertFalse(textual["additional_temporal_resampling_any"] if "additional_temporal_resampling_any" in textual else textual["additional_temporal_resampling"])

    def test_dataset_message_dispatch_keeps_strict_default_and_textual_override(self):
        row = build_gt_free_row(sample(n=3), 0, new_data_root=None)
        row["timestamp_spacing_audit"] = audit_uniform_spacing([0.0, 1.0, 1.0]).to_dict()
        with self.assertRaises(ValueError):
            build_messages_for_adapter(row, MIN_TOKENS, TOTAL_TOKENS, STRICT_SINGLE_FPS_ADAPTER)
        textual = build_messages_for_adapter(row, MIN_TOKENS, TOTAL_TOKENS, TEXTUAL_TIMESTAMP_IMAGE_SEQUENCE_ADAPTER)
        self.assertEqual(textual[0]["content"][0]["type"], "text")
        self.assertEqual(textual[0]["content"][1]["type"], "image")

    def test_textual_prediction_path_is_separate_from_strict_cache(self):
        root = Path("/tmp/out")
        self.assertEqual(
            prediction_path_for(root, smoke=True, timestamp_adapter=STRICT_SINGLE_FPS_ADAPTER),
            root / "predictions" / "timelens8b_tal_smoke_predictions.jsonl",
        )
        self.assertEqual(
            prediction_path_for(root, smoke=True, timestamp_adapter=TEXTUAL_TIMESTAMP_IMAGE_SEQUENCE_ADAPTER),
            root / "predictions" / "timelens8b_tal_smoke_predictions_textual_timestamp_image_sequence.jsonl",
        )

    def test_cache_key_changes_with_timestamp_adapter(self):
        row = build_gt_free_row(sample(n=3), 0, new_data_root=None)
        fp = {"model_path": "/m", "config": "h"}
        prompt = build_official_prompt(row["human_question"])
        pixel = {"min_tokens": 64, "total_tokens": 14336}
        decoding = {"do_sample": False}
        strict = build_cache_key(row, fp, prompt, pixel, decoding, timestamp_adapter=STRICT_SINGLE_FPS_ADAPTER)
        textual = build_cache_key(row, fp, prompt, pixel, decoding, timestamp_adapter=TEXTUAL_TIMESTAMP_IMAGE_SEQUENCE_ADAPTER)
        self.assertNotEqual(strict, textual)


if __name__ == "__main__":
    unittest.main()
