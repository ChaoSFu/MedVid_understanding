from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from baselines.timechat7b_medvidu.cache import build_cache_key
from baselines.timechat7b_medvidu.checkpoint import verify_component_paths
from baselines.timechat7b_medvidu.config import GenerationConfig, OFFICIAL_SYSTEM_PROMPT, OFFICIAL_VTUNE_GROUNDING_PROMPT
from baselines.timechat7b_medvidu.evaluate_timechat_medvidu_tal import evaluate_predictions
from baselines.timechat7b_medvidu.frame_loader import (
    assert_timestamp_msg_consistency,
    build_timechat_msg,
    build_timechat_visual_message,
    load_frames_as_timechat_tensor,
    official_uniform_indices,
    round_timechat_timestamp,
    select_frame_inputs,
    timestamp_texts,
)
from baselines.timechat7b_medvidu.io_utils import append_jsonl, assert_no_gt_leak, completed_cache, write_json, write_jsonl
from baselines.timechat7b_medvidu.manifest import build_gt_free_row, build_manifest, choose_smoke_rows, extract_human_question
from baselines.timechat7b_medvidu.model_adapter import MedVidUTimeChatVTune
from baselines.timechat7b_medvidu.parser import parse_timechat_answer


def sample(qa_type: str = "tal", n: int = 5, dataset_name: str = "AVOS") -> dict:
    frames = list(range(10, 10 + n))
    video = ["/root/data/AVOS/frames_15fps/v/dup.jpg" if i in (2, 3) else f"/root/data/AVOS/frames_15fps/v/{i:04d}.jpg" for i in range(n)]
    if dataset_name == "NurViD":
        frames = [40 + 2 * i for i in range(n)]
        video = [f"/root/data/NurViD/frames_2fps/v/{x:06d}.jpg" for x in frames]
    return {
        "id": f"id-{dataset_name}-{qa_type}",
        "qa_type": qa_type,
        "dataset_name": dataset_name,
        "data_source": dataset_name,
        "video": video,
        "sampled_video_frames": frames,
        "metadata": {"fps": "1.0", "video_id": "v", "input_video_start_time": "0.0", "input_video_end_time": "8.0"},
        "conversations": [
            {"from": "human", "value": "<video>\nWhen does the action happen?"},
            {"from": "gpt", "value": "1-2 seconds."},
        ],
        "struc_info": [{"spans": [{"start": 1.0, "end": 2.0}], "action": "leaky"}],
    }


class FakeParser:
    def extract_time(self, answer: str):
        if answer == "":
            return [0, 0]
        if answer == "fallback":
            return [0, 0]
        return [3.0, 1.0] if answer == "swap" else [1.0, 3.0]

    def extract_time2(self, answer: str):
        return [4.0, 6.0] if answer == "fallback" else [0, 0]


class TimeChatMedVidUTests(unittest.TestCase):
    def test_tal_filtering_inventory(self):
        with tempfile.TemporaryDirectory() as td:
            data_path = Path(td) / "data.json"
            out_root = Path(td) / "out"
            write_json(data_path, [sample("tal"), sample("stg")])
            inventory = build_manifest(data_path, out_root, new_data_root=None)
            self.assertEqual(inventory["n_total"], 2)
            self.assertEqual(inventory["n_tal"], 1)
            self.assertEqual(inventory["manifest_rows"], 1)

    def test_gt_stripping_and_human_question_verbatim(self):
        row = build_gt_free_row(sample(), 0, new_data_root=None)
        self.assertEqual(extract_human_question(sample()), "<video>\nWhen does the action happen?")
        self.assertEqual(row["human_question"], "<video>\nWhen does the action happen?")
        self.assertNotIn("conversations", row)
        self.assertNotIn("struc_info", row)
        self.assertNotIn("answer", row)
        self.assertFalse(row["gt_fields_present"])
        assert_no_gt_leak(row)

    def test_frame_order_and_logical_duplicate_preservation(self):
        row = build_gt_free_row(sample(n=5), 0, new_data_root=None)
        self.assertEqual(row["ordered_frame_paths"][2], row["ordered_frame_paths"][3])
        self.assertEqual([x["logical_position"] for x in row["logical_frame_identities"]], [0, 1, 2, 3, 4])

    def test_uniform_sampling_boundaries(self):
        self.assertEqual(official_uniform_indices(3), [0, 1, 2])
        self.assertEqual(len(official_uniform_indices(96)), 96)
        expected = __import__("numpy").arange(0, 120, 120 / 96).astype(int).tolist()
        self.assertEqual(official_uniform_indices(120), expected)

    def test_timestamp_rounding_and_msg_consistency(self):
        self.assertEqual(round_timechat_timestamp(3.27), "3.3")
        texts = timestamp_texts([0.0, 0.49, 1.04])
        msg = build_timechat_msg(["0.0", "0.5", "1.0"])
        assert_timestamp_msg_consistency(msg, texts)
        visual_msg = build_timechat_visual_message(msg)
        self.assertEqual(visual_msg.count("<ImageHere>"), 1)
        self.assertIn(msg, visual_msg)

    def test_select_frame_inputs_preserves_msg_and_indices(self):
        selected = select_frame_inputs(["a", "b", "b"], [0.0, 1.25, 1.25])
        self.assertEqual(selected["selected_frame_paths"], ["a", "b", "b"])
        self.assertEqual(selected["selected_rounded_timestamps"], ["0.0", "1.2", "1.2"])
        self.assertIn("The video contains 3 frames sampled at", selected["msg"])

    def test_frame_loader_shape_and_rgb(self):
        try:
            from PIL import Image
            import torch  # noqa: F401
        except Exception as exc:
            self.skipTest(f"frame loader image dependencies unavailable: {exc!r}")
        with tempfile.TemporaryDirectory() as td:
            p1 = Path(td) / "a.png"
            p2 = Path(td) / "b.png"
            Image.new("L", (9, 7), color=128).save(p1)
            Image.new("RGB", (4, 5), color=(1, 2, 3)).save(p2)
            tensor = load_frames_as_timechat_tensor([str(p1), str(p2)], image_size=8)
            self.assertEqual(tuple(tensor.shape), (3, 2, 8, 8))
            self.assertEqual(str(tensor.dtype), "torch.uint8")

    def test_timestamp_mapper_and_clip_local_normalization(self):
        row = build_gt_free_row(sample(n=2), 0, new_data_root=None)
        self.assertEqual(row["local_timestamps"][0], 0.0)
        self.assertGreaterEqual(row["local_timestamps"][1], row["local_timestamps"][0])

    def test_official_prompt_system_generation_exactness(self):
        self.assertEqual(
            OFFICIAL_VTUNE_GROUNDING_PROMPT.format(event="Q"),
            "Localize the visual content described by the given textual query 'Q' in the video, and output the start and end timestamps in seconds.",
        )
        self.assertIn("Follow the instructions carefully", OFFICIAL_SYSTEM_PROMPT)
        gen = GenerationConfig()
        self.assertEqual((gen.num_beams, gen.do_sample, gen.temperature, gen.max_new_tokens, gen.max_length), (1, False, 0.05, 300, 2000))

    def test_official_parser_fallback_and_parse_valid(self):
        parsed = parse_timechat_answer(FakeParser(), "fallback")
        self.assertEqual(parsed["official_parsed_span"], [4.0, 6.0])
        self.assertTrue(parsed["extract_time2_fallback_used"])
        self.assertTrue(parsed["parse_valid"])
        invalid = parse_timechat_answer(FakeParser(), "")
        self.assertFalse(invalid["parse_valid"])

    def test_no_gt_in_raw_prediction_shape(self):
        pred = {
            "sample_id": "s",
            "question": "q",
            "official_parsed_span": [1, 2],
            "parse_valid": True,
            "gt_information_available_to_model": False,
        }
        assert_no_gt_leak(pred)
        with self.assertRaises(AssertionError):
            assert_no_gt_leak({"sample_id": "s", "ground_truth": []})

    def test_cache_key_deterministic_and_cache_rerun(self):
        row = build_gt_free_row(sample(n=3), 0, new_data_root=None)
        gen = {"do_sample": False}
        key1 = build_cache_key(row, "ckpt", "prompt", gen)
        key2 = build_cache_key(row, "ckpt", "prompt", gen)
        self.assertEqual(key1, key2)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "pred.jsonl"
            append_jsonl(path, {"cache_key": key1, "inference_error": None})
            self.assertEqual(set(completed_cache(path)), {key1})

    def test_evaluation_join_and_duplicate_detection(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data_path = root / "data.json"
            pred_path = root / "pred.jsonl"
            write_json(data_path, [sample()])
            row = {
                "sample_id": "000000::id-AVOS-tal::tal",
                "official_parsed_span": [1.0, 2.0],
                "parse_valid": True,
            }
            write_jsonl(pred_path, [row, row])
            result = evaluate_predictions(pred_path, data_path, root / "eval")
            self.assertEqual(result["join_audit"]["matched"], 1)
            self.assertEqual(result["join_audit"]["duplicate"], 1)
            self.assertEqual(result["medvidu_official_tal_metrics"]["Recall@0.50"], 1.0)

    def test_missing_checkpoint_failure(self):
        with tempfile.TemporaryDirectory() as td:
            missing = Path(td) / "missing.pth"
            with self.assertRaises(FileNotFoundError):
                verify_component_paths(missing, missing, missing, missing)

    def test_smoke_selection_is_gt_blind(self):
        rows = [build_gt_free_row(sample(n=3 + i, dataset_name="AVOS"), i, new_data_root=None) for i in range(5)]
        selected = choose_smoke_rows(rows, size=3, seed=42)
        self.assertEqual(len(selected), 3)
        self.assertTrue(all("struc_info" not in row for row in selected))

    def test_no_source_video_backtracking_or_pseudo_mp4(self):
        row = build_gt_free_row(sample(n=3), 0, new_data_root=None)
        self.assertFalse(row["source_video_accessed"])
        self.assertTrue(all(not path.endswith(".mp4") for path in row["ordered_frame_paths"]))

    def test_adapter_parser_methods_match_official_path_contract(self):
        adapter = MedVidUTimeChatVTune.__new__(MedVidUTimeChatVTune)
        self.assertEqual(adapter.extract_time("The event happens in 2 - 4 seconds."), [2.0, 4.0])
        self.assertEqual(adapter.extract_time2("from 5 to 8 seconds"), [5.0, 8.0])


if __name__ == "__main__":
    unittest.main()
