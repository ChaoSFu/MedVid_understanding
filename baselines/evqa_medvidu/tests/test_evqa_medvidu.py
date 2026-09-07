from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from baselines.evqa_medvidu.cache import build_cache_key
from baselines.evqa_medvidu.backend import EVQABackend, _build_ref_box_kwargs, _forbid_segmentation_output, _needs_sam2_frames, _reset_segmentation_output
from baselines.evqa_medvidu.config import RunConfig
from baselines.evqa_medvidu.evaluation.join import gt_index, join_audit, prediction_index
from baselines.evqa_medvidu.frame_adapter import nearest_indices_for_target_times, select_model_frames
from baselines.evqa_medvidu.io_utils import append_jsonl, assert_no_gt_leak, completed_cache, write_json, write_jsonl
from baselines.evqa_medvidu.manifest import build_gt_free_row, build_manifests, canonical_task, choose_smoke_rows
from baselines.evqa_medvidu.raw_output import parse_cvs_components, parse_temporal_segments
from baselines.evqa_medvidu.run_full import run_full_tasks
from baselines.evqa_medvidu.run_smoke import cap_model_frames_for_dry_run
from baselines.evqa_medvidu.spatial.mask_frame_alignment import build_mask_frame_alignment
from baselines.evqa_medvidu.spatial.mask_to_bbox import tight_bbox
from baselines.evqa_medvidu.stg_time_spec import build_stg_target_alignment, parse_stg_target_schedule
from baselines.evqa_medvidu.temporal_mapper import TemporalMapper
from baselines.evqa_medvidu.tasks import get_adapter
from baselines.evqa_medvidu.tasks.base import SchemaMismatch


def sample(qa_type: str = "stg", n: int = 5, dataset_name: str = "AVOS") -> dict:
    frames = list(range(10, 10 + n))
    question = "<video>\nQuestion?"
    if qa_type == "stg":
        question = "<video>\nGive the bounding boxes every 1 second from 0 to 0 seconds."
    return {
        "id": f"id-{qa_type}",
        "qa_type": qa_type,
        "dataset_name": dataset_name,
        "data_source": dataset_name,
        "video": ["/root/data/AVOS/frames_15fps/v/dup.jpg" if i in (2, 3) else f"/root/data/AVOS/frames_15fps/v/{i:04d}.jpg" for i in range(n)],
        "sampled_video_frames": frames,
        "metadata": {"fps": "1.0", "video_id": "v"},
        "conversations": [
            {"from": "human", "value": question},
            {"from": "gpt", "value": "leaky answer"},
        ],
        "struc_info": [{"start": 1, "end": 2, "bbox_dict": {"1": [1, 2, 3, 4]}}],
        "is_RC": False,
        "RC_info": None,
    }


def rc_sample() -> dict:
    s = sample("region_caption_gpt", dataset_name="EgoSurgery")
    s["is_RC"] = True
    s["RC_info"] = {"start_frame": s["video"][1], "start_frame_bbox": [1, 2, 30, 40]}
    return s


def cvs_sample() -> dict:
    s = sample("cvs_assessment", dataset_name="Cholec80_CVS")
    s["conversations"][0]["value"] = "<video>\nScore CVS."
    s["conversations"][1]["value"] = "Two structures: 0, Cystic plate: 1, Hepatocystic triangle: 2"
    s["struc_info"] = [{"cvs_scores": {"two_structures": 0, "cystic_plate": 1, "hepatocystic_triangle": 2}}]
    return s


class EVQAMedVidUTests(unittest.TestCase):
    def test_task_filtering_aliases(self):
        self.assertEqual(canonical_task("stg"), "stg")
        self.assertEqual(canonical_task("region_caption_gemini"), "rc")
        self.assertEqual(canonical_task("cvs_assessment"), "cvs")

    def test_gt_stripping_and_question_preservation(self):
        row = build_gt_free_row(sample(), 0, new_data_root=None)
        self.assertEqual(row["human_question"], "<video>\nGive the bounding boxes every 1 second from 0 to 0 seconds.")
        self.assertNotIn("conversations", row)
        self.assertNotIn("struc_info", row)
        assert_no_gt_leak(row)

    def test_rc_region_allowed_but_target_stripped(self):
        row = build_gt_free_row(rc_sample(), 1, new_data_root=None)
        self.assertTrue(row["provided_region"]["provided_region_is_task_input"])
        self.assertEqual(row["provided_region"]["start_frame_bbox"], [1.0, 2.0, 30.0, 40.0])
        assert_no_gt_leak(row, allowed_paths={"$.provided_region"})

    def test_frame_order_and_duplicate_preservation(self):
        row = build_gt_free_row(sample(n=4), 0, new_data_root=None)
        self.assertEqual(row["ordered_frame_paths"][2], row["ordered_frame_paths"][3])
        self.assertEqual([x["logical_position"] for x in row["logical_frame_identities"]], [0, 1, 2, 3])

    def test_sampling_deterministic_nearest(self):
        self.assertEqual(nearest_indices_for_target_times([0.0, 0.4, 1.2], [0.5, 1.0]), [1, 2])
        result = select_model_frames(["a", "b", "c", "d"], [0.0, 1.0, 2.0, 3.0], 3.0, fps=1.0, max_frames=2)
        self.assertEqual(result.selected_logical_indices, [0, 1])
        self.assertEqual(result.selected_frame_paths, ["a", "b"])

    def test_default_sampling_uses_every_benchmark_frame(self):
        paths = [f"frame-{i}" for i in range(180)]
        timestamps = [float(i) / 30.0 for i in range(180)]
        result = select_model_frames(paths, timestamps, timestamps[-1])
        self.assertEqual(result.selected_logical_indices, list(range(180)))
        self.assertEqual(result.selected_frame_paths, paths)

    def test_dry_run_frame_cap_changes_sampling_and_hash(self):
        row = build_gt_free_row(sample(n=200, dataset_name="SyntheticLongClip"), 0, new_data_root=None)
        capped = cap_model_frames_for_dry_run(row, 16)
        self.assertEqual(capped["model_sampling"]["n_selected"], 16)
        self.assertNotEqual(capped["selected_frame_hash"], row["selected_frame_hash"])
        self.assertFalse(capped["formal_result_allowed"])
        self.assertIn("not_formal", capped["model_sampling"]["policy"])

    def test_cache_key_deterministic_and_resume_cache(self):
        row = build_gt_free_row(sample(n=3), 0, new_data_root=None)
        key1 = build_cache_key(row, {"model": "m"}, "prompt", {"adapter": 1})
        key2 = build_cache_key(row, {"model": "m"}, "prompt", {"adapter": 1})
        self.assertEqual(key1, key2)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "raw.jsonl"
            append_jsonl(path, {"cache_key": key1, "error": None})
            self.assertEqual(completed_cache(path), {key1})

    def test_visual_resolution_policy_changes_cache_key(self):
        row = build_gt_free_row(sample(n=3), 0, new_data_root=None)
        low = build_cache_key(
            row,
            {"model": "m"},
            "prompt",
            {"adapter": 1},
            visual_input_config={"max_pixels_per_frame": 128 * 28 * 28},
        )
        high = build_cache_key(
            row,
            {"model": "m"},
            "prompt",
            {"adapter": 1},
            visual_input_config={"max_pixels_per_frame": 256 * 28 * 28},
        )
        self.assertNotEqual(low, high)

    def test_inference_constraint_changes_cache_key(self):
        row = build_gt_free_row(sample(n=3), 0, new_data_root=None)
        unrestricted = build_cache_key(
            row,
            {"model": "m"},
            "prompt",
            {"adapter": 1},
            inference_config={"forbid_segmentation_output": False},
        )
        text_only = build_cache_key(
            row,
            {"model": "m"},
            "prompt",
            {"adapter": 1},
            inference_config={"forbid_segmentation_output": True},
        )
        self.assertNotEqual(unrestricted, text_only)

    def test_mask_bbox_edge_cases(self):
        self.assertIsNone(tight_bbox(np.zeros((3, 4), dtype=np.uint8)))
        mask = np.zeros((5, 6), dtype=np.uint8)
        mask[2, 3] = 1
        self.assertEqual(tight_bbox(mask), [3.0, 2.0, 3.0, 2.0])
        mask[1:4, 2:5] = 1
        self.assertEqual(tight_bbox(mask), [2.0, 1.0, 4.0, 3.0])

    def test_mask_frame_alignment(self):
        row = build_gt_free_row(sample(n=2), 0, new_data_root=None)
        alignment = build_mask_frame_alignment(row, 2)
        self.assertEqual(alignment[1]["logical_medvidu_frame_index"], 1)
        with self.assertRaises(ValueError):
            build_mask_frame_alignment(row, 3)

    def test_stg_multiple_masklets_stop(self):
        adapter = get_adapter("stg")
        row = build_gt_free_row(sample(n=2), 0, new_data_root=None)
        raw = {"sample_id": row["sample_id"], "manifest_row": row, "raw_textual_response": "[[0, 1]]", "raw_masklets": [{"mask": np.zeros((2, 3, 3))}, {"mask": np.zeros((2, 3, 3))}]}
        with self.assertRaises(SchemaMismatch):
            adapter.parse_evqa_output(raw)

    def test_stg_single_masklet_to_medvidu_answer(self):
        adapter = get_adapter("stg")
        row = build_gt_free_row(sample(n=2), 0, new_data_root=None)
        row["stg_target_alignment"] = [
            {
                "target_timestamp": 1.0,
                "logical_medvidu_frame_index": 1,
                "source_frame_index": 11,
                "matched_frame_timestamp": 1.0,
                "absolute_timing_error_seconds": 0.0,
                "frame_path": row["ordered_frame_paths"][1],
            }
        ]
        masklet = np.zeros((2, 4, 5), dtype=np.uint8)
        masklet[1, 1:3, 2:4] = 1
        raw = {"sample_id": row["sample_id"], "manifest_row": row, "raw_textual_response": "Answer [[0, 1]]", "raw_masklets": [{"mask": masklet}]}
        parsed = adapter.parse_evqa_output(raw)
        pred = adapter.to_medvidu_prediction(row, parsed)
        self.assertEqual(pred["answer"], "1.0 seconds: [2.0, 1.0, 3.0, 2.0]")

    def test_cholectrack20_uses_25_hz_source_timebase(self):
        result = TemporalMapper.map("CholecTrack20", [20751, 20776, 20801], ["a", "b", "c"], {"fps": "1.0"})
        self.assertEqual([obs.local_time for obs in result.observations], [0.0, 1.0, 2.0])

    def test_stg_schedule_and_nearest_alignment(self):
        schedule = parse_stg_target_schedule("What are the boxes sampled every 4 seconds from 4.0 to 12.0 seconds?")
        self.assertEqual(schedule.target_timestamps, (4.0, 8.0, 12.0))
        alignment = build_stg_target_alignment(
            [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
            list(range(7)),
            [f"f{i}" for i in range(7)],
            schedule,
        )
        self.assertEqual([item["logical_medvidu_frame_index"] for item in alignment], [2, 4, 6])

    def test_stg_schedule_covers_between_seconds_wording(self):
        schedule = parse_stg_target_schedule("Between 150.0 seconds and 206.0 seconds, where is it every 8 seconds?")
        self.assertEqual(schedule.target_timestamps, (150.0, 158.0, 166.0, 174.0, 182.0, 190.0, 198.0, 206.0))

    def test_stg_schedule_does_not_append_non_grid_end(self):
        schedule = parse_stg_target_schedule("Track it every 8 seconds from 66 to 78 seconds.")
        self.assertEqual(schedule.target_timestamps, (66.0, 74.0))

    def test_full_runner_is_ordered_resumable_and_isolates_sample_errors(self):
        class FakeBackend:
            def run_row(self, row):
                if row["sample_id"] == "rc-bad":
                    raise RuntimeError("synthetic failure")
                status = "cached" if row["sample_id"].endswith("cached") else "new"
                return {"status": status, "sample_id": row["sample_id"]}, None, None

        with tempfile.TemporaryDirectory() as td:
            cfg = RunConfig(output_root=Path(td))
            cfg.make_dirs()
            for task, ids in {"stg": ["stg-new"], "rc": ["rc-bad"], "cvs": ["cvs-cached"]}.items():
                write_jsonl(
                    cfg.manifest_dir / f"{task}_gt_free.jsonl",
                    [{"sample_id": sample_id, "task": task} for sample_id in ids],
                )

            calls = []

            def evaluator(pred_path, data_path, output_path):
                calls.append(pred_path.stem)
                write_json(output_path, {"evaluated": pred_path.stem})
                return {"evaluated": pred_path.stem}

            report = run_full_tasks(cfg, FakeBackend(), evaluators={task: evaluator for task in ("stg", "rc", "cvs")})
            self.assertEqual(report["task_order"], ["stg", "rc", "cvs"])
            self.assertEqual(report["tasks"]["stg"]["new"], 1)
            self.assertEqual(report["tasks"]["cvs"]["cached"], 1)
            self.assertEqual(report["tasks"]["rc"]["errors"], 1)
            self.assertEqual(calls, ["stg", "rc", "cvs"])
            self.assertEqual(report["status"], "COMPLETE_WITH_SAMPLE_ERRORS")

    def test_temporal_and_cvs_parsers(self):
        self.assertEqual(parse_temporal_segments("evidence [[1.5, 2.0], [3, 4]]"), [[1.5, 2.0], [3.0, 4.0]])
        self.assertEqual(
            parse_cvs_components("Two structures: 0, Cystic plate: 1, Hepatocystic triangle: 2"),
            {"two_structures": 0, "cystic_plate": 1, "hepatocystic_triangle": 2},
        )

    def test_cvs_parse_invalid_preserved(self):
        adapter = get_adapter("cvs")
        row = build_gt_free_row(cvs_sample(), 2, new_data_root=None)
        parsed = adapter.parse_evqa_output({"raw_textual_response": "unclear", "manifest_row": row})
        pred = adapter.to_medvidu_prediction(row, parsed)
        self.assertFalse(pred["parse_valid"])
        self.assertEqual(pred["parse_status"], "PARSE_INVALID")

    def test_rc_ref_box_kwargs_match_official_batch_nesting(self):
        try:
            from PIL import Image
        except Exception as exc:
            self.skipTest(f"PIL unavailable: {exc!r}")

        with tempfile.TemporaryDirectory() as td:
            paths = []
            for i in range(3):
                path = Path(td) / f"{i}.jpg"
                Image.new("RGB", (100, 50), color=(i, i, i)).save(path)
                paths.append(str(path))
            s = rc_sample()
            s["video"] = paths
            s["sampled_video_frames"] = [10, 11, 12]
            s["RC_info"]["start_frame"] = paths[1]
            row = build_gt_free_row(s, 1, new_data_root=None)
            kwargs = _build_ref_box_kwargs(row, sam2_image_size=768, n_selected_frames=3)
            self.assertEqual(len(kwargs["point_coords"]), 1)
            self.assertEqual(len(kwargs["point_coords"][0]), 1)
            self.assertEqual(tuple(kwargs["point_coords"][0][0].shape), (1, 2, 2))
            self.assertEqual(tuple(kwargs["point_labels"][0][0].shape), (1, 2))
            self.assertEqual(tuple(kwargs["point_frames"][0][0].shape), (1,))

    def test_cvs_generation_omits_sam2_frames(self):
        self.assertTrue(_needs_sam2_frames("stg"))
        self.assertFalse(_needs_sam2_frames("rc"))
        self.assertFalse(_needs_sam2_frames("cvs"))
        self.assertFalse(_forbid_segmentation_output("stg"))
        self.assertTrue(_forbid_segmentation_output("rc"))
        self.assertTrue(_forbid_segmentation_output("cvs"))

    def test_rc_and_cvs_prompts_use_text_output_contracts(self):
        backend = EVQABackend(Path("/unused-model"), Path("/unused-output"), {"model": "test"})
        rc_row = build_gt_free_row(rc_sample(), 1, new_data_root=None)
        cvs_row = build_gt_free_row(cvs_sample(), 2, new_data_root=None)
        rc_prompt = backend.build_prompt(rc_row)
        cvs_prompt = backend.build_prompt(cvs_row)
        self.assertIn("<|ref|>", rc_prompt)
        self.assertIn("plain-English description", rc_prompt)
        self.assertIn("Do not output masks or special tokens", rc_prompt)
        self.assertIn("Two structures: <0|1|2>", cvs_prompt)
        self.assertIn("Do not output masks, temporal evidence, special tokens, or explanations", cvs_prompt)

    def test_rc_special_token_output_is_invalid(self):
        adapter = get_adapter("rc")
        row = build_gt_free_row(rc_sample(), 1, new_data_root=None)
        parsed = adapter.parse_evqa_output(
            {"raw_textual_response": "The green object <|seg|>.", "manifest_row": row}
        )
        prediction = adapter.to_medvidu_prediction(row, parsed)
        self.assertFalse(prediction["parse_valid"])
        self.assertEqual(prediction["parse_status"], "PARSE_INVALID")
        self.assertEqual(prediction["answer"], "")

    def test_stg_resets_segmentation_output_buffer(self):
        class Model:
            pass

        model = Model()
        model.seg = ["previous output"]
        _reset_segmentation_output(model, "stg")
        self.assertEqual(model.seg, [])
        model.seg = ["non-stg output"]
        _reset_segmentation_output(model, "cvs")
        self.assertEqual(model.seg, ["non-stg output"])

    def test_manifest_inventory_and_smoke_gt_blind(self):
        with tempfile.TemporaryDirectory() as td:
            data_path = Path(td) / "data.json"
            out = Path(td) / "out"
            write_json(data_path, [sample(), rc_sample(), cvs_sample()])
            inv = build_manifests(RunConfig(data_path=data_path, output_root=out, new_data_root=None))
            self.assertEqual(inv["manifest_rows"], {"stg": 1, "rc": 1, "cvs": 1})
            rows = [build_gt_free_row(sample(n=3 + i), i, new_data_root=None) for i in range(4)]
            smoke = choose_smoke_rows(rows, size=2, seed=42)
            self.assertEqual(len(smoke), 2)
            self.assertTrue(all("struc_info" not in row for row in smoke))

    def test_evaluation_join_duplicate_missing_extra(self):
        gt = gt_index([sample(), cvs_sample()], "stg")
        pred, duplicates = prediction_index([{"sample_id": next(iter(gt)), "answer": "a"}, {"sample_id": next(iter(gt)), "answer": "b"}, {"sample_id": "extra", "answer": "c"}])
        audit = join_audit(gt, pred, duplicates)
        self.assertEqual(audit["matched"], 1)
        self.assertEqual(audit["extra"], 1)
        self.assertEqual(audit["duplicates"], 1)

    def test_no_source_video_backtracking_or_fake_mp4(self):
        row = build_gt_free_row(sample(n=3), 0, new_data_root=None)
        self.assertFalse(row["source_video_accessed"])
        self.assertFalse(row["fake_mp4_used"])
        self.assertTrue(all(not path.endswith(".mp4") for path in row["ordered_frame_paths"]))


if __name__ == "__main__":
    unittest.main()
