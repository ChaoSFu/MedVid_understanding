from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path

from baselines.dig32_medvidu.config import REQUESTED_K, VIDEO_REFINEMENT_WLEN
from baselines.dig32_medvidu.manifest.build_gt_free_manifest import build_gt_free_row, frame_count_audit, rc_compatibility_audit
from baselines.dig32_medvidu.models.base import (
    build_frame_prompt,
    load_inputs,
    selected_positions_hash,
    sha256_file,
    target_cache_key,
)
from baselines.dig32_medvidu.run_target import validate_same_evidence
from baselines.dig32_medvidu.selector.dig_adapter import (
    build_selector_record,
    global_uniform_positions,
    positions_hash,
    refine_and_select,
)
from baselines.dig32_medvidu.selector.freeze_selector_manifest import assert_selector_no_gt_leak, selector_cache_key
from baselines.dig32_medvidu.selector.query_identifier import parse_query_type
from baselines.dig32_medvidu.selector.reward_adapter import parse_reward
from baselines.videoitg_qwen35.io_utils import read_jsonl, write_jsonl


REPO_ROOT = Path(__file__).resolve().parents[3]
DIG_REPO = REPO_ROOT / "third_party" / "DIG"


def sample(qa_type: str = "tal", n: int = 40, dataset_name: str = "AVOS") -> dict:
    return {
        "id": f"id-{qa_type}",
        "qa_type": qa_type,
        "dataset_name": dataset_name,
        "data_source": dataset_name,
        "video": ["/frames/dup.jpg" if i in (2, 3) else f"/frames/{i:04d}.jpg" for i in range(n)],
        "sampled_video_frames": list(range(10, 10 + n)),
        "metadata": {"fps": 1.0, "video_id": "v"},
        "conversations": [
            {"from": "human", "value": "<video>\nWhere is the action?"},
            {"from": "gpt", "value": "10-20 seconds."},
        ],
        "struc_info": [{"start": 10, "end": 20, "answer": "leaky"}],
        "RC_info": None,
        "is_RC": False,
    }


def selector_cfg() -> dict:
    return {
        "selector_version": "dig32_fixed_v1",
        "requested_k": 32,
        "video_refinement_wlen": 2,
        "query_identifier_model": "Qwen/Qwen3-Next-80B-A3B-Instruct",
        "reward_lmm": "Qwen3-VL-8B-Instruct",
        "cafs_model": "facebook/dinov2-base",
        "cafs_sample_per_sec": 2,
        "cafs_infer_batch_size": 64,
        "question_hash": "qh",
    }


class Dig32MedVidUTests(unittest.TestCase):
    def test_gt_stripping_reuses_existing_manifest_builder(self):
        row = build_gt_free_row(sample(), new_data_root=None)
        self.assertNotIn("struc_info", row)
        self.assertNotIn("conversations", row)
        self.assertEqual(row["question"], "<video>\nWhere is the action?")

    def test_ordered_logical_positions_preserve_duplicate_paths(self):
        row = build_gt_free_row(sample(n=5), new_data_root=None)
        self.assertEqual(row["video"][2], row["video"][3])
        selected = [2, 3]
        record = build_selector_record(row, "global", selected, selected, "commit", selector_cfg())
        self.assertEqual(record["selected_frame_paths_chronological"], ["/frames/dup.jpg", "/frames/dup.jpg"])
        self.assertEqual(record["selected_original_positions_chronological"], [2, 3])

    def test_global_uniform_sampling_exact_k_for_n_ge_32(self):
        selected = global_uniform_positions(100, REQUESTED_K)
        self.assertEqual(len(selected), 32)
        self.assertEqual(selected[0], 0)
        self.assertEqual(selected[-1], 99)
        self.assertEqual(selected, sorted(selected))

    def test_global_uniform_sampling_effective_k_for_short_video(self):
        selected = global_uniform_positions(10, REQUESTED_K)
        self.assertEqual(selected, list(range(10)))

    def test_official_video_refinement_semantic_equivalence(self):
        intervals, selected = refine_and_select([0, 10, 100, 0], [0, 5, 10, 15, 20], DIG_REPO, k=4, wlen=2)
        self.assertEqual(VIDEO_REFINEMENT_WLEN, 2)
        self.assertEqual(intervals, [[0, 15]])
        self.assertEqual(selected, [0, 5, 10, 15])

    def test_local_selected_indices_bounds_and_chronological_reorder(self):
        row = build_gt_free_row(sample(n=50), new_data_root=None)
        selected_score_order = [20, 5, 10]
        record = build_selector_record(row, "local", selected_score_order, sorted(selected_score_order), "commit", selector_cfg())
        self.assertEqual(record["selected_original_positions"], [20, 5, 10])
        self.assertEqual(record["selected_original_positions_chronological"], [5, 10, 20])
        self.assertTrue(all(0 <= x < 50 for x in record["selected_original_positions_chronological"]))

    def test_source_frame_index_and_timestamp_mapping(self):
        row = build_gt_free_row(sample(n=3), new_data_root=None)
        record = build_selector_record(row, "global", [0, 2], [0, 2], "commit", selector_cfg())
        self.assertEqual(record["selected_source_frame_indices_chronological"], [10, 12])
        self.assertEqual(record["selected_local_times_chronological"], [0.0, 2.0 / 15.0])

    def test_selector_cache_key_contains_scientific_controls(self):
        row = build_gt_free_row(sample(n=40), new_data_root=None)
        key1 = selector_cache_key(row, "commit-a", selector_cfg())
        key2 = selector_cache_key(row, "commit-b", selector_cfg())
        self.assertNotEqual(key1, key2)

    def test_selector_output_deterministic(self):
        row = build_gt_free_row(sample(n=40), new_data_root=None)
        r1 = build_selector_record(row, "global", [0, 1], [0, 1], "commit", selector_cfg())
        r2 = build_selector_record(row, "global", [0, 1], [0, 1], "commit", selector_cfg())
        self.assertEqual(json.dumps(r1, sort_keys=True), json.dumps(r2, sort_keys=True))

    def test_manifest_sha256_stable(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "m.jsonl"
            write_jsonl(path, [{"a": 1}, {"b": 2}])
            self.assertEqual(sha256_file(path), sha256_file(path))

    def test_selector_no_gt_fields_allows_required_false_audit_flag(self):
        row = build_gt_free_row(sample(n=40), new_data_root=None)
        record = build_selector_record(row, "global", [0, 1], [0, 1], "commit", selector_cfg())
        assert_selector_no_gt_leak(record)
        record["gt_answer"] = "bad"
        with self.assertRaises(AssertionError):
            assert_selector_no_gt_leak(record)

    def test_target_model_cannot_import_selector(self):
        for rel in ("models/base.py", "models/qwen_hf.py", "models/qwen3vl.py", "models/qwen35.py", "models/qwen38.py"):
            tree = ast.parse((REPO_ROOT / "baselines" / "dig32_medvidu" / rel).read_text())
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    imports.append(node.module)
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
            self.assertFalse(any("selector" in item for item in imports), rel)

    def test_target_backbones_receive_same_positions(self):
        row = build_gt_free_row(sample(n=40), new_data_root=None)
        record = build_selector_record(row, "global", [0, 10, 20], [0, 10, 20], "commit", selector_cfg())
        hashes = {positions_hash(record["selected_original_positions_chronological"]) for _ in range(4)}
        self.assertEqual(len(hashes), 1)

    def test_selected_positions_hash_mismatch_rejected(self):
        row = build_gt_free_row(sample(n=40), new_data_root=None)
        record = build_selector_record(row, "global", [0, 1], [0, 1], "commit", selector_cfg())
        validate_same_evidence(record)
        record["selected_positions_hash"] = "bad"
        with self.assertRaises(RuntimeError):
            validate_same_evidence(record)

    def test_selector_manifest_sha_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            manifest = Path(td) / "manifest.jsonl"
            selector = Path(td) / "selector.jsonl"
            write_jsonl(manifest, [{"sample_id": "s"}])
            write_jsonl(selector, [{"sample_id": "s"}])
            with self.assertRaises(RuntimeError):
                load_inputs(manifest, selector, expected_selector_sha256="bad")

    def test_same_task_prompt_and_timestamps(self):
        row = build_gt_free_row(sample(n=40), new_data_root=None)
        record = build_selector_record(row, "global", [0, 2], [0, 2], "commit", selector_cfg())
        prompt1, ts1 = build_frame_prompt(row, record)
        prompt2, ts2 = build_frame_prompt(row, record)
        self.assertEqual(prompt1, prompt2)
        self.assertEqual(ts1, ts2)

    def test_deterministic_target_cache_key_changes_with_selector_sha(self):
        row = build_gt_free_row(sample(n=40), new_data_root=None)
        record = build_selector_record(row, "global", [0, 1], [0, 1], "commit", selector_cfg())
        prompt, _ = build_frame_prompt(row, record)
        k1 = target_cache_key(row, record, {"model": "qwen"}, "sha-a", prompt, {"do_sample": False}, {"total_pixels": 1})
        k2 = target_cache_key(row, record, {"model": "qwen"}, "sha-b", prompt, {"do_sample": False}, {"total_pixels": 1})
        self.assertNotEqual(k1, k2)

    def test_empty_output_parser_failure_isolated(self):
        row = build_gt_free_row(sample(n=40), new_data_root=None)
        status, parsed = __import__("baselines.dig32_medvidu.tasks", fromlist=["get_adapter"]).get_adapter(row["qa_type"]).parse_prediction("")
        self.assertEqual(status, "EMPTY")
        self.assertIsNone(parsed)

    def test_duplicate_selected_indices_detectable(self):
        row = build_gt_free_row(sample(n=40), new_data_root=None)
        record = build_selector_record(row, "local", [1, 1, 2], [1, 1, 2], "commit", selector_cfg())
        selected = record["selected_original_positions_chronological"]
        self.assertNotEqual(len(set(selected)), len(selected))

    def test_rc_mandatory_input_audit(self):
        s = sample(qa_type="region_caption_gemini", n=3, dataset_name="EgoSurgery")
        s["is_RC"] = True
        s["RC_info"] = {"start_frame": "/frames/0001.jpg", "start_frame_bbox": [1, 2, 3, 4]}
        row = build_gt_free_row(s, new_data_root=None)
        audit = rc_compatibility_audit([row])
        self.assertEqual(audit["status"], "RC_SELECTOR_ADAPTATION_REQUIRED")
        self.assertTrue(audit["mandatory_region_frame_schema"])

    def test_frame_count_audit(self):
        rows = [build_gt_free_row(sample(n=n), i, new_data_root=None) for i, n in enumerate([10, 32, 40])]
        audit = frame_count_audit(rows)
        self.assertEqual(audit["n_lt_32"], 1)
        self.assertEqual(audit["n_eq_32"], 1)
        self.assertEqual(audit["n_gt_32"], 1)

    def test_query_identifier_official_parser(self):
        self.assertEqual(parse_query_type('{"isGlobal": true}'), "global")
        self.assertEqual(parse_query_type('{"isGlobal": "false"}'), "local")

    def test_reward_official_parser(self):
        self.assertEqual(parse_reward('{"reward": 42}'), 42.0)

    def test_jsonl_resume_rows_are_read_once(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "x.jsonl"
            write_jsonl(path, [{"sample_id": "a"}, {"sample_id": "b"}])
            self.assertEqual([row["sample_id"] for row in read_jsonl(path)], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
