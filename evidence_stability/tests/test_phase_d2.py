import argparse
import contextlib
import io
import importlib.util
import tempfile
import unittest
from pathlib import Path

from evidence_stability.cache import make_intervention_probe_cache_key, stable_hash
from evidence_stability.phase_d2 import (
    assert_no_forbidden_model_fields,
    assert_raw_result_gt_free,
    deterministic_smoke_select,
    is_forbidden_field_name,
    logical_relative_frame_path,
    project_intervention_for_model,
    projection_path_audit,
    remap_projection_frame_paths,
    runtime_frame_path,
    sanitize_processor_metadata,
    select_generation_valid_interventions,
    smoke_selection_artifact,
)
from evidence_stability.prompts import PROMPT_VERSION, build_evidence_presence_prompt, parse_yes_no
from evidence_stability.utils import write_jsonl


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "06_probe_interventions.py"
SPEC = importlib.util.spec_from_file_location("phase_d2_probe_script", SCRIPT_PATH)
phase_d2_script = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(phase_d2_script)


def make_row(
    idx: int,
    *,
    generation_valid: bool = True,
    strict_valid: bool = False,
    label: str = "TRUE_SUPPORT",
    target_field: str = "action",
    intervention_type: str = "SHIFT_LEFT_2",
    dataset_name: str = "AVOS",
    n_frames: int = 16,
    duplicate: bool = False,
) -> dict:
    frames = [f"/tmp/frame_{idx}_{i}.jpg" for i in range(n_frames)]
    if duplicate and n_frames > 2:
        frames[2] = frames[1]
    family = "SHIFT"
    if intervention_type.startswith("RESAMPLE"):
        family = "RESAMPLE"
    if intervention_type.startswith("CONTEXT"):
        family = "CONTEXT"
    return {
        "qa_id": f"{idx:04d}::clip-{idx // 2}",
        "clip_id": f"clip-{idx // 2}",
        "window_id": f"{idx:04d}::clip-{idx // 2}::pos0000-0015",
        "intervention_id": f"{idx:04d}::clip-{idx // 2}::pos0000-0015::{intervention_type}",
        "dataset_name": dataset_name,
        "target_action": "suturing" if target_field == "action" else "dissection phase",
        "target_field": target_field,
        "intervention_family": family,
        "intervention_type": intervention_type,
        "intervened_frame_paths": frames,
        "intervened_n_frames": n_frames,
        "intervened_n_unique_frames": len(set(frames)),
        "generation_valid": generation_valid,
        "strict_valid": strict_valid,
        "any_gt_valid": strict_valid,
        "validity_reason": "STRICT_VALID_STRONG_GT_ALIGNED" if strict_valid else "GT_CONTAMINATED",
        "original_candidate_label": label,
        "original_gt_alignment_class": "STRONG_GT_ALIGNED",
        "original_evidence_density": 1.0,
        "original_gt_evidence_recall": 1.0,
        "original_n_gt_visible": 16,
    }


class PhaseD2Tests(unittest.TestCase):
    def test_existing_result_cache_index_rejects_duplicate_intervention_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "results.jsonl"
            write_jsonl(
                path,
                [
                    {"intervention_id": "a", "cache_key": "key-a"},
                    {"intervention_id": "a", "cache_key": "key-b"},
                ],
            )
            with self.assertRaises(RuntimeError):
                phase_d2_script.load_existing_result_cache_index(path)

    def test_existing_result_cache_index_preserves_unique_records(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "results.jsonl"
            write_jsonl(
                path,
                [
                    {"intervention_id": "a", "cache_key": "key-a"},
                    {"intervention_id": "b", "cache_key": "key-b"},
                ],
            )
            self.assertEqual(
                phase_d2_script.load_existing_result_cache_index(path),
                {"a": "key-a", "b": "key-b"},
            )

    def test_generation_valid_only_filter_ignores_strict_valid(self):
        rows = [
            make_row(1, generation_valid=True, strict_valid=False),
            make_row(2, generation_valid=False, strict_valid=True),
        ]
        selected = select_generation_valid_interventions(rows)
        self.assertEqual([r["intervention_id"] for r in selected], [rows[0]["intervention_id"]])

    def test_projection_strips_gt_and_label_fields(self):
        row = make_row(1, strict_valid=True, label="SPURIOUS_SUPPORT", n_frames=8)
        projection = project_intervention_for_model(row)
        assert_no_forbidden_model_fields(projection, "test projection")
        self.assertNotIn("strict_valid", projection)
        self.assertNotIn("original_candidate_label", projection)
        self.assertEqual(projection["ordered_frame_paths"], row["intervened_frame_paths"])
        self.assertEqual(projection["n_frames"], 8)

    def test_gt_field_detection_is_token_aware(self):
        allowed = [
            "input_token_length",
            "length",
            "image_grid_thw",
            "image_grid_thw_rows",
            "pixel_values_shape",
            "input_frame_count",
            "expected_frame_count",
            "attention_mask",
        ]
        forbidden = [
            "gt",
            "gt_alignment_class",
            "original_gt_alignment_class",
            "n_gt_visible",
            "candidate_label",
            "strict_valid",
            "gt_evidence_recall",
        ]
        for field in allowed:
            self.assertFalse(is_forbidden_field_name(field), field)
            assert_no_forbidden_model_fields({field: 1}, f"allowed {field}")
        for field in forbidden:
            self.assertTrue(is_forbidden_field_name(field), field)
            with self.assertRaises(RuntimeError):
                assert_no_forbidden_model_fields({field: 1}, f"forbidden {field}")

    def test_processor_metadata_allows_engineering_length_fields(self):
        metadata = sanitize_processor_metadata(
            {
                "input_token_length": 123,
                "input_ids_shape": [1, 123],
                "image_grid_thw_rows": 16,
                "pixel_values_shape": [4096, 1176],
                "input_keys": ["input_ids", "attention_mask", "image_grid_thw"],
                "dropped_private_field": "not copied",
            },
            expected_frame_count=16,
        )
        self.assertEqual(metadata["input_token_length"], 123)
        self.assertEqual(metadata["expected_frame_count"], 16)
        self.assertNotIn("dropped_private_field", metadata)

    def test_projection_preserves_duplicate_ordered_frames(self):
        row = make_row(1, n_frames=16, duplicate=True)
        projection = project_intervention_for_model(row)
        self.assertEqual(projection["ordered_frame_paths"], row["intervened_frame_paths"])
        self.assertLess(projection["n_unique_frames"], projection["n_frames"])

    def test_frame_root_remap_preserves_order_and_duplicates(self):
        row = make_row(1, n_frames=4, duplicate=True)
        row["intervened_frame_paths"] = [
            "/root/data/AVOS/frames_15fps/v/0001.jpg",
            "/root/data/AVOS/frames_15fps/v/0002.jpg",
            "/root/data/AVOS/frames_15fps/v/0002.jpg",
            "/root/data/AVOS/frames_15fps/v/0003.jpg",
        ]
        row["intervened_n_unique_frames"] = 3
        projection = project_intervention_for_model(row)
        remapped = phase_d2_script.resolve_projection_frame_paths(
            projection,
            "/mnt/hdd3/huihui/hh_datas/MedVidU/valdata",
            "/root/data",
        )
        self.assertEqual(
            remapped["ordered_frame_paths"],
            [
                "/mnt/hdd3/huihui/hh_datas/MedVidU/valdata/AVOS/frames_15fps/v/0001.jpg",
                "/mnt/hdd3/huihui/hh_datas/MedVidU/valdata/AVOS/frames_15fps/v/0002.jpg",
                "/mnt/hdd3/huihui/hh_datas/MedVidU/valdata/AVOS/frames_15fps/v/0002.jpg",
                "/mnt/hdd3/huihui/hh_datas/MedVidU/valdata/AVOS/frames_15fps/v/0003.jpg",
            ],
        )
        self.assertEqual(remapped["n_frames"], 4)
        self.assertEqual(remapped["n_unique_frames"], 3)

    def test_old_hdd3_path_remaps_to_runtime_hdd_root(self):
        frozen = "/mnt/hdd3/huihui/hh_datas/MedVidU/valdata/AVOS/a.jpg"
        runtime_root = "/mnt/hdd/huihui/hh_datas/MedVidU/valdata"
        self.assertEqual(logical_relative_frame_path(frozen, runtime_root), "AVOS/a.jpg")
        self.assertEqual(
            runtime_frame_path(frozen, runtime_root),
            "/mnt/hdd/huihui/hh_datas/MedVidU/valdata/AVOS/a.jpg",
        )

    def test_cholec_camma_path_remaps_by_dataset_relative_suffix(self):
        frozen = "/mnt/hdd3/huihui/hh_datas/MedVidU/valdata/CholecT50/CAMMA_data/video/000001.jpg"
        runtime_root = "/mnt/hdd/huihui/hh_datas/MedVidU/valdata"
        self.assertEqual(
            runtime_frame_path(frozen, runtime_root),
            "/mnt/hdd/huihui/hh_datas/MedVidU/valdata/CholecT50/CAMMA_data/video/000001.jpg",
        )

    def test_runtime_frame_remap_preserves_order_duplicates_and_logical_identity(self):
        projection = project_intervention_for_model(make_row(1, n_frames=4))
        projection["ordered_frame_paths"] = [
            "/mnt/hdd3/huihui/hh_datas/MedVidU/valdata/AVOS/v/0001.jpg",
            "/mnt/hdd3/huihui/hh_datas/MedVidU/valdata/AVOS/v/0002.jpg",
            "/mnt/hdd3/huihui/hh_datas/MedVidU/valdata/AVOS/v/0002.jpg",
            "/mnt/hdd3/huihui/hh_datas/MedVidU/valdata/AVOS/v/0003.jpg",
        ]
        projection["n_frames"] = 4
        remapped = remap_projection_frame_paths(
            projection,
            "/mnt/hdd/huihui/hh_datas/MedVidU/valdata",
        )
        self.assertEqual(
            remapped["logical_relative_frame_paths"],
            ["AVOS/v/0001.jpg", "AVOS/v/0002.jpg", "AVOS/v/0002.jpg", "AVOS/v/0003.jpg"],
        )
        self.assertEqual(remapped["frozen_ordered_frame_paths"], projection["ordered_frame_paths"])
        self.assertEqual(
            remapped["ordered_frame_paths"],
            [
                "/mnt/hdd/huihui/hh_datas/MedVidU/valdata/AVOS/v/0001.jpg",
                "/mnt/hdd/huihui/hh_datas/MedVidU/valdata/AVOS/v/0002.jpg",
                "/mnt/hdd/huihui/hh_datas/MedVidU/valdata/AVOS/v/0002.jpg",
                "/mnt/hdd/huihui/hh_datas/MedVidU/valdata/AVOS/v/0003.jpg",
            ],
        )
        self.assertEqual(remapped["n_unique_frames"], 3)

    def test_runtime_frame_remap_rejects_path_traversal(self):
        with self.assertRaises(ValueError):
            logical_relative_frame_path("../AVOS/a.jpg", "/mnt/hdd/huihui/hh_datas/MedVidU/valdata")
        with self.assertRaises(ValueError):
            logical_relative_frame_path(
                "/mnt/hdd3/huihui/hh_datas/MedVidU/valdata/AVOS/../a.jpg",
                "/mnt/hdd/huihui/hh_datas/MedVidU/valdata",
            )

    def test_projection_path_audit_reports_missing_runtime_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            projection = project_intervention_for_model(make_row(1, n_frames=1))
            projection["ordered_frame_paths"] = [
                "/mnt/hdd3/huihui/hh_datas/MedVidU/valdata/AVOS/missing.jpg",
            ]
            projection["n_frames"] = 1
            remapped = remap_projection_frame_paths(projection, tmp)
            audit = projection_path_audit([remapped])
            self.assertEqual(audit["n_missing_frame_references"], 1)
            self.assertEqual(audit["n_interventions_with_missing_frames"], 1)
            self.assertEqual(audit["runtime_path_previews"][0]["logical_first_frame"], "AVOS/missing.jpg")

    def test_projection_path_audit_accepts_existing_runtime_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime_root = Path(tmp)
            frame = runtime_root / "AVOS" / "present.jpg"
            frame.parent.mkdir(parents=True)
            frame.write_text("not an actual image", encoding="utf-8")
            projection = project_intervention_for_model(make_row(1, n_frames=1))
            projection["ordered_frame_paths"] = [
                "/mnt/hdd3/huihui/hh_datas/MedVidU/valdata/AVOS/present.jpg",
            ]
            projection["n_frames"] = 1
            remapped = remap_projection_frame_paths(projection, str(runtime_root))
            audit = projection_path_audit([remapped])
            self.assertEqual(audit["n_missing_frame_references"], 0)

    def test_smoke_selection_has_no_origin_or_strict_valid_artifact(self):
        rows = [
            make_row(1, label="TRUE_SUPPORT", target_field="action", intervention_type="RESAMPLE_50", n_frames=8),
            make_row(2, label="SPURIOUS_SUPPORT", target_field="phase", intervention_type="RESAMPLE_75", n_frames=12),
            make_row(3, label="TRUE_SUPPORT", target_field="action", intervention_type="SHIFT_LEFT_2", n_frames=16, dataset_name="NurViD"),
            make_row(4, label="SPURIOUS_SUPPORT", target_field="phase", intervention_type="CONTEXT_1P5X", n_frames=24, dataset_name="EgoSurgery"),
        ]
        selected = deterministic_smoke_select(rows, n=4, seed=42)
        artifact = smoke_selection_artifact(selected, seed=42)
        for item in artifact["interventions"]:
            self.assertNotIn("strict_valid", item)
            self.assertNotIn("original_candidate_label", item)
            self.assertNotIn("candidate_label", item)

    def test_prompt_hash_matches_phase_c_builder_and_parser(self):
        prompt_a = build_evidence_presence_prompt("suturing")
        prompt_b = build_evidence_presence_prompt("suturing")
        self.assertEqual(stable_hash({"prompt": prompt_a}), stable_hash({"prompt": prompt_b}))
        self.assertEqual(parse_yes_no("YES"), "YES")
        self.assertEqual(parse_yes_no("No."), "NO")
        self.assertEqual(parse_yes_no("YES because visible"), "INVALID")

    def test_no_original_phase_c_response_in_prompt(self):
        projection = project_intervention_for_model(make_row(1))
        prompt, _ = phase_d2_script.prompt_for_projection(projection)
        self.assertIn("suturing", prompt)
        self.assertNotIn("YES because", prompt)
        self.assertNotIn("TRUE_SUPPORT", prompt)
        self.assertNotIn("intervention", prompt.lower())

    def test_intervention_cache_key_stability_and_collision_separation(self):
        row = project_intervention_for_model(make_row(1, intervention_type="RESAMPLE_50", n_frames=8))
        prompt_hash = stable_hash({"prompt": build_evidence_presence_prompt(row["target_action"])})
        base = ("modelhash", prompt_hash, row["qa_id"], row["window_id"], row["intervention_id"], row["ordered_frame_paths"], row["n_frames"], {"max_new_tokens": 8})
        key_a = make_intervention_probe_cache_key(*base)
        key_b = make_intervention_probe_cache_key(*base)
        self.assertEqual(key_a, key_b)

        different_type = dict(row)
        different_type["intervention_id"] = different_type["intervention_id"].replace("RESAMPLE_50", "RESAMPLE_75")
        different_type["n_frames"] = 12
        different_type["ordered_frame_paths"] = different_type["ordered_frame_paths"] + ["extra.jpg"] * 4
        key_c = make_intervention_probe_cache_key(
            "modelhash",
            prompt_hash,
            different_type["qa_id"],
            different_type["window_id"],
            different_type["intervention_id"],
            different_type["ordered_frame_paths"],
            different_type["n_frames"],
            {"max_new_tokens": 8},
        )
        self.assertNotEqual(key_a, key_c)

        shifted = dict(row)
        shifted["intervention_id"] = shifted["intervention_id"].replace("RESAMPLE_50", "SHIFT_RIGHT_2")
        shifted["ordered_frame_paths"] = list(reversed(shifted["ordered_frame_paths"]))
        key_d = make_intervention_probe_cache_key(
            "modelhash",
            prompt_hash,
            shifted["qa_id"],
            shifted["window_id"],
            shifted["intervention_id"],
            shifted["ordered_frame_paths"],
            shifted["n_frames"],
            {"max_new_tokens": 8},
        )
        self.assertNotEqual(key_a, key_d)

    def test_intervention_cache_key_uses_logical_frame_paths_across_machine_roots(self):
        projection = project_intervention_for_model(make_row(1, intervention_type="SHIFT_LEFT_2", n_frames=2))
        projection["ordered_frame_paths"] = [
            "/mnt/hdd3/huihui/hh_datas/MedVidU/valdata/AVOS/v/0001.jpg",
            "/mnt/hdd3/huihui/hh_datas/MedVidU/valdata/AVOS/v/0002.jpg",
        ]
        projection["n_frames"] = 2
        on_old_root = remap_projection_frame_paths(
            projection,
            "/mnt/hdd3/huihui/hh_datas/MedVidU/valdata",
        )
        on_new_root = remap_projection_frame_paths(
            projection,
            "/mnt/hdd/huihui/hh_datas/MedVidU/valdata",
        )
        model_hash = "modelhash"
        decoding_config = {"max_new_tokens": 8, "do_sample": False}
        keys_a, _ = phase_d2_script.cache_key_audit([on_old_root], model_hash, decoding_config)
        keys_b, _ = phase_d2_script.cache_key_audit([on_new_root], model_hash, decoding_config)
        self.assertNotEqual(on_old_root["ordered_frame_paths"], on_new_root["ordered_frame_paths"])
        self.assertEqual(on_old_root["logical_relative_frame_paths"], on_new_root["logical_relative_frame_paths"])
        self.assertEqual(keys_a[projection["intervention_id"]], keys_b[projection["intervention_id"]])

    def test_raw_result_schema_gt_free(self):
        record = {
            "qa_id": "0001::clip",
            "clip_id": "clip",
            "window_id": "0001::clip::pos0000-0015",
            "intervention_id": "0001::clip::pos0000-0015::SHIFT_LEFT_2",
            "dataset_name": "AVOS",
            "target_action": "suturing",
            "intervention_family": "SHIFT",
            "intervention_type": "SHIFT_LEFT_2",
            "prompt_hash": "abc",
            "decoding_config": {"do_sample": False, "max_new_tokens": 8},
            "ordered_frame_paths": ["a.jpg"],
            "n_frames": 1,
            "n_unique_frames": 1,
            "raw_response": "NO",
            "parsed_prediction": "NO",
            "model_name": "qwen3_vl_8b",
            "model_revision": "hash",
            "model_identity_hash": "hash",
            "cache_key": "cache",
            "created_at": "2026-09-03 00:00:00",
            "processor_metadata": {
                "input_token_length": 123,
                "image_grid_thw_rows": 1,
                "expected_frame_count": 1,
            },
        }
        assert_raw_result_gt_free(record)
        bad = dict(record)
        bad["candidate_label"] = "TRUE_SUPPORT"
        with self.assertRaises(RuntimeError):
            assert_raw_result_gt_free(bad)
        bad_extra = dict(record)
        bad_extra["model_fingerprint"] = {"model_class": "Qwen3VLForConditionalGeneration"}
        with self.assertRaises(RuntimeError):
            assert_raw_result_gt_free(bad_extra)

    def test_phase_c_consistency_uses_same_model_identity(self):
        args = argparse.Namespace(
            skip_phase_c_consistency=False,
            phase_c_probe_summary="",
            phase_c_probe_results="",
            prompt_version=PROMPT_VERSION,
        )
        with tempfile.TemporaryDirectory() as tmp:
            summary = Path(tmp) / "summary.json"
            summary.write_text(
                '{"prompt_version":"evidence_presence_v1","model_fingerprint":{"model_identity_hash":"h"},"decoding_config":{"do_sample":false,"max_new_tokens":8,"enable_thinking":false}}',
                encoding="utf-8",
            )
            args.phase_c_probe_summary = str(summary)
            checks = phase_d2_script.phase_c_consistency_check(
                args,
                {"model_identity_hash": "h"},
                {"do_sample": False, "max_new_tokens": 8, "enable_thinking": False},
                [project_intervention_for_model(make_row(1))],
            )
            self.assertTrue(checks["model_identity_hash_matches"])
            self.assertTrue(checks["prompt_version_matches"])
            self.assertTrue(checks["decoding_config_matches"])

    def test_phase_c_consistency_allows_runtime_hash_only_difference(self):
        args = argparse.Namespace(
            skip_phase_c_consistency=False,
            phase_c_probe_summary="",
            phase_c_probe_results="",
            prompt_version=PROMPT_VERSION,
        )
        old_fp = {
            "model_identity_hash": "old-hash",
            "model_path": "/mnt/hdd3/huihui/models/Qwen3-VL-8B-Instruct",
            "config_sha256": "config",
            "generation_config_sha256": "gen",
            "architectures": ["Qwen3VLForConditionalGeneration"],
            "model_type": "qwen3_vl",
            "processor_class": "Qwen3VLProcessor",
            "model_class": "Qwen3VLForConditionalGeneration",
            "dtype": "bfloat16",
            "do_sample": False,
            "max_new_tokens": 8,
            "enable_thinking": False,
            "processor_min_pixels": None,
            "processor_max_pixels": None,
            "device": "cuda:1",
            "torch_version": "2.6.0+cu124",
        }
        new_fp = dict(old_fp)
        new_fp["model_identity_hash"] = "new-hash"
        new_fp["device"] = "cuda:0"
        with tempfile.TemporaryDirectory() as tmp:
            summary = Path(tmp) / "summary.json"
            summary.write_text(
                '{"prompt_version":"evidence_presence_v1","model_fingerprint":'
                + phase_d2_script.json.dumps(old_fp)
                + ',"decoding_config":{"do_sample":false,"max_new_tokens":8,"enable_thinking":false}}',
                encoding="utf-8",
            )
            args.phase_c_probe_summary = str(summary)
            checks = phase_d2_script.phase_c_consistency_check(
                args,
                new_fp,
                {"do_sample": False, "max_new_tokens": 8, "enable_thinking": False},
                [project_intervention_for_model(make_row(1))],
            )
            self.assertFalse(checks["model_identity_hash_matches"])
            self.assertTrue(checks["core_model_fingerprint_matches"])
            phase_d2_script.assert_phase_c_consistency(checks)

    def test_phase_c_consistency_rejects_scientific_environment_difference(self):
        args = argparse.Namespace(
            skip_phase_c_consistency=False,
            phase_c_probe_summary="",
            phase_c_probe_results="",
            prompt_version=PROMPT_VERSION,
        )
        old_fp = {
            "model_identity_hash": "old-hash",
            "model_path": "/mnt/hdd3/huihui/models/Qwen3-VL-8B-Instruct",
            "config_sha256": "config",
            "generation_config_sha256": "gen",
            "architectures": ["Qwen3VLForConditionalGeneration"],
            "model_type": "qwen3_vl",
            "processor_class": "Qwen3VLProcessor",
            "model_class": "Qwen3VLForConditionalGeneration",
            "dtype": "bfloat16",
            "do_sample": False,
            "max_new_tokens": 8,
            "enable_thinking": False,
            "processor_min_pixels": None,
            "processor_max_pixels": None,
            "transformers_version": "5.16.1",
            "torch_version": "2.6.0+cu124",
        }
        new_fp = dict(old_fp)
        new_fp["model_identity_hash"] = "new-hash"
        new_fp["transformers_version"] = "4.57.1"
        new_fp["torch_version"] = "2.9.1+cu128"
        with tempfile.TemporaryDirectory() as tmp:
            summary = Path(tmp) / "summary.json"
            summary.write_text(
                '{"prompt_version":"evidence_presence_v1","model_fingerprint":'
                + phase_d2_script.json.dumps(old_fp)
                + ',"decoding_config":{"do_sample":false,"max_new_tokens":8,"enable_thinking":false}}',
                encoding="utf-8",
            )
            args.phase_c_probe_summary = str(summary)
            checks = phase_d2_script.phase_c_consistency_check(
                args,
                new_fp,
                {"do_sample": False, "max_new_tokens": 8, "enable_thinking": False},
                [project_intervention_for_model(make_row(1))],
            )
            self.assertTrue(checks["core_model_fingerprint_matches"])
            self.assertFalse(checks["scientific_environment_matches"])
            self.assertTrue(checks["model_fingerprint_differences"]["transformers_version"]["scientific_environment_field"])
            self.assertTrue(checks["model_fingerprint_differences"]["torch_version"]["scientific_environment_field"])
            with self.assertRaises(RuntimeError):
                phase_d2_script.assert_phase_c_consistency(checks)
            phase_d2_script.assert_phase_c_consistency(checks, allow_scientific_environment_mismatch=True)

    def test_frozen_qwen_model_path_allows_relocated_model_when_loaded_path_matches_arg(self):
        args = argparse.Namespace(
            model_backend="qwen3_vl",
            model_path="/new/server/path/Qwen3-VL-8B-Instruct/",
        )
        fingerprint = {
            "model_path": "/new/server/path/Qwen3-VL-8B-Instruct",
            "architectures": ["Qwen3VLForConditionalGeneration"],
            "model_class": "Qwen3VLForConditionalGeneration",
            "processor_class": "Qwen3VLProcessor",
            "dtype": "bfloat16",
            "do_sample": False,
            "max_new_tokens": 8,
            "enable_thinking": False,
        }
        checks = phase_d2_script.fingerprint_is_frozen_qwen(args, fingerprint)
        self.assertTrue(checks["model_path_argument_matches_loaded_model"])

    def test_phase_c_consistency_allows_relocated_model_path_when_content_matches(self):
        args = argparse.Namespace(
            skip_phase_c_consistency=False,
            phase_c_probe_summary="",
            phase_c_probe_results="",
            prompt_version=PROMPT_VERSION,
        )
        old_fp = {
            "model_identity_hash": "old-hash",
            "model_path": "/mnt/hdd3/huihui/models/Qwen3-VL-8B-Instruct",
            "config_sha256": "config",
            "generation_config_sha256": "gen",
            "architectures": ["Qwen3VLForConditionalGeneration"],
            "model_type": "qwen3_vl",
            "processor_class": "Qwen3VLProcessor",
            "model_class": "Qwen3VLForConditionalGeneration",
            "dtype": "bfloat16",
            "do_sample": False,
            "max_new_tokens": 8,
            "enable_thinking": False,
            "processor_min_pixels": None,
            "processor_max_pixels": None,
        }
        new_fp = dict(old_fp)
        new_fp["model_identity_hash"] = "new-hash"
        new_fp["model_path"] = "/new/server/path/Qwen3-VL-8B-Instruct"
        with tempfile.TemporaryDirectory() as tmp:
            summary = Path(tmp) / "summary.json"
            summary.write_text(
                '{"prompt_version":"evidence_presence_v1","model_fingerprint":'
                + phase_d2_script.json.dumps(old_fp)
                + ',"decoding_config":{"do_sample":false,"max_new_tokens":8,"enable_thinking":false}}',
                encoding="utf-8",
            )
            args.phase_c_probe_summary = str(summary)
            checks = phase_d2_script.phase_c_consistency_check(
                args,
                new_fp,
                {"do_sample": False, "max_new_tokens": 8, "enable_thinking": False},
                [project_intervention_for_model(make_row(1))],
            )
            self.assertFalse(checks["model_identity_hash_matches"])
            self.assertTrue(checks["core_model_fingerprint_matches"])
            self.assertIn("model_path", checks["model_fingerprint_differences"])
            phase_d2_script.assert_phase_c_consistency(checks)

    def test_phase_c_consistency_rejects_core_model_difference(self):
        args = argparse.Namespace(
            skip_phase_c_consistency=False,
            phase_c_probe_summary="",
            phase_c_probe_results="",
            prompt_version=PROMPT_VERSION,
        )
        old_fp = {
            "model_identity_hash": "old-hash",
            "model_path": "/mnt/hdd3/huihui/models/Qwen3-VL-8B-Instruct",
            "config_sha256": "config-a",
            "generation_config_sha256": "gen",
            "architectures": ["Qwen3VLForConditionalGeneration"],
            "model_type": "qwen3_vl",
            "processor_class": "Qwen3VLProcessor",
            "model_class": "Qwen3VLForConditionalGeneration",
            "dtype": "bfloat16",
            "do_sample": False,
            "max_new_tokens": 8,
            "enable_thinking": False,
            "processor_min_pixels": None,
            "processor_max_pixels": None,
        }
        new_fp = dict(old_fp)
        new_fp["model_identity_hash"] = "new-hash"
        new_fp["config_sha256"] = "config-b"
        with tempfile.TemporaryDirectory() as tmp:
            summary = Path(tmp) / "summary.json"
            summary.write_text(
                '{"prompt_version":"evidence_presence_v1","model_fingerprint":'
                + phase_d2_script.json.dumps(old_fp)
                + ',"decoding_config":{"do_sample":false,"max_new_tokens":8,"enable_thinking":false}}',
                encoding="utf-8",
            )
            args.phase_c_probe_summary = str(summary)
            checks = phase_d2_script.phase_c_consistency_check(
                args,
                new_fp,
                {"do_sample": False, "max_new_tokens": 8, "enable_thinking": False},
                [project_intervention_for_model(make_row(1))],
            )
            self.assertFalse(checks["core_model_fingerprint_matches"])
            with self.assertRaises(RuntimeError):
                phase_d2_script.assert_phase_c_consistency(checks)

    def test_dry_run_resume_skips_cached(self):
        rows = [
            make_row(1, label="TRUE_SUPPORT", target_field="action", intervention_type="RESAMPLE_50", n_frames=8),
            make_row(2, label="SPURIOUS_SUPPORT", target_field="phase", intervention_type="RESAMPLE_75", n_frames=12),
            make_row(3, label="TRUE_SUPPORT", target_field="action", intervention_type="SHIFT_LEFT_2", n_frames=16, dataset_name="NurViD"),
            make_row(4, label="SPURIOUS_SUPPORT", target_field="phase", intervention_type="CONTEXT_1P5X", n_frames=24, dataset_name="EgoSurgery"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            interventions = Path(tmp) / "interventions.jsonl"
            results = Path(tmp) / "results.jsonl"
            out_dir = Path(tmp) / "out"
            write_jsonl(interventions, rows)
            projection = project_intervention_for_model(rows[0])
            prompt_hash = stable_hash({"prompt": build_evidence_presence_prompt(projection["target_action"])})
            cache_key = make_intervention_probe_cache_key(
                "v1",
                prompt_hash,
                projection["qa_id"],
                projection["window_id"],
                projection["intervention_id"],
                projection["ordered_frame_paths"],
                projection["n_frames"],
                {"deterministic_dummy": True},
            )
            write_jsonl(
                results,
                [{"intervention_id": projection["intervention_id"], "cache_key": cache_key}],
            )
            argv = [
                "06_probe_interventions.py",
                "--interventions",
                str(interventions),
                "--output_dir",
                str(out_dir),
                "--probe_results",
                str(results),
                "--model_backend",
                "dummy",
                "--expected_generation_valid",
                "-1",
                "--skip_phase_c_consistency",
                "--dry_run",
                "--max_interventions",
                "4",
            ]
            old_argv = phase_d2_script.sys.argv
            try:
                phase_d2_script.sys.argv = argv
                with contextlib.redirect_stdout(io.StringIO()):
                    phase_d2_script.main()
            finally:
                phase_d2_script.sys.argv = old_argv
            summary = (out_dir / "phase_d2_smoke_summary.json").read_text(encoding="utf-8")
            self.assertIn('"skipped_cached_count": 1', summary)
            self.assertIn('"dry_run_count": 3', summary)

    def test_full_mode_uses_full_summary_name_and_generation_valid_only(self):
        rows = [
            make_row(1, generation_valid=True, strict_valid=False, n_frames=8),
            make_row(2, generation_valid=True, strict_valid=True, n_frames=12),
            make_row(3, generation_valid=False, strict_valid=True, n_frames=16),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            interventions = Path(tmp) / "interventions.jsonl"
            out_dir = Path(tmp) / "out"
            write_jsonl(interventions, rows)
            argv = [
                "06_probe_interventions.py",
                "--interventions",
                str(interventions),
                "--output_dir",
                str(out_dir),
                "--model_backend",
                "dummy",
                "--expected_generation_valid",
                "-1",
                "--skip_phase_c_consistency",
                "--dry_run",
                "--run_all_generation_valid",
            ]
            old_argv = phase_d2_script.sys.argv
            try:
                phase_d2_script.sys.argv = argv
                with contextlib.redirect_stdout(io.StringIO()):
                    phase_d2_script.main()
            finally:
                phase_d2_script.sys.argv = old_argv
            self.assertTrue((out_dir / "phase_d2_full_summary.json").exists())
            summary = phase_d2_script.json.loads((out_dir / "phase_d2_full_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["n_total_interventions"], 3)
            self.assertEqual(summary["n_generation_valid"], 2)
            self.assertEqual(summary["n_generation_invalid"], 1)
            self.assertEqual(summary["n_selected_interventions"], 2)
            self.assertEqual(summary["dry_run_count"], 2)

    def test_full_mode_missing_frames_stop_before_inference(self):
        rows = [make_row(1, generation_valid=True, strict_valid=False, n_frames=8)]
        rows[0]["intervened_frame_paths"] = [
            f"/mnt/hdd3/huihui/hh_datas/MedVidU/valdata/AVOS/missing_{i}.jpg"
            for i in range(8)
        ]
        rows[0]["intervened_n_frames"] = 8
        rows[0]["intervened_n_unique_frames"] = 8
        with tempfile.TemporaryDirectory() as tmp:
            interventions = Path(tmp) / "interventions.jsonl"
            out_dir = Path(tmp) / "out"
            runtime_root = Path(tmp) / "runtime_valdata"
            runtime_root.mkdir()
            write_jsonl(interventions, rows)
            argv = [
                "06_probe_interventions.py",
                "--interventions",
                str(interventions),
                "--output_dir",
                str(out_dir),
                "--model_backend",
                "dummy",
                "--expected_generation_valid",
                "-1",
                "--skip_phase_c_consistency",
                "--run_all_generation_valid",
                "--frame_root",
                str(runtime_root),
            ]
            old_argv = phase_d2_script.sys.argv
            try:
                phase_d2_script.sys.argv = argv
                with self.assertRaises(RuntimeError):
                    with contextlib.redirect_stdout(io.StringIO()):
                        phase_d2_script.main()
            finally:
                phase_d2_script.sys.argv = old_argv
            summary = phase_d2_script.json.loads((out_dir / "phase_d2_full_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["path_audit"]["n_missing_frame_references"], 8)
            self.assertFalse(summary["real_model_inference_executed"])
            self.assertIn("Missing frame", summary["stop_reason"])

    def test_reuse_selection_from_preserves_exact_ids(self):
        rows = [
            make_row(1, label="TRUE_SUPPORT", target_field="action", intervention_type="RESAMPLE_50", n_frames=8),
            make_row(2, label="SPURIOUS_SUPPORT", target_field="phase", intervention_type="RESAMPLE_75", n_frames=12),
            make_row(3, label="TRUE_SUPPORT", target_field="action", intervention_type="SHIFT_LEFT_2", n_frames=16, dataset_name="NurViD"),
            make_row(4, label="SPURIOUS_SUPPORT", target_field="phase", intervention_type="CONTEXT_1P5X", n_frames=24, dataset_name="EgoSurgery"),
        ]
        artifact = smoke_selection_artifact([rows[2], rows[0], rows[3], rows[1]], seed=42)
        selected = phase_d2_script.select_from_smoke_artifact(rows, artifact)
        self.assertEqual(
            [row["intervention_id"] for row in selected],
            [rows[2]["intervention_id"], rows[0]["intervention_id"], rows[3]["intervention_id"], rows[1]["intervention_id"]],
        )


if __name__ == "__main__":
    unittest.main()
