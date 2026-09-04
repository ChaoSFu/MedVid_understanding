import contextlib
import io
import importlib.util
import math
import tempfile
import unittest
from pathlib import Path

from evidence_stability.join import d2_final_prediction_audit, join_intervention_results
from evidence_stability.stability import (
    candidate_stability_rows,
    include_in_primary_denominator,
    missingness_rows,
    phase_e_summary,
    qa_stability_rows,
    retention_matrix_rows,
    stability_by_group_rows,
)
from evidence_stability.utils import read_json, write_jsonl


JOIN_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "07_join_intervention_results.py"
JOIN_SPEC = importlib.util.spec_from_file_location("phase_d3_join_script", JOIN_SCRIPT_PATH)
phase_d3_join_script = importlib.util.module_from_spec(JOIN_SPEC)
assert JOIN_SPEC and JOIN_SPEC.loader
JOIN_SPEC.loader.exec_module(phase_d3_join_script)

STABILITY_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "08_compute_stability.py"
STABILITY_SPEC = importlib.util.spec_from_file_location("phase_e_stability_script", STABILITY_SCRIPT_PATH)
phase_e_script = importlib.util.module_from_spec(STABILITY_SPEC)
assert STABILITY_SPEC and STABILITY_SPEC.loader
STABILITY_SPEC.loader.exec_module(phase_e_script)


TYPES = ["SHIFT_LEFT_2", "SHIFT_RIGHT_2", "RESAMPLE_75", "RESAMPLE_50", "CONTEXT_1P5X"]
FAMILY = {
    "SHIFT_LEFT_2": "SHIFT",
    "SHIFT_RIGHT_2": "SHIFT",
    "RESAMPLE_75": "RESAMPLE",
    "RESAMPLE_50": "RESAMPLE",
    "CONTEXT_1P5X": "CONTEXT",
}


def is_nan(value):
    return isinstance(value, float) and math.isnan(value)


def manifest_row(
    qa_id,
    clip_id,
    window_id,
    label,
    intervention_type,
    *,
    generation_valid=True,
    strict_valid=True,
    any_gt_valid=True,
    dataset_name="AVOS",
    target_field="action",
    target_action="suturing",
    gt_duration_total=6.0,
):
    return {
        "qa_id": qa_id,
        "clip_id": clip_id,
        "window_id": window_id,
        "intervention_id": f"{window_id}::{intervention_type}",
        "dataset_name": dataset_name,
        "target_action": target_action,
        "target_field": target_field,
        "original_candidate_label": label,
        "gt_duration_total": gt_duration_total,
        "gt_duration_bin": None,
        "intervention_family": FAMILY[intervention_type],
        "intervention_type": intervention_type,
        "generation_valid": generation_valid,
        "intervened_n_frames": {"RESAMPLE_75": 12, "RESAMPLE_50": 8, "CONTEXT_1P5X": 24}.get(intervention_type, 16),
        "intervened_n_unique_frames": {"RESAMPLE_75": 12, "RESAMPLE_50": 8, "CONTEXT_1P5X": 24}.get(intervention_type, 16),
        "boundary_asymmetric": False,
        "original_gt_alignment_class": "STRONG_GT_ALIGNED" if label == "TRUE_SUPPORT" else "NO_GT_OVERLAP",
        "intervention_gt_alignment_class": "STRONG_GT_ALIGNED" if strict_valid and label == "TRUE_SUPPORT" else "NO_GT_OVERLAP",
        "strict_valid": strict_valid,
        "any_gt_valid": any_gt_valid,
        "validity_reason": "STRICT_VALID" if strict_valid else "GT_CONTAMINATED",
        "original_evidence_density": 1.0 if label == "TRUE_SUPPORT" else 0.0,
        "intervention_evidence_density": 1.0 if strict_valid and label == "TRUE_SUPPORT" else 0.0,
        "original_gt_evidence_recall": 1.0 if label == "TRUE_SUPPORT" else 0.0,
        "intervention_gt_evidence_recall": 1.0 if strict_valid and label == "TRUE_SUPPORT" else 0.0,
        "original_n_gt_visible": 16 if label == "TRUE_SUPPORT" else 0,
        "intervention_n_gt_visible": 16 if strict_valid and label == "TRUE_SUPPORT" else 0,
    }


def probe_row(row, parsed):
    return {
        "qa_id": row["qa_id"],
        "clip_id": row["clip_id"],
        "window_id": row["window_id"],
        "intervention_id": row["intervention_id"],
        "dataset_name": row["dataset_name"],
        "target_action": row["target_action"],
        "intervention_family": row["intervention_family"],
        "intervention_type": row["intervention_type"],
        "model_name": "qwen3_vl_8b",
        "model_revision": "model-rev",
        "model_identity_hash": "model-hash",
        "prompt_version": "evidence_presence_v1",
        "prompt_hash": "prompt-hash",
        "decoding_config": {"do_sample": False, "max_new_tokens": 8, "enable_thinking": False},
        "ordered_frame_paths": ["a.jpg"] * row["intervened_n_frames"],
        "n_frames": row["intervened_n_frames"],
        "n_unique_frames": 1,
        "raw_response": parsed if parsed in {"YES", "NO"} else "",
        "parsed_prediction": parsed,
        "cache_key": f"cache::{row['intervention_id']}",
        "created_at": "2026-09-03 00:00:00",
        "processor_metadata": {
            "input_token_length": 100,
            "image_grid_thw_rows": row["intervened_n_frames"],
            "expected_frame_count": row["intervened_n_frames"],
        },
    }


def make_candidate(qa_id, window_id, label, predictions, strict_valid=None, generation_valid=None, **kwargs):
    rows = []
    probes = []
    strict_valid = strict_valid or {}
    generation_valid = generation_valid or {}
    for intervention_type in TYPES:
        row = manifest_row(
            qa_id,
            f"clip-{qa_id}",
            window_id,
            label,
            intervention_type,
            strict_valid=strict_valid.get(intervention_type, True),
            any_gt_valid=strict_valid.get(intervention_type, True),
            generation_valid=generation_valid.get(intervention_type, True),
            **kwargs,
        )
        rows.append(row)
        if row["generation_valid"] and intervention_type in predictions:
            probes.append(probe_row(row, predictions[intervention_type]))
    return rows, probes


def synthetic_joined():
    manifest = []
    probes = []
    rows, pred = make_candidate(
        "q1",
        "q1::pos0000-0015",
        "TRUE_SUPPORT",
        {
            "SHIFT_LEFT_2": "YES",
            "SHIFT_RIGHT_2": "YES",
            "RESAMPLE_75": "YES",
            "RESAMPLE_50": "NO",
            "CONTEXT_1P5X": "NO",
        },
        strict_valid={"CONTEXT_1P5X": False},
    )
    manifest += rows
    probes += pred
    rows, pred = make_candidate(
        "q1",
        "q1::pos0016-0031",
        "SPURIOUS_SUPPORT",
        {
            "SHIFT_LEFT_2": "NO",
            "SHIFT_RIGHT_2": "YES",
            "RESAMPLE_75": "YES",
            "RESAMPLE_50": "NO",
            "CONTEXT_1P5X": "NO",
        },
        strict_valid={"SHIFT_RIGHT_2": False, "CONTEXT_1P5X": False},
    )
    manifest += rows
    probes += pred
    rows, pred = make_candidate(
        "q2",
        "q2::pos0000-0015",
        "TRUE_SUPPORT",
        {
            "SHIFT_LEFT_2": "ERROR",
            "SHIFT_RIGHT_2": "YES",
            "RESAMPLE_75": "INVALID",
            "RESAMPLE_50": "YES",
            "CONTEXT_1P5X": "NO",
        },
        target_field="phase",
        dataset_name="NurViD",
    )
    manifest += rows
    probes += pred
    rows, pred = make_candidate(
        "q2",
        "q2::pos0016-0031",
        "SPURIOUS_SUPPORT",
        {
            "SHIFT_LEFT_2": "NO",
            "SHIFT_RIGHT_2": "NO",
            "RESAMPLE_75": "NO",
            "RESAMPLE_50": "NO",
        },
        strict_valid={"CONTEXT_1P5X": False},
        generation_valid={"CONTEXT_1P5X": False},
        target_field="phase",
        dataset_name="NurViD",
    )
    manifest += rows
    probes += pred
    joined, _ = join_intervention_results(manifest, probes, expected_generation_valid=19)
    return joined, manifest, probes


class PhaseDETests(unittest.TestCase):
    def test_exact_intervention_id_join(self):
        manifest, probes = make_candidate("q", "q::pos0000-0015", "TRUE_SUPPORT", {t: "YES" for t in TYPES})
        joined, summary = join_intervention_results(manifest, probes, expected_generation_valid=5)
        self.assertEqual(summary["n_matched"], 5)
        self.assertEqual({row["intervention_id"] for row in joined}, {row["intervention_id"] for row in manifest})
        self.assertEqual({row["candidate_id"] for row in joined}, {"q::pos0000-0015"})
        self.assertEqual({row["candidate_type"] for row in joined}, {"TRUE_SUPPORT"})
        self.assertEqual({row["intervention_prediction"] for row in joined}, {"YES"})

    def test_duplicate_intervention_probe_id_fails(self):
        manifest, probes = make_candidate("q", "q::pos0000-0015", "TRUE_SUPPORT", {t: "YES" for t in TYPES})
        with self.assertRaises(RuntimeError):
            join_intervention_results(manifest, probes + [probes[0]], expected_generation_valid=5)

    def test_duplicate_manifest_id_fails(self):
        manifest, probes = make_candidate("q", "q::pos0000-0015", "TRUE_SUPPORT", {t: "YES" for t in TYPES})
        with self.assertRaises(RuntimeError):
            join_intervention_results(manifest + [manifest[0]], probes, expected_generation_valid=5)

    def test_missing_generation_valid_prediction_fails(self):
        manifest, probes = make_candidate("q", "q::pos0000-0015", "TRUE_SUPPORT", {t: "YES" for t in TYPES})
        with self.assertRaises(RuntimeError):
            join_intervention_results(manifest, probes[:-1], expected_generation_valid=5)

    def test_generation_invalid_missing_prediction_allowed(self):
        manifest, probes = make_candidate(
            "q",
            "q::pos0000-0015",
            "TRUE_SUPPORT",
            {t: "YES" for t in TYPES if t != "CONTEXT_1P5X"},
            generation_valid={"CONTEXT_1P5X": False},
        )
        joined, summary = join_intervention_results(manifest, probes, expected_generation_valid=4)
        self.assertEqual(summary["n_missing_generation_valid_predictions"], 0)
        context = [row for row in joined if row["intervention_type"] == "CONTEXT_1P5X"][0]
        self.assertEqual(context["parsed_prediction"], "NOT_RUN_GENERATION_INVALID")

    def test_historical_errors_do_not_override_valid_final_results(self):
        manifest, probes = make_candidate("q", "q::pos0000-0015", "TRUE_SUPPORT", {t: "YES" for t in TYPES})
        error_rows = [
            {
                "intervention_id": probes[0]["intervention_id"],
                "qa_id": "q",
                "error_type": "CUDA_OOM_STOP",
            }
        ]
        joined, summary = join_intervention_results(manifest, probes, error_rows=error_rows, expected_generation_valid=5)
        by_id = {row["intervention_id"]: row for row in joined}
        self.assertEqual(by_id[probes[0]["intervention_id"]]["intervention_prediction"], "YES")
        self.assertEqual(summary["historical_error_records"], 1)
        self.assertEqual(summary["historical_errors_resolved_by_final_result"], 1)
        self.assertEqual(summary["unresolved_errors"], 0)
        audit = d2_final_prediction_audit(manifest, probes, error_rows)
        self.assertEqual(audit["historical_errors_resolved_by_final_result"], 1)

    def test_raw_gt_leakage_detection_fails(self):
        manifest, probes = make_candidate("q", "q::pos0000-0015", "TRUE_SUPPORT", {t: "YES" for t in TYPES})
        probes[0]["candidate_label"] = "TRUE_SUPPORT"
        with self.assertRaises(RuntimeError):
            join_intervention_results(manifest, probes, expected_generation_valid=5)

    def test_synthetic_fixture_scores_exactly(self):
        joined, _, _ = synthetic_joined()
        candidates = candidate_stability_rows(joined)
        by_window = {row["window_id"]: row for row in candidates}
        a = by_window["q1::pos0000-0015"]
        self.assertEqual(a["original_candidate_label"], "TRUE_SUPPORT")
        self.assertEqual(a["n_shift_valid"], 2)
        self.assertEqual(a["n_shift_yes"], 2)
        self.assertEqual(a["S_shift"], 1.0)
        self.assertEqual(a["S_resample"], 0.5)
        self.assertTrue(is_nan(a["S_context"]))
        self.assertEqual(a["S_overall_micro"], 0.75)
        self.assertEqual(a["S_overall_macro"], 0.75)
        self.assertEqual(a["candidate_id"], "q1::pos0000-0015")
        self.assertEqual(a["candidate_type"], "TRUE_SUPPORT")
        self.assertEqual(a["n_valid_shift"], 2)
        self.assertEqual(a["n_yes_resample"], 1)
        self.assertEqual(len(a["strict_valid_intervention_ids"]), 4)

        b = by_window["q1::pos0016-0031"]
        self.assertEqual(b["original_candidate_label"], "SPURIOUS_SUPPORT")
        self.assertEqual(b["S_shift"], 0.0)
        self.assertEqual(b["S_resample"], 0.5)
        self.assertTrue(is_nan(b["S_context"]))

        c = by_window["q2::pos0000-0015"]
        self.assertEqual(c["S_resample"], 1.0)
        self.assertEqual(c["n_resample_valid"], 1)
        self.assertEqual(c["n_missing_binary_outputs"], 2)

        d = by_window["q2::pos0016-0031"]
        self.assertTrue(is_nan(d["S_context"]))

    def test_invalid_interventions_never_encoded_as_zero(self):
        joined, _, _ = synthetic_joined()
        candidates = candidate_stability_rows(joined)
        a = [row for row in candidates if row["window_id"] == "q1::pos0000-0015"][0]
        b = [row for row in candidates if row["window_id"] == "q1::pos0016-0031"][0]
        self.assertTrue(is_nan(a["R_CONTEXT_1P5X"]))
        self.assertTrue(is_nan(b["R_SHIFT_RIGHT_2"]))
        self.assertTrue(is_nan(b["R_CONTEXT_1P5X"]))

    def test_generation_invalid_excluded(self):
        joined, _, _ = synthetic_joined()
        invalid = [row for row in joined if not row["generation_valid"]][0]
        self.assertFalse(include_in_primary_denominator(invalid))

    def test_model_invalid_and_error_excluded_from_denominator(self):
        joined, _, _ = synthetic_joined()
        candidates = candidate_stability_rows(joined)
        c = [row for row in candidates if row["window_id"] == "q2::pos0000-0015"][0]
        self.assertTrue(is_nan(c["R_SHIFT_LEFT_2"]))
        self.assertTrue(is_nan(c["R_RESAMPLE_75"]))
        self.assertEqual(c["R_RESAMPLE_50"], 1)

    def test_qa_level_true_spurious_and_delta(self):
        joined, _, _ = synthetic_joined()
        qa_rows = qa_stability_rows(candidate_stability_rows(joined))
        q1 = [row for row in qa_rows if row["qa_id"] == "q1"][0]
        self.assertEqual(q1["n_true_candidates"], 1)
        self.assertEqual(q1["n_spurious_candidates"], 1)
        self.assertEqual(q1["mean_S_true_macro"], 0.75)
        self.assertEqual(q1["mean_S_spurious_macro"], 0.25)
        self.assertEqual(q1["delta_S_macro_mean"], 0.5)
        self.assertEqual(q1["delta_macro"], 0.5)
        self.assertEqual(q1["mean_true_macro"], 0.75)
        self.assertEqual(q1["mean_spurious_macro"], 0.25)
        self.assertEqual(q1["target_field"], "action")
        self.assertEqual(q1["dataset_name"], "AVOS")

    def test_synthetic_protocol_example_matches_expected_scores(self):
        manifest = []
        probes = []
        strict = {"CONTEXT_1P5X": False}
        predictions = {
            "SHIFT_LEFT_2": "YES",
            "SHIFT_RIGHT_2": "NO",
            "RESAMPLE_75": "YES",
            "RESAMPLE_50": "YES",
            "CONTEXT_1P5X": "NO",
        }
        rows, pred = make_candidate("q", "q::candidateA", "TRUE_SUPPORT", predictions, strict_valid=strict)
        manifest += rows
        probes += pred
        joined, _ = join_intervention_results(manifest, probes, expected_generation_valid=5)
        candidate = candidate_stability_rows(joined)[0]
        self.assertEqual(candidate["S_shift"], 0.5)
        self.assertEqual(candidate["S_resample"], 1.0)
        self.assertTrue(is_nan(candidate["S_context"]))
        self.assertEqual(candidate["S_overall_micro"], 0.75)
        self.assertEqual(candidate["S_overall_macro"], 0.75)

    def test_paired_usability_recomputed_after_model_outputs(self):
        joined, _, _ = synthetic_joined()
        qa_rows = qa_stability_rows(candidate_stability_rows(joined))
        q1 = [row for row in qa_rows if row["qa_id"] == "q1"][0]
        q2 = [row for row in qa_rows if row["qa_id"] == "q2"][0]
        self.assertTrue(q1["paired_macro_usable"])
        self.assertTrue(q1["paired_shift_analysis_usable"])
        self.assertFalse(q1["paired_context_analysis_usable"])
        self.assertTrue(q2["paired_macro_usable"])
        self.assertFalse(q2["paired_context_analysis_usable"])

    def test_retention_matrix_and_missingness_outputs(self):
        joined, _, _ = synthetic_joined()
        candidates = candidate_stability_rows(joined)
        matrix = retention_matrix_rows(candidates)
        self.assertEqual(len(matrix), 4)
        self.assertTrue(any(row["SHIFT_LEFT_2"] == 1 for row in matrix))
        missing = missingness_rows(joined, candidates)
        self.assertTrue(any(row["model_INVALID"] > 0 for row in missing))
        self.assertTrue(any(row["model_ERROR"] > 0 for row in missing))
        self.assertTrue(any(row["family_NaN"] > 0 for row in missing))

    def test_dataset_and_target_summaries_preserve_groups(self):
        joined, _, _ = synthetic_joined()
        candidates = candidate_stability_rows(joined)
        qa_rows = qa_stability_rows(candidates)
        by_dataset = stability_by_group_rows(candidates, qa_rows, "dataset_name")
        by_target = stability_by_group_rows(candidates, qa_rows, "target_field")
        self.assertTrue(any(row["dataset_name"] == "AVOS" for row in by_dataset))
        self.assertTrue(any(row["dataset_name"] == "NurViD" for row in by_dataset))
        self.assertTrue(any(row["target_field"] == "action" for row in by_target))
        self.assertTrue(any(row["target_field"] == "phase" for row in by_target))

    def test_phase_e_summary_keeps_primary_protocol(self):
        joined, _, _ = synthetic_joined()
        candidates = candidate_stability_rows(joined)
        qa_rows = qa_stability_rows(candidates)
        summary = phase_e_summary(candidates, qa_rows, "strict")
        self.assertEqual(summary["primary_h2_analysis"]["target_type"], "action")
        self.assertEqual(summary["primary_h2_analysis"]["primary_unit"], "qa")
        self.assertEqual(summary["primary_h2_analysis"]["preferred_summary_score"], "S_overall_macro")
        self.assertFalse(summary["primary_h2_analysis"]["inferential_statistics_run"])

    def test_phase_e_script_rejects_threshold_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_argv = phase_e_script.sys.argv
            try:
                phase_e_script.sys.argv = [
                    "08_compute_stability.py",
                    "--joined_results",
                    str(Path(tmp) / "missing.jsonl"),
                    "--output_dir",
                    str(Path(tmp) / "phase_e"),
                    "--stable_hallucination_threshold",
                    "0.8",
                ]
                with self.assertRaises(RuntimeError):
                    phase_e_script.main()
            finally:
                phase_e_script.sys.argv = old_argv

    def test_join_cli_exits_cleanly_without_full_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_argv = phase_d3_join_script.sys.argv
            try:
                phase_d3_join_script.sys.argv = [
                    "07_join_intervention_results.py",
                    "--manifest",
                    str(Path(tmp) / "manifest.jsonl"),
                    "--probe_results",
                    str(Path(tmp) / "missing.jsonl"),
                    "--output_dir",
                    str(Path(tmp) / "out"),
                ]
                with contextlib.redirect_stdout(io.StringIO()):
                    phase_d3_join_script.main()
            finally:
                phase_d3_join_script.sys.argv = old_argv
            summary = read_json(Path(tmp) / "out" / "phase_d3_join_audit.json")
            self.assertFalse(summary["full_phase_d2b_predictions_available"])
            self.assertFalse(summary["real_h2_analysis_executed"])

    def test_phase_e_cli_exits_cleanly_without_joined_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_argv = phase_e_script.sys.argv
            try:
                phase_e_script.sys.argv = [
                    "08_compute_stability.py",
                    "--joined_results",
                    str(Path(tmp) / "missing.jsonl"),
                    "--output_dir",
                    str(Path(tmp) / "phase_e"),
                ]
                with contextlib.redirect_stdout(io.StringIO()):
                    phase_e_script.main()
            finally:
                phase_e_script.sys.argv = old_argv
            summary = read_json(Path(tmp) / "phase_e" / "summary" / "phase_e_summary.json")
            self.assertFalse(summary["joined_intervention_results_available"])
            self.assertFalse(summary["real_h2_analysis_executed"])

    def test_phase_e_cli_writes_synthetic_outputs_without_statistics(self):
        joined, _, _ = synthetic_joined()
        with tempfile.TemporaryDirectory() as tmp:
            joined_path = Path(tmp) / "joined.jsonl"
            out_dir = Path(tmp) / "phase_e"
            write_jsonl(joined_path, joined)
            old_argv = phase_e_script.sys.argv
            try:
                phase_e_script.sys.argv = [
                    "08_compute_stability.py",
                    "--joined_results",
                    str(joined_path),
                    "--output_dir",
                    str(out_dir),
                ]
                with contextlib.redirect_stdout(io.StringIO()):
                    phase_e_script.main()
            finally:
                phase_e_script.sys.argv = old_argv
            self.assertTrue((out_dir / "candidate_stability.csv").exists())
            self.assertTrue((out_dir / "candidate" / "candidate_stability.jsonl").exists())
            self.assertTrue((out_dir / "candidate" / "candidate_stability_distribution.csv").exists())
            self.assertTrue((out_dir / "candidate" / "spurious_high_stability.csv").exists())
            self.assertTrue((out_dir / "candidate" / "true_low_stability.csv").exists())
            self.assertTrue((out_dir / "qa_stability.csv").exists())
            self.assertTrue((out_dir / "qa" / "h2_primary_action_qa.csv").exists())
            self.assertTrue((out_dir / "qa" / "h2_all_paired_qa.csv").exists())
            self.assertTrue((out_dir / "qa" / "h2_phase_only_qa.csv").exists())
            self.assertTrue((out_dir / "summary" / "yes_stickiness_diagnostic.csv").exists())
            self.assertTrue((out_dir / "provenance" / "h2_protocol_provenance.json").exists())
            self.assertTrue((out_dir / "intervention_retention_matrix.csv").exists())
            summary = read_json(out_dir / "summary" / "phase_e_summary.json")
            self.assertFalse(summary["real_h2_analysis_executed"])
            self.assertFalse(summary["threshold_selection_run"])
            primary = read_json(out_dir / "summary" / "h2_primary_action_summary.json")
            phase = read_json(out_dir / "summary" / "h2_phase_only_summary.json")
            self.assertEqual(primary["n_paired_QA"], 1)
            self.assertEqual(phase["n_paired_QA"], 1)


if __name__ == "__main__":
    unittest.main()
