import inspect
import importlib.util
import unittest
from pathlib import Path

from evidence_stability.interventions import (
    generate_context,
    generate_resample,
    generate_shift,
    intervention_position_plan,
)


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "05_generate_interventions.py"
SPEC = importlib.util.spec_from_file_location("phase_d_generate_interventions_script", SCRIPT_PATH)
phase_d_script = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(phase_d_script)


class InterventionGenerationTests(unittest.TestCase):
    def test_shift_left_normal_window(self):
        plan = generate_shift(10, 25, 60, -2)
        self.assertTrue(plan.generation_valid)
        self.assertEqual(plan.positions, tuple(range(8, 24)))

    def test_shift_right_normal_window(self):
        plan = generate_shift(10, 25, 60, 2)
        self.assertTrue(plan.generation_valid)
        self.assertEqual(plan.positions, tuple(range(12, 28)))

    def test_shift_at_left_boundary(self):
        plan = generate_shift(0, 15, 60, -2)
        self.assertFalse(plan.generation_valid)
        self.assertEqual(plan.generation_invalid_reason, "GENERATION_INVALID_BOUNDARY")

    def test_shift_at_right_boundary(self):
        plan = generate_shift(44, 59, 60, 2)
        self.assertFalse(plan.generation_valid)
        self.assertEqual(plan.generation_invalid_reason, "GENERATION_INVALID_BOUNDARY")

    def test_resample_75_returns_12_ordered_positions(self):
        plan = generate_resample(10, 25, 12)
        self.assertTrue(plan.generation_valid)
        self.assertEqual(len(plan.positions), 12)
        self.assertEqual(list(plan.positions), sorted(plan.positions))

    def test_resample_50_returns_8_ordered_positions(self):
        plan = generate_resample(10, 25, 8)
        self.assertTrue(plan.generation_valid)
        self.assertEqual(len(plan.positions), 8)
        self.assertEqual(list(plan.positions), sorted(plan.positions))

    def test_resampling_is_deterministic(self):
        self.assertEqual(generate_resample(10, 25, 12), generate_resample(10, 25, 12))

    def test_context_expansion_centered_case(self):
        plan = generate_context(10, 25, 60, 24)
        self.assertTrue(plan.generation_valid)
        self.assertFalse(plan.boundary_asymmetric)
        self.assertEqual(plan.positions, tuple(range(6, 30)))

    def test_context_expansion_left_boundary_asymmetric_case(self):
        plan = generate_context(0, 15, 60, 24)
        self.assertTrue(plan.generation_valid)
        self.assertTrue(plan.boundary_asymmetric)
        self.assertEqual(plan.positions, tuple(range(0, 24)))

    def test_context_expansion_right_boundary_asymmetric_case(self):
        plan = generate_context(44, 59, 60, 24)
        self.assertTrue(plan.generation_valid)
        self.assertTrue(plan.boundary_asymmetric)
        self.assertEqual(plan.positions, tuple(range(36, 60)))

    def test_impossible_24_frame_expansion(self):
        plan = generate_context(0, 15, 20, 24)
        self.assertFalse(plan.generation_valid)
        self.assertEqual(plan.generation_invalid_reason, "GENERATION_INVALID_BOUNDARY")

    def test_duplicate_frames_preserved_by_position_selection(self):
        frame_paths = ["a.jpg", "b.jpg", "b.jpg", "c.jpg"]
        plan = generate_shift(0, 3, 6, 0)
        selected = [frame_paths[i] for i in plan.positions]
        self.assertEqual(selected, frame_paths)

    def test_intervention_id_uniqueness_pattern(self):
        window_id = "0001::clip::pos0000-0015"
        ids = {f"{window_id}::{name}" for name in ["SHIFT_LEFT_2", "SHIFT_RIGHT_2", "RESAMPLE_75", "RESAMPLE_50", "CONTEXT_1P5X"]}
        self.assertEqual(len(ids), 5)
        self.assertNotIn(window_id, ids)

    def test_generation_functions_do_not_accept_gt_arguments(self):
        forbidden = {
            "gt_spans",
            "gt_alignment_class",
            "evidence_density",
            "gt_evidence_recall",
            "n_gt_visible_in_window",
            "n_gt_visible_total",
            "candidate_label",
        }
        for fn in [generate_shift, generate_resample, generate_context, intervention_position_plan]:
            self.assertFalse(set(inspect.signature(fn).parameters) & forbidden)

    def test_spurious_intervention_entering_gt_becomes_invalid(self):
        result = phase_d_script.classify_validity(
            "SPURIOUS_SUPPORT",
            {"n_gt_visible_in_window": 1, "gt_alignment_class": "WEAK_GT_ALIGNED"},
        )
        self.assertFalse(result["strict_valid"])
        self.assertEqual(result["validity_reason"], "GT_CONTAMINATED")

    def test_true_intervention_losing_strong_support_becomes_strict_invalid(self):
        result = phase_d_script.classify_validity(
            "TRUE_SUPPORT",
            {"n_gt_visible_in_window": 1, "gt_alignment_class": "WEAK_GT_ALIGNED"},
        )
        self.assertFalse(result["strict_valid"])
        self.assertTrue(result["any_gt_valid"])
        self.assertEqual(result["validity_reason"], "WEAKENED_GT_SUPPORT")

    def test_original_candidate_label_normalization_does_not_mutate_row(self):
        row = {"candidate_label": "TRUE_SUPPORT"}
        self.assertEqual(phase_d_script.normalize_candidate_label(row), "TRUE_SUPPORT")
        self.assertEqual(row["candidate_label"], "TRUE_SUPPORT")


if __name__ == "__main__":
    unittest.main()
