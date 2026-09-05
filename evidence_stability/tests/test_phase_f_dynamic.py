import math
import importlib.util
import unittest
from collections import Counter
from pathlib import Path

from evidence_stability.cache import make_h3_dynamic_probe_cache_key, stable_hash
from evidence_stability.dynamic import (
    BLOCK_ORDER,
    FREEZE_MID_IDX,
    INTERVENTION_TYPES,
    assert_dynamic_intervention_valid,
    assert_model_manifest_gt_free,
    candidate_dynamic_rows,
    clean_mean,
    deterministic_smoke_candidate_ids,
    dynamic_sensitivity,
    final_records_by_intervention,
    generate_dynamic_interventions_for_candidate,
    make_full_shuffle_permutation,
    multiset_preserved,
    ordered_frame_hash,
    qa_dynamic_rows,
    reconstruct_h3_candidates,
)
from evidence_stability.prompts import PROMPT_VERSION, build_evidence_presence_prompt


PROBE_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "11_probe_h3_dynamic_interventions.py"
PROBE_SPEC = importlib.util.spec_from_file_location("h3_dynamic_probe_script", PROBE_SCRIPT_PATH)
h3_probe_script = importlib.util.module_from_spec(PROBE_SPEC)
assert PROBE_SPEC and PROBE_SPEC.loader
PROBE_SPEC.loader.exec_module(h3_probe_script)


def is_nan(value):
    return isinstance(value, float) and math.isnan(value)


def synthetic_frozen_inputs():
    phase_c = []
    paired = []
    true_remaining = 64
    spur_remaining = 94
    datasets = ["AVOS", "CholecT50", "EgoSurgery", "NurViD"]
    for q in range(16):
        qa_id = f"{q:04d}::clip-{q}"
        target_field = "action" if q < 9 else "phase"
        n_true = 4
        n_spur = 6
        if q == 15:
            n_true = true_remaining
            n_spur = spur_remaining
        true_remaining -= n_true
        spur_remaining -= n_spur
        true_ids = []
        spur_ids = []
        for i in range(n_true + n_spur):
            window_id = f"{qa_id}::pos{i * 8:04d}-{i * 8 + 15:04d}"
            if i < n_true:
                true_ids.append(window_id)
            else:
                spur_ids.append(window_id)
            phase_c.append(
                {
                    "qa_id": qa_id,
                    "clip_id": f"clip-{q}",
                    "window_id": window_id,
                    "dataset_name": datasets[q % len(datasets)],
                    "target_action": "suturing" if target_field == "action" else "preparation phase",
                    "target_field": target_field,
                    "frame_paths": [f"/root/data/{datasets[q % len(datasets)]}/v/{i:02d}_{j:02d}.jpg" for j in range(16)],
                    "parsed_prediction": "YES",
                    "prompt_version": PROMPT_VERSION,
                    "prompt_hash": stable_hash({"prompt": build_evidence_presence_prompt("suturing")}),
                    "cache_key": f"phase-c::{window_id}",
                }
            )
        paired.append(
            {
                "qa_id": qa_id,
                "clip_id": f"clip-{q}",
                "dataset_name": datasets[q % len(datasets)],
                "target_action": "suturing" if target_field == "action" else "preparation phase",
                "target_field": target_field,
                "paired_h2_eligible": True,
                "true_support_window_ids": true_ids,
                "spurious_support_window_ids": spur_ids,
            }
        )
    return phase_c, paired


def candidate(candidate_id="q::pos0000-0015", frames=None):
    frames = frames or [f"f{i}.jpg" for i in range(16)]
    return {
        "candidate_id": candidate_id,
        "qa_id": "q",
        "clip_id": "clip",
        "window_id": candidate_id,
        "dataset_name": "AVOS",
        "target_action": "suturing",
        "target_field": "action",
        "ordered_frame_paths": frames,
        "original_prediction": "YES",
    }


def joined_row(candidate_type, intervention_type, prediction, qa_id="q", candidate_id=None):
    candidate_id = candidate_id or f"{qa_id}::{candidate_type}"
    return {
        "intervention_id": f"{candidate_id}::{intervention_type}",
        "candidate_id": candidate_id,
        "qa_id": qa_id,
        "clip_id": f"clip-{qa_id}",
        "dataset_name": "AVOS",
        "target_action": "suturing",
        "target_field": "action",
        "candidate_type": candidate_type,
        "original_prediction": "YES",
        "intervention_type": intervention_type,
        "prediction": prediction,
    }


class PhaseFDynamicTests(unittest.TestCase):
    def test_candidate_reconstruction_frozen_counts_and_yes_predictions(self):
        phase_c, paired = synthetic_frozen_inputs()
        candidates, audit = reconstruct_h3_candidates(phase_c, paired)
        self.assertEqual(audit["n_total_candidates"], 158)
        self.assertEqual(audit["candidate_type_counts"]["TRUE_SUPPORT"], 64)
        self.assertEqual(audit["candidate_type_counts"]["SPURIOUS_SUPPORT"], 94)
        self.assertEqual(audit["original_prediction_counts"], {"YES": 158})
        self.assertEqual(audit["n_primary_action_qa"], 9)
        self.assertEqual(audit["n_all_paired_qa"], 16)
        self.assertEqual(audit["n_phase_only_qa"], 7)
        self.assertEqual(len({row["candidate_id"] for row in candidates}), 158)

    def test_candidate_reconstruction_fails_if_original_prediction_not_yes(self):
        phase_c, paired = synthetic_frozen_inputs()
        phase_c[0]["parsed_prediction"] = "NO"
        with self.assertRaises(RuntimeError):
            reconstruct_h3_candidates(phase_c, paired)

    def test_full_shuffle_preserves_length_multiset_duplicates_and_is_deterministic(self):
        frames = [f"f{i}.jpg" for i in range(16)]
        frames[5] = frames[4]
        c = candidate(frames=frames)
        first = [row for row in generate_dynamic_interventions_for_candidate(c) if row["intervention_type"] == "FULL_SHUFFLE_V1"][0]
        second = [row for row in generate_dynamic_interventions_for_candidate(c) if row["intervention_type"] == "FULL_SHUFFLE_V1"][0]
        self.assertEqual(first["ordered_frame_paths"], second["ordered_frame_paths"])
        self.assertEqual(first["n_frames"], 16)
        self.assertTrue(multiset_preserved(frames, first["ordered_frame_paths"]))
        self.assertEqual(Counter(frames), Counter(first["ordered_frame_paths"]))
        self.assertNotEqual(first["intervention_provenance"]["permutation"], list(range(16)))
        self.assertGreaterEqual(first["intervention_provenance"]["n_positions_changed"], 12)
        assert_dynamic_intervention_valid(first, frames)

    def test_full_shuffle_different_candidate_ids_have_deterministic_valid_permutations(self):
        a = make_full_shuffle_permutation("candidate-a")
        b = make_full_shuffle_permutation("candidate-b")
        self.assertTrue(a["generation_valid"])
        self.assertTrue(b["generation_valid"])
        self.assertEqual(a, make_full_shuffle_permutation("candidate-a"))
        self.assertNotEqual(a["permutation"], b["permutation"])

    def test_block_shuffle_fixed_order_and_preserves_internal_blocks(self):
        c = candidate()
        row = [r for r in generate_dynamic_interventions_for_candidate(c) if r["intervention_type"] == "BLOCK_SHUFFLE_4X4_V1"][0]
        self.assertEqual(row["n_frames"], 16)
        self.assertEqual(row["intervention_provenance"]["block_order"], BLOCK_ORDER)
        self.assertEqual(row["ordered_frame_paths"][0:4], c["ordered_frame_paths"][8:12])
        self.assertEqual(row["ordered_frame_paths"][4:8], c["ordered_frame_paths"][0:4])
        self.assertTrue(multiset_preserved(c["ordered_frame_paths"], row["ordered_frame_paths"]))
        assert_dynamic_intervention_valid(row, c["ordered_frame_paths"])

    def test_freeze_mid_uses_index_7_and_preserves_16_logical_entries(self):
        c = candidate()
        row = [r for r in generate_dynamic_interventions_for_candidate(c) if r["intervention_type"] == "FREEZE_MID_V1"][0]
        self.assertEqual(row["n_frames"], 16)
        self.assertEqual(row["intervention_provenance"]["mid_idx"], FREEZE_MID_IDX)
        self.assertEqual(row["ordered_frame_paths"], [c["ordered_frame_paths"][7]] * 16)
        self.assertEqual(len(row["ordered_frame_paths"]), 16)
        assert_dynamic_intervention_valid(row, c["ordered_frame_paths"])

    def test_model_manifest_is_gt_free_and_has_no_candidate_type(self):
        row = generate_dynamic_interventions_for_candidate(candidate())[0]
        self.assertNotIn("candidate_type", row)
        self.assertNotIn("gt_alignment_class", row)
        self.assertNotIn("evidence_density", row)
        self.assertFalse(row["gt_information_used_in_model_inference"])
        assert_model_manifest_gt_free(row)
        leaked = dict(row)
        leaked["candidate_type"] = "TRUE_SUPPORT"
        with self.assertRaises(RuntimeError):
            assert_model_manifest_gt_free(leaked)

    def test_prompt_has_no_intervention_or_gt_terms_and_hash_is_stable(self):
        prompt = build_evidence_presence_prompt("suturing")
        lowered = prompt.lower()
        for forbidden in ["freeze", "shuffle", "block_shuffle", "intervention", "dynamic", "true_support", "spurious_support"]:
            self.assertNotIn(forbidden, lowered)
        self.assertEqual(stable_hash({"prompt": prompt}), stable_hash({"prompt": prompt}))

    def test_h3_cache_key_includes_protocol_variables(self):
        base = {
            "model_identity_hash": "model",
            "prompt_hash": "prompt",
            "qa_id": "q",
            "candidate_id": "c",
            "intervention_id": "c::FULL_SHUFFLE_V1",
            "intervention_type": "FULL_SHUFFLE_V1",
            "ordered_frame_paths": ["a.jpg", "b.jpg"],
            "ordered_frame_hash": ordered_frame_hash(["a.jpg", "b.jpg"]),
            "target_action": "suturing",
            "decoding_config": {"do_sample": False, "max_new_tokens": 8, "enable_thinking": False},
            "intervention_protocol": {"permutation": [1, 0]},
        }
        a = make_h3_dynamic_probe_cache_key(**base)
        b = make_h3_dynamic_probe_cache_key(**{**base, "intervention_protocol": {"permutation": [0, 1]}})
        c = make_h3_dynamic_probe_cache_key(**{**base, "ordered_frame_paths": ["b.jpg", "a.jpg"]})
        self.assertNotEqual(a, b)
        self.assertNotEqual(a, c)

    def test_synthetic_dynamic_scores_match_protocol_examples(self):
        rows = [
            joined_row("TRUE_SUPPORT", "FULL_SHUFFLE_V1", "NO", candidate_id="c1"),
            joined_row("TRUE_SUPPORT", "BLOCK_SHUFFLE_4X4_V1", "YES", candidate_id="c1"),
            joined_row("TRUE_SUPPORT", "FREEZE_MID_V1", "NO", candidate_id="c1"),
            joined_row("SPURIOUS_SUPPORT", "FULL_SHUFFLE_V1", "YES", candidate_id="c2"),
            joined_row("SPURIOUS_SUPPORT", "BLOCK_SHUFFLE_4X4_V1", "YES", candidate_id="c2"),
            joined_row("SPURIOUS_SUPPORT", "FREEZE_MID_V1", "YES", candidate_id="c2"),
        ]
        candidates = candidate_dynamic_rows(rows)
        c1 = [row for row in candidates if row["candidate_id"] == "c1"][0]
        c2 = [row for row in candidates if row["candidate_id"] == "c2"][0]
        self.assertEqual(c1["D_full_shuffle"], 1)
        self.assertEqual(c1["D_block_shuffle"], 0)
        self.assertEqual(c1["D_order"], 0.5)
        self.assertEqual(c1["D_freeze"], 1)
        self.assertEqual(c1["D_macro_exploratory"], 0.75)
        self.assertEqual(c2["D_order"], 0)
        self.assertEqual(c2["D_freeze"], 0)
        self.assertEqual(c2["D_macro_exploratory"], 0)
        self.assertEqual(dynamic_sensitivity("YES"), 0)
        self.assertEqual(dynamic_sensitivity("NO"), 1)
        self.assertTrue(is_nan(dynamic_sensitivity("INVALID")))

    def test_freeze_invalid_does_not_alter_d_order_and_macro_ignores_nan(self):
        rows = [
            joined_row("TRUE_SUPPORT", "FULL_SHUFFLE_V1", "NO", candidate_id="c1"),
            joined_row("TRUE_SUPPORT", "BLOCK_SHUFFLE_4X4_V1", "YES", candidate_id="c1"),
            joined_row("TRUE_SUPPORT", "FREEZE_MID_V1", "INVALID", candidate_id="c1"),
        ]
        c1 = candidate_dynamic_rows(rows)[0]
        self.assertEqual(c1["D_order"], 0.5)
        self.assertTrue(is_nan(c1["D_freeze"]))
        self.assertEqual(c1["D_macro_exploratory"], 0.5)
        self.assertEqual(clean_mean([0.5, float("nan")]), 0.5)

    def test_qa_aggregation_averages_candidates_within_type_before_delta(self):
        rows = [
            joined_row("TRUE_SUPPORT", "FULL_SHUFFLE_V1", "NO", qa_id="q", candidate_id="t1"),
            joined_row("TRUE_SUPPORT", "BLOCK_SHUFFLE_4X4_V1", "NO", qa_id="q", candidate_id="t1"),
            joined_row("TRUE_SUPPORT", "FREEZE_MID_V1", "YES", qa_id="q", candidate_id="t1"),
            joined_row("TRUE_SUPPORT", "FULL_SHUFFLE_V1", "YES", qa_id="q", candidate_id="t2"),
            joined_row("TRUE_SUPPORT", "BLOCK_SHUFFLE_4X4_V1", "YES", qa_id="q", candidate_id="t2"),
            joined_row("TRUE_SUPPORT", "FREEZE_MID_V1", "YES", qa_id="q", candidate_id="t2"),
            joined_row("SPURIOUS_SUPPORT", "FULL_SHUFFLE_V1", "YES", qa_id="q", candidate_id="s1"),
            joined_row("SPURIOUS_SUPPORT", "BLOCK_SHUFFLE_4X4_V1", "YES", qa_id="q", candidate_id="s1"),
            joined_row("SPURIOUS_SUPPORT", "FREEZE_MID_V1", "YES", qa_id="q", candidate_id="s1"),
        ]
        qa = qa_dynamic_rows(candidate_dynamic_rows(rows), [{"qa_id": "q", "paired_h2_eligible": True}])[0]
        self.assertEqual(qa["n_true_candidates"], 2)
        self.assertEqual(qa["n_spurious_candidates"], 1)
        self.assertEqual(qa["mean_true_D_order"], 0.5)
        self.assertEqual(qa["mean_spur_D_order"], 0)
        self.assertEqual(qa["delta_D_order"], 0.5)

    def test_historical_error_does_not_override_successful_final_result(self):
        expected = {"i1", "i2"}
        results = [{"intervention_id": "i1", "prediction": "YES"}, {"intervention_id": "i2", "prediction": "NO"}]
        errors = [{"intervention_id": "i1", "error_type": "CUDA_OOM_STOP"}]
        final, audit = final_records_by_intervention(results, errors, expected)
        self.assertEqual(final["i1"]["prediction"], "YES")
        self.assertEqual(audit["historical_errors"], 1)
        self.assertEqual(audit["resolved_by_final_result"], 1)
        self.assertEqual(audit["unresolved_errors"], 0)

    def test_duplicate_intervention_results_fail_loudly(self):
        with self.assertRaises(RuntimeError):
            final_records_by_intervention(
                [{"intervention_id": "i1", "prediction": "YES"}, {"intervention_id": "i1", "prediction": "NO"}],
                [],
                {"i1"},
            )

    def test_processor_frame_audit_accepts_nested_expected_frame_count(self):
        audit = h3_probe_script.processor_frame_audit(
            [
                {
                    "intervention_id": "i1",
                    "intervention_type": "FREEZE_MID_V1",
                    "processor_metadata": {
                        "expected_frame_count": 16,
                        "input_frame_count": 16,
                        "image_grid_thw_rows": 16,
                    },
                }
            ]
        )
        self.assertEqual(audit["FREEZE_MID_V1"]["n_checked"], 1)
        self.assertTrue(audit["FREEZE_MID_V1"]["all_input_frame_count_16"])
        self.assertEqual(audit["frame_count_processor_audit"], {16: True})

    def test_smoke_selection_is_gt_blind_and_dataset_covering(self):
        phase_c, paired = synthetic_frozen_inputs()
        candidates, _ = reconstruct_h3_candidates(phase_c, paired)
        selected = deterministic_smoke_candidate_ids(candidates, n=12)
        self.assertEqual(len(selected), 12)
        by_id = {row["candidate_id"]: row for row in candidates}
        self.assertGreaterEqual(len({by_id[cid]["dataset_name"] for cid in selected}), 4)


if __name__ == "__main__":
    unittest.main()
