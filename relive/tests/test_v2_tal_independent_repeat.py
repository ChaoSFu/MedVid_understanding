from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path
import tempfile
import unittest

from relive.storage.artifacts import canonical_json, stable_hash
from relive.v2.independent_repeat import compare_independent_runs, safe_candidate_summary


def _write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, list):
        path.write_bytes(b"".join((canonical_json(row) + "\n").encode() for row in value))
    else:
        path.write_bytes((canonical_json(value) + "\n").encode())


def _read(path: Path):
    raw = path.read_text()
    return [json.loads(line) for line in raw.splitlines() if line] if path.suffix == ".jsonl" else json.loads(raw)


def _stage3b_root(root: Path, *, revision: str = "r1") -> None:
    calls, packets, likelihoods, ranking = [], [], [], []
    for index in range(23):
        window_id, packet_id = f"window_{index:02d}", f"packet_{index:02d}"
        role = "GLOBAL_CONTEXT_DIAGNOSTIC" if index == 22 else "ORDINARY_TEMPORAL_WINDOW"
        packet = {"window_id": window_id, "window_role": role, "start_seconds": float(index), "end_seconds": float(index + 2),
                  "scale_seconds": 56.0 if index == 22 else 8.0, "requirement_id": "requirement_x",
                  "frame_selection_policy": "chronological_endpoints_uniform_interior_v1", "maximum_unique_frames": 16 if index == 22 else 8,
                  "duplicate_visual_frames_removed_from_packet": True,
                  "frames": [{"logical_frame_order": index, "source_frame_reference": f"public/{index}.jpg", "frame_sha256": f"{index:064x}",
                              "timestamp_seconds": float(index), "logical_aliases": [index]}]}
        packets.append(packet)
        calls.append({"window_id": window_id, "packet_id": packet_id, "requirement_id": "requirement_x", "prompt_version": "p1",
                      "prompt_sha256": f"{index + 100:064x}", "frame_ids": [f"frame_{index:02d}"], "frame_sha256": [f"{index:064x}"]})
        logs = {"A": float(30 - index), "B": -2.0, "C": -3.0, "D": -4.0}
        likelihoods.append({"window_id": window_id, "log_probabilities": logs, "support_margin": float(30 - index),
                            "raw_logits": logs, "cache_hit": False, "cache_key": f"cache_{index:02d}"})
        ranking.append({"window_id": window_id, "window_role": role, "scale_seconds": packet["scale_seconds"], "start_seconds": packet["start_seconds"], "end_seconds": packet["end_seconds"],
                        "rank": None if index == 22 else index + 1, "positive_hypothesis_eligible": index != 22,
                        "nms_status": "GLOBAL_CONTEXT_NOT_ELIGIBLE" if index == 22 else ("SELECTED" if index < 5 else "NOT_SELECTED_BUDGET"),
                        "nms_suppressor_window_id": None})
    plan = {"format": "relive-v2-tal-coarse-hypothesis-generation-v1", "planned_model_calls": 23,
            "input_hashes": {"requirement_manifest_sha256": "a" * 64, "selection_manifest_sha256": "b" * 64,
                             "video_index_manifest_sha256": "c" * 64, "timestamp_manifest_sha256": "d" * 64,
                             "timestamp_provenance_sha256": "e" * 64, "search_plan_sha256": "f" * 64,
                             "claim_graph_schema_sha256": "1" * 64, "hypothesis_contract_sha256": "2" * 64,
                             "temporal_policy_sha256": "3" * 64}, "calls": calls}
    _write(root / "v2_stage3b_call_plan.json", plan)
    _write(root / "v2_stage3b_visual_packets.jsonl", packets)
    _write(root / "v2_stage3b_prompt_spec.json", {"format": plan["format"], "prompt_version": "p1", "target_event": "secure_the_base", "required_evidence": ["event"]})
    _write(root / "v2_stage3b_choice_spec.json", {"choice_labels": ["A", "B", "C", "D"], "support_margin_formula": "fixed"})
    _write(root / "v2_stage3b_ranking_policy.json", {"policy_version": "fixed", "temporal_iou_suppression_threshold": .5, "max_positive_hypotheses": 5, "include_no_visible_event_hypothesis": True})
    audit = {"artifact_sha256": {}, "model_calls_made": 0, "backend_loaded": False, "cache_opened": False, "gt_used": False}
    audit["audit_content_sha256"] = stable_hash(audit)
    _write(root / "v2_stage3b_freeze_audit.json", audit)
    contracts = {call["packet_id"]: {"choice_labels": ["A", "B", "C", "D"], "choice_token_ids": {"A": 1, "B": 2, "C": 3, "D": 4}} for call in calls}
    _write(root / "v2_stage3b_preflight.json", {"choice_token_contracts": contracts, "model_fingerprint": {"scientific_identity": {"model": "fixture", "revision": revision, "checkpoint_metadata_sha256": "z" * 64, "actual_model_class": "Fixture", "actual_processor_class": "FixtureProcessor", "chat_template_sha256": "y" * 64, "chat_template_source": "fixture", "generation": {"do_sample": False}}, "preprocessing": {"decode": "Pillow_RGB", "resize": False}}})
    hypotheses = []
    for index in range(5):
        hypotheses.append({"hypothesis_id": f"hypothesis_{index:02d}", "requirement_id": "requirement_x", "parent_claim_id": None, "claim_role": "TARGET_HYPOTHESIS", "target_event": "secure_the_base", "polarity": "POSITIVE", "candidate_interval": {"start_seconds": float(index), "end_seconds": float(index + 2), "timestamp_domain": "CLIP_LOCAL"}, "generation_source": "CLAIM_CONDITIONED_COARSE_RETRIEVAL", "status": "CANDIDATE_UNVERIFIED", "supporting_window_ids": [f"window_{index:02d}"], "retrieval_rank": index + 1, "coarse_support_margin": float(30 - index), "provenance": {}})
    hypotheses.append({"hypothesis_id": "hypothesis_null", "requirement_id": "requirement_x", "parent_claim_id": None, "claim_role": "TARGET_HYPOTHESIS", "target_event": "secure_the_base", "polarity": "NO_VISIBLE_EVENT", "candidate_interval": None, "generation_source": "CLAIM_CONDITIONED_COARSE_RETRIEVAL", "status": "CANDIDATE_UNVERIFIED", "supporting_window_ids": [], "retrieval_rank": None, "coarse_support_margin": None, "provenance": {}})
    graph = {"requirement_binding": {"requirement_id": "requirement_x"}, "nodes": copy.deepcopy(hypotheses), "observation_claim_count": 0, "verified_hypothesis_count": 0, "certificate_created": False}
    _write(root / "run" / "v2_stage3b_window_likelihoods.jsonl", likelihoods)
    _write(root / "run" / "v2_stage3b_window_ranking.jsonl", ranking)
    _write(root / "run" / "v2_stage3b_hypothesis_claims.jsonl", hypotheses)
    _write(root / "run" / "v2_stage3b_claim_graph.json", graph)
    _write(root / "v2_stage3b_run_summary.json", {"mode": "run", "planned_model_calls": 23, "new_model_calls": 23, "cache_hits": 0, "logical_model_calls": 23, "window_count": 23, "model_calls_made": 23})


class IndependentRepeatTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.a, self.b = self.root / "run_a", self.root / "run_b"; _stage3b_root(self.a); _stage3b_root(self.b)

    def tearDown(self): self.temp.cleanup()

    def compare(self, name="comparison"):
        return compare_independent_runs(run_a_dir=self.a, run_b_dir=self.b, output_dir=self.root / name)

    def mutate_json(self, path, callback):
        value = _read(path); callback(value); _write(path, value)

    def test_exact_reproduction_safe_summary_and_deterministic_outputs(self):
        result = self.compare("one")
        self.assertEqual(result["reproducibility_status"], "EXACT_NUMERIC_AND_DISCRETE_REPRODUCTION")
        self.assertTrue(result["ready_for_stage3c"])
        repeat = self.compare("two")
        self.assertEqual(result["comparison_content_sha256"], repeat["comparison_content_sha256"])
        self.assertEqual((self.root / "one" / "v2_stage3b_independent_repeat_comparison.json").read_bytes(), (self.root / "two" / "v2_stage3b_independent_repeat_comparison.json").read_bytes())
        summary = safe_candidate_summary(run_dir=self.a)
        self.assertEqual((len(summary["windows"]), len(summary["positive_hypotheses"]), len(summary["no_visible_event"])), (23, 5, 1))
        self.assertNotIn("temporal_gt_sentinel", canonical_json(summary))
        self.assertNotIn("relive.backends", inspect.getsource(compare_independent_runs))

    def test_numeric_variation_and_operational_fields(self):
        def change(rows): rows[0]["log_probabilities"]["A"] += 1e-9; rows[0]["support_margin"] += 1e-9; rows[0]["cache_hit"] = True; rows[0]["cache_key"] = "different"
        self.mutate_json(self.b / "run" / "v2_stage3b_window_likelihoods.jsonl", change)
        result = self.compare()
        self.assertEqual(result["status"], "PASS_WITH_NUMERIC_VARIATION")
        self.assertTrue(result["ready_for_stage3c"])

    def test_discrete_instability_cases(self):
        cases = {
            "argmax": lambda: self.mutate_json(self.b / "run" / "v2_stage3b_window_likelihoods.jsonl", lambda rows: rows[0]["log_probabilities"].update({"A": -9.0, "B": 9.0})),
            "ranking": lambda: self.mutate_json(self.b / "run" / "v2_stage3b_window_ranking.jsonl", lambda rows: rows.__setitem__(0, {**rows[0], "rank": 2})),
            "nms": lambda: self.mutate_json(self.b / "run" / "v2_stage3b_window_ranking.jsonl", lambda rows: rows.__setitem__(6, {**rows[6], "nms_status": "SUPPRESSED_CROSS_SCALE_DUPLICATE", "nms_suppressor_window_id": "window_00"})),
            "selected": lambda: self.mutate_json(self.b / "run" / "v2_stage3b_window_ranking.jsonl", lambda rows: rows.__setitem__(5, {**rows[5], "nms_status": "SELECTED"})),
            "interval": lambda: self.mutate_json(self.b / "run" / "v2_stage3b_hypothesis_claims.jsonl", lambda rows: rows[0]["candidate_interval"].update({"end_seconds": 99.0})),
            "null": lambda: self.mutate_json(self.b / "run" / "v2_stage3b_hypothesis_claims.jsonl", lambda rows: rows.pop()),
            "graph": lambda: self.mutate_json(self.b / "run" / "v2_stage3b_claim_graph.json", lambda graph: graph["nodes"][0]["supporting_window_ids"].append("window_21")),
        }
        for name, mutation in cases.items():
            with self.subTest(name=name):
                self.tearDown(); self.setUp(); mutation()
                result = self.compare(name)
                self.assertEqual(result["status"], "UNSTABLE")
                self.assertFalse(result["ready_for_stage3c"])

    def test_input_and_fresh_contract_fail_closed(self):
        for name, relative_path, mutation in (
            ("prompt", "v2_stage3b_prompt_spec.json", lambda value: value.update({"prompt_version": "different"})),
            ("packet", "v2_stage3b_visual_packets.jsonl", lambda value: value[0].update({"scale_seconds": 9.0})),
            ("model", "v2_stage3b_preflight.json", lambda value: value["model_fingerprint"]["scientific_identity"].update({"revision": "different"})),
        ):
            with self.subTest(name=name):
                self.tearDown(); self.setUp(); self.mutate_json(self.b / relative_path, mutation)
                result = self.compare(name)
                self.assertEqual((result["status"], result["reason"]), ("FAIL", "INDEPENDENT_RUN_INPUT_MISMATCH"))
        for name, mutation in (
            ("cache", lambda: self.mutate_json(self.b / "v2_stage3b_run_summary.json", lambda value: value.update({"cache_hits": 1}))),
            ("calls", lambda: self.mutate_json(self.b / "v2_stage3b_run_summary.json", lambda value: value.update({"new_model_calls": 22}))),
            ("missing", lambda: self.mutate_json(self.b / "run" / "v2_stage3b_window_likelihoods.jsonl", lambda rows: rows.pop())),
            ("duplicate", lambda: self.mutate_json(self.b / "run" / "v2_stage3b_window_likelihoods.jsonl", lambda rows: rows.append(copy.deepcopy(rows[0])))),
        ):
            with self.subTest(name=name):
                self.tearDown(); self.setUp(); mutation()
                result = self.compare(name)
                self.assertEqual((result["status"], result["reason"]), ("FAIL", "NOT_AN_INDEPENDENT_FRESH_RUN"))


if __name__ == "__main__": unittest.main()
