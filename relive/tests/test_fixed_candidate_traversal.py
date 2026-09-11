"""GPU-free Phase 2 fixed sliding-window traversal tests."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from relive.acquisition import acquire
from relive.config import load_config, validate_config
from relive.data.schemas import FIELD_SOURCES, load_runtime
from relive.runner import run
from relive.traversal import audit_fixed_candidate_pool, build_fixed_candidate_pool


ROOT = Path(__file__).resolve().parents[1]


class FixedCandidateTraversalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.runtime = self._runtime()

    def tearDown(self):
        self.temp.cleanup()

    def _runtime(self) -> Path:
        frames = []
        for order in range(6):
            image = Image.new("RGB", (32, 24), (20 + order * 20, 50, 120))
            ImageDraw.Draw(image).rectangle((4, 3, 20, 18), fill=(230, 80 + order, 30))
            path = self.root / f"frame-{order}.png"
            image.save(path)
            frames.append({"frame_id": f"f{order}", "path": str(path), "order": order})
        row = {"sample_id": "phase2-case", "task": "claim_verification", "question": "What is visible?",
               "frames": frames, "target_claim": {"claim_id": "phase2-claim", "text": "A visible object is present."},
               "metadata": {"dataset_name": "synthetic-public"}}
        path = self.root / "runtime.jsonl"
        payload = json.dumps(row, separators=(",", ":")) + "\n"
        path.write_text(payload, encoding="utf-8")
        Path(str(path) + ".provenance.json").write_text(json.dumps({
            "schema_version": "relive-runtime-v1", "source_kind": "synthetic",
            "runtime_sha256": hashlib.sha256(payload.encode()).hexdigest(), "field_sources": FIELD_SOURCES,
        }), encoding="utf-8")
        return path

    def config(self) -> dict:
        config = deepcopy(load_config(ROOT / "configs" / "mock_smoke.yaml"))
        config["claims"]["contrast_fixtures"] = []
        config["policy"]["name"] = "semantic_spatial"
        config["acquisition"].update(method="sliding_windows", window_size=3, stride=2, max_candidates=4)
        config["adaptation"]["enabled"] = False
        config["traversal"]["mode"] = "fixed_sliding_window_pool"
        config["budget"].update(max_calls=100, max_candidates=4, max_spatial_proposals=4, max_rounds=4)
        return validate_config(config)

    def candidate_ids(self, config):
        sample = load_runtime(self.runtime)[0]
        return [candidate.candidate_id for candidate in acquire(sample, config["acquisition"])]

    def semantic_rule(self, candidate_id, status, variant="ORIGINAL"):
        return {"match": {"stage": "semantic", "candidate_id": candidate_id, "variant": variant},
                "response": {"status": status}}

    def record(self, output):
        return json.loads(next((output / "samples").glob("*.json")).read_text())

    def test_pool_order_is_deterministic_and_gt_free(self):
        config = self.config()
        samples = load_runtime(self.runtime)
        _, first, first_hash = build_fixed_candidate_pool(samples, config)
        _, second, second_hash = build_fixed_candidate_pool(samples, config)
        self.assertEqual(first, second)
        self.assertEqual(first_hash, second_hash)
        self.assertEqual([row["candidate_rank"] for row in first], [0, 1, 2])
        self.assertTrue(first[-1]["tail_short_window"])
        self.assertEqual(audit_fixed_candidate_pool(first, first_hash)["status"], "PASS")
        self.assertNotIn("reference_answer", json.dumps(first))
        self.assertNotIn("temporal_gt", json.dumps(first))

    def test_uncertain_advances_and_verified_early_exits(self):
        config = self.config()
        ids = self.candidate_ids(config)
        config["backend"]["rules"].append(self.semantic_rule(ids[0], "INSUFFICIENT", "KEEP_TARGET"))
        output = self.root / "uncertain-next"
        run(config, self.runtime, output, max_samples=1)
        record = self.record(output)
        trace = record["candidate_traversal"]["trace"]
        self.assertEqual([row["candidate_rank"] for row in trace], [0, 1])
        self.assertEqual(trace[0]["outer_loop_state"], "CANDIDATE_UNCERTAIN")
        self.assertEqual(trace[1]["outer_loop_state"], "VERIFIED_FOUND")
        self.assertTrue(trace[1]["accepted"])
        self.assertEqual(record["termination_reason"], "VERIFIED_FOUND")
        self.assertEqual(record["strict"]["certificate_ids"], [trace[1]["certificate_id"]])

    def test_rejected_and_original_insufficient_advance(self):
        for status, expected in (("CONTRADICTED", "CANDIDATE_REJECTED"), ("INSUFFICIENT", "ORIGINAL_INSUFFICIENT")):
            with self.subTest(status=status):
                config = self.config()
                ids = self.candidate_ids(config)
                config["backend"]["rules"].append(self.semantic_rule(ids[0], status))
                output = self.root / status.lower()
                run(config, self.runtime, output, max_samples=1)
                trace = self.record(output)["candidate_traversal"]["trace"]
                self.assertEqual(trace[0]["outer_loop_state"], expected)
                self.assertEqual(trace[1]["outer_loop_state"], "VERIFIED_FOUND")

    def test_pool_exhaustion_and_max_candidate_budget_are_distinct(self):
        config = self.config()
        for candidate_id in self.candidate_ids(config):
            config["backend"]["rules"].append(self.semantic_rule(candidate_id, "INSUFFICIENT"))
        output = self.root / "exhausted"
        run(config, self.runtime, output, max_samples=1)
        record = self.record(output)
        self.assertEqual(record["termination_reason"], "CANDIDATE_POOL_EXHAUSTED")
        self.assertEqual(len(record["candidate_traversal"]["trace"]), 3)
        limited = self.config()
        limited["budget"]["max_candidates"] = 1
        for candidate_id in self.candidate_ids(limited):
            limited["backend"]["rules"].append(self.semantic_rule(candidate_id, "INSUFFICIENT"))
        limited = validate_config(limited)
        run(limited, self.runtime, self.root / "limited", max_samples=1)
        self.assertEqual(self.record(self.root / "limited")["termination_reason"], "MAX_CANDIDATES")

    def test_candidate_is_never_visited_twice_and_replay_uses_cache(self):
        config = self.config()
        ids = self.candidate_ids(config)
        config["backend"]["rules"].append(self.semantic_rule(ids[0], "INSUFFICIENT", "KEEP_TARGET"))
        cache = self.root / "cache"
        first = run(config, self.runtime, self.root / "first", cache_dir=cache, max_samples=1)
        replay = run(config, self.runtime, self.root / "replay", cache_dir=cache, max_samples=1)
        self.assertGreater(first["new_model_calls"], 0)
        self.assertEqual(replay["new_model_calls"], 0)
        self.assertGreater(replay["cache_hits"], 0)
        trace = self.record(self.root / "replay")["candidate_traversal"]["trace"]
        self.assertEqual(len({row["candidate_id"] for row in trace}), len(trace))

    def test_changed_window_config_changes_manifest_hash(self):
        config = self.config()
        _, _, original_hash = build_fixed_candidate_pool(load_runtime(self.runtime), config)
        changed = self.config()
        changed["acquisition"]["stride"] = 1
        changed = validate_config(changed)
        _, _, changed_hash = build_fixed_candidate_pool(load_runtime(self.runtime), changed)
        self.assertNotEqual(original_hash, changed_hash)

    def test_fixed_mode_rejects_non_sliding_or_enabled_adaptation(self):
        config = self.config()
        config["acquisition"]["method"] = "uniform"
        with self.assertRaisesRegex(ValueError, "requires acquisition.method"):
            validate_config(config)
        config = self.config()
        config["adaptation"]["enabled"] = True
        with self.assertRaisesRegex(ValueError, "requires adaptation.enabled=false"):
            validate_config(config)

    def test_benchmark_forced_fallback_stays_outside_strict_evidence(self):
        config = self.config()
        config["reasoning"]["mode"] = "benchmark_forced"
        for candidate_id in self.candidate_ids(config):
            config["backend"]["rules"].append(self.semantic_rule(candidate_id, "INSUFFICIENT"))
        output = self.root / "forced"
        run(config, self.runtime, output, max_samples=1)
        record = self.record(output)
        self.assertEqual(record["strict"]["status"], "ABSTAIN")
        self.assertTrue(record["benchmark_forced"]["fallback_used"])
        self.assertEqual(record["benchmark_forced"]["certificate_ids"], [])


if __name__ == "__main__":
    unittest.main()
