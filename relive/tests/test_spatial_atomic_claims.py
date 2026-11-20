from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from relive.spatial_reliability.atomic_claims import (
    ATOMIC_CLAIM_INSTRUCTION,
    atomic_claim_prompt,
    atomic_claim_static_checks,
    generate_atomic_claim_candidates,
    parse_atomic_claim,
)
from relive.spatial_reliability.common import SpatialReliabilityError, stable_hash


class SpatialAtomicClaimTests(unittest.TestCase):
    def _fixture(self, root: Path, response: str = "A knife is visible."):
        source_row = {
            "id": "1.0$video$1$1",
            "qa_type": "region_caption_gpt",
            "dataset_name": "fixture",
            "metadata": {"video_id": "video"},
            "conversations": [
                {"from": "human", "value": "Describe the marked region."},
                {"from": "gpt", "value": "The knife advances along the tissue margin."},
            ],
        }
        source = root / "source.json"
        source.write_text(json.dumps([source_row]), encoding="utf-8")
        runtime_row = {
            "format": "relive-spatial-rc-dataset-v1",
            "sample_id": "relive-spatial-fixture",
            "source_record_index": 0,
            "source_record_sha256": stable_hash(source_row),
            "dataset_name": "fixture",
            "video_id": "video",
            "claim_id": "claim-old",
            "claim_text": "The knife advances along the tissue margin.",
            "split": "development",
        }
        runtime = root / "runtime.jsonl"
        runtime.write_text(json.dumps(runtime_row) + "\n", encoding="utf-8")
        config = root / "config.json"
        config.write_text(json.dumps({
            "backend": {
                "kind": "mock",
                "rules": [{
                    "match": {"stage": "relive_spatial_atomic_claim_generation"},
                    "response": response,
                }],
            },
        }), encoding="utf-8")
        return source, runtime, config

    def test_prompt_is_caption_only_and_contract_is_explicit(self):
        prompt = atomic_claim_prompt("The knife advances along the margin.")
        self.assertIn("The knife advances along the margin.", prompt)
        self.assertIn("exactly EXCLUDED", prompt)
        self.assertIn("Do not return JSON", prompt)
        for forbidden in ("gt_bbox", "certificate", "DEPENDENCY_VERIFIED", "oracle_manifest"):
            self.assertNotIn(forbidden, prompt)
        self.assertTrue(ATOMIC_CLAIM_INSTRUCTION.startswith("Given the source region caption"))

    def test_strict_plain_text_parser(self):
        self.assertEqual(parse_atomic_claim("A knife is visible.")["status"], "PROPOSED")
        self.assertEqual(parse_atomic_claim("EXCLUDED")["status"], "EXCLUDED")
        for raw in (
            "```A knife is visible.```",
            '"A knife is visible."',
            "Claim: A knife is visible.",
            "A knife is visible.\nExplanation",
            "A knife is visible. It is metallic.",
            '{"claim":"A knife is visible."}',
        ):
            with self.subTest(raw=raw):
                self.assertEqual(parse_atomic_claim(raw)["status"], "PARSE_FAILURE")

    def test_static_checks_flag_temporal_action(self):
        checks = atomic_claim_static_checks(
            "The knife advances during the procedure.",
            "The knife advances during the procedure.",
        )
        self.assertEqual(checks["status"], "FLAGGED_FOR_HUMAN_REVIEW")
        self.assertIn("advances", checks["temporal_or_procedural_term_hits"])
        self.assertIn("during", checks["temporal_or_procedural_term_hits"])
        self.assertTrue(checks["source_content_token_overlap"])

    def test_fake_run_writes_review_only_artifacts_and_exact_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, runtime, config = self._fixture(root)
            source_before, runtime_before = source.read_bytes(), runtime.read_bytes()
            run = generate_atomic_claim_candidates(
                runtime_manifest=runtime, source_json=source, config_path=config,
                cache_dir=root / "cache", output_dir=root / "run", expected_count=1,
                execution_mode="run",
            )
            replay = generate_atomic_claim_candidates(
                runtime_manifest=runtime, source_json=source, config_path=config,
                cache_dir=root / "cache", output_dir=root / "replay", expected_count=1,
                execution_mode="replay",
            )
            self.assertEqual(run["model_calls_made"], 1)
            self.assertEqual(replay["model_calls_made"], 0)
            self.assertEqual(replay["cache_hits"], 1)
            self.assertFalse(run["runtime_manifest_modified"])
            self.assertFalse(run["certificate_created"])
            self.assertEqual(run["new_verified_count"], 0)
            self.assertEqual(source.read_bytes(), source_before)
            self.assertEqual(runtime.read_bytes(), runtime_before)
            for output in (root / "run", root / "replay"):
                for name in (
                    "atomic_claim_candidates.jsonl",
                    "atomic_claim_human_review_list.jsonl",
                    "atomic_claim_generation_summary.json",
                    "atomic_claim_generation_audit.json",
                    "README.md",
                ):
                    self.assertTrue((output / name).is_file(), name)
            review = json.loads((root / "run/atomic_claim_human_review_list.jsonl").read_text())
            self.assertEqual(review["generated_atomic_claim"], "A knife is visible.")
            self.assertEqual(review["human_review"]["decision"], "")
            self.assertNotIn("gt_bbox", json.dumps(review))

    def test_source_tamper_and_count_mismatch_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, runtime, config = self._fixture(root)
            with self.assertRaisesRegex(SpatialReliabilityError, "COHORT_COUNT_MISMATCH"):
                generate_atomic_claim_candidates(
                    runtime_manifest=runtime, source_json=source, config_path=config,
                    cache_dir=root / "cache", output_dir=root / "out", expected_count=2,
                )
            changed = json.loads(source.read_text())
            changed[0]["conversations"][1]["value"] = "Tampered caption."
            source.write_text(json.dumps(changed))
            with self.assertRaisesRegex(SpatialReliabilityError, "SOURCE_BINDING_INVALID"):
                generate_atomic_claim_candidates(
                    runtime_manifest=runtime, source_json=source, config_path=config,
                    cache_dir=root / "cache", output_dir=root / "out2", expected_count=1,
                )

    def test_replay_cache_miss_is_fail_closed_before_model_call(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, runtime, config = self._fixture(root)
            with self.assertRaisesRegex(SpatialReliabilityError, "REPLAY_CACHE_MISS"):
                generate_atomic_claim_candidates(
                    runtime_manifest=runtime, source_json=source, config_path=config,
                    cache_dir=root / "empty-cache", output_dir=root / "replay",
                    expected_count=1, execution_mode="replay",
                )
            self.assertFalse((root / "replay").exists())


if __name__ == "__main__":
    unittest.main()
