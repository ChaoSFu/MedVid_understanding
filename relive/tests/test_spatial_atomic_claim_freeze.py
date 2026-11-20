from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from relive.spatial_reliability.atomic_claim_freeze import freeze_reviewed_atomic_claims
from relive.spatial_reliability.atomic_claims import ATOMIC_CLAIM_FORMAT
from relive.spatial_reliability.common import SpatialReliabilityError


class SpatialAtomicClaimFreezeTests(unittest.TestCase):
    def _inputs(self, root: Path):
        runtime, candidates, reviews = [], [], []
        for index in range(1, 33):
            sample_id = f"sample-{index:02d}"
            generated = f"Instrument {index} is visible."
            source_sha = f"{index:064x}"
            candidate_id = f"candidate-{index:02d}"
            split = "development" if index <= 20 else "frozen_evaluation"
            runtime.append({
                "format": "relive-spatial-rc-dataset-v1",
                "sample_id": sample_id,
                "source_record_index": index,
                "source_record_sha256": source_sha,
                "video_id": f"video-{index}",
                "dataset_name": "fixture",
                "frame_order": 0,
                "frame_path": f"/public/frame-{index}.jpg",
                "frame_sha256": None,
                "claim_id": f"old-claim-{index}",
                "claim_text": f"Old claim {index}.",
                "claim_scope": "LOCAL_ATOMIC",
                "split": split,
                "normalizer_version": "old",
                "scope_router_version": "old",
            })
            candidates.append({
                "format": ATOMIC_CLAIM_FORMAT,
                "candidate_id": candidate_id,
                "sample_id": sample_id,
                "source_record_index": index,
                "source_record_sha256": source_sha,
                "current_claim_id": f"old-claim-{index}",
                "current_claim_text": f"Old claim {index}.",
                "source_region_caption": f"Instrument {index} is visible in the region.",
                "candidate": {"status": "PROPOSED", "claim_text": generated,
                              "reason": "REVIEW_REQUIRED"},
            })
            reviews.append({
                "format": ATOMIC_CLAIM_FORMAT + "-human-review",
                "candidate_id": candidate_id,
                "sample_id": sample_id,
                "source_region_caption": f"Instrument {index} is visible in the region.",
                "generated_status": "PROPOSED",
                "generated_atomic_claim": generated,
            })
        paths = {}
        for name, rows in (("runtime", runtime), ("candidates", candidates), ("review", reviews)):
            path = root / f"{name}.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            paths[name] = path
        feedback = root / "feedback.json"
        feedback.write_text(json.dumps({
            "format": "relive-spatial-atomic-claim-author-feedback-v1",
            "expected_candidate_count": 32,
            "position_index_base": 1,
            "default_action": "ACCEPT_GENERATED",
            "exclusions": [
                {"position": 31, "rationale": "No non-invented atomic claim is available."},
            ],
            "human_feedback_source": "DIRECT_AUTHOR_VISUAL_REVIEW_2026-11-20",
            "reviewer_id": "AUTHOR_CONFIRMED_IN_CHAT",
            "replacements": [
                {"position": 5, "final_claim": "Forceps is visible.", "rationale": "confirmed"},
                {"position": 9, "final_claim": "The electric cautery is held by the right hand.", "rationale": "confirmed"},
                {"position": 16, "final_claim": "Forceps is visible.", "rationale": "confirmed"},
                {"position": 28, "final_claim": "Forceps is visible.", "rationale": "confirmed"},
            ],
        }))
        paths["feedback"] = feedback
        return paths

    def test_freezes_27_accepts_four_replacements_and_one_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._inputs(root)
            before = {name: path.read_bytes() for name, path in paths.items()}
            result = freeze_reviewed_atomic_claims(
                runtime_manifest=paths["runtime"],
                candidates_jsonl=paths["candidates"],
                review_jsonl=paths["review"],
                feedback_json=paths["feedback"],
                output_dir=root / "frozen",
            )
            self.assertEqual(result["status"], "FROZEN")
            self.assertEqual(result["decision_counts"], {
                "ACCEPT_GENERATED": 27,
                "AUTHOR_REPLACEMENT": 4,
                "EXCLUDED_NO_VALID_ATOMIC_CLAIM": 1,
            })
            self.assertEqual(result["final_split_counts"], {
                "curated_evaluation": 11, "development": 20,
            })
            self.assertEqual(result["source_candidate_count"], 32)
            self.assertEqual(result["final_runtime_count"], 31)
            self.assertFalse(result["former_frozen_evaluation_is_blind"])
            frozen = [json.loads(line) for line in
                      (root / "frozen/reviewed_atomic_claim_runtime_manifest.jsonl").read_text().splitlines()]
            self.assertEqual(frozen[4]["claim_text"], "Forceps is visible.")
            self.assertEqual(frozen[8]["claim_text"], "The electric cautery is held by the right hand.")
            self.assertEqual(frozen[15]["claim_text"], "Forceps is visible.")
            self.assertEqual(frozen[27]["claim_text"], "Forceps is visible.")
            self.assertEqual(frozen[0]["claim_text"], "Instrument 1 is visible.")
            self.assertEqual(frozen[20]["split"], "curated_evaluation")
            self.assertEqual(len(frozen), 31)
            self.assertNotIn("sample-31", {row["sample_id"] for row in frozen})
            audit = json.loads((root / "frozen/reviewed_atomic_claim_audit.json").read_text())
            self.assertEqual(audit["model_calls_made"], 0)
            self.assertEqual(audit["cache_writes"], 0)
            self.assertFalse(audit["certificate_created"])
            self.assertEqual(audit["new_verified_count"], 0)
            self.assertEqual(before, {name: path.read_bytes() for name, path in paths.items()})

    def test_binding_and_feedback_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._inputs(root)
            rows = [json.loads(line) for line in paths["review"].read_text().splitlines()]
            rows[4]["candidate_id"] = "wrong"
            paths["review"].write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(SpatialReliabilityError, "BINDING_INVALID:5"):
                freeze_reviewed_atomic_claims(
                    runtime_manifest=paths["runtime"], candidates_jsonl=paths["candidates"],
                    review_jsonl=paths["review"], feedback_json=paths["feedback"],
                    output_dir=root / "out",
                )


if __name__ == "__main__":
    unittest.main()
