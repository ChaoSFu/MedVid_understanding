"""GPU-free Phase 3.5 claim-scope routing and prospective-cohort checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from phase35_claim_scope_audit import Phase35Error, run_phase35
from relive.claim_scope import (GLOBAL_DISTRIBUTED, LOCAL_ATOMIC, MULTI_SUPPORT_POSSIBLE,
                                UNRESOLVED_SCOPE, ClaimScopeError, route_claim_scope)
from relive.data.schemas import FIELD_SOURCES


class ClaimScopeTests(unittest.TestCase):
    def test_clear_scope_taxonomy_is_deterministic(self):
        cases = {
            "A dark metallic forceps jaw is visibly contacting tissue at the center-left.": LOCAL_ATOMIC,
            "At least one laparoscopic surgical instrument is visibly present in the supplied frames.": MULTI_SUPPORT_POSSIBLE,
            "The supplied frames visibly depict an open surgical procedure from an egocentric viewpoint.": GLOBAL_DISTRIBUTED,
            "The procedure is underway.": UNRESOLVED_SCOPE,
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                first = route_claim_scope(text, qa_type="public")
                second = route_claim_scope(text, qa_type="other")
                self.assertEqual(first.claim_scope, expected)
                self.assertEqual(first.as_dict(), second.as_dict())
                self.assertEqual(first.single_roi_certificate_applicable, expected == LOCAL_ATOMIC)
                self.assertFalse(first.gt_used)

    def test_router_rejects_gt_and_outcome_shaped_inputs(self):
        with self.assertRaisesRegex(ClaimScopeError, "prohibited"):
            route_claim_scope("A local forceps jaw contacts tissue.", task_metadata={"bbox": [0, 0, 1, 1]})
        with self.assertRaisesRegex(ClaimScopeError, "prohibited"):
            route_claim_scope("A local forceps jaw contacts tissue.", task_metadata={"certificate": "VERIFIED"})

    def _runtime(self, root: Path, name: str, rows: list[tuple[str, str]]) -> Path:
        runtime_rows = []
        for index, (sample_id, claim) in enumerate(rows):
            image = root / f"{name}-{index}.png"
            Image.new("RGB", (8, 8), "white").save(image)
            runtime_rows.append({"sample_id": sample_id, "task": "claim_verification", "question": "public question",
                                 "frames": [{"frame_id": f"{sample_id}:f", "path": str(image), "order": 0}],
                                 "target_claim": {"claim_id": f"{sample_id}:claim", "text": claim},
                                 "metadata": {"dataset_name": "public", "source_qa_type": "claim"}})
        payload = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in runtime_rows)
        runtime = root / f"{name}.jsonl"
        runtime.write_text(payload, encoding="utf-8")
        sha = hashlib.sha256(payload.encode()).hexdigest()
        Path(str(runtime) + ".provenance.json").write_text(json.dumps({"schema_version": "relive-runtime-v1", "source_kind": "public_runtime", "runtime_sha256": sha, "field_sources": FIELD_SOURCES}), encoding="utf-8")
        Path(str(runtime) + ".gt_isolation_audit.json").write_text(json.dumps({"status": "PASS", "runtime_sha256": sha, "adapter": "test"}), encoding="utf-8")
        return runtime

    def test_prospective_cohort_is_fresh_and_outcome_independent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            development = self._runtime(root, "development", [
                ("old-local", "A forceps jaw is visibly contacting tissue."),
                ("old-multi", "At least one laparoscopic surgical instrument is visibly present."),
                ("old-global", "The supplied frames visibly depict an open surgical procedure from an egocentric viewpoint."),
            ])
            future = self._runtime(root, "future", [
                ("new-global", "The supplied frames visibly depict an open surgical procedure from an egocentric viewpoint."),
                ("new-local", "A white specimen-retrieval bag is visibly open in the lower field."),
            ])
            phase2 = root / "phase2"
            phase2.mkdir()
            manifest = []
            for rank, sample_id in enumerate(("old-local", "old-multi", "old-global")):
                manifest.append({"sample_id": sample_id, "qa_id": sample_id, "candidate_id": f"candidate-{rank}", "candidate_rank": 0,
                                 "frame_ids": [f"{sample_id}:f"], "manifest_hash": "frozen"})
            (phase2 / "fixed_candidate_pool.jsonl").write_text("".join(json.dumps(row) + "\n" for row in manifest), encoding="utf-8")
            phase25 = root / "phase25.jsonl"
            # The artificial certificate field demonstrates that only IDs bind
            # the development cohort; router outcomes cannot depend on it.
            phase25.write_text("".join(json.dumps({"qa_id": row["sample_id"], "candidate_id": row["candidate_id"], "certificate_status": "VERIFIED"}) + "\n" for row in manifest), encoding="utf-8")
            output = root / "out"
            report = run_phase35(development_runtime=development, phase2_run=phase2, phase25_diagnostics=phase25,
                                 prospective_runtime=future, output_dir=output, max_prospective_samples=5)
            self.assertEqual(report["model_calls_made"], 0)
            frozen = json.loads((output / "prospective_local_atomic_manifest.json").read_text())
            self.assertEqual([row["sample_id"] for row in frozen["selected"]], ["new-local"])
            self.assertTrue(frozen["not_a_temporal_candidate_pool"])
            rows = [json.loads(line) for line in (output / "claim_scope_development_audit.jsonl").read_text().splitlines()]
            self.assertEqual({row["claim_scope"] for row in rows}, {LOCAL_ATOMIC, MULTI_SUPPORT_POSSIBLE, GLOBAL_DISTRIBUTED})

    def test_prospective_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = self._runtime(root, "same", [("old", "A forceps jaw is visibly contacting tissue.")])
            phase2 = root / "phase2"; phase2.mkdir()
            row = {"sample_id": "old", "qa_id": "old", "candidate_id": "c", "candidate_rank": 0, "frame_ids": ["old:f"], "manifest_hash": "frozen"}
            (phase2 / "fixed_candidate_pool.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            phase25 = root / "phase25.jsonl"; phase25.write_text(json.dumps({"qa_id": "old", "candidate_id": "c"}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(Phase35Error, "overlaps"):
                run_phase35(development_runtime=runtime, phase2_run=phase2, phase25_diagnostics=phase25,
                            prospective_runtime=runtime, output_dir=root / "out", max_prospective_samples=5)


if __name__ == "__main__":
    unittest.main()
