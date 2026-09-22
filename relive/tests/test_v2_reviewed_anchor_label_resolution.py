"""GPU-free tests for explicit duplicate-grounding canonical labels."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from relive.storage.artifacts import canonical_json, stable_hash
from relive.v2.reviewed_anchor_formal_intervention import (
    ReviewedAnchorInterventionError, prepare_warning_adjudication_template,
    validate_label_resolution_manifest,
)
from relive.v2.reviewed_anchor_label_resolution import (
    ReviewedAnchorLabelResolutionError, apply_canonical_labels, prepare_canonical_label_template,
)


def write_rows(path: Path, rows) -> None:
    path.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")


def raw(anchor: str, frame: str) -> dict:
    return {"anchor_candidate_id": anchor, "spatial_grounding_task_id": "task-" + anchor,
            "observation_claim_id": "obs-" + anchor, "observation_claim": "visible public action",
            "observation_role": "ACTION_CORE_PRESSING", "geometry_type": "DYNAMIC_SUPPORT_TUBE",
            "required_component_roles": ["BASE_PLATE"], "contextual_requirements": [],
            "image_path": "/tmp/public-frame-not-opened.png", "frame_sha256": frame,
            "timestamp_seconds": 1.0, "source_frame_reference": 1,
            "components": [{"role": "BASE_PLATE", "visibility": "VISIBLE", "bbox_2d_raw": [100, 100, 400, 400],
                            "bbox_normalized_xyxy": [.1, .1, .4, .4], "bbox_pixel_xyxy": [10, 10, 40, 40]}]}


class ReviewedAnchorLabelResolutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        raws, reviews = [], []
        for index in range(75):
            # Five pairs share exactly the same grounding signature but have a
            # deliberately inconsistent initial review label.
            pair = index // 2 if index < 10 else index
            frame = f"{pair:064x}"
            anchor = f"anchor-{index:02d}"
            raws.append(raw(anchor, frame))
            label = "BOX_TOO_BROAD" if index < 10 and index % 2 else "ACCEPTED"
            reviews.append({"anchor_candidate_id": anchor, "role": "BASE_PLATE", "model_visibility": "VISIBLE",
                            "human_overlay_label": label, "reason_code": ""})
        self.raw, self.review = self.root / "raw.jsonl", self.root / "review.jsonl"
        write_rows(self.raw, raws); write_rows(self.review, reviews)
        # Create the source warning queue through the existing adjudicator.
        from relive.v2.human_overlay_adjudication import adjudicate
        source = self.root / "source"
        adjudicate(raw_anchor_manifest=self.raw, review_jsonl=self.review, output_dir=source)
        self.queue = source / "schema_issue_candidates.jsonl"
        self.v1 = self.root / "v1.jsonl"
        prepare_warning_adjudication_template(self.queue, self.v1)
        v1_rows = [json.loads(line) for line in self.v1.read_text().splitlines()]
        for row in v1_rows:
            row.update(decision="UNIFY_LABELS", reviewer_id="reviewer-cora-01", rationale="Canonical label selected after overlay review.")
        write_rows(self.v1, v1_rows)

    def tearDown(self):
        self.temp.cleanup()

    def _canonical(self) -> Path:
        canonical = self.root / "canonical.jsonl"
        prepare_canonical_label_template(warning_adjudication=self.v1, output_path=canonical)
        rows = [json.loads(line) for line in canonical.read_text().splitlines()]
        for row in rows:
            row["resolved_label"] = "ACCEPTED"
        write_rows(canonical, rows)
        return canonical

    def test_explicit_labels_create_derived_review_and_recomputed_artifacts(self):
        original_raw, original_review, original_v1 = self.raw.read_bytes(), self.review.read_bytes(), self.v1.read_bytes()
        result = apply_canonical_labels(raw_grounding=self.raw, human_review=self.review, warning_queue=self.queue,
            warning_adjudication=self.v1, canonical_label_adjudication=self._canonical(), output_dir=self.root / "resolution")
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(self.raw.read_bytes(), original_raw)
        self.assertEqual(self.review.read_bytes(), original_review)
        self.assertEqual(self.v1.read_bytes(), original_v1)
        derived = [json.loads(line) for line in Path(result["derived_human_review"]).read_text().splitlines()]
        self.assertEqual(sum(row["reason_code"] == "CANONICAL_DUPLICATE_GROUNDING_LABEL_APPLIED" for row in derived), 10)
        manifest = json.loads(Path(result["resolution_manifest"]).read_text())
        self.assertEqual(manifest["canonical_label_count"], 5)
        self.assertEqual(manifest["review_rows_targeted"], 10)
        self.assertEqual(manifest["review_rows_changed"], 5)
        self.assertEqual(manifest["model_calls_made"], 0)
        self.assertFalse(manifest["certificate_created"])

    def test_prose_cannot_substitute_for_explicit_label_or_mismatch_source(self):
        canonical = self._canonical()
        rows = [json.loads(line) for line in canonical.read_text().splitlines()]
        rows[0]["resolved_label"] = ""
        write_rows(canonical, rows)
        with self.assertRaisesRegex(ReviewedAnchorLabelResolutionError, "CANONICAL_LABEL"):
            apply_canonical_labels(raw_grounding=self.raw, human_review=self.review, warning_queue=self.queue,
                warning_adjudication=self.v1, canonical_label_adjudication=canonical, output_dir=self.root / "bad")

    def test_deterministic_recomputed_outputs(self):
        canonical = self._canonical()
        first = apply_canonical_labels(raw_grounding=self.raw, human_review=self.review, warning_queue=self.queue,
            warning_adjudication=self.v1, canonical_label_adjudication=canonical, output_dir=self.root / "one")
        second = apply_canonical_labels(raw_grounding=self.raw, human_review=self.review, warning_queue=self.queue,
            warning_adjudication=self.v1, canonical_label_adjudication=canonical, output_dir=self.root / "two")
        # The derived review is path-independent.  The resolution manifest
        # deliberately binds the recomputed report byte hash, whose report
        # records its immutable output directory, so it is path-specific.
        self.assertEqual((self.root / "one" / "derived_human_overlay_review.jsonl").read_bytes(),
                         (self.root / "two" / "derived_human_overlay_review.jsonl").read_bytes())
        self.assertEqual(first["recomputed_route_counts"], second["recomputed_route_counts"])

    def test_resolution_manifest_binds_every_recomputed_input(self):
        result = apply_canonical_labels(raw_grounding=self.raw, human_review=self.review, warning_queue=self.queue,
            warning_adjudication=self.v1, canonical_label_adjudication=self._canonical(), output_dir=self.root / "resolution")
        root = Path(result["output_dir"]); recomputed = Path(result["recomputed_output_dir"])
        manifest = root / "canonical_label_resolution_manifest.json"
        validate_label_resolution_manifest(manifest,
            raw_grounding_sha256=hashlib.sha256(self.raw.read_bytes()).hexdigest(),
            derived_human_review_sha256=hashlib.sha256((root / "derived_human_overlay_review.jsonl").read_bytes()).hexdigest(),
            validation_report_sha256=hashlib.sha256((recomputed / "human_review_validation_report.json").read_bytes()).hexdigest(),
            observation_decisions_sha256=hashlib.sha256((recomputed / "observation_anchor_decisions.jsonl").read_bytes()).hexdigest(),
            eligible_manifest_sha256=hashlib.sha256((recomputed / "eligible_anchor_manifest.jsonl").read_bytes()).hexdigest(),
            warning_queue_sha256=hashlib.sha256(self.queue.read_bytes()).hexdigest(),
            warning_adjudication_sha256=hashlib.sha256(self.v1.read_bytes()).hexdigest())
        payload = json.loads(manifest.read_text())
        payload["recomputed_artifact_sha256"]["eligible_anchor_manifest.jsonl"] = "0" * 64
        checksum = dict(payload); checksum.pop("manifest_content_sha256", None)
        payload["manifest_content_sha256"] = stable_hash(checksum)
        manifest.write_text(canonical_json(payload) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ReviewedAnchorInterventionError, "RECOMPUTED_BINDING"):
            validate_label_resolution_manifest(manifest,
                raw_grounding_sha256=hashlib.sha256(self.raw.read_bytes()).hexdigest(),
                derived_human_review_sha256=hashlib.sha256((root / "derived_human_overlay_review.jsonl").read_bytes()).hexdigest(),
                validation_report_sha256=hashlib.sha256((recomputed / "human_review_validation_report.json").read_bytes()).hexdigest(),
                observation_decisions_sha256=hashlib.sha256((recomputed / "observation_anchor_decisions.jsonl").read_bytes()).hexdigest(),
                eligible_manifest_sha256=hashlib.sha256((recomputed / "eligible_anchor_manifest.jsonl").read_bytes()).hexdigest(),
                warning_queue_sha256=hashlib.sha256(self.queue.read_bytes()).hexdigest(),
                warning_adjudication_sha256=hashlib.sha256(self.v1.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
