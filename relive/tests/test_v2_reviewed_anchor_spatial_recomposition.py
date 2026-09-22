"""GPU-free contracts for immutable R0 auditing and one-round composite R1."""
from __future__ import annotations

import json
from pathlib import Path

from tests.test_v2_reviewed_anchor_formal_intervention import (
    ReviewedAnchorFormalInterventionTests, write_json, write_jsonl,
)
from relive.config import load_config
from relive.interventions import apply_spatial_intervention_mask, generate_composite_control_regions
from relive.v2.reviewed_anchor_formal_intervention import execute as r0_execute
from relive.v2.reviewed_anchor_spatial_recomposition import (
    ReviewedAnchorRecompositionError, audit_r0_cohort, execute, preflight,
)


class ReviewedAnchorSpatialRecompositionTests(ReviewedAnchorFormalInterventionTests):
    def setUp(self):
        super().setUp()
        cfg = load_config(self.config)
        cfg["backend"]["rules"] = [
            {"match": {"stage": "semantic", "variant": "KEEP_TARGET"}, "response": {"status": "INSUFFICIENT"}},
            {"match": {"stage": "semantic", "variant": "DROP_TARGET"}, "response": {"status": "SUPPORTED"}},
            {"match": {"stage": "semantic", "variant": "DROP_MATCHED_CONTROL"}, "response": {"status": "SUPPORTED"}},
        ]
        write_json(self.config, cfg)
        rows = [json.loads(line) for line in self.eligible.read_text().splitlines()]
        for row in rows:
            row["observation_role"] = "ACTION_CORE_PRESSING"
            boxes = [[.10, .10, .20, .20], [.30, .10, .40, .20], [.50, .10, .60, .20]]
            for component, box in zip(row["adjudicated_components"], boxes):
                component["adjudicated_bbox"] = box
        write_jsonl(self.eligible, rows)

    def _r0(self):
        out = self.root / "r0"
        self.assertEqual(self._preflight(out, self._approved_adjudication())["status"], "PASS")
        r0_execute(output_dir=out, config_path=self.config, mode="run")
        r0_execute(output_dir=out, config_path=self.config, mode="replay")
        return out

    def _r1_preflight(self, r0, out):
        return preflight(r0_output_dir=r0, eligible_manifest=self.eligible, config_path=self.config,
                         output_dir=out, warning_queue=self.queue,
                         warning_adjudication=self.root / "ready.jsonl")

    def test_union_mask_preserves_sparse_geometry_and_matched_control(self):
        regions = [[.05, .05, .2, .2], [.55, .35, .75, .55], [.35, .6, .45, .9]]
        controls = generate_composite_control_regions(regions)
        self.assertTrue(controls["available"])
        self.assertEqual(len(controls["regions"][0]), 3)
        for before, after in zip(regions, controls["regions"][0]):
            self.assertAlmostEqual(after[2] - after[0], before[2] - before[0])
            self.assertAlmostEqual(after[3] - after[1], before[3] - before[1])
        from PIL import Image
        image = Image.new("RGB", (100, 100), (10, 20, 30))
        spec = {"operator": "opaque_gray", "operator_version": "relive-opaque-gray-hard-mask-v1", "parameters": {"fill_rgb": [127, 127, 127]}}
        _, audit = apply_spatial_intervention_mask(image, regions, "DROP_TARGET", spec)
        self.assertTrue(audit["pixel_audit_pass"])
        self.assertEqual(audit["geometry_type"], "COMPOSITE_BINARY_MASK_UNION")

    def test_complete_r0_routes_exactly_once_then_r1_run_and_replay(self):
        r0 = self._r0(); r1 = self.root / "r1"
        result = self._r1_preflight(r0, r1)
        self.assertEqual(result["candidate_count"], 5)
        plan = json.loads((r1 / "reviewed_anchor_r1_recomposition_plan.json").read_text())
        self.assertTrue(all(row["component_roles"] == ["OPERATOR_HAND", "BASE_PLATE", "HAND_BASE_INTERFACE"] for row in plan["candidates"]))
        self.assertNotIn("ACCEPTED", json.dumps(plan))
        first = execute(output_dir=r1, config_path=self.config, mode="run")
        replay = execute(output_dir=r1, config_path=self.config, mode="replay")
        self.assertEqual(first["summary"]["new_model_calls"], 20)
        self.assertEqual(replay["summary"]["new_model_calls"], 0)
        traces = [json.loads(line) for line in (r1 / "run" / "reviewed_anchor_r1_trace.jsonl").read_text().splitlines()]
        self.assertTrue(all(row["spatial"]["composition"]["geometry_type"] == "COMPOSITE_BINARY_MASK_UNION" for row in traces))
        self.assertTrue(all(len(row["spatial"]["pixel_audit_refs"]) == 4 for row in traces))
        self.assertTrue(all(row["certificate"]["certificate_status"] == "NOT_APPLICABLE_SYNTHETIC_TEST" for row in traces))

    def test_incomplete_r0_blocks_r1_and_unify_without_derived_recompute_is_diagnostic_only(self):
        r0 = self.root / "incomplete"
        self._preflight(r0, self._approved_adjudication())
        r0_execute(output_dir=r0, config_path=self.config, mode="run")
        with self.assertRaisesRegex(ReviewedAnchorRecompositionError, "R0_COHORT_INCOMPLETE"):
            self._r1_preflight(r0, self.root / "blocked")
        # Full R0 plus an UNIFY decision cannot masquerade as a re-labelled,
        # re-routed input because the historical adjudication schema has no
        # canonical resolved_label field.
        r0_execute(output_dir=r0, config_path=self.config, mode="replay")
        rows = [json.loads(line) for line in (self.root / "ready.jsonl").read_text().splitlines()]
        rows[0]["decision"] = "UNIFY_LABELS"
        unify = self.root / "unify.jsonl"; write_jsonl(unify, rows)
        # The R0 plan must itself bind the reviewed adjudication file; a later
        # substitution is rejected rather than used to rewrite its meaning.
        r0_unify = self.root / "r0-unify"
        self._preflight(r0_unify, unify)
        r0_execute(output_dir=r0_unify, config_path=self.config, mode="run")
        r0_execute(output_dir=r0_unify, config_path=self.config, mode="replay")
        audit = audit_r0_cohort(r0_output_dir=r0_unify, output_dir=self.root / "audit", warning_queue=self.queue, warning_adjudication=unify)
        self.assertEqual(audit["unify_labels_audit"]["status"], "DIAGNOSTIC_ONLY_INPUT_ADJUDICATION_NOT_APPLIED")
        with self.assertRaisesRegex(ReviewedAnchorRecompositionError, "DIAGNOSTIC_ONLY_INPUT_ADJUDICATION_NOT_APPLIED"):
            preflight(r0_output_dir=r0_unify, eligible_manifest=self.eligible, config_path=self.config, output_dir=self.root / "blocked2",
                      warning_queue=self.queue, warning_adjudication=unify)

    def test_template_approved_preflight_run_replay_and_no_label_prompt_leakage(self):
        # This class deliberately changes the synthetic DROP verdict to create
        # the pre-registered R1 condition; the inherited R0 expectation is
        # exercised unchanged in its source test class above.
        pass
