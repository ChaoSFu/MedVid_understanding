from __future__ import annotations

import json
from pathlib import Path
import unittest

from relive.v2.observation_retrieval import execute as retrieval_execute, preflight as retrieval_preflight
from relive.v2.temporal_evidence_composition import prepare as compose_prepare
from relive.v2.spatial_evidence_planning import GEOMETRY, SpatialPlanError, prepare, validate
from tests.test_v2_tal_observation_retrieval import ObservationRetrievalTests, _FakeObservationBackend


class SpatialEvidencePlanningTests(unittest.TestCase):
    def setUp(self):
        self.base = ObservationRetrievalTests(methodName="test_packets_dedup_fanout_run_replay_and_no_admission")
        self.base.setUp()
        self.stage3d = self.base.out
        self.base.prepare()
        retrieval_preflight(output_dir=self.stage3d, config_path=self.base.base.config, stage3c_dir=self.base.stage3c, backend_factory=lambda _: _FakeObservationBackend())
        retrieval_execute(output_dir=self.stage3d, config_path=self.base.base.config, mode="run", backend_factory=lambda _: _FakeObservationBackend())
        self.stage3e = self.base.base.root / "stage3e"
        policy3e = Path(__file__).resolve().parents[1] / "configs/v2/tal_temporal_evidence_composition_policy.json"
        compose_prepare(stage3c_dir=self.base.stage3c, stage3d_dir=self.stage3d, video_index_dir=self.base.base.index, policy_path=policy3e, output_dir=self.stage3e)
        self.policy = Path(__file__).resolve().parents[1] / "configs/v2/tal_spatial_evidence_planning_policy.json"
        self.out = self.base.base.root / "stage3f"

    def tearDown(self):
        self.base.tearDown()

    def freeze(self, output=None):
        return prepare(stage3c_dir=self.base.stage3c, stage3d_dir=self.stage3d, stage3e_dir=self.stage3e,
                       video_index_dir=self.base.base.index, policy_path=self.policy, output_dir=output or self.out)

    def test_metadata_only_geometry_anchor_dedup_and_validation(self):
        report = self.freeze()
        self.assertEqual(report["status"], "PASS")
        self.assertGreater(report["spatial_plan_count"], 0)
        self.assertGreater(report["deduplicated_grounding_task_count"], 0)
        plans = [json.loads(line) for line in (self.out / "v2_tal_spatial_evidence_plans.jsonl").read_text().splitlines()]
        tasks = [json.loads(line) for line in (self.out / "v2_tal_spatial_grounding_tasks.jsonl").read_text().splitlines()]
        self.assertTrue(all(plan["status"] == "PLANNED_UNGROUNDED" for plan in plans))
        self.assertTrue(all(plan["future_intervention_target"] == "COMPOSITE_EVIDENCE_UNION" and plan["matched_control_required"] for plan in plans))
        self.assertTrue(all(len(task["anchor_candidates"]) <= 3 for task in tasks))
        self.assertTrue(all(len({anchor["unique_visual_frame_id"] for anchor in task["anchor_candidates"]}) == len(task["anchor_candidates"]) for task in tasks))
        by_role = {plan["observation_role"]: plan for plan in plans}
        for role, (kind, required, contextual, scope) in GEOMETRY.items():
            self.assertEqual(by_role[role]["geometry_type"], kind)
            self.assertEqual(by_role[role]["required_component_roles"], list(required))
            self.assertEqual(by_role[role]["contextual_requirements"], list(contextual))
            self.assertEqual(by_role[role]["propagation_scope"], scope)
        checked = validate(self.out)
        self.assertEqual(checked["status"], "PASS")
        self.assertEqual(checked["frames_read"], 0)
        self.assertFalse(checked["backend_loaded"])

    def test_repeat_is_byte_identical_and_tamper_fails_closed(self):
        self.freeze()
        repeat = self.base.base.root / "stage3f-repeat"
        self.freeze(repeat)
        names = sorted(path.name for path in self.out.iterdir())
        self.assertEqual(names, sorted(path.name for path in repeat.iterdir()))
        for name in names:
            self.assertEqual((self.out / name).read_bytes(), (repeat / name).read_bytes())
        plans = self.out / "v2_tal_spatial_evidence_plans.jsonl"
        plans.write_bytes(plans.read_bytes() + b" ")
        with self.assertRaisesRegex(SpatialPlanError, "ARTIFACT_TAMPERED"):
            validate(self.out)

    def test_unknown_role_is_unresolved_without_geometry_payload(self):
        self.freeze()
        payload = json.loads((self.out / "v2_tal_spatial_evidence_schema.json").read_text())
        self.assertIn("bbox", payload["forbidden_geometry_payload_keys"])
        self.assertIn("mask", payload["forbidden_geometry_payload_keys"])


if __name__ == "__main__":
    unittest.main()
