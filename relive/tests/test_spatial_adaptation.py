"""GPU-free Phase 3 routing, provenance, and cache-isolation tests."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from relive.backends.base import Backend
from relive.certificate import build_certificate
from relive.claims import PROMPT_VERSIONS, prompt
from relive.spatial_adaptation import (MAX_SPATIAL_REFINEMENT_ROUNDS,
                                       SpatialAdaptationError, SpatialEvidenceAdaptationController, candidate_from_manifest,
                                       parse_refined_proposal, route_refinement)
from relive.storage.artifacts import ArtifactStore
from relive.storage.cache import Budget, CachedInference
from relive.types import (Claim, EvidenceCandidate, ExecutionStatus, Frame,
                          RuntimeSample, SemanticStatus, SpatialProposal,
                          VerificationResult)


class JsonBackend(Backend):
    synthetic = True

    def infer(self, request):
        self.calls += 1
        return json.dumps({"support_region": [100, 200, 400, 600],
                           "coordinate_system": "normalized_0_1000_xyxy"})


class SpatialAdaptationTests(unittest.TestCase):
    def setUp(self):
        self.claim = Claim("claim-1", "A local object is visible.")
        self.candidate = EvidenceCandidate("candidate-1", "sample-1", ("frame-1",), (None,), 0, "chronological")
        self.parent = SpatialProposal("proposal-r0", "candidate-1", "claim-1", (0.1, 0.1, 0.5, 0.5))

    def diagnosis(self, reasons, **extra):
        return {"original_failure_reasons": reasons, **extra}

    def test_reason_specific_routes_and_forbidden_states(self):
        self.assertEqual(route_refinement(self.diagnosis(["DEPENDENCE_UNRESOLVED"]))["stage"],
                         "spatial_refine_dependence")
        self.assertEqual(route_refinement(self.diagnosis(["KEEP_SUPPORT_LOST"]))["stage"],
                         "spatial_refine_sufficiency")
        geometry = self.diagnosis(["CONTROL_RESULT_COUNT_MISMATCH"],
                                  control_count_audit={"classification": "B_LEGITIMATE_PROTOCOL_UNAVAILABLE"})
        self.assertEqual(route_refinement(geometry)["routing_reason"], "CONTROL_GEOMETRY_UNAVAILABLE")
        unresolved = self.diagnosis(["CONTROL_RESULT_COUNT_MISMATCH"],
                                    control_count_audit={"classification": "UNRESOLVED"})
        self.assertFalse(route_refinement(unresolved)["allowed"])
        self.assertFalse(route_refinement(self.diagnosis(["ORIGINAL_INSUFFICIENT"]))["allowed"])
        self.assertFalse(route_refinement(self.diagnosis(["TECHNICAL_FAILURE"], root_cause_class="TECHNICAL_ERROR"))["allowed"])

    def test_r1_parser_binds_parent_and_rejects_non_integer_schema(self):
        route = route_refinement(self.diagnosis(["DEPENDENCE_UNRESOLVED"]))
        result = parse_refined_proposal(
            '{"support_region":[100,200,400,600],"coordinate_system":"normalized_0_1000_xyxy"}',
            self.candidate, self.claim, parent=self.parent, route=route, raw_response_ref="raw-r1")
        self.assertEqual(result.support_region, (0.1, 0.2, 0.4, 0.6))
        self.assertEqual(result.provenance["parent_proposal_id"], self.parent.proposal_id)
        self.assertEqual(result.provenance["spatial_round"], 1)
        controller = SpatialEvidenceAdaptationController()
        unchanged = controller.parse(
            '{"support_region":[100,100,500,500],"coordinate_system":"normalized_0_1000_xyxy"}',
            self.candidate, self.claim, parent=self.parent, route=route, raw_response_ref="raw-same")
        self.assertEqual(unchanged.parser_status, ExecutionStatus.PARSE_ERROR)
        self.assertEqual(unchanged.provenance["failure_reason"], "REFINEMENT_BBOX_UNCHANGED")
        invalid = parse_refined_proposal(
            '{"support_region":[100.0,200,400,600],"coordinate_system":"normalized_0_1000_xyxy"}',
            self.candidate, self.claim, parent=self.parent, route=route, raw_response_ref="raw-bad")
        self.assertEqual(invalid.parser_status, ExecutionStatus.PARSE_ERROR)

    def test_frozen_candidate_rejects_gt_shaped_fields_and_path_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp) / "public.png"
            Image.new("RGB", (10, 10), "purple").save(image)
            frame = Frame("frame-1", str(image), 0)
            sample = RuntimeSample("sample-1", "claim_verification", "q", (frame,), target_claim=self.claim)
            row = {"sample_id": "sample-1", "candidate_id": "candidate-1", "candidate_rank": 0,
                   "frame_ids": ["frame-1"], "source_frame_paths": [str(image)]}
            recovered = candidate_from_manifest(row, sample)
            self.assertEqual(recovered.frame_ids, ("frame-1",))
            with self.assertRaisesRegex(SpatialAdaptationError, "GT_SHAPED"):
                candidate_from_manifest({**row, "temporal_gt": [1, 2]}, sample)
            with self.assertRaisesRegex(SpatialAdaptationError, "SOURCE_PATH_MISMATCH"):
                candidate_from_manifest({**row, "source_frame_paths": ["wrong.png"]}, sample)

    def test_r0_r1_prompt_contexts_have_distinct_cache_identity_and_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "public.png"
            Image.new("RGB", (8, 8), "purple").save(image)
            cache = CachedInference(JsonBackend({"kind": "unit", "max_retries": 0}), ArtifactStore(root / "cache"))
            r0 = {"stage": "spatial_refine_dependence",
                  "prompt": prompt("spatial_refine_dependence", {"frame_ids": ["frame-1"], "timing": {},
                      "claim": {"text": self.claim.text, "entity": None, "time_scope": {}},
                      "previous_bbox_normalized_0_1_xyxy": [0.1, 0.1, 0.5, 0.5],
                      "previous_failure_reason": "DEPENDENCE_UNRESOLVED", "previous_original_semantic_status": "SUPPORTED",
                      "previous_keep_semantic_status": "SUPPORTED", "previous_drop_semantic_status": "SUPPORTED", "refinement_round": 1}),
                  "prompt_version": PROMPT_VERSIONS["spatial_refine_dependence"], "image_paths": [str(image)],
                  "frame_ids": ["frame-1"], "context": {"proposal_index": 0}}
            r1 = {**r0, "prompt": prompt("spatial_refine_dependence", {"frame_ids": ["frame-1"], "timing": {},
                      "claim": {"text": self.claim.text, "entity": None, "time_scope": {}},
                      "previous_bbox_normalized_0_1_xyxy": [0.2, 0.1, 0.5, 0.5],
                      "previous_failure_reason": "DEPENDENCE_UNRESOLVED", "previous_original_semantic_status": "SUPPORTED",
                      "previous_keep_semantic_status": "SUPPORTED", "previous_drop_semantic_status": "SUPPORTED", "refinement_round": 1}),
                  "context": {"proposal_index": 1}}
            self.assertNotEqual(cache.cache_key(r0), cache.cache_key(r1))
            first = Budget(4)
            cache.call(r1, first)
            self.assertEqual(first.new_calls, 1)
            replay = Budget(4)
            cache.call(r1, replay)
            self.assertEqual(replay.new_calls, 0)
            self.assertEqual(replay.cache_hits, 1)

    def test_certificate_builder_remains_sole_verified_authority(self):
        original = VerificationResult(SemanticStatus.SUPPORTED, ExecutionStatus.OK, "raw", "semantic", {
            "candidate_id": self.candidate.candidate_id, "claim_id": self.claim.claim_id,
            "frame_ids": list(self.candidate.frame_ids)})
        certificate = build_certificate(self.candidate, self.claim, original, "semantic_spatial", {
            "pass": False, "status": "DEPENDENCE_UNRESOLVED", "reasons": ["DEPENDENCE_UNRESOLVED"],
            "require_controls": True})
        self.assertEqual(certificate.final_status.value, "UNCERTAIN")
        self.assertEqual(MAX_SPATIAL_REFINEMENT_ROUNDS, 1)
        controller = SpatialEvidenceAdaptationController()
        terminal = controller.route(self.diagnosis(["DEPENDENCE_UNRESOLVED"]), current_round=1)
        self.assertFalse(terminal["allowed"])
        self.assertEqual(terminal["routing_reason"], "MAX_SPATIAL_REFINEMENT_ROUNDS")


if __name__ == "__main__":
    unittest.main()
