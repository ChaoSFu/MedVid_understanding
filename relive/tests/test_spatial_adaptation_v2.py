"""GPU-free Phase 3-v2 coordinate-contract and formal-path checks."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from relive.backends.base import Backend
from relive.backends.mock import MockBackend
from relive.claims import PROMPT_VERSIONS, prompt
from relive.config import load_config, validate_config
from relive.runner import SampleRunner
from relive.spatial_adaptation_v2 import (SpatialEvidenceAdaptationControllerV2,
                                          canonicalize_region_1000,
                                          parse_refinement_v2, route_refinement_v2)
from relive.storage.artifacts import ArtifactStore
from relive.storage.cache import Budget, CachedInference
from relive.types import Claim, EvidenceCandidate, ExecutionStatus, Frame, RuntimeSample, SpatialProposal

ROOT = Path(__file__).resolve().parents[1]


class JsonBackend(Backend):
    synthetic = True
    def infer(self, request):
        self.calls += 1
        return '{"status":"UNRESOLVED","bbox_normalized_0_1000":null,"reason":"no distinct region"}'


class Phase3V2Tests(unittest.TestCase):
    def setUp(self):
        self.claim = Claim("c", "A local object is visible.")
        self.candidate = EvidenceCandidate("e", "s", ("f",), (None,), 0, "chronological")
        self.parent = SpatialProposal("r0", "e", "c", (0.432, 0.274, 0.682, 0.882))
        self.route = route_refinement_v2({"original_failure_reasons": ["DEPENDENCE_UNRESOLVED"]})

    def parse(self, value):
        return parse_refinement_v2(json.dumps(value), self.candidate, self.claim, parent=self.parent,
                                   route=self.route, raw_response_ref="raw")

    def test_canonicalizes_previous_coordinate_contract(self):
        self.assertEqual(canonicalize_region_1000(self.parent.support_region), [432, 274, 682, 882])
        rendered = prompt("spatial_refine_dependence_v2", {
            "frame_ids": ["f"], "timing": {}, "claim": {"text": self.claim.text, "entity": None, "time_scope": {}},
            "previous_bbox_normalized_0_1000": [432, 274, 682, 882], "diagnosed_failure_reason": "DEPENDENCE_UNRESOLVED",
            "previous_original_semantic_status": "SUPPORTED", "previous_keep_semantic_status": "SUPPORTED",
            "previous_drop_semantic_status": "SUPPORTED", "refinement_round": 1})
        self.assertIn('"previous_bbox_normalized_0_1000": [432, 274, 682, 882]', rendered)
        self.assertNotIn("previous_bbox_normalized_0_1_xyxy", rendered)

    def test_proposed_and_unresolved_closed_schema(self):
        proposed = self.parse({"status": "PROPOSED", "bbox_normalized_0_1000": [400, 250, 700, 900], "reason": "include relation"})
        self.assertEqual(proposed["outcome"], "PROPOSED")
        self.assertEqual(proposed["proposal"].support_region, (0.4, 0.25, 0.7, 0.9))
        unresolved = self.parse({"status": "UNRESOLVED", "bbox_normalized_0_1000": None, "reason": "no distinct support"})
        self.assertEqual(unresolved["outcome"], "UNRESOLVED")
        self.assertIsNone(unresolved["proposal"])

    def test_invalid_coordinates_and_noop_are_not_formal_proposals(self):
        floating = self.parse({"status": "PROPOSED", "bbox_normalized_0_1000": [0.4, 250, 700, 900], "reason": "bad"})
        self.assertEqual(floating["failure_code"], "REFINEMENT_COORDINATE_VIOLATION")
        invalid = self.parse({"status": "PROPOSED", "bbox_normalized_0_1000": [900, 250, 700, 900], "reason": "bad"})
        self.assertEqual(invalid["failure_code"], "REFINEMENT_COORDINATE_VIOLATION")
        noop = self.parse({"status": "PROPOSED", "bbox_normalized_0_1000": [432, 274, 682, 882], "reason": "same"})
        self.assertEqual(noop["outcome"], "REFINEMENT_NO_OP")
        self.assertIsNone(noop["proposal"])

    def test_one_round_controller_and_v1_v2_cache_isolation_replay(self):
        controller = SpatialEvidenceAdaptationControllerV2()
        self.assertFalse(controller.route({"original_failure_reasons": ["DEPENDENCE_UNRESOLVED"]}, current_round=1)["allowed"])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "f.png"
            Image.new("RGB", (8, 8), "purple").save(image)
            cache = CachedInference(JsonBackend({"kind": "unit", "max_retries": 0}), ArtifactStore(root / "cache"))
            base = {"image_paths": [str(image)], "frame_ids": ["f"], "context": {"candidate_id": "e"}}
            v1 = {**base, "stage": "spatial_refine_dependence", "prompt_version": "relive-spatial-refine-dependence-v1", "prompt": "v1"}
            v2 = {**base, "stage": "spatial_refine_dependence_v2", "prompt_version": PROMPT_VERSIONS["spatial_refine_dependence_v2"], "prompt": "v2"}
            self.assertNotEqual(cache.cache_key(v1), cache.cache_key(v2))
            first, replay = Budget(2), Budget(2)
            cache.call(v2, first); cache.call(v2, replay)
            self.assertEqual(first.new_calls, 1)
            self.assertEqual(replay.new_calls, 0)
            self.assertEqual(replay.cache_hits, 1)

    def test_changed_v2_proposal_enters_existing_formal_path(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "f.png"
            source = Image.new("RGB", (20, 16), (20, 80, 140))
            ImageDraw.Draw(source).rectangle((2, 2, 10, 10), fill=(230, 40, 20))
            source.save(image)
            sample = RuntimeSample("s", "claim_verification", "q", (Frame("f", str(image), 0),),
                                   target_claim=self.claim, provenance={"runtime_sha256": "test-runtime"})
            cfg = load_config(ROOT / "configs" / "mock_smoke.yaml")
            cfg["policy"]["name"] = "semantic_spatial"
            cfg["budget"].update(max_calls=20, max_spatial_proposals=1)
            cfg = validate_config(cfg)
            store = ArtifactStore(root / "run")
            engine = SampleRunner(cfg, store, CachedInference(MockBackend(cfg["backend"]), ArtifactStore(root / "cache")))
            proposed = self.parse({"status": "PROPOSED", "bbox_normalized_0_1000": [100, 100, 500, 700], "reason": "changed"})
            original = engine.infer(sample, self.candidate, self.claim, "semantic")
            spatial = engine.spatial_with_proposal(sample, self.candidate, self.claim, original, proposed["proposal"], proposal_index=1)
            certificate = engine.certificate(sample, self.candidate, self.claim, original, spatial, None)
            self.assertIn("keep", spatial["references"])
            self.assertIn("drop", spatial["references"])
            self.assertIsNotNone(certificate.final_status)


if __name__ == "__main__":
    unittest.main()
