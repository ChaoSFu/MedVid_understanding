"""GPU-free Phase 3.7 frozen residual-support diagnostic checks."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import unittest

from PIL import Image, ImageChops

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from relive.phase36 import execute as phase36_execute, preflight as phase36_preflight
from relive.phase37 import (TARGET_CLAIM_ID, execute, geometry_audit, parse_residual_proposal,
                            preflight, residual_keep, union_drop)
from test_phase36 import Phase36Tests


class Phase37Tests(unittest.TestCase):
    def setUp(self):
        self.fixture = Phase36Tests(methodName="test_routes_contract_and_replay")
        self.fixture.setUp()
        self.root = self.fixture.root; self.config = self.fixture.config; self.runtime = self.fixture.runtime
        self.manifest = self.fixture.manifest; self.v3 = self.fixture.v3
        self.phase36 = self.root / "p36"
        phase36_preflight(config_path=self.config, runtime_path=self.runtime, prospective_manifest_path=self.manifest,
                          phase35_v3_run_dir=self.v3 / "run", output_dir=self.phase36, require_real=False)
        phase36_execute(config_path=self.config, runtime_path=self.runtime, prospective_manifest_path=self.manifest,
                        phase35_v3_run_dir=self.v3 / "run", output_dir=self.phase36, mode="run", require_real=False)
        cfg = json.loads(self.config.read_text())
        cfg["backend"]["rules"].append({"match": {"stage": "residual_support_localize"}, "response": {
            "status": "PROPOSED", "bbox_normalized_0_1000": [500, 500, 700, 700], "reason": "visible local detail"}})
        self.config.write_text(json.dumps(cfg))

    def tearDown(self):
        self.fixture.tearDown()

    def _kwargs(self, output):
        return {"config_path": self.config, "runtime_path": self.runtime, "prospective_manifest_path": self.manifest,
                "phase36_run_dir": self.phase36 / "run", "output_dir": output, "require_real": False}

    def test_frozen_cohort_drop_input_and_replay(self):
        out = self.root / "phase37"
        plan = preflight(**self._kwargs(out))
        self.assertEqual(plan["model_calls_made"], 0)
        self.assertEqual(plan["cohort_claim_ids"], [TARGET_CLAIM_ID])
        self.assertEqual(plan["excluded_claim_ids"], ["phase35-local-002", "phase35-local-003"])
        first = execute(**self._kwargs(out), mode="run")
        replay = execute(**self._kwargs(out), mode="replay")
        self.assertEqual(first["status"], "PASS")
        self.assertGreater(first["summary"]["new_model_calls"], 0)
        self.assertEqual(replay["summary"]["new_model_calls"], 0)
        self.assertGreater(replay["summary"]["cache_hits"], 0)
        trace = json.loads((out / "run" / "phase37_trace.jsonl").read_text())
        self.assertEqual(trace["claim_id"], TARGET_CLAIM_ID)
        self.assertEqual(trace["residual_proposal_attempt_count"], 1)
        self.assertEqual(trace["formal_certificate_status_unchanged"], "UNCERTAIN")
        self.assertEqual(trace["residual_proposal"]["outcome"], "PROPOSED")
        self.assertTrue(trace["residual_geometry"]["valid"])
        self.assertTrue(all(item["pixel_audit_pass"] for item in trace["interventions"]))

    def test_closed_contract_overlap_and_deterministic_images(self):
        self.assertEqual(parse_residual_proposal('{"status":"UNRESOLVED","bbox_normalized_0_1000":null,"reason":"none"}')["outcome"], "UNRESOLVED")
        self.assertEqual(parse_residual_proposal('{"status":"PROPOSED","bbox_normalized_0_1000":[1,2,3.0,4],"reason":"bad"}')["failure_code"], "RESIDUAL_COORDINATE_VIOLATION")
        overlap = geometry_audit((.1,.1,.4,.4), (.2,.2,.3,.3), (40, 30))
        self.assertFalse(overlap["valid"])
        self.assertEqual(overlap["invalid_code"], "RESIDUAL_PROPOSAL_INVALID_MASK_OVERLAP")
        spec = {"operator": "opaque_gray", "operator_version": "relive-opaque-gray-hard-mask-v1", "parameters": {"fill_rgb": [127,127,127]}}
        original = Image.new("RGB", (40, 30), (10, 20, 30))
        drop, _ = __import__("relive.interventions", fromlist=["apply_spatial_intervention"]).apply_spatial_intervention(original, (.1,.1,.4,.4), "DROP_TARGET", spec)
        first, keep_audit = residual_keep(drop, (.5,.5,.7,.7), spec)
        second, _ = residual_keep(drop, (.5,.5,.7,.7), spec)
        self.assertIsNone(ImageChops.difference(first, second).getbbox())
        union, union_audit = union_drop(original, (.1,.1,.4,.4), (.5,.5,.7,.7), spec)
        self.assertTrue(keep_audit["pixel_audit_pass"])
        self.assertTrue(union_audit["pixel_audit_pass"])
        self.assertEqual(union.getpixel((5,5)), (127,127,127))
        self.assertEqual(union.getpixel((22,17)), (127,127,127))


if __name__ == "__main__":
    unittest.main()
