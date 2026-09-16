from __future__ import annotations
import json
from pathlib import Path
import unittest
import tests.test_v2_tal_coarse_hypothesis as coarse_test
from relive.storage.artifacts import canonical_json
from relive.v2.coarse_hypothesis import execute, preflight
from relive.v2.observation_planning import ObservationPlanningError, prepare, validate

class ObservationPlanningTests(unittest.TestCase):
 def setUp(self):
  self.base=coarse_test.CoarseHypothesisTests(methodName="test_packets_preflight_run_replay_and_no_admission");self.base.setUp();self.base.prepare();preflight(output_dir=self.base.out,config_path=self.base.config,backend_factory=lambda _: coarse_test._FakeChoiceBackend());execute(output_dir=self.base.out,config_path=self.base.config,mode="run",backend_factory=lambda _: coarse_test._FakeChoiceBackend())
  self.compare=self.base.root/"comparison.json";self.compare.write_text(canonical_json({"ready_for_stage3c":True})+"\n")
  self.policy=Path(__file__).resolve().parents[1]/"configs/v2/tal_observation_temporal_planning_policy.json"
 def tearDown(self):self.base.tearDown()
 def prepare_stage(self,out):return prepare(requirement_dir=self.base.requirements,selection_manifest=self.base.selection,video_index_dir=self.base.index,stage3b_dir=self.base.out,comparison_path=self.compare,policy_path=self.policy,output_dir=out)
 def test_decomposition_domains_windows_dedup_and_determinism(self):
  one,two=self.base.root/"one",self.base.root/"two";r=self.prepare_stage(one);self.assertEqual((r["positive_hypothesis_count"],r["observation_claim_count"],r["negative_hypothesis_count"]),(5,15,1));self.assertGreater(r["claim_window_binding_count"],r["physical_acquisition_window_count"])
  claims=[json.loads(x) for x in (one/"v2_stage3c_observation_claims.jsonl").read_text().splitlines()];self.assertEqual({x["template_id"] for x in claims},{"PRECONDITION_ALIGNMENT","ACTION_CORE_PRESSING","POSTCONDITION_ATTACHMENT"});self.assertTrue(all(x["status"]=="CANDIDATE_UNVERIFIED" for x in claims));self.assertEqual(validate(one)["status"],"PASS")
  self.prepare_stage(two);self.assertEqual((one/"v2_stage3c_observation_claims.jsonl").read_bytes(),(two/"v2_stage3c_observation_claims.jsonl").read_bytes())
  domains=[json.loads(x) for x in (one/"v2_stage3c_observation_domains.jsonl").read_text().splitlines()];self.assertTrue(all(x["domain_coverage"]==1.0 for x in domains))
 def test_budget_parent_mutation_and_artifact_tamper_fail_closed(self):
  bad=self.base.root/"bad.json";policy=json.loads(self.policy.read_text());policy["max_fine_windows_per_claim"]=1;bad.write_text(canonical_json(policy)+"\n")
  with self.assertRaisesRegex(ObservationPlanningError,"BUDGET"):prepare(requirement_dir=self.base.requirements,selection_manifest=self.base.selection,video_index_dir=self.base.index,stage3b_dir=self.base.out,comparison_path=self.compare,policy_path=bad,output_dir=self.base.root/"bad")
  graph=self.base.out/"run"/"v2_stage3b_claim_graph.json";value=json.loads(graph.read_text());value["nodes"][0]["status"]="SUPPORTED";graph.write_text(canonical_json(value)+"\n")
  with self.assertRaisesRegex(ObservationPlanningError,"PARENT"):self.prepare_stage(self.base.root/"mutated")
if __name__=="__main__":unittest.main()
