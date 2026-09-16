from __future__ import annotations
import hashlib
import json
from pathlib import Path
import unittest

import tests.test_v2_tal_coarse_hypothesis as coarse_test
from relive.storage.artifacts import canonical_json
from relive.v2.coarse_hypothesis import execute as coarse_execute, preflight as coarse_preflight
from relive.v2.observation_planning import prepare as plan_prepare
from relive.v2.observation_retrieval import (CHOICES, ObservationRetrievalError, execute, preflight, prepare, support_margin, validate)

class _FakeObservationBackend:
    synthetic=True
    def fingerprint(self): return {"adapter_version":"fake-stage3d-v1","model":"fake"}
    def forced_choice_token_contract(self, request, choices=CHOICES):
        return {"choice_labels":list(choices),"choice_token_ids":{label:index+1 for index,label in enumerate(choices)},"context_input_ids_sha256":hashlib.sha256(request["prompt"].encode()).hexdigest(),"context_token_count":11}
    def forced_choice_likelihood(self, request, *, choice_token_ids=None, choices=CHOICES):
        contract=self.forced_choice_token_contract(request,choices=choices)
        assert choice_token_ids==contract["choice_token_ids"]
        base=int(hashlib.sha256(request["retrieval_task_id"].encode()).hexdigest()[:4],16)/65535
        return {"choice_contract":contract,"choices":{label:{"token_id":contract["choice_token_ids"][label],"raw_logit":base-index,"log_probability":base-index} for index,label in enumerate(choices)}}

class ObservationRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.base=coarse_test.CoarseHypothesisTests(methodName="test_packets_preflight_run_replay_and_no_admission"); self.base.setUp(); self.base.prepare()
        coarse_preflight(output_dir=self.base.out,config_path=self.base.config,backend_factory=lambda _:coarse_test._FakeChoiceBackend())
        coarse_execute(output_dir=self.base.out,config_path=self.base.config,mode="run",backend_factory=lambda _:coarse_test._FakeChoiceBackend())
        self.compare=self.base.root/"comparison.json"; self.compare.write_text(canonical_json({"ready_for_stage3c":True})+"\n")
        self.stage3c=self.base.root/"stage3c"; policy3c=Path(__file__).resolve().parents[1]/"configs/v2/tal_observation_temporal_planning_policy.json"
        plan_prepare(requirement_dir=self.base.requirements,selection_manifest=self.base.selection,video_index_dir=self.base.index,stage3b_dir=self.base.out,comparison_path=self.compare,policy_path=policy3c,output_dir=self.stage3c)
        self.policy=Path(__file__).resolve().parents[1]/"configs/v2/tal_observation_retrieval_policy.json"; self.out=self.base.root/"stage3d"
    def tearDown(self): self.base.tearDown()
    def prepare(self,out=None):
        return prepare(requirement_dir=self.base.requirements,selection_manifest_path=self.base.selection,video_index_dir=self.base.index,stage3c_dir=self.stage3c,config_path=self.base.config,policy_path=self.policy,output_dir=out or self.out)
    def test_packets_dedup_fanout_run_replay_and_no_admission(self):
        result=self.prepare(); self.assertEqual(result["observation_claim_count"],15)
        stage3c_manifest=json.loads((self.stage3c/"v2_stage3c_manifest.json").read_text())
        self.assertEqual((result["physical_window_count"],result["claim_window_binding_count"]),(stage3c_manifest["physical_acquisition_window_count"],stage3c_manifest["claim_window_binding_count"]))
        packets=[json.loads(row) for row in (self.out/"v2_stage3d_visual_packets.jsonl").read_text().splitlines()]
        tasks=[json.loads(row) for row in (self.out/"v2_stage3d_retrieval_tasks.jsonl").read_text().splitlines()]
        fanout=[json.loads(row) for row in (self.out/"v2_stage3d_binding_fanout.jsonl").read_text().splitlines()]
        self.assertEqual(len(packets),result["physical_window_count"]); self.assertEqual(len(fanout),result["claim_window_binding_count"]); self.assertLess(len(tasks),len(fanout)); self.assertTrue(all(len(row["frames"])<=8 for row in packets)); self.assertTrue(all(row["status"]=="PLANNED_UNSCORED" for row in fanout))
        self.assertTrue(all("parent_hypothesis" not in row["prompt"].lower() and "no_visible" not in row["prompt"].lower() for row in tasks))
        preflight(output_dir=self.out,config_path=self.base.config,stage3c_dir=self.stage3c,backend_factory=lambda _: _FakeObservationBackend())
        run=execute(output_dir=self.out,config_path=self.base.config,mode="run",backend_factory=lambda _: _FakeObservationBackend())
        replay=execute(output_dir=self.out,config_path=self.base.config,mode="replay",backend_factory=lambda _: _FakeObservationBackend())
        self.assertEqual((run["new_model_calls"],run["cache_hits"],replay["new_model_calls"],replay["cache_hits"]),(run["planned_model_calls"],0,0,run["planned_model_calls"]))
        for key in ("numeric_results_sha256","binding_results_scientific_sha256","observation_rankings_sha256","candidate_evidence_sets_sha256"): self.assertEqual(run[key],replay[key])
        candidates=[json.loads(row) for row in (self.out/"run"/"v2_stage3d_candidate_evidence_sets.jsonl").read_text().splitlines()]
        self.assertEqual(len({row["observation_claim_id"] for row in candidates}),15); self.assertTrue(all(row["status"]=="CANDIDATE_EVIDENCE_UNVERIFIED" for row in candidates)); self.assertEqual(validate(self.out)["status"],"PASS")
    def test_margin_formula_and_tamper_fail_closed(self):
        self.assertAlmostEqual(support_margin({"A":0.,"B":0.,"C":0.,"D":0.}),-__import__("math").log(3))
        self.prepare(); data=(self.out/"v2_stage3d_visual_packets.jsonl").read_bytes(); (self.out/"v2_stage3d_visual_packets.jsonl").write_bytes(data+b" ")
        with self.assertRaisesRegex(ObservationRetrievalError,"ARTIFACT_TAMPERED"): preflight(output_dir=self.out,config_path=self.base.config,stage3c_dir=self.stage3c,backend_factory=lambda _: _FakeObservationBackend())

if __name__=="__main__": unittest.main()

class ObservationRetrievalContractTests(ObservationRetrievalTests):
    def test_temporal_insufficiency_has_no_model_call_and_nonunique_tokens_fail(self):
        policy=json.loads(self.policy.read_text()); policy["template_min_unique_timestamps"]["ACTION_CORE_PRESSING"]=99
        altered=self.base.root/"altered-policy.json"; altered.write_text(canonical_json(policy)+"\n")
        prepare(requirement_dir=self.base.requirements,selection_manifest_path=self.base.selection,video_index_dir=self.base.index,stage3c_dir=self.stage3c,config_path=self.base.config,policy_path=altered,output_dir=self.out)
        tasks=[json.loads(row) for row in (self.out/"v2_stage3d_retrieval_tasks.jsonl").read_text().splitlines()]
        self.assertTrue(any(row["retrieval_status"]=="TEMPORAL_PACKET_INSUFFICIENT" and not row["model_call_allowed"] for row in tasks))
        class Broken(_FakeObservationBackend):
            def forced_choice_token_contract(self, request, choices=CHOICES):
                result=super().forced_choice_token_contract(request,choices=choices); result["choice_token_ids"]={label:1 for label in choices}; return result
        with self.assertRaisesRegex(ObservationRetrievalError,"ABCD_NEXT_TOKEN_CONTRACT_NOT_UNIQUE"):
            preflight(output_dir=self.out,config_path=self.base.config,stage3c_dir=self.stage3c,backend_factory=lambda _:Broken())
