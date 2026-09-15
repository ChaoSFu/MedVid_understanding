from __future__ import annotations
import hashlib,json,tempfile,unittest
from pathlib import Path
from relive.storage.artifacts import canonical_json
from relive.v2.adaptation_controller import default_policy
from relive.v2.fixture_replay import *

class V2FixtureReplayTests(unittest.TestCase):
    def setUp(self): self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
    def tearDown(self): self.temp.cleanup()
    def row(self,index,code,allowed,forbidden="CREATE_VERIFIED", **extra):
        value={"case_id":f"opaque-{index}","source_trace_sha256":hashlib.sha256(f"{index}".encode()).hexdigest(),"observed_diagnostic_class":"class","auxiliary_diagnostic":None,"proposed_v2_failure_code":code,"allowed_action_family":allowed,"forbidden_action_family":forbidden,"analysis_rule_version":"v1","diagnostic_only":True,"certificate_created":False,"certificate_unchanged":True,"new_verified_count":0,"gt_used":False,"model_calls_made":0,"backend_loaded":False}
        value.update(extra); return value
    def write(self,rows,name="fixture.jsonl"):
        path=self.root/name; path.write_text("".join(canonical_json(row)+"\n" for row in rows)); return path
    def test_synthetic_ten_case_routes_without_case_id_logic(self):
        rows=[]
        rows += [self.row(i,"NO_FAILURE_DIAGNOSTIC_COMPLETE","STOP_DIAGNOSTIC") for i in range(5)]
        rows += [self.row(5,"ORIGINAL_INSUFFICIENT","TEMPORAL_REACQUIRE")]
        rows += [self.row(i,"DROP_SUPPORT_PERSISTS","LOCALIZE_RESIDUAL_COMPONENT") for i in range(6,8)]
        rows += [self.row(i,"KEEP_SUPPORT_LOST","RECOMPOSE_SPATIAL_EVIDENCE") for i in range(8,10)]
        path=self.write(rows); first=self.root/"first"; second=self.root/"second"
        audit=write_replay_artifacts(fixtures_path=path,output_dir=first,policy=default_policy()); again=write_replay_artifacts(fixtures_path=path,output_dir=second,policy=default_policy())
        self.assertEqual(audit["action_counts"],{"RESIDUAL_LOCALIZE":2,"SPATIAL_RECOMPOSE":2,"STOP_DIAGNOSTIC":5,"TEMPORAL_REACQUIRE":1})
        self.assertEqual(audit["routing_trace_sha256"],again["routing_trace_sha256"]); self.assertFalse(audit["backend_loaded"])
        self.assertNotIn("VERIFIED",(first/"v2_failure_routing_trace.jsonl").read_text())
    def test_strict_unknown_forbidden_tamper_and_legacy(self):
        path=self.write([self.row(1,"UNKNOWN","STOP_DIAGNOSTIC")])
        with self.assertRaisesRegex(FixtureReplayError,"UNKNOWN_FAILURE") : load_phase4a_failure_fixtures(path)
        path=self.write([self.row(1,"DROP_SUPPORT_PERSISTS","UNKNOWN_ACTION")],"action.jsonl")
        with self.assertRaisesRegex(FixtureReplayError,"UNKNOWN_ACTION") : load_phase4a_failure_fixtures(path)
        good=self.write([self.row(1,"NO_FAILURE_DIAGNOSTIC_COMPLETE","STOP_DIAGNOSTIC",legacy_posthoc_gate=True,legacy_status="POSTHOC_VERIFIED_FOR_DIAGNOSTIC_FINALIZATION_ONLY")],"good.jsonl")
        with self.assertRaisesRegex(FixtureReplayError,"SHA256") : load_phase4a_failure_fixtures(good,expected_sha256="0"*64)
        output=self.root/"legacy"; audit=write_replay_artifacts(fixtures_path=good,output_dir=output,policy=default_policy())
        self.assertTrue(audit["legacy_status_seen"]); self.assertEqual(audit["legacy_status_interpretation"],"NO_CERTIFICATE_VERIFICATION")
    def test_duplicate_key_is_rejected(self):
        path=self.root/"duplicate.jsonl"; path.write_text('{"case_id":"a","case_id":"b"}\n')
        with self.assertRaisesRegex(FixtureReplayError,"DUPLICATE") : load_phase4a_failure_fixtures(path)

if __name__ == "__main__": unittest.main()
