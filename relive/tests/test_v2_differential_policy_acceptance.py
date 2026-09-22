from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from relive.storage.artifacts import canonical_json, stable_hash
from relive.v2.differential_evidence import DifferentialEvidenceError, variant_set_from_mapping
from relive.v2.differential_policy_acceptance import DifferentialAcceptanceError, run_acceptance_audit

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "configs/v2/differential_evidence_policy.yaml"
FIXTURE = ROOT / "tests/fixtures/v2_differential_policy_acceptance_fixture.json"


def load_fixture(): return json.loads(FIXTURE.read_text())
def write_json(path, value): path.write_text(canonical_json(value) + "\n", encoding="utf-8")


class DifferentialAcceptanceTests(unittest.TestCase):
    def closure(self, root: Path) -> Path:
        row={"status":"PILOT_CLOSED","source_artifacts_unchanged":True,"label_resolution_applied":True,
             "r0_replay_new_model_calls":0,"r1_replay_new_model_calls":0}
        row["closure_manifest_sha256"]=stable_hash(row); path=root/"closure.json"; write_json(path,row); return path

    def test_fixed_fixture_runs_twice_byte_identically(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); closure=self.closure(root)
            first=run_acceptance_audit(policy_path=POLICY,fixture_path=FIXTURE,output_dir=root/"one",pilot_closure=closure)
            second=run_acceptance_audit(policy_path=POLICY,fixture_path=FIXTURE,output_dir=root/"two",pilot_closure=closure)
            self.assertEqual(first["status"],"PASS"); self.assertEqual(second["status"],"PASS")
            for name in ("differential_policy_acceptance_report.json","differential_policy_test_manifest.json","pilot_regression_report.json","README.md"):
                self.assertEqual((root/"one"/name).read_bytes(),(root/"two"/name).read_bytes())
            report=json.loads((root/"one"/"differential_policy_acceptance_report.json").read_text())
            self.assertAlmostEqual(report["metrics"]["g_keep"], .4)
            self.assertAlmostEqual(report["metrics"]["g_drop"], .45)
            self.assertEqual(report["certificate_status"], "DIAGNOSTIC_ONLY")
            self.assertFalse(report["certificate_created"])

    def test_pilot_regression_is_read_only_and_reports_zero_replay_calls(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); closure=self.closure(root); before=closure.read_bytes()
            run_acceptance_audit(policy_path=POLICY,fixture_path=FIXTURE,output_dir=root/"accept",pilot_closure=closure)
            report=json.loads((root/"accept"/"pilot_regression_report.json").read_text())
            self.assertTrue(report["pilot_checked"]); self.assertEqual(report["r0_replay_new_model_calls"],0); self.assertEqual(report["r1_replay_new_model_calls"],0)
            self.assertEqual(report["cache_writes"],0); self.assertEqual(report["certificate_writes"],0); self.assertEqual(before,closure.read_bytes())

    def test_missing_target_or_required_variant_fails_closed(self):
        fixture=load_fixture(); fixture["variant_set"].pop("DROP_TARGET")
        with self.assertRaisesRegex(DifferentialEvidenceError,"REQUIRED_VARIANT_SET_INCOMPLETE"):
            variant_set_from_mapping(fixture["variant_set"])

    def test_unparseable_variant_fails_closed(self):
        fixture=load_fixture(); fixture["variant_set"]["ORIGINAL"]={"variant":"ORIGINAL"}
        with self.assertRaisesRegex(DifferentialEvidenceError,"UNPARSEABLE"):
            variant_set_from_mapping(fixture["variant_set"])

    def test_required_variant_not_executed_cannot_reach_metrics_or_certificate(self):
        fixture=load_fixture(); fixture["variant_set"]["KEEP_TARGET"]["score"]=None
        values=variant_set_from_mapping(fixture["variant_set"])
        with self.assertRaisesRegex(DifferentialEvidenceError,"CONTINUOUS_SUPPORT_SCORE_UNAVAILABLE"):
            values.compute_metrics()

    def test_control_tier_and_cardinality_fail_closed(self):
        fixture=load_fixture(); fixture["variant_set"]["control_quality"]["tier"]="TIER_3_RELAXED"
        with self.assertRaisesRegex(DifferentialEvidenceError,"MATCHED_CONTROL_SET_REJECTED"):
            variant_set_from_mapping(fixture["variant_set"])
        fixture=load_fixture(); fixture["variant_set"]["KEEP_MATCHED_CONTROL"]=fixture["variant_set"]["KEEP_MATCHED_CONTROL"][:1]
        with self.assertRaisesRegex(DifferentialEvidenceError,"MATCHED_CONTROL_SET_REJECTED"):
            variant_set_from_mapping(fixture["variant_set"])

    def test_acceptance_rejects_tainted_automatic_fixture(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); fixture=load_fixture(); fixture["provenance"]=["HUMAN_ORACLE"]; source=root/"tainted.json"; write_json(source,fixture)
            result=run_acceptance_audit(policy_path=POLICY,fixture_path=source,output_dir=root/"accept")
            report=json.loads(Path(result["report"]).read_text())
            self.assertEqual(report["certificate_status"],"ORACLE_DIAGNOSTIC_ONLY")
            self.assertEqual(report["certificate_failure_reason"],"HUMAN_ORACLE_CONTAMINATION")
            self.assertEqual(report["new_verified_count"],0)

    def test_no_overwrite_output_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); output=root/"out"; output.mkdir(); (output/"existing").write_text("x")
            with self.assertRaisesRegex(DifferentialAcceptanceError,"NOT_EMPTY"):
                run_acceptance_audit(policy_path=POLICY,fixture_path=FIXTURE,output_dir=output)


if __name__ == "__main__": unittest.main()
