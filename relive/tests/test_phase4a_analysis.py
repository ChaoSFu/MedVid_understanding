"""CPU-only tests for the read-only Phase 4A descriptive finalizer."""
from __future__ import annotations
import json, math, tempfile, unittest
from pathlib import Path

from relive.phase4a_analysis import (ANALYSIS_RULE_VERSION, Phase4AAnalysisError,
                                     classify_case, finalize)
from relive.phase4a0 import ELIGIBLE_MANIFEST_FORMAT, ELIGIBLE_RECORD_FORMAT
from relive.storage.artifacts import canonical_json
import hashlib


def digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


class Phase4AAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.phase4 = self.root / "phase4"; (self.phase4 / "run").mkdir(parents=True)
        self.eligible_rows, trace = [], []
        groups = (["FULL_LOCAL_DEPENDENCE_PATTERN"] * 5 + ["DROP_SUPPORT_PERSISTS"] * 2 +
                  ["KEEP_PRESERVATION_FAILURE"] * 2 + ["ORIGINAL_LIKELIHOOD_INSUFFICIENT"])
        for number, group in enumerate(groups, 1):
            source = f"development:development-dev-{number:03d}"
            margins = {"ORIGINAL": 1., "KEEP_TARGET": 1., "DROP_TARGET": -1., "DROP_MATCHED_CONTROL": 1., "FULL_GRAY": -1., "MISMATCHED_PUBLIC": .25}
            if group == "DROP_SUPPORT_PERSISTS":
                margins["DROP_TARGET"] = .25
                margins["FULL_GRAY"] = -1. if number == 6 else .25
            elif group == "KEEP_PRESERVATION_FAILURE": margins["KEEP_TARGET"] = -.25
            elif group == "ORIGINAL_LIKELIHOOD_INSUFFICIENT": margins["ORIGINAL"] = -.25
            delta = {"delta_drop": margins["ORIGINAL"] - margins["DROP_TARGET"], "delta_full_gray": margins["ORIGINAL"] - margins["FULL_GRAY"],
                     "delta_mismatch": margins["ORIGINAL"] - margins["MISMATCHED_PUBLIC"], "delta_specificity": margins["DROP_MATCHED_CONTROL"] - margins["DROP_TARGET"]}
            trace.append({"source_id": source, "claim_id": f"claim-{number}", "variants": {key: {"support_margin": value} for key, value in margins.items()}, "deltas": delta})
            self.eligible_rows.append({"format": ELIGIBLE_RECORD_FORMAT, "audit_case_id": f"development-dev-{number:03d}"})
        self.trace_path = self.phase4 / "run" / "phase4a_trace.jsonl"; self.trace_path.write_text("".join(canonical_json(row) + "\n" for row in trace))
        eligible_path = self.root / "eligible.jsonl"; eligible_path.write_text("".join(canonical_json(row) + "\n" for row in self.eligible_rows))
        self.eligibility = self.root / "eligible.manifest.json"
        manifest = {"format": ELIGIBLE_MANIFEST_FORMAT, "selection_status": "FROZEN_ELIGIBLE_CONTROLS", "eligible_controls_path": str(eligible_path),
                    "eligible_controls_sha256": hashlib.sha256(eligible_path.read_bytes()).hexdigest(), "eligible_count": 10}
        manifest["manifest_content_sha256"] = digest(manifest); self.eligibility.write_text(canonical_json(manifest))
        frozen = {"source_mode": "DEVELOPMENT_POSITIVE_CONTROL", "formal_positive_control": True, "entries": []}
        frozen["phase4a_frozen_manifest_sha256"] = digest(frozen); (self.phase4 / "phase4a_frozen_manifest.json").write_text(canonical_json(frozen))
        (self.phase4 / "phase4a_preflight.json").write_text(canonical_json({"source_mode": "DEVELOPMENT_POSITIVE_CONTROL", "eligibility_manifest": str(self.eligibility)}))

    def tearDown(self): self.temp.cleanup()

    def test_ten_case_descriptive_groups_and_deterministic_outputs(self):
        first, second = self.root / "out1", self.root / "out2"
        audit = finalize(phase4a_output_dir=self.phase4, eligibility_manifest_path=self.eligibility, output_dir=first)
        again = finalize(phase4a_output_dir=self.phase4, eligibility_manifest_path=self.eligibility, output_dir=second)
        summary = json.loads((first / "phase4a_descriptive_summary.json").read_text())
        self.assertEqual(summary["classification_counts"], {"DROP_SUPPORT_PERSISTS": 2, "FULL_LOCAL_DEPENDENCE_PATTERN": 5,
                         "KEEP_PRESERVATION_FAILURE": 2, "ORIGINAL_LIKELIHOOD_INSUFFICIENT": 1})
        rows = [json.loads(line) for line in (first / "phase4a_failure_routing_fixtures.jsonl").read_text().splitlines()]
        self.assertEqual(rows[5]["allowed_action_family"], "LOCALIZE_RESIDUAL_COMPONENT")
        self.assertEqual(rows[6]["allowed_action_family"], "STOP_OR_PREDECLARED_VERIFIER_SWITCH")
        self.assertTrue(audit["diagnostic_only"]); self.assertFalse(audit["backend_loaded"])
        self.assertEqual(audit["artifact_sha256"], again["artifact_sha256"])

    def test_invalid_variants_nonfinite_and_delta_mismatch_fail_closed(self):
        rows = [json.loads(line) for line in self.trace_path.read_text().splitlines()]
        rows[0]["variants"].pop("FULL_GRAY")
        self.trace_path.write_text("".join(canonical_json(row) + "\n" for row in rows))
        with self.assertRaisesRegex(Phase4AAnalysisError, "VARIANT_SET"):
            finalize(phase4a_output_dir=self.phase4, eligibility_manifest_path=self.eligibility, output_dir=self.root / "bad")
        # Restore and check non-finite JSON input is rejected.
        self.temp.cleanup(); self.setUp()
        rows = [json.loads(line) for line in self.trace_path.read_text().splitlines()]
        rows[0]["variants"]["ORIGINAL"]["support_margin"] = float("nan")
        self.trace_path.write_text("\n".join(json.dumps(row, allow_nan=True) for row in rows) + "\n")
        with self.assertRaisesRegex(Phase4AAnalysisError, "FINITE"):
            finalize(phase4a_output_dir=self.phase4, eligibility_manifest_path=self.eligibility, output_dir=self.root / "nan")
        self.temp.cleanup(); self.setUp()
        rows = [json.loads(line) for line in self.trace_path.read_text().splitlines()]
        rows[0]["deltas"]["delta_drop"] = 123.
        self.trace_path.write_text("".join(canonical_json(row) + "\n" for row in rows))
        with self.assertRaisesRegex(Phase4AAnalysisError, "DELTA_BINDING"):
            finalize(phase4a_output_dir=self.phase4, eligibility_manifest_path=self.eligibility, output_dir=self.root / "delta")

    def test_classifier_uses_margin_signs_not_case_ids(self):
        self.assertEqual(classify_case({"ORIGINAL": 1., "KEEP_TARGET": 1., "DROP_TARGET": -1., "DROP_MATCHED_CONTROL": 1.}), "FULL_LOCAL_DEPENDENCE_PATTERN")
        self.assertEqual(classify_case({"ORIGINAL": 1., "KEEP_TARGET": 1., "DROP_TARGET": .1, "DROP_MATCHED_CONTROL": 1.}), "DROP_SUPPORT_PERSISTS")

if __name__ == "__main__": unittest.main()
