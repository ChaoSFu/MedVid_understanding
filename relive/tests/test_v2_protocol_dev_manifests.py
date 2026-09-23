from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from relive.storage.artifacts import canonical_json, stable_hash
from relive.v2.protocol_dev_manifests import (
    PILOT_VIDEO, ProtocolDevManifestError, assert_automatic_runtime_safe,
    build_protocol_dev_manifests, freeze_preconditions, prefilter_csv_to_stage_a_preview,
    source_record_uid, validate_protocol_dev_freeze, derive_stage_a_source_uids,
)


ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "configs/v2/differential_evidence_policy.yaml"
ACCEPTED_COMMIT = "e9e2949f5a2050e140a9cd36d52bc09edcb1d907"


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canonical_json(value) + "\n" for value in values), encoding="utf-8")


class ProtocolDevManifestTests(unittest.TestCase):
    def precondition(self, root: Path) -> Path:
        policy_hash = hashlib.sha256(POLICY.read_bytes()).hexdigest()
        acceptance = root / "acceptance"; acceptance.mkdir()
        write_json(acceptance / "differential_policy_acceptance_report.json", {
            "status": "PASS", "policy_sha256": policy_hash, "model_calls_made": 0,
            "certificate_created": False,
        })
        write_json(acceptance / "pilot_regression_report.json", {
            "status": "PASS", "new_model_calls": 0, "cache_writes": 0, "certificate_writes": 0,
        })
        closure = {"status": "PILOT_CLOSED", "source_artifacts_unchanged": True,
                   "label_resolution_applied": True, "r0_replay_new_model_calls": 0,
                   "r1_replay_new_model_calls": 0, "cache_opened": False,
                   "certificate_created": False, "new_verified_count": 0}
        closure["closure_manifest_sha256"] = stable_hash(closure)
        closure_path = root / "closure.json"; write_json(closure_path, closure)
        freeze_preconditions(policy_path=POLICY, acceptance_dir=acceptance, pilot_closure=closure_path,
                             accepted_code_commit=ACCEPTED_COMMIT, output_dir=root / "precondition")
        return root / "precondition" / "protocol_dev_precondition_freeze_report.json"

    def stage_a(self, root: Path, *, pilot: bool = False, duplicate_video: bool = False,
                missing_media: bool = False, blank_decision: bool = False) -> list[dict]:
        rows = []
        types = ["SPATIAL_RELATION"] * 3 + ["CONTACT_ACTION"] * 3 + ["POSTCONDITION_PERSISTENCE"] * 3
        for index, claim_type in enumerate(types):
            media = root / f"media-{index}.bin"; media.write_bytes(b"frame")
            dataset, video = ("Fixture", f"video-{index}")
            if pilot and index == 0: dataset, video = PILOT_VIDEO
            if duplicate_video and index == 1: dataset, video = "Fixture", "video-0"
            source_hashes = [digest(f"source-{index}")]
            rows.append({"candidate_id": f"candidate-{index}", "dataset": dataset, "video_id": video,
                         "source_record_uid": source_record_uid(dataset=dataset, video_id=video,
                            source_record_canonical_hashes=source_hashes, source_task="tal"),
                         "source_record_canonical_hashes": source_hashes, "source_task": "tal",
                         "proposed_claim_type": claim_type,
                         "source_temporal_span": {"start_seconds": 0.0, "end_seconds": 4.0},
                         "media_path": str(root / "missing.bin") if missing_media and index == 0 else str(media),
                         "decision": "" if blank_decision and index == 0 else "ADMIT",
                         "reviewer_id": "reviewer-1", "rationale": "independent full-video review"})
        return rows

    def stage_b(self, stage_a: list[dict], *, multiple_predicate: bool = False) -> list[dict]:
        rows = []
        for index, parent in enumerate(stage_a):
            pair = f"pair-{index}"
            shared = {"subject": f"subject-{index}", "object": f"object-{index}", "qualifiers": {"context": str(index)}}
            for label, predicate in (("TRUE", "present"), ("FALSE", "absent")):
                claim = {**shared, "predicate": predicate}
                if multiple_predicate and index == 0 and label == "FALSE": claim["object"] = "different-object"
                rows.append({"case_id": f"case-{index}-{label.lower()}", "pair_id": pair,
                             "dataset": parent["dataset"], "video_id": parent["video_id"],
                             "source_record_uid": parent["source_record_uid"], "claim_type": parent["proposed_claim_type"],
                             "truth_label": label, "claim": claim, "changed_predicate": "predicate",
                             "temporal_window": {"start_seconds": 0.0, "end_seconds": 3.0},
                             "interaction_window": {"start_seconds": 1.0, "end_seconds": 2.0},
                             "after_window": {"start_seconds": 2.0, "end_seconds": 3.0},
                             "involved_entities": [shared["subject"], shared["object"]],
                             "required_evidence_roles": ["SUBJECT", "REFERENCE"],
                             "oracle_evidence_tube_or_mask": {"kind": "human_mask", "sha256": digest(f"oracle-{index}")},
                             "observability": "DIRECTLY_VISIBLE", "media_path": parent["media_path"],
                             "reviewer_id": "reviewer-2", "rationale": "paired one-predicate contrast"})
        return rows

    def build(self, root: Path, stage_a: list[dict], stage_b: list[dict], **kwargs):
        a, b = root / "stage_a.jsonl", root / "stage_b.jsonl"
        write_jsonl(a, stage_a); write_jsonl(b, stage_b)
        if "external_split_registry" not in kwargs:
            registry = root / "external_splits.jsonl"; write_jsonl(registry, [])
            kwargs["external_split_registry"] = registry
        return build_protocol_dev_manifests(policy_path=POLICY, precondition_freeze_report=self.precondition(root),
            stage_a_path=a, stage_b_path=b, output_dir=root / "output", **kwargs)

    def test_complete_cohort_freezes_runtime_gt_and_oracle_separately(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); stage_a = self.stage_a(root); result = self.build(root, stage_a, self.stage_b(stage_a))
            self.assertEqual(result["freeze_status"], "FROZEN_PROTOCOL_DEV")
            self.assertEqual(result["case_counts"], {"runtime": 18, "ground_truth": 18, "oracle": 18})
            runtime = (root / "output/protocol_dev_runtime_manifest.jsonl").read_text()
            self.assertNotIn("oracle", runtime); self.assertNotIn("truth_label", runtime); self.assertNotIn("reviewer", runtime)
            self.assertIn("HUMAN_ORACLE", (root / "output/protocol_dev_oracle_evidence_manifest.jsonl").read_text())
            self.assertEqual(validate_protocol_dev_freeze(freeze_manifest=root / "output/protocol_dev_freeze_manifest.json", policy_path=POLICY)["status"], "PASS")

    def test_incomplete_human_decision_only_creates_preview(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); stage_a = self.stage_a(root, blank_decision=True); result = self.build(root, stage_a, self.stage_b(stage_a))
            self.assertEqual(result["freeze_status"], "PREVIEW_ONLY")
            audit = json.loads((root / "output/protocol_dev_audit_report.json").read_text())
            self.assertIn("HUMAN_DECISION_INCOMPLETE", audit["issues"])
            self.assertEqual((root / "output/protocol_dev_runtime_manifest.jsonl").read_bytes(), b"")

    def test_source_record_uid_cannot_be_old_id_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); stage_a = self.stage_a(root); stage_a[0]["source_record_uid"] = "source-legacy-id"
            result = self.build(root, stage_a, self.stage_b(stage_a))
            self.assertEqual(result["freeze_status"], "PREVIEW_ONLY")
            self.assertIn("SOURCE_RECORD_UID_NOT_CANONICALLY_BOUND", json.loads((root / "output/protocol_dev_audit_report.json").read_text())["issues"])

    def test_source_uid_derivation_does_not_fill_human_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); stage_a = self.stage_a(root, blank_decision=True)
            stage_a[0]["source_record_uid"] = ""; source = root / "source.jsonl"; write_jsonl(source, stage_a)
            output = root / "derived.jsonl"; derive_stage_a_source_uids(stage_a_path=source, output_path=output)
            result = json.loads(output.read_text().splitlines()[0])
            self.assertTrue(result["source_record_uid"].startswith("source_record_")); self.assertEqual(result["decision"], "")

    def test_duplicate_video_and_pilot_video_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); stage_a = self.stage_a(root, duplicate_video=True)
            result = self.build(root, stage_a, self.stage_b(stage_a)); audit = json.loads((root / "output/protocol_dev_audit_report.json").read_text())
            self.assertEqual(result["freeze_status"], "PREVIEW_ONLY"); self.assertIn("DUPLICATE_PROTOCOL_DEV_VIDEO", audit["issues"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); stage_a = self.stage_a(root, pilot=True)
            self.build(root, stage_a, self.stage_b(stage_a)); audit = json.loads((root / "output/protocol_dev_audit_report.json").read_text())
            self.assertIn("PILOT_VIDEO_LEAKAGE", audit["issues"])

    def test_false_claim_must_change_only_predicate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); stage_a = self.stage_a(root); result = self.build(root, stage_a, self.stage_b(stage_a, multiple_predicate=True))
            self.assertEqual(result["freeze_status"], "PREVIEW_ONLY")
            self.assertIn("FALSE_CLAIM_MUST_CHANGE_EXACTLY_ONE_PREDICATE", json.loads((root / "output/protocol_dev_audit_report.json").read_text())["issues"])

    def test_count_media_and_split_leakage_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); stage_a = self.stage_a(root, missing_media=True)
            result = self.build(root, stage_a[:-1], self.stage_b(stage_a[:-1]))
            self.assertEqual(result["freeze_status"], "PREVIEW_ONLY")
            audit = json.loads((root / "output/protocol_dev_audit_report.json").read_text())
            self.assertIn("FROZEN_COHORT_CARDINALITY_INVALID", audit["issues"]); self.assertIn("MEDIA_PATH_UNRESOLVED", audit["issues"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); stage_a = self.stage_a(root); split = root / "split.jsonl"
            write_jsonl(split, [{"dataset": stage_a[0]["dataset"], "video_id": stage_a[0]["video_id"], "split": "blind_test"}])
            result = self.build(root, stage_a, self.stage_b(stage_a), external_split_registry=split)
            self.assertEqual(result["freeze_status"], "PREVIEW_ONLY")
            self.assertIn("CROSS_SPLIT_VIDEO_LEAKAGE", json.loads((root / "output/protocol_dev_audit_report.json").read_text())["issues"])

    def test_human_oracle_cannot_enter_automatic_runtime(self):
        with self.assertRaisesRegex(ProtocolDevManifestError, "HUMAN_ORACLE"):
            assert_automatic_runtime_safe({"case_id": "x", "oracle_evidence_tube_or_mask": {}})
        with self.assertRaisesRegex(ProtocolDevManifestError, "HUMAN_ORACLE"):
            assert_automatic_runtime_safe({"truth_label": "TRUE"})

    def test_policy_or_precondition_tampering_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); stage_a = self.stage_a(root); a, b = root / "a.jsonl", root / "b.jsonl"
            write_jsonl(a, stage_a); write_jsonl(b, self.stage_b(stage_a)); precondition = self.precondition(root)
            value = json.loads(precondition.read_text()); value["policy_sha256"] = digest("wrong")
            value["freeze_report_sha256"] = stable_hash({key: item for key, item in value.items() if key != "freeze_report_sha256"}); write_json(precondition, value)
            with self.assertRaisesRegex(ProtocolDevManifestError, "PRECONDITION_FREEZE_INVALID"):
                build_protocol_dev_manifests(policy_path=POLICY, precondition_freeze_report=precondition,
                    stage_a_path=a, stage_b_path=b, output_dir=root / "out")

    def test_frozen_artifact_tampering_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); stage_a = self.stage_a(root); result = self.build(root, stage_a, self.stage_b(stage_a))
            runtime = root / "output/protocol_dev_runtime_manifest.jsonl"; runtime.write_text(runtime.read_text() + "{}\n")
            with self.assertRaisesRegex(ProtocolDevManifestError, "ARTIFACT_TAMPERED"):
                validate_protocol_dev_freeze(freeze_manifest=root / "output/protocol_dev_freeze_manifest.json", policy_path=POLICY)

    def test_deterministic_frozen_outputs_and_csv_preview(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); stage_a = self.stage_a(root); stage_b = self.stage_b(stage_a)
            a, b = root / "a.jsonl", root / "b.jsonl"; write_jsonl(a, stage_a); write_jsonl(b, stage_b)
            precondition = self.precondition(root)
            first = build_protocol_dev_manifests(policy_path=POLICY, precondition_freeze_report=precondition, stage_a_path=a, stage_b_path=b, output_dir=root / "one")
            second = build_protocol_dev_manifests(policy_path=POLICY, precondition_freeze_report=precondition, stage_a_path=a, stage_b_path=b, output_dir=root / "two")
            self.assertEqual(first["manifest_content_sha256"], second["manifest_content_sha256"])
            for name in ("protocol_dev_runtime_manifest.jsonl", "protocol_dev_ground_truth_manifest.jsonl", "protocol_dev_oracle_evidence_manifest.jsonl", "protocol_dev_freeze_manifest.json", "protocol_dev_audit_report.json"):
                self.assertEqual((root / "one" / name).read_bytes(), (root / "two" / name).read_bytes())
            csv_path = root / "prefilter.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["protocol_dev_prefilter_id", "claim_type", "dataset_name", "source_video_id", "source_qa_types", "human_review_decision", "human_review_rationale"])
                writer.writeheader(); writer.writerow({"protocol_dev_prefilter_id": "p1", "claim_type": "SPATIAL_RELATION", "dataset_name": "Fixture", "source_video_id": "vid", "source_qa_types": "tal", "human_review_decision": "", "human_review_rationale": ""})
            preview = prefilter_csv_to_stage_a_preview(prefilter_csv=csv_path, output_dir=root / "preview")
            self.assertEqual(preview["status"], "PREVIEW_ONLY")
            self.assertEqual(json.loads((root / "preview/protocol_dev_prefilter_preview_report.json").read_text())["model_calls_made"], 0)

    def test_precondition_rejects_acceptance_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); acceptance = root / "acceptance"; acceptance.mkdir()
            write_json(acceptance / "differential_policy_acceptance_report.json", {"status": "PASS", "policy_sha256": digest("bad"), "model_calls_made": 0, "certificate_created": False})
            write_json(acceptance / "pilot_regression_report.json", {"status": "PASS", "new_model_calls": 0, "cache_writes": 0, "certificate_writes": 0})
            closure = {"status": "PILOT_CLOSED", "source_artifacts_unchanged": True, "label_resolution_applied": True}
            closure["closure_manifest_sha256"] = stable_hash(closure); path = root / "closure.json"; write_json(path, closure)
            with self.assertRaisesRegex(ProtocolDevManifestError, "ACCEPTANCE_OR_PILOT"):
                freeze_preconditions(policy_path=POLICY, acceptance_dir=acceptance, pilot_closure=path,
                    accepted_code_commit=ACCEPTED_COMMIT, output_dir=root / "out")


if __name__ == "__main__":
    unittest.main()
