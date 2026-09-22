from __future__ import annotations

import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path

from relive.storage.artifacts import canonical_json, stable_hash
from relive.v2.pilot_closure import PilotClosureError, freeze_pilot_closure
from relive.v2.protocol import (
    CLAIM_ROLE_TEMPLATES, PROTOCOL_VERSION, AdaptationDecision, ClaimType,
    EvidenceNode, EvidenceProgram, EvidenceTube, InterventionPlan,
    OcclusionState, ProtocolContractError, ProtocolManifest, RelationEdge,
)
from relive.v2.protocol_manifests import ProtocolManifestError, validate_split_manifests, write_manifest


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def program(claim_type: ClaimType) -> EvidenceProgram:
    roles = CLAIM_ROLE_TEMPLATES[claim_type]
    nodes = tuple(EvidenceNode(f"node-{i}", role, digest(f"frame-{i}"), None, OcclusionState.VISIBLE)
                  for i, role in enumerate(roles))
    return EvidenceProgram(
        "program", claim_type, roles, nodes,
        (EvidenceTube("tube", (nodes[0].node_id,), (0.0, 1.0), None),),
        (RelationEdge("edge", nodes[0].node_id, nodes[-1].node_id, "RELATED", None),),
        (InterventionPlan("plan", "MASK_UNION", (nodes[0].node_id,), "TIER_1", "operator-v1", 1),),
        AdaptationDecision("adapt", "STOP", 0, "cycle", "protocol fixture"),
    )


class ProtocolContractTests(unittest.TestCase):
    def test_pilot_closure_binds_applied_resolution_without_modifying_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); r0, r1 = root / "r0", root / "r1"
            def write(path, value):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(canonical_json(value) + "\n", encoding="utf-8")
            def rows(path, count):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("".join(canonical_json({"index": index}) + "\n" for index in range(count)), encoding="utf-8")
            artifacts = {name: digest(name) for name in ("human_review_validation_report.json", "observation_anchor_decisions.jsonl", "eligible_anchor_manifest.jsonl")}
            resolution = root / "resolution.json"
            resolution_data = {"format": "reviewed-anchor-duplicate-grounding-label-resolution-v1", "status": "PASS",
                "raw_grounding_sha256": digest("raw"), "source_human_review_sha256": digest("source review"),
                "source_warning_queue_sha256": digest("queue"), "source_warning_adjudication_sha256": digest("adjudication"),
                "derived_human_review_sha256": digest("derived review"), "recomputed_artifact_sha256": artifacts,
                "canonical_labels": [{"grounding_signature": digest(str(index)), "resolved_label": "ACCEPTED"} for index in range(5)],
                "canonical_label_count": 5, "review_rows_changed": 1,
                "source_raw_grounding_unchanged": True, "source_human_review_unchanged": True,
                "source_warning_queue_unchanged": True, "source_warning_adjudication_unchanged": True}
            resolution_data["manifest_content_sha256"] = stable_hash(resolution_data); write(resolution, resolution_data)
            r0_plan = {"bindings": {"label_resolution_manifest": hashlib.sha256(resolution.read_bytes()).hexdigest()}}
            write(r0 / "reviewed_anchor_intervention_plan.json", r0_plan)
            for mode, calls in (("run", 20), ("replay", 0)):
                write(r0 / mode / "reviewed_anchor_summary.json", {"candidate_count": 5, "full_formal_cohort_complete": True, "new_model_calls": calls})
                rows(r0 / mode / "reviewed_anchor_intervention_trace.jsonl", 5)
            write(r1 / "reviewed_anchor_r1_recomposition_plan.json", {"bindings": {"r0_plan_sha256": hashlib.sha256((r0 / "reviewed_anchor_intervention_plan.json").read_bytes()).hexdigest()}})
            for mode, calls in (("run", 2), ("replay", 0)):
                write(r1 / mode / "reviewed_anchor_r1_summary.json", {"new_verified_count": 0, "new_model_calls": calls})
                rows(r1 / mode / "reviewed_anchor_r1_trace.jsonl", 1)
            before = hashlib.sha256((r0 / "reviewed_anchor_intervention_plan.json").read_bytes()).hexdigest()
            result = freeze_pilot_closure(r0_output_dir=r0, r1_output_dir=r1, label_resolution_manifest=resolution, output_dir=root / "closure")
            self.assertEqual(result["status"], "PILOT_CLOSED")
            self.assertTrue(result["label_resolution_applied"])
            self.assertEqual(before, hashlib.sha256((r0 / "reviewed_anchor_intervention_plan.json").read_bytes()).hexdigest())

    def test_claim_type_routes_are_deterministic(self):
        for claim_type, roles in CLAIM_ROLE_TEMPLATES.items():
            self.assertEqual(program(claim_type).required_roles, roles)

    def test_unsupported_claim_type_is_rejected(self):
        with self.assertRaises(ValueError):
            ClaimType("UNSUPPORTED")
        with self.assertRaises(ProtocolContractError):
            EvidenceProgram("id", ClaimType.ENTITY_STATE, ("SUBJECT",), (), (), (), (),
                            AdaptationDecision("d", "STOP", 0, "key", "reason"))

    def test_schema_round_trip_and_protocol_hash_are_stable(self):
        source = program(ClaimType.CONTACT_ACTION)
        rebuilt = EvidenceProgram.from_dict(source.to_dict())
        self.assertEqual(rebuilt.to_dict(), source.to_dict())
        self.assertEqual(rebuilt.sha256, source.sha256)
        self.assertEqual(stable_hash(source.to_dict()), stable_hash(json.loads(canonical_json(source.to_dict()))))

    def test_split_video_and_frame_leakage_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_video = digest("same video")
            frame = digest("same frame")
            manifests = []
            for split in ("protocol_dev", "calibration", "blind_test"):
                row = ProtocolManifest(PROTOCOL_VERSION, digest(split), split,
                                       source_video if split in {"protocol_dev", "calibration"} else digest(split + " video"),
                                       (frame if split == "blind_test" else digest(split + " frame"),), "automatic")
                path = root / f"{split}.jsonl"; write_manifest(path, [row]); manifests.append(path)
            with self.assertRaisesRegex(ProtocolManifestError, "CROSS_SPLIT"):
                validate_split_manifests(protocol_dev=manifests[0], calibration=manifests[1], blind_test=manifests[2])

    def test_valid_disjoint_split_manifests_are_hash_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); paths = {}
            for split in ("protocol_dev", "calibration", "blind_test"):
                path = root / f"{split}.jsonl"
                write_manifest(path, [ProtocolManifest(PROTOCOL_VERSION, digest(split + " program"), split,
                    digest(split + " video"), (digest(split + " frame"),), "automatic")])
                paths[split] = path
            result = validate_split_manifests(**paths)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["model_calls_made"], 0)
            self.assertIn("validation_sha256", result)

    def test_pilot_closure_never_overwrites_existing_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "closure"; output.mkdir(); (output / "prior.txt").write_text("immutable")
            with self.assertRaisesRegex(PilotClosureError, "IMMUTABLE_OUTPUT_EXISTS"):
                freeze_pilot_closure(r0_output_dir=Path(temporary) / "r0", r1_output_dir=Path(temporary) / "r1",
                    label_resolution_manifest=Path(temporary) / "resolution.json", output_dir=output)
            self.assertEqual((output / "prior.txt").read_text(), "immutable")

    def test_protocol_modules_have_no_identity_specific_branch(self):
        root = Path(__file__).resolve().parents[1] / "src" / "relive" / "v2"
        for filename in ("protocol.py", "protocol_manifests.py", "pilot_closure.py"):
            source = (root / filename).read_text(encoding="utf-8")
            self.assertIsNone(re.search(r"(?:video_id|anchor_candidate_id)\s*(?:==|!=|in)", source))


if __name__ == "__main__":
    unittest.main()
