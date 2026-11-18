from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from relive.backends.base import Backend
from relive.config import load_config
from relive.interventions import OPAQUE_GRAY_OPERATOR, OPAQUE_GRAY_VERSION, resolve_intervention_spec
from relive.spatial_reliability.common import runtime_gt_audit
from relive.spatial_reliability.dataset import normalize_caption, parse_proposal, prepare_rc_dataset, route_claim_scope
from relive.spatial_reliability.engine import oracle_classification, run_state_machine
from relive.spatial_reliability.operators import apply_union_intervention, keep_residual
from relive.spatial_reliability.oracle import run_oracle_feasibility
from relive.spatial_reliability.protocol import (
    VERIFIER_PROMPT_VERSION,
    ScientificStatus,
    classify_formal_result,
    controller_action,
    verifier_prompt,
)
from relive.spatial_reliability.runtime import run_runtime
from relive.storage.artifacts import ArtifactStore
from relive.storage.cache import Budget, CachedInference
from relive.types import ExecutionStatus, SemanticStatus
from relive.verification import parse_single_frame_verification


OPERATOR = resolve_intervention_spec({"operator": OPAQUE_GRAY_OPERATOR,
                                      "operator_version": OPAQUE_GRAY_VERSION,
                                      "parameters": {"fill_rgb": [127, 127, 127]}}, 1.0)


class StubBackend(Backend):
    synthetic = True

    def __init__(self):
        super().__init__({"kind": "unit", "model": "stub", "revision": "v1", "max_retries": 0})

    def infer(self, request):
        self.calls += 1
        return '{"status":"SUPPORTED","frame_index":0}'


def proposal(box=(100, 100, 400, 400)):
    return json.dumps({"status": "PROPOSED", "bbox_normalized_0_1000": list(box), "reason": "fixture"})


def formal(original="SUPPORTED", keep="SUPPORTED", drop="INSUFFICIENT", control_constructible=True, control="SUPPORTED"):
    return {"original": original, "keep": keep, "drop": drop,
            "control_constructible": control_constructible, "control": control}


class SpatialReliabilityTests(unittest.TestCase):
    def test_rc_dedup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = root / "frame.png"; Image.new("RGB", (20, 20), "red").save(frame)
            def row(caption, qa="region_caption_gemini"):
                return {"id": "1.0$video$1$2", "qa_type": qa, "dataset_name": "fixture",
                        "metadata": {"video_id": "video"}, "video": [str(frame)],
                        "RC_info": {"start_frame": str(frame), "start_frame_bbox": [1, 1, 10, 10]},
                        "conversations": [{"from": "human", "value": "question"}, {"from": "gpt", "value": caption}]}
            source = root / "source.json"; source.write_text(json.dumps([row("A metallic needle is visible."), row("A curved needle is visible.")]))
            report = prepare_rc_dataset(source_json=source, output_dir=root / "out", dev_count=20)
            self.assertEqual(report["raw_rc_count"], 2)
            dedup = json.loads((root / "out/dedup_report.json").read_text())
            self.assertEqual(dedup["unique_spatial_regions"], 1)
            self.assertEqual(dedup["duplicate_group_count"], 1)

    def test_claim_normalizer_schema(self):
        accepted = normalize_caption("The region contains a curved metallic needle.")
        self.assertEqual(set(accepted), {"status", "claim_text", "scope", "reason", "normalizer_version"})
        self.assertEqual(accepted["status"], "ACCEPTED")
        self.assertEqual(normalize_caption("The tool moves throughout the sequence.")["status"], "EXCLUDED")

    def test_claim_scope_router(self):
        self.assertEqual(route_claim_scope("A metallic needle is visible.")[0], "LOCAL_ATOMIC")
        self.assertEqual(route_claim_scope("The needle is adjacent to the incision.")[0], "LOCAL_RELATIONAL")
        self.assertEqual(route_claim_scope("The tool moves throughout the sequence.")[0], "TEMPORAL")

    def test_runtime_manifest_has_no_gt(self):
        self.assertEqual(runtime_gt_audit([{"sample_id": "s", "claim_text": "A needle is visible."}])["status"], "PASS")
        self.assertEqual(runtime_gt_audit([{"sample_id": "s", "gt_bbox": [1, 2, 3, 4]}])["status"], "FAIL")

    def test_oracle_manifest_not_importable_by_runtime(self):
        source = (Path(__file__).parents[1] / "src/relive/spatial_reliability/runtime.py").read_text()
        self.assertNotIn("from .oracle", source)
        self.assertNotIn("import oracle", source)

    def test_bbox_schema(self):
        self.assertEqual(parse_proposal(proposal())["status"], "PROPOSED")
        self.assertEqual(parse_proposal(proposal((100, 100, 100, 400)))["status"], "PARSE_FAILURE")
        self.assertEqual(parse_proposal('{"status":"UNRESOLVED","bbox_normalized_0_1000":null,"reason":"none"}')["status"], "UNRESOLVED")

    def test_keep_operator(self):
        image = Image.new("RGB", (20, 20), "red")
        kept, audit = apply_union_intervention(image, [[0.1, 0.1, 0.5, 0.5]], "KEEP_UNION", OPERATOR)
        self.assertTrue(audit["pixel_audit_pass"])
        self.assertEqual(kept.getpixel((3, 3)), (255, 0, 0))
        self.assertEqual(kept.getpixel((18, 18)), (127, 127, 127))

    def test_drop_operator(self):
        image = Image.new("RGB", (20, 20), "red")
        dropped, audit = apply_union_intervention(image, [[0.1, 0.1, 0.5, 0.5]], "DROP_UNION", OPERATOR)
        self.assertTrue(audit["pixel_audit_pass"])
        self.assertEqual(dropped.getpixel((3, 3)), (127, 127, 127))

    def test_union_operator(self):
        image = Image.new("RGB", (20, 20), "blue")
        _, audit = apply_union_intervention(image, [[0, 0, .2, .2], [.8, .8, 1, 1]], "DROP_UNION", OPERATOR)
        self.assertTrue(audit["pixel_audit_pass"])
        self.assertEqual(audit["region_count"], 2)
        with self.assertRaisesRegex(ValueError, "ONE_OR_TWO"):
            apply_union_intervention(image, [[0, 0, .1, .1]] * 3, "DROP_UNION", OPERATOR)

    def test_keep_residual_operator(self):
        image = Image.new("RGB", (20, 20), "green")
        kept, audit = keep_residual(image, [.5, .5, .9, .9], OPERATOR)
        self.assertTrue(audit["pixel_audit_pass"])
        self.assertEqual(kept.getpixel((15, 15)), (0, 128, 0))

    def test_pixel_audit(self):
        image = Image.new("RGB", (20, 20), (127, 127, 127))
        _, audit = apply_union_intervention(image, [[.1, .1, .4, .4]], "DROP_UNION", OPERATOR)
        self.assertFalse(audit["pixel_audit_pass"])

    def test_certificate_state_machine(self):
        self.assertEqual(classify_formal_result(**formal()).value, "DEPENDENCY_VERIFIED")
        self.assertEqual(classify_formal_result(**formal(drop="SUPPORTED")).value, "SUPPORTED_EVIDENCE")
        self.assertEqual(classify_formal_result(**formal(keep="INSUFFICIENT")).value, "SPATIAL_INSUFFICIENT")
        self.assertEqual(classify_formal_result(**formal(original="INSUFFICIENT")).value, "BASE_UNSUPPORTED")

    def test_sufficiency_repair_max_one_attempt(self):
        result = run_state_machine(sample_id="s", initial_proposal_raw=proposal(),
                                   verify_region=lambda _: formal(keep="INSUFFICIENT"),
                                   refine=lambda _: proposal((50, 50, 500, 500)))
        self.assertEqual(result["repair_attempts"], 1)
        self.assertEqual(result["status"], "SPATIAL_SUFFICIENCY_UNRESOLVED")
        self.assertEqual(controller_action(ScientificStatus.SPATIAL_INSUFFICIENT, repair_attempts=1), "STOP")

    def test_residual_search_max_one_attempt(self):
        result = run_state_machine(sample_id="s", initial_proposal_raw=proposal(),
                                   verify_region=lambda _: formal(drop="SUPPORTED"),
                                   residual=lambda _: '{"status":"UNRESOLVED","bbox_normalized_0_1000":null,"reason":"none"}')
        self.assertEqual(result["residual_attempts"], 1)
        self.assertEqual(result["status"], "DEPENDENCY_UNRESOLVED")

    def test_residual_region_must_be_outside_primary(self):
        result = run_state_machine(sample_id="s", initial_proposal_raw=proposal(),
                                   verify_region=lambda _: formal(drop="SUPPORTED"),
                                   residual=lambda _: proposal((200, 200, 500, 500)),
                                   verify_residual=lambda _a, _b: "SUPPORTED")
        self.assertEqual(result["status"], "DEPENDENCY_UNRESOLVED")
        self.assertIn("RESIDUAL_PROPOSAL_OVERLAPS_PRIMARY", str(result["trace"]))

    def test_evidence_set_max_two_regions(self):
        result = run_state_machine(sample_id="s", initial_proposal_raw=proposal(),
                                   verify_region=lambda _: formal(drop="SUPPORTED"),
                                   residual=lambda _: proposal((600, 600, 900, 900)),
                                   verify_residual=lambda _a, _b: "SUPPORTED",
                                   verify_union=lambda _: {"keep": "SUPPORTED", "drop": "INSUFFICIENT"})
        self.assertEqual(len(result["regions"]), 2)
        self.assertEqual(result["status"], "EVIDENCE_SET_VERIFIED")

    def test_control_unavailable_not_semantic_failure(self):
        status = classify_formal_result(**formal(control_constructible=False, control=None))
        self.assertEqual(status, ScientificStatus.DEPENDENCY_SUPPORTED_NO_CONTROL)

    def test_control_failure_stops_adaptation(self):
        result = run_state_machine(sample_id="s", initial_proposal_raw=proposal(),
                                   verify_region=lambda _: formal(control="INSUFFICIENT"))
        self.assertEqual(result["status"], "INTERVENTION_UNSTABLE")
        self.assertEqual(result["repair_attempts"], 0)
        self.assertEqual(result["residual_attempts"], 0)
        self.assertEqual(classify_formal_result(**formal(keep="INSUFFICIENT", control="INSUFFICIENT")),
                         ScientificStatus.INTERVENTION_UNSTABLE)
        self.assertEqual(classify_formal_result(**formal(control=None)), ScientificStatus.TECHNICAL_FAILURE)

    def test_oracle_classification(self):
        self.assertEqual(oracle_classification("SUPPORTED", "SUPPORTED", "SUPPORTED"), "ORACLE_RESIDUAL_SUPPORT")
        self.assertEqual(oracle_classification("SUPPORTED", "SUPPORTED", "INSUFFICIENT"), "ORACLE_SINGLE_REGION_FEASIBLE")
        self.assertEqual(oracle_classification(None, "SUPPORTED", "INSUFFICIENT"), "ORACLE_TECHNICAL_FAILURE")

    def test_spatial_verifier_v3_closed_single_frame_contract(self):
        inputs = {"frame_ids": ["opaque-frame-id"], "image_paths": ["frame.png"], "variant": "ORIGINAL"}
        valid = parse_single_frame_verification(
            '{"status":"SUPPORTED","observation":"Visible evidence supports the claim.","frame_index":0}',
            input_references=inputs,
        )
        self.assertEqual(valid.execution_status, ExecutionStatus.OK)
        self.assertEqual(valid.semantic_status, SemanticStatus.SUPPORTED)
        self.assertEqual(valid.frame_references, ("opaque-frame-id",))

        fenced = parse_single_frame_verification(
            '```json\n{"status":"INSUFFICIENT","observation":"Evidence is not visible.","frame_index":0}\n```',
            input_references=inputs,
        )
        self.assertEqual(fenced.execution_status, ExecutionStatus.OK)
        self.assertEqual(fenced.semantic_status, SemanticStatus.INSUFFICIENT)

        invalid = {
            "legacy_numeric_references": '{"status":"SUPPORTED","observation":"evidence","frame_references":[0]}',
            "partial_markdown_fence": '```json\n{"status":"SUPPORTED","observation":"evidence","frame_index":0}',
            "text_after_fence": '```json\n{"status":"SUPPORTED","observation":"evidence","frame_index":0}\n```\nextra',
            "truncated": '{"status":"SUPPORTED","observation":"unfinished',
            "wrong_index": '{"status":"SUPPORTED","observation":"evidence","frame_index":1}',
            "missing_observation": '{"status":"SUPPORTED","frame_index":0}',
        }
        for name, raw in invalid.items():
            with self.subTest(name=name):
                self.assertEqual(
                    parse_single_frame_verification(raw, input_references=inputs).execution_status,
                    ExecutionStatus.PARSE_ERROR,
                )

    def test_spatial_verifier_v3_prompt_and_config_are_versioned(self):
        prompt = verifier_prompt(claim="A needle is visible.", variant="ORIGINAL")
        self.assertEqual(VERIFIER_PROMPT_VERSION, "relive-spatial-semantic-verifier-v3")
        self.assertIn("frame_index must be the integer 0", prompt)
        self.assertIn("Do not return frame_references", prompt)
        self.assertIn("single complete ```json Markdown wrapper is allowed", prompt)
        config_path = Path(__file__).parents[1] / "configs/qwen35_9b_relive_spatial_verifier_v3.yaml"
        config = load_config(config_path)
        self.assertEqual(config["backend"]["generation"]["max_new_tokens"], 256)
        self.assertEqual(config["spatial"]["intervention"]["operator"], "opaque_gray")

    def test_exact_cache_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); image = root / "image.png"; Image.new("RGB", (4, 4), "red").save(image)
            backend = StubBackend(); cached = CachedInference(backend, ArtifactStore(root / "cache"))
            request = {"stage": "relive_spatial_test", "prompt": "fixed", "prompt_version": "v1",
                       "image_paths": [str(image)], "frame_ids": ["f1"], "operator_version": OPAQUE_GRAY_VERSION,
                       "bbox_normalized_0_1000": [100, 100, 400, 400], "verification_mode": "ORIGINAL"}
            first = cached.call(request, Budget(2)); second_budget = Budget(2); second = cached.call(request, second_budget)
            self.assertFalse(first["cache_hit"])
            self.assertTrue(second["cache_hit"])
            self.assertEqual(second_budget.new_calls, 0)
            self.assertEqual(backend.calls, 1)

    def test_baseline_modes_are_bounded(self):
        one = run_state_machine(sample_id="s", initial_proposal_raw=proposal(), verify_region=lambda _: formal(), mode="one_shot")
        verify = run_state_machine(sample_id="s", initial_proposal_raw=proposal(), verify_region=lambda _: formal(), mode="verify_only")
        self.assertEqual(one["status"], "PROPOSAL_ONLY")
        self.assertEqual(verify["status"], "DEPENDENCY_VERIFIED")

    def test_mock_runtime_smoke_and_exact_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime_rows = []
            for index in range(3):
                frame = root / f"frame-{index}.png"; Image.new("RGB", (32, 32), (20 + index, 80, 140)).save(frame)
                runtime_rows.append({"format": "relive-spatial-rc-dataset-v1", "sample_id": f"sample-{index}",
                    "source_record_index": index, "source_record_sha256": f"{index:064x}", "video_id": f"v{index}",
                    "dataset_name": "fixture", "frame_order": 0, "frame_path": str(frame), "frame_sha256": None,
                    "claim_id": f"claim-{index}", "claim_text": "A metallic needle is visible.",
                    "claim_scope": "LOCAL_ATOMIC", "split": "development",
                    "normalizer_version": "relive-caption-to-claim-normalizer-v1",
                    "scope_router_version": "relive-spatial-claim-scope-router-v1"})
            manifest = root / "runtime.jsonl"
            manifest.write_text("".join(json.dumps(row) + "\n" for row in runtime_rows))
            config = root / "config.json"
            config.write_text(json.dumps({
                "backend": {"kind": "mock", "rules": [
                    {"match": {"stage": "relive_spatial_proposal"}, "response": {"status": "PROPOSED", "bbox_normalized_0_1000": [100,100,400,400], "reason": "fixture"}},
                    {"match": {"stage": "relive_spatial_residual"}, "response": {"status": "PROPOSED", "bbox_normalized_0_1000": [600,600,900,900], "reason": "fixture"}},
                    {"match": {"stage": "relive_spatial_verify"}, "response": {"status": "SUPPORTED", "observation": "fixture", "frame_index": 0}}
                ]},
                "spatial": {"intervention": {"operator": "opaque_gray", "operator_version": "relive-opaque-gray-hard-mask-v1", "parameters": {"fill_rgb": [127,127,127]}}}
            }))
            first = run_runtime(runtime_manifest=manifest, config_path=config, cache_dir=root / "cache",
                                output_dir=root / "run", max_samples=3, execution_mode="run")
            replay = run_runtime(runtime_manifest=manifest, config_path=config, cache_dir=root / "cache",
                                 output_dir=root / "replay", max_samples=3, execution_mode="replay")
            self.assertEqual(first["sample_count"], 3)
            self.assertGreater(first["new_model_calls"], 0)
            self.assertEqual(replay["new_model_calls"], 0)
            self.assertEqual(first["scientific_status_counts"], {"DEPENDENCY_UNRESOLVED": 3})
            self.assertEqual(replay["scientific_status_counts"], first["scientific_status_counts"])

    def test_oracle_fake_verifier_artifacts_and_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); frame = root / "frame.png"; Image.new("RGB", (32, 32), "red").save(frame)
            public = {"sample_id": "s", "source_record_sha256": "a" * 64, "split": "development",
                      "frame_path": str(frame), "frame_sha256": "b" * 64,
                      "claim_id": "c", "claim_text": "A metallic needle is visible."}
            hidden = {"sample_id": "s", "source_record_sha256": "a" * 64, "gt_bbox": [2, 2, 20, 20]}
            runtime = root / "runtime.jsonl"; runtime.write_text(json.dumps(public) + "\n")
            oracle = root / "oracle.jsonl"; oracle.write_text(json.dumps(hidden) + "\n")
            config = root / "config.json"; config.write_text(json.dumps({
                "backend": {"kind": "mock", "rules": [
                    {"match": {"stage": "relive_spatial_verify"}, "response": {"status": "SUPPORTED", "observation": "fixture", "frame_index": 0}}
                ]},
                "spatial": {"intervention": {"operator": "opaque_gray", "operator_version": "relive-opaque-gray-hard-mask-v1", "parameters": {"fill_rgb": [127,127,127]}}}
            }))
            first = run_oracle_feasibility(runtime_manifest=runtime, oracle_manifest=oracle, config_path=config,
                                           cache_dir=root / "cache", output_dir=root / "oracle-run")
            replay = run_oracle_feasibility(runtime_manifest=runtime, oracle_manifest=oracle, config_path=config,
                                            cache_dir=root / "cache", output_dir=root / "oracle-replay")
            self.assertEqual(first["classification_counts"], {"ORACLE_RESIDUAL_SUPPORT": 1})
            self.assertGreater(first["usage"]["new_calls"], 0)
            self.assertEqual(replay["usage"]["new_calls"], 0)
            for name in ("protocol.json", "sample_results.jsonl", "summary.json", "summary.md",
                         "failure_breakdown.json", "cache_audit.json", "gt_isolation_audit.json"):
                self.assertTrue((root / "oracle-run" / name).is_file())

    def test_oracle_parse_failures_are_technical_not_base_unsupported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = root / "frame.png"
            Image.new("RGB", (32, 32), "red").save(frame)
            public = {
                "sample_id": "s", "source_record_sha256": "a" * 64, "split": "development",
                "frame_path": str(frame), "frame_sha256": "b" * 64,
                "claim_id": "c", "claim_text": "A metallic needle is visible.",
            }
            hidden = {"sample_id": "s", "source_record_sha256": "a" * 64, "gt_bbox": [2, 2, 20, 20]}
            runtime = root / "runtime.jsonl"
            runtime.write_text(json.dumps(public) + "\n")
            oracle = root / "oracle.jsonl"
            oracle.write_text(json.dumps(hidden) + "\n")
            config = root / "config.json"
            config.write_text(json.dumps({
                "backend": {"kind": "mock", "rules": [{
                    "match": {"stage": "relive_spatial_verify"},
                    "response": {"status": "SUPPORTED", "observation": "fixture", "frame_references": [0]},
                }]},
                "spatial": {"intervention": {
                    "operator": "opaque_gray",
                    "operator_version": "relive-opaque-gray-hard-mask-v1",
                    "parameters": {"fill_rgb": [127, 127, 127]},
                }},
            }))
            report = run_oracle_feasibility(
                runtime_manifest=runtime, oracle_manifest=oracle, config_path=config,
                cache_dir=root / "cache", output_dir=root / "oracle-run",
            )
            self.assertEqual(report["status"], "FAIL")
            self.assertFalse(report["scientific_result_available"])
            self.assertEqual(report["classification_counts"], {"ORACLE_TECHNICAL_FAILURE": 1})
            self.assertEqual(report["N_technical_failure_cases"], 1)
            self.assertEqual(report["N_technical_failure_verdicts"], 3)
            self.assertEqual(report["N_gt_keep_failed"], 0)
            self.assertEqual(report["N_gt_drop_not_supported"], 0)


if __name__ == "__main__":
    unittest.main()
