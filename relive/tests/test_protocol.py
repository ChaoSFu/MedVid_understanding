"""Synthetic protocol regression tests; these do not measure scientific efficacy."""
from __future__ import annotations

import ast
from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from relive.certificate import POLICY_VERSION, build_certificate
from relive.contrasts import evaluate_contrast
from relive.claim_aggregation import aggregate_claim
from relive.claims import parse_claims, parse_contrasts, prompt
from relive.coverage import assess_coverage
from relive.adaptation import choose_action
from relive.interventions import apply_intervention, check_spatial, generate_control_regions
from relive.reasoning import strict_answer
from relive.spatial import normalized_to_pixel_bbox, parse_proposal, validate_region
from relive.types import (Claim, ContrastGroup, EvidenceCandidate, ExecutionStatus,
                          ExclusivityStatus, FinalStatus, RuntimeSample, SemanticStatus,
                          VerificationResult)
from relive.verification import execution_failure, parse_verification, strict_json


CANDIDATE = EvidenceCandidate("candidate-a", "sample-a", ("f0", "f1"), (None, None), 0, "uniform")
CLAIM = Claim("claim-a", "The instrument grasps the tissue.", required_for_question=True)
ALTERNATIVE = Claim("claim-b", "The instrument is separated from the tissue.")


def verdict(status="SUPPORTED", claim_id=CLAIM.claim_id, candidate_id=CANDIDATE.candidate_id):
    payload = {"status": status}
    if status != "INSUFFICIENT":
        payload["frame_references"] = ["f0"]
    return parse_verification(json.dumps(payload), "raw-synthetic",
                              input_references={"candidate_id": candidate_id, "claim_id": claim_id,
                                                "frame_ids": list(CANDIDATE.frame_ids)})


def candidate_verdict(status, candidate, claim=CLAIM):
    payload = {"status": status}
    if status != "INSUFFICIENT":
        payload["frame_references"] = [candidate.frame_ids[0]]
    return parse_verification(json.dumps(payload), "raw-synthetic",
                              input_references={"candidate_id": candidate.candidate_id,
                                                "claim_id": claim.claim_id,
                                                "frame_ids": list(candidate.frame_ids)})


def group(exclusivity=ExclusivityStatus.DECLARED_EXCLUSIVE):
    return ContrastGroup("contrast-a", (CLAIM.claim_id, ALTERNATIVE.claim_id), "visible contact",
                         exclusivity, {"source": "human_synthetic_fixture"})


def spatial_pass(require_controls=True):
    return check_spatial(verdict(), verdict(), verdict("INSUFFICIENT"),
                         [verdict(), verdict()] if require_controls else [],
                         require_controls, require_controls, 2 if require_controls else 0)


class VerificationParserTests(unittest.TestCase):
    def test_all_three_semantic_states_are_preserved(self):
        for status in SemanticStatus:
            with self.subTest(status=status):
                result = verdict(status.value)
                self.assertEqual(result.semantic_status, status)
                self.assertEqual(result.execution_status, ExecutionStatus.OK)
                self.assertEqual(result.raw_response_ref, "raw-synthetic")

    def test_invalid_responses_are_technical_not_semantic(self):
        invalid = ["", "```json\n{}\n```", "[]", "null", "{}", '{"status": true}',
                   '{"status":"NO"}', '{"status":"SUPPORTED","reasoning":"invented"}',
                   '{"status":"SUPPORTED","observation":42}',
                   '{"status":"SUPPORTED","frame_references":[1]}',
                   '{"status":"SUPPORTED"}', '{"status":"CONTRADICTED","frame_references":[]}',
                   '{"status":"SUPPORTED","status":"CONTRADICTED"}',
                   '{"status":"SUPPORTED","observation":NaN}']
        for raw in invalid:
            with self.subTest(raw=raw):
                result = parse_verification(raw)
                self.assertIsNone(result.semantic_status)
                self.assertEqual(result.execution_status, ExecutionStatus.PARSE_ERROR)
                self.assertTrue(result.failure_reason)

    def test_frame_references_must_belong_to_provided_evidence(self):
        inputs = {"frame_ids": ["f0", "f1"]}
        result = parse_verification('{"status":"INSUFFICIENT","frame_references":["f1"]}',
                                    input_references=inputs)
        self.assertEqual(result.frame_references, ("f1",))
        for frame_refs in (["other"], ["f0", "f0"]):
            result = parse_verification(json.dumps({"status": "SUPPORTED", "frame_references": frame_refs}),
                                        input_references=inputs)
            self.assertEqual(result.execution_status, ExecutionStatus.PARSE_ERROR)
        self.assertEqual(inputs, {"frame_ids": ["f0", "f1"]})

    def test_compact_frame_index_resolves_to_a_supplied_runtime_id(self):
        inputs = {"frame_ids": ["a-very-long-runtime-frame-id-0", "a-very-long-runtime-frame-id-1"]}
        result = parse_verification('{"status":"SUPPORTED","frame_index":1}', input_references=inputs)
        self.assertEqual(result.execution_status, ExecutionStatus.OK)
        self.assertEqual(result.frame_references, ("a-very-long-runtime-frame-id-1",))
        for raw in ('{"status":"SUPPORTED","frame_index":-1}',
                    '{"status":"SUPPORTED","frame_index":2}',
                    '{"status":"SUPPORTED","frame_index":true}',
                    '{"status":"SUPPORTED","frame_index":0,"frame_references":["a-very-long-runtime-frame-id-0"]}'):
            with self.subTest(raw=raw):
                self.assertEqual(parse_verification(raw, input_references=inputs).execution_status,
                                 ExecutionStatus.PARSE_ERROR)

    def test_semantic_prompt_uses_compact_frame_indices(self):
        rendered = prompt("semantic", {"frame_count": 2})
        self.assertIn('"frame_index":0', rendered)
        self.assertIn('"frame_count": 2', rendered)
        self.assertNotIn("frame_references", rendered)

    def test_complete_json_markdown_fence_is_accepted_but_partial_fence_is_not(self):
        complete = parse_verification("```json\n{\"status\":\"SUPPORTED\",\"frame_references\":[\"f0\"]}\n```",
                                      input_references={"frame_ids": ["f0"]})
        self.assertEqual(complete.execution_status, ExecutionStatus.OK)
        self.assertEqual(complete.semantic_status, SemanticStatus.SUPPORTED)
        partial = parse_verification("```json\n{\"status\":\"SUPPORTED\"}")
        self.assertEqual(partial.execution_status, ExecutionStatus.PARSE_ERROR)

    def test_explicit_technical_failures_do_not_imply_contradiction(self):
        for status in (ExecutionStatus.INFERENCE_ERROR, ExecutionStatus.PARSE_ERROR, ExecutionStatus.INVALID_INPUT):
            result = execution_failure(status, "synthetic transport/parse failure")
            self.assertIsNone(result.semantic_status)
            self.assertEqual(result.execution_status, status)
        with self.assertRaises(ValueError):
            execution_failure(ExecutionStatus.OK, "invalid")
        with self.assertRaises(ValueError):
            VerificationResult(SemanticStatus.CONTRADICTED, ExecutionStatus.PARSE_ERROR, None, "test")
        with self.assertRaises(ValueError):
            VerificationResult(None, ExecutionStatus.OK, None, "test")

    def test_strict_json_rejects_nested_duplicate_keys_and_nonfinite(self):
        for raw in ('{"outer":{"x":1,"x":2}}', '{"x":Infinity}', '{"x":-Infinity}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                strict_json(raw)

    def test_claim_and_contrast_parsers_share_closed_json_rules(self):
        for raw in ('{"claims":[{"text":"x","text":"y"}]}',
                    '{"claims":[{"text":NaN}]}',
                    '{"alternatives":[],"comparison_dimension":"d","extra":true}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                if "claims" in raw:
                    parse_claims(raw, "sample", CANDIDATE, 2)
                else:
                    parse_contrasts(raw, CLAIM, CANDIDATE, 2)


class ContrastTests(unittest.TestCase):
    def test_nonexclusive_claims_allow_joint_support(self):
        result = evaluate_contrast(group(ExclusivityStatus.NONEXCLUSIVE),
                                   {CLAIM.claim_id: verdict(), ALTERNATIVE.claim_id: verdict(claim_id=ALTERNATIVE.claim_id)},
                                   CLAIM.claim_id)
        self.assertTrue(result["pass"])
        self.assertFalse(result["unique_supported"])
        self.assertFalse(result["exactly_one_rule_applied"])

    def test_unresolved_exclusivity_never_forces_a_winner(self):
        for alternative in SemanticStatus:
            result = evaluate_contrast(group(ExclusivityStatus.UNRESOLVED),
                                       {CLAIM.claim_id: verdict(), ALTERNATIVE.claim_id: verdict(alternative.value, ALTERNATIVE.claim_id)},
                                       CLAIM.claim_id)
            self.assertFalse(result["pass"])
            self.assertFalse(result["exactly_one_rule_applied"])
            self.assertIn("EXCLUSIVITY_UNRESOLVED", result["reasons"])

    def test_unique_support_is_distinct_from_alternative_contradiction(self):
        results = {CLAIM.claim_id: verdict(), ALTERNATIVE.claim_id: verdict("INSUFFICIENT", ALTERNATIVE.claim_id)}
        strict = evaluate_contrast(group(), results, CLAIM.claim_id, strict_alternatives=True)
        self.assertTrue(strict["unique_supported"])
        self.assertFalse(strict["alternatives_contradicted"])
        self.assertFalse(strict["pass"])
        self.assertIn("ALTERNATIVES_NOT_CONTRADICTED", strict["reasons"])
        self.assertTrue(evaluate_contrast(group(), results, CLAIM.claim_id, strict_alternatives=False)["pass"])

    def test_declared_exclusive_requires_supported_target(self):
        for target_status, other_status, reason in (("INSUFFICIENT", "INSUFFICIENT", "NO_SUPPORTED_CLAIM"),
                                                    ("SUPPORTED", "SUPPORTED", "MULTIPLE_EXCLUSIVE_CLAIMS_SUPPORTED"),
                                                    ("CONTRADICTED", "SUPPORTED", "TARGET_NOT_SUPPORTED")):
            result = evaluate_contrast(group(), {CLAIM.claim_id: verdict(target_status),
                                                ALTERNATIVE.claim_id: verdict(other_status, ALTERNATIVE.claim_id)}, CLAIM.claim_id)
            self.assertFalse(result["pass"])
            self.assertIn(reason, result["reasons"])

    def test_exclusive_pass_retains_declaration_and_references(self):
        result = evaluate_contrast(group(), {CLAIM.claim_id: verdict(),
                                            ALTERNATIVE.claim_id: verdict("CONTRADICTED", ALTERNATIVE.claim_id)}, CLAIM.claim_id)
        self.assertTrue(result["pass"])
        self.assertTrue(result["unique_supported"])
        self.assertTrue(result["alternatives_contradicted"])
        self.assertEqual(result["exclusivity_provenance"]["source"], "human_synthetic_fixture")
        self.assertEqual(set(result["references"]), {CLAIM.claim_id, ALTERNATIVE.claim_id})

    def test_missing_or_failed_alternatives_cannot_pass(self):
        for extra in ({}, {ALTERNATIVE.claim_id: execution_failure(ExecutionStatus.INFERENCE_ERROR, "offline")}):
            result = evaluate_contrast(group(), {CLAIM.claim_id: verdict(), **extra}, CLAIM.claim_id)
            self.assertFalse(result["pass"])
            self.assertIn("CONTRAST_TECHNICAL_FAILURE_OR_MISSING_RESULT", result["reasons"])


class SpatialProposalTests(unittest.TestCase):
    def test_half_open_floor_ceil_mapping(self):
        self.assertEqual(normalized_to_pixel_bbox((0.12, 0.23, 0.81, 0.91), 10, 20), (1, 4, 9, 19))
        self.assertEqual(normalized_to_pixel_bbox((0, 0, 1, 1), 13, 7), (0, 0, 13, 7))
        self.assertEqual(normalized_to_pixel_bbox((0.5, 0.5, 0.5001, 0.5001), 10, 10), (5, 5, 6, 6))

    def test_invalid_coordinates_are_rejected_without_clipping(self):
        for region in ((-0.1, 0, 1, 1), (0, 0, 1.1, 1), (0.5, 0, 0.5, 1),
                       (0.9, 0, 0.1, 1), (False, 0, 1, 1), (0, float("nan"), 1, 1),
                       (0, 0, float("inf"), 1), (0, 1), "0,0,1,1"):
            with self.subTest(region=region), self.assertRaises(ValueError):
                validate_region(region)
        for dimensions in ((0, 5), (4, -1), (True, 5), (2.5, 3)):
            with self.subTest(dimensions=dimensions), self.assertRaises(ValueError):
                normalized_to_pixel_bbox((0, 0, 1, 1), *dimensions)

    def test_target_and_support_remain_distinct(self):
        raw = json.dumps({"target_bbox": [0.1, 0.1, 0.2, 0.2], "support_region": [0, 0, 0.5, 0.5]})
        result = parse_proposal(raw, CANDIDATE, CLAIM, "raw-proposal")
        self.assertEqual(result.parser_status, ExecutionStatus.OK)
        self.assertNotEqual(result.target_bbox, result.support_region)
        self.assertEqual(result.frame_mapping["timestamps"], [None, None])
        self.assertEqual(result.frame_mapping["frame_ids"], ["f0", "f1"])
        self.assertEqual(result.provenance["raw_response_ref"], "raw-proposal")
        self.assertEqual(result.proposal_id, parse_proposal(raw, CANDIDATE, CLAIM).proposal_id)

    def test_full_frame_regions_are_retained_and_reported(self):
        result = parse_proposal('{"support_region":[0,0,1,1]}', CANDIDATE, CLAIM)
        self.assertEqual(result.parser_status, ExecutionStatus.OK)
        self.assertEqual(result.support_region, (0.0, 0.0, 1.0, 1.0))
        self.assertTrue(result.provenance["large_region"])
        self.assertTrue(result.provenance["full_frame_region"])

    def test_declared_1000_coordinate_system_is_converted_and_audited(self):
        raw = json.dumps({"support_region": [382, 400, 900, 999], "target_bbox": None,
                          "coordinate_system": "normalized_0_1000_xyxy"})
        result = parse_proposal(raw, CANDIDATE, CLAIM)
        self.assertEqual(result.parser_status, ExecutionStatus.OK)
        self.assertEqual(result.support_region, (0.382, 0.4, 0.9, 0.999))
        self.assertEqual(result.coordinate_system, "normalized_0_1_xyxy")
        self.assertEqual(result.provenance["original_coordinates"], [382, 400, 900, 999])
        self.assertEqual(result.provenance["source_coordinate_system"], "normalized_0_1000_xyxy")
        self.assertEqual(result.provenance["coordinate_conversion"], "divide_by_1000")

    def test_malformed_proposal_is_technical_error(self):
        for raw in ('{"support_region":[0,0,1000,1000]}',
                    '{"support_region":[0,0,1,1],"coordinate_system":"pixels"}',
                    '{"target_bbox":[0,0,1,1]}', '{"support_region":[0,0,1,1],"support_region":[0,0,0.5,0.5]}'):
            result = parse_proposal(raw, CANDIDATE, CLAIM)
            self.assertEqual(result.parser_status, ExecutionStatus.PARSE_ERROR)
            self.assertIsNone(result.support_region)


class InterventionTests(unittest.TestCase):
    @staticmethod
    def patterned_image(mode="RGB"):
        image = Image.new("RGB", (19, 13))
        image.putdata([((x * 73 + y * 91) % 256, (x * 101 + y * 37) % 256, (x * 43 + y * 61) % 256)
                       for y in range(13) for x in range(19)])
        return image.convert(mode)

    def test_original_and_invariant_pixels_for_every_supported_mode(self):
        region = (0.21, 0.18, 0.62, 0.79)
        for mode in ("RGB", "RGBA", "L"):
            image = self.patterned_image(mode)
            original_bytes = image.tobytes()
            for variant in ("ORIGINAL", "KEEP_TARGET", "DROP_TARGET", "DROP_MATCHED_CONTROL"):
                with self.subTest(mode=mode, variant=variant):
                    output, audit = apply_intervention(image, region, variant, 2.0)
                    self.assertEqual(output.size, image.size)
                    self.assertEqual(image.tobytes(), original_bytes)
                    self.assertTrue(audit["pixel_audit_pass"])
                    self.assertTrue(audit["resolution_preserved"])
                    x1, y1, x2, y2 = audit["pixel_bbox"]
                    for y in range(image.height):
                        for x in range(image.width):
                            inside = x1 <= x < x2 and y1 <= y < y2
                            invariant = variant == "ORIGINAL" or (inside if variant == "KEEP_TARGET" else not inside)
                            if invariant:
                                self.assertEqual(output.getpixel((x, y)), image.getpixel((x, y)))
                    self.assertEqual(audit["any_pixel_changed"], variant != "ORIGINAL")

    def test_extracted_h4_pixel_operation_matches_history_without_imports(self):
        path = Path(__file__).resolve().parents[2] / "evidence_stability/scripts/16_generate_h4_spatial_interventions.py"
        if not path.exists():
            self.skipTest("Historical repository is unavailable in the standalone package")
        tree = ast.parse(path.read_text())
        pure_functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                          and node.name in {"make_roi_image", "gaussian_blur_radius"}]
        self.assertEqual(len(pure_functions), 2)
        namespace = {"Any": object}
        exec(compile(ast.Module(body=pure_functions, type_ignores=[]), str(path), "exec"), namespace)
        image = self.patterned_image()
        region = (0.2, 0.2, 0.6, 0.7)
        pixel_box = normalized_to_pixel_bbox(region, *image.size)
        for relive_variant, historical_variant in (("KEEP_TARGET", "KEEP_ROI_V1"), ("DROP_TARGET", "DROP_ROI_V1")):
            output, _ = apply_intervention(image, region, relive_variant, 0.05 * min(image.size))
            historical = namespace["make_roi_image"](image, pixel_box, historical_variant)
            self.assertEqual(output.tobytes(), historical.tobytes())

    def test_invalid_interventions_fail_explicitly(self):
        image = self.patterned_image()
        for radius in (0, -1, True, float("nan")):
            with self.assertRaises(ValueError):
                apply_intervention(image, (0, 0, 1, 1), "KEEP_TARGET", radius)
        with self.assertRaises(ValueError):
            apply_intervention(image, (0, 0, 1, 1), "SHUFFLE", 2)

    def test_uniform_blur_is_a_structured_no_effect_not_an_exception(self):
        image = Image.new("RGB", (12, 12), "white")
        for variant in ("KEEP_TARGET", "DROP_TARGET", "DROP_MATCHED_CONTROL"):
            with self.subTest(variant=variant):
                output, audit = apply_intervention(image, (0.2, 0.2, 0.8, 0.8), variant, 2.0)
                self.assertEqual(output.tobytes(), image.tobytes())
                self.assertFalse(audit["pixel_audit_pass"])
                self.assertEqual(audit["audit_status"], "INTERVENTION_NO_EFFECT")
                self.assertFalse(audit["expected_region_changed"])

    def test_controls_are_deterministic_matched_and_disjoint(self):
        region = (0.4, 0.4, 0.6, 0.6)
        first = generate_control_regions(region, 3)
        self.assertEqual(first, generate_control_regions(region, 3))
        self.assertTrue(first["available"])
        self.assertEqual(first["valid_count"], 3)
        placed = [region]
        for control in first["regions"]:
            self.assertAlmostEqual(control[2] - control[0], 0.2)
            self.assertAlmostEqual(control[3] - control[1], 0.2)
            for previous in placed:
                overlap = max(0, min(previous[2], control[2]) - max(previous[0], control[0])) * max(0, min(previous[3], control[3]) - max(previous[1], control[1]))
                self.assertEqual(overlap, 0)
            placed.append(control)

    def test_unavailable_controls_are_explicit_and_partial_set_is_retained(self):
        full = generate_control_regions((0, 0, 1, 1), 2)
        self.assertFalse(full["available"])
        self.assertEqual(full["status"], "CONTROL_UNAVAILABLE")
        self.assertEqual(full["regions"], [])
        partial = generate_control_regions((0, 0, 0.5, 0.5), 4)
        self.assertFalse(partial["available"])
        self.assertEqual(partial["valid_count"], 3)
        self.assertEqual(len(partial["regions"]), 3)

    def test_strict_spatial_pass_and_control_less_ablation_are_separate(self):
        full = spatial_pass()
        self.assertTrue(full["pass"])
        self.assertEqual(full["status"], "REGION_SPECIFICITY_PASS")
        ablation = spatial_pass(False)
        self.assertTrue(ablation["pass"])
        self.assertEqual(ablation["status"], "KEEP_DROP_PASS")
        self.assertNotEqual(full["aggregation"], ablation["aggregation"])

    def test_spatial_failure_taxonomy(self):
        cases = [("SUPPORTED", "SUPPORTED", ["SUPPORTED"], True, "DEPENDENCE_UNRESOLVED"),
                 ("INSUFFICIENT", "INSUFFICIENT", ["SUPPORTED"], True, "KEEP_SUPPORT_LOST"),
                 ("SUPPORTED", "INSUFFICIENT", ["INSUFFICIENT"], True, "NONSPECIFIC_INTERVENTION_RESPONSE"),
                 ("SUPPORTED", "CONTRADICTED", ["SUPPORTED"], True, "INTERVENTION_CONTRADICTION"),
                 ("SUPPORTED", "INSUFFICIENT", [], False, "CONTROL_UNAVAILABLE")]
        for keep, drop, controls, available, expected in cases:
            with self.subTest(expected=expected):
                result = check_spatial(verdict(), verdict(keep), verdict(drop), [verdict(status) for status in controls], available)
                self.assertFalse(result["pass"])
                self.assertEqual(result["status"], expected)

    def test_technical_and_missing_control_results_cannot_pass(self):
        technical = execution_failure(ExecutionStatus.PARSE_ERROR, "bad output")
        for controls in ([technical], [None]):
            result = check_spatial(verdict(), verdict(), verdict("INSUFFICIENT"), controls, True)
            self.assertEqual(result["status"], "TECHNICAL_FAILURE")
            self.assertFalse(result["pass"])
            self.assertNotIn("INTERVENTION_CONTRADICTION", result["reasons"])
        result = check_spatial(verdict(), technical, verdict("INSUFFICIENT"), [verdict()], True)
        self.assertNotIn("KEEP_SUPPORT_LOST", result["reasons"])
        result = check_spatial(verdict(), verdict(), verdict("INSUFFICIENT"), [verdict()], True,
                               expected_control_count=2)
        self.assertEqual(result["status"], "CONTROL_RESULT_COUNT_MISMATCH")
        self.assertFalse(result["pass"])


class CertificateCoverageTests(unittest.TestCase):
    def test_semantic_policies_and_rejection(self):
        for status, expected in (("SUPPORTED", FinalStatus.VERIFIED), ("INSUFFICIENT", FinalStatus.UNCERTAIN),
                                 ("CONTRADICTED", FinalStatus.REJECTED)):
            certificate = build_certificate(CANDIDATE, CLAIM, verdict(status), "semantic_only")
            self.assertEqual(certificate.final_status, expected)
            self.assertEqual(certificate.policy_version, POLICY_VERSION)
            self.assertEqual(certificate.candidate_id, CANDIDATE.candidate_id)
            self.assertEqual(certificate.time_scope["timestamps"], [None, None])
        technical = execution_failure(ExecutionStatus.INFERENCE_ERROR, "offline")
        self.assertEqual(build_certificate(CANDIDATE, CLAIM, technical, "semantic_only").final_status, FinalStatus.UNCERTAIN)

    def test_required_checks_never_become_dynamically_inapplicable(self):
        for check in (None, {"pass": False, "applicability": "NOT_APPLICABLE", "required": False}):
            certificate = build_certificate(CANDIDATE, CLAIM, verdict(), "semantic_spatial", check)
            self.assertEqual(certificate.final_status, FinalStatus.UNCERTAIN)
            self.assertTrue(certificate.checks["spatial"]["required"])
            self.assertEqual(certificate.checks["spatial"]["applicability"], "REQUIRED")

    def test_control_less_policy_cannot_substitute_for_matched_controls(self):
        self.assertEqual(build_certificate(CANDIDATE, CLAIM, verdict(), "semantic_keep_drop", spatial_pass(False)).final_status,
                         FinalStatus.VERIFIED)
        certificate = build_certificate(CANDIDATE, CLAIM, verdict(), "semantic_spatial", spatial_pass(False))
        self.assertEqual(certificate.final_status, FinalStatus.UNCERTAIN)
        self.assertIn("MATCHED_CONTROLS_REQUIRED", certificate.failure_reasons)

    def test_full_protocol_and_stable_content_identity(self):
        contrast = evaluate_contrast(group(), {CLAIM.claim_id: verdict(),
                                              ALTERNATIVE.claim_id: verdict("CONTRADICTED", ALTERNATIVE.claim_id)}, CLAIM.claim_id)
        certificate = build_certificate(CANDIDATE, CLAIM, verdict(), "semantic_contrast_spatial", spatial_pass(), contrast)
        self.assertEqual(certificate.final_status, FinalStatus.VERIFIED)
        self.assertEqual(certificate.certificate_id,
                         build_certificate(CANDIDATE, CLAIM, verdict(), "semantic_contrast_spatial", spatial_pass(), contrast).certificate_id)
        self.assertNotEqual(certificate.certificate_id,
                            build_certificate(CANDIDATE, CLAIM, verdict(), "semantic_spatial", spatial_pass()).certificate_id)
        self.assertEqual(certificate.provenance["requirements_frozen"]["controls"], True)

    def test_unbound_or_mismatched_original_never_verifies_or_rejects_this_pair(self):
        for original in (parse_verification('{"status":"SUPPORTED"}'), verdict(candidate_id="other"),
                         verdict(claim_id="other"), verdict("CONTRADICTED", candidate_id="other"),
                         replace(verdict(), input_references={"candidate_id": CANDIDATE.candidate_id,
                                                            "claim_id": CLAIM.claim_id, "frame_ids": ["wrong"]})):
            certificate = build_certificate(CANDIDATE, CLAIM, original, "semantic_only")
            self.assertEqual(certificate.final_status, FinalStatus.UNCERTAIN)
            self.assertIn("ORIGINAL_REFERENCE_BINDING_MISMATCH", certificate.failure_reasons)

    def test_other_claim_spatial_and_contrast_evidence_cannot_admit_target(self):
        spatial = spatial_pass()
        spatial["references"]["keep"]["input_references"]["claim_id"] = "other"
        certificate = build_certificate(CANDIDATE, CLAIM, verdict(), "semantic_spatial", spatial)
        self.assertEqual(certificate.final_status, FinalStatus.UNCERTAIN)
        self.assertIn("SPATIAL_REFERENCE_BINDING_MISMATCH", certificate.failure_reasons)
        contrast = evaluate_contrast(group(), {CLAIM.claim_id: verdict(), ALTERNATIVE.claim_id: verdict("CONTRADICTED", ALTERNATIVE.claim_id)}, CLAIM.claim_id)
        contrast["references"][ALTERNATIVE.claim_id]["input_references"]["candidate_id"] = "other"
        certificate = build_certificate(CANDIDATE, CLAIM, verdict(), "semantic_contrast_spatial", spatial_pass(), contrast)
        self.assertIn("CONTRAST_REFERENCE_BINDING_MISMATCH", certificate.failure_reasons)

    def test_acquisition_only_has_no_reliability_certificate(self):
        certificate = build_certificate(CANDIDATE, CLAIM, verdict(), "acquisition_only")
        self.assertEqual(certificate.final_status, FinalStatus.UNCERTAIN)
        self.assertIn("ACQUISITION_ONLY_NO_VERIFICATION", certificate.failure_reasons)
        with self.assertRaises(ValueError):
            build_certificate(CANDIDATE, CLAIM, verdict(), "invented_policy")

    def test_local_support_does_not_complete_multiclaim_coverage(self):
        certificate = build_certificate(CANDIDATE, CLAIM, verdict(), "semantic_only")
        coverage = assess_coverage([CLAIM, ALTERNATIVE], [certificate])
        self.assertEqual(coverage.status, "INCOMPLETE")
        self.assertEqual(coverage.supported_claims, (CLAIM.claim_id,))
        self.assertEqual(coverage.missing_claims, (ALTERNATIVE.claim_id,))
        sample = RuntimeSample("sample-a", "action_qa", "Which two actions occur?", (), required_claims=(CLAIM, ALTERNATIVE))
        answer = strict_answer(sample, [CLAIM, ALTERNATIVE], [certificate], coverage)
        self.assertEqual(answer["status"], "ABSTAIN")
        self.assertEqual(answer["certificate_ids"], [])
        self.assertIsNone(answer["answer"])

    def test_empty_requirements_or_unresolved_relation_prevent_completion(self):
        certificate = build_certificate(CANDIDATE, CLAIM, verdict(), "semantic_only")
        self.assertEqual(assess_coverage([], [certificate]).status, "INCOMPLETE")
        coverage = assess_coverage([CLAIM], [certificate], ["temporal_order_unavailable"])
        self.assertEqual(coverage.status, "INCOMPLETE")
        self.assertEqual(coverage.missing_claims, ())

    def test_contradiction_for_same_input_is_not_overwritten_by_later_pass(self):
        accepted = build_certificate(CANDIDATE, CLAIM, verdict(), "semantic_only")
        rejected = build_certificate(CANDIDATE, CLAIM, verdict("CONTRADICTED"), "semantic_only")
        for certificates in ([rejected, accepted], [accepted, rejected]):
            coverage = assess_coverage([CLAIM], certificates)
            self.assertEqual(coverage.status, "INCOMPLETE")
            self.assertEqual(coverage.supported_claims, ())
            self.assertTrue(coverage.unresolved_relations[0].startswith("CONFLICTING_CERTIFICATES:"))

    def test_complete_strict_answer_only_uses_verified_pair_references(self):
        certificate = build_certificate(CANDIDATE, CLAIM, verdict(), "semantic_only")
        unverified = build_certificate(CANDIDATE, ALTERNATIVE, verdict("INSUFFICIENT", ALTERNATIVE.claim_id), "semantic_only")
        coverage = assess_coverage([CLAIM], [certificate, unverified])
        sample = RuntimeSample("sample-a", "action_qa", "What action is visible?", (), required_claims=(CLAIM,))
        answer = strict_answer(sample, [CLAIM, ALTERNATIVE], [certificate, unverified], coverage)
        self.assertEqual(answer["status"], "ANSWERED")
        self.assertEqual(answer["answer"], CLAIM.text)
        self.assertEqual(answer["certificate_ids"], [certificate.certificate_id])
        self.assertEqual(answer["input_references"][0]["candidate_id"], CANDIDATE.candidate_id)
        self.assertFalse(answer["fallback_used"])


class ClaimAggregationTests(unittest.TestCase):
    SECOND = EvidenceCandidate("candidate-b", "sample-a", ("f2", "f3"), (None, None), 1, "uniform")

    def certificate(self, candidate, status, claim=CLAIM):
        return build_certificate(candidate, claim, candidate_verdict(status, candidate, claim), "semantic_only")

    def test_rejected_candidate_then_verified_candidate_is_supported(self):
        certificates = [self.certificate(CANDIDATE, "CONTRADICTED"), self.certificate(self.SECOND, "SUPPORTED")]
        result = aggregate_claim(CLAIM, certificates)
        self.assertEqual(result["status"], "SUPPORTED")
        self.assertEqual(assess_coverage([CLAIM], certificates).status, "COMPLETE")

    def test_rejected_candidate_then_insufficient_candidate_abstains(self):
        certificates = [self.certificate(CANDIDATE, "CONTRADICTED"), self.certificate(self.SECOND, "INSUFFICIENT")]
        result = aggregate_claim(CLAIM, certificates)
        self.assertEqual(result["status"], "INSUFFICIENT")
        coverage = assess_coverage([CLAIM], certificates)
        sample = RuntimeSample("sample-a", "claim_verification", "Is it visible?", (), target_claim=CLAIM)
        self.assertEqual(strict_answer(sample, [CLAIM], certificates, coverage)["judgment"], "UNCERTAIN")

    def test_global_claim_rejection_is_not_global_contradiction(self):
        result = aggregate_claim(CLAIM, [self.certificate(CANDIDATE, "CONTRADICTED")])
        self.assertEqual(result["status"], "INSUFFICIENT")

    def test_explicit_exact_scope_can_be_contradicted(self):
        scoped = Claim("scoped", "The action occurs in these frames.", time_scope={"frame_ids": ["f0", "f1"]})
        certificate = self.certificate(CANDIDATE, "CONTRADICTED", scoped)
        self.assertEqual(aggregate_claim(scoped, [certificate])["status"], "CONTRADICTED")

    def test_same_scope_verified_and_rejected_is_conflict(self):
        certificates = [self.certificate(CANDIDATE, "SUPPORTED"), self.certificate(CANDIDATE, "CONTRADICTED")]
        self.assertEqual(aggregate_claim(CLAIM, certificates)["status"], "CONFLICT")
        self.assertEqual(assess_coverage([CLAIM], certificates).status, "INCOMPLETE")

    def test_missing_real_regrounding_is_auditable_stop_not_fixed_box(self):
        spatial = {"pass": False, "status": "KEEP_SUPPORT_LOST", "reasons": ["KEEP_SUPPORT_LOST"],
                   "require_controls": True}
        certificate = build_certificate(CANDIDATE, CLAIM, verdict(), "semantic_spatial", spatial)
        self.assertEqual(certificate.final_status, FinalStatus.UNCERTAIN)
        self.assertEqual(choose_action(certificate, enabled=True,
                                       allowed=["TRY_ALTERNATE_SUPPORT_REGION", "NEXT_CANDIDATE"],
                                       can_expand=False, can_alternate=False, can_next=True),
                         ("STOP", "RE_GROUNDING_NOT_IMPLEMENTED"))


if __name__ == "__main__":
    unittest.main()
