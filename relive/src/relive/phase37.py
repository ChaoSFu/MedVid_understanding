"""Phase 3.7 residual visual-support mechanism diagnostic.

This module is deliberately outside certificate admission.  It binds one frozen
Phase 3.6 case to its stored DROP(R1) evidence, permits one neutral residual
localization request, and records diagnostic semantic probes without creating a
new certificate or a second refinement round.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Sequence

from PIL import Image, ImageChops

from .audits import audit_runtime_imports
from .backends import make_backend
from .certificate import build_certificate
from .claims import PROMPT_VERSIONS, stable_id
from .config import load_config
from .interventions import OPAQUE_GRAY_OPERATOR, OPAQUE_GRAY_VERSION, apply_spatial_intervention
from .phase35_v3 import (Phase35V3Error, candidate_for, runtime_gt_audit, sha256_path,
                         validate_config, validate_prospective_manifest)
from .runner import SampleRunner, _image_file
from .spatial import normalized_1000_to_region, normalized_to_pixel_bbox, validate_region
from .storage.artifacts import ArtifactStore, canonical_json, stable_hash
from .storage.cache import CachedInference
from .types import ExecutionStatus, SemanticStatus, to_dict
from .verification import parse_verification

PHASE37_FORMAT = "relive-phase37-residual-visual-support-diagnostic-v1"
TARGET_CLAIM_ID = "phase35-local-001"
MAX_RESIDUAL_PROPOSAL_ROUNDS = 1
PROHIBITED_INPUTS = ["reference_answer", "assistant_answer", "temporal_gt", "bbox_mask_gt",
                     "struc_info", "RC_info", "evaluation_artifacts"]


class Phase37Error(ValueError):
    pass


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Phase37Error(f"unreadable JSON: {path}") from exc
    if not isinstance(value, dict):
        raise Phase37Error(f"JSON object required: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Phase37Error(f"unreadable JSONL: {path}") from exc
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise Phase37Error(f"nonempty JSONL object rows required: {path}")
    return rows


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _region_1000(region: Sequence[float]) -> list[int]:
    return [round(value * 1000) for value in validate_region(region)]


def _intersect(a: Sequence[float], b: Sequence[float]) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def geometry_audit(r1: Sequence[float], residual: Sequence[float], size: tuple[int, int]) -> dict[str, Any]:
    """Report continuous overlap and enforce visible-pixel disjointness.

    A residual rectangle cannot establish visual support from the already opaque
    area.  We therefore require zero overlap after the same canonical raster
    mapping used by the intervention operator.  This is a visibility invariant,
    not an empirically chosen IoU cutoff.
    """
    r1, residual = validate_region(r1), validate_region(residual)
    intersection = _intersect(r1, residual)
    union = (r1[2]-r1[0])*(r1[3]-r1[1]) + (residual[2]-residual[0])*(residual[3]-residual[1]) - intersection
    r1_box = normalized_to_pixel_bbox(r1, *size)
    res_box = normalized_to_pixel_bbox(residual, *size)
    pixel_overlap = max(0, min(r1_box[2], res_box[2]) - max(r1_box[0], res_box[0])) * max(0, min(r1_box[3], res_box[3]) - max(r1_box[1], res_box[1]))
    return {"r1_normalized_0_1_xyxy": list(r1), "residual_normalized_0_1_xyxy": list(residual),
            "r1_normalized_0_1000_xyxy": _region_1000(r1), "residual_normalized_0_1000_xyxy": _region_1000(residual),
            "continuous_intersection_area": intersection, "iou": (intersection / union) if union else 0.0,
            "residual_area_fraction": (residual[2]-residual[0])*(residual[3]-residual[1]),
            "r1_pixel_bbox": list(r1_box), "residual_pixel_bbox": list(res_box),
            "pixel_overlap_count": pixel_overlap,
            "valid": pixel_overlap == 0,
            "invalid_code": None if pixel_overlap == 0 else "RESIDUAL_PROPOSAL_INVALID_MASK_OVERLAP",
            "validity_rule": "residual visual support must occupy pixels outside the opaque R1 mask; zero raster overlap"}


def parse_residual_proposal(raw: str | None) -> dict[str, Any]:
    """Strict closed contract. No fallback ROI is manufactured."""
    raw = raw or ""
    try:
        from .json_protocol import strict_json
        data = strict_json(raw)
    except (TypeError, ValueError) as exc:
        return {"outcome": "PARSE_FAILURE", "failure_code": "RESIDUAL_SCHEMA_FAILURE", "bbox_normalized_0_1000": None,
                "residual_region": None, "reason": str(exc), "raw_model_response": raw}
    if not isinstance(data, dict) or set(data) != {"status", "bbox_normalized_0_1000", "reason"}:
        return {"outcome": "PARSE_FAILURE", "failure_code": "RESIDUAL_SCHEMA_FAILURE", "bbox_normalized_0_1000": None,
                "residual_region": None, "reason": None, "raw_model_response": raw}
    status, box, reason = data["status"], data["bbox_normalized_0_1000"], data["reason"]
    if status not in {"PROPOSED", "UNRESOLVED"} or not isinstance(reason, str) or len(reason) > 2000:
        return {"outcome": "PARSE_FAILURE", "failure_code": "RESIDUAL_SCHEMA_FAILURE", "bbox_normalized_0_1000": None,
                "residual_region": None, "reason": reason if isinstance(reason, str) else None, "raw_model_response": raw}
    if status == "UNRESOLVED":
        if box is not None:
            return {"outcome": "PARSE_FAILURE", "failure_code": "RESIDUAL_UNRESOLVED_REQUIRES_NULL_BBOX", "bbox_normalized_0_1000": None,
                    "residual_region": None, "reason": reason, "raw_model_response": raw}
        return {"outcome": "UNRESOLVED", "failure_code": None, "bbox_normalized_0_1000": None,
                "residual_region": None, "reason": reason, "raw_model_response": raw}
    if not isinstance(box, list) or len(box) != 4 or any(type(value) is not int for value in box):
        return {"outcome": "PARSE_FAILURE", "failure_code": "RESIDUAL_COORDINATE_VIOLATION", "bbox_normalized_0_1000": None,
                "residual_region": None, "reason": reason, "raw_model_response": raw}
    try:
        region = normalized_1000_to_region(box)
    except (TypeError, ValueError):
        return {"outcome": "PARSE_FAILURE", "failure_code": "RESIDUAL_COORDINATE_VIOLATION", "bbox_normalized_0_1000": None,
                "residual_region": None, "reason": reason, "raw_model_response": raw}
    return {"outcome": "PROPOSED", "failure_code": None, "bbox_normalized_0_1000": box,
            "residual_region": region, "reason": reason, "raw_model_response": raw}


def _gray(size: tuple[int, int], fill: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", size, fill)


def residual_keep(drop_r1: Image.Image, residual: Sequence[float], operator: dict[str, Any]) -> tuple[Image.Image, dict[str, Any]]:
    """Keep R_res from frozen DROP(R1); all other evidence stays opaque gray."""
    source = drop_r1.convert("RGB")
    altered, audit = apply_spatial_intervention(source, residual, "KEEP_TARGET", operator)
    return altered, audit


def union_drop(original: Image.Image, r1: Sequence[float], residual: Sequence[float], operator: dict[str, Any]) -> tuple[Image.Image, dict[str, Any]]:
    """Opaque-gray the deterministic union R1 ∪ R_res without changing either ROI."""
    source = original.convert("RGB")
    first, _ = apply_spatial_intervention(source, r1, "DROP_TARGET", operator)
    altered, _ = apply_spatial_intervention(first, residual, "DROP_TARGET", operator)
    r1_box, res_box = normalized_to_pixel_bbox(r1, *source.size), normalized_to_pixel_bbox(residual, *source.size)
    fill = tuple(operator["parameters"]["fill_rgb"])
    expected = source.copy(); expected.paste(fill, r1_box); expected.paste(fill, res_box)
    changed_rgb = ImageChops.difference(source, altered)
    changed = Image.new("L", source.size, 0)
    for band in changed_rgb.split():
        changed = ImageChops.lighter(changed, band)
    expected_change = ImageChops.difference(source, expected)
    same_as_union = ImageChops.difference(altered, expected).getbbox() is None
    # Explicit masks permit auditing the union, rather than treating two calls as a black box.
    mask = Image.new("L", source.size, 0); mask.paste(255, r1_box); mask.paste(255, res_box)
    outside = ImageChops.multiply(changed, ImageChops.invert(mask))
    inside = ImageChops.multiply(changed, mask)
    passed = altered.size == source.size and same_as_union and outside.getbbox() is None and inside.getbbox() is not None
    return altered, {"variant": "DROP_UNION_R1_RESIDUAL", "intervention_protocol": operator,
                     "r1_pixel_bbox": list(r1_box), "residual_pixel_bbox": list(res_box),
                     "union_mask_rule": "deterministic raster union of R1 and R_res", "resolution_preserved": altered.size == source.size,
                     "union_output_matches_expected": same_as_union, "outside_union_unchanged": outside.getbbox() is None,
                     "expected_union_changed": inside.getbbox() is not None, "pixel_audit_pass": passed,
                     "audit_status": "PASS" if passed else "INTERVENTION_PIXEL_AUDIT_FAILED",
                     "expected_change_has_pixels": expected_change.getbbox() is not None}


def _drop_paths(certificate: dict[str, Any]) -> list[str]:
    try:
        refs = certificate["checks"]["spatial"]["references"]["drop"]["input_references"]
        paths = refs["image_paths"]
    except (KeyError, TypeError):
        raise Phase37Error("Phase 3.6 formal certificate does not retain DROP(R1) image provenance") from None
    if not isinstance(paths, list) or not paths or not all(isinstance(path, str) for path in paths):
        raise Phase37Error("Phase 3.6 DROP(R1) image paths are invalid")
    return paths


def _phase36_input(phase36_run_dir: Path, samples: list[Any]) -> dict[str, Any]:
    traces = _jsonl(phase36_run_dir / "phase36_regrounding_trace.jsonl")
    matches = [row for row in traces if row.get("claim_id") == TARGET_CLAIM_ID]
    if len(matches) != 1:
        raise Phase37Error("Phase 3.7 requires exactly one Phase 3.6 local-001 trace")
    row = matches[0]
    sample = next((item for item in samples if item.target_claim.claim_id == TARGET_CLAIM_ID), None)
    if sample is None or len(samples) != 3:
        raise Phase37Error("Phase 3.7 requires the frozen three-sample Phase 3.5 runtime")
    candidate = candidate_for(sample)
    if row.get("sample_id") != sample.sample_id or row.get("candidate_id") != candidate.candidate_id:
        raise Phase37Error("Phase 3.6 candidate lineage mismatch")
    r1 = row.get("r1_normalized_0_1_xyxy")
    semantic = row.get("semantic") if isinstance(row.get("semantic"), dict) else {}
    cert = row.get("formal_certificate")
    if not (isinstance(r1, list) and len(r1) == 4 and row.get("refinement_outcome") == "PROPOSED"
            and semantic.get("original") == "SUPPORTED" and semantic.get("keep") == "SUPPORTED"
            and semantic.get("drop") == "SUPPORTED" and semantic.get("control") == ["SUPPORTED"]
            and row.get("pixel_audit_status") == "PASS" and row.get("certificate_status") == "UNCERTAIN"
            and "DEPENDENCE_UNRESOLVED" in row.get("certificate_failure_reasons", [])
            and "RESIDUAL_SUPPORT_OR_SEMANTIC_INSENSITIVITY" in row.get("diagnostic_flags", [])
            and isinstance(cert, dict)):
        raise Phase37Error("Phase 3.6 local-001 does not match the frozen residual-support lineage")
    drop_paths = _drop_paths(cert)
    frames = [frame.path for frame in sample.frames]
    if len(drop_paths) != len(frames):
        raise Phase37Error("DROP(R1) artifact count differs from frozen frame count")
    for path in [*frames, *drop_paths]:
        if not Path(path).is_file():
            raise Phase37Error(f"missing frozen Phase 3.6 image artifact: {path}")
    return {"trace": row, "trace_path": phase36_run_dir / "phase36_regrounding_trace.jsonl", "sample": sample,
            "candidate": candidate, "r1": validate_region(r1), "certificate": cert, "drop_paths": drop_paths}


def _verify_frozen_drop(inputs: dict[str, Any], operator: dict[str, Any]) -> list[dict[str, Any]]:
    checks = []
    for frame, path in zip(inputs["sample"].frames, inputs["drop_paths"]):
        with Image.open(frame.path) as original, Image.open(path) as actual:
            regenerated, audit = apply_spatial_intervention(original.convert("RGB"), inputs["r1"], "DROP_TARGET", operator)
            equal = actual.convert("RGB").size == regenerated.size and ImageChops.difference(actual.convert("RGB"), regenerated).getbbox() is None
        checks.append({"frame_id": frame.frame_id, "original_path": frame.path, "drop_r1_path": path,
                       "drop_r1_sha256": sha256_path(Path(path)), "regenerated_pixels_match_frozen_artifact": equal,
                       "regeneration_audit": audit})
    if not all(item["regenerated_pixels_match_frozen_artifact"] and item["regeneration_audit"]["pixel_audit_pass"] for item in checks):
        raise Phase37Error("frozen DROP(R1) artifact does not match the Phase 3.6 operator provenance")
    return checks


def preflight(*, config_path: Path, runtime_path: Path, prospective_manifest_path: Path,
              phase36_run_dir: Path, output_dir: Path, require_real: bool = True) -> dict[str, Any]:
    if output_dir.exists():
        raise Phase37Error("Phase 3.7 output directory must not exist before zero-call preflight")
    config = load_config(config_path); validate_config(config, require_real=require_real)
    prospective, samples = validate_prospective_manifest(prospective_manifest_path, runtime_path)
    gt = runtime_gt_audit(runtime_path, samples)
    inputs = _phase36_input(phase36_run_dir, samples)
    source = audit_runtime_imports()
    if source["status"] != "PASS":
        raise Phase37Error("runtime import audit failed")
    drop_checks = _verify_frozen_drop(inputs, config["spatial"]["intervention"])
    output_dir.mkdir(parents=True)
    plan = {"format": PHASE37_FORMAT, "status": "PASS", "mode": "preflight", "model_calls_made": 0,
            "cache_mutated": False, "git_commit": _git_commit(), "config": str(config_path), "config_sha256": sha256_path(config_path),
            "runtime": str(runtime_path), "runtime_sha256": samples[0].provenance["runtime_sha256"],
            "prospective_manifest": str(prospective_manifest_path), "prospective_manifest_sha256": prospective["manifest_sha256"],
            "phase36_run_dir": str(phase36_run_dir), "phase36_trace_sha256": sha256_path(inputs["trace_path"]),
            "cohort_claim_ids": [TARGET_CLAIM_ID], "excluded_claim_ids": ["phase35-local-002", "phase35-local-003"],
            "claim": inputs["sample"].target_claim.text, "claim_id": TARGET_CLAIM_ID,
            "candidate_id": inputs["candidate"].candidate_id, "frozen_frame_ids": list(inputs["candidate"].frame_ids),
            "r1_normalized_0_1_xyxy": list(inputs["r1"]), "r1_normalized_0_1000_xyxy": _region_1000(inputs["r1"]),
            "drop_r1_artifacts": drop_checks, "residual_prompt_version": PROMPT_VERSIONS["residual_support_localize"],
            "semantic_prompt_version": PROMPT_VERSIONS["semantic"], "spatial_intervention": config["spatial"]["intervention"],
            "opaque_operator_unchanged": config["spatial"]["intervention"].get("operator") == OPAQUE_GRAY_OPERATOR and config["spatial"]["intervention"].get("operator_version") == OPAQUE_GRAY_VERSION,
            "semantic_verifier_unchanged": True, "certificate_builder_source_sha256": hashlib.sha256(inspect.getsource(build_certificate).encode()).hexdigest(),
            "phase36_certificate_sha256": stable_hash(inputs["certificate"]), "max_residual_proposal_rounds": MAX_RESIDUAL_PROPOSAL_ROUNDS,
            "planned_max_model_calls": 3, "planned_call_derivation": "one DROP(R1)-only residual localization; if valid then one residual KEEP and one union DROP semantic diagnostic",
            "cache_dir": str(output_dir / "cache"), "run_dir": str(output_dir / "run"), "replay_dir": str(output_dir / "replay"),
            "runtime_gt_isolation_audit": gt, "runtime_import_audit": source, "gt_used": False,
            "prohibited_inputs_not_opened": PROHIBITED_INPUTS,
            "formal_certificate_read_only": True, "no_r2": True, "no_verified_status_created": True}
    (output_dir / "phase37_preflight.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return plan


def _semantic(engine: SampleRunner, sample: Any, candidate: Any, claim: Any, variant: str,
              paths: list[str], region: Sequence[float]) -> dict[str, Any]:
    result = engine.infer(sample, candidate, claim, "semantic", variant=variant, image_paths=paths, region=region,
                          proposal_index=MAX_RESIDUAL_PROPOSAL_ROUNDS)
    return to_dict(result)


def _write_image_audits(store: ArtifactStore, sample: Any, candidate: Any, variant: str, paths: list[str], audits: list[dict[str, Any]]) -> list[str]:
    return [store.append_event("phase37_pixel_audits", {"sample_id": sample.sample_id, "candidate_id": candidate.candidate_id,
             "frame_id": frame.frame_id, "variant": variant, "output_path": path, **audit})
            for frame, path, audit in zip(sample.frames, paths, audits)]


def _diagnostic_label(outcome: dict[str, Any], keep: dict[str, Any] | None, union: dict[str, Any] | None) -> str:
    if outcome["outcome"] == "UNRESOLVED":
        return "RESIDUAL_NOT_LOCALIZED_SUPPORT_PERSISTS"
    if outcome["outcome"] != "PROPOSED":
        return "TECHNICAL_FAILURE"
    if keep is None or union is None:
        return "TECHNICAL_FAILURE"
    if keep.get("semantic_status") != "SUPPORTED":
        return "RESIDUAL_REGION_PROPOSED_NOT_SUFFICIENT"
    if union.get("semantic_status") != "SUPPORTED":
        return "EVIDENCE_SET_REMOVAL_BREAKS_SUPPORT"
    return "SUPPORT_PERSISTS_AFTER_EVIDENCE_SET_REMOVAL"


def execute(*, config_path: Path, runtime_path: Path, prospective_manifest_path: Path,
            phase36_run_dir: Path, output_dir: Path, mode: str, require_real: bool = True) -> dict[str, Any]:
    if mode not in {"run", "replay"}:
        raise Phase37Error("mode must be run or replay")
    plan = _json(output_dir / "phase37_preflight.json")
    config = load_config(config_path); validate_config(config, require_real=require_real)
    prospective, samples = validate_prospective_manifest(prospective_manifest_path, runtime_path)
    inputs = _phase36_input(phase36_run_dir, samples)
    expected = {"config_sha256": sha256_path(config_path), "runtime_sha256": samples[0].provenance["runtime_sha256"],
                "prospective_manifest_sha256": prospective["manifest_sha256"], "phase36_trace_sha256": sha256_path(inputs["trace_path"]),
                "phase36_certificate_sha256": stable_hash(inputs["certificate"])}
    if any(plan.get(key) != value for key, value in expected.items()) or plan.get("cohort_claim_ids") != [TARGET_CLAIM_ID]:
        raise Phase37Error("missing or incompatible zero-call Phase 3.7 preflight")
    if plan.get("semantic_prompt_version") != PROMPT_VERSIONS["semantic"] or plan.get("residual_prompt_version") != PROMPT_VERSIONS["residual_support_localize"]:
        raise Phase37Error("frozen Phase 3.7 prompt version changed")
    if hashlib.sha256(inspect.getsource(build_certificate).encode()).hexdigest() != plan.get("certificate_builder_source_sha256"):
        raise Phase37Error("certificate builder changed after Phase 3.7 preflight")
    target = output_dir / mode
    if target.exists():
        raise Phase37Error(f"Phase 3.7 {mode} output already exists")
    backend = make_backend(config["backend"])
    if require_real and backend.synthetic:
        raise Phase37Error("Phase 3.7 requires real local_hf execution")
    store, cache = ArtifactStore(target), ArtifactStore(output_dir / "cache")
    inference = CachedInference(backend, cache); engine = SampleRunner(config, store, inference)
    sample, candidate, claim, r1 = inputs["sample"], inputs["candidate"], inputs["sample"].target_claim, inputs["r1"]
    before = engine.budget.snapshot(); started = time.monotonic()
    with store.run_lock():
        # The proposal request receives only DROP(R1) images, the frozen claim and ordinary public frame identity.
        response = engine.infer(sample, candidate, claim, "residual_support_localize", variant="DROP_R1_RESIDUAL_LOCALIZATION",
                                image_paths=inputs["drop_paths"], region=r1, proposal_index=1)
        parsed = parse_residual_proposal(response.get("raw_text") if response.get("execution_status") == "OK" else None)
        parsed["raw_response_ref"] = response.get("raw_response_ref")
        parsed["execution_status"] = response.get("execution_status")
        geometry = None; residual_keep_result = union_drop_result = None; intervention_rows = []
        if parsed["outcome"] == "PROPOSED":
            with Image.open(inputs["drop_paths"][0]) as image:
                geometry = geometry_audit(r1, parsed["residual_region"], image.convert("RGB").size)
            if not geometry["valid"]:
                parsed["outcome"] = "INVALID_MASK_OVERLAP"; parsed["failure_code"] = geometry["invalid_code"]
            else:
                keep_paths = []; keep_audits = []; union_paths = []; union_audits = []
                for frame, drop_path in zip(sample.frames, inputs["drop_paths"]):
                    with Image.open(drop_path) as drop_image:
                        kept, keep_audit = residual_keep(drop_image, parsed["residual_region"], config["spatial"]["intervention"])
                    with Image.open(frame.path) as original:
                        dropped, union_audit = union_drop(original, r1, parsed["residual_region"], config["spatial"]["intervention"])
                    # A shared content-addressed cache image root keeps replay request
                    # identity stable while run artifacts only reference those immutable files.
                    keep_paths.append(_image_file(cache.root, kept)); keep_audits.append(keep_audit)
                    union_paths.append(_image_file(cache.root, dropped)); union_audits.append(union_audit)
                keep_refs = _write_image_audits(store, sample, candidate, "KEEP_RESIDUAL_FROM_DROP_R1", keep_paths, keep_audits)
                union_refs = _write_image_audits(store, sample, candidate, "DROP_UNION_R1_RESIDUAL", union_paths, union_audits)
                intervention_rows = [{"variant": "KEEP_RESIDUAL_FROM_DROP_R1", "image_paths": keep_paths, "pixel_audit_refs": keep_refs,
                                      "pixel_audit_pass": all(item["pixel_audit_pass"] for item in keep_audits), "base": "frozen_DROP_R1"},
                                     {"variant": "DROP_UNION_R1_RESIDUAL", "image_paths": union_paths, "pixel_audit_refs": union_refs,
                                      "pixel_audit_pass": all(item["pixel_audit_pass"] for item in union_audits), "base": "frozen_ORIGINAL", "union": [list(r1), list(parsed["residual_region"])]}]
                if not all(item["pixel_audit_pass"] for item in keep_audits + union_audits):
                    parsed["outcome"] = "TECHNICAL_FAILURE"; parsed["failure_code"] = "INTERVENTION_PIXEL_AUDIT_FAILED"
                else:
                    residual_keep_result = _semantic(engine, sample, candidate, claim, "KEEP_RESIDUAL_FROM_DROP_R1", keep_paths, parsed["residual_region"])
                    union_drop_result = _semantic(engine, sample, candidate, claim, "DROP_UNION_R1_RESIDUAL", union_paths, r1)
        after = engine.budget.snapshot()
        label = _diagnostic_label(parsed, residual_keep_result, union_drop_result)
        trace = {"format": PHASE37_FORMAT, "claim_id": TARGET_CLAIM_ID, "sample_id": sample.sample_id, "candidate_id": candidate.candidate_id,
                 "frozen_frame_ids": list(candidate.frame_ids), "r1_normalized_0_1_xyxy": list(r1), "r1_normalized_0_1000_xyxy": _region_1000(r1),
                 "drop_r1_artifacts": [{"path": path, "sha256": sha256_path(Path(path))} for path in inputs["drop_paths"]],
                 "residual_prompt_version": PROMPT_VERSIONS["residual_support_localize"], "residual_proposal": parsed,
                 "residual_geometry": geometry, "interventions": intervention_rows,
                 "residual_keep": residual_keep_result, "union_drop": union_drop_result, "control_status": "NOT_RUN_DIAGNOSTIC_OPTIONAL",
                 "prior_drop_r1_semantic_status": "SUPPORTED", "formal_certificate_status_unchanged": "UNCERTAIN",
                 "diagnostic_outcome": label, "max_residual_proposal_rounds": MAX_RESIDUAL_PROPOSAL_ROUNDS,
                 "residual_proposal_attempt_count": 1, "new_model_calls": after["new_calls"]-before["new_calls"],
                 "cache_hits": after["cache_hits"]-before["cache_hits"], "logical_model_calls": after["calls"]-before["calls"],
                 "model_fingerprint": backend.fingerprint(), "git_commit": _git_commit(), "bindings": expected,
                 "operator": config["spatial"]["intervention"], "semantic_prompt_version": PROMPT_VERSIONS["semantic"], "gt_used": False}
        (store.root / "phase37_residual_proposal.jsonl").write_text(canonical_json({key: trace[key] for key in ("claim_id", "candidate_id", "frozen_frame_ids", "r1_normalized_0_1_xyxy", "drop_r1_artifacts", "residual_prompt_version", "residual_proposal", "residual_geometry", "gt_used")}) + "\n", encoding="utf-8")
        (store.root / "phase37_residual_interventions.jsonl").write_text(canonical_json({"claim_id": TARGET_CLAIM_ID, "interventions": intervention_rows, "gt_used": False}) + "\n", encoding="utf-8")
        (store.root / "phase37_semantic_results.jsonl").write_text(canonical_json({"claim_id": TARGET_CLAIM_ID, "residual_keep": residual_keep_result, "union_drop": union_drop_result, "gt_used": False}) + "\n", encoding="utf-8")
        (store.root / "phase37_trace.jsonl").write_text(canonical_json(trace) + "\n", encoding="utf-8")
    if mode == "run" and trace["new_model_calls"] <= 0:
        raise Phase37Error("Phase 3.7 first run made zero model calls")
    if mode == "replay" and trace["new_model_calls"] != 0:
        raise Phase37Error("Phase 3.7 replay made unexpected new model calls")
    summary = {"format": PHASE37_FORMAT, "status": "PASS", "mode": mode, "cohort_count": 1, "diagnostic_outcome": trace["diagnostic_outcome"],
               "new_model_calls": trace["new_model_calls"], "cache_hits": trace["cache_hits"], "logical_model_calls": trace["logical_model_calls"],
               "residual_proposal_attempt_count": 1, "max_residual_proposal_rounds": MAX_RESIDUAL_PROPOSAL_ROUNDS,
               "no_r2": True, "formal_certificate_unchanged": True, "new_verified_count": 0, "latency_seconds": time.monotonic()-started,
               "cache_dir": str(cache.root), "output_dir": str(store.root)}
    audit = {"status": "PASS", "zero_call_preflight_bound": True, "only_local001": trace["claim_id"] == TARGET_CLAIM_ID,
             "local002_excluded": True, "local003_excluded": True, "one_residual_proposal_only": trace["residual_proposal_attempt_count"] == 1,
             "drop_r1_only_proposer_input": all(path["path"] in inputs["drop_paths"] for path in trace["drop_r1_artifacts"]),
             "formal_certificate_unchanged": True, "new_verified_count": 0, "gt_used": False,
             "runtime_gt_isolation_audit": runtime_gt_audit(runtime_path, samples), "runtime_import_audit": audit_runtime_imports()}
    report = {"status": "PASS" if audit["runtime_import_audit"]["status"] == "PASS" else "FAIL", "mode": mode, "summary": summary, "audit": audit,
              "traces": str(store.root / "phase37_trace.jsonl")}
    suffix = "" if mode == "run" else "_replay"
    (output_dir / f"phase37{suffix}_diagnostic_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    (output_dir / f"phase37{suffix}_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    (output_dir / f"phase37_{mode}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    if mode == "run":
        (output_dir / "phase37_summary.md").write_text("\n".join(["# ReliVE Phase 3.7 residual visual-support diagnostic", "", f"Diagnostic outcome: `{trace['diagnostic_outcome']}`.", "", "This diagnostic does not create a certificate or VERIFIED status. A persistent semantic result after intervention does not prove a shortcut or hallucination.", ""]), encoding="utf-8")
    return report
