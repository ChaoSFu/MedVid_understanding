#!/usr/bin/env python3
"""Calibration-only fixed-ROI semantic intervention runner.

This deliberately lives outside the ReliVE core runner.  It reads one frozen,
human-authored public-frame calibration manifest and exercises the existing
semantic prompt, parser, local-HF backend, cache, and pixel intervention
implementation.  It never calls spatial grounding, claim acquisition, the
certificate builder, or any evaluation/GT loader.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from PIL import Image

from relive.audits import audit_runtime_imports
from relive.backends import make_backend
from relive.claims import PROMPT_VERSIONS, prompt
from relive.config import load_config
from relive.interventions import CONTROL_VERSION, INTERVENTION_VERSION, apply_intervention
from relive.storage.artifacts import ArtifactStore
from relive.storage.cache import Budget, CachedInference
from relive.types import ExecutionStatus, SemanticStatus, to_dict
from relive.verification import execution_failure, parse_verification


class CalibrationError(RuntimeError):
    pass


VARIANTS = ("ORIGINAL", "KEEP_TARGET", "DROP_TARGET", "DROP_MATCHED_CONTROL")
POSITIVES = "positive_control_candidate"
ANCILLARY = {"broad_multiple_support_uncertainty_control", "visually_exclusive_negative_control"}


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CalibrationError(f"unreadable JSON: {path}") from exc
    if not isinstance(value, dict):
        raise CalibrationError(f"JSON object required: {path}")
    return value


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def _valid_box(value: Any, name: str) -> tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) != 4:
        raise CalibrationError(f"{name} must be a normalized xyxy list")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in value):
        raise CalibrationError(f"{name} coordinates must be numbers")
    x1, y1, x2, y2 = (float(x) for x in value)
    if not 0 <= x1 < x2 <= 1 or not 0 <= y1 < y2 <= 1:
        raise CalibrationError(f"{name} must be in bounds with positive area")
    return x1, y1, x2, y2


def _intersection(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def _manifest(path: Path, git_commit: str) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    manifest = _json(path)
    if manifest.get("format") != "relive-calibration-manifest-frozen-v1":
        raise CalibrationError("requires relive-calibration-manifest-frozen-v1")
    if manifest.get("calibration_only") is not True or manifest.get("selection_status") != "FROZEN_PRE_INFERENCE":
        raise CalibrationError("manifest is not frozen calibration-only input")
    manifest_commit = manifest.get("git_commit")
    if not isinstance(manifest_commit, str) or len(manifest_commit) != 40:
        raise CalibrationError("frozen manifest must bind a full git commit")
    if manifest_commit == git_commit:
        commit_relation = "EXACT_EXECUTION_HEAD"
    else:
        # The runner may be added after an operator freezes inputs.  The frozen
        # source commit must still be in this execution history; immutable
        # manifest, frame, claim, and ROI hashes remain independently checked.
        relation = subprocess.run(["git", "merge-base", "--is-ancestor", manifest_commit, git_commit],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        if relation.returncode != 0:
            raise CalibrationError("frozen manifest commit is not an ancestor of current HEAD")
        commit_relation = "FROZEN_COMMIT_ANCESTOR_OF_EXECUTION_HEAD"
    supplied_hash = manifest.get("frozen_manifest_sha256")
    without_hash = dict(manifest)
    without_hash.pop("frozen_manifest_sha256", None)
    if supplied_hash != _canonical_hash(without_hash):
        raise CalibrationError("frozen manifest SHA-256 does not bind its content")
    selected = manifest.get("selected_positive_control")
    ancillary = manifest.get("ancillary_controls")
    if not isinstance(selected, dict) or selected.get("kind") != POSITIVES or not isinstance(ancillary, list):
        raise CalibrationError("manifest must contain one selected positive and ancillary controls")
    controls = [selected, *ancillary]
    if len(controls) != 3 or {row.get("kind") for row in ancillary} != ANCILLARY:
        raise CalibrationError("requires exactly the frozen positive, broad control, and negative control")
    ids = [row.get("candidate_id") for row in controls]
    if any(not isinstance(value, str) or not value for value in ids) or len(set(ids)) != 3:
        raise CalibrationError("calibration control IDs must be unique")
    for row in controls:
        _validate_control(row)
    return manifest, controls, commit_relation


def _validate_control(row: dict[str, Any]) -> None:
    ids, paths, orders = row.get("frame_ids"), row.get("frame_paths"), row.get("frame_orders")
    if not all(isinstance(value, list) and 1 <= len(value) <= 3 for value in (ids, paths, orders)):
        raise CalibrationError("each control must contain 1-3 frozen frames")
    if not (len(ids) == len(paths) == len(orders) and all(isinstance(x, str) and x for x in [*ids, *paths])
            and all(type(x) is int and x >= 0 for x in orders)):
        raise CalibrationError("invalid frozen frame identity")
    if orders != list(range(orders[0], orders[0] + len(orders))):
        raise CalibrationError("frozen calibration frames must be consecutive")
    if any(not Path(path).expanduser().is_file() for path in paths):
        raise CalibrationError("frozen public calibration frame is missing")
    expected_frame_hash = _canonical_hash([{"frame_id": fid, "path": path, "order": order}
                                           for fid, path, order in zip(ids, paths, orders)])
    if row.get("frame_sha256") != expected_frame_hash:
        raise CalibrationError("frozen frame hash mismatch")
    claim = row.get("atomic_claim_en")
    if not isinstance(claim, str) or not claim.strip() or row.get("claim_sha256") != hashlib.sha256(claim.encode()).hexdigest():
        raise CalibrationError("frozen claim hash mismatch")
    target, control = _valid_box(row.get("diagnostic_only_bbox_normalized_xyxy"), "target ROI"), _valid_box(row.get("matched_control_bbox_normalized_xyxy"), "matched control")
    target_area = (target[2] - target[0]) * (target[3] - target[1])
    control_area = (control[2] - control[0]) * (control[3] - control[1])
    if not 0.05 <= target_area <= 0.30 or abs(target_area - control_area) > 1e-12 or _intersection(target, control) != 0:
        raise CalibrationError("frozen ROI/control geometry is invalid")
    expected = row.get("expected_outcomes_frozen_pre_inference")
    if not isinstance(expected, dict) or set(expected) != set(VARIANTS) or any(value not in SemanticStatus._value2member_map_ for value in expected.values()):
        raise CalibrationError("frozen expected semantic outcomes are invalid")


def _runtime_rows(controls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in controls:
        rows.append({
            "sample_id": f"calibration:{row['candidate_id']}", "task": "claim_verification",
            "question": row["atomic_claim_en"],
            "frames": [{"frame_id": fid, "path": path, "order": order}
                       for fid, path, order in zip(row["frame_ids"], row["frame_paths"], row["frame_orders"])],
            "target_claim": {"claim_id": f"calibration:{row['candidate_id']}:claim", "text": row["atomic_claim_en"],
                             "time_scope": {"frame_ids": row["frame_ids"]}},
            "metadata": {"runtime_adapter": "fixed_roi_calibration_only", "source_qa_type": row["source_qa_type"],
                         "dataset_name": row["dataset_name"], "nonofficial_protocol": True},
        })
    return rows


def _write_preflight(manifest_path: Path, manifest: dict[str, Any], controls: list[dict[str, Any]],
                     config: dict[str, Any], output: Path, cache_dir: Path, git_commit: str,
                     commit_relation: str) -> dict[str, Any]:
    if config["backend"]["kind"] != "local_hf":
        raise CalibrationError("fixed-ROI calibration requires the reviewed local_hf backend")
    output.mkdir(parents=True, exist_ok=False)
    runtime = _runtime_rows(controls)
    runtime_path = output / "calibration.runtime.jsonl"
    payload = b"".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode() + b"\n" for row in runtime)
    runtime_path.write_bytes(payload)
    runtime_sha = hashlib.sha256(payload).hexdigest()
    # The sidecar must remain the existing closed runtime schema.  Calibration
    # context is recorded in the separate preflight report, not here.
    sidecar = {"schema_version": "relive-runtime-v1", "source_kind": "public_runtime", "runtime_sha256": runtime_sha,
               "field_sources": {"question": "public_question", "target_claim": "user_query", "required_claims": "user_query",
                                 "frames": "public_frames", "metadata": "public_metadata"}}
    Path(str(runtime_path) + ".provenance.json").write_text(json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n")
    isolation = {"status": "PASS", "calibration_only": True, "runtime_sha256": runtime_sha,
                 "manifest_path": str(manifest_path), "manifest_sha256": manifest["frozen_manifest_sha256"],
                 "allowed_inputs": ["frozen_human_authored_public_frame_calibration_manifest", "public_frame_images", "existing_local_hf_configuration"],
                 "prohibited_inputs_not_opened": ["reference_answer", "assistant_answer", "temporal_gt", "bbox_mask_gt", "struc_info", "RC_info", "evaluation_artifacts"],
                 "time_policy": "frame order only; no timestamps, FPS, or temporal GT", "classification": "calibration_only_not_official_medvidu_qa"}
    audit_path = output / "calibration.gt_isolation_audit.json"
    audit_path.write_text(json.dumps(isolation, ensure_ascii=False, indent=2) + "\n")
    source_audit = audit_runtime_imports()
    plan = {"status": "PASS", "mode": "preflight", "model_calls_made": 0, "cache_mutated": False,
            "git_commit": git_commit, "frozen_manifest_git_commit": manifest["git_commit"],
            "commit_relation": commit_relation, "manifest_sha256": manifest["frozen_manifest_sha256"],
            "runtime": str(runtime_path), "runtime_sha256": runtime_sha, "provenance": str(Path(str(runtime_path) + ".provenance.json")),
            "gt_isolation_audit": str(audit_path), "output_directory": str(output), "cache_directory": str(cache_dir),
            "backend": {key: config["backend"].get(key) for key in ("kind", "model", "revision", "model_path", "dtype", "device", "device_map", "input_device", "generation", "frame_encoding")},
            "fixed_roi_support": "ISOLATED_CALIBRATION_ONLY_ADAPTER", "formal_certificate_created": False,
            "planned_unique_semantic_calls": len(controls) * len(VARIANTS),
            "planned_call_derivation": "3 frozen controls x ORIGINAL, KEEP_TARGET, DROP_TARGET, DROP_MATCHED_CONTROL",
            "controls": [{"candidate_id": row["candidate_id"], "kind": row["kind"], "frame_ids": row["frame_ids"],
                          "target_roi": row["diagnostic_only_bbox_normalized_xyxy"], "matched_control_roi": row["matched_control_bbox_normalized_xyxy"],
                          "expected_outcomes": row["expected_outcomes_frozen_pre_inference"]} for row in controls],
            "runtime_source_audit": source_audit}
    (output / "calibration_preflight.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    return plan


def _image_file(root: Path, image: Image.Image) -> str:
    from io import BytesIO
    data = BytesIO(); image.save(data, format="PNG"); raw = data.getvalue()
    path = root / "images" / f"{hashlib.sha256(raw).hexdigest()}.png"; path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists(): path.write_bytes(raw)
    elif path.read_bytes() != raw: raise CalibrationError("image content identity collision")
    return str(path)


def _semantic(inference: CachedInference, budget: Budget, control: dict[str, Any], variant: str,
              image_paths: list[str], region: tuple[float, float, float, float] | None) -> dict[str, Any]:
    ids = control["frame_ids"]
    request = {"stage": "semantic", "prompt": prompt("semantic", {"frame_count": len(ids),
                                                             "claim": {"text": control["atomic_claim_en"]}}),
               "prompt_version": PROMPT_VERSIONS["semantic"], "image_paths": image_paths, "frame_ids": ids,
               "context": {"calibration_only": True, "calibration_control_id": control["candidate_id"],
                           "claim_text": control["atomic_claim_en"], "variant": variant,
                           "fixed_roi": list(region) if region else None,
                           "intervention_version": INTERVENTION_VERSION, "control_version": CONTROL_VERSION}}
    raw = inference.call(request, budget)
    refs = {"sample_id": f"calibration:{control['candidate_id']}", "candidate_id": control["candidate_id"],
            "claim_id": f"calibration:{control['candidate_id']}:claim", "frame_ids": ids, "image_paths": image_paths,
            "variant": variant, "region": list(region) if region else None, "control_index": 0 if variant == "DROP_MATCHED_CONTROL" else None,
            "cache_key": raw["cache_key"]}
    verdict = (parse_verification(raw["raw_text"], raw_response_ref=raw["raw_response_ref"], prompt_version=PROMPT_VERSIONS["semantic"], input_references=refs)
               if raw["execution_status"] == ExecutionStatus.OK.value else execution_failure(ExecutionStatus(raw["execution_status"]), raw.get("failure_reason") or "INFERENCE_FAILURE", raw["raw_response_ref"], PROMPT_VERSIONS["semantic"], refs))
    return {"result": to_dict(verdict), "cache_hit": raw["cache_hit"], "latency_seconds": raw["latency_seconds"], "attempts": raw["attempts"]}


def _run(manifest: dict[str, Any], controls: list[dict[str, Any]], config: dict[str, Any], output: Path, cache_dir: Path,
         git_commit: str, preflight: dict[str, Any], commit_relation: str) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    store, cache = ArtifactStore(output), ArtifactStore(cache_dir)
    backend = make_backend(config["backend"])
    if backend.synthetic: raise CalibrationError("real fixed-ROI calibration requires the local-HF backend")
    if config["backend"]["kind"] != "local_hf": raise CalibrationError("fixed-ROI calibration requires local_hf")
    budget, inference = Budget(preflight["planned_unique_semantic_calls"]), CachedInference(backend, cache)
    all_rows = []
    with store.run_lock():
        for control in controls:
            target = tuple(control["diagnostic_only_bbox_normalized_xyxy"])
            matched = tuple(control["matched_control_bbox_normalized_xyxy"])
            variants = {"ORIGINAL": (list(control["frame_paths"]), None, []), "KEEP_TARGET": ([], target, []),
                        "DROP_TARGET": ([], target, []), "DROP_MATCHED_CONTROL": ([], matched, [])}
            pixel_audits = []
            for variant, (paths, region, _) in list(variants.items()):
                if variant == "ORIGINAL":
                    for frame_id, source_path in zip(control["frame_ids"], control["frame_paths"]):
                        with Image.open(source_path) as image: _, audit = apply_intervention(image.convert("RGB"), target, "ORIGINAL", config["spatial"]["blur_radius"])
                        pixel_audits.append({"frame_id": frame_id, "variant": variant, "output_path": None, **audit})
                else:
                    altered_paths = []
                    for frame_id, source_path in zip(control["frame_ids"], control["frame_paths"]):
                        with Image.open(source_path) as image: altered, audit = apply_intervention(image.convert("RGB"), region, variant, config["spatial"]["blur_radius"])
                        path = _image_file(cache.root, altered); altered_paths.append(path)
                        pixel_audits.append({"frame_id": frame_id, "variant": variant, "output_path": path, **audit})
                    variants[variant] = (altered_paths, region, [])
            for audit in pixel_audits: store.append_event("pixel_audits", {"calibration_only": True, "candidate_id": control["candidate_id"], **audit})
            outputs = {}
            for variant in VARIANTS:
                paths, region, _ = variants[variant]
                result = _semantic(inference, budget, control, variant, paths, region)
                outputs[variant] = result
                store.append_event("verification", {"calibration_only": True, "candidate_id": control["candidate_id"], "variant": variant, **result})
            statuses = {variant: outputs[variant]["result"]["semantic_status"] for variant in VARIANTS}
            expected = control["expected_outcomes_frozen_pre_inference"]
            pixel_ok = all(audit["pixel_audit_pass"] for audit in pixel_audits)
            matches = {variant: statuses[variant] == expected[variant] for variant in VARIANTS}
            kind = control["kind"]
            if kind == POSITIVES:
                verdict = "CALIBRATION_POSITIVE_PASS" if pixel_ok and all(matches.values()) else "CALIBRATION_POSITIVE_FAIL"
            elif kind == "broad_multiple_support_uncertainty_control":
                verdict = "CALIBRATION_UNCERTAINTY_CONTROL_PASS" if pixel_ok and all(matches.values()) else "CALIBRATION_UNCERTAINTY_CONTROL_FAIL"
            else:
                verdict = "CALIBRATION_NEGATIVE_CONTROL_PASS" if pixel_ok and all(matches.values()) else "CALIBRATION_NEGATIVE_CONTROL_FAIL"
            all_rows.append({"candidate_id": control["candidate_id"], "kind": kind, "claim": control["atomic_claim_en"],
                             "frame_ids": control["frame_ids"], "expected": expected, "observed": statuses, "expected_match": matches,
                             "pixel_audits": pixel_audits, "pixel_audit_pass": pixel_ok, "variant_results": outputs,
                             "calibration_verdict": verdict, "formal_certificate": None,
                             "not_benchmark_evidence": True})
        report = {"status": "PASS", "calibration_only": True, "formal_certificate_created": False,
                  "git_commit": git_commit, "frozen_manifest_git_commit": manifest["git_commit"],
                  "commit_relation": commit_relation, "manifest_sha256": manifest["frozen_manifest_sha256"], "preflight": preflight,
                  "backend": backend.fingerprint(), "usage": budget.snapshot(), "controls": all_rows,
                  "termination_reason": "FIXED_FROZEN_CONTROLS_COMPLETE", "model_parameters_updated": False,
                  "prohibited_inputs_not_opened": manifest["prohibited_inputs_not_opened"]}
        store.put_json("results", "calibration", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="ReliVE calibration-only frozen fixed-ROI runner")
    parser.add_argument("--mode", choices=("preflight", "run"), required=True)
    parser.add_argument("--manifest", required=True); parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True); parser.add_argument("--cache-dir", required=True)
    args = parser.parse_args()
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        manifest_path, output, cache_dir = Path(args.manifest).resolve(), Path(args.output_dir).resolve(), Path(args.cache_dir).resolve()
        manifest, controls, commit_relation = _manifest(manifest_path, git_commit)
        config = load_config(args.config)
        if args.mode == "preflight": report = _write_preflight(manifest_path, manifest, controls, config, output, cache_dir, git_commit, commit_relation)
        else:
            preflight_path = output.parent / "preflight" / "calibration_preflight.json"
            preflight = _json(preflight_path)
            if preflight.get("manifest_sha256") != manifest.get("frozen_manifest_sha256") or preflight.get("planned_unique_semantic_calls") != 12:
                raise CalibrationError("missing or incompatible zero-call preflight")
            report = _run(manifest, controls, config, output, cache_dir, git_commit, preflight, commit_relation)
        print(json.dumps(report if args.mode == "preflight" else {"status": report["status"], "usage": report["usage"], "termination_reason": report["termination_reason"], "output_directory": str(output)}, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, CalibrationError) as exc:
        print(f"ReliVE fixed-ROI calibration error: {exc}", file=sys.stderr); return 2


if __name__ == "__main__":
    raise SystemExit(main())
