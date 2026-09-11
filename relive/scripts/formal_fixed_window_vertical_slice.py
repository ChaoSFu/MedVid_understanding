#!/usr/bin/env python3
"""Prepare and run one formal fixed-window spatial-certificate vertical slice.

The frozen calibration manifest supplies a public atomic claim and public frame
window only. Its diagnostic human ROI is deliberately excluded from the formal
runtime, core config, spatial proposal input, and certificate input. It is read
only after the core run to produce an explicitly diagnostic ROI comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from relive.audits import audit_run, audit_runtime_imports
from relive.config import load_config
from relive.data.schemas import FIELD_SOURCES
from relive.interventions import OPAQUE_GRAY_OPERATOR, OPAQUE_GRAY_VERSION
from relive.runner import run
from relive.storage.artifacts import stable_hash


class VerticalSliceError(RuntimeError):
    pass


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerticalSliceError(f"unreadable JSON: {path}") from exc
    if not isinstance(value, dict):
        raise VerticalSliceError(f"JSON object required: {path}")
    return value


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise VerticalSliceError("cannot determine current git HEAD") from exc


def _check_manifest(manifest: dict[str, Any], head: str) -> dict[str, Any]:
    if manifest.get("format") != "relive-calibration-manifest-frozen-v1":
        raise VerticalSliceError("requires relive-calibration-manifest-frozen-v1")
    if manifest.get("calibration_only") is not True or manifest.get("selection_status") != "FROZEN_PRE_INFERENCE":
        raise VerticalSliceError("requires a frozen calibration-only manifest")
    supplied = manifest.get("frozen_manifest_sha256")
    content = dict(manifest); content.pop("frozen_manifest_sha256", None)
    if supplied != _hash(content):
        raise VerticalSliceError("frozen manifest SHA-256 does not bind its content")
    prior = manifest.get("git_commit")
    if not isinstance(prior, str) or len(prior) != 40:
        raise VerticalSliceError("frozen manifest must bind a full git commit")
    relation = subprocess.run(["git", "merge-base", "--is-ancestor", prior, head],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    if relation.returncode != 0:
        raise VerticalSliceError("frozen manifest commit is not an ancestor of execution HEAD")
    control = manifest.get("selected_positive_control")
    if not isinstance(control, dict) or control.get("kind") != "positive_control_candidate":
        raise VerticalSliceError("frozen manifest lacks its selected positive control")
    _check_public_control(control)
    return control


def _check_public_control(control: dict[str, Any]) -> None:
    required = {"candidate_id", "atomic_claim_en", "frame_ids", "frame_paths", "frame_orders", "dataset_name", "source_qa_type"}
    if not required <= set(control):
        raise VerticalSliceError("selected positive lacks public runtime fields")
    ids, paths, orders = control["frame_ids"], control["frame_paths"], control["frame_orders"]
    if not (isinstance(ids, list) and isinstance(paths, list) and isinstance(orders, list)
            and 1 <= len(ids) <= 3 and len(ids) == len(paths) == len(orders)):
        raise VerticalSliceError("formal fixed window requires 1-3 frozen aligned frames")
    if any(not isinstance(x, str) or not x for x in [*ids, *paths]) or any(type(x) is not int for x in orders):
        raise VerticalSliceError("frozen frame identities must be explicit")
    if orders != list(range(orders[0], orders[0] + len(orders))):
        raise VerticalSliceError("formal fixed-window frames must be consecutive")
    if any(not Path(path).expanduser().is_file() for path in paths):
        raise VerticalSliceError("a selected public frame is missing")
    if not isinstance(control["atomic_claim_en"], str) or not control["atomic_claim_en"].strip():
        raise VerticalSliceError("selected positive needs an atomic claim")


def _runtime_row(control: dict[str, Any]) -> dict[str, Any]:
    """Project strictly public fields; never read or propagate diagnostic ROI keys."""
    return {
        "sample_id": f"formal-fixed-window:{control['candidate_id']}",
        "task": "claim_verification",
        "question": control["atomic_claim_en"],
        "frames": [{"frame_id": fid, "path": path, "order": order}
                   for fid, path, order in zip(control["frame_ids"], control["frame_paths"], control["frame_orders"])],
        "target_claim": {"claim_id": f"formal-fixed-window:{control['candidate_id']}:claim",
                         "text": control["atomic_claim_en"],
                         "time_scope": {"frame_ids": list(control["frame_ids"])}},
        "metadata": {"runtime_adapter": "formal_fixed_window_vertical_slice",
                     "source_qa_type": control["source_qa_type"], "dataset_name": control["dataset_name"],
                     "nonofficial_protocol": True},
    }


def _core_config(source_config: Path, frame_count: int) -> dict[str, Any]:
    config = load_config(source_config)
    # Backend config/generation is copied exactly. This slice constrains only
    # candidate traversal and selects the registered core operator.
    config["acquisition"] = {"method": "sliding_windows", "window_size": frame_count,
                             "stride": frame_count, "max_candidates": 1}
    config["claims"]["max_claims"] = 1
    config["claims"]["contrast_fixtures"] = []
    config["spatial"]["intervention"] = {
        "operator": OPAQUE_GRAY_OPERATOR, "operator_version": OPAQUE_GRAY_VERSION,
        "parameters": {"fill_rgb": [127, 127, 127]},
    }
    config["policy"] = {"name": "semantic_spatial", "version": "relive-v1-policy-1", "strict_alternatives": True}
    config["adaptation"] = {"enabled": False, "actions": [], "expand_frames": config["adaptation"]["expand_frames"]}
    config["budget"] = {"max_calls": 5, "max_candidates": 1, "max_spatial_proposals": 1, "max_rounds": 1}
    return config


def _write_preflight(root: Path, manifest_path: Path, source_config: Path, manifest: dict[str, Any], control: dict[str, Any], head: str) -> dict[str, Any]:
    preflight = root / "preflight"
    if root.exists():
        raise VerticalSliceError("output root must not already exist")
    preflight.mkdir(parents=True)
    row = _runtime_row(control)
    payload = (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
    runtime = preflight / "formal_fixed_window.runtime.jsonl"
    runtime.write_bytes(payload)
    runtime_sha = hashlib.sha256(payload).hexdigest()
    sidecar = {"schema_version": "relive-runtime-v1", "source_kind": "public_runtime",
               "runtime_sha256": runtime_sha, "field_sources": FIELD_SOURCES}
    Path(str(runtime) + ".provenance.json").write_text(json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n")
    isolation = {"status": "PASS", "runtime_sha256": runtime_sha,
                 "adapter": "formal_fixed_window_vertical_slice",
                 "classification": "nonofficial_public_claim_verification_vertical_slice",
                 "allowed_inputs": ["frozen_manifest_public_claim_and_frames", "public_runtime_frames", "reviewed_local_hf_config"],
                 "prohibited_inputs_not_opened": ["reference_answer", "assistant_answer", "temporal_gt", "bbox_mask_gt", "struc_info", "RC_info", "evaluation_artifacts"],
                 "human_diagnostic_roi_policy": "excluded from runtime, config, proposal input, and certificate input; post-run comparison only"}
    Path(str(runtime) + ".gt_isolation_audit.json").write_text(json.dumps(isolation, ensure_ascii=False, indent=2) + "\n")
    config = _core_config(source_config, len(control["frame_ids"]))
    config_path = preflight / "formal_fixed_window.config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    plan = {"status": "PASS", "mode": "preflight", "model_calls_made": 0, "cache_mutated": False,
            "git_commit": head, "manifest": str(manifest_path), "manifest_sha256": manifest["frozen_manifest_sha256"],
            "runtime": str(runtime), "runtime_sha256": runtime_sha, "config": str(config_path),
            "config_sha256": stable_hash(config), "cache_directory": str(root / "cache"),
            "run_directory": str(root / "run"), "replay_directory": str(root / "replay"),
            "selected_positive_control": control["candidate_id"], "frame_ids": list(control["frame_ids"]),
            "claim": control["atomic_claim_en"], "candidate_pool": "exactly one fixed public window",
            "spatial_intervention": config["spatial"]["intervention"],
            "planned_max_model_calls": 5,
            "planned_call_derivation": "ORIGINAL semantic + automatic spatial proposal + KEEP + DROP + one matched control; later stages are conditional on earlier formal results",
            "human_roi_not_in_core_inputs": True, "gt_isolation_audit": isolation,
            "runtime_source_audit": audit_runtime_imports()}
    (preflight / "formal_vertical_slice_preflight.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    return plan


def _iou(a: list[float], b: list[float]) -> dict[str, float]:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa, ab = (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
    return {"automatic_area_fraction": aa, "human_diagnostic_area_fraction": ab, "intersection_fraction": inter,
            "iou": inter / (aa + ab - inter) if aa + ab - inter else 0.0,
            "automatic_covered_by_human": inter / aa if aa else 0.0,
            "human_covered_by_automatic": inter / ab if ab else 0.0}


def _report(root: Path, manifest: dict[str, Any], control: dict[str, Any], summary: dict[str, Any], mode: str) -> dict[str, Any]:
    result_files = list((root / mode / "samples").glob("*.json"))
    if len(result_files) != 1:
        raise VerticalSliceError("formal core run did not produce exactly one sample")
    sample = json.loads(result_files[0].read_text(encoding="utf-8"))
    certificates = sample.get("certificates", [])
    certificate = certificates[0] if certificates else None
    spatial = (certificate or {}).get("checks", {}).get("spatial", {})
    proposal = spatial.get("proposal") if isinstance(spatial, dict) else None
    automatic = proposal.get("support_region") if isinstance(proposal, dict) else None
    # This is deliberately post-run diagnostic-only access. The value is never
    # written to the core runtime/config or supplied to the runner.
    human = control.get("diagnostic_only_bbox_normalized_xyxy")
    comparison = _iou(automatic, human) if isinstance(automatic, list) and isinstance(human, list) else None
    payload = {"status": "PASS", "mode": mode, "summary": summary, "sample_id": sample["sample_id"],
               "termination_reason": sample["termination_reason"], "strict": sample["strict"],
               "formal_certificate": certificate, "automatic_spatial_proposal": proposal,
               "automatic_vs_human_diagnostic_roi": comparison,
               "human_roi_use": "post_run_diagnostic_only", "pixel_audit_events": list((root / mode / "events" / "pixel_audits").glob("*.json")),
               "strict_audit": audit_run(root / mode), "manifest_sha256": manifest["frozen_manifest_sha256"],
               "formal_fixed_window_positive_verified": bool(certificate and certificate.get("final_status") == "VERIFIED")}
    safe = {**payload, "pixel_audit_events": [str(x) for x in payload["pixel_audit_events"]]}
    (root / f"formal_vertical_slice_{mode}.json").write_text(json.dumps(safe, ensure_ascii=False, indent=2) + "\n")
    return safe


def main() -> int:
    parser = argparse.ArgumentParser(description="Formal fixed-window ReliVE spatial-certificate vertical slice")
    parser.add_argument("--mode", choices=("preflight", "run", "replay"), required=True)
    parser.add_argument("--manifest", required=True); parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    try:
        root, manifest_path, config_path = Path(args.output_dir).resolve(), Path(args.manifest).resolve(), Path(args.config).resolve()
        head = _git_commit(); manifest = _json(manifest_path); control = _check_manifest(manifest, head)
        if args.mode == "preflight":
            report = _write_preflight(root, manifest_path, config_path, manifest, control, head)
        else:
            preflight = _json(root / "preflight" / "formal_vertical_slice_preflight.json")
            if preflight.get("manifest_sha256") != manifest.get("frozen_manifest_sha256"):
                raise VerticalSliceError("missing or incompatible zero-call preflight")
            mode = args.mode
            output = root / mode
            if output.exists():
                raise VerticalSliceError(f"{mode} output already exists")
            config = load_config(root / "preflight" / "formal_fixed_window.config.json")
            summary = run(config, root / "preflight" / "formal_fixed_window.runtime.jsonl", output,
                          cache_dir=root / "cache", max_samples=1, phase="smoke")
            report = _report(root, manifest, control, summary, mode)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, VerticalSliceError) as exc:
        print(f"ReliVE formal fixed-window vertical slice error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
