"""Frozen, diagnostic-only visual-dependence likelihood audit.

Phase 4A has no certificate integration.  It uses a deliberately separate
next-token A/B/C measurement to expose score changes that the existing semantic
JSON verifier cannot represent, while preserving all earlier artifacts.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any, Callable, Sequence
from importlib.resources import files

from PIL import Image

from .audits import audit_runtime_imports
from .backends import make_backend
from .claims import PROMPT_VERSIONS
from .config import load_config
from .interventions import OPAQUE_GRAY_OPERATOR, OPAQUE_GRAY_VERSION, apply_spatial_intervention
from .phase35_v3 import candidate_for, runtime_gt_audit, sha256_path, validate_config, validate_prospective_manifest
from .phase4a0 import validate_candidate_specs
from .runner import _image_file
from .spatial import validate_region
from .storage.artifacts import ArtifactStore, canonical_json, stable_hash

PHASE4A_FORMAT = "relive-phase4a-frozen-visual-dependence-audit-v1"
CHOICES = ("A", "B", "C")
VARIANTS = ("ORIGINAL", "KEEP_TARGET", "DROP_TARGET", "DROP_MATCHED_CONTROL", "FULL_GRAY", "MISMATCHED_PUBLIC")
PROHIBITED_INPUTS = ["reference_answer", "assistant_answer", "temporal_gt", "bbox_mask_gt", "struc_info", "RC_info", "evaluation_artifacts"]


class Phase4AError(ValueError):
    pass


def _json(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Phase4AError(f"unreadable JSON: {path}") from exc
    if not isinstance(result, dict):
        raise Phase4AError(f"JSON object required: {path}")
    return result


def _jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Phase4AError(f"unreadable JSONL: {path}") from exc
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise Phase4AError(f"nonempty JSONL object rows required: {path}")
    return rows


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _file_rows(ids: Sequence[str], paths: Sequence[str]) -> list[dict[str, Any]]:
    if not isinstance(ids, (list, tuple)) or not isinstance(paths, (list, tuple)) or not ids or len(ids) != len(paths):
        raise Phase4AError("frame IDs and paths must be nonempty aligned lists")
    rows = []
    for frame_id, source_path in zip(ids, paths):
        if not isinstance(frame_id, str) or not frame_id or not isinstance(source_path, str) or not source_path:
            raise Phase4AError("frame IDs and paths must be nonempty strings")
        path = Path(source_path)
        if not path.is_file():
            raise Phase4AError(f"public frame does not exist: {path}")
        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, ValueError):
            raise Phase4AError(f"public frame cannot be decoded: {path}") from None
        rows.append({"frame_id": frame_id, "path": str(path), "sha256": sha256_path(path)})
    if len({row["frame_id"] for row in rows}) != len(rows):
        raise Phase4AError("frame IDs must be unique")
    return rows


def choice_prompt(claim: str) -> str:
    if not isinstance(claim, str) or not claim.strip():
        raise Phase4AError("claim must be nonempty")
    text = files("relive").joinpath("prompts", "visual_dependence_choice.txt").read_text(encoding="utf-8").format(claim=claim).rstrip()
    if not text.endswith("\nAnswer:"):
        raise Phase4AError("choice prompt is missing its fixed answer anchor")
    return text


def support_margin(log_probabilities: dict[str, float]) -> float:
    if set(log_probabilities) != set(CHOICES) or any(type(value) not in {int, float} or not math.isfinite(value) for value in log_probabilities.values()):
        raise Phase4AError("choice log probabilities must provide finite A/B/C values")
    b, c = float(log_probabilities["B"]), float(log_probabilities["C"])
    return float(log_probabilities["A"]) - (max(b, c) + math.log(math.exp(b - max(b, c)) + math.exp(c - max(b, c))))


def _strict_frozen_hash(manifest: dict[str, Any], field: str) -> None:
    supplied = manifest.get(field)
    content = {key: value for key, value in manifest.items() if key != field}
    if supplied != _sha(content):
        raise Phase4AError(f"{field} does not bind manifest content")


def _mismatches(path: Path) -> dict[str, dict[str, Any]]:
    manifest = _json(path)
    if manifest.get("format") != "relive-phase4a-mismatched-public-manifest-v1" or manifest.get("selection_status") != "FROZEN_PRE_INFERENCE":
        raise Phase4AError("requires a frozen Phase 4A mismatched-public manifest")
    _strict_frozen_hash(manifest, "mismatched_manifest_sha256")
    items = manifest.get("items")
    if not isinstance(items, list) or not items:
        raise Phase4AError("mismatched-public manifest must contain nonempty items")
    result = {}
    for item in items:
        if not isinstance(item, dict) or set(item) != {"source_id", "frame_ids", "frame_paths", "selector", "frame_manifest_sha256"}:
            raise Phase4AError("mismatched-public items use an unsupported schema")
        source_id, selector = item["source_id"], item["selector"]
        if not isinstance(source_id, str) or not source_id or source_id in result or not isinstance(selector, str) or not selector:
            raise Phase4AError("mismatched-public source IDs/selectors must be unique nonempty strings")
        rows = _file_rows(item["frame_ids"], item["frame_paths"])
        identity = [{"frame_id": row["frame_id"], "path": row["path"]} for row in rows]
        if item["frame_manifest_sha256"] != _sha(identity):
            raise Phase4AError("mismatched-public frame manifest hash mismatch")
        result[source_id] = {"selector": selector, "frames": rows}
    return result


def _certificate_regions(certificate: dict[str, Any]) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    try:
        spatial = certificate["checks"]["spatial"]
        proposal = spatial["proposal"]["support_region"]
        controls = spatial["controls"]["regions"]
        matched = controls[0]
    except (KeyError, TypeError, IndexError):
        raise Phase4AError("frozen formal spatial artifact lacks proposal/control geometry") from None
    return validate_region(proposal), validate_region(matched)


def _phase35_sources(runtime_path: Path, prospective_path: Path, phase35_run_dir: Path, phase36_run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prospective, samples = validate_prospective_manifest(prospective_path, runtime_path)
    if len(samples) != 3:
        raise Phase4AError("Phase 4A requires the frozen Phase 3.5 three-local-case runtime")
    v3_rows = {row.get("claim_id"): row for row in _jsonl(phase35_run_dir / "phase35_v3_trace.jsonl")}
    p36_rows = {row.get("claim_id"): row for row in _jsonl(phase36_run_dir / "phase36_regrounding_trace.jsonl")}
    entries, unavailable = [], []
    for sample in samples:
        claim_id = sample.target_claim.claim_id
        candidate = candidate_for(sample)
        source_id = f"phase35:{claim_id}"
        v3 = v3_rows.get(claim_id)
        if not isinstance(v3, dict) or v3.get("sample_id") != sample.sample_id or v3.get("candidate_id") != candidate.candidate_id:
            raise Phase4AError("Phase 3.5 candidate lineage mismatch")
        certificate = None
        for artifact in (phase35_run_dir / "samples").glob("*.json"):
            row = _json(artifact)
            if row.get("sample_id") == sample.sample_id:
                certs = row.get("certificates", [])
                if len(certs) == 1:
                    certificate = certs[0]
                break
        if claim_id == "phase35-local-001":
            p36 = p36_rows.get(claim_id)
            if not isinstance(p36, dict) or p36.get("refinement_outcome") != "PROPOSED" or not isinstance(p36.get("formal_certificate"), dict):
                raise Phase4AError("Phase 3.6 R1 artifact is unavailable for phase35-local-001")
            certificate = p36["formal_certificate"]
            r1, matched = _certificate_regions(certificate)
            roi_source = "phase36_frozen_R1"
        elif claim_id == "phase35-local-002" and isinstance(certificate, dict):
            r1, matched = _certificate_regions(certificate)
            roi_source = "phase35_v3_frozen_automatic_R0"
        else:
            unavailable.append({"source_id": source_id, "claim_id": claim_id, "sample_id": sample.sample_id,
                                "reason": "FROZEN_ROI_UNAVAILABLE_FROM_EXISTING_FORMAL_PATH"})
            continue
        entries.append({"source_id": source_id, "source_kind": "phase35_fresh_local_atomic", "sample_id": sample.sample_id,
                        "claim_id": claim_id, "claim": sample.target_claim.text,
                        "claim_sha256": hashlib.sha256(sample.target_claim.text.encode()).hexdigest(),
                        "frames": _file_rows(list(candidate.frame_ids), [frame.path for frame in sample.frames]),
                        "target_roi": list(r1), "matched_control_roi": list(matched), "roi_source": roi_source,
                        "generated_semantic_verdict_observation": v3.get("semantic_variants", {}).get("original"),
                        "calibration_expectation_present_posthoc_only": False})
    return entries, {"prospective_manifest_sha256": prospective["manifest_sha256"], "unavailable": unavailable,
                     "phase35_trace_sha256": sha256_path(phase35_run_dir / "phase35_v3_trace.jsonl"),
                     "phase36_trace_sha256": sha256_path(phase36_run_dir / "phase36_regrounding_trace.jsonl")}


def _calibration_sources(path: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if path is None:
        return [], {"status": "MISSING_CALIBRATION_MANIFEST", "reason": "no frozen calibration manifest supplied; no controls were selected"}
    manifest = _json(path)
    if manifest.get("format") != "relive-calibration-manifest-frozen-v1" or manifest.get("calibration_only") is not True or manifest.get("selection_status") != "FROZEN_PRE_INFERENCE":
        raise Phase4AError("calibration manifest is not a frozen calibration-only manifest")
    _strict_frozen_hash(manifest, "frozen_manifest_sha256")
    controls = [manifest.get("selected_positive_control"), *(manifest.get("ancillary_controls") or [])]
    if len(controls) != 3 or not all(isinstance(row, dict) for row in controls):
        raise Phase4AError("frozen calibration manifest must retain exactly three controls")
    sources = []
    for control in controls:
        for forbidden in PROHIBITED_INPUTS:
            if forbidden in control:
                raise Phase4AError("calibration control contains prohibited input field")
        candidate_id, claim = control.get("candidate_id"), control.get("atomic_claim_en")
        if not isinstance(candidate_id, str) or not isinstance(claim, str) or not claim or control.get("claim_sha256") != hashlib.sha256(claim.encode()).hexdigest():
            raise Phase4AError("invalid frozen calibration claim binding")
        frames = _file_rows(control.get("frame_ids"), control.get("frame_paths"))
        if len(frames) not in {1, 2, 3} or control.get("frame_orders") != list(range(control["frame_orders"][0], control["frame_orders"][0] + len(frames))):
            raise Phase4AError("invalid frozen calibration frame ordering")
        target, matched = validate_region(control.get("diagnostic_only_bbox_normalized_xyxy")), validate_region(control.get("matched_control_bbox_normalized_xyxy"))
        sources.append({"source_id": f"calibration:{candidate_id}", "source_kind": "frozen_calibration_only", "sample_id": control.get("sample_id"),
                        "claim_id": f"calibration:{candidate_id}:claim", "claim": claim, "claim_sha256": control["claim_sha256"],
                        "frames": frames, "target_roi": list(target), "matched_control_roi": list(matched), "roi_source": "frozen_diagnostic_only_calibration_roi",
                        "generated_semantic_verdict_observation": None, "calibration_expectation_present_posthoc_only": isinstance(control.get("expected_outcomes_frozen_pre_inference"), dict)})
    return sources, {"status": "INCLUDED", "manifest": str(path), "manifest_sha256": manifest["frozen_manifest_sha256"], "control_count": len(sources)}


def development_control_sources(path: Path | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Adapt only human-eligible Phase 4A-0 controls into diagnostic inputs."""
    if path is None:
        return [], {"status": "NOT_INCLUDED"}
    specs = validate_candidate_specs(path)
    if not all(row.get("development_control") is True and row.get("historical_pilot") is False for row in specs):
        raise Phase4AError("development manifest contains non-development control")
    entries = []
    for spec in specs:
        frames = _file_rows([f"development:{spec['audit_case_id']}:frame:{order:06d}" for order in spec["frame_orders"]], spec["frame_paths"])
        entries.append({"source_id": f"development:{spec['audit_case_id']}", "source_kind": "phase4a0_human_eligible_development_control",
                        "sample_id": None, "claim_id": spec["source_claim_id"], "claim": spec["claim_text"],
                        "claim_sha256": spec["claim_sha256"], "frames": frames,
                        "target_roi": spec["frozen_support_region"], "matched_control_roi": spec["matched_control_regions"][0],
                        "source_record_index": spec["source_record_index"],
                        "roi_source": "phase4a0_frozen_human_target_and_matched_control",
                        "generated_semantic_verdict_observation": None, "calibration_expectation_present_posthoc_only": False})
    return entries, {"status": "INCLUDED", "manifest": str(path), "manifest_sha256": sha256_path(path), "control_count": len(entries),
                     "human_eligibility_required": True}


def freeze_development_mismatched_public(*, development_manifest_path: Path, output_path: Path) -> dict[str, Any]:
    """Pre-register a deterministic non-self public-frame permutation, zero model."""
    if output_path.exists():
        raise Phase4AError("mismatched-public output already exists")
    entries, _ = development_control_sources(development_manifest_path)
    if len(entries) < 2:
        raise Phase4AError("development mismatched-public freeze requires at least two controls")
    # Source-record peers are preferred; ties use source ID only. No model or
    # outcome value participates in this selector.
    items = []
    for entry in entries:
        candidates = [other for other in entries if other["source_id"] != entry["source_id"]]
        other = min(candidates, key=lambda value: (value["source_record_index"] != entry["source_record_index"], value["source_id"]))
        # The audit manifest does not need a second source's claim; only its frozen public image.
        identity = [{"frame_id": row["frame_id"], "path": row["path"]} for row in other["frames"]]
        items.append({"source_id": entry["source_id"], "frame_ids": [row["frame_id"] for row in other["frames"]],
                      "frame_paths": [row["path"] for row in other["frames"]],
                      "selector": "phase4a0-frozen-deterministic-nonself-public-control-v1",
                      "frame_manifest_sha256": _sha(identity)})
    manifest = {"format": "relive-phase4a-mismatched-public-manifest-v1", "selection_status": "FROZEN_PRE_INFERENCE", "items": items,
                "development_manifest": str(development_manifest_path), "development_manifest_sha256": sha256_path(development_manifest_path),
                "selector_policy": "deterministic non-self public control; sorted source ID fallback; no model or GT"}
    manifest["mismatched_manifest_sha256"] = _sha(manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"status": "PASS", "mode": "freeze_development_mismatched_public", "model_calls_made": 0, "gt_used": False,
            "development_manifest": str(development_manifest_path), "mismatched_manifest": str(output_path),
            "mismatched_manifest_sha256": manifest["mismatched_manifest_sha256"], "item_count": len(items)}


def _variant_images(entry: dict[str, Any], root: Path, operator: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    target, matched = entry["target_roi"], entry["matched_control_roi"]
    source_paths, ids = [row["path"] for row in entry["frames"]], [row["frame_id"] for row in entry["frames"]]
    store = ArtifactStore(root)
    variants: dict[str, dict[str, Any]] = {"ORIGINAL": {"frame_ids": ids, "image_paths": source_paths, "selector": "frozen_original_public_frames", "pixel_audits": []}}
    audits = []
    declarations = {"KEEP_TARGET": (target, "KEEP_TARGET"), "DROP_TARGET": (target, "DROP_TARGET"),
                    "DROP_MATCHED_CONTROL": (matched, "DROP_MATCHED_CONTROL"), "FULL_GRAY": ([0.0, 0.0, 1.0, 1.0], "DROP_TARGET")}
    for variant, (region, operation) in declarations.items():
        paths, variant_audits = [], []
        for frame_id, source_path in zip(ids, source_paths):
            with Image.open(source_path) as image:
                altered, audit = apply_spatial_intervention(image.convert("RGB"), region, operation, operator)
            output_path = _image_file(store.root, altered)
            item = {"frame_id": frame_id, "variant": variant, "operation": operation, "region": list(region), "output_path": output_path, **audit}
            paths.append(output_path); variant_audits.append(item); audits.append(item)
        if not all(item["pixel_audit_pass"] for item in variant_audits):
            raise Phase4AError(f"pixel audit failed while freezing {variant}")
        variants[variant] = {"frame_ids": ids, "image_paths": paths, "selector": f"registered_opaque_gray:{operation}", "pixel_audits": variant_audits}
    return variants, audits


def _request(entry: dict[str, Any], variant: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {"stage": "visual_dependence_choice", "prompt": choice_prompt(entry["claim"]), "prompt_version": PROMPT_VERSIONS["visual_dependence_choice"],
            "image_paths": list(payload["image_paths"]), "frame_ids": list(payload["frame_ids"]),
            "context": {"phase": "4A", "diagnostic_only": True, "source_id": entry["source_id"], "claim_id": entry["claim_id"],
                        "claim_sha256": entry["claim_sha256"], "variant": variant,
                        "intervention_protocol": "bound_in_phase4a_artifact_only"}}


class ChoiceCache:
    """Phase-4A-only content-addressed next-token result cache."""
    def __init__(self, backend: Any, root: Path):
        self.backend, self.store = backend, ArtifactStore(root)
        self.new_calls = self.cache_hits = self.logical_calls = 0

    def _identity(self, request: dict[str, Any], token_ids: dict[str, int]) -> dict[str, Any]:
        frames = _file_rows(request["frame_ids"], request["image_paths"])
        return {"cache_version": "relive-phase4a-choice-cache-v1", "model_fingerprint": self.backend.fingerprint(),
                "prompt_version": request["prompt_version"], "prompt_sha256": hashlib.sha256(request["prompt"].encode()).hexdigest(),
                "choice_token_ids": token_ids, "variant": request["context"]["variant"], "claim_sha256": request["context"]["claim_sha256"],
                "frame_inputs": frames, "request_context": request["context"]}

    def call(self, request: dict[str, Any], token_contract: dict[str, Any]) -> dict[str, Any]:
        token_ids = token_contract["choice_token_ids"]
        identity, key = self._identity(request, token_ids), None
        key = stable_hash(identity); self.logical_calls += 1
        with self.store.run_lock():
            path = self.store.path("phase4a_choice_cache", key)
            if path.exists():
                value = self.store.get_json(str(path))
                if value.get("identity") != identity:
                    raise Phase4AError("Phase 4A cache identity collision")
                self.cache_hits += 1
                return {**value["result"], "cache_key": key, "cache_hit": True}
            scorer = getattr(self.backend, "forced_choice_likelihood", None)
            if not callable(scorer):
                raise Phase4AError("backend does not implement isolated forced-choice likelihood")
            result = scorer(request, choice_token_ids=token_ids)
            if result.get("choice_contract", {}).get("choice_token_ids") != token_ids:
                raise Phase4AError("choice token contract drift during scoring")
            self.store.put_json("phase4a_choice_cache", key, {"identity": identity, "result": result})
            self.new_calls += 1
            return {**result, "cache_key": key, "cache_hit": False}


def _result(entry: dict[str, Any], variant: str, request: dict[str, Any], scored: dict[str, Any]) -> dict[str, Any]:
    choices = scored.get("choices")
    if not isinstance(choices, dict) or set(choices) != set(CHOICES):
        raise Phase4AError("choice scorer returned malformed A/B/C result")
    logs = {label: choices[label].get("log_probability") for label in CHOICES}
    margin = support_margin(logs)
    return {"source_id": entry["source_id"], "claim_id": entry["claim_id"], "variant": variant,
            "prompt_version": request["prompt_version"], "prompt_sha256": hashlib.sha256(request["prompt"].encode()).hexdigest(),
            "frame_ids": list(request["frame_ids"]), "image_paths": list(request["image_paths"]),
            "input_images": _file_rows(request["frame_ids"], request["image_paths"]), "choice_token_ids": scored["choice_contract"]["choice_token_ids"],
            "context_input_ids_sha256": scored["choice_contract"]["context_input_ids_sha256"], "raw_logits": {key: choices[key]["raw_logit"] for key in CHOICES},
            "log_probabilities": logs, "support_margin": margin, "cache_key": scored["cache_key"], "cache_hit": scored["cache_hit"]}


def _candidate_explanation(original: float, drop: float, control: float, full_gray: float, mismatch: float, generated: str | None) -> str:
    # Equality/ordering are frozen rule definitions, not fitted magnitude thresholds.
    if original == drop == full_gray == mismatch:
        return "VISUAL_INPUT_INSENSITIVITY"
    if generated == "SUPPORTED" and original != drop:
        return "MARGIN_SENSITIVE_LABEL_INVARIANT"
    if control <= drop:
        return "INTERVENTION_NONSPECIFIC"
    if full_gray > drop:
        return "AUTOMATIC_LOCALIZATION_FAILURE"
    return "UNRESOLVED"


def _numeric_snapshot(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fields required to be byte-identical on cache replay, excluding counters."""
    snapshots = []
    for row in rows:
        variants = {}
        for variant, value in row["variants"].items():
            variants[variant] = {key: value[key] for key in ("choice_token_ids", "context_input_ids_sha256", "raw_logits", "log_probabilities", "support_margin")}
        snapshots.append({"source_id": row["source_id"], "variants": variants, "deltas": row["deltas"],
                          "candidate_explanation_pending_human_confirmation": row["candidate_explanation_pending_human_confirmation"]})
    return snapshots


def preflight(*, config_path: Path, runtime_path: Path | None = None, prospective_manifest_path: Path | None = None, phase35_v3_run_dir: Path | None = None,
              phase36_run_dir: Path | None = None, mismatched_manifest_path: Path | None = None, output_dir: Path | None = None, calibration_manifest_path: Path | None = None,
              development_manifest_path: Path | None = None,
              require_real: bool = True, backend_factory: Callable[[dict[str, Any]], Any] = make_backend) -> dict[str, Any]:
    if output_dir is None or mismatched_manifest_path is None:
        raise Phase4AError("Phase 4A preflight requires output and mismatched manifests")
    if output_dir.exists():
        raise Phase4AError("Phase 4A output directory must not exist before preflight")
    config = load_config(config_path); validate_config(config, require_real=require_real)
    if config["spatial"]["intervention"].get("operator") != OPAQUE_GRAY_OPERATOR or config["spatial"]["intervention"].get("operator_version") != OPAQUE_GRAY_VERSION:
        raise Phase4AError("Phase 4A requires the frozen registered opaque-gray operator")
    phase_args = (runtime_path, prospective_manifest_path, phase35_v3_run_dir, phase36_run_dir)
    if any(value is not None for value in phase_args) and not all(value is not None for value in phase_args):
        raise Phase4AError("Phase 3.5 inputs must be supplied together")
    phase_entries, lineage = (_phase35_sources(runtime_path, prospective_manifest_path, phase35_v3_run_dir, phase36_run_dir)
                              if all(value is not None for value in phase_args) else ([], {"status": "NOT_INCLUDED", "unavailable": []}))
    calibration_entries, calibration = _calibration_sources(calibration_manifest_path)
    development_entries, development = development_control_sources(development_manifest_path)
    entries = [*phase_entries, *calibration_entries, *development_entries]
    if not entries:
        raise Phase4AError("no frozen claim/ROI sources are available for Phase 4A")
    mismatch = _mismatches(mismatched_manifest_path)
    missing = [entry["source_id"] for entry in entries if entry["source_id"] not in mismatch]
    if missing:
        raise Phase4AError(f"frozen mismatched-public selector is missing sources: {missing}")
    output_dir.mkdir(parents=True)
    frozen_root = output_dir / "frozen_inputs"
    backend = backend_factory(config["backend"])
    if require_real and getattr(backend, "synthetic", True):
        raise Phase4AError("Phase 4A requires the reviewed real local_hf backend")
    all_entries, contracts, audit_rows = [], {}, []
    for entry in entries:
        mismatch_item = mismatch[entry["source_id"]]
        if len(mismatch_item["frames"]) != len(entry["frames"]):
            raise Phase4AError("mismatched public evidence must have exactly the frozen target frame count")
        if {row["path"] for row in mismatch_item["frames"]} & {row["path"] for row in entry["frames"]}:
            raise Phase4AError("mismatched public evidence cannot reuse a target public frame")
        variants, pixel_audits = _variant_images(entry, frozen_root / entry["source_id"].replace(":", "_"), config["spatial"]["intervention"])
        variants["MISMATCHED_PUBLIC"] = {"frame_ids": [row["frame_id"] for row in mismatch_item["frames"]], "image_paths": [row["path"] for row in mismatch_item["frames"]],
                                         "selector": mismatch_item["selector"], "pixel_audits": []}
        entry = {**entry, "variants": variants, "mismatched_selector": mismatch_item["selector"]}
        all_entries.append(entry); audit_rows.extend(pixel_audits)
        for variant in VARIANTS:
            request = _request(entry, variant, variants[variant])
            contract_fn = getattr(backend, "forced_choice_token_contract", None)
            if not callable(contract_fn):
                raise Phase4AError("backend does not implement isolated forced-choice token validation")
            contract = contract_fn(request)
            if contract.get("choice_labels") != list(CHOICES) or set(contract.get("choice_token_ids", {})) != set(CHOICES) or len(set(contract["choice_token_ids"].values())) != 3:
                raise Phase4AError("A/B/C next-token contract is not unique in actual prompt context")
            contracts[f"{entry['source_id']}:{variant}"] = contract
    gt = (runtime_gt_audit(runtime_path, validate_prospective_manifest(prospective_manifest_path, runtime_path)[1])
          if runtime_path is not None else {"status": "NOT_APPLICABLE_DEVELOPMENT_ONLY", "gt_used": False})
    imports = audit_runtime_imports()
    if imports["status"] != "PASS":
        raise Phase4AError("runtime import audit failed")
    frozen_manifest = {"format": PHASE4A_FORMAT, "diagnostic_only": True, "selection_status": "FROZEN_PRE_INFERENCE", "entries": all_entries,
                       "mismatched_manifest": str(mismatched_manifest_path), "mismatched_manifest_sha256": _json(mismatched_manifest_path)["mismatched_manifest_sha256"],
                       "variant_order": list(VARIANTS), "operator": config["spatial"]["intervention"], "choice_prompt_version": PROMPT_VERSIONS["visual_dependence_choice"]}
    frozen_manifest["phase4a_frozen_manifest_sha256"] = _sha(frozen_manifest)
    (output_dir / "phase4a_frozen_manifest.json").write_text(json.dumps(frozen_manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    (output_dir / "phase4a_frozen_pixel_audits.jsonl").write_text("".join(canonical_json(row)+"\n" for row in audit_rows), encoding="utf-8")
    plan = {"format": PHASE4A_FORMAT, "status": "PASS", "mode": "preflight", "model_calls_made": 0, "cache_mutated": False,
            "diagnostic_only": True, "certificate_unchanged": True, "new_verified_count": 0, "gt_used": False, "git_commit": _git_commit(),
            "config": str(config_path), "config_sha256": sha256_path(config_path), "runtime": str(runtime_path),
            "prospective_manifest": str(prospective_manifest_path) if prospective_manifest_path else None, "lineage": lineage, "calibration": calibration, "development_controls": development,
            "frozen_manifest": str(output_dir / "phase4a_frozen_manifest.json"), "frozen_manifest_sha256": frozen_manifest["phase4a_frozen_manifest_sha256"],
            "choice_prompt_version": PROMPT_VERSIONS["visual_dependence_choice"], "choice_prompt_hashes": {entry["source_id"]: hashlib.sha256(choice_prompt(entry["claim"]).encode()).hexdigest() for entry in all_entries},
            "choice_token_contracts": contracts, "spatial_intervention": config["spatial"]["intervention"], "semantic_verifier_unchanged": True,
            "planned_model_calls": len(all_entries)*len(VARIANTS), "planned_call_derivation": "one forced-choice next-token likelihood call per frozen source/variant; no semantic verifier or certificate call",
            "cache_dir": str(output_dir / "cache"), "run_dir": str(output_dir / "run"), "replay_dir": str(output_dir / "replay"),
            "runtime_gt_isolation_audit": gt, "runtime_import_audit": imports, "prohibited_inputs_not_opened": PROHIBITED_INPUTS,
            "unavailable_phase35_sources": lineage["unavailable"]}
    (output_dir / "phase4a_preflight.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return plan


def execute(*, config_path: Path, runtime_path: Path | None = None, prospective_manifest_path: Path | None = None, phase35_v3_run_dir: Path | None = None,
            phase36_run_dir: Path | None = None, mismatched_manifest_path: Path | None = None, output_dir: Path | None = None, mode: str = "run", calibration_manifest_path: Path | None = None,
            development_manifest_path: Path | None = None,
            require_real: bool = True, backend_factory: Callable[[dict[str, Any]], Any] = make_backend) -> dict[str, Any]:
    if mode not in {"run", "replay"}:
        raise Phase4AError("mode must be run or replay")
    if output_dir is None: raise Phase4AError("output directory is required")
    plan = _json(output_dir / "phase4a_preflight.json"); frozen = _json(output_dir / "phase4a_frozen_manifest.json")
    _strict_frozen_hash(frozen, "phase4a_frozen_manifest_sha256")
    config = load_config(config_path); validate_config(config, require_real=require_real)
    if plan.get("config_sha256") != sha256_path(config_path) or plan.get("choice_prompt_version") != PROMPT_VERSIONS["visual_dependence_choice"]:
        raise Phase4AError("incompatible Phase 4A zero-call preflight")
    if frozen.get("operator") != config["spatial"]["intervention"]:
        raise Phase4AError("frozen Phase 4A operator changed")
    target = output_dir / mode
    if target.exists():
        raise Phase4AError(f"Phase 4A {mode} output already exists")
    backend = backend_factory(config["backend"])
    if require_real and getattr(backend, "synthetic", True):
        raise Phase4AError("Phase 4A requires the reviewed real local_hf backend")
    cache = ChoiceCache(backend, output_dir / "cache"); store = ArtifactStore(target); rows = []
    started = time.monotonic()
    with store.run_lock():
        for entry in frozen["entries"]:
            scored = {}
            for variant in VARIANTS:
                payload, request = entry["variants"][variant], _request(entry, variant, entry["variants"][variant])
                token_contract = plan["choice_token_contracts"].get(f"{entry['source_id']}:{variant}")
                if not isinstance(token_contract, dict):
                    raise Phase4AError("missing frozen token contract")
                scored[variant] = _result(entry, variant, request, cache.call(request, token_contract))
            margins = {variant: scored[variant]["support_margin"] for variant in VARIANTS}
            deltas = {"delta_drop": margins["ORIGINAL"]-margins["DROP_TARGET"], "delta_full_gray": margins["ORIGINAL"]-margins["FULL_GRAY"],
                      "delta_mismatch": margins["ORIGINAL"]-margins["MISMATCHED_PUBLIC"], "delta_specificity": margins["DROP_MATCHED_CONTROL"]-margins["DROP_TARGET"]}
            rows.append({"source_id": entry["source_id"], "source_kind": entry["source_kind"], "sample_id": entry.get("sample_id"), "claim_id": entry["claim_id"],
                         "claim_sha256": entry["claim_sha256"], "frozen_frame_ids": [item["frame_id"] for item in entry["frames"]],
                         "target_roi": entry["target_roi"], "matched_control_roi": entry["matched_control_roi"], "roi_source": entry["roi_source"],
                         "mismatched_selector": entry["mismatched_selector"], "generated_semantic_verdict_observation": entry["generated_semantic_verdict_observation"],
                         "calibration_expectation_present_posthoc_only": entry["calibration_expectation_present_posthoc_only"], "variants": scored, "deltas": deltas,
                         "candidate_explanation_pending_human_confirmation": _candidate_explanation(margins["ORIGINAL"], margins["DROP_TARGET"], margins["DROP_MATCHED_CONTROL"], margins["FULL_GRAY"], margins["MISMATCHED_PUBLIC"], entry["generated_semantic_verdict_observation"]),
                         "model_fingerprint": backend.fingerprint(),
                         "diagnostic_only": True, "certificate_unchanged": True, "new_verified_count": 0, "gt_used": False})
        (store.root / "phase4a_trace.jsonl").write_text("".join(canonical_json(row)+"\n" for row in rows), encoding="utf-8")
    if mode == "run" and cache.new_calls <= 0: raise Phase4AError("fresh Phase 4A run made zero model calls")
    if mode == "replay" and cache.new_calls != 0: raise Phase4AError("Phase 4A replay made unexpected new model calls")
    values_hash = stable_hash(_numeric_snapshot(rows))
    if mode == "replay":
        first = _jsonl(output_dir / "run" / "phase4a_trace.jsonl")
        first_hash = stable_hash(_numeric_snapshot(first))
        if values_hash != first_hash: raise Phase4AError("Phase 4A replay numeric results differ from first run")
    summary = {"format": PHASE4A_FORMAT, "status": "PASS", "mode": mode, "diagnostic_only": True, "certificate_unchanged": True, "new_verified_count": 0, "gt_used": False,
               "source_count": len(rows), "new_model_calls": cache.new_calls, "cache_hits": cache.cache_hits, "logical_model_calls": cache.logical_calls,
               "numeric_results_sha256": values_hash, "cache_dir": str(output_dir / "cache"), "output_dir": str(store.root), "latency_seconds": time.monotonic()-started,
               "candidate_explanation_counts": {label: sum(row["candidate_explanation_pending_human_confirmation"] == label for row in rows) for label in ("MARGIN_SENSITIVE_LABEL_INVARIANT", "VISUAL_INPUT_INSENSITIVITY", "AUTOMATIC_LOCALIZATION_FAILURE", "INTERVENTION_NONSPECIFIC", "UNRESOLVED")}}
    audit = {"status": "PASS", "diagnostic_only": True, "certificate_unchanged": True, "new_verified_count": 0, "gt_used": False,
             "no_certificate_builder_called": True, "semantic_verifier_not_called": True, "all_variants_same_choice_prompt_per_claim": all(len({row["variants"][variant]["prompt_sha256"] for variant in VARIANTS}) == 1 for row in rows),
             "replay_numeric_identical": mode != "replay" or True, "runtime_import_audit": audit_runtime_imports(),
             "runtime_gt_isolation_audit": (runtime_gt_audit(runtime_path, validate_prospective_manifest(prospective_manifest_path, runtime_path)[1])
                                              if runtime_path is not None else {"status": "NOT_APPLICABLE_DEVELOPMENT_ONLY", "gt_used": False})}
    report = {"status": "PASS" if audit["runtime_import_audit"]["status"] == "PASS" else "FAIL", "mode": mode, "summary": summary, "audit": audit, "traces": str(store.root / "phase4a_trace.jsonl")}
    (output_dir / f"phase4a_{mode}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    (output_dir / f"phase4a_{mode}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    (output_dir / f"phase4a_{mode}_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    if mode == "run":
        (output_dir / "phase4a_report.md").write_text("\n".join(["# ReliVE Phase 4A frozen visual-dependence audit", "", "This is diagnostic-only likelihood evidence. It does not modify certificates or establish VERIFIED.", "", "Raw A/B/C log probabilities, margins, and deltas are in `run/phase4a_trace.jsonl`; candidate explanation labels require human confirmation.", ""]), encoding="utf-8")
    return report
