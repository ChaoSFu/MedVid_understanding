#!/usr/bin/env python3
"""Phase 3.5: GT-free spatial-certificate applicability audit and cohort freeze."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from relive.audits import audit_runtime_imports
from relive.claim_scope import (CLAIM_SCOPE_ROUTER_VERSION, LOCAL_ATOMIC, canonical_scope_record,
                                stable_scope_hash)
from relive.data.schemas import load_runtime
from relive.storage.artifacts import canonical_json


class Phase35Error(ValueError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase35Error(f"unreadable JSON: {path}") from exc
    if not isinstance(data, dict):
        raise Phase35Error(f"JSON object required: {path}")
    return data


def _jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase35Error(f"unreadable JSONL: {path}") from exc
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise Phase35Error(f"nonempty JSONL object rows required: {path}")
    return rows


def _gt_audit(runtime: Path, samples) -> dict[str, Any]:
    sidecar = Path(str(runtime) + ".gt_isolation_audit.json")
    audit = _json(sidecar)
    runtime_sha = samples[0].provenance.get("runtime_sha256")
    if audit.get("status") != "PASS" or audit.get("runtime_sha256") != runtime_sha:
        raise Phase35Error("runtime GT-isolation audit is missing, failed, or not bound to this runtime")
    return {"status": "PASS", "path": str(sidecar), "runtime_sha256": runtime_sha,
            "adapter": audit.get("adapter"), "classification": audit.get("classification"),
            "prohibited_inputs_not_opened": ["reference_answer", "assistant_answer", "temporal_gt",
                                               "bbox_mask_gt", "struc_info", "RC_info", "evaluation_artifacts"]}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise Phase35Error(f"refusing to overwrite immutable artifact: {path}")
    path.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise Phase35Error(f"refusing to overwrite immutable artifact: {path}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _candidate_rows(phase2_run: Path, samples_by_id: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    path = phase2_run / "fixed_candidate_pool.jsonl"
    rows = _jsonl(path)
    hashes = {row.get("manifest_hash") for row in rows}
    if len(hashes) != 1 or not isinstance(next(iter(hashes)), str):
        raise Phase35Error("Phase 2 candidate manifest lacks one immutable hash")
    for row in rows:
        sample_id = row.get("sample_id")
        if sample_id not in samples_by_id or not isinstance(row.get("candidate_id"), str):
            raise Phase35Error("Phase 2 candidate manifest cannot bind to development runtime")
    return rows, next(iter(hashes))


def _phase25_binding(path: Path, candidate_keys: set[tuple[str, str]]) -> dict[str, Any]:
    rows = _jsonl(path)
    keys = {(row.get("qa_id"), row.get("candidate_id")) for row in rows}
    if not keys or not keys.issubset(candidate_keys):
        raise Phase35Error("Phase 2.5 diagnostic IDs do not bind to the frozen Phase 2 candidate manifest")
    # Read only IDs for historical-cohort binding. Values such as failure codes
    # and certificate outcomes are intentionally neither copied nor routed on.
    return {"path": str(path), "sha256": _sha(path), "candidate_count": len(keys),
            "routing_inputs_used": ["qa_id", "candidate_id"], "certificate_outcomes_used": False}


def _development_rows(manifest: list[dict[str, Any]], samples_by_id: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for candidate in manifest:
        rows.append(canonical_scope_record(samples_by_id[candidate["sample_id"]], candidate_id=candidate["candidate_id"],
                                           candidate_rank=candidate.get("candidate_rank"), development_only=True))
    return rows


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_scope = Counter(row["claim_scope"] for row in rows)
    by_sample: dict[str, str] = {}
    for row in rows:
        known = by_sample.setdefault(row["sample_id"], row["claim_scope"])
        if known != row["claim_scope"]:
            raise Phase35Error("one frozen claim received inconsistent scope decisions")
    return {"candidate_count": len(rows), "sample_count": len(by_sample),
            "scope_distribution_by_candidate": dict(sorted(by_scope.items())),
            "scope_distribution_by_sample": dict(sorted(Counter(by_sample.values()).items())),
            "single_roi_eligible_candidate_count": by_scope[LOCAL_ATOMIC],
            "single_roi_eligible_sample_count": sum(scope == LOCAL_ATOMIC for scope in by_sample.values())}


def _markdown(summary: dict[str, Any], records: list[dict[str, Any]], prospective: dict[str, Any]) -> str:
    lines = ["# ReliVE Phase 3.5 claim-scope applicability audit", "",
             "This is a GT-free development diagnostic. Scope routing determines whether the current single-ROI protocol applies; it does not verify a claim or reinterpret historical certificates.", "",
             "## Historical development cohort", "", "| Scope | Candidate count |", "| --- | ---: |"]
    lines.extend(f"| `{scope}` | {count} |" for scope, count in summary["scope_distribution_by_candidate"].items())
    lines += ["", "## Routed examples", "", "| Sample | QA type | Scope | Reason |", "| --- | --- | --- | --- |"]
    seen = set()
    for row in records:
        if row["claim_scope"] in seen:
            continue
        seen.add(row["claim_scope"])
        lines.append(f"| `{row['sample_id']}` | `{row.get('source_qa_type')}` | `{row['claim_scope']}` | `{row['reason_code']}` |")
    lines += ["", "## Prospective frozen cohort", "", f"Selected LOCAL_ATOMIC samples: {prospective['selected_sample_count']}.",
              "Selection occurred before spatial proposals, interventions, semantic outcomes, or certificate outcomes. Phase 3-v3 is not run by this command.", ""]
    return "\n".join(lines)


def run_phase35(*, development_runtime: Path, phase2_run: Path, phase25_diagnostics: Path,
                prospective_runtime: Path, output_dir: Path, max_prospective_samples: int) -> dict[str, Any]:
    if output_dir.exists():
        raise Phase35Error("Phase 3.5 output directory must not exist")
    if type(max_prospective_samples) is not int or max_prospective_samples < 1:
        raise Phase35Error("max_prospective_samples must be a positive integer")
    dev_samples = load_runtime(development_runtime)
    future_samples = load_runtime(prospective_runtime)
    development_gt = _gt_audit(development_runtime, dev_samples)
    prospective_gt = _gt_audit(prospective_runtime, future_samples)
    source_audit = audit_runtime_imports()
    if source_audit["status"] != "PASS":
        raise Phase35Error("runtime import audit failed")
    dev_by_id = {sample.sample_id: sample for sample in dev_samples}
    manifest, manifest_hash = _candidate_rows(phase2_run, dev_by_id)
    phase25_binding = _phase25_binding(phase25_diagnostics, {(row["sample_id"], row["candidate_id"]) for row in manifest})
    historical_ids = {row["sample_id"] for row in manifest}
    if any(sample.sample_id in historical_ids for sample in future_samples):
        overlap = sorted(sample.sample_id for sample in future_samples if sample.sample_id in historical_ids)
        raise Phase35Error(f"prospective runtime overlaps Phase 3-v1/v2 development samples: {overlap}")
    records = _development_rows(manifest, dev_by_id)
    summary = _summary(records)
    prospective_all = [canonical_scope_record(sample, development_only=False) for sample in future_samples]
    selected = [row for row in prospective_all if row["claim_scope"] == LOCAL_ATOMIC][:max_prospective_samples]
    if not selected:
        raise Phase35Error("prospective runtime has no LOCAL_ATOMIC claim; no eligible cohort can be frozen")
    prospective_payload = {
        "format": "relive-phase35-prospective-local-atomic-manifest-v1",
        "router_version": CLAIM_SCOPE_ROUTER_VERSION,
        "selection_status": "FROZEN_PRE_SPATIAL_CERTIFICATE_OUTCOMES",
        "selection_rule": {"input_order": "public runtime JSONL order", "include_scope": LOCAL_ATOMIC,
                           "max_samples": max_prospective_samples, "exclude_sample_ids_from": "Phase 2 frozen candidate manifest"},
        "prospective_runtime": str(prospective_runtime), "prospective_runtime_sha256": future_samples[0].provenance["runtime_sha256"],
        "excluded_historical_sample_ids_sha256": stable_scope_hash([{"sample_id": value} for value in sorted(historical_ids)]),
        "selected_sample_count": len(selected), "selected": selected,
        "not_a_temporal_candidate_pool": True, "phase3_v3_executed": False, "gt_used": False,
        "prohibited_inputs_not_opened": ["reference_answer", "assistant_answer", "temporal_gt", "bbox_mask_gt",
                                           "struc_info", "RC_info", "evaluation_artifacts"],
    }
    prospective_payload["manifest_sha256"] = stable_scope_hash([{key: value for key, value in prospective_payload.items() if key != "manifest_sha256"}])
    output_dir.mkdir(parents=True)
    _write_jsonl(output_dir / "claim_scope_development_audit.jsonl", records)
    _write_jsonl(output_dir / "claim_scope_prospective_audit.jsonl", prospective_all)
    _write_json(output_dir / "prospective_local_atomic_manifest.json", prospective_payload)
    report = {"status": "PASS", "phase": "ReliVE Phase 3.5 Spatial Certificate Applicability Audit + Claim-Scope Router",
              "model_calls_made": 0, "cache_mutated": False, "router_version": CLAIM_SCOPE_ROUTER_VERSION,
              "development_runtime": str(development_runtime), "development_runtime_sha256": dev_samples[0].provenance["runtime_sha256"],
              "phase2_candidate_manifest": str(phase2_run / "fixed_candidate_pool.jsonl"), "phase2_candidate_manifest_sha256": _sha(phase2_run / "fixed_candidate_pool.jsonl"),
              "phase2_candidate_manifest_hash": manifest_hash, "phase25_binding": phase25_binding,
              "historical_development_distribution": summary, "prospective_manifest": str(output_dir / "prospective_local_atomic_manifest.json"),
              "prospective_manifest_sha256": prospective_payload["manifest_sha256"], "proposed_fresh_local_atomic_sample_count": len(selected),
              "development_gt_isolation_audit": development_gt, "prospective_gt_isolation_audit": prospective_gt,
              "runtime_import_audit": source_audit, "historical_certificates_reinterpreted": False,
              "phase3_v3_executed": False}
    _write_json(output_dir / "phase35_claim_scope_report.json", report)
    (output_dir / "phase35_claim_scope_report.md").write_text(_markdown(summary, records, prospective_payload), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="ReliVE Phase 3.5 GT-free claim-scope applicability audit")
    parser.add_argument("--development-runtime", required=True)
    parser.add_argument("--phase2-run-dir", required=True)
    parser.add_argument("--phase25-diagnostics", required=True)
    parser.add_argument("--prospective-runtime", required=True,
                        help="Fresh public runtime whose sample IDs do not occur in the Phase 2 development cohort")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-prospective-samples", type=int, default=5)
    args = parser.parse_args()
    try:
        report = run_phase35(development_runtime=Path(args.development_runtime).resolve(),
                             phase2_run=Path(args.phase2_run_dir).resolve(),
                             phase25_diagnostics=Path(args.phase25_diagnostics).resolve(),
                             prospective_runtime=Path(args.prospective_runtime).resolve(),
                             output_dir=Path(args.output_dir).resolve(), max_prospective_samples=args.max_prospective_samples)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, Phase35Error) as exc:
        print(f"ReliVE Phase 3.5 error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
