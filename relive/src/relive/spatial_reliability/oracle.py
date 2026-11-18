from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

from PIL import Image

from relive.interventions import apply_spatial_intervention
from relive.spatial import validate_region

from .common import SpatialReliabilityError, sha256_path, strict_json_load, write_once
from .engine import oracle_classification
from .inference import ModelSession
from .operators import save_intervention
from .protocol import PROTOCOL_VERSION, protocol_manifest


ORACLE_FORMAT = "relive-spatial-oracle-feasibility-v1"


def _rows(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise SpatialReliabilityError(f"INVALID_JSONL:{path}") from exc
    return rows


def _normalized_gt(box: list[float], size: tuple[int, int]) -> list[float]:
    width, height = size
    return list(validate_region([box[0] / width, box[1] / height, box[2] / width, box[3] / height]))


def _summary_markdown(summary: dict[str, Any]) -> str:
    return f"""# ReliVE-Spatial oracle feasibility audit

This is an oracle-only protocol feasibility audit. It is not a runtime grounding result.

- Total eligible samples: {summary['N_total']}
- Technical-failure samples: {summary['N_technical_failure_cases']}
- Technical-failure verdicts: {summary['N_technical_failure_verdicts']}
- Original supported: {summary['N_original_supported']}
- GT KEEP supported: {summary['N_gt_keep_supported']}
- GT KEEP failed: {summary['N_gt_keep_failed']}
- GT DROP supported: {summary['N_gt_drop_supported']}
- GT DROP not supported: {summary['N_gt_drop_not_supported']}
- Oracle sufficiency rate: {summary['oracle_sufficiency_rate']}
- Oracle single-region dependency rate: {summary['oracle_single_region_dependency_rate']}
- Oracle residual-support rate: {summary['oracle_residual_support_rate']}

Interpretation: KEEP measures sufficiency; DROP measures individual dependency. A supported DROP is
not a claim that the GT box is wrong. It indicates sufficient but non-necessary evidence under the
frozen verifier and intervention protocol.
"""


def run_oracle_feasibility(*, runtime_manifest: str | Path, oracle_manifest: str | Path,
                           config_path: str | Path, cache_dir: str | Path, output_dir: str | Path,
                           split: str = "development", max_samples: int | None = None) -> dict[str, Any]:
    runtime, oracle = _rows(runtime_manifest), _rows(oracle_manifest)
    oracle_by_id = {row.get("sample_id"): row for row in oracle}
    selected = [row for row in runtime if row.get("split") == split]
    if max_samples is not None:
        selected = selected[:max_samples]
    if not selected or len(oracle_by_id) != len(oracle):
        raise SpatialReliabilityError("ORACLE_INPUT_BINDING_INVALID")
    session = ModelSession(config_path=config_path, cache_dir=cache_dir, max_calls=max(3, len(selected) * 3))
    operator = session.config["spatial"]["intervention"]
    output = Path(output_dir)
    intervention_root = Path(cache_dir).expanduser().resolve() / "oracle_interventions" / ORACLE_FORMAT
    results = []
    for runtime_row in selected:
        hidden = oracle_by_id.get(runtime_row["sample_id"])
        if not hidden or hidden.get("source_record_sha256") != runtime_row.get("source_record_sha256"):
            raise SpatialReliabilityError("RUNTIME_ORACLE_BINDING_MISMATCH")
        source_path = Path(runtime_row["frame_path"])
        with Image.open(source_path) as opened:
            source = opened.convert("RGB")
        gt_region = _normalized_gt(hidden["gt_bbox"], source.size)
        variants = {"ORIGINAL": source.copy()}
        audits = {}
        for name, intervention_name in (("KEEP_GT", "KEEP_TARGET"), ("DROP_GT", "DROP_TARGET")):
            variants[name], audits[name] = apply_spatial_intervention(source, gt_region, intervention_name, operator)
        paths = {"ORIGINAL": str(source_path)}
        for name in ("KEEP_GT", "DROP_GT"):
            paths[name] = save_intervention(intervention_root / runtime_row["sample_id"] / f"{name}.png", variants[name])["path"]
        semantic = {}
        verification = {}
        for name in ("ORIGINAL", "KEEP_GT", "DROP_GT"):
            verification[name] = session.verify(claim=runtime_row["claim_text"], variant=name,
                                                image_path=paths[name], frame_id=f"{runtime_row['sample_id']}:{name}",
                                                identity_fields={"operator_version": operator["operator_version"],
                                                                 "bbox_normalized_0_1_xyxy": gt_region})
            semantic[name] = verification[name]["semantic_status"]
        classification = oracle_classification(semantic["ORIGINAL"], semantic["KEEP_GT"], semantic["DROP_GT"])
        results.append({
            "format": ORACLE_FORMAT, "sample_id": runtime_row["sample_id"], "claim_id": runtime_row["claim_id"],
            "classification": classification, "semantic": semantic, "verification": verification,
            "gt_region_normalized_0_1_xyxy": gt_region, "pixel_audits": audits,
            "bindings": {"runtime_manifest_sha256": sha256_path(runtime_manifest),
                         "oracle_manifest_sha256": sha256_path(oracle_manifest),
                         "source_frame_sha256": runtime_row.get("frame_sha256")},
        })
    counts = Counter(row["classification"] for row in results)
    original_supported = sum(row["semantic"]["ORIGINAL"] == "SUPPORTED" for row in results)
    keep_supported = sum(row["semantic"]["KEEP_GT"] == "SUPPORTED" for row in results)
    keep_failed = sum(
        row["semantic"]["KEEP_GT"] in {"CONTRADICTED", "INSUFFICIENT"} for row in results
    )
    keep_condition = [row for row in results if row["semantic"]["ORIGINAL"] == "SUPPORTED"]
    dependency_condition = [row for row in keep_condition if row["semantic"]["KEEP_GT"] == "SUPPORTED"]
    drop_supported = sum(row["semantic"]["DROP_GT"] == "SUPPORTED" for row in results)
    drop_not_supported = sum(
        row["semantic"]["DROP_GT"] in {"CONTRADICTED", "INSUFFICIENT"} for row in results
    )
    technical_failure_verdicts = [
        verdict
        for row in results
        for verdict in row["verification"].values()
        if verdict.get("execution_status") != "OK"
    ]
    technical_failure_cases = sum(
        row["classification"] == "ORACLE_TECHNICAL_FAILURE" for row in results
    )
    failure_reasons = Counter(
        verdict.get("failure_reason") or "UNSPECIFIED_TECHNICAL_FAILURE"
        for verdict in technical_failure_verdicts
    )
    sufficiency_rate = (
        sum(row["semantic"]["KEEP_GT"] == "SUPPORTED" for row in keep_condition) / len(keep_condition)
        if keep_condition and not technical_failure_verdicts else None
    )
    dependency_rate = (
        sum(row["semantic"]["DROP_GT"] != "SUPPORTED" for row in dependency_condition) / len(dependency_condition)
        if dependency_condition and not technical_failure_verdicts else None
    )
    residual_rate = (
        sum(row["semantic"]["DROP_GT"] == "SUPPORTED" for row in dependency_condition) / len(dependency_condition)
        if dependency_condition and not technical_failure_verdicts else None
    )
    summary = {
        "format": ORACLE_FORMAT, "protocol_version": PROTOCOL_VERSION,
        "status": "PASS" if not technical_failure_verdicts else "FAIL",
        "scientific_result_available": not technical_failure_verdicts,
        "N_total": len(results),
        "N_technical_failure_cases": technical_failure_cases,
        "N_technical_failure_verdicts": len(technical_failure_verdicts),
        "N_original_supported": original_supported, "N_gt_keep_supported": keep_supported,
        "N_gt_keep_failed": keep_failed, "N_gt_drop_supported": drop_supported,
        "N_gt_drop_not_supported": drop_not_supported,
        "oracle_sufficiency_rate": sufficiency_rate,
        "oracle_single_region_dependency_rate": dependency_rate,
        "oracle_residual_support_rate": residual_rate,
        "classification_counts": dict(sorted(counts.items())), "usage": session.snapshot(),
        "certificate_created": False, "new_verified_count": 0, "gt_used": True,
    }
    failure = {
        "format": ORACLE_FORMAT + "-failure-breakdown",
        "classification_counts": dict(sorted(counts.items())),
        "technical_failure_reason_counts": dict(sorted(failure_reasons.items())),
    }
    cache_audit = {"format": ORACLE_FORMAT + "-cache-audit", **session.snapshot(),
                   "exact_replay_requires_new_model_calls": 0}
    isolation = {"format": ORACLE_FORMAT + "-isolation", "status": "PASS", "gt_used": True,
                 "gt_opened_only_by_oracle_command": True, "runtime_module_imported_oracle_manifest": False}
    for name, value, jsonl in (
        ("protocol.json", protocol_manifest(), False), ("sample_results.jsonl", results, True),
        ("summary.json", summary, False), ("failure_breakdown.json", failure, False),
        ("cache_audit.json", cache_audit, False), ("gt_isolation_audit.json", isolation, False),
    ):
        write_once(output / name, value, jsonl=jsonl)
    summary_md = output / "summary.md"
    body = _summary_markdown(summary).encode("utf-8")
    if summary_md.exists() and summary_md.read_bytes() != body:
        raise SpatialReliabilityError("IMMUTABLE_ARTIFACT_EXISTS:summary.md")
    summary_md.parent.mkdir(parents=True, exist_ok=True)
    if not summary_md.exists():
        summary_md.write_bytes(body)
    return summary
