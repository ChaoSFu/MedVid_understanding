from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

from PIL import Image

from relive.interventions import apply_spatial_intervention, generate_control_regions

from .common import SpatialReliabilityError, runtime_gt_audit, sha256_path, stable_hash, write_once
from .dataset import parse_token_constrained_proposal
from .engine import run_state_machine
from .inference import ModelSession
from .operators import apply_union_intervention, keep_residual, save_intervention
from .protocol import (PROPOSAL_PROMPT, PROPOSAL_PROMPT_VERSION, REFINEMENT_PROMPT,
                       REFINEMENT_PROMPT_VERSION, RESIDUAL_PROMPT, RESIDUAL_PROMPT_VERSION,
                       PROTOCOL_VERSION, protocol_manifest)


RUNTIME_FORMAT = "relive-spatial-runtime-v1"


def _rows(path: str | Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SpatialReliabilityError("RUNTIME_MANIFEST_INVALID") from exc
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise SpatialReliabilityError("RUNTIME_MANIFEST_INVALID")
    audit = runtime_gt_audit(rows)
    if audit["status"] != "PASS":
        raise SpatialReliabilityError("RUNTIME_GT_ISOLATION_FAILED")
    return rows


def _proposal_prompt(claim: str, extra: str = "") -> str:
    return f"Claim: {claim}\n{extra}\n{PROPOSAL_PROMPT}"


def _proposal_call_audit(call: dict[str, Any], stage: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "execution_status": call["execution_status"],
        "cache_hit": call["cache_hit"],
        "cache_key": call["cache_key"],
        "raw_response_ref": call["raw_response_ref"],
        "failure_reason": call["failure_reason"],
        "constraint_enforcement": call.get("constraint_enforcement"),
        "constraint_failure": call.get("constraint_failure"),
        "generation_metadata": call.get("generation_metadata"),
        "constraint_metadata": call.get("constraint_metadata"),
    }


def run_runtime(*, runtime_manifest: str | Path, config_path: str | Path, cache_dir: str | Path,
                output_dir: str | Path, mode: str = "relive_spatial", split: str = "development",
                max_samples: int | None = None, execution_mode: str = "run") -> dict[str, Any]:
    if execution_mode not in {"run", "replay"}:
        raise SpatialReliabilityError("EXECUTION_MODE_MUST_BE_RUN_OR_REPLAY")
    rows = [row for row in _rows(runtime_manifest) if row.get("split") == split]
    if max_samples is not None:
        rows = rows[:max_samples]
    if not rows:
        raise SpatialReliabilityError("EMPTY_RUNTIME_SPLIT")
    output = Path(output_dir)
    session = ModelSession(config_path=config_path, cache_dir=cache_dir, max_calls=max(1, len(rows) * 20))
    operator = session.config["spatial"]["intervention"]
    intervention_root = Path(cache_dir).expanduser().resolve() / "spatial_interventions" / PROTOCOL_VERSION
    sample_results = []
    for row in rows:
        sample_dir = intervention_root / row["sample_id"]
        frame_path = row["frame_path"]
        if not Path(frame_path).is_file():
            raise SpatialReliabilityError(f"RUNTIME_FRAME_MISSING:{row['sample_id']}")
        context: dict[str, Any] = {"formal": [], "proposal_generation_calls": []}
        drop_paths: dict[tuple[float, ...], str] = {}
        proposal_call = session.propose(
            stage="relive_spatial_proposal", prompt=_proposal_prompt(row["claim_text"]),
            prompt_version=PROPOSAL_PROMPT_VERSION, image_path=frame_path,
            frame_id=f"{row['sample_id']}:ORIGINAL",
        )
        context["proposal_generation_calls"].append(_proposal_call_audit(proposal_call, "R0_PROPOSAL"))
        proposal_raw = (proposal_call["raw_text"]
                        if proposal_call["execution_status"] == "OK" and not proposal_call.get("constraint_failure")
                        else None)

        def verify_region(region: list[float]) -> dict[str, Any]:
            tag = "region-" + stable_hash(region)[:12]
            with Image.open(frame_path) as opened:
                source = opened.convert("RGB")
            keep, keep_audit = apply_spatial_intervention(source, region, "KEEP_TARGET", operator)
            drop, drop_audit = apply_spatial_intervention(source, region, "DROP_TARGET", operator)
            keep_info = save_intervention(sample_dir / tag / "KEEP_TARGET.png", keep)
            drop_info = save_intervention(sample_dir / tag / "DROP_TARGET.png", drop)
            drop_paths[tuple(region)] = drop_info["path"]
            controls = generate_control_regions(region, 1)
            control_info = control_audit = None
            if controls["available"]:
                control_image, control_audit = apply_spatial_intervention(source, controls["regions"][0], "DROP_MATCHED_CONTROL", operator)
                control_info = save_intervention(sample_dir / tag / "DROP_MATCHED_CONTROL.png", control_image)
            paths = {"ORIGINAL": frame_path, "KEEP_TARGET": keep_info["path"], "DROP_TARGET": drop_info["path"]}
            if control_info:
                paths["CONTROL"] = control_info["path"]
            verification = {name: session.verify(claim=row["claim_text"], variant=name, image_path=path,
                                                 frame_id=f"{row['sample_id']}:{tag}:{name}",
                                                 identity_fields={"operator_version": operator["operator_version"],
                                                                  "bbox_normalized_0_1_xyxy": region})
                            for name, path in paths.items()}
            semantic = {name: value["semantic_status"] for name, value in verification.items()}
            detail = {"region": region, "tag": tag, "semantic": semantic, "verification": verification,
                      "control_geometry": controls, "pixel_audits": {"KEEP_TARGET": keep_audit, "DROP_TARGET": drop_audit,
                                                                      "CONTROL": control_audit},
                      "artifacts": {"KEEP_TARGET": keep_info, "DROP_TARGET": drop_info, "CONTROL": control_info}}
            context["formal"].append(detail)
            return {"original": semantic["ORIGINAL"], "keep": semantic["KEEP_TARGET"],
                    "drop": semantic["DROP_TARGET"], "control_constructible": controls["available"],
                    "control": semantic.get("CONTROL")}

        def refine(region: list[float]) -> str:
            generic = "Try again and propose another evidence region." if mode == "generic_retry" else REFINEMENT_PROMPT
            prompt = _proposal_prompt(row["claim_text"], f"Previous bbox normalized 0-1: {region}. Failure: KEEP_INSUFFICIENT. {generic}")
            call = session.propose(
                stage="relive_spatial_refine", prompt=prompt,
                prompt_version=("relive-spatial-generic-retry-v1" if mode == "generic_retry" else REFINEMENT_PROMPT_VERSION),
                image_path=frame_path, frame_id=f"{row['sample_id']}:R1",
            )
            context["proposal_generation_calls"].append(_proposal_call_audit(call, "R1_PROPOSAL"))
            return (call["raw_text"]
                    if call["execution_status"] == "OK" and not call.get("constraint_failure") else "")

        def residual(primary: list[float]) -> str:
            drop_path = drop_paths.get(tuple(primary))
            if not drop_path:
                return ""
            prompt = f"Claim: {row['claim_text']}\n{RESIDUAL_PROMPT}"
            call = session.propose(
                stage="relive_spatial_residual", prompt=prompt, prompt_version=RESIDUAL_PROMPT_VERSION,
                image_path=drop_path, frame_id=f"{row['sample_id']}:DROP_PRIMARY",
            )
            context["proposal_generation_calls"].append(_proposal_call_audit(call, "R2_RESIDUAL_PROPOSAL"))
            return (call["raw_text"]
                    if call["execution_status"] == "OK" and not call.get("constraint_failure") else "")

        def verify_residual(primary: list[float], residual_region: list[float]) -> str | None:
            drop_path = drop_paths.get(tuple(primary))
            if not drop_path:
                return None
            with Image.open(drop_path) as opened:
                image, audit = keep_residual(opened.convert("RGB"), residual_region, operator)
            info = save_intervention(sample_dir / "evidence-set" / "KEEP_RESIDUAL.png", image)
            verification = session.verify(claim=row["claim_text"], variant="KEEP_RESIDUAL",
                                          image_path=info["path"], frame_id=f"{row['sample_id']}:KEEP_RESIDUAL",
                                          identity_fields={"operator_version": operator["operator_version"],
                                                           "primary_bbox": primary, "residual_bbox": residual_region})
            context["keep_residual"] = {"region": residual_region, "audit": audit, "artifact": info, "verification": verification}
            return verification["semantic_status"]

        def verify_union(regions: list[list[float]]) -> dict[str, str | None]:
            with Image.open(frame_path) as opened:
                source = opened.convert("RGB")
            results, details = {}, {}
            for variant in ("KEEP_UNION", "DROP_UNION"):
                image, audit = apply_union_intervention(source, regions, variant, operator)
                info = save_intervention(sample_dir / "evidence-set" / f"{variant}.png", image)
                verification = session.verify(claim=row["claim_text"], variant=variant, image_path=info["path"],
                                              frame_id=f"{row['sample_id']}:{variant}",
                                              identity_fields={"operator_version": operator["operator_version"],
                                                               "evidence_set_regions": regions})
                results["keep" if variant == "KEEP_UNION" else "drop"] = verification["semantic_status"]
                details[variant] = {"audit": audit, "artifact": info, "verification": verification}
            context["union"] = details
            return results

        result = run_state_machine(sample_id=row["sample_id"], initial_proposal_raw=proposal_raw or "",
                                   verify_region=verify_region, refine=refine, residual=residual,
                                   verify_residual=verify_residual, verify_union=verify_union, mode=mode,
                                   proposal_parser=parse_token_constrained_proposal)
        sample_results.append({"format": RUNTIME_FORMAT, **result, "claim_id": row["claim_id"],
                               "proposal_call": _proposal_call_audit(proposal_call, "R0_PROPOSAL"),
                               "details": context, "source_frame_sha256": row.get("frame_sha256"),
                               "gt_used": False, "certificate_created": False, "new_verified_count": 0})
    counts = Counter(row["status"] for row in sample_results)
    proposal_calls = [call for row in sample_results
                      for call in row["details"]["proposal_generation_calls"]]
    proposal_failures = Counter(
        call.get("constraint_failure") or call.get("failure_reason")
        for call in proposal_calls
        if call.get("constraint_failure") or call.get("failure_reason")
    )
    usage = session.snapshot()
    summary = {"format": RUNTIME_FORMAT, "protocol_version": PROTOCOL_VERSION, "status": "PASS",
               "mode": mode, "execution_mode": execution_mode, "sample_count": len(sample_results),
               "scientific_status_counts": dict(sorted(counts.items())), "usage": usage,
               "new_model_calls": usage["new_calls"], "cache_hits": usage["cache_hits"],
               "proposal_generation_call_count": len(proposal_calls),
               "proposal_constraint_failure_count": sum(
                   call.get("constraint_failure") is not None for call in proposal_calls
               ),
               "proposal_generation_failure_reason_counts": dict(sorted(proposal_failures.items())),
               "gt_used": False, "certificate_created": False, "new_verified_count": 0}
    adaptation = {"format": RUNTIME_FORMAT + "-adaptation", "initial_proposal_count": len(sample_results),
                  "initial_keep_failure_count": sum(any(x.get("stage") == "CONTROLLER" and x.get("action") == "SUFFICIENCY_REPAIR" for x in row["trace"]) for row in sample_results),
                  "sufficiency_repair_attempted": sum(row["repair_attempts"] for row in sample_results),
                  "sufficiency_repair_recovered": sum(any(x.get("stage") == "R1_FORMAL" and x.get("status") not in {"SPATIAL_INSUFFICIENT", "TECHNICAL_FAILURE"} for x in row["trace"]) for row in sample_results),
                  "residual_search_attempted": sum(row["residual_attempts"] for row in sample_results),
                  "residual_r2_proposed": sum(any(x.get("stage") == "R2_RESIDUAL_PROPOSAL" and x.get("result", {}).get("status") == "PROPOSED" for x in row["trace"]) for row in sample_results),
                  "residual_r2_sufficient": sum(any(x.get("stage") == "KEEP_RESIDUAL" and x.get("status") == "SUPPORTED" for x in row["trace"]) for row in sample_results),
                  "evidence_set_verified": counts.get("EVIDENCE_SET_VERIFIED", 0),
                  "dependency_unresolved": counts.get("DEPENDENCY_UNRESOLVED", 0)}
    adaptation["adaptation_recovery_rate"] = (adaptation["sufficiency_repair_recovered"] / adaptation["sufficiency_repair_attempted"]
                                               if adaptation["sufficiency_repair_attempted"] else None)
    adaptation["evidence_set_recovery_rate"] = (adaptation["evidence_set_verified"] / adaptation["residual_search_attempted"]
                                                 if adaptation["residual_search_attempted"] else None)
    failure = {"format": RUNTIME_FORMAT + "-failure-breakdown",
               "scientific_status_counts": dict(sorted(counts.items())),
               "proposal_generation_failure_reason_counts": dict(sorted(proposal_failures.items()))}
    replay = {"format": RUNTIME_FORMAT + "-replay-audit", "execution_mode": execution_mode,
              "new_model_calls": usage["new_calls"], "cache_hits": usage["cache_hits"],
              "exact_replay_pass": execution_mode != "replay" or usage["new_calls"] == 0}
    gt_audit = runtime_gt_audit(rows)
    for name, value, jsonl in (
        ("protocol.json", protocol_manifest(), False), ("sample_results.jsonl", sample_results, True),
        ("runtime_summary.json", summary, False), ("adaptation_summary.json", adaptation, False),
        ("failure_breakdown.json", failure, False), ("cache_audit.json", replay, False),
        ("gt_isolation_audit.json", gt_audit, False),
    ):
        write_once(output / name, value, jsonl=jsonl)
    markdown = (
        "# ReliVE-Spatial runtime summary\n\n"
        f"- Mode: {mode}\n- Execution: {execution_mode}\n- Samples: {len(sample_results)}\n"
        f"- Scientific states: `{json.dumps(dict(sorted(counts.items())), sort_keys=True)}`\n"
        f"- New model calls: {usage['new_calls']}\n- Cache hits: {usage['cache_hits']}\n\n"
        "These are engineering and frozen-protocol runtime outcomes. No GT was opened, and no legacy "
        "VERIFIED certificate was created. SUPPORTED_EVIDENCE means sufficient but not individually "
        "necessary evidence.\n"
    ).encode("utf-8")
    summary_path = output / "summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if summary_path.exists() and summary_path.read_bytes() != markdown:
        raise SpatialReliabilityError("IMMUTABLE_ARTIFACT_EXISTS:summary.md")
    if not summary_path.exists():
        summary_path.write_bytes(markdown)
    return summary
