"""Apply explicit author feedback and freeze a reviewed atomic-claim cohort."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from .atomic_claims import (ATOMIC_CLAIM_FORMAT, atomic_claim_static_checks,
                            parse_atomic_claim)
from .common import (SpatialReliabilityError, canonical_bytes, runtime_gt_audit,
                     sha256_path, stable_hash, strict_json_load, write_once)


FREEZE_FORMAT = "relive-spatial-reviewed-atomic-claims-v1"
REVIEWED_CLAIM_VERSION = "relive-spatial-reviewed-atomic-claim-v1"
FEEDBACK_FORMAT = "relive-spatial-atomic-claim-author-feedback-v1"


def _strict_jsonl(path: str | Path, code: str) -> list[dict[str, Any]]:
    def reject_constant(value: str) -> None:
        raise ValueError(value)

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(key)
            value[key] = item
        return value

    try:
        rows = [
            json.loads(line, parse_constant=reject_constant, object_pairs_hook=reject_duplicates)
            for line in Path(path).read_bytes().decode("utf-8").splitlines() if line.strip()
        ]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise SpatialReliabilityError(code) from exc
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise SpatialReliabilityError(code)
    return rows


def _feedback(path: str | Path, expected_count: int) -> tuple[
        dict[int, dict[str, Any]], dict[int, dict[str, Any]], dict[str, Any]]:
    value = strict_json_load(path)
    required = {
        "format", "expected_candidate_count", "position_index_base", "default_action",
        "reviewer_id", "human_feedback_source", "replacements", "exclusions",
    }
    if (not isinstance(value, dict) or set(value) != required
            or value.get("format") != FEEDBACK_FORMAT
            or value.get("expected_candidate_count") != expected_count
            or value.get("position_index_base") != 1
            or value.get("default_action") != "ACCEPT_GENERATED"
            or not isinstance(value.get("reviewer_id"), str) or not value["reviewer_id"]
            or not isinstance(value.get("human_feedback_source"), str)
            or not isinstance(value.get("replacements"), list)
            or not isinstance(value.get("exclusions"), list)):
        raise SpatialReliabilityError("ATOMIC_CLAIM_FEEDBACK_INVALID")
    replacements: dict[int, dict[str, Any]] = {}
    for item in value["replacements"]:
        if (not isinstance(item, dict) or set(item) != {"position", "final_claim", "rationale"}
                or type(item.get("position")) is not int
                or not 1 <= item["position"] <= expected_count
                or item["position"] in replacements
                or not isinstance(item.get("final_claim"), str)
                or parse_atomic_claim(item["final_claim"])["status"] != "PROPOSED"
                or not isinstance(item.get("rationale"), str) or not item["rationale"].strip()):
            raise SpatialReliabilityError("ATOMIC_CLAIM_FEEDBACK_INVALID")
        replacements[item["position"]] = item
    if not replacements:
        raise SpatialReliabilityError("ATOMIC_CLAIM_FEEDBACK_INVALID")
    exclusions: dict[int, dict[str, Any]] = {}
    for item in value["exclusions"]:
        if (not isinstance(item, dict) or set(item) != {"position", "rationale"}
                or type(item.get("position")) is not int
                or not 1 <= item["position"] <= expected_count
                or item["position"] in exclusions or item["position"] in replacements
                or not isinstance(item.get("rationale"), str) or not item["rationale"].strip()):
            raise SpatialReliabilityError("ATOMIC_CLAIM_FEEDBACK_INVALID")
        exclusions[item["position"]] = item
    if not exclusions:
        raise SpatialReliabilityError("ATOMIC_CLAIM_FEEDBACK_INVALID")
    return replacements, exclusions, value


def _markdown(decisions: list[dict[str, Any]]) -> str:
    lines = [
        "# Frozen reviewed atomic claims",
        "",
        "These claims are author-reviewed inputs for the replacement experiment. The former frozen-evaluation rows are marked curated because the author inspected them.",
        "",
    ]
    for row in decisions:
        lines.extend([
            f"## {row['position']:02d}. `{row['sample_id']}`",
            "",
            f"- Split: `{row['final_split']}` (source: `{row['source_split']}`)",
            f"- RC Caption: {row['source_region_caption']}",
            f"- Generated claim: {row['generated_atomic_claim']}",
            f"- Final claim: **{row['final_claim'] or 'EXCLUDED'}**",
            f"- Decision: `{row['decision']}`",
            f"- Rationale: {row['rationale']}",
            "",
        ])
    return "\n".join(lines)


def freeze_reviewed_atomic_claims(*, runtime_manifest: str | Path,
                                  candidates_jsonl: str | Path,
                                  review_jsonl: str | Path,
                                  feedback_json: str | Path,
                                  output_dir: str | Path,
                                  expected_count: int = 32) -> dict[str, Any]:
    if type(expected_count) is not int or expected_count < 1:
        raise SpatialReliabilityError("ATOMIC_CLAIM_FREEZE_COUNT_INVALID")
    runtime = _strict_jsonl(runtime_manifest, "ATOMIC_CLAIM_FREEZE_RUNTIME_INVALID")
    candidates = _strict_jsonl(candidates_jsonl, "ATOMIC_CLAIM_FREEZE_CANDIDATES_INVALID")
    reviews = _strict_jsonl(review_jsonl, "ATOMIC_CLAIM_FREEZE_REVIEW_INVALID")
    if runtime_gt_audit(runtime)["status"] != "PASS":
        raise SpatialReliabilityError("ATOMIC_CLAIM_FREEZE_RUNTIME_GT_INVALID")
    if not (len(runtime) == len(candidates) == len(reviews) == expected_count):
        raise SpatialReliabilityError("ATOMIC_CLAIM_FREEZE_COUNT_MISMATCH")
    replacements, exclusions, feedback = _feedback(feedback_json, expected_count)
    final_runtime: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    seen_samples: set[str] = set()
    seen_candidates: set[str] = set()
    for position, (public, candidate, review) in enumerate(zip(runtime, candidates, reviews), start=1):
        sample_id = public.get("sample_id")
        candidate_id = candidate.get("candidate_id")
        parsed = candidate.get("candidate")
        if (not isinstance(sample_id, str) or sample_id in seen_samples
                or not isinstance(candidate_id, str) or candidate_id in seen_candidates
                or candidate.get("format") != ATOMIC_CLAIM_FORMAT
                or review.get("format") != ATOMIC_CLAIM_FORMAT + "-human-review"
                or candidate.get("sample_id") != sample_id or review.get("sample_id") != sample_id
                or review.get("candidate_id") != candidate_id
                or candidate.get("source_record_index") != public.get("source_record_index")
                or candidate.get("source_record_sha256") != public.get("source_record_sha256")
                or candidate.get("current_claim_id") != public.get("claim_id")
                or candidate.get("current_claim_text") != public.get("claim_text")
                or not isinstance(parsed, dict)
                or review.get("generated_status") != parsed.get("status")
                or review.get("generated_atomic_claim") != parsed.get("claim_text")
                or review.get("source_region_caption") != candidate.get("source_region_caption")):
            raise SpatialReliabilityError(f"ATOMIC_CLAIM_FREEZE_BINDING_INVALID:{position}")
        generated = parsed.get("claim_text")
        exclusion = exclusions.get(position)
        if exclusion:
            decisions.append({
                "format": FREEZE_FORMAT + "-decision",
                "position": position,
                "candidate_id": candidate_id,
                "sample_id": sample_id,
                "source_record_index": public["source_record_index"],
                "source_record_sha256": public["source_record_sha256"],
                "source_split": public.get("split"),
                "final_split": None,
                "source_region_caption": candidate["source_region_caption"],
                "generated_atomic_claim": generated,
                "final_claim": None,
                "decision": "EXCLUDED_NO_VALID_ATOMIC_CLAIM",
                "rationale": exclusion["rationale"],
                "reviewer_id": feedback["reviewer_id"],
                "human_feedback_source": feedback["human_feedback_source"],
                "final_claim_static_checks": None,
            })
            seen_samples.add(sample_id)
            seen_candidates.add(candidate_id)
            continue
        if parsed.get("status") != "PROPOSED" or not isinstance(generated, str):
            raise SpatialReliabilityError(f"ATOMIC_CLAIM_UNREVIEWED_NONPROPOSAL:{position}")
        replacement = replacements.get(position)
        final_claim = replacement["final_claim"] if replacement else generated
        if parse_atomic_claim(final_claim)["status"] != "PROPOSED":
            raise SpatialReliabilityError(f"ATOMIC_CLAIM_FINAL_CLAIM_INVALID:{position}")
        source_split = public.get("split")
        final_split = "curated_evaluation" if source_split == "frozen_evaluation" else source_split
        if final_split not in {"development", "curated_evaluation"}:
            raise SpatialReliabilityError(f"ATOMIC_CLAIM_SPLIT_INVALID:{position}")
        frozen = dict(public)
        frozen.update({
            "claim_id": "claim-" + stable_hash({
                "sample_id": sample_id,
                "text": final_claim,
                "version": REVIEWED_CLAIM_VERSION,
            })[:24],
            "claim_text": final_claim,
            "claim_scope": "LOCAL_ATOMIC",
            "split": final_split,
            "normalizer_version": REVIEWED_CLAIM_VERSION,
            "scope_router_version": REVIEWED_CLAIM_VERSION,
        })
        final_runtime.append(frozen)
        decisions.append({
            "format": FREEZE_FORMAT + "-decision",
            "position": position,
            "candidate_id": candidate_id,
            "sample_id": sample_id,
            "source_record_index": public["source_record_index"],
            "source_record_sha256": public["source_record_sha256"],
            "source_split": source_split,
            "final_split": final_split,
            "source_region_caption": candidate["source_region_caption"],
            "generated_atomic_claim": generated,
            "final_claim": final_claim,
            "decision": "AUTHOR_REPLACEMENT" if replacement else "ACCEPT_GENERATED",
            "rationale": (replacement["rationale"] if replacement
                          else "Author reviewed the generated claim and reported no correction."),
            "reviewer_id": feedback["reviewer_id"],
            "human_feedback_source": feedback["human_feedback_source"],
            "final_claim_static_checks": atomic_claim_static_checks(
                final_claim, candidate["source_region_caption"]
            ),
        })
        seen_samples.add(sample_id)
        seen_candidates.add(candidate_id)
    if set(replacements) != {row["position"] for row in decisions if row["decision"] == "AUTHOR_REPLACEMENT"}:
        raise SpatialReliabilityError("ATOMIC_CLAIM_REPLACEMENT_APPLICATION_INVALID")
    if set(exclusions) != {row["position"] for row in decisions if row["decision"] == "EXCLUDED_NO_VALID_ATOMIC_CLAIM"}:
        raise SpatialReliabilityError("ATOMIC_CLAIM_EXCLUSION_APPLICATION_INVALID")
    isolation = runtime_gt_audit(final_runtime)
    if isolation["status"] != "PASS":
        raise SpatialReliabilityError("ATOMIC_CLAIM_FINAL_RUNTIME_GT_INVALID")
    decision_counts = Counter(row["decision"] for row in decisions)
    split_counts = Counter(row["split"] for row in final_runtime)
    payloads = {
        "reviewed_atomic_claim_runtime_manifest.jsonl": final_runtime,
        "reviewed_atomic_claim_decisions.jsonl": decisions,
        "reviewed_atomic_claim_audit.json": {
            "format": FREEZE_FORMAT + "-audit",
            "status": "PASS",
            "source_candidate_count": len(decisions),
            "final_runtime_count": len(final_runtime),
            "decision_counts": dict(sorted(decision_counts.items())),
            "final_split_counts": dict(sorted(split_counts.items())),
            "replacement_positions": sorted(replacements),
            "excluded_positions": sorted(exclusions),
            "former_frozen_evaluation_is_blind": False,
            "former_frozen_evaluation_use": "CURATED_DIAGNOSTIC_ONLY",
            "runtime_gt_isolation": isolation,
            "model_calls_made": 0,
            "cache_writes": 0,
            "certificate_created": False,
            "new_verified_count": 0,
        },
    }
    artifact_hashes = {
        name: hashlib.sha256(
            (b"".join(canonical_bytes(row) + b"\n" for row in value)
             if name.endswith(".jsonl") else canonical_bytes(value) + b"\n")
        ).hexdigest()
        for name, value in payloads.items()
    }
    manifest = {
        "format": FREEZE_FORMAT,
        "status": "FROZEN",
        "runtime_manifest_sha256": sha256_path(runtime_manifest),
        "candidates_jsonl_sha256": sha256_path(candidates_jsonl),
        "review_jsonl_sha256": sha256_path(review_jsonl),
        "feedback_json_sha256": sha256_path(feedback_json),
        "artifact_sha256": artifact_hashes,
        "source_candidate_count": len(decisions),
        "final_runtime_count": len(final_runtime),
        "decision_counts": dict(sorted(decision_counts.items())),
        "final_split_counts": dict(sorted(split_counts.items())),
        "replacement_positions": sorted(replacements),
        "excluded_positions": sorted(exclusions),
        "former_frozen_evaluation_is_blind": False,
        "model_calls_made": 0,
        "cache_writes": 0,
        "certificate_created": False,
        "new_verified_count": 0,
    }
    output = Path(output_dir)
    for name, value in payloads.items():
        write_once(output / name, value, jsonl=name.endswith(".jsonl"))
    write_once(output / "reviewed_atomic_claim_freeze_manifest.json", manifest)
    markdown = _markdown(decisions).encode("utf-8")
    markdown_path = output / "reviewed_atomic_claims.md"
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    if markdown_path.exists():
        if markdown_path.read_bytes() != markdown:
            raise SpatialReliabilityError(f"IMMUTABLE_ARTIFACT_EXISTS:{markdown_path}")
    else:
        markdown_path.write_bytes(markdown)
    return {**manifest, "output_dir": str(output)}
