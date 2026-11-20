"""Versioned RC-caption to atomic-claim candidate generation.

This is a review-preparation stage, not a runtime or certificate stage.  It
binds the existing 32-row public runtime cohort back to canonical source
records, gives the model only the source region caption, and writes a blank
human-review list.  Generated text is never applied to a runtime manifest by
this module.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from relive.backends import make_backend
from relive.config import load_config
from relive.storage.artifacts import ArtifactStore
from relive.storage.cache import Budget, CachedInference

from .common import (SpatialReliabilityError, runtime_gt_audit,
                     sha256_path, stable_hash, strict_json_load, write_once)
from .dataset import _caption


ATOMIC_CLAIM_FORMAT = "relive-spatial-atomic-claim-candidates-v1"
ATOMIC_CLAIM_PROMPT_VERSION = "relive-spatial-rc-to-atomic-claim-v1"
ATOMIC_CLAIM_PARSER_VERSION = "relive-spatial-atomic-claim-parser-v1"
ALLOWED_REVIEW_DECISIONS = [
    "ACCEPT_CANDIDATE",
    "CONFIRM_EXCLUDED",
    "REJECT_CANDIDATE",
    "AUTHOR_EDIT_REQUIRED",
]

ATOMIC_CLAIM_INSTRUCTION = """Given the source region caption below, generate one minimal single-frame spatial claim suitable for visual evidence verification.

The claim must describe exactly one directly visible local fact. Prefer object presence or one simple visible state, for example: A knife is visible. Or: The forceps jaws are open.

Remove temporal actions, motion, procedural steps, intention, purpose, causal interpretation, surgical phase, global scene information, and unnecessary relations. Do not add any object, attribute, location, or medical information that is not explicitly supported by the source caption. Prefer the simplest valid claim. If no local single-frame visual claim can be produced without inventing information, return EXCLUDED.

Output contract:
- Return exactly one plain-text claim ending in a period, or exactly EXCLUDED.
- Do not return JSON, Markdown, quotation marks, labels, bullets, or explanation.
- Do not mention the caption, question, answer, frame, sequence, or video.

Source region caption:
"""

_TEMPORAL_OR_PROCEDURAL = re.compile(
    r"\b(?:across|throughout|during|before|after|while|then|sequence|frames?|video|"
    r"procedure|phase|advances?|moves?|moving|perform(?:s|ed|ing)?|used?\s+to|"
    r"guided|maintain(?:s|ed|ing)?|adjust(?:s|ed|ing)?|separat(?:es|ed|ing)|"
    r"dissect(?:s|ed|ing)?|retract(?:s|ed|ing)?|insert(?:s|ed|ing)?|"
    r"remove(?:s|d|ing)?|caus(?:es|ed|ing))\b",
    re.IGNORECASE,
)
_PURPOSE_OR_CAUSAL = re.compile(
    r"\b(?:in order to|so that|for optimal|for access|therefore|because|resulting in)\b",
    re.IGNORECASE,
)
_GLOBAL_SCENE = re.compile(
    r"\b(?:overall|entire scene|surgical field|procedure|phase)\b", re.IGNORECASE
)
_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "on", "at", "for", "with", "and", "or", "by",
    "this", "that", "these", "those", "it", "its", "visible",
})


def atomic_claim_prompt(caption: str) -> str:
    if not isinstance(caption, str) or not caption.strip():
        raise SpatialReliabilityError("ATOMIC_CLAIM_CAPTION_INVALID")
    return ATOMIC_CLAIM_INSTRUCTION + " ".join(caption.split())


def parse_atomic_claim(raw: Any) -> dict[str, Any]:
    """Parse the deliberately minimal, non-JSON generation surface."""
    if not isinstance(raw, str) or not raw:
        return {"status": "PARSE_FAILURE", "claim_text": None, "reason": "EMPTY_OUTPUT"}
    text = raw.strip()
    if text == "EXCLUDED":
        return {"status": "EXCLUDED", "claim_text": None,
                "reason": "MODEL_FOUND_NO_NONINVENTED_LOCAL_FACT"}
    if (text != raw or "\n" in text or "\r" in text or "`" in text
            or text.startswith(("\"", "'", "-", "*", "{"))
            or text.endswith(("\"", "'", "}"))
            or re.match(r"^(?:claim|answer|output)\s*:", text, flags=re.IGNORECASE)):
        return {"status": "PARSE_FAILURE", "claim_text": None,
                "reason": "NONCANONICAL_PLAIN_TEXT_CONTRACT"}
    if len(text) > 200 or not text.endswith(".") or text.count(".") != 1 or "?" in text or "!" in text:
        return {"status": "PARSE_FAILURE", "claim_text": None,
                "reason": "SINGLE_SENTENCE_CONTRACT_VIOLATION"}
    words = _TOKEN.findall(text)
    if not 3 <= len(words) <= 24:
        return {"status": "PARSE_FAILURE", "claim_text": None,
                "reason": "CLAIM_LENGTH_CONTRACT_VIOLATION"}
    return {"status": "PROPOSED", "claim_text": text, "reason": "REVIEW_REQUIRED"}


def atomic_claim_static_checks(claim: str | None, caption: str) -> dict[str, Any]:
    if claim is None:
        return {
            "status": "NOT_APPLICABLE",
            "single_sentence": None,
            "temporal_or_procedural_term_hits": [],
            "purpose_or_causal_term_hits": [],
            "global_scene_term_hits": [],
            "source_content_token_overlap": [],
        }
    temporal = sorted({match.group(0).casefold() for match in _TEMPORAL_OR_PROCEDURAL.finditer(claim)})
    purpose = sorted({match.group(0).casefold() for match in _PURPOSE_OR_CAUSAL.finditer(claim)})
    global_hits = sorted({match.group(0).casefold() for match in _GLOBAL_SCENE.finditer(claim)})
    source_tokens = {token.casefold() for token in _TOKEN.findall(caption)} - _STOPWORDS
    claim_tokens = {token.casefold() for token in _TOKEN.findall(claim)} - _STOPWORDS
    overlap = sorted(source_tokens & claim_tokens)
    passed = not temporal and not purpose and not global_hits and bool(overlap)
    return {
        "status": "PASS" if passed else "FLAGGED_FOR_HUMAN_REVIEW",
        "single_sentence": claim.endswith(".") and claim.count(".") == 1,
        "temporal_or_procedural_term_hits": temporal,
        "purpose_or_causal_term_hits": purpose,
        "global_scene_term_hits": global_hits,
        "source_content_token_overlap": overlap,
    }


def _strict_jsonl(path: str | Path) -> list[dict[str, Any]]:
    def reject_constant(value: str) -> None:
        raise ValueError(value)

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(key)
            result[key] = value
        return result

    try:
        rows = [
            json.loads(line, parse_constant=reject_constant, object_pairs_hook=reject_duplicates)
            for line in Path(path).read_bytes().decode("utf-8").splitlines() if line.strip()
        ]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise SpatialReliabilityError("ATOMIC_CLAIM_RUNTIME_MANIFEST_INVALID") from exc
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise SpatialReliabilityError("ATOMIC_CLAIM_RUNTIME_MANIFEST_INVALID")
    if runtime_gt_audit(rows)["status"] != "PASS":
        raise SpatialReliabilityError("ATOMIC_CLAIM_RUNTIME_GT_ISOLATION_FAILED")
    return rows


def _bound_inputs(runtime_manifest: str | Path, source_json: str | Path,
                  expected_count: int) -> list[dict[str, Any]]:
    runtime = _strict_jsonl(runtime_manifest)
    source = strict_json_load(source_json)
    if not isinstance(source, list):
        raise SpatialReliabilityError("ATOMIC_CLAIM_SOURCE_INVALID")
    if len(runtime) != expected_count:
        raise SpatialReliabilityError(
            f"ATOMIC_CLAIM_COHORT_COUNT_MISMATCH:{len(runtime)}:{expected_count}"
        )
    seen_samples: set[str] = set()
    seen_indexes: set[int] = set()
    bound: list[dict[str, Any]] = []
    for row in runtime:
        sample_id = row.get("sample_id")
        index = row.get("source_record_index")
        if (not isinstance(sample_id, str) or not sample_id or sample_id in seen_samples
                or type(index) is not int or not 0 <= index < len(source) or index in seen_indexes):
            raise SpatialReliabilityError("ATOMIC_CLAIM_SOURCE_BINDING_INVALID")
        record = source[index]
        if (not isinstance(record, dict)
                or stable_hash(record) != row.get("source_record_sha256")
                or not str(record.get("qa_type", "")).startswith("region_caption")):
            raise SpatialReliabilityError("ATOMIC_CLAIM_SOURCE_BINDING_INVALID")
        caption = _caption(record)
        seen_samples.add(sample_id)
        seen_indexes.add(index)
        bound.append({"runtime": row, "caption": caption})
    return bound


def _write_text_once(path: Path, text: str) -> None:
    body = text.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != body:
            raise SpatialReliabilityError(f"IMMUTABLE_ARTIFACT_EXISTS:{path}")
        return
    descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def _review_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Atomic claim human review list",
        "",
        "Every row is review-only. `PASS` below means only that deterministic lexical checks did not flag the text; it is not human approval and does not admit the claim to runtime.",
        "",
    ]
    for index, row in enumerate(rows, start=1):
        checks = row["static_checks"]
        hits = sorted({
            *checks["temporal_or_procedural_term_hits"],
            *checks["purpose_or_causal_term_hits"],
            *checks["global_scene_term_hits"],
        })
        lines.extend([
            f"## {index:02d}. `{row['sample_id']}`",
            "",
            f"- Split: `{row['split']}`",
            f"- Dataset / video: `{row['dataset_name']}` / `{row['video_id']}`",
            f"- Source record index: `{row['source_record_index']}`",
            f"- RC Caption: {row['source_region_caption']}",
            f"- Previous claim: {row['current_claim_text']}",
            f"- Generated status: `{row['generated_status']}`",
            f"- Generated atomic claim: {row['generated_atomic_claim'] or 'EXCLUDED / unavailable'}",
            f"- Static check: `{checks['status']}`",
            f"- Flagged terms: `{', '.join(hits) if hits else 'none'}`",
            "- Human decision: **PENDING**",
            "",
        ])
    return "\n".join(lines)


def generate_atomic_claim_candidates(*, runtime_manifest: str | Path,
                                     source_json: str | Path,
                                     config_path: str | Path,
                                     cache_dir: str | Path,
                                     output_dir: str | Path,
                                     expected_count: int = 32,
                                     execution_mode: str = "run") -> dict[str, Any]:
    if execution_mode not in {"run", "replay"}:
        raise SpatialReliabilityError("ATOMIC_CLAIM_EXECUTION_MODE_INVALID")
    if type(expected_count) is not int or expected_count < 1:
        raise SpatialReliabilityError("ATOMIC_CLAIM_EXPECTED_COUNT_INVALID")
    bound = _bound_inputs(runtime_manifest, source_json, expected_count)
    config = load_config(config_path)
    backend = make_backend(config["backend"])
    inference = CachedInference(backend, ArtifactStore(cache_dir))
    budget = Budget(expected_count)
    contract_identity = {
        "format": ATOMIC_CLAIM_FORMAT,
        "prompt_version": ATOMIC_CLAIM_PROMPT_VERSION,
        "parser_version": ATOMIC_CLAIM_PARSER_VERSION,
        "output_contract": "EXACT_EXCLUDED_OR_ONE_PLAIN_TEXT_SENTENCE",
        "human_review_required": True,
        "automatic_runtime_admission": False,
    }
    results: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    for item in bound:
        row, caption = item["runtime"], item["caption"]
        prompt = atomic_claim_prompt(caption)
        request = {
            "stage": "relive_spatial_atomic_claim_generation",
            "prompt": prompt,
            "prompt_version": ATOMIC_CLAIM_PROMPT_VERSION,
            "image_paths": [],
            "frame_ids": [],
            "atomic_claim_contract_identity": {
                **contract_identity,
                "sample_id": row["sample_id"],
                "source_record_sha256": row["source_record_sha256"],
                "source_caption_sha256": hashlib.sha256(caption.encode("utf-8")).hexdigest(),
            },
        }
        if execution_mode == "replay" and not inference.has_terminal_result(request):
            raise SpatialReliabilityError(f"ATOMIC_CLAIM_REPLAY_CACHE_MISS:{row['sample_id']}")
        call = inference.call(request, budget)
        if execution_mode == "replay" and not call["cache_hit"]:
            raise SpatialReliabilityError(f"ATOMIC_CLAIM_REPLAY_CACHE_MISS:{row['sample_id']}")
        parsed = (parse_atomic_claim(call["raw_text"])
                  if call["execution_status"] == "OK"
                  else {"status": "INFERENCE_ERROR", "claim_text": None,
                        "reason": call["failure_reason"]})
        checks = atomic_claim_static_checks(parsed["claim_text"], caption)
        candidate_id = "atomic-claim-candidate-" + stable_hash({
            "sample_id": row["sample_id"],
            "source_record_sha256": row["source_record_sha256"],
            "prompt_version": ATOMIC_CLAIM_PROMPT_VERSION,
            "cache_key": call["cache_key"],
            "parsed": parsed,
        })[:24]
        result = {
            "format": ATOMIC_CLAIM_FORMAT,
            "candidate_id": candidate_id,
            "sample_id": row["sample_id"],
            "split": row.get("split"),
            "dataset_name": row.get("dataset_name"),
            "video_id": row.get("video_id"),
            "source_record_index": row["source_record_index"],
            "source_record_sha256": row["source_record_sha256"],
            "current_claim_id": row.get("claim_id"),
            "current_claim_text": row.get("claim_text"),
            "source_region_caption": caption,
            "generation": {
                "execution_status": call["execution_status"],
                "failure_reason": call["failure_reason"],
                "cache_hit": call["cache_hit"],
                "cache_key": call["cache_key"],
                "raw_response_ref": call["raw_response_ref"],
            },
            "candidate": parsed,
            "static_checks": checks,
            "review_status": "HUMAN_REVIEW_REQUIRED",
            "runtime_manifest_modified": False,
            "gt_used": False,
            "oracle_manifest_opened": False,
            "certificate_created": False,
            "new_verified_count": 0,
        }
        results.append(result)
        reviews.append({
            "format": ATOMIC_CLAIM_FORMAT + "-human-review",
            "candidate_id": candidate_id,
            "sample_id": row["sample_id"],
            "split": row.get("split"),
            "dataset_name": row.get("dataset_name"),
            "video_id": row.get("video_id"),
            "source_record_index": row["source_record_index"],
            "source_region_caption": caption,
            "current_claim_text": row.get("claim_text"),
            "generated_status": parsed["status"],
            "generated_atomic_claim": parsed["claim_text"],
            "generation_reason": parsed["reason"],
            "static_checks": checks,
            "allowed_decisions": ALLOWED_REVIEW_DECISIONS,
            "allowed_checklist_values": ["YES", "NO", "NOT_APPLICABLE"],
            "review_checklist": {
                "one_directly_visible_local_fact": "",
                "single_frame_judgable": "",
                "no_added_object_attribute_location_or_medical_fact": "",
                "no_temporal_action_purpose_cause_phase_or_global_scene": "",
            },
            "human_review": {
                "decision": "",
                "approved_claim": "",
                "reviewer_id": "",
                "rationale": "",
            },
        })
    usage = budget.snapshot()
    counts = Counter(result["candidate"]["status"] for result in results)
    static_counts = Counter(result["static_checks"]["status"] for result in results)
    summary = {
        "format": ATOMIC_CLAIM_FORMAT,
        "status": "PASS",
        "execution_mode": execution_mode,
        "candidate_count": len(results),
        "generation_status_counts": dict(sorted(counts.items())),
        "static_check_status_counts": dict(sorted(static_counts.items())),
        "human_review_required_count": len(reviews),
        "runtime_manifest_modified": False,
        "automatic_runtime_admission": False,
        "usage": usage,
        "model_calls_made": usage["new_calls"],
        "cache_hits": usage["cache_hits"],
        "gt_used": False,
        "oracle_manifest_opened": False,
        "certificate_created": False,
        "new_verified_count": 0,
    }
    audit = {
        "format": ATOMIC_CLAIM_FORMAT + "-audit",
        "status": "PASS",
        "runtime_manifest_sha256": sha256_path(runtime_manifest),
        "source_json_sha256": sha256_path(source_json),
        "config_sha256": sha256_path(config_path),
        "prompt_sha256": hashlib.sha256(ATOMIC_CLAIM_INSTRUCTION.encode("utf-8")).hexdigest(),
        "prompt_version": ATOMIC_CLAIM_PROMPT_VERSION,
        "parser_version": ATOMIC_CLAIM_PARSER_VERSION,
        "backend_fingerprint": backend.fingerprint(),
        "source_records_checked": len(bound),
        "source_binding_pass": True,
        "prompt_inputs": ["source_region_caption"],
        "forbidden_prompt_inputs_not_opened": [
            "oracle manifest", "RC bbox", "GT", "certificate", "runtime result",
            "human review decision", "prior model classification",
        ],
        "runtime_gt_isolation": runtime_gt_audit([item["runtime"] for item in bound]),
        "model_calls_made": usage["new_calls"],
        "cache_writes": usage["new_calls"],
        "certificate_created": False,
        "new_verified_count": 0,
    }
    output = Path(output_dir)
    write_once(output / "atomic_claim_candidates.jsonl", results, jsonl=True)
    write_once(output / "atomic_claim_human_review_list.jsonl", reviews, jsonl=True)
    write_once(output / "atomic_claim_generation_summary.json", summary)
    write_once(output / "atomic_claim_generation_audit.json", audit)
    _write_text_once(output / "atomic_claim_human_review_list.md", _review_markdown(reviews))
    readme = (
        "# Atomic claim candidate review\n\n"
        "This directory contains model-generated review candidates derived only from the "
        "bound source RC captions. It does not modify the existing runtime manifest, open "
        "spatial GT/oracle artifacts, create certificates, or admit any candidate.\n\n"
        "Read `atomic_claim_human_review_list.md`, then fill the authoritative "
        "`atomic_claim_human_review_list.jsonl`. Every row requires a human decision and "
        "the four checklist fields. For ACCEPT_CANDIDATE all four checks must be YES and "
        "approved_claim must equal generated_atomic_claim. For CONFIRM_EXCLUDED the four "
        "checks are NOT_APPLICABLE and approved_claim remains empty. REJECT_CANDIDATE "
        "keeps approved_claim empty; AUTHOR_EDIT_REQUIRED records the reviewed replacement "
        "in approved_claim. Do not start a replacement experiment until a "
        "separate validation/apply stage has frozen the completed review.\n"
    )
    _write_text_once(output / "README.md", readme)
    return {**summary, "output_dir": str(output)}
