"""Frozen, GT-free temporal candidate-pool traversal.

This module deliberately owns only the outer temporal loop.  It reuses the
normal :class:`relive.runner.SampleRunner` for semantic verification, spatial
proposal, intervention, and certificate construction.  It neither proposes a
second region nor changes a candidate after the candidate manifest is frozen.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, TYPE_CHECKING

from PIL import Image
from relive.acquisition import acquire
from relive.audits import audit_strict_result
from relive.claims import PROMPT_VERSIONS
from relive.storage.artifacts import ArtifactError, ArtifactStore, canonical_json, stable_hash
from relive.types import ExecutionStatus, FinalStatus, SemanticStatus, RuntimeSample, to_dict

if TYPE_CHECKING:  # Avoid a runtime import cycle with ``runner``.
    from relive.runner import SampleRunner


TRAVERSAL_MODE = "fixed_sliding_window_pool"
TRAVERSAL_VERSION = "relive-fixed-sliding-window-traversal-v1"


class TraversalError(ValueError):
    """Raised before model inference when a frozen-pool invariant is broken."""


def _temporal_extent(sample: RuntimeSample, candidate) -> tuple[int | float, int | float, str]:
    """Use public timestamps when available, otherwise immutable frame order."""
    selected = {frame.frame_id: frame for frame in sample.frames}
    frames = [selected[frame_id] for frame_id in candidate.frame_ids]
    if all(frame.timestamp is not None for frame in frames):
        return frames[0].timestamp, frames[-1].timestamp, "public_timestamp"
    return frames[0].order, frames[-1].order, "frame_order"


def _candidate_row(sample: RuntimeSample, candidate, config: dict[str, Any]) -> dict[str, Any]:
    start, end, source = _temporal_extent(sample, candidate)
    acquisition = config["acquisition"]
    return {
        "qa_id": sample.sample_id,
        "clip_id": sample.metadata.get("clip_id", sample.sample_id),
        "sample_id": sample.sample_id,
        "dataset_name": sample.metadata.get("dataset_name"),
        "candidate_id": candidate.candidate_id,
        "candidate_rank": candidate.acquisition_rank,
        "frame_ids": list(candidate.frame_ids),
        "source_frame_paths": [frame.path for frame in sample.frames if frame.frame_id in set(candidate.frame_ids)],
        "source_frame_references": list(candidate.provenance.get("frame_references", [])),
        "temporal_start_runtime": start,
        "temporal_end_runtime": end,
        "temporal_coordinate_source": source,
        "window_generator_version": candidate.provenance.get("acquisition_version"),
        "window_strategy": "sliding_windows",
        "window_length": acquisition["window_size"],
        "stride": acquisition["stride"],
        "frame_count": len(candidate.frame_ids),
        "tail_short_window": len(candidate.frame_ids) < acquisition["window_size"],
        "rank_source": candidate.rank_source,
        "source_selector": "public_runtime_frames_in_explicit_order",
        "source_provenance": {
            "runtime_sha256": sample.provenance.get("runtime_sha256"),
            "source_kind": sample.provenance.get("source_kind"),
        },
    }


def build_fixed_candidate_pool(samples: list[RuntimeSample], config: dict[str, Any]) -> tuple[dict[str, list], list[dict[str, Any]], str]:
    """Generate every candidate before inference and bind it to one hash."""
    if config["traversal"]["mode"] != TRAVERSAL_MODE:
        raise TraversalError("fixed candidate pool requires fixed_sliding_window_pool mode")
    if config["acquisition"]["method"] != "sliding_windows":
        raise TraversalError("fixed candidate pool requires acquisition.method=sliding_windows")
    if config["adaptation"]["enabled"] is not False:
        raise TraversalError("fixed candidate pool requires adaptation.enabled=false")
    by_sample: dict[str, list] = {}
    rows: list[dict[str, Any]] = []
    for sample in samples:
        if sample.task != "claim_verification" or sample.target_claim is None:
            raise TraversalError("fixed candidate pool currently supports claim_verification with one frozen target_claim")
        candidates = acquire(sample, config["acquisition"])
        if not candidates:
            raise TraversalError("fixed candidate pool is empty")
        if [candidate.acquisition_rank for candidate in candidates] != list(range(len(candidates))):
            raise TraversalError("candidate ranks are not a contiguous chronological sequence")
        if any(candidate.rank_source != "chronological" for candidate in candidates):
            raise TraversalError("fixed candidate pool requires chronological candidate ordering")
        if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
            raise TraversalError("duplicate candidate IDs in generated candidate pool")
        by_sample[sample.sample_id] = candidates
        rows.extend(_candidate_row(sample, candidate, config) for candidate in candidates)
    pool_hash = stable_hash(rows)
    bound_rows = [{**row, "manifest_hash": pool_hash} for row in rows]
    audit_fixed_candidate_pool(bound_rows, pool_hash)
    return by_sample, bound_rows, pool_hash


def audit_fixed_candidate_pool(rows: list[dict[str, Any]], expected_hash: str | None = None) -> dict[str, Any]:
    """Audit ordering, identity, and the absence of annotation-shaped fields."""
    issues: list[str] = []
    ids: set[str] = set()
    by_sample: dict[str, list[dict[str, Any]]] = {}
    prohibited = {"reference_answer", "assistant_answer", "temporal_gt", "bbox", "mask", "struc_info", "rc_info",
                  "true_support", "spurious_support", "gt_iou", "evaluation_artifact"}
    for row in rows:
        if not isinstance(row, dict):
            issues.append("candidate manifest row is not an object")
            continue
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            issues.append("candidate manifest lacks candidate_id")
        elif candidate_id in ids:
            issues.append("duplicate candidate ID")
        else:
            ids.add(candidate_id)
        if not isinstance(row.get("frame_ids"), list) or not row["frame_ids"]:
            issues.append("candidate manifest lacks a nonempty frame list")
        if any(key.lower() in prohibited for key in row):
            issues.append("candidate manifest contains a prohibited GT-shaped field")
        if expected_hash is not None and row.get("manifest_hash") != expected_hash:
            issues.append("candidate manifest hash binding mismatch")
        by_sample.setdefault(str(row.get("sample_id")), []).append(row)
    for sample_id, sample_rows in by_sample.items():
        ranks = [row.get("candidate_rank") for row in sample_rows]
        starts = [row.get("temporal_start_runtime") for row in sample_rows]
        if ranks != list(range(len(sample_rows))):
            issues.append(f"noncontiguous candidate rank for {sample_id}")
        if starts != sorted(starts):
            issues.append(f"candidate temporal order is not ascending for {sample_id}")
        frames = [tuple(row.get("frame_ids", [])) for row in sample_rows]
        if len(set(frames)) != len(frames):
            issues.append(f"duplicate candidate frame list for {sample_id}")
    return {"status": "PASS" if not issues else "FAIL", "issues": issues,
            "rows": len(rows), "samples": len(by_sample), "manifest_hash": expected_hash}


def audit_candidate_frames(samples: list[RuntimeSample], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Decode every public candidate frame before a Phase 2 model call."""
    issues: list[str] = []
    expected = {frame.frame_id: frame.path for sample in samples for frame in sample.frames}
    referenced = {frame_id for row in rows for frame_id in row.get("frame_ids", [])}
    for frame_id in sorted(referenced):
        path = expected.get(frame_id)
        if path is None:
            issues.append(f"candidate references unknown frame: {frame_id}")
            continue
        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, ValueError):
            issues.append(f"candidate frame is not decodable: {frame_id}")
    return {"status": "PASS" if not issues else "FAIL", "issues": issues,
            "candidate_frame_references": len(referenced), "decoder": "Pillow.verify"}


def write_jsonl_immutable(path: Path, rows: list[dict[str, Any]]) -> None:
    """Atomically create an immutable JSONL artifact, validating a prior copy."""
    body = "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != body:
                raise ArtifactError(f"Immutable artifact collision: {path}") from None
    finally:
        os.unlink(temporary)


def write_fixed_candidate_pool(store: ArtifactStore, rows: list[dict[str, Any]], pool_hash: str) -> dict[str, Any]:
    audit = audit_fixed_candidate_pool(rows, pool_hash)
    if audit["status"] != "PASS":
        raise TraversalError("candidate manifest audit failed")
    path = store.root / "fixed_candidate_pool.jsonl"
    write_jsonl_immutable(path, rows)
    reference = store.put_json("candidate_manifests", "fixed_candidate_pool", {
        "format": "relive-fixed-candidate-pool-v1", "traversal_version": TRAVERSAL_VERSION,
        "manifest_hash": pool_hash, "jsonl_path": str(path), "audit": audit,
    })
    return {"manifest_hash": pool_hash, "jsonl_path": str(path), "artifact_ref": reference, "audit": audit}


def _verification_status(certificate: dict[str, Any], variant: str) -> str | None:
    spatial = certificate.get("checks", {}).get("spatial", {})
    references = spatial.get("references", {}) if isinstance(spatial, dict) else {}
    if variant == "ORIGINAL":
        value = certificate.get("checks", {}).get("semantic", {}).get("status")
    elif variant == "KEEP":
        value = references.get("keep", {}).get("semantic_status")
    elif variant == "DROP":
        value = references.get("drop", {}).get("semantic_status")
    else:
        controls = references.get("controls", [])
        values = [item.get("semantic_status") for item in controls if isinstance(item, dict)]
        value = values[0] if len(values) == 1 else (values if values else None)
    return value


def _pixel_audit_status(store: ArtifactStore, certificate: dict[str, Any]) -> str:
    spatial = certificate.get("checks", {}).get("spatial", {})
    refs = spatial.get("pixel_audit_refs", []) if isinstance(spatial, dict) else []
    if not refs:
        return "NOT_RUN"
    audits = []
    for ref in refs:
        try:
            audit = store.get_json(ref)
        except (ArtifactError, OSError, ValueError):
            return "UNREADABLE"
        audits.append(audit)
    return "PASS" if all(audit.get("pixel_audit_pass") for audit in audits) else "FAIL"


def _technical_invalid(certificate: dict[str, Any]) -> bool:
    semantic = certificate.get("checks", {}).get("semantic", {})
    refs = semantic.get("references", {}) if isinstance(semantic, dict) else {}
    if refs.get("execution_status") not in {None, ExecutionStatus.OK.value}:
        return True
    spatial = certificate.get("checks", {}).get("spatial", {})
    if not isinstance(spatial, dict):
        return False
    return spatial.get("status") in {"SPATIAL_PROPOSAL_FAILURE", "INTERVENTION_PIXEL_AUDIT_FAILED", "SPATIAL_REFERENCE_BINDING_MISMATCH"}


class FixedCandidateTraversalController:
    """Run one immutable temporal candidate at a time; first VERIFIED wins."""

    def __init__(self, engine: "SampleRunner", sample: RuntimeSample, candidates: list,
                 candidate_manifest_hash: str):
        self.engine, self.sample = engine, sample
        self.pool, self.candidate_manifest_hash = list(candidates), candidate_manifest_hash
        self.trace: list[dict[str, Any]] = []

    def _trace_row(self, candidate, certificate, before: dict[str, Any], state: str,
                   accepted: bool, termination_reason: str) -> dict[str, Any]:
        cert = to_dict(certificate)
        spatial = cert.get("checks", {}).get("spatial", {})
        proposal = spatial.get("proposal", {}) if isinstance(spatial, dict) else {}
        after = self.engine.budget.snapshot()
        row = {
            "qa_id": self.sample.sample_id,
            "sample_id": self.sample.sample_id,
            "candidate_id": candidate.candidate_id,
            "candidate_rank": candidate.acquisition_rank,
            "attempt_index": len(self.trace),
            "outer_loop_state": state,
            "spatial_proposal_id": proposal.get("proposal_id") if isinstance(proposal, dict) else None,
            "bbox": proposal.get("support_region") if isinstance(proposal, dict) else None,
            "bbox_area_fraction": spatial.get("support_area_fraction") if isinstance(spatial, dict) else None,
            "original_semantic_status": _verification_status(cert, "ORIGINAL"),
            "keep_semantic_status": _verification_status(cert, "KEEP"),
            "drop_semantic_status": _verification_status(cert, "DROP"),
            "control_semantic_status": _verification_status(cert, "CONTROL"),
            "pixel_audit_status": _pixel_audit_status(self.engine.store, cert),
            "control_availability": spatial.get("control_available") if isinstance(spatial, dict) else None,
            "certificate_id": cert.get("certificate_id"),
            "certificate_status": cert.get("final_status"),
            "failure_reasons": list(cert.get("failure_reasons", [])),
            "model_call_count_increment": after["new_calls"] - before["new_calls"],
            "cache_hits_increment": after["cache_hits"] - before["cache_hits"],
            "logical_model_call_count_increment": after["calls"] - before["calls"],
            "accepted": accepted,
            "termination_reason": termination_reason,
            "provenance": {"candidate_manifest_hash": self.candidate_manifest_hash,
                           "prompt_versions": PROMPT_VERSIONS,
                           "operator": self.engine.intervention,
                           "model_fingerprint": self.engine.inference.backend.fingerprint(),
                           "traversal_version": TRAVERSAL_VERSION},
        }
        return row

    def run(self) -> dict[str, Any]:
        if self.sample.task != "claim_verification" or self.sample.target_claim is None:
            raise TraversalError("fixed traversal requires claim_verification with a frozen target_claim")
        self.engine.claims = [self.sample.target_claim]
        attempted = self.pool[:self.engine.cfg["budget"]["max_candidates"]]
        if not attempted:
            raise TraversalError("fixed traversal budget permits no candidates")
        accepted_certificate = None
        terminal = "CANDIDATE_POOL_EXHAUSTED"
        for candidate in attempted:
            if candidate.candidate_id in self.engine.seen_candidates:
                raise TraversalError("candidate visited twice")
            self.engine.seen_candidates.add(candidate.candidate_id)
            self.engine.candidates.append(candidate)
            self.engine.record("candidates", candidate)
            before = self.engine.budget.snapshot()
            try:
                certificate = self.engine.pair(self.sample, candidate, self.sample.target_claim, 0)
                self.engine.certificates.append(certificate)
            except Exception as exc:
                # ``BudgetExceeded`` is deliberately not translated into a
                # semantic state.  No recovery/retry policy is added here.
                from relive.storage.cache import BudgetExceeded
                if not isinstance(exc, BudgetExceeded):
                    raise
                terminal = "CALL_BUDGET_EXHAUSTED"
                break
            cert = to_dict(certificate)
            if certificate.final_status == FinalStatus.VERIFIED:
                state, accepted, terminal = "VERIFIED_FOUND", True, "VERIFIED_FOUND"
                accepted_certificate = certificate
            elif _technical_invalid(cert):
                state, accepted, terminal = "TECHNICAL_INVALID", False, "TECHNICAL_INVALID"
            elif _verification_status(cert, "ORIGINAL") == SemanticStatus.INSUFFICIENT.value:
                state, accepted = "ORIGINAL_INSUFFICIENT", False
            elif certificate.final_status == FinalStatus.REJECTED:
                state, accepted = "CANDIDATE_REJECTED", False
            else:
                state, accepted = "CANDIDATE_UNCERTAIN", False
            row = self._trace_row(candidate, certificate, before, state, accepted, terminal if accepted or state == "TECHNICAL_INVALID" else state)
            self.trace.append(row)
            self.engine.record("candidate_traversal", row)
            if accepted or state == "TECHNICAL_INVALID":
                break
        else:
            if len(attempted) < len(self.pool):
                terminal = "MAX_CANDIDATES"
            else:
                terminal = "CANDIDATE_POOL_EXHAUSTED"
        coverage, strict = self.engine.coverage_and_answer(self.sample)
        if strict["status"] == "ANSWERED" and terminal != "VERIFIED_FOUND":
            raise TraversalError("strict answer admitted without a fixed-traversal VERIFIED candidate")
        return {
            "coverage": coverage, "strict": strict, "termination_reason": terminal,
            "accepted_certificate_id": accepted_certificate.certificate_id if accepted_certificate else None,
            "accepted_candidate_id": accepted_certificate.candidate_id if accepted_certificate else None,
            "candidates_total": len(self.pool), "candidates_examined": len(self.trace),
            "candidate_manifest_hash": self.candidate_manifest_hash,
            "trace": self.trace,
        }


def audit_traversal_results(results: list[dict[str, Any]], pool_rows: list[dict[str, Any]], pool_hash: str) -> dict[str, Any]:
    """Check actual rank order, early exit, and strict evidence binding."""
    issues: list[str] = []
    rows_by_sample: dict[str, list[dict[str, Any]]] = {}
    for row in pool_rows:
        rows_by_sample.setdefault(row["sample_id"], []).append(row)
    for result in results:
        traversal = result.get("candidate_traversal", {})
        trace = traversal.get("trace", [])
        sample_id = result.get("sample_id")
        ranks = [row.get("candidate_rank") for row in trace]
        ids = [row.get("candidate_id") for row in trace]
        if ranks != list(range(len(trace))):
            issues.append(f"traversal order mismatch for {sample_id}")
        if len(set(ids)) != len(ids):
            issues.append(f"candidate revisited for {sample_id}")
        if any(row.get("provenance", {}).get("candidate_manifest_hash") != pool_hash for row in trace):
            issues.append(f"trace manifest binding mismatch for {sample_id}")
        verified = [index for index, row in enumerate(trace) if row.get("accepted")]
        if len(verified) > 1 or (verified and verified[0] != len(trace) - 1):
            issues.append(f"early exit violated for {sample_id}")
        if traversal.get("termination_reason") == "VERIFIED_FOUND" and not verified:
            issues.append(f"missing accepted VERIFIED candidate for {sample_id}")
        if traversal.get("termination_reason") == "CANDIDATE_POOL_EXHAUSTED" and len(trace) != len(rows_by_sample.get(sample_id, [])):
            issues.append(f"pool exhaustion without visiting every candidate for {sample_id}")
        strict_audit = audit_strict_result(result)
        if strict_audit["status"] != "PASS":
            issues.append(f"strict evidence audit failed for {sample_id}")
    return {"status": "PASS" if not issues else "FAIL", "issues": issues,
            "candidate_manifest_hash": pool_hash, "samples_checked": len(results),
            "traversal_version": TRAVERSAL_VERSION}


def failure_reason_distribution(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate only runtime-generated certificate statuses and reasons."""
    by_reason: dict[str, int] = {}
    by_rank: dict[str, dict[str, int]] = {}
    by_qa: dict[str, dict[str, int]] = {}
    by_qa_type: dict[str, dict[str, int]] = {}
    for result in results:
        qa_id = result["sample_id"]
        qa_type = result.get("provenance", {}).get("source_qa_type", result.get("task"))
        for row in result.get("candidate_traversal", {}).get("trace", []):
            labels = [f"CERTIFICATE_{row.get('certificate_status')}", *row.get("failure_reasons", [])]
            for label in labels:
                by_reason[label] = by_reason.get(label, 0) + 1
                for target, key in ((by_rank, str(row.get("candidate_rank"))), (by_qa, qa_id), (by_qa_type, str(qa_type))):
                    target.setdefault(key, {})[label] = target.setdefault(key, {}).get(label, 0) + 1
    return {"format": "relive-phase2-failure-reason-distribution-v1", "by_reason": dict(sorted(by_reason.items())),
            "by_candidate_rank": by_rank, "by_qa": by_qa, "by_qa_type": by_qa_type}


def failure_reason_markdown(distribution: dict[str, Any]) -> str:
    lines = ["# Phase 2 fixed sliding-window traversal failure distribution", "",
             "This is a GT-free engineering-smoke diagnostic. It does not establish benchmark performance.", "",
             "| Runtime certificate state or failure reason | Attempts |", "| --- | ---: |"]
    lines.extend(f"| `{label}` | {count} |" for label, count in distribution["by_reason"].items())
    return "\n".join(lines) + "\n"
