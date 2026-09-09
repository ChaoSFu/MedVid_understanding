"""Deterministic GT-independent acquisition; order is not relevance."""
from __future__ import annotations

import hashlib
import json
from typing import Protocol

from relive.data.frames import select_frames
from relive.types import EvidenceCandidate, Frame, RuntimeSample


class ExternalRetriever(Protocol):
    """A retriever may rank relevance, but cannot grant verification admission."""

    def retrieve(self, sample: RuntimeSample, max_candidates: int) -> list[EvidenceCandidate]: ...


def _positive(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _candidate(sample: RuntimeSample, frames: tuple[Frame, ...], rank: int, source: str, parameters: dict, parent: EvidenceCandidate | None = None) -> EvidenceCandidate:
    identity = {"sample_id": sample.sample_id, "frame_ids": [f.frame_id for f in frames], "source": source, "parameters": parameters, "parent_id": parent.candidate_id if parent else None}
    cid = "ev_" + hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:20]
    return EvidenceCandidate(cid, sample.sample_id, tuple(f.frame_id for f in frames), tuple(f.timestamp for f in frames), rank, source,
        parent_id=parent.candidate_id if parent else None, round_index=parent.round_index + 1 if parent else 0,
        provenance={"acquisition_version": "relive-acquisition-v1", "parameters": parameters, "source_runtime_sha256": sample.provenance.get("runtime_sha256"), "frame_references": [f.source_reference for f in frames], "semantic_relevance_scored": False})


def acquire(sample: RuntimeSample, config: dict) -> list[EvidenceCandidate]:
    cfg = config.get("acquisition", config)
    strategy = cfg.get("method", cfg.get("strategy", "sliding_windows"))
    size = _positive(cfg.get("window_size", 3), "window_size")
    stride = _positive(cfg.get("stride", 2), "stride")
    maximum = _positive(cfg.get("max_candidates", 4), "max_candidates")
    if "budget" in config and "max_candidates" in config["budget"]:
        maximum = min(maximum, _positive(config["budget"]["max_candidates"], "budget.max_candidates"))
    frames = sample.frames
    if not frames:
        raise ValueError("acquisition requires frames")
    params = {"strategy": strategy, "window_size": size, "stride": stride, "max_candidates": maximum}
    if strategy == "uniform":
        count = min(size, len(frames))
        indexes = [0] if count == 1 else [(i * (len(frames) - 1)) // (count - 1) for i in range(count)]
        return [_candidate(sample, tuple(frames[i] for i in indexes), 0, "uniform", params)]
    if strategy not in {"sliding_windows", "sliding_window"}:
        raise ValueError(f"unsupported acquisition strategy: {strategy}")
    windows = []
    seen = set()
    for start in range(0, len(frames), stride):
        window = frames[start:start + size]
        key = tuple(f.frame_id for f in window)
        if key in seen:
            continue
        seen.add(key)
        windows.append(_candidate(sample, window, len(windows), "chronological", params))
        if len(windows) == maximum or start + size >= len(frames):
            break
    return windows


def expand_candidate(sample: RuntimeSample, candidate: EvidenceCandidate, extra_frames: int) -> EvidenceCandidate | None:
    _positive(extra_frames, "extra_frames")
    selected = select_frames(sample, candidate)
    indexes = {f.frame_id: i for i, f in enumerate(sample.frames)}
    start, end = indexes[selected[0].frame_id], indexes[selected[-1].frame_id]
    expanded = sample.frames[max(0, start - extra_frames):min(len(sample.frames), end + extra_frames + 1)]
    if tuple(f.frame_id for f in expanded) == candidate.frame_ids:
        return None
    return _candidate(sample, expanded, candidate.acquisition_rank, "chronological", {"action": "EXPAND_TEMPORAL_CONTEXT", "extra_frames_each_side": extra_frames}, candidate)
