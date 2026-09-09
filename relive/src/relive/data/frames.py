"""Frame selection with original order and explicit timestamp provenance."""
from __future__ import annotations

from PIL import Image

from relive.types import EvidenceCandidate, Frame, RuntimeSample


def select_frames(sample: RuntimeSample, candidate: EvidenceCandidate) -> tuple[Frame, ...]:
    if candidate.sample_id != sample.sample_id:
        raise ValueError("candidate belongs to a different sample")
    lookup = {f.frame_id: f for f in sample.frames}
    if not candidate.frame_ids or len(set(candidate.frame_ids)) != len(candidate.frame_ids) or any(fid not in lookup for fid in candidate.frame_ids):
        raise ValueError("candidate requires nonempty, unique and known frame IDs")
    frames = tuple(lookup[fid] for fid in candidate.frame_ids)
    if [f.order for f in frames] != sorted(f.order for f in frames):
        raise ValueError("candidate changes original frame order")
    if candidate.timestamps != tuple(f.timestamp for f in frames):
        raise ValueError("candidate timestamps disagree with the source frames")
    return frames


def open_frame(frame: Frame) -> Image.Image:
    with Image.open(frame.path) as image:
        return image.convert("RGB").copy()


def load_frames(candidate: EvidenceCandidate, sample: RuntimeSample) -> list[Image.Image]:
    return [open_frame(frame) for frame in select_frames(sample, candidate)]


def timing_summary(frames: tuple[Frame, ...]) -> dict:
    complete = bool(frames) and all(f.timestamp is not None and f.timestamp_source for f in frames)
    return {"original_order_preserved": True, "seconds_available": complete,
            "timestamps": [f.timestamp for f in frames],
            "timestamp_sources": [f.timestamp_source for f in frames],
            "limitation": None if complete else "No second-level localization or inferred temporal relation from absent timestamps."}
