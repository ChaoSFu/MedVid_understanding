from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any, Iterable


@dataclass(frozen=True)
class GTSpan:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, float]:
        return {"start": self.start, "end": self.end}


@dataclass(frozen=True)
class FrameObservation:
    frame_position: int
    source_frame_index: int
    frame_path: str
    local_time: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_position": self.frame_position,
            "source_frame_index": self.source_frame_index,
            "frame_path": self.frame_path,
            "local_time": self.local_time,
        }


@dataclass(frozen=True)
class TemporalCell:
    source_frame_index: int
    local_time: float
    cell_start: float
    cell_end: float
    frame_positions: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_frame_index": self.source_frame_index,
            "local_time": self.local_time,
            "cell_start": self.cell_start,
            "cell_end": self.cell_end,
            "frame_positions": list(self.frame_positions),
        }


@dataclass(frozen=True)
class TemporalValidation:
    temporal_status: str
    spans: tuple[GTSpan, ...]
    gt_boundary_clipped: bool
    excluded_reason: str | None
    tolerance: float


@dataclass(frozen=True)
class AlignmentMetrics:
    n_gt_visible_in_window: int
    n_gt_visible_total: int
    n_unique_frames_window: int
    evidence_density: float | None
    gt_evidence_recall: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_gt_visible_in_window": self.n_gt_visible_in_window,
            "n_gt_visible_total": self.n_gt_visible_total,
            "n_unique_frames_window": self.n_unique_frames_window,
            "evidence_density": self.evidence_density,
            "gt_evidence_recall": self.gt_evidence_recall,
        }


def infer_clip_duration(metadata: dict[str, Any], n_frames: int) -> float:
    fps = float(metadata["fps"])
    start_time = metadata.get("input_video_start_time")
    end_time = metadata.get("input_video_end_time")
    if start_time is not None and end_time is not None:
        return float(end_time) - float(start_time)
    return (n_frames - 1) / fps if n_frames > 0 else 0.0


def map_frames_mapping_c(
    sampled_video_frames: list[int],
    frame_paths: list[str],
    metadata: dict[str, Any],
) -> tuple[float, list[FrameObservation]]:
    """Endpoint-anchored frame-index interpolation.

    GT remains in clip-local coordinates. `input_video_start_time` is used only
    to infer clip duration when both endpoint times are available.
    """
    if len(sampled_video_frames) != len(frame_paths):
        raise ValueError("len(sampled_video_frames) must equal len(frame_paths)")
    if not sampled_video_frames:
        raise ValueError("sampled_video_frames is empty")

    fps = float(metadata["fps"])
    clip_duration = infer_clip_duration(metadata, len(sampled_video_frames))
    first = sampled_video_frames[0]
    last = sampled_video_frames[-1]

    observations: list[FrameObservation] = []
    for i, frame_index in enumerate(sampled_video_frames):
        if last > first:
            local_time = (frame_index - first) / (last - first) * clip_duration
        else:
            local_time = i / fps
        observations.append(
            FrameObservation(
                frame_position=i,
                source_frame_index=int(frame_index),
                frame_path=frame_paths[i],
                local_time=float(local_time),
            )
        )
    return float(clip_duration), observations


def build_temporal_cells(
    observations: Iterable[FrameObservation],
    clip_duration: float,
) -> list[TemporalCell]:
    grouped: dict[int, dict[str, Any]] = {}
    for obs in observations:
        bucket = grouped.setdefault(
            obs.source_frame_index,
            {"time": obs.local_time, "positions": []},
        )
        bucket["positions"].append(obs.frame_position)
        if obs.local_time < bucket["time"]:
            bucket["time"] = obs.local_time

    unique = sorted(
        (
            int(frame_index),
            float(value["time"]),
            tuple(sorted(value["positions"])),
        )
        for frame_index, value in grouped.items()
    )
    unique.sort(key=lambda x: (x[1], x[0]))

    if not unique:
        return []
    if len(unique) == 1:
        frame_index, t, positions = unique[0]
        return [TemporalCell(frame_index, t, 0.0, clip_duration, positions)]

    cells: list[TemporalCell] = []
    times = [x[1] for x in unique]
    for i, (frame_index, t, positions) in enumerate(unique):
        if i == 0:
            start = 0.0
            end = (times[0] + times[1]) / 2.0
        elif i == len(unique) - 1:
            start = (times[-2] + times[-1]) / 2.0
            end = clip_duration
        else:
            start = (times[i - 1] + times[i]) / 2.0
            end = (times[i] + times[i + 1]) / 2.0
        cells.append(TemporalCell(frame_index, t, start, end, positions))
    return cells


def intervals_intersect_inclusive(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return max(a_start, b_start) <= min(a_end, b_end)


def cell_intersects_span(cell: TemporalCell, span: GTSpan) -> bool:
    return intervals_intersect_inclusive(cell.cell_start, cell.cell_end, span.start, span.end)


def visible_cells_for_spans(cells: Iterable[TemporalCell], spans: Iterable[GTSpan]) -> set[int]:
    visible: set[int] = set()
    span_list = list(spans)
    for cell in cells:
        if any(cell_intersects_span(cell, span) for span in span_list):
            visible.add(cell.source_frame_index)
    return visible


def validate_and_clip_gt_spans(
    spans: Iterable[GTSpan],
    clip_duration: float,
    metadata_fps: float,
) -> TemporalValidation:
    tolerance = max(1.0, 1.0 / metadata_fps)
    clipped: list[GTSpan] = []
    boundary_clipped = False

    for span in spans:
        if span.end < span.start:
            return TemporalValidation(
                temporal_status="GT_INVALID",
                spans=tuple(clipped),
                gt_boundary_clipped=boundary_clipped,
                excluded_reason="GT_END_BEFORE_START",
                tolerance=tolerance,
            )

        overflow = max(0.0, -span.start, span.end - clip_duration)
        if overflow > tolerance:
            return TemporalValidation(
                temporal_status="GT_OUT_OF_RANGE",
                spans=tuple(clipped),
                gt_boundary_clipped=boundary_clipped,
                excluded_reason="GT_OUT_OF_RANGE",
                tolerance=tolerance,
            )

        start = span.start
        end = span.end
        if start < 0.0:
            start = 0.0
            boundary_clipped = True
        if end > clip_duration:
            end = clip_duration
            boundary_clipped = True
        clipped.append(GTSpan(start=start, end=end))

    return TemporalValidation(
        temporal_status="OK",
        spans=tuple(clipped),
        gt_boundary_clipped=boundary_clipped,
        excluded_reason=None,
        tolerance=tolerance,
    )


def compute_alignment_metrics(
    all_cells: Iterable[TemporalCell],
    window_source_frame_indices: Iterable[int],
    gt_spans: Iterable[GTSpan],
) -> AlignmentMetrics:
    cells = list(all_cells)
    spans = tuple(gt_spans)
    visible_total = visible_cells_for_spans(cells, spans)
    unique_window = set(int(x) for x in window_source_frame_indices)
    visible_window = visible_total.intersection(unique_window)
    n_unique = len(unique_window)
    density = len(visible_window) / n_unique if n_unique else None
    recall = len(visible_window) / len(visible_total) if visible_total else None
    return AlignmentMetrics(
        n_gt_visible_in_window=len(visible_window),
        n_gt_visible_total=len(visible_total),
        n_unique_frames_window=n_unique,
        evidence_density=density,
        gt_evidence_recall=recall,
    )


def label_support(
    prediction: str,
    metrics: AlignmentMetrics,
    min_density: float = 0.25,
    min_gt_recall: float = 0.50,
) -> str:
    if prediction != "YES":
        return "NOT_SUPPORT_CANDIDATE"
    if metrics.n_gt_visible_total == 0:
        return "NO_VISIBLE_GT_FRAME"
    if metrics.n_gt_visible_in_window == 0:
        return "SPURIOUS_SUPPORT"
    density = metrics.evidence_density or 0.0
    recall = metrics.gt_evidence_recall or 0.0
    if density >= min_density or recall >= min_gt_recall:
        return "STRONG_GT_SUPPORT"
    return "WEAK_GT_SUPPORT"


def frame_quality_stats(frames: list[int]) -> dict[str, Any]:
    diffs = [frames[i + 1] - frames[i] for i in range(len(frames) - 1)]
    positive_diffs = [d for d in diffs if d > 0]
    duplicates = len(frames) - len(set(frames))
    return {
        "duplicate_count": duplicates,
        "duplicate_ratio": duplicates / len(frames) if frames else 0.0,
        "n_unique_frames": len(set(frames)),
        "nonmonotonic_count": sum(1 for d in diffs if d < 0),
        "frame_delta_min": min(diffs) if diffs else None,
        "frame_delta_median": median(diffs) if diffs else None,
        "frame_delta_max": max(diffs) if diffs else None,
        "positive_frame_delta_median": median(positive_diffs) if positive_diffs else None,
    }
