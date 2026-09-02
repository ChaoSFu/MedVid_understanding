from __future__ import annotations

from typing import Any

from .temporal import (
    GTSpan,
    TemporalCell,
    classify_gt_alignment,
    compute_alignment_metrics,
)


def generate_position_windows(
    sample: dict[str, Any],
    window_size: int = 16,
    stride: int = 8,
    drop_last: bool = True,
) -> list[dict[str, Any]]:
    frame_paths = sample["frame_paths"]
    frame_indices = sample["sampled_frame_indices"]
    observations = sample["frame_observations"]
    cells_by_frame = {
        int(cell["source_frame_index"]): cell for cell in sample.get("temporal_cells", [])
    }
    n = len(frame_paths)
    windows: list[dict[str, Any]] = []
    start = 0
    while start < n:
        end = start + window_size
        if end > n:
            if drop_last:
                break
            end = n
        if end <= start:
            break
        indices = frame_indices[start:end]
        unique_indices = sorted(set(indices))
        unique_cells = [cells_by_frame[i] for i in unique_indices if i in cells_by_frame]
        if unique_cells:
            start_time = min(float(c["cell_start"]) for c in unique_cells)
            end_time = max(float(c["cell_end"]) for c in unique_cells)
        else:
            obs_times = [float(o["local_time"]) for o in observations[start:end]]
            start_time = min(obs_times) if obs_times else None
            end_time = max(obs_times) if obs_times else None
        duplicate_count = len(indices) - len(set(indices))
        window = {
            "window_id": f"{sample['qa_id']}::pos{start:04d}-{end - 1:04d}",
            "qa_id": sample["qa_id"],
            "clip_id": sample["clip_id"],
            "sample_id": sample["qa_id"],
            "dataset_name": sample.get("dataset_name"),
            "metadata_fps": sample.get("metadata_fps"),
            "analysis_eligible": bool(sample.get("analysis_eligible")),
            "temporal_status": sample.get("temporal_status"),
            "start_pos": start,
            "end_pos": end - 1,
            "start_time": start_time,
            "end_time": end_time,
            "frame_paths": frame_paths[start:end],
            "frame_indices": indices,
            "frame_positions": list(range(start, end)),
            "frame_times": [float(o["local_time"]) for o in observations[start:end]],
            "n_frames": len(indices),
            "n_unique_frames": len(set(indices)),
            "duplicate_count": duplicate_count,
            "duplicate_ratio": duplicate_count / len(indices) if indices else 0.0,
        }
        windows.append(window)
        start += stride
    return windows


def _cells_from_sample(sample: dict[str, Any]) -> list[TemporalCell]:
    return [
        TemporalCell(
            source_frame_index=int(cell["source_frame_index"]),
            local_time=float(cell["local_time"]),
            cell_start=float(cell["cell_start"]),
            cell_end=float(cell["cell_end"]),
            frame_positions=tuple(int(x) for x in cell.get("frame_positions", [])),
        )
        for cell in sample.get("temporal_cells", [])
    ]


def _spans_from_sample(sample: dict[str, Any]) -> list[GTSpan]:
    return [
        GTSpan(float(span["start"]), float(span["end"]))
        for span in sample.get("gt_spans", [])
    ]


def compute_window_alignment(sample: dict[str, Any], window: dict[str, Any]) -> dict[str, Any]:
    metrics = compute_alignment_metrics(
        _cells_from_sample(sample),
        window["frame_indices"],
        _spans_from_sample(sample),
    )
    out = metrics.to_dict()
    out["gt_alignment_class"] = classify_gt_alignment(
        metrics,
        analysis_eligible=bool(sample.get("analysis_eligible")),
    )
    return out
