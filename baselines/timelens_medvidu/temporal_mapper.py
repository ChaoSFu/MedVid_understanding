from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import median
from typing import Any


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
class MappingResult:
    clip_duration: float
    observations: tuple[FrameObservation, ...]
    time_mapping_method: str
    source_timebase_hz: float | None
    clip_duration_source: str


@dataclass(frozen=True)
class SpacingAudit:
    n_frames: int
    n_positive_deltas: int
    min_delta: float | None
    max_delta: float | None
    median_delta: float | None
    std_delta: float | None
    effective_fps: float | None
    max_timing_error: float | None
    median_timing_error: float | None
    uniform_positive_spacing: bool
    duplicate_timestamps_present: bool
    qwen_frame_list_single_fps_compatible: bool
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_frames": self.n_frames,
            "n_positive_deltas": self.n_positive_deltas,
            "min_delta": self.min_delta,
            "max_delta": self.max_delta,
            "median_delta": self.median_delta,
            "std_delta": self.std_delta,
            "effective_fps": self.effective_fps,
            "max_timing_error": self.max_timing_error,
            "median_timing_error": self.median_timing_error,
            "uniform_positive_spacing": self.uniform_positive_spacing,
            "duplicate_timestamps_present": self.duplicate_timestamps_present,
            "qwen_frame_list_single_fps_compatible": self.qwen_frame_list_single_fps_compatible,
            "status": self.status,
        }


class TemporalMapper:
    """Dataset-aware, GT-free MedVidU frame index to clip-local time mapper."""

    SOURCE_TIMEBASE_HZ = {
        "AVOS": 15.0,
        "CholecT50": 1.0,
        "CoPESD": 1.0,
        "EgoSurgery": 0.5,
    }

    @classmethod
    def map(
        cls,
        dataset_name: str,
        sampled_video_frames: list[int],
        frame_paths: list[str],
        metadata: dict[str, Any],
    ) -> MappingResult:
        if len(sampled_video_frames) != len(frame_paths):
            raise ValueError("len(sampled_video_frames) must equal len(frame_paths)")
        if not sampled_video_frames:
            raise ValueError("sampled_video_frames is empty")
        if dataset_name == "NurViD":
            return cls._map_nurvid(sampled_video_frames, frame_paths, metadata)
        if dataset_name in cls.SOURCE_TIMEBASE_HZ:
            return cls._map_source_timebase(sampled_video_frames, frame_paths, cls.SOURCE_TIMEBASE_HZ[dataset_name])
        if "fps" in metadata:
            return cls._map_source_timebase(sampled_video_frames, frame_paths, float(metadata["fps"]))
        raise ValueError(f"Unsupported TAL dataset for temporal mapping: {dataset_name}")

    @staticmethod
    def _map_nurvid(
        sampled_video_frames: list[int],
        frame_paths: list[str],
        metadata: dict[str, Any],
    ) -> MappingResult:
        start_time = metadata.get("input_video_start_time")
        end_time = metadata.get("input_video_end_time")
        if start_time is None or end_time is None:
            raise ValueError("NurViD requires input_video_start_time and input_video_end_time")

        clip_duration = float(end_time) - float(start_time)
        first = sampled_video_frames[0]
        last = sampled_video_frames[-1]
        metadata_fps = float(metadata.get("fps", 1.0))
        source_timebase_hz = (last - first) / clip_duration if clip_duration > 0 and last > first else None

        observations: list[FrameObservation] = []
        for i, frame_index in enumerate(sampled_video_frames):
            if last > first:
                local_time = (frame_index - first) / (last - first) * clip_duration
            else:
                local_time = i / metadata_fps
            observations.append(
                FrameObservation(
                    frame_position=i,
                    source_frame_index=int(frame_index),
                    frame_path=frame_paths[i],
                    local_time=float(local_time),
                )
            )
        return MappingResult(
            clip_duration=float(clip_duration),
            observations=tuple(observations),
            time_mapping_method="known_duration_endpoint_interpolation",
            source_timebase_hz=source_timebase_hz,
            clip_duration_source="metadata_input_video_times",
        )

    @staticmethod
    def _map_source_timebase(
        sampled_video_frames: list[int],
        frame_paths: list[str],
        source_timebase_hz: float,
    ) -> MappingResult:
        first = sampled_video_frames[0]
        last = sampled_video_frames[-1]
        clip_duration = (last - first) / source_timebase_hz if last >= first else 0.0
        observations = tuple(
            FrameObservation(
                frame_position=i,
                source_frame_index=int(frame_index),
                frame_path=frame_paths[i],
                local_time=float(frame_index - first) / float(source_timebase_hz),
            )
            for i, frame_index in enumerate(sampled_video_frames)
        )
        return MappingResult(
            clip_duration=float(clip_duration),
            observations=observations,
            time_mapping_method="source_frame_rate",
            source_timebase_hz=float(source_timebase_hz),
            clip_duration_source="source_frame_index_range",
        )


def audit_uniform_spacing(local_times: list[float], tolerance: float = 1e-6) -> SpacingAudit:
    if len(local_times) < 2:
        return SpacingAudit(
            n_frames=len(local_times),
            n_positive_deltas=0,
            min_delta=None,
            max_delta=None,
            median_delta=None,
            std_delta=None,
            effective_fps=None,
            max_timing_error=0.0 if local_times else None,
            median_timing_error=0.0 if local_times else None,
            uniform_positive_spacing=True,
            duplicate_timestamps_present=False,
            qwen_frame_list_single_fps_compatible=True,
            status="OK",
        )

    deltas = [local_times[i + 1] - local_times[i] for i in range(len(local_times) - 1)]
    if any(not math.isfinite(x) for x in local_times + deltas):
        raise ValueError("local timestamps must be finite")
    if any(delta < -tolerance for delta in deltas):
        raise ValueError("local timestamps must be non-decreasing")

    positive = [delta for delta in deltas if delta > tolerance]
    duplicate = any(abs(delta) <= tolerance for delta in deltas)
    if not positive:
        return SpacingAudit(
            n_frames=len(local_times),
            n_positive_deltas=0,
            min_delta=0.0,
            max_delta=0.0,
            median_delta=None,
            std_delta=0.0,
            effective_fps=None,
            max_timing_error=None,
            median_timing_error=None,
            uniform_positive_spacing=False,
            duplicate_timestamps_present=duplicate,
            qwen_frame_list_single_fps_compatible=False,
            status="NO_POSITIVE_DELTAS",
        )

    med = float(median(positive))
    mean = sum(positive) / len(positive)
    std = (sum((x - mean) ** 2 for x in positive) / len(positive)) ** 0.5
    uniform = max(abs(x - med) for x in positive) <= max(tolerance, abs(med) * 1e-6)
    effective_fps = 1.0 / med if med > 0 else None
    normalized = [t - local_times[0] for t in local_times]
    predicted = [i / effective_fps for i in range(len(local_times))] if effective_fps else []
    errors = [abs(a - b) for a, b in zip(normalized, predicted)]
    max_error = max(errors) if errors else None
    median_error = float(median(errors)) if errors else None
    compatible = bool(uniform and not duplicate and (max_error is not None and max_error <= max(tolerance, med * 1e-6)))
    if compatible:
        status = "OK"
    elif duplicate:
        status = "DUPLICATE_TIMESTAMPS_NOT_SINGLE_FPS_COMPATIBLE"
    else:
        status = "NON_UNIFORM_TIMESTAMPS_NOT_SINGLE_FPS_COMPATIBLE"
    return SpacingAudit(
        n_frames=len(local_times),
        n_positive_deltas=len(positive),
        min_delta=min(deltas),
        max_delta=max(deltas),
        median_delta=med,
        std_delta=std,
        effective_fps=effective_fps,
        max_timing_error=max_error,
        median_timing_error=median_error,
        uniform_positive_spacing=uniform,
        duplicate_timestamps_present=duplicate,
        qwen_frame_list_single_fps_compatible=compatible,
        status=status,
    )

