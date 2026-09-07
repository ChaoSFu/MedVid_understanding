from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any

from .frame_adapter import nearest_indices_for_target_times


STG_TIME_SPEC_VERSION = "stg_question_schedule_and_nearest_frame_v1"


@dataclass(frozen=True)
class STGTargetSchedule:
    start_seconds: float
    end_seconds: float
    interval_seconds: float
    target_timestamps: tuple[float, ...]
    parse_pattern: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": STG_TIME_SPEC_VERSION,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "interval_seconds": self.interval_seconds,
            "target_timestamps": list(self.target_timestamps),
            "parse_pattern": self.parse_pattern,
        }


_NUMBER = r"(?P<{name}>\d+(?:\.\d+)?)"
_SCHEDULE_PATTERNS = (
    (
        "every_then_from",
        re.compile(
            rf"every\s+{_NUMBER.format(name='step')}\s+seconds?\s+from\s+"
            rf"{_NUMBER.format(name='start')}\s+to\s+{_NUMBER.format(name='end')}\s+seconds?",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "from_then_every",
        re.compile(
            rf"from\s+{_NUMBER.format(name='start')}\s+to\s+{_NUMBER.format(name='end')}\s+seconds?"
            rf".*?every\s+{_NUMBER.format(name='step')}\s+seconds?",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "every_then_between",
        re.compile(
            rf"every\s+{_NUMBER.format(name='step')}\s+seconds?\s+between\s+"
            rf"{_NUMBER.format(name='start')}\s+and\s+{_NUMBER.format(name='end')}\s+seconds?",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "between_then_every",
        re.compile(
            rf"between\s+{_NUMBER.format(name='start')}\s+and\s+{_NUMBER.format(name='end')}\s+seconds?"
            rf".*?every\s+{_NUMBER.format(name='step')}\s+seconds?",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "between_seconds_then_every",
        re.compile(
            rf"between\s+{_NUMBER.format(name='start')}\s+seconds?\s+and\s+{_NUMBER.format(name='end')}\s+seconds?"
            rf".*?every\s+{_NUMBER.format(name='step')}\s+seconds?",
            flags=re.IGNORECASE,
        ),
    ),
)


def parse_stg_target_schedule(question: str) -> STGTargetSchedule:
    """Extract the benchmark-requested STG timestamps from the task input."""
    for pattern_name, pattern in _SCHEDULE_PATTERNS:
        match = pattern.search(question)
        if match is None:
            continue
        values = {key: float(value) for key, value in match.groupdict().items()}
        start = values["start"]
        end = values["end"]
        step = values["step"]
        if step <= 0.0 or end < start:
            raise ValueError(f"invalid STG target schedule: start={start}, end={end}, step={step}")
        count = math.floor((end - start) / step + 1e-8)
        targets = tuple(round(start + i * step, 8) for i in range(count + 1))
        return STGTargetSchedule(start, end, step, targets, pattern_name)
    raise ValueError(f"could not parse STG target schedule from question: {question!r}")


def build_stg_target_alignment(
    local_timestamps: list[float],
    source_frame_indices: list[int],
    frame_paths: list[str],
    schedule: STGTargetSchedule,
) -> list[dict[str, Any]]:
    if not (len(local_timestamps) == len(source_frame_indices) == len(frame_paths)):
        raise ValueError("STG timestamp, source-index, and path lists must align")
    if not local_timestamps:
        raise ValueError("STG target alignment requires at least one benchmark frame")
    if schedule.start_seconds < local_timestamps[0] - 1e-6 or schedule.end_seconds > local_timestamps[-1] + 1e-6:
        raise ValueError(
            "STG target schedule falls outside available clip-local time range: "
            f"requested [{schedule.start_seconds}, {schedule.end_seconds}], "
            f"available [{local_timestamps[0]}, {local_timestamps[-1]}]"
        )
    indices = nearest_indices_for_target_times(local_timestamps, list(schedule.target_timestamps))
    return [
        {
            "target_timestamp": float(target),
            "logical_medvidu_frame_index": int(index),
            "source_frame_index": int(source_frame_indices[index]),
            "matched_frame_timestamp": float(local_timestamps[index]),
            "absolute_timing_error_seconds": abs(float(local_timestamps[index]) - float(target)),
            "frame_path": frame_paths[index],
        }
        for target, index in zip(schedule.target_timestamps, indices)
    ]
