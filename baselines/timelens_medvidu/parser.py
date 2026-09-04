from __future__ import annotations

import sys
from pathlib import Path

from .config import TIMELENS_ROOT


if str(TIMELENS_ROOT) not in sys.path:
    sys.path.insert(0, str(TIMELENS_ROOT))

from timelens.utils import extract_time as official_extract_time  # noqa: E402


def parse_timelens_answer(answer: str) -> tuple[str, list[list[float]]]:
    timestamps = official_extract_time(answer or "")
    if not timestamps:
        return "NO_TIMESTAMP", []
    parsed = [[float(start), float(end)] for start, end in timestamps]
    return "OK", parsed

