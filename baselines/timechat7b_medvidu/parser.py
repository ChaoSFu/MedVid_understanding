from __future__ import annotations

import math
from typing import Any


def parse_timechat_answer(timechat_runner: Any, answer: str) -> dict[str, Any]:
    first = timechat_runner.extract_time(answer or "")
    used_fallback = False
    parsed = first
    if parsed == [0, 0]:
        parsed = timechat_runner.extract_time2(answer or "")
        used_fallback = True
    span = [float(parsed[0]), float(parsed[1])] if isinstance(parsed, (list, tuple)) and len(parsed) >= 2 else [0.0, 0.0]
    parse_valid = all(math.isfinite(x) for x in span) and span[1] > span[0] and span != [0.0, 0.0]
    return {
        "raw_answer": answer,
        "official_parsed_span": span,
        "parse_valid": parse_valid,
        "official_parser": "TimeChat.extract_time_then_extract_time2_fallback",
        "extract_time2_fallback_used": used_fallback,
    }

