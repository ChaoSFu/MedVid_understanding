from __future__ import annotations

from pathlib import Path
from typing import Any

from baselines.evqa_medvidu.io_utils import first_gt_leak, read_jsonl, write_json


def audit_manifest_gt_leakage(manifest_dir: Path, output_path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {"tasks": {}, "leak_count": 0}
    for task in ("stg", "rc", "cvs"):
        rows = read_jsonl(manifest_dir / f"{task}_gt_free.jsonl")
        leaks = []
        for row in rows:
            allowed = {"$.provided_region", "$.provided_region.start_frame_bbox"} if task == "rc" else set()
            leak = first_gt_leak(row, allowed_paths=allowed)
            if leak:
                leaks.append({"sample_id": row.get("sample_id"), "path": leak})
        payload["tasks"][task] = {
            "rows_checked": len(rows),
            "leaks": leaks[:20],
            "leak_count": len(leaks),
            "gt_visible": bool(leaks),
            "provided_region_visible": task == "rc",
            "target_answer_visible": False,
        }
        payload["leak_count"] += len(leaks)
    write_json(output_path, payload)
    return payload

