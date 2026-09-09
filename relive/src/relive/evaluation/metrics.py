"""Independent descriptive evaluation; no metrics are called official."""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
from typing import Any


def _read_results(run_dir: str | Path) -> list[dict]:
    directory = Path(run_dir)
    files = sorted((directory / "samples").glob("*.json"))
    if files:
        rows = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    elif (directory / "results.jsonl").exists():
        rows = [json.loads(line) for line in (directory / "results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        raise ValueError("no completed sample result files found")
    if not rows or any(not isinstance(r, dict) or not isinstance(r.get("sample_id"), str) or not r["sample_id"] for r in rows):
        raise ValueError("malformed sample result")
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate sample IDs in result files")
    if any(not isinstance(r.get("strict"), dict) for r in rows):
        raise ValueError("sample results require a structured strict prediction")
    return rows


def _load_ground_truth(path: str | Path) -> dict[str, dict]:
    """Only this independent evaluation module opens ground truth files."""
    source = Path(path)
    payload = source.read_text(encoding="utf-8")
    if source.suffix == ".jsonl":
        rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
    else:
        rows = json.loads(payload)
    if not isinstance(rows, list) or any(not isinstance(r, dict) or not isinstance(r.get("sample_id"), str) or not r["sample_id"] for r in rows):
        raise ValueError("evaluation GT requires a list/JSONL of records with sample_id")
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate GT sample IDs")
    return {r["sample_id"]: r for r in rows}


def _metric(value: Any = None, count: int = 0, reason: str | None = None) -> dict:
    return {"status": "AVAILABLE" if reason is None else "NOT_EVALUABLE", "value": value, "n": count, **({"reason": reason} if reason else {})}


def _answered(strict: dict) -> bool:
    return isinstance(strict, dict) and strict.get("status") == "ANSWERED"


def _exact_answer(answer: Any) -> str:
    return json.dumps(answer, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _conditional_score(rows: list[dict], truth: dict[str, dict], mode: str = "strict") -> dict:
    evaluable = []
    by_task: dict[str, list[bool]] = {}
    for row in rows:
        strict = row.get(mode) or {}
        gt = truth.get(row["sample_id"])
        if not _answered(strict) or gt is None or gt.get("answer") is None or ("task" in gt and gt["task"] != row.get("task")):
            continue
        hit = _exact_answer(strict.get("answer")) == _exact_answer(gt["answer"])
        evaluable.append(hit)
        by_task.setdefault(row.get("task", "unknown"), []).append(hit)
    reason = None if evaluable else "No answered predictions with corresponding answer labels."
    return {"metric": "strict_task_answer_exact_match", "official": False,
            "answered_samples": sum(_answered(row.get(mode)) for row in rows), "labelled_answered_samples": len(evaluable),
            "exact_match": _metric(sum(evaluable) / len(evaluable) if evaluable else None, len(evaluable), reason),
            "answered_error_rate": _metric(1 - sum(evaluable) / len(evaluable) if evaluable else None, len(evaluable), reason),
            "by_task": {task: {"n": len(hits), "exact_match": sum(hits) / len(hits)} for task, hits in by_task.items()},
            "limitation": "Exact structural JSON/text equality only; no medical semantic accuracy or official downstream metric is inferred."}


def _box(value: Any, coordinate_system: str | None = None) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4 or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value):
        return None
    if min(value) < 0 or value[2] <= value[0] or value[3] <= value[1]:
        return None
    if coordinate_system == "normalized_0_1_xyxy" and max(value) > 1:
        return None
    return value


def _iou(a: list[float], b: list[float]) -> float:
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a, area_b = (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area_a + area_b - intersection)


def _alignments(rows: list[dict], truth: dict[str, dict]) -> dict:
    spatial, temporal = [], []
    for row in rows:
        # Only explicitly task-typed predictions can be compared to corresponding
        # labels: support regions are not object boxes or action time spans.
        prediction = row.get("evaluation_prediction") or {}
        if not isinstance(prediction, dict):
            continue
        labels = truth.get(row["sample_id"], {})
        p, g = prediction.get("target_bbox"), labels.get("target_bbox")
        pb, gb = _box(p.get("xyxy"), p.get("coordinate_system")) if isinstance(p, dict) else None, _box(g.get("xyxy"), g.get("coordinate_system")) if isinstance(g, dict) else None
        if pb and gb and p.get("coordinate_system") == g.get("coordinate_system") and p.get("coordinate_system") in {"normalized_0_1_xyxy", "original_pixels_xyxy"} and p.get("frame_id") == g.get("frame_id") and p.get("frame_id"):
            spatial.append(_iou(pb, gb))
        p, g = prediction.get("temporal_span"), labels.get("temporal_span")
        if isinstance(p, dict) and isinstance(g, dict) and isinstance(p.get("timebase"), str) and p["timebase"] and p["timebase"] == g.get("timebase"):
            vals = [p.get("start"), p.get("end"), g.get("start"), g.get("end")]
            if all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x >= 0 for x in vals):
                ps, pe, gs, ge = vals
                if pe > ps and ge > gs:
                    intersection = max(0, min(pe, ge) - max(ps, gs))
                    temporal.append(intersection / ((pe - ps) + (ge - gs) - intersection))
    return {"target_box_iou": _metric(sum(spatial) / len(spatial) if spatial else None, len(spatial), None if spatial else "No paired target boxes with the same frame and coordinate system."),
            "temporal_span_iou": _metric(sum(temporal) / len(temporal) if temporal else None, len(temporal), None if temporal else "No paired predicted/annotated temporal spans with matching explicit timebase."),
            "limitation": "Spatial overlap does not establish semantic truth; temporal nonoverlap does not establish action absence."}


def _collect_statuses(value: Any, prefix: str, out: Counter) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            label = f"{prefix}.{key}" if prefix else key
            if isinstance(item, str) and (key.endswith("status") or key in {"applicability", "region_specificity"}):
                out[f"{label}={item}"] += 1
            elif isinstance(item, (dict, list)):
                _collect_statuses(item, label, out)
    elif isinstance(value, list):
        for item in value:
            _collect_statuses(item, prefix, out)


def evaluate(run_dir: str | Path, gt_path: str | Path | None = None) -> dict:
    rows = _read_results(run_dir)
    truth = _load_ground_truth(gt_path) if gt_path is not None else {}
    count = len(rows)
    answered = sum(_answered(row.get("strict", {})) for row in rows)
    paired_rows = [row for row in rows if isinstance(row.get("before_adaptation_strict"), dict)]
    before = sum(_answered(row["before_adaptation_strict"]) for row in paired_rows)
    before_n = len(paired_rows)
    paired_after = sum(_answered(row["strict"]) for row in paired_rows)
    checks: Counter = Counter()
    statuses: Counter = Counter()
    for row in rows:
        for cert in row.get("certificates", []):
            statuses[cert.get("final_status", "MISSING")] += 1
            _collect_statuses(cert.get("checks", {}), "", checks)
    fallback_n = sum((row.get("benchmark_forced") or {}).get("fallback_used") is True for row in rows)
    usage = [r.get("usage") or {} for r in rows]

    def measured_total(field: str) -> dict:
        values = [u.get(field) for u in usage if isinstance(u, dict)]
        valid = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0 and (field == "latency_seconds" or isinstance(v, int))]
        if len(valid) != count:
            return _metric(None, len(valid), f"Missing or invalid {field} for {count - len(valid)} sample(s); total is not inferred.")
        return _metric(sum(valid), count)

    measured = {name: measured_total(name) for name in ("calls", "latency_seconds", "new_calls", "cache_hits")}
    calls, latency = measured["calls"]["value"], measured["latency_seconds"]["value"]
    return {
        "evaluation_version": "relive-evaluation-v1", "n_samples": count,
        "synthetic": all(r.get("synthetic") is True for r in rows),
        "source_kinds": dict(Counter("synthetic" if r.get("synthetic") is True else "real" if r.get("synthetic") is False else "unknown" for r in rows)),
        "gt_loaded": gt_path is not None, "gt_joined_samples": sum(r["sample_id"] in truth for r in rows),
        "official_downstream_metrics": {"status": "NOT_CONNECTED", "reason": "Official evaluator and task prediction mapping have not been integrated."},
        "strict_coverage": _metric(answered / count, count),
        "coverage_status_distribution": dict(Counter((r.get("coverage") or {}).get("status", "MISSING") for r in rows)),
        "strict_task_metrics": _conditional_score(rows, truth),
        "fallback_rate": _metric(fallback_n / count, count),
        "fallback_enabled_samples": sum(r.get("benchmark_forced") is not None for r in rows),
        "certificate_status_distribution": dict(statuses), "check_status_distribution": dict(checks),
        "termination_distribution": dict(Counter(r.get("termination_reason", "MISSING") for r in rows)),
        "compute": {"calls": calls, "mean_calls_per_sample": calls / count if calls is not None else None, "latency_seconds": latency,
                    "new_calls_recorded_at_sample_completion": measured["new_calls"]["value"],
                    "cache_hits_recorded_at_sample_completion": measured["cache_hits"]["value"], "measurements": measured},
        "adaptation": {"before_coverage": _metric(before / before_n if before_n else None, before_n, None if before_n else "No stored before-adaptation output."),
                       "after_coverage": _metric(paired_after / before_n if before_n else None, before_n, None if before_n else "No stored before-adaptation output for a paired comparison."),
                       "before_task_metrics": _conditional_score(paired_rows, truth, "before_adaptation_strict"),
                       "after_task_metrics": _conditional_score(paired_rows, truth),
                       "events": sum(len(r.get("adaptation_events", [])) for r in rows),
                       "comparison_limitation": "Descriptive within-run results; comparison is not a matched-budget causal experiment."},
        "alignment": _alignments(rows, truth),
        "limitations": ["Synthetic outputs validate engineering behavior only.", "Coverage and compute must accompany conditional answered accuracy.", "Forced fallback predictions are excluded from strict task metrics."],
    }
