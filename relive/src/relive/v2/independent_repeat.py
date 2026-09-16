"""Zero-model comparison and safe summary tools for Stage 3B fresh repeats.

This module intentionally never imports a backend, cache, dataset adapter, or
Stage 3B inference implementation.  It compares only emitted, canonical
Stage 3B artifacts from two independently created run roots.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

from relive.storage.artifacts import canonical_json, stable_hash
from .task_selection import TALSelectionError, strict_json_loads, strict_jsonl


FORMAT = "relive-v2-tal-stage3b-independent-repeat-v1"
CHOICES = ("A", "B", "C", "D")
WINDOW_COUNT = 23


class IndependentRepeatError(ValueError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict[str, Any] | list[dict[str, Any]]) -> None:
    if path.exists():
        raise IndependentRepeatError("IMMUTABLE_OUTPUT_EXISTS")
    payload = (canonical_json(value) + "\n").encode("utf-8") if isinstance(value, dict) else b"".join(
        (canonical_json(row) + "\n").encode("utf-8") for row in value)
    path.write_bytes(payload)


def _object(path: Path, code: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = strict_json_loads(raw.decode("utf-8"), error_code=code)
    except (OSError, UnicodeDecodeError, TALSelectionError) as exc:
        raise IndependentRepeatError(f"{code}_INVALID") from exc
    if not isinstance(value, dict) or raw != (canonical_json(value) + "\n").encode("utf-8"):
        raise IndependentRepeatError(f"{code}_NONCANONICAL")
    return value


def _rows(path: Path, code: str) -> list[dict[str, Any]]:
    try:
        value = list(strict_jsonl(path, error_code=code))
        raw = path.read_bytes()
    except (OSError, TALSelectionError) as exc:
        raise IndependentRepeatError(f"{code}_INVALID") from exc
    expected = b"".join((canonical_json(row) + "\n").encode("utf-8") for row in value)
    if raw != expected or any(not isinstance(row, dict) for row in value):
        raise IndependentRepeatError(f"{code}_NONCANONICAL")
    return value


def _finite(value: Any, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise IndependentRepeatError(code)
    return float(value)


def _without_operational(value: Any) -> Any:
    """Remove paths and execution-only values from an equality projection."""
    if isinstance(value, list):
        return [_without_operational(item) for item in value]
    if not isinstance(value, dict):
        return value
    excluded = {
        "cache_key", "cache_hit", "cache_dir", "output_dir", "latency_seconds",
        "resolved_frame_path", "image_paths", "model_path", "device", "input_device",
        "device_map", "max_memory", "timeout_seconds", "runtime_placement",
    }
    return {key: _without_operational(item) for key, item in value.items() if key not in excluded}


def _packet_projection(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        frames = [{key: frame[key] for key in ("logical_frame_order", "source_frame_reference", "frame_sha256", "timestamp_seconds", "logical_aliases")}
                  for frame in row.get("frames", [])]
        result.append({key: row.get(key) for key in ("window_id", "window_role", "start_seconds", "end_seconds", "scale_seconds", "requirement_id", "frame_selection_policy", "maximum_unique_frames", "duplicate_visual_frames_removed_from_packet")}
                      | {"frames": frames})
    return sorted(result, key=lambda item: item["window_id"])


def _call_plan_projection(plan: Mapping[str, Any]) -> dict[str, Any]:
    calls = []
    for call in plan.get("calls", []):
        calls.append({key: call.get(key) for key in ("window_id", "requirement_id", "prompt_version", "prompt_sha256", "frame_ids", "frame_sha256")})
    return {"format": plan.get("format"), "planned_model_calls": plan.get("planned_model_calls"),
            "input_hashes": plan.get("input_hashes"), "calls": sorted(calls, key=lambda item: item["window_id"])}


def _model_projection(preflight: Mapping[str, Any]) -> dict[str, Any]:
    fingerprint = preflight.get("model_fingerprint")
    if not isinstance(fingerprint, dict):
        raise IndependentRepeatError("MODEL_FINGERPRINT_INVALID")
    identity = fingerprint.get("scientific_identity")
    if not isinstance(identity, dict):
        raise IndependentRepeatError("MODEL_SCIENTIFIC_IDENTITY_INVALID")
    fields = ("model", "revision", "checkpoint_metadata_sha256", "actual_model_class", "actual_processor_class",
              "model_class", "processor_class", "chat_template_source", "chat_template_sha256", "chat_template_kwargs",
              "chat_message_layout", "frame_encoding", "image_order", "processor_call_mode", "processor_min_pixels",
              "processor_max_pixels", "generation", "local_files_only", "trust_remote_code")
    return {key: identity.get(key) for key in fields} | {"preprocessing": fingerprint.get("preprocessing")}


def _token_projection(plan: Mapping[str, Any], preflight: Mapping[str, Any]) -> list[dict[str, Any]]:
    contracts = preflight.get("choice_token_contracts")
    packet_window = {item.get("packet_id"): item.get("window_id") for item in plan.get("calls", [])}
    if not isinstance(contracts, dict) or set(contracts) != set(packet_window):
        raise IndependentRepeatError("CHOICE_TOKEN_CONTRACT_BINDING_INVALID")
    output = []
    for packet_id, contract in contracts.items():
        if not isinstance(contract, dict) or contract.get("choice_labels") != list(CHOICES):
            raise IndependentRepeatError("CHOICE_TOKEN_CONTRACT_INVALID")
        tokens = contract.get("choice_token_ids")
        if not isinstance(tokens, dict) or set(tokens) != set(CHOICES) or len(set(tokens.values())) != 4:
            raise IndependentRepeatError("CHOICE_TOKEN_CONTRACT_INVALID")
        output.append({"window_id": packet_window[packet_id], "choice_token_ids": tokens})
    return sorted(output, key=lambda item: item["window_id"])


def _load_bundle(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise IndependentRepeatError("RUN_DIRECTORY_INVALID")
    plan = _object(root / "v2_stage3b_call_plan.json", "CALL_PLAN")
    preflight = _object(root / "v2_stage3b_preflight.json", "PREFLIGHT")
    freeze = _object(root / "v2_stage3b_freeze_audit.json", "FREEZE_AUDIT")
    prompt = _object(root / "v2_stage3b_prompt_spec.json", "PROMPT_SPEC")
    choice = _object(root / "v2_stage3b_choice_spec.json", "CHOICE_SPEC")
    policy = _object(root / "v2_stage3b_ranking_policy.json", "RANKING_POLICY")
    packets = _rows(root / "v2_stage3b_visual_packets.jsonl", "VISUAL_PACKETS")
    summary = _object(root / "v2_stage3b_run_summary.json", "RUN_SUMMARY")
    likelihoods = _rows(root / "run" / "v2_stage3b_window_likelihoods.jsonl", "LIKELIHOODS")
    ranking = _rows(root / "run" / "v2_stage3b_window_ranking.jsonl", "RANKING")
    hypotheses = _rows(root / "run" / "v2_stage3b_hypothesis_claims.jsonl", "HYPOTHESES")
    graph = _object(root / "run" / "v2_stage3b_claim_graph.json", "CLAIM_GRAPH")
    return {"root": root, "plan": plan, "preflight": preflight, "freeze": freeze, "prompt": prompt,
            "choice": choice, "policy": policy, "packets": packets, "summary": summary,
            "likelihoods": likelihoods, "ranking": ranking, "hypotheses": hypotheses, "graph": graph}


def _scientific_inputs(bundle: Mapping[str, Any]) -> dict[str, Any]:
    freeze = bundle["freeze"]
    if freeze.get("audit_content_sha256") != stable_hash({key: value for key, value in freeze.items() if key != "audit_content_sha256"}):
        raise IndependentRepeatError("FREEZE_AUDIT_HASH_MISMATCH")
    return {
        "upstream_artifact_hashes": bundle["plan"].get("input_hashes"),
        "visual_packet_scientific_projection_sha256": stable_hash(_packet_projection(bundle["packets"])),
        "prompt_specification_sha256": stable_hash(_without_operational(bundle["prompt"])),
        "choice_specification_sha256": stable_hash(_without_operational(bundle["choice"])),
        "ranking_nms_policy_sha256": stable_hash(_without_operational(bundle["policy"])),
        "call_plan_scientific_projection_sha256": stable_hash(_call_plan_projection(bundle["plan"])),
        "model_scientific_projection_sha256": stable_hash(_model_projection(bundle["preflight"])),
        "choice_token_ids_sha256": stable_hash(_token_projection(bundle["plan"], bundle["preflight"])),
        "freeze_audit_scientific_projection_sha256": stable_hash(_without_operational(freeze)),
    }


def _fresh_audit(bundle: Mapping[str, Any], *, distinct_root: bool) -> dict[str, Any]:
    summary, plan, rows = bundle["summary"], bundle["plan"], bundle["likelihoods"]
    expected_ids = [item.get("window_id") for item in plan.get("calls", [])]
    seen = [item.get("window_id") for item in rows]
    duplicates = len(seen) - len(set(seen))
    missing = len(set(expected_ids) - set(seen))
    failed = 0
    for row in rows:
        choices = row.get("log_probabilities")
        try:
            if not isinstance(choices, dict) or set(choices) != set(CHOICES):
                raise IndependentRepeatError("CHOICE_RESULT_INVALID")
            for label in CHOICES:
                _finite(choices[label], "CHOICE_RESULT_INVALID")
            _finite(row.get("support_margin"), "SUPPORT_MARGIN_INVALID")
        except IndependentRepeatError:
            failed += 1
    errors = []
    if not distinct_root: errors.append("RUN_ROOT_NOT_DISTINCT")
    if plan.get("planned_model_calls") != WINDOW_COUNT: errors.append("PLANNED_MODEL_CALL_COUNT_INVALID")
    required = {"mode": "run", "new_model_calls": WINDOW_COUNT, "cache_hits": 0,
                "logical_model_calls": WINDOW_COUNT, "window_count": WINDOW_COUNT, "model_calls_made": WINDOW_COUNT}
    if any(summary.get(key) != value for key, value in required.items()): errors.append("FRESH_SUMMARY_CONTRACT_INVALID")
    if len(rows) != WINDOW_COUNT or len(set(expected_ids)) != WINDOW_COUNT or missing or duplicates or failed:
        errors.append("WINDOW_RESULT_COMPLETENESS_INVALID")
    return {"independent_fresh_run": not errors, "errors": errors, "planned_model_calls": plan.get("planned_model_calls"),
            "new_model_calls": summary.get("new_model_calls"), "cache_hits": summary.get("cache_hits"),
            "failed_model_call_count": failed, "missing_window_count": missing,
            "duplicate_window_result_count": duplicates}


def _numeric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        logs = row["log_probabilities"]
        output.append({"window_id": row["window_id"], "log_likelihood_A": _finite(logs["A"], "LOG_LIKELIHOOD_INVALID"),
                       "log_likelihood_B": _finite(logs["B"], "LOG_LIKELIHOOD_INVALID"), "log_likelihood_C": _finite(logs["C"], "LOG_LIKELIHOOD_INVALID"),
                       "log_likelihood_D": _finite(logs["D"], "LOG_LIKELIHOOD_INVALID"), "support_margin": _finite(row["support_margin"], "SUPPORT_MARGIN_INVALID"),
                       "choice_argmax": max(CHOICES, key=lambda label: logs[label])})
    if len(output) != WINDOW_COUNT or len({item["window_id"] for item in output}) != WINDOW_COUNT:
        raise IndependentRepeatError("NUMERIC_WINDOW_ALIGNMENT_INVALID")
    return sorted(output, key=lambda item: item["window_id"])


def _interval_iou(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    start = max(float(left["start_seconds"]), float(right["start_seconds"]))
    end = min(float(left["end_seconds"]), float(right["end_seconds"]))
    union = max(float(left["end_seconds"]), float(right["end_seconds"])) - min(float(left["start_seconds"]), float(right["start_seconds"]))
    return 0.0 if union == 0 else max(0.0, end - start) / union


def _ranking_projection(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {item.get("window_id"): item for item in rows}
    if len(rows) != WINDOW_COUNT or len(by_id) != WINDOW_COUNT:
        raise IndependentRepeatError("RANKING_WINDOW_ALIGNMENT_INVALID")
    output = []
    for row in rows:
        suppressor = row.get("nms_suppressor_window_id")
        iou = None if suppressor is None else _interval_iou(row, by_id.get(suppressor, {})) if suppressor in by_id else "INVALID_SUPPRESSOR"
        output.append({key: row.get(key) for key in ("window_id", "window_role", "scale_seconds", "start_seconds", "end_seconds", "rank", "positive_hypothesis_eligible", "nms_status", "nms_suppressor_window_id")}
                      | {"nms_temporal_iou": iou})
    return sorted(output, key=lambda item: (item["rank"] is None, item["rank"] if item["rank"] is not None else 10**9, item["window_id"]))


def _hypothesis_projection(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ("hypothesis_id", "requirement_id", "polarity", "candidate_interval", "supporting_window_ids", "retrieval_rank", "generation_source", "status")
    return sorted(({key: row.get(key) for key in keys} for row in rows), key=lambda item: item.get("hypothesis_id") or "")


def _graph_projection(graph: Mapping[str, Any]) -> dict[str, Any]:
    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        raise IndependentRepeatError("CLAIM_GRAPH_NODES_INVALID")
    return {"requirement_binding": graph.get("requirement_binding"), "nodes": _hypothesis_projection(nodes),
            "observation_claim_count": graph.get("observation_claim_count"), "verified_hypothesis_count": graph.get("verified_hypothesis_count"),
            "certificate_created": graph.get("certificate_created")}


def _first_mismatch(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any] | None:
    for a, b in zip(left, right):
        if a != b:
            return {"left": a, "right": b}
    return None if len(left) == len(right) else {"left_count": len(left), "right_count": len(right)}


def _compare_ready(a: Mapping[str, Any], b: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    a_numeric, b_numeric = _numeric_rows(a["likelihoods"]), _numeric_rows(b["likelihoods"])
    a_by_id, b_by_id = {row["window_id"]: row for row in a_numeric}, {row["window_id"]: row for row in b_numeric}
    differences, values, support = [], [], []
    per_choice: dict[str, list[float]] = {label: [] for label in CHOICES}
    exact_windows = exact_values = argmax = 0
    for window_id in sorted(a_by_id):
        left, right = a_by_id[window_id], b_by_id[window_id]
        deltas = {key: abs(left[key] - right[key]) for key in ("log_likelihood_A", "log_likelihood_B", "log_likelihood_C", "log_likelihood_D", "support_margin")}
        exact = all(value == 0.0 for value in deltas.values())
        exact_windows += int(exact); exact_values += sum(value == 0.0 for value in deltas.values()); argmax += int(left["choice_argmax"] == right["choice_argmax"])
        values.extend(deltas.values()); support.append(deltas["support_margin"])
        for label, key in zip(CHOICES, ("log_likelihood_A", "log_likelihood_B", "log_likelihood_C", "log_likelihood_D")):
            per_choice[label].append(deltas[key])
        differences.append({"window_id": window_id, "run_a": left, "run_b": right, "absolute_difference": deltas,
                            "choice_argmax_equal": left["choice_argmax"] == right["choice_argmax"], "numeric_exact": exact})
    a_rank, b_rank = _ranking_projection(a["ranking"]), _ranking_projection(b["ranking"])
    nms_keys = ("window_id", "nms_status", "nms_suppressor_window_id", "nms_temporal_iou")
    nms_exact = [{key: row[key] for key in nms_keys} for row in a_rank] == [{key: row[key] for key in nms_keys} for row in b_rank]
    selected_a = [row["window_id"] for row in a_rank if row["nms_status"] == "SELECTED"]
    selected_b = [row["window_id"] for row in b_rank if row["nms_status"] == "SELECTED"]
    a_hyp, b_hyp = _hypothesis_projection(a["hypotheses"]), _hypothesis_projection(b["hypotheses"])
    a_graph, b_graph = _graph_projection(a["graph"]), _graph_projection(b["graph"])
    no_visible_a = any(row.get("polarity") == "NO_VISIBLE_EVENT" and row.get("status") == "CANDIDATE_UNVERIFIED" for row in a_hyp)
    no_visible_b = any(row.get("polarity") == "NO_VISIBLE_EVENT" and row.get("status") == "CANDIDATE_UNVERIFIED" for row in b_hyp)
    claims_valid = all(row.get("status") == "CANDIDATE_UNVERIFIED" for row in a_hyp + b_hyp)
    numeric_hash_a, numeric_hash_b = stable_hash(a_numeric), stable_hash(b_numeric)
    return {
        "window_count_compared": len(differences), "numeric_results_sha256_equal": numeric_hash_a == numeric_hash_b,
        "numeric_results_sha256_a": numeric_hash_a, "numeric_results_sha256_b": numeric_hash_b,
        "exact_numeric_window_count": exact_windows, "exact_numeric_value_count": exact_values,
        "max_absolute_difference": max(values), "mean_absolute_difference": sum(values) / len(values),
        "per_choice_max_absolute_difference": {key: max(items) for key, items in per_choice.items()},
        "support_margin_max_absolute_difference": max(support), "support_margin_mean_absolute_difference": sum(support) / len(support),
        "choice_argmax_agreement_count": argmax, "choice_argmax_agreement_rate": argmax / len(differences),
        "window_ranking_exact_match": a_rank == b_rank, "first_ranking_mismatch": _first_mismatch(a_rank, b_rank),
        "nms_projection_exact_match": nms_exact, "selected_positive_window_ids_exact_match": selected_a == selected_b,
        "selected_positive_window_ids_a": selected_a, "selected_positive_window_ids_b": selected_b,
        "hypothesis_claims_byte_hash_equal": _sha(a["root"] / "run" / "v2_stage3b_hypothesis_claims.jsonl") == _sha(b["root"] / "run" / "v2_stage3b_hypothesis_claims.jsonl"),
        "hypothesis_semantic_projection_equal": a_hyp == b_hyp, "positive_hypothesis_count_a": sum(row.get("polarity") == "POSITIVE" for row in a_hyp),
        "positive_hypothesis_count_b": sum(row.get("polarity") == "POSITIVE" for row in b_hyp), "no_visible_event_present_a": no_visible_a,
        "no_visible_event_present_b": no_visible_b, "hypothesis_statuses_valid": claims_valid,
        "claim_graph_byte_hash_equal": _sha(a["root"] / "run" / "v2_stage3b_claim_graph.json") == _sha(b["root"] / "run" / "v2_stage3b_claim_graph.json"),
        "claim_graph_structural_projection_equal": a_graph == b_graph,
    }, differences


def compare_independent_runs(*, run_a_dir: Path, run_b_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Compare two distinct Stage 3B fresh runs without opening a model or cache."""
    if output_dir.exists():
        raise IndependentRepeatError("COMPARISON_OUTPUT_DIRECTORY_MUST_BE_NEW")
    a, b = _load_bundle(run_a_dir), _load_bundle(run_b_dir)
    inputs_a, inputs_b = _scientific_inputs(a), _scientific_inputs(b)
    base = {"format": FORMAT, "run_a_manifest_sha256": _sha(run_a_dir / "v2_stage3b_run_summary.json"),
            "run_b_manifest_sha256": _sha(run_b_dir / "v2_stage3b_run_summary.json"),
            "scientific_input_projection_sha256": stable_hash(inputs_a), "model_calls_made": 0, "backend_loaded": False,
            "cache_opened": False, "gt_used": False, "certificate_created": False, "new_verified_count": 0,
            "certificate_status": "NOT_APPLICABLE"}
    output_dir.mkdir(parents=True, exist_ok=False)
    if inputs_a != inputs_b:
        summary = base | {"status": "FAIL", "reason": "INDEPENDENT_RUN_INPUT_MISMATCH", "reproducibility_status": "NOT_COMPARABLE_INPUT_MISMATCH", "ready_for_stage3c": False}
        differences: list[dict[str, Any]] = []
    else:
        fresh_a, fresh_b = _fresh_audit(a, distinct_root=run_a_dir.resolve() != run_b_dir.resolve()), _fresh_audit(b, distinct_root=run_a_dir.resolve() != run_b_dir.resolve())
        if not fresh_a["independent_fresh_run"] or not fresh_b["independent_fresh_run"]:
            summary = base | {"status": "FAIL", "reason": "NOT_AN_INDEPENDENT_FRESH_RUN", "reproducibility_status": "FRESH_RUN_CONTRACT_FAILED", "ready_for_stage3c": False,
                              "fresh_run_a": fresh_a, "fresh_run_b": fresh_b}
            differences = []
        else:
            metrics, differences = _compare_ready(a, b)
            exact = all(metrics[key] for key in ("numeric_results_sha256_equal", "window_ranking_exact_match", "nms_projection_exact_match", "hypothesis_semantic_projection_equal", "claim_graph_structural_projection_equal"))
            discrete = (metrics["choice_argmax_agreement_rate"] == 1.0 and metrics["window_ranking_exact_match"] and metrics["nms_projection_exact_match"]
                        and metrics["selected_positive_window_ids_exact_match"] and metrics["hypothesis_semantic_projection_equal"]
                        and metrics["claim_graph_structural_projection_equal"] and metrics["no_visible_event_present_a"] and metrics["no_visible_event_present_b"] and metrics["hypothesis_statuses_valid"])
            if exact:
                decision = {"status": "PASS", "reproducibility_status": "EXACT_NUMERIC_AND_DISCRETE_REPRODUCTION", "ready_for_stage3c": True}
            elif discrete:
                decision = {"status": "PASS_WITH_NUMERIC_VARIATION", "reproducibility_status": "DISCRETE_PIPELINE_STABLE_NUMERIC_VARIATION_REPORTED", "ready_for_stage3c": True}
            else:
                decision = {"status": "UNSTABLE", "reproducibility_status": "INDEPENDENT_FRESH_RUN_DISCRETE_INSTABILITY", "ready_for_stage3c": False}
            summary = base | fresh_a | {"fresh_run_b": fresh_b} | metrics | decision
    summary["comparison_content_sha256"] = stable_hash(summary)
    audit = {"format": FORMAT, "status": summary["status"], "run_a_manifest_sha256": base["run_a_manifest_sha256"],
             "run_b_manifest_sha256": base["run_b_manifest_sha256"], "scientific_input_projection_sha256": base["scientific_input_projection_sha256"],
             "allowed_inputs_opened": ["Stage 3B frozen run artifacts only"],
             "forbidden_inputs_not_opened": ["backend", "inference cache", "public frames", "raw dataset", "assistant answer", "reference answer", "temporal GT", "bbox", "mask", "ROI", "certificate"],
             "model_calls_made": 0, "backend_loaded": False, "cache_opened": False, "hypothesis_created": False,
             "certificate_created": False, "new_verified_count": 0, "gt_used": False}
    audit["audit_content_sha256"] = stable_hash(audit)
    _write(output_dir / "v2_stage3b_independent_repeat_comparison.json", summary)
    _write(output_dir / "v2_stage3b_independent_repeat_window_differences.jsonl", differences)
    _write(output_dir / "v2_stage3b_independent_repeat_audit.json", audit)
    return summary


def safe_candidate_summary(*, run_dir: Path) -> dict[str, Any]:
    """Return only non-GT Stage 3B candidate structure and likelihood values."""
    bundle = _load_bundle(run_dir)
    numeric = {row["window_id"]: row for row in _numeric_rows(bundle["likelihoods"])}
    ranking = _ranking_projection(bundle["ranking"])
    windows = []
    for row in ranking:
        score = numeric[row["window_id"]]
        windows.append({"window_id": row["window_id"], "role": row["window_role"], "scale_seconds": row["scale_seconds"],
                        "start_seconds": row["start_seconds"], "end_seconds": row["end_seconds"],
                        "log_likelihood_A": score["log_likelihood_A"], "log_likelihood_B": score["log_likelihood_B"],
                        "log_likelihood_C": score["log_likelihood_C"], "log_likelihood_D": score["log_likelihood_D"],
                        "support_margin": score["support_margin"], "retrieval_rank": row["rank"], "nms_status": row["nms_status"],
                        "nms_suppressor_window_id": row["nms_suppressor_window_id"]})
    positives = [{key: item.get(key) for key in ("hypothesis_id", "candidate_interval", "supporting_window_ids", "retrieval_rank", "coarse_support_margin")}
                 for item in bundle["hypotheses"] if item.get("polarity") == "POSITIVE"]
    nulls = [{key: item.get(key) for key in ("hypothesis_id", "polarity", "status")}
             for item in bundle["hypotheses"] if item.get("polarity") == "NO_VISIBLE_EVENT"]
    return {"format": "relive-v2-tal-stage3b-safe-candidate-summary-v1", "status": "PASS", "window_count": len(windows),
            "windows": windows, "positive_hypotheses": sorted(positives, key=lambda item: item.get("retrieval_rank") or 10**9),
            "no_visible_event": nulls, "model_calls_made": 0, "backend_loaded": False, "cache_opened": False,
            "gt_used": False, "certificate_created": False, "new_verified_count": 0, "certificate_status": "NOT_APPLICABLE"}
