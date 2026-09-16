"""ReliVE-v2 Stage 3B: frozen-window, coarse hypothesis generation only.

The module has no free-text temporal answer path.  It scores exactly the
immutable Stage 3A windows with an A/B/C/D next-token likelihood protocol and
emits only immutable ``CANDIDATE_UNVERIFIED`` alternatives.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import time
from typing import Any, Callable, Mapping
from importlib.resources import files

from relive.backends import make_backend
from relive.config import load_config
from relive.storage.artifacts import ArtifactStore, canonical_json, stable_hash
from .task_selection import TALSelectionError, strict_json_loads, strict_jsonl
from .temporal_search_plan import (TemporalSearchPlanError, validate_hypothesis_claim,
                                   validate_temporal_search_plan_artifacts)
from .video_index import VideoIndexError, validate_video_index_artifacts


FORMAT = "relive-v2-tal-coarse-hypothesis-generation-v1"
PROMPT_VERSION = "relive-v2-tal-coarse-choice-v1"
CHOICES = ("A", "B", "C", "D")
ORDINARY_LIMIT, FULL_LIMIT = 8, 16
PLANNED_CALLS = 23
class CoarseHypothesisError(ValueError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: bytes) -> None:
    if path.exists():
        raise CoarseHypothesisError("IMMUTABLE_OUTPUT_EXISTS")
    path.write_bytes(payload)


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join((canonical_json(row) + "\n").encode("utf-8") for row in rows)


def _object(path: Path, code: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes(); value = strict_json_loads(raw.decode("utf-8"), error_code=code)
    except (OSError, UnicodeDecodeError, TALSelectionError) as exc:
        raise CoarseHypothesisError(f"{code}_INVALID") from exc
    if not isinstance(value, dict) or raw != (canonical_json(value) + "\n").encode("utf-8"):
        raise CoarseHypothesisError(f"{code}_NONCANONICAL_OR_NONOBJECT")
    return value


def _rows(path: Path, code: str) -> tuple[dict[str, Any], ...]:
    try:
        raw = path.read_bytes(); result = strict_jsonl(path, error_code=code)
    except (OSError, TALSelectionError) as exc:
        raise CoarseHypothesisError(f"{code}_INVALID") from exc
    if raw != _jsonl(list(result)):
        raise CoarseHypothesisError(f"{code}_NONCANONICAL_BYTES")
    return result


def _finite(value: Any, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise CoarseHypothesisError(code)
    return float(value)


def support_margin(log_probabilities: Mapping[str, Any]) -> float:
    """Frozen Stage 3B margin: logP(A)-logsumexp(B,C,D)."""
    if set(log_probabilities) != set(CHOICES):
        raise CoarseHypothesisError("CHOICE_LOG_PROBABILITIES_INVALID")
    values = {key: _finite(value, "CHOICE_LOG_PROBABILITIES_INVALID") for key, value in log_probabilities.items()}
    rest = (values["B"], values["C"], values["D"]); maximum = max(rest)
    return values["A"] - (maximum + math.log(sum(math.exp(item - maximum) for item in rest)))


def temporal_iou(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    start_a, end_a = _finite(left["start_seconds"], "TEMPORAL_INTERVAL_INVALID"), _finite(left["end_seconds"], "TEMPORAL_INTERVAL_INVALID")
    start_b, end_b = _finite(right["start_seconds"], "TEMPORAL_INTERVAL_INVALID"), _finite(right["end_seconds"], "TEMPORAL_INTERVAL_INVALID")
    if end_a < start_a or end_b < start_b:
        raise CoarseHypothesisError("TEMPORAL_INTERVAL_INVALID")
    union = max(end_a, end_b) - min(start_a, start_b)
    return 0.0 if union == 0 else max(0.0, min(end_a, end_b) - max(start_a, start_b)) / union


def _prompt(*, target_event: str, evidence: list[str], start: float, end: float, timestamps: list[float]) -> str:
    template = files("relive").joinpath("prompts", "v2_tal_coarse_choice.txt").read_text(encoding="utf-8")
    text = template.format(target_event=target_event, required_evidence="\n".join(f"- {item.lower().replace('_', ' ')}" for item in evidence),
                           window_start=start, window_end=end, frame_timestamps=", ".join(format(item, ".12g") for item in timestamps)).rstrip()
    if not text.endswith("\nAnswer:"):
        raise CoarseHypothesisError("COARSE_PROMPT_ANCHOR_INVALID")
    return text


def _packet_indices(count: int, limit: int) -> list[int]:
    if type(count) is not int or count < 2 or type(limit) is not int or limit < 2:
        raise CoarseHypothesisError("PACKET_FRAME_COUNT_INVALID")
    if count <= limit:
        return list(range(count))
    denominator = limit - 1
    indices = [(slot * (count - 1) + denominator // 2) // denominator for slot in range(limit)]
    if indices[0] != 0 or indices[-1] != count - 1 or len(set(indices)) != limit:
        raise CoarseHypothesisError("PACKET_SELECTION_NOT_UNIQUE")
    return indices


def _input_hashes(stage3a_dir: Path, video_index_dir: Path, timestamp_path: Path, provenance_path: Path) -> dict[str, str]:
    try:
        validate_temporal_search_plan_artifacts(stage3a_dir)
        validated = validate_video_index_artifacts(video_index_dir, materialize_frames=False)
    except (TemporalSearchPlanError, VideoIndexError) as exc:
        raise CoarseHypothesisError("FROZEN_UPSTREAM_ARTIFACT_INVALID") from exc
    if validated.get("index_status") != "RESOLVED_DATASET_NATIVE_CLIP_LOCAL":
        raise CoarseHypothesisError("VIDEO_INDEX_NOT_CLIP_LOCAL_RESOLVED")
    manifest = _object(stage3a_dir / "v2_temporal_search_manifest.json", "STAGE3A_MANIFEST")
    required = {"video_index_manifest_sha256", "timestamp_manifest_sha256", "timestamp_provenance_sha256", "search_plan_sha256", "hypothesis_contract_sha256", "claim_graph_schema_sha256", "temporal_policy_sha256"}
    if not required.issubset(manifest) or manifest["video_index_manifest_sha256"] != _sha(video_index_dir / "v2_video_index_manifest.json") or manifest["timestamp_manifest_sha256"] != _sha(timestamp_path) or manifest["timestamp_provenance_sha256"] != _sha(provenance_path):
        raise CoarseHypothesisError("FROZEN_UPSTREAM_BINDING_MISMATCH")
    return {"stage3a_manifest_sha256": _sha(stage3a_dir / "v2_temporal_search_manifest.json"),
            "requirement_manifest_sha256": manifest["requirement_manifest_sha256"], "selection_manifest_sha256": manifest["selection_manifest_sha256"],
            "video_index_manifest_sha256": manifest["video_index_manifest_sha256"], "timestamp_manifest_sha256": manifest["timestamp_manifest_sha256"],
            "timestamp_provenance_sha256": manifest["timestamp_provenance_sha256"], "claim_graph_schema_sha256": manifest["claim_graph_schema_sha256"],
            "hypothesis_contract_sha256": manifest["hypothesis_contract_sha256"], "temporal_policy_sha256": manifest["temporal_policy_sha256"],
            "search_plan_sha256": manifest["search_plan_sha256"]}


def _build_packets(*, stage3a_dir: Path, video_index_dir: Path, requirement_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    search = _rows(stage3a_dir / "v2_temporal_search_plan.jsonl", "STAGE3A_SEARCH_PLAN")
    indexes = _rows(video_index_dir / "v2_video_index.jsonl", "VIDEO_INDEX")
    if len(indexes) != 1 or len(search) != PLANNED_CALLS:
        raise CoarseHypothesisError("STAGE3B_REQUIRES_EXACTLY_23_FROZEN_WINDOWS")
    index = indexes[0]
    frames = {item["frame_order"]: item for item in index.get("frames", [])}
    if len(frames) != index.get("frame_count"):
        raise CoarseHypothesisError("VIDEO_INDEX_FRAME_SCHEMA_INVALID")
    specs = _rows(requirement_dir / "v2_requirement_specs.jsonl", "REQUIREMENT_SPECS")
    if len(specs) != 1:
        raise CoarseHypothesisError("STAGE3B_REQUIRES_ONE_FROZEN_REQUIREMENT")
    spec = specs[0]
    requirement_id = index["requirement_id"]
    target_event, required_evidence = spec.get("target_event"), spec.get("required_evidence")
    if not isinstance(target_event, str) or not target_event or not isinstance(required_evidence, list) or not required_evidence or any(not isinstance(item, str) for item in required_evidence):
        raise CoarseHypothesisError("REQUIREMENT_SEMANTICS_INVALID")
    logical_aliases: dict[str, list[int]] = {}
    for item in frames.values():
        key = stable_hash({"frame_sha256": item["frame_sha256"], "source_frame_reference": item["source_frame_reference"], "timestamp_seconds": item["timestamp_seconds"]})
        logical_aliases.setdefault(key, []).append(item["frame_order"])
    packets, calls = [], []
    for window in search:
        orders = window["included_logical_frame_orders"]
        unique: list[dict[str, Any]] = []
        seen = set()
        for order in orders:
            frame = frames.get(order)
            if frame is None:
                raise CoarseHypothesisError("WINDOW_FRAME_MEMBERSHIP_INVALID")
            visual = stable_hash({"frame_sha256": frame["frame_sha256"], "source_frame_reference": frame["source_frame_reference"], "timestamp_seconds": frame["timestamp_seconds"]})
            if visual not in seen:
                seen.add(visual)
                unique.append({"unique_visual_key": visual, **frame})
        limit = FULL_LIMIT if window["window_role"] == "FULL_CLIP" else ORDINARY_LIMIT
        selected = [unique[position] for position in _packet_indices(len(unique), limit)]
        frame_rows = [{"unique_visual_frame_id": "unique_visual_frame_" + item["unique_visual_key"][:24], "logical_aliases": logical_aliases[item["unique_visual_key"]],
                       "logical_frame_order": item["frame_order"], "source_frame_reference": item["source_frame_reference"], "frame_sha256": item["frame_sha256"],
                       "resolved_frame_path": item["resolved_frame_path"], "timestamp_seconds": item["timestamp_seconds"]} for item in selected]
        prompt = _prompt(target_event=target_event, evidence=required_evidence, start=window["start_seconds"], end=window["end_seconds"], timestamps=[item["timestamp_seconds"] for item in frame_rows])
        packet = {"window_id": window["window_id"], "window_role": "GLOBAL_CONTEXT_DIAGNOSTIC" if window["window_role"] == "FULL_CLIP" else "ORDINARY_TEMPORAL_WINDOW",
                  "start_seconds": window["start_seconds"], "end_seconds": window["end_seconds"], "scale_seconds": window["scale_seconds"],
                  "requirement_id": requirement_id, "frame_selection_policy": "chronological_endpoints_uniform_interior_v1", "maximum_unique_frames": limit,
                  "duplicate_visual_frames_removed_from_packet": True, "frames": frame_rows}
        packet["packet_id"] = "visual_packet_" + stable_hash(packet)[:24]
        packets.append(packet)
        calls.append({"window_id": window["window_id"], "packet_id": packet["packet_id"], "requirement_id": requirement_id,
                      "prompt": prompt, "prompt_version": PROMPT_VERSION, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                      "image_paths": [item["resolved_frame_path"] for item in frame_rows], "frame_ids": [item["unique_visual_frame_id"] for item in frame_rows],
                      "frame_sha256": [item["frame_sha256"] for item in frame_rows], "context": {"stage": "v2_tal_coarse_hypothesis", "window_id": window["window_id"], "packet_id": packet["packet_id"], "requirement_id": requirement_id}})
    if len(packets) != PLANNED_CALLS or len({item["packet_id"] for item in packets}) != PLANNED_CALLS:
        raise CoarseHypothesisError("VISUAL_PACKET_COUNT_OR_ID_INVALID")
    return packets, calls, {"target_event": target_event, "required_evidence": required_evidence, "requirement_id": requirement_id}


def _ranking_policy() -> dict[str, Any]:
    return {"format": FORMAT, "policy_version": "relive-v2-tal-coarse-ranking-nms-v1", "candidate_generation_basis": "RANK_ONLY_NO_ADMISSION_THRESHOLD",
            "ranking": ["support_margin_desc", "shorter_window_first", "start_seconds_asc", "window_id_asc"],
            "temporal_iou_suppression_threshold": 0.5, "max_positive_hypotheses": 5,
            "include_no_visible_event_hypothesis": True, "max_total_hypotheses": 6,
            "full_clip_positive_hypothesis_eligible": False}


def _choice_spec() -> dict[str, Any]:
    return {"format": FORMAT, "prompt_version": PROMPT_VERSION, "choice_labels": list(CHOICES),
            "choice_semantics": {"A": "clear visible evidence of target event within this time window", "B": "related, preparatory, partial, or post-event evidence only", "C": "no visible evidence relevant to target event", "D": "insufficient or unreadable frames"},
            "support_margin_formula": "logP(A)-logsumexp(logP(B),logP(C),logP(D))", "free_text_generation_allowed": False}


def prepare(*, requirement_dir: Path, selection_manifest_path: Path, stage3a_dir: Path, video_index_dir: Path,
            timestamp_manifest_path: Path, timestamp_provenance_path: Path, config_path: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise CoarseHypothesisError("OUTPUT_DIRECTORY_MUST_BE_EMPTY")
    inputs = _input_hashes(stage3a_dir, video_index_dir, timestamp_manifest_path, timestamp_provenance_path)
    if _sha(requirement_dir / "v2_requirement_manifest.json") != inputs["requirement_manifest_sha256"] or _sha(selection_manifest_path) != inputs["selection_manifest_sha256"]:
        raise CoarseHypothesisError("REQUIREMENT_OR_SELECTION_BINDING_MISMATCH")
    config = load_config(config_path)
    if config.get("backend", {}).get("kind") != "local_hf":
        raise CoarseHypothesisError("STAGE3B_REQUIRES_VERSIONED_LOCAL_HF_CONFIG")
    packets, calls, requirement = _build_packets(stage3a_dir=stage3a_dir, video_index_dir=video_index_dir, requirement_dir=requirement_dir)
    output_dir.mkdir(parents=True)
    prompt_spec, choice_spec, rank = {"format": FORMAT, "prompt_version": PROMPT_VERSION, **requirement}, _choice_spec(), _ranking_policy()
    _write(output_dir / "v2_stage3b_visual_packets.jsonl", _jsonl(packets))
    _write(output_dir / "v2_stage3b_prompt_spec.json", (canonical_json(prompt_spec) + "\n").encode())
    _write(output_dir / "v2_stage3b_choice_spec.json", (canonical_json(choice_spec) + "\n").encode())
    _write(output_dir / "v2_stage3b_ranking_policy.json", (canonical_json(rank) + "\n").encode())
    call_plan = {"format": FORMAT, "status": "FROZEN_PRE_MODEL", "planned_model_calls": PLANNED_CALLS, "calls": calls,
                 "input_hashes": inputs, "config_sha256": _sha(config_path), "requirement_specs_sha256": _sha(requirement_dir / "v2_requirement_specs.jsonl"),
                 "model_configuration": config["backend"],
                 "visual_packets_sha256": _sha(output_dir / "v2_stage3b_visual_packets.jsonl"), "prompt_spec_sha256": _sha(output_dir / "v2_stage3b_prompt_spec.json"),
                 "choice_spec_sha256": _sha(output_dir / "v2_stage3b_choice_spec.json"), "ranking_policy_sha256": _sha(output_dir / "v2_stage3b_ranking_policy.json"),
                 "model_calls_made": 0, "backend_loaded": False, "cache_opened": False, "hypothesis_count": 0,
                 "observation_claim_count": 0, "claim_graph_created": False, "gt_used": False, "certificate_created": False,
                 "new_verified_count": 0, "certificate_status": "NOT_APPLICABLE"}
    call_plan["call_plan_content_sha256"] = stable_hash(call_plan)
    _write(output_dir / "v2_stage3b_call_plan.json", (canonical_json(call_plan) + "\n").encode())
    # Produced before any backend or cache access.  This binds the sole
    # permitted Stage 3B inputs and every frozen pre-model contract.
    audit = {
        "format": FORMAT, "status": "PASS", "mode": "prepare",
        "allowed_inputs_opened": [
            "frozen RequirementSpec artifacts", "frozen selection manifest",
            "frozen Stage 2 VideoIndex and timestamp provenance",
            "frozen Stage 3A temporal search plan", "reviewed local-HF configuration",
        ],
        "allowed_inputs_permitted_for_run": ["selected public frame bytes"],
        "forbidden_inputs_not_opened": [
            "assistant answer", "reference answer", "temporal GT", "bbox", "mask", "ROI",
            "model output", "certificate", "observation claim", "raw training source container",
        ],
        "input_hashes": inputs,
        "artifact_sha256": {
            "v2_stage3b_visual_packets.jsonl": _sha(output_dir / "v2_stage3b_visual_packets.jsonl"),
            "v2_stage3b_prompt_spec.json": _sha(output_dir / "v2_stage3b_prompt_spec.json"),
            "v2_stage3b_choice_spec.json": _sha(output_dir / "v2_stage3b_choice_spec.json"),
            "v2_stage3b_ranking_policy.json": _sha(output_dir / "v2_stage3b_ranking_policy.json"),
            "v2_stage3b_call_plan.json": _sha(output_dir / "v2_stage3b_call_plan.json"),
        },
        "model_calls_made": 0, "backend_loaded": False, "cache_opened": False,
        "observation_claim_count": 0, "certificate_created": False,
        "new_verified_count": 0, "certificate_status": "NOT_APPLICABLE", "gt_used": False,
    }
    audit["audit_content_sha256"] = stable_hash(audit)
    _write(output_dir / "v2_stage3b_freeze_audit.json", (canonical_json(audit) + "\n").encode())
    summary = {"format": FORMAT, "status": "PASS", "mode": "prepare", "stage_status": "FROZEN_PRE_MODEL", "window_count": PLANNED_CALLS,
               "packet_count": PLANNED_CALLS, "planned_model_calls": PLANNED_CALLS, "model_calls_made": 0, "backend_loaded": False,
               "cache_opened": False, "hypothesis_count": 0, "observation_claim_count": 0, "claim_graph_created": False,
               "gt_used": False, "certificate_created": False, "new_verified_count": 0, "certificate_status": "NOT_APPLICABLE"}
    _write(output_dir / "v2_stage3b_prepare.json", (canonical_json(summary) + "\n").encode())
    return summary


def _prepared(output_dir: Path, config_path: Path) -> tuple[dict[str, Any], tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], dict[str, Any]]:
    call_plan = _object(output_dir / "v2_stage3b_call_plan.json", "CALL_PLAN")
    packets = _rows(output_dir / "v2_stage3b_visual_packets.jsonl", "VISUAL_PACKETS")
    calls = tuple(call_plan.get("calls", [])); ranking = _object(output_dir / "v2_stage3b_ranking_policy.json", "RANKING_POLICY")
    if call_plan.get("call_plan_content_sha256") != stable_hash({key: value for key, value in call_plan.items() if key != "call_plan_content_sha256"}) or call_plan.get("config_sha256") != _sha(config_path) or len(packets) != PLANNED_CALLS or len(calls) != PLANNED_CALLS or call_plan.get("planned_model_calls") != PLANNED_CALLS:
        raise CoarseHypothesisError("PREPARED_STAGE3B_ARTIFACT_DRIFT")
    for key, name in (("visual_packets_sha256", "v2_stage3b_visual_packets.jsonl"), ("prompt_spec_sha256", "v2_stage3b_prompt_spec.json"), ("choice_spec_sha256", "v2_stage3b_choice_spec.json"), ("ranking_policy_sha256", "v2_stage3b_ranking_policy.json")):
        if call_plan.get(key) != _sha(output_dir / name):
            raise CoarseHypothesisError("PREPARED_STAGE3B_ARTIFACT_DRIFT")
    audit = _object(output_dir / "v2_stage3b_freeze_audit.json", "FREEZE_AUDIT")
    if audit.get("audit_content_sha256") != stable_hash({key: value for key, value in audit.items() if key != "audit_content_sha256"}):
        raise CoarseHypothesisError("FREEZE_AUDIT_HASH_MISMATCH")
    for name, expected in audit.get("artifact_sha256", {}).items():
        if _sha(output_dir / name) != expected:
            raise CoarseHypothesisError("FREEZE_AUDIT_ARTIFACT_DRIFT")
    return call_plan, packets, calls, ranking


def preflight(*, output_dir: Path, config_path: Path, backend_factory: Callable[[dict[str, Any]], Any] = make_backend) -> dict[str, Any]:
    if (output_dir / "v2_stage3b_preflight.json").exists():
        raise CoarseHypothesisError("PREFLIGHT_ALREADY_EXISTS")
    call_plan, packets, calls, _ = _prepared(output_dir, config_path)
    cache_dir = output_dir / "cache"
    if cache_dir.exists() and any(cache_dir.iterdir()):
        raise CoarseHypothesisError("PREFLIGHT_REQUIRES_EMPTY_CACHE")
    backend = backend_factory(load_config(config_path)["backend"])
    contracts = {}
    for call in calls:
        contract = backend.forced_choice_token_contract(call, choices=CHOICES)
        if contract.get("choice_labels") != list(CHOICES) or set(contract.get("choice_token_ids", {})) != set(CHOICES) or len(set(contract["choice_token_ids"].values())) != 4:
            raise CoarseHypothesisError("ABCD_NEXT_TOKEN_CONTRACT_NOT_UNIQUE")
        contracts[call["packet_id"]] = contract
    report = {"format": FORMAT, "status": "PASS", "mode": "preflight", "planned_model_calls": PLANNED_CALLS, "model_calls_made": 0,
              "backend_loaded": True, "cache_opened": False, "cache_empty": True, "window_count": len(packets), "packet_count": len(packets),
              "model_fingerprint": backend.fingerprint(), "choice_token_contracts": contracts, "call_plan_sha256": _sha(output_dir / "v2_stage3b_call_plan.json"),
              "hypothesis_count": 0, "observation_claim_count": 0, "claim_graph_created": False, "gt_used": False,
              "certificate_created": False, "new_verified_count": 0, "certificate_status": "NOT_APPLICABLE"}
    report["preflight_content_sha256"] = stable_hash(report)
    _write(output_dir / "v2_stage3b_preflight.json", (canonical_json(report) + "\n").encode())
    return report


class ChoiceCache:
    def __init__(self, backend: Any, root: Path): self.backend, self.store, self.new_calls, self.cache_hits, self.logical_calls = backend, ArtifactStore(root), 0, 0, 0
    def call(self, request: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
        token_ids = contract["choice_token_ids"]
        identity = {"cache_version": "relive-v2-tal-coarse-choice-cache-v1", "model_fingerprint": self.backend.fingerprint(), "prompt_sha256": request["prompt_sha256"],
                    "choice_token_ids": token_ids, "window_id": request["window_id"], "packet_id": request["packet_id"], "requirement_id": request["requirement_id"],
                    "frame_sha256": request["frame_sha256"], "frame_ids": request["frame_ids"]}
        key = stable_hash(identity); self.logical_calls += 1
        with self.store.run_lock():
            path = self.store.path("v2_tal_coarse_choice", key)
            if path.exists():
                value = self.store.get_json(str(path))
                if value.get("identity") != identity: raise CoarseHypothesisError("CHOICE_CACHE_IDENTITY_COLLISION")
                self.cache_hits += 1; return {**value["result"], "cache_key": key, "cache_hit": True}
            result = self.backend.forced_choice_likelihood(request, choice_token_ids=token_ids, choices=CHOICES)
            if result.get("choice_contract", {}).get("choice_token_ids") != token_ids: raise CoarseHypothesisError("CHOICE_TOKEN_CONTRACT_DRIFT")
            self.store.put_json("v2_tal_coarse_choice", key, {"identity": identity, "result": result})
            self.new_calls += 1; return {**result, "cache_key": key, "cache_hit": False}


def _rank(rows: list[dict[str, Any]], policy: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordinary = [row for row in rows if row["positive_hypothesis_eligible"]]
    ordered = sorted(ordinary, key=lambda row: (-row["support_margin"], row["scale_seconds"], row["start_seconds"], row["window_id"]))
    selected = []
    for rank, row in enumerate(ordered, start=1):
        row["rank"] = rank; suppressor = next((item for item in selected if temporal_iou(row, item) >= policy["temporal_iou_suppression_threshold"]), None)
        if suppressor is not None:
            row["nms_status"] = "SUPPRESSED_CROSS_SCALE_DUPLICATE"; row["nms_suppressor_window_id"] = suppressor["window_id"]
        elif len(selected) < policy["max_positive_hypotheses"]:
            row["nms_status"] = "SELECTED"; row["nms_suppressor_window_id"] = None; selected.append(row)
        else:
            row["nms_status"] = "NOT_SELECTED_BUDGET"; row["nms_suppressor_window_id"] = None
    for row in rows:
        if not row["positive_hypothesis_eligible"]:
            row["rank"] = None; row["nms_status"] = "GLOBAL_CONTEXT_NOT_ELIGIBLE"; row["nms_suppressor_window_id"] = None
    return rows, selected


def _hypotheses(selected: list[dict[str, Any]], requirement_id: str, target_event: str) -> list[dict[str, Any]]:
    output = []
    for row in selected:
        payload = {"requirement_id": requirement_id, "parent_claim_id": None, "claim_role": "TARGET_HYPOTHESIS", "target_event": target_event,
                   "polarity": "POSITIVE", "candidate_interval": {"start_seconds": row["start_seconds"], "end_seconds": row["end_seconds"], "timestamp_domain": "CLIP_LOCAL"},
                   "generation_source": "CLAIM_CONDITIONED_COARSE_RETRIEVAL", "status": "CANDIDATE_UNVERIFIED", "supporting_window_ids": [row["window_id"]],
                   "retrieval_rank": row["rank"], "coarse_support_margin": row["support_margin"], "provenance": {"packet_id": row["packet_id"], "ranking_policy": "relive-v2-tal-coarse-ranking-nms-v1"}}
        payload["hypothesis_id"] = "hypothesis_" + stable_hash(payload)[:24]; validate_hypothesis_claim(payload); output.append(payload)
    null = {"requirement_id": requirement_id, "parent_claim_id": None, "claim_role": "TARGET_HYPOTHESIS", "target_event": target_event,
            "polarity": "NO_VISIBLE_EVENT", "candidate_interval": None, "generation_source": "CLAIM_CONDITIONED_COARSE_RETRIEVAL",
            "status": "CANDIDATE_UNVERIFIED", "supporting_window_ids": [], "retrieval_rank": None, "coarse_support_margin": None,
            "provenance": {"alternative_policy": "always_include_no_visible_event_v1"}}
    null["hypothesis_id"] = "hypothesis_" + stable_hash(null)[:24]; validate_hypothesis_claim(null); output.append(null)
    return output


def _numeric(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: row[key] for key in ("window_id", "raw_logits", "log_probabilities", "support_margin", "rank", "nms_status", "nms_suppressor_window_id")} for row in rows]


def execute(*, output_dir: Path, config_path: Path, mode: str, backend_factory: Callable[[dict[str, Any]], Any] = make_backend) -> dict[str, Any]:
    if mode not in {"run", "replay"}: raise CoarseHypothesisError("RUN_MODE_INVALID")
    target = output_dir / mode
    if target.exists(): raise CoarseHypothesisError("RUN_OUTPUT_ALREADY_EXISTS")
    call_plan, packets, calls, policy = _prepared(output_dir, config_path)
    pre = _object(output_dir / "v2_stage3b_preflight.json", "PREFLIGHT")
    if pre.get("call_plan_sha256") != _sha(output_dir / "v2_stage3b_call_plan.json") or pre.get("preflight_content_sha256") != stable_hash({key: value for key, value in pre.items() if key != "preflight_content_sha256"}):
        raise CoarseHypothesisError("PREFLIGHT_ARTIFACT_DRIFT")
    backend = backend_factory(load_config(config_path)["backend"])
    if backend.fingerprint() != pre.get("model_fingerprint"): raise CoarseHypothesisError("MODEL_FINGERPRINT_DRIFT")
    cache, packet_by_id, rows = ChoiceCache(backend, output_dir / "cache"), {item["packet_id"]: item for item in packets}, []
    started = time.monotonic()
    for call in calls:
        contract = pre["choice_token_contracts"].get(call["packet_id"])
        if not isinstance(contract, dict): raise CoarseHypothesisError("PREFLIGHT_TOKEN_CONTRACT_MISSING")
        scored = cache.call(call, contract); choices = scored.get("choices", {})
        if set(choices) != set(CHOICES): raise CoarseHypothesisError("CHOICE_RESULT_SCHEMA_INVALID")
        logs = {label: choices[label].get("log_probability") for label in CHOICES}
        packet = packet_by_id[call["packet_id"]]
        rows.append({"window_id": call["window_id"], "window_role": packet["window_role"], "scale_seconds": packet["scale_seconds"], "start_seconds": packet["start_seconds"], "end_seconds": packet["end_seconds"],
                     "packet_id": packet["packet_id"], "frame_hashes": call["frame_sha256"], "choice_token_ids": scored["choice_contract"]["choice_token_ids"],
                     "raw_logits": {key: choices[key]["raw_logit"] for key in CHOICES}, "log_probabilities": logs, "support_margin": support_margin(logs),
                     "positive_hypothesis_eligible": packet["window_role"] != "GLOBAL_CONTEXT_DIAGNOSTIC", "rank": None, "nms_status": None, "nms_suppressor_window_id": None,
                     "cache_key": scored["cache_key"], "cache_hit": scored["cache_hit"]})
    if cache.logical_calls != PLANNED_CALLS or (mode == "run" and (cache.new_calls != PLANNED_CALLS or cache.cache_hits != 0)) or (mode == "replay" and (cache.new_calls != 0 or cache.cache_hits != PLANNED_CALLS)):
        raise CoarseHypothesisError("MODEL_CALL_BUDGET_OR_CACHE_REPLAY_MISMATCH")
    rows, selected = _rank(rows, policy)
    prompt = _object(output_dir / "v2_stage3b_prompt_spec.json", "PROMPT_SPEC")
    hypotheses = _hypotheses(selected, prompt["requirement_id"], prompt["target_event"])
    graph = {"format": FORMAT, "requirement_binding": {"requirement_id": prompt["requirement_id"], **call_plan["input_hashes"]}, "nodes": hypotheses,
             "observation_claim_count": 0, "verified_hypothesis_count": 0, "certificate_created": False}
    graph["claim_graph_sha256"] = stable_hash({key: value for key, value in graph.items() if key != "claim_graph_sha256"})
    target.mkdir(parents=True)
    _write(target / "v2_stage3b_window_likelihoods.jsonl", _jsonl(rows))
    # Cache hit bookkeeping differs by design on replay and therefore is not
    # part of the immutable ranking result or its replay-equivalence hash.
    ranking_rows = [{key: value for key, value in row.items() if key not in {"cache_key", "cache_hit"}}
                    for row in sorted(rows, key=lambda row: (row["rank"] is None, row["rank"] or 0, row["window_id"]))]
    _write(target / "v2_stage3b_window_ranking.jsonl", _jsonl(ranking_rows))
    _write(target / "v2_stage3b_hypothesis_claims.jsonl", _jsonl(hypotheses))
    _write(target / "v2_stage3b_claim_graph.json", (canonical_json(graph) + "\n").encode())
    likelihood_hash = _sha(target / "v2_stage3b_window_likelihoods.jsonl")
    numeric_hash, ranking_hash, hypothesis_hash = stable_hash(_numeric(rows)), _sha(target / "v2_stage3b_window_ranking.jsonl"), _sha(target / "v2_stage3b_hypothesis_claims.jsonl")
    summary = {"format": FORMAT, "status": "PASS", "mode": mode, "stage_status": "COARSE_HYPOTHESES_GENERATED_UNVERIFIED", "window_count": PLANNED_CALLS,
               "model_calls_made": cache.new_calls, "new_model_calls": cache.new_calls, "cache_hits": cache.cache_hits, "logical_model_calls": cache.logical_calls,
               "hypothesis_count": len(hypotheses), "observation_claim_count": 0, "claim_graph_created": True, "verified_hypothesis_count": 0,
               "certificate_created": False, "new_verified_count": 0, "certificate_status": "NOT_APPLICABLE", "gt_used": False,
               "numeric_results_sha256": numeric_hash, "window_likelihoods_sha256": likelihood_hash,
               "window_ranking_sha256": ranking_hash, "hypothesis_claims_sha256": hypothesis_hash,
               "claim_graph_sha256": graph["claim_graph_sha256"], "claim_graph_artifact_sha256": _sha(target / "v2_stage3b_claim_graph.json"), "latency_seconds": time.monotonic() - started,
               "ready_for_observation_claim_planning": True}
    if mode == "replay":
        first = _object(output_dir / "v2_stage3b_run_summary.json", "RUN_SUMMARY")
        for key in ("numeric_results_sha256", "window_ranking_sha256", "hypothesis_claims_sha256", "claim_graph_sha256"):
            if summary[key] != first.get(key): raise CoarseHypothesisError("REPLAY_RESULT_HASH_MISMATCH")
    _write(output_dir / f"v2_stage3b_{mode}_summary.json", (canonical_json(summary) + "\n").encode())
    return summary


def validate(output_dir: Path) -> dict[str, Any]:
    # Validator intentionally uses only emitted Stage 3B bytes; it never loads a model, cache, image, or upstream source.
    _object(output_dir / "v2_stage3b_preflight.json", "PREFLIGHT")
    audit = _object(output_dir / "v2_stage3b_freeze_audit.json", "FREEZE_AUDIT")
    run = _object(output_dir / "v2_stage3b_run_summary.json", "RUN_SUMMARY")
    likelihoods = _rows(output_dir / "run" / "v2_stage3b_window_likelihoods.jsonl", "LIKELIHOODS")
    ranking = _rows(output_dir / "run" / "v2_stage3b_window_ranking.jsonl", "RANKING")
    hypotheses = _rows(output_dir / "run" / "v2_stage3b_hypothesis_claims.jsonl", "HYPOTHESES")
    graph = _object(output_dir / "run" / "v2_stage3b_claim_graph.json", "CLAIM_GRAPH")
    if len(likelihoods) != PLANNED_CALLS or run.get("new_model_calls") != PLANNED_CALLS or run.get("cache_hits") != 0 or run.get("hypothesis_count") != len(hypotheses) or any(item.get("status") != "CANDIDATE_UNVERIFIED" for item in hypotheses) or graph.get("observation_claim_count") != 0 or graph.get("verified_hypothesis_count") != 0:
        raise CoarseHypothesisError("STAGE3B_RESULT_SCHEMA_INVALID")
    if graph.get("claim_graph_sha256") != stable_hash({key: value for key, value in graph.items() if key != "claim_graph_sha256"}):
        raise CoarseHypothesisError("CLAIM_GRAPH_HASH_MISMATCH")
    if audit.get("audit_content_sha256") != stable_hash({key: value for key, value in audit.items() if key != "audit_content_sha256"}):
        raise CoarseHypothesisError("FREEZE_AUDIT_HASH_MISMATCH")
    expected_files = {
        "window_likelihoods_sha256": output_dir / "run" / "v2_stage3b_window_likelihoods.jsonl",
        "window_ranking_sha256": output_dir / "run" / "v2_stage3b_window_ranking.jsonl",
        "hypothesis_claims_sha256": output_dir / "run" / "v2_stage3b_hypothesis_claims.jsonl",
        "claim_graph_artifact_sha256": output_dir / "run" / "v2_stage3b_claim_graph.json",
    }
    if len(ranking) != PLANNED_CALLS or any(run.get(key) != _sha(path) for key, path in expected_files.items()):
        raise CoarseHypothesisError("STAGE3B_RESULT_ARTIFACT_HASH_MISMATCH")
    if (output_dir / "v2_stage3b_replay_summary.json").is_file():
        replay = _object(output_dir / "v2_stage3b_replay_summary.json", "REPLAY_SUMMARY")
        if replay.get("new_model_calls") != 0 or replay.get("cache_hits") != PLANNED_CALLS or any(replay.get(key) != run.get(key) for key in ("numeric_results_sha256", "window_ranking_sha256", "hypothesis_claims_sha256", "claim_graph_sha256")):
            raise CoarseHypothesisError("REPLAY_RESULT_HASH_MISMATCH")
    return {"status": "PASS", "stage_status": run["stage_status"], "window_count": PLANNED_CALLS, "hypothesis_count": len(hypotheses),
            "observation_claim_count": 0, "verified_hypothesis_count": 0, "model_calls_made": 0, "backend_loaded": False,
            "cache_opened": False, "certificate_created": False, "new_verified_count": 0, "certificate_status": "NOT_APPLICABLE", "gt_used": False}
