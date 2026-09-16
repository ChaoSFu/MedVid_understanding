"""ReliVE-v2 Stage 3D claim-conditioned observation evidence retrieval.

This module consumes only frozen Stage 1--3C products and selected public
frame bytes.  It produces ranked *unverified* evidence candidates; it never
changes a claim, hypothesis, certificate, or final TAL answer.
"""
from __future__ import annotations

import hashlib
import math
from importlib.resources import files
from pathlib import Path
import time
from typing import Any, Callable, Mapping

from relive.backends import make_backend
from relive.config import load_config
from relive.storage.artifacts import ArtifactStore, canonical_json, stable_hash
from .coarse_hypothesis import support_margin
from .observation_planning import ObservationPlanningError, validate as validate_stage3c
from .task_selection import TALSelectionError, strict_json_loads, strict_jsonl
from .video_index import VideoIndexError, validate_video_index_artifacts

FORMAT = "relive-v2-tal-observation-evidence-retrieval-v1"
PROMPT_VERSION = "relive-v2-tal-observation-choice-v1"
CHOICES = ("A", "B", "C", "D")
PREPARED = (
    "v2_stage3d_visual_packets.jsonl", "v2_stage3d_prompt_specs.json",
    "v2_stage3d_retrieval_tasks.jsonl", "v2_stage3d_binding_fanout.jsonl",
    "v2_stage3d_candidate_ranking_policy.json", "v2_stage3d_call_plan.json",
)


class ObservationRetrievalError(ValueError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join((canonical_json(row) + "\n").encode("utf-8") for row in rows)


def _write(path: Path, value: Any) -> None:
    if path.exists():
        raise ObservationRetrievalError("IMMUTABLE_OUTPUT_EXISTS")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (canonical_json(value) + "\n").encode("utf-8") if isinstance(value, dict) else _jsonl(value)
    path.write_bytes(raw)


def _obj(path: Path, code: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = strict_json_loads(raw.decode("utf-8"), error_code=code)
    except (OSError, UnicodeDecodeError, TALSelectionError) as exc:
        raise ObservationRetrievalError(f"{code}_INVALID") from exc
    if not isinstance(value, dict) or raw != (canonical_json(value) + "\n").encode("utf-8"):
        raise ObservationRetrievalError(f"{code}_NONCANONICAL")
    return value


def _rows(path: Path, code: str) -> list[dict[str, Any]]:
    try:
        rows = list(strict_jsonl(path, error_code=code)); raw = path.read_bytes()
    except (OSError, TALSelectionError) as exc:
        raise ObservationRetrievalError(f"{code}_INVALID") from exc
    if raw != _jsonl(rows):
        raise ObservationRetrievalError(f"{code}_NONCANONICAL")
    return rows


def _finite(value: Any, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ObservationRetrievalError(code)
    return float(value)


def _stage3c(stage3c_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        validated = validate_stage3c(stage3c_dir)
    except ObservationPlanningError as exc:
        raise ObservationRetrievalError("FROZEN_STAGE3C_ARTIFACT_INVALID") from exc
    if validated.get("status") != "PASS" or not validated.get("ready_for_observation_retrieval"):
        raise ObservationRetrievalError("STAGE3C_NOT_READY")
    manifest = _obj(stage3c_dir / "v2_stage3c_manifest.json", "STAGE3C_MANIFEST")
    claims = _rows(stage3c_dir / "v2_stage3c_observation_claims.jsonl", "OBSERVATION_CLAIMS")
    windows = _rows(stage3c_dir / "v2_stage3c_physical_acquisition_windows.jsonl", "PHYSICAL_WINDOWS")
    bindings = _rows(stage3c_dir / "v2_stage3c_claim_window_bindings.jsonl", "CLAIM_WINDOW_BINDINGS")
    if (len(claims), len(windows), len(bindings)) != (manifest.get("observation_claim_count"), manifest.get("physical_acquisition_window_count"), manifest.get("claim_window_binding_count")):
        raise ObservationRetrievalError("STAGE3C_COUNT_BINDING_INVALID")
    if any(item.get("status") != "CANDIDATE_UNVERIFIED" for item in claims):
        raise ObservationRetrievalError("OBSERVATION_CLAIM_STATUS_CHANGED")
    return manifest, claims, windows, bindings


def _input_hashes(requirement_dir: Path, selection_path: Path, video_index_dir: Path, stage3c_dir: Path) -> dict[str, str]:
    try:
        index = validate_video_index_artifacts(video_index_dir, materialize_frames=False)
    except VideoIndexError as exc:
        raise ObservationRetrievalError("FROZEN_VIDEO_INDEX_INVALID") from exc
    if index.get("index_status") != "RESOLVED_DATASET_NATIVE_CLIP_LOCAL":
        raise ObservationRetrievalError("VIDEO_INDEX_NOT_RESOLVED")
    manifest, _, _, _ = _stage3c(stage3c_dir)
    needed = ("v2_requirement_manifest.json", "v2_requirement_specs.jsonl")
    if any(not (requirement_dir / name).is_file() for name in needed):
        raise ObservationRetrievalError("FROZEN_REQUIREMENT_INVALID")
    return {
        "requirement_manifest_sha256": _sha(requirement_dir / "v2_requirement_manifest.json"),
        "requirement_specs_sha256": _sha(requirement_dir / "v2_requirement_specs.jsonl"),
        "selection_manifest_sha256": _sha(selection_path),
        "video_index_manifest_sha256": _sha(video_index_dir / "v2_video_index_manifest.json"),
        "stage3c_manifest_sha256": _sha(stage3c_dir / "v2_stage3c_manifest.json"),
        "stage3c_artifact_map_sha256": stable_hash(manifest["artifact_sha256"]),
    }


_TEMPLATE_INSTRUCTIONS = {
    "PRECONDITION_ALIGNMENT": "Judge only the visible spatial relation. Do not infer alignment from procedure order or expected nursing workflow.",
    "ACTION_CORE_PRESSING": "A requires temporal visual evidence of fixing or pressing motion across the ordered frames. A single static hand-base overlap is not sufficient. Do not infer pressing from contact alone.",
    "POSTCONDITION_ATTACHMENT": "A requires ordered-frame evidence of a post-interaction state, not merely a single frame in which the base and skin overlap. Do not infer attachment from nursing workflow knowledge.",
}


def _prompt(claim: Mapping[str, Any]) -> str:
    template = claim.get("template_id")
    surface = claim.get("surface")
    if template not in _TEMPLATE_INSTRUCTIONS or not isinstance(surface, str) or not surface:
        raise ObservationRetrievalError("OBSERVATION_TEMPLATE_INVALID")
    rendered = files("relive").joinpath("prompts", "v2_tal_observation_choice.txt").read_text(encoding="utf-8").format(
        observation_claim=surface, template_instruction=_TEMPLATE_INSTRUCTIONS[template]).rstrip()
    if not rendered.endswith("\nAnswer:"):
        raise ObservationRetrievalError("OBSERVATION_PROMPT_ANCHOR_INVALID")
    return rendered


def _packet_frames(window: Mapping[str, Any], index_frames: Mapping[int, Mapping[str, Any]]) -> list[dict[str, Any]]:
    orders = window.get("included_logical_frame_orders")
    if not isinstance(orders, list) or not orders:
        raise ObservationRetrievalError("PHYSICAL_WINDOW_FRAME_MEMBERSHIP_INVALID")
    alias: dict[str, list[int]] = {}
    for frame in index_frames.values():
        key = stable_hash({"frame_sha256": frame["frame_sha256"], "source_frame_reference": frame["source_frame_reference"], "timestamp_seconds": frame["timestamp_seconds"]})
        alias.setdefault(key, []).append(frame["frame_order"])
    unique, seen = [], set()
    for order in orders:
        frame = index_frames.get(order)
        if frame is None:
            raise ObservationRetrievalError("PHYSICAL_WINDOW_FRAME_MEMBERSHIP_INVALID")
        key = stable_hash({"frame_sha256": frame["frame_sha256"], "source_frame_reference": frame["source_frame_reference"], "timestamp_seconds": frame["timestamp_seconds"]})
        if key not in seen:
            seen.add(key)
            unique.append({"unique_visual_key": key, **frame})
    if len(unique) > 8:
        # Endpoints plus deterministic uniformly-spaced interior samples.
        slots, denominator = 8, 7
        picked = [(slot * (len(unique) - 1) + denominator // 2) // denominator for slot in range(slots)]
        if len(set(picked)) != slots or picked[0] != 0 or picked[-1] != len(unique) - 1:
            raise ObservationRetrievalError("PACKET_SELECTION_INVALID")
        unique = [unique[item] for item in picked]
    rows = []
    for frame in unique:
        rows.append({"unique_visual_frame_id": "unique_visual_frame_" + frame["unique_visual_key"][:24],
                     "logical_aliases": sorted(alias[frame["unique_visual_key"]]), "logical_frame_order": frame["frame_order"],
                     "source_frame_reference": frame["source_frame_reference"], "frame_sha256": frame["frame_sha256"],
                     "resolved_frame_path": frame["resolved_frame_path"], "timestamp_seconds": frame["timestamp_seconds"]})
    return rows


def _policy(path: Path) -> dict[str, Any]:
    policy = _obj(path, "RETRIEVAL_POLICY")
    if policy.get("format") != "relive-v2-tal-observation-retrieval-policy-v1" or policy.get("choice_labels") != list(CHOICES) or policy.get("max_unique_frames_per_packet") != 8 or policy.get("max_candidate_evidence_windows_per_claim") != 3:
        raise ObservationRetrievalError("RETRIEVAL_POLICY_INVALID")
    minimum = policy.get("template_min_unique_timestamps")
    if not isinstance(minimum, dict) or set(minimum) != set(_TEMPLATE_INSTRUCTIONS) or any(type(value) is not int or value < 1 for value in minimum.values()):
        raise ObservationRetrievalError("RETRIEVAL_POLICY_INVALID")
    return policy


def prepare(*, requirement_dir: Path, selection_manifest_path: Path, video_index_dir: Path, stage3c_dir: Path,
            config_path: Path, policy_path: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ObservationRetrievalError("OUTPUT_DIRECTORY_MUST_BE_EMPTY")
    config = load_config(config_path)
    if config.get("backend", {}).get("kind") != "local_hf":
        raise ObservationRetrievalError("STAGE3D_REQUIRES_LOCAL_HF")
    hashes = _input_hashes(requirement_dir, selection_manifest_path, video_index_dir, stage3c_dir)
    manifest, claims, windows, bindings = _stage3c(stage3c_dir)
    policy = _policy(policy_path); policy_sha = _sha(policy_path)
    index_rows = _rows(video_index_dir / "v2_video_index.jsonl", "VIDEO_INDEX")
    if len(index_rows) != 1:
        raise ObservationRetrievalError("STAGE3D_REQUIRES_ONE_VIDEO_INDEX")
    frame_map = {item["frame_order"]: item for item in index_rows[0].get("frames", [])}
    if len(frame_map) != index_rows[0].get("frame_count"):
        raise ObservationRetrievalError("VIDEO_INDEX_FRAME_SCHEMA_INVALID")
    packets = []
    for window in sorted(windows, key=lambda item: item["acquisition_window_id"]):
        frames = _packet_frames(window, frame_map)
        base = {"physical_window_id": window["acquisition_window_id"], "start_seconds": window["start_seconds"], "end_seconds": window["end_seconds"],
                "ordered_chronological": True, "duplicate_visual_frames_removed": True, "logical_aliases_preserved": True,
                "max_unique_frames_per_packet": 8, "frames": frames, "video_index_manifest_sha256": hashes["video_index_manifest_sha256"],
                "stage3c_physical_windows_sha256": manifest["artifact_sha256"]["v2_stage3c_physical_acquisition_windows.jsonl"]}
        base["packet_content_sha256"] = stable_hash(base)
        base["visual_packet_id"] = "visual_packet_" + stable_hash(base)[:24]
        packets.append(base)
    by_window = {item["physical_window_id"]: item for item in packets}
    claim_by_id = {item["claim_id"]: item for item in claims}
    if len(by_window) != len(windows) or any(item.get("acquisition_window_id") not in by_window or item.get("claim_id") not in claim_by_id for item in bindings):
        raise ObservationRetrievalError("STAGE3C_BINDING_REFERENCE_INVALID")
    prompt_specs = {"format": FORMAT, "prompt_version": PROMPT_VERSION, "choice_labels": list(CHOICES),
                    "choice_semantics": {"A": "ordered frames clearly support the observation claim", "B": "related or partial evidence only", "C": "no visible supporting evidence", "D": "unreadable or unusable"},
                    "template_instructions": _TEMPLATE_INSTRUCTIONS, "free_text_generation_allowed": False,
                    "support_margin_formula": "logP(A)-logsumexp(logP(B),logP(C),logP(D))"}
    prompt_sha = stable_hash(prompt_specs)
    tasks_by_key: dict[str, dict[str, Any]] = {}
    fanout: list[dict[str, Any]] = []
    for binding in sorted(bindings, key=lambda item: (item["claim_id"], item["acquisition_window_id"])):
        claim, packet = claim_by_id[binding["claim_id"]], by_window[binding["acquisition_window_id"]]
        prompt = _prompt(claim)
        timestamps = {frame["timestamp_seconds"] for frame in packet["frames"]}
        needed = policy["template_min_unique_timestamps"][claim["template_id"]]
        status = "READY_FOR_MODEL" if len(timestamps) >= needed else "TEMPORAL_PACKET_INSUFFICIENT"
        key_material = {"observation_template_id": claim["template_id"], "physical_window_id": packet["physical_window_id"],
                        "visual_packet_sha256": packet["packet_content_sha256"], "prompt_spec_sha256": prompt_sha,
                        "model_config_sha256": _sha(config_path), "choice_policy_sha256": policy_sha}
        key = stable_hash(key_material)
        task = tasks_by_key.setdefault(key, {"retrieval_task_id": "retrieval_task_" + key[:24], "retrieval_task_key": key,
                                               **key_material, "observation_template_id": claim["template_id"], "physical_window_id": packet["physical_window_id"],
                                               "visual_packet_id": packet["visual_packet_id"], "prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                                               "image_paths": [frame["resolved_frame_path"] for frame in packet["frames"]], "frame_ids": [frame["unique_visual_frame_id"] for frame in packet["frames"]],
                                               "frame_sha256": [frame["frame_sha256"] for frame in packet["frames"]], "unique_timestamp_count": len(timestamps),
                                               "min_unique_timestamps_required": needed, "retrieval_status": status, "model_call_allowed": status == "READY_FOR_MODEL"})
        if task["retrieval_status"] != status:
            raise ObservationRetrievalError("RETRIEVAL_TASK_DEDUP_STATUS_CONFLICT")
        binding_id = "binding_" + stable_hash({"claim_id": binding["claim_id"], "parent_hypothesis_id": binding["parent_hypothesis_id"], "window": binding["acquisition_window_id"]})[:24]
        fanout.append({"binding_id": binding_id, "observation_claim_id": binding["claim_id"], "parent_hypothesis_id": binding["parent_hypothesis_id"],
                       "observation_template_id": claim["template_id"], "physical_window_id": packet["physical_window_id"],
                       "retrieval_task_id": task["retrieval_task_id"], "status": "PLANNED_UNSCORED"})
    tasks = sorted(tasks_by_key.values(), key=lambda item: item["retrieval_task_id"])
    task_id_by_key = {item["retrieval_task_key"]: item["retrieval_task_id"] for item in tasks}
    fanout_count: dict[str, int] = {}
    for item in fanout: fanout_count[item["retrieval_task_id"]] = fanout_count.get(item["retrieval_task_id"], 0) + 1
    for item in fanout: item["shared_task_binding_count"] = fanout_count[item["retrieval_task_id"]]
    planned = sum(1 for item in tasks if item["model_call_allowed"])
    rank = {"format": FORMAT, "policy_version": policy["policy_version"], "max_candidate_evidence_windows_per_claim": 3,
            "candidate_selection_basis": "RANK_ONLY_NO_ADMISSION_THRESHOLD", "ranking": policy["ranking"], "temporal_nms_applied": False}
    output_dir.mkdir(parents=True)
    artifacts: dict[str, Any] = {"v2_stage3d_visual_packets.jsonl": packets, "v2_stage3d_prompt_specs.json": prompt_specs,
        "v2_stage3d_retrieval_tasks.jsonl": tasks, "v2_stage3d_binding_fanout.jsonl": fanout, "v2_stage3d_candidate_ranking_policy.json": rank}
    for name, value in artifacts.items(): _write(output_dir / name, value)
    call_plan = {"format": FORMAT, "status": "FROZEN_PRE_MODEL", "input_hashes": hashes, "config_sha256": _sha(config_path), "policy_sha256": policy_sha,
                 "planned_model_calls": planned, "unique_retrieval_task_count": len(tasks), "claim_window_binding_count": len(bindings),
                 "deduplicated_model_call_savings": len(bindings)-len(tasks), "task_status_counts": {"READY_FOR_MODEL": planned, "TEMPORAL_PACKET_INSUFFICIENT": len(tasks)-planned},
                 "model_configuration": config["backend"], "artifacts": {name: _sha(output_dir / name) for name in artifacts},
                 "model_calls_made": 0, "backend_loaded": False, "cache_opened": False, "gt_used": False, "assistant_or_gt_values_accessed": False,
                 "certificate_created": False, "new_verified_count": 0, "certificate_status": "NOT_APPLICABLE", "no_visible_event_status": "CANDIDATE_UNVERIFIED"}
    call_plan["call_plan_content_sha256"] = stable_hash(call_plan)
    _write(output_dir / "v2_stage3d_call_plan.json", call_plan)
    artifact_hashes = {name: _sha(output_dir / name) for name in PREPARED}
    manifest_out = {"format": FORMAT, "status": "PASS", "stage_status": "FROZEN_PRE_MODEL", "artifact_sha256": artifact_hashes,
                    "physical_window_count": len(windows), "visual_packet_count": len(packets), "claim_window_binding_count": len(bindings),
                    "unique_retrieval_task_count": len(tasks), "planned_model_calls": planned, "deduplicated_model_call_savings": len(bindings)-len(tasks),
                    "observation_claim_count": len(claims), "model_calls_made": 0, "backend_loaded": False, "cache_opened": False,
                    "gt_used": False, "assistant_or_gt_values_accessed": False, "certificate_created": False, "new_verified_count": 0,
                    "certificate_status": "NOT_APPLICABLE", "no_visible_event_status": "CANDIDATE_UNVERIFIED"}
    manifest_out["manifest_content_sha256"] = stable_hash(manifest_out)
    _write(output_dir / "v2_stage3d_manifest.json", manifest_out)
    audit = {"format": FORMAT, "status": "PASS", "mode": "prepare", "allowed_inputs_opened": ["frozen RequirementSpec", "selection manifest", "Immutable VideoIndex", "Stage 3C ObservationClaims and physical windows", "Stage 3C binding and ClaimGraph", "reviewed local-HF configuration", "versioned Stage 3D retrieval policy"],
             "allowed_inputs_permitted_for_run": ["selected public frame bytes"], "forbidden_inputs_not_opened": ["raw trainval JSON", "assistant answer", "reference answer", "conversations[1]", "struc_info", "temporal GT", "GT interval", "evaluation", "bbox", "mask", "ROI", "certificate", "previous final answer"],
             "input_hashes": hashes, "artifact_sha256": {**artifact_hashes, "v2_stage3d_manifest.json": _sha(output_dir / "v2_stage3d_manifest.json")},
             "model_calls_made": 0, "backend_loaded": False, "cache_opened": False, "gt_used": False, "assistant_or_gt_values_accessed": False, "certificate_created": False, "new_verified_count": 0, "certificate_status": "NOT_APPLICABLE"}
    audit["audit_content_sha256"] = stable_hash(audit); _write(output_dir / "v2_stage3d_freeze_audit.json", audit)
    summary = {"format": FORMAT, "status": "PASS", "mode": "prepare", "stage_status": "FROZEN_PRE_MODEL", "physical_window_count": len(windows), "claim_window_binding_count": len(bindings), "unique_retrieval_task_count": len(tasks), "planned_model_calls": planned, "observation_claim_count": len(claims), "model_calls_made": 0, "backend_loaded": False, "cache_opened": False, "gt_used": False, "certificate_created": False, "new_verified_count": 0, "certificate_status": "NOT_APPLICABLE"}
    _write(output_dir / "v2_stage3d_prepare.json", summary)
    return summary


def _prepared(output_dir: Path, config_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = _obj(output_dir / "v2_stage3d_manifest.json", "MANIFEST")
    if manifest.get("manifest_content_sha256") != stable_hash({key: value for key, value in manifest.items() if key != "manifest_content_sha256"}):
        raise ObservationRetrievalError("MANIFEST_TAMPERED")
    for name, expected in manifest.get("artifact_sha256", {}).items():
        if _sha(output_dir / name) != expected: raise ObservationRetrievalError("ARTIFACT_TAMPERED")
    plan = _obj(output_dir / "v2_stage3d_call_plan.json", "CALL_PLAN")
    if plan.get("call_plan_content_sha256") != stable_hash({key: value for key, value in plan.items() if key != "call_plan_content_sha256"}) or plan.get("config_sha256") != _sha(config_path):
        raise ObservationRetrievalError("PREPARED_ARTIFACT_DRIFT")
    if any(plan.get("artifacts", {}).get(name) != _sha(output_dir / name) for name in PREPARED if name != "v2_stage3d_call_plan.json"):
        raise ObservationRetrievalError("PREPARED_ARTIFACT_DRIFT")
    tasks = _rows(output_dir / "v2_stage3d_retrieval_tasks.jsonl", "RETRIEVAL_TASKS")
    fanout = _rows(output_dir / "v2_stage3d_binding_fanout.jsonl", "BINDING_FANOUT")
    packets = _rows(output_dir / "v2_stage3d_visual_packets.jsonl", "VISUAL_PACKETS")
    if len(tasks) != plan.get("unique_retrieval_task_count") or len(fanout) != plan.get("claim_window_binding_count") or len(packets) != manifest.get("visual_packet_count"):
        raise ObservationRetrievalError("PREPARED_COUNT_DRIFT")
    return plan, tasks, fanout, packets


def preflight(*, output_dir: Path, config_path: Path, stage3c_dir: Path, backend_factory: Callable[[dict[str, Any]], Any] = make_backend) -> dict[str, Any]:
    if (output_dir / "v2_stage3d_preflight.json").exists(): raise ObservationRetrievalError("PREFLIGHT_ALREADY_EXISTS")
    plan, tasks, fanout, packets = _prepared(output_dir, config_path)
    current = _stage3c(stage3c_dir)[0]
    if _sha(stage3c_dir / "v2_stage3c_manifest.json") != plan["input_hashes"]["stage3c_manifest_sha256"] or stable_hash(current["artifact_sha256"]) != plan["input_hashes"]["stage3c_artifact_map_sha256"]:
        raise ObservationRetrievalError("FROZEN_STAGE3C_ARTIFACT_DRIFT")
    cache = output_dir / "cache"
    if cache.exists() and any(cache.iterdir()): raise ObservationRetrievalError("PREFLIGHT_REQUIRES_EMPTY_CACHE")
    backend = backend_factory(load_config(config_path)["backend"])
    contracts = {}
    for task in tasks:
        if task["model_call_allowed"]:
            contract = backend.forced_choice_token_contract(task, choices=CHOICES)
            if contract.get("choice_labels") != list(CHOICES) or set(contract.get("choice_token_ids", {})) != set(CHOICES) or len(set(contract["choice_token_ids"].values())) != 4:
                raise ObservationRetrievalError("ABCD_NEXT_TOKEN_CONTRACT_NOT_UNIQUE")
            contracts[task["retrieval_task_id"]] = contract
    if len(contracts) != plan["planned_model_calls"]: raise ObservationRetrievalError("PREFLIGHT_TASK_CONTRACT_COUNT_INVALID")
    report = {"format": FORMAT, "status": "PASS", "mode": "preflight", "planned_model_calls": plan["planned_model_calls"], "unique_retrieval_task_count": len(tasks), "claim_window_binding_count": len(fanout), "model_calls_made": 0, "backend_loaded": True, "cache_opened": False, "cache_empty": True, "model_fingerprint": backend.fingerprint(), "choice_token_contracts": contracts, "call_plan_sha256": _sha(output_dir / "v2_stage3d_call_plan.json"), "gt_used": False, "assistant_or_gt_values_accessed": False, "certificate_created": False, "new_verified_count": 0, "certificate_status": "NOT_APPLICABLE"}
    report["preflight_content_sha256"] = stable_hash(report); _write(output_dir / "v2_stage3d_preflight.json", report)
    return report


class ChoiceCache:
    def __init__(self, backend: Any, root: Path): self.backend,self.store,self.new_calls,self.cache_hits,self.logical_calls=backend,ArtifactStore(root),0,0,0
    def call(self, task: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
        identity={"cache_version":"relive-v2-tal-observation-choice-cache-v1","model_fingerprint":self.backend.fingerprint(),"prompt_sha256":task["prompt_sha256"],"choice_token_ids":contract["choice_token_ids"],"visual_packet_sha256":task["visual_packet_sha256"],"observation_template_id":task["observation_template_id"],"model_config_sha256":task["model_config_sha256"],"choice_policy_sha256":task["choice_policy_sha256"],"frame_sha256":task["frame_sha256"]}
        key=stable_hash(identity); self.logical_calls+=1
        with self.store.run_lock():
            path=self.store.path("v2_tal_observation_choice",key)
            if path.exists():
                value=self.store.get_json(str(path))
                if value.get("identity")!=identity: raise ObservationRetrievalError("CHOICE_CACHE_IDENTITY_COLLISION")
                self.cache_hits+=1; return {**value["result"],"cache_key":key,"cache_hit":True}
            result=self.backend.forced_choice_likelihood(dict(task),choice_token_ids=contract["choice_token_ids"],choices=CHOICES)
            if result.get("choice_contract",{}).get("choice_token_ids")!=contract["choice_token_ids"]: raise ObservationRetrievalError("CHOICE_TOKEN_CONTRACT_DRIFT")
            self.store.put_json("v2_tal_observation_choice",key,{"identity":identity,"result":result}); self.new_calls+=1
            return {**result,"cache_key":key,"cache_hit":False}


def _result_rows(tasks: list[dict[str, Any]], contracts: Mapping[str, Any], cache: ChoiceCache) -> list[dict[str, Any]]:
    rows=[]
    for task in tasks:
        base={"retrieval_task_id":task["retrieval_task_id"],"observation_template_id":task["observation_template_id"],"physical_window_id":task["physical_window_id"],"visual_packet_id":task["visual_packet_id"],"retrieval_status":task["retrieval_status"],"status":"CANDIDATE_EVIDENCE_UNVERIFIED"}
        if not task["model_call_allowed"]:
            rows.append({**base,"choice_token_ids":None,"raw_logits":None,"log_probabilities":None,"choice_argmax":None,"support_margin":None,"numeric_projection_sha256":None,"cache_key":None,"cache_hit":False}); continue
        scored=cache.call(task,contracts[task["retrieval_task_id"]]); choices=scored.get("choices",{})
        if set(choices)!=set(CHOICES): raise ObservationRetrievalError("CHOICE_RESULT_SCHEMA_INVALID")
        logs={item:choices[item].get("log_probability") for item in CHOICES}; margin=support_margin(logs)
        argmax=sorted(CHOICES,key=lambda label:(-logs[label],label))[0]
        numeric={"log_likelihood_A":logs["A"],"log_likelihood_B":logs["B"],"log_likelihood_C":logs["C"],"log_likelihood_D":logs["D"],"support_margin":margin,"choice_argmax":argmax}
        rows.append({**base,"choice_token_ids":scored["choice_contract"]["choice_token_ids"],"raw_logits":{label:choices[label]["raw_logit"] for label in CHOICES},"log_probabilities":logs,"choice_argmax":argmax,"support_margin":margin,"numeric_projection_sha256":stable_hash(numeric),"cache_key":scored["cache_key"],"cache_hit":scored["cache_hit"]})
    return rows


def _bindings_and_rankings(*, task_rows: list[dict[str, Any]], fanout: list[dict[str, Any]], packets: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    task={item["retrieval_task_id"]:item for item in task_rows}; packet={item["physical_window_id"]:item for item in packets}; results=[]
    for fan in fanout:
        result=task.get(fan["retrieval_task_id"])
        if result is None: raise ObservationRetrievalError("FANOUT_TASK_MISSING")
        results.append({**fan,"log_likelihood_A":None if result["log_probabilities"] is None else result["log_probabilities"]["A"],"log_likelihood_B":None if result["log_probabilities"] is None else result["log_probabilities"]["B"],"log_likelihood_C":None if result["log_probabilities"] is None else result["log_probabilities"]["C"],"log_likelihood_D":None if result["log_probabilities"] is None else result["log_probabilities"]["D"],"support_margin":result["support_margin"],"choice_argmax":result["choice_argmax"],"retrieval_status":result["retrieval_status"],"status":"CANDIDATE_EVIDENCE_UNVERIFIED"})
    groups: dict[str,list[dict[str,Any]]]={}
    for row in results: groups.setdefault(row["observation_claim_id"],[]).append(row)
    rankings=[]; candidates=[]
    for claim_id, rows in sorted(groups.items()):
        ordered=sorted(rows,key=lambda row:(row["retrieval_status"]!="READY_FOR_MODEL", -(row["support_margin"] if row["support_margin"] is not None else float("-inf")), packet[row["physical_window_id"]]["start_seconds"], row["physical_window_id"], row["retrieval_task_id"]))
        ranked=[]
        for rank,row in enumerate(ordered,start=1):
            ranked.append({"observation_claim_id":claim_id,"rank":rank,"binding_id":row["binding_id"],"physical_window_id":row["physical_window_id"],"retrieval_task_id":row["retrieval_task_id"],"retrieval_status":row["retrieval_status"],"support_margin":row["support_margin"],"status":"CANDIDATE_EVIDENCE_UNVERIFIED"})
        rankings.extend(ranked); candidates.extend(ranked[:3])
    return results, rankings, candidates


def _scientific_task_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key:value for key,value in row.items() if key not in {"cache_key","cache_hit"}} for row in rows]


def execute(*, output_dir: Path, config_path: Path, mode: str, backend_factory: Callable[[dict[str, Any]], Any] = make_backend) -> dict[str, Any]:
    if mode not in {"run","replay"}: raise ObservationRetrievalError("RUN_MODE_INVALID")
    target=output_dir/mode
    if target.exists(): raise ObservationRetrievalError("RUN_OUTPUT_ALREADY_EXISTS")
    plan,tasks,fanout,packets=_prepared(output_dir,config_path); pre=_obj(output_dir/"v2_stage3d_preflight.json","PREFLIGHT")
    if pre.get("preflight_content_sha256")!=stable_hash({key:value for key,value in pre.items() if key!="preflight_content_sha256"}) or pre.get("call_plan_sha256")!=_sha(output_dir/"v2_stage3d_call_plan.json"): raise ObservationRetrievalError("PREFLIGHT_ARTIFACT_DRIFT")
    backend=backend_factory(load_config(config_path)["backend"])
    if backend.fingerprint()!=pre.get("model_fingerprint"): raise ObservationRetrievalError("MODEL_FINGERPRINT_DRIFT")
    started=time.monotonic(); cache=ChoiceCache(backend,output_dir/"cache"); rows=_result_rows(tasks,pre["choice_token_contracts"],cache)
    if cache.logical_calls!=plan["planned_model_calls"] or (mode=="run" and (cache.new_calls!=plan["planned_model_calls"] or cache.cache_hits!=0)) or (mode=="replay" and (cache.new_calls!=0 or cache.cache_hits!=plan["planned_model_calls"])): raise ObservationRetrievalError("MODEL_CALL_BUDGET_OR_CACHE_REPLAY_MISMATCH")
    binding,ranking,candidates=_bindings_and_rankings(task_rows=rows,fanout=fanout,packets=packets)
    target.mkdir(parents=True); _write(target/"v2_stage3d_task_likelihoods.jsonl",rows); _write(target/"v2_stage3d_binding_results.jsonl",binding); _write(target/"v2_stage3d_observation_rankings.jsonl",ranking); _write(target/"v2_stage3d_candidate_evidence_sets.jsonl",candidates)
    scientific={"numeric_results_sha256":stable_hash(_scientific_task_rows(rows)),"binding_results_scientific_sha256":stable_hash(binding),"observation_rankings_sha256":_sha(target/"v2_stage3d_observation_rankings.jsonl"),"candidate_evidence_sets_sha256":_sha(target/"v2_stage3d_candidate_evidence_sets.jsonl")}
    summary={"format":FORMAT,"status":"PASS","mode":mode,"stage_status":"OBSERVATION_EVIDENCE_CANDIDATES_RETRIEVED_UNVERIFIED","physical_window_count":len(packets),"claim_window_binding_count":len(fanout),"unique_retrieval_task_count":len(tasks),"planned_model_calls":plan["planned_model_calls"],"deduplicated_model_call_savings":len(fanout)-len(tasks),"observation_claim_count":plan["claim_window_binding_count"] and len({x["observation_claim_id"] for x in fanout}),"candidate_evidence_set_count":len({x["observation_claim_id"] for x in fanout}),"supported_observation_count":0,"verified_observation_count":0,"hypothesis_status_unchanged":True,"observation_claim_status_unchanged":True,"negative_coverage_obligation_status":"UNTESTED","no_visible_event_status":"CANDIDATE_UNVERIFIED","model_calls_made":cache.new_calls,"new_model_calls":cache.new_calls,"cache_hits":cache.cache_hits,"logical_model_calls":cache.logical_calls,"failed_model_calls":0,"certificate_created":False,"new_verified_count":0,"certificate_status":"NOT_APPLICABLE","gt_used":False,"assistant_or_gt_values_accessed":False,"latency_seconds":time.monotonic()-started,"ready_for_temporal_evidence_composition":True,**scientific}
    if mode=="replay":
        first=_obj(output_dir/"v2_stage3d_run_summary.json","RUN_SUMMARY")
        if any(summary[key]!=first.get(key) for key in scientific): raise ObservationRetrievalError("REPLAY_RESULT_HASH_MISMATCH")
    _write(output_dir/f"v2_stage3d_{mode}_summary.json",summary); return summary


def validate(output_dir: Path) -> dict[str, Any]:
    # Validator uses emitted bytes only; config/model/cache/images/upstream inputs are never opened.
    manifest=_obj(output_dir/"v2_stage3d_manifest.json","MANIFEST")
    if manifest.get("manifest_content_sha256")!=stable_hash({key:value for key,value in manifest.items() if key!="manifest_content_sha256"}): raise ObservationRetrievalError("MANIFEST_TAMPERED")
    for name,expected in manifest.get("artifact_sha256",{}).items():
        if _sha(output_dir/name)!=expected: raise ObservationRetrievalError("ARTIFACT_TAMPERED")
    plan=_obj(output_dir/"v2_stage3d_call_plan.json","CALL_PLAN")
    if plan.get("call_plan_content_sha256")!=stable_hash({key:value for key,value in plan.items() if key!="call_plan_content_sha256"}): raise ObservationRetrievalError("CALL_PLAN_TAMPERED")
    audit=_obj(output_dir/"v2_stage3d_freeze_audit.json","FREEZE_AUDIT")
    if audit.get("audit_content_sha256")!=stable_hash({key:value for key,value in audit.items() if key!="audit_content_sha256"}): raise ObservationRetrievalError("FREEZE_AUDIT_TAMPERED")
    run=_obj(output_dir/"v2_stage3d_run_summary.json","RUN_SUMMARY"); likelihoods=_rows(output_dir/"run"/"v2_stage3d_task_likelihoods.jsonl","TASK_LIKELIHOODS"); binding=_rows(output_dir/"run"/"v2_stage3d_binding_results.jsonl","BINDING_RESULTS"); ranking=_rows(output_dir/"run"/"v2_stage3d_observation_rankings.jsonl","RANKINGS"); candidates=_rows(output_dir/"run"/"v2_stage3d_candidate_evidence_sets.jsonl","CANDIDATES")
    if len(likelihoods)!=run.get("unique_retrieval_task_count") or len(binding)!=run.get("claim_window_binding_count") or len({row["observation_claim_id"] for row in candidates})!=run.get("candidate_evidence_set_count") or any(row.get("status")!="CANDIDATE_EVIDENCE_UNVERIFIED" for row in binding+candidates) or run.get("supported_observation_count")!=0 or run.get("verified_observation_count")!=0: raise ObservationRetrievalError("RESULT_SCHEMA_INVALID")
    if (output_dir/"v2_stage3d_replay_summary.json").is_file():
        replay=_obj(output_dir/"v2_stage3d_replay_summary.json","REPLAY_SUMMARY")
        keys=("numeric_results_sha256","binding_results_scientific_sha256","observation_rankings_sha256","candidate_evidence_sets_sha256")
        if replay.get("new_model_calls")!=0 or replay.get("cache_hits")!=run.get("planned_model_calls") or any(replay.get(key)!=run.get(key) for key in keys): raise ObservationRetrievalError("REPLAY_RESULT_HASH_MISMATCH")
    return {"status":"PASS","stage_status":run["stage_status"],"physical_window_count":run["physical_window_count"],"claim_window_binding_count":run["claim_window_binding_count"],"unique_retrieval_task_count":run["unique_retrieval_task_count"],"candidate_evidence_set_count":run["candidate_evidence_set_count"],"model_calls_made":0,"backend_loaded":False,"cache_opened":False,"gt_used":False,"assistant_or_gt_values_accessed":False,"certificate_created":False,"new_verified_count":0,"certificate_status":"NOT_APPLICABLE","ready_for_temporal_evidence_composition":True}
