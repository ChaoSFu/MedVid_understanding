from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np

from ..config import REQUESTED_K, VIDEO_REFINEMENT_WLEN


def load_official_video_refinement(dig_repo_dir: Path):
    path = Path(dig_repo_dir) / "pipeline" / "video_refinement.py"
    if "decord" not in sys.modules and importlib.util.find_spec("decord") is None:
        decord_stub = types.ModuleType("decord")
        decord_stub.VideoReader = object
        sys.modules["decord"] = decord_stub
    spec = importlib.util.spec_from_file_location("dig_official_video_refinement", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load official DIG video_refinement.py from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    old_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old_dont_write_bytecode
    return module


def global_uniform_positions(n_frames: int, k: int = REQUESTED_K) -> list[int]:
    if n_frames < 0:
        raise ValueError("n_frames must be non-negative")
    if n_frames == 0 or k <= 0:
        return []
    effective_k = min(k, n_frames)
    return [int(x) for x in np.linspace(0, n_frames - 1, effective_k, dtype=int).tolist()]


def refine_and_select(
    rewards: list[float],
    boundaries: list[int],
    dig_repo_dir: Path,
    k: int = REQUESTED_K,
    wlen: int = VIDEO_REFINEMENT_WLEN,
) -> tuple[list[list[int]], list[int]]:
    module = load_official_video_refinement(dig_repo_dir)
    intervals = module.video_refinement(rewards, boundaries, wlen=wlen)
    selected = module.select_k_indices(intervals, k)
    return [[int(a), int(b)] for a, b in intervals], [int(x) for x in selected]


def positions_hash(positions: list[int]) -> str:
    from ..io_utils import sha256_json

    return sha256_json([int(x) for x in positions])


def build_selector_record(
    manifest_row: dict[str, Any],
    query_type: str,
    selected_score_order: list[int],
    selected_chronological: list[int],
    dig_commit: str,
    selector_config: dict[str, Any],
    r_frame_positions: list[int] | None = None,
    reward_values: list[float] | None = None,
    reward_boundaries: list[int] | None = None,
    refined_intervals: list[list[int]] | None = None,
) -> dict[str, Any]:
    observations = {int(obs["frame_position"]): obs for obs in manifest_row.get("frame_observations", [])}
    selected_chronological = [int(x) for x in selected_chronological]
    record = {
        "sample_id": manifest_row["sample_id"],
        "original_index": manifest_row["original_index"],
        "id": manifest_row.get("id"),
        "qa_type": manifest_row["qa_type"],
        "dataset_name": manifest_row.get("dataset_name"),
        "data_source": manifest_row.get("data_source"),
        "question_hash": selector_config["question_hash"],
        "n_available_frames": int(manifest_row["n_medvidu_frames"]),
        "n_medvidu_frames": int(manifest_row["n_medvidu_frames"]),
        "query_type": query_type,
        "r_frame_original_positions": [int(x) for x in (r_frame_positions or [])],
        "reward_values": [float(x) for x in (reward_values or [])],
        "reward_boundaries": [int(x) for x in (reward_boundaries or [])],
        "refined_intervals_original_positions": refined_intervals or [],
        "selected_original_positions": [int(x) for x in selected_score_order],
        "selected_original_positions_chronological": selected_chronological,
        "selected_source_frame_indices_chronological": [
            int(observations[pos]["source_frame_index"]) for pos in selected_chronological
        ],
        "selected_local_times_chronological": [
            float(observations[pos]["local_time"]) for pos in selected_chronological
        ],
        "selected_frame_paths_chronological": [
            str(observations[pos]["frame_path"]) for pos in selected_chronological
        ],
        "requested_k": int(selector_config["requested_k"]),
        "effective_k": len(selected_chronological),
        "selector_version": selector_config["selector_version"],
        "dig_commit": dig_commit,
        "query_identifier_model": selector_config["query_identifier_model"],
        "reward_lmm": selector_config["reward_lmm"],
        "cafs_model": selector_config["cafs_model"],
        "wlen": int(selector_config["video_refinement_wlen"]),
        "selector_applicable": bool(manifest_row.get("selector_applicable", False)),
        "selector_reason": manifest_row.get("selector_reason"),
        "selected_positions_hash": positions_hash(selected_chronological),
        "gt_information_used": False,
    }
    return record
