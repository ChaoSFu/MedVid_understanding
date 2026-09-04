from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SelectionResult:
    sample_id: str
    qa_type: str
    dataset_name: str | None
    n_medvidu_frames: int
    n_selector_candidates: int
    effective_k: int
    candidate_original_positions: list[int]
    ranked_original_positions: list[int]
    ranked_scores: list[float]
    topk_original_positions_score_order: list[int]
    selected_original_positions_chronological: list[int]
    selected_source_frame_indices_chronological: list[int]
    selected_frame_paths_chronological: list[str]
    selected_local_times_chronological: list[float]
    selector_model: str
    selector_commit: str
    selector_applicable: bool
    selector_reason: str | None = None
    cache_key: str | None = None

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class FrameSelector(Protocol):
    selector_name: str

    def select(self, manifest_row: dict) -> SelectionResult:
        ...


def official_uniform_candidate_positions(n_frames: int, candidate_limit: int) -> list[int]:
    if n_frames < 0:
        raise ValueError("n_frames must be non-negative")
    if candidate_limit <= 0:
        raise ValueError("candidate_limit must be positive")
    if n_frames <= candidate_limit:
        return list(range(n_frames))
    scale = float(n_frames) / float(candidate_limit)
    return [round((i + 1) * scale - 1) for i in range(candidate_limit)]


def topk_chronological(candidate_positions: list[int], scores: list[float], k: int) -> tuple[list[int], list[float], list[int], list[int]]:
    if len(candidate_positions) != len(scores):
        raise ValueError("candidate_positions and scores must have equal length")
    if k <= 0:
        raise ValueError("k must be positive")
    order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
    ranked_positions = [candidate_positions[i] for i in order]
    ranked_scores = [float(scores[i]) for i in order]
    effective_k = min(k, len(candidate_positions))
    topk_score_order = ranked_positions[:effective_k]
    selected_chronological = sorted(topk_score_order)
    return ranked_positions, ranked_scores, topk_score_order, selected_chronological
