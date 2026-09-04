from __future__ import annotations

from pathlib import Path

import pytest

from baselines.videoitg_qwen35.config import ALL_QA_TYPES, CANDIDATE_LIMIT, TOP_K
from baselines.videoitg_qwen35.io_utils import append_jsonl, assert_no_gt_leak, load_completed_keys
from baselines.videoitg_qwen35.manifest import build_gt_free_row
from baselines.videoitg_qwen35.models.qwen35 import qwen_cache_key
from baselines.videoitg_qwen35.selectors.base import official_uniform_candidate_positions, topk_chronological
from baselines.videoitg_qwen35.selectors.videoitg import build_selection_result, bypass_selection, validate_selection
from baselines.videoitg_qwen35.tasks import get_adapter


def sample(qa_type: str = "tal", n: int = 40, dataset_name: str = "AVOS") -> dict:
    return {
        "id": f"id-{qa_type}",
        "qa_type": qa_type,
        "dataset_name": dataset_name,
        "data_source": dataset_name,
        "video": ["/frames/dup.jpg" if i in (2, 3) else f"/frames/{i:04d}.jpg" for i in range(n)],
        "sampled_video_frames": list(range(10, 10 + n)),
        "metadata": {"fps": 1.0, "video_id": "v"},
        "conversations": [
            {"from": "human", "value": "<video>\nWhere is the action?"},
            {"from": "gpt", "value": "10-20 seconds."},
        ],
        "struc_info": [{"start": 10, "end": 20, "answer": "leaky"}],
        "RC_info": None,
        "is_RC": False,
    }


def test_all_qa_types_dispatch():
    for qa_type in ALL_QA_TYPES:
        assert get_adapter(qa_type).applies_to(qa_type)


def test_gt_stripping_and_manifest_fields():
    row = build_gt_free_row(sample())
    assert row["question"] == "<video>\nWhere is the action?"
    assert "struc_info" not in row
    assert "conversations" not in row
    assert_no_gt_leak(row)


def test_logical_positions_preserve_duplicate_paths():
    row = build_gt_free_row(sample(n=5))
    positions = official_uniform_candidate_positions(row["n_medvidu_frames"], CANDIDATE_LIMIT)
    paths = [row["video"][pos] for pos in positions]
    assert positions == [0, 1, 2, 3, 4]
    assert paths[2] == paths[3]
    assert len(paths) == 5


@pytest.mark.parametrize("n,expected", [(10, 10), (32, 32), (100, 32)])
def test_top_k_boundaries(n: int, expected: int):
    candidates = official_uniform_candidate_positions(n, CANDIDATE_LIMIT)
    scores = [float(i) for i in range(len(candidates))]
    _, _, _, selected = topk_chronological(candidates, scores, TOP_K)
    assert len(selected) == expected
    assert selected == sorted(selected)


def test_candidate_limit_over_512():
    candidates = official_uniform_candidate_positions(1000, 512)
    assert len(candidates) == 512
    assert min(candidates) >= 0
    assert max(candidates) < 1000
    assert candidates == sorted(candidates)


def test_score_sort_topk_and_chronological_reorder():
    candidates = [0, 1, 2, 3, 4]
    scores = [0.1, 0.9, 0.3, 0.8, 0.2]
    ranked, ranked_scores, topk, selected = topk_chronological(candidates, scores, 3)
    assert ranked == [1, 3, 2, 4, 0]
    assert ranked_scores == [0.9, 0.8, 0.3, 0.2, 0.1]
    assert topk == [1, 3, 2]
    assert selected == [1, 2, 3]


def test_build_selection_result_mapping_and_validation():
    row = build_gt_free_row(sample(n=5))
    result = build_selection_result(
        row,
        candidate_positions=[0, 1, 2, 3, 4],
        scores=[0.1, 0.9, 0.3, 0.8, 0.2],
        selector_model="nvidia/VideoITG-8B",
        selector_commit="commit",
        top_k=3,
    ).to_dict()
    validate_selection(result)
    assert result["topk_original_positions_score_order"] == [1, 3, 2]
    assert result["selected_original_positions_chronological"] == [1, 2, 3]
    assert result["selected_source_frame_indices_chronological"] == [11, 12, 13]
    assert result["selected_frame_paths_chronological"][1] == "/frames/dup.jpg"


def test_timestamp_mapping_for_metadata_fps_fallback():
    row = build_gt_free_row(sample(qa_type="skill_assessment", n=3, dataset_name="jigsaws"))
    assert row["time_mapping_method"] == "metadata_fps_source_frame_rate"
    assert [obs["local_time"] for obs in row["frame_observations"]] == [0.0, 1.0, 2.0]


def test_selector_jsonl_restart(tmp_path: Path):
    path = tmp_path / "results.jsonl"
    append_jsonl(path, {"cache_key": "abc", "sample_id": "s"})
    assert load_completed_keys(path) == {"abc"}


def test_qwen_cache_key_changes_with_positions():
    row = build_gt_free_row(sample(n=5))
    selection = bypass_selection(row, "m", "c").to_dict()
    fp = {"model": "qwen"}
    decode = {"do_sample": False}
    pixel = {"total_pixels": 1}
    key1 = qwen_cache_key(row, selection, fp, "prompt", decode, pixel)
    selection["selected_original_positions_chronological"] = [1]
    key2 = qwen_cache_key(row, selection, fp, "prompt", decode, pixel)
    assert key1 != key2


def test_rc_selector_bypass():
    s = sample(qa_type="region_caption_gpt", n=3, dataset_name="EgoSurgery")
    s["is_RC"] = True
    s["RC_info"] = {"start_frame": "/frames/0001.jpg", "start_frame_bbox": [1, 2, 3, 4]}
    row = build_gt_free_row(s)
    result = bypass_selection(row, "m", "c").to_dict()
    assert row["selector_applicable"] is False
    assert result["selector_applicable"] is False
    assert result["effective_k"] == 0
    assert "region-conditioned" in result["selector_reason"]


def test_empty_model_response_parser_failure_not_exception():
    adapter = get_adapter("tal")
    status, parsed = adapter.parse_prediction("")
    assert status == "EMPTY"
    assert parsed is None


def test_gt_leak_detection():
    with pytest.raises(AssertionError):
        assert_no_gt_leak({"safe": {"gt_answer": "nope"}})
