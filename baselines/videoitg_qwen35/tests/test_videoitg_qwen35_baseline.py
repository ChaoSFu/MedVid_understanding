from __future__ import annotations

from pathlib import Path

import pytest

from baselines.videoitg_qwen35.config import ALL_QA_TYPES, CANDIDATE_LIMIT, TOP_K
from baselines.videoitg_qwen35.io_utils import append_jsonl, assert_no_gt_leak, load_completed_keys, repair_jsonl
from baselines.videoitg_qwen35.manifest import build_gt_free_row, remap_nested_paths
from baselines.videoitg_qwen35.models.qwen35 import qwen_cache_key
from baselines.videoitg_qwen35.models.qwen_sglang import build_sglang_messages, resize_to_max_pixels, sglang_cache_key
from baselines.videoitg_qwen35.selectors.base import official_uniform_candidate_positions, topk_chronological
from baselines.videoitg_qwen35.selectors.videoitg import (
    build_selection_result,
    bypass_selection,
    filter_rows_for_shard,
    repair_selector_outputs,
    selector_output_paths,
    validate_selection,
    validate_shard_args,
)
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
    row = build_gt_free_row(sample(), new_data_root=None)
    assert row["question"] == "<video>\nWhere is the action?"
    assert "struc_info" not in row
    assert "conversations" not in row
    assert_no_gt_leak(row)


def test_logical_positions_preserve_duplicate_paths():
    row = build_gt_free_row(sample(n=5), new_data_root=None)
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
    row = build_gt_free_row(sample(n=5), new_data_root=None)
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
    row = build_gt_free_row(sample(qa_type="skill_assessment", n=3, dataset_name="jigsaws"), new_data_root=None)
    assert row["time_mapping_method"] == "metadata_fps_source_frame_rate"
    assert [obs["local_time"] for obs in row["frame_observations"]] == [0.0, 1.0, 2.0]


def test_selector_jsonl_restart(tmp_path: Path):
    path = tmp_path / "results.jsonl"
    append_jsonl(path, {"cache_key": "abc", "sample_id": "s"})
    assert load_completed_keys(path) == {"abc"}


def test_repair_jsonl_keeps_valid_lines_and_backs_up(tmp_path: Path):
    path = tmp_path / "broken.jsonl"
    path.write_text('{"cache_key":"abc"}\nnot-json\n{"cache_key":"def"}\n', encoding="utf-8")
    report = repair_jsonl(path)
    assert report["valid_lines"] == 2
    assert len(report["bad_lines"]) == 1
    assert Path(report["backup_path"]).exists()
    assert load_completed_keys(path) == {"abc", "def"}


def test_repair_selector_outputs_reports_only_bad_existing_files(tmp_path: Path):
    good = tmp_path / "top.jsonl"
    bad = tmp_path / "scores.jsonl"
    missing = tmp_path / "errors.jsonl"
    good.write_text('{"cache_key":"abc"}\n', encoding="utf-8")
    bad.write_text('{"cache_key":"abc"}\nnot-json\n', encoding="utf-8")
    reports = repair_selector_outputs(good, bad, missing)
    assert len(reports) == 1
    assert reports[0]["path"] == str(bad)
    assert load_completed_keys(bad) == {"abc"}


def test_qwen_cache_key_changes_with_positions():
    row = build_gt_free_row(sample(n=5), new_data_root=None)
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
    row = build_gt_free_row(s, new_data_root=None)
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


def test_manifest_path_remap_video_and_rc_info():
    s = sample(qa_type="region_caption_gpt", n=2, dataset_name="EgoSurgery")
    s["video"] = [
        "/root/data/CoPESD/high_light_images/000937/1116.jpg",
        "/root/data/AVOS/frames_15fps/rzKGRcXNXm4/4162.jpg",
    ]
    s["RC_info"] = {
        "start_frame": "/root/data/AVOS/frames_15fps/rzKGRcXNXm4/4162.jpg",
        "start_frame_bbox": [1, 2, 3, 4],
    }
    row = build_gt_free_row(
        s,
        old_data_root="/root/data",
        new_data_root="/mnt/hdd3/huihui/hh_datas/MedVidU/valdata",
    )
    assert row["video"][0] == "/mnt/hdd3/huihui/hh_datas/MedVidU/valdata/CoPESD/high_light_images/000937/1116.jpg"
    assert row["video"][1] == "/mnt/hdd3/huihui/hh_datas/MedVidU/valdata/AVOS/frames_15fps/rzKGRcXNXm4/4162.jpg"
    assert row["RC_info"]["start_frame"] == "/mnt/hdd3/huihui/hh_datas/MedVidU/valdata/AVOS/frames_15fps/rzKGRcXNXm4/4162.jpg"


def test_remap_nested_paths_leaves_non_matching_values():
    obj = {"a": "/root/data/x.jpg", "b": "/other/y.jpg", "c": ["/root/data/z.jpg", 3]}
    assert remap_nested_paths(obj, "/root/data", "/new/root") == {
        "a": "/new/root/x.jpg",
        "b": "/other/y.jpg",
        "c": ["/new/root/z.jpg", 3],
    }


def test_sglang_messages_are_timestamped_image_sequence():
    messages = build_sglang_messages(
        prompt="Question?",
        frame_data_urls=["data:image/png;base64,aaa", "data:image/png;base64,bbb"],
        timestamps=[1.25, 2.5],
        media_schema="image_sequence",
        min_pixels=None,
        max_pixels=None,
    )
    content = messages[0]["content"]
    assert content[0] == {"type": "text", "text": "Frame 1: 1.250 seconds"}
    assert content[1]["type"] == "image_url"
    assert content[2] == {"type": "text", "text": "Frame 2: 2.500 seconds"}
    assert content[-1] == {"type": "text", "text": "Question?"}


def test_sglang_cache_key_changes_with_served_model():
    row = build_gt_free_row(sample(n=5), new_data_root=None)
    selection = bypass_selection(row, "m", "c").to_dict()
    decode = {"temperature": 0.0}
    media = {"media_schema": "image_sequence"}
    key1 = sglang_cache_key(row, selection, "qwen-a", "prompt", decode, media)
    key2 = sglang_cache_key(row, selection, "qwen-b", "prompt", decode, media)
    assert key1 != key2


def test_resize_to_max_pixels_preserves_small_images():
    class ImageStub:
        size = (10, 10)

        def resize(self, size):
            raise AssertionError("small image should not resize")

    assert resize_to_max_pixels(ImageStub(), 100) is not None


def test_resize_to_max_pixels_scales_large_images():
    class ImageStub:
        size = (400, 100)

        def resize(self, size):
            out = ImageStub()
            out.size = size
            return out

    resized = resize_to_max_pixels(ImageStub(), 10000)
    assert resized.size == (200, 50)


def test_selector_shard_filter_uses_original_index():
    rows = [{"original_index": i, "sample_id": str(i)} for i in range(10)]
    assert [r["original_index"] for r in filter_rows_for_shard(rows, 3, 0)] == [0, 3, 6, 9]
    assert [r["original_index"] for r in filter_rows_for_shard(rows, 3, 1)] == [1, 4, 7]
    assert [r["original_index"] for r in filter_rows_for_shard(rows, 3, 2)] == [2, 5, 8]


def test_selector_shard_args_validation():
    validate_shard_args(1, 0)
    with pytest.raises(ValueError):
        validate_shard_args(0, 0)
    with pytest.raises(ValueError):
        validate_shard_args(2, 2)


def test_selector_shard_output_paths():
    from baselines.videoitg_qwen35.config import RunConfig

    cfg = RunConfig(output_root=Path("/tmp/out"))
    top, scores, errors = selector_output_paths(cfg, 4, 2)
    assert top.name == "videoitg_top32.shard0002-of0004.jsonl"
    assert scores.name == "videoitg_scores.shard0002-of0004.jsonl"
    assert errors.name == "selector_errors.shard0002-of0004.jsonl"
