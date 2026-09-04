from __future__ import annotations

import argparse
import math
import sys
import traceback
from pathlib import Path
from typing import Any

from ..config import CANDIDATE_LIMIT, TOP_K, VIDEOITG_CHECKPOINT, VIDEOITG_COMMIT, RunConfig
from ..io_utils import append_jsonl, assert_no_gt_leak, environment_snapshot, load_completed_keys, read_jsonl, repair_jsonl, sha256_json, write_json
from .base import SelectionResult, official_uniform_candidate_positions, topk_chronological


class VideoITGSelector:
    selector_name = "videoitg"

    def __init__(
        self,
        videoitg_repo_dir: Path,
        checkpoint: str = VIDEOITG_CHECKPOINT,
        commit: str = VIDEOITG_COMMIT,
        candidate_limit: int = CANDIDATE_LIMIT,
        top_k: int = TOP_K,
        device: str = "cuda:0",
    ) -> None:
        self.videoitg_repo_dir = videoitg_repo_dir
        self.checkpoint = checkpoint
        self.commit = commit
        self.candidate_limit = candidate_limit
        self.top_k = top_k
        self.device = device
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        if str(self.videoitg_repo_dir) not in sys.path:
            sys.path.insert(0, str(self.videoitg_repo_dir))
        try:
            import torch
            from eagle.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
            from eagle.mm_utils import get_model_name_from_path, tokenizer_image_token
            from eagle.model.builder import load_pretrained_model
        except Exception as exc:
            raise ImportError(
                "Could not import official VideoITG/eagle modules. Run this in the separate "
                "VideoITG environment with the official repository on --videoitg-repo-dir."
            ) from exc

        self.torch = torch
        self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self.tokenizer_image_token = tokenizer_image_token
        tokenizer, model, image_processor, _ = load_pretrained_model(
            self.checkpoint,
            None,
            get_model_name_from_path(self.checkpoint),
            device_map=self.device,
        )
        self.tokenizer = tokenizer
        self.model = model.half().eval().to(torch.device(self.device))
        self.image_processor = image_processor
        self._loaded = True

    def score_medvidu_frame_list(
        self,
        prompt: str,
        ordered_frame_paths: list[str],
        original_positions: list[int],
    ) -> list[float]:
        if len(ordered_frame_paths) != len(original_positions):
            raise ValueError("ordered_frame_paths and original_positions must have equal length")
        self._load()
        from PIL import Image

        images = []
        for path, original_position in zip(ordered_frame_paths, original_positions):
            try:
                images.append(Image.open(path).convert("RGB"))
            except Exception as exc:
                raise RuntimeError(
                    "Failed to read MedVidU frame for VideoITG selector: "
                    f"original_position={original_position} path={path!r} error={exc!r}"
                ) from exc
        video_tensor = self.image_processor.preprocess(images, return_tensors="pt")["pixel_values"]
        video_tensor = video_tensor.half().to(self.torch.device(self.device))
        processed_video = [video_tensor]

        full_prompt = self.DEFAULT_IMAGE_TOKEN + prompt + "\n"
        input_ids = self.tokenizer_image_token(
            full_prompt,
            self.tokenizer,
            self.IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        )
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        input_ids = pad_sequence(self.torch, self.tokenizer, [input_ids], True, pad_token_id).to(self.torch.device(self.device))
        attention_mask = input_ids.ne(pad_token_id).to(self.torch.device(self.device))

        with self.torch.inference_mode():
            response = self.model(
                input_ids,
                attention_mask=attention_mask,
                images=processed_video,
            )
        scores = response.logits[0].sigmoid().view(-1).detach().float().cpu().tolist()
        if len(scores) != len(original_positions):
            raise RuntimeError(f"VideoITG returned {len(scores)} scores for {len(original_positions)} candidate frames")
        if not all(math.isfinite(float(x)) for x in scores):
            raise RuntimeError("VideoITG returned non-finite relevance score")
        return [float(x) for x in scores]

    def select(self, manifest_row: dict[str, Any]) -> SelectionResult:
        if not manifest_row.get("selector_applicable", False):
            return bypass_selection(manifest_row, self.checkpoint, self.commit)

        n_frames = int(manifest_row["n_medvidu_frames"])
        candidate_positions = official_uniform_candidate_positions(n_frames, self.candidate_limit)
        frame_paths = list(manifest_row["video"])
        candidate_paths = [frame_paths[pos] for pos in candidate_positions]
        scores = self.score_medvidu_frame_list(
            str(manifest_row["question"]),
            candidate_paths,
            candidate_positions,
        )
        return build_selection_result(
            manifest_row=manifest_row,
            candidate_positions=candidate_positions,
            scores=scores,
            selector_model=self.checkpoint,
            selector_commit=self.commit,
            top_k=self.top_k,
            candidate_limit=self.candidate_limit,
        )


def pad_sequence(torch_module, tokenizer, input_ids, batch_first, padding_value):
    if tokenizer.padding_side == "left":
        input_ids = [torch_module.flip(_input_ids, [0]) for _input_ids in input_ids]
    input_ids = torch_module.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
    if tokenizer.padding_side == "left":
        input_ids = torch_module.flip(input_ids, [1])
    return input_ids


def selection_cache_key(
    manifest_row: dict[str, Any],
    candidate_positions: list[int],
    selector_model: str,
    selector_commit: str,
    candidate_limit: int,
    top_k: int,
) -> str:
    return sha256_json(
        {
            "sample_id": manifest_row["sample_id"],
            "ordered_candidate_logical_positions": candidate_positions,
            "ordered_candidate_frame_paths_hash": sha256_json(
                [manifest_row["video"][pos] for pos in candidate_positions]
            ),
            "question_hash": sha256_json(manifest_row.get("question", "")),
            "selector_model": selector_model,
            "selector_commit": selector_commit,
            "candidate_limit": candidate_limit,
            "top_k": top_k,
        }
    )


def build_selection_result(
    manifest_row: dict[str, Any],
    candidate_positions: list[int],
    scores: list[float],
    selector_model: str,
    selector_commit: str,
    top_k: int,
    candidate_limit: int = CANDIDATE_LIMIT,
) -> SelectionResult:
    ranked_positions, ranked_scores, topk_score_order, selected_chronological = topk_chronological(candidate_positions, scores, top_k)
    observations = {int(obs["frame_position"]): obs for obs in manifest_row.get("frame_observations", [])}
    selected_source = [int(observations[pos]["source_frame_index"]) for pos in selected_chronological]
    selected_paths = [str(observations[pos]["frame_path"]) for pos in selected_chronological]
    selected_times = [float(observations[pos]["local_time"]) for pos in selected_chronological]
    result = SelectionResult(
        sample_id=str(manifest_row["sample_id"]),
        qa_type=str(manifest_row["qa_type"]),
        dataset_name=manifest_row.get("dataset_name"),
        n_medvidu_frames=int(manifest_row["n_medvidu_frames"]),
        n_selector_candidates=len(candidate_positions),
        effective_k=min(top_k, len(candidate_positions)),
        candidate_original_positions=list(candidate_positions),
        ranked_original_positions=ranked_positions,
        ranked_scores=ranked_scores,
        topk_original_positions_score_order=topk_score_order,
        selected_original_positions_chronological=selected_chronological,
        selected_source_frame_indices_chronological=selected_source,
        selected_frame_paths_chronological=selected_paths,
        selected_local_times_chronological=selected_times,
        selector_model=selector_model,
        selector_commit=selector_commit,
        selector_applicable=True,
        selector_reason=None,
        cache_key=selection_cache_key(manifest_row, candidate_positions, selector_model, selector_commit, candidate_limit, top_k),
    )
    assert_no_gt_leak(result.to_dict())
    return result


def bypass_selection(manifest_row: dict[str, Any], selector_model: str, selector_commit: str) -> SelectionResult:
    result = SelectionResult(
        sample_id=str(manifest_row["sample_id"]),
        qa_type=str(manifest_row["qa_type"]),
        dataset_name=manifest_row.get("dataset_name"),
        n_medvidu_frames=int(manifest_row["n_medvidu_frames"]),
        n_selector_candidates=0,
        effective_k=0,
        candidate_original_positions=[],
        ranked_original_positions=[],
        ranked_scores=[],
        topk_original_positions_score_order=[],
        selected_original_positions_chronological=[],
        selected_source_frame_indices_chronological=[],
        selected_frame_paths_chronological=[],
        selected_local_times_chronological=[],
        selector_model=selector_model,
        selector_commit=selector_commit,
        selector_applicable=False,
        selector_reason=manifest_row.get("selector_reason"),
        cache_key=selection_cache_key(manifest_row, [], selector_model, selector_commit, CANDIDATE_LIMIT, TOP_K),
    )
    assert_no_gt_leak(result.to_dict())
    return result


def validate_selection(row: dict[str, Any]) -> None:
    if not row.get("selector_applicable", False):
        return
    n_frames = int(row["n_medvidu_frames"])
    candidates = [int(x) for x in row["candidate_original_positions"]]
    scores = [float(x) for x in row["ranked_scores"]]
    ranked = [int(x) for x in row["ranked_original_positions"]]
    selected = [int(x) for x in row["selected_original_positions_chronological"]]
    if len(candidates) != int(row["n_selector_candidates"]):
        raise AssertionError("candidate count mismatch")
    if len(scores) != len(ranked):
        raise AssertionError("score count mismatch")
    if any(pos < 0 or pos >= n_frames for pos in candidates + ranked + selected):
        raise AssertionError("selected/candidate position out of range")
    if selected != sorted(selected):
        raise AssertionError("selected positions are not chronological")
    if len(selected) != int(row["effective_k"]):
        raise AssertionError("effective_k mismatch")
    if [int(x) for x in row["topk_original_positions_score_order"]] != ranked[: int(row["effective_k"])]:
        raise AssertionError("top-k score order mismatch")
    if selected != sorted(row["topk_original_positions_score_order"]):
        raise AssertionError("chronological reorder mismatch")
    if not all(math.isfinite(x) for x in scores):
        raise AssertionError("non-finite score")
    assert_no_gt_leak(row)


def validate_shard_args(num_shards: int, shard_index: int) -> None:
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards")


def filter_rows_for_shard(rows: list[dict[str, Any]], num_shards: int, shard_index: int) -> list[dict[str, Any]]:
    validate_shard_args(num_shards, shard_index)
    if num_shards == 1:
        return rows
    return [
        row
        for row in rows
        if int(row.get("original_index", 0)) % num_shards == shard_index
    ]


def selector_output_paths(cfg: RunConfig, num_shards: int, shard_index: int) -> tuple[Path, Path, Path]:
    validate_shard_args(num_shards, shard_index)
    if num_shards == 1:
        return (
            cfg.selector_dir / "videoitg_top32.jsonl",
            cfg.selector_dir / "videoitg_scores.jsonl",
            cfg.selector_dir / "selector_errors.jsonl",
        )
    suffix = f"shard{shard_index:04d}-of{num_shards:04d}.jsonl"
    return (
        cfg.selector_dir / f"videoitg_top32.{suffix}",
        cfg.selector_dir / f"videoitg_scores.{suffix}",
        cfg.selector_dir / f"selector_errors.{suffix}",
    )


def repair_selector_outputs(output_path: Path, scores_path: Path, errors_path: Path) -> list[dict[str, Any]]:
    reports = [
        repair_jsonl(output_path),
        repair_jsonl(scores_path),
        repair_jsonl(errors_path),
    ]
    return [report for report in reports if report["exists"] and report["bad_lines"]]


def parse_args() -> argparse.Namespace:
    cfg = RunConfig()
    p = argparse.ArgumentParser(description="Run frozen VideoITG selector on a MedVidU frame-list manifest.")
    p.add_argument("--manifest", type=Path, default=cfg.manifest_dir / "medvidu_videoitg_manifest_gt_free.smoke.jsonl")
    p.add_argument("--output-root", type=Path, default=cfg.output_root)
    p.add_argument("--videoitg-repo-dir", type=Path, required=True)
    p.add_argument("--checkpoint", default=cfg.selector_model)
    p.add_argument("--commit", default=cfg.videoitg_commit)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--candidate-limit", type=int, default=cfg.candidate_limit)
    p.add_argument("--top-k", type=int, default=cfg.top_k)
    p.add_argument("--num-shards", type=int, default=1, help="Total selector shards for multi-GPU parallel runs.")
    p.add_argument("--shard-index", type=int, default=0, help="This process shard index, 0-based.")
    p.add_argument(
        "--no-auto-repair-jsonl",
        action="store_true",
        help="Disable startup repair of interrupted selector JSONL outputs.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    validate_shard_args(args.num_shards, args.shard_index)
    cfg = RunConfig(output_root=args.output_root)
    cfg.make_dirs()
    (cfg.provenance_dir / "videoitg_commit.txt").write_text(args.commit + "\n", encoding="utf-8")
    (cfg.provenance_dir / "videoitg_environment.txt").write_text(environment_snapshot(), encoding="utf-8")

    selector = VideoITGSelector(
        videoitg_repo_dir=args.videoitg_repo_dir,
        checkpoint=args.checkpoint,
        commit=args.commit,
        candidate_limit=args.candidate_limit,
        top_k=args.top_k,
        device=args.device,
    )
    rows = filter_rows_for_shard(read_jsonl(args.manifest), args.num_shards, args.shard_index)
    output_path, scores_path, errors_path = selector_output_paths(cfg, args.num_shards, args.shard_index)
    if not args.no_auto_repair_jsonl:
        repair_reports = repair_selector_outputs(output_path, scores_path, errors_path)
        if repair_reports:
            report_path = cfg.selector_dir / f"selector_startup_repair_shard{args.shard_index:04d}-of{args.num_shards:04d}.json"
            write_json(report_path, repair_reports)
    completed = load_completed_keys(output_path)
    for row in rows:
        candidate_positions = official_uniform_candidate_positions(int(row["n_medvidu_frames"]), args.candidate_limit) if row.get("selector_applicable") else []
        cache_key = selection_cache_key(row, candidate_positions, args.checkpoint, args.commit, args.candidate_limit, args.top_k)
        if cache_key in completed:
            continue
        try:
            result = selector.select(row).to_dict()
            validate_selection(result)
            append_jsonl(output_path, result)
            append_jsonl(scores_path, result)
            completed.add(str(result["cache_key"]))
        except Exception as exc:
            append_jsonl(
                errors_path,
                {
                    "sample_id": row.get("sample_id"),
                    "stage": "selector",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                },
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
