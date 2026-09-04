from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = REPO_ROOT / "data_json" / "init_datas" / "medvidu_eccv2026_trainval.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "baselines" / "videoitg_qwen35"
DEFAULT_EVALUATOR = REPO_ROOT / "MedVidBench-Leaderboard" / "evaluation" / "evaluate_predictions.py"

VIDEOITG_REPO = "https://github.com/NVlabs/VideoITG"
VIDEOITG_COMMIT = "50a60a822c0e362bfd8747c45ba34e66e9c9d650"
VIDEOITG_CHECKPOINT = "nvidia/VideoITG-8B"
QWEN35_CHECKPOINT = "Qwen/Qwen3.5-4B"

CANDIDATE_LIMIT = 512
TOP_K = 32
SMOKE_PER_TASK = 2
SMOKE_SEED = 42

TAL_QA_TYPES = {"tal"}
STG_QA_TYPES = {"stg"}
DENSE_CAPTION_QA_TYPES = {"dense_captioning_gpt", "dense_captioning_gemini"}
NEXT_ACTION_QA_TYPES = {"next_action"}
CVS_QA_TYPES = {"cvs_assessment"}
VIDEO_SUMMARY_QA_TYPES = {"video_summary_gpt", "video_summary_gemini"}
REGION_CAPTION_QA_TYPES = {"region_caption_gpt", "region_caption_gemini"}
SKILL_ASSESSMENT_QA_TYPES = {"skill_assessment"}

SELECTOR_APPLICABLE_QA_TYPES = (
    TAL_QA_TYPES
    | STG_QA_TYPES
    | DENSE_CAPTION_QA_TYPES
    | NEXT_ACTION_QA_TYPES
    | CVS_QA_TYPES
    | VIDEO_SUMMARY_QA_TYPES
    | SKILL_ASSESSMENT_QA_TYPES
)
ALL_QA_TYPES = SELECTOR_APPLICABLE_QA_TYPES | REGION_CAPTION_QA_TYPES

TASK_NAME_BY_QA_TYPE = {
    "tal": "tal",
    "stg": "stg",
    "dense_captioning_gpt": "dvc",
    "dense_captioning_gemini": "dvc",
    "next_action": "next_action",
    "cvs_assessment": "cvs_assessment",
    "video_summary_gpt": "vs",
    "video_summary_gemini": "vs",
    "region_caption_gpt": "rc",
    "region_caption_gemini": "rc",
    "skill_assessment": "skill_assessment",
}

LEADERBOARD_TASK_BY_ADAPTER = {
    "tal": "tal",
    "stg": "stg",
    "dvc": "dvc",
    "next_action": "next_action",
    "cvs_assessment": "cvs_assessment",
    "vs": "vs",
    "rc": "rc",
    "skill_assessment": "skill_assessment",
}

MAX_NEW_TOKENS_BY_TASK = {
    "tal": 128,
    "stg": 160,
    "dvc": 384,
    "next_action": 64,
    "cvs_assessment": 96,
    "vs": 384,
    "rc": 160,
    "skill_assessment": 160,
}

GT_FORBIDDEN_KEYS = {
    "answer",
    "gnd",
    "gt",
    "gt_answer",
    "gt_spans",
    "processed_gt_spans",
    "raw_gt_spans",
    "invalid_gt_spans",
    "gt_visible_source_frame_indices",
    "n_gt_visible_total",
    "TRUE_SUPPORT",
    "SPURIOUS_SUPPORT",
    "evidence_stability",
    "reliability_score",
    "alignment_class",
    "gt_alignment_class",
    "IoU",
    "iou",
}

GT_DERIVED_SUBSTRINGS = (
    "gt_",
    "_gt",
    "ground_truth",
    "true_support",
    "spurious_support",
    "evidence_bank",
    "reliability",
)


@dataclass(frozen=True)
class DecodeConfig:
    do_sample: bool = False
    temperature: float | None = None
    top_p: float | None = None
    num_beams: int = 1
    thinking: bool = False


@dataclass(frozen=True)
class PixelConfig:
    total_pixels: int | None = 32000 * 32 * 32
    min_pixels: int | None = None
    max_pixels: int | None = None
    do_resize: bool | None = False


@dataclass(frozen=True)
class RunConfig:
    data_path: Path = DEFAULT_DATA_PATH
    output_root: Path = DEFAULT_OUTPUT_ROOT
    evaluator: Path = DEFAULT_EVALUATOR
    videoitg_repo: str = VIDEOITG_REPO
    videoitg_commit: str = VIDEOITG_COMMIT
    selector_model: str = VIDEOITG_CHECKPOINT
    qwen_model: str = QWEN35_CHECKPOINT
    candidate_limit: int = CANDIDATE_LIMIT
    top_k: int = TOP_K
    smoke_seed: int = SMOKE_SEED
    smoke_per_task: int = SMOKE_PER_TASK
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    pixel: PixelConfig = field(default_factory=PixelConfig)

    @property
    def manifest_dir(self) -> Path:
        return self.output_root / "manifest"

    @property
    def selector_dir(self) -> Path:
        return self.output_root / "selector"

    @property
    def prediction_dir(self) -> Path:
        return self.output_root / "predictions"

    @property
    def evaluation_dir(self) -> Path:
        return self.output_root / "evaluation"

    @property
    def logs_dir(self) -> Path:
        return self.output_root / "logs"

    @property
    def provenance_dir(self) -> Path:
        return self.output_root / "provenance"

    def make_dirs(self) -> None:
        for path in (
            self.manifest_dir,
            self.selector_dir,
            self.prediction_dir,
            self.evaluation_dir,
            self.logs_dir,
            self.provenance_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def to_jsonable(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("data_path", "output_root", "evaluator"):
            data[key] = str(data[key])
        return data
