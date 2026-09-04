from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
TIMELENS_ROOT = REPO_ROOT / "third_party" / "TimeLens-main"

DEFAULT_DATA_PATH = REPO_ROOT / "data_json" / "init_datas" / "medvidu_eccv2026_trainval.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "baselines" / "timelens8b_medvidu_tal"
DEFAULT_OLD_DATA_ROOT = "/root/data"
DEFAULT_NEW_DATA_ROOT = os.environ.get("MEDVIDU_DATA_ROOT", "/mnt/hdd3/huihui/hh_datas/MedVidU/valdata")

MODEL_NAME = "TimeLens-8B"
TIMELENS_MODEL_ID = "TencentARC/TimeLens-8B"
TIMESTAMP_ADAPTER_VERSION = "medvidu_frame_list_effective_fps_v1"
PROMPT_VERSION = "official_timelens_grounder_prompt"
GROUNDER_PROMPT = (
    "Please find the visual event described by the sentence '{}', determining its starting and ending times. "
    "The format should be: 'The event happens in <start time> - <end time> seconds'."
)

MIN_TOKENS = 64
TOTAL_TOKENS = 14336
QWEN3_IMAGE_PATCH_SIZE = 16
QWEN3_DOWNSAMPLE_RATE = 32
NATIVE_TIMELENS_RAW_VIDEO_FPS = 2
SMOKE_SIZE = 20
SMOKE_SEED = 42

GT_FORBIDDEN_KEYS = {
    "answer",
    "gnd",
    "gt",
    "gt_answer",
    "gt_spans",
    "processed_gt_spans",
    "raw_gt_spans",
    "invalid_gt_spans",
    "ground_truth",
    "TRUE_SUPPORT",
    "SPURIOUS_SUPPORT",
    "evidence_stability",
    "evidence_density",
    "gt_evidence_recall",
    "reliability_score",
    "alignment_class",
    "gt_alignment_class",
    "IoU",
    "iou",
    "accuracy",
}

GT_DERIVED_SUBSTRINGS = (
    "true_support",
    "spurious_support",
    "ground_truth",
    "evidence_bank",
    "phase_b",
    "phase_c",
    "phase_d",
    "phase_f",
    "reliability",
)


@dataclass(frozen=True)
class DecodingConfig:
    do_sample: bool = False
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_new_tokens: int = 512


@dataclass(frozen=True)
class PixelConfig:
    min_tokens: int = MIN_TOKENS
    total_tokens: int = TOTAL_TOKENS
    downsample_rate: int = QWEN3_DOWNSAMPLE_RATE

    @property
    def min_pixels(self) -> int:
        return self.min_tokens * self.downsample_rate * self.downsample_rate

    @property
    def total_pixels(self) -> int:
        return self.total_tokens * self.downsample_rate * self.downsample_rate

    def to_jsonable(self) -> dict[str, int]:
        return {
            "min_tokens": self.min_tokens,
            "total_tokens": self.total_tokens,
            "downsample_rate": self.downsample_rate,
            "min_pixels": self.min_pixels,
            "total_pixels": self.total_pixels,
        }


@dataclass(frozen=True)
class RunConfig:
    data_path: Path = DEFAULT_DATA_PATH
    output_root: Path = DEFAULT_OUTPUT_ROOT
    old_data_root: str = DEFAULT_OLD_DATA_ROOT
    new_data_root: str | None = DEFAULT_NEW_DATA_ROOT
    timelens_root: Path = TIMELENS_ROOT
    smoke_size: int = SMOKE_SIZE
    smoke_seed: int = SMOKE_SEED
    pixel: PixelConfig = field(default_factory=PixelConfig)
    decoding: DecodingConfig = field(default_factory=DecodingConfig)

    @property
    def manifest_dir(self) -> Path:
        return self.output_root / "manifest"

    @property
    def prediction_dir(self) -> Path:
        return self.output_root / "predictions"

    @property
    def evaluation_dir(self) -> Path:
        return self.output_root / "evaluation"

    @property
    def audit_dir(self) -> Path:
        return self.output_root / "audit"

    @property
    def provenance_dir(self) -> Path:
        return self.output_root / "provenance"

    def make_dirs(self) -> None:
        for path in (
            self.manifest_dir,
            self.prediction_dir,
            self.evaluation_dir,
            self.audit_dir,
            self.provenance_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def to_jsonable(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("data_path", "output_root", "timelens_root"):
            data[key] = str(data[key])
        data["pixel"] = self.pixel.to_jsonable()
        return data

