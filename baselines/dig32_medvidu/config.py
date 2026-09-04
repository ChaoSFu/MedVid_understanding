from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
from typing import Any

from baselines.videoitg_qwen35.config import (
    ALL_QA_TYPES,
    DEFAULT_DATA_PATH,
    DEFAULT_EVALUATOR,
    DEFAULT_NEW_DATA_ROOT,
    DEFAULT_OLD_DATA_ROOT,
    REGION_CAPTION_QA_TYPES,
    SELECTOR_APPLICABLE_QA_TYPES,
    TASK_NAME_BY_QA_TYPE,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "baselines" / "dig32_medvidu"
DEFAULT_DIG_REPO_DIR = REPO_ROOT / "third_party" / "DIG"

SELECTOR_VERSION = "dig32_fixed_v1"
REQUESTED_K = 32
VIDEO_REFINEMENT_WLEN = 2
QUERY_IDENTIFIER_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"
REWARD_LMM = "Qwen3-VL-8B-Instruct"
REWARD_LMM_PATH = "Qwen/Qwen3-VL-8B-Instruct"
CAFS_MODEL = "facebook/dinov2-base"
CAFS_SAMPLE_PER_SEC = 2
CAFS_INFER_BATCH_SIZE = 64
SMOKE_PER_TASK = 2
SMOKE_SEED = 42

TARGET_MODELS = {
    "qwen3vl_4b": "Qwen/Qwen3-VL-4B-Instruct",
    "qwen3vl_8b": "Qwen/Qwen3-VL-8B-Instruct",
    "qwen35_4b": "Qwen/Qwen3.5-4B",
    "qwen38_27b": "Qwen/Qwen3.8-27B",
}


@dataclass(frozen=True)
class DecodeConfig:
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 1
    repetition_penalty: float = 1.0
    num_beams: int = 1
    thinking: bool = False


@dataclass(frozen=True)
class PixelConfig:
    total_pixels: int | None = 32000 * 32 * 32
    min_pixels: int | None = None
    max_pixels: int | None = None
    do_resize: bool | None = False


@dataclass(frozen=True)
class SelectorConfig:
    selector_version: str = SELECTOR_VERSION
    requested_k: int = REQUESTED_K
    video_refinement_wlen: int = VIDEO_REFINEMENT_WLEN
    query_identifier_model: str = QUERY_IDENTIFIER_MODEL
    reward_lmm: str = REWARD_LMM
    reward_lmm_path: str = REWARD_LMM_PATH
    cafs_model: str = CAFS_MODEL
    cafs_sample_per_sec: int = CAFS_SAMPLE_PER_SEC
    cafs_infer_batch_size: int = CAFS_INFER_BATCH_SIZE

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunConfig:
    data_path: Path = DEFAULT_DATA_PATH
    output_root: Path = DEFAULT_OUTPUT_ROOT
    evaluator: Path = DEFAULT_EVALUATOR
    dig_repo_dir: Path = DEFAULT_DIG_REPO_DIR
    old_data_root: str = DEFAULT_OLD_DATA_ROOT
    new_data_root: str | None = DEFAULT_NEW_DATA_ROOT
    smoke_seed: int = SMOKE_SEED
    smoke_per_task: int = SMOKE_PER_TASK
    selector: SelectorConfig = field(default_factory=SelectorConfig)
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    pixel: PixelConfig = field(default_factory=PixelConfig)
    query_base_url: str = os.environ.get("DIG_QUERY_BASE_URL", "http://localhost:8000/v1")
    query_api_key: str = os.environ.get("DIG_QUERY_API_KEY", "token-abc123")
    reward_base_url: str = os.environ.get("DIG_REWARD_BASE_URL", "http://localhost:8000/v1")
    reward_api_key: str = os.environ.get("DIG_REWARD_API_KEY", "token-abc123")

    @property
    def selector_dir(self) -> Path:
        return self.output_root / "selector"

    @property
    def targets_dir(self) -> Path:
        return self.output_root / "targets"

    @property
    def evaluation_dir(self) -> Path:
        return self.output_root / "evaluation"

    @property
    def provenance_dir(self) -> Path:
        return self.output_root / "provenance"

    @property
    def logs_dir(self) -> Path:
        return self.output_root / "logs"

    def make_dirs(self) -> None:
        for path in (
            self.selector_dir,
            self.targets_dir,
            self.evaluation_dir,
            self.provenance_dir,
            self.logs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def to_jsonable(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("data_path", "output_root", "evaluator", "dig_repo_dir"):
            data[key] = str(data[key])
        return data
