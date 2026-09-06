from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = REPO_ROOT / "data_json" / "init_datas" / "medvidu_eccv2026_trainval.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "baselines" / "timechat7b_activitynet_vtune_medvidu_tal"
DEFAULT_OLD_DATA_ROOT = "/root/data"
DEFAULT_NEW_DATA_ROOT = os.environ.get("MEDVIDU_DATA_ROOT", "/mnt/hdd3/huihui/hh_datas/MedVidU/valdata")

BASELINE_NAME = "TimeChat-7B-ActivityNet-VTune"
TASK_NAME = "MedVidU TAL"
SCIENTIFIC_NAME = "TimeChat-7B-ActivityNet-VTune adapted to MedVidU TAL"

MAX_FRAMES = 96
IMAGE_SIZE = 224
TIMESTAMP_ROUNDING_SECONDS = 0.1
QUERY_ADAPTER_VERSION = "medvidu_human_question_verbatim_v1"
PROMPT_VERSION = "official_vtune_grounding"
TIMESTAMP_ADAPTER_VERSION = "medvidu_gt_free_clip_local_timechat_qformer_text_v1"
FRAME_LOADER_VERSION = "medvidu_frame_list_timechat_uint8_rgb_224_v1"
CACHE_VERSION = "timechat_medvidu_cache_v1"
SMOKE_SIZE = 20
SMOKE_SEED = 42

OFFICIAL_VTUNE_GROUNDING_PROMPT = (
    "Localize the visual content described by the given textual query '{event}' in the video, "
    "and output the start and end timestamps in seconds."
)
OFFICIAL_SYSTEM_PROMPT = (
    "You are able to understand the visual content that the user provides. "
    "Follow the instructions carefully and explain your answers in detail."
)

GT_FORBIDDEN_KEYS = {
    "answer",
    "assistant",
    "conversations",
    "gnd",
    "gt",
    "gt_answer",
    "gt_spans",
    "processed_gt_spans",
    "raw_gt_spans",
    "invalid_gt_spans",
    "ground_truth",
    "struc_info",
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
class GenerationConfig:
    num_beams: int = 1
    do_sample: bool = False
    temperature: float = 0.05
    max_new_tokens: int = 300
    max_length: int = 2000


@dataclass(frozen=True)
class RunConfig:
    data_path: Path = DEFAULT_DATA_PATH
    output_root: Path = DEFAULT_OUTPUT_ROOT
    old_data_root: str = DEFAULT_OLD_DATA_ROOT
    new_data_root: str | None = DEFAULT_NEW_DATA_ROOT
    smoke_size: int = SMOKE_SIZE
    smoke_seed: int = SMOKE_SEED
    generation: GenerationConfig = field(default_factory=GenerationConfig)

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
        for path in (self.manifest_dir, self.prediction_dir, self.evaluation_dir, self.audit_dir, self.provenance_dir):
            path.mkdir(parents=True, exist_ok=True)

    def to_jsonable(self) -> dict[str, Any]:
        data = asdict(self)
        data["data_path"] = str(self.data_path)
        data["output_root"] = str(self.output_root)
        data.update(
            {
                "baseline": BASELINE_NAME,
                "task": TASK_NAME,
                "max_frames": MAX_FRAMES,
                "sampling": "uniform_official_style",
                "image_size": IMAGE_SIZE,
                "timestamp_rounding_seconds": TIMESTAMP_ROUNDING_SECONDS,
                "query_adapter": QUERY_ADAPTER_VERSION,
                "prompt": PROMPT_VERSION,
                "training_on_medvidu": False,
                "gt_available_to_inference": False,
            }
        )
        return data

