from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
EVQA_ROOT = REPO_ROOT / "third_party" / "EVQA"
EVQA_TRAIN_ROOT = EVQA_ROOT / "train"
DEFAULT_DATA_PATH = REPO_ROOT / "data_json" / "init_datas" / "medvidu_eccv2026_trainval.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "baselines" / "evqa_medvidu_v3_text_contracts_no_seg"
DEFAULT_OLD_DATA_ROOT = "/root/data"
DEFAULT_NEW_DATA_ROOT = os.environ.get("MEDVIDU_DATA_ROOT", "/mnt/hdd3/huihui/hh_datas/MedVidU/valdata")

BASELINE_NAME = "E-VQA (ST-Evidence-7B)"
STG_SCIENTIFIC_LABEL = "E-VQA (ST-Evidence-7B), adapted to MedVidU STG"
RC_SCIENTIFIC_LABEL = "E-VQA (ST-Evidence-7B), region-conditioned adaptation"
CVS_SCIENTIFIC_LABEL = "E-VQA (ST-Evidence-7B), zero-shot structured reasoning"

OFFICIAL_FPS = 1.0
OFFICIAL_MAX_FRAMES = 128
OFFICIAL_MIN_PIXELS = 128 * 28 * 28
OFFICIAL_MAX_PIXELS_PER_FRAME = 256 * 28 * 28
VISUAL_RESOLUTION_POLICY_VERSION = "per_frame_max_pixels_v1"

PROMPT_VERSION = "medvidu_task_output_contracts_v3_no_seg_for_text_tasks"
CVS_OUTPUT_CONTRACT = (
    "Output exactly one line in this format: Two structures: <0|1|2>, Cystic plate: <0|1|2>, "
    "Hepatocystic triangle: <0|1|2>. Do not output masks, temporal evidence, special tokens, or explanations."
)
RC_OUTPUT_CONTRACT = (
    "Answer with only a concise plain-English description of the referenced object and its action. "
    "Do not output masks or special tokens."
)
SAMPLING_VERSION = "medvidu_all_benchmark_frames_no_downsampling_v1"
FRAME_ADAPTER_VERSION = "ordered_benchmark_frame_list_no_fake_mp4_v1"
TIMESTAMP_ADAPTER_VERSION = "medvidu_gt_free_clip_local_timelens_mapper_v1"
MASK_TO_BBOX_VERSION = "tight_bbox_v1"
CACHE_VERSION = "evqa_medvidu_cache_v1"
SMOKE_SIZE = 10
SMOKE_SEED = 42

GT_FORBIDDEN_KEYS = {
    "answer",
    "assistant",
    "conversations",
    "gnd",
    "gt",
    "gt_answer",
    "ground_truth",
    "struc_info",
    "bbox_dict",
    "cvs_scores",
    "critical_view_achieved",
    "target_caption",
    "target_answer",
    "label",
}
GT_DERIVED_SUBSTRINGS = (
    "ground_truth",
    "bbox_dict",
    "cvs_score",
    "critical_view",
    "target_caption",
    "target_answer",
)


@dataclass(frozen=True)
class GenerationConfig:
    do_sample: bool = False
    temperature: float | None = None
    top_k: float | None = None
    top_p: float | None = None
    repetition_penalty: float | None = None
    max_new_tokens: int = 512


@dataclass(frozen=True)
class RunConfig:
    data_path: Path = DEFAULT_DATA_PATH
    output_root: Path = DEFAULT_OUTPUT_ROOT
    evqa_root: Path = EVQA_ROOT
    old_data_root: str = DEFAULT_OLD_DATA_ROOT
    new_data_root: str | None = DEFAULT_NEW_DATA_ROOT
    smoke_size: int = SMOKE_SIZE
    smoke_seed: int = SMOKE_SEED
    fps: float = OFFICIAL_FPS
    max_frames: int | None = None
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    @property
    def provenance_dir(self) -> Path:
        return self.output_root / "provenance"

    @property
    def official_reproduction_dir(self) -> Path:
        return self.output_root / "official_reproduction"

    @property
    def manifest_dir(self) -> Path:
        return self.output_root / "manifest"

    @property
    def prediction_dir(self) -> Path:
        return self.output_root / "predictions"

    @property
    def raw_prediction_dir(self) -> Path:
        return self.prediction_dir / "raw"

    @property
    def medvidu_prediction_dir(self) -> Path:
        return self.prediction_dir / "medvidu"

    @property
    def error_dir(self) -> Path:
        return self.prediction_dir / "errors"

    @property
    def evaluation_dir(self) -> Path:
        return self.output_root / "evaluation"

    @property
    def audit_dir(self) -> Path:
        return self.output_root / "audit"

    @property
    def visualization_dir(self) -> Path:
        return self.output_root / "visualizations"

    def make_dirs(self) -> None:
        for path in (
            self.provenance_dir,
            self.official_reproduction_dir,
            self.manifest_dir,
            self.raw_prediction_dir,
            self.medvidu_prediction_dir,
            self.error_dir,
            self.evaluation_dir,
            self.audit_dir,
            self.visualization_dir / "stg",
            self.visualization_dir / "rc",
            self.visualization_dir / "cvs",
        ):
            path.mkdir(parents=True, exist_ok=True)

    def to_jsonable(self) -> dict[str, Any]:
        data = asdict(self)
        data["data_path"] = str(self.data_path)
        data["output_root"] = str(self.output_root)
        data["evqa_root"] = str(self.evqa_root)
        data.update(
            {
                "baseline": BASELINE_NAME,
                "stg_label": STG_SCIENTIFIC_LABEL,
                "rc_label": RC_SCIENTIFIC_LABEL,
                "cvs_label": CVS_SCIENTIFIC_LABEL,
                "training_on_medvidu": False,
                "lora_on_medvidu": False,
                "gt_available_to_inference": False,
                "source_raw_video_access": False,
                "fake_mp4_used": False,
                "prompt_version": PROMPT_VERSION,
                "sampling_version": SAMPLING_VERSION,
                "visual_resolution_policy_version": VISUAL_RESOLUTION_POLICY_VERSION,
                "visual_min_pixels": OFFICIAL_MIN_PIXELS,
                "visual_max_pixels_per_frame": OFFICIAL_MAX_PIXELS_PER_FRAME,
                "frame_adapter_version": FRAME_ADAPTER_VERSION,
                "timestamp_adapter_version": TIMESTAMP_ADAPTER_VERSION,
                "mask_to_bbox_version": MASK_TO_BBOX_VERSION,
                "cache_version": CACHE_VERSION,
            }
        )
        return data
