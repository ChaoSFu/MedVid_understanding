from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

from .config import EVQA_ROOT
from .io_utils import environment_snapshot, run_command, sha256_file, write_json


OFFICIAL_BEHAVIOR_MD = """# Official EVQA Behavior Audit

Upstream inspected from `third_party/EVQA` at the recorded commit.

1. Official model loading: `benchmark/st_evidence_gen/ours_st_evidence.py` calls `train/unipixel/model/builder.py::build_model(model_path, is_trainable=False, merge_adapter=False)`.
2. ST-Evidence-7B architecture: UniPixel-style patched `PixelQwen2_5_VLForConditionalGeneration`.
3. Base VLM: Qwen2.5-VL config path when the checkpoint contains adapter/partial weights; full checkpoints load through `AutoModel` or `Qwen2_5_VLForConditionalGeneration` depending on config.
4. Segmentation backend: integrated SAM2 video predictor/decoder from `train/sam2`; inference masklets are stored on `model.seg`.
5. Native input format: chat messages containing a video payload plus text, processed with `AutoProcessor.apply_chat_template` and `unipixel.dataset.utils.process_vision_info`.
6. Native video/frame loader: benchmark scripts use Decord MP4 loading. UniPixel utilities also support ordered image directories/lists, but official ST-Evidence generation scripts load videos with `VideoReader`.
7. Native FPS / sampling policy: default `--fps 1.0`, `--max-frames 128`; official script computes `sample_frames = int(video_duration * fps)`, clipped to `[1, min(max_frames, total_frames)]`, then uniform samples for LLM.
8. Maximum visual frame behavior: default 128 LLM frames; SAM2 receives the loaded frame tensor when `sample_for_llm_only=True`.
9. Prompt format: single-turn prompt is `Question + Options + Answer the question and provide evidence in the form of temporal (...) and spatial evidence (masks).`; multi-turn path separately asks answer, temporal evidence, then segmentation.
10. Generation config: deterministic by default: `do_sample=False`, `temperature=None`, `top_k=None`, `top_p=None`, `repetition_penalty=None`, `max_new_tokens=512`.
11. Temporal evidence output schema: free text containing `[[start, end], ...]`, parsed by regex into float pairs.
12. Referring expression schema: general MLLM baselines emit textual referring expressions for UniPixel segmentation; UniPixel direct path emits masks from generated `<seg>` tokens rather than textual object IDs.
13. Spatial masklet output schema: `model.seg` is a list of predicted object masklets; each item contains a propagated tensor shaped like frames over `H x W`; official benchmark scripts merge object masks before saving PNGs.
14. Mask propagation logic: generated segmentation token hidden states are passed to SAM2, then `propagate_in_video` produces per-frame masks.
15. Answer output schema: ST-Evidence native benchmark uses MCQ answer letters A-E; parser extracts a letter from the generated text.
16. Official evaluation metrics: ST-Evidence reports QA accuracy, temporal IoU/IoP metrics, and optional mask J/F/J&F. These are secondary diagnostics for MedVidU, not primary MedVidU metrics.

Observed differences from the MedVidU adaptation prompt:

- The current upstream has both `unipixel_st_evidence.py` and `ours_st_evidence.py`; `ours_st_evidence.py` single-turn and `unipixel_st_evidence.py` multi-turn differ slightly.
- Official benchmark scripts are MP4-first, but UniPixel lower-level utilities include frame-directory/frame-list loaders. The MedVidU adapter must avoid creating fake MP4s.
- Official ST-Evidence is MCQ-oriented; MedVidU STG/RC/CVS are adaptations, not native E-VQA benchmark tasks.
"""


def record_upstream_provenance(evqa_root: Path, provenance_dir: Path) -> dict[str, Any]:
    provenance_dir.mkdir(parents=True, exist_ok=True)
    commit = run_command(["git", "rev-parse", "HEAD"], evqa_root).strip()
    status = run_command(["git", "status", "--short"], evqa_root)
    remote = run_command(["git", "remote", "-v"], evqa_root)
    (provenance_dir / "evqa_git_commit.txt").write_text(commit + "\n", encoding="utf-8")
    (provenance_dir / "evqa_git_status_before.txt").write_text(status, encoding="utf-8")
    (provenance_dir / "evqa_git_remote.txt").write_text(remote, encoding="utf-8")
    (provenance_dir / "official_evqa_behavior.md").write_text(OFFICIAL_BEHAVIOR_MD, encoding="utf-8")
    diff = run_command(["git", "diff"], evqa_root)
    (provenance_dir / "evqa_git_diff_after.txt").write_text(diff, encoding="utf-8")
    return {"commit": commit, "status": status, "remote": remote, "diff_after": diff}


def model_fingerprint(model_path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model_path": str(model_path),
        "exists": model_path.exists(),
        "expected_model": "Salesforce/ST-Evidence-7B",
        "auto_download_attempted": False,
    }
    if not model_path.exists():
        payload["error"] = "MODEL_PATH_NOT_FOUND"
        return payload
    for name in ("config.json", "generation_config.json", "processor_config.json", "preprocessor_config.json"):
        path = model_path / name
        if path.exists():
            payload[f"{name}_sha256"] = sha256_file(path)
    try:
        evqa_train_root = EVQA_ROOT / "train"
        if evqa_train_root.exists() and str(evqa_train_root) not in sys.path:
            sys.path.insert(0, str(evqa_train_root))
        try:
            import unipixel.model  # noqa: F401
        except Exception as exc:
            payload["evqa_model_registration_error"] = repr(exc)

        from transformers import AutoConfig, AutoProcessor

        cfg = AutoConfig.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=True)
        proc = AutoProcessor.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=True)
        payload.update(
            {
                "model_config_class": cfg.__class__.__name__,
                "model_type": getattr(cfg, "model_type", None),
                "base_model_path": getattr(cfg, "base_model_path", None),
                "sam2_config": getattr(cfg, "sam2_config", None),
                "sam2_checkpoint": getattr(cfg, "sam2_checkpoint", None),
                "sam2_image_size": getattr(cfg, "sam2_image_size", None),
                "processor_class": proc.__class__.__name__,
            }
        )
    except Exception as exc:
        payload["config_read_error"] = repr(exc)
    return payload


def write_environment_snapshot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(environment_snapshot(), encoding="utf-8")


def ensure_evqa_present(evqa_root: Path = EVQA_ROOT) -> None:
    if not evqa_root.exists():
        raise FileNotFoundError(f"Official EVQA repository is missing: {evqa_root}")
