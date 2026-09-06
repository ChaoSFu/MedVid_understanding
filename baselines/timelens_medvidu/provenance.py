from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

from .config import (
    DEFAULT_OUTPUT_ROOT,
    GROUNDER_PROMPT,
    MIN_TOKENS,
    NATIVE_TIMELENS_RAW_VIDEO_FPS,
    QWEN3_DOWNSAMPLE_RATE,
    QWEN3_IMAGE_PATCH_SIZE,
    TIMELENS_ROOT,
    TOTAL_TOKENS,
)
from .io_utils import environment_snapshot, git_text, sha256_file, write_json


OFFICIAL_FILES = [
    "README.md",
    "scripts/eval_timelens_bench.sh",
    "evaluation/eval_dataloader.py",
    "evaluation/utils.py",
    "evaluation/compute_metrics.py",
    "timelens/dataset/timelens_data.py",
    "timelens/utils.py",
]


def write_timelens_git_provenance(timelens_root: Path, provenance_dir: Path) -> dict[str, Any]:
    provenance_dir.mkdir(parents=True, exist_ok=True)
    top_level = git_text(timelens_root, ["rev-parse", "--show-toplevel"]).strip()
    commit = git_text(timelens_root, ["rev-parse", "HEAD"]).strip()
    status = git_text(timelens_root, ["status", "--short"])
    diff = git_text(timelens_root, ["diff", "--", *OFFICIAL_FILES])
    has_own_git = (timelens_root / ".git").exists()
    status_text = status if status else "CLEAN\n"
    text = (
        f"timelens_path: {timelens_root}\n"
        f"git_top_level: {top_level}\n"
        f"has_own_git_dir: {has_own_git}\n"
        f"git_commit: {commit}\n"
        "git_status_short:\n"
        f"{status_text}"
    )
    (provenance_dir / "timelens_git_commit.txt").write_text(text, encoding="utf-8")
    (provenance_dir / "timelens_git_diff.txt").write_text(diff, encoding="utf-8")
    return {
        "timelens_path": str(timelens_root),
        "git_top_level": top_level,
        "has_own_git_dir": has_own_git,
        "git_commit": commit,
        "git_status_short": status,
        "official_file_diff_empty": diff.strip() == "",
    }


def write_official_behavior(timelens_root: Path, provenance_dir: Path, git_info: dict[str, Any]) -> None:
    hashes = {
        rel: sha256_file(timelens_root / rel)
        for rel in OFFICIAL_FILES
        if (timelens_root / rel).exists()
    }
    body = f"""# Official TimeLens Behavior

This document records the local TimeLens code inspected before implementing the MedVidU adapter.

## Upstream Location

- TimeLens path: `{timelens_root}`
- Git top level observed from that path: `{git_info.get("git_top_level")}`
- Has own `.git` directory: `{git_info.get("has_own_git_dir")}`
- Commit recorded by `git rev-parse HEAD`: `{git_info.get("git_commit")}`
- Official-file diff at inspection time: `{"empty" if git_info.get("official_file_diff_empty") else "non-empty"}`

Because the local `third_party/TimeLens-main` directory does not expose a nested `.git` directory in this checkout, the commit above is the enclosing MedVidU repository commit observed by Git from that path.

## Files Read

{json.dumps(hashes, indent=2, ensure_ascii=False, sort_keys=True)}

## Confirmed Behavior In Local Code

- Model loading is in `evaluation/eval_dataloader.py`: `AutoModelForImageTextToText.from_pretrained(..., dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map=args.device).eval()`.
- Processor loading is in `evaluation/eval_dataloader.py`: `AutoProcessor.from_pretrained(..., padding_side="left", do_resize=False, trust_remote_code=True)`.
- TimeLens-8B/Qwen3-VL path is in `evaluation/utils.py`: model path containing `qwen3` or `timelens-8b` uses downsample rate 32, `image_patch_size=16`, `return_video_kwargs=True`, and `return_video_metadata=True`, then passes `video_metadata` to the processor.
- Official TimeLens-8B grounding prompt in `evaluation/utils.py` is exactly:

```text
{GROUNDER_PROMPT}
```

- TimeLens-7B has a separate textual timestamp prompt. The MedVidU adapter does not use that path for TimeLens-8B.
- Official decoding in `evaluation/eval_dataloader.py`: `do_sample=False`, `temperature=None`, `top_p=None`, `top_k=None`, `max_new_tokens=512`.
- `scripts/eval_timelens_bench.sh` defaults are `min_tokens={MIN_TOKENS}`, `total_tokens={TOTAL_TOKENS}`, and raw-video `FPS={NATIVE_TIMELENS_RAW_VIDEO_FPS}`.
- For TimeLens-8B/Qwen3-VL, MedVidU adapter pixel budget is `min_tokens * {QWEN3_DOWNSAMPLE_RATE}^2` and `total_tokens * {QWEN3_DOWNSAMPLE_RATE}^2`; `image_patch_size={QWEN3_IMAGE_PATCH_SIZE}`.
- Official timestamp parsing is `timelens.utils.extract_time`. The MedVidU adapter reuses it and does not convert parse failures into fake raw predictions.

## MedVidU Adaptation Boundary

This baseline is TimeLens-8B adapted to MedVidU TAL using the benchmark-provided visual frame sequence and the official frozen TimeLens grounding model. It is not an official TimeLens result on MedVidU.
"""
    (provenance_dir / "official_timelens_behavior.md").write_text(body, encoding="utf-8")


def model_fingerprint(model_path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"model_path": str(model_path), "exists": model_path.exists()}
    for name in ("config.json", "generation_config.json", "preprocessor_config.json", "processor_config.json"):
        path = model_path / name
        out[f"{name}_sha256"] = sha256_file(path) if path.exists() else None
    try:
        from transformers import AutoConfig, AutoProcessor

        cfg = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(str(model_path), padding_side="left", do_resize=False, trust_remote_code=True)
        out["model_class"] = cfg.architectures[0] if getattr(cfg, "architectures", None) else type(cfg).__name__
        out["processor_class"] = type(processor).__name__
    except Exception as exc:
        out["model_introspection_error"] = repr(exc)
    try:
        import torch

        out["torch_version"] = torch.__version__
        out["dtype"] = "torch.bfloat16"
        out["cuda_available"] = torch.cuda.is_available()
        out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as exc:
        out["torch_error"] = repr(exc)
    for module_name in ("transformers", "qwen_vl_utils", "flash_attn"):
        try:
            module = __import__(module_name)
            out[f"{module_name}_version"] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            out[f"{module_name}_version"] = f"UNAVAILABLE ({exc!r})"
    return out


def write_environment_and_model(model_path: Path | None, provenance_dir: Path) -> dict[str, Any] | None:
    provenance_dir.mkdir(parents=True, exist_ok=True)
    (provenance_dir / "environment_snapshot.txt").write_text(environment_snapshot(), encoding="utf-8")
    if model_path is None:
        fp = {
            "model_path": None,
            "exists": False,
            "status": "NOT_RECORDED_NO_MODEL_PATH_PROVIDED",
        }
        write_json(provenance_dir / "timelens8b_model_fingerprint.json", fp)
        return fp
    fp = model_fingerprint(model_path)
    write_json(provenance_dir / "timelens8b_model_fingerprint.json", fp)
    return fp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write TimeLens-MedVidU provenance files.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--timelens-root", type=Path, default=TIMELENS_ROOT)
    parser.add_argument("--model-path", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    provenance_dir = args.output_root / "provenance"
    git_info = write_timelens_git_provenance(args.timelens_root, provenance_dir)
    write_official_behavior(args.timelens_root, provenance_dir, git_info)
    write_environment_and_model(args.model_path, provenance_dir)
    print(json.dumps(git_info, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
