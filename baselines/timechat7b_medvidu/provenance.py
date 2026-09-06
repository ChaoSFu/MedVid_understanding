from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import (
    DEFAULT_OUTPUT_ROOT,
    FRAME_LOADER_VERSION,
    IMAGE_SIZE,
    MAX_FRAMES,
    OFFICIAL_SYSTEM_PROMPT,
    OFFICIAL_VTUNE_GROUNDING_PROMPT,
    PROMPT_VERSION,
    QUERY_ADAPTER_VERSION,
    SCIENTIFIC_NAME,
    TIMESTAMP_ADAPTER_VERSION,
    TIMESTAMP_ROUNDING_SECONDS,
    RunConfig,
)
from .io_utils import environment_snapshot, git_text, sha256_file, sha256_text, write_json


OFFICIAL_FILES = [
    "README.md",
    "run.py",
    "utils/cons_utils.py",
    "utils/prompts.py",
    "task/grounding.py",
    "eval/eval.py",
    "timechat/utils.py",
    "timechat/eval_configs/timechat.yaml",
    "timechat/models/timechat.py",
    "timechat/processors/video_processor.py",
]


def require_official_files(timechat_repo: Path) -> list[str]:
    missing = [rel for rel in OFFICIAL_FILES if not (timechat_repo / rel).exists()]
    if missing:
        raise FileNotFoundError("STOP: official TimeChat/Consistency files missing: " + ", ".join(missing))
    return missing


def write_upstream_git_provenance(timechat_repo: Path, provenance_dir: Path, suffix: str = "before") -> dict[str, Any]:
    provenance_dir.mkdir(parents=True, exist_ok=True)
    require_official_files(timechat_repo)
    top_level = git_text(timechat_repo, ["rev-parse", "--show-toplevel"]).strip()
    commit = git_text(timechat_repo, ["rev-parse", "HEAD"]).strip()
    status = git_text(timechat_repo, ["status", "--short"])
    remote = git_text(timechat_repo, ["remote", "-v"])
    diff = git_text(timechat_repo, ["diff", "--", *OFFICIAL_FILES])
    (provenance_dir / "upstream_git_commit.txt").write_text(
        (
            f"repository_path: {timechat_repo}\n"
            f"git_top_level: {top_level}\n"
            f"commit_sha: {commit}\n"
            "git_status_short:\n"
            f"{status if status else 'CLEAN\\n'}"
            "git_remote_v:\n"
            f"{remote}"
        ),
        encoding="utf-8",
    )
    (provenance_dir / f"upstream_git_diff_{suffix}.txt").write_text(diff, encoding="utf-8")
    return {
        "repository_path": str(timechat_repo),
        "git_top_level": top_level,
        "commit_sha": commit,
        "git_status_short": status,
        "git_remote_v": remote,
        "official_file_diff_empty": diff.strip() == "",
    }


def official_file_hashes(timechat_repo: Path) -> dict[str, str]:
    require_official_files(timechat_repo)
    return {rel: sha256_file(timechat_repo / rel) for rel in OFFICIAL_FILES}


def write_official_behavior(timechat_repo: Path, provenance_dir: Path, git_info: dict[str, Any]) -> None:
    hashes = official_file_hashes(timechat_repo)
    body = f"""# Official TimeChat Behavior

This file records the official local implementation inspected for `{SCIENTIFIC_NAME}`.

## Upstream

- Repository path: `{timechat_repo}`
- Commit SHA: `{git_info.get("commit_sha")}`
- Dirty official files before adapter run: `{not git_info.get("official_file_diff_empty")}`

## Files Read

```json
{json.dumps(hashes, indent=2, ensure_ascii=False, sort_keys=True)}
```

## Confirmed Behavior Required By This Adapter

- Grounding execution follows `run.py -> TimeChat(args) -> run_grounding -> model.load_video_features(...) -> model.run(task="grounding", ...)`.
- When `--fine_tuned` is active, official `timechat/utils.py` sets the grounding prompt to:

```text
{OFFICIAL_VTUNE_GROUNDING_PROMPT}
```

- The adapter does not pretend that MedVidU is ActivityNet. It uses `dset_name=medvidu` only as an adapter label and overrides the checkpoint path explicitly.
- Official video behavior is preserved as a frame budget: `n_frames={MAX_FRAMES}`, image size `{IMAGE_SIZE}x{IMAGE_SIZE}`, all frames if `N <= {MAX_FRAMES}`, and official-style uniform sampling otherwise.
- Official timestamp text is preserved: `This frame is sampled at {{t}} second.` with `round(local_seconds, 1)`.
- The user-facing TimeChat message is preserved: `The video contains {{N}} frames sampled at ... seconds.`
- Official deterministic decoding is preserved: `num_beams=1`, `do_sample=False`, `temperature=0.05`, `max_new_tokens=300`, `max_length=2000`.
- Official system prompt is preserved:

```text
{OFFICIAL_SYSTEM_PROMPT}
```

- Official architecture/config invariants expected from `timechat/eval_configs/timechat.yaml`: `max_frame_pos=96`, `window_size=32`, `stride=32`, `qformer_text_input=True`, `lora=True`, `lora_inference_mode=True`.
- Official parser behavior is preserved by calling `TimeChat.extract_time`, then falling back to `TimeChat.extract_time2` only when the first parser returns `[0, 0]`.

## MedVidU Adapter Boundary

- Query adapter: `{QUERY_ADAPTER_VERSION}`.
- Timestamp adapter: `{TIMESTAMP_ADAPTER_VERSION}`.
- Frame loader: `{FRAME_LOADER_VERSION}`.
- Timestamp rounding: `{TIMESTAMP_ROUNDING_SECONDS}` seconds.
- Visual evidence is restricted to MedVidU-provided frame sequences. No source-video backtracking and no pseudo-MP4 construction are used.
- GT is not loaded during inference; evaluation is an offline phase that joins predictions to GT by `sample_id`/`qa_id`.
"""
    (provenance_dir / "official_timechat_behavior.md").write_text(body, encoding="utf-8")


def write_run_config(output_root: Path, vtune_ckpt: Path | None = None) -> dict[str, Any]:
    cfg = RunConfig(output_root=output_root)
    payload = cfg.to_jsonable()
    payload["checkpoint"] = str(vtune_ckpt) if vtune_ckpt else None
    payload["prompt_text"] = OFFICIAL_VTUNE_GROUNDING_PROMPT
    payload["prompt_sha256"] = sha256_text(OFFICIAL_VTUNE_GROUNDING_PROMPT)
    write_json(cfg.provenance_dir / "run_config.json", payload)
    return payload


def write_environment(output_root: Path) -> None:
    cfg = RunConfig(output_root=output_root)
    cfg.make_dirs()
    (cfg.provenance_dir / "environment_snapshot.txt").write_text(environment_snapshot(), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write provenance for TimeChat-7B ActivityNet VTune MedVidU TAL.")
    parser.add_argument("--timechat-repo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--vtune-ckpt", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = RunConfig(output_root=args.output_root)
    cfg.make_dirs()
    git_info = write_upstream_git_provenance(args.timechat_repo, cfg.provenance_dir)
    write_official_behavior(args.timechat_repo, cfg.provenance_dir, git_info)
    write_run_config(args.output_root, args.vtune_ckpt)
    write_environment(args.output_root)
    print(json.dumps(git_info, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
