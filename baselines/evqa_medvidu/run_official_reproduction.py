from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
from typing import Any

from .config import DEFAULT_OUTPUT_ROOT, EVQA_ROOT, RunConfig
from .io_utils import write_json


def run_official_reproduction(
    evqa_root: Path,
    model_path: Path,
    data_file: Path,
    video_dir: Path,
    output_root: Path,
    samples: int,
    fps: float,
    max_frames: int,
) -> dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(f"model path does not exist: {model_path}")
    if not data_file.exists():
        raise FileNotFoundError(f"official ST-Evidence data file does not exist: {data_file}")
    if not video_dir.exists():
        raise FileNotFoundError(f"official ST-Evidence video directory does not exist: {video_dir}")
    script = evqa_root / "benchmark" / "st_evidence_gen" / "ours_st_evidence.py"
    if not script.exists():
        raise FileNotFoundError(f"official inference script does not exist: {script}")
    cfg = RunConfig(output_root=output_root, evqa_root=evqa_root)
    cfg.make_dirs()
    official_out = cfg.official_reproduction_dir / "official_script_outputs"
    cmd = [
        "python",
        str(script),
        "--model",
        str(model_path),
        "--data-file",
        str(data_file),
        "--video-dir",
        str(video_dir),
        "--fps",
        str(fps),
        "--max-frames",
        str(max_frames),
        "--output-dir",
        str(official_out),
        "--start-idx",
        "0",
        "--end-idx",
        str(samples),
        "--save-every",
        "1",
    ]
    result = subprocess.run(cmd, cwd=script.parent, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    report = {
        "samples": samples,
        "success": result.returncode == 0,
        "command": cmd,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-12000:],
        "output_dir": str(official_out),
        "temporal_output": "requires inspection of official JSON parsed_response.segments",
        "mask_output": "requires inspection of official mask output directory",
    }
    write_json(cfg.official_reproduction_dir / "report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run official E-VQA ST-Evidence reproduction on 2-5 samples before MedVidU adaptation.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--evqa-root", type=Path, default=EVQA_ROOT)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=128)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 2 <= args.samples <= 5:
        raise ValueError("--samples must be between 2 and 5 for Phase A")
    report = run_official_reproduction(args.evqa_root, args.model_path, args.data_file, args.video_dir, args.output_root, args.samples, args.fps, args.max_frames)
    print(report)
    return 0 if report["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

