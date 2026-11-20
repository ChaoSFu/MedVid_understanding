"""Explicitly synthetic 2-per-task smoke, including pixel IO and exact replay.

Creates its own labeled toy source and images; never substitutes them into
MedVidU. This command makes no real model calls.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from .contracts import QA_TYPES
from .data import audit_dataset, load_json, write_once
from .pipeline import generate, validate_bank


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--config", default="configs/claim_bank_mock_v1.yaml")
    a = p.parse_args()
    root = Path(a.output_dir).resolve()
    if root.exists():
        p.error("Synthetic smoke requires a new output directory")
    root.mkdir(parents=True)
    records = []
    for n, qa in enumerate(QA_TYPES):
        for instance in range(2):
            paths = []
            for frame in range(3):
                path = root / "synthetic_images" / f"{n}-{instance}-{frame}.png"
                path.parent.mkdir(exist_ok=True)
                Image.new("RGB", (48, 32), (30 + n, 40 + frame, 50 + instance)).save(path)
                paths.append(str(path))
            records.append({"id": "intentionally-duplicated-fixture-id", "qa_type": qa,
                            "data_source": "SYNTHETIC_FIXTURE", "dataset_name": "SYNTHETIC_FIXTURE",
                            "video": paths, "sampled_video_frames": [0, 1, 2], "metadata": {},
                            "conversations": [{"from": "human", "value": "Inspect the synthetic marker."},
                                              {"from": "gpt", "value": "FORBIDDEN_GT_SENTINEL"}],
                            "struc_info": "FORBIDDEN_GT_SENTINEL", "RC_info": "FORBIDDEN_GT_SENTINEL"})
    write_once(root / "synthetic_source.json", records)
    audit_dataset(str(root / "synthetic_source.json"), str(root / "data"), expected_count=22, prepare=True)
    shared = {"runtime_manifest": str(root / "data/runtime_manifest.jsonl"), "config_path": a.config,
              "cache_dir": str(root / "synthetic_cache"), "per_qa_type": 2}
    first = generate(**shared, stage=1, output_dir=str(root / "stage1"))
    second = generate(**shared, stage=2, stage1_dir=str(root / "stage1"), output_dir=str(root / "stage2"))
    replay = generate(**shared, stage=2, stage1_dir=str(root / "stage1"), output_dir=str(root / "replay"),
                      replay_only=True, replay_from=str(root / "stage2"))
    validation = validate_bank(root / "stage2")
    for name in ("claim_bank.jsonl", "claims.jsonl", "requirements.jsonl", "frame_sampling_manifest.jsonl"):
        assert (root / "stage2" / name).read_bytes() == (root / "replay" / name).read_bytes()
    report = {"status": "SYNTHETIC_SMOKE_PASS", "synthetic_test": True, "qa_types": len(QA_TYPES),
              "samples_per_qa_type": 2, "source_record_count": 22,
              "stage1": first, "stage2": second, "replay": replay, "validation": validation,
              "real_model_calls": 0, "certificate_writes": 0, "new_verified_count": 0}
    write_once(root / "synthetic_smoke_report.json", report)
    print(json.dumps({k: report[k] for k in ("status", "source_record_count", "real_model_calls", "certificate_writes", "new_verified_count")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
