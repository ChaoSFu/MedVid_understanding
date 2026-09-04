from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ..config import CAFS_INFER_BATCH_SIZE, CAFS_MODEL, CAFS_SAMPLE_PER_SEC


def load_official_cafs_module(dig_repo_dir: Path):
    repo = Path(dig_repo_dir).resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from pipeline import cafs as official_cafs  # type: ignore

    return official_cafs


class OfficialCAFSAdapter:
    """MedVidU frame-list adapter for DIG CAFS using official DINOv2 primitives."""

    def __init__(
        self,
        dig_repo_dir: Path,
        model_name: str = CAFS_MODEL,
        samples_per_sec: int = CAFS_SAMPLE_PER_SEC,
        infer_batch_size: int = CAFS_INFER_BATCH_SIZE,
        seed: int = 42,
        device: str = "cuda",
    ) -> None:
        self.dig_repo_dir = Path(dig_repo_dir)
        self.model_name = model_name
        self.samples_per_sec = samples_per_sec
        self.infer_batch_size = infer_batch_size
        self.seed = seed
        self.device = device
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        import torch
        import torch.nn.functional as F
        from PIL import Image
        from scipy.signal import find_peaks
        from transformers import AutoImageProcessor, Dinov2Model

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.torch = torch
        self.F = F
        self.Image = Image
        self.find_peaks = find_peaks
        self.official_cafs = load_official_cafs_module(self.dig_repo_dir)
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = Dinov2Model.from_pretrained(self.model_name).to(self.device).eval()
        self._loaded = True

    def _target_positions(self, row: dict[str, Any]) -> list[int]:
        observations = list(row.get("frame_observations") or [])
        n = int(row["n_medvidu_frames"])
        if n <= 0:
            return []
        duration = float(row.get("clip_duration") or 0.0)
        if duration <= 0:
            return list(range(n))
        target_count = max(1, int(duration) * int(self.samples_per_sec))
        target_times = np.linspace(0.0, duration, target_count)
        obs_times = np.array([float(obs["local_time"]) for obs in observations], dtype=float)
        positions = [int(np.abs(obs_times - t).argmin()) for t in target_times]
        out = []
        for pos in positions:
            if pos not in out:
                out.append(pos)
        return out

    def get_r_frames(self, row: dict[str, Any]) -> dict[str, Any]:
        self.load()
        n = int(row["n_medvidu_frames"])
        if n <= 1:
            return {"r_frame_original_positions": list(range(n)), "boundaries_original_positions": [0, max(0, n - 1)]}

        target_positions = self._target_positions(row)
        if not target_positions:
            target_positions = list(range(n))

        duration = float(row.get("clip_duration") or max(1, n - 1))
        num_segments = max(1, math.ceil(duration / 60.0))
        segment_edges = np.linspace(0, n - 1, num_segments + 1, dtype=int)

        frame_paths = list(row["video"])
        cuts: list[int] = [0]
        result_frames: list[int] = []

        for seg_i, (start, end) in enumerate(zip(segment_edges[:-1], segment_edges[1:])):
            if seg_i == len(segment_edges) - 2:
                chunk_positions = [p for p in target_positions if start <= p <= end]
            else:
                chunk_positions = [p for p in target_positions if start <= p < end]
            if not chunk_positions:
                continue

            chunk_features = []
            for j in range(0, len(chunk_positions), self.infer_batch_size):
                paths = [frame_paths[pos] for pos in chunk_positions[j : j + self.infer_batch_size]]
                images = [self.Image.open(path).convert("RGB") for path in paths]
                features = self.official_cafs.extract_batch_features(self.model, self.processor, images, self.device)
                chunk_features.append(features.mean(dim=1))
            if not chunk_features:
                continue
            features = self.torch.cat(chunk_features, dim=0)
            if features.shape[0] > 1:
                diffs = 1 - self.F.cosine_similarity(features[:-1], features[1:], dim=1)
                peaks, _ = self.find_peaks(diffs.detach().cpu().numpy(), prominence=0.1)
                current_peaks = [chunk_positions[int(p)] for p in peaks]
                temp_cuts = [cuts[-1]] + current_peaks
                cuts.extend(current_peaks)
            else:
                temp_cuts = [cuts[-1]]

            if seg_i == len(segment_edges) - 2:
                cuts.append(n - 1)
                temp_cuts.append(n - 1)

            mids = [(a + b) // 2 for a, b in zip(temp_cuts[:-1], temp_cuts[1:])]
            if mids:
                target_np = np.array(target_positions)
                nearest = [int(np.abs(target_np - m).argmin()) for m in mids]
                result_frames.extend([int(target_np[i]) for i in nearest])

        if not cuts or cuts[-1] != n - 1:
            cuts.append(n - 1)
        return {
            "r_frame_original_positions": [int(x) for x in result_frames],
            "boundaries_original_positions": [int(x) for x in cuts],
        }
