from __future__ import annotations

import base64
import time
from io import BytesIO
from pathlib import Path
from typing import Any

from ..config import DecodeConfig, PixelConfig
from ..io_utils import assert_no_gt_leak, read_jsonl, sha256_json
from ..tasks import get_adapter


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_positions_hash(selection_row: dict[str, Any]) -> str:
    return sha256_json([int(x) for x in selection_row.get("selected_original_positions_chronological", [])])


def target_cache_key(
    manifest_row: dict[str, Any],
    selection_row: dict[str, Any],
    model_fingerprint: dict[str, Any],
    selector_manifest_sha256: str,
    prompt: str,
    decode_config: dict[str, Any],
    pixel_config: dict[str, Any],
) -> str:
    return sha256_json(
        {
            "sample_id": manifest_row["sample_id"],
            "target_model_fingerprint": model_fingerprint,
            "selector_manifest_sha256": selector_manifest_sha256,
            "selected_positions_hash": selected_positions_hash(selection_row),
            "task_prompt_hash": sha256_json(prompt),
            "decoding_config": decode_config,
            "pixel_config": pixel_config,
        }
    )


def load_inputs(manifest_path: Path, selector_path: Path, expected_selector_sha256: str | None = None) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], str]:
    actual_sha = sha256_file(selector_path)
    if expected_selector_sha256 is not None and expected_selector_sha256 != actual_sha:
        raise RuntimeError(
            "SELECTOR_MANIFEST_SHA_MISMATCH: "
            f"expected={expected_selector_sha256} actual={actual_sha} path={selector_path}"
        )
    manifest_by_id = {str(row["sample_id"]): row for row in read_jsonl(manifest_path)}
    selections = read_jsonl(selector_path)
    return manifest_by_id, selections, actual_sha


def build_frame_prompt(manifest_row: dict[str, Any], selection_row: dict[str, Any]) -> tuple[str, list[float]]:
    adapter = get_adapter(str(manifest_row["qa_type"]))
    timestamps = [float(x) for x in selection_row.get("selected_local_times_chronological") or []]
    frame_lines = [f"Frame {i + 1}: {timestamp:.3f} seconds" for i, timestamp in enumerate(timestamps)]
    return adapter.build_prompt(manifest_row, frame_lines=frame_lines), timestamps


def open_selected_frames(selection_row: dict[str, Any]) -> tuple[list[Any], list[list[int]]]:
    from PIL import Image

    frames = []
    sizes = []
    for path in selection_row.get("selected_frame_paths_chronological") or []:
        image = Image.open(path).convert("RGB")
        frames.append(image)
        sizes.append([int(image.size[0]), int(image.size[1])])
    return frames, sizes


def pil_to_data_url(image: Any) -> str:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG", optimize=False)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_config_for_task(adapter: Any, decode: DecodeConfig, max_new_tokens: int | None = None) -> dict[str, Any]:
    return {
        "do_sample": decode.do_sample,
        "temperature": decode.temperature,
        "top_p": decode.top_p,
        "top_k": decode.top_k,
        "repetition_penalty": decode.repetition_penalty,
        "num_beams": decode.num_beams,
        "thinking": decode.thinking,
        "max_new_tokens": adapter.max_new_tokens() if max_new_tokens is None else int(max_new_tokens),
    }


def pixel_config_dict(pixel: PixelConfig) -> dict[str, Any]:
    return dict(pixel.__dict__)


def prediction_row(
    manifest_row: dict[str, Any],
    selection_row: dict[str, Any],
    answer: str,
    model_name: str,
    backend: str,
    selector_manifest_sha256: str,
    model_fingerprint: dict[str, Any],
    prompt: str,
    decode_config: dict[str, Any],
    pixel_config: dict[str, Any],
    image_sizes: list[list[int]],
    elapsed_sec: float,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    adapter = get_adapter(str(manifest_row["qa_type"]))
    parser_status, parsed_prediction = adapter.parse_prediction(answer)
    positions = [int(x) for x in selection_row.get("selected_original_positions_chronological", [])]
    timestamps = [float(x) for x in selection_row.get("selected_local_times_chronological", [])]
    row = {
        "sample_id": manifest_row["sample_id"],
        "original_index": manifest_row["original_index"],
        "id": manifest_row.get("id"),
        "qa_type": manifest_row["qa_type"],
        "dataset_name": manifest_row.get("dataset_name"),
        "data_source": manifest_row.get("data_source"),
        "selector_applicable": selection_row.get("selector_applicable", False),
        "selector_reason": selection_row.get("selector_reason"),
        "selector_manifest_sha256": selector_manifest_sha256,
        "selected_positions_hash": sha256_json(positions),
        "selected_original_positions_chronological": positions,
        "selected_source_frame_indices_chronological": selection_row.get("selected_source_frame_indices_chronological", []),
        "selected_local_times_chronological": timestamps,
        "prediction": answer,
        "parser_status": parser_status,
        "parsed_prediction": parsed_prediction,
        "cache_key": target_cache_key(
            manifest_row=manifest_row,
            selection_row=selection_row,
            model_fingerprint=model_fingerprint,
            selector_manifest_sha256=selector_manifest_sha256,
            prompt=prompt,
            decode_config=decode_config,
            pixel_config=pixel_config,
        ),
        "inference_info": {
            "model": model_name,
            "backend": backend,
            "selector_version": selection_row.get("selector_version"),
            "reward_lmm": selection_row.get("reward_lmm"),
            "decode_config": decode_config,
            "pixel_config": pixel_config,
            "num_selected_frames": len(positions),
            "original_image_sizes": image_sizes,
            "elapsed_sec": round(elapsed_sec, 3),
            "model_fingerprint": model_fingerprint,
            **(extra_info or {}),
        },
    }
    assert_no_gt_leak({k: v for k, v in row.items() if k != "inference_info"})
    return row


class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        return False
