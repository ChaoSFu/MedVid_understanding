from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any

from .cache import build_cache_key
from .config import EVQA_TRAIN_ROOT, OFFICIAL_MAX_PIXELS_PER_FRAME, OFFICIAL_MIN_PIXELS, GenerationConfig
from .io_utils import append_jsonl, completed_cache, sha256_json
from .tasks import get_adapter
from .tasks.base import SchemaMismatch


class EVQABackend:
    """Shared frozen ST-Evidence-7B backend for all MedVidU task adapters."""

    def __init__(
        self,
        model_path: Path,
        output_root: Path,
        model_fingerprint: dict[str, Any],
        device: str = "auto",
        dtype: str = "bfloat16",
        generation: GenerationConfig | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.output_root = Path(output_root)
        self.model_fingerprint = model_fingerprint
        self.device_arg = device
        self.dtype = dtype
        self.generation = generation or GenerationConfig()
        self.model = None
        self.processor = None
        self.sam2_transform = None
        self.device = None

    def load(self) -> None:
        if self.model is not None:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(f"model path does not exist: {self.model_path}")
        if str(EVQA_TRAIN_ROOT) not in sys.path:
            sys.path.insert(0, str(EVQA_TRAIN_ROOT))
        from unipixel.model.builder import build_model
        from unipixel.utils.transforms import get_sam2_transform

        self.model, self.processor = build_model(
            str(self.model_path),
            device=self.device_arg,
            dtype=self.dtype,
            is_trainable=False,
            merge_adapter=False,
        )
        self.device = next(self.model.parameters()).device
        self.sam2_transform = get_sam2_transform(self.model.config.sam2_image_size)

    def build_prompt(self, row: dict[str, Any]) -> str:
        question = _strip_video_token(row["human_question"])
        if row["task"] == "stg":
            return f"{question} Answer the question and provide evidence in the form of temporal ([[start1, end1], [start2, end2], ...]) and spatial evidence (masks)."
        if row["task"] == "rc":
            return f"Here is a video with a provided region denoted by [0] <|ref|>. {question}"
        if row["task"] == "cvs":
            return f"{question} Provide the answer. If temporal or spatial evidence is relevant, include it in the native E-VQA evidence format."
        raise KeyError(row["task"])

    def run_row(self, row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
        adapter = get_adapter(row["task"])
        model_input = adapter.build_model_input(row)
        prompt = self.build_prompt(row)
        cache_key = build_cache_key(row, self.model_fingerprint, prompt, model_input)
        raw_path = self.output_root / "predictions" / "raw" / f"{row['task']}.jsonl"
        medvidu_path = self.output_root / "predictions" / "medvidu" / f"{row['task']}.jsonl"
        err_path = self.output_root / "predictions" / "errors" / f"{row['task']}.jsonl"
        if cache_key in completed_cache(raw_path):
            return {"status": "cached", "cache_key": cache_key, "sample_id": row["sample_id"]}, None, None

        try:
            raw = self._generate(row, prompt, model_input)
            raw["cache_key"] = cache_key
            parsed = adapter.parse_evqa_output(raw)
            medvidu = adapter.to_medvidu_prediction(row, parsed)
            medvidu["cache_key"] = cache_key
            adapter.validate_prediction(medvidu)
            append_jsonl(raw_path, _json_safe_raw(raw))
            append_jsonl(medvidu_path, medvidu)
            return {"status": "new", "cache_key": cache_key, "sample_id": row["sample_id"]}, raw, medvidu
        except SchemaMismatch as exc:
            error = {"sample_id": row["sample_id"], "cache_key": cache_key, "task": row["task"], "error_type": "SCHEMA_MISMATCH", "error": str(exc)}
            append_jsonl(err_path, error)
            raise
        except Exception as exc:
            error = {"sample_id": row["sample_id"], "cache_key": cache_key, "task": row["task"], "error_type": exc.__class__.__name__, "error": repr(exc)}
            append_jsonl(err_path, error)
            raise

    def _generate(self, row: dict[str, Any], prompt: str, model_input: dict[str, Any]) -> dict[str, Any]:
        self.load()
        assert self.model is not None and self.processor is not None and self.sam2_transform is not None and self.device is not None
        import torch
        from PIL import Image
        from unipixel.dataset.utils import process_vision_info

        frame_paths = row["ordered_frame_paths"]
        selected_paths = row["model_sampling"]["selected_frame_paths"]
        frames = torch.stack([_pil_to_tensor(Image.open(path).convert("RGB")) for path in frame_paths], dim=0)
        images = [Image.open(path).convert("RGB") for path in selected_paths]
        content: list[dict[str, Any]] = [
            {
                "type": "video",
                "video": images,
                "min_pixels": OFFICIAL_MIN_PIXELS,
                "max_pixels": OFFICIAL_MAX_PIXELS_PER_FRAME * len(images),
            },
            {"type": "text", "text": prompt},
        ]
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        images_proc, videos_proc, kwargs = process_vision_info(messages, return_video_kwargs=True)
        data = self.processor(text=[text], images=images_proc, videos=videos_proc, return_tensors="pt", **kwargs)
        data["frames"] = [self.sam2_transform(frames).to(self.model.sam2.dtype)]
        data["frame_size"] = [frames.shape[1:3]]
        if row["task"] == "rc":
            data.update(_build_ref_box_kwargs(row, self.model.config.sam2_image_size, len(selected_paths)))
        batch = data.to(self.device)
        for key in ("point_coords", "point_labels", "point_frames", "frames"):
            if key in batch:
                batch[key] = _move_nested(batch[key], self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **batch,
                do_sample=self.generation.do_sample,
                temperature=self.generation.temperature,
                top_k=self.generation.top_k,
                top_p=self.generation.top_p,
                repetition_penalty=self.generation.repetition_penalty,
                max_new_tokens=self.generation.max_new_tokens,
            )
        output_ids = output_ids[0, data.input_ids.size(1) :]
        if len(output_ids) and output_ids[-1] == self.processor.tokenizer.eos_token_id:
            output_ids = output_ids[:-1]
        response = self.processor.decode(output_ids, clean_up_tokenization_spaces=False)
        raw_masklets = []
        if hasattr(self.model, "seg"):
            for i, item in enumerate(self.model.seg):
                mask = item[0] if isinstance(item, (list, tuple)) else item
                if getattr(mask, "ndim", None) == 4 and mask.shape[0] == 1:
                    mask = mask[0]
                raw_masklets.append({"masklet_id": f"masklet_{i:03d}", "mask": mask.cpu().numpy(), "metadata": {"source": "model.seg"}})
        return {
            "sample_id": row["sample_id"],
            "qa_id": row.get("qa_id"),
            "task": row["task"],
            "dataset": row.get("dataset_name") or row.get("data_source"),
            "question": row["human_question"],
            "prompt": prompt,
            "selected_frame_identities": [row["logical_frame_identities"][i] for i in row["model_sampling"]["selected_logical_indices"]],
            "selected_timestamps": row["model_sampling"]["selected_timestamps"],
            "raw_textual_response": response,
            "semantic_answer": response,
            "raw_temporal_segments": [],
            "raw_referring_expressions": [],
            "raw_masklets": raw_masklets,
            "inference_status": "ok",
            "error": None,
            "manifest_row": row,
            "generation_config": asdict(self.generation),
        }


def _strip_video_token(text: str) -> str:
    return text.replace("<video>\n", "").replace("<video>", "").strip()


def _pil_to_tensor(image: Any) -> Any:
    import numpy as np
    import torch

    return torch.from_numpy(np.asarray(image))


def _move_nested(value: Any, device: Any) -> Any:
    if hasattr(value, "to"):
        return value.to(device)
    if isinstance(value, list):
        return [_move_nested(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_nested(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _move_nested(item, device) for key, item in value.items()}
    return value


def _build_ref_box_kwargs(row: dict[str, Any], sam2_image_size: int, n_selected_frames: int) -> dict[str, Any]:
    import torch

    region = row["provided_region"]
    bbox = region["start_frame_bbox"]
    frame_path = region.get("start_frame")
    selected = row["model_sampling"]["selected_frame_paths"]
    if frame_path in selected:
        prompt_frame = selected.index(frame_path)
    else:
        all_paths = row["ordered_frame_paths"]
        logical = all_paths.index(frame_path) if frame_path in all_paths else 0
        prompt_frame = min(range(len(row["model_sampling"]["selected_logical_indices"])), key=lambda i: abs(row["model_sampling"]["selected_logical_indices"][i] - logical))
    width, height = _image_size(row["ordered_frame_paths"][0])
    scale_x = float(sam2_image_size) / float(width)
    scale_y = float(sam2_image_size) / float(height)
    x1, y1, x2, y2 = bbox
    coords = torch.tensor([[[x1 * scale_x, y1 * scale_y], [x2 * scale_x, y2 * scale_y]]], dtype=torch.float32)
    labels = torch.tensor([[2, 3]], dtype=torch.int)
    frames = [torch.LongTensor([min(prompt_frame, max(0, n_selected_frames - 1))])]
    return {"point_coords": [coords], "point_labels": [labels], "point_frames": frames}


def _image_size(path: str) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as im:
        return im.size


def _json_safe_raw(raw: dict[str, Any]) -> dict[str, Any]:
    out = dict(raw)
    out.pop("manifest_row", None)
    out["manifest_row_hash"] = sha256_json(raw.get("manifest_row", {}))
    out["raw_masklets"] = [
        {
            "masklet_id": m.get("masklet_id"),
            "metadata": m.get("metadata", {}),
            "mask_shape": list(getattr(m.get("mask"), "shape", [])),
            "mask_dtype": str(getattr(m.get("mask"), "dtype", "")),
        }
        for m in raw.get("raw_masklets", [])
    ]
    return out
