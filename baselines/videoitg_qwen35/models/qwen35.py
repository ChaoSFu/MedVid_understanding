from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from ..config import QWEN35_CHECKPOINT, RunConfig
from ..io_utils import append_jsonl, assert_no_gt_leak, environment_snapshot, load_completed_keys, read_jsonl, sha256_json, write_json
from ..tasks import get_adapter


def qwen_cache_key(
    manifest_row: dict[str, Any],
    selection_row: dict[str, Any],
    model_fingerprint: dict[str, Any],
    prompt: str,
    decode_config: dict[str, Any],
    pixel_config: dict[str, Any],
) -> str:
    return sha256_json(
        {
            "sample_id": manifest_row["sample_id"],
            "qa_type": manifest_row["qa_type"],
            "selected_ordered_positions": selection_row.get("selected_original_positions_chronological", []),
            "timestamps": selection_row.get("selected_local_times_chronological", []),
            "qwen35_fingerprint": model_fingerprint,
            "prompt_hash": sha256_json(prompt),
            "decode_config": decode_config,
            "pixel_config": pixel_config,
        }
    )


class Qwen35Runner:
    def __init__(
        self,
        checkpoint: str = QWEN35_CHECKPOINT,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        use_flash_attention_2: bool = False,
        run_config: RunConfig | None = None,
    ) -> None:
        self.checkpoint = checkpoint
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.use_flash_attention_2 = use_flash_attention_2
        self.run_config = run_config or RunConfig(qwen_model=checkpoint)
        self._loaded = False
        self.model_fingerprint: dict[str, Any] = {}

    def load(self) -> None:
        if self._loaded:
            return
        import torch
        import transformers
        from transformers import AutoModelForImageTextToText, AutoProcessor

        model_kwargs: dict[str, Any] = {
            "device_map": self.device_map,
            "torch_dtype": self.torch_dtype,
        }
        if self.use_flash_attention_2:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        processor = AutoProcessor.from_pretrained(self.checkpoint)
        model = AutoModelForImageTextToText.from_pretrained(self.checkpoint, **model_kwargs).eval()
        self.processor = processor
        self.model = model
        self.torch = torch
        self.transformers = transformers
        self.model_fingerprint = build_model_fingerprint(
            checkpoint=self.checkpoint,
            processor=processor,
            model=model,
            transformers_version=transformers.__version__,
            torch_version=torch.__version__,
            dtype=str(next(model.parameters()).dtype),
        )
        self._loaded = True

    def generate(self, manifest_row: dict[str, Any], selection_row: dict[str, Any]) -> dict[str, Any]:
        self.load()
        adapter = get_adapter(str(manifest_row["qa_type"]))
        frame_paths = list(selection_row.get("selected_frame_paths_chronological") or [])
        timestamps = [float(x) for x in selection_row.get("selected_local_times_chronological") or []]
        if selection_row.get("selector_applicable", False) and len(frame_paths) == 0:
            raise ValueError("selector-applicable sample has no selected frames")
        if len(frame_paths) != len(timestamps):
            raise ValueError("selected frame paths and timestamps differ in length")

        from PIL import Image

        frames = [Image.open(path).convert("RGB") for path in frame_paths]
        frame_lines = [
            f"Frame {i + 1}: {timestamp:.3f} seconds"
            for i, timestamp in enumerate(timestamps)
        ]
        prompt = adapter.build_prompt(manifest_row, frame_lines=frame_lines)
        messages = build_image_sequence_messages(prompt, frames, timestamps, self.run_config.pixel.__dict__)
        inputs = self._build_inputs(messages)

        decode_config = {
            "do_sample": self.run_config.decode.do_sample,
            "temperature": self.run_config.decode.temperature,
            "top_p": self.run_config.decode.top_p,
            "num_beams": self.run_config.decode.num_beams,
            "thinking": self.run_config.decode.thinking,
            "max_new_tokens": adapter.max_new_tokens(),
        }
        pixel_config = self.run_config.pixel.__dict__
        cache_key = qwen_cache_key(
            manifest_row,
            selection_row,
            self.model_fingerprint,
            prompt,
            decode_config,
            pixel_config,
        )
        start = time.time()
        if self.torch.cuda.is_available():
            self.torch.cuda.reset_peak_memory_stats()
        with self.torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=adapter.max_new_tokens(),
                do_sample=False,
                num_beams=1,
                use_cache=True,
            )
        elapsed = time.time() - start
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        answers = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        answer = answers[0].strip() if answers else ""
        parser_status, parsed_prediction = adapter.parse_prediction(answer)
        memory = {}
        if self.torch.cuda.is_available():
            memory = {
                "peak_allocated_bytes": int(self.torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(self.torch.cuda.max_memory_reserved()),
            }
        row = {
            "sample_id": manifest_row["sample_id"],
            "original_index": manifest_row["original_index"],
            "id": manifest_row.get("id"),
            "qa_type": manifest_row["qa_type"],
            "dataset_name": manifest_row.get("dataset_name"),
            "data_source": manifest_row.get("data_source"),
            "selector_applicable": selection_row.get("selector_applicable", False),
            "selector_reason": selection_row.get("selector_reason"),
            "selected_original_positions_chronological": selection_row.get("selected_original_positions_chronological", []),
            "selected_source_frame_indices_chronological": selection_row.get("selected_source_frame_indices_chronological", []),
            "selected_local_times_chronological": timestamps,
            "prediction": answer,
            "parser_status": parser_status,
            "parsed_prediction": parsed_prediction,
            "cache_key": cache_key,
            "inference_info": {
                "model": self.checkpoint,
                "backend": "AutoModelForImageTextToText",
                "processor_class": self.model_fingerprint.get("processor_class"),
                "model_class": self.model_fingerprint.get("model_class"),
                "config_model_type": self.model_fingerprint.get("config_model_type"),
                "decode_config": decode_config,
                "pixel_config": pixel_config,
                "num_selected_frames": len(frames),
                "original_image_sizes": [list(image.size) for image in frames],
                "input_length": int(inputs.input_ids.shape[-1]),
                "elapsed_sec": round(elapsed, 3),
                "gpu_memory": memory,
            },
        }
        assert_no_gt_leak(row)
        return row

    def _build_inputs(self, messages: list[dict[str, Any]]):
        from qwen_vl_utils import process_vision_info

        image_inputs, video_inputs = process_vision_info(messages)
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.run_config.decode.thinking,
        )
        kwargs: dict[str, Any] = {}
        if self.run_config.pixel.total_pixels is not None:
            kwargs["total_pixels"] = self.run_config.pixel.total_pixels
        if self.run_config.pixel.min_pixels is not None:
            kwargs["min_pixels"] = self.run_config.pixel.min_pixels
        if self.run_config.pixel.max_pixels is not None:
            kwargs["max_pixels"] = self.run_config.pixel.max_pixels
        inputs = self.processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            **kwargs,
        )
        try:
            return inputs.to(self.model.device)
        except Exception:
            return inputs.to("cuda" if self.torch.cuda.is_available() else "cpu")


def build_image_sequence_messages(prompt: str, frames: list[Any], timestamps: list[float], pixel_config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for i, (image, timestamp) in enumerate(zip(frames, timestamps), start=1):
        content.append({"type": "text", "text": f"Frame {i}: {timestamp:.3f} seconds"})
        item: dict[str, Any] = {"type": "image", "image": image}
        if pixel_config:
            for key in ("min_pixels", "max_pixels"):
                if pixel_config.get(key) is not None:
                    item[key] = pixel_config[key]
        content.append(item)
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def build_model_fingerprint(
    checkpoint: str,
    processor: Any,
    model: Any,
    transformers_version: str,
    torch_version: str,
    dtype: str,
) -> dict[str, Any]:
    config = getattr(model, "config", None)
    vision_config = getattr(config, "vision_config", None)
    if hasattr(vision_config, "to_dict"):
        vision_config = vision_config.to_dict()
    return {
        "checkpoint": checkpoint,
        "processor_class": processor.__class__.__name__,
        "model_class": model.__class__.__name__,
        "config_model_type": getattr(config, "model_type", None),
        "architectures": getattr(config, "architectures", None),
        "transformers_version": transformers_version,
        "torch_version": torch_version,
        "dtype": dtype,
        "vision_config": vision_config,
        "supported_multimodal_input": {
            "processor_has_apply_chat_template": hasattr(processor, "apply_chat_template"),
            "processor_call_supports_images_videos": True,
            "runner_schema": "timestamped_image_sequence",
        },
    }


def preflight(checkpoint: str, output_path: Path, device_map: str = "auto", torch_dtype: str = "auto") -> dict[str, Any]:
    runner = Qwen35Runner(checkpoint=checkpoint, device_map=device_map, torch_dtype=torch_dtype)
    runner.load()
    write_json(output_path, runner.model_fingerprint)
    return runner.model_fingerprint


def run_inference(
    manifest_path: Path,
    selector_path: Path,
    output_path: Path,
    errors_path: Path,
    checkpoint: str,
    device_map: str,
    torch_dtype: str,
    include_rc_direct: bool = False,
) -> dict[str, Any]:
    manifest_by_id = {row["sample_id"]: row for row in read_jsonl(manifest_path)}
    selections = read_jsonl(selector_path)
    cfg = RunConfig(qwen_model=checkpoint, output_root=output_path.parents[1])
    runner = Qwen35Runner(checkpoint=checkpoint, device_map=device_map, torch_dtype=torch_dtype, run_config=cfg)
    runner.load()
    write_json(cfg.provenance_dir / "qwen35_model_fingerprint.json", runner.model_fingerprint)
    completed = load_completed_keys(output_path)
    counts = {"completed": 0, "skipped_completed": 0, "errors": 0, "parser_failures": 0, "empty": 0}
    for selection in selections:
        sample_id = selection["sample_id"]
        manifest_row = manifest_by_id[sample_id]
        if not selection.get("selector_applicable", False) and not include_rc_direct:
            continue
        adapter = get_adapter(str(manifest_row["qa_type"]))
        prompt = adapter.build_prompt(
            manifest_row,
            [
                f"Frame {i + 1}: {float(t):.3f} seconds"
                for i, t in enumerate(selection.get("selected_local_times_chronological", []))
            ],
        )
        cache_key = qwen_cache_key(
            manifest_row,
            selection,
            runner.model_fingerprint,
            prompt,
            {
                "do_sample": cfg.decode.do_sample,
                "temperature": cfg.decode.temperature,
                "top_p": cfg.decode.top_p,
                "num_beams": cfg.decode.num_beams,
                "thinking": cfg.decode.thinking,
                "max_new_tokens": adapter.max_new_tokens(),
            },
            cfg.pixel.__dict__,
        )
        if cache_key in completed:
            counts["skipped_completed"] += 1
            continue
        try:
            row = runner.generate(manifest_row, selection)
            append_jsonl(output_path, row)
            completed.add(str(row["cache_key"]))
            counts["completed"] += 1
            if row["parser_status"] != "OK":
                counts["parser_failures"] += 1
            if not row["prediction"]:
                counts["empty"] += 1
        except Exception as exc:
            append_jsonl(errors_path, {"sample_id": sample_id, "stage": "qwen35", "error": repr(exc)})
            counts["errors"] += 1
    return counts


def parse_args() -> argparse.Namespace:
    cfg = RunConfig()
    p = argparse.ArgumentParser(description="Run deterministic Qwen3.5-4B inference on VideoITG-selected MedVidU frames.")
    p.add_argument("--manifest", type=Path, default=cfg.manifest_dir / "medvidu_videoitg_manifest_gt_free.smoke.jsonl")
    p.add_argument("--selector", type=Path, default=cfg.selector_dir / "videoitg_top32.jsonl")
    p.add_argument("--output-root", type=Path, default=cfg.output_root)
    p.add_argument("--checkpoint", default=cfg.qwen_model)
    p.add_argument("--device-map", default="auto")
    p.add_argument("--torch-dtype", default="auto")
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--include-rc-direct", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = RunConfig(output_root=args.output_root, qwen_model=args.checkpoint)
    cfg.make_dirs()
    (cfg.provenance_dir / "qwen35_environment.txt").write_text(environment_snapshot(), encoding="utf-8")
    fingerprint_path = cfg.provenance_dir / "qwen35_model_fingerprint.json"
    if args.preflight_only:
        preflight(args.checkpoint, fingerprint_path, device_map=args.device_map, torch_dtype=args.torch_dtype)
        return 0
    counts = run_inference(
        manifest_path=args.manifest,
        selector_path=args.selector,
        output_path=cfg.prediction_dir / "qwen35_videoitg_predictions.jsonl",
        errors_path=cfg.prediction_dir / "qwen35_errors.jsonl",
        checkpoint=args.checkpoint,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        include_rc_direct=args.include_rc_direct,
    )
    print(counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
