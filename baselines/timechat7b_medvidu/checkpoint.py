from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .config import DEFAULT_OUTPUT_ROOT, RunConfig
from .io_utils import sha256_file, write_json


CRITICAL_KEY_PREFIXES = ("video_Qformer.", "llama_proj.", "llama_model.", "Qformer.")
CRITICAL_EXACT_KEYS = ("video_frame_position_embedding.weight",)


def inspect_checkpoint(vtune_ckpt: Path) -> dict[str, Any]:
    if not vtune_ckpt.exists():
        raise FileNotFoundError(f"STOP: VTune checkpoint not found: {vtune_ckpt}")
    import torch

    loaded = torch.load(vtune_ckpt, map_location="cpu")
    model_state = loaded.get("model") if isinstance(loaded, dict) else None
    keys = list(model_state.keys()) if isinstance(model_state, dict) else []
    result = {
        "checkpoint_path": str(vtune_ckpt),
        "filename": vtune_ckpt.name,
        "file_size_bytes": vtune_ckpt.stat().st_size,
        "sha256": sha256_file(vtune_ckpt),
        "torch_load_success": True,
        "top_level_keys": sorted(list(loaded.keys())) if isinstance(loaded, dict) else [],
        "has_model_key": isinstance(model_state, dict),
        "number_of_state_dict_keys": len(keys),
        "critical_key_presence": {
            key: key in keys for key in CRITICAL_EXACT_KEYS
        }
        | {
            prefix + "*": any(k.startswith(prefix) for k in keys)
            for prefix in CRITICAL_KEY_PREFIXES
        },
    }
    if not result["has_model_key"]:
        raise ValueError("STOP: checkpoint does not contain ckpt['model']")
    return result


def audit_load_message(msg: Any) -> dict[str, Any]:
    missing = list(getattr(msg, "missing_keys", []) or [])
    unexpected = list(getattr(msg, "unexpected_keys", []) or [])
    critical_missing = [
        key
        for key in missing
        if key in CRITICAL_EXACT_KEYS or any(key.startswith(prefix) for prefix in CRITICAL_KEY_PREFIXES)
    ]
    return {
        "n_missing_keys": len(missing),
        "n_unexpected_keys": len(unexpected),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "critical_missing_keys": critical_missing,
        "critical_mismatch": bool(critical_missing),
        "status": "STOP" if critical_missing else "OK",
    }


def verify_component_paths(vtune_ckpt: Path, vit_model: Path, q_former_model: Path, llama_model: Path) -> dict[str, Any]:
    components = {
        "vtune_ckpt": vtune_ckpt,
        "vit_model": vit_model,
        "q_former_model": q_former_model,
        "llama_model": llama_model,
    }
    result = {name: {"path": str(path), "exists": path.exists()} for name, path in components.items()}
    missing = [name for name, payload in result.items() if not payload["exists"]]
    result["all_found"] = len(missing) == 0
    result["missing_components"] = missing
    if missing:
        raise FileNotFoundError("STOP: missing TimeChat component(s): " + ", ".join(missing))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect the TimeChat ActivityNet VTune checkpoint without modifying it.")
    parser.add_argument("--vtune-ckpt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = RunConfig(output_root=args.output_root)
    cfg.make_dirs()
    result = inspect_checkpoint(args.vtune_ckpt)
    write_json(cfg.provenance_dir / "checkpoint_inspection.json", result)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
