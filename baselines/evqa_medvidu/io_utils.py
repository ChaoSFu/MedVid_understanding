from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .config import GT_DERIVED_SUBSTRINGS, GT_FORBIDDEN_KEYS


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_no} is not a JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            f.write("\n")
    os.replace(tmp, path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        f.write("\n")


def stable_json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(stable_json_dumps(value))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def first_gt_leak(value: Any, path: str = "$", allowed_paths: set[str] | None = None) -> str | None:
    allowed_paths = allowed_paths or set()
    if path in allowed_paths:
        return None
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            key_lower = key_text.lower()
            child_path = f"{path}.{key_text}"
            if child_path in allowed_paths:
                continue
            if key_text in GT_FORBIDDEN_KEYS or key_lower in {k.lower() for k in GT_FORBIDDEN_KEYS}:
                return child_path
            if any(part in key_lower for part in GT_DERIVED_SUBSTRINGS):
                return child_path
            leak = first_gt_leak(child, child_path, allowed_paths)
            if leak is not None:
                return leak
    elif isinstance(value, list):
        for i, child in enumerate(value):
            leak = first_gt_leak(child, f"{path}[{i}]", allowed_paths)
            if leak is not None:
                return leak
    return None


def assert_no_gt_leak(value: Any, allowed_paths: set[str] | None = None) -> None:
    leak = first_gt_leak(value, allowed_paths=allowed_paths)
    if leak is not None:
        raise AssertionError(f"GT-derived field leaked into inference artifact at {leak}")


def completed_cache(path: Path) -> set[str]:
    return {str(row["cache_key"]) for row in read_jsonl(path) if row.get("cache_key") and not row.get("error")}


def run_command(args: list[str], cwd: Path) -> str:
    result = subprocess.run(args, cwd=cwd, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return result.stdout


def environment_snapshot(extra_packages: tuple[str, ...] = ()) -> str:
    modules = (
        "torch",
        "torchvision",
        "transformers",
        "peft",
        "accelerate",
        "flash_attn",
        "cv2",
        "PIL",
        "numpy",
        "decord",
        "nncore",
        "pycocotools",
        "hydra",
        *extra_packages,
    )
    lines = [f"python: {sys.version}", f"executable: {sys.executable}", f"cwd: {Path.cwd()}"]
    for module_name in modules:
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "unknown")
            lines.append(f"{module_name}: {version}")
        except Exception as exc:
            lines.append(f"{module_name}: UNAVAILABLE ({exc!r})")
    for command in (["nvidia-smi"], ["nvcc", "--version"]):
        try:
            result = subprocess.run(command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            lines.append(f"{' '.join(command)}:")
            lines.append(result.stdout.strip())
        except Exception as exc:
            lines.append(f"{' '.join(command)}: UNAVAILABLE ({exc!r})")
    return "\n".join(lines) + "\n"

