"""Explicit OpenAI-compatible Chat Completions transport, with no SDK retries.

The existing evidence_stability adapter established the image-list content
contract. This implementation deliberately requires all deployment settings
instead of inheriting its guessed defaults. No model weights are updated.
"""
from __future__ import annotations

import base64
from io import BytesIO
import json
import os
from pathlib import Path
import socket
from typing import Any
from urllib import error, request as urllib_request

from PIL import Image

from relive.config import validate_backend
from .base import Backend, BackendError


class _NoRedirect(urllib_request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise BackendError("HTTP_REDIRECT_REFUSED")


class OpenAICompatibleBackend(Backend):
    synthetic = False

    def __init__(self, config: dict[str, Any]):
        validated = validate_backend(config)
        if validated["kind"] != "openai_compatible":
            raise ValueError("OpenAICompatibleBackend requires kind=openai_compatible")
        super().__init__(validated)
        self.endpoint = self.config["base_url"].rstrip("/") + self.config["endpoint_path"]

    def _image_url(self, path: str) -> str:
        encoding = self.config["frame_encoding"]
        with Image.open(path) as source:
            source.load()
            if encoding == "source":
                mime = Image.MIME.get(source.format)
                if mime not in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
                    raise BackendError("UNSUPPORTED_SOURCE_IMAGE_FORMAT")
                if getattr(source, "n_frames", 1) != 1:
                    raise BackendError("ANIMATED_FRAME_NOT_ALLOWED")
                data = Path(path).read_bytes()
            else:
                converted = source.convert("RGB")
                buffer = BytesIO()
                options = {"quality": self.config["jpeg_quality"]} if encoding == "jpeg" else {}
                converted.save(buffer, format=encoding.upper(), **options)
                data = buffer.getvalue()
                mime = "image/jpeg" if encoding == "jpeg" else "image/png"
        return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"

    def _payload(self, request: dict[str, Any]) -> dict[str, Any]:
        content = [{"type": "text", "text": request["prompt"]}]
        paths = request.get("image_paths", [])
        frame_ids = request.get("frame_ids", [])
        if len(paths) != len(frame_ids):
            raise BackendError("FRAME_IDENTITY_LENGTH_MISMATCH")
        # No sorting by filename or fabricated timestamps: caller passes source order.
        for frame_id, path in zip(frame_ids, paths):
            content.append({"type": "text", "text": f"Frame reference: {frame_id}"})
            image = {"url": self._image_url(path)}
            if "image_detail" in self.config:
                image["detail"] = self.config["image_detail"]
            content.append({"type": "image_url", "image_url": image})
        return {"model": self.config["model"], "messages": [{"role": "user", "content": content}],
                **self.config["generation"], **self.config["extra"]}

    def infer(self, request: dict[str, Any]) -> str:
        credential = os.environ.get(self.config["credential_env"])
        if not credential:
            raise BackendError("CREDENTIAL_ENV_UNSET")
        payload = self._payload(request)
        http_request = urllib_request.Request(
            self.endpoint, data=json.dumps(payload, allow_nan=False).encode("utf-8"), method="POST",
            headers={"Authorization": f"Bearer {credential}", "Content-Type": "application/json"},
        )
        self.calls += 1
        try:
            with urllib_request.build_opener(_NoRedirect()).open(
                http_request, timeout=self.config["timeout_seconds"]
            ) as response:
                body = response.read()
        except error.HTTPError as exc:
            # Never log response bodies, headers, or request repr (may hold credentials).
            raise BackendError(f"HTTP_{exc.code}", retryable=exc.code in {408, 429, 500, 502, 503, 504}) from None
        except (error.URLError, TimeoutError, socket.timeout, OSError):
            raise BackendError("TRANSPORT_FAILURE", retryable=True) from None
        try:
            decoded = json.loads(body)
            result = decoded["choices"][0]["message"]["content"]
            if not isinstance(result, str):
                raise ValueError("Nontext content")
            return result
        except (json.JSONDecodeError, UnicodeDecodeError, KeyError, IndexError, TypeError, ValueError):
            raise BackendError("INVALID_COMPLETION_ENVELOPE") from None
