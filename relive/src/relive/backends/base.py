"""An infer() invocation makes exactly one transport/model attempt.

Bounded technical retries belong to CachedInference so every attempt is paid
from the sample budget and retained in immutable artifacts.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any


class BackendError(RuntimeError):
    """Safe public diagnostic; do not place credentials or response bodies here."""

    def __init__(self, reason: str, *, retryable: bool = False):
        super().__init__(reason)
        self.retryable = retryable


class Backend(ABC):
    synthetic = False

    def __init__(self, config: dict[str, Any]):
        self.config = deepcopy(config)
        self.calls = 0

    def fingerprint(self) -> dict[str, Any]:
        return {"adapter_version": "relive-backend-v1", "synthetic": self.synthetic,
                "config": deepcopy(self.config)}

    @abstractmethod
    def infer(self, request: dict[str, Any]) -> str:
        """Return raw model text, without semantic retries or interpretation."""
        raise NotImplementedError
