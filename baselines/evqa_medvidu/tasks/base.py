from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class SchemaMismatch(RuntimeError):
    """Raised when a GT-free conversion rule must be frozen before proceeding."""


class MedVidUTaskAdapter(ABC):
    task: str

    @abstractmethod
    def build_model_input(self, sample: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def parse_evqa_output(self, raw: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def to_medvidu_prediction(self, sample: dict[str, Any], parsed: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def validate_prediction(self, prediction: dict[str, Any]) -> None:
        raise NotImplementedError

