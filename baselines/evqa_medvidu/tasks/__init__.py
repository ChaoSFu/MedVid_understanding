from __future__ import annotations

from .base import MedVidUTaskAdapter, SchemaMismatch
from .cvs import CVSAdapter
from .rc import RCAdapter
from .stg import STGAdapter


def get_adapter(task: str) -> MedVidUTaskAdapter:
    if task == "stg":
        return STGAdapter()
    if task == "rc":
        return RCAdapter()
    if task == "cvs":
        return CVSAdapter()
    raise KeyError(f"unknown E-VQA MedVidU task: {task}")


__all__ = ["CVSAdapter", "MedVidUTaskAdapter", "RCAdapter", "STGAdapter", "SchemaMismatch", "get_adapter"]

