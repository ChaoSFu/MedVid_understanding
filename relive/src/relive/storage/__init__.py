"""Immutable research artifacts and inference replay.

Cache symbols stay lazily imported so CPU-only protocol readers can reuse the
canonical artifact hash without importing the inference/cache implementation.
"""
from .artifacts import ArtifactError, ArtifactStore, stable_hash, stable_id

__all__ = ["ArtifactError", "ArtifactStore", "Budget", "BudgetExceeded", "CachedInference", "stable_hash", "stable_id"]

def __getattr__(name: str):
    if name in {"Budget", "BudgetExceeded", "CachedInference"}:
        from .cache import Budget, BudgetExceeded, CachedInference
        return {"Budget": Budget, "BudgetExceeded": BudgetExceeded, "CachedInference": CachedInference}[name]
    raise AttributeError(name)
