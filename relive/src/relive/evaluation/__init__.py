"""Offline evaluation: the only ReliVE component allowed to load hidden GT.

Nothing in runtime imports this package. Call evaluate from the independent
evaluation CLI handler after predictions have been written.
"""

from .metrics import evaluate

__all__ = ["evaluate"]
