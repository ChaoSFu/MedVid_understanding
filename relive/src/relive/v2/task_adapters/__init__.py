"""CPU-only ReliVE-v2 task-adapter contracts."""
from .base import TaskAdapter
from ..temporal_localization import TemporalLocalizationTaskAdapter
__all__=["TaskAdapter","TemporalLocalizationTaskAdapter"]
