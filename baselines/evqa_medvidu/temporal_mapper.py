from __future__ import annotations

"""E-VQA-specific temporal mapping over MedVidU benchmark frame lists."""

from baselines.timelens_medvidu.temporal_mapper import (
    FrameObservation,
    MappingResult,
    SpacingAudit,
    TemporalMapper as _BaseTemporalMapper,
    audit_uniform_spacing,
)


class TemporalMapper(_BaseTemporalMapper):
    """Keep E-VQA's source-timebase contract independent from TimeLens."""

    SOURCE_TIMEBASE_HZ = {
        **_BaseTemporalMapper.SOURCE_TIMEBASE_HZ,
        # CholecTrack20 frame identifiers refer to the 25 Hz source video,
        # while metadata.fps describes the benchmark's image sampling rate.
        "CholecTrack20": 25.0,
    }

__all__ = ["FrameObservation", "MappingResult", "SpacingAudit", "TemporalMapper", "audit_uniform_spacing"]
