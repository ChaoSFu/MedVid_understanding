from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InterventionPositions:
    positions: tuple[int, ...]
    generation_valid: bool
    generation_invalid_reason: str | None = None
    boundary_asymmetric: bool = False


def _invalid_boundary() -> InterventionPositions:
    return InterventionPositions(
        positions=tuple(),
        generation_valid=False,
        generation_invalid_reason="GENERATION_INVALID_BOUNDARY",
    )


def generate_shift(
    original_start_pos: int,
    original_end_pos: int,
    clip_n_sampled_positions: int,
    delta: int,
) -> InterventionPositions:
    start = original_start_pos + delta
    end = original_end_pos + delta
    if start < 0 or end >= clip_n_sampled_positions:
        return _invalid_boundary()
    return InterventionPositions(
        positions=tuple(range(start, end + 1)),
        generation_valid=True,
    )


def evenly_spaced_indices(input_length: int, output_length: int) -> tuple[int, ...]:
    if output_length <= 0:
        raise ValueError("output_length must be positive")
    if output_length > input_length:
        raise ValueError("output_length cannot exceed input_length")
    if output_length == 1:
        return (0,)
    indices = tuple(round(i * (input_length - 1) / (output_length - 1)) for i in range(output_length))
    if len(set(indices)) != output_length:
        raise RuntimeError(f"Evenly spaced rule produced duplicate positions: {indices}")
    return indices


def generate_resample(
    original_start_pos: int,
    original_end_pos: int,
    output_length: int,
) -> InterventionPositions:
    input_length = original_end_pos - original_start_pos + 1
    relative = evenly_spaced_indices(input_length, output_length)
    return InterventionPositions(
        positions=tuple(original_start_pos + i for i in relative),
        generation_valid=True,
    )


def generate_context(
    original_start_pos: int,
    original_end_pos: int,
    clip_n_sampled_positions: int,
    target_length: int = 24,
) -> InterventionPositions:
    original_length = original_end_pos - original_start_pos + 1
    if original_length > target_length or target_length > clip_n_sampled_positions:
        return _invalid_boundary()

    extra = target_length - original_length
    preferred_before = extra // 2
    preferred_after = extra - preferred_before
    before = min(preferred_before, original_start_pos)
    after = min(preferred_after, clip_n_sampled_positions - original_end_pos - 1)
    missing_before = preferred_before - before
    missing_after = preferred_after - after

    if missing_before:
        after += min(missing_before, clip_n_sampled_positions - original_end_pos - 1 - after)
    if missing_after:
        before += min(missing_after, original_start_pos - before)

    if before + original_length + after != target_length:
        return _invalid_boundary()

    start = original_start_pos - before
    end = original_end_pos + after
    return InterventionPositions(
        positions=tuple(range(start, end + 1)),
        generation_valid=True,
        boundary_asymmetric=(before != preferred_before or after != preferred_after),
    )


def intervention_position_plan(
    original_start_pos: int,
    original_end_pos: int,
    clip_n_sampled_positions: int,
    intervention_type: str,
) -> InterventionPositions:
    if intervention_type == "SHIFT_LEFT_2":
        return generate_shift(original_start_pos, original_end_pos, clip_n_sampled_positions, delta=-2)
    if intervention_type == "SHIFT_RIGHT_2":
        return generate_shift(original_start_pos, original_end_pos, clip_n_sampled_positions, delta=2)
    if intervention_type == "RESAMPLE_75":
        return generate_resample(original_start_pos, original_end_pos, output_length=12)
    if intervention_type == "RESAMPLE_50":
        return generate_resample(original_start_pos, original_end_pos, output_length=8)
    if intervention_type == "CONTEXT_1P5X":
        return generate_context(original_start_pos, original_end_pos, clip_n_sampled_positions, target_length=24)
    raise ValueError(f"Unsupported intervention_type: {intervention_type}")
