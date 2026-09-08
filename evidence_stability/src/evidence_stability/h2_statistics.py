from __future__ import annotations

import itertools
import math
from statistics import mean, median
from typing import Any


def _average_abs_ranks(values: list[float]) -> list[float]:
    """Return average ranks for absolute nonzero paired differences."""
    ordered = sorted(enumerate(values), key=lambda item: abs(item[1]))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and math.isclose(abs(ordered[end][1]), abs(ordered[start][1]), abs_tol=1e-12):
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        for index, _ in ordered[start:end]:
            ranks[index] = average_rank
        start = end
    return ranks


def exact_wilcoxon_signed_rank(deltas: list[float], *, max_nonzero_pairs: int = 20) -> dict[str, Any]:
    """Compute a two-sided exact Wilcoxon signed-rank test by sign enumeration.

    Zero differences are removed (the Wilcox convention). This exact conditional
    distribution also handles tied absolute differences through average ranks.
    """
    clean = [float(value) for value in deltas]
    if not clean or any(not math.isfinite(value) for value in clean):
        raise ValueError("deltas must contain at least one finite numeric value")

    nonzero = [value for value in clean if not math.isclose(value, 0.0, abs_tol=1e-12)]
    if len(nonzero) > max_nonzero_pairs:
        raise ValueError(
            f"Exact signed-rank enumeration is limited to {max_nonzero_pairs} nonzero pairs; got {len(nonzero)}."
        )
    if not nonzero:
        return {
            "n_pairs": len(clean),
            "n_zero_deltas": len(clean),
            "n_nonzero_deltas": 0,
            "W_plus": 0.0,
            "W_minus": 0.0,
            "rank_biserial_correlation": 0.0,
            "p_value_two_sided_exact": 1.0,
            "mean_delta": mean(clean),
            "median_delta": median(clean),
        }

    ranks = _average_abs_ranks(nonzero)
    total_rank = sum(ranks)
    w_plus = sum(rank for value, rank in zip(nonzero, ranks) if value > 0)
    w_minus = total_rank - w_plus
    observed_distance = abs(w_plus - total_rank / 2.0)
    extreme = 0
    for signs in itertools.product((0, 1), repeat=len(nonzero)):
        simulated_w_plus = sum(rank for sign, rank in zip(signs, ranks) if sign)
        if abs(simulated_w_plus - total_rank / 2.0) >= observed_distance - 1e-12:
            extreme += 1
    total_permutations = 2 ** len(nonzero)
    return {
        "n_pairs": len(clean),
        "n_zero_deltas": len(clean) - len(nonzero),
        "n_nonzero_deltas": len(nonzero),
        "W_plus": w_plus,
        "W_minus": w_minus,
        "rank_biserial_correlation": (w_plus - w_minus) / total_rank,
        "p_value_two_sided_exact": extreme / total_permutations,
        "mean_delta": mean(clean),
        "median_delta": median(clean),
    }
