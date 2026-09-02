from __future__ import annotations


def phase_c_label(parsed_prediction: str, gt_alignment_class: str) -> str:
    if parsed_prediction == "YES" and gt_alignment_class == "STRONG_GT_ALIGNED":
        return "TRUE_SUPPORT"
    if parsed_prediction == "YES" and gt_alignment_class == "WEAK_GT_ALIGNED":
        return "WEAK_TRUE_SUPPORT"
    if parsed_prediction == "YES" and gt_alignment_class == "NO_GT_OVERLAP":
        return "SPURIOUS_SUPPORT"
    if parsed_prediction == "NO" and gt_alignment_class == "NO_GT_OVERLAP":
        return "TRUE_NEGATIVE_CONTROL"
    if parsed_prediction == "NO" and gt_alignment_class == "STRONG_GT_ALIGNED":
        return "NO_ON_STRONG_GT_ALIGNED"
    if parsed_prediction == "NO" and gt_alignment_class == "WEAK_GT_ALIGNED":
        return "NO_ON_WEAK_GT_ALIGNED"
    if parsed_prediction == "INVALID":
        return "INVALID_RESPONSE"
    if parsed_prediction == "ERROR":
        return "ERROR"
    return "OTHER"
