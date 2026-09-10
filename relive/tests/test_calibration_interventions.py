"""Tests for isolated, frozen calibration-only intervention operators."""
from __future__ import annotations

import unittest

from PIL import Image

from relive.calibration_interventions import (
    OPAQUE_GRAY_VERSION,
    apply_calibration_intervention,
    calibration_intervention_spec,
)


class CalibrationInterventionTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (40, 30), (20, 50, 90))
        for x in range(10, 20):
            for y in range(8, 20):
                self.image.putpixel((x, y), (220, 30, 10))
        self.region = [0.25, 0.25, 0.5, 0.75]
        self.spec = calibration_intervention_spec({"calibration_intervention": {
            "operator": "opaque_gray", "operator_version": OPAQUE_GRAY_VERSION,
            "fill_rgb": [127, 127, 127], "selection_basis": "human_visual_review_pre_inference",
        }}, 4)

    def test_opaque_gray_drop_is_local_and_keep_preserves_target(self):
        dropped, drop_audit = apply_calibration_intervention(self.image, self.region, "DROP_TARGET", self.spec)
        kept, keep_audit = apply_calibration_intervention(self.image, self.region, "KEEP_TARGET", self.spec)
        original, original_audit = apply_calibration_intervention(self.image, self.region, "ORIGINAL", self.spec)
        self.assertEqual(drop_audit["audit_status"], "PASS")
        self.assertTrue(drop_audit["inside_changed"])
        self.assertTrue(drop_audit["outside_unchanged"])
        self.assertEqual(keep_audit["audit_status"], "PASS")
        self.assertTrue(keep_audit["inside_unchanged"])
        self.assertTrue(keep_audit["outside_changed"])
        self.assertEqual(original, self.image)
        self.assertEqual(original_audit["audit_status"], "PASS")
        self.assertEqual(dropped.getpixel((11, 10)), (127, 127, 127))
        self.assertEqual(kept.getpixel((11, 10)), self.image.getpixel((11, 10)))

    def test_rejects_non_frozen_opaque_parameters(self):
        with self.assertRaisesRegex(ValueError, "NOT_FROZEN_OPAQUE_GRAY"):
            calibration_intervention_spec({"calibration_intervention": {
                "operator": "opaque_gray", "operator_version": OPAQUE_GRAY_VERSION,
                "fill_rgb": [0, 0, 0], "selection_basis": "human_visual_review_pre_inference",
            }}, 4)


if __name__ == "__main__":
    unittest.main()
