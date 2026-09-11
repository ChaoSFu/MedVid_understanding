"""Versioned core spatial-operator contracts."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image, ImageChops, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from relive.calibration_interventions import (  # noqa: E402
    OPAQUE_GRAY_VERSION as CALIBRATION_OPAQUE_VERSION,
    apply_calibration_intervention,
    calibration_intervention_spec,
)
from relive.config import validate_config  # noqa: E402
from relive.interventions import (  # noqa: E402
    INTERVENTION_VERSION, OPAQUE_GRAY_OPERATOR, OPAQUE_GRAY_VERSION,
    apply_intervention, apply_spatial_intervention, resolve_intervention_spec,
)
from relive.backends.base import Backend  # noqa: E402
from relive.storage import ArtifactStore, CachedInference  # noqa: E402


class StubBackend(Backend):
    synthetic = True

    def __init__(self):
        super().__init__({"kind": "stub", "model": "stub", "revision": "v1", "max_retries": 0})

    def infer(self, request):
        return '{"status":"SUPPORTED"}'


def request(path: Path) -> dict:
    return {"stage": "semantic", "prompt": "fixed prompt", "prompt_version": "p1",
            "image_paths": [str(path)], "frame_ids": ["f0"],
            "context": {"claim": "c", "region": [0.1, 0.1, 0.4, 0.4]}}


class InterventionProtocolTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (40, 30), (10, 30, 50))
        for x in range(8, 29):
            for y in range(5, 24):
                self.image.putpixel((x, y), (220, (x * 11) % 255, (y * 19) % 255))
        self.region = [0.2, 0.2, 0.7, 0.8]

    def test_gaussian_legacy_pixels_are_unchanged(self):
        # This is the pre-protocol GaussianBlur + Image.composite definition.
        mask = Image.new("L", self.image.size, 0)
        box = (8, 6, 28, 24)
        mask.paste(255, box=box)
        blurred = self.image.filter(ImageFilter.GaussianBlur(radius=2.0))
        expected = Image.composite(blurred, self.image, mask)
        actual, audit = apply_intervention(self.image, self.region, "DROP_TARGET", 2.0)
        self.assertEqual(ImageChops.difference(expected, actual).getbbox(), None)
        self.assertEqual(audit["version"], INTERVENTION_VERSION)

    def test_opaque_operator_is_closed_and_preserves_the_expected_regions(self):
        spec = resolve_intervention_spec({"operator": OPAQUE_GRAY_OPERATOR,
                                          "operator_version": OPAQUE_GRAY_VERSION,
                                          "parameters": {"fill_rgb": [127, 127, 127]}}, 4.0)
        drop, drop_audit = apply_spatial_intervention(self.image, self.region, "DROP_TARGET", spec)
        keep, keep_audit = apply_spatial_intervention(self.image, self.region, "KEEP_TARGET", spec)
        control, control_audit = apply_spatial_intervention(self.image, [0.0, 0.0, 0.5, 0.6], "DROP_MATCHED_CONTROL", spec)
        self.assertTrue(drop_audit["inside_changed"] and drop_audit["outside_unchanged"])
        self.assertTrue(keep_audit["inside_unchanged"] and keep_audit["outside_changed"])
        self.assertTrue(control_audit["inside_changed"] and control_audit["outside_unchanged"])
        self.assertEqual(drop_audit["audit_status"], "PASS")
        self.assertEqual(drop.getpixel((10, 10)), (127, 127, 127))
        self.assertEqual(keep.getpixel((10, 10)), self.image.getpixel((10, 10)))

    def test_unregistered_operator_or_parameters_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "UNSUPPORTED_SPATIAL_INTERVENTION_OPERATOR"):
            resolve_intervention_spec({"operator": "anything", "operator_version": "v1", "parameters": {}}, 4.0)
        with self.assertRaisesRegex(ValueError, "UNSUPPORTED_OPAQUE_GRAY"):
            resolve_intervention_spec({"operator": OPAQUE_GRAY_OPERATOR, "operator_version": OPAQUE_GRAY_VERSION,
                                       "parameters": {"fill_rgb": [1, 2, 3]}}, 4.0)

    def test_config_normalizes_legacy_gaussian_and_binds_explicit_operator(self):
        config = validate_config({"spatial": {"blur_radius": 2.0}})
        self.assertEqual(config["spatial"]["intervention"]["parameters"], {"blur_radius": 2.0})
        opaque = validate_config({"spatial": {"blur_radius": 4.0, "intervention": {
            "operator": OPAQUE_GRAY_OPERATOR, "operator_version": OPAQUE_GRAY_VERSION,
            "parameters": {"fill_rgb": [127, 127, 127]},
        }}})
        self.assertEqual(opaque["spatial"]["intervention"]["operator"], OPAQUE_GRAY_OPERATOR)

    def test_calibration_adapter_and_core_share_opaque_pixels(self):
        calibration = calibration_intervention_spec({"calibration_intervention": {
            "operator": "opaque_gray", "operator_version": CALIBRATION_OPAQUE_VERSION,
            "fill_rgb": [127, 127, 127], "selection_basis": "human_visual_review_pre_inference",
        }}, 4.0)
        from_calibration, _ = apply_calibration_intervention(self.image, self.region, "DROP_TARGET", calibration)
        core_spec = resolve_intervention_spec({"operator": OPAQUE_GRAY_OPERATOR,
                                               "operator_version": OPAQUE_GRAY_VERSION,
                                               "parameters": {"fill_rgb": [127, 127, 127]}}, 4.0)
        from_core, _ = apply_spatial_intervention(self.image, self.region, "DROP_TARGET", core_spec)
        self.assertEqual(ImageChops.difference(from_calibration, from_core).getbbox(), None)

    def test_operator_identity_prevents_cache_collision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "image.png"; self.image.save(path)
            cache = CachedInference(StubBackend(), ArtifactStore(root / "cache"))
            base = request(path)
            gaussian = deepcopy(base); gaussian["context"]["intervention_protocol"] = resolve_intervention_spec(None, 4.0)
            opaque = deepcopy(base); opaque["context"]["intervention_protocol"] = resolve_intervention_spec({
                "operator": OPAQUE_GRAY_OPERATOR, "operator_version": OPAQUE_GRAY_VERSION,
                "parameters": {"fill_rgb": [127, 127, 127]}}, 4.0)
            self.assertNotEqual(cache.cache_key(gaussian), cache.cache_key(opaque))


if __name__ == "__main__":
    unittest.main()
