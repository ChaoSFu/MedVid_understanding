import tempfile
import unittest
import importlib.util
from pathlib import Path

from evidence_stability.cache import make_probe_cache_key
from evidence_stability.phase_c import phase_c_label
from evidence_stability.prompts import build_evidence_presence_prompt, parse_yes_no


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "03_probe_evidence.py"
SPEC = importlib.util.spec_from_file_location("phase_c_probe_script", SCRIPT_PATH)
phase_c_probe_script = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(phase_c_probe_script)


class PhaseCTests(unittest.TestCase):
    def test_parse_yes_no_conservative(self):
        self.assertEqual(parse_yes_no("YES"), "YES")
        self.assertEqual(parse_yes_no("Yes."), "YES")
        self.assertEqual(parse_yes_no("NO"), "NO")
        self.assertEqual(parse_yes_no("No."), "NO")
        self.assertEqual(parse_yes_no("Yes, because the tool is visible."), "INVALID")
        self.assertEqual(parse_yes_no("maybe"), "INVALID")

    def test_prompt_has_no_gt_terms(self):
        prompt = build_evidence_presence_prompt("suturing")
        lower = prompt.lower()
        self.assertIn("suturing", prompt)
        self.assertNotIn("gt", lower)
        self.assertNotIn("ground truth", lower)
        self.assertNotIn("span", lower)

    def test_cache_key_separates_same_clip_different_qa(self):
        frames = ["a.jpg", "b.jpg"]
        prompt = build_evidence_presence_prompt("suturing")
        key_a = make_probe_cache_key("m", "r1", "evidence_presence_v1", "0001::clip", "0001::clip::pos0000-0001", frames, prompt)
        key_b = make_probe_cache_key("m", "r1", "evidence_presence_v1", "0002::clip", "0002::clip::pos0000-0001", frames, prompt)
        self.assertNotEqual(key_a, key_b)

    def test_cache_key_separates_ordered_frames(self):
        prompt = build_evidence_presence_prompt("suturing")
        key_a = make_probe_cache_key("m", "r1", "evidence_presence_v1", "q", "w", ["a.jpg", "b.jpg"], prompt)
        key_b = make_probe_cache_key("m", "r1", "evidence_presence_v1", "q", "w", ["b.jpg", "a.jpg"], prompt)
        self.assertNotEqual(key_a, key_b)

    def test_cache_key_separates_model_fingerprint_and_decoding(self):
        prompt = build_evidence_presence_prompt("suturing")
        base = ("m", "r1", "evidence_presence_v1", "q", "w", ["a.jpg", "b.jpg"], prompt)
        key_a = make_probe_cache_key(*base, model_fingerprint={"config_sha256": "a"}, decoding_config={"max_new_tokens": 8})
        key_b = make_probe_cache_key(*base, model_fingerprint={"config_sha256": "b"}, decoding_config={"max_new_tokens": 8})
        key_c = make_probe_cache_key(*base, model_fingerprint={"config_sha256": "a"}, decoding_config={"max_new_tokens": 16})
        self.assertNotEqual(key_a, key_b)
        self.assertNotEqual(key_a, key_c)

    def test_frame_root_remap(self):
        resolved = phase_c_probe_script.resolve_frame_path(
            "/root/data/NurViD/foo/0001.jpg",
            "/mnt/hdd3/huihui/hh_datas/MedVidU/valdata",
            "/root/data",
        )
        self.assertEqual(resolved, "/mnt/hdd3/huihui/hh_datas/MedVidU/valdata/NurViD/foo/0001.jpg")

    def test_posthoc_phase_c_labels(self):
        self.assertEqual(phase_c_label("YES", "STRONG_GT_ALIGNED"), "TRUE_SUPPORT")
        self.assertEqual(phase_c_label("YES", "WEAK_GT_ALIGNED"), "WEAK_TRUE_SUPPORT")
        self.assertEqual(phase_c_label("YES", "NO_GT_OVERLAP"), "SPURIOUS_SUPPORT")
        self.assertEqual(phase_c_label("NO", "NO_GT_OVERLAP"), "TRUE_NEGATIVE_CONTROL")
        self.assertEqual(phase_c_label("NO", "STRONG_GT_ALIGNED"), "NO_ON_STRONG_GT_ALIGNED")


if __name__ == "__main__":
    unittest.main()
