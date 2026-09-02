import unittest

from evidence_stability.dataset import normalize_tal_sample
from evidence_stability.windows import compute_window_alignment, generate_position_windows


def make_sample():
    return {
        "id": "sample-1",
        "qa_type": "tal",
        "dataset_name": "AVOS",
        "data_source": "AVOS",
        "metadata": {"fps": "1.0"},
        "video": [f"/root/data/AVOS/frames_15fps/v/{i}.jpg" for i in [0, 15, 30, 45, 60, 75]],
        "sampled_video_frames": [0, 15, 30, 45, 60, 75],
        "conversations": [
            {"from": "human", "value": "<video>\nWhen does action happen?"},
            {"from": "gpt", "value": "1.0-2.0 seconds."},
        ],
        "struc_info": [{"action": "action", "spans": [{"start": 1.0, "end": 2.0}]}],
    }


class WindowTests(unittest.TestCase):
    def test_sliding_window_boundaries_drop_last(self):
        normalized, failure = normalize_tal_sample(make_sample(), 0, verify_paths=False)
        self.assertIsNone(failure)
        windows = generate_position_windows(normalized, window_size=4, stride=2, drop_last=True)
        self.assertEqual([(w["start_pos"], w["end_pos"]) for w in windows], [(0, 3), (2, 5)])

    def test_sliding_window_boundaries_keep_last(self):
        normalized, _ = normalize_tal_sample(make_sample(), 0, verify_paths=False)
        windows = generate_position_windows(normalized, window_size=4, stride=3, drop_last=False)
        self.assertEqual([(w["start_pos"], w["end_pos"]) for w in windows], [(0, 3), (3, 5)])

    def test_window_alignment_metrics(self):
        normalized, _ = normalize_tal_sample(make_sample(), 0, verify_paths=False)
        windows = generate_position_windows(normalized, window_size=2, stride=2, drop_last=True)
        metrics = compute_window_alignment(normalized, windows[0])
        self.assertGreater(metrics["n_gt_visible_in_window"], 0)
        self.assertGreater(metrics["evidence_density"], 0)
        self.assertIn(metrics["gt_alignment_class"], {"STRONG_GT_ALIGNED", "WEAK_GT_ALIGNED"})

    def test_same_clip_different_qa_have_different_window_ids(self):
        sample_a = make_sample()
        sample_b = make_sample()
        normalized_a, _ = normalize_tal_sample(sample_a, 0, verify_paths=False)
        normalized_b, _ = normalize_tal_sample(sample_b, 1, verify_paths=False)
        self.assertEqual(normalized_a["clip_id"], normalized_b["clip_id"])
        self.assertNotEqual(normalized_a["qa_id"], normalized_b["qa_id"])
        window_a = generate_position_windows(normalized_a, window_size=4, stride=2)[0]
        window_b = generate_position_windows(normalized_b, window_size=4, stride=2)[0]
        self.assertNotEqual(window_a["window_id"], window_b["window_id"])


if __name__ == "__main__":
    unittest.main()
