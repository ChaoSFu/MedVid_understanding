import unittest

from evidence_stability.dataset import normalize_tal_sample
from evidence_stability.windows import compute_window_alignment, generate_position_windows


def make_sample():
    return {
        "id": "sample-1",
        "qa_type": "tal",
        "dataset_name": "Toy",
        "data_source": "Toy",
        "metadata": {"fps": "1.0"},
        "video": [f"/root/data/Toy/{i}.jpg" for i in range(6)],
        "sampled_video_frames": [0, 1, 2, 3, 4, 5],
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


if __name__ == "__main__":
    unittest.main()
