import unittest

from evidence_stability.dataset import extract_gt_spans, extract_target, normalize_tal_sample


class DatasetTests(unittest.TestCase):
    def test_extract_action_or_phase(self):
        action, field = extract_target({"struc_info": [{"phase": "closure", "spans": []}]})
        self.assertEqual(action, "closure")
        self.assertEqual(field, "phase")

    def test_gt_answer_fallback_parser(self):
        spans, source = extract_gt_spans({"struc_info": []}, "start: 1.0, end: 2.0; 4.0 to 5.0 seconds")
        self.assertEqual(source, "gt_answer")
        self.assertEqual([(s.start, s.end) for s in spans], [(1.0, 2.0), (4.0, 5.0)])

    def test_normalize_duplicate_fields(self):
        sample = {
            "id": "dup",
            "qa_type": "tal",
            "dataset_name": "Toy",
            "data_source": "Toy",
            "metadata": {"fps": "1.0"},
            "video": ["a.jpg", "a.jpg", "b.jpg"],
            "sampled_video_frames": [1, 1, 2],
            "conversations": [
                {"from": "human", "value": "When does grasping happen?"},
                {"from": "gpt", "value": "0.0-1.0 seconds."},
            ],
            "struc_info": [{"action": "grasping", "spans": [{"start": 0.0, "end": 1.0}]}],
        }
        normalized, failure = normalize_tal_sample(sample, 0, verify_paths=False)
        self.assertIsNone(failure)
        self.assertEqual(normalized["duplicate_count"], 1)
        self.assertEqual(normalized["n_unique_frames"], 2)


if __name__ == "__main__":
    unittest.main()
