import unittest

from evidence_stability.h2_statistics import exact_wilcoxon_signed_rank


class H2StatisticsTests(unittest.TestCase):
    def test_exact_signed_rank_all_positive_three_pairs(self):
        result = exact_wilcoxon_signed_rank([1.0, 2.0, 3.0])
        self.assertEqual(result["n_pairs"], 3)
        self.assertEqual(result["n_nonzero_deltas"], 3)
        self.assertEqual(result["W_plus"], 6.0)
        self.assertEqual(result["W_minus"], 0.0)
        self.assertEqual(result["rank_biserial_correlation"], 1.0)
        self.assertEqual(result["p_value_two_sided_exact"], 0.25)

    def test_exact_signed_rank_excludes_zero_deltas(self):
        result = exact_wilcoxon_signed_rank([0.0, 1.0, -2.0])
        self.assertEqual(result["n_pairs"], 3)
        self.assertEqual(result["n_zero_deltas"], 1)
        self.assertEqual(result["n_nonzero_deltas"], 2)
        self.assertEqual(result["W_plus"], 1.0)
        self.assertEqual(result["W_minus"], 2.0)

    def test_exact_signed_rank_rejects_more_than_twenty_nonzero_pairs(self):
        with self.assertRaises(ValueError):
            exact_wilcoxon_signed_rank([1.0] * 21)
