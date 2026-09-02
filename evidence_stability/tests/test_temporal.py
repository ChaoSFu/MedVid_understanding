import unittest

from evidence_stability.temporal import (
    GTSpan,
    TemporalMapper,
    build_temporal_cells,
    classify_gt_alignment,
    compute_alignment_metrics,
    label_support,
    map_frames_mapping_c,
    validate_and_clip_gt_spans,
)


class TemporalMappingTests(unittest.TestCase):
    def test_avos_source_timebase_adapter(self):
        result = TemporalMapper.map(
            "AVOS",
            [150, 165, 180],
            ["a.jpg", "b.jpg", "c.jpg"],
            {"fps": "1.0"},
        )
        self.assertEqual(result.time_mapping_method, "source_frame_rate")
        self.assertEqual(result.source_timebase_hz, 15.0)
        self.assertEqual(result.clip_duration, 2.0)
        self.assertEqual([round(x.local_time, 3) for x in result.observations], [0.0, 1.0, 2.0])

    def test_cholect50_source_timebase_adapter(self):
        result = TemporalMapper.map(
            "CholecT50",
            [100, 110],
            ["a.jpg", "b.jpg"],
            {"fps": "0.1"},
        )
        self.assertEqual(result.source_timebase_hz, 1.0)
        self.assertEqual(result.clip_duration, 10.0)

    def test_endpoint_anchored_mapping_c(self):
        duration, obs = map_frames_mapping_c(
            [100, 110, 120],
            ["a.jpg", "b.jpg", "c.jpg"],
            {"fps": "1.0"},
        )
        self.assertEqual(duration, 2.0)
        self.assertEqual([round(x.local_time, 3) for x in obs], [0.0, 1.0, 2.0])

    def test_nurvid_known_duration_mapping(self):
        duration, obs = map_frames_mapping_c(
            [153, 156, 249],
            ["a.jpg", "b.jpg", "c.jpg"],
            {
                "fps": "1.0",
                "input_video_start_time": "76.4",
                "input_video_end_time": "124.2",
            },
        )
        self.assertAlmostEqual(duration, 47.8)
        self.assertAlmostEqual(obs[-1].local_time, 47.8)

    def test_duplicate_source_frame_indices_unique_cells(self):
        duration, obs = map_frames_mapping_c(
            [10, 10, 20],
            ["a.jpg", "a.jpg", "b.jpg"],
            {"fps": "1.0"},
        )
        cells = build_temporal_cells(obs, duration)
        self.assertEqual(len(cells), 2)
        self.assertEqual(cells[0].source_frame_index, 10)
        self.assertEqual(cells[0].frame_positions, (0, 1))

    def test_zero_duration_gt_point_intersects_cell(self):
        duration, obs = map_frames_mapping_c(
            [0, 10, 20],
            ["a.jpg", "b.jpg", "c.jpg"],
            {"fps": "1.0"},
        )
        cells = build_temporal_cells(obs, duration)
        metrics = compute_alignment_metrics(cells, [10], [GTSpan(1.0, 1.0)])
        self.assertEqual(metrics.n_gt_visible_in_window, 1)

    def test_multi_span_visible_count(self):
        duration, obs = map_frames_mapping_c(
            [0, 10, 20, 30],
            ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
            {"fps": "1.0"},
        )
        cells = build_temporal_cells(obs, duration)
        metrics = compute_alignment_metrics(cells, [0, 10, 20, 30], [GTSpan(0, 0), GTSpan(3, 3)])
        self.assertEqual(metrics.n_gt_visible_total, 2)

    def test_small_gt_overflow_clips_within_tolerance(self):
        validation = validate_and_clip_gt_spans([GTSpan(-0.5, 2.5)], 2.0, 1.0)
        self.assertEqual(validation.temporal_status, "OK")
        self.assertTrue(validation.gt_boundary_clipped)
        self.assertEqual(validation.processed_spans[0], GTSpan(0.0, 2.0))
        self.assertTrue(all(0 <= s.start <= s.end <= 2.0 for s in validation.processed_spans))

    def test_large_gt_overflow_excluded(self):
        validation = validate_and_clip_gt_spans([GTSpan(0.0, 5.0)], 2.0, 1.0)
        self.assertEqual(validation.temporal_status, "GT_OUT_OF_RANGE")

    def test_outside_near_boundary_not_snapped_to_boundary(self):
        validation = validate_and_clip_gt_spans([GTSpan(185.0, 185.0)], 184.0, 1.0)
        self.assertEqual(validation.temporal_status, "GT_NEAR_BOUNDARY_NOT_VISIBLE")
        self.assertEqual(validation.processed_spans, ())
        self.assertEqual(validation.invalid_gt_spans[0]["reason"], "OUTSIDE_CLIP_NEAR_BOUNDARY")

    def test_density_recall_and_labels(self):
        duration, obs = map_frames_mapping_c(
            [0, 10, 20, 30],
            ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
            {"fps": "1.0"},
        )
        cells = build_temporal_cells(obs, duration)
        strong = compute_alignment_metrics(cells, [10, 20], [GTSpan(1.0, 2.0)])
        self.assertEqual(strong.n_gt_visible_in_window, 2)
        self.assertAlmostEqual(strong.evidence_density, 1.0)
        self.assertEqual(classify_gt_alignment(strong), "STRONG_GT_ALIGNED")
        self.assertEqual(label_support("YES", strong), "STRONG_GT_SUPPORT")

        weak = compute_alignment_metrics(cells, [0], [GTSpan(1.0, 2.0)])
        self.assertEqual(classify_gt_alignment(weak), "NO_GT_OVERLAP")
        self.assertEqual(label_support("YES", weak), "SPURIOUS_SUPPORT")


if __name__ == "__main__":
    unittest.main()
