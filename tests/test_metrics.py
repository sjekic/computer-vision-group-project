import math
import unittest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metrics import average_precision_at_k, relevant_counts_by_location, topk_hit


class MetricsTests(unittest.TestCase):
    def test_average_precision_uses_total_relevant_database_items(self):
        retrieved = ["hall", "cafeteria", "hall", "hall"]

        score = average_precision_at_k(
            retrieved_locations=retrieved,
            gt_location="hall",
            total_relevant=4,
            k=4,
        )

        expected = (1 / 1 + 2 / 3 + 3 / 4) / 4
        self.assertTrue(math.isclose(score, expected, rel_tol=1e-9))

    def test_average_precision_caps_relevant_count_by_k(self):
        retrieved = ["hall", "hall", "hall"]

        score = average_precision_at_k(
            retrieved_locations=retrieved,
            gt_location="hall",
            total_relevant=20,
            k=3,
        )

        self.assertEqual(score, 1.0)

    def test_topk_hit_returns_false_for_missing_ground_truth(self):
        self.assertFalse(topk_hit(["hall", "stairs"], None, 2))
        self.assertFalse(topk_hit(["hall", "stairs"], "cafeteria", 2))
        self.assertTrue(topk_hit(["hall", "stairs"], "stairs", 2))

    def test_relevant_counts_are_grouped_from_image_labels(self):
        labels = ["hall/img001", "hall/img002", "stairs/img001", "flat_label"]

        counts = relevant_counts_by_location(labels)

        self.assertEqual(counts["hall"], 2)
        self.assertEqual(counts["stairs"], 1)
        self.assertEqual(counts["flat_label"], 1)


if __name__ == "__main__":
    unittest.main()
