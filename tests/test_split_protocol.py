import unittest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from split_protocol import exclude_indexed_queries, select_database_records, select_synthetic_query_records


class SplitProtocolTests(unittest.TestCase):
    def test_database_records_default_to_original_images_only(self):
        records = [
            {"image_id": "hall/img001", "augmented": False, "aug_type": "original"},
            {"image_id": "hall/img001_aug_blur", "augmented": True, "aug_type": "aug_blur"},
        ]

        selected = select_database_records(records)

        self.assertEqual([r["image_id"] for r in selected], ["hall/img001"])

    def test_synthetic_queries_are_selected_by_augmentation_type(self):
        records = [
            {"image_id": "hall/img001", "augmented": False, "aug_type": "original"},
            {"image_id": "hall/img001_aug_blur", "augmented": True, "aug_type": "aug_blur"},
            {"image_id": "hall/img001_aug_warm", "augmented": True, "aug_type": "aug_warm"},
        ]

        selected = select_synthetic_query_records(records, ["aug_blur"])

        self.assertEqual([r["image_id"] for r in selected], ["hall/img001_aug_blur"])

    def test_indexed_queries_are_removed_from_evaluation(self):
        queries = [
            (Path("data/processed/hall/img001_aug_blur.jpg"), "hall", "hall/img001_aug_blur"),
            (Path("data/processed/hall/img002_aug_blur.jpg"), "hall", "hall/img002_aug_blur"),
        ]

        selected = exclude_indexed_queries(queries, {"hall/img001_aug_blur"})

        self.assertEqual(selected, [(Path("data/processed/hall/img002_aug_blur.jpg"), "hall", "hall/img002_aug_blur")])


if __name__ == "__main__":
    unittest.main()
