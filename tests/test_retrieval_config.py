import unittest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retrieval_config import descriptor_filename, index_filename, label_filename, resolve_methods


class RetrievalConfigTests(unittest.TestCase):
    def test_resolve_methods_keeps_orb_in_all_user_workflows(self):
        self.assertEqual(resolve_methods("sift"), ["sift"])
        self.assertEqual(resolve_methods("orb"), ["orb"])
        self.assertEqual(resolve_methods("dinov2"), ["dinov2"])
        self.assertEqual(resolve_methods("anyloc"), ["anyloc"])
        self.assertEqual(resolve_methods("both"), ["sift", "dinov2"])
        self.assertEqual(resolve_methods("all"), ["sift", "orb", "dinov2", "anyloc"])

    def test_index_filename_preserves_flat_defaults_and_suffixes_scalable_indexes(self):
        self.assertEqual(index_filename("sift", "flat"), "sift_vlad.index")
        self.assertEqual(index_filename("orb", "flat"), "orb_bow.index")
        self.assertEqual(index_filename("dinov2", "flat"), "dinov2.index")
        self.assertEqual(index_filename("anyloc", "flat"), "anyloc_dinov2_vlad.index")
        self.assertEqual(index_filename("dinov2", "ivf"), "dinov2_ivf.index")
        self.assertEqual(index_filename("sift", "hnsw"), "sift_vlad_hnsw.index")
        self.assertEqual(index_filename("anyloc", "hnsw"), "anyloc_dinov2_vlad_hnsw.index")

    def test_descriptor_and_label_filenames_match_existing_artifacts(self):
        self.assertEqual(descriptor_filename("sift"), "sift_vlad_descriptors.npy")
        self.assertEqual(label_filename("sift"), "sift_vlad_labels.npy")
        self.assertEqual(descriptor_filename("orb"), "orb_descriptors.npy")
        self.assertEqual(label_filename("orb"), "orb_labels.npy")
        self.assertEqual(descriptor_filename("dinov2"), "dinov2_descriptors.npy")
        self.assertEqual(label_filename("dinov2"), "dinov2_labels.npy")
        self.assertEqual(descriptor_filename("anyloc"), "anyloc_dinov2_vlad_descriptors.npy")
        self.assertEqual(label_filename("anyloc"), "anyloc_dinov2_vlad_labels.npy")


if __name__ == "__main__":
    unittest.main()
