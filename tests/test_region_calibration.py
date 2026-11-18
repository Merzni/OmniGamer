import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from profile_fixtures import load_profile
from tools.calibrate_region_metric import choose_threshold, load_samples, write_threshold


class RegionCalibrationTests(unittest.TestCase):
    def test_separable_threshold_has_zero_synthetic_false_positives_and_misses(self):
        result = choose_threshold([
            ("no_change", 0.0), ("no_change", 0.01),
            ("target_change", 0.2), ("target_change", 0.6),
        ])
        self.assertTrue(result["separable"])
        self.assertEqual(result["false_positives"], 0)
        self.assertEqual(result["false_negatives"], 0)
        self.assertGreater(result["threshold_margin_over_no_change"], 0)

    def test_overlapping_labels_are_reported_instead_of_hidden(self):
        result = choose_threshold([
            ("no_change", 0.1), ("no_change", 0.3),
            ("target_change", 0.2), ("target_change", 0.25),
        ])
        self.assertFalse(result["separable"])
        self.assertGreaterEqual(result["false_negatives"], 1)

    def test_manifest_uses_profile_region_and_saved_pairs(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            before = Image.new("RGB", (20, 20), "black")
            after = before.copy()
            ImageDraw.Draw(after).rectangle((5, 17, 14, 19), fill="white")
            before.save(base / "before.png")
            after.save(base / "after.png")
            (base / "pairs.csv").write_text(
                "sample_id,before,after,class\npositive,before.png,after.png,target_change\n",
                encoding="utf-8")
            profile = load_profile("minecraft")
            rows = load_samples(base / "pairs.csv", profile.regions["hotbar"])
            self.assertEqual(rows[0]["class"], "target_change")
            self.assertGreater(rows[0]["delta"], 0.08)

    def test_profile_threshold_write_validates_and_updates_only_requested_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "profile.json"
            profile = load_profile("calibration-test")
            write_threshold(path, profile, 0.123)
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["thresholds"]["region_change"], 0.123)
            self.assertEqual(profile.thresholds, {})


if __name__ == "__main__":
    unittest.main()
