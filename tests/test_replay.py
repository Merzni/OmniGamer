import tempfile
import unittest
from pathlib import Path

from PIL import Image

from omnigamer.replay import FrameDirectoryCapture


class FrameReplayTests(unittest.TestCase):
    def test_replays_sorted_frames_and_repeats_last_without_mutating_files(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            Image.new("RGB", (4, 3), "red").save(directory / "02.png")
            Image.new("RGB", (4, 3), "blue").save(directory / "01.png")
            capture = FrameDirectoryCapture(directory)
            colors = [capture.grab_frame().getpixel((0, 0)) for _ in range(3)]
            self.assertEqual(colors[0], (0, 0, 255))
            self.assertEqual(colors[1], (255, 0, 0))
            self.assertEqual(colors[2], (255, 0, 0))
            self.assertEqual(len(list(directory.glob("*.png"))), 2)

    def test_empty_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                FrameDirectoryCapture(temp)


if __name__ == "__main__":
    unittest.main()
