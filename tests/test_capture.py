"""Window capture fallback regressions use fake pixels and never touch the desktop."""
import unittest
from unittest.mock import MagicMock, Mock, patch

from PIL import Image

from omnigamer.capture import WindowCapture


class WindowCaptureTests(unittest.TestCase):
    def test_window_capture_falls_back_when_mss_bitblt_fails(self):
        mss_context = MagicMock()
        mss_context.__enter__.return_value.grab.side_effect = OSError("BitBlt failed")
        image = Image.new("RGB", (10, 8), "blue")
        capture = WindowCapture(hwnd=123)
        capture.get_client_rect = Mock(return_value=(0, 0, 10, 8))

        with patch("omnigamer.capture.mss.mss", return_value=mss_context), \
                patch("omnigamer.capture.ImageGrab.grab", return_value=image) as grab_window:
            result = capture.grab_frame()

        self.assertEqual(result.size, (10, 8))
        self.assertEqual(result.getpixel((0, 0)), (0, 0, 255))
        grab_window.assert_called_once_with(window=123)

    def test_monitor_capture_does_not_use_window_fallback(self):
        mss_context = MagicMock()
        mss_context.__enter__.return_value.grab.side_effect = OSError("BitBlt failed")
        capture = WindowCapture(hwnd=0)
        capture.get_client_rect = Mock(return_value=(0, 0, 10, 8))

        with patch("omnigamer.capture.mss.mss", return_value=mss_context), \
                patch("omnigamer.capture.ImageGrab.grab") as grab_window:
            with self.assertRaisesRegex(OSError, "BitBlt failed"):
                capture.grab_frame()

        grab_window.assert_not_called()

    def test_window_capture_reports_both_failure_causes(self):
        mss_context = MagicMock()
        mss_context.__enter__.return_value.grab.side_effect = OSError("BitBlt failed")
        capture = WindowCapture(hwnd=123)
        capture.get_client_rect = Mock(return_value=(0, 0, 10, 8))

        with patch("omnigamer.capture.mss.mss", return_value=mss_context), \
                patch("omnigamer.capture.ImageGrab.grab", side_effect=OSError("PrintWindow failed")):
            with self.assertRaisesRegex(RuntimeError, "BitBlt.*PrintWindow failed"):
                capture.grab_frame()


if __name__ == "__main__":
    unittest.main()
