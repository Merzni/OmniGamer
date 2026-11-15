"""Execution regressions with mocked Windows/MSS; never send actual input."""
import unittest
from unittest.mock import Mock, patch

from omnigamer.capture import WindowCapture
from omnigamer import capture, controls


class CaptureTests(unittest.TestCase):
    def test_cannot_silently_continue_when_foreground_change_is_denied(self):
        with patch.object(capture, "user32") as windows, patch.object(capture.time, "sleep"):
            windows.IsWindow.return_value = True
            windows.IsIconic.return_value = False
            windows.GetForegroundWindow.return_value = 999
            with self.assertRaisesRegex(RuntimeError, "lost focus"):
                WindowCapture(hwnd=123).focus()

    def test_invalid_target_does_not_fall_back_to_capturing_other_apps(self):
        with patch.object(capture, "user32") as windows, patch.object(capture.mss, "mss") as mss:
            windows.IsWindow.return_value = False
            target = WindowCapture(hwnd=123)
            with self.assertRaisesRegex(RuntimeError, "no longer exists"):
                target.grab_frame()
            mss.assert_not_called()

    def test_monitor_fallback_caches_coordinate_rect(self):
        with patch.object(capture.mss, "mss") as mss:
            mss.return_value.__enter__.return_value.monitors = [None, dict(left=100, top=50, width=800, height=600)]
            target = WindowCapture()
            self.assertEqual(target.norm_to_screen(500, 500), (500, 350))
            self.assertEqual(target.last_rect, (100, 50, 900, 650))

    def test_each_frame_creates_and_closes_its_own_mss_context(self):
        target = WindowCapture(hwnd=123)
        with patch.object(target, "get_client_rect", return_value=(0, 0, 2, 2)), \
                patch.object(capture.mss, "mss") as mss:
            raw = mss.return_value.__enter__.return_value.grab.return_value
            raw.size, raw.bgra = (2, 2), bytes(16)
            self.assertEqual(target.grab_frame().size, (2, 2))
            target.grab_frame()
            self.assertEqual(mss.call_count, 2)
            self.assertEqual(mss.return_value.__exit__.call_count, 2)


class InputTests(unittest.TestCase):
    def test_jump_chord_holds_both_keys_and_releases_in_reverse_order(self):
        controller = controls.HumanInputController(failsafe_corner=False)
        events = []
        def sent(packet):
            events.append((packet.union.ki.wScan, packet.union.ki.dwFlags))
        def held(duration):
            self.assertEqual(len(events), 2)  # Both down before time advances.
            self.assertEqual(duration, 0.6)
        with patch.object(controller, "_send", side_effect=sent), patch.object(controller, "_hold_for", side_effect=held):
            controller.press_keys(["w", "space"], 0.6)
        self.assertEqual(events, [(0x11, 8), (0x39, 8), (0x39, 10), (0x11, 10)])

    def test_chord_releases_all_keys_after_interruption(self):
        controller = controls.HumanInputController(failsafe_corner=False)
        with patch.object(controller, "_send") as send, \
                patch.object(controller, "_hold_for", side_effect=InterruptedError("stop")):
            with self.assertRaises(InterruptedError):
                controller.press_keys(["w", "space"])
        self.assertEqual([c.args[0].union.ki.dwFlags for c in send.call_args_list], [8, 8, 10, 10])

    def test_camera_preserves_relative_signed_delta(self):
        controller = controls.HumanInputController(failsafe_corner=False)
        with patch.object(controller, "_send") as send, patch.object(controls.time, "sleep"):
            controller.move_camera(-123, 31)
        packets = [call.args[0].union.mi for call in send.call_args_list]
        self.assertEqual(sum(p.dx for p in packets), -123)
        self.assertEqual(sum(p.dy for p in packets), 31)
        self.assertTrue(all(p.dwFlags == controls.MOUSEEVENTF_MOVE for p in packets))

    def test_sendinput_failure_is_reported(self):
        with patch.object(controls, "user32") as windows:
            windows.SendInput.return_value = 0
            controller = controls.HumanInputController(failsafe_corner=False)
            with self.assertRaisesRegex(RuntimeError, "SendInput rejected"):
                controller._send(controls.INPUT())

    def test_mouse_button_released_when_focus_is_lost_mid_hold(self):
        target_check = Mock(side_effect=[None, None, RuntimeError("focus lost")])
        controller = controls.HumanInputController(failsafe_corner=False, target_check=target_check)
        with patch.object(controller, "_send") as send, patch.object(controls.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "focus lost"):
                controller.hold_mouse(duration=1)
            flags = [call.args[0].union.mi.dwFlags for call in send.call_args_list]
            self.assertEqual(flags, [controls.MOUSEEVENTF_LEFTDOWN, controls.MOUSEEVENTF_LEFTUP])

    def test_key_released_when_focus_is_lost_mid_hold(self):
        target_check = Mock(side_effect=[None, None, RuntimeError("focus lost")])
        controller = controls.HumanInputController(failsafe_corner=False, target_check=target_check)
        with patch.object(controller, "_send") as send, patch.object(controls.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "focus lost"):
                controller.press_key("w", duration=1)
            flags = [call.args[0].union.ki.dwFlags for call in send.call_args_list]
            self.assertEqual(flags, [controls.KEYEVENTF_SCANCODE, controls.KEYEVENTF_SCANCODE | controls.KEYEVENTF_KEYUP])

    def test_unknown_key_is_not_silently_counted_as_an_action(self):
        controller = controls.HumanInputController(failsafe_corner=False)
        with patch.object(controller, "_send") as send:
            with self.assertRaisesRegex(ValueError, "Unsupported key"):
                controller.press_key("teleport")
            send.assert_not_called()


if __name__ == "__main__":
    unittest.main()
