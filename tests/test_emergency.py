"""Emergency stop tests use only mocked key state and input packets."""
import threading
import time
import unittest
from unittest.mock import Mock, patch

from omnigamer import controls
from omnigamer.agent import OmniAgent
from omnigamer.emergency import EmergencyHotkey


class EmergencyHotkeyTests(unittest.TestCase):
    def test_global_hotkey_triggers_once_per_press(self):
        pressed = [False]
        callback = Mock()
        hotkey = EmergencyHotkey("F12", callback, key_state=lambda: pressed[0])

        self.assertFalse(hotkey.poll_once())
        pressed[0] = True
        self.assertTrue(hotkey.poll_once())
        self.assertFalse(hotkey.poll_once())
        pressed[0] = False
        self.assertFalse(hotkey.poll_once())
        pressed[0] = True
        self.assertTrue(hotkey.poll_once())
        self.assertEqual(callback.call_count, 2)

    def test_request_stop_latches_flag_and_releases_held_input(self):
        agent = OmniAgent.__new__(OmniAgent)
        agent._stop_event = threading.Event()
        agent._stop_reason = None
        agent.is_running = True
        agent.controls = Mock()
        agent.controls.release_all.return_value = []
        agent.last_emergency_release_error = ""

        agent.request_stop()

        self.assertTrue(agent._stop_event.is_set())
        self.assertFalse(agent.is_running)
        agent.controls.release_all.assert_called_once_with()
        with self.assertRaises(InterruptedError):
            agent._check_stop()

    def test_key_state_failure_requests_stop_and_records_failure(self):
        stopped = threading.Event()
        on_error = Mock(side_effect=lambda error: stopped.set())
        hotkey = EmergencyHotkey("F12", Mock(), key_state=Mock(side_effect=OSError("access denied")),
                                 poll_interval=0.001, on_error=on_error)

        with self.assertLogs("omnigamer.emergency", level="ERROR"):
            hotkey.start()
            self.assertTrue(stopped.wait(1))
            hotkey._thread.join(1)

        self.assertFalse(hotkey._thread.is_alive())
        self.assertIn("access denied", hotkey.last_error)
        self.assertEqual(on_error.call_count, 1)
        hotkey.callback.assert_not_called()

    def test_monitor_failure_stops_agent_with_distinct_reason(self):
        agent = OmniAgent.__new__(OmniAgent)
        agent._stop_event = threading.Event()
        agent._stop_reason = None
        agent.is_running = True
        agent.controls = Mock()
        agent.controls.release_all.return_value = []
        agent.last_emergency_release_error = ""
        agent.last_emergency_hotkey_error = ""
        agent.last_error = ""

        agent._emergency_hotkey_failed(RuntimeError("poll failed"))

        self.assertTrue(agent._stop_event.is_set())
        self.assertEqual(agent._stop_reason, "emergency_hotkey_error")
        self.assertIn("poll failed", agent.last_emergency_hotkey_error)
        agent.controls.release_all.assert_called_once_with()

    def test_release_all_sends_up_for_held_key_without_real_input(self):
        controller = controls.HumanInputController(failsafe_corner=False)
        down = controls.INPUT(type=controls.INPUT_KEYBOARD)
        down.union.ki = controls.KEYBDINPUT(0, controls.SCAN_CODES["w"], controls.KEYEVENTF_SCANCODE, 0, None)
        up = controls.INPUT(type=controls.INPUT_KEYBOARD)
        up.union.ki = controls.KEYBDINPUT(0, controls.SCAN_CODES["w"],
                                          controls.KEYEVENTF_SCANCODE | controls.KEYEVENTF_KEYUP, 0, None)
        mouse_down = controls.INPUT(type=controls.INPUT_MOUSE)
        mouse_down.union.mi = controls.MOUSEINPUT(0, 0, 0, controls.MOUSEEVENTF_LEFTDOWN, 0, None)
        mouse_up = controls.INPUT(type=controls.INPUT_MOUSE)
        mouse_up.union.mi = controls.MOUSEINPUT(0, 0, 0, controls.MOUSEEVENTF_LEFTUP, 0, None)
        held_key = controller._pressed(down, up)
        held_mouse = controller._pressed(mouse_down, mouse_up)
        with patch.object(controller, "_send") as send:
            held_key.__enter__()
            held_mouse.__enter__()
            self.assertEqual(controller.release_all(), [])
            held_mouse.__exit__(None, None, None)
            held_key.__exit__(None, None, None)
        packets = [call.args[0] for call in send.call_args_list]
        self.assertEqual([packet.union.ki.dwFlags for packet in packets[:1]], [controls.KEYEVENTF_SCANCODE])
        self.assertEqual([packet.union.mi.dwFlags for packet in packets[1:2]], [controls.MOUSEEVENTF_LEFTDOWN])
        self.assertEqual([packet.union.mi.dwFlags for packet in packets[2:3]], [controls.MOUSEEVENTF_LEFTUP])
        self.assertEqual([packet.union.ki.dwFlags for packet in packets[3:]],
                         [controls.KEYEVENTF_SCANCODE | controls.KEYEVENTF_KEYUP])
        self.assertEqual(controller._active_releases, [])

    def test_stop_interrupts_wait_for_blocked_vlm_request(self):
        agent = OmniAgent.__new__(OmniAgent)
        agent._stop_event = threading.Event()
        agent._stop_reason = None
        started = threading.Event()

        def blocked_request():
            started.set()
            threading.Event().wait(2)

        timer = threading.Thread(target=lambda: (started.wait(1), agent._stop_event.set()), daemon=True)
        timer.start()
        begin = time.perf_counter()
        with self.assertRaises(InterruptedError):
            agent._call_interruptibly(blocked_request)
        self.assertLess(time.perf_counter() - begin, 0.5)


if __name__ == "__main__":
    unittest.main()
