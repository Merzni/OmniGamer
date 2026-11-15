"""Addressed input must never move the user's system cursor or focus."""
import unittest
from unittest.mock import patch
from omnigamer.window_input import WindowInputController
from omnigamer.pointer import BotPointer
from omnigamer import controls
import win32con


class WindowInputTests(unittest.TestCase):
    def setUp(self):
        self.pointer=BotPointer(desktop=False)
        self.controller=WindowInputController(lambda:123,lambda:(100,200,900,800),self.pointer,failsafe_corner=False)

    def test_click_uses_client_messages_and_separate_pointer(self):
        with patch('omnigamer.window_input.win32gui.IsWindow',return_value=True), \
             patch('omnigamer.window_input.win32gui.PostMessage') as post, \
             patch.object(controls,'user32') as system,patch.object(controls.time,'sleep'):
            self.controller.click(500,500)
        system.SetCursorPos.assert_not_called();system.SendInput.assert_not_called();system.SetForegroundWindow.assert_not_called()
        downs=[c for c in post.call_args_list if c.args[1]==win32con.WM_LBUTTONDOWN]
        self.assertEqual(len(downs),1)
        self.assertEqual(downs[0].args[3],400|(300<<16))
        pointer=self.pointer.snapshot()
        self.assertEqual((pointer['screen_x'],pointer['screen_y']),(500,500))
        self.assertFalse(pointer['pressed']);self.assertEqual(pointer['event'],'up')

    def test_message_failure_does_not_fall_back_to_shared_input(self):
        with patch('omnigamer.window_input.win32gui.IsWindow',return_value=True), \
             patch('omnigamer.window_input.win32gui.PostMessage',side_effect=RuntimeError('rejected')), \
             patch.object(controls,'user32') as system:
            with self.assertRaisesRegex(RuntimeError,'rejected'):self.controller.set_cursor_pos(500,500)
        system.SetCursorPos.assert_not_called();system.SendInput.assert_not_called()

    def test_keyboard_chord_is_addressed_to_target_and_all_keys_release(self):
        with patch('omnigamer.window_input.win32gui.IsWindow',return_value=True), \
             patch('omnigamer.window_input.win32gui.PostMessage') as post, \
             patch('omnigamer.window_input.win32api.MapVirtualKey',side_effect=lambda scan,mode:{0x11:0x57,0x39:0x20}[scan]), \
             patch.object(controls,'user32') as system,patch.object(controls.time,'sleep'):
            self.controller.press_keys(['w','space'],.1)
        self.assertEqual([c.args[1] for c in post.call_args_list],[256,256,257,257])
        self.assertTrue(all(c.args[0]==123 for c in post.call_args_list))
        self.assertTrue(post.call_args_list[-1].args[3]&(1<<31))
        system.SendInput.assert_not_called()

    def test_unsupported_camera_is_explicit_without_fake_marker_motion_or_shared_input(self):
        with patch('omnigamer.window_input.win32gui.IsWindow',return_value=True), \
             patch('omnigamer.window_input.win32gui.PostMessage') as post, \
             patch.object(controls,'user32') as system,patch.object(controls.time,'sleep'):
            with self.assertRaisesRegex(RuntimeError,'native'):
                self.controller.move_camera(120,-30)
        post.assert_not_called()
        system.SetCursorPos.assert_not_called();system.SendInput.assert_not_called()

    def test_pointer_state_and_hide_do_not_need_a_desktop_window(self):
        self.pointer.update(300,400,(100,200,900,800),'down',True)
        self.assertEqual((self.pointer.snapshot()['x'],self.pointer.snapshot()['y']),(250,333))
        self.pointer.hide();self.assertFalse(self.pointer.snapshot()['visible'])


if __name__=='__main__':unittest.main()
