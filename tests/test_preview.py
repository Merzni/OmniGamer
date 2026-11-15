"""Fresh preview must neither wait for inference nor focus/control the desktop."""
import unittest
from unittest.mock import Mock, patch
from PIL import Image
from fastapi import HTTPException
from omnigamer.dashboard import app as dashboard


class PreviewTests(unittest.TestCase):
    def test_virtual_preview_marks_bot_position_without_focus_or_mutating_capture(self):
        import asyncio,io
        agent=Mock()
        agent.capture.hwnd=123
        agent.input_backend="window"
        agent.pointer.snapshot.return_value={"visible":True,"x":500,"y":500,"pressed":False}
        frame=Image.new("RGB",(800,500))
        with patch.object(dashboard,"agent",agent),patch.object(dashboard,"WindowCapture") as capture:
            capture.return_value.assert_visible=Mock()
            capture.return_value.grab_frame.return_value=frame
            response=dashboard.get_live_frame()
            async def body():
                return b''.join([part async for part in response.body_iterator])
            rendered=Image.open(io.BytesIO(asyncio.run(body())))
            self.assertGreater(rendered.getpixel((400,250))[0],100)
            self.assertEqual(frame.getpixel((400,250)),(0,0,0))
            capture.return_value.assert_visible.assert_called()
            capture.return_value.focus.assert_not_called()
            agent.step.assert_not_called()

    def test_preview_captures_fresh_image_without_agent_step_or_focus(self):
        agent = Mock()
        agent.capture.hwnd = 123
        agent.pointer.snapshot.return_value={"visible":False}
        with patch.object(dashboard, "agent", agent), patch.object(dashboard, "WindowCapture") as capture:
            capture.return_value.assert_foreground = Mock()
            capture.return_value.grab_frame.return_value = Image.new("RGB", (1600, 900))
            response = dashboard.get_live_frame()
            self.assertEqual(response.media_type, "image/jpeg")
            self.assertEqual(response.headers["cache-control"], "no-store")
            capture.assert_called_once_with(hwnd=123)
            capture.return_value.focus.assert_not_called()
            capture.return_value.grab_frame.assert_called_once()
            agent.step.assert_not_called()
            agent.capture.grab_frame.assert_not_called()

    def test_preview_does_not_display_an_unfocused_target_as_live(self):
        agent = Mock()
        agent.capture.hwnd = 123
        with patch.object(dashboard, "agent", agent), patch.object(dashboard, "WindowCapture") as capture:
            capture.return_value.assert_foreground = Mock(side_effect=RuntimeError("lost focus"))
            with self.assertRaises(HTTPException) as error:
                dashboard.get_live_frame()
            self.assertEqual(error.exception.status_code, 409)
            capture.return_value.grab_frame.assert_not_called()

    def test_preview_requires_an_explicit_target_window(self):
        agent = Mock()
        agent.capture.hwnd = 0
        with patch.object(dashboard, "agent", agent):
            with self.assertRaises(HTTPException):
                dashboard.get_live_frame()
