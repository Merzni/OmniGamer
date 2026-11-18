"""Hardware-level input controller with human-like motion, 3D mouse look, and failsafe guards."""
import ctypes
from ctypes import wintypes
import time
import math
import random
from threading import Lock
from contextlib import contextmanager, ExitStack
from typing import Tuple, Optional

user32 = ctypes.windll.user32

# Input constants for SendInput
INPUT_MOUSE = 0
INPUT_KEYBOARD = 1

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_ABSOLUTE = 0x8000

KEYEVENTF_KEYDOWN = 0x0000
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG))
    ]

class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG))
    ]

class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD)
    ]

class _INPUT_UNION(ctypes.Union):
    _fields_ = [
        ("mi", MOUSEINPUT),
        ("ki", KEYBDINPUT),
        ("hi", HARDWAREINPUT)
    ]

class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", wintypes.DWORD),
        ("union", _INPUT_UNION)
    ]

# Hardware scan codes (DirectInput/DirectX/OpenGL/LWJGL compatible)
SCAN_CODES = {
    "esc": 0x01, "escape": 0x01, "1": 0x02, "2": 0x03, "3": 0x04,
    "4": 0x05, "5": 0x06, "6": 0x07, "7": 0x08, "8": 0x09,
    "9": 0x0A, "0": 0x0B, "minus": 0x0C, "equals": 0x0D,
    "backspace": 0x0E, "tab": 0x0F,
    "q": 0x10, "w": 0x11, "e": 0x12, "r": 0x13, "t": 0x14,
    "y": 0x15, "u": 0x16, "i": 0x17, "o": 0x18, "p": 0x19,
    "left_bracket": 0x1A, "right_bracket": 0x1B, "enter": 0x1C, "ctrl": 0x1D,
    "a": 0x1E, "s": 0x1F, "d": 0x20, "f": 0x21, "g": 0x22,
    "h": 0x23, "j": 0x24, "k": 0x25, "l": 0x26, "semicolon": 0x27,
    "apostrophe": 0x28, "grave": 0x29, "shift": 0x2A, "backslash": 0x2B,
    "z": 0x2C, "x": 0x2D, "c": 0x2E, "v": 0x2F, "b": 0x30,
    "n": 0x31, "m": 0x32, "comma": 0x33, "period": 0x34,
    "slash": 0x35, "right_shift": 0x36, "keypad_multiply": 0x37,
    "alt": 0x38, "space": 0x39, "capslock": 0x3A,
    "f1": 0x3B, "f2": 0x3C, "f3": 0x3D, "f4": 0x3E,
    "f5": 0x3F, "f6": 0x40, "f7": 0x41, "f8": 0x42,
    "f9": 0x43, "f10": 0x44, "numlock": 0x45, "scrolllock": 0x46,
    "f11": 0x57, "f12": 0x58,
}

# Virtual key fallback
VK_MAP = {
    "esc": 0x1B, "escape": 0x1B,
    "enter": 0x0D, "return": 0x0D,
    "space": 0x20,
    "w": 0x57, "a": 0x41, "s": 0x53, "d": 0x44,
    "e": 0x45, "q": 0x51, "r": 0x52, "f": 0x46, "c": 0x43,
    "1": 0x31, "2": 0x32, "3": 0x33, "4": 0x34, "5": 0x35,
    "tab": 0x09, "shift": 0x10, "ctrl": 0x11, "alt": 0x12,
    "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
}

class HumanInputController:
    """Controls mouse and keyboard with safety checks, 3D camera look, and human-like natural movement."""

    def __init__(self, failsafe_corner: bool = True, target_check=None):
        self.failsafe_corner = failsafe_corner
        self.aborted = False
        self.target_check = target_check
        self.emergency_check = None
        self._pressed_lock = Lock()
        self._active_releases = []

    def _send(self, inp: INPUT):
        if user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT)) != 1:
            raise RuntimeError("SendInput rejected an input event; check desktop and game privilege levels")

    @contextmanager
    def _pressed(self, down: INPUT, up: INPUT):
        self.check_failsafe()
        token = object()
        with self._pressed_lock:
            self._send(down)
            self._active_releases.append((token, up))
        try:
            yield
        finally:
            # An emergency thread may already have released this packet.
            with self._pressed_lock:
                for index, (active_token, _) in enumerate(self._active_releases):
                    if active_token is token:
                        self._send(up)
                        self._active_releases.pop(index)
                        break

    def release_all(self):
        """Release only synthetic inputs currently held by this controller."""
        with self._pressed_lock:
            active = list(reversed(self._active_releases))
            self._active_releases.clear()
        errors = []
        for token, packet in active:
            try:
                self._send(packet)
            except Exception as exc:
                errors.append(exc)
                with self._pressed_lock:
                    if not any(active_token is token for active_token, _ in self._active_releases):
                        self._active_releases.append((token, packet))
        return errors

    def _hold_for(self, duration: float):
        elapsed = 0.0
        while elapsed < duration:
            chunk = min(0.1, duration - elapsed)
            time.sleep(chunk)
            elapsed += chunk
            self.check_failsafe()

    def check_failsafe(self):
        """Check if user moved cursor to screen corner (0,0) to instantly abort."""
        if self.emergency_check:
            self.emergency_check()
        if self.aborted:
            raise InterruptedError("Failsafe is latched; restart or reattach the agent")
        if self.target_check:
            self.target_check()
        if not self.failsafe_corner:
            return
        pt = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(pt))
        if pt.x <= 5 and pt.y <= 5:
            self.aborted = True
            raise InterruptedError("EMERGENCY FAILSAFE: Cursor moved to top-left corner (0,0). Halting.")

    def get_cursor_pos(self) -> Tuple[int, int]:
        pt = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(pt))
        return pt.x, pt.y

    def set_cursor_pos(self, x: int, y: int):
        if not user32.SetCursorPos(x, y):
            raise RuntimeError("SetCursorPos failed")

    def move_to(self, target_x: int, target_y: int, steps: int = 12, natural: bool = True):
        """Move cursor smoothly towards target with human micro-jitter (for 2D menus)."""
        self.check_failsafe()
        cur_x, cur_y = self.get_cursor_pos()
        if not natural or steps <= 1:
            self.set_cursor_pos(target_x, target_y)
            return

        for step in range(1, steps + 1):
            t = step / steps
            ease_t = 0.5 - 0.5 * math.cos(t * math.pi)
            jitter_x = random.randint(-1, 1) if step < steps else 0
            jitter_y = random.randint(-1, 1) if step < steps else 0
            ix = int(cur_x + (target_x - cur_x) * ease_t + jitter_x)
            iy = int(cur_y + (target_y - cur_y) * ease_t + jitter_y)
            self.set_cursor_pos(ix, iy)
            time.sleep(0.008 + random.uniform(0.0, 0.005))
            self.check_failsafe()

    def move_camera(self, dx: int, dy: int, steps: int = 6):
        """Rotate a camera using relative mouse delta packets."""
        self.check_failsafe()
        step_dx = dx // steps
        step_dy = dy // steps
        remainder_x = dx % steps
        remainder_y = dy % steps

        for i in range(steps):
            cur_dx = step_dx + (remainder_x if i == steps - 1 else 0)
            cur_dy = step_dy + (remainder_y if i == steps - 1 else 0)
            
            inp = INPUT(type=INPUT_MOUSE)
            inp.union.mi = MOUSEINPUT(cur_dx, cur_dy, 0, MOUSEEVENTF_MOVE, 0, None)
            self._send(inp)
            time.sleep(0.012)
            self.check_failsafe()

    def click(self, x: Optional[int] = None, y: Optional[int] = None, button: str = "left", hold_time: float = 0.08):
        """Perform a natural click at (x, y) or current pos."""
        if x is not None and y is not None:
            self.move_to(x, y)
        self.check_failsafe()
        
        down_flag = MOUSEEVENTF_LEFTDOWN if button == "left" else MOUSEEVENTF_RIGHTDOWN
        up_flag = MOUSEEVENTF_LEFTUP if button == "left" else MOUSEEVENTF_RIGHTUP
        
        inp_down = INPUT(type=INPUT_MOUSE)
        inp_down.union.mi = MOUSEINPUT(0, 0, 0, down_flag, 0, None)
        inp_up = INPUT(type=INPUT_MOUSE)
        inp_up.union.mi = MOUSEINPUT(0, 0, 0, up_flag, 0, None)
        with self._pressed(inp_down, inp_up):
            self._hold_for(hold_time + random.uniform(0.01, 0.03))

    def double_click(self, x: Optional[int] = None, y: Optional[int] = None):
        """Perform a rapid double click."""
        self.click(x, y, hold_time=0.04)
        time.sleep(random.uniform(0.08, 0.12))
        self.click(hold_time=0.04)

    def hold_mouse(self, button: str = "left", duration: float = 1.5):
        """Hold a mouse button continuously."""
        self.check_failsafe()
        down_flag = MOUSEEVENTF_LEFTDOWN if button == "left" else MOUSEEVENTF_RIGHTDOWN
        up_flag = MOUSEEVENTF_LEFTUP if button == "left" else MOUSEEVENTF_RIGHTUP

        inp_down = INPUT(type=INPUT_MOUSE)
        inp_down.union.mi = MOUSEINPUT(0, 0, 0, down_flag, 0, None)
        inp_up = INPUT(type=INPUT_MOUSE)
        inp_up.union.mi = MOUSEINPUT(0, 0, 0, up_flag, 0, None)
        with self._pressed(inp_down, inp_up):
            self._hold_for(duration)

    def drag(self, start_x: int, start_y: int, end_x: int, end_y: int, duration: float = 0.4):
        """Click and drag (swipe) from (start_x, start_y) to (end_x, end_y)."""
        self.move_to(start_x, start_y)
        self.check_failsafe()
        
        inp_down = INPUT(type=INPUT_MOUSE)
        inp_down.union.mi = MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTDOWN, 0, None)
        inp_up = INPUT(type=INPUT_MOUSE)
        inp_up.union.mi = MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTUP, 0, None)
        with self._pressed(inp_down, inp_up):
            self._hold_for(0.05)
            steps = int(max(10, duration * 60))
            for step in range(1, steps + 1):
                t = step / steps
                ease_t = 0.5 - 0.5 * math.cos(t * math.pi)
                self.check_failsafe()
                self.set_cursor_pos(int(start_x + (end_x - start_x) * ease_t),
                                    int(start_y + (end_y - start_y) * ease_t))
                self._hold_for(duration / steps)
            self._hold_for(0.05)

    def press_key(self, key_name: str, duration: float = 0.1):
        """Simulate a hardware scan-code keyboard press."""
        self.check_failsafe()
        k = key_name.lower().strip()
        scan = SCAN_CODES.get(k)
        vk = VK_MAP.get(k)

        if scan is not None:
            # Send hardware scancode packet
            inp_down = INPUT(type=INPUT_KEYBOARD)
            inp_down.union.ki = KEYBDINPUT(wVk=0, wScan=scan, dwFlags=KEYEVENTF_SCANCODE, time=0, dwExtraInfo=None)
            inp_up = INPUT(type=INPUT_KEYBOARD)
            inp_up.union.ki = KEYBDINPUT(wVk=0, wScan=scan, dwFlags=KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP, time=0, dwExtraInfo=None)
            with self._pressed(inp_down, inp_up):
                self._hold_for(duration)
        elif vk is not None:
            inp_down = INPUT(type=INPUT_KEYBOARD)
            inp_down.union.ki = KEYBDINPUT(wVk=vk, wScan=0, dwFlags=KEYEVENTF_KEYDOWN, time=0, dwExtraInfo=None)
            inp_up = INPUT(type=INPUT_KEYBOARD)
            inp_up.union.ki = KEYBDINPUT(wVk=vk, wScan=0, dwFlags=KEYEVENTF_KEYUP, time=0, dwExtraInfo=None)
            with self._pressed(inp_down, inp_up):
                self._hold_for(duration)
        else:
            raise ValueError(f"Unsupported key: {key_name}")

    def press_keys(self, key_names, duration: float = 0.6):
        """Hold a chord together, releasing every pressed key on interruption."""
        keys = list(dict.fromkeys(name.lower().strip() for name in key_names))
        packets = []
        # Validate everything before pressing even the first key.
        for key in keys:
            scan, vk = SCAN_CODES.get(key), VK_MAP.get(key)
            if scan is None and vk is None:
                raise ValueError(f"Unsupported key: {key}")
            flags = KEYEVENTF_SCANCODE if scan is not None else 0
            down, up = INPUT(type=INPUT_KEYBOARD), INPUT(type=INPUT_KEYBOARD)
            vk_code = (vk or 0) if scan is None else 0
            down.union.ki = KEYBDINPUT(vk_code, scan or 0, flags, 0, None)
            up.union.ki = KEYBDINPUT(vk_code, scan or 0, flags | KEYEVENTF_KEYUP, 0, None)
            packets.append((down, up))
        with ExitStack() as stack:
            for down, up in packets:
                stack.enter_context(self._pressed(down, up))
            self._hold_for(duration)
