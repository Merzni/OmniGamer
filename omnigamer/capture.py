"""Window management and screen capture for OmniGamer."""
import ctypes
from ctypes import wintypes
import time
from typing import Optional, List, Dict, Tuple
from PIL import Image, ImageGrab
import mss
import win32gui
import win32service
import win32con

user32 = ctypes.windll.user32

class RECT(ctypes.Structure):
    _fields_ = [
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG)
    ]

class POINT(ctypes.Structure):
    _fields_ = [
        ("x", wintypes.LONG),
        ("y", wintypes.LONG)
    ]

def list_visible_windows() -> List[Dict]:
    """Find all top-level visible windows with non-empty titles across user desktop."""
    results = []
    seen_hwnds = set()

    def process_hwnd(hwnd):
        int_hwnd = int(hwnd)
        if int_hwnd in seen_hwnds:
            return
        seen_hwnds.add(int_hwnd)

        if win32gui.IsWindowVisible(int_hwnd):
            title = win32gui.GetWindowText(int_hwnd).strip()
            if title and not title.startswith("Default IME") and title != "Program Manager":
                rect = RECT()
                user32.GetWindowRect(int_hwnd, ctypes.byref(rect))
                w = rect.right - rect.left
                h = rect.bottom - rect.top
                if w > 100 and h > 100:
                    results.append({
                        "hwnd": int_hwnd,
                        "title": title,
                        "rect": (rect.left, rect.top, rect.right, rect.bottom),
                        "width": w,
                        "height": h
                    })

    # Try default interactive desktop first
    try:
        h_desk = win32service.OpenDesktop("default", 0, False, win32con.GENERIC_ALL)
        for h in h_desk.EnumDesktopWindows():
            process_hwnd(h)
    except Exception:
        pass

    # Fallback to standard EnumWindows
    if not results:
        def enum_cb(h, _):
            process_hwnd(h)
            return True
        win32gui.EnumWindows(enum_cb, None)

    return sorted(results, key=lambda x: x["title"].lower())

def find_window_by_title(query: str) -> Optional[Dict]:
    """Find the best matching window by substring (case-insensitive)."""
    q = query.lower()
    windows = list_visible_windows()
    for w in windows:
        if q in w["title"].lower():
            return w
    return None

class WindowCapture:
    """Handles continuous, fast screen capture of a target window."""
    
    def __init__(self, hwnd: int = 0):
        self.hwnd = hwnd
        self.last_rect: Optional[Tuple[int, int, int, int]] = None

    def get_window_title(self) -> str:
        return win32gui.GetWindowText(self.hwnd) if self.hwnd and user32.IsWindow(self.hwnd) else ""

    def focus(self):
        """Bring window to foreground if minimized or behind."""
        if self.hwnd:
            if not user32.IsWindow(self.hwnd):
                raise RuntimeError("Target window no longer exists; attach the game again")
            if user32.IsIconic(self.hwnd):
                user32.ShowWindow(self.hwnd, 9)  # SW_RESTORE
            if user32.GetForegroundWindow() != self.hwnd:
                user32.SetForegroundWindow(self.hwnd)
                time.sleep(0.1)
            self.assert_foreground()

    def assert_foreground(self):
        if self.hwnd and (not user32.IsWindow(self.hwnd) or user32.GetForegroundWindow() != self.hwnd):
            raise RuntimeError("Game window lost focus; action cancelled")

    def assert_visible(self):
        """Window-message mode never steals focus; require unobscured screen capture."""
        if not self.hwnd or not win32gui.IsWindow(self.hwnd):
            raise RuntimeError("Attach a valid game window for virtual input")
        if not win32gui.IsWindowVisible(self.hwnd) or user32.IsIconic(self.hwnd):
            raise RuntimeError("The game must remain visible and not minimized")
        left,top,right,bottom=self.get_client_rect()
        for fx,fy in ((.1,.1),(.5,.1),(.9,.1),(.1,.5),(.5,.5),(.9,.5),(.1,.9),(.5,.9),(.9,.9)):
            owner=win32gui.WindowFromPoint((int(left+(right-left)*fx),int(top+(bottom-top)*fy)))
            if owner==self.hwnd or win32gui.IsChild(self.hwnd,owner):continue
            if owner and win32gui.GetClassName(owner).startswith("OmniGamerBotPointer_"):continue
            raise RuntimeError("Game window is obscured; virtual mode will not focus it or capture another app")

    def get_client_rect(self) -> Tuple[int, int, int, int]:
        """Get absolute screen coordinates for the client area (excluding title bar borders)."""
        if self.hwnd and not user32.IsWindow(self.hwnd):
            raise RuntimeError("Target window no longer exists; attach the game again")
        if not self.hwnd:
            with mss.mss() as sct:
                mon = sct.monitors[1]
                self.last_rect = (mon["left"], mon["top"], mon["left"] + mon["width"], mon["top"] + mon["height"])
            return self.last_rect
        
        rect = RECT()
        user32.GetClientRect(self.hwnd, ctypes.byref(rect))
        
        top_left = POINT(rect.left, rect.top)
        bot_right = POINT(rect.right, rect.bottom)
        
        user32.ClientToScreen(self.hwnd, ctypes.byref(top_left))
        user32.ClientToScreen(self.hwnd, ctypes.byref(bot_right))
        
        self.last_rect = (top_left.x, top_left.y, bot_right.x, bot_right.y)
        if self.last_rect[2] <= self.last_rect[0] or self.last_rect[3] <= self.last_rect[1]:
            raise RuntimeError("Target window has no visible client area")
        return self.last_rect

    def grab_frame(self) -> Image.Image:
        """Capture current frame of the target window as a PIL Image RGB."""
        left, top, right, bottom = self.get_client_rect()
        width = right - left
        height = bottom - top
        
        bbox = {"left": left, "top": top, "width": width, "height": height}
        try:
            # MSS handles are thread-local; FastAPI may use a different worker each step.
            with mss.mss() as sct:
                raw = sct.grab(bbox)
            return Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
        except Exception as screen_error:
            if not self.hwnd:
                raise
            try:
                # Pillow uses a target-window capture path that can work when desktop
                # BitBlt is unavailable for an accelerated window.
                img = ImageGrab.grab(window=int(self.hwnd)).convert("RGB")
                if img.size != (width, height):
                    raise RuntimeError(
                        f"Window capture size mismatch: expected {width}x{height}, "
                        f"got {img.width}x{img.height}")
                return img
            except Exception as window_error:
                raise RuntimeError(
                    f"Screen capture failed ({type(screen_error).__name__}: {screen_error}); "
                    f"window capture fallback failed ({type(window_error).__name__}: {window_error})"
                ) from window_error

    def norm_to_screen(self, norm_x: float, norm_y: float) -> Tuple[int, int]:
        """Convert normalized (0..1000, 0..1000) coords to absolute screen pixels."""
        if not self.last_rect:
            self.get_client_rect()
        left, top, right, bottom = self.last_rect
        w = right - left
        h = bottom - top
        abs_x = max(left,min(right-1,int(left + (norm_x / 1000.0) * w)))
        abs_y = max(top,min(bottom-1,int(top + (norm_y / 1000.0) * h)))
        return abs_x, abs_y
