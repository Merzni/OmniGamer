"""A bot-owned pointer state and an optional click-through desktop marker."""
import threading
import time


class BotPointer:
    def __init__(self, desktop=False):
        self._lock = threading.Lock()
        self._state = {"visible": False, "x": 500, "y": 500, "pressed": False,
                       "button": "left", "event": "idle", "sequence": 0}
        self.desktop = DesktopPointer() if desktop else None

    def update(self, screen_x, screen_y, rect, event="move", pressed=False, button="left"):
        left, top, right, bottom = rect
        with self._lock:
            self._state.update(visible=True, screen_x=screen_x, screen_y=screen_y,
                x=round(1000*(screen_x-left)/max(1,right-left)),
                y=round(1000*(screen_y-top)/max(1,bottom-top)), event=event,
                pressed=pressed, button=button, timestamp=time.time(),
                sequence=self._state["sequence"]+1)
            snapshot = dict(self._state)
        if self.desktop:
            self.desktop.publish(snapshot)

    def snapshot(self):
        with self._lock:
            state = dict(self._state)
        state["overlay_error"] = self.desktop.error if self.desktop else ""
        return state

    def hide(self):
        with self._lock:
            self._state["visible"] = False
        if self.desktop:
            self.desktop.publish(self.snapshot())


class DesktopPointer:
    """Own non-activating window; excluded from screenshots to avoid confusing VLM."""
    def __init__(self):
        self.hwnd = None
        self.state = {"visible": False}
        self.error = ""
        self._thread = None
        self._ready = threading.Event()

    def publish(self, state):
        self.state = state
        if not self._thread:
            if not state.get("visible"):
                return
            self._thread = threading.Thread(target=self._run, daemon=True, name="OmniGamerBotPointer")
            self._thread.start()
            self._ready.wait(1)
        if self.hwnd:
            import win32gui, win32con
            try:
                win32gui.PostMessage(self.hwnd, win32con.WM_APP+1, 0, 0)
            except Exception as exc:
                # A failed cosmetic marker must not interrupt button-up delivery.
                self.error=str(exc)

    def _run(self):
        import ctypes
        from ctypes import wintypes
        import win32api, win32gui, win32con
        try:
            wc = win32gui.WNDCLASS()
            wc.hInstance = win32api.GetModuleHandle(None)
            wc.lpszClassName = f"OmniGamerBotPointer_{id(self)}"
            wc.lpfnWndProc = self._proc
            win32gui.RegisterClass(wc)
            style = (win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT | win32con.WS_EX_TOPMOST
                     | win32con.WS_EX_TOOLWINDOW | win32con.WS_EX_NOACTIVATE)
            self.hwnd = win32gui.CreateWindowEx(style, wc.lpszClassName, "OmniGamer Bot Cursor",
                win32con.WS_POPUP, 0, 0, 72, 72, 0, 0, wc.hInstance, None)
            win32gui.SetLayeredWindowAttributes(self.hwnd, 0, 255, win32con.LWA_COLORKEY)
            affinity = ctypes.windll.user32.SetWindowDisplayAffinity
            affinity.argtypes, affinity.restype = [wintypes.HWND,wintypes.DWORD], wintypes.BOOL
            if not affinity(self.hwnd, 0x11):  # WDA_EXCLUDEFROMCAPTURE
                raise RuntimeError("Desktop marker cannot be excluded from capture; dashboard pointer remains available")
            self._ready.set()
            self._proc(self.hwnd,win32con.WM_APP+1,0,0)
            win32gui.PumpMessages()
        except Exception as exc:
            self.error = str(exc)
            if self.hwnd:
                try: win32gui.DestroyWindow(self.hwnd)
                except Exception: pass
            self.hwnd = None
            self._ready.set()

    def _proc(self, hwnd, message, wparam, lparam):
        import win32api, win32gui, win32con
        if message == win32con.WM_NCHITTEST:
            return win32con.HTTRANSPARENT
        if message == win32con.WM_MOUSEACTIVATE:
            return win32con.MA_NOACTIVATE
        if message == win32con.WM_APP+1:
            state = self.state
            if state.get("visible"):
                win32gui.SetWindowPos(hwnd,win32con.HWND_TOPMOST,state["screen_x"]-30,state["screen_y"]-30,
                                     72,72,win32con.SWP_NOACTIVATE|win32con.SWP_SHOWWINDOW)
                win32gui.InvalidateRect(hwnd,None,True)
            else: win32gui.ShowWindow(hwnd,win32con.SW_HIDE)
            return 0
        if message == win32con.WM_PAINT:
            dc,paint = win32gui.BeginPaint(hwnd)
            brush = win32gui.CreateSolidBrush(0)
            win32gui.FillRect(dc,(0,0,72,72),brush)
            win32gui.DeleteObject(brush)
            pen = win32gui.CreatePen(win32con.PS_SOLID,3,win32api.RGB(255,40,180))
            old_pen = win32gui.SelectObject(dc,pen)
            old_brush = win32gui.SelectObject(dc,win32gui.GetStockObject(win32con.NULL_BRUSH))
            radius = 22 if self.state.get("pressed") else 14
            win32gui.Ellipse(dc,30-radius,30-radius,30+radius,30+radius)
            for end in ((30,8),(30,52),(8,30),(52,30)):
                win32gui.MoveToEx(dc,30,30);win32gui.LineTo(dc,*end)
            win32gui.SetTextColor(dc,win32api.RGB(255,120,210))
            win32gui.SetBkMode(dc,win32con.TRANSPARENT)
            win32gui.DrawText(dc,"AI",-1,(42,42,70,70),win32con.DT_LEFT)
            win32gui.SelectObject(dc,old_pen);win32gui.SelectObject(dc,old_brush)
            win32gui.DeleteObject(pen)
            win32gui.EndPaint(hwnd,paint)
            return 0
        return win32gui.DefWindowProc(hwnd,message,wparam,lparam)
