"""Independent logical pointer using addressed window messages, never SendInput."""
import win32api
import win32gui
import win32con
from .controls import HumanInputController, INPUT_MOUSE, INPUT_KEYBOARD, KEYEVENTF_KEYUP


class WindowInputController(HumanInputController):
    """Delivery is not consumption: Raw Input/DirectInput games may ignore these events."""
    def __init__(self, hwnd_provider, rect_provider, pointer, **kwargs):
        super().__init__(**kwargs)
        self.hwnd_provider, self.rect_provider, self.pointer = hwnd_provider, rect_provider, pointer
        self.client_x = self.client_y = None
        self.mouse_flags = 0

    def _post(self, message, wparam, lparam):
        hwnd = self.hwnd_provider()
        if not hwnd or not win32gui.IsWindow(hwnd):
            raise RuntimeError("Virtual input requires a valid target window")
        # PostMessage raises on failure; it never changes the system pointer.
        win32gui.PostMessage(hwnd,message,wparam,lparam)

    def move_camera(self,dx,dy,steps=6):
        raise RuntimeError("Оконный ввод не поддерживает относительную 3D-камеру. Выбери общий ввод Windows (native) или отдельную среду игры.")

    def _position(self):
        rect = self.rect_provider()
        if self.client_x is None:
            self.client_x = (rect[2]-rect[0])//2
            self.client_y = (rect[3]-rect[1])//2
        return rect

    def get_cursor_pos(self):
        rect = self._position()
        return rect[0]+self.client_x, rect[1]+self.client_y

    def _lparam(self):
        return (self.client_x & 0xffff) | ((self.client_y & 0xffff)<<16)

    def _publish(self, event="move", pressed=None, button="left"):
        rect = self._position()
        self.pointer.update(rect[0]+self.client_x,rect[1]+self.client_y,rect,event,
                            bool(self.mouse_flags) if pressed is None else pressed,button)

    def set_cursor_pos(self, x, y):
        rect = self._position()
        self.client_x = max(0,min(rect[2]-rect[0]-1,x-rect[0]))
        self.client_y = max(0,min(rect[3]-rect[1]-1,y-rect[1]))
        self._post(win32con.WM_MOUSEMOVE,self.mouse_flags,self._lparam())
        self._publish()

    def _send(self, packet):
        self._position()
        if packet.type == INPUT_MOUSE:
            flags = packet.union.mi.dwFlags
            if flags & 1:
                x,y = self.get_cursor_pos()
                self.set_cursor_pos(x+packet.union.mi.dx,y+packet.union.mi.dy)
            for bit,message,mask,button in (
                (2,win32con.WM_LBUTTONDOWN,win32con.MK_LBUTTON,"left"),
                (4,win32con.WM_LBUTTONUP,win32con.MK_LBUTTON,"left"),
                (8,win32con.WM_RBUTTONDOWN,win32con.MK_RBUTTON,"right"),
                (16,win32con.WM_RBUTTONUP,win32con.MK_RBUTTON,"right")):
                if flags & bit:
                    pressed = bit in (2,8)
                    next_flags = self.mouse_flags | mask if pressed else self.mouse_flags & ~mask
                    self._post(message,next_flags,self._lparam())
                    self.mouse_flags = next_flags
                    self._publish("down" if pressed else "up",pressed,button)
        elif packet.type == INPUT_KEYBOARD:
            key = packet.union.ki
            vk = key.wVk or win32api.MapVirtualKey(key.wScan,3)
            if not vk: raise ValueError("Unsupported scan code for window input")
            up = bool(key.dwFlags & KEYEVENTF_KEYUP)
            lparam = 1 | (key.wScan<<16) | ((3<<30) if up else 0)
            self._post(win32con.WM_KEYUP if up else win32con.WM_KEYDOWN,vk,lparam)
        else:
            raise ValueError("Unsupported input packet")
