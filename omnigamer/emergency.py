"""Global emergency-key polling for the Windows desktop session."""
import threading

import win32con
import win32api


def virtual_key_code(key_name):
    name = str(key_name).strip().upper()
    aliases = {"ESC": "ESCAPE", "CTRL": "CONTROL", "ALT": "MENU", "SPACEBAR": "SPACE"}
    name = aliases.get(name, name)
    if len(name) == 1 and name.isalnum():
        return ord(name)
    virtual_key = getattr(win32con, f"VK_{name}", None)
    if virtual_key is None:
        raise ValueError(f"Unsupported emergency key: {key_name}")
    return virtual_key


class EmergencyHotkey:
    """Poll a global virtual key and invoke its callback once per key press."""

    def __init__(self, key_name, callback, key_state=None, poll_interval=0.01):
        self.key_name = str(key_name)
        self.virtual_key = virtual_key_code(key_name)
        self.callback = callback
        self.key_state = key_state or self._read_key_state
        self.poll_interval = poll_interval
        self._stop = threading.Event()
        self._thread = None
        self._was_pressed = False

    def _read_key_state(self):
        # GetAsyncKeyState reads the active desktop's global key state without
        # registering a reserved hotkey or requiring an administrator token.
        return bool(win32api.GetAsyncKeyState(self.virtual_key) & 0x8000)

    def poll_once(self):
        pressed = bool(self.key_state())
        triggered = pressed and not self._was_pressed
        self._was_pressed = pressed
        if triggered:
            self.callback()
        return triggered

    def _run(self):
        while not self._stop.is_set():
            try:
                self.poll_once()
            except Exception:
                # A polling failure must not kill the agent thread or desktop app.
                pass
            self._stop.wait(self.poll_interval)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="OmniGamerEmergencyKey", daemon=True)
        self._thread.start()

    def close(self):
        self._stop.set()

