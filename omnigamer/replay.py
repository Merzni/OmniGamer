"""Read-only frame-directory playback for dry-run planning and diagnosis."""
from pathlib import Path

from PIL import Image


class FrameDirectoryCapture:
    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(self, directory):
        self.directory = Path(directory).expanduser().resolve()
        if not self.directory.is_dir():
            raise ValueError(f"Frame directory does not exist: {self.directory}")
        self.frames = sorted(path for path in self.directory.iterdir()
                             if path.is_file() and path.suffix.casefold() in self.EXTENSIONS)
        if not self.frames:
            raise ValueError(f"No supported frame images in {self.directory}")
        self.index = 0
        self.hwnd = 0
        self.last_frame = None

    def grab_frame(self):
        path = self.frames[min(self.index, len(self.frames) - 1)]
        self.index += 1
        with Image.open(path) as image:
            self.last_frame = image.convert("RGB").copy()
        return self.last_frame.copy()

    def get_window_title(self):
        return f"Replay: {self.directory.name}"
