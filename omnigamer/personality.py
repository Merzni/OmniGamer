"""Neuro-sama style personality, live streaming dialogue, and commentary logger."""
import logging
import sys
import threading
import time

def configure_stdio():
    """Use UTF-8 for text streams when the current stream supports reconfiguration."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass

configure_stdio()

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

logger = logging.getLogger(__name__)
_console_warning_lock = threading.Lock()
_console_warning_emitted = False


def configure_console():
    """Refresh Rich after configuring the current standard output streams."""
    global console
    configure_stdio()
    console = Console()
    return console


def _warn_console_failure(exc):
    global _console_warning_emitted
    with _console_warning_lock:
        if _console_warning_emitted:
            return
        _console_warning_emitted = True
    logger.warning("Cosmetic console output failed (%s); suppressing further print warnings",
                   type(exc).__name__)


console = configure_console()

class StreamerPersona:
    """Manages the agent's live personality, speech bubbles, and mood."""

    def __init__(self, name: str = "Omni-chan"):
        self.name = name
        self.recent_dialogues = []

    def speak(self, text: str, mood: str = "curious"):
        """Output personality dialogue to terminal with streaming styling."""
        try:
            timestamp = time.strftime("%H:%M:%S")
            entry = {"time": timestamp, "speaker": self.name, "text": text, "mood": mood}
            self.recent_dialogues.append(entry)
            if len(self.recent_dialogues) > 50:
                self.recent_dialogues.pop(0)

            color_map = {
                "curious": "cyan",
                "excited": "magenta",
                "confused": "yellow",
                "triumphant": "green",
                "upset": "red"
            }
            border_style = color_map.get(mood, "cyan")

            panel = Panel(
                Text(f"💬 \"{text}\"", style="bold white"),
                title=f"🌸 [bold {border_style}]{self.name}[/bold {border_style}] ({timestamp})",
                subtitle=f"[dim]Mood: {mood}[/dim]",
                border_style=border_style,
                padding=(0, 2)
            )
            console.print(panel)
        except Exception as exc:
            _warn_console_failure(exc)

    def log_action(self, action_desc: str, result_desc: str):
        """Log a concise action summary."""
        try:
            console.print(f"[bold blue]⚡ Action:[/bold blue] {action_desc} -> [dim]{result_desc}[/dim]")
        except Exception as exc:
            _warn_console_failure(exc)
