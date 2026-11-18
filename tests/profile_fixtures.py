"""Hermetic profile loading for tests; never reads local runtime profiles."""
from pathlib import Path
from unittest.mock import patch

from omnigamer.config import config
from omnigamer.profiles import load_profile as _load_profile


_FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures"
_PROFILE_DIR = _FIXTURE_ROOT / "profiles"


def load_profile(game_id="", profile_path=None):
    """Load fixture-backed profiles without consulting user runtime data."""
    if profile_path is not None:
        return _load_profile(game_id, profile_path)
    if (game_id or "").strip().casefold() == "minecraft":
        return _load_profile(game_id, _PROFILE_DIR / "minecraft.json")
    with patch.object(config, "project_root", _FIXTURE_ROOT):
        return _load_profile(game_id)
