"""Validated, game-specific configuration loaded before any input is sent."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .config import config
from .controls import SCAN_CODES, VK_MAP


class NormalizedRegion(BaseModel):
    """A normalized rectangle in the captured client image (0..1)."""

    model_config = ConfigDict(extra="forbid")
    x1: float = Field(ge=0, le=1)
    y1: float = Field(ge=0, le=1)
    x2: float = Field(gt=0, le=1)
    y2: float = Field(gt=0, le=1)

    def model_post_init(self, __context):
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("region x2/y2 must be greater than x1/y1")


class InputAction(BaseModel):
    """A named low-level input action described by a key or mouse button."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["key", "mouse"]
    value: str
    duration: float = Field(default=0.1, gt=0, le=10)

    @model_validator(mode="after")
    def validate_input(self):
        value = self.value.strip().casefold()
        if not value:
            raise ValueError("input action value must not be empty")
        if self.kind == "mouse" and value not in {"left", "right"}:
            raise ValueError("mouse action value must be left or right")
        if self.kind == "key" and value not in SCAN_CODES and value not in VK_MAP:
            raise ValueError(f"unsupported key name in profile action: {value}")
        self.value = value
        return self


class MotionLimits(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_action_seconds: float = Field(default=0.75, ge=0.1, le=10)
    max_run_seconds: float = Field(default=120, gt=0, le=3600)
    max_actions: int = Field(default=60, ge=1, le=10000)
    max_turn_counts: int = Field(default=30, ge=1, le=1000)
    max_sweep_counts: int = Field(default=360, ge=1, le=10000)
    max_jump_recovery_attempts: int = Field(default=config.approach_target_jump_recovery_attempts,
                                            ge=0, le=3)
    interaction_retry_step_seconds: float = Field(default=0.25, ge=0, le=0.75)


class GameProfile(BaseModel):
    """Game-specific controls and visual vocabulary; safe to validate offline."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    comment: str = Field(default="", alias="_comment", exclude=True)
    game_id: str = Field(min_length=1, max_length=80)
    window: str = ""
    input_backend: Literal["window", "native"] = config.input_backend
    camera_type: Literal["first_person", "third_person", "none"] = "none"
    move_keys: Dict[str, str] = Field(default_factory=lambda: {
        "forward": "w", "backward": "s", "left": "a", "right": "d"})
    actions: Dict[str, InputAction] = Field(default_factory=dict)
    regions: Dict[str, NormalizedRegion] = Field(default_factory=dict)
    forbidden_keys: list[str] = Field(default_factory=list)
    motion_limits: MotionLimits = Field(default_factory=MotionLimits)
    detector_queries: Dict[str, list[str]] = Field(default_factory=dict)
    ui_adapters: list[Literal["world_selection", "stacked_main_menu"]] = Field(default_factory=list)
    main_menu_target_transitions: Dict[str, Literal["world_select"]] = Field(default_factory=dict)
    mine_requires_grounded_level_view: bool | None = None
    mine_requires_target_at_crosshair: bool = False
    thresholds: Dict[str, float] = Field(default_factory=dict)
    prompt_addendum: str = Field(default="", max_length=4000)
    calibration: Dict[str, float | bool] = Field(default_factory=dict)

    @model_validator(mode="after")
    def default_mining_view_guard(self):
        # Preserve the old 3D mining guard for legacy first-person profiles,
        # while leaving non-camera profiles free of that assumption.
        if self.mine_requires_grounded_level_view is None:
            self.mine_requires_grounded_level_view = self.camera_type == "first_person"
        return self

    @field_validator("game_id")
    @classmethod
    def valid_game_id(cls, value):
        value = value.strip().casefold()
        if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,79}", value):
            raise ValueError("game_id may contain only lowercase letters, digits, dot, underscore, and hyphen")
        return value

    @field_validator("move_keys")
    @classmethod
    def validate_move_keys(cls, value):
        required = {"forward", "backward", "left", "right"}
        if not required.issubset(value):
            raise ValueError(f"move_keys must define {sorted(required)}")
        for direction, key in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(f"move_keys[{direction!r}] must be a non-empty key name")
            if key.strip().casefold() not in SCAN_CODES and key.strip().casefold() not in VK_MAP:
                raise ValueError(f"unsupported key name in move_keys[{direction!r}]: {key}")
        return {direction: key.strip().casefold() for direction, key in value.items()}

    @field_validator("thresholds")
    @classmethod
    def finite_thresholds(cls, value):
        for name, threshold in value.items():
            if not math.isfinite(threshold) or not 0 <= threshold <= 1:
                raise ValueError(f"threshold {name!r} must be finite and in 0..1")
        return value

    @field_validator("calibration")
    @classmethod
    def finite_calibration(cls, value):
        for name, setting in value.items():
            if isinstance(setting, float) and not math.isfinite(setting):
                raise ValueError(f"calibration value {name!r} must be finite")
        return value

    @field_validator("forbidden_keys")
    @classmethod
    def valid_forbidden_keys(cls, value):
        if any(not isinstance(key, str) or not key.strip() for key in value):
            raise ValueError("forbidden_keys must contain non-empty key names")
        return [key.strip().casefold() for key in value]


def _profiles_dir() -> Path:
    return config.project_root / "data" / "profiles"


def _generic_profile(game_id: str = "default") -> GameProfile:
    return GameProfile(game_id=game_id, input_backend=config.input_backend)


def load_profile(game_id: str = "", profile_path: str | Path | None = None) -> GameProfile:
    """Load and validate a profile. Errors are raised before agent attachment/input."""
    if profile_path:
        path = Path(profile_path).expanduser().resolve()
        try:
            raw = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read profile {path}: {exc}") from exc
        try:
            profile = GameProfile.model_validate(raw)
        except Exception as exc:
            raise ValueError(f"Invalid profile {path}: {exc}") from exc
        if game_id and profile.game_id != game_id.strip().casefold():
            raise ValueError(f"--game-id {game_id!r} does not match profile game_id {profile.game_id!r}")
        return profile

    normalized = (game_id or "default").strip().casefold()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,79}", normalized):
        raise ValueError("game_id may contain only letters, digits, dot, underscore, and hyphen")
    path = _profiles_dir() / f"{normalized}.json"
    if not path.exists():
        return _generic_profile(normalized)
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        profile = GameProfile.model_validate(raw)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Invalid profile {path}: {exc}") from exc
    if profile.game_id != normalized:
        raise ValueError(f"Profile {path} declares game_id {profile.game_id!r}, expected {normalized!r}")
    return profile
