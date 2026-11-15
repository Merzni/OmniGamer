"""Global configuration for OmniGamer."""
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field

class OmniConfig(BaseModel):
    # Paths
    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    data_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data")
    screenshots_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "screenshots")
    db_path: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "memory.db")

    # Vision & Model
    vlm_api_url: str = "http://127.0.0.1:8081/v1/chat/completions"
    vlm_model_name: str = "qwen3.5-4b-local"
    request_timeout: float = 110.0
    image_max_size: int = 640  # Smaller VLM input; raise if small UI text becomes unreadable
    vlm_max_tokens: int = 192

    # Optional local open-vocabulary target detector
    detector_url: str = "http://127.0.0.1:8082"
    detector_score_min: float = Field(default=0.15, ge=0, le=1)
    detector_max_box_area: float = Field(default=0.6, gt=0, le=1)
    detector_preprocess: Literal["none", "gamma", "clahe", "stretch"] = "stretch"

    # Action & Safety
    action_delay_seconds: float = 0.1  # Let the rendered frame settle after an action
    max_motion_seconds: float = Field(default=0.75, ge=0.5, le=3.0)
    navigation_stuck_delta: float = 0.015
    verification_interval: int = Field(default=4, ge=1, le=10)
    recovery_after_steps: int = Field(default=20, ge=4, le=100)
    recovery_after_seconds: float = Field(default=90, ge=10, le=600)
    state_repeat_detection_enabled: bool = True
    state_repeat_window: int = Field(default=8, ge=2, le=64)
    state_repeat_triggers_recovery: bool = False
    assess_look_with_vlm: bool = False
    click_hold_seconds: float = 0.08
    failsafe_corner: bool = True  # Move mouse to (0,0) to abort immediately
    emergency_key: str = "F12"   # Key to immediately halt the agent
    input_backend: Literal["window", "native"] = "native"
    desktop_bot_pointer: bool = True

    # approach_target skill
    approach_target_deadzone_x: float = Field(default=0.06, ge=0, le=0.5)
    approach_target_align_exit_x: float = Field(default=0.09, ge=0, le=0.5)
    approach_target_calibration_counts: int = Field(default=12, ge=1, le=30)
    approach_target_walk_duration: float = Field(default=0.5, ge=0.5, le=0.75)
    approach_target_reach_height: float = Field(default=0.8, gt=0, le=1)
    approach_target_reach_top_y: float = Field(default=0.02, ge=0, le=0.1)
    approach_target_reach_min_width: float = Field(default=0.1, gt=0, le=1)
    approach_target_reach_center_x: float = Field(default=0.15, ge=0, le=0.5)
    approach_target_reach_center_stability: float = Field(default=0.05, ge=0, le=0.5)
    approach_target_reach_crosshair_margin: float = Field(default=0.05, ge=0, le=0.24)
    approach_target_track_max_center_shift: float = Field(default=0.35, gt=0, le=1)
    approach_target_max_look_counts: int = Field(default=30, ge=10, le=90)
    approach_target_close_target_held_impulses: int = Field(default=2, ge=1, le=6)
    approach_target_close_target_held_duration: float = Field(default=0.25, gt=0, le=0.5)
    approach_target_jump_recovery_attempts: int = Field(default=1, ge=0, le=3)
    approach_target_max_impulses: int = Field(default=40, ge=1, le=200)
    approach_target_max_seconds: float = Field(default=120, gt=0, le=1800)
    approach_target_lost_frames: int = Field(default=3, ge=1, le=10)
    approach_target_lost_search_turns: int = Field(default=6, ge=1, le=12)

    # Learning & Exploration
    change_threshold: float = 0.06 # Normalized pixel difference to count as real state transition
    memory_hash_distance: int = Field(default=4, ge=0, le=16)
    save_screenshots: bool = True  # Keep visual history for inspection

    # Passive run telemetry
    telemetry_enabled: bool = True
    telemetry_frame_every_n: int = Field(default=20, ge=1)
    telemetry_resource_interval_s: float = Field(default=10.0, gt=0)
    telemetry_frames_max_mb: float = Field(default=500.0, gt=0)
    telemetry_runs_max_count: int = Field(default=100, ge=1, le=10000)
    telemetry_runs_max_mb: float = Field(default=5000.0, gt=0)

    # Goal planner limits
    goal_validation_retries: int = Field(default=1, ge=0, le=1)
    goal_max_replans: int = Field(default=3, ge=0, le=10)
    goal_max_skill_steps: int = Field(default=12, ge=1, le=100)
    goal_plan_max_tokens: int = Field(default=256, ge=64, le=512)

    # Personality
    persona_name: str = "Omni-chan"

config = OmniConfig()
config.data_dir.mkdir(parents=True, exist_ok=True)
config.screenshots_dir.mkdir(parents=True, exist_ok=True)
