"""Screen-grounded, single-click menu interaction for goal plans."""
from __future__ import annotations

import re
import time

from pydantic import BaseModel, ConfigDict, Field

from ..vision_brain import BrainDecision
from .contracts import SkillResult, SkillStatus


def _known_menu_result(value):
    text = " ".join(re.findall(r"[a-z0-9а-яё]+", str(value).casefold()))
    if "world" in text and "selected" in text:
        return "world_selected"
    if "gameplay" in text or ("world" in text and "load" in text) or "game started" in text:
        return "gameplay"
    if "world" in text and any(word in text for word in ("list", "selection", "select")):
        return "world_select"
    return None


class ClickMenuTargetParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_label: str = Field(min_length=1, max_length=80)
    intent: str = Field(min_length=1, max_length=200)
    expected_result: str = Field(min_length=1, max_length=100)


def click_menu_target(target_label, intent, expected_result, *, capture, controls, profile, grounder,
                      prepare_target, check_target, stop_check=None, recorder=None):
    """Ground a requested label on the current menu, then send exactly one click."""
    started = time.monotonic()
    if not profile.ui_adapters:
        raise ValueError("Profile has no enabled menu UI adapter")
    if stop_check:
        stop_check()
    prepare_target()
    if stop_check:
        stop_check()

    before_rect = capture.get_client_rect()
    frame = capture.grab_frame()
    if capture.last_rect != before_rect:
        raise RuntimeError("Game window moved/resized while capturing the menu")

    request = BrainDecision(action_type="click", target_label=target_label, intent=intent,
                            expected_result=expected_result)
    grounded = grounder(frame, request)
    if (grounded.scene_type != "menu" or grounded.x is None or grounded.y is None
            or grounded.visual_confidence < 0.85 or not grounded.target_label.strip()):
        raise ValueError("Menu target was not confidently grounded on a verified menu")
    requested_result = _known_menu_result(expected_result)
    localized_result = _known_menu_result(grounded.expected_result)
    adapter_conflict = ("world_selection" in profile.ui_adapters
                        and grounded.menu_id == "world_select"
                        and localized_result is not None
                        and requested_result != localized_result)
    if adapter_conflict or (requested_result is not None and localized_result is not None
                            and requested_result != localized_result):
        raise ValueError("Localized menu action leads to a different expected result")
    if not (0 <= grounded.x <= 1000 and 0 <= grounded.y <= 1000):
        raise ValueError("Grounded menu coordinates are outside the captured image")

    if stop_check:
        stop_check()
    prepare_target()
    if capture.get_client_rect() != before_rect:
        raise RuntimeError("Game window moved/resized during menu localization; click cancelled")
    check_target()
    screen_x, screen_y = capture.norm_to_screen(grounded.x, grounded.y)
    controls.click(screen_x, screen_y, button="left", hold_time=0.08)
    if stop_check:
        stop_check()
    capture.grab_frame()
    result = SkillResult(
        status=SkillStatus.UNCONFIRMED,
        result_code="MENU_ACTION_UNCONFIRMED",
        reason="menu_click_sent_postcondition_pending",
        duration_seconds=max(0.0, time.monotonic() - started),
        evidence={
            "target_label": grounded.target_label,
            "expected_result": grounded.expected_result,
            "menu_id": grounded.menu_id,
            "menu_state": grounded.menu_state,
            "visual_confidence": grounded.visual_confidence,
            "normalized_target": [grounded.x, grounded.y],
            "screen_target": [screen_x, screen_y],
            "input_actions": 1,
            "postcondition": "pending_visual_verification",
        },
    )
    return result
