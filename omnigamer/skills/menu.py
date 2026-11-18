"""Screen-grounded, single-click menu interaction for goal plans."""
from __future__ import annotations

import re
import time

from PIL import ImageChops
from pydantic import BaseModel, ConfigDict, Field

from ..vision_brain import BrainDecision
from .contracts import SkillResult, SkillStatus


def classify_menu_result(value):
    words = re.findall(r"[a-z0-9а-яё]+", str(value).casefold())
    text = " ".join(words)
    word_set = set(words)
    has_world = bool(word_set & {"world", "мир", "мира", "мире", "миров"})
    if (("world" in word_set and "selected" in word_set)
            or (has_world and word_set & {"выбран", "выбрана", "выбрано", "выбраны"})):
        return "world_selected"
    if ("gameplay" in word_set or "геймплей" in word_set
            or ("игровой" in word_set and "процесс" in word_set)
            or ("игра" in word_set and word_set & {"началась", "запущена", "стартовала"})
            or ("world" in word_set and "load" in word_set)
            or "game started" in text):
        return "gameplay"
    if ("world" in word_set and word_set & {"list", "selection", "select"}) or (
            has_world and word_set & {"список", "списке", "выбор", "выбору", "выбора"}):
        return "world_select"
    return None


def _label_words(value):
    return re.findall(r"[a-z0-9а-яё]+", str(value).casefold())


def _menu_target_matches(requested, grounded, expected_result, profile):
    requested_words = _label_words(requested)
    localized_words = _label_words(grounded.target_label)
    localized = " ".join(localized_words)
    if localized in {"visible world row", "play selected world"}:
        if "world_selection" not in profile.ui_adapters:
            return False
        result = classify_menu_result(expected_result)
        if (localized == "visible world row" and grounded.menu_id == "world_select"
                and grounded.menu_state == "no_selection" and result == "world_selected"):
            return True
        if (localized == "play selected world" and grounded.menu_id == "world_select"
                and grounded.menu_state == "world_selected" and result == "gameplay"):
            return True
        return False
    return bool(requested_words and requested_words == localized_words)


class ClickMenuTargetParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_label: str = Field(min_length=1, max_length=80)
    intent: str = Field(min_length=1, max_length=200)
    expected_result: str = Field(min_length=1, max_length=100)


def _target_patch(frame, x, y):
    width, height = frame.size
    center_x = round(x * width / 1000)
    center_y = round(y * height / 1000)
    half_width = max(12, round(width * 0.20))
    half_height = max(12, round(height * 0.08))
    return frame.convert("RGB").crop((
        max(0, center_x - half_width), max(0, center_y - half_height),
        min(width, center_x + half_width), min(height, center_y + half_height),
    ))


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
    requested_result = classify_menu_result(expected_result)
    localized_result = classify_menu_result(grounded.expected_result)
    adapter_conflict = ("world_selection" in profile.ui_adapters
                        and grounded.menu_id == "world_select"
                        and localized_result is not None
                        and requested_result != localized_result)
    if adapter_conflict or (requested_result is not None and localized_result is not None
                            and requested_result != localized_result):
        raise ValueError("Localized menu action leads to a different expected result")
    if not _menu_target_matches(target_label, grounded, expected_result, profile):
        raise ValueError("Localized menu target does not match the requested target")
    if not (0 <= grounded.x <= 1000 and 0 <= grounded.y <= 1000):
        raise ValueError("Grounded menu coordinates are outside the captured image")

    if stop_check:
        stop_check()
    prepare_target()
    if capture.get_client_rect() != before_rect:
        raise RuntimeError("Game window moved/resized during menu localization; click cancelled")
    current_frame = capture.grab_frame()
    if capture.last_rect != before_rect:
        raise RuntimeError("Game window moved/resized while rechecking the menu target")
    before_patch = _target_patch(frame, grounded.x, grounded.y)
    current_patch = _target_patch(current_frame, grounded.x, grounded.y)
    if before_patch.size != current_patch.size or ImageChops.difference(
            before_patch, current_patch).getbbox() is not None:
        raise RuntimeError("Menu target changed during localization; click cancelled")
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
