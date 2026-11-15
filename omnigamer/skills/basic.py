"""Small bounded skills shared by game profiles and deterministic test doubles."""
from __future__ import annotations

import time
from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .contracts import SkillResult, SkillStatus


class SearchTargetParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_text: str = Field(min_length=1, max_length=200)
    sweep: int = Field(default=6, ge=1, le=12)
    vertical: bool = False


class MovePulseParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    direction: Literal["forward", "backward", "left", "right"]
    seconds: float = Field(gt=0, le=10)


class TurnParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    direction: str = Field(pattern="^(left|right|up|down)$")
    amount: int = Field(gt=0, le=1000)


class InteractUntilChangeParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    region: str = Field(min_length=1, max_length=80)
    action: str = Field(min_length=1, max_length=80)
    max_repeats: int = Field(default=3, ge=1, le=10)


class WaitForConditionParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    timeout: float = Field(gt=0, le=300)
    query: Optional[str] = Field(default=None, min_length=1, max_length=200)
    region: Optional[str] = Field(default=None, min_length=1, max_length=80)
    present: bool = True

    @model_validator(mode="after")
    def requires_visual_condition(self):
        if not self.query and not self.region:
            raise ValueError("wait_for_condition requires a query or a region")
        return self


def _emit(recorder, name, step, status, reason, evidence, before=None, after=None):
    if recorder:
        recorder({
            "schema_version": 1,
            "skill": name,
            "iteration": step,
            "state": status.value,
            "reason": reason,
            "action": evidence.get("action"),
            "result": {"status": status.value, "reason": reason, "evidence": evidence},
            "evidence": evidence,
        }, before, after)


def _result(status, code, reason, evidence, started):
    return SkillResult(status=status, result_code=code, reason=reason,
                       evidence=evidence, duration_seconds=max(0.0, time.monotonic() - started))


def region_delta(before, after, region, noise_floor=4):
    """97th-percentile RGB change in a normalized crop, with a small noise floor."""
    if before.size != after.size:
        after = after.resize(before.size)
    x1 = int(region.x1 * before.width)
    y1 = int(region.y1 * before.height)
    x2 = int(region.x2 * before.width)
    y2 = int(region.y2 * before.height)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Region resolves to an empty crop")
    left = np.asarray(before.convert("RGB").crop((x1, y1, x2, y2)), dtype=np.float32)
    right = np.asarray(after.convert("RGB").crop((x1, y1, x2, y2)), dtype=np.float32)
    channel_delta = np.max(np.abs(left - right), axis=2)
    noise_suppressed = np.maximum(channel_delta - noise_floor, 0)
    return float(np.percentile(noise_suppressed, 97) / (255 - noise_floor))


def queries_for_target(profile, target):
    queries = [target]
    normalized = target.casefold()
    for label, synonyms in (getattr(profile, "detector_queries", {}) or {}).items():
        if label.replace("_", " ").casefold() in normalized or any(
                str(value).casefold() == normalized for value in synonyms):
            for value in synonyms:
                if str(value).casefold() not in {item.casefold() for item in queries}:
                    queries.append(str(value))
    return queries


def _valid_detection(response, profile):
    if not response:
        return []
    score_min = profile.thresholds.get("detector_score_min", 0.15)
    boxes = []
    for box in response.get("boxes", []):
        try:
            x1, y1, x2, y2 = (float(box[k]) for k in ("x1", "y1", "x2", "y2"))
            score = float(box.get("score", 0))
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1 and score >= score_min:
            boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                          "score": score, "query": str(box.get("query", ""))})
    return boxes


def move_pulse(direction, seconds, *, controls, profile, camera_guard, stop_check=None, recorder=None,
               capture=None):
    params = MovePulseParams(direction=direction, seconds=seconds)
    if params.direction not in profile.move_keys:
        raise ValueError(f"Unknown movement direction: {params.direction}")
    if params.seconds > profile.motion_limits.max_action_seconds:
        raise ValueError("Movement pulse exceeds profile max_action_seconds")
    started = time.monotonic()
    before = capture.grab_frame() if capture else None
    key = profile.move_keys[params.direction]
    proposal = {"action_type": "walk", "key": key, "duration": params.seconds,
                "scene_type": "gameplay", "camera_pitch": "unknown", "visual_confidence": 0}
    filtered, guard_reason = camera_guard.filter(proposal)
    if filtered.get("action_type") != "walk":
        evidence = {"direction": params.direction, "key": key, "action": None,
                    "guard_action": filtered, "guard_reason": guard_reason}
        return _result(SkillStatus.FAILED, "FAILED", guard_reason or "camera_guard_blocked_move",
                       evidence, started)
    try:
        if stop_check:
            stop_check()
        controls.press_key(key, params.seconds)
        after = capture.grab_frame() if capture else None
        evidence = {"direction": params.direction, "key": key, "seconds": params.seconds,
                    "action": {"type": "move_pulse", "key": key, "seconds": params.seconds}}
        _emit(recorder, "move_pulse", 1, SkillStatus.SUCCESS, "movement_sent", evidence, before, after)
        return _result(SkillStatus.SUCCESS, "SUCCESS", "movement_sent", evidence, started)
    except InterruptedError as exc:
        controls.release_all()
        evidence = {"direction": params.direction, "key": key, "action": None}
        _emit(recorder, "move_pulse", 1, SkillStatus.ABORT, str(exc), evidence, before, None)
        return _result(SkillStatus.ABORT, "ABORT", str(exc), evidence, started)


def turn(direction, amount, *, controls, camera_guard, profile, stop_check=None,
         recorder=None, capture=None):
    params = TurnParams(direction=direction, amount=amount)
    if profile.camera_type == "none":
        return SkillResult(status=SkillStatus.FAILED, result_code="FAILED",
                           reason="profile_has_no_camera", evidence={"direction": direction})
    bounded = min(params.amount, profile.motion_limits.max_turn_counts)
    started = time.monotonic()
    before = capture.grab_frame() if capture else None
    amount_name = "small" if bounded <= 40 else "medium" if bounded <= 80 else "large"
    proposal = {"action_type": "look", "look_direction": params.direction,
                "look_amount": amount_name, "x": None, "y": None}
    filtered, reason = camera_guard.filter(proposal)
    if reason and filtered.get("action_type") != "look":
        return _result(SkillStatus.FAILED, "FAILED", reason, {"direction": direction}, started)
    dx, dy = int(filtered.get("x") or 0), int(filtered.get("y") or 0)
    if params.direction in {"left", "right"}:
        dx = (1 if params.direction == "right" else -1) * min(bounded, abs(dx))
        dy = 0
    else:
        dy = (1 if params.direction == "down" else -1) * min(bounded, abs(dy))
        dx = 0
    try:
        if stop_check:
            stop_check()
        controls.move_camera(dx, dy)
        filtered.update(x=dx, y=dy)
        camera_guard.executed(filtered)
        after = capture.grab_frame() if capture else None
        evidence = {"direction": direction, "requested_counts": params.amount,
                    "sent_counts": [dx, dy], "clamped": bounded != params.amount,
                    "action": {"type": "turn", "counts": [dx, dy]}}
        _emit(recorder, "turn", 1, SkillStatus.SUCCESS, "turn_sent", evidence, before, after)
        return _result(SkillStatus.SUCCESS, "SUCCESS", "turn_sent", evidence, started)
    except InterruptedError as exc:
        controls.release_all()
        evidence = {"direction": direction, "sent_counts": [0, 0], "action": None}
        _emit(recorder, "turn", 1, SkillStatus.ABORT, str(exc), evidence, before, None)
        return _result(SkillStatus.ABORT, "ABORT", str(exc), evidence, started)


def search_target(target_text, sweep, *, capture, detector, controls, camera_guard,
                  profile, stop_check=None, recorder=None, vertical=False):
    params = SearchTargetParams(target_text=target_text, sweep=sweep, vertical=vertical)
    started = time.monotonic()
    queries = queries_for_target(profile, params.target_text)
    per_turn = profile.motion_limits.max_turn_counts
    sweep_turn_limit = max(1, profile.motion_limits.max_sweep_counts // max(1, per_turn))
    max_turns = min(params.sweep, 12, profile.motion_limits.max_actions, sweep_turn_limit)
    for index in range(max_turns + 1):
        if stop_check:
            try:
                stop_check()
            except InterruptedError as exc:
                controls.release_all()
                return _result(SkillStatus.ABORT, "ABORT", str(exc), {"turns": index}, started)
        frame = capture.grab_frame()
        response = detector.detect(frame, queries)
        if response is None:
            evidence = {"target_text": params.target_text, "queries": queries,
                        "turns": index, "action": None}
            _emit(recorder, "search_target", index + 1, SkillStatus.FAILED,
                  "detector_unavailable", evidence, frame, None)
            return _result(SkillStatus.FAILED, "FAILED", "detector_unavailable", evidence, started)
        boxes = _valid_detection(response, profile)
        if boxes:
            evidence = {"target_text": params.target_text, "queries": queries,
                        "boxes": boxes, "turns": index, "action": None}
            _emit(recorder, "search_target", index + 1, SkillStatus.SUCCESS,
                  "target_detected", evidence, frame, frame)
            return _result(SkillStatus.SUCCESS, "TARGET_DETECTED", "target_detected", evidence, started)
        if index == max_turns:
            break
        turn_result = turn("right", profile.motion_limits.max_turn_counts,
                           controls=controls, camera_guard=camera_guard,
                           profile=profile, stop_check=stop_check)
        if turn_result.status != SkillStatus.SUCCESS:
            return turn_result
        _emit(recorder, "search_target", index + 1, SkillStatus.FAILED,
              "target_not_visible", {"target_text": params.target_text,
              "queries": queries, "turn": "right", "action": turn_result.evidence.get("action")},
              frame, None)
    if params.vertical:
        for direction in ("up", "down"):
            frame = capture.grab_frame()
            turn_result = turn(direction, min(20, profile.motion_limits.max_turn_counts),
                               controls=controls, camera_guard=camera_guard,
                               profile=profile, stop_check=stop_check)
            if turn_result.status != SkillStatus.SUCCESS:
                return turn_result
            response = detector.detect(capture.grab_frame(), queries)
            if response is None:
                return _result(SkillStatus.FAILED, "FAILED", "detector_unavailable",
                               {"target_text": params.target_text, "queries": queries}, started)
            boxes = _valid_detection(response, profile)
            if boxes:
                evidence = {"target_text": params.target_text, "queries": queries,
                            "boxes": boxes, "turns": max_turns, "vertical_turn": direction,
                            "action": turn_result.evidence.get("action")}
                _emit(recorder, "search_target", max_turns + 1, SkillStatus.SUCCESS,
                      "target_detected", evidence, frame, None)
                return _result(SkillStatus.SUCCESS, "TARGET_DETECTED", "target_detected", evidence, started)
    evidence = {"target_text": params.target_text, "queries": queries,
                "turns": max_turns, "action": None}
    return _result(SkillStatus.FAILED, "FAILED", "target_not_found_within_sweep", evidence, started)


def _perform_profile_action(action, controls, profile):
    configured = profile.actions.get(action)
    if configured is None:
        raise ValueError(f"Unknown profile action: {action}")
    if configured.kind == "mouse":
        controls.hold_mouse(button=configured.value, duration=configured.duration)
    else:
        key = configured.value.casefold()
        if key in {value.casefold() for value in profile.forbidden_keys}:
            raise ValueError(f"Profile forbids requested action key: {key}")
        controls.press_key(key, configured.duration)
    return {"name": action, "kind": configured.kind, "value": configured.value,
            "duration": configured.duration}


def interact_until_change(region, action, max_repeats, *, capture, controls, profile, camera_guard,
                          stop_check=None, recorder=None):
    params = InteractUntilChangeParams(region=region, action=action, max_repeats=max_repeats)
    if params.region not in profile.regions:
        raise ValueError(f"Unknown profile region: {params.region}")
    region_spec = profile.regions[params.region]
    threshold = profile.thresholds.get("region_change", 0.08)
    started = time.monotonic()
    before = capture.grab_frame()
    for attempt in range(1, params.max_repeats + 1):
        if stop_check:
            try:
                stop_check()
            except InterruptedError as exc:
                controls.release_all()
                evidence = {"attempt": attempt, "action": None}
                _emit(recorder, "interact_until_change", attempt, SkillStatus.ABORT,
                      str(exc), evidence, before, None)
                return _result(SkillStatus.ABORT, "ABORT", str(exc), evidence, started)
        sent = _perform_profile_action(params.action, controls, profile)
        after = capture.grab_frame()
        delta = region_delta(before, after, region_spec)
        evidence = {"region": params.region, "action": sent, "attempt": attempt,
                    "region_delta": delta, "threshold": threshold,
                    "retry_step_seconds": profile.motion_limits.interaction_retry_step_seconds,
                    "action_sent": {"type": "interact", **sent}}
        if delta >= threshold:
            _emit(recorder, "interact_until_change", attempt, SkillStatus.UNCONFIRMED,
                  "region_changed_unverified", evidence, before, after)
            return _result(SkillStatus.UNCONFIRMED, "REGION_CHANGE_UNCONFIRMED",
                           "region_changed_unverified", evidence, started)
        _emit(recorder, "interact_until_change", attempt, SkillStatus.FAILED,
              "region_unchanged", evidence, before, after)
        if attempt < params.max_repeats:
            retry_step = profile.motion_limits.interaction_retry_step_seconds
            if retry_step > 0:
                moved = move_pulse("forward", min(retry_step, profile.motion_limits.max_action_seconds),
                                   controls=controls, profile=profile, camera_guard=camera_guard,
                                   stop_check=stop_check)
                if moved.status != SkillStatus.SUCCESS:
                    return moved
                before = capture.grab_frame()
            else:
                before = after
    return _result(SkillStatus.FAILED, "FAILED", "region_unchanged_after_repeats", evidence, started)


def wait_for_condition(*, capture, profile, timeout, query=None, region=None,
                       present=True, detector=None, stop_check=None, recorder=None,
                       poll_seconds=0.2):
    params = WaitForConditionParams(timeout=timeout, query=query, region=region, present=present)
    if not params.query and not params.region:
        raise ValueError("wait_for_condition requires a query or a region")
    if params.region and params.region not in profile.regions:
        raise ValueError(f"Unknown profile region: {params.region}")
    started = time.monotonic()
    deadline = started + params.timeout
    baseline = capture.grab_frame() if params.region else None
    step = 0
    last_evidence = {}
    while time.monotonic() < deadline:
        if stop_check:
            try:
                stop_check()
            except InterruptedError as exc:
                return _result(SkillStatus.ABORT, "ABORT", str(exc), last_evidence, started)
        frame = capture.grab_frame()
        visible = None
        if params.query:
            if detector is None:
                return _result(SkillStatus.FAILED, "FAILED", "detector_unavailable", {}, started)
            boxes = _valid_detection(detector.detect(frame, queries_for_target(profile, params.query)), profile)
            visible = bool(boxes)
            last_evidence["boxes"] = boxes
        if params.region:
            delta = region_delta(baseline, frame, profile.regions[params.region])
            changed = delta >= profile.thresholds.get("region_change", 0.08)
            region_condition = changed if params.present else not changed
            visible = region_condition if visible is None else visible and region_condition
            last_evidence["region_delta"] = delta
        condition_met = visible is params.present
        step += 1
        last_evidence.update(query=params.query, region=params.region, present=params.present,
                             condition_met=condition_met, action=None)
        if condition_met:
            _emit(recorder, "wait_for_condition", step, SkillStatus.SUCCESS,
                  "condition_met", last_evidence, baseline, frame)
            return _result(SkillStatus.SUCCESS, "CONDITION_MET", "condition_met", last_evidence, started)
        time.sleep(min(poll_seconds, max(0, deadline - time.monotonic())))
    return _result(SkillStatus.FAILED, "FAILED", "condition_timeout", last_evidence, started)
