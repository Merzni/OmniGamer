"""A bounded, detector-driven state machine for approaching a visible target."""
import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..config import config

logger = logging.getLogger(__name__)


@dataclass
class ApproachTargetSettings:
    deadzone_x: float = config.approach_target_deadzone_x
    align_exit_x: float = config.approach_target_align_exit_x
    calibration_counts: int = config.approach_target_calibration_counts
    walk_duration: float = config.approach_target_walk_duration
    reach_height: float = config.approach_target_reach_height
    reach_top_y: float = config.approach_target_reach_top_y
    reach_min_width: float = config.approach_target_reach_min_width
    reach_center_x: float = config.approach_target_reach_center_x
    reach_center_stability: float = config.approach_target_reach_center_stability
    reach_crosshair_margin: float = config.approach_target_reach_crosshair_margin
    track_max_center_shift: float = config.approach_target_track_max_center_shift
    track_ambiguity_margin: float = 0.02
    max_look_counts: int = config.approach_target_max_look_counts
    close_target_held_impulses: int = config.approach_target_close_target_held_impulses
    close_target_held_duration: float = config.approach_target_close_target_held_duration
    jump_recovery_attempts: int = config.approach_target_jump_recovery_attempts
    max_impulses: int = config.approach_target_max_impulses
    max_seconds: float = config.approach_target_max_seconds
    lost_frames: int = config.approach_target_lost_frames
    lost_search_turns: int = config.approach_target_lost_search_turns
    score_min: float = config.detector_score_min
    max_box_area: float = config.detector_max_box_area
    max_boxes: int = 10

    @classmethod
    def from_profile(cls, profile):
        """Map profile thresholds and motion limits onto this skill's settings."""
        if profile is None:
            return cls()
        values = profile.thresholds
        limits = profile.motion_limits
        mapped = {
            "deadzone_x": "approach_deadzone_x",
            "align_exit_x": "approach_align_exit_x",
            "reach_height": "approach_reach_height",
            "reach_top_y": "approach_reach_top_y",
            "reach_min_width": "approach_reach_min_width",
            "reach_center_x": "approach_reach_center_x",
            "reach_center_stability": "approach_reach_center_stability",
            "reach_crosshair_margin": "approach_reach_crosshair_margin",
            "track_max_center_shift": "approach_track_max_center_shift",
            "track_ambiguity_margin": "approach_track_ambiguity_margin",
            "score_min": "detector_score_min",
            "max_box_area": "detector_max_box_area",
        }
        overrides = {setting: values[key] for setting, key in mapped.items() if key in values}
        overrides["max_seconds"] = limits.max_run_seconds
        overrides["max_impulses"] = limits.max_actions
        overrides["jump_recovery_attempts"] = limits.max_jump_recovery_attempts
        overrides["max_look_counts"] = max(10, limits.max_turn_counts)
        overrides["walk_duration"] = min(config.approach_target_walk_duration,
                                         max(0.5, limits.max_action_seconds))
        return cls(**overrides)

    def validate(self):
        if self.align_exit_x < self.deadzone_x:
            raise ValueError("align_exit_x must be greater than or equal to deadzone_x")
        if not 0.5 <= self.walk_duration <= 0.75:
            raise ValueError("walk_duration must be in 0.5..0.75 seconds")
        if (not 0 < self.reach_height <= 1 or not 0 < self.max_box_area <= 1
                or not 0 < self.reach_min_width <= 1 or self.max_look_counts < 10
                or not 0 < self.track_max_center_shift <= 1
                or not 0 <= self.track_ambiguity_margin < self.track_max_center_shift
                or not 0 <= self.reach_crosshair_margin < 0.25):
            raise ValueError("reach_height and max_box_area must be in (0,1]")
        if self.max_impulses < 1 or self.max_seconds <= 0:
            raise ValueError("approach_target limits must be positive")
        if not 0 <= self.jump_recovery_attempts <= 3:
            raise ValueError("jump_recovery_attempts must be in 0..3")


@dataclass
class ApproachTargetResult:
    state: str
    reason: str
    last_box: dict | None
    iterations: int
    impulses: int
    elapsed_seconds: float
    criterion: str | None = None
    input_actions: int = 0

    def to_dict(self):
        return asdict(self)


def box_metrics(box):
    if not box:
        return None, None, None
    width = max(0.0, float(box["x2"]) - float(box["x1"]))
    height = max(0.0, float(box["y2"]) - float(box["y1"]))
    center_x = (float(box["x1"]) + float(box["x2"])) / 2
    return height, center_x - 0.5, width * height


def action_for_box(box, settings=None, previous_box=None):
    """Pure no-input action classification used by the saved-frame dry run."""
    settings = settings or ApproachTargetSettings()
    h, e, _ = box_metrics(box)
    if h is None:
        return "acquire"
    width = float(box["x2"]) - float(box["x1"])
    center = (float(box["x1"]) + float(box["x2"])) / 2
    prev_width = (float(previous_box["x2"]) - float(previous_box["x1"])) if previous_box else 0
    prev_center = ((float(previous_box["x1"]) + float(previous_box["x2"])) / 2) \
        if previous_box else None
    if (float(box["y1"]) <= settings.reach_top_y and width >= settings.reach_min_width
            and abs(center - 0.5) <= settings.reach_center_x and previous_box
            and float(previous_box["y1"]) <= settings.reach_top_y
            and prev_width >= settings.reach_min_width
            and abs(prev_center - 0.5) <= settings.reach_center_x
            and abs(center - prev_center) <= settings.reach_center_stability):
        return "reached"
    return "align" if abs(e) > settings.align_exit_x else "advance"


class ApproachTargetSkill:
    def __init__(self, capture, target_text, detector, controls, camera_guard, game_id,
                 stop_check=None, recorder=None, settings=None, calibration_path=None,
                 clock=None, sleep=None, profile=None):
        self.capture = capture
        self.target_text = " ".join(str(target_text or "").split()).strip()
        if not self.target_text:
            raise ValueError("target_text is required")
        self.detector = detector
        self.controls = controls
        self.camera_guard = camera_guard
        self.game_id = str(game_id or "unknown")
        self.profile = profile
        self.move_keys = getattr(profile, "move_keys", None) or {
            "forward": "w", "backward": "s", "left": "a", "right": "d"}
        self.stop_check = stop_check or (lambda: None)
        self.recorder = recorder
        self.settings = settings or ApproachTargetSettings()
        self.settings.validate()
        self.calibration_path = Path(calibration_path or (config.data_dir / "calibration.json"))
        self.clock = clock or time.monotonic
        self.sleep = sleep or time.sleep
        self._started = None
        self._iterations = 0
        self._impulses = 0
        # Profile max_actions is shared by movement and camera input. Keep the
        # movement-only counter separately for diagnostics.
        self._input_actions = 0
        self._last_box = None
        self._last_valid_box = None
        self._last_center = None
        self._selection_uncertain = False
        self._selection_uncertain_streak = 0
        self._missing_frames = 0
        self._lost_search_turns = 0
        self._aligning = False
        self._plateau_streak = 0
        self._blocked_streak = 0
        self._blocked_recoveries = 0
        self._side_steps_attempted = 0
        self._pending_side_step = None
        self._pending_jump_recovery = False
        self._jump_recoveries_used = 0
        self._side_direction = self.move_keys["left"]
        self._reach_streak = 0
        self._last_reach_center = None
        self._close_target_held_remaining = 0
        self._close_target_held_active = False
        self._calibration = self._read_calibration()

    def _read_calibration(self):
        try:
            rows = json.loads(self.calibration_path.read_text(encoding="utf-8"))
            value = rows.get(self.game_id, {})
            k = float(value.get("k", 0))
            if math.isfinite(k) and k > 0:
                return {"k": k, "invert": bool(value.get("invert", False))}
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
        configured = getattr(self.profile, "calibration", {}) or {}
        try:
            k = float(configured.get("k", configured.get("camera_counts_per_radian", 0)))
            if math.isfinite(k) and k > 0:
                return {"k": k, "invert": bool(configured.get("invert", configured.get("invert_horizontal", False)))}
        except (TypeError, ValueError):
            pass
        return None

    def _save_calibration(self, k, invert):
        self.calibration_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            rows = json.loads(self.calibration_path.read_text(encoding="utf-8"))
            if not isinstance(rows, dict):
                rows = {}
        except (OSError, ValueError, json.JSONDecodeError):
            rows = {}
        rows[self.game_id] = {"k": round(float(k), 8), "invert": bool(invert),
                              "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        temporary = self.calibration_path.with_suffix(self.calibration_path.suffix + ".tmp")
        temporary.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.calibration_path)

    def _detect(self, frame):
        queries = [self.target_text]
        target = self.target_text.casefold()
        for label, synonyms in getattr(self.profile, "detector_queries", {}).items():
            if label.replace("_", " ").casefold() in target or any(
                    str(query).casefold() == target for query in synonyms):
                existing = {item.casefold() for item in queries}
                queries.extend(str(query) for query in synonyms if str(query).casefold() not in existing)
        response = self.detector.detect(
            frame, queries, score_min=self.settings.score_min,
            max_boxes=self.settings.max_boxes,
        )
        if response is None:
            return None, None
        raw_boxes = response.get("boxes", [])
        valid = []
        for item in response.get("boxes", []):
            try:
                box = {key: float(item[key]) for key in ("x1", "y1", "x2", "y2", "score")}
                if not all(math.isfinite(value) for value in box.values()):
                    continue
                if (box["score"] < self.settings.score_min or box["x2"] <= box["x1"]
                        or box["y2"] <= box["y1"]):
                    continue
                area = (box["x2"] - box["x1"]) * (box["y2"] - box["y1"])
                if area > self.settings.max_box_area:
                    continue
                box["query"] = str(item.get("query", self.target_text))
                valid.append(box)
            except (KeyError, TypeError, ValueError):
                continue
        primary = [box for box in valid if box["query"] == self.target_text]
        selected = primary
        source = "primary"
        if not selected and len(queries) > 1:
            selected = []
            for box in valid:
                if box["query"] not in queries[1:]:
                    continue
                width = box["x2"] - box["x1"]
                height = box["y2"] - box["y1"]
                if width > 0.5 or box["y2"] < 1 / 3 or (width > 0.65 and height < 0.25):
                    continue
                selected.append(box)
            if selected:
                source = "fallback_query"
                target_words = self._content_words(self.target_text)
                if target_words:
                    def specificity(box):
                        query_words = self._content_words(box["query"])
                        return (len(target_words & query_words), -len(query_words))
                    best_specificity = max(specificity(box) for box in selected)
                    if best_specificity[0] > 0:
                        selected = [box for box in selected
                                    if specificity(box) == best_specificity]
        return {"boxes": selected, "raw_boxes": raw_boxes, "queries": queries,
                "selected_query_source": source,
                "preprocess": response.get("preprocess"),
                "latency_ms": response.get("latency_ms"), "model": response.get("model")}, \
            response.get("latency_ms")

    @staticmethod
    def _content_words(text):
        stop_words = {"a", "an", "the", "of", "to", "in", "on", "with"}
        words = set()
        current = []
        for char in str(text).casefold():
            if char.isalnum():
                current.append(char)
            elif current:
                word = "".join(current)
                if word not in stop_words:
                    words.add(word)
                current = []
        if current:
            word = "".join(current)
            if word not in stop_words:
                words.add(word)
        return words

    def _select(self, detection):
        boxes = detection.get("boxes", []) if detection else []
        self._selection_uncertain = False
        if not boxes:
            self._selection_uncertain_streak = 0
            return None
        if self._last_center is None:
            selected = max(boxes, key=lambda item: item["score"])
        else:
            ranked = sorted(((math.hypot(
                (item["x1"] + item["x2"]) / 2 - self._last_center[0],
                (item["y1"] + item["y2"]) / 2 - self._last_center[1]), item)
                for item in boxes), key=lambda row: row[0])
            nearest_distance, selected = ranked[0]
            if nearest_distance > self.settings.track_max_center_shift:
                self._selection_uncertain_streak = 0
                return None
            if (len(ranked) > 1 and
                    ranked[1][0] - nearest_distance <= self.settings.track_ambiguity_margin):
                self._selection_uncertain = True
                self._selection_uncertain_streak += 1
                return None
        self._selection_uncertain_streak = 0
        self._last_center = ((selected["x1"] + selected["x2"]) / 2,
                             (selected["y1"] + selected["y2"]) / 2)
        self._last_box = dict(selected)
        self._last_valid_box = dict(selected)
        return dict(selected)

    def _close_and_centered(self, box):
        if not box:
            return False
        height, center_error, _ = box_metrics(box)
        return (height > 0.5 or float(box.get("y1", 1.0)) <= self.settings.reach_top_y) \
            and abs(center_error) <= self.settings.reach_center_x

    def _reach_geometry(self, box):
        if not box:
            return False, None
        width = float(box["x2"]) - float(box["x1"])
        center_x = (float(box["x1"]) + float(box["x2"])) / 2
        valid = (float(box["y1"]) <= self.settings.reach_top_y
                 and width >= self.settings.reach_min_width
                 and abs(center_x - 0.5) <= self.settings.reach_center_x)
        return valid, center_x

    def _crosshair_horizontally_aimed(self, box):
        if not box:
            return False
        margin = self.settings.reach_crosshair_margin
        return float(box["x1"]) + margin <= 0.5 <= float(box["x2"]) - margin

    def _crosshair_aimed(self, box):
        if not box or not self._crosshair_horizontally_aimed(box):
            return False
        margin = self.settings.reach_crosshair_margin
        return float(box["y1"]) + margin <= 0.5 <= float(box["y2"]) - margin

    def _camera(self, direction, counts):
        action = {"action_type": "look", "look_direction": direction, "look_amount": "small",
                  "x": None, "y": None}
        filtered, reason = self.camera_guard.filter(action)
        if reason and filtered.get("action_type") != "look":
            raise RuntimeError(f"CameraGuard rejected the skill look: {reason}")
        bounded = max(10, min(self.settings.max_look_counts, int(abs(counts))))
        dx = bounded if direction == "right" else -bounded if direction == "left" else 0
        dy = -bounded if direction == "up" else bounded if direction == "down" else 0
        filtered.update(x=dx, y=dy)
        self.controls.move_camera(dx, dy)
        self._input_actions += 1
        self.camera_guard.executed(filtered)
        return {"type": "look", "direction": direction, "counts": dx if dx else dy,
                "sent_counts": [dx, dy]}

    def _align_counts(self, error_x):
        k = self._calibration["k"]
        sign = -1 if self._calibration.get("invert") else 1
        requested = round(error_x / k * sign)
        magnitude = max(10, min(90, abs(requested)))
        return magnitude if requested >= 0 else -magnitude

    def _record(self, state, before, after, before_detection, after_detection, before_box, after_box,
                action, reason="", h_delta=None, progress=None):
        if not self.recorder:
            return
        before_h, before_e, _ = box_metrics(before_box)
        after_h, after_e, _ = box_metrics(after_box)
        row = {
            "iteration": self._iterations,
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "state": state,
            "detector_latency_ms": {
                "before": before_detection.get("latency_ms") if before_detection else None,
                "after": after_detection.get("latency_ms") if after_detection else None,
            },
            "detector": {
                "before": {"queries": before_detection.get("queries"),
                           "raw_boxes": before_detection.get("raw_boxes"),
                           "preprocess": before_detection.get("preprocess"),
                           "selected_query_source": before_detection.get("selected_query_source")}
                           if before_detection else None,
                "after": {"queries": after_detection.get("queries"),
                          "raw_boxes": after_detection.get("raw_boxes"),
                          "preprocess": after_detection.get("preprocess"),
                          "selected_query_source": after_detection.get("selected_query_source"),
                          "retry_attempts": after_detection.get("retry_attempts")}
                          if after_detection else None,
            },
            "box": {"before": before_box, "after": after_box,
                    "score": after_box.get("score") if after_box else (before_box.get("score") if before_box else None)},
            "h": {"before": before_h, "after": after_h},
            "e": {"before": before_e, "after": after_e},
            "h_delta": h_delta,
            "progress": progress,
            "input_actions": self._input_actions,
            "movement_impulses": self._impulses,
            "action": action,
            "reason": reason or None,
            "elapsed_seconds": round(self.clock() - self._started, 3),
        }
        try:
            self.recorder(row, before, after)
        except Exception as exc:
            logger.warning("Skill telemetry record failed (%s): %s", type(exc).__name__, exc)

    def _result(self, state, reason, criterion=None):
        return ApproachTargetResult(
            state=state, reason=reason, last_box=self._last_box,
            iterations=self._iterations, impulses=self._impulses,
            elapsed_seconds=round(max(0.0, self.clock() - self._started), 3),
            criterion=criterion, input_actions=self._input_actions,
        )

    def run(self):
        self._started = self.clock()
        current_frame = None
        current_detection = None
        current_box = None
        try:
            self.stop_check()
            if hasattr(self.capture, "focus"):
                self.capture.focus()
            current_frame = self.capture.grab_frame()
            current_detection, _ = self._detect(current_frame)
            if current_detection is None:
                self._iterations = 1
                self._record("ABORT", current_frame, current_frame, None, None, None, None,
                             None, "detector_unavailable")
                return self._result("ABORT", "detector_unavailable")
            current_box = self._select(current_detection)
            if current_box is None:
                self._missing_frames = 1

            while True:
                self.stop_check()
                if self.clock() - self._started >= self.settings.max_seconds:
                    return self._result("TIMEOUT", "max_seconds")
                if self._input_actions >= self.settings.max_impulses:
                    return self._result("TIMEOUT", "max_actions")

                self._iterations += 1
                before = current_frame
                before_detection = current_detection
                before_box = current_box
                before_h, before_e, _ = box_metrics(before_box)
                action = None
                action_reason = ""
                state = "ACQUIRE"
                terminal = None

                if self._reach_streak >= 2:
                    terminal = ("REACHED", "top_edge_centered_stable", "top_edge_centered_stable")
                elif self._pending_side_step:
                    direction = self._pending_side_step
                    self._pending_side_step = None
                    self.controls.press_key(direction, 0.5)
                    self._impulses += 1
                    self._input_actions += 1
                    action = {"type": "side_step", "key": direction, "duration": 0.5}
                    state = "ACQUIRE"
                elif self._pending_jump_recovery:
                    self._pending_jump_recovery = False
                    jump = self.profile.actions["jump"]
                    duration = min(jump.duration, self.profile.motion_limits.max_action_seconds, 0.75)
                    self.controls.press_keys([self.move_keys["forward"], jump.value], duration)
                    self._impulses += 1
                    self._input_actions += 1
                    action = {"type": "jump_recovery", "keys": [self.move_keys["forward"], jump.value],
                              "duration": duration}
                    state = "JUMP_RECOVERY"
                elif before_box is None:
                    if self._selection_uncertain:
                        state = "TRACK_UNCERTAIN"
                        action = {"type": "observe", "duration": 0}
                    elif self._close_target_held_active:
                        state = "CLOSE_TARGET_HELD"
                        if self._close_target_held_remaining <= 0:
                            terminal = ("REACHED_UNCONFIRMED", "target_occluded_after_close_approach", None)
                        else:
                            self.controls.press_key(self.move_keys["forward"], self.settings.close_target_held_duration)
                            self._impulses += 1
                            self._input_actions += 1
                            action = {"type": "cautious_walk", "key": self.move_keys["forward"],
                                      "duration": self.settings.close_target_held_duration}
                    elif (self._missing_frames >= self.settings.lost_frames
                          and self._close_and_centered(self._last_valid_box)):
                        self._close_target_held_active = True
                        self._close_target_held_remaining = self.settings.close_target_held_impulses
                        state = "CLOSE_TARGET_HELD"
                        self.controls.press_key(self.move_keys["forward"], self.settings.close_target_held_duration)
                        self._impulses += 1
                        self._input_actions += 1
                        action = {"type": "cautious_walk", "key": self.move_keys["forward"],
                                  "duration": self.settings.close_target_held_duration}
                    else:
                        state = "LOST" if self._missing_frames >= self.settings.lost_frames else "ACQUIRE"
                    if state == "ACQUIRE":
                        action = {"type": "observe", "duration": 0}
                    elif state == "LOST" and self._lost_search_turns >= self.settings.lost_search_turns:
                        terminal = ("LOST", "target_not_reacquired", None)
                    elif state == "LOST":
                        direction = "left" if self._lost_search_turns % 2 == 0 else "right"
                        action = self._camera(direction, 12)
                        action["type"] = "search_look"
                        self._lost_search_turns += 1
                elif before_box and self._calibration is None:
                    state = "ACQUIRE"
                    action = self._camera("right", self.settings.calibration_counts)
                    action["type"] = "calibration_look"
                else:
                    reaches_before, _ = self._reach_geometry(before_box)
                    if (reaches_before and not self._crosshair_horizontally_aimed(before_box)):
                        state = "ALIGN"
                        self._aligning = True
                    elif abs(before_e) > self.settings.align_exit_x:
                        state = "ALIGN"
                        self._aligning = True
                    elif (reaches_before and self._crosshair_horizontally_aimed(before_box)
                          and not self._crosshair_aimed(before_box)):
                        state = "ALIGN_VERTICAL"
                        direction = "up" if before_box["y2"] < 0.5 + self.settings.reach_crosshair_margin else "down"
                        action = self._camera(direction, 12)
                        action["type"] = "vertical_align"
                    elif self._aligning and abs(before_e) > self.settings.deadzone_x:
                        state = "ALIGN"
                    else:
                        state = "ADVANCE"
                        self._aligning = False

                    if state == "ALIGN":
                        counts = self._align_counts(before_e)
                        direction = "right" if counts > 0 else "left"
                        action = self._camera(direction, counts)
                    elif state == "ADVANCE":
                        self.controls.press_key(self.move_keys["forward"], self.settings.walk_duration)
                        self._impulses += 1
                        self._input_actions += 1
                        action = {"type": "walk", "key": self.move_keys["forward"],
                                  "duration": self.settings.walk_duration}

                if terminal:
                    state, action_reason, criterion = terminal
                    self._record(state, before, before, before_detection, None, before_box, before_box,
                                 action, action_reason)
                    return self._result(state, action_reason, criterion)

                self.stop_check()
                after = self.capture.grab_frame()
                after_detection, _ = self._detect(after)
                if after_detection is None:
                    self._record("ABORT", before, after, before_detection, None, before_box, None,
                                 action, "detector_unavailable")
                    return self._result("ABORT", "detector_unavailable")
                after_box = self._select(after_detection)
                if (after_box is None and action and "look" in action.get("type", "")):
                    # Recheck at the new camera position before counting a miss.
                    first_after_detection = after_detection
                    after = self.capture.grab_frame()
                    after_detection, _ = self._detect(after)
                    if after_detection is None:
                        self._record("ABORT", before, after, before_detection, None,
                                     before_box, None, action, "detector_unavailable")
                        return self._result("ABORT", "detector_unavailable")
                    after_detection["retry_attempts"] = [
                        {"queries": first_after_detection.get("queries"),
                         "raw_boxes": first_after_detection.get("raw_boxes"),
                         "preprocess": first_after_detection.get("preprocess")},
                        {"queries": after_detection.get("queries"),
                         "raw_boxes": after_detection.get("raw_boxes"),
                         "preprocess": after_detection.get("preprocess")},
                    ]
                    after_box = self._select(after_detection)
                after_h, after_e, _ = box_metrics(after_box)
                h_delta = (after_h - before_h) if before_h is not None and after_h is not None else None
                progress = h_delta is not None and h_delta >= 0.01

                if action and action["type"] == "calibration_look" and before_box and after_box:
                    delta_e = after_e - before_e
                    measured = abs(-delta_e / action["counts"]) if action["counts"] else 0.0
                    if measured >= 0.0005:
                        invert = delta_e > 0
                        self._calibration = {"k": measured, "invert": invert}
                        self._save_calibration(measured, invert)
                        action_reason = "camera_sign_inverted" if invert else "camera_calibrated"
                    else:
                        action_reason = "camera_calibration_low_motion"

                if after_box is None:
                    self._missing_frames += 1
                    current_box = None
                    if action and action.get("type") == "cautious_walk":
                        self._close_target_held_remaining -= 1
                else:
                    self._missing_frames = 0
                    self._lost_search_turns = 0
                    self._close_target_held_active = False
                    self._close_target_held_remaining = 0
                    current_box = after_box

                reaches, reach_center = self._reach_geometry(after_box)
                reaches = reaches and self._crosshair_aimed(after_box)
                if reaches and (self._last_reach_center is None or
                                abs(reach_center - self._last_reach_center)
                                <= self.settings.reach_center_stability):
                    self._reach_streak += 1
                elif reaches:
                    self._reach_streak = 1
                else:
                    self._reach_streak = 0
                self._last_reach_center = reach_center if reaches else None

                if action and action["type"] in {"walk", "jump_recovery"} and h_delta is not None:
                    if h_delta < 0.005:
                        self._plateau_streak += 1
                    else:
                        self._plateau_streak = 0
                    if h_delta < 0.005 and before_h < 0.5 and after_h < 0.5:
                        self._blocked_streak += 1
                    else:
                        self._blocked_streak = 0

                    if self._blocked_streak >= 3:
                        self._blocked_streak = 0
                        if self._blocked_recoveries:
                            jump = (getattr(self.profile, "actions", {}) or {}).get("jump")
                            forbidden = {key.casefold() for key in getattr(self.profile, "forbidden_keys", [])}
                            if (self._jump_recoveries_used < self.settings.jump_recovery_attempts
                                    and jump and jump.kind == "key"
                                    and jump.value.casefold() not in forbidden):
                                self._jump_recoveries_used += 1
                                self._pending_jump_recovery = True
                                action_reason = "three_low_growth_impulses_jump_recovery_next"
                            elif self._side_steps_attempted < 2:
                                self._pending_side_step = self._side_direction
                                self._side_steps_attempted += 1
                                left_key, right_key = self.move_keys["left"], self.move_keys["right"]
                                self._side_direction = right_key if self._side_direction == left_key else left_key
                                action_reason = "three_low_growth_impulses_alternate_side_step_next"
                            else:
                                terminal = ("BLOCKED", "blocked_after_side_step", None)
                        else:
                            self._blocked_recoveries = 1
                            self._pending_side_step = self._side_direction
                            self._side_steps_attempted += 1
                            left_key, right_key = self.move_keys["left"], self.move_keys["right"]
                            self._side_direction = right_key if self._side_direction == left_key else left_key
                            action_reason = "three_low_growth_impulses_side_step_next"

                if self._reach_streak >= 2:
                    terminal = ("REACHED", "top_edge_centered_stable", "top_edge_centered_stable")
                elif self._selection_uncertain_streak >= 2:
                    terminal = ("UNCERTAIN", "ambiguous_target_tracking", None)

                if terminal:
                    state, action_reason, criterion = terminal
                self._record(state, before, after, before_detection, after_detection, before_box,
                             after_box, action, action_reason, h_delta, progress)
                if terminal:
                    return self._result(state, action_reason, criterion)

                current_frame, current_detection = after, after_detection
        except InterruptedError as exc:
            return self._result("ABORT", f"{type(exc).__name__}: {exc}")
        finally:
            try:
                self.controls.release_all()
            except Exception as exc:
                logger.warning("ApproachTarget input release failed (%s): %s", type(exc).__name__, exc)
