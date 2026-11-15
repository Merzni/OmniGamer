"""Bounded camera commands and persistent, visually checked small goals."""
import math

from .look import look_counts


GOALS = {
    "enter_game": "Войти в игровой мир; меню и загрузка должны закончиться.",
    "restore_view": "Вернуть взгляд примерно к горизонту, чтобы видеть путь перед собой.",
    "reach_open_ground": "Выйти на открытое ровное место; не копать под собой.",
    "collect_resource": "Получить дополнительный ресурс в видимый быстрый слот.",
}


class CameraGuard:
    """Relative-count budget, NOT an estimate of actual pitch in degrees."""
    def __init__(self):
        self.vertical_offset = 0

    def filter(self, action, facts=None, grounded_confident=False):
        result = dict(action)
        pitch = facts.pitch if facts else "unknown"
        scene = facts.scene if facts else result.get("scene_type", "unknown")
        if pitch == "unknown" and result.get("visual_confidence", 0) >= .85:
            pitch = result.get("camera_pitch", "unknown")
        kind = result["action_type"]
        reason = ""
        if scene == "gameplay" and pitch in {"up", "down"} and kind in {"walk", "jump", "mine"}:
            direction = "down" if pitch == "up" else "up"
            result.update(action_type="look", look_direction=direction, look_amount="medium",
                          x=None, y=None, key=None)
            reason = "Сначала вернуть взгляд к горизонту; движение/копание при экстремальном наклоне отменено."
        elif kind == "mine" and (pitch != "level" or not grounded_confident):
            result.update(action_type="wait", duration=.5, key=None, x=None, y=None)
            reason = "Копание отменено: направление взгляда и безопасная опора не подтверждены."
        if result["action_type"] == "look":
            direction = result.get("look_direction")
            amount = result.get("look_amount")
            if direction not in {"left", "right", "up", "down"} or amount not in {"small", "medium", "large"}:
                result.update(action_type="wait", duration=.5, key=None, x=None, y=None,
                              look_direction=None, look_amount=None)
                reason = reason or "Некорректный look: отсутствует направление или величина."
                return result, reason
            dx, dy = look_counts(direction, amount)
            dx = max(-140, min(140, dx))
            dy = max(-60, min(60, dy))
            correcting_extreme = (scene == "gameplay" and
                                  ((pitch == "down" and dy < 0) or (pitch == "up" and dy > 0)))
            if not correcting_extreme:
                dy = max(-180 - self.vertical_offset, min(180 - self.vertical_offset, dy))
            requested_dx, requested_dy = look_counts(direction, amount)
            if (dx, dy) != (requested_dx, requested_dy):
                reason = reason or "Поворот ограничен по размеру и накопленным вертикальным смещениям."
            result.update(x=dx, y=dy, look_direction=direction, look_amount=amount)
        return result, reason

    def executed(self, action):
        if action["action_type"] == "look":
            self.vertical_offset = max(-180, min(180, self.vertical_offset + int(action.get("y") or 0)))

    def observe(self, assessment):
        if assessment.usable and assessment.after.pitch == "level":
            self.vertical_offset = 0


class GoalSupervisor:
    """Legacy goal verifier for historical lesson records; not the live policy."""
    def __init__(self, state=None, include_resource_goal=True, max_steps=20, max_seconds=90):
        self.plan = ["enter_game", "restore_view", "reach_open_ground"] + (
            ["collect_resource"] if include_resource_goal else [])
        self.index = 0
        self.target_item = "log"
        self.baseline = None
        self.confirmations = 0
        self.no_progress_steps = 0
        self.no_progress_seconds = 0.0
        self.halted = False
        self.reason = ""
        self.completed = []
        self.max_steps, self.max_seconds = max_steps, max_seconds
        self.last_assessment = None
        self.pending_evidence = None
        if state:
            for key in ("plan", "index", "target_item", "baseline", "confirmations", "no_progress_steps",
                        "no_progress_seconds", "halted", "reason", "completed"):
                if key in state:
                    setattr(self, key, state[key])
            # A restart cannot serve as the second visual confirmation.
            self.confirmations = 0

    @property
    def goal_id(self):
        return self.plan[self.index] if self.index < len(self.plan) else "done"

    def snapshot(self):
        return {key: getattr(self, key) for key in (
            "plan", "index", "target_item", "baseline", "confirmations", "no_progress_steps",
            "no_progress_seconds", "halted", "reason", "completed")}

    def set_goal(self, goal_id, target_item="log"):
        if goal_id not in GOALS:
            raise ValueError("Unknown goal")
        self.plan, self.index = [goal_id], 0
        self.target_item = target_item.strip()[:40] or "log"
        self.baseline, self.confirmations = None, 0
        self.no_progress_steps, self.no_progress_seconds = 0, 0.0
        self.halted, self.reason, self.completed = False, "", []
        self.last_assessment = self.pending_evidence = None

    def context(self):
        description = GOALS.get(self.goal_id, "Все поставленные цели проверены.")
        if self.goal_id == "collect_resource":
            description += f" Целевой предмет: {self.target_item}; исходное количество: {self.baseline}."
        return (f"\nCURRENT PERSISTENT GOAL: {self.goal_id}: {description}\n"
                f"Steps without verified goal progress: {self.no_progress_steps}/{self.max_steps}. "
                "Changing the camera or narrating success does not complete the goal. "
                "Do not mine the block under your feet. Past scene descriptions are hypotheses."
                + ("\nThe goal state is a candidate; hold still for a second visual check."
                   if self.confirmations else "")
                + ("\nRECOVERY REQUIRED: change your strategy; inspect a safe route sideways/backward, "
                   "restore the horizon, and stop repeating variants of the same plan."
                   if self.no_progress_steps >= 6 else ""))

    def _met(self, facts):
        if self.goal_id == "enter_game":
            return facts.scene == "gameplay"
        if self.goal_id == "restore_view":
            return facts.scene == "gameplay" and facts.pitch == "level"
        if self.goal_id == "reach_open_ground":
            return (facts.scene == "gameplay" and facts.pitch == "level"
                    and facts.open_ground is True and facts.safe_footing is True)
        if self.goal_id == "collect_resource":
            count = facts.item_count(self.target_item)
            return count is not None and self.baseline is not None and count > self.baseline
        return False

    def observe(self, assessment, seconds, step, check_id=None):
        """Two time-separated assessments confirm a state, not causal learning."""
        self.no_progress_steps += 1
        self.no_progress_seconds += max(0, seconds) if math.isfinite(seconds) else 0
        event = {"status": "unverified", "goal_id": self.goal_id, "completed": False}
        if assessment is not None:
            self.last_assessment = assessment
            if assessment.usable:
                if self.goal_id == "collect_resource" and self.baseline is None:
                    self.baseline = assessment.before.item_count(self.target_item)
                event.update(target_item=self.target_item, baseline=self.baseline)
                met = self._met(assessment.after) and assessment.result != "regressed"
                event["status"] = "goal_candidate" if met else (
                    "suspected_regression" if assessment.result == "regressed" else "not_confirmed")
                if met:
                    if self.confirmations == 0:
                        self.confirmations = 1
                        self.pending_evidence = {"step": step, "check_id": check_id, "evidence": assessment.evidence}
                    elif step != self.pending_evidence["step"] and (
                        check_id is None or check_id != self.pending_evidence["check_id"]
                    ):
                        self.confirmations += 1
                else:
                    self.confirmations = 0
                    self.pending_evidence = None
                if self.confirmations >= 2:
                    event.update(status="visual_confirmed", completed=True,
                                 evidence=assessment.evidence, first_evidence=self.pending_evidence)
                    self.completed.append({"goal_id": self.goal_id, "step": step,
                                           "verification": "two_visual_model_checks"})
                    self.index += 1
                    self.confirmations = 0
                    self.baseline = None
                    self.pending_evidence = None
                    self.no_progress_steps, self.no_progress_seconds = 0, 0.0
            else:
                self.confirmations = 0
                self.pending_evidence = None
        if self.goal_id == "done":
            self.halted, self.reason = True, "Поставленные цели подтверждены визуальными проверками."
        elif self.no_progress_steps >= self.max_steps or self.no_progress_seconds >= self.max_seconds:
            self.halted = True
            self.reason = "Лимит попыток без подтверждённого продвижения. Проверь кадр и задай новую цель."
        return event


class PlaySupervisor:
    """Open-ended play: learn checked outcomes and recover without a finish line."""
    def __init__(self, state=None, max_steps=20, max_seconds=90):
        self.mode = "exploring"
        self.no_progress_steps = 0
        self.no_progress_seconds = 0.0
        self.recovery_count = 0
        self.recovery_remaining = 0
        self.verified_changes = 0
        self.experience_steps = 0
        self.last_change = ""
        self.pending = None
        self.max_steps, self.max_seconds = max_steps, max_seconds
        # Old prescribed plans and their halted flag must not stop free play.
        if state and state.get("schema_version") == 3:
            for key in ("mode", "no_progress_steps", "no_progress_seconds", "recovery_count",
                        "recovery_remaining", "verified_changes", "experience_steps", "last_change"):
                if key in state:
                    setattr(self, key, state[key])

    @property
    def halted(self):
        return False  # Runtime errors/failsafe still stop execution in the agent.

    @property
    def goal_id(self):
        return "free_play"  # Compatibility with historical journal scope column.

    def snapshot(self):
        return {"schema_version": 3, "mode": self.mode, "halted": False, **{
            key: getattr(self, key) for key in ("no_progress_steps", "no_progress_seconds", "recovery_count",
                "recovery_remaining", "verified_changes", "experience_steps", "last_change")}}

    def context(self):
        return (
            "\nCONTINUOUS AUTONOMOUS PLAY: learn this game through observed outcomes. "
            "Choose your own useful next experiment/short plan from the current screen and memory. "
            "Explore mechanics, develop skills, acquire useful resources and progress through the game. "
            "After an ending or achievement, keep playing: explore remaining content or try new strategies. "
            "There is no prescribed checklist and no terminal completed-goal state. "
            "Past narration is a hypothesis; only visually checked changes are evidence. "
            "Avoid random repeated movement; test a different explanation when an action fails. "
            f"Current mode={self.mode}; recent attempts without checked progress={self.no_progress_steps}; "
            f"strategy changes tried={self.recovery_count}."
            + ("\nRECOVERY: inspect a different direction, backtrack or try a safe lateral route. "
               "Do not repeat the same wall/sky/ground loop; use the new view to replan."
               if self.mode == "recovering" else "")
        )

    def trigger_recovery(self):
        """Start the existing recovery sequence when an external loop detector fires."""
        if self.mode == "recovering":
            return False
        self.mode = "recovering"
        self.recovery_remaining = 4
        self.recovery_count += 1
        self.no_progress_steps, self.no_progress_seconds = 0, 0.0
        return True

    @staticmethod
    def _candidate(assessment):
        before, after = assessment.before, assessment.after
        if before.scene in {"menu", "loading"} and after.scene == "gameplay":
            return {"kind": "entered_gameplay"}
        if before.pitch in {"up", "down"} and after.pitch == "level" and after.scene == "gameplay":
            return {"kind": "restored_forward_view"}
        if (before.open_ground is False and after.open_ground is True
                and after.safe_footing is True and after.pitch == "level"):
            return {"kind": "reached_open_ground"}
        if before.visible_items_readable and after.visible_items_readable:
            for item in after.items:
                old, new = before.item_count(item.name), after.item_count(item.name)
                if old is not None and new is not None and new > old:
                    return {"kind": "resource_increased", "item": item.name, "before_count": old, "after_count": new}
        return None

    @staticmethod
    def _still_visible(candidate, facts):
        kind = candidate["kind"]
        if kind == "entered_gameplay":
            return facts.scene == "gameplay"
        if kind == "restored_forward_view":
            return facts.scene == "gameplay" and facts.pitch == "level"
        if kind == "reached_open_ground":
            return facts.scene == "gameplay" and facts.pitch == "level" and facts.open_ground is True and facts.safe_footing is True
        count = facts.item_count(candidate["item"])
        return count is not None and count >= candidate["after_count"]

    def observe(self, assessment, seconds, step, check_id=None):
        self.experience_steps += 1
        self.no_progress_steps += 1
        self.no_progress_seconds += max(0, seconds) if math.isfinite(seconds) else 0
        event = {"status": "unverified", "completed": False, "verified_change": False}
        if assessment is not None:
            if assessment.usable and assessment.result != "regressed":
                if (self.pending and step != self.pending["step"] and check_id != self.pending["check_id"]
                        and self._still_visible(self.pending["change"], assessment.after)):
                    change = self.pending["change"]
                    event.update(status="visual_confirmed", verified_change=True,
                                 change=change, first_check_id=self.pending["check_id"],
                                 second_check_id=check_id, evidence=assessment.evidence)
                    self.verified_changes += 1
                    self.last_change = change["kind"]
                    self.pending = None
                    self.no_progress_steps, self.no_progress_seconds = 0, 0.0
                    self.mode, self.recovery_remaining = "exploring", 0
                else:
                    candidate = self._candidate(assessment)
                    self.pending = ({"change": candidate, "step": step, "check_id": check_id}
                                    if candidate else None)
                    event["status"] = "change_candidate" if candidate else "not_confirmed"
            else:
                self.pending = None
                event["status"] = "suspected_regression" if assessment.result == "regressed" else "unverified"
        if self.mode == "recovering" and not event["verified_change"]:
            self.recovery_remaining = max(0, self.recovery_remaining - 1)
            if self.recovery_remaining == 0:
                self.mode = "exploring"
        elif self.no_progress_steps >= self.max_steps or self.no_progress_seconds >= self.max_seconds:
            self.trigger_recovery()
            event["recovery_started"] = True
        return event

    def recovery_action(self, proposal, facts):
        if self.mode != "recovering" or facts.scene != "gameplay":
            return proposal, ""
        phase = 4 - self.recovery_remaining
        result = dict(proposal)
        direction = -1 if self.recovery_count % 2 else 1
        if phase == 0:
            result.update(action_type="look", look_direction="left" if direction < 0 else "right",
                          look_amount="medium", x=None, y=None, key=None)
        elif phase in {1, 2} and facts.pitch == "level" and facts.safe_footing is True:
            result.update(action_type="walk", key="s" if phase == 1 else ("a" if direction < 0 else "d"),
                          duration=.5, x=None, y=None)
        elif phase == 1:
            result.update(action_type="look", look_direction="right" if direction < 0 else "left",
                          look_amount="medium", x=None, y=None, key=None)
        else:
            return proposal, ""
        return result, "Сменить подход: осмотреть другой путь и выйти из повторяющейся последовательности."


class StateRepeatDetector:
    """Detect recently repeated dHash states and alternating look commands."""

    HASH_DISTANCE = 4

    def __init__(self, window=8):
        self.window = max(1, int(window))
        self.history = []

    @staticmethod
    def _distance(left, right):
        try:
            if len(left) != 16 or len(right) != 16:
                return 65
            return (int(left, 16) ^ int(right, 16)).bit_count()
        except (TypeError, ValueError):
            return 65

    @staticmethod
    def _look_direction(action):
        if not action or action.get("action_type") != "look":
            return None
        direction = action.get("look_direction")
        if direction in {"left", "right", "up", "down"}:
            return direction
        return None

    def detect(self, state_hash, step):
        recent = [row for row in self.history if 0 < step - row["step"] <= self.window]
        matches = [row for row in recent
                   if self._distance(row["state_hash"], state_hash) <= self.HASH_DISTANCE]
        repeat_of_step = max((row["step"] for row in matches), default=None)
        tail = self.history[-4:]
        directions = [self._look_direction(row["action"]) for row in tail]
        look_oscillation = (len(directions) == 4 and all(directions)
                            and directions[0] == directions[2]
                            and directions[1] == directions[3]
                            and directions[0] != directions[1]
                            and ({directions[0], directions[1]} in
                                 ({"left", "right"}, {"up", "down"})))
        return {
            "state_repeat": repeat_of_step is not None,
            "repeat_of_step": repeat_of_step,
            "look_oscillation": bool(look_oscillation),
        }

    def remember(self, state_hash, step, action):
        self.history.append({"step": int(step), "state_hash": state_hash,
                             "action": dict(action) if action else None})
        self.history = self.history[-self.window:]
