"""Validated goal plans, bounded execution, and per-game episodic outcomes."""
from __future__ import annotations

import json
import math
import sqlite3
import time
from copy import deepcopy
from contextlib import closing, contextmanager
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .config import config
from .memory import GameMemory
from .skills.contracts import (InputActionLimitExceeded, SkillDeadlineExceeded,
                               SkillStatus)
from .skills.menu import classify_menu_result, normalize_menu_label
from .ui_grounding import (button_candidates, has_profile_menu_layout,
                           first_world_row, stacked_main_menu_layout,
                           world_selection_cues)

OUTCOME_SUCCESS_CODES = frozenset({
    "SUCCESS_VERIFIED", "TARGET_DETECTED", "APPROACH_REACHED", "CONDITION_MET",
})
CAMERA_DEPENDENT_SKILLS = frozenset({"approach_target", "search_target", "turn"})
MENU_UNSAFE_SKILLS = frozenset({"interact_until_change", "move_pulse", "turn",
                                "approach_target", "search_target"})
MINING_PREFLIGHT_STABLE_SCREEN_DELTA = 0.02


def _verified_example(args_json, expected_result, success_check):
    """Return only recipe data that satisfies the planner's visible-check contract."""
    if (not isinstance(expected_result, str) or not expected_result.strip()
            or len(expected_result) > 300 or not isinstance(success_check, str)
            or not success_check.strip() or len(success_check) > 300):
        return None
    try:
        args = json.loads(args_json)
    except (TypeError, ValueError):
        return None
    if not isinstance(args, dict):
        return None
    return {"args": args, "expected_result": expected_result,
            "success_check": success_check}


class PlannedStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    skill: str = Field(min_length=1, max_length=80)
    args: dict
    expected_result: str = Field(min_length=1, max_length=300)
    success_check: str = Field(min_length=1, max_length=300)


class GoalPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    steps: list[PlannedStep] = Field(min_length=1, max_length=6)


class PlanningError(RuntimeError):
    """A plan could not be obtained and validated; execution must not begin."""


def skill_time_remaining(deadline, clock=time.monotonic):
    remaining = deadline - clock()
    if remaining <= 0:
        raise SkillDeadlineExceeded("skill time budget exhausted")
    return remaining


class DeadlineControls:
    """Clamp held-input durations to the remaining skill/run deadline."""
    def __init__(self, controls, deadline, clock=time.monotonic):
        self.controls, self.deadline, self.clock = controls, deadline, clock

    def _duration(self, requested):
        remaining = skill_time_remaining(self.deadline, self.clock)
        return min(float(requested), remaining)

    def press_key(self, key, duration):
        return self.controls.press_key(key, self._duration(duration))

    def press_keys(self, keys, duration):
        return self.controls.press_keys(keys, self._duration(duration))

    def hold_mouse(self, button, duration):
        return self.controls.hold_mouse(button=button, duration=self._duration(duration))

    def click(self, x=None, y=None, button="left", hold_time=0.08):
        return self.controls.click(x=x, y=y, button=button,
                                   hold_time=self._duration(hold_time))

    def move_camera(self, dx, dy):
        skill_time_remaining(self.deadline, self.clock)
        return self.controls.move_camera(dx, dy)

    def release_all(self):
        return self.controls.release_all()


class InputActionBudget:
    def __init__(self, max_actions):
        self.max_actions = max(1, int(max_actions))
        self.actions_used = 0

    @property
    def exhausted(self):
        return self.actions_used >= self.max_actions

    def consume(self):
        if self.exhausted:
            raise InputActionLimitExceeded("input action limit reached",
                                           actions_used=self.actions_used)
        self.actions_used += 1


class BudgetControls:
    """Count input operations against one shared attempt-level budget."""
    def __init__(self, controls, budget):
        self.controls, self.budget = controls, budget

    def __getattr__(self, name):
        return getattr(self.controls, name)

    def press_key(self, key, duration):
        self.budget.consume()
        return self.controls.press_key(key, duration)

    def press_keys(self, keys, duration):
        self.budget.consume()
        return self.controls.press_keys(keys, duration)

    def hold_mouse(self, button, duration):
        self.budget.consume()
        return self.controls.hold_mouse(button=button, duration=duration)

    def click(self, x=None, y=None, button="left", hold_time=0.08):
        self.budget.consume()
        return self.controls.click(x=x, y=y, button=button, hold_time=hold_time)

    def move_camera(self, dx, dy):
        self.budget.consume()
        return self.controls.move_camera(dx, dy)

    def release_all(self):
        return self.controls.release_all()


class DeadlineDetector:
    """Pass the remaining deadline through to HTTP request timeouts."""
    def __init__(self, detector, deadline, clock=time.monotonic):
        self.detector, self.deadline, self.clock = detector, deadline, clock

    def __getattr__(self, name):
        return getattr(self.detector, name)

    def detect(self, image, queries, score_min=None, max_boxes=None):
        remaining = skill_time_remaining(self.deadline, self.clock)
        return self.detector.detect(image, queries, score_min=score_min, max_boxes=max_boxes,
                                    timeout=remaining)


class SkillOutcomeStore:
    """Small SQLite store for episode-level skill success rates, isolated by game_id."""
    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self._memory_connection = sqlite3.connect(":memory:") if str(self.db_path) == ":memory:" else None
        if self._memory_connection is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS skill_outcomes (
                id INTEGER PRIMARY KEY, game_id TEXT NOT NULL, skill TEXT NOT NULL,
                target TEXT NOT NULL, result TEXT NOT NULL, duration REAL NOT NULL,
                ts REAL NOT NULL, run_id TEXT NOT NULL)""")
            conn.execute("CREATE INDEX IF NOT EXISTS ix_skill_outcomes_lookup "
                         "ON skill_outcomes(game_id,skill,target)")
            columns = {row[1] for row in conn.execute("PRAGMA table_info(skill_outcomes)")}
            for name, sql_type in (("args_json", "TEXT"), ("expected_result", "TEXT"),
                                   ("success_check", "TEXT"), ("state_hash", "TEXT")):
                if name not in columns:
                    conn.execute(f"ALTER TABLE skill_outcomes ADD COLUMN {name} {sql_type}")
            conn.execute("CREATE INDEX IF NOT EXISTS ix_skill_outcomes_state "
                         "ON skill_outcomes(game_id,state_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS ix_skill_outcomes_target "
                         "ON skill_outcomes(game_id,target,id)")
            conn.commit()

    @contextmanager
    def _connection(self):
        if self._memory_connection is not None:
            yield self._memory_connection
        else:
            with closing(sqlite3.connect(self.db_path)) as conn:
                yield conn

    def record(self, game_id, skill, target, result, duration, run_id, *, args=None,
               expected_result="", success_check="", state_hash=None):
        args_json = json.dumps(args, ensure_ascii=False, sort_keys=True) if args is not None else None
        with self._connection() as conn:
            conn.execute("INSERT INTO skill_outcomes(game_id,skill,target,result,duration,ts,run_id,"
                         "args_json,expected_result,success_check,state_hash) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                         (game_id, skill, target, result, max(0.0, float(duration)), time.time(),
                          run_id, args_json, str(expected_result or ""), str(success_check or ""),
                          state_hash))
            conn.commit()

    def summary(self, game_id, limit=200, max_groups=20, *, state_hash=None, target=None):
        with self._connection() as conn:
            columns = ("SELECT skill,target,result,args_json,expected_result,success_check,state_hash "
                       "FROM skill_outcomes WHERE game_id=? ")
            if target is None:
                rows = conn.execute(columns + "ORDER BY id DESC LIMIT ?", (game_id, limit)).fetchall()
            else:
                # Keep retrieval bounded while preserving the active goal's
                # history when unrelated outcomes push it past the global window.
                rows = conn.execute(
                    columns + "AND (id IN (SELECT id FROM skill_outcomes WHERE game_id=? "
                    "ORDER BY id DESC LIMIT ?) OR id IN (SELECT id FROM skill_outcomes "
                    "WHERE game_id=? AND target=? ORDER BY id DESC LIMIT ?)) ORDER BY id DESC",
                    (game_id, game_id, limit, game_id, target, limit),
                ).fetchall()
        grouped = {}
        state_examples = {}
        for skill, row_target, result, args_json, expected_result, success_check, example_state_hash in rows:
            item = grouped.setdefault((skill, row_target), [0, 0, 0, None, False])
            item[1] += 1
            is_success = result in OUTCOME_SUCCESS_CODES
            item[0] += is_success
            item[2] += result in {SkillStatus.UNCONFIRMED.value, "SUCCESS_REGION_CHANGE",
                                  "REGION_CHANGE_UNCONFIRMED", "MENU_TARGET_UNCONFIRMED"}
            # Rows are newest first. A later failure, uncertain attempt, or generic
            # command success invalidates the old recipe until a fresh verified result.
            if state_hash is None and not item[4]:
                item[4] = True
                if result not in OUTCOME_SUCCESS_CODES or not args_json:
                    continue
                item[3] = _verified_example(args_json, expected_result, success_check)
            elif (state_hash is not None and example_state_hash
                  and GameMemory._hash_distance(state_hash, example_state_hash) <= 4):
                state_item = state_examples.setdefault((skill, row_target), [False, None])
                if state_item[0]:
                    continue
                state_item[0] = True
                if result not in OUTCOME_SUCCESS_CODES or not args_json:
                    continue
                state_item[1] = _verified_example(args_json, expected_result, success_check)
        # ``rows`` is newest-first, so the first time a group is inserted into
        # ``grouped`` is its latest outcome. Preserve that order when limiting
        # prompt context; alphabetical truncation can discard fresh recipes.
        ordered_groups = list(grouped.items())
        if target is not None:
            # Keep recipes for the active goal ahead of unrelated recent goals.
            # Python's stable sort preserves newest-first order within each tier.
            ordered_groups.sort(key=lambda entry: entry[0][1] != target)
        summaries = [{"skill": skill, "target": group_target, "successes": values[0], "attempts": values[1],
                      "uncertain": values[2], "success_rate": round(values[0] / values[1], 3),
                      "verified_example": (state_examples.get((skill, group_target), [False, None])[1]
                                           if state_hash is not None else values[3])}
                     for (skill, group_target), values in ordered_groups]
        return summaries[:max(1, int(max_groups))]


class GoalPlanner:
    def __init__(self, registry, request_plan, max_validation_retries=None):
        self.registry = registry
        self.request_plan = request_plan
        self.max_validation_retries = (config.goal_validation_retries if max_validation_retries is None
                                       else max_validation_retries)
        self.last_plan_metrics = {"duration_seconds": 0.0, "request_durations_seconds": []}

    def _profile_catalog(self, profile, menu_visible=False):
        catalog = deepcopy(self.registry.planner_catalog())
        if not profile:
            return catalog
        if menu_visible:
            catalog = [skill for skill in catalog if skill["name"] not in MENU_UNSAFE_SKILLS]
        if profile.camera_type == "none" or profile.input_backend == "window":
            catalog = [skill for skill in catalog
                       if skill["name"] not in CAMERA_DEPENDENT_SKILLS]
        if not profile.ui_adapters:
            catalog = [skill for skill in catalog if skill["name"] != "click_menu_target"]
        if not profile.actions or not profile.regions:
            catalog = [skill for skill in catalog if skill["name"] != "interact_until_change"]
        for skill in catalog:
            properties = skill.get("parameters", {}).get("properties", {})
            if skill["name"] == "interact_until_change":
                if profile.regions:
                    properties["region"]["enum"] = sorted(profile.regions)
                if profile.actions:
                    properties["action"]["enum"] = sorted(profile.actions)
            elif skill["name"] == "wait_for_condition" and profile.regions:
                properties["region"]["enum"] = sorted(profile.regions)
        return catalog

    @staticmethod
    def _goal_explicitly_requests_world_creation(goal):
        words = set(normalize_menu_label(goal).split())
        creation = words & {"create", "creating", "creation", "создай", "создать",
                            "создайте", "создание", "создания", "создании", "созданию",
                            "new", "новый", "нового", "новую", "новом"}
        world = words & {"world", "мир", "мира", "мире", "миров"}
        return bool(creation and world)

    def validate(self, raw, profile=None, menu_visible=False, frame=None, goal=""):
        plan = raw if isinstance(raw, GoalPlan) else GoalPlan.model_validate(raw)
        explicit_world_creation = self._goal_explicitly_requests_world_creation(goal)
        main_menu_layout = bool(
            profile and frame is not None
            and "world_selection" in (profile.ui_adapters or ())
            and "stacked_main_menu" in (profile.ui_adapters or ())
            and stacked_main_menu_layout(button_candidates(frame)))
        world_selection_next = None
        if (profile and frame is not None
                and "world_selection" in (profile.ui_adapters or ())):
            selection, play_enabled = world_selection_cues(frame)
            if selection is False and play_enabled is False and first_world_row(frame):
                world_selection_next = "select_existing_world"
            elif selection is True and play_enabled is True:
                world_selection_next = "start_selected_world"
        menu_targets = {
            normalize_menu_label(label)
            for label in (getattr(profile, "main_menu_target_transitions", {}) or {})
        }
        if menu_targets:
            for index, step in enumerate(plan.steps[:-1]):
                if step.skill != "wait_for_condition":
                    continue
                wait_args = self.registry.get(step.skill).validate_args(step.args).model_dump()
                if (not wait_args.get("present", True)
                        or normalize_menu_label(wait_args.get("query", "")) not in menu_targets):
                    continue
                if any(later.skill in MENU_UNSAFE_SKILLS for later in plan.steps[index + 1:]):
                    raise ValueError(
                        "A visible configured menu target was detected by the plan; "
                        "gameplay skills cannot follow it without a fresh menu-grounded plan")
        validated = []
        for step_index, step in enumerate(plan.steps):
            if menu_visible and step.skill in MENU_UNSAFE_SKILLS:
                raise ValueError(f"Skill {step.skill!r} cannot operate a recognized menu screen")
            if (profile and (profile.camera_type == "none" or profile.input_backend == "window")
                    and step.skill in CAMERA_DEPENDENT_SKILLS):
                raise ValueError(f"Skill {step.skill!r} requires a camera; relative camera input is unavailable in profile "
                                 f"{profile.game_id!r}")
            spec = self.registry.get(step.skill)
            params = spec.validate_args(step.args)
            values = params.model_dump()
            if step.skill == "click_menu_target":
                if not profile or not profile.ui_adapters:
                    raise ValueError("Skill 'click_menu_target' requires a profile UI adapter")
                target_label = normalize_menu_label(values["target_label"])
                result_class = classify_menu_result(values["expected_result"])
                if (step_index == 0 and world_selection_next == "select_existing_world"
                        and not explicit_world_creation
                        and (target_label != "visible world row" or result_class != "world_selected"
                             or self._goal_explicitly_requests_world_creation(values["intent"]))):
                    values["target_label"] = "visible_world_row"
                    values["intent"] = "Select the visible existing world before starting it"
                    values["expected_result"] = "world_selected"
                    step = step.model_copy(update={
                        "args": values,
                        "expected_result": "world_selected",
                        "success_check": "The existing world is visibly selected",
                    })
                    target_label = normalize_menu_label(values["target_label"])
                    result_class = "world_selected"
                if (step_index == 0 and world_selection_next == "start_selected_world"
                        and not explicit_world_creation
                        and (target_label != "play selected world" or result_class != "gameplay"
                             or self._goal_explicitly_requests_world_creation(values["intent"]))):
                    values["target_label"] = "Play selected world"
                    values["intent"] = "Start playing the selected world"
                    values["expected_result"] = "gameplay"
                    step = step.model_copy(update={
                        "args": values,
                        "expected_result": "gameplay",
                        "success_check": "Gameplay is visible",
                    })
                if main_menu_layout:
                    mapped_transition = next((transition for label, transition in
                                              profile.main_menu_target_transitions.items()
                                              if normalize_menu_label(label) == target_label), None)
                    if menu_targets and mapped_transition is None:
                        raise ValueError(
                            "click_menu_target from the recognized stacked main menu must use a configured target transition")
                    if mapped_transition == "world_select":
                        canonical_result = "World selection screen appears"
                        canonical_check = "World selection screen is visible"
                        values["expected_result"] = canonical_result
                        step = step.model_copy(update={
                            "args": {**step.args, "expected_result": canonical_result},
                            "expected_result": canonical_result,
                            "success_check": canonical_check,
                        })
                if values["expected_result"].strip().casefold() != step.expected_result.strip().casefold():
                    raise ValueError("click_menu_target expected_result must match the step postcondition")
                if (main_menu_layout
                        and classify_menu_result(values["expected_result"]) != "world_select"):
                    raise ValueError(
                        "click_menu_target from the recognized stacked main menu must expect the immediate world-selection screen")
            if profile and step.skill == "interact_until_change":
                if not profile.actions or not profile.regions:
                    raise ValueError(f"Skill 'interact_until_change' requires configured actions and regions "
                                     f"in profile {profile.game_id!r}")
                if values["action"] not in profile.actions:
                    raise ValueError(f"Action {values['action']!r} is not configured in profile {profile.game_id!r}")
                if values["region"] not in profile.regions:
                    raise ValueError(f"Region {values['region']!r} is not configured in profile {profile.game_id!r}")
            if (profile and step.skill == "wait_for_condition" and values.get("region")
                    and values["region"] not in profile.regions):
                raise ValueError(f"Region {values['region']!r} is not configured in profile {profile.game_id!r}")
            validated.append(step)
        return GoalPlan(steps=validated)

    @staticmethod
    def _canonical_args(skill, args, profile=None):
        """Normalize detector aliases so wording changes do not hide a repeated strategy."""
        values = dict(args)
        field = "target_text" if "target_text" in values else "query" if "query" in values else None
        if (profile is None or field is None or
                skill not in {"search_target", "approach_target", "wait_for_condition"}):
            return values

        def normalize(text):
            normalized = " ".join(str(text).casefold().split()).strip(" .,!?:;")
            for article in ("a ", "an ", "the "):
                if normalized.startswith(article):
                    return normalized[len(article):]
            return normalized

        target = normalize(values[field])
        values[field] = target
        for label, synonyms in (profile.detector_queries or {}).items():
            canonical = normalize(label.replace("_", " "))
            aliases = {normalize(alias) for alias in synonyms}
            aliases.add(canonical)
            if target in aliases:
                values[field] = f"@detector:{canonical}"
                break
        return values

    def signature(self, plan, profile=None):
        """Return skill/argument pairs with profile detector aliases collapsed."""
        return tuple((step.skill, json.dumps(self._canonical_args(
            step.skill,
            self.registry.get(step.skill).validate_args(step.args).model_dump(mode="json"),
            profile), sort_keys=True, separators=(",", ":"))) for step in plan.steps)

    def plan(self, *, frame, goal, profile, outcome_summary=(), failure="", timeout_seconds=None,
             avoid_plan=None, failed_step=None):
        error = ""
        plan_started = time.monotonic()
        request_durations = []
        deadline = None if timeout_seconds is None else time.monotonic() + max(0.0, timeout_seconds)
        menu_visible = has_profile_menu_layout(frame, profile)
        try:
            for attempt in range(self.max_validation_retries + 1):
                request = dict(frame=frame, goal=goal, profile=profile,
                               skills=self._profile_catalog(profile, menu_visible=menu_visible),
                               outcome_summary=list(outcome_summary),
                               validation_error=error if attempt else failure)
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise PlanningError("Goal time budget exhausted before a valid plan was returned")
                    request["timeout_seconds"] = remaining
                request_started = time.monotonic()
                try:
                    raw = self.request_plan(**request)
                except InterruptedError:
                    raise
                except Exception as exc:
                    detail = f"{type(exc).__name__}: {exc}"[:300]
                    raise PlanningError(f"Goal planner request failed: {detail}") from exc
                finally:
                    request_durations.append(max(0.0, time.monotonic() - request_started))
                try:
                    plan = self.validate(raw, profile=profile, menu_visible=menu_visible, frame=frame,
                                          goal=goal)
                    signature = self.signature(plan, profile=profile)
                    if avoid_plan is not None and signature == avoid_plan:
                        raise ValueError("Plan exactly repeats the previous failed plan; choose a different strategy.")
                    if failed_step is not None and plan.steps:
                        failed_name, failed_args = failed_step
                        first = plan.steps[0]
                        first_args = self.registry.get(first.skill).validate_args(first.args).model_dump(mode="json")
                        if (first.skill == failed_name and self._canonical_args(
                                first.skill, first_args, profile) == self._canonical_args(
                                failed_name, failed_args, profile)):
                            raise ValueError("Plan starts by repeating the immediately failed skill with identical arguments.")
                    return plan
                except (ValidationError, ValueError, TypeError) as exc:
                    error = str(exc)
            raise PlanningError(f"Planner returned an invalid plan after {self.max_validation_retries + 1} request(s): {error}")
        finally:
            self.last_plan_metrics = {
                "duration_seconds": max(0.0, time.monotonic() - plan_started),
                "request_durations_seconds": request_durations,
            }


class GoalExecutor:
    """Execute validated steps with global action/time caps and bounded replanning."""
    def __init__(self, planner, registry, outcome_store, *, profile, game_id,
                 execute_skill, frame_provider, stop_check, dry_run=False,
                 max_replans=None, max_steps=None, input_action_budget=None,
                 run_id=None, clock=time.monotonic, result_verifier=None,
                 logger=None, detector=None, menu_preview=None, skill_preflight=None,
                 skill_context=None, vlm_request_counter=None):
        self.planner, self.registry, self.store = planner, registry, outcome_store
        self.profile, self.game_id = profile, game_id
        self.execute_skill, self.frame_provider, self.stop_check = execute_skill, frame_provider, stop_check
        self.dry_run = dry_run
        self.max_replans = config.goal_max_replans if max_replans is None else max_replans
        self.max_steps = config.goal_max_skill_steps if max_steps is None else max_steps
        self.input_action_budget = input_action_budget
        self.result_verifier = result_verifier
        self.run_id, self.clock, self.logger, self.detector = run_id or uuid4().hex, clock, logger, detector
        self.menu_preview = menu_preview
        self.skill_preflight = skill_preflight
        self.skill_context = skill_context
        self.vlm_request_counter = vlm_request_counter
        self.memory_reuse_count = 0
        self._detector_calls_start = None
        self._vlm_requests_start = None

    def _detector_call_count(self):
        count = getattr(self.detector, "detect_requests", None)
        return count if isinstance(count, int) and not isinstance(count, bool) and count >= 0 else None

    def _vlm_request_count(self):
        if self.vlm_request_counter is None:
            return None
        count = self.vlm_request_counter()
        return count if isinstance(count, int) and not isinstance(count, bool) and count >= 0 else None

    @staticmethod
    def _handoff_target_seed(search_result, target_text):
        """Return a search seed only when a successful result matches the next target."""
        if not isinstance(search_result, dict) or search_result.get("status") != SkillStatus.SUCCESS.value:
            return None
        evidence = search_result.get("evidence")
        if not isinstance(evidence, dict):
            return None
        normalize = lambda text: " ".join(str(text or "").casefold().split())
        if not normalize(evidence.get("target_text")) or normalize(evidence.get("target_text")) != normalize(target_text):
            return None
        seed = evidence.get("target_seed")
        return dict(seed) if isinstance(seed, dict) else None

    def _requires_mining_preflight(self, step, params):
        if step.skill != "interact_until_change" or self.profile is None:
            return False
        if not (getattr(self.profile, "mine_requires_grounded_level_view", False)
                or getattr(self.profile, "mine_requires_target_at_crosshair", False)):
            return False
        mine_actions = set(getattr(self.profile, "mine_actions", ()) or ())
        return params.get("action") in mine_actions

    def _check_mining_preflight(self, goal, step, frame, timeout_seconds):
        """Require fresh, confident profile-specific visual facts before mining input."""
        started = self.clock()
        if self.skill_preflight is None:
            return ({"status": SkillStatus.UNCONFIRMED.value,
                     "result_code": "MINING_PREFLIGHT_UNAVAILABLE",
                     "reason": "mining_preflight_callback_unavailable",
                     "evidence": {"failure_stage": "mining_preflight",
                                  "postcondition": "not_checked_no_input"},
                     "duration_seconds": 0.0}, None, None)
        try:
            assessment = self.skill_preflight(goal, step, frame, timeout_seconds)
        except InterruptedError as exc:
            return ({"status": SkillStatus.ABORT.value, "result_code": "ABORT",
                     "reason": str(exc)[:300],
                     "evidence": {"failure_stage": "mining_preflight",
                                  "postcondition": "not_checked_no_input"},
                     "duration_seconds": max(0.0, self.clock() - started)}, None, None)
        except TimeoutError as exc:
            return ({"status": SkillStatus.UNCONFIRMED.value,
                     "result_code": "MINING_PREFLIGHT_TIMEOUT",
                     "reason": f"{type(exc).__name__}: {exc}"[:300],
                     "evidence": {"failure_stage": "mining_preflight",
                                  "postcondition": "not_checked_no_input"},
                     "duration_seconds": max(0.0, self.clock() - started)}, None, None)
        except Exception as exc:
            return ({"status": SkillStatus.UNCONFIRMED.value,
                     "result_code": "MINING_PREFLIGHT_ERROR",
                     "reason": f"{type(exc).__name__}: {exc}"[:300],
                     "evidence": {"failure_stage": "mining_preflight",
                                  "postcondition": "not_checked_no_input"},
                     "duration_seconds": max(0.0, self.clock() - started)}, None, None)

        def value(source, name, default=None):
            if isinstance(source, dict):
                return source.get(name, default)
            return getattr(source, name, default)

        after = value(assessment, "after")
        try:
            confidence = float(value(assessment, "confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        verdict = value(assessment, "result", "unknown")
        usable = bool(value(assessment, "usable", False))
        evidence = {"failure_stage": "mining_preflight",
                    "confidence": confidence, "assessment_result": verdict,
                    "usable": usable,
                    "evidence": str(value(assessment, "evidence", ""))[:500],
                    "uncertainty": str(value(assessment, "uncertainty", ""))[:300],
                    "facts": {name: value(after, name) for name in
                              ("scene", "pitch", "safe_footing", "target_at_crosshair")}}
        if (not usable or not math.isfinite(confidence) or confidence < 0.85
                or verdict == "regressed" or after is None):
            return ({"status": SkillStatus.UNCONFIRMED.value,
                     "result_code": "MINING_PREFLIGHT_UNCONFIRMED",
                     "reason": "mining_safety_facts_not_confident",
                     "evidence": {**evidence, "postcondition": "not_checked_no_input"},
                     "duration_seconds": max(0.0, self.clock() - started)}, None, evidence)

        requirements = []
        blocked, unknown = [], []
        if (getattr(self.profile, "mine_requires_grounded_level_view", False)
                or getattr(self.profile, "mine_requires_target_at_crosshair", False)):
            scene = value(after, "scene", "unknown")
            if scene != "gameplay":
                (blocked if scene in {"menu", "loading"} else unknown).append("scene_not_gameplay")
        if getattr(self.profile, "mine_requires_grounded_level_view", False):
            requirements.extend(("pitch_level", "safe_footing"))
            pitch, footing = value(after, "pitch"), value(after, "safe_footing")
            if pitch != "level":
                (blocked if pitch in {"up", "down"} else unknown).append("pitch_not_level")
            if footing is not True:
                (blocked if footing is False else unknown).append("safe_footing_unconfirmed")
        if getattr(self.profile, "mine_requires_target_at_crosshair", False):
            requirements.append("target_at_crosshair")
            target_at_crosshair = value(after, "target_at_crosshair")
            if target_at_crosshair is not True:
                (blocked if target_at_crosshair is False else unknown).append(
                    "target_at_crosshair_unconfirmed")

        evidence.update(requirements=requirements, blocked_conditions=blocked,
                        unknown_conditions=unknown)
        if blocked or unknown:
            status = SkillStatus.FAILED.value if blocked else SkillStatus.UNCONFIRMED.value
            code = "MINING_PREFLIGHT_BLOCKED" if blocked else "MINING_PREFLIGHT_UNCONFIRMED"
            reason = ",".join(blocked + unknown)
            return ({"status": status, "result_code": code, "reason": reason,
                     "evidence": {**evidence, "postcondition": "not_checked_no_input"},
                     "duration_seconds": max(0.0, self.clock() - started)}, None, evidence)

        if not hasattr(frame, "convert"):
            return ({"status": SkillStatus.UNCONFIRMED.value,
                     "result_code": "MINING_PREFLIGHT_UNCONFIRMED",
                     "reason": "preflight_frame_unavailable_for_staleness_check",
                     "evidence": {**evidence, "postcondition": "not_checked_no_input"},
                     "duration_seconds": max(0.0, self.clock() - started)}, None, evidence)
        fresh_frame = self.frame_provider()
        if not hasattr(fresh_frame, "convert"):
            return ({"status": SkillStatus.UNCONFIRMED.value,
                     "result_code": "MINING_PREFLIGHT_UNCONFIRMED",
                     "reason": "fresh_frame_unavailable_after_preflight",
                     "evidence": {**evidence, "postcondition": "not_checked_no_input"},
                     "duration_seconds": max(0.0, self.clock() - started)}, None, evidence)
        screen_delta = GameMemory.compute_screen_delta(frame, fresh_frame)
        evidence["screen_delta_after_preflight"] = screen_delta
        evidence["screen_delta_limit"] = MINING_PREFLIGHT_STABLE_SCREEN_DELTA
        if screen_delta >= MINING_PREFLIGHT_STABLE_SCREEN_DELTA:
            return ({"status": SkillStatus.UNCONFIRMED.value,
                     "result_code": "STALE_MINING_PREFLIGHT",
                     "reason": "screen_changed_during_mining_preflight",
                     "evidence": {**evidence, "postcondition": "not_checked_no_input"},
                     "duration_seconds": max(0.0, self.clock() - started)}, None, evidence)
        return None, fresh_frame, evidence

    @staticmethod
    def _memory_reuse_metrics(goal, outcomes, plan):
        """Count exact skill-argument reuse of a latest verified example for this goal."""
        verified_examples = {}
        for outcome in outcomes:
            example = outcome.get("verified_example")
            if outcome.get("target") != goal or not isinstance(example, dict):
                continue
            args = example.get("args")
            if isinstance(args, dict):
                signature = json.dumps(args, sort_keys=True, separators=(",", ":"))
                verified_examples[(outcome.get("skill"), signature)] = True
        reused = 0
        for step in plan.steps:
            args = step.model_dump(mode="json")["args"]
            signature = json.dumps(args, sort_keys=True, separators=(",", ":"))
            reused += (step.skill, signature) in verified_examples
        return len(verified_examples), reused

    def _verify_result(self, step, result, before, after, timeout_seconds, *, force=False, scope=None):
        """Verify interaction changes, intermediate transitions, and final postconditions."""
        result_code = result.get("result_code")
        needs_check = result_code in {
            "REGION_CHANGE_UNCONFIRMED", "SUCCESS_REGION_CHANGE",
            "INTERACTION_OUTCOME_UNCONFIRMED", "MENU_ACTION_UNCONFIRMED"}
        if not force and not needs_check:
            return result
        verification_scope = scope or ("menu_transition" if result_code == "MENU_ACTION_UNCONFIRMED" else
                              "goal_postcondition" if force else
                              "interaction_outcome" if result_code == "INTERACTION_OUTCOME_UNCONFIRMED" else
                              "region_change")
        if force and result.get("status") != SkillStatus.SUCCESS.value:
            return result
        if self.result_verifier is None:
            if not force:
                return {**result, "status": SkillStatus.UNCONFIRMED.value,
                        "result_code": "UNCONFIRMED", "reason": "interaction_without_success_check",
                        "evidence": {**(result.get("evidence") or {}), "verification": {
                            "scope": verification_scope, "result": "unknown", "source": "no_verifier"}}}
            return {**result, "status": SkillStatus.UNCONFIRMED.value,
                    "result_code": "UNCONFIRMED", "reason": "goal_result_unverified",
                    "evidence": {**(result.get("evidence") or {}), "verification": {
                        "scope": verification_scope, "result": "unknown", "source": "no_verifier"}}}
        evidence = dict(result.get("evidence") or {})
        if timeout_seconds <= 0:
            return {**result, "status": SkillStatus.UNCONFIRMED.value,
                    "result_code": "UNCONFIRMED", "reason": "success_check_time_budget_exhausted",
                    "evidence": {**evidence, "verification": {"result": "unknown",
                        "scope": verification_scope, "source": "time_budget"}}}
        verify_started = self.clock()
        try:
            assessment = self.result_verifier(step, before, after, result, timeout_seconds)
            verify_duration = max(0.0, self.clock() - verify_started)
            usable = bool(getattr(assessment, "usable", False))
            verdict = getattr(assessment, "result", "unknown")
            confidence = float(getattr(assessment, "confidence", 0.0))
            proof = str(getattr(assessment, "evidence", ""))
            verification = {"scope": verification_scope, "result": verdict, "confidence": confidence,
                            "evidence": proof,
                            "evidence_source": getattr(assessment, "evidence_source", "unknown"),
                            "usable": usable, "duration_seconds": verify_duration}
            total_duration = float(result.get("duration_seconds", 0.0)) + verify_duration
            if verify_duration > timeout_seconds:
                return {**result, "status": SkillStatus.UNCONFIRMED.value,
                        "result_code": "UNCONFIRMED", "reason": "success_check_time_budget_exhausted",
                        "duration_seconds": total_duration,
                        "evidence": {**evidence, "verification": verification}}
            if verification_scope == "menu_transition" or step.skill == "click_menu_target":
                menu_action_succeeded = getattr(assessment, "menu_action_succeeded", None)
                verification["menu_action_succeeded"] = menu_action_succeeded
                if usable and menu_action_succeeded is False:
                    return {**result, "status": SkillStatus.FAILED.value,
                            "result_code": "VERIFICATION_FAILED", "reason": "menu_action_not_succeeded",
                            "duration_seconds": total_duration,
                            "evidence": {**evidence, "verification": verification}}
                if not usable or menu_action_succeeded is not True:
                    return {**result, "status": SkillStatus.UNCONFIRMED.value,
                            "result_code": "UNCONFIRMED", "reason": "menu_action_unconfirmed",
                            "duration_seconds": total_duration,
                            "evidence": {**evidence, "verification": verification}}
            if usable and verdict == "progress":
                return {**result, "status": SkillStatus.SUCCESS.value,
                        "result_code": "SUCCESS_VERIFIED", "reason": "success_check_confirmed",
                        "duration_seconds": total_duration,
                        "evidence": {**evidence, "verification": verification}}
            if usable and verdict in {"blocked", "regressed"}:
                return {**result, "status": SkillStatus.FAILED.value,
                        "result_code": "VERIFICATION_FAILED", "reason": verdict,
                        "duration_seconds": total_duration,
                        "evidence": {**evidence, "verification": verification}}
            return {**result, "status": SkillStatus.UNCONFIRMED.value,
                    "result_code": "UNCONFIRMED", "reason": "success_check_uncertain",
                    "duration_seconds": total_duration,
                    "evidence": {**evidence, "verification": verification}}
        except InterruptedError as exc:
            return {**result, "status": "ABORT", "result_code": "ABORT",
                    "reason": str(exc)[:300],
                    "duration_seconds": float(result.get("duration_seconds", 0.0)) +
                        max(0.0, self.clock() - verify_started),
                    "evidence": {**evidence, "postcondition": "not_checked_emergency_stop",
                        "verification": {"result": "unknown", "scope": verification_scope,
                                         "source": "interrupted"}}}
        except Exception as exc:
            return {**result, "status": SkillStatus.UNCONFIRMED.value,
                    "result_code": "UNCONFIRMED", "reason": "success_check_error",
                    "duration_seconds": float(result.get("duration_seconds", 0.0)) +
                        max(0.0, self.clock() - verify_started),
                    "evidence": {**evidence, "verification": {"result": "unknown",
                        "scope": verification_scope, "source": "error",
                        "error": f"{type(exc).__name__}: {exc}"[:300]}}}

    def _finish(self, result):
        result.setdefault("memory_reuse_count", self.memory_reuse_count)
        current_detector_calls = self._detector_call_count()
        result.setdefault("detector_calls",
                          max(0, current_detector_calls - self._detector_calls_start)
                          if current_detector_calls is not None
                          and self._detector_calls_start is not None else None)
        current_vlm_requests = self._vlm_request_count()
        result.setdefault("vlm_calls",
                          max(0, current_vlm_requests - self._vlm_requests_start)
                          if current_vlm_requests is not None
                          and self._vlm_requests_start is not None else None)
        if self.input_action_budget is not None:
            result.setdefault("input_actions_used", self.input_action_budget.actions_used)
            result.setdefault("input_actions_limit", self.input_action_budget.max_actions)
        if self.logger:
            self.logger({"event": "run_finish", **result})
        return result

    def _recover_blocked_approach(self, goal, target_text, started, steps_used):
        """Try one bounded camera sweep and a fresh approach before a slow VLM replan."""
        if self.dry_run:
            return None, 0, None, None
        turn_amount = min(60, self.profile.motion_limits.max_turn_counts)
        recovery = (
            ("turn", {"direction": "right", "amount": turn_amount}),
            ("search_target", {"target_text": target_text, "sweep": 2, "vertical": False}),
            ("approach_target", {"target_text": target_text}),
        )
        used = 0
        last_result = None
        last_skill = None
        previous_skill = None
        last_state_hash = None
        for name, args in recovery:
            self.stop_check()
            if self.input_action_budget is not None and self.input_action_budget.exhausted:
                return {"status": "FAILED", "result_code": "FAILED", "reason": "max_actions",
                        "evidence": {"recovery": "approach"}}, used, last_skill, last_state_hash
            if steps_used + used >= self.max_steps:
                return {"status": "FAILED", "result_code": "FAILED", "reason": "max_skill_steps",
                        "evidence": {"recovery": "approach"}}, used, last_skill, last_state_hash
            remaining = self.profile.motion_limits.max_run_seconds - (self.clock() - started)
            spec = self.registry.get(name)
            available = min(remaining, float(spec.timeout_seconds))
            if available <= 0:
                return {"status": "FAILED", "result_code": "FAILED", "reason": "max_run_seconds",
                        "evidence": {"recovery": "approach"}}, used, last_skill, last_state_hash
            last_state_hash = self._current_state_hash()
            params = spec.validate_args(args).model_dump(mode="json")
            begin = self.clock()
            last_skill = name
            skill_context = {}
            if name == "approach_target" and previous_skill == "search_target":
                seed = self._handoff_target_seed(last_result, target_text)
                if seed:
                    skill_context["approach_target_seed"] = seed
            last_result = self._execute_skill(spec, params, available, begin, skill_context)
            previous_skill = name
            duration = last_result.get("duration_seconds", max(0.0, self.clock() - begin))
            if name != "approach_target":
                self.store.record(self.game_id, name, goal,
                                  last_result.get("result_code") or last_result.get("status", "FAILED"),
                                  duration, self.run_id, args=params, state_hash=last_state_hash)
            if self.logger:
                self.logger({"event": "local_recovery_skill", "recovery": "approach",
                             "skill": name, "args": params, "result": last_result})
            used += 1
            if last_result.get("status") != SkillStatus.SUCCESS.value:
                return last_result, used, last_skill, last_state_hash
        return last_result, used, last_skill, last_state_hash

    def _current_state_hash(self):
        frame = self.frame_provider()
        return GameMemory.compute_state_hash(frame) if hasattr(frame, "convert") else None

    def _execute_skill(self, spec, params, available, begin, skill_context=None):
        if self.skill_context is not None:
            self.skill_context.clear()
            self.skill_context.update(skill_context or {})
        try:
            result = self.execute_skill(spec, params, available)
        except InterruptedError as exc:
            evidence = {"postcondition": "not_checked_emergency_stop"}
            if self.input_action_budget is not None:
                evidence["input_actions"] = self.input_action_budget.actions_used
            return {"status": "ABORT", "result_code": "ABORT",
                    "reason": str(exc)[:300], "evidence": evidence,
                    "duration_seconds": max(0.0, self.clock() - begin)}
        except InputActionLimitExceeded as exc:
            evidence = {"failure_stage": "input_action_budget",
                        "postcondition": "not_checked_budget_exhausted",
                        "input_actions": (self.input_action_budget.actions_used
                                          if self.input_action_budget is not None
                                          else exc.actions_used)}
            return {"status": SkillStatus.FAILED.value, "result_code": "INPUT_ACTION_LIMIT",
                    "reason": f"max_actions: {exc}"[:300], "evidence": evidence,
                    "duration_seconds": max(0.0, self.clock() - begin)}
        except SkillDeadlineExceeded as exc:
            evidence = {"failure_stage": "skill_deadline",
                        "postcondition": "not_checked_skill_timeout"}
            if self.input_action_budget is not None:
                evidence["input_actions"] = self.input_action_budget.actions_used
            return {"status": SkillStatus.FAILED.value, "result_code": "SKILL_TIMEOUT",
                    "reason": f"skill_time_budget_exhausted: {exc}"[:300],
                    "evidence": evidence,
                    "duration_seconds": max(0.0, self.clock() - begin)}
        finally:
            if self.skill_context is not None:
                self.skill_context.clear()
        return result.model_dump() if hasattr(result, "model_dump") else result

    def run(self, goal):
        self._steps_used = 0
        self._detector_calls_start = self._detector_call_count()
        self._vlm_requests_start = self._vlm_request_count()
        try:
            return self._run(goal)
        except InterruptedError as exc:
            return self._finish({"status": "ABORT", "reason": str(exc)[:300],
                                 "steps": self._steps_used})

    def _run(self, goal):
        started, steps_used, replans, failure = self.clock(), 0, 0, ""
        avoid_plan, failed_step = None, None
        while True:
            if self.input_action_budget is not None and self.input_action_budget.exhausted:
                return self._finish({"status": "FAILED", "reason": "max_actions", "steps": steps_used})
            self.stop_check()
            if self.clock() - started >= self.profile.motion_limits.max_run_seconds:
                return self._finish({"status": "FAILED", "reason": "max_run_seconds", "steps": steps_used})
            plan_budget = self.profile.motion_limits.max_run_seconds - (self.clock() - started)
            if plan_budget <= 0:
                return self._finish({"status": "FAILED", "reason": "max_run_seconds", "steps": steps_used})
            frame = self.frame_provider()
            state_hash = (GameMemory.compute_state_hash(frame)
                          if hasattr(frame, "convert") else None)
            outcome_summary = self.store.summary(self.game_id, state_hash=state_hash, target=goal)
            try:
                plan = self.planner.plan(frame=frame, goal=goal, profile=self.profile,
                                         outcome_summary=outcome_summary, failure=failure,
                                         timeout_seconds=plan_budget, avoid_plan=avoid_plan,
                                         failed_step=failed_step)
            except PlanningError as exc:
                prior_failure = f"; previous_failure={failure}" if failure else ""
                if self.logger:
                    self.logger({"event": "plan_rejected", "reason": str(exc),
                                 **self.planner.last_plan_metrics})
                return self._finish({"status": "FAILED",
                                     "reason": f"planner_failed{prior_failure}: {exc}",
                                     "steps": steps_used})
            verified_example_count, memory_reuse_count = self._memory_reuse_metrics(
                goal, outcome_summary, plan)
            self.memory_reuse_count += memory_reuse_count
            current_plan_signature = self.planner.signature(plan, profile=self.profile)
            if self.logger:
                self.logger({"event": "plan", "replan_index": replans, "goal": goal,
                             "steps": [step.model_dump(mode="json") for step in plan.steps],
                             "verified_example_count": verified_example_count,
                             "memory_reuse_count": memory_reuse_count,
                             **self.planner.last_plan_metrics,
                             "request_count": len(self.planner.last_plan_metrics["request_durations_seconds"])})
            failure = ""
            previous_step = None
            previous_result = None
            for step_index, step in enumerate(plan.steps):
                is_final_step = step_index == len(plan.steps) - 1
                self.stop_check()
                if self.input_action_budget is not None and self.input_action_budget.exhausted:
                    return self._finish({"status": "FAILED", "reason": "max_actions", "steps": steps_used})
                if steps_used >= self.max_steps:
                    return self._finish({"status": "FAILED", "reason": "max_skill_steps", "steps": steps_used})
                if self.clock() - started >= self.profile.motion_limits.max_run_seconds:
                    return self._finish({"status": "FAILED", "reason": "max_run_seconds", "steps": steps_used})
                spec = self.registry.get(step.skill)
                begin = self.clock()
                step_frame = self.frame_provider()
                step_state_hash = (GameMemory.compute_state_hash(step_frame)
                                   if hasattr(step_frame, "convert") else None)
                remaining = max(0.0, self.profile.motion_limits.max_run_seconds - (begin - started))
                available = min(remaining, float(spec.timeout_seconds))
                params = spec.validate_args(step.args).model_dump(mode="json")
                if step.skill == "wait_for_condition":
                    if available < 0.1:
                        return self._finish({"status": "FAILED", "reason": "time_budget_exhausted",
                                             "steps": steps_used})
                    params["timeout"] = min(params["timeout"], available)
                menu_unsafe_after_plan = (
                    step.skill in MENU_UNSAFE_SKILLS
                    and has_profile_menu_layout(step_frame, self.profile))
                if menu_unsafe_after_plan:
                    result = {"status": SkillStatus.FAILED.value,
                              "result_code": "STALE_MENU_STATE",
                              "reason": "menu_appeared_after_plan; unsafe_skill_not_executed",
                              "evidence": {"failure_stage": "pre_skill_menu_recheck",
                                           "input_actions": 0,
                                           "postcondition": "not_checked_no_action_sent"},
                              "duration_seconds": max(0.0, self.clock() - begin)}
                    if self.logger:
                        self.logger({"event": "stale_menu_plan_rejected", "skill": step.skill,
                                     "reason": result["reason"]})
                elif self.dry_run:
                    resolved = {"skill": step.skill, "args": params}
                    preview_failed = False
                    if step.skill == "click_menu_target" and self.menu_preview is not None:
                        try:
                            preview = self.menu_preview(step, step_frame, available)
                            preview = preview.model_dump(mode="json") if hasattr(preview, "model_dump") else preview
                            if not isinstance(preview, dict) or preview.get("result_code") != "TARGET_GROUNDED_PREVIEW":
                                reason = preview.get("reason") if isinstance(preview, dict) else ""
                                raise ValueError(reason or "Menu preview did not return a grounded target")
                            resolved["grounded_target"] = preview
                        except InterruptedError as exc:
                            # Preserve the emergency-stop signal; an interrupted
                            # preview is an abort, not a localization failure.
                            preview_failed = True
                            result = {"status": "ABORT", "result_code": "ABORT",
                                      "reason": str(exc)[:300],
                                      "duration_seconds": max(0.0, self.clock() - begin)}
                            if self.logger:
                                self.logger({"event": "menu_preview_aborted", "skill": step.skill,
                                             "reason": result["reason"]})
                        except Exception as exc:
                            preview_failed = True
                            result = {"status": SkillStatus.UNCONFIRMED.value,
                                      "result_code": "MENU_TARGET_PREVIEW_FAILED",
                                      "reason": f"menu_target_not_grounded: {type(exc).__name__}: {exc}"[:300],
                                      "duration_seconds": max(0.0, self.clock() - begin)}
                            if self.logger:
                                self.logger({"event": "menu_preview_failed", "skill": step.skill,
                                             "reason": result["reason"]})
                    if step.skill == "move_pulse":
                        resolved["key"] = self.profile.move_keys[params["direction"]]
                    elif step.skill == "interact_until_change":
                        action = self.profile.actions.get(params["action"])
                        resolved["action"] = action.model_dump(mode="json") if action else None
                    if step.skill in {"approach_target", "search_target", "wait_for_condition"} and self.detector:
                        query = params.get("target_text") or params.get("query")
                        if query:
                            try:
                                from .skills.basic import _valid_detection, queries_for_target
                                queries = (queries_for_target(self.profile, query)
                                           if self.profile else [query])
                                score_min = (self.profile.thresholds.get("detector_score_min")
                                             if self.profile else None)
                                found = self.detector.detect(step_frame, queries,
                                                             score_min=score_min, max_boxes=10)
                                raw_boxes = found.get("boxes", []) if found else []
                                valid_boxes = (_valid_detection(found, self.profile)
                                               if found and self.profile else raw_boxes)
                                resolved["detector"] = {"query": query,
                                    "queries": queries,
                                    "available": found is not None,
                                    "box_count": len(raw_boxes),
                                    "candidate_box_count": len(valid_boxes)}
                            except Exception as exc:
                                resolved["detector"] = {"query": query, "available": False,
                                    "error": type(exc).__name__}
                    if not preview_failed:
                        result = {"status": "SUCCESS", "result_code": "WOULD_SEND",
                                  "reason": "dry_run", "evidence": {"would_send": {"skill": step.skill,
                                  "args": params, "resolved": resolved}}, "duration_seconds": 0.0}
                        if self.logger:
                            self.logger({"event": "would_send", **resolved})
                else:
                    verification_before = (step_frame if self.result_verifier and
                                           (step.skill in {"interact_until_change", "click_menu_target"} or is_final_step)
                                           else None)
                    skip_verification = False
                    if self._requires_mining_preflight(step, params):
                        preflight_result, checked_frame, preflight_evidence = (
                            self._check_mining_preflight(goal, step, step_frame, available))
                        if self.logger:
                            self.logger({"event": "skill_preflight", "skill": step.skill,
                                         "result": preflight_result or {
                                             "status": "PASSED",
                                             "result_code": "MINING_PREFLIGHT_PASSED",
                                             "reason": "profile_mining_conditions_confirmed",
                                             "evidence": preflight_evidence}})
                        if preflight_result is not None:
                            result = preflight_result
                            skip_verification = True
                        elif has_profile_menu_layout(checked_frame, self.profile):
                            result = {"status": SkillStatus.FAILED.value,
                                      "result_code": "STALE_MENU_STATE",
                                      "reason": "menu_appeared_during_mining_preflight",
                                      "evidence": {"failure_stage": "post_mining_preflight_menu_recheck",
                                                   "input_actions": 0,
                                                   "postcondition": "not_checked_no_input"},
                                      "duration_seconds": max(0.0, self.clock() - begin)}
                            skip_verification = True
                        else:
                            step_frame = checked_frame
                            verification_before = step_frame if self.result_verifier else None
                            step_state_hash = (GameMemory.compute_state_hash(step_frame)
                                               if hasattr(step_frame, "convert") else None)
                            remaining = max(0.0, self.profile.motion_limits.max_run_seconds - (
                                self.clock() - started))
                            available = min(remaining, float(spec.timeout_seconds))
                            if available <= 0:
                                result = {"status": SkillStatus.FAILED.value,
                                          "result_code": "SKILL_TIMEOUT",
                                          "reason": "time_budget_exhausted_before_mining_input",
                                          "evidence": {"failure_stage": "mining_preflight",
                                                       "input_actions": 0,
                                                       "postcondition": "not_checked_no_input"},
                                          "duration_seconds": max(0.0, self.clock() - begin)}
                                skip_verification = True
                            else:
                                self.stop_check()
                                begin = self.clock()
                                skill_context = {}
                                if (step.skill == "approach_target" and step_index > 0
                                        and plan.steps[step_index - 1].skill == "search_target"
                                        and previous_step == "search_target"):
                                    seed = self._handoff_target_seed(previous_result,
                                                                     params.get("target_text"))
                                    if seed:
                                        skill_context["approach_target_seed"] = seed
                                result = self._execute_skill(spec, params, available, begin,
                                                             skill_context)
                    else:
                        skill_context = {}
                        if (step.skill == "approach_target" and step_index > 0
                                and plan.steps[step_index - 1].skill == "search_target"
                                and previous_step == "search_target"):
                            seed = self._handoff_target_seed(previous_result,
                                                             params.get("target_text"))
                            if seed:
                                skill_context["approach_target_seed"] = seed
                        result = self._execute_skill(spec, params, available, begin,
                                                     skill_context)

                    if not skip_verification:
                        verification_candidate = result.get("result_code") in {
                            "REGION_CHANGE_UNCONFIRMED", "SUCCESS_REGION_CHANGE",
                            "INTERACTION_OUTCOME_UNCONFIRMED", "MENU_ACTION_UNCONFIRMED"}
                        verify_final = is_final_step and result.get("status") == SkillStatus.SUCCESS.value
                        if (step.skill in {"interact_until_change", "click_menu_target"}
                                and verification_candidate) or verify_final:
                            verification_after = self.frame_provider() if self.result_verifier else None
                            verification_budget = self.profile.motion_limits.max_run_seconds - (
                                self.clock() - started)
                            result = self._verify_result(step, result, verification_before,
                                                         verification_after, verification_budget,
                                                         force=verify_final)
                        elif (not is_final_step and result.get("status") == SkillStatus.SUCCESS.value
                              and result.get("result_code") not in OUTCOME_SUCCESS_CODES):
                            # Low-level commands report that input was sent, not that the
                            # predicted transition happened. Do not execute the next
                            # preplanned step until its visible postcondition is confirmed.
                            verification_after = self.frame_provider()
                            verification_budget = self.profile.motion_limits.max_run_seconds - (
                                self.clock() - started)
                            result = self._verify_result(
                                step, result, step_frame, verification_after, verification_budget,
                                force=True, scope="plan_transition")
                status = result.get("status", "FAILED")
                duration = result.get("duration_seconds", max(0.0, self.clock() - begin))
                if not self.dry_run:
                    self.store.record(self.game_id, step.skill, goal,
                                      result.get("result_code") or status, duration, self.run_id,
                                      args=params, expected_result=step.expected_result,
                                      success_check=step.success_check, state_hash=step_state_hash)
                if self.logger:
                    self.logger({"event": "skill_result", "skill": step.skill,
                                 "args": step.args, "result": result})
                previous_step, previous_result = step.skill, result
                steps_used += 1
                self._steps_used = steps_used
                if result.get("result_code") == "MENU_TARGET_PREVIEW_FAILED":
                    return self._finish({"status": SkillStatus.UNCONFIRMED.value,
                                         "reason": result["reason"], "steps": steps_used})
                if (self.dry_run and step.skill == "click_menu_target"
                        and result.get("result_code") == "WOULD_SEND"
                        and "grounded_target" in resolved and not is_final_step):
                    reason = "menu_transition_previewed_without_input; next step needs a new screen"
                    if self.logger:
                        self.logger({"event": "menu_preview_stopped_before_dependent_step",
                                     "skill": step.skill, "reason": reason})
                    return self._finish({"status": "DRY_RUN", "reason": reason,
                                         "steps": steps_used})
                if result.get("reason") == "goal_result_unverified":
                    return self._finish({"status": SkillStatus.UNCONFIRMED.value,
                                         "reason": result["reason"], "steps": steps_used})
                if (self.input_action_budget is not None and self.input_action_budget.exhausted
                        and status != SkillStatus.SUCCESS.value):
                    return self._finish({"status": "FAILED", "reason": "max_actions", "steps": steps_used})
                if status in {SkillStatus.ABORT.value, "ABORT"}:
                    return self._finish({"status": "ABORT", "reason": result.get("reason", "aborted"), "steps": steps_used})
                if status != SkillStatus.SUCCESS.value:
                    if (step.skill == "approach_target"
                            and result.get("reason") in {
                                "blocked_after_side_step", "target_not_reacquired"}):
                        recovered, recovery_steps, recovery_skill, recovery_state_hash = (
                            self._recover_blocked_approach(goal, params["target_text"], started,
                                                           steps_used))
                        steps_used += recovery_steps
                        self._steps_used = steps_used
                        recovery_approach_ran = recovery_skill == "approach_target"
                        if recovered and recovered.get("status") == SkillStatus.SUCCESS.value:
                            if is_final_step:
                                if self.result_verifier:
                                    verification_after = self.frame_provider()
                                    verification_budget = self.profile.motion_limits.max_run_seconds - (
                                        self.clock() - started)
                                else:
                                    verification_after = None
                                    verification_budget = 0
                                recovered = self._verify_result(
                                    step, recovered, verification_before, verification_after,
                                    verification_budget, force=True)
                                if not self.dry_run:
                                    self.store.record(
                                        self.game_id, step.skill, goal,
                                        recovered.get("result_code") or recovered.get("status", "FAILED"),
                                        recovered.get("duration_seconds", 0.0), self.run_id,
                                        args=params, expected_result=step.expected_result,
                                        success_check=step.success_check, state_hash=recovery_state_hash)
                                if self.logger:
                                    self.logger({"event": "goal_postcondition", "skill": step.skill,
                                                 "result": recovered})
                                if recovered.get("reason") == "goal_result_unverified":
                                    return self._finish({"status": SkillStatus.UNCONFIRMED.value,
                                        "reason": recovered["reason"], "steps": steps_used})
                                if recovered.get("status") == SkillStatus.SUCCESS.value:
                                    return self._finish({"status": "SUCCESS", "reason": "plan_completed",
                                                         "steps": steps_used})
                                result = recovered
                                status = result.get("status", "FAILED")
                            else:
                                if recovery_approach_ran:
                                    self.store.record(
                                        self.game_id, step.skill, goal,
                                        recovered.get("result_code") or recovered.get("status", "FAILED"),
                                        recovered.get("duration_seconds", 0.0), self.run_id,
                                        args=params, expected_result=step.expected_result,
                                        success_check=step.success_check, state_hash=recovery_state_hash)
                                continue
                        elif recovered and recovery_approach_ran:
                            self.store.record(
                                self.game_id, step.skill, goal,
                                recovered.get("result_code") or recovered.get("status", "FAILED"),
                                recovered.get("duration_seconds", 0.0), self.run_id,
                                args=params, expected_result=step.expected_result,
                                success_check=step.success_check, state_hash=recovery_state_hash)
                        if recovered:
                            result = recovered
                    failure = f"{step.skill}: {status}: {result.get('reason', '')}; expected={step.expected_result}; check={step.success_check}"
                    avoid_plan = current_plan_signature
                    failed_step = (step.skill, params)
                    break
            else:
                return self._finish({"status": "DRY_RUN" if self.dry_run else "SUCCESS",
                                     "reason": "plan_completed", "steps": steps_used})
            if replans >= self.max_replans:
                return self._finish({"status": "FAILED", "reason": failure or "replans_exhausted", "steps": steps_used})
            replans += 1
