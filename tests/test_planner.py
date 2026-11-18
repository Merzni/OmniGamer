import sqlite3
import tempfile
from contextlib import closing
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image
from omnigamer.memory import GameMemory
from omnigamer.perception import FrameFacts, VisualAssessment
from omnigamer.planner import (BudgetControls, DeadlineControls, GoalExecutor, GoalPlan, GoalPlanner,
                               InputActionBudget, InputActionLimitExceeded,
                               PlanningError, SkillOutcomeStore)
from omnigamer.profiles import GameProfile
from omnigamer.skills.contracts import SkillResult, SkillStatus
from omnigamer.skills.registry import default_registry
from omnigamer.vision_brain import VisionBrain


def plan_for(skill="move_pulse", args=None):
    return {"steps": [{"skill": skill, "args": args or {"direction": "forward", "seconds": 0.5},
                       "expected_result": "move", "success_check": "result status"}]}


class ConfirmedAssessment:
    result = "progress"
    confidence = 0.95
    evidence = "Expected visible goal state is present"
    evidence_source = "test_fixture"
    usable = True


def confirm_goal(*_args):
    return ConfirmedAssessment()


class GoalPlannerTests(unittest.TestCase):
    def setUp(self):
        self.profile = GameProfile(game_id="sim", camera_type="first_person",
                                   motion_limits={"max_run_seconds": 20, "max_actions": 3})
        self.registry = default_registry()

    def _mining_plan(self):
        return {"steps": [{"skill": "interact_until_change", "args": {
            "region": "hotbar", "action": "primary_hold", "max_repeats": 1},
            "expected_result": "the visible log disappears",
            "success_check": "the target log is no longer visible"}]}

    def _mining_profile(self):
        return GameProfile(game_id="sim", camera_type="first_person",
            actions={"primary_hold": {"kind": "mouse", "value": "left", "duration": 0.1}},
            mine_actions=["primary_hold"],
            regions={"hotbar": {"x1": 0.3, "y1": 0.9, "x2": 0.7, "y2": 1}},
            mine_requires_grounded_level_view=True,
            mine_requires_target_at_crosshair=True,
            motion_limits={"max_run_seconds": 20, "max_actions": 3})

    @staticmethod
    def _mining_assessment(*, target=True):
        facts = FrameFacts(scene="gameplay", pitch="level", safe_footing=True,
                           target_at_crosshair=target)
        return VisualAssessment(before=facts, after=facts, result="unknown", confidence=0.95,
                                evidence="Current frame shows the mining safety facts.")

    def test_rejects_unknown_skill_and_invalid_arguments(self):
        planner = GoalPlanner(self.registry, lambda **_: plan_for("ghost"))
        with self.assertRaises(PlanningError):
            planner.plan(frame=None, goal="x", profile=self.profile)
        planner = GoalPlanner(self.registry, lambda **_: plan_for(args={"direction": "up", "seconds": 99}))
        with self.assertRaises(PlanningError):
            planner.plan(frame=None, goal="x", profile=self.profile)

    def test_plan_schema_accepts_six_steps_and_rejects_seven(self):
        step = plan_for()["steps"][0]
        valid = GoalPlan.model_validate({"steps": [step] * 6})
        self.assertEqual(len(valid.steps), 6)
        with self.assertRaises(ValueError):
            GoalPlan.model_validate({"steps": [step] * 7})

    def test_adjacent_search_to_approach_hands_off_selected_target(self):
        profile = self.profile
        registry = default_registry(profile=profile)
        target_seed = {"x1": .2, "y1": .2, "x2": .4, "y2": .7,
                       "score": .8, "query": "a tree trunk"}
        plan = {"steps": [
            {"skill": "search_target", "args": {"target_text": "a tree trunk", "sweep": 2},
             "expected_result": "target detected", "success_check": "target is visible"},
            {"skill": "approach_target", "args": {"target_text": "a tree trunk"},
             "expected_result": "target reached", "success_check": "target is centered"},
        ]}
        context = {}
        observed_contexts = []

        def execute(spec, _args, _remaining):
            observed_contexts.append(dict(context))
            if spec.name == "search_target":
                return SkillResult(status=SkillStatus.SUCCESS, result_code="TARGET_DETECTED",
                                   reason="target_detected",
                                   evidence={"target_text": "a tree trunk",
                                             "target_seed": target_seed})
            return SkillResult(status=SkillStatus.SUCCESS, result_code="APPROACH_REACHED",
                               reason="target_reached")

        runner = GoalExecutor(GoalPlanner(registry, lambda **_: plan), registry,
            SkillOutcomeStore(":memory:"), profile=profile, game_id="sim",
            execute_skill=execute, frame_provider=lambda: Image.new("RGB", (32, 24)),
            stop_check=lambda: None, result_verifier=confirm_goal, max_replans=0,
            skill_context=context)

        result = runner.run("approach a tree trunk")

        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(observed_contexts[0], {})
        self.assertEqual(observed_contexts[1]["approach_target_seed"], target_seed)
        self.assertEqual(context, {})

    def test_search_seed_is_not_handed_off_across_intervening_skill(self):
        profile = self.profile
        registry = default_registry(profile=profile)
        plan = {"steps": [
            {"skill": "search_target", "args": {"target_text": "a tree trunk", "sweep": 2},
             "expected_result": "target detected", "success_check": "target is visible"},
            {"skill": "turn", "args": {"direction": "right", "amount": 10},
             "expected_result": "camera turned", "success_check": "turn sent"},
            {"skill": "approach_target", "args": {"target_text": "a tree trunk"},
             "expected_result": "target reached", "success_check": "target is centered"},
        ]}
        context = {}
        approach_context = []

        def execute(spec, _args, _remaining):
            if spec.name == "approach_target":
                approach_context.append(dict(context))
            if spec.name == "search_target":
                return SkillResult(status=SkillStatus.SUCCESS, result_code="TARGET_DETECTED",
                                   reason="target_detected", evidence={
                                       "target_text": "a tree trunk",
                                       "target_seed": {"x1": .2, "y1": .2, "x2": .4,
                                                       "y2": .7, "score": .8,
                                                       "query": "a tree trunk"}})
            return SkillResult(status=SkillStatus.SUCCESS,
                               result_code=("APPROACH_REACHED" if spec.name == "approach_target"
                                            else "SUCCESS"), reason="ok")

        runner = GoalExecutor(GoalPlanner(registry, lambda **_: plan), registry,
            SkillOutcomeStore(":memory:"), profile=profile, game_id="sim",
            execute_skill=execute, frame_provider=lambda: Image.new("RGB", (32, 24)),
            stop_check=lambda: None, result_verifier=confirm_goal, max_replans=0,
            skill_context=context)

        runner.run("approach a tree trunk")

        self.assertEqual(approach_context, [{}])

    def test_search_seed_requires_matching_target_text(self):
        result = {"status": "SUCCESS", "evidence": {
            "target_text": "a tree trunk",
            "target_seed": {"x1": .2, "y1": .2, "x2": .4, "y2": .7,
                            "score": .8, "query": "a tree trunk"}}}

        self.assertIsNone(GoalExecutor._handoff_target_seed(result, "a rock"))
        self.assertIsNotNone(GoalExecutor._handoff_target_seed(result, " A TREE TRUNK "))

    def test_local_recovery_hands_off_selected_target_to_approach(self):
        registry = default_registry(profile=self.profile)
        context = {}
        observed = []
        target_seed = {"x1": .2, "y1": .2, "x2": .4, "y2": .7,
                       "score": .8, "query": "a tree trunk"}

        def execute(spec, _args, _remaining):
            observed.append((spec.name, dict(context)))
            if spec.name == "search_target":
                return SkillResult(status=SkillStatus.SUCCESS, result_code="TARGET_DETECTED",
                                   reason="target_detected", evidence={
                                       "target_text": "a tree trunk", "target_seed": target_seed})
            return SkillResult(status=SkillStatus.SUCCESS,
                               result_code=("APPROACH_REACHED" if spec.name == "approach_target"
                                            else "SUCCESS"), reason="ok")

        runner = GoalExecutor(GoalPlanner(registry, lambda **_: plan_for()), registry,
            SkillOutcomeStore(":memory:"), profile=self.profile, game_id="sim",
            execute_skill=execute, frame_provider=lambda: Image.new("RGB", (32, 24)),
            stop_check=lambda: None, skill_context=context)

        recovered, used, skill, _ = runner._recover_blocked_approach(
            "approach a tree trunk", "a tree trunk", runner.clock(), 0)

        self.assertEqual((recovered["status"], used, skill), ("SUCCESS", 3, "approach_target"))
        self.assertEqual(observed[-1][0], "approach_target")
        self.assertEqual(observed[-1][1]["approach_target_seed"], target_seed)

    def test_menu_skill_requires_profile_adapter_and_matching_expected_result(self):
        menu_profile = GameProfile(game_id="minecraft", ui_adapters=["world_selection"])
        planner = GoalPlanner(default_registry(), lambda **_: None)
        args = {"target_label": "Одиночная игра", "intent": "Open world list",
                "expected_result": "world list"}
        valid = plan_for("click_menu_target", args)
        valid["steps"][0]["expected_result"] = "world list"
        self.assertEqual(planner.validate(valid, profile=menu_profile).steps[0].skill,
                         "click_menu_target")
        with self.assertRaisesRegex(ValueError, "match the step postcondition"):
            mismatched = plan_for("click_menu_target", args)
            mismatched["steps"][0]["expected_result"] = "delete world"
            planner.validate(mismatched, profile=menu_profile)
        self.assertNotIn("click_menu_target", {
            item["name"] for item in planner._profile_catalog(self.profile)})

    def test_stacked_main_menu_click_must_expect_immediate_world_selection(self):
        profile = GameProfile(game_id="minecraft", ui_adapters=[
            "world_selection", "stacked_main_menu"])
        planner = GoalPlanner(default_registry(), lambda **_: None)
        frame = Image.new("RGB", (800, 600))

        def click_plan(expected_result):
            return {"steps": [{
                "skill": "click_menu_target",
                "args": {"target_label": "Одиночная игра", "intent": "Open single-player",
                         "expected_result": expected_result},
                "expected_result": expected_result,
                "success_check": "world selection screen is visible",
            }]}

        with patch("omnigamer.planner.stacked_main_menu_layout", return_value=True):
            with self.assertRaisesRegex(ValueError, "immediate world-selection screen"):
                planner.validate(click_plan("Game world loads into selected world"),
                                 profile=profile, menu_visible=True, frame=frame)
            accepted = planner.validate(click_plan("World selection screen appears"),
                                        profile=profile, menu_visible=True, frame=frame)
        self.assertEqual(accepted.steps[0].args["expected_result"],
                         "World selection screen appears")

    def test_profiled_main_menu_transition_corrects_skipped_world_selection(self):
        profile = GameProfile(game_id="minecraft", ui_adapters=[
            "world_selection", "stacked_main_menu"],
            main_menu_target_transitions={"Одиночная игра": "world_select"})
        planner = GoalPlanner(default_registry(), lambda **_: None)
        frame = Image.new("RGB", (800, 600))
        skipped = {"steps": [{
            "skill": "click_menu_target",
            "args": {"target_label": " ОДИНОЧНАЯ ИГРА! ",
                     "intent": "Start single-player and locate a tree",
                     "expected_result": "Game world loads, camera faces the selected world"},
            "expected_result": "Game world loads, camera faces the selected world",
            "success_check": "Gameplay is visible, with no menu overlay",
        }]}
        with patch("omnigamer.planner.stacked_main_menu_layout", return_value=True):
            corrected = planner.validate(skipped, profile=profile,
                                         menu_visible=True, frame=frame)
        step = corrected.steps[0]
        self.assertEqual(step.args["expected_result"], "World selection screen appears")
        self.assertEqual(step.expected_result, "World selection screen appears")
        self.assertEqual(step.success_check, "World selection screen is visible")

    def test_profiled_main_menu_transition_does_not_apply_without_layout_evidence(self):
        profile = GameProfile(game_id="minecraft", ui_adapters=[
            "world_selection", "stacked_main_menu"],
            main_menu_target_transitions={"Одиночная игра": "world_select"})
        planner = GoalPlanner(default_registry(), lambda **_: None)
        frame = Image.new("RGB", (800, 600))
        skipped = {"steps": [{
            "skill": "click_menu_target",
            "args": {"target_label": "Одиночная игра", "intent": "Start single-player",
                     "expected_result": "Game world loads"},
            "expected_result": "Game world loads",
            "success_check": "Gameplay is visible",
        }]}
        with patch("omnigamer.planner.stacked_main_menu_layout", return_value=False):
            accepted = planner.validate(skipped, profile=profile,
                                        menu_visible=False, frame=frame)
        self.assertEqual(accepted.steps[0].expected_result, "Game world loads")

    def test_recognized_menu_filters_and_rejects_gameplay_interactions(self):
        profile = GameProfile(game_id="menu", camera_type="first_person",
                              ui_adapters=["stacked_main_menu"],
                              actions={"interact": {"kind": "key", "value": "e"}},
                              regions={"hotbar": {"x1": 0.3, "y1": 0.9, "x2": 0.7, "y2": 1}})
        calls = []
        planner = GoalPlanner(self.registry, lambda **kwargs: calls.append(kwargs) or plan_for(
            "interact_until_change", {"region": "hotbar", "action": "interact", "max_repeats": 2}),
            max_validation_retries=0)

        with patch("omnigamer.planner.has_profile_menu_layout", return_value=True):
            with self.assertRaisesRegex(PlanningError, "recognized menu screen"):
                planner.plan(frame=object(), goal="Open single-player world selection",
                             profile=profile)

        offered = {skill["name"] for skill in calls[0]["skills"]}
        self.assertNotIn("interact_until_change", offered)
        self.assertNotIn("move_pulse", offered)
        self.assertNotIn("turn", offered)
        self.assertIn("click_menu_target", offered)

    def test_known_menu_label_wait_cannot_be_followed_by_gameplay_interaction(self):
        profile = GameProfile(game_id="minecraft", camera_type="first_person",
            ui_adapters=["world_selection", "stacked_main_menu"],
            main_menu_target_transitions={"Одиночная игра": "world_select"},
            actions={"interact": {"kind": "mouse", "value": "right"}},
            regions={"hotbar": {"x1": 0.3, "y1": 0.9, "x2": 0.7, "y2": 1}})
        calls = []
        bad_plan = {"steps": [
            {"skill": "wait_for_condition", "args": {
                "timeout": 2, "query": "Одиночная игра", "present": True},
                "expected_result": "MENU_VISIBLE", "success_check": "Menu row visible"},
            {"skill": "interact_until_change", "args": {
                "region": "hotbar", "action": "interact", "max_repeats": 1},
                "expected_result": "PLAY_SELECTED", "success_check": "Play button highlighted"},
        ]}
        safe_plan = {"steps": [bad_plan["steps"][0]]}
        planner = GoalPlanner(self.registry,
            lambda **request: calls.append(request) or (bad_plan if len(calls) == 1 else safe_plan),
            max_validation_retries=1)

        accepted = planner.plan(frame=None,
            goal="Inspect the title menu; do not enter a world", profile=profile)

        self.assertEqual([step.skill for step in accepted.steps], ["wait_for_condition"])
        self.assertEqual(len(calls), 2)
        self.assertIn("visible configured menu target", calls[1]["validation_error"])

    def test_pre_execution_menu_recheck_replans_stale_gameplay_skill(self):
        profile = GameProfile(game_id="minecraft", camera_type="first_person",
            ui_adapters=["world_selection", "stacked_main_menu"],
            actions={"interact": {"kind": "key", "value": "e"}},
            regions={"hotbar": {"x1": 0.3, "y1": 0.9, "x2": 0.7, "y2": 1}},
            motion_limits={"max_run_seconds": 30, "max_actions": 5})
        registry = default_registry(profile)
        gameplay_frame, menu_frame = Image.new("RGB", (64, 48)), Image.new("RGB", (64, 48), "white")
        interact_plan = plan_for("interact_until_change", {
            "region": "hotbar", "action": "interact", "max_repeats": 1})
        click_plan = {"steps": [{"skill": "click_menu_target", "args": {
            "target_label": "Одиночная игра", "intent": "Open world list",
            "expected_result": "World selection screen appears"},
            "expected_result": "World selection screen appears",
            "success_check": "world selection menu is visible"}]}
        requested_frames = []

        def request_plan(**request):
            requested_frames.append(request["frame"])
            return interact_plan if len(requested_frames) == 1 else click_plan

        planner = GoalPlanner(registry, request_plan, max_validation_retries=0)
        frames = iter((gameplay_frame, menu_frame, menu_frame, menu_frame))
        logs = []
        runner = GoalExecutor(planner, registry, SkillOutcomeStore(":memory:"), profile=profile,
            game_id="minecraft", execute_skill=lambda *_: self.fail("stale gameplay skill executed"),
            frame_provider=lambda: next(frames, menu_frame), stop_check=lambda: None,
            dry_run=True, logger=logs.append,
            menu_preview=lambda *_: {"result_code": "TARGET_GROUNDED_PREVIEW",
                                     "evidence": {"input_actions": 0}})

        with patch("omnigamer.planner.has_profile_menu_layout",
                   side_effect=lambda frame, _profile: frame is menu_frame):
            result = runner.run("Open a single-player world")

        self.assertEqual(result["status"], "DRY_RUN")
        self.assertEqual(requested_frames, [gameplay_frame, menu_frame])
        self.assertTrue(any(row["event"] == "stale_menu_plan_rejected" for row in logs))
        would_send = [row for row in logs if row["event"] == "would_send"]
        self.assertEqual([row["skill"] for row in would_send], ["click_menu_target"])

    def test_input_action_budget_caps_real_control_calls(self):
        controls = Mock()
        budget = InputActionBudget(2)
        bounded = BudgetControls(controls, budget)
        bounded.press_key("w", 0.5)
        bounded.move_camera(10, 0)
        with self.assertRaises(InputActionLimitExceeded):
            bounded.hold_mouse("left", 0.1)
        bounded.release_all()
        self.assertEqual(budget.actions_used, 2)
        self.assertEqual(controls.press_key.call_count, 1)
        self.assertEqual(controls.move_camera.call_count, 1)
        self.assertEqual(controls.hold_mouse.call_count, 0)
        self.assertEqual(controls.release_all.call_count, 1)

    def test_exhausted_input_budget_stops_goal_without_replanning(self):
        with tempfile.TemporaryDirectory() as temp:
            budget = InputActionBudget(2)
            raw_controls = Mock()
            calls = []
            logs = []
            profile = GameProfile(game_id="sim", camera_type="first_person",
                actions={"interact": {"kind": "key", "value": "e"}},
                regions={"hotbar": {"x1": 0.3, "y1": 0.9, "x2": 0.7, "y2": 1}},
                motion_limits={"max_run_seconds": 20, "max_actions": 2,
                               "interaction_retry_step_seconds": 0})
            plan = {"steps": [{"skill": "interact_until_change", "args": {
                "region": "hotbar", "action": "interact", "max_repeats": 3},
                "expected_result": "item acquired", "success_check": "item count increased"}]}
            planner = GoalPlanner(self.registry, lambda **_: calls.append("plan") or plan)
            capture = Mock()
            capture.grab_frame.return_value = Image.new("RGB", (20, 20))
            camera_guard = Mock()
            camera_guard.filter.side_effect = lambda action: (action, None)
            recorder = Mock()

            def execute_skill(spec, args, _remaining):
                return spec.executor(args, capture=capture,
                    controls=BudgetControls(raw_controls, budget), profile=profile,
                    camera_guard=camera_guard, stop_check=lambda: None, recorder=recorder)

            store = SkillOutcomeStore(Path(temp) / "outcomes.db")
            runner = GoalExecutor(planner, self.registry, store,
                profile=profile, game_id="sim", execute_skill=execute_skill,
                frame_provider=lambda: object(), stop_check=lambda: None,
                input_action_budget=budget, logger=logs.append)
            result = runner.run("move")
            self.assertEqual(result["reason"], "max_actions")
            self.assertEqual(result["input_actions_used"], 2)
            self.assertEqual(len(calls), 1)
            self.assertEqual(raw_controls.press_key.call_count, 2)
            skill_result = next(row["result"] for row in logs if row["event"] == "skill_result")
            self.assertEqual(skill_result["result_code"], "INPUT_ACTION_LIMIT")
            self.assertEqual(skill_result["evidence"]["input_actions"], 2)
            recorded_result = recorder.finish.call_args.args[0]
            self.assertEqual(recorded_result.result_code, "INPUT_ACTION_LIMIT")
            self.assertEqual(recorded_result.evidence["failure_stage"], "input_action_budget")
            self.assertEqual(recorded_result.evidence["input_actions"], 2)
            self.assertEqual(store.summary("sim")[0]["successes"], 0)

    def test_skill_deadline_timeout_is_recorded_by_registered_executor(self):
        with tempfile.TemporaryDirectory() as temp:
            budget = InputActionBudget(3)
            raw_controls = Mock()
            logs = []
            planner = GoalPlanner(self.registry, lambda **_: plan_for())
            camera_guard = Mock()
            camera_guard.filter.side_effect = lambda action: (action, None)
            recorder = Mock()

            def execute_skill(spec, args, _remaining):
                controls = BudgetControls(DeadlineControls(
                    raw_controls, deadline=1.0, clock=lambda: 2.0), budget)
                return spec.executor(args, controls=controls, profile=self.profile,
                    camera_guard=camera_guard, stop_check=lambda: None, recorder=recorder)

            store = SkillOutcomeStore(Path(temp) / "outcomes.db")
            runner = GoalExecutor(planner, self.registry, store, profile=self.profile, game_id="sim",
                execute_skill=execute_skill, frame_provider=lambda: object(),
                stop_check=lambda: None, input_action_budget=budget, logger=logs.append,
                max_replans=0)

            result = runner.run("move")

            self.assertEqual(result["status"], "FAILED")
            self.assertIn("skill_time_budget_exhausted", result["reason"])
            self.assertEqual(budget.actions_used, 1)
            self.assertEqual(raw_controls.press_key.call_count, 0)
            skill_result = next(row["result"] for row in logs if row["event"] == "skill_result")
            self.assertEqual(skill_result["result_code"], "SKILL_TIMEOUT")
            self.assertEqual(skill_result["evidence"]["input_actions"], 1)
            recorded_result = recorder.finish.call_args.args[0]
            self.assertEqual(recorded_result.result_code, "SKILL_TIMEOUT")
            self.assertEqual(recorded_result.evidence["failure_stage"], "skill_deadline")
            self.assertEqual(recorded_result.evidence["input_actions"], 1)
            self.assertEqual(store.summary("sim")[0]["successes"], 0)

    def test_skill_time_remaining_raises_typed_deadline(self):
        from omnigamer.planner import skill_time_remaining
        from omnigamer.skills.contracts import SkillDeadlineExceeded

        self.assertEqual(skill_time_remaining(3.0, clock=lambda: 2.0), 1.0)
        with self.assertRaises(SkillDeadlineExceeded):
            skill_time_remaining(2.0, clock=lambda: 2.0)

    def test_profiled_mining_fails_closed_without_preflight(self):
        with tempfile.TemporaryDirectory() as temp:
            profile = self._mining_profile()
            frame = Image.new("RGB", (16, 16), (30, 40, 50))
            sends = []
            logs = []
            runner = GoalExecutor(
                GoalPlanner(self.registry, lambda **_: self._mining_plan()), self.registry,
                SkillOutcomeStore(Path(temp) / "outcomes.db"), profile=profile, game_id="sim",
                execute_skill=lambda *args: sends.append(args), frame_provider=lambda: frame,
                stop_check=lambda: None, logger=logs.append, max_replans=0)

            runner.run("collect a visible log")

            self.assertEqual(sends, [])
            preflight = next(row for row in logs if row["event"] == "skill_preflight")
            self.assertEqual(preflight["result"]["result_code"], "MINING_PREFLIGHT_UNAVAILABLE")
            self.assertEqual(preflight["result"]["evidence"]["postcondition"],
                             "not_checked_no_input")

    def test_profiled_mining_preflight_timeout_never_sends_input(self):
        with tempfile.TemporaryDirectory() as temp:
            profile = self._mining_profile()
            frame = Image.new("RGB", (16, 16), (30, 40, 50))
            sends = []
            logs = []

            def timeout(*_args):
                raise TimeoutError("preflight inference timed out")

            runner = GoalExecutor(
                GoalPlanner(self.registry, lambda **_: self._mining_plan()), self.registry,
                SkillOutcomeStore(Path(temp) / "outcomes.db"), profile=profile, game_id="sim",
                execute_skill=lambda *args: sends.append(args), frame_provider=lambda: frame,
                stop_check=lambda: None, logger=logs.append, max_replans=0,
                skill_preflight=timeout)

            runner.run("collect a visible log")

            self.assertEqual(sends, [])
            preflight = next(row for row in logs if row["event"] == "skill_preflight")
            self.assertEqual(preflight["result"]["result_code"], "MINING_PREFLIGHT_TIMEOUT")

    def test_profiled_right_click_is_not_treated_as_mining(self):
        with tempfile.TemporaryDirectory() as temp:
            profile = self._mining_profile()
            profile.actions = {"interact": {"kind": "mouse", "value": "right", "duration": 0.1}}
            frame = Image.new("RGB", (16, 16), (30, 40, 50))
            sends = []
            plan = {"steps": [{"skill": "interact_until_change", "args": {
                "region": "hotbar", "action": "interact", "max_repeats": 1},
                "expected_result": "the door opens", "success_check": "the room is visible"}]}

            def execute(*_args):
                sends.append("right-click action executed")
                return SkillResult(status=SkillStatus.UNCONFIRMED,
                    result_code="INTERACTION_OUTCOME_UNCONFIRMED", reason="needs post-check")

            runner = GoalExecutor(
                GoalPlanner(self.registry, lambda **_: plan), self.registry,
                SkillOutcomeStore(Path(temp) / "outcomes.db"), profile=profile, game_id="sim",
                execute_skill=execute, frame_provider=lambda: frame, stop_check=lambda: None,
                max_replans=0)

            runner.run("open the door")

            self.assertEqual(sends, ["right-click action executed"])

    def test_profiled_mining_rejects_target_away_from_crosshair_without_input(self):
        with tempfile.TemporaryDirectory() as temp:
            profile = self._mining_profile()
            frame = Image.new("RGB", (16, 16), (30, 40, 50))
            sends = []
            logs = []
            runner = GoalExecutor(
                GoalPlanner(self.registry, lambda **_: self._mining_plan()), self.registry,
                SkillOutcomeStore(Path(temp) / "outcomes.db"), profile=profile, game_id="sim",
                execute_skill=lambda *args: sends.append(args), frame_provider=lambda: frame,
                stop_check=lambda: None, logger=logs.append, max_replans=0,
                skill_preflight=lambda *_: self._mining_assessment(target=False))

            runner.run("collect a visible log")

            self.assertEqual(sends, [])
            preflight = next(row for row in logs if row["event"] == "skill_preflight")
            self.assertEqual(preflight["result"]["result_code"], "MINING_PREFLIGHT_BLOCKED")
            self.assertIn("target_at_crosshair_unconfirmed",
                          preflight["result"]["evidence"]["blocked_conditions"])

    def test_profiled_mining_keeps_ambiguous_target_fact_unconfirmed(self):
        with tempfile.TemporaryDirectory() as temp:
            profile = self._mining_profile()
            frame = Image.new("RGB", (16, 16), (30, 40, 50))
            sends = []
            logs = []
            runner = GoalExecutor(
                GoalPlanner(self.registry, lambda **_: self._mining_plan()), self.registry,
                SkillOutcomeStore(Path(temp) / "outcomes.db"), profile=profile, game_id="sim",
                execute_skill=lambda *args: sends.append(args), frame_provider=lambda: frame,
                stop_check=lambda: None, logger=logs.append, max_replans=0,
                skill_preflight=lambda *_: self._mining_assessment(target=None))

            runner.run("collect a visible log")

            self.assertEqual(sends, [])
            preflight = next(row for row in logs if row["event"] == "skill_preflight")
            self.assertEqual(preflight["result"]["result_code"],
                             "MINING_PREFLIGHT_UNCONFIRMED")
            self.assertIn("target_at_crosshair_unconfirmed",
                          preflight["result"]["evidence"]["unknown_conditions"])

    def test_profiled_mining_rejects_stale_facts_before_input(self):
        with tempfile.TemporaryDirectory() as temp:
            profile = self._mining_profile()
            original = Image.new("RGB", (16, 16), (30, 40, 50))
            changed = Image.new("RGB", (16, 16), (220, 210, 200))
            frames = iter([original, original.copy(), changed])
            sends = []
            logs = []
            runner = GoalExecutor(
                GoalPlanner(self.registry, lambda **_: self._mining_plan()), self.registry,
                SkillOutcomeStore(Path(temp) / "outcomes.db"), profile=profile, game_id="sim",
                execute_skill=lambda *args: sends.append(args), frame_provider=lambda: next(frames),
                stop_check=lambda: None, logger=logs.append, max_replans=0,
                skill_preflight=lambda *_: self._mining_assessment(target=True))

            runner.run("collect a visible log")

            self.assertEqual(sends, [])
            preflight = next(row for row in logs if row["event"] == "skill_preflight")
            self.assertEqual(preflight["result"]["result_code"], "STALE_MINING_PREFLIGHT")
            self.assertGreaterEqual(
                preflight["result"]["evidence"]["screen_delta_after_preflight"], 0.02)

    def test_profiled_mining_proceeds_only_on_confident_matching_preflight_facts(self):
        with tempfile.TemporaryDirectory() as temp:
            profile = self._mining_profile()
            frame = Image.new("RGB", (16, 16), (30, 40, 50))
            sends = []
            logs = []

            def execute(*_args):
                sends.append("input path reached")
                return SkillResult(status=SkillStatus.UNCONFIRMED,
                    result_code="INTERACTION_OUTCOME_UNCONFIRMED",
                    reason="interaction outcome requires post-check")

            runner = GoalExecutor(
                GoalPlanner(self.registry, lambda **_: self._mining_plan()), self.registry,
                SkillOutcomeStore(Path(temp) / "outcomes.db"), profile=profile, game_id="sim",
                execute_skill=execute, frame_provider=lambda: frame, stop_check=lambda: None,
                logger=logs.append, max_replans=0,
                skill_preflight=lambda *_: self._mining_assessment(target=True))

            runner.run("collect a visible log")

            self.assertEqual(sends, ["input path reached"])
            preflight = next(row for row in logs if row["event"] == "skill_preflight")
            self.assertEqual(preflight["result"]["result_code"], "MINING_PREFLIGHT_PASSED")
            self.assertEqual(preflight["result"]["evidence"]["facts"]["target_at_crosshair"], True)

    def test_unrelated_timeout_error_keeps_generic_failure_code(self):
        recorder = Mock()
        from omnigamer.skills.registry import _run_recorded

        result = _run_recorded(recorder, lambda: (_ for _ in ()).throw(TimeoutError("transport timeout")))
        self.assertEqual(result.result_code, "FAILED")
        self.assertEqual(result.evidence["exception"], "TimeoutError")
        self.assertEqual(recorder.finish.call_args.args[0], result)

    def test_invalid_plan_gets_one_retry_with_error(self):
        calls = []
        def respond(**kwargs):
            calls.append(kwargs)
            return plan_for("unknown") if len(calls) == 1 else plan_for()
        planner = GoalPlanner(self.registry, respond)
        plan = planner.plan(frame=None, goal="move", profile=self.profile)
        self.assertEqual(plan.steps[0].skill, "move_pulse")
        self.assertEqual(len(calls), 2)
        self.assertIn("Unknown skill", calls[1]["validation_error"])
        self.assertEqual(len(planner.last_plan_metrics["request_durations_seconds"]), 2)
        self.assertGreaterEqual(planner.last_plan_metrics["duration_seconds"], 0)

    def test_replan_rejects_same_failed_plan_before_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            store = SkillOutcomeStore(Path(temp) / "outcomes.db")
            repeated = plan_for()
            alternative = plan_for(args={"direction": "backward", "seconds": 0.5})
            requests = []
            responses = [repeated, repeated, alternative]

            def request(**kwargs):
                requests.append(kwargs)
                return responses.pop(0)

            planner = GoalPlanner(self.registry, request)
            executions = []

            def execute(spec, args, _remaining):
                executions.append((spec.name, args["direction"]))
                if len(executions) == 1:
                    return SkillResult(status=SkillStatus.FAILED, result_code="FAILED", reason="no_effect")
                return SkillResult(status=SkillStatus.SUCCESS, result_code="SUCCESS", reason="moved")

            runner = GoalExecutor(planner, self.registry, store, profile=self.profile, game_id="sim",
                execute_skill=execute, frame_provider=lambda: object(), stop_check=lambda: None,
                max_replans=1, result_verifier=confirm_goal)
            result = runner.run("move")
            self.assertEqual(result["status"], "SUCCESS")
            self.assertEqual(executions, [("move_pulse", "forward"), ("move_pulse", "backward")])
            self.assertIn("exactly repeats", requests[2]["validation_error"])

    def test_replan_rejects_detector_alias_repeat(self):
        with tempfile.TemporaryDirectory() as temp:
            self.profile.detector_queries = {
                "tree_trunk": ["tree trunk", "a tree trunk", "a tree"]}
            store = SkillOutcomeStore(Path(temp) / "outcomes.db")

            def approach(target):
                return {"steps": [{"skill": "approach_target", "args": {"target_text": target},
                                   "expected_result": "target reached", "success_check": "aligned"}]}

            requests = []
            responses = [approach("tree trunk"), approach("a tree trunk"), approach("rock")]

            def request(**kwargs):
                requests.append(kwargs)
                return responses.pop(0)

            planner = GoalPlanner(self.registry, request)
            executions = []

            def execute(spec, args, _remaining):
                executions.append(args["target_text"])
                if len(executions) == 1:
                    return SkillResult(status=SkillStatus.FAILED, result_code="FAILED", reason="lost")
                return SkillResult(status=SkillStatus.SUCCESS, result_code="SUCCESS", reason="reached")

            runner = GoalExecutor(planner, self.registry, store, profile=self.profile, game_id="sim",
                execute_skill=execute, frame_provider=lambda: object(), stop_check=lambda: None,
                max_replans=1, result_verifier=confirm_goal)
            result = runner.run("approach target")
            self.assertEqual(result["status"], "SUCCESS")
            self.assertEqual(executions, ["tree trunk", "rock"])
            self.assertIn("exactly repeats", requests[2]["validation_error"])

    def test_plan_signature_ignores_leading_english_article_in_target(self):
        self.profile.detector_queries = {"tree_trunk": ["tree trunk"]}
        planner = GoalPlanner(self.registry, lambda **_: None)

        def signature(target):
            plan = GoalPlan.model_validate({"steps": [{
                "skill": "approach_target", "args": {"target_text": target},
                "expected_result": "target reached", "success_check": "aligned",
            }]})
            return planner.signature(plan, profile=self.profile)

        self.assertEqual(signature("tree trunk"), signature("the tree trunk"))
        self.assertEqual(signature("an oak"), signature("oak"))

    def test_profile_actions_and_regions_are_schema_constrained(self):
        self.profile.actions = {"primary_hold": {"kind": "mouse", "value": "left"},
                                "interact": {"kind": "mouse", "value": "right"}}
        self.profile.regions = {"hotbar": {"x1": .3, "y1": .9, "x2": .7, "y2": 1}}
        calls = []
        def respond(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return {"steps": [{"skill": "interact_until_change", "args": {
                    "region": "hotbar", "action": "right", "max_repeats": 2},
                    "expected_result": "change", "success_check": "region delta"}]}
            return {"steps": [{"skill": "interact_until_change", "args": {
                "region": "hotbar", "action": "interact", "max_repeats": 2},
                "expected_result": "change", "success_check": "region delta"}]}
        plan = GoalPlanner(self.registry, respond).plan(frame=None, goal="get item", profile=self.profile)
        self.assertEqual(plan.steps[0].args["action"], "interact")
        self.assertEqual(len(calls), 2)
        schema = {item["name"]: item for item in calls[0]["skills"]}["interact_until_change"]
        self.assertEqual(schema["parameters"]["properties"]["action"]["enum"],
                         ["interact", "primary_hold"])
        self.assertIn("not configured", calls[1]["validation_error"])

    def test_camera_free_profile_hides_and_rejects_camera_dependent_skills(self):
        profile = GameProfile(game_id="flat-game", camera_type="none")
        requests = []
        responses = [
            {"steps": [{"skill": "turn", "args": {"direction": "right", "amount": 20},
                        "expected_result": "screen scrolls", "success_check": "landmark changes"}]},
            plan_for(),
        ]
        planner = GoalPlanner(self.registry,
                              lambda **request: requests.append(request) or responses.pop(0))

        plan = planner.plan(frame=None, goal="move", profile=profile)

        offered = {skill["name"] for skill in requests[0]["skills"]}
        self.assertTrue({"approach_target", "search_target", "turn"}.isdisjoint(offered))
        self.assertEqual(plan.steps[0].skill, "move_pulse")
        self.assertIn("requires a camera", requests[1]["validation_error"])
        for skill, args in (
                ("approach_target", {"target_text": "target"}),
                ("search_target", {"target_text": "target", "sweep": 2}),
                ("turn", {"direction": "right", "amount": 20})):
            with self.subTest(skill=skill), self.assertRaisesRegex(ValueError, "requires a camera"):
                planner.validate({"steps": [{"skill": skill, "args": args,
                                             "expected_result": "changed",
                                             "success_check": "visible change"}]}, profile)

    def test_window_backend_hides_and_rejects_camera_dependent_skills(self):
        profile = GameProfile(game_id="window-camera", camera_type="first_person",
                              input_backend="window", ui_adapters=["stacked_main_menu"])
        planner = GoalPlanner(self.registry, lambda **_: plan_for())
        offered = {skill["name"] for skill in planner._profile_catalog(profile)}
        self.assertTrue({"approach_target", "search_target", "turn"}.isdisjoint(offered))
        self.assertIn("move_pulse", offered)
        self.assertIn("click_menu_target", offered)
        for skill, args in (
                ("approach_target", {"target_text": "target"}),
                ("search_target", {"target_text": "target", "sweep": 2}),
                ("turn", {"direction": "right", "amount": 20})):
            with self.subTest(skill=skill), self.assertRaisesRegex(
                    ValueError, "relative camera input is unavailable"):
                planner.validate({"steps": [{"skill": skill, "args": args,
                                             "expected_result": "changed",
                                             "success_check": "visible change"}]}, profile)

    def test_interaction_skill_requires_profile_action_and_region(self):
        planner = GoalPlanner(self.registry, lambda **_: plan_for())
        step = {"steps": [{"skill": "interact_until_change",
                            "args": {"region": "area", "action": "activate", "max_repeats": 1},
                            "expected_result": "screen changes", "success_check": "target disappears"}]}
        profiles = (
            GameProfile(game_id="no-interaction", camera_type="none"),
            GameProfile(game_id="missing-region", camera_type="none",
                        actions={"activate": {"kind": "key", "value": "e"}}),
            GameProfile(game_id="missing-action", camera_type="none",
                        regions={"area": {"x1": 0, "y1": 0, "x2": 1, "y2": 1}}),
        )
        for profile in profiles:
            with self.subTest(game_id=profile.game_id):
                offered = {skill["name"] for skill in planner._profile_catalog(profile)}
                self.assertNotIn("interact_until_change", offered)
                with self.assertRaisesRegex(ValueError, "requires configured actions and regions"):
                    planner.validate(step, profile)

    def test_outcome_store_separates_games_and_summarizes_success_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SkillOutcomeStore(Path(tmp) / "outcomes.db")
            store.record("a", "move_pulse", "move", "SUCCESS_VERIFIED", 1, "r1",
                         args={"direction": "forward", "seconds": 0.5},
                         expected_result="near object", success_check="target remains centered")
            store.record("a", "move_pulse", "move", "FAILED", 2, "r2")
            store.record("a", "move_pulse", "move", "SUCCESS_REGION_CHANGE", 3, "r4")
            store.record("a", "click_menu_target", "menu", "MENU_TARGET_UNCONFIRMED", 4, "r5")
            store.record("b", "move_pulse", "move", "SUCCESS", 1, "r3")
            summary = {item["skill"]: item for item in store.summary("a")}
            self.assertEqual(summary["move_pulse"], {"skill": "move_pulse", "target": "move",
                "successes": 1, "attempts": 3, "uncertain": 1,
                "success_rate": 0.333, "verified_example": None})
            self.assertEqual(summary["click_menu_target"]["uncertain"], 1)
            self.assertIsNone(summary["click_menu_target"]["verified_example"])
            self.assertEqual((store.summary("b")[0]["successes"],
                              store.summary("b")[0]["attempts"],
                              store.summary("b")[0]["success_rate"]), (0, 1, 0.0))

    def test_outcome_summary_limit_keeps_most_recent_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SkillOutcomeStore(Path(tmp) / "outcomes.db")
            for index in range(25):
                store.record("sim", "move_pulse", f"target-{index:02d}", "SUCCESS_VERIFIED",
                             1, f"run-{index}", args={"direction": "forward", "seconds": 0.5})

            summary = store.summary("sim", max_groups=20)
            targets = [item["target"] for item in summary]

            self.assertEqual(targets[0], "target-24")
            self.assertIn("target-24", targets)
            self.assertNotIn("target-00", targets)
            self.assertEqual(len(targets), 20)

    def test_outcome_summary_prioritizes_active_goal_over_newer_other_goals(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SkillOutcomeStore(Path(tmp) / "outcomes.db")
            frame = Image.new("RGB", (16, 16), (0, 0, 0))
            state_hash = GameMemory.compute_state_hash(frame)
            for index in range(25):
                store.record("sim", "move_pulse", f"goal-{index:02d}", "SUCCESS_VERIFIED",
                             1, f"run-{index}", args={"direction": "forward", "seconds": 0.5},
                             expected_result="closer to goal", success_check="visible distance decreased",
                             state_hash=state_hash)

            requests = []
            planner = GoalPlanner(self.registry,
                                  lambda **request: requests.append(request) or plan_for())
            executor = GoalExecutor(planner, self.registry, store, profile=self.profile, game_id="sim",
                                    execute_skill=lambda *_: self.fail("dry run executed"),
                                    frame_provider=lambda: frame, stop_check=lambda: None,
                                    dry_run=True)
            result = executor.run("goal-00")

            summary = requests[0]["outcome_summary"]
            targets = [item["target"] for item in summary]

            self.assertEqual(targets[0], "goal-00")
            self.assertIn("goal-00", targets)
            self.assertNotIn("goal-01", targets)
            self.assertEqual(len(targets), 20)
            self.assertEqual(summary[0]["verified_example"]["args"],
                             {"direction": "forward", "seconds": 0.5})
            self.assertEqual(result["memory_reuse_count"], 1)

    def test_outcome_summary_retains_active_goal_past_recent_history_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SkillOutcomeStore(Path(tmp) / "outcomes.db")
            for index in range(250):
                store.record("sim", "move_pulse", f"other-goal-{index:03d}", "FAILED",
                             1, f"other-{index}")
            store.record("sim", "move_pulse", "active-goal", "SUCCESS_VERIFIED", 1, "active",
                         args={"direction": "forward", "seconds": 0.5},
                         expected_result="closer to goal", success_check="visible distance decreased",
                         state_hash="0123456789abcdef")

            summary = store.summary("sim", target="active-goal", state_hash="0123456789abcdef")

            self.assertEqual(len(summary), 20)
            self.assertEqual(summary[0]["target"], "active-goal")
            self.assertEqual(summary[0]["verified_example"]["args"],
                             {"direction": "forward", "seconds": 0.5})

    def test_latest_success_reenables_example_after_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SkillOutcomeStore(Path(tmp) / "outcomes.db")
            store.record("sim", "approach_target", "reach a tree", "SUCCESS_VERIFIED", 1, "r1",
                         args={"target_text": "tree"}, expected_result="near the trunk",
                         success_check="crosshair stays on trunk")
            store.record("sim", "approach_target", "reach a tree", "FAILED", 2, "r2")
            self.assertIsNone(store.summary("sim")[0]["verified_example"])

            store.record("sim", "approach_target", "reach a tree", "APPROACH_REACHED", 1, "r3",
                         args={"target_text": "birch trunk"}, expected_result="near the trunk",
                         success_check="target remains centered")
            self.assertEqual(store.summary("sim")[0]["verified_example"]["args"],
                             {"target_text": "birch trunk"})

    def test_verified_examples_are_filtered_by_matching_screen_state(self):
        with tempfile.TemporaryDirectory() as temp:
            store = SkillOutcomeStore(Path(temp) / "outcomes.db")
            left_to_right = Image.new("RGB", (16, 16), (0, 0, 0))
            right_to_left = Image.new("RGB", (16, 16), (0, 0, 0))
            for x in range(8, 16):
                for y in range(16):
                    left_to_right.putpixel((x, y), (255, 255, 255))
            for x in range(8):
                for y in range(16):
                    right_to_left.putpixel((x, y), (255, 255, 255))
            state_a = GameMemory.compute_state_hash(left_to_right)
            state_b = GameMemory.compute_state_hash(right_to_left)
            self.assertGreater(GameMemory._hash_distance(state_a, state_b), 4)
            store.record("sim", "move_pulse", "move", "SUCCESS_VERIFIED", 1, "a",
                         args={"direction": "forward", "seconds": 0.5},
                         expected_result="closer", success_check="target is closer", state_hash=state_a)
            store.record("sim", "move_pulse", "move", "SUCCESS_VERIFIED", 1, "b",
                         args={"direction": "left", "seconds": 0.5},
                         expected_result="turned", success_check="view changed", state_hash=state_b)
            self.assertEqual(store.summary("sim", state_hash=state_a)[0]["verified_example"]["args"],
                             {"direction": "forward", "seconds": 0.5})
            self.assertIsNone(store.summary("sim", state_hash="ffffffffffffffff")[0]["verified_example"])

            nearby_state_a = f"{int(state_a, 16) ^ 1:016x}"
            store.record("sim", "move_pulse", "move", "FAILED", 1, "a-failed",
                         args={"direction": "forward", "seconds": 0.5}, state_hash=nearby_state_a)
            self.assertIsNone(store.summary("sim", state_hash=state_a)[0]["verified_example"])

    def test_outcome_summary_requires_visible_check_for_verified_examples(self):
        with tempfile.TemporaryDirectory() as temp:
            store = SkillOutcomeStore(Path(temp) / "outcomes.db")
            state_hash = "0123456789abcdef"
            store.record("sim", "search_target", "find target", "TARGET_DETECTED", 0.2, "recovery",
                         args={"target_text": "target"}, state_hash=state_hash)

            self.assertEqual(store.summary("sim")[0]["successes"], 1)
            self.assertIsNone(store.summary("sim")[0]["verified_example"])
            self.assertIsNone(store.summary("sim", state_hash=state_hash)[0]["verified_example"])

            store.record("sim", "search_target", "find target", "TARGET_DETECTED", 0.2, "checked",
                         args={"target_text": "target"}, expected_result="target is visible",
                         success_check="detector returns a matching target", state_hash=state_hash)
            example = store.summary("sim", state_hash=state_hash)[0]["verified_example"]
            self.assertEqual(example["expected_result"], "target is visible")
            self.assertEqual(example["success_check"], "detector returns a matching target")

    def test_skill_outcome_store_additively_migrates_legacy_schema(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "legacy.db"
            with closing(sqlite3.connect(path)) as conn:
                with conn:
                    conn.execute("CREATE TABLE skill_outcomes (id INTEGER PRIMARY KEY, game_id TEXT NOT NULL, "
                                 "skill TEXT NOT NULL, target TEXT NOT NULL, result TEXT NOT NULL, "
                                 "duration REAL NOT NULL, ts REAL NOT NULL, run_id TEXT NOT NULL)")
            store = SkillOutcomeStore(path)
            with closing(sqlite3.connect(path)) as conn:
                columns = {row[1] for row in conn.execute("PRAGMA table_info(skill_outcomes)")}
            self.assertTrue({"args_json", "expected_result", "success_check", "state_hash"}.issubset(columns))
            store.record("sim", "turn", "look around", "SUCCESS_VERIFIED", 0.2, "r1",
                         args={"direction": "left", "amount": 20},
                         expected_result="new area visible", success_check="landmark changed")
            example = store.summary("sim")[0]["verified_example"]
            self.assertEqual(example["args"], {"direction": "left", "amount": 20})

    def test_legacy_success_rows_survive_migration_without_becoming_examples(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "legacy.db"
            with closing(sqlite3.connect(path)) as conn:
                with conn:
                    conn.execute("CREATE TABLE skill_outcomes (id INTEGER PRIMARY KEY, game_id TEXT NOT NULL, "
                                 "skill TEXT NOT NULL, target TEXT NOT NULL, result TEXT NOT NULL, "
                                 "duration REAL NOT NULL, ts REAL NOT NULL, run_id TEXT NOT NULL)")
                    conn.execute("INSERT INTO skill_outcomes(game_id,skill,target,result,duration,ts,run_id) "
                                 "VALUES(?,?,?,?,?,?,?)",
                                 ("sim", "move_pulse", "move", "SUCCESS", 0.5, 1.0, "legacy"))

            store = SkillOutcomeStore(path)
            summary = store.summary("sim")[0]
            self.assertEqual((summary["successes"], summary["attempts"]), (0, 1))
            self.assertIsNone(summary["verified_example"])

            store.record("sim", "move_pulse", "move", "SUCCESS_VERIFIED", 0.4, "new",
                         args={"direction": "forward", "seconds": 0.4},
                         expected_result="nearer", success_check="target is closer",
                         state_hash="0123456789abcdef")
            reopened = SkillOutcomeStore(path).summary("sim")[0]
            self.assertEqual((reopened["successes"], reopened["attempts"]), (1, 2))
            self.assertEqual(reopened["verified_example"]["args"],
                             {"direction": "forward", "seconds": 0.4})

    def test_generic_success_is_not_an_outcome_or_verified_example(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SkillOutcomeStore(Path(tmp) / "outcomes.db")
            store.record("sim", "move_pulse", "move", "SUCCESS_VERIFIED", 1, "r1",
                         args={"direction": "forward", "seconds": 0.5},
                         expected_result="nearer", success_check="target is closer")
            store.record("sim", "move_pulse", "move", "SUCCESS", 0.5, "r2",
                         args={"direction": "forward", "seconds": 0.5},
                         expected_result="nearer", success_check="target is closer")
            summary = store.summary("sim")[0]
            self.assertEqual((summary["successes"], summary["attempts"], summary["success_rate"]),
                             (1, 2, 0.5))
            self.assertIsNone(summary["verified_example"])

    def test_dry_run_logs_would_send_and_never_calls_executor(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SkillOutcomeStore(Path(tmp) / "outcomes.db")
            planner = GoalPlanner(self.registry, lambda **_: plan_for())
            logs, calls = [], []
            executor = GoalExecutor(planner, self.registry, store, profile=self.profile, game_id="sim",
                                    execute_skill=lambda *_: calls.append("execute"),
                                    frame_provider=lambda: object(), stop_check=lambda: None,
                                    dry_run=True, logger=logs.append)
            result = executor.run("move")
            self.assertEqual(result["status"], "DRY_RUN")
            self.assertEqual(calls, [])
            self.assertEqual(logs[1]["event"], "would_send")

    def test_dry_run_menu_uses_read_only_target_preview(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile = GameProfile(game_id="minecraft", ui_adapters=["world_selection"],
                motion_limits={"max_run_seconds": 20, "max_actions": 3})
            args = {"target_label": "Одиночная игра", "intent": "Open world list",
                    "expected_result": "World selection screen appears"}
            plan = {"steps": [{"skill": "click_menu_target", "args": args,
                "expected_result": "World selection screen appears",
                "success_check": "world selection menu is visible"}]}
            planner = GoalPlanner(default_registry(profile), lambda **_: plan)
            frame = Image.new("RGB", (800, 600))
            preview = Mock(return_value={"result_code": "TARGET_GROUNDED_PREVIEW",
                                         "evidence": {"target_label": "Одиночная игра",
                                                      "input_actions": 0}})
            logs = []
            executor = GoalExecutor(planner, default_registry(profile),
                SkillOutcomeStore(Path(tmp) / "outcomes.db"), profile=profile, game_id="minecraft",
                execute_skill=lambda *_: self.fail("dry-run executed input skill"),
                frame_provider=lambda: frame, stop_check=lambda: None,
                dry_run=True, logger=logs.append, menu_preview=preview)

            result = executor.run("Добудь дерево")

            self.assertEqual(result["status"], "DRY_RUN")
            preview.assert_called_once()
            would_send = next(row for row in logs if row["event"] == "would_send")
            self.assertEqual(would_send["grounded_target"]["evidence"]["input_actions"], 0)

    def test_dry_run_menu_preview_failure_never_reports_would_send(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile = GameProfile(game_id="minecraft", ui_adapters=["world_selection"],
                motion_limits={"max_run_seconds": 20, "max_actions": 3})
            args = {"target_label": "Одиночная игра", "intent": "Open world list",
                    "expected_result": "World selection screen appears"}
            plan = {"steps": [{"skill": "click_menu_target", "args": args,
                "expected_result": "World selection screen appears",
                "success_check": "world selection menu is visible"}]}
            planner = GoalPlanner(default_registry(profile), lambda **_: plan)
            frame = Image.new("RGB", (800, 600))
            logs = []
            executor = GoalExecutor(planner, default_registry(profile),
                SkillOutcomeStore(Path(tmp) / "outcomes.db"), profile=profile, game_id="minecraft",
                execute_skill=lambda *_: self.fail("dry-run executed input skill"),
                frame_provider=lambda: frame, stop_check=lambda: None,
                dry_run=True, logger=logs.append,
                menu_preview=lambda *_: {"result_code": "MENU_TARGET_UNCONFIRMED",
                    "reason": "menu_target_unconfirmed:ValueError:target unreadable"})

            result = executor.run("Добудь дерево")

            self.assertEqual(result["status"], SkillStatus.UNCONFIRMED.value)
            self.assertIn("target unreadable", result["reason"])
            self.assertFalse(any(row["event"] == "would_send" for row in logs))
            self.assertTrue(any(row["event"] == "menu_preview_failed" for row in logs))

    def test_dry_run_menu_preview_preserves_emergency_abort(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile = GameProfile(game_id="minecraft", ui_adapters=["world_selection"],
                motion_limits={"max_run_seconds": 20, "max_actions": 3})
            args = {"target_label": "Одиночная игра", "intent": "Open world list",
                    "expected_result": "World selection screen appears"}
            plan = {"steps": [{"skill": "click_menu_target", "args": args,
                "expected_result": "World selection screen appears",
                "success_check": "world selection menu is visible"}]}
            planner = GoalPlanner(default_registry(profile), lambda **_: plan)
            frame = Image.new("RGB", (800, 600))
            logs = []

            def interrupted_preview(*_):
                raise InterruptedError("Emergency stop requested (F12)")

            executor = GoalExecutor(planner, default_registry(profile),
                SkillOutcomeStore(Path(tmp) / "outcomes.db"), profile=profile, game_id="minecraft",
                execute_skill=lambda *_: self.fail("dry-run executed input skill"),
                frame_provider=lambda: frame, stop_check=lambda: None,
                dry_run=True, logger=logs.append, menu_preview=interrupted_preview)

            result = executor.run("Добудь дерево")

            self.assertEqual(result["status"], "ABORT")
            self.assertIn("Emergency stop requested", result["reason"])
            self.assertFalse(any(row["event"] == "would_send" for row in logs))
            self.assertTrue(any(row["event"] == "menu_preview_aborted" for row in logs))

    def test_goal_telemetry_counts_only_exact_verified_example_reuse_for_same_goal(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SkillOutcomeStore(Path(tmp) / "outcomes.db")
            frame_a = Image.new("RGB", (16, 16), (0, 0, 0))
            frame_b = Image.new("RGB", (16, 16), (0, 0, 0))
            for x in range(8, 16):
                for y in range(16):
                    frame_a.putpixel((x, y), (255, 255, 255))
            for x in range(8):
                for y in range(16):
                    frame_b.putpixel((x, y), (255, 255, 255))
            state_hash = GameMemory.compute_state_hash(frame_a)
            store.record("sim", "move_pulse", "move", "SUCCESS_VERIFIED", 0.5, "prior",
                         args={"direction": "forward", "seconds": 0.5},
                         expected_result="closer", success_check="target moved closer",
                         state_hash=state_hash)
            planner = GoalPlanner(self.registry, lambda **_: plan_for())
            logs = []
            runner = GoalExecutor(planner, self.registry, store, profile=self.profile, game_id="sim",
                execute_skill=lambda *_: self.fail("dry run executed"),
                frame_provider=lambda: frame_a, stop_check=lambda: None, dry_run=True,
                logger=logs.append)

            result = runner.run("move")

            plan_event = next(event for event in logs if event["event"] == "plan")
            self.assertEqual(plan_event["verified_example_count"], 1)
            self.assertEqual(plan_event["memory_reuse_count"], 1)
            self.assertEqual(result["memory_reuse_count"], 1)

            logs.clear()
            other_state = GoalExecutor(planner, self.registry, store, profile=self.profile, game_id="sim",
                execute_skill=lambda *_: self.fail("dry run executed"),
                frame_provider=lambda: frame_b, stop_check=lambda: None, dry_run=True,
                logger=logs.append)
            other_state_result = other_state.run("move")
            other_state_plan = next(event for event in logs if event["event"] == "plan")
            self.assertEqual(other_state_plan["verified_example_count"], 0)
            self.assertEqual(other_state_plan["memory_reuse_count"], 0)
            self.assertEqual(other_state_result["memory_reuse_count"], 0)

            logs.clear()
            other_goal = GoalExecutor(planner, self.registry, store, profile=self.profile, game_id="sim",
                execute_skill=lambda *_: self.fail("dry run executed"),
                frame_provider=lambda: object(), stop_check=lambda: None, dry_run=True,
                logger=logs.append)
            other_result = other_goal.run("move elsewhere")
            other_plan = next(event for event in logs if event["event"] == "plan")
            self.assertEqual(other_plan["verified_example_count"], 0)
            self.assertEqual(other_plan["memory_reuse_count"], 0)
            self.assertEqual(other_result["memory_reuse_count"], 0)

    def test_each_plan_step_memory_uses_its_own_pre_action_screen(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame_a = Image.new("RGB", (16, 16), (0, 0, 0))
            frame_b = Image.new("RGB", (16, 16), (0, 0, 0))
            for x in range(8, 16):
                for y in range(16):
                    frame_a.putpixel((x, y), (255, 255, 255))
            for x in range(8):
                for y in range(16):
                    frame_b.putpixel((x, y), (255, 255, 255))
            state_a = GameMemory.compute_state_hash(frame_a)
            state_b = GameMemory.compute_state_hash(frame_b)
            plan = {"steps": [
                {"skill": "move_pulse", "args": {"direction": "forward", "seconds": 0.5},
                 "expected_result": "move", "success_check": "result status"},
                {"skill": "move_pulse", "args": {"direction": "left", "seconds": 0.5},
                 "expected_result": "move", "success_check": "result status"},
            ]}
            store = SkillOutcomeStore(Path(tmp) / "outcomes.db")
            runner = GoalExecutor(
                GoalPlanner(self.registry, lambda **_: plan), self.registry, store,
                profile=self.profile, game_id="sim",
                execute_skill=lambda *_: SkillResult(status=SkillStatus.SUCCESS,
                    result_code="SUCCESS_VERIFIED", reason="verified"),
                frame_provider=Mock(side_effect=[frame_a, frame_a, frame_b, frame_b]),
                stop_check=lambda: None, result_verifier=confirm_goal)

            self.assertEqual(runner.run("move")["status"], "SUCCESS")
            a_example = store.summary("sim", state_hash=state_a)[0]["verified_example"]
            b_example = store.summary("sim", state_hash=state_b)[0]["verified_example"]
            self.assertEqual(a_example["args"]["direction"], "forward")
            self.assertEqual(b_example["args"]["direction"], "left")

    def test_unverified_low_level_transition_blocks_next_planned_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame_a = Image.new("RGB", (16, 16), (0, 0, 0))
            frame_b = Image.new("RGB", (16, 16), (255, 255, 255))
            plan = {"steps": [
                {"skill": "turn", "args": {"direction": "right", "amount": 20},
                 "expected_result": "view turns right", "success_check": "landmark shifts right"},
                {"skill": "move_pulse", "args": {"direction": "forward", "seconds": 0.5},
                 "expected_result": "closer to tree", "success_check": "tree grows in frame"},
            ]}
            calls = []

            class BlockedAssessment:
                usable = True
                result = "blocked"
                confidence = 0.95
                evidence = "The landmark did not shift"
                evidence_source = "test_fixture"

            runner = GoalExecutor(
                GoalPlanner(self.registry, lambda **_: plan), self.registry,
                SkillOutcomeStore(Path(tmp) / "outcomes.db"), profile=self.profile,
                game_id="sim",
                execute_skill=lambda spec, *_: calls.append(spec.name) or SkillResult(
                    status=SkillStatus.SUCCESS, result_code="SUCCESS", reason="command sent"),
                frame_provider=Mock(side_effect=[frame_a, frame_a, frame_b]),
                stop_check=lambda: None, result_verifier=lambda *_: BlockedAssessment(),
                max_replans=0)

            result = runner.run("reach the tree")
            self.assertEqual(result["status"], "FAILED")
            self.assertEqual(calls, ["turn"])
            self.assertIn("blocked", result["reason"])

    def test_verified_low_level_transition_allows_next_planned_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame_a = Image.new("RGB", (16, 16), (0, 0, 0))
            frame_b = Image.new("RGB", (16, 16), (255, 255, 255))
            plan = {"steps": [
                {"skill": "turn", "args": {"direction": "right", "amount": 20},
                 "expected_result": "view turns right", "success_check": "landmark shifts right"},
                {"skill": "move_pulse", "args": {"direction": "forward", "seconds": 0.5},
                 "expected_result": "closer to tree", "success_check": "tree grows in frame"},
            ]}
            calls = []
            logs = []
            runner = GoalExecutor(
                GoalPlanner(self.registry, lambda **_: plan), self.registry,
                SkillOutcomeStore(Path(tmp) / "outcomes.db"), profile=self.profile,
                game_id="sim",
                execute_skill=lambda spec, *_: calls.append(spec.name) or SkillResult(
                    status=SkillStatus.SUCCESS, result_code="SUCCESS", reason="command sent"),
                frame_provider=Mock(side_effect=[frame_a, frame_a, frame_b, frame_b, frame_b]),
                stop_check=lambda: None, result_verifier=confirm_goal, logger=logs.append)

            result = runner.run("reach the tree")
            scopes = [event["result"]["evidence"]["verification"]["scope"]
                      for event in logs if event["event"] == "skill_result"]
            self.assertEqual(result["status"], "SUCCESS")
            self.assertEqual(calls, ["turn", "move_pulse"])
            self.assertEqual(scopes, ["plan_transition", "goal_postcondition"])

    def test_dry_run_detector_uses_profile_synonyms(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.profile.detector_queries = {"tree_trunk": ["a tree trunk", "a tree"]}
            self.profile.thresholds["detector_score_min"] = 0.15
            seen = []

            class Detector:
                def detect(self, _image, queries, **kwargs):
                    seen.append((queries, kwargs))
                    boxes = ([{"query": "a tree", "score": 0.259, "x1": 0.302,
                               "y1": 0.0, "x2": 0.746, "y2": 0.956}]
                             if "a tree" in queries else [])
                    return {"boxes": boxes}

            plan = {"steps": [{"skill": "approach_target", "args": {"target_text": "a tree trunk"},
                                "expected_result": "near", "success_check": "target centered"}]}
            planner = GoalPlanner(self.registry, lambda **_: plan)
            logs = []
            runner = GoalExecutor(planner, self.registry, SkillOutcomeStore(Path(tmp) / "outcomes.db"),
                profile=self.profile, game_id="sim", execute_skill=lambda *_: self.fail("dry run executed"),
                frame_provider=lambda: object(), stop_check=lambda: None, dry_run=True,
                detector=Detector(), logger=logs.append)
            result = runner.run("approach a tree")
            self.assertEqual(result["status"], "DRY_RUN")
            self.assertEqual(seen[0][0], ["a tree trunk", "a tree"])
            self.assertEqual(seen[0][1], {"score_min": 0.15, "max_boxes": 10})
            detection = logs[1]["detector"]
            self.assertEqual(detection["box_count"], 1)
            self.assertEqual(detection["candidate_box_count"], 1)

    def test_failed_skill_replans_then_abort_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SkillOutcomeStore(Path(tmp) / "outcomes.db")
            planner_calls = []
            planner = GoalPlanner(self.registry, lambda **_: planner_calls.append(1) or plan_for())
            executor = GoalExecutor(planner, self.registry, store, profile=self.profile, game_id="sim",
                                    execute_skill=lambda *_: SkillResult(status=SkillStatus.ABORT,
                                        result_code="ABORT", reason="emergency"),
                                    frame_provider=lambda: object(), stop_check=lambda: None)
            result = executor.run("move")
            self.assertEqual(result["status"], "ABORT")
            self.assertEqual(len(planner_calls), 1)

    def test_interrupted_skill_execution_becomes_recorded_abort(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SkillOutcomeStore(Path(tmp) / "outcomes.db")
            logs = []
            planner = GoalPlanner(self.registry, lambda **_: plan_for())

            def interrupted_skill(*_):
                raise InterruptedError("Emergency stop requested (F12)")

            executor = GoalExecutor(planner, self.registry, store, profile=self.profile, game_id="sim",
                execute_skill=interrupted_skill, frame_provider=lambda: object(),
                stop_check=lambda: None, logger=logs.append, max_replans=0)

            result = executor.run("move")

            self.assertEqual(result["status"], "ABORT")
            self.assertIn("Emergency stop requested", result["reason"])
            self.assertEqual(store.summary("sim")[0]["successes"], 0)
            skill_result = next(row["result"] for row in logs if row["event"] == "skill_result")
            self.assertEqual(skill_result["result_code"], "ABORT")
            self.assertEqual(skill_result["evidence"]["postcondition"], "not_checked_emergency_stop")

    def test_interrupted_planning_request_returns_abort_not_planning_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs = []

            def interrupted_request(**_):
                raise InterruptedError("Emergency stop requested (F12)")

            planner = GoalPlanner(self.registry, interrupted_request)
            runner = GoalExecutor(planner, self.registry, SkillOutcomeStore(Path(tmp) / "outcomes.db"),
                profile=self.profile, game_id="sim",
                execute_skill=lambda *_: self.fail("interrupted planning reached a skill"),
                frame_provider=lambda: object(), stop_check=lambda: None, logger=logs.append)

            result = runner.run("move")

            self.assertEqual(result["status"], "ABORT")
            self.assertEqual(result["steps"], 0)
            self.assertFalse(any(row["event"] == "plan_rejected" for row in logs))

    def test_interrupted_stop_check_before_planning_returns_abort(self):
        with tempfile.TemporaryDirectory() as tmp:
            planner = GoalPlanner(self.registry,
                lambda **_: self.fail("emergency stop should prevent planning"))

            def interrupted_stop_check():
                raise InterruptedError("Emergency stop requested (F12)")

            runner = GoalExecutor(planner, self.registry, SkillOutcomeStore(Path(tmp) / "outcomes.db"),
                profile=self.profile, game_id="sim", execute_skill=lambda *_: None,
                frame_provider=lambda: object(), stop_check=interrupted_stop_check)

            result = runner.run("move")

            self.assertEqual(result["status"], "ABORT")
            self.assertEqual(result["steps"], 0)

    def test_interrupted_postcondition_check_records_aborted_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SkillOutcomeStore(Path(tmp) / "outcomes.db")
            logs = []
            planner = GoalPlanner(self.registry, lambda **_: plan_for())

            def interrupted_verification(*_):
                raise InterruptedError("Emergency stop requested (F12)")

            runner = GoalExecutor(planner, self.registry, store, profile=self.profile, game_id="sim",
                execute_skill=lambda *_: SkillResult(status=SkillStatus.SUCCESS, result_code="SUCCESS",
                    reason="input sent", evidence={"input_actions": 1}),
                frame_provider=lambda: object(), stop_check=lambda: None,
                result_verifier=interrupted_verification, logger=logs.append, max_replans=0)

            result = runner.run("move")

            self.assertEqual(result["status"], "ABORT")
            self.assertEqual(result["steps"], 1)
            skill_result = next(row["result"] for row in logs if row["event"] == "skill_result")
            self.assertEqual(skill_result["result_code"], "ABORT")
            self.assertEqual(skill_result["evidence"]["postcondition"], "not_checked_emergency_stop")
            self.assertEqual(store.summary("sim")[0]["successes"], 0)

    def test_region_delta_is_unconfirmed_without_a_success_verifier(self):
        with tempfile.TemporaryDirectory() as temp:
            self.profile.actions = {"primary_hold": {"kind": "mouse", "value": "left"}}
            self.profile.regions = {"hotbar": {"x1": .3, "y1": .9, "x2": .7, "y2": 1}}
            store = SkillOutcomeStore(Path(temp) / "outcomes.db")
            step = {"steps": [{"skill": "interact_until_change", "args": {
                "region": "hotbar", "action": "primary_hold", "max_repeats": 1},
                "expected_result": "item acquired", "success_check": "visible log count increased"}]}
            planner = GoalPlanner(self.registry, lambda **_: step)
            runner = GoalExecutor(planner, self.registry, store, profile=self.profile, game_id="sim",
                execute_skill=lambda *_: SkillResult(status=SkillStatus.UNCONFIRMED,
                    result_code="REGION_CHANGE_UNCONFIRMED", reason="region_changed_unverified"),
                frame_provider=lambda: Image.new("RGB", (8, 8)), stop_check=lambda: None,
                max_replans=0)
            result = runner.run("collect a log")
            self.assertEqual(result["status"], "FAILED")
            summary = store.summary("sim")[0]
            self.assertEqual(summary["successes"], 0)
            self.assertEqual(summary["uncertain"], 1)
            self.assertIsNone(summary["verified_example"])

    def test_interaction_without_region_delta_stays_uncertain_without_verifier(self):
        with tempfile.TemporaryDirectory() as temp:
            self.profile.actions = {"primary_hold": {"kind": "mouse", "value": "left"}}
            self.profile.regions = {"hotbar": {"x1": .3, "y1": .9, "x2": .7, "y2": 1}}
            step = {"steps": [{"skill": "interact_until_change", "args": {
                "region": "hotbar", "action": "primary_hold", "max_repeats": 3},
                "expected_result": "target removed", "success_check": "target disappeared"}]}
            store = SkillOutcomeStore(Path(temp) / "outcomes.db")
            runner = GoalExecutor(
                GoalPlanner(self.registry, lambda **_: step), self.registry, store,
                profile=self.profile, game_id="sim",
                execute_skill=lambda *_: SkillResult(status=SkillStatus.UNCONFIRMED,
                    result_code="INTERACTION_OUTCOME_UNCONFIRMED",
                    reason="interaction_without_region_change"),
                frame_provider=lambda: Image.new("RGB", (8, 8)), stop_check=lambda: None,
                max_replans=0)
            runner.run("remove target")
            summary = store.summary("sim")[0]
            self.assertEqual(summary["successes"], 0)
            self.assertEqual(summary["uncertain"], 1)
            self.assertIsNone(summary["verified_example"])

    def test_goal_result_is_unconfirmed_without_final_postcondition_verifier(self):
        with tempfile.TemporaryDirectory() as temp:
            store = SkillOutcomeStore(Path(temp) / "outcomes.db")
            runner = GoalExecutor(
                GoalPlanner(self.registry, lambda **_: plan_for()), self.registry, store,
                profile=self.profile, game_id="sim",
                execute_skill=lambda *_: SkillResult(status=SkillStatus.SUCCESS,
                    result_code="SUCCESS", reason="movement_sent"),
                frame_provider=lambda: object(), stop_check=lambda: None, max_replans=0)
            result = runner.run("reach the target")
            summary = store.summary("sim")[0]
            self.assertEqual((result["status"], result["reason"]),
                             ("UNCONFIRMED", "goal_result_unverified"))
            self.assertEqual(summary["successes"], 0)
            self.assertEqual(summary["uncertain"], 1)

    def test_final_postcondition_verification_promotes_command_to_verified_goal(self):
        with tempfile.TemporaryDirectory() as temp:
            store = SkillOutcomeStore(Path(temp) / "outcomes.db")
            calls = []

            def verify(step, _before, _after, result, _timeout):
                calls.append((step.expected_result, step.success_check, result["result_code"]))
                return ConfirmedAssessment()

            runner = GoalExecutor(
                GoalPlanner(self.registry, lambda **_: plan_for()), self.registry, store,
                profile=self.profile, game_id="sim",
                execute_skill=lambda *_: SkillResult(status=SkillStatus.SUCCESS,
                    result_code="SUCCESS", reason="movement_sent"),
                frame_provider=lambda: object(), stop_check=lambda: None,
                result_verifier=verify)
            result = runner.run("reach the target")
            summary = store.summary("sim")[0]
            self.assertEqual(result["status"], "SUCCESS")
            self.assertEqual(calls, [("move", "result status", "SUCCESS")])
            self.assertEqual(summary["successes"], 1)
            self.assertIsNotNone(summary["verified_example"])

    def test_success_check_can_confirm_or_reject_region_delta(self):
        class Assessment:
            def __init__(self, verdict, confidence=.95, evidence="visible item count increased"):
                self.result, self.confidence, self.evidence = verdict, confidence, evidence
                self.evidence_source = "visual_model"
                self.usable = confidence >= .85 and len(evidence.strip()) >= 8

        for verdict, confidence, evidence, expected in (
                ("progress", .95, "visible item count increased", "SUCCESS"),
                ("unknown", .95, "criterion remains unclear", "FAILED"),
                ("blocked", .95, "the target remains unchanged", "FAILED"),
                ("progress", .5, "Weak.", "FAILED")):
            with self.subTest(verdict=verdict), tempfile.TemporaryDirectory() as temp:
                self.profile.actions = {"primary_hold": {"kind": "mouse", "value": "left"}}
                self.profile.regions = {"hotbar": {"x1": .3, "y1": .9, "x2": .7, "y2": 1}}
                step = {"steps": [{"skill": "interact_until_change", "args": {
                    "region": "hotbar", "action": "primary_hold", "max_repeats": 1},
                    "expected_result": "item acquired", "success_check": "visible log count increased"}]}
                planner = GoalPlanner(self.registry, lambda **_: step)
                frames = [Image.new("RGB", (8, 8), (i, 0, 0)) for i in range(3)]
                calls = []
                def verify(plan_step, before, after, result, timeout_seconds):
                    calls.append((plan_step.success_check, before.getpixel((0, 0)), after.getpixel((0, 0))))
                    self.assertGreater(timeout_seconds, 0)
                    return Assessment(verdict, confidence, evidence)
                runner = GoalExecutor(planner, self.registry,
                    SkillOutcomeStore(Path(temp) / "outcomes.db"), profile=self.profile, game_id="sim",
                    execute_skill=lambda *_: SkillResult(status=SkillStatus.UNCONFIRMED,
                        result_code="REGION_CHANGE_UNCONFIRMED", reason="region_changed_unverified"),
                    frame_provider=lambda: frames.pop(0), stop_check=lambda: None,
                    result_verifier=verify, max_replans=0)
                result = runner.run("collect a log")
                self.assertEqual(result["status"], expected)
                self.assertEqual(calls[0][0], "visible log count increased")

    def test_success_check_can_confirm_world_progress_without_region_delta(self):
        class Assessment:
            result = "progress"
            confidence = .95
            evidence = "the target block disappeared"
            evidence_source = "visual_model"
            usable = True

        with tempfile.TemporaryDirectory() as temp:
            self.profile.actions = {"primary_hold": {"kind": "mouse", "value": "left"}}
            self.profile.regions = {"hotbar": {"x1": .3, "y1": .9, "x2": .7, "y2": 1}}
            step = {"steps": [{"skill": "interact_until_change", "args": {
                "region": "hotbar", "action": "primary_hold", "max_repeats": 3},
                "expected_result": "target removed", "success_check": "the target block disappeared"}]}
            calls = []
            planning_frame = Image.new("RGB", (8, 8), (0, 0, 0))
            before = Image.new("RGB", (8, 8), (0, 0, 0))
            after = before.copy()
            after.putpixel((2, 2), (255, 255, 255))  # Target area changed; hotbar pixels did not.
            frames = [planning_frame, before, after]

            def verify(plan_step, observed_before, observed_after, result, _timeout):
                calls.append((plan_step.success_check, result["result_code"]))
                self.assertEqual(observed_before.getpixel((0, 7)), observed_after.getpixel((0, 7)))
                self.assertNotEqual(observed_before.getpixel((2, 2)), observed_after.getpixel((2, 2)))
                return Assessment()

            store = SkillOutcomeStore(Path(temp) / "outcomes.db")
            runner = GoalExecutor(
                GoalPlanner(self.registry, lambda **_: step), self.registry,
                store, profile=self.profile, game_id="sim",
                execute_skill=lambda *_: SkillResult(status=SkillStatus.UNCONFIRMED,
                    result_code="INTERACTION_OUTCOME_UNCONFIRMED",
                    reason="interaction_without_region_change"),
                frame_provider=lambda: frames.pop(0), stop_check=lambda: None,
                result_verifier=verify, max_replans=0)
            result = runner.run("remove target")
            self.assertEqual(result["status"], "SUCCESS")
            self.assertEqual(calls, [("the target block disappeared", "INTERACTION_OUTCOME_UNCONFIRMED")])
            recorded = runner.store.summary("sim", state_hash=GameMemory.compute_state_hash(
                planning_frame))[0]
            self.assertEqual(recorded["successes"], 1)
            self.assertIsNotNone(recorded["verified_example"])

    def test_blocked_approach_uses_bounded_local_recovery_before_replanning(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SkillOutcomeStore(Path(tmp) / "outcomes.db")
            planner_calls = []
            plan = {"steps": [{"skill": "approach_target", "args": {"target_text": "a tree"},
                               "expected_result": "near", "success_check": "stable target"}]}
            planner = GoalPlanner(self.registry, lambda **_: planner_calls.append(1) or plan)
            calls = []

            def execute(spec, args, _remaining):
                calls.append((spec.name, args))
                if spec.name == "approach_target" and len(calls) == 1:
                    return SkillResult(status=SkillStatus.FAILED, result_code="FAILED",
                                       reason="blocked_after_side_step")
                return SkillResult(status=SkillStatus.SUCCESS, result_code="SUCCESS", reason="ok")

            runner = GoalExecutor(planner, self.registry, store, profile=self.profile, game_id="sim",
                execute_skill=execute, frame_provider=lambda: object(), stop_check=lambda: None,
                max_replans=0, result_verifier=confirm_goal)
            result = runner.run("get a tree")
            self.assertEqual(result["status"], "SUCCESS")
            self.assertEqual([name for name, _ in calls],
                             ["approach_target", "turn", "search_target", "approach_target"])
            self.assertEqual(len(planner_calls), 1)
            self.assertEqual(result["steps"], 4)
            approach_summary = next(item for item in store.summary("sim")
                                    if item["skill"] == "approach_target")
            self.assertEqual((approach_summary["successes"], approach_summary["attempts"],
                              approach_summary["success_rate"]), (1, 2, 0.5))
            self.assertIsNotNone(approach_summary["verified_example"])

    def test_lost_target_uses_local_search_and_reapproach(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SkillOutcomeStore(Path(tmp) / "outcomes.db")
            plan = {"steps": [{"skill": "approach_target", "args": {"target_text": "a tree"},
                               "expected_result": "near", "success_check": "stable target"}]}
            planner_calls = []
            planner = GoalPlanner(self.registry, lambda **_: planner_calls.append(1) or plan)
            calls = []

            def execute(spec, args, _remaining):
                calls.append(spec.name)
                if spec.name == "approach_target" and calls.count("approach_target") == 1:
                    return SkillResult(status=SkillStatus.FAILED, result_code="FAILED",
                                       reason="target_not_reacquired")
                return SkillResult(status=SkillStatus.SUCCESS, result_code="SUCCESS", reason="ok")

            runner = GoalExecutor(planner, self.registry, store, profile=self.profile, game_id="sim",
                execute_skill=execute, frame_provider=lambda: object(), stop_check=lambda: None,
                max_replans=0, result_verifier=confirm_goal)
            result = runner.run("get a tree")
            self.assertEqual(result["status"], "SUCCESS")
            self.assertEqual(calls, ["approach_target", "turn", "search_target", "approach_target"])
            self.assertEqual(len(planner_calls), 1)
            approach_summary = next(item for item in store.summary("sim")
                                    if item["skill"] == "approach_target")
            self.assertEqual((approach_summary["successes"], approach_summary["attempts"],
                              approach_summary["success_rate"]), (1, 2, 0.5))
            self.assertIsNotNone(approach_summary["verified_example"])

    def test_planner_failure_and_action_limit_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SkillOutcomeStore(Path(tmp) / "outcomes.db")
            calls = []
            logs = []
            unavailable = GoalPlanner(self.registry, lambda **_: (_ for _ in ()).throw(TimeoutError("offline")))
            runner = GoalExecutor(unavailable, self.registry, store, profile=self.profile, game_id="sim",
                execute_skill=lambda *_: calls.append("input"), frame_provider=lambda: object(),
                stop_check=lambda: None, logger=logs.append)
            result = runner.run("goal")
            self.assertEqual(result["status"], "FAILED")
            self.assertIn("planner_failed: Goal planner request failed: TimeoutError: offline",
                          result["reason"])
            self.assertEqual(calls, [])
            self.assertEqual(logs[0]["event"], "plan_rejected")
            self.assertEqual(len(logs[0]["request_durations_seconds"]), 1)
            self.assertGreaterEqual(logs[0]["request_durations_seconds"][0], 0)

            two_steps = {"steps": [plan_for()["steps"][0], plan_for()["steps"][0]]}
            planner = GoalPlanner(self.registry, lambda **_: two_steps)
            ran = []
            runner = GoalExecutor(planner, self.registry, store, profile=self.profile, game_id="sim",
                execute_skill=lambda *_: ran.append(1) or SkillResult(status=SkillStatus.SUCCESS,
                    result_code="SUCCESS", reason="ok"), frame_provider=lambda: object(), stop_check=lambda: None,
                max_steps=1, result_verifier=confirm_goal)
            self.assertEqual(runner.run("goal")["reason"], "max_skill_steps")
            self.assertEqual(len(ran), 1)

    def test_total_time_limit_prevents_next_skill(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SkillOutcomeStore(Path(tmp) / "outcomes.db")
            self.profile.motion_limits.max_run_seconds = 1
            planner = GoalPlanner(self.registry, lambda **_: plan_for())
            tick_values = iter((0.0, 0.0, 2.0))
            ran = []
            runner = GoalExecutor(planner, self.registry, store, profile=self.profile, game_id="sim",
                execute_skill=lambda *_: ran.append(1), frame_provider=lambda: object(),
                stop_check=lambda: None, clock=lambda: next(tick_values))
            self.assertEqual(runner.run("goal")["reason"], "max_run_seconds")
            self.assertEqual(ran, [])

    def test_vlm_goal_request_uses_strict_plan_schema(self):
        brain = VisionBrain()
        response = Mock()
        response.json.return_value = {"choices": [{"message": {"content": '{"steps": []}'}}]}
        response.raise_for_status.return_value = None
        requests = []
        brain._post = lambda *args, **kwargs: requests.append((args, kwargs)) or response
        result = brain.plan_goal(Image.new("RGB", (16, 16)), goal="go", profile=self.profile,
                                 skills=self.registry.planner_catalog())
        self.assertEqual(result, {"steps": []})
        payload = requests[0][1]["json"]
        self.assertIn("immediate next visible screen/state after that one click",
                      payload["messages"][0]["content"])
        self.assertEqual(payload["response_format"]["json_schema"]["name"], "GoalPlan")
        self.assertEqual(payload["response_format"]["json_schema"]["schema"]["properties"]["steps"]["maxItems"], 6)
        self.assertEqual(payload["max_tokens"], 256)
        self.assertEqual(payload["messages"][1]["content"][0]["type"], "text")


if __name__ == "__main__":
    unittest.main()
