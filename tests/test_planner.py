import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from PIL import Image
from omnigamer.planner import (BudgetControls, GoalExecutor, GoalPlan, GoalPlanner,
                               InputActionBudget, InputActionLimitExceeded,
                               PlanningError, SkillOutcomeStore)
from omnigamer.profiles import GameProfile
from omnigamer.skills.contracts import SkillResult, SkillStatus
from omnigamer.skills.registry import default_registry
from omnigamer.vision_brain import VisionBrain


def plan_for(skill="move_pulse", args=None):
    return {"steps": [{"skill": skill, "args": args or {"direction": "forward", "seconds": 0.5},
                       "expected_result": "move", "success_check": "result status"}]}


class GoalPlannerTests(unittest.TestCase):
    def setUp(self):
        self.profile = GameProfile(game_id="sim", motion_limits={"max_run_seconds": 20, "max_actions": 3})
        self.registry = default_registry()

    def test_rejects_unknown_skill_and_invalid_arguments(self):
        planner = GoalPlanner(self.registry, lambda **_: plan_for("ghost"))
        with self.assertRaises(PlanningError):
            planner.plan(frame=None, goal="x", profile=self.profile)

    def test_plan_schema_accepts_six_steps_and_rejects_seven(self):
        step = plan_for()["steps"][0]
        valid = GoalPlan.model_validate({"steps": [step] * 6})
        self.assertEqual(len(valid.steps), 6)
        with self.assertRaises(ValueError):
            GoalPlan.model_validate({"steps": [step] * 7})

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
            planner = GoalPlanner(self.registry, lambda **_: calls.append("plan") or plan_for())

            def execute_skill(*_args):
                controls = BudgetControls(raw_controls, budget)
                try:
                    controls.press_key("w", 0.1)
                    controls.move_camera(10, 0)
                    controls.hold_mouse("left", 0.1)
                except InputActionLimitExceeded as exc:
                    return SkillResult(status=SkillStatus.FAILED, result_code="FAILED", reason=str(exc))

            runner = GoalExecutor(planner, self.registry, SkillOutcomeStore(Path(temp) / "outcomes.db"),
                profile=self.profile, game_id="sim", execute_skill=execute_skill,
                frame_provider=lambda: object(), stop_check=lambda: None,
                input_action_budget=budget)
            result = runner.run("move")
            self.assertEqual(result["reason"], "max_actions")
            self.assertEqual(result["input_actions_used"], 2)
            self.assertEqual(len(calls), 1)
            self.assertEqual(raw_controls.hold_mouse.call_count, 0)
        planner = GoalPlanner(self.registry, lambda **_: plan_for(args={"direction": "up", "seconds": 99}))
        with self.assertRaises(PlanningError):
            planner.plan(frame=None, goal="x", profile=self.profile)

    def test_invalid_plan_gets_one_retry_with_error(self):
        calls = []
        def respond(**kwargs):
            calls.append(kwargs)
            return plan_for("unknown") if len(calls) == 1 else plan_for()
        plan = GoalPlanner(self.registry, respond).plan(frame=None, goal="move", profile=self.profile)
        self.assertEqual(plan.steps[0].skill, "move_pulse")
        self.assertEqual(len(calls), 2)
        self.assertIn("Unknown skill", calls[1]["validation_error"])

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

    def test_outcome_store_separates_games_and_summarizes_success_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SkillOutcomeStore(Path(tmp) / "outcomes.db")
            store.record("a", "move_pulse", "move", "SUCCESS", 1, "r1")
            store.record("a", "move_pulse", "move", "FAILED", 2, "r2")
            store.record("b", "move_pulse", "move", "SUCCESS", 1, "r3")
            self.assertEqual(store.summary("a"), [{"skill": "move_pulse", "target": "move",
                                                    "successes": 1, "attempts": 2, "success_rate": 0.5}])
            self.assertEqual(store.summary("b")[0]["attempts"], 1)

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
                max_replans=0)
            result = runner.run("get a tree")
            self.assertEqual(result["status"], "SUCCESS")
            self.assertEqual([name for name, _ in calls],
                             ["approach_target", "turn", "search_target", "approach_target"])
            self.assertEqual(len(planner_calls), 1)
            self.assertEqual(result["steps"], 4)

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
                max_replans=0)
            result = runner.run("get a tree")
            self.assertEqual(result["status"], "SUCCESS")
            self.assertEqual(calls, ["approach_target", "turn", "search_target", "approach_target"])
            self.assertEqual(len(planner_calls), 1)

    def test_planner_failure_and_action_limit_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SkillOutcomeStore(Path(tmp) / "outcomes.db")
            calls = []
            unavailable = GoalPlanner(self.registry, lambda **_: (_ for _ in ()).throw(TimeoutError("offline")))
            runner = GoalExecutor(unavailable, self.registry, store, profile=self.profile, game_id="sim",
                execute_skill=lambda *_: calls.append("input"), frame_provider=lambda: object(), stop_check=lambda: None)
            with self.assertRaises(TimeoutError):
                runner.run("goal")
            self.assertEqual(calls, [])

            two_steps = {"steps": [plan_for()["steps"][0], plan_for()["steps"][0]]}
            planner = GoalPlanner(self.registry, lambda **_: two_steps)
            ran = []
            runner = GoalExecutor(planner, self.registry, store, profile=self.profile, game_id="sim",
                execute_skill=lambda *_: ran.append(1) or SkillResult(status=SkillStatus.SUCCESS,
                    result_code="SUCCESS", reason="ok"), frame_provider=lambda: object(), stop_check=lambda: None,
                max_steps=1)
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
        self.assertEqual(payload["response_format"]["json_schema"]["name"], "GoalPlan")
        self.assertEqual(payload["response_format"]["json_schema"]["schema"]["properties"]["steps"]["maxItems"], 6)
        self.assertEqual(payload["max_tokens"], 256)
        self.assertEqual(payload["messages"][1]["content"][0]["type"], "text")


if __name__ == "__main__":
    unittest.main()
