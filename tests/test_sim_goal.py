import tempfile
import unittest
from pathlib import Path

from omnigamer.planner import GoalExecutor, GoalPlanner, SkillOutcomeStore
from omnigamer.perception import VisualAssessment
from omnigamer.skills.registry import default_registry
from omnigamer.supervision import CameraGuard
from tests.sim.environment import (SimCapture, SimControls, SimDetector, SimRecorder,
                                  SimState, make_profile_a, make_profile_b)


class SyntheticGoalPathTests(unittest.TestCase):
    def test_sim_target_shifts_on_turn_and_grows_on_forward_motion(self):
        profile = make_profile_a()
        state = SimState(profile, target_after_turns=1)
        state.heading = 1
        capture, controls, detector = SimCapture(state), SimControls(state), SimDetector(state)
        before = detector.detect(capture.grab_frame(), ["marker"])["boxes"][0]
        controls.move_camera(5, 0)
        turned = detector.detect(capture.grab_frame(), ["marker"])["boxes"][0]
        controls.press_key(profile.move_keys["forward"], 0.5)
        advanced = detector.detect(capture.grab_frame(), ["marker"])["boxes"][0]
        self.assertLess((turned["x1"] + turned["x2"]) / 2,
                        (before["x1"] + before["x2"]) / 2)
        self.assertGreater(advanced["x2"] - advanced["x1"],
                           turned["x2"] - turned["x1"])

    def run_profile(self, profile):
        state = SimState(profile)
        capture, controls, detector = SimCapture(state), SimControls(state), SimDetector(state)
        registry = default_registry()
        answer = {"steps": [
            {"skill": "search_target", "args": {"target_text": "marker", "sweep": 3},
             "expected_result": "marker detected", "success_check": "detector returned a box"},
            {"skill": "interact_until_change", "args": {
                "region": next(iter(profile.regions)), "action": next(iter(profile.actions)), "max_repeats": 2},
             "expected_result": "panel changes", "success_check": "region metric exceeds threshold"},
        ]}
        planner = GoalPlanner(registry, lambda **_: answer)

        def execute(spec, args, remaining_seconds=None):
            recorder = SimRecorder()
            deps = {"capture": capture, "controls": controls, "camera_guard": CameraGuard(),
                    "profile": profile, "stop_check": lambda: None, "recorder": recorder}
            if spec.name in {"search_target", "wait_for_condition"}:
                deps["detector"] = detector
            return spec.executor(args, **deps)

        with tempfile.TemporaryDirectory() as temp:
            store = SkillOutcomeStore(Path(temp) / "outcomes.db")
            def verify_visible_result(step, before, after, result, timeout_seconds):
                region = profile.regions[step.args["region"]]
                x = int((region.x1 + region.x2) * .5 * after.width)
                y = int((region.y1 + region.y2) * .5 * after.height)
                visible = after.getpixel((x, y)) == (245, 190, 35)
                return VisualAssessment(result="progress" if visible else "unknown",
                    confidence=.95 if visible else .2,
                    evidence="Synthetic status marker appeared" if visible else "No marker evidence")
            runner = GoalExecutor(planner, registry, store, profile=profile, game_id=profile.game_id,
                                  execute_skill=execute, frame_provider=capture.grab_frame,
                                  stop_check=lambda: None, result_verifier=verify_visible_result)
            result = runner.run("find the visible marker and activate it")
            self.assertEqual(result["status"], "SUCCESS")
            self.assertTrue(state.item)
            self.assertEqual(state.actions, 2)
            self.assertEqual(len(store.summary(profile.game_id)), 2)
            return result

    def test_complete_goal_pipeline_with_two_distinct_profiles(self):
        self.run_profile(make_profile_a())
        self.run_profile(make_profile_b())

    def test_detector_failure_does_not_report_success(self):
        profile = make_profile_a()
        state = SimState(profile)
        state.detector_available = False
        self.assertIsNone(SimDetector(state).detect(SimCapture(state).grab_frame(), ["marker"]))
        result = self._run_one_skill(profile, state, {"skill": "search_target",
            "args": {"target_text": "marker", "sweep": 2}, "expected_result": "visible",
            "success_check": "detector box"})
        self.assertEqual(result["status"], "FAILED")
        self.assertIn("detector_unavailable", result["reason"])

    def test_emergency_check_aborts_before_planning(self):
        profile, registry = make_profile_a(), default_registry()
        calls = []
        planner = GoalPlanner(registry, lambda **_: calls.append("plan"))
        with tempfile.TemporaryDirectory() as temp:
            runner = GoalExecutor(planner, registry, SkillOutcomeStore(Path(temp) / "x.db"),
                profile=profile, game_id=profile.game_id, execute_skill=lambda *_: None,
                frame_provider=lambda: None,
                stop_check=lambda: (_ for _ in ()).throw(InterruptedError("emergency")))
            result = runner.run("goal")
            self.assertEqual(result["status"], "ABORT")
            self.assertIn("emergency", result["reason"])
        self.assertEqual(calls, [])

    def test_lost_target_is_reported_failed(self):
        profile = make_profile_a()
        state = SimState(profile, target_after_turns=100)
        result = self._run_one_skill(profile, state, {"skill": "search_target",
            "args": {"target_text": "marker", "sweep": 2}, "expected_result": "visible",
            "success_check": "detector box"})
        self.assertEqual(result["status"], "FAILED")
        self.assertIn("target_not_found", result["reason"])

    def test_forbidden_profile_action_key_is_rejected(self):
        profile = make_profile_b()
        profile.forbidden_keys = ["enter"]
        state = SimState(profile)
        result = self._run_one_skill(profile, state, {"skill": "interact_until_change",
            "args": {"region": "status_strip", "action": "activate", "max_repeats": 1},
            "expected_result": "changed", "success_check": "region delta"})
        self.assertEqual(result["status"], "FAILED")
        self.assertFalse(state.item)
        self.assertEqual(state.actions, 0)

    def _run_one_skill(self, profile, state, step):
        capture, controls, detector = SimCapture(state), SimControls(state), SimDetector(state)
        registry = default_registry()
        planner = GoalPlanner(registry, lambda **_: {"steps": [step]})
        def execute(spec, args, remaining_seconds=None):
            recorder = SimRecorder()
            deps = {"capture": capture, "controls": controls, "camera_guard": CameraGuard(),
                    "profile": profile, "stop_check": lambda: None, "recorder": recorder}
            if spec.name in {"search_target", "wait_for_condition"}:
                deps["detector"] = detector
            return spec.executor(args, **deps)
        with tempfile.TemporaryDirectory() as temp:
            runner = GoalExecutor(planner, registry, SkillOutcomeStore(Path(temp) / "x.db"),
                profile=profile, game_id=profile.game_id, execute_skill=execute,
                frame_provider=capture.grab_frame, stop_check=lambda: None)
            return runner.run("target")


if __name__ == "__main__":
    unittest.main()
