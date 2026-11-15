import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from omnigamer.skills.approach_target import (
    ApproachTargetSettings, ApproachTargetSkill, action_for_box,
)
from omnigamer.detector_client import preprocess_image
from omnigamer.profiles import GameProfile, load_profile
from tools.summarize_run import summarize


def box(center, height, score=0.9):
    return {"x1": center - 0.04, "x2": center + 0.04,
            "y1": 0.5 - height / 2, "y2": 0.5 + height / 2, "score": score}


def reach_box(center=0.5):
    return {"x1": center - 0.1, "x2": center + 0.1,
            "y1": 0.0, "y2": 0.82, "score": 0.9}


class FakeCapture:
    def __init__(self):
        self.index = 0

    def focus(self):
        pass

    def grab_frame(self):
        self.index += 1
        return Image.new("RGB", (100, 100), (self.index % 255, 0, 0))


class FakeDetector:
    def __init__(self, sequence):
        self.sequence = list(sequence)
        self.calls = 0

    def detect(self, image, queries, **kwargs):
        result = self.sequence[min(self.calls, len(self.sequence) - 1)]
        self.calls += 1
        if result is None:
            return None
        return {"boxes": result, "latency_ms": 3.0, "model": "fake"}


class FakeControls:
    def __init__(self):
        self.calls = []
        self.released = False

    def press_key(self, key, duration):
        self.calls.append(("key", key, duration))

    def press_keys(self, keys, duration):
        self.calls.append(("chord", list(keys), duration))

    def move_camera(self, dx, dy):
        self.calls.append(("look", dx, dy))

    def release_all(self):
        self.released = True


class FakeGuard:
    def filter(self, action):
        return (dict(action, action_type="look", x=0, y=0), None)

    def executed(self, action):
        pass


class StepClock:
    def __init__(self, increment=0.01):
        self.value = 0.0
        self.increment = increment

    def __call__(self):
        self.value += self.increment
        return self.value


class ApproachTargetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.calibration = Path(self.temp.name) / "calibration.json"
        self.calibration.write_text(json.dumps({"game:test": {"k": 0.01, "invert": False}}),
                                    encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def make_skill(self, sequence, settings=None, **kwargs):
        capture = FakeCapture()
        detector = FakeDetector(sequence)
        controls = FakeControls()
        calibration_path = kwargs.pop("calibration_path", self.calibration)
        skill = ApproachTargetSkill(
            capture=capture, target_text="a tree trunk", detector=detector,
            controls=controls, camera_guard=FakeGuard(), game_id="game:test",
            calibration_path=calibration_path, settings=settings or ApproachTargetSettings(),
            **kwargs,
        )
        return skill, controls, detector

    def test_reaches_by_height_and_releases_controls(self):
        skill, controls, detector = self.make_skill([
            [box(0.5, 0.2)], [reach_box()], [reach_box()],
        ])
        result = skill.run()
        self.assertEqual(result.state, "REACHED")
        self.assertEqual(result.criterion, "top_edge_centered_stable")
        self.assertEqual([call[1] for call in controls.calls if call[0] == "key"], ["w"])
        self.assertTrue(controls.released)
        self.assertEqual(detector.calls, 3)

    def test_reaches_by_centered_height_plateau(self):
        skill, _, _ = self.make_skill([
            [box(0.5, 0.50)], [reach_box()], [reach_box()],
        ])
        result = skill.run()
        self.assertEqual(result.state, "REACHED")
        self.assertEqual(result.criterion, "top_edge_centered_stable")

    def test_reach_requires_crosshair_margin_and_corrects_vertical_aim(self):
        near_crosshair = {"x1": 0.4, "x2": 0.6, "y1": 0.0, "y2": 0.53, "score": 0.9}
        aimed = {"x1": 0.4, "x2": 0.6, "y1": 0.0, "y2": 0.60, "score": 0.9}
        skill, controls, _ = self.make_skill(
            [[box(0.5, 0.2)], [near_crosshair], [aimed], [aimed]])
        result = skill.run()
        self.assertEqual(result.state, "REACHED")
        self.assertIn(("look", 0, -12), controls.calls)
        self.assertEqual(sum(call == ("key", "w", 0.5) for call in controls.calls), 1)

    def test_reached_pose_gets_neutral_confirmation_without_extra_movement(self):
        skill, controls, detector = self.make_skill([[reach_box()], [reach_box()], [reach_box()]])
        result = skill.run()
        self.assertEqual(result.state, "REACHED")
        self.assertEqual(result.iterations, 2)
        self.assertEqual(detector.calls, 3)
        self.assertEqual(controls.calls, [])

    def test_fallback_query_can_guide_approach_but_cannot_confirm_reached(self):
        skill, _, _ = self.make_skill([[]])
        broad_fallback = {**reach_box(), "query": "a tree"}
        exact_target = {**reach_box(), "query": "a tree trunk"}
        self.assertFalse(skill._reach_geometry(broad_fallback)[0])
        self.assertTrue(skill._reach_geometry(exact_target)[0])

    def test_preprocess_modes_return_image_and_parameters(self):
        image = Image.new("RGB", (64, 64), (20, 30, 40))
        for mode in ("none", "gamma", "clahe", "stretch"):
            processed, params = preprocess_image(image, mode)
            self.assertEqual(processed.size, image.size)
            self.assertEqual(params["mode"], mode)
        with self.assertRaises(ValueError):
            preprocess_image(image, "unrecognized")

    def test_detector_fallback_query_uses_geometry_checks_and_logs_raw_boxes(self):
        class ResponseDetector:
            def detect(self, image, queries, **kwargs):
                self.queries = queries
                return {"boxes": [
                    {"query": "a tree", "score": 0.8, "x1": 0.1, "y1": 0.1,
                     "x2": 0.9, "y2": 0.8},
                    {"query": "a tree", "score": 0.7, "x1": 0.35, "y1": 0.1,
                     "x2": 0.55, "y2": 0.8},
                ], "latency_ms": 5, "preprocess": {"mode": "stretch"}}
        skill, _, _ = self.make_skill([[]], profile=load_profile("minecraft"))
        detector = ResponseDetector()
        skill.detector = detector
        detection, _ = skill._detect(Image.new("RGB", (10, 10)))
        self.assertEqual(detector.queries, ["a tree trunk", "a tree"])
        self.assertEqual(len(detection["raw_boxes"]), 2)
        self.assertEqual(len(detection["boxes"]), 1)
        self.assertEqual(detection["selected_query_source"], "fallback_query")
        self.assertEqual(detection["boxes"][0]["x1"], 0.35)

    def test_detector_prefers_more_specific_fallback_query(self):
        class ResponseDetector:
            def detect(self, image, queries, **kwargs):
                return {"boxes": [
                    {"query": "a tree", "score": 0.8, "x1": 0.4, "y1": 0.1,
                     "x2": 0.6, "y2": 0.8},
                    {"query": "a tree trunk", "score": 0.2, "x1": 0.47, "y1": 0.4,
                     "x2": 0.53, "y2": 0.8},
                ]}
        skill, _, _ = self.make_skill([[]], profile=load_profile("minecraft"))
        skill.detector = ResponseDetector()
        detection, _ = skill._detect(Image.new("RGB", (10, 10)))
        self.assertEqual([box["query"] for box in detection["boxes"]], ["a tree trunk"])
        self.assertEqual(detection["boxes"][0]["score"], 0.2)

    def test_target_tracker_rejects_distant_box_without_rebinding(self):
        skill, _, _ = self.make_skill([[]], settings=ApproachTargetSettings(
            track_max_center_shift=0.1))
        first = box(0.5, 0.2)
        self.assertEqual(skill._select({"boxes": [first]}), first)
        original_center = skill._last_center
        self.assertIsNone(skill._select({"boxes": [box(0.685, 0.2)]}))
        self.assertEqual(skill._last_center, original_center)
        nearby = box(0.58, 0.2)
        self.assertEqual(skill._select({"boxes": [nearby]}), nearby)

    def test_target_tracker_reports_symmetric_candidates_as_uncertain(self):
        skill, _, _ = self.make_skill([[]], settings=ApproachTargetSettings(
            track_max_center_shift=0.2, track_ambiguity_margin=0.02))
        original = box(0.5, 0.2)
        self.assertEqual(skill._select({"boxes": [original]}), original)
        original_center = skill._last_center
        left, right = box(0.45, 0.2), box(0.55, 0.2)
        self.assertIsNone(skill._select({"boxes": [left, right]}))
        self.assertTrue(skill._selection_uncertain)
        self.assertEqual(skill._last_center, original_center)

    def test_target_tracker_uses_box_scale_to_break_close_center_tie(self):
        skill, _, _ = self.make_skill([[]], settings=ApproachTargetSettings(
            track_max_center_shift=0.2, track_ambiguity_margin=0.02))
        original = box(0.5, 0.2)
        skill._select({"boxes": [original]})
        similar_shape = box(0.53, 0.2)
        nearer_but_mismatched = {"x1": 0.36, "x2": 0.66, "y1": 0.2, "y2": 0.8,
                                 "score": 0.99}

        selected = skill._select({"boxes": [nearer_but_mismatched, similar_shape]})

        self.assertEqual(selected, similar_shape)
        self.assertFalse(skill._selection_uncertain)
        self.assertAlmostEqual(skill._last_center[0], 0.53)

    def test_target_tracker_ignores_geometry_match_outside_center_shift_limit(self):
        skill, _, _ = self.make_skill([[]], settings=ApproachTargetSettings(
            track_max_center_shift=0.1, track_ambiguity_margin=0.005))
        original = box(0.5, 0.2)
        skill._select({"boxes": [original]})
        eligible = {"x1": 0.44, "x2": 0.74, "y1": 0.2, "y2": 0.8, "score": 0.9}
        beyond_limit = box(0.61, 0.2)

        selected = skill._select({"boxes": [beyond_limit, eligible]})

        self.assertEqual(selected, eligible)

    def test_repeated_ambiguous_tracking_is_unconfirmed_without_blind_movement(self):
        original = box(0.5, 0.2)
        ambiguous = [box(0.45, 0.2), box(0.55, 0.2)]
        skill, controls, _ = self.make_skill(
            [[original], ambiguous, ambiguous],
            settings=ApproachTargetSettings(track_max_center_shift=0.2,
                                            track_ambiguity_margin=0.02))
        result = skill.run()
        self.assertEqual(result.state, "UNCERTAIN")
        self.assertEqual(result.reason, "ambiguous_target_tracking")
        self.assertEqual([call for call in controls.calls if call[0] == "key"],
                         [("key", "w", 0.5)])
        self.assertTrue(controls.released)

    def test_profile_action_budget_counts_camera_commands(self):
        settings = ApproachTargetSettings(max_impulses=1)
        skill, controls, _ = self.make_skill([[box(0.8, 0.2)]], settings=settings)
        result = skill.run()
        self.assertEqual(result.state, "TIMEOUT")
        self.assertEqual(result.reason, "max_actions")
        self.assertEqual(result.input_actions, 1)
        self.assertEqual(result.impulses, 0)
        self.assertEqual(len([call for call in controls.calls if call[0] == "look"]), 1)

    def test_profile_jump_action_recovers_from_repeated_no_growth_after_side_step(self):
        profile = load_profile("minecraft")
        stable = box(0.5, 0.3)
        skill, controls, _ = self.make_skill(
            [[stable]] * 12 + [[reach_box()], [reach_box()]], profile=profile,
            settings=ApproachTargetSettings.from_profile(profile))
        result = skill.run()
        self.assertEqual(result.state, "REACHED")
        self.assertIn(("chord", ["w", "space"], 0.6), controls.calls)
        self.assertEqual(sum(call[0] == "chord" for call in controls.calls), 2)
        self.assertTrue(controls.released)

    def test_forbidden_jump_key_is_not_used_for_blocked_recovery(self):
        profile = load_profile("minecraft")
        profile.forbidden_keys.append("space")
        stable = box(0.5, 0.3)
        skill, controls, _ = self.make_skill([[stable]] * 12, profile=profile)
        result = skill.run()
        self.assertEqual(result.reason, "blocked_after_side_step")
        self.assertFalse(any(call[0] == "chord" for call in controls.calls))

    def test_close_target_held_is_bounded_and_never_claims_success(self):
        settings = ApproachTargetSettings(close_target_held_impulses=2,
                                          close_target_held_duration=0.2)
        skill, controls, _ = self.make_skill([[reach_box()], [], [], [], [], []], settings=settings)
        rows = []
        skill.recorder = lambda row, before, after: rows.append(row)
        result = skill.run()
        cautious = [call for call in controls.calls if call[:2] == ("key", "w") and call[2] == 0.2]
        self.assertEqual(result.state, "REACHED_UNCONFIRMED")
        self.assertEqual(result.reason, "target_occluded_after_close_approach")
        self.assertEqual(len(cautious), 2)
        self.assertTrue(all(call[2] == 0.2 for call in cautious))
        self.assertTrue(any(row["state"] == "CLOSE_TARGET_HELD" for row in rows))

    def test_look_retry_rechecks_without_counting_a_miss(self):
        class RetryDetector(FakeDetector):
            pass
        skill, controls, detector = self.make_skill([
            [box(0.95, 0.2)], [], [box(0.5, 0.2)], [box(0.5, 0.2)],
        ], settings=ApproachTargetSettings(max_impulses=2))
        rows = []
        skill.recorder = lambda row, before, after: rows.append(row)
        skill.run()
        self.assertGreaterEqual(detector.calls, 4)
        self.assertEqual(rows[0]["detector"]["after"]["raw_boxes"][0]["x1"], 0.46)
        self.assertEqual(len(rows[0]["detector"]["after"]["retry_attempts"]), 2)
        self.assertEqual([call for call in controls.calls if call[0] == "look"][0][1], 30)

    def test_lost_target_search_is_bounded(self):
        skill, controls, _ = self.make_skill([[], [], [], [], [], [], [], []])
        result = skill.run()
        looks = [call for call in controls.calls if call[0] == "look"]
        self.assertEqual(result.state, "LOST")
        self.assertEqual(len(looks), 6)
        self.assertEqual([call[1] > 0 for call in looks], [False, True, False, True, False, True])

    def test_blocked_uses_alternating_side_step_then_exits(self):
        sequence = [[box(0.5, 0.2)]] * 12
        skill, controls, _ = self.make_skill(sequence)
        result = skill.run()
        side_steps = [call[1] for call in controls.calls if call[0] == "key" and call[1] in {"a", "d"}]
        self.assertEqual(result.state, "BLOCKED")
        self.assertEqual(side_steps, ["a", "d"])

    def test_calibration_inverts_camera_sign_and_saves_by_game(self):
        self.calibration.unlink()
        skill, controls, _ = self.make_skill([
            [box(0.60, 0.2)], [box(0.62, 0.2)], [box(0.62, 0.2)], [box(0.50, 0.2)],
        ])
        skill.run()
        stored = json.loads(self.calibration.read_text(encoding="utf-8"))["game:test"]
        self.assertTrue(stored["invert"])
        looks = [call[1] for call in controls.calls if call[0] == "look"]
        self.assertGreater(looks[0], 0)
        self.assertLess(looks[1], 0)

    def test_profile_camera_calibration_is_used_when_no_saved_calibration_exists(self):
        missing = Path(self.temp.name) / "missing-calibration.json"
        skill, _, _ = self.make_skill([[box(0.5, 0.2)]], profile=GameProfile(
            game_id="test", calibration={"camera_counts_per_radian": 0.02,
                                           "invert_horizontal": True}),
            calibration_path=missing)
        self.assertEqual(skill._calibration, {"k": 0.02, "invert": True})

    def test_emergency_aborts_and_releases_inputs(self):
        def stop():
            raise InterruptedError("F12")
        skill, controls, _ = self.make_skill([[box(0.5, 0.2)]], stop_check=stop)
        result = skill.run()
        self.assertEqual(result.state, "ABORT")
        self.assertIn("InterruptedError", result.reason)
        self.assertTrue(controls.released)

    def test_timeout_is_bounded(self):
        settings = ApproachTargetSettings(max_seconds=0.04, max_impulses=40)
        skill, _, _ = self.make_skill([[box(0.5, 0.2)]], settings=settings,
                                      clock=StepClock(0.02))
        result = skill.run()
        self.assertEqual(result.state, "TIMEOUT")
        self.assertEqual(result.reason, "max_seconds")

    def test_detector_unavailable_aborts_without_vlm(self):
        skill, controls, detector = self.make_skill([None])
        result = skill.run()
        self.assertEqual(result.state, "ABORT")
        self.assertEqual(result.reason, "detector_unavailable")
        self.assertEqual(detector.calls, 1)
        self.assertFalse(hasattr(skill, "brain"))
        self.assertTrue(controls.released)

    def test_dry_run_classification(self):
        settings = ApproachTargetSettings()
        self.assertEqual(action_for_box(None, settings), "acquire")
        self.assertEqual(action_for_box(box(0.8, 0.2), settings), "align")
        self.assertEqual(action_for_box(box(0.5, 0.2), settings), "advance")
        self.assertEqual(action_for_box(reach_box(), settings, reach_box()), "reached")
        self.assertEqual(action_for_box(reach_box(), settings), "advance")

    def test_summary_includes_skill_result_latency_and_height_curve(self):
        run_dir = Path(self.temp.name) / "run"
        run_dir.mkdir()
        (run_dir / "run_meta.json").write_text(json.dumps({
            "run_id": "run", "started_at": "2026-01-01T00:00:00+00:00",
            "ended_at": "2026-01-01T00:00:02+00:00", "termination_reason": "skill_reached",
            "steps_recorded": 0, "skill_result": {"state": "REACHED", "reason": "reach_height",
                                                     "iterations": 2, "elapsed_seconds": 1.8},
        }), encoding="utf-8")
        (run_dir / "skill_steps.jsonl").write_text(
            '{"iteration":1,"detector_latency_ms":{"before":100,"after":120},"h":{"after":0.4}}\n'
            '{"iteration":2,"detector_latency_ms":{"before":130,"after":150},"h":{"after":0.8}}\n',
            encoding="utf-8")
        text = summarize(run_dir)
        self.assertIn("Состояние: REACHED", text)
        self.assertIn("Средняя задержка детектора: 125.00 мс", text)
        self.assertIn("1=0.400, 2=0.800", text)

    def test_summary_includes_goal_plans_and_dry_run_actions(self):
        run_dir = Path(self.temp.name) / "goal-run"
        run_dir.mkdir()
        (run_dir / "run_meta.json").write_text(json.dumps({"run_id": "goal-run"}), encoding="utf-8")
        (run_dir / "goal_steps.jsonl").write_text(
            '{"schema_version":1,"event":"plan","steps":[]}\n'
            '{"schema_version":1,"event":"plan","replan_index":1,"steps":[]}\n'
            '{"schema_version":1,"event":"would_send","skill":"turn"}\n'
            '{"schema_version":1,"event":"run_finish","status":"DRY_RUN","reason":"plan_completed"}\n',
            encoding="utf-8")
        text = summarize(run_dir)
        self.assertIn("Проверенных планов: 2; перепланирований: 1", text)
        self.assertIn("would_send: 1", text)
        self.assertIn("DRY_RUN — plan_completed", text)


if __name__ == "__main__":
    unittest.main()
