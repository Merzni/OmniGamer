import unittest
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

from omnigamer.profiles import NormalizedRegion, load_profile
from omnigamer.skills.basic import (
    interact_until_change,
    move_pulse,
    region_delta,
    search_target,
    turn,
    wait_for_condition,
)
from omnigamer.supervision import CameraGuard
from omnigamer.skills.contracts import SkillStatus
from omnigamer.skills.registry import execute_move_pulse


class FakeCapture:
    def __init__(self, frames):
        self.frames = list(frames)
        self.index = 0

    def grab_frame(self):
        frame = self.frames[min(self.index, len(self.frames) - 1)]
        self.index += 1
        return frame.copy()


class FakeControls:
    def __init__(self):
        self.calls = []

    def press_key(self, key, duration):
        self.calls.append(("key", key, duration))

    def hold_mouse(self, button, duration):
        self.calls.append(("mouse", button, duration))

    def move_camera(self, dx, dy):
        self.calls.append(("look", dx, dy))

    def release_all(self):
        self.calls.append(("release",))


class FakeDetector:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def detect(self, frame, queries, **kwargs):
        self.calls += 1
        return self.responses[min(self.calls - 1, len(self.responses) - 1)]


class BasicSkillTests(unittest.TestCase):
    def setUp(self):
        self.profile = load_profile("minecraft")
        self.guard = CameraGuard()

    def test_move_pulse_resolves_profile_direction_and_records_common_row(self):
        controls = FakeControls()
        rows = []
        result = move_pulse("forward", 0.5, controls=controls, profile=self.profile,
                            camera_guard=self.guard, recorder=lambda row, before, after: rows.append(row))
        self.assertEqual(result.status, SkillStatus.SUCCESS)
        self.assertEqual(controls.calls, [("key", "w", 0.5)])
        self.assertEqual(rows[0]["schema_version"], 1)
        self.assertEqual(rows[0]["skill"], "move_pulse")

    def test_move_pulse_rejects_profile_duration_over_limit(self):
        with self.assertRaisesRegex(ValueError, "max_action_seconds"):
            move_pulse("forward", 1.0, controls=FakeControls(), profile=self.profile,
                       camera_guard=self.guard)

    def test_region_delta_measures_only_requested_normalized_crop(self):
        before = Image.new("RGB", (20, 20), (0, 0, 0))
        after = before.copy()
        ImageDraw.Draw(after).rectangle((5, 17, 15, 19), fill=(255, 255, 255))
        item_delta = region_delta(before, after, self.profile.regions["hotbar"])
        self.assertGreater(item_delta, 0.08)
        top_half = NormalizedRegion(x1=0, y1=0, x2=1, y2=0.5)
        self.assertEqual(region_delta(before, after, top_half), 0)

    def test_region_delta_suppresses_low_amplitude_pixel_noise(self):
        before = Image.new("RGB", (20, 20), (100, 100, 100))
        noise = Image.new("RGB", (20, 20), (102, 102, 102))
        self.assertEqual(region_delta(before, noise, self.profile.regions["hotbar"]), 0)

    def test_turn_is_clamped_by_profile(self):
        controls = FakeControls()
        result = turn("right", 500, controls=controls, camera_guard=self.guard,
                      profile=self.profile)
        self.assertEqual(result.status, SkillStatus.SUCCESS)
        self.assertTrue(result.evidence["clamped"])
        self.assertLessEqual(abs(controls.calls[0][1]), self.profile.motion_limits.max_turn_counts)

    def test_search_stops_after_first_visible_target(self):
        frame = Image.new("RGB", (20, 20))
        detector = FakeDetector([{"boxes": [{"x1": .2, "y1": .2, "x2": .5, "y2": .6,
                                               "score": .8, "query": "a tree trunk"}]}])
        controls = FakeControls()
        result = search_target("a tree trunk", 6, capture=FakeCapture([frame]), detector=detector,
                               controls=controls, camera_guard=self.guard, profile=self.profile)
        self.assertEqual(result.status, SkillStatus.SUCCESS)
        self.assertEqual(detector.calls, 1)
        self.assertEqual(controls.calls, [])

    def test_search_is_bounded_and_reports_detector_failure(self):
        frame = Image.new("RGB", (20, 20))
        controls = FakeControls()
        no_match = search_target("unlisted object", 12, capture=FakeCapture([frame]),
                                 detector=FakeDetector([{"boxes": []}]), controls=controls,
                                 camera_guard=self.guard, profile=self.profile)
        self.assertEqual(no_match.status, SkillStatus.FAILED)
        self.assertEqual(no_match.reason, "target_not_found_within_sweep")
        self.assertLessEqual(sum(abs(call[1]) for call in controls.calls if call[0] == "look"),
                             self.profile.motion_limits.max_sweep_counts)
        failed = search_target("unlisted object", 4, capture=FakeCapture([frame]),
                               detector=FakeDetector([None]), controls=FakeControls(),
                               camera_guard=self.guard, profile=self.profile)
        self.assertEqual(failed.reason, "detector_unavailable")

    def test_interact_until_change_reports_region_evidence(self):
        before = Image.new("RGB", (20, 20), (0, 0, 0))
        after = before.copy()
        ImageDraw.Draw(after).rectangle((5, 17, 15, 19), fill=(255, 255, 255))
        controls = FakeControls()
        rows = []
        result = interact_until_change("hotbar", "primary_hold", 2,
                                      capture=FakeCapture([before, after]), controls=controls,
                                      profile=self.profile, camera_guard=self.guard,
                                      recorder=lambda row, old, new: rows.append(row))
        self.assertEqual(result.status, SkillStatus.UNCONFIRMED)
        self.assertEqual(result.result_code, "REGION_CHANGE_UNCONFIRMED")
        self.assertEqual(result.reason, "region_changed_unverified")
        self.assertEqual(controls.calls, [("mouse", "left", 2.5)])
        self.assertGreaterEqual(result.evidence["region_delta"], result.evidence["threshold"])
        self.assertEqual(rows[0]["skill"], "interact_until_change")

    def test_interact_until_change_can_repeat_without_moving_off_target(self):
        frame = Image.new("RGB", (20, 20), (0, 0, 0))
        profile = self.profile.model_copy(deep=True)
        profile.motion_limits.interaction_retry_step_seconds = 0
        controls = FakeControls()
        result = interact_until_change("hotbar", "primary_hold", 2,
                                      capture=FakeCapture([frame, frame, frame]), controls=controls,
                                      profile=profile, camera_guard=self.guard)
        self.assertEqual(result.reason, "region_unchanged_after_repeats")
        self.assertEqual(controls.calls, [("mouse", "left", 2.5), ("mouse", "left", 2.5)])

    def test_wait_for_condition_returns_on_detector_match(self):
        frame = Image.new("RGB", (20, 20))
        detector = FakeDetector([{"boxes": [{"x1": .1, "y1": .1, "x2": .4, "y2": .5,
                                               "score": .9, "query": "a tree trunk"}]}])
        result = wait_for_condition(capture=FakeCapture([frame]), profile=self.profile,
                                    timeout=1, query="a tree trunk", detector=detector,
                                    poll_seconds=0)
        self.assertEqual(result.status, SkillStatus.SUCCESS)

    def test_registry_executor_creates_and_finishes_common_telemetry(self):
        with tempfile.TemporaryDirectory() as temp:
            class Telemetry:
                directory = Path(temp)

            controls = FakeControls()
            result = execute_move_pulse(
                {"direction": "forward", "seconds": 0.5}, controls=controls,
                camera_guard=self.guard, profile=self.profile, telemetry=Telemetry())
            self.assertEqual(result.status, SkillStatus.SUCCESS)
            rows = [line for line in (Path(temp) / "skill_steps.jsonl").read_text(
                encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertIn('"event":"finish"', rows[-1])

    def test_registry_executor_requires_telemetry_before_sending_input(self):
        controls = FakeControls()
        with self.assertRaisesRegex(ValueError, "requires a recorder"):
            execute_move_pulse({"direction": "forward", "seconds": 0.5},
                               controls=controls, camera_guard=self.guard, profile=self.profile)
        self.assertEqual(controls.calls, [])


if __name__ == "__main__":
    unittest.main()
