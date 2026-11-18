import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import run_agent


class FakeControls:
    def __init__(self):
        self.releases = 0
    def release_all(self):
        self.releases += 1
        raise AssertionError("dry-run must not send even a release event")


class FakeHotkey:
    def close(self):
        pass


class FakeDetector:
    def detect(self, image, queries, **kwargs):
        return {"boxes": [{"score": 0.9}], "query": queries[0]}


class FakeAgent:
    instance = None
    run_dir = None
    def __init__(self, **kwargs):
        self.db_path = kwargs["db_path"]
        self.capture = None
        self.controls = FakeControls()
        self.camera_guard = object()
        self.brain = self
        self.memory = type("Memory", (), {"db_path": Path(tempfile.gettempdir()) / "unused-memory.db"})()
        self.telemetry = type("Telemetry", (), {})()
        self.telemetry.directory = FakeAgent.run_dir
        self.telemetry.meta = {"run_id": "run-cli"}
        self.emergency_hotkey = FakeHotkey()
        self.ended = None
        self.goal_plan_requests = 0
        self.profile = kwargs["profile"]
        self.game_id = "game:" + self.profile.game_id
        FakeAgent.instance = self
    def plan_goal(self, **kwargs):
        return {"steps": [{"skill": "search_target", "args": {"target_text": "target", "sweep": 2},
                           "expected_result": "target detected", "success_check": "detector result"}]}
    def request_goal_plan(self, **kwargs):
        self.goal_plan_requests += 1
        return self.plan_goal(**kwargs)
    def begin_telemetry_run(self):
        return None
    def start_emergency_hotkey(self):
        pass
    def _check_stop(self):
        pass
    def end_telemetry_run(self, reason, error_message=None):
        self.ended = reason


class CliGoalDryRunTests(unittest.TestCase):
    def test_frame_directory_goal_pipeline_logs_without_any_controller_input(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frames, runs = root / "frames", root / "runs"
            frames.mkdir()
            runs.mkdir()
            Image.new("RGB", (24, 24), "black").save(frames / "01.png")
            profile = root / "profile.json"
            profile.write_text(json.dumps({"game_id": "cli-test", "camera_type": "first_person"}),
                               encoding="utf-8")
            FakeAgent.run_dir = runs / "run-cli"
            FakeAgent.run_dir.mkdir()
            with patch.object(run_agent, "OmniAgent", FakeAgent), \
                 patch("omnigamer.detector_client.DetectorClient", return_value=FakeDetector()):
                with patch.object(sys, "argv", ["run_agent.py", "--profile", str(profile),
                    "--frame-dir", str(frames), "--goal", "test movement", "--dry-run"]):
                    with contextlib.redirect_stdout(io.StringIO()) as stdout:
                        run_agent.main()
            instance = FakeAgent.instance
            self.assertEqual(instance.db_path, ":memory:")
            self.assertEqual(instance.controls.releases, 0)
            self.assertEqual(instance.ended, "dry_run")
            self.assertEqual(instance.goal_plan_requests, 1)
            self.assertIn("would send", stdout.getvalue())
            events = [json.loads(line) for line in (runs / "run-cli" / "goal_steps.jsonl").read_text(
                encoding="utf-8").splitlines()]
            self.assertEqual([row["event"] for row in events],
                             ["plan", "would_send", "skill_result", "run_finish", "input_action_budget"])
            self.assertEqual(events[0]["request_count"], 1)
            self.assertEqual(len(events[0]["request_durations_seconds"]), 1)
            self.assertGreaterEqual(events[0]["duration_seconds"],
                                    events[0]["request_durations_seconds"][0])
            self.assertTrue(events[1]["detector"]["available"])
            self.assertEqual(events[1]["detector"]["box_count"], 1)
            self.assertEqual(events[-1]["actions_used"], 0)
            self.assertEqual(events[-1]["max_actions"], 60)


if __name__ == "__main__":
    unittest.main()
