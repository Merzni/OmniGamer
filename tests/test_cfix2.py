"""Console, presentation and run-finalization regressions without game input."""
import io
import importlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from omnigamer.agent import OmniAgent
from omnigamer.personality import StreamerPersona
from omnigamer.telemetry import RunTelemetry


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class QuietSampler:
    def __init__(self, *_args):
        self.closed = False

    def start(self):
        pass

    def close(self):
        self.closed = True


class ConsoleAndFinalizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data_dir = Path(self.temp.name) / "data"

    def make_agent(self):
        agent = OmniAgent.__new__(OmniAgent)
        agent.capture = Mock()
        agent.capture.hwnd = 77
        agent.capture.get_window_title.return_value = "Fake Minecraft"
        agent.game_id = "game:test"
        agent.goal = ""
        agent.input_backend = "native"
        agent.telemetry = RunTelemetry(self.data_dir, resource_sampler_factory=QuietSampler)
        agent._telemetry_no_effect_streak = 0
        agent.emergency_hotkey = Mock()
        agent.controls = Mock()
        agent.controls.release_all.return_value = []
        agent.persona = StreamerPersona()
        agent.memory = Mock()
        agent.memory.get_stats.return_value = {
            "total_actions": 0, "effective_actions": 0, "unique_screens_discovered": 0,
        }
        agent._stop_event = threading.Event()
        agent.is_running = False
        agent.start_emergency_hotkey = Mock()
        return agent

    def read_run_meta(self):
        run_dirs = list((self.data_dir / "runs").iterdir())
        self.assertEqual(len(run_dirs), 1)
        return json.loads((run_dirs[0] / "run_meta.json").read_text(encoding="utf-8"))

    def test_cp1251_strict_console_is_reconfigured_before_emoji_speech(self):
        from omnigamer import personality

        stdout_buffer, stderr_buffer = io.BytesIO(), io.BytesIO()
        stdout = io.TextIOWrapper(stdout_buffer, encoding="cp1251", errors="strict")
        stderr = io.TextIOWrapper(stderr_buffer, encoding="cp1251", errors="strict")
        try:
            with patch.object(sys, "stdout", stdout), patch.object(sys, "stderr", stderr):
                personality.configure_console()
                StreamerPersona().speak("Привет, 🌸")
                stdout.flush()
                rendered = stdout_buffer.getvalue().decode("utf-8")
                self.assertIn("🌸", rendered)
                self.assertEqual(sys.stdout.encoding.lower(), "utf-8")
                self.assertEqual(sys.stdout.errors, "replace")
                self.assertEqual(sys.stderr.encoding.lower(), "utf-8")
                self.assertEqual(sys.stderr.errors, "replace")
        finally:
            personality.configure_console()

    def test_console_exception_warns_once_and_does_not_abort_steps(self):
        from omnigamer import personality

        previous = personality._console_warning_emitted
        personality._console_warning_emitted = False
        self.addCleanup(setattr, personality, "_console_warning_emitted", previous)
        agent = self.make_agent()
        agent.step = Mock()
        with patch.object(personality.console, "print", side_effect=UnicodeEncodeError(
                "cp1251", "🌸", 0, 1, "unmappable")), \
                self.assertLogs("omnigamer.personality", level="WARNING") as captured:
            agent.run_loop(max_steps=2)
        self.assertEqual(agent.step.call_count, 2)
        self.assertEqual(len(captured.output), 1)
        self.assertIn("UnicodeEncodeError", captured.output[0])
        self.assertEqual(self.read_run_meta()["termination_reason"], "step_limit")

    def test_startup_backend_exception_finalizes_metadata_and_propagates(self):
        agent = self.make_agent()
        agent.start_emergency_hotkey.side_effect = RuntimeError("backend startup failed")
        with self.assertRaisesRegex(RuntimeError, "backend startup failed"):
            agent.run_loop(max_steps=2)
        meta = self.read_run_meta()
        self.assertIsNotNone(meta["ended_at"])
        self.assertEqual(meta["termination_reason"], "error")
        self.assertEqual(meta["error"], "RuntimeError: backend startup failed")
        agent.controls.release_all.assert_called_once_with()
        agent.emergency_hotkey.close.assert_called_once_with()
        self.assertTrue(agent.telemetry.resource_sampler.closed)

    def test_step_three_exception_finalizes_metadata_and_propagates(self):
        agent = self.make_agent()
        agent.step = Mock(side_effect=[None, None, RuntimeError("step 3 failed")])
        with self.assertRaisesRegex(RuntimeError, "step 3 failed"):
            agent.run_loop(max_steps=5)
        meta = self.read_run_meta()
        self.assertIsNotNone(meta["ended_at"])
        self.assertEqual(meta["termination_reason"], "error")
        self.assertEqual(meta["error"], "RuntimeError: step 3 failed")
        self.assertEqual(agent.step.call_count, 3)
        agent.controls.release_all.assert_called_once_with()
        self.assertTrue(agent.telemetry.resource_sampler.closed)

    def test_dashboard_attach_exception_finalizes_run(self):
        dashboard = importlib.import_module("omnigamer.dashboard.app")
        agent = self.make_agent()
        agent.attach = Mock(side_effect=RuntimeError("panel attach failed"))
        with patch.object(dashboard, "agent", agent), self.assertRaisesRegex(RuntimeError, "panel attach failed"):
            dashboard.attach_window(dashboard.AttachRequest(hwnd=88, game_id="test"))
        meta = self.read_run_meta()
        self.assertEqual(meta["termination_reason"], "error")
        self.assertEqual(meta["error"], "RuntimeError: panel attach failed")

    def test_dashboard_step_exception_finalizes_run(self):
        from fastapi import HTTPException
        dashboard = importlib.import_module("omnigamer.dashboard.app")
        agent = self.make_agent()
        agent.step = Mock(side_effect=RuntimeError("panel step failed"))
        with patch.object(dashboard, "agent", agent), self.assertRaises(HTTPException) as caught:
            dashboard.trigger_step(single_step=True)
        self.assertEqual(caught.exception.status_code, 409)
        meta = self.read_run_meta()
        self.assertEqual(meta["termination_reason"], "error")
        self.assertEqual(meta["error"], "RuntimeError: panel step failed")

    def test_cli_subprocess_reconfigures_strict_cp1251_and_finalizes_step_limit(self):
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            data_dir = temp_path / "agent-data"
            probe_path = temp_path / "initial_encoding.txt"
            sitecustomize = temp_path / "sitecustomize.py"
            sitecustomize.write_text(r'''import os, sys
for stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure:
        reconfigure(encoding="cp1251", errors="strict")
open(os.environ["OMNIGAMER_CFIX2_PROBE"], "w").write(sys.stdout.encoding + ":" + sys.stdout.errors)
from omnigamer import agent as agent_module
from omnigamer import capture as capture_module
from omnigamer.personality import StreamerPersona
from omnigamer.telemetry import RunTelemetry
class QuietSampler:
    def __init__(self, *_args): pass
    def start(self): pass
    def close(self): pass
capture_module.find_window_by_title = lambda _query: {"hwnd": 77, "title": "Fake Minecraft"}
class FakeAgent:
    def __init__(self, hwnd=0, game_id="", **_kwargs):
        self.hwnd = hwnd
        self.game_id = "game:" + game_id
        self.persona = StreamerPersona()
        self.telemetry = RunTelemetry(os.environ["OMNIGAMER_CFIX2_DATA"], resource_sampler_factory=QuietSampler)
    def begin_telemetry_run(self):
        return bool(self.telemetry.start({"window_title": "Fake Minecraft", "hwnd": self.hwnd, "game_id": self.game_id}))
    def attach(self, hwnd, game_id="", *, focus=True):
        self.hwnd = hwnd
        if game_id:
            self.game_id = "game:" + game_id
        self.telemetry.update_metadata(hwnd=hwnd, window_title="Fake Minecraft", game_id=self.game_id)
    def end_telemetry_run(self, reason, error_message=None):
        self.telemetry.finish(reason, error_message=error_message)
    def run_loop(self, max_steps=2):
        self.begin_telemetry_run()
        self.persona.speak("🌸 CLI smoke greeting")
        for index in range(1, max_steps + 1):
            trace = self.telemetry.begin_step(index)
            trace.data["decision"]["action_type"] = "wait"
            self.telemetry.finish_step(trace)
        self.telemetry.finish("step_limit")
agent_module.OmniAgent = FakeAgent
''', encoding="utf-8")
            env = os.environ.copy()
            env.pop("PYTHONIOENCODING", None)
            env["PYTHONPATH"] = str(temp_path) + os.pathsep + str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
            env["OMNIGAMER_CFIX2_PROBE"] = str(probe_path)
            env["OMNIGAMER_CFIX2_DATA"] = str(data_dir)
            result = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "run_agent.py"), "--window", "Fake",
                 "--game-id", "cfix2-test", "--steps", "2", "--legacy"],
                cwd=PROJECT_ROOT, env=env, capture_output=True, timeout=20,
            )
            output = result.stdout.decode("utf-8")
            stderr = result.stderr.decode("utf-8", errors="replace")
            self.assertEqual(probe_path.read_text(encoding="utf-8"), "cp1251:strict")
            self.assertEqual(result.returncode, 0, stderr)
            self.assertIn("🌸", output)
            meta_paths = list((data_dir / "runs").glob("*/run_meta.json"))
            self.assertEqual(len(meta_paths), 1)
            meta = json.loads(meta_paths[0].read_text(encoding="utf-8"))
            self.assertIsNotNone(meta["ended_at"])
            self.assertEqual(meta["termination_reason"], "step_limit")
            if os.environ.get("OMNIGAMER_CFIX2_PRINT_EVIDENCE") == "1":
                print("CLI_SMOKE_STDOUT_FIRST_20_LINES:")
                print("\n".join(output.splitlines()[:20]))
                print("CLI_SMOKE_RUN_META:")
                print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    unittest.main()
