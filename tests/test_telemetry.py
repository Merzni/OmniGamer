"""Telemetry and report tests use temporary run folders and synthetic data only."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from omnigamer.telemetry import RunTelemetry, StepTrace, rotate_run_data
from omnigamer.skills.contracts import SkillResult, SkillStatus
from omnigamer.skills.telemetry import SkillStepRecorder
from tools.summarize_run import summarize


class QuietSampler:
    def __init__(self, *_args):
        self.closed = False

    def start(self):
        pass

    def close(self):
        self.closed = True


class TelemetryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data_dir = Path(self.temp.name)

    def recorder(self, **kwargs):
        return RunTelemetry(self.data_dir, resource_sampler_factory=QuietSampler, **kwargs)

    def test_run_rotation_keeps_newest_completed_runs_and_ignores_unmanaged_dirs(self):
        runs = self.data_dir / "runs"
        runs.mkdir()
        for index in range(3):
            folder = runs / f"run-{index}"
            folder.mkdir()
            (folder / "run_meta.json").write_text("{}", encoding="utf-8")
            (folder / "steps.jsonl").write_text("x" * 10, encoding="utf-8")
        unmanaged = runs / "keep-me"
        unmanaged.mkdir()
        rotate_run_data(runs, max_count=2, max_bytes=1024)
        self.assertEqual(sorted(path.name for path in runs.iterdir()), ["keep-me", "run-1", "run-2"])

    def test_generic_skill_recorder_and_summary_use_common_result_format(self):
        class Run:
            directory = self.data_dir / "skill-run"

        Run.directory.mkdir()
        (Run.directory / "run_meta.json").write_text(json.dumps({"run_id": "skill-run"}), encoding="utf-8")
        recorder = SkillStepRecorder(Run, "move_pulse", {"direction": "forward", "seconds": 0.5})
        recorder({"skill": "move_pulse", "state": "SUCCESS", "evidence": {"key": "w"}}, None, None)
        recorder.finish(SkillResult(status=SkillStatus.SUCCESS, result_code="SUCCESS",
                                    reason="movement_sent", evidence={"key": "w"}))
        rows = [json.loads(line) for line in (Run.directory / "skill_steps.jsonl").read_text(
            encoding="utf-8").splitlines()]
        self.assertEqual(rows[0]["schema_version"], 1)
        self.assertEqual(rows[1]["event"], "finish")
        self.assertEqual(rows[1]["result"]["status"], "SUCCESS")
        self.assertIn("move_pulse: SUCCESS: 1", summarize(Run.directory))

    def test_camera_actions_pause_no_effect_streak_and_count_separately(self):
        no_effect, camera_only = 4, 0
        no_effect, camera_only = StepTrace.update_loop_streaks(
            no_effect, camera_only, "look", False, "camera_motion")
        self.assertEqual((no_effect, camera_only), (4, 1))
        no_effect, camera_only = StepTrace.update_loop_streaks(
            no_effect, camera_only, "wait", True, "camera_motion")
        self.assertEqual((no_effect, camera_only), (4, 2))
        no_effect, camera_only = StepTrace.update_loop_streaks(no_effect, camera_only, "walk", False)
        self.assertEqual((no_effect, camera_only), (5, 0))
        no_effect, camera_only = StepTrace.update_loop_streaks(no_effect, camera_only, "walk", True)
        self.assertEqual((no_effect, camera_only), (0, 0))
        no_effect, camera_only = StepTrace.update_loop_streaks(
            4, 0, "wait", False, "refusal")
        self.assertEqual((no_effect, camera_only), (5, 1))

    def test_step_record_keeps_unavailable_fields_null(self):
        telemetry = self.recorder()
        run_dir = Path(telemetry.start({"game_id": "game:fake"}))
        trace = telemetry.begin_step(1)
        trace.data["decision"]["action_type"] = "wait"
        trace.data["decision"]["fallback_reason"] = "Timeout: offline"
        trace.data["decision"]["raw_json"] = '{"action_type":"wait"}'
        trace.data["decision"]["final"] = {"action_type": "wait"}
        trace.add_prompt_context("memory context", "Find a safe path")
        trace.data["outcome"]["status"] = "no_effect"
        row = telemetry.finish_step(trace)
        telemetry.finish("single_step")

        saved = json.loads((run_dir / "steps.jsonl").read_text(encoding="utf-8"))
        prompt = json.loads((run_dir / "prompts.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(row["timings_ms"]["capture"], None)
        self.assertIsNone(saved["decision"]["parameters"])
        self.assertIsNone(saved["memory"]["similar_records"])
        self.assertEqual(saved["decision"]["action_type"], "wait")
        self.assertEqual(saved["decision"]["fallback_reason"], "Timeout: offline")
        self.assertEqual(saved["decision"]["raw_json"], '{"action_type":"wait"}')
        self.assertEqual(saved["decision"]["final"]["action_type"], "wait")
        self.assertEqual(prompt["goal"], "Find a safe path")
        self.assertEqual(prompt["context_hint"], "memory context")

    def test_telemetry_write_error_is_swallowed(self):
        telemetry = self.recorder()
        telemetry.start({"game_id": "game:fake"})
        trace = telemetry.begin_step(1)
        real_steps_file = telemetry.steps_file

        class BrokenStream:
            def write(self, _value):
                raise OSError("synthetic disk error")

            def flush(self):
                pass

            def close(self):
                pass

        telemetry.steps_file = BrokenStream()
        with patch.object(telemetry, "_warn_once") as warn:
            telemetry.finish_step(trace)
            warn.assert_called_once()
        telemetry.steps_file = None
        real_steps_file.close()
        telemetry.finish("error")
        self.assertTrue(telemetry.closed)

    def test_frame_size_limit_stops_future_frame_saves(self):
        telemetry = self.recorder(frames_max_mb=0.0001)
        run_dir = Path(telemetry.start({}))
        trace = telemetry.begin_step(1)
        trace.data["outcome"]["status"] = "success"
        with patch.object(telemetry, "_warn_once") as warn:
            telemetry.finish_step(trace, Image.new("RGB", (640, 640), "navy"), None)
            self.assertTrue(telemetry.frames_disabled)
            warn.assert_called_once()
        self.assertEqual(list((run_dir / "frames").glob("*.jpg")), [])
        telemetry.finish("single_step")

    def test_finish_flushes_step_and_writes_termination_reason(self):
        telemetry = self.recorder()
        run_dir = Path(telemetry.start({"game_id": "game:fake"}))
        trace = telemetry.begin_step(7)
        trace.data["decision"]["action_type"] = "walk"
        telemetry.finish("panel_stop")
        self.assertFalse(telemetry.closed)  # The active step is flushed before finalization.
        telemetry.finish_step(trace)
        self.assertTrue(telemetry.closed)
        meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["termination_reason"], "panel_stop")
        self.assertIsNotNone(meta["ended_at"])
        self.assertEqual(meta["steps_recorded"], 1)

    def test_error_finalization_overrides_prior_stop_and_records_short_message(self):
        telemetry = self.recorder()
        run_dir = Path(telemetry.start({"game_id": "game:fake"}))
        trace = telemetry.begin_step(1)
        telemetry.finish("panel_stop")
        telemetry.finish("error", error_message="RuntimeError: startup failed")
        telemetry.finish_step(trace)
        meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
        self.assertIsNotNone(meta["ended_at"])
        self.assertEqual(meta["termination_reason"], "error")
        self.assertEqual(meta["error"], "RuntimeError: startup failed")

    def test_summary_p95_percentages_and_resource_stats_on_synthetic_run(self):
        run_dir = self.data_dir / "synthetic-run"
        run_dir.mkdir()
        meta = {"run_id": "synthetic-run", "started_at": "2026-01-01T00:00:00+00:00",
                "ended_at": "2026-01-01T00:01:00+00:00", "termination_reason": "step_limit"}
        (run_dir / "run_meta.json").write_text(json.dumps(meta), encoding="utf-8")
        steps = []
        for index, (action, status, capture_ms, total_ms) in enumerate((
            ("walk", "success", 10, 100), ("walk", "no_effect", 20, 200),
            ("jump", "uncertain", 30, 300), ("jump", "refusal", 40, 400),
        ), 1):
            steps.append({
                "step": index,
                "decision": {"action_type": action, "refusal_reason": "guard" if status == "refusal" else None,
                             "fallback_reason": "Timeout: offline" if index == 2 else None,
                             "action_source": "vlm"},
                "outcome": {"status": status},
                "loop": {"no_effect_streak": index, "camera_only_streak": index - 1},
                "timings_ms": {"capture": capture_ms, "memory_lookup": None, "vlm_decide": None,
                               "grounding": None, "input_send": None, "post_wait": None,
                               "result_eval": None, "memory_write": None, "total": total_ms},
                "error": None,
            })
        (run_dir / "steps.jsonl").write_text("".join(json.dumps(row) + "\n" for row in steps), encoding="utf-8")
        resources = [{"process_ram_mb": 100, "gpu_vram_used_mb": 500},
                     {"process_ram_mb": 200, "gpu_vram_used_mb": 1000}]
        (run_dir / "resources.jsonl").write_text("".join(json.dumps(row) + "\n" for row in resources), encoding="utf-8")

        script = Path(__file__).resolve().parents[1] / "tools" / "summarize_run.py"
        result = subprocess.run([sys.executable, str(script), str(run_dir)], capture_output=True,
                                text=True, encoding="utf-8", check=True)
        self.assertIn("Захват кадра: 25.00 мс / 25.00 мс / 40.00 мс / 10.0%", result.stdout)
        self.assertIn("walk: 2; success 50.0%, no_effect 50.0%", result.stdout)
        self.assertIn("Максимальная серия шагов без видимого эффекта: 4", result.stdout)
        self.assertIn("Максимальная серия look/wait: 3", result.stdout)
        self.assertIn("RAM процесса, пик/среднее: 200.00 МБ / 150.00 МБ", result.stdout)
        self.assertIn("VRAM, пик/среднее: 1000.00 МБ / 750.00 МБ", result.stdout)
        self.assertIn("Шагов с fallback: 1 (25.0%)", result.stdout)


if __name__ == "__main__":
    unittest.main()
