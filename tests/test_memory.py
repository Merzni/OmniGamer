"""Memory regression tests: temporary databases only, no desktop input or live VLM."""
import json
from contextlib import closing
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests
from PIL import Image

from omnigamer.memory import GameMemory
from omnigamer.config import config
from omnigamer.vision_brain import BrainDecision, VisionBrain
from omnigamer.profiles import GameProfile, InputAction, load_profile


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "memory.db"
        self.memory = GameMemory(self.path)

    def record(self, **overrides):
        values = dict(timestamp=1.0, state_hash="0123456789abcdef", action_type="click",
                      x=500, y=700, key=None, screen_delta=0.0, effective=False,
                      commentary="test", intent="open menu", game_id="game:test",
                      next_state_hash="0123456789abcdef", duration=1.5,
                      observation="menu", session_id="first")
        values.update(overrides)
        self.memory.record_experience(**values)

    def test_in_memory_database_survives_separate_connection_scopes(self):
        memory = GameMemory(":memory:")
        self.assertEqual(memory.get_recent_experiences("game:dry-run"), [])
        memory.record_experience(timestamp=1.0, state_hash="abc", action_type="wait",
            x=None, y=None, key=None, screen_delta=0, effective=False, commentary="", intent="",
            game_id="game:dry-run")
        self.assertEqual(memory.get_stats("game:dry-run")["total_actions"], 1)

    def test_migration_preserves_unattributed_old_rows_and_is_repeatable(self):
        old_path = Path(self.temp.name) / "old.db"
        with closing(sqlite3.connect(old_path)) as conn, conn:
            conn.execute("CREATE TABLE experiences (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                         "timestamp REAL, state_hash TEXT, action_type TEXT, x INTEGER, y INTEGER, "
                         "key TEXT, screen_delta REAL, effective INTEGER, commentary TEXT, intent TEXT)")
            conn.execute("INSERT INTO experiences(state_hash, commentary) VALUES ('old', 'preserve')")
            conn.execute("CREATE TABLE dead_clicks (state_hash TEXT, x INTEGER, y INTEGER, "
                         "fail_count INTEGER, PRIMARY KEY(state_hash, x, y))")
            conn.execute("INSERT INTO dead_clicks VALUES ('old', 50, 100, 3)")
        GameMemory(old_path)
        migrated = GameMemory(old_path)
        self.assertEqual(migrated.get_stats()["total_actions"], 1)
        self.assertEqual(migrated.get_stats("game:test")["total_actions"], 0)
        with closing(sqlite3.connect(old_path)) as conn:
            self.assertEqual(conn.execute("SELECT commentary, game_id FROM experiences").fetchone(),
                             ("preserve", "legacy"))
            self.assertEqual(conn.execute("SELECT fail_count FROM dead_clicks").fetchone()[0], 3)

    def test_restart_retrieves_changed_and_unchanged_actions(self):
        self.record()
        self.record()
        self.record(x=200, effective=True, screen_delta=0.3, next_state_hash="fedcba9876543210")
        reopened = GameMemory(self.path)
        context = reopened.get_learning_context("0123456789abcdef", "game:test")
        self.assertIn("AVOID REPEATING", context)
        self.assertIn("changed=1/1", context)
        self.assertIn("fedcba9876543210", context)
        self.assertEqual(len(reopened.get_recent_experiences("game:test")), 3)
        self.assertEqual(reopened.get_stats("game:test")["observed_transitions"], 1)

    def test_profiles_isolate_retrieval_history_and_statistics(self):
        self.record(intent="UNIQUE_SECRET_OTHER_GAME", game_id="game:other")
        self.record(game_id="legacy")
        context = self.memory.get_learning_context("0123456789abcdef", "game:test")
        self.assertNotIn("UNIQUE_SECRET_OTHER_GAME", context)
        self.assertNotIn("Recent history", context)
        self.assertEqual(self.memory.get_stats("game:test")["total_actions"], 0)

    def test_fuzzy_hash_retrieval_but_guard_needs_exact_hash(self):
        self.record()
        self.record()
        near = "0123456789abcdee"  # One differing bit.
        self.assertIn("AVOID REPEATING", self.memory.get_learning_context(near, "game:test"))
        action = dict(action_type="click", x=500, y=700)
        self.assertEqual(self.memory.get_repetition_warning(near, "game:test", action), "")
        self.assertNotIn("AVOID REPEATING", self.memory.get_learning_context("ffffffffffffffff", "game:test"))

    def test_thumbnail_filters_hash_collisions_and_aspect_ratio(self):
        dark = GameMemory.compute_visual_signature(Image.new("RGB", (100, 100), "black"))
        light = GameMemory.compute_visual_signature(Image.new("RGB", (100, 100), "white"))
        wide = GameMemory.compute_visual_signature(Image.new("RGB", (200, 100), "black"))
        self.record(visual_signature=dark)
        self.assertTrue(self.memory._matching_experiences("0123456789abcdef", "game:test", dark))
        self.assertFalse(self.memory._matching_experiences("0123456789abcdef", "game:test", light))
        self.assertFalse(self.memory._matching_experiences("0123456789abcdef", "game:test", wide))

    def test_new_changed_attempt_clears_old_failure_warning(self):
        action = dict(action_type="click", x=500, y=700)
        self.record()
        self.record()
        self.assertTrue(self.memory.get_repetition_warning("0123456789abcdef", "game:test", action))
        self.record(effective=True, screen_delta=0.1)
        self.assertEqual(self.memory.get_repetition_warning("0123456789abcdef", "game:test", action), "")

    def test_camera_and_duration_are_distinct_actions(self):
        self.record(action_type="look", x=-300, y=0)
        self.record(action_type="look", x=-300, y=0)
        self.assertFalse(self.memory.get_repetition_warning(
            "0123456789abcdef", "game:test", dict(action_type="look", x=300, y=0)))
        self.record(action_type="walk", key="w", duration=1.0)
        self.record(action_type="walk", key="w", duration=1.0)
        self.assertFalse(self.memory.get_repetition_warning(
            "0123456789abcdef", "game:test", dict(action_type="walk", key="w", duration=3.0)))

    def test_look_and_wait_records_do_not_motivate_from_memory(self):
        self.record(action_type="look", intent="LOOK_ONLY_SECRET", commentary="look secret")
        self.record(action_type="wait", intent="WAIT_ONLY_SECRET", commentary="wait secret")
        context = self.memory.get_learning_context("0123456789abcdef", "game:test")
        self.assertNotIn("LOOK_ONLY_SECRET", context)
        self.assertNotIn("WAIT_ONLY_SECRET", context)

    def test_unlearnable_fallback_does_not_create_action_evidence(self):
        self.record(learnable=False)
        self.record(learnable=False)
        self.assertFalse(self.memory._matching_experiences("0123456789abcdef", "game:test"))

    def test_context_stays_bounded_with_large_history(self):
        for index in range(100):
            self.record(x=index * 10, observation="x" * 5000, intent="y" * 5000)
        context = self.memory.get_learning_context("0123456789abcdef", "game:test")
        self.assertLess(len(context), 4000)
        self.assertEqual(context.count("(screen-change counts are not goal success)"), 6)
        self.assertNotIn("y" * 100, context)
        self.assertNotIn("x" * 100, context)

    def test_drag_endpoint_and_duration_survive_restart(self):
        self.record(action_type="drag", x2=300, y2=800, duration=2.0)
        row = GameMemory(self.path).get_recent_experiences("game:test")[0]
        self.assertEqual((row["x2"], row["y2"], row["duration"]), (300, 800, 2.0))

    def test_animated_background_does_not_erase_consecutive_failed_clicks(self):
        signature = self.memory.compute_visual_signature(Image.new("RGB", (100, 100), (60, 60, 60)))
        nearby = self.memory.compute_visual_signature(Image.new("RGB", (100, 100), (65, 65, 65)))
        self.record(state_hash="0000000000000000", visual_signature=signature)
        self.record(state_hash="ffffffffffffffff", visual_signature=nearby)
        self.assertIn("REPLAN", self.memory.get_recent_failure_hint(
            "game:test", signature, dict(action_type="click", x=500, y=700)))
        self.assertFalse(self.memory.get_recent_failure_hint(
            "game:test", signature, dict(action_type="click", x=300, y=250)))
        self.record(action_type="wait", learnable=False)
        self.assertTrue(self.memory.get_recent_failure_hint("game:test", signature))

    def test_recent_failure_hint_does_not_cross_different_scenes_or_success(self):
        signature = self.memory.compute_visual_signature(Image.new("RGB", (100, 100), "black"))
        other = self.memory.compute_visual_signature(Image.new("RGB", (100, 100), "white"))
        self.record(visual_signature=signature)
        self.record(visual_signature=signature)
        self.assertFalse(self.memory.get_recent_failure_hint("game:test", other))
        self.record(visual_signature=signature, effective=True)
        self.assertFalse(self.memory.get_recent_failure_hint("game:test", signature))

    def test_blocked_navigation_uses_recent_results_despite_hash_drift(self):
        signature = self.memory.compute_visual_signature(Image.new("RGB", (100, 100), (20, 30, 20)))
        self.record(action_type="walk", key="w", state_hash="0000000000000000", screen_delta=0.001, visual_signature=signature)
        self.record(action_type="walk", key="w", state_hash="ffffffffffffffff", screen_delta=0.012, visual_signature=signature)
        self.assertIn("NAVIGATION REPLAN", self.memory.get_navigation_hint("game:test", signature))
        self.assertFalse(self.memory.get_navigation_hint("game:test", signature, action=dict(action_type="look",x=120,y=0)))
        self.record(action_type="walk", key="w", effective=True, screen_delta=0.1, visual_signature=signature)
        self.assertFalse(self.memory.get_navigation_hint("game:test", signature))

    def test_jump_directions_are_distinct_in_memory(self):
        self.assertNotEqual(self.memory._action_key(dict(action_type="jump",key="w",duration=0.6)),
                            self.memory._action_key(dict(action_type="jump",key="a",duration=0.6)))


class AgentMemoryTests(unittest.TestCase):
    def setUp(self):
        from omnigamer import agent as agent_module
        self.module = agent_module
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "memory.db"
        self.frame = Image.new("RGB", (100, 100), "black")
        for name in ("WindowCapture", "HumanInputController", "VisionBrain", "StreamerPersona"):
            patcher = patch.object(agent_module, name)
            setattr(self, name, patcher.start())
            self.addCleanup(patcher.stop)
        self.WindowCapture.return_value.get_window_title.return_value = "Minecraft 1.21"
        self.WindowCapture.return_value.assert_foreground = Mock()
        self.WindowCapture.return_value.grab_frame.return_value = self.frame
        self.WindowCapture.return_value.norm_to_screen.return_value = (50, 70)
        self.WindowCapture.return_value.last_rect = (0, 0, 100, 100)
        self.WindowCapture.return_value.get_client_rect.return_value = (0, 0, 100, 100)
        self.VisionBrain.return_value.last_error = None
        from omnigamer.perception import VisualAssessment
        self.VisionBrain.return_value.assess.return_value = VisualAssessment()
        self.VisionBrain.return_value.last_assessment_error = None
        self.addCleanup(setattr, agent_module.config, "save_screenshots", agent_module.config.save_screenshots)
        self.addCleanup(setattr, agent_module.config, "action_delay_seconds", agent_module.config.action_delay_seconds)
        agent_module.config.save_screenshots = False
        agent_module.config.action_delay_seconds = 0
        self.addCleanup(setattr,agent_module.config,"desktop_bot_pointer",agent_module.config.desktop_bot_pointer)
        agent_module.config.desktop_bot_pointer=False

    def create(self, game_id="survival", mine_requires_target_at_crosshair=False,
               mine_requires_grounded_level_view=True, camera_type="first_person"):
        profile = GameProfile(game_id=game_id, camera_type=camera_type,
            actions={"primary_hold": InputAction(kind="mouse", value="left", duration=3),
                     "interact": InputAction(kind="mouse", value="right", duration=0.15),
                     "jump": InputAction(kind="key", value="space", duration=0.75)},
            mine_requires_grounded_level_view=mine_requires_grounded_level_view,
            mine_requires_target_at_crosshair=mine_requires_target_at_crosshair)
        return self.module.OmniAgent(db_path=self.path, hwnd=123, game_id=game_id,input_backend="native",
                                     emergency_hotkey_enabled=False, telemetry_enabled=False,
                                     profile=profile)

    def test_experience_reaches_brain_after_restart_and_replans(self):
        click = BrainDecision(action_type="click", x=500, y=700, observation="menu")
        other = BrainDecision(action_type="click", x=200, y=300, observation="menu")
        self.VisionBrain.return_value.decide.return_value = click
        first = self.create()
        first.step()
        first.step()
        self.HumanInputController.return_value.click.reset_mock()
        second = self.create()
        self.assertEqual(len(second.recent_history), 2)
        self.VisionBrain.return_value.decide.side_effect = [click, other]
        second.step()
        calls = self.VisionBrain.return_value.decide.call_args_list
        self.assertIn("AVOID REPEATING", calls[-2].kwargs["context_hint"])
        self.assertIn("REPLAN:", calls[-1].kwargs["context_hint"])
        self.assertEqual(second.last_decision.x, 200)
        self.assertEqual(self.HumanInputController.return_value.click.call_count, 1)
        self.assertEqual(second.memory.get_stats(second.game_id)["total_actions"], 3)
        row = second.memory.get_recent_experiences(second.game_id)[-1]
        self.assertEqual(row["next_state_hash"], second.memory.compute_state_hash(self.frame))
        self.assertEqual(row["observation"], "menu")

    def test_default_look_skips_vlm_result_assessment_and_is_not_learned_as_progress(self):
        self.addCleanup(setattr, self.module.config, "assess_look_with_vlm",
                        self.module.config.assess_look_with_vlm)
        self.module.config.assess_look_with_vlm = False
        self.VisionBrain.return_value.decide.return_value = BrainDecision(
            action_type="look", look_direction="left", look_amount="small")
        agent = self.create()
        result = agent.step()
        self.VisionBrain.return_value.assess.assert_not_called()
        self.assertEqual(result["decision"]["action_type"], "look")
        row = agent.memory.get_recent_experiences(agent.game_id)[-1]
        self.assertFalse(row["effective"])
        self.assertFalse(row["learnable"])
        self.assertEqual(row["outcome_status"], "camera_motion")

    def test_repeated_state_starts_existing_recovery_policy(self):
        self.addCleanup(setattr, self.module.config, "state_repeat_detection_enabled",
                        self.module.config.state_repeat_detection_enabled)
        self.addCleanup(setattr, self.module.config, "state_repeat_triggers_recovery",
                        self.module.config.state_repeat_triggers_recovery)
        self.module.config.state_repeat_detection_enabled = True
        self.module.config.state_repeat_triggers_recovery = True
        self.VisionBrain.return_value.decide.return_value = BrainDecision(
            action_type="walk", key="w", duration=0.5)
        agent = self.create()
        agent.step()
        second = agent.step()
        self.assertTrue(second["supervision"]["mode"] == "recovering"
                        or agent.supervisor.recovery_count >= 1)

    def test_ignored_replan_never_executes_known_failed_click(self):
        click = BrainDecision(action_type="click", x=500, y=700)
        self.VisionBrain.return_value.decide.return_value = click
        agent = self.create()
        agent.step()
        agent.step()
        self.HumanInputController.return_value.click.reset_mock()
        with patch.object(self.module.time, "sleep"):
            agent.step()
        self.HumanInputController.return_value.click.assert_not_called()
        self.assertEqual(agent.last_decision.action_type, "wait")
        self.assertFalse(agent.memory.get_recent_experiences(agent.game_id)[-1]["learnable"])

    def test_attach_switches_and_restores_profile_history(self):
        self.VisionBrain.return_value.decide.return_value = BrainDecision(action_type="jump")
        agent = self.create()
        agent.step()
        agent.attach(456, "creative")
        self.assertEqual(agent.game_id, "game:creative")
        self.assertEqual(agent.recent_history, [])
        agent.attach(789, "  SURVIVAL ")
        self.assertEqual(agent.game_id, "game:survival")
        self.assertEqual(len(agent.recent_history), 1)
        self.assertIsNone(agent.last_decision)

    def test_attach_without_focus_is_available_for_read_only_goal_runs(self):
        agent = self.create()

        class FakeCapture:
            hwnd = 0
            last_rect = None
            visible_checks = 0
            focus_calls = 0

            def get_window_title(self):
                return "Minecraft"

            def assert_visible(self):
                self.visible_checks += 1

            def assert_foreground(self):
                raise AssertionError("read-only attach must not require foreground")

            def get_client_rect(self):
                return (0, 0, 100, 100)

            def focus(self):
                self.focus_calls += 1

        capture = FakeCapture()
        agent.capture = capture
        agent.attach(456, "creative", focus=False)
        self.assertEqual(capture.focus_calls, 0)
        self.assertEqual(capture.visible_checks, 1)

    def test_focus_failure_prevents_capture_inference_and_experience(self):
        agent = self.create()
        self.WindowCapture.return_value.focus.side_effect = RuntimeError("focus denied")
        with self.assertRaisesRegex(RuntimeError, "focus denied"):
            agent.step()
        self.WindowCapture.return_value.grab_frame.assert_not_called()
        self.VisionBrain.return_value.decide.assert_not_called()
        self.assertEqual(agent.memory.get_stats(agent.game_id)["total_actions"], 0)
        self.assertEqual(agent.memory.get_stats(agent.game_id)["failed_attempts"], 1)
        self.assertIn("focus denied", agent.last_error)

    def test_vlm_failure_stops_before_input_instead_of_repeating_wait(self):
        from omnigamer.telemetry import RunTelemetry

        agent = self.create()
        class QuietSampler:
            def __init__(self, *_args): pass
            def start(self): pass
            def close(self): pass

        agent.telemetry = RunTelemetry(self.temp.name, resource_sampler_factory=QuietSampler)
        run_dir = Path(agent.telemetry.start({"game_id": agent.game_id}))
        self.VisionBrain.return_value.decide.return_value = BrainDecision(action_type="wait", duration=0.5)
        self.VisionBrain.return_value.last_error = "ValidationError: invalid coordinates"
        self.VisionBrain.return_value.telemetry_grounding_seconds = 0.0
        self.VisionBrain.return_value.telemetry_http_requests = 0
        self.VisionBrain.return_value.telemetry_decision_calls = 0
        with self.assertRaisesRegex(RuntimeError, "VLM decision failed"):
            agent.step()
        agent.end_telemetry_run("error")
        self.assertIn("invalid coordinates", agent.last_error)
        self.assertEqual(agent.last_decision.action_type, "wait")
        self.HumanInputController.return_value.press_key.assert_not_called()
        step = json.loads((run_dir / "steps.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(step["decision"]["fallback_reason"], "ValidationError: invalid coordinates")
        self.HumanInputController.return_value.click.assert_not_called()
        self.assertEqual(agent.memory.get_stats(agent.game_id)["total_actions"], 0)

    def test_window_resize_during_inference_does_not_execute_stale_coordinates(self):
        agent = self.create()
        self.VisionBrain.return_value.decide.return_value = BrainDecision(action_type="click", x=500, y=700)
        self.WindowCapture.return_value.get_client_rect.return_value = (20, 20, 200, 200)
        with self.assertRaisesRegex(RuntimeError, "moved/resized"):
            agent.step()
        self.HumanInputController.return_value.click.assert_not_called()
        self.assertEqual(agent.memory.get_stats(agent.game_id)["total_actions"], 0)

    def test_jump_really_combines_forward_and_space_with_short_duration(self):
        agent = self.create()
        self.VisionBrain.return_value.decide.return_value = BrainDecision(action_type="jump",key="space",duration=2)
        agent.step()
        self.HumanInputController.return_value.press_keys.assert_called_once_with(["w","space"],duration=0.75)
        row = agent.memory.get_recent_experiences(agent.game_id)[-1]
        self.assertEqual((row["key"], row["duration"]), ("w", 0.75))
        self.assertIn("vision_seconds", agent.last_timings)

    def test_ignored_navigation_replan_inspects_instead_of_walking_again(self):
        agent = self.create()
        signature = agent.memory.compute_visual_signature(self.frame)
        for state in ("0000000000000000", "ffffffffffffffff"):
            agent.memory.record_experience(1, state, "walk",None,None,"w",0.001,False,"test","explore",
                                           game_id=agent.game_id,visual_signature=signature,duration=2)
        self.VisionBrain.return_value.decide.return_value = BrainDecision(action_type="walk",key="w",duration=2)
        agent.step()
        self.HumanInputController.return_value.press_key.assert_not_called()
        self.HumanInputController.return_value.move_camera.assert_called_once_with(90, 0)
        self.assertEqual(agent.last_action_source, "navigation_recovery")

    def test_stagnation_starts_recovery_and_restart_continues_play(self):
        agent = self.create()
        self.VisionBrain.return_value.decide.return_value = BrainDecision(action_type="walk",key="w",duration=.5)
        white=Image.new("RGB",(100,100),"white")
        self.WindowCapture.return_value.grab_frame.side_effect = [self.frame,white]*20
        for _ in range(20):
            result=agent.step()
        self.assertFalse(result["supervision"]["halted"])
        self.assertEqual(result["supervision"]["mode"],"recovering")
        self.assertEqual(agent.memory.get_stats(agent.game_id)["effective_actions"],20)
        resumed=self.create()
        self.WindowCapture.return_value.grab_frame.side_effect=None
        resumed.step()
        self.assertEqual(self.WindowCapture.return_value.grab_frame.call_count,42)
        self.assertGreaterEqual(resumed.supervisor.recovery_count,1)
        self.assertFalse(resumed.supervisor.halted)
        self.assertLessEqual(resumed.supervisor.recovery_remaining,4)

    def test_two_visual_checks_store_outcome_and_play_continues(self):
        from omnigamer.perception import FrameFacts,VisualAssessment
        self.addCleanup(setattr,self.module.config,"verification_interval",self.module.config.verification_interval)
        self.addCleanup(setattr,self.module.config,"assess_look_with_vlm",self.module.config.assess_look_with_vlm)
        self.addCleanup(setattr,self.module.config,"state_repeat_detection_enabled",
                        self.module.config.state_repeat_detection_enabled)
        self.module.config.verification_interval=1
        self.module.config.assess_look_with_vlm=True
        self.module.config.state_repeat_detection_enabled=False
        assessment=VisualAssessment(before=FrameFacts(scene="gameplay",pitch="down"),
            after=FrameFacts(scene="gameplay",pitch="level"),confidence=.95,
            evidence="The horizon is visible in the after frame.",result="progress")
        self.VisionBrain.return_value.assess.return_value=assessment
        self.VisionBrain.return_value.decide.return_value=BrainDecision(
            action_type="look",look_direction="up",look_amount="medium")
        agent=self.create()
        first=agent.step()
        self.assertFalse(first["verification"]["completed"])
        with patch.object(self.module.time,"sleep"):
            second=agent.step()
        self.assertTrue(second["verification"]["verified_change"])
        self.assertFalse(second["verification"]["completed"])
        self.assertFalse(second["supervision"]["halted"])
        self.assertEqual(agent.last_action_source,"vlm")
        self.assertEqual(agent.memory.get_stats(agent.game_id)["checked_play_outcomes"],1)
        self.assertEqual(self.HumanInputController.return_value.move_camera.call_count,2)
        self.assertEqual(agent.memory.get_stats(agent.game_id)["effective_actions"],0)

    def test_mining_beneath_feet_is_prechecked_and_never_executed(self):
        from omnigamer.perception import FrameFacts,VisualAssessment
        down=VisualAssessment(before=FrameFacts(scene="gameplay",pitch="down"),
            after=FrameFacts(scene="gameplay",pitch="down",safe_footing=False),confidence=.8,
            evidence="Ground under the feet fills the screen.",result="regressed")
        self.VisionBrain.return_value.assess.return_value=down
        self.VisionBrain.return_value.decide.return_value=BrainDecision(action_type="mine",duration=3)
        agent=self.create();agent.step()
        self.HumanInputController.return_value.hold_mouse.assert_not_called()
        self.HumanInputController.return_value.move_camera.assert_called_once_with(0,-50)
        self.assertEqual(agent.last_action_source,"camera_guard")
        self.assertEqual(agent.memory.get_stats(agent.game_id)["visually_confirmed_goals"],0)

    def test_telemetry_records_action_source_and_mine_guard_evidence(self):
        from omnigamer.perception import FrameFacts, VisualAssessment
        from omnigamer.telemetry import RunTelemetry

        class QuietSampler:
            def __init__(self, *_args): pass
            def start(self): pass
            def close(self): pass

        assessment = VisualAssessment(
            before=FrameFacts(scene="gameplay", pitch="level"),
            after=FrameFacts(scene="gameplay", pitch="level", safe_footing=False),
            confidence=.95, result="unknown", evidence="The ground is not confirmed.")
        self.VisionBrain.return_value.decide.return_value = BrainDecision(action_type="mine", duration=1.5)
        self.VisionBrain.return_value.assess.return_value = assessment
        self.VisionBrain.return_value.telemetry_grounding_seconds = 0.0
        self.VisionBrain.return_value.telemetry_http_requests = 1
        self.VisionBrain.return_value.telemetry_decision_calls = 1
        self.VisionBrain.return_value.telemetry_raw_json_responses = []
        self.VisionBrain.return_value.assess.side_effect = lambda *_args, **_kwargs: (
            self.VisionBrain.return_value.telemetry_raw_json_responses.append('{"raw":"preflight"}') or assessment)

        agent = self.create()
        agent.telemetry = RunTelemetry(self.temp.name, resource_sampler_factory=QuietSampler)
        run_dir = Path(agent.telemetry.start({"game_id": agent.game_id}))
        agent.step()
        agent.end_telemetry_run("step_limit")
        row = json.loads((run_dir / "steps.jsonl").read_text(encoding="utf-8"))

        self.assertEqual(row["decision"]["action_type"], "wait")
        self.assertEqual(row["decision"]["action_source"], "camera_guard")
        self.assertEqual(row["decision"]["guard"]["facts"]["safe_footing"], False)
        self.assertFalse(row["decision"]["guard"]["facts"]["grounded_confident"])
        self.assertEqual(row["decision"]["guard"]["refusal_condition"], ["grounded_confident_false"])
        self.assertEqual(row["decision"]["guard"]["preflight"]["raw_response"], '{"raw":"preflight"}')

    def test_state_repeat_is_telemetried_but_does_not_recover_by_default(self):
        from omnigamer.telemetry import RunTelemetry

        class QuietSampler:
            def __init__(self, *_args): pass
            def start(self): pass
            def close(self): pass

        self.addCleanup(setattr, self.module.config, "state_repeat_detection_enabled",
                        self.module.config.state_repeat_detection_enabled)
        self.addCleanup(setattr, self.module.config, "state_repeat_triggers_recovery",
                        self.module.config.state_repeat_triggers_recovery)
        self.module.config.state_repeat_detection_enabled = True
        self.module.config.state_repeat_triggers_recovery = False
        self.VisionBrain.return_value.decide.return_value = BrainDecision(action_type="wait", duration=.1)
        self.VisionBrain.return_value.telemetry_grounding_seconds = 0.0
        self.VisionBrain.return_value.telemetry_http_requests = 1
        self.VisionBrain.return_value.telemetry_decision_calls = 1

        agent = self.create()
        agent.telemetry = RunTelemetry(self.temp.name, resource_sampler_factory=QuietSampler)
        run_dir = Path(agent.telemetry.start({"game_id": agent.game_id}))
        state_hash = agent.memory.compute_state_hash(self.frame)
        agent.state_repeat_detector.remember(state_hash, 0, {"action_type": "look", "look_direction": "left"})
        agent.step()
        agent.end_telemetry_run("step_limit")
        row = json.loads((run_dir / "steps.jsonl").read_text(encoding="utf-8"))

        self.assertTrue(row["loop"]["state_repeat"])
        self.assertEqual(row["loop"]["repeat_of_step"], 0)
        self.assertEqual(agent.supervisor.mode, "exploring")

    def test_independent_check_can_allow_safe_forward_mining(self):
        from omnigamer.perception import FrameFacts,VisualAssessment
        facts=FrameFacts(scene="gameplay",pitch="level",safe_footing=True)
        self.VisionBrain.return_value.assess.return_value=VisualAssessment(before=facts,after=facts,
            confidence=.95,evidence="Solid footing and a forward tree trunk are visible.",result="unknown")
        self.VisionBrain.return_value.decide.return_value=BrainDecision(action_type="mine",duration=3)
        agent=self.create();agent.step()
        self.HumanInputController.return_value.hold_mouse.assert_called_once_with(button="left",duration=3)
        self.assertEqual(agent.brain.assess.call_count,2)

    def test_generic_profile_skips_mining_specific_preflight(self):
        self.VisionBrain.return_value.decide.return_value = BrainDecision(
            action_type="mine", duration=0.5)
        agent = self.create(game_id="platformer", mine_requires_grounded_level_view=False,
                            camera_type="none")
        agent.step()
        self.VisionBrain.return_value.assess.assert_called_once()
        self.HumanInputController.return_value.hold_mouse.assert_called_once()

    def test_mining_profile_requires_fresh_crosshair_target_evidence(self):
        from omnigamer.perception import FrameFacts, VisualAssessment
        facts=FrameFacts(scene="gameplay",pitch="level",safe_footing=True,
                         target_at_crosshair=None)
        self.VisionBrain.return_value.assess.return_value=VisualAssessment(
            before=facts, after=facts, confidence=.95,
            evidence="Solid footing, target under the crosshair is unclear.", result="unknown")
        self.VisionBrain.return_value.decide.return_value=BrainDecision(action_type="mine",duration=3)
        agent=self.create(mine_requires_target_at_crosshair=True)
        result=agent.step()
        self.HumanInputController.return_value.hold_mouse.assert_not_called()
        self.assertEqual(result["decision"]["action_type"], "wait")
        self.assertIn("под прицелом", agent.last_guard_reason)

    def test_unknown_check_retains_executed_experience_without_promoting_success(self):
        self.addCleanup(setattr,self.module.config,"verification_interval",self.module.config.verification_interval)
        self.module.config.verification_interval=1
        self.VisionBrain.return_value.decide.return_value=BrainDecision(action_type="jump",key="w")
        self.VisionBrain.return_value.last_assessment_error="invalid critic JSON"
        self.WindowCapture.return_value.grab_frame.side_effect=[self.frame,Image.new("RGB",(100,100),"white")]
        agent=self.create();agent.step()
        row=agent.memory.get_recent_experiences(agent.game_id)[-1]
        self.assertTrue(row["effective"])
        self.assertEqual(row["outcome_status"],"unverified")
        self.assertEqual(agent.memory.get_stats(agent.game_id)["visually_confirmed_goals"],0)
        self.assertEqual(agent.last_assessment_error,"invalid critic JSON")


    def test_menu_exploration_confirmation_restart_replay_and_invalidation(self):
        from tests.test_menu_memory import menu
        from omnigamer.perception import FrameFacts, VisualAssessment
        source=FrameFacts(scene="menu",menu_id="main_menu")
        destination=FrameFacts(scene="gameplay")
        success=VisualAssessment(before=source,after=destination,confidence=.95,result="progress",
            evidence="The Play button opened gameplay.",menu_action_succeeded=True)
        confirm=VisualAssessment(before=destination,after=destination,confidence=.95,
            evidence="Gameplay remains visible in the new observation.")
        action=BrainDecision(action_type="click",x=500,y=500,scene_type="menu",menu_id="main_menu",
            visual_confidence=.95,target_label="PLAY",expected_result="gameplay")
        frame=menu()
        capture=self.WindowCapture.return_value
        capture.grab_frame.side_effect=[frame,self.frame,self.frame,self.frame,frame,self.frame]
        self.VisionBrain.return_value.decide.side_effect=[action,BrainDecision(
            action_type="look",look_direction="right",look_amount="small"),
            action.model_copy(update={"x":200})]
        self.VisionBrain.return_value.assess.side_effect=[success,confirm,confirm,VisualAssessment(
            before=source,after=source,confidence=.95,result="blocked",evidence="Play did not open gameplay.",
            menu_action_succeeded=False)]
        first=self.create()
        first.step()
        self.assertIsNotNone(first._pending_menu)
        self.assertIsNone(first.menu_memory.recall(first.game_id,frame,action))
        first.step()
        self.assertEqual(first.last_menu_memory["mode"],"learned")
        second=self.create()
        second.step()
        self.assertEqual(second.last_action_source,"menu_memory")
        self.assertEqual(second.last_decision.x,500)
        self.assertEqual(second.last_menu_memory["mode"],"invalidated")
        self.assertIsNone(second.menu_memory.recall(second.game_id,frame,action))

    def test_virtual_agent_checks_visibility_without_stealing_focus(self):
        self.WindowCapture.return_value.assert_visible=Mock()
        from omnigamer.profiles import GameProfile
        profile = GameProfile(game_id="virtual", camera_type="first_person")
        with patch.object(self.module,"WindowInputController") as controller:
            agent=self.module.OmniAgent(db_path=self.path,hwnd=123,game_id="virtual",input_backend="window",
                                        emergency_hotkey_enabled=False, telemetry_enabled=False,
                                        profile=profile)
            agent.attach(123,"virtual")
            self.WindowCapture.return_value.last_rect=(0,0,100,100)
        self.VisionBrain.return_value.decide.return_value=BrainDecision(
            action_type="look",look_direction="right",look_amount="small")
        agent.step()
        self.WindowCapture.return_value.focus.assert_not_called()
        self.WindowCapture.return_value.assert_foreground.assert_not_called()
        self.assertGreaterEqual(self.WindowCapture.return_value.assert_visible.call_count,3)
        controller.return_value.move_camera.assert_called_once()

    def test_unknown_menu_click_still_triggers_menu_verifier(self):
        self.VisionBrain.return_value.decide.return_value=BrainDecision(action_type="click",x=500,y=500)
        agent=self.create()
        agent.step()
        assessment_action=self.VisionBrain.return_value.assess.call_args.args[2]
        self.assertEqual(assessment_action["scene_type"],"menu")

    def test_minecraft_title_status_suffix_does_not_switch_memory_profile(self):
        self.WindowCapture.return_value.get_window_title.return_value="Minecraft* 26.3"
        agent=self.module.OmniAgent(db_path=self.path,hwnd=123,input_backend="native",
                                     emergency_hotkey_enabled=False, telemetry_enabled=False,
                                     profile=load_profile("minecraft"))
        original=agent.game_id
        self.WindowCapture.return_value.get_window_title.return_value="Minecraft* 26.3 - Одиночная игра"
        agent.attach(123)
        self.assertEqual(agent.game_id,original)

    def test_profile_forbidden_key_is_rejected_before_dispatch(self):
        agent = self.create()
        agent.profile = load_profile("minecraft")
        with self.assertRaisesRegex(ValueError, "forbids requested key"):
            agent._check_profile_forbidden_key(BrainDecision(action_type="key", key="esc"))
        agent._check_profile_forbidden_key(BrainDecision(action_type="walk", key="w"))


class BrainMemoryTests(unittest.TestCase):
    def test_visual_check_uses_before_after_and_hud_without_actor_narrative(self):
        response=Mock()
        response.json.return_value={"choices":[{"message":{"content":json.dumps({
            "before":{"scene":"gameplay","pitch":"down"},
            "after":{"scene":"gameplay","pitch":"down"},"result":"regressed",
            "confidence":.8,"evidence":"The player dug into a hole beneath the feet."})}}]}
        brain=VisionBrain()
        with patch("omnigamer.vision_brain.requests.post",return_value=response) as post:
            result=brain.assess(Image.new("RGB",(100,100)),Image.new("RGB",(100,100)),{"action_type":"mine"},"reach_open_ground")
        self.assertEqual(result.result,"regressed")
        content=post.call_args.kwargs["json"]["messages"][1]["content"]
        self.assertEqual(sum(part["type"]=="image_url" for part in content),4)
        self.assertIn("BEFORE",[part.get("text") for part in content])
        self.assertIsNone(brain.last_assessment_error)

    def test_malformed_visual_check_is_unknown_not_a_success(self):
        response=Mock()
        response.json.return_value={"choices":[{"message":{"content":"not JSON"}}]}
        brain=VisionBrain()
        with patch("omnigamer.vision_brain.requests.post",return_value=response):
            result=brain.assess(Image.new("RGB",(100,100)),Image.new("RGB",(100,100)),{},"restore_view")
        self.assertFalse(result.usable)
        self.assertEqual(result.result,"unknown")
        self.assertIn("ValueError",brain.last_assessment_error)
    def test_unit_fraction_click_coordinates_become_1000_scale(self):
        decision = VisionBrain()._parse_json_response('{"action_type":"click","x":0.5,"y":0.6}')
        self.assertEqual((decision.x, decision.y), (500, 600))

    def test_fraction_drag_uses_same_scale_for_both_endpoints(self):
        decision = VisionBrain()._parse_json_response('{"action_type":"drag","x":0,"y":0.2,"x2":1,"y2":0.8}')
        self.assertEqual((decision.x,decision.y,decision.x2,decision.y2), (0,200,1000,800))

    def test_native_coordinates_are_rounded_and_not_rescaled(self):
        decision = VisionBrain()._parse_json_response('{"action_type":"click","x":500.2,"y":600.7}')
        self.assertEqual((decision.x, decision.y), (500, 601))

    def test_relative_camera_is_never_scaled_as_screen_coordinates(self):
        decision = VisionBrain()._parse_json_response(
            '{"action_type":"look","look_direction":"left","look_amount":"large"}')
        self.assertEqual((decision.x, decision.y), (-90, 0))
        with self.assertRaises(ValueError):
            VisionBrain()._parse_json_response('{"action_type":"look","x":-120.4,"y":0.6}')

    def test_regex_fallback_preserves_fractions_until_conversion(self):
        decision = VisionBrain()._parse_json_response('{"action_type":"click","x":0.5,"y":0.6,}')
        self.assertEqual((decision.x, decision.y), (500, 600))

    def test_invalid_or_mixed_units_are_not_silently_executed(self):
        for values in ({"x":0.5,"y":600}, {"x":-1,"y":600}, {"x":1001,"y":600},
                       {"x":float('nan'),"y":600}, {"x":True,"y":600}, {"x":500}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                VisionBrain()._parse_json_response(json.dumps(dict(action_type="click",**values)))

    def test_null_optional_timing_does_not_discard_a_click(self):
        decision = VisionBrain()._parse_json_response(
            '{"action_type":"click","x":450,"y":260,"duration":null,"confidence":null}'
        )
        self.assertEqual(decision.action_type, "click")
        self.assertEqual(decision.duration, 1.5)
    def test_retrieved_evidence_is_in_actual_http_payload(self):
        with tempfile.TemporaryDirectory() as folder:
            brain = VisionBrain()
            brain.log_file = Path(folder) / "raw.log"
            response = Mock()
            response.json.return_value = {"choices": [{"message": {"content": json.dumps({
                "action_type":"jump","scene_type":"gameplay","camera_pitch":"level","visual_confidence":.9,
                "menu_id":"unknown","menu_state":"default","target_label":"","expected_result":"move",
                "observation":"Ground ahead","intent":"Cross a low step","commentary":"Попробую прыгнуть."})}}]}
            with patch("omnigamer.vision_brain.requests.post", return_value=response) as post:
                decision = brain.decide(Image.new("RGB", (8, 8)), "PERSISTED_ACTION_EVIDENCE")
            payload = post.call_args.kwargs["json"]
            self.assertIn("PERSISTED_ACTION_EVIDENCE", payload["messages"][1]["content"][0]["text"])
            self.assertEqual(payload["max_tokens"], brain.max_tokens)
            self.assertEqual(post.call_args.kwargs["timeout"], config.request_timeout)
            self.assertEqual(decision.action_type, "jump")

    def test_network_failure_waits_and_reports_nonlearnable_error(self):
        brain = VisionBrain()
        with patch("omnigamer.vision_brain.requests.post", side_effect=requests.Timeout("offline")), \
                patch("omnigamer.vision_brain.Path.write_text"):
            decision = brain.decide(Image.new("RGB", (8, 8)))
        self.assertEqual(decision.action_type, "wait")
        self.assertIn("Timeout", brain.last_error)

    def test_unparseable_response_cannot_silently_train_forward_walking(self):
        brain = VisionBrain()
        for text in ("not an action", "{}", '{"action_type":"teleport"}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                brain._parse_json_response(text)


if __name__ == "__main__":
    unittest.main()
