import unittest
from unittest.mock import Mock

from PIL import Image

from omnigamer.profiles import GameProfile
from omnigamer.planner import GoalExecutor, GoalPlanner, InputActionBudget, BudgetControls, SkillOutcomeStore
from omnigamer.skills.contracts import SkillStatus
from omnigamer.skills.registry import default_registry
from omnigamer.skills.menu import click_menu_target
from omnigamer.vision_brain import BrainDecision


class MenuSkillTests(unittest.TestCase):
    def setUp(self):
        self.profile = GameProfile(game_id="ui", ui_adapters=["world_selection"])
        self.capture = Mock()
        self.capture.last_rect = (10, 20, 1010, 820)
        self.capture.get_client_rect.return_value = self.capture.last_rect
        self.capture.grab_frame.return_value = Image.new("RGB", (1000, 800))
        self.capture.norm_to_screen.side_effect = lambda x, y: (x + 10, y + 20)
        self.controls = Mock()
        self.grounded = BrainDecision(action_type="click", x=410, y=520,
            scene_type="menu", menu_id="main_menu", target_label="Одиночная игра",
            expected_result="world_select", visual_confidence=.98)

    def invoke(self, grounder=None):
        return click_menu_target("Одиночная игра", "Open the world list", "World selection screen appears",
            capture=self.capture, controls=self.controls, profile=self.profile,
            grounder=grounder or Mock(return_value=self.grounded),
            prepare_target=Mock(), check_target=Mock())

    def test_verified_menu_target_sends_one_click_and_defers_outcome(self):
        result = self.invoke()
        self.controls.click.assert_called_once_with(420, 540, button="left", hold_time=0.08)
        self.assertEqual(result.result_code, "MENU_ACTION_UNCONFIRMED")
        self.assertEqual(result.evidence["target_label"], "Одиночная игра")
        self.assertEqual(result.evidence["input_actions"], 1)
        self.assertEqual(result.evidence["postcondition"], "pending_visual_verification")

    def test_non_menu_grounding_sends_no_input(self):
        unsafe = self.grounded.model_copy(update={"scene_type": "unknown"})
        with self.assertRaisesRegex(ValueError, "not confidently grounded"):
            self.invoke(Mock(return_value=unsafe))
        self.controls.click.assert_not_called()

    def test_localized_menu_transition_cannot_override_requested_result(self):
        unsafe = self.grounded.model_copy(update={"expected_result": "gameplay"})
        with self.assertRaisesRegex(ValueError, "different expected result"):
            self.invoke(Mock(return_value=unsafe))
        self.controls.click.assert_not_called()

    def test_world_selection_adapter_refuses_unspecified_transition(self):
        unsafe = self.grounded.model_copy(update={"menu_id": "world_select",
                                                  "menu_state": "no_selection",
                                                  "expected_result": "world_selected"})
        with self.assertRaisesRegex(ValueError, "different expected result"):
            click_menu_target("visible_world_row", "Select a world", "selected state",
                capture=self.capture, controls=self.controls, profile=self.profile,
                grounder=Mock(return_value=unsafe), prepare_target=Mock(), check_target=Mock())
        self.controls.click.assert_not_called()

    def test_window_geometry_change_cancels_click(self):
        self.capture.get_client_rect.side_effect = [
            (10, 20, 1010, 820), (11, 20, 1011, 820)]
        with self.assertRaisesRegex(RuntimeError, "moved/resized"):
            self.invoke()
        self.controls.click.assert_not_called()

    def test_menu_click_consumes_one_shared_input_action(self):
        budget = InputActionBudget(1)
        controls = BudgetControls(self.controls, budget)
        controls.click(30, 40)
        self.assertEqual(budget.actions_used, 1)
        self.controls.click.assert_called_once_with(x=30, y=40, button="left", hold_time=0.08)

    def test_menu_transition_uses_visible_postcondition_verifier(self):
        class Assessment:
            usable = True
            result = "progress"
            confidence = .95
            evidence = "World list appeared"
            evidence_source = "test_fixture"

        registry = default_registry(self.profile)
        planner = GoalPlanner(registry, lambda **_: None)
        executor = GoalExecutor(planner, registry, SkillOutcomeStore(":memory:"),
            profile=self.profile, game_id=self.profile.game_id, execute_skill=lambda *_: None,
            frame_provider=lambda: self.capture.grab_frame(), stop_check=lambda: None,
            result_verifier=lambda *_: Assessment())
        from omnigamer.planner import PlannedStep
        step = PlannedStep(skill="click_menu_target",
            args={"target_label": "Одиночная игра", "intent": "Open world list",
                  "expected_result": "world list"},
            expected_result="world list", success_check="world selection menu is visible")
        result = executor._verify_result(step, {
            "status": SkillStatus.UNCONFIRMED.value,
            "result_code": "MENU_ACTION_UNCONFIRMED",
            "reason": "menu_click_sent_postcondition_pending", "evidence": {}},
            self.capture.grab_frame(), self.capture.grab_frame(), 5)
        self.assertEqual(result["status"], SkillStatus.SUCCESS.value)
        self.assertEqual(result["evidence"]["verification"]["scope"], "menu_transition")


if __name__ == "__main__":
    unittest.main()
