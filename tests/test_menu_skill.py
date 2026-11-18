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

    def test_unrelated_localized_label_is_rejected_before_click(self):
        unrelated = self.grounded.model_copy(update={"target_label": "Выйти из игры"})
        with self.assertRaisesRegex(ValueError, "does not match the requested target"):
            self.invoke(Mock(return_value=unrelated))
        self.controls.click.assert_not_called()

    def test_localized_menu_transition_cannot_override_requested_result(self):
        unsafe = self.grounded.model_copy(update={"expected_result": "gameplay"})
        with self.assertRaisesRegex(ValueError, "different expected result"):
            self.invoke(Mock(return_value=unsafe))
        self.controls.click.assert_not_called()

    def test_russian_localized_menu_transition_cannot_override_requested_result(self):
        unsafe = self.grounded.model_copy(update={"expected_result": "Мир выбран"})
        with self.assertRaisesRegex(ValueError, "different expected result"):
            self.invoke(Mock(return_value=unsafe))
        self.controls.click.assert_not_called()

    def test_equivalent_russian_world_selection_result_is_accepted(self):
        localized = self.grounded.model_copy(update={
            "expected_result": "Список доступных миров"})
        result = click_menu_target("Одиночная игра", "Open the world list",
            "Переход к выбору мира", capture=self.capture, controls=self.controls,
            profile=self.profile, grounder=Mock(return_value=localized),
            prepare_target=Mock(), check_target=Mock())
        self.assertEqual(result.result_code, "MENU_ACTION_UNCONFIRMED")
        self.controls.click.assert_called_once()

    def test_equivalent_russian_gameplay_result_is_accepted(self):
        localized = self.grounded.model_copy(update={"expected_result": "Игра запущена"})
        result = click_menu_target("Одиночная игра", "Start the game",
            "Переход в игровой процесс", capture=self.capture, controls=self.controls,
            profile=self.profile, grounder=Mock(return_value=localized),
            prepare_target=Mock(), check_target=Mock())
        self.assertEqual(result.result_code, "MENU_ACTION_UNCONFIRMED")
        self.controls.click.assert_called_once()

    def test_world_selection_adapter_refuses_unspecified_transition(self):
        unsafe = self.grounded.model_copy(update={"menu_id": "world_select",
                                                  "menu_state": "no_selection",
                                                  "expected_result": "world_selected"})
        with self.assertRaisesRegex(ValueError, "different expected result"):
            click_menu_target("visible_world_row", "Select a world", "selected state",
                capture=self.capture, controls=self.controls, profile=self.profile,
                grounder=Mock(return_value=unsafe), prepare_target=Mock(), check_target=Mock())
        self.controls.click.assert_not_called()

    def test_measured_world_row_alias_is_allowed_for_selection_postcondition(self):
        measured = self.grounded.model_copy(update={
            "target_label": "visible_world_row", "menu_id": "world_select",
            "menu_state": "no_selection", "expected_result": "world_selected"})
        result = click_menu_target("visible existing world", "Select a world",
            "world_selected", capture=self.capture, controls=self.controls,
            profile=self.profile, grounder=Mock(return_value=measured),
            prepare_target=Mock(), check_target=Mock())
        self.assertEqual(result.result_code, "MENU_ACTION_UNCONFIRMED")
        self.controls.click.assert_called_once()

    def test_measured_play_alias_requires_selected_world_and_gameplay_postcondition(self):
        measured = self.grounded.model_copy(update={
            "target_label": "Play selected world", "menu_id": "world_select",
            "menu_state": "world_selected", "expected_result": "gameplay"})
        result = click_menu_target("Play", "Start the selected world", "gameplay",
            capture=self.capture, controls=self.controls, profile=self.profile,
            grounder=Mock(return_value=measured), prepare_target=Mock(),
            check_target=Mock())
        self.assertEqual(result.result_code, "MENU_ACTION_UNCONFIRMED")
        self.controls.click.assert_called_once()

    def test_window_geometry_change_cancels_click(self):
        self.capture.get_client_rect.side_effect = [
            (10, 20, 1010, 820), (11, 20, 1011, 820)]
        with self.assertRaisesRegex(RuntimeError, "moved/resized"):
            self.invoke()
        self.controls.click.assert_not_called()

    def test_target_change_during_localization_cancels_stale_click(self):
        before = Image.new("RGB", (1000, 800), (20, 20, 20))
        changed = before.copy()
        from PIL import ImageDraw
        ImageDraw.Draw(changed).rectangle((250, 350, 600, 580), fill=(220, 220, 220))
        self.capture.grab_frame.side_effect = [before, changed, changed]

        with self.assertRaisesRegex(RuntimeError, "target changed during localization"):
            self.invoke()

        self.assertEqual(self.capture.grab_frame.call_count, 2)
        self.controls.click.assert_not_called()

    def test_animation_outside_target_patch_does_not_cancel_click(self):
        before = Image.new("RGB", (1000, 800), (20, 20, 20))
        changed = before.copy()
        from PIL import ImageDraw
        ImageDraw.Draw(changed).rectangle((0, 0, 100, 100), fill=(220, 220, 220))
        self.capture.grab_frame.side_effect = [before, changed, changed]

        result = self.invoke()

        self.assertEqual(result.result_code, "MENU_ACTION_UNCONFIRMED")
        self.controls.click.assert_called_once_with(420, 540, button="left", hold_time=0.08)

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
