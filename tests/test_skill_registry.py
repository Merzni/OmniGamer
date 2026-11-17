import unittest

from omnigamer.skills.approach_target import ApproachTargetResult
from omnigamer.skills.registry import (
    SkillStatus, _approach_result, default_registry,
)


class SkillRegistryTests(unittest.TestCase):
    def test_catalog_exposes_validated_contract_for_approach(self):
        registry = default_registry()
        self.assertEqual(registry.names(), (
            "approach_target", "click_menu_target", "interact_until_change", "move_pulse",
            "search_target", "turn", "wait_for_condition"))
        schema = next(item for item in registry.planner_catalog() if item["name"] == "approach_target")
        self.assertEqual(schema["resources"], ["detector", "input", "camera_guard"])
        self.assertIn("APPROACH_REACHED", schema["results"])
        self.assertNotIn("SUCCESS", schema["results"])
        self.assertEqual(registry.validate("approach_target", {"target_text": "a target"}).target_text,
                         "a target")

    def test_menu_skill_is_only_offered_to_profiles_with_ui_adapter(self):
        from omnigamer.profiles import GameProfile

        without_ui = default_registry(GameProfile(game_id="plain"))
        with_ui = default_registry(GameProfile(game_id="ui", ui_adapters=["world_selection"]))
        self.assertNotIn("click_menu_target", without_ui.names())
        self.assertIn("click_menu_target", with_ui.names())
        params = with_ui.validate("click_menu_target", {
            "target_label": "Singleplayer", "intent": "Open the world list",
            "expected_result": "world selection screen"})
        self.assertEqual(params.target_label, "Singleplayer")

    def test_unknown_skill_and_extra_parameters_are_rejected(self):
        registry = default_registry()
        with self.assertRaisesRegex(ValueError, "Unknown skill"):
            registry.get("missing")
        with self.assertRaises(Exception):
            registry.validate("approach_target", {"target_text": "target", "extra": True})

    def test_legacy_skill_states_map_to_explicit_contract_results(self):
        def raw(state):
            return ApproachTargetResult(state, "reason", None, 2, 1, 0.5)

        self.assertEqual(_approach_result(raw("REACHED")).status, SkillStatus.SUCCESS)
        self.assertEqual(_approach_result(raw("REACHED")).result_code, "APPROACH_REACHED")
        self.assertEqual(_approach_result(raw("REACHED_UNCONFIRMED")).status, SkillStatus.UNCONFIRMED)
        self.assertEqual(_approach_result(raw("UNCERTAIN")).status, SkillStatus.UNCONFIRMED)
        self.assertEqual(_approach_result(raw("ABORT")).status, SkillStatus.ABORT)
        self.assertEqual(_approach_result(raw("LOST")).status, SkillStatus.FAILED)


if __name__ == "__main__":
    unittest.main()
