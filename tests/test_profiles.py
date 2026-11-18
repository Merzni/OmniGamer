import json
import tempfile
import unittest
from pathlib import Path

from omnigamer.profiles import GameProfile, load_profile
from omnigamer.skills.approach_target import ApproachTargetSettings
from omnigamer.agent import OmniAgent
from omnigamer.vision_brain import BrainDecision


class GameProfileTests(unittest.TestCase):
    def test_builtin_profile_validates_game_controls_and_regions(self):
        profile = load_profile("minecraft")
        self.assertEqual(profile.game_id, "minecraft")
        self.assertEqual(profile.move_keys["forward"], "w")
        self.assertEqual(profile.actions["primary_hold"].value, "left")
        self.assertEqual(profile.detector_queries["tree_trunk"], ["a tree trunk", "a tree"])
        self.assertIn("hotbar", profile.regions)

    def test_main_menu_transition_mapping_is_configurable_and_validated(self):
        profile = GameProfile(game_id="menu", ui_adapters=["stacked_main_menu"],
                              main_menu_target_transitions={"Single Player": "world_select"})
        self.assertEqual(profile.main_menu_target_transitions["Single Player"], "world_select")
        with self.assertRaises(ValueError):
            GameProfile(game_id="menu", main_menu_target_transitions={"Single Player": "gameplay"})

    def test_crosshair_target_precondition_is_explicitly_profiled(self):
        self.assertFalse(GameProfile(game_id="generic").mine_requires_target_at_crosshair)
        self.assertFalse(GameProfile(game_id="generic").mine_requires_grounded_level_view)
        self.assertTrue(GameProfile(game_id="legacy-first-person", camera_type="first_person")
                        .mine_requires_grounded_level_view)
        self.assertFalse(GameProfile(game_id="flat-game", camera_type="first_person",
                                     mine_requires_grounded_level_view=False)
                         .mine_requires_grounded_level_view)
        self.assertTrue(load_profile("minecraft").mine_requires_grounded_level_view)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "mining.json"
            path.write_text(json.dumps({"game_id": "mining-game",
                                        "mine_requires_target_at_crosshair": True}),
                            encoding="utf-8")
            profile = load_profile("mining-game", path)
            self.assertTrue(profile.mine_requires_target_at_crosshair)

    def test_explicit_profile_requires_matching_game_id(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "profile.json"
            path.write_text(json.dumps({"game_id": "other-game"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_profile("requested-game", path)

    def test_rejects_invalid_normalized_region_and_bad_id(self):
        with self.assertRaises(ValueError):
            GameProfile.model_validate({
                "game_id": "bad/id",
                "regions": {"hud": {"x1": 0.8, "y1": 0, "x2": 0.2, "y2": 1}},
            })

    def test_rejects_unsupported_profile_keys_and_nonfinite_thresholds(self):
        with self.assertRaisesRegex(ValueError, "unsupported key name"):
            GameProfile.model_validate({"game_id": "bad-keys", "move_keys": {
                "forward": "not-a-key", "backward": "s", "left": "a", "right": "d"}})
        with self.assertRaisesRegex(ValueError, "finite"):
            GameProfile.model_validate({"game_id": "bad-threshold", "thresholds": {"region_change": float("nan")}})

    def test_unknown_profile_id_uses_safe_generic_defaults(self):
        profile = load_profile("new-game")
        self.assertEqual(profile.game_id, "new-game")
        self.assertEqual(profile.detector_queries, {})
        self.assertEqual(profile.forbidden_keys, [])

    def test_documented_template_is_valid_profile_data(self):
        template = Path(__file__).resolve().parents[1] / "data" / "profiles" / "_template.json"
        profile = load_profile("example-game", template)
        self.assertEqual(profile.game_id, "example-game")
        self.assertTrue(profile.comment)

    def test_profile_thresholds_and_limits_configure_target_skill(self):
        settings = ApproachTargetSettings.from_profile(load_profile("minecraft"))
        self.assertEqual(settings.deadzone_x, 0.06)
        self.assertEqual(settings.score_min, 0.15)
        self.assertEqual(settings.max_seconds, 120)
        self.assertEqual(settings.max_impulses, 60)
        self.assertEqual(settings.max_look_counts, 30)
        self.assertEqual(settings.track_max_center_shift, 0.2)
        self.assertEqual(settings.reach_crosshair_margin, 0.08)
        self.assertEqual(settings.jump_recovery_attempts, 2)
        self.assertEqual(load_profile("minecraft").motion_limits.interaction_retry_step_seconds, 0)

    def test_minecraft_profile_allows_repeated_verified_harvest(self):
        profile = load_profile("minecraft")
        self.assertEqual(profile.actions["primary_hold"].duration, 2.5)
        self.assertEqual(profile.thresholds["region_change"], 0.18525896966457367)
        self.assertIn("max_repeats=3", profile.prompt_addendum)
        self.assertIn("stops as soon as it changes", profile.prompt_addendum)

    def test_legacy_motion_uses_profile_mapping_and_caps_duration(self):
        profile = GameProfile(game_id="arrow-game", camera_type="first_person",
            move_keys={"forward": "up", "backward": "down", "left": "left", "right": "right"},
            actions={"jump": {"kind": "key", "value": "space", "duration": 0.6}},
            motion_limits={"max_action_seconds": 0.65})
        agent = OmniAgent.__new__(OmniAgent)
        agent.profile = profile
        normalized = agent._normalize_motion(BrainDecision(action_type="walk", duration=2.0))
        self.assertEqual(normalized.key, "up")
        self.assertEqual(normalized.duration, 0.65)
        jump = agent._normalize_motion(BrainDecision(action_type="jump", key="space", duration=0.75))
        self.assertEqual(jump.key, "up")
        self.assertEqual(jump.duration, 0.6)

    def test_core_python_has_no_game_specific_control_vocabulary(self):
        source_root = Path(__file__).resolve().parents[1] / "omnigamer"
        banned = ("minecraft", "hotbar", "key=e", "tree_trunk", "a tree trunk", "a tree")
        for path in source_root.rglob("*.py"):
            contents = path.read_text(encoding="utf-8-sig").casefold()
            for marker in banned:
                with self.subTest(path=path.name, marker=marker):
                    self.assertNotIn(marker, contents)


if __name__ == "__main__":
    unittest.main()
