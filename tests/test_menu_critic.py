"""Menu critic schema compatibility must not invent world selection."""
import json
import unittest
from unittest.mock import Mock,patch
from PIL import Image
from omnigamer.vision_brain import VisionBrain
from omnigamer.profiles import load_profile


class MenuCriticTests(unittest.TestCase):
    def test_gameplay_assessment_schema_requires_crosshair_target_field(self):
        schema=VisionBrain()._assessment_format(False)["json_schema"]["schema"]["$defs"]["FrameFacts"]
        self.assertIn("target_at_crosshair",schema["required"])
        self.assertIn("target_at_crosshair",schema["properties"])

    def assess(self, facts):
        payload={"before":{"scene":"main_menu"},"after":facts,"confidence":.95,
                 "menu_action_succeeded":False,"evidence":"The world selection menu is visible."}
        response=Mock()
        response.json.return_value={"choices":[{"message":{"content":json.dumps(payload)}}]}
        with patch("omnigamer.vision_brain.requests.post",return_value=response):
            frame=Image.new("RGB",(800,500))
            brain=VisionBrain();brain.active_profile=load_profile("minecraft")
            return brain.assess(frame,frame,{"scene_type":"menu"},"Free play")

    def test_disabled_play_overrides_claimed_selection_and_scene_alias(self):
        result=self.assess({"scene":"world_select","menu_state":"world_selected",
                            "selection_visible":True,"play_enabled":False})
        self.assertTrue(result.usable)
        self.assertEqual(result.before.scene,"menu")
        self.assertEqual(result.before.menu_id,"main_menu")
        self.assertEqual(result.after.menu_state,"no_selection")

    def test_missing_selection_cues_cannot_claim_known_selected_state(self):
        result=self.assess({"scene":"menu","menu_id":"world_select","menu_state":"world_selected"})
        self.assertEqual(result.after.menu_state,"unknown_selection")

    def test_goal_menu_click_verification_uses_menu_critic_without_scene_type(self):
        payload={"before":{"scene":"menu","menu_id":"main_menu","menu_state":"default"},
                 "after":{"scene":"menu","menu_id":"world_select","menu_state":"no_selection",
                          "selection_visible":False,"play_enabled":False},
                 "confidence":.95,"menu_action_succeeded":False,"result":"blocked",
                 "evidence":"The world selection screen is visible.","uncertainty":""}
        response=Mock()
        response.json.return_value={"choices":[{"message":{"content":json.dumps(payload)}}]}
        frame=Image.new("RGB",(800,500))
        brain=VisionBrain();brain.active_profile=load_profile("minecraft")
        with patch("omnigamer.vision_brain.requests.post",return_value=response) as post:
            result=brain.assess(frame,frame,{"action_type":"click_menu_target",
                "expected_result":"World selection screen appears",
                "success_check":"World selection screen is visible"},"Goal step")
        self.assertFalse(result.menu_action_succeeded)
        request=post.call_args.kwargs["json"]
        self.assertIn("You verify MENU transitions",request["messages"][0]["content"])
        required=request["response_format"]["json_schema"]["schema"]["required"]
        self.assertIn("menu_action_succeeded",required)
