"""Menu critic schema compatibility must not invent world selection."""
import json
import unittest
from unittest.mock import Mock,patch
from PIL import Image
from omnigamer.vision_brain import VisionBrain
from omnigamer.profiles import load_profile


class MenuCriticTests(unittest.TestCase):
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
