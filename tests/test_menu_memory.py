"""Remember a menu transition, reopen storage, and replay only supported targets."""
import tempfile
import unittest
from pathlib import Path
from PIL import Image,ImageDraw
from omnigamer.memory import GameMemory
from omnigamer.menu_memory import MenuMemory,menu_key
from omnigamer.perception import FrameFacts,VisualAssessment
from omnigamer.vision_brain import BrainDecision


def menu(background='navy',enabled=True,text='PLAY'):
    frame=Image.new('RGB',(800,500),background)
    draw=ImageDraw.Draw(frame)
    draw.rectangle((200,225,600,275),fill=(140,140,140) if enabled else (45,45,45))
    draw.text((370,244),text,fill='white' if enabled else (65,65,65))
    return frame


class MenuMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'memory.db'
        self.memory=GameMemory(self.path);self.menus=MenuMemory(self.memory)
        self.action=dict(action_type='click',x=500,y=500,duration=1.5,intent='Start playing',target_label='PLAY',expected_result='gameplay')
        self.source=FrameFacts(scene='menu',menu_id='main_menu',menu_state='default')

    def learn(self,source=None,target=None,action=None,game='g',success=True):
        source=source or self.source
        target=target or FrameFacts(scene='gameplay')
        first=VisualAssessment(before=source,after=target,result='progress',confidence=.95,
            menu_action_succeeded=success,evidence='The intended next screen appeared after the click.')
        confirm=VisualAssessment(before=target,after=target,result='unknown',confidence=.95,evidence='The next screen is still visible.')
        a=self.memory.record_visual_check(1,game,'s',1,'post','free_play',first.model_dump())
        b=self.memory.record_visual_check(2,game,'s',2,'menu_confirm','free_play',confirm.model_dump())
        return self.menus.remember(game,menu(),action or self.action,a,b)

    def observation(self,**kwargs):
        return BrainDecision(scene_type='menu',menu_id='main_menu',menu_state='default',visual_confidence=.95,**kwargs)

    def test_checked_action_is_recalled_after_restart_with_animated_background(self):
        self.learn()
        reopened=MenuMemory(GameMemory(self.path))
        recalled=reopened.recall('g',menu('darkred'),self.observation())
        self.assertEqual(recalled['action']['x'],500)
        self.assertEqual(recalled['destination'],'gameplay')

    def test_disabled_or_changed_target_is_not_replayed(self):
        self.learn()
        self.assertIsNone(self.menus.recall('g',menu(enabled=False),self.observation()))
        self.assertIsNone(self.menus.recall('g',menu(text='DELETE WORLD'),self.observation()))

    def test_other_profile_or_selection_state_is_not_the_same_menu(self):
        self.learn()
        self.assertIsNone(self.menus.recall('other',menu(),self.observation()))
        observed=self.observation().model_copy(update={'menu_state':'world_selected'})
        self.assertIsNone(self.menus.recall('g',menu(),observed))

    def test_screen_change_alone_cannot_teach_a_success(self):
        self.assertIsNone(self.learn(success=False))
        self.assertIsNone(self.menus.recall('g',menu(),self.observation()))

    def test_failed_replay_is_disabled_until_relearned(self):
        action_id=self.learn();self.menus.invalidate(action_id)
        self.assertIsNone(self.menus.recall('g',menu(),self.observation()))
        self.learn()
        self.assertIsNotNone(self.menus.recall('g',menu(),self.observation()))

    def test_route_is_remembered_but_only_first_action_is_recalled(self):
        selected=FrameFacts(scene='menu',menu_id='world_select',menu_state='world_selected')
        self.learn(target=selected)
        self.learn(source=selected)
        recalled=self.menus.recall('g',menu(),self.observation())
        self.assertEqual(recalled['route'],['menu:world_select:world_selected','gameplay'])
        self.assertEqual(recalled['action']['action_type'],'click')

    def test_uncertain_menu_identity_does_not_trigger_replay(self):
        self.learn()
        self.assertIsNone(self.menus.recall('g',menu(),self.observation().model_copy(update={'visual_confidence':.5})))

    def test_selection_is_part_of_state_identity(self):
        self.assertNotEqual(menu_key('menu','world_select','no_selection'),menu_key('menu','world_select','world_selected'))

    def test_uncertain_or_conflicting_evidence_is_declined_without_crashing(self):
        first=VisualAssessment(before=self.source,after=FrameFacts(scene='gameplay'),confidence=.5,
            evidence='The destination is unclear.',menu_action_succeeded=True)
        confirm=VisualAssessment(after=FrameFacts(scene='gameplay'),confidence=.95,evidence='Gameplay is visible.')
        a=self.memory.record_visual_check(1,'g','s',1,'post','free_play',first.model_dump())
        b=self.memory.record_visual_check(2,'g','s',2,'menu_confirm','free_play',confirm.model_dump())
        self.assertIsNone(self.menus.remember('g',menu(),self.action,a,b))


if __name__=='__main__':unittest.main()
