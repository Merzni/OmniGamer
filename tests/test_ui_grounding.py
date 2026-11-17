"""Actor intent must become a measured, checked target before input."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import Mock,patch
from PIL import Image,ImageDraw
from omnigamer.vision_brain import VisionBrain,BrainDecision
from omnigamer.ui_grounding import button_candidates,world_selection_cues
from omnigamer.camera_feedback import camera_feedback
from omnigamer.profiles import load_profile
import numpy as np


class GroundingTests(unittest.TestCase):
    def setUp(self):
        self.frame=Image.new('RGB',(960,540),'navy')
        ImageDraw.Draw(self.frame).rectangle((300,200,660,240),fill=(130,130,130))
        self.brain=VisionBrain()

    def target(self,**changes):
        boxes=button_candidates(self.frame)
        self.assertTrue(boxes)
        payload=dict(scene_type='menu',menu_id='main_menu',menu_state='default',candidate_id=1,
            bbox=boxes[0],label='PLAY',expected_result='gameplay',enabled=True,confidence=.95)
        payload.update(changes)
        response=Mock();response.json.return_value={'choices':[{'message':{'content':json.dumps(payload)}}]}
        return response,boxes

    def test_actor_miss_is_replaced_by_measured_button_center(self):
        response,boxes=self.target()
        with patch('omnigamer.vision_brain.requests.post',return_value=response) as post:
            decision=self.brain.ground_click(self.frame,BrainDecision(action_type='click',x=10,y=900,target_label='PLAY'))
        x,y,z,b=boxes[0]
        self.assertEqual((decision.x,decision.y),(round((x+z)/2),round((y+b)/2)))
        self.assertEqual(decision.scene_type,'menu')
        self.assertEqual(decision.expected_result,'gameplay')
        self.assertIn('response_format',post.call_args.kwargs['json'])

    def test_disabled_or_unlocalized_control_is_never_clicked(self):
        for changes in ({'enabled':False},{'confidence':.5},{'candidate_id':0,'bbox':[0,0,1000,1000]},
                        {'menu_id':'unknown'}):
            response,_=self.target(**changes)
            with patch('omnigamer.vision_brain.requests.post',return_value=response),self.assertRaises(ValueError):
                self.brain.ground_click(self.frame,BrainDecision(action_type='click',x=500,y=500))

    def test_candidate_id_must_match_localizer_bbox(self):
        response,_=self.target(bbox=[700,700,900,850])
        with patch('omnigamer.vision_brain.requests.post',return_value=response), \
             self.assertRaisesRegex(ValueError,'does not match returned bbox'):
            self.brain.ground_click(self.frame,BrainDecision(action_type='click',target_label='PLAY'))

    def test_unreadable_localized_label_is_never_clicked(self):
        response,_=self.target(label='\ufffd\ufffd\ufffd')
        with patch('omnigamer.vision_brain.requests.post',return_value=response), \
             self.assertRaisesRegex(ValueError,'label is unreadable'):
            self.brain.ground_click(self.frame,BrainDecision(action_type='click',target_label='PLAY'))

    def test_localizer_rejection_reports_the_confidence_gate(self):
        response,_=self.target(confidence=.84)
        with patch('omnigamer.vision_brain.requests.post',return_value=response), \
             self.assertRaisesRegex(ValueError,'confidence 0.84 is below 0.85'):
            self.brain.ground_click(self.frame,BrainDecision(action_type='click',target_label='PLAY'))

    def test_missing_schema_fields_and_truncated_action_do_not_execute(self):
        with tempfile.TemporaryDirectory() as td:
            self.brain.log_file=Path(td)/'raw.log'
            for content in ('{"action_type":"jump"}','{"action_type":"click","x":500,'):
                response=Mock();response.json.return_value={'choices':[{'message':{'content':content}}]}
                with patch('omnigamer.vision_brain.requests.post',return_value=response),patch('omnigamer.vision_brain.Path.write_text'):
                    result=self.brain.decide(self.frame)
                self.assertTrue(self.brain.last_error)
                self.assertEqual(result.action_type,'wait')

    def test_unknown_layout_cannot_prove_world_selection(self):
        self.assertEqual(world_selection_cues(self.frame),(None,None))

    @staticmethod
    def world_list(selected):
        frame=Image.new('RGB',(960,481),(90,45,20));draw=ImageDraw.Draw(frame)
        for box in ((215,395,476,425),(484,395,744,425),(215,436,341,466),(350,436,476,466),
                    (484,436,610,466),(619,436,744,466)):
            draw.rectangle(box,fill=(110,110,110),outline='black',width=2)
        draw.rectangle((215,395,476,425),fill=(110,110,110) if selected else (44,44,44),outline='black',width=2)
        draw.rectangle((260,407,430,418),fill='white' if selected else (90,90,90))
        for y,color in ((95,'white'),(112,(100,100,100)),(128,(100,100,100))):
            draw.rectangle((320,y,500,y+7),fill=color)
        if selected:draw.rectangle((250,88,710,150),outline='white',width=2)
        return frame

    def test_existing_world_is_selected_then_played_from_measured_layout(self):
        action=BrainDecision(action_type='click',x=1,y=999,intent='Start playing')
        with patch('omnigamer.vision_brain.requests.post') as post:
            row=self.brain.ground_click(self.world_list(False),action,profile=load_profile('minecraft'))
            play=self.brain.ground_click(self.world_list(True),action,profile=load_profile('minecraft'))
        post.assert_not_called()
        self.assertEqual(row.menu_state,'no_selection')
        self.assertEqual(row.expected_result,'world_selected')
        self.assertLess(row.y,400)
        self.assertEqual(play.menu_state,'world_selected')
        self.assertTrue(790<play.y<900)
        self.assertEqual(play.expected_result,'gameplay')

    def test_world_list_geometry_adapter_is_disabled_without_profile_opt_in(self):
        response,_=self.target(menu_id='world_select',menu_state='unknown_selection')
        profile=load_profile('example-game')
        with patch('omnigamer.vision_brain.requests.post',return_value=response) as post, \
             self.assertRaises(ValueError):
            self.brain.ground_click(self.world_list(True),BrainDecision(action_type='click',
                x=500,y=650,intent='Start playing'),profile=profile)
        post.assert_called_once()

    def test_row_geometry_proof_is_labeled_and_still_rechecked(self):
        self.brain.active_profile=load_profile('minecraft')
        before,after=self.world_list(False),self.world_list(True)
        action=self.brain._world_list_action(before,BrainDecision(action_type='click',x=500,y=650),button_candidates(before))
        self.assertIsNotNone(action)
        initial={'before':{'scene':'menu','menu_id':'world_select'},'after':{'scene':'menu','menu_id':'world_select'},
                 'menu_action_succeeded':False,'confidence':.8,'evidence':'Model missed the selection change.'}
        inspection={'facts':{'scene':'menu','menu_id':'world_select','menu_state':'world_selected',
                            'selection_visible':True,'play_enabled':True},'confidence':.95,'evidence':'A row outline is visible.'}
        responses=[]
        for payload in (initial,inspection):
            response=Mock();response.json.return_value={'choices':[{'message':{'content':json.dumps(payload)}}]};responses.append(response)
        with patch('omnigamer.vision_brain.requests.post',side_effect=responses) as post:
            result=self.brain.assess(before,after,action.model_dump(),'Free play')
        self.assertEqual(post.call_count,2)
        self.assertTrue(result.menu_action_succeeded)
        self.assertEqual(result.evidence_source,'screen_geometry')
        self.assertEqual(result.after.menu_state,'world_selected')


class CameraFeedbackTests(unittest.TestCase):
    def test_scene_shift_is_distinguished_from_ui_animation(self):
        rng=np.random.default_rng(2)
        # Coherent scene features survive downsampling, unlike sensor noise.
        image=Image.fromarray(rng.integers(0,255,(54,96,3),dtype=np.uint8)).resize((960,540),Image.Resampling.NEAREST)
        self.assertEqual(camera_feedback(image,image)['status'],'unconfirmed')
        moved=Image.fromarray(np.roll(np.asarray(image),30,axis=1))
        self.assertEqual(camera_feedback(image,moved)['status'],'shift_observed')
        ui=image.copy();ImageDraw.Draw(ui).rectangle((0,450,960,540),fill='white')
        self.assertEqual(camera_feedback(image,ui)['status'],'unconfirmed')
