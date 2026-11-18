"""Actor intent must become a measured, checked target before input."""
import base64,json,io,tempfile,unittest
from pathlib import Path
from unittest.mock import Mock,patch
from PIL import Image,ImageDraw
from omnigamer.vision_brain import VisionBrain,BrainDecision
from omnigamer.ui_grounding import (button_candidates,world_selection_cues,stacked_main_menu_layout,
                                    has_profile_menu_layout)
from omnigamer.camera_feedback import camera_feedback
from profile_fixtures import load_profile
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
        payload=post.call_args.kwargs['json']
        self.assertIn('response_format',payload)
        first=payload['messages'][1]['content'][0]['image_url']['url']
        context=Image.open(io.BytesIO(base64.b64decode(first.split(',',1)[1])))
        self.assertEqual(context.size,(960,540))

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

    def test_measured_candidate_tolerates_bbox_edge_error_and_uses_geometry(self):
        boxes=[[425,190,701,213],[342,192,420,232],[340,405,719,470],
               [340,487,719,551],[340,568,719,633],[340,732,527,798],[531,732,718,798]]
        response,_=self.target(candidate_id=3,bbox=[250,350,750,420],
                               label='Одиночная игра',expected_result='world_select')
        with patch('omnigamer.vision_brain.button_candidates',return_value=boxes), \
             patch('omnigamer.vision_brain.requests.post',return_value=response):
            result=self.brain.ground_click(self.frame,BrainDecision(action_type='click',
                target_label='Одиночная игра',intent='Open the world list'),
                profile=load_profile('minecraft'))
        self.assertEqual((result.x,result.y),(530,438))
        self.assertEqual(result.target_label,'Одиночная игра')

    def test_dark_outlines_recover_textured_menu_buttons(self):
        rng=np.random.default_rng(5);w,h=960,526
        pixels=np.empty((h,w,3),dtype=np.uint8);pixels[:]=[70,110,150]
        texture=rng.integers(-28,29,(36,360,3))
        pixels[213:249,325:685]=np.clip(130+texture,0,255)
        frame=Image.fromarray(pixels)
        ImageDraw.Draw(frame).rectangle((325,213,684,248),outline=(0,0,0),width=2)
        boxes=button_candidates(frame)
        target=next((box for box in boxes if 390<box[1]<420),None)
        self.assertIsNotNone(target)
        left,top,right,bottom=target
        self.assertTrue(left<350<right and top<445<bottom)
        self.assertLess(bottom-top,100)

    def test_stacked_main_menu_geometry_confirms_unknown_vlm_scene(self):
        rng=np.random.default_rng(17);w,h=960,526
        pixels=np.empty((h,w,3),dtype=np.uint8);pixels[:]=[70,110,150]
        for top in (213,256,299):
            texture=rng.integers(-28,29,(36,360,3))
            pixels[top:top+36,325:685]=np.clip(130+texture,0,255)
        frame=Image.fromarray(pixels);draw=ImageDraw.Draw(frame)
        for top in (213,256,299):draw.rectangle((325,top,684,top+35),outline=(0,0,0),width=2)
        for box in ((325,385,504,420),(510,385,689,420)):
            draw.rectangle(box,fill=(125,125,125),outline=(0,0,0),width=2)
        boxes=button_candidates(frame)
        self.assertTrue(stacked_main_menu_layout(boxes))
        self.assertTrue(has_profile_menu_layout(frame,load_profile('minecraft')))
        target=next(box for box in boxes if 390<box[1]<420)
        response,_=self.target(scene_type='unknown',menu_id='unknown',candidate_id=boxes.index(target)+1,
            bbox=target,label='Одиночная игра',expected_result='world_select')
        with patch('omnigamer.vision_brain.requests.post',return_value=response) as post:
            result=self.brain.ground_click(frame,BrainDecision(action_type='click',
                target_label='Одиночная игра',intent='Open world list'),
                profile=load_profile('minecraft'))
        self.assertEqual(result.scene_type,'menu')
        self.assertEqual(result.menu_id,'main_menu')
        payload=post.call_args.kwargs['json']
        images=payload['messages'][1]['content']
        sizes=[Image.open(io.BytesIO(base64.b64decode(item['image_url']['url'].split(',',1)[1]))).size
               for item in images]
        self.assertEqual(sizes[0],(640,351))
        self.assertEqual(sizes[1][0],960)
        self.assertIn('bbox MUST describe that same numbered rectangle',payload['messages'][0]['content'])

    def test_stacked_main_menu_geometry_rejects_incomplete_layout(self):
        boxes=[[340,405,719,470],[340,487,719,551],[340,568,719,633],
               [340,732,527,798]]
        self.assertFalse(stacked_main_menu_layout(boxes))

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
        self.assertFalse(has_profile_menu_layout(self.frame,load_profile('minecraft')))

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
        self.assertTrue(has_profile_menu_layout(self.world_list(False),load_profile('minecraft')))
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

    def test_world_list_geometry_can_reuse_candidates_for_the_same_frame(self):
        from omnigamer.ui_grounding import first_world_row
        frame=self.world_list(False)
        boxes=button_candidates(frame)
        cues=world_selection_cues(frame,boxes)
        expected_row=first_world_row(frame,boxes,cues)
        with patch('omnigamer.ui_grounding.button_candidates',side_effect=AssertionError('recomputed')):
            self.assertEqual(cues,(False,False))
            self.assertEqual(first_world_row(frame,boxes,cues),expected_row)

    def test_explicit_world_creation_target_is_not_replaced_with_existing_row(self):
        frame=self.world_list(False)
        action=BrainDecision(action_type='click',target_label='Создать новый мир',
                             intent='Create a new world')
        self.assertIsNone(self.brain._world_list_action(
            frame,action,button_candidates(frame)))

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
        inspection={'facts':{'scene':'menu','menu_id':'world_select','menu_state':'world_selected',
                            'selection_visible':True,'play_enabled':True},'confidence':.95,'evidence':'A row outline is visible.'}
        response=Mock();response.json.return_value={'choices':[{'message':{'content':json.dumps(inspection)}}]}
        with patch('omnigamer.vision_brain.button_candidates',wraps=button_candidates) as candidates, \
             patch('omnigamer.vision_brain.requests.post',return_value=response) as post:
            result=self.brain.assess(before,after,action.model_dump(),'Free play',timeout_seconds=60)
        self.assertEqual(post.call_count,1)
        self.assertEqual(sum(call.args[0] is before for call in candidates.call_args_list),1)
        self.assertEqual(sum(call.args[0] is after for call in candidates.call_args_list),2)
        self.assertGreater(post.call_args.kwargs['timeout'],50)
        self.assertLessEqual(post.call_args.kwargs['timeout'],60)
        self.assertTrue(result.menu_action_succeeded)
        self.assertEqual(result.evidence_source,'screen_geometry')
        self.assertEqual(result.after.menu_state,'world_selected')
        self.assertEqual(len(post.call_args.kwargs['json']['messages'][1]['content']),1)

    def test_missing_click_coordinates_fall_back_to_visual_assessment(self):
        self.brain.active_profile=load_profile('minecraft')
        frame=self.world_list(False)
        action=self.brain._world_list_action(frame,BrainDecision(action_type='click'),button_candidates(frame))
        action=action.model_copy(update={'x':None,'y':None})
        payload={'before':{'scene':'menu','menu_id':'world_select','menu_state':'no_selection',
                           'selection_visible':False,'play_enabled':False},
                 'after':{'scene':'menu','menu_id':'world_select','menu_state':'no_selection',
                          'selection_visible':False,'play_enabled':False},
                 'menu_action_succeeded':False,'result':'blocked','confidence':.95,
                 'evidence':'The world row remains unselected.','uncertainty':''}
        response=Mock();response.json.return_value={'choices':[{'message':{'content':json.dumps(payload)}}]}
        with patch('omnigamer.vision_brain.requests.post',return_value=response) as post:
            result=self.brain.assess(frame,frame,action.model_dump(),'Free play')
        self.assertFalse(result.menu_action_succeeded)
        post.assert_called_once()
        self.assertIn('You verify MENU transitions',post.call_args.kwargs['json']['messages'][0]['content'])

    def test_unusable_independent_menu_inspection_cannot_preserve_progress(self):
        self.brain.active_profile=load_profile('minecraft')
        before,after=self.world_list(False),self.world_list(True)
        action=self.brain._world_list_action(before,BrainDecision(action_type='click',x=500,y=650),button_candidates(before))
        inspection={'facts':{'scene':'unknown','menu_id':'unknown','menu_state':'default',
                             'selection_visible':None,'play_enabled':None},
                    'confidence':.1,'evidence':'Unclear.'}
        response=Mock();response.json.return_value={'choices':[{'message':{'content':json.dumps(inspection)}}]}
        with patch('omnigamer.vision_brain.requests.post',return_value=response):
            result=self.brain.assess(before,after,action.model_dump(),'Free play')
        self.assertFalse(result.usable)
        self.assertEqual(result.result,'unknown')
        self.assertIsNone(result.menu_action_succeeded)


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
