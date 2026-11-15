"""Goal/camera/evidence regressions; all databases and inputs are isolated."""
import tempfile
import unittest
from pathlib import Path
from omnigamer.perception import FrameFacts, VisibleItem, VisualAssessment
from omnigamer.supervision import CameraGuard, GoalSupervisor, PlaySupervisor
from omnigamer.memory import GameMemory


def checked(**after):
    return VisualAssessment(before=FrameFacts(scene="gameplay", pitch="down"),
                            after=FrameFacts(**after), confidence=.95,
                            evidence="The after frame shows the specified visible state.", result="progress")


class CameraGuardTests(unittest.TestCase):
    def test_digging_down_is_replaced_by_upward_reorientation(self):
        action, reason = CameraGuard().filter(dict(action_type="mine"),
            FrameFacts(scene="gameplay",pitch="down"), grounded_confident=True)
        self.assertEqual((action["action_type"],action["y"]), ("look",-50))
        self.assertTrue(reason)

    def test_unknown_view_or_support_cannot_authorize_mining(self):
        for pitch, footing in (("unknown",True),("level",False)):
            with self.subTest(pitch=pitch):
                action, _ = CameraGuard().filter(dict(action_type="mine"),
                    FrameFacts(scene="gameplay",pitch=pitch), grounded_confident=footing)
                self.assertEqual(action["action_type"], "wait")

    def test_fresh_level_view_and_support_allow_forward_mining(self):
        action, _ = CameraGuard().filter(dict(action_type="mine"),
            FrameFacts(scene="gameplay",pitch="level"), grounded_confident=True)
        self.assertEqual(action["action_type"], "mine")

    def test_extreme_vertical_command_is_bounded(self):
        guard = CameraGuard()
        total = 0
        for _ in range(10):
            action, _ = guard.filter(dict(action_type="look", look_direction="down", look_amount="large"))
            self.assertLessEqual(abs(action["y"]), 60)
            total += action["y"]
            guard.executed(action)
        self.assertEqual(total, 180)

    def test_fresh_visual_correction_can_continue_until_horizon_is_found(self):
        guard = CameraGuard()
        for _ in range(8):
            action, _ = guard.filter(dict(action_type="walk",key="w"),
                FrameFacts(scene="gameplay",pitch="up"))
            self.assertEqual(action["y"], 50)
            guard.executed(action)
        guard.observe(checked(scene="gameplay",pitch="level"))
        self.assertEqual(guard.vertical_offset, 0)

    def test_invalid_look_is_replaced_by_safe_wait_without_alternation(self):
        guard = CameraGuard()
        first, _ = guard.filter(dict(action_type="look",x=0,y=0))
        second, _ = guard.filter(dict(action_type="look",x=0,y=0))
        self.assertEqual((first["action_type"], second["action_type"]), ("wait", "wait"))
        self.assertIsNone(first["look_direction"])


class StateRepeatDetectorTests(unittest.TestCase):
    def test_recent_hash_repeat_and_abab_look_oscillation_are_reported(self):
        from omnigamer.supervision import StateRepeatDetector
        detector = StateRepeatDetector(window=8)
        for step, direction in enumerate(("left", "right", "left", "right"), 1):
            detector.remember("0123456789abcdef", step, {
                "action_type": "look", "look_direction": direction,
            })
        detected = detector.detect("0123456789abcdef", 5)
        self.assertEqual(detected, {"state_repeat": True, "repeat_of_step": 4, "look_oscillation": True})

    def test_menu_navigation_is_not_turned_into_camera_motion(self):
        action, _ = CameraGuard().filter(dict(action_type="click",x=500,y=400),
            FrameFacts(scene="menu",pitch="up"))
        self.assertEqual(action["action_type"], "click")


class GoalTests(unittest.TestCase):
    def test_mixed_actions_cannot_reset_global_no_progress_budget(self):
        goal = GoalSupervisor(max_steps=20)
        for step in range(1,21):
            goal.observe(None,.5,step)
        self.assertTrue(goal.halted)
        self.assertEqual(goal.no_progress_steps,20)

    def test_active_time_budget_is_independent_of_step_count(self):
        goal = GoalSupervisor(max_seconds=90)
        goal.observe(None,45,1)
        goal.observe(None,46,2)
        self.assertTrue(goal.halted)

    def test_completion_needs_two_distinct_confident_post_checks(self):
        goal = GoalSupervisor()
        goal.set_goal("restore_view")
        assessment = checked(scene="gameplay",pitch="level")
        self.assertFalse(goal.observe(assessment,1,1,10)["completed"])
        self.assertFalse(goal.observe(assessment,1,1,10)["completed"])
        event=goal.observe(assessment,1,2,11)
        self.assertTrue(event["completed"])
        self.assertEqual(event["first_evidence"]["check_id"],10)
        self.assertEqual(goal.goal_id,"done")

    def test_unknown_low_confidence_or_regression_breaks_confirmation(self):
        for bad in (VisualAssessment(), checked(scene="gameplay",pitch="down"),
                    checked(scene="gameplay",pitch="level").model_copy(update={"confidence":.5}),
                    checked(scene="gameplay",pitch="level").model_copy(update={"result":"regressed"})):
            goal=GoalSupervisor();goal.set_goal("restore_view")
            goal.observe(checked(scene="gameplay",pitch="level"),1,1)
            goal.observe(bad,1,2)
            self.assertFalse(goal.observe(checked(scene="gameplay",pitch="level"),1,3)["completed"])

    def test_completed_goal_advances_plan_and_resets_budget(self):
        goal=GoalSupervisor()
        goal.observe(checked(scene="gameplay",pitch="down"),1,1)
        event=goal.observe(checked(scene="gameplay",pitch="down"),1,2)
        self.assertTrue(event["completed"])
        self.assertEqual(goal.goal_id,"restore_view")
        self.assertEqual(goal.no_progress_steps,0)

    def test_resource_goal_requires_visible_count_increase_not_narration(self):
        goal=GoalSupervisor();goal.set_goal("collect_resource","log")
        before=FrameFacts(scene="gameplay",visible_items_readable=True,items=[VisibleItem(slot=1,name="dirt",count=18)])
        no_gain=VisualAssessment(before=before,after=before,confidence=.95,evidence="Dirt is visible, but no wood was acquired.",result="progress")
        goal.observe(no_gain,1,1)
        self.assertEqual(goal.confirmations,0)
        after=FrameFacts(scene="gameplay",visible_items_readable=True,items=[VisibleItem(slot=2,name="oak_log",count=1)])
        gained=VisualAssessment(before=before,after=after,confidence=.95,evidence="One oak log appears in the second visible slot.",result="progress")
        self.assertFalse(goal.observe(gained,1,2)["completed"])
        self.assertTrue(goal.observe(gained,1,3)["completed"])

    def test_unreadable_hotbar_does_not_become_zero_items(self):
        facts=FrameFacts(visible_items_readable=False,items=[])
        self.assertIsNone(facts.item_count("log"))
        self.assertIsNone(FrameFacts(visible_items_readable=True,items=[VisibleItem(slot=1,name="log",count=1),
            VisibleItem(slot=1,name="dirt",count=2)]).item_count("log"))

    def test_open_ground_goal_also_requires_visible_support(self):
        goal=GoalSupervisor();goal.set_goal("reach_open_ground")
        unsafe=checked(scene="gameplay",pitch="level",open_ground=True,safe_footing=None)
        goal.observe(unsafe,1,1);goal.observe(unsafe,1,2)
        self.assertEqual(goal.confirmations,0)

    def test_collecting_dirt_by_falling_into_hole_is_not_a_completed_goal(self):
        goal=GoalSupervisor();goal.set_goal("collect_resource","dirt")
        before=FrameFacts(scene="gameplay",visible_items_readable=True,items=[VisibleItem(slot=1,name="dirt",count=12)])
        after=FrameFacts(scene="gameplay",pitch="down",visible_items_readable=True,items=[VisibleItem(slot=1,name="dirt",count=14)])
        regression=VisualAssessment(before=before,after=after,result="regressed",confidence=.95,
            evidence="Mining under the feet increased dirt but dropped the player into a hole.")
        goal.observe(regression,1,1);goal.observe(regression,1,2)
        self.assertEqual(goal.confirmations,0)
        self.assertFalse(goal.completed)

    def test_restart_preserves_halt_but_does_not_supply_second_confirmation(self):
        goal=GoalSupervisor(max_steps=4)
        for step in range(4):goal.observe(None,1,step)
        self.assertTrue(GoalSupervisor(goal.snapshot()).halted)
        goal.set_goal("restore_view")
        goal.observe(checked(scene="gameplay",pitch="level"),1,1)
        restored=GoalSupervisor(goal.snapshot())
        self.assertFalse(restored.observe(checked(scene="gameplay",pitch="level"),1,2)["completed"])


class EvidenceMemoryTests(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path=Path(self.folder.name)/"memory.db"
        self.memory=GameMemory(self.path)

    def test_goal_and_budget_survive_database_reopen_and_stay_game_scoped(self):
        goal=GoalSupervisor();goal.observe(None,5,1)
        self.memory.save_supervisor("minecraft",goal.snapshot())
        restored=GameMemory(self.path).load_supervisor("minecraft")
        self.assertEqual(restored["no_progress_steps"],1)
        self.assertIsNone(self.memory.load_supervisor("other-game"))

    def test_verified_goal_has_two_traceable_checks_and_not_raw_actor_story(self):
        goal=GoalSupervisor();goal.set_goal("restore_view")
        assessment=checked(scene="gameplay",pitch="level")
        first=self.memory.record_visual_check(1,"game","session",1,"post","restore_view",assessment.model_dump())
        goal.observe(assessment,1,1,first)
        second=self.memory.record_visual_check(2,"game","session",2,"post","restore_view",assessment.model_dump())
        event=goal.observe(assessment,1,2,second)
        self.memory.record_goal_lesson(2,"game","restore_view",first,second,event)
        self.assertEqual(self.memory.get_stats("game")["visually_confirmed_goals"],1)
        self.assertIn("Historical visual state: restore_view",self.memory.get_learning_context("0000000000000000","game"))
        with self.assertRaises(ValueError):
            self.memory.record_goal_lesson(2,"other-game","restore_view",first,second,event)
        with self.assertRaises(ValueError):
            self.memory.record_goal_lesson(2,"game","restore_view",first,first,event)

    def test_actor_hallucinations_are_preserved_for_audit_but_not_retrieved_as_facts(self):
        self.memory.record_experience(1,"0000000000000000","walk",None,None,"w",.5,True,
            "FAKE_PICKAXE", "FAKE_EXIT",game_id="game",observation="FAKE_ZOMBIE")
        context=self.memory.get_learning_context("0000000000000000","game")
        self.assertNotIn("FAKE_",context)
        self.assertIn("changed=1/1",context)
        self.assertIn("not goal success",context)

    def test_low_confidence_assessment_cannot_become_confirmed_lesson(self):
        weak=checked(scene="gameplay",pitch="level").model_copy(update={"confidence":.5})
        first=self.memory.record_visual_check(1,"g","s",1,"post","restore_view",weak.model_dump())
        second=self.memory.record_visual_check(2,"g","s",2,"post","restore_view",weak.model_dump())
        with self.assertRaises(ValueError):
            self.memory.record_goal_lesson(2,"g","restore_view",first,second,
                {"goal_id":"restore_view","status":"visual_confirmed","completed":True})


class ContinuousPlayTests(unittest.TestCase):
    def test_old_completed_or_halted_goals_do_not_stop_continuous_play(self):
        for state in ({"plan":[],"index":4,"halted":True,"reason":"done"},
                      {"plan":["collect_resource"],"no_progress_steps":20,"halted":True}):
            play=PlaySupervisor(state)
            self.assertFalse(play.halted)
            self.assertEqual(play.mode,"exploring")
            self.assertNotIn("plan",play.snapshot())

    def test_many_recovery_cycles_keep_playing_without_external_tasks(self):
        play=PlaySupervisor(max_steps=4,max_seconds=10)
        for step in range(1,201):
            event=play.observe(None,1,step,step)
            self.assertFalse(play.halted)
            self.assertFalse(event["completed"])
        self.assertGreater(play.recovery_count,5)

    def test_checked_achievement_does_not_become_a_terminal_state(self):
        play=PlaySupervisor()
        play.observe(checked(scene="gameplay",pitch="level"),1,1,10)
        event=play.observe(checked(scene="gameplay",pitch="level"),1,2,11)
        self.assertTrue(event["verified_change"])
        self.assertFalse(play.halted)
        self.assertFalse(event["completed"])
        self.assertEqual(play.verified_changes,1)
        self.assertIn("After an ending or achievement, keep playing",play.context())

    def test_unchanged_forward_view_is_not_repeatedly_rewarded(self):
        play=PlaySupervisor()
        same=VisualAssessment(before=FrameFacts(scene="gameplay",pitch="level"),
            after=FrameFacts(scene="gameplay",pitch="level"),confidence=.95,evidence="The same level view is visible.",result="progress")
        for step in range(10):play.observe(same,1,step,step+1)
        self.assertEqual(play.verified_changes,0)

    def test_regression_never_becomes_positive_resource_learning(self):
        play=PlaySupervisor()
        bad=VisualAssessment(before=FrameFacts(visible_items_readable=True,items=[]),
            after=FrameFacts(visible_items_readable=True,items=[VisibleItem(slot=1,name="dirt",count=2)]),
            confidence=.95,result="regressed",evidence="Dirt increased while falling into a hole.")
        for step in range(3):play.observe(bad,1,step,step+1)
        self.assertEqual(play.verified_changes,0)

    def test_recovery_backtracking_requires_known_support_and_pose(self):
        play=PlaySupervisor(max_steps=4)
        for step in range(4):play.observe(None,1,step,step+1)
        proposal=dict(action_type="walk",key="w",duration=.6)
        action,_=play.recovery_action(proposal,FrameFacts(scene="gameplay",pitch="level",safe_footing=True))
        self.assertEqual(action["action_type"],"look")
        play.observe(None,1,5,6)
        action,_=play.recovery_action(proposal,FrameFacts(scene="gameplay",pitch="level",safe_footing=True))
        self.assertEqual(action["key"],"s")
        unsafe,_=play.recovery_action(proposal,FrameFacts(scene="gameplay",pitch="unknown"))
        self.assertEqual(unsafe["action_type"],"look")

    def test_recovery_state_persists_without_resurrecting_a_stop_flag(self):
        play=PlaySupervisor(max_steps=4)
        for step in range(4):play.observe(None,1,step,step+1)
        resumed=PlaySupervisor(play.snapshot())
        self.assertEqual(resumed.mode,"recovering")
        self.assertEqual(resumed.recovery_remaining,4)
        self.assertFalse(resumed.halted)

    def test_new_outcome_keeps_two_traceable_checks_without_goal_assignment(self):
        with tempfile.TemporaryDirectory() as folder:
            memory=GameMemory(Path(folder)/"memory.db")
            play=PlaySupervisor()
            assessment=checked(scene="gameplay",pitch="level")
            first=memory.record_visual_check(1,"g","s",1,"post","free_play",assessment.model_dump())
            play.observe(assessment,1,1,first)
            second=memory.record_visual_check(2,"g","s",2,"post","free_play",assessment.model_dump())
            event=play.observe(assessment,1,2,second)
            memory.record_play_lesson(2,"g",event)
            self.assertEqual(memory.get_stats("g")["checked_play_outcomes"],1)
            self.assertIn("Previously checked play outcome",memory.get_learning_context("0000000000000000","g"))
            with self.assertRaises(ValueError):memory.record_play_lesson(2,"another-game",event)


if __name__=="__main__":unittest.main()
