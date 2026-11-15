"""Core autonomous game loop: Observe -> Reason -> Act -> Evaluate -> Learn."""
import logging
import sys
import time
from threading import Event, RLock, Thread
from uuid import uuid4
from typing import Optional, Dict
from PIL import Image

from .config import config
from .capture import WindowCapture
from .controls import HumanInputController
from .vision_brain import VisionBrain, BrainDecision, VisionDecisionError
from .memory import GameMemory
from .personality import StreamerPersona
from .perception import FrameFacts, VisualAssessment
from .supervision import CameraGuard, PlaySupervisor, StateRepeatDetector
from .pointer import BotPointer
from .window_input import WindowInputController
from .menu_memory import MenuMemory, facts_key
from .camera_feedback import camera_feedback
from .emergency import EmergencyHotkey
from .telemetry import RunTelemetry, StepTrace

class OmniAgent:
    """Autonomous universal gamer agent."""

    def __init__(
        self,
        hwnd: int = 0,
        vlm_url: str = config.vlm_api_url,
        db_path = config.db_path,
        game_id: str = "",
        input_backend: str = None,
        emergency_hotkey_enabled: bool = True,
        telemetry_enabled: Optional[bool] = None,
        goal: str = "",
        profile=None,
    ):
        self.capture = WindowCapture(hwnd=hwnd)
        self.pointer = BotPointer(desktop=config.desktop_bot_pointer)
        if profile is None and game_id:
            from .profiles import load_profile
            profile = load_profile(game_id)
        self.profile = profile
        self.input_backend = input_backend or getattr(profile, "input_backend", None) or config.input_backend
        self._build_controls()
        self.brain = VisionBrain(api_url=vlm_url, model_name=config.vlm_model_name,
                                 image_max_size=config.image_max_size, max_tokens=config.vlm_max_tokens,
                                 timeout=config.request_timeout)
        self.brain.active_profile = self.profile
        self.memory = GameMemory(db_path=db_path)
        self.menu_memory = MenuMemory(self.memory)
        self._pending_menu = None
        self.last_menu_memory = {}
        self.persona = StreamerPersona(name=config.persona_name)
        self._stop_event = Event()
        self.emergency_hotkey = (EmergencyHotkey(config.emergency_key, self.request_stop)
                                 if emergency_hotkey_enabled else None)
        self.last_emergency_release_error = ""
        self.telemetry = RunTelemetry(
            config.data_dir,
            enabled=config.telemetry_enabled if telemetry_enabled is None else telemetry_enabled,
            frame_every_n=config.telemetry_frame_every_n,
            resource_interval_s=config.telemetry_resource_interval_s,
            frames_max_mb=config.telemetry_frames_max_mb,
            runs_max_count=config.telemetry_runs_max_count,
            runs_max_mb=config.telemetry_runs_max_mb,
        )
        self._telemetry_trace = None
        self._telemetry_before_frame = None
        self._telemetry_after_frame = None
        self._telemetry_no_effect_streak = 0
        self._telemetry_camera_only_streak = 0
        
        self.is_running = False
        self.step_count = 0
        self.last_frame: Optional[Image.Image] = None
        self.last_decision: Optional[BrainDecision] = None
        self.last_delta: float = 0.0
        self.session_id = uuid4().hex
        self._step_lock = RLock()
        self.game_id = self._resolve_game_id(game_id)
        self.goal = str(goal or "").strip()[:500]
        self.recent_history = self.memory.get_recent_experiences(self.game_id)
        self.state_repeat_detector = StateRepeatDetector(config.state_repeat_window)
        self._last_repeat_recovery_key = None
        self.last_memory_context = ""
        self.last_memory_warning = ""
        self.last_error = ""
        self.last_timings = {}
        self.last_action_source = "vlm"
        self.last_camera_feedback = {}
        self._load_supervision()

    def _build_controls(self):
        if self.input_backend == "window":
            self.controls = WindowInputController(lambda:self.capture.hwnd,self.capture.get_client_rect,self.pointer,
                failsafe_corner=config.failsafe_corner,target_check=self.capture.assert_visible)
        elif self.input_backend == "native":
            self.controls = HumanInputController(failsafe_corner=config.failsafe_corner,
                                                target_check=self.capture.assert_foreground)
        else:
            raise ValueError("Input backend must be window or native")
        self.controls.emergency_check = self._check_stop

    def start_emergency_hotkey(self):
        if self.emergency_hotkey:
            self.emergency_hotkey.start()

    def _check_stop(self):
        if self._stop_event.is_set():
            raise InterruptedError(f"Emergency stop requested ({config.emergency_key})")

    def _check_profile_forbidden_key(self, decision):
        if not self.profile:
            return
        forbidden = {str(key).strip().casefold() for key in self.profile.forbidden_keys}
        requested = {str(decision.key).strip().casefold()} if decision.key else set()
        if decision.action_type in {"walk", "jump"}:
            requested.add(str(decision.key or self.profile.move_keys["forward"]).strip().casefold())
        if decision.action_type == "jump":
            jump = self.profile.actions.get("jump")
            requested.add(jump.value if jump and jump.kind == "key" else "space")
        action_name = {"mine": "primary_hold", "use": "interact"}.get(decision.action_type)
        configured = self.profile.actions.get(action_name) if action_name else None
        if configured and configured.kind == "key":
            requested.add(configured.value)
        blocked = requested & forbidden
        if blocked:
            raise ValueError(f"Profile forbids requested key: {sorted(blocked)[0]}")

    def _send_profile_action(self, name):
        action = self.profile.actions.get(name) if self.profile else None
        if action is None:
            raise ValueError(f"Profile action is not configured: {name}")
        if action.kind == "mouse":
            self.controls.hold_mouse(button=action.value, duration=action.duration)
        else:
            self.controls.press_key(action.value, duration=action.duration)
        return action

    def _apply_profile_motion(self, decision):
        if not self.profile or decision.action_type not in {"walk", "jump"}:
            return decision
        key = decision.key or self.profile.move_keys["forward"]
        if key.casefold() not in set(self.profile.move_keys.values()):
            raise ValueError(f"Profile does not permit movement key: {key}")
        max_duration = min(config.max_motion_seconds, self.profile.motion_limits.max_action_seconds)
        if max_duration < 0.5:
            raise ValueError("Profile max_action_seconds is too short for the legacy movement contract")
        duration = max(0.5, min(decision.duration, max_duration))
        if decision.action_type == "jump":
            jump = self.profile.actions.get("jump")
            if jump is None or jump.kind != "key":
                raise ValueError("Profile jump action must be configured as a key")
            return decision.model_copy(update={"key": key, "duration": min(duration, jump.duration)})
        return decision.model_copy(update={"key": key, "duration": duration})

    def request_stop(self):
        """Latch the stop flag and immediately release bot-held inputs."""
        self._stop_event.set()
        self.is_running = False
        try:
            errors = self.controls.release_all() if hasattr(self, "controls") else []
            self.last_emergency_release_error = "; ".join(str(error) for error in errors)
        except Exception as exc:
            self.last_emergency_release_error = f"{type(exc).__name__}: {exc}"

    def _call_interruptibly(self, function, *args, **kwargs):
        """Run a blocking model request without making the stop wait for its timeout."""
        self._check_stop()
        result = {}
        completed = Event()

        def invoke():
            try:
                result["value"] = function(*args, **kwargs)
            except BaseException as exc:
                result["error"] = exc
            finally:
                completed.set()

        Thread(target=invoke, name="OmniGamerVLMRequest", daemon=True).start()
        while not completed.wait(0.02):
            self._check_stop()
        self._check_stop()
        if "error" in result:
            raise result["error"]
        return result.get("value")

    def begin_telemetry_run(self):
        try:
            title = self.capture.get_window_title()
        except Exception:
            title = None
        metadata = {
            "window_title": title,
            "hwnd": self.capture.hwnd or None,
            "game_id": self.game_id,
            "goal": self.goal,
            "input_backend": self.input_backend,
            "vlm_model_alias": config.vlm_model_name,
            "vlm_api_url": config.vlm_api_url,
            "config": {
                "telemetry_enabled": config.telemetry_enabled,
                "action_delay_seconds": config.action_delay_seconds,
                "max_motion_seconds": config.max_motion_seconds,
                "verification_interval": config.verification_interval,
                "recovery_after_steps": config.recovery_after_steps,
                "recovery_after_seconds": config.recovery_after_seconds,
                "state_repeat_detection_enabled": config.state_repeat_detection_enabled,
                "state_repeat_window": config.state_repeat_window,
                "state_repeat_triggers_recovery": config.state_repeat_triggers_recovery,
                "assess_look_with_vlm": config.assess_look_with_vlm,
                "image_max_size": config.image_max_size,
                "vlm_max_tokens": config.vlm_max_tokens,
                "request_timeout": config.request_timeout,
                "change_threshold": config.change_threshold,
                "memory_hash_distance": config.memory_hash_distance,
                "telemetry_frame_every_n": config.telemetry_frame_every_n,
                "telemetry_resource_interval_s": config.telemetry_resource_interval_s,
                "telemetry_frames_max_mb": config.telemetry_frames_max_mb,
                "telemetry_runs_max_count": config.telemetry_runs_max_count,
                "telemetry_runs_max_mb": config.telemetry_runs_max_mb,
            },
        }
        was_active = not self.telemetry.closed
        self.telemetry.start(metadata)
        started = not was_active and not self.telemetry.closed
        if started:
            self._telemetry_no_effect_streak = 0
            self._telemetry_camera_only_streak = 0
        return started

    def end_telemetry_run(self, reason, error_message=None):
        if reason == "error" and hasattr(self, "controls"):
            try:
                self.controls.release_all()
            except Exception as exc:
                logging.getLogger(__name__).warning("Input release during error finalization failed (%s)",
                                                    type(exc).__name__)
        self.telemetry.finish(reason, error_message=error_message)

    def _trace_start(self, stage):
        if self._telemetry_trace:
            self._telemetry_trace.start(stage)

    def _trace_stop(self, stage):
        if self._telemetry_trace:
            self._telemetry_trace.stop(stage)

    def set_input_backend(self, backend):
        if backend not in {"window","native"}:raise ValueError("Unknown input backend")
        with self._step_lock:
            self.pointer.hide()
            self.input_backend = backend
            self._build_controls()

    def prepare_target(self):
        if self.input_backend == "window":self.capture.assert_visible()
        else:self.capture.focus()

    def check_target(self):
        if self.input_backend == "window":self.capture.assert_visible()
        else:self.capture.assert_foreground()

    def _load_supervision(self):
        self.last_camera_feedback = {}
        self.supervisor = PlaySupervisor(
            self.memory.load_supervisor(self.game_id),
            max_steps=config.recovery_after_steps, max_seconds=config.recovery_after_seconds,
        )
        self.camera_guard = CameraGuard()
        self.last_assessment = None
        self.last_assessment_error = ""
        self._verified_frame = None
        self._verified_facts = None
        self.last_guard_reason = ""

    def supervision_state(self):
        state = self.supervisor.snapshot()
        state.update(context=self.supervisor.context(),
                     max_no_progress_steps=self.supervisor.max_steps,
                     max_no_progress_seconds=self.supervisor.max_seconds,
                     camera_vertical_offset=self.camera_guard.vertical_offset,
                     camera_note="Relative counts since last visual level check; not degrees")
        return state

    def _resolve_game_id(self, game_id: str) -> str:
        explicit = " ".join(game_id.split()).casefold()
        title = " ".join(self.capture.get_window_title().split()).casefold()
        profile_title = " ".join(getattr(self.profile, "window", "").split()).casefold()
        if profile_title and title.startswith(profile_title):
            title=title.split(" - ",1)[0]
        return f"game:{explicit}" if explicit else (f"window:{title}" if title else "monitor:primary")

    def attach(self, hwnd: int, game_id: str = ""):
        """Switch memory and target together; never mix games' recent histories."""
        if game_id and self.profile is None:
            from .profiles import load_profile
            next_profile = load_profile(game_id)
        else:
            next_profile = self.profile
        with self._step_lock:
            self.capture.hwnd = hwnd
            self.capture.last_rect = None
            self.pointer.hide()
            self.profile = next_profile
            self.brain.active_profile = self.profile
            self.game_id = self._resolve_game_id(game_id)
            self.recent_history = self.memory.get_recent_experiences(self.game_id)
            self.state_repeat_detector = StateRepeatDetector(config.state_repeat_window)
            self._last_repeat_recovery_key = None
            self.last_memory_context = self.last_memory_warning = ""
            self.last_frame = self.last_decision = None
            self.last_delta = 0.0
            self.last_error = ""
            self.last_timings = {}
            self.controls.aborted = False
            self._stop_event.clear()
            self.last_emergency_release_error = ""
            self._build_controls()
            self._pending_menu = None
            self.last_menu_memory = {}
            self._load_supervision()
            if not self.telemetry.closed and self.telemetry.meta:
                self.telemetry.update_metadata(window_title=self.capture.get_window_title(),
                                               hwnd=self.capture.hwnd or None, game_id=self.game_id,
                                               input_backend=self.input_backend)
            self.prepare_target()

    def step(self) -> Dict:
        self.start_emergency_hotkey()
        self.begin_telemetry_run()
        with self._step_lock:
            self.last_error = ""
            self._phase = "before_capture"
            self._attempt_decision = None
            self._input_started = False
            self._active_menu_recall = None
            trace = self.telemetry.begin_step(self.step_count + 1) if not self.telemetry.closed else None
            self._telemetry_trace = trace
            self._telemetry_before_frame = None
            self._telemetry_after_frame = None
            self._telemetry_assess_calls = 0
            request_start = getattr(self.brain, "telemetry_http_requests", None)
            decision_start = getattr(self.brain, "telemetry_decision_calls", None)
            request_events = getattr(self.brain, "telemetry_http_request_events", [])
            raw_responses = getattr(self.brain, "telemetry_raw_json_responses", [])
            request_event_start = len(request_events) if isinstance(request_events, list) else 0
            raw_response_start = len(raw_responses) if isinstance(raw_responses, list) else 0
            failure = None
            try:
                return self._step()
            except Exception as exc:
                failure = exc
                self.last_error = f"{type(exc).__name__}: {exc}"
                if trace:
                    trace.data["error"] = {"type": type(exc).__name__, "message": str(exc)[:240]}
                    trace.data["outcome"]["status"] = "refusal" if isinstance(exc, InterruptedError) else "uncertain"
                    trace.data["decision"]["refusal_reason"] = self.last_error[:240]
                    fallback_reason = getattr(self.brain, "last_error", "") or getattr(
                        self.brain, "last_assessment_error", ""
                    )
                    if fallback_reason:
                        trace.data["decision"]["fallback_reason"] = str(fallback_reason)[:240]
                if self._active_menu_recall and self._input_started:
                    self.menu_memory.invalidate(self._active_menu_recall["id"])
                self.memory.record_failure(time.time(), self.game_id, self.session_id, self.step_count,
                                           self._phase, self.supervisor.goal_id, self.last_error,
                                           {"action_source": self.last_action_source, "timings": self.last_timings,
                                            "decision": self._attempt_decision,
                                            "input_may_have_been_sent": self._input_started})
                raise
            finally:
                if trace:
                    request_end = getattr(self.brain, "telemetry_http_requests", None)
                    decision_end = getattr(self.brain, "telemetry_decision_calls", None)
                    if request_start is not None and request_end is not None:
                        trace.data["decision"]["vlm_http_requests"] = request_end - request_start
                    if decision_start is not None and decision_end is not None:
                        trace.data["decision"]["decision_calls"] = decision_end - decision_start
                        trace.data["decision"]["vlm_retries"] = max(0, decision_end - decision_start - 1)
                    request_events = getattr(self.brain, "telemetry_http_request_events", [])
                    raw_responses = getattr(self.brain, "telemetry_raw_json_responses", [])
                    if not isinstance(request_events, list):
                        request_events = []
                    if not isinstance(raw_responses, list):
                        raw_responses = []
                    trace.data["decision"]["http_requests"] = request_events[request_event_start:]
                    trace.data["decision"]["http_request_counts"] = {
                        kind: sum(event.get("type") == kind for event in trace.data["decision"]["http_requests"])
                        for kind in ("decide", "preflight", "post_check")
                    }
                    raw_attempts = raw_responses[raw_response_start:]
                    trace.data["decision"]["raw_json"] = raw_attempts[-1] if raw_attempts else None
                    trace.data["decision"]["raw_json_attempts"] = raw_attempts
                    if self._attempt_decision is not None:
                        trace.data["decision"]["final"] = self.last_decision.model_dump()
                    self.telemetry.finish_step(trace, self._telemetry_before_frame, self._telemetry_after_frame)
                    if request_event_start:
                        del request_events[:request_event_start]
                    if raw_response_start:
                        del raw_responses[:raw_response_start]
                    del request_events[:]
                    del raw_responses[:]
                self._telemetry_trace = None
                if failure is not None and not isinstance(failure, VisionDecisionError):
                    reason = "f12" if isinstance(failure, InterruptedError) and self._stop_event.is_set() else "error"
                    self.end_telemetry_run(reason, error_message=self.last_error if reason == "error" else None)

    def _step(self) -> Dict:
        """Run a single perception-decision-action-evaluation cycle."""
        self._check_stop()
        self.step_count += 1
        started = time.perf_counter()
        verification_seconds = 0.0
        goal_id = self.supervisor.goal_id
        
        # 1. Capture screen before action
        self._trace_start("capture")
        self.prepare_target()
        before_frame = self.capture.grab_frame()
        self._telemetry_before_frame = before_frame
        self._trace_stop("capture")
        self._check_stop()
        observed_rect = self.capture.last_rect
        self.last_frame = before_frame
        self._trace_start("memory_lookup")
        state_hash = self.memory.compute_state_hash(before_frame)
        visual_signature = self.memory.compute_visual_signature(before_frame)
        repeat = ({"state_repeat": False, "repeat_of_step": None, "look_oscillation": False})
        if config.state_repeat_detection_enabled:
            repeat = self.state_repeat_detector.detect(state_hash, self.step_count)
            if self._telemetry_trace:
                self._telemetry_trace.data["loop"].update(repeat)
            repeat_key = (state_hash, repeat.get("repeat_of_step"))
            should_recover = ((repeat["state_repeat"] and repeat_key != self._last_repeat_recovery_key)
                              or repeat["look_oscillation"])
            if config.state_repeat_triggers_recovery and should_recover:
                self.supervisor.trigger_recovery()
                self._last_repeat_recovery_key = repeat_key
        if self._pending_menu:
            pending=self._pending_menu
            check_started=time.perf_counter()
            confirmation,confirmation_id=self._assess(before_frame,before_frame,BrainDecision(**pending["action"]),
                                                       goal_id,"menu_confirm")
            verification_seconds += time.perf_counter()-check_started
            remembered=self.menu_memory.remember(self.game_id,pending["frame"],pending["action"],
                                                  pending["check_id"],confirmation_id) if confirmation.usable else None
            if not remembered and pending.get("replay_id"):
                self.menu_memory.invalidate(pending["replay_id"])
            self.last_menu_memory={"mode":"learned" if remembered else "unconfirmed","action_id":remembered,
                                   "message":"Переход меню запомнен" if remembered else "Переход меню не подтверждён"}
            self._pending_menu=None
        
        # Check dead clicks and loop detection
        context_hint = self.memory.get_learning_context(
            state_hash, self.game_id, visual_signature, config.memory_hash_distance
        )
        context_hint += self.memory.get_recent_failure_hint(self.game_id, visual_signature)
        context_hint += self.memory.get_navigation_hint(
            self.game_id, visual_signature, config.navigation_stuck_delta
        )
        context_hint += self.supervisor.context()
        if self.last_guard_reason:
            context_hint += "\nPREVIOUS CONTROL GUARD: " + self.last_guard_reason
        if self.last_assessment:
            context_hint += "\nSEPARATE VISUAL CHECK (fallible model inference, not actor narrative): " + str({
                "after": self.last_assessment["after"], "confidence": self.last_assessment["confidence"],
                "uncertainty": self.last_assessment["uncertainty"],
                "result": self.last_assessment["result"], "evidence": self.last_assessment["evidence"],
            })
        # Do not report a stale stuck loop after resuming on a different screen.
        same_screen = self.recent_history and self.recent_history[-1].get("next_state_hash") == state_hash
        stuck_warning = self.memory.check_stuck_loop(self.recent_history) if same_screen else None
        
        if stuck_warning:
            context_hint += "\n" + stuck_warning
            self.persona.speak(
                "Кажется, я повторяюсь. Проверю прошлые попытки и поищу другой ход.",
                mood="confused"
            )
        if self._telemetry_trace:
            self._telemetry_trace.data["memory"]["similar_records"] = getattr(
                self.memory, "last_learning_match_count", None)
        self._trace_stop("memory_lookup")
        
        # 2. VLM Perception & Planning
        self._phase = "decision"
        vision_started = time.perf_counter()
        decision = self._decide(before_frame, context_hint)
        mine_proposed = decision.action_type == "mine"
        self._check_stop()
        vision_seconds = time.perf_counter() - vision_started
        self._trace_start("memory_lookup")
        self.last_memory_warning = self.memory.get_repetition_warning(
            state_hash, self.game_id, decision.model_dump(), visual_signature
        ) or self.memory.get_recent_failure_hint(self.game_id, visual_signature, decision.model_dump()) or self.memory.get_navigation_hint(
            self.game_id, visual_signature, config.navigation_stuck_delta, decision.model_dump()
        )
        self._trace_stop("memory_lookup")
        self.last_action_source = "vlm"
        learnable = not bool(self.brain.last_error)
        if self.last_memory_warning:
            context_hint += "\n" + self.last_memory_warning
            vision_started = time.perf_counter()
            decision = self._decide(before_frame, context_hint)
            mine_proposed = mine_proposed or decision.action_type == "mine"
            vision_seconds += time.perf_counter() - vision_started
            learnable = not bool(self.brain.last_error)
            self._trace_start("memory_lookup")
            if learnable and self.memory.get_navigation_hint(
                self.game_id, visual_signature, config.navigation_stuck_delta, decision.model_dump()
            ):
                decision = BrainDecision(
                    action_type="look", look_direction="right", look_amount="large",
                    duration=0.1, confidence=0.3,
                    observation=decision.observation, intent="Inspect a route around a possible obstacle",
                    commentary="Вперёд не получается. Осмотрюсь сбоку.",
                )
                self.last_action_source = "navigation_recovery"
            elif self.memory.get_repetition_warning(
                state_hash, self.game_id, decision.model_dump(), visual_signature
            ) or self.memory.get_recent_failure_hint(self.game_id, visual_signature, decision.model_dump()):
                # A small VLM may ignore retrieval. Do not execute a known loop.
                decision = BrainDecision(
                    action_type="wait", duration=0.5, confidence=0.0,
                    observation=decision.observation,
                    intent="Reobserve before choosing a different action",
                    commentary="Я уже пробовала это без результата. Осмотрюсь и выберу другой ход.",
                )
                learnable = False
            self._trace_stop("memory_lookup")
        self.last_memory_context = context_hint
        recalled=self.menu_memory.recall(self.game_id,before_frame,decision)
        if recalled:
            saved={key:value for key,value in recalled["action"].items() if value is not None}
            decision=decision.model_copy(update={**saved,
                "commentary":"Я уже была в этом меню. Этот шаг сработал — повторю.","confidence":.95})
            self._active_menu_recall=recalled
            self.last_action_source="menu_memory"
            self.last_menu_memory={"mode":"replay",**recalled,
                                   "message":"Знакомое меню: повторяю проверенный шаг"}
        # Mining requires a fresh, separate visual safety check. Actor narration
        # and the mere fact that the image changed cannot authorize it.
        facts = None
        facts_confident = False
        facts_source = ""
        preflight_assessment = None
        preflight_raw_response = None
        if self._verified_frame is not None and self.memory.compute_screen_delta(before_frame, self._verified_frame) < .02:
            facts = self._verified_facts
            facts_confident = True
            facts_source = "cached_verified"
        if decision.action_type == "mine" and facts is None:
            check_started = time.perf_counter()
            raw_responses = getattr(self.brain, "telemetry_raw_json_responses", [])
            raw_start = len(raw_responses) if isinstance(raw_responses, list) else 0
            preflight_assessment, _ = self._assess(before_frame, before_frame, decision, goal_id, "preflight")
            raw_responses = getattr(self.brain, "telemetry_raw_json_responses", [])
            if isinstance(raw_responses, list) and len(raw_responses) > raw_start:
                preflight_raw_response = str(raw_responses[-1])[:4000]
            self._check_stop()
            verification_seconds += time.perf_counter() - check_started
            if preflight_assessment.confidence >= .6:
                facts = preflight_assessment.after
                facts_confident = preflight_assessment.usable and preflight_assessment.result != "regressed"
                facts_source = "preflight"
        if facts is None:
            facts = FrameFacts(scene=decision.scene_type,
                               pitch=decision.camera_pitch if decision.visual_confidence >= .85 else "unknown")
            facts_source = "decision_fallback"
        grounded_confident = (facts_confident and facts.scene == "gameplay" and facts.safe_footing is True)
        recovered, recovery_reason = self.supervisor.recovery_action(decision.model_dump(), facts)
        if recovery_reason:
            decision = BrainDecision(**recovered).model_copy(update={
                "intent": "Test an alternative route after repeated ineffective actions",
                "commentary": "Попробую другой подход, чтобы разобраться, что мешает.",
            })
            self.last_action_source = "strategy_recovery"
        profile_camera_reason = ""
        if self.profile and self.profile.camera_type == "none" and decision.action_type == "look":
            decision = decision.model_copy(update={"action_type": "wait", "duration": 0.2,
                "intent": "Wait because this profile has no camera", "look_direction": None,
                "look_amount": None, "x": None, "y": None})
            profile_camera_reason = "profile_has_no_camera"
        filtered, guard_reason = self.camera_guard.filter(
            decision.model_dump(), facts=facts,
            grounded_confident=grounded_confident,
        )
        self.last_guard_reason = guard_reason or profile_camera_reason
        if self.last_guard_reason:
            decision = BrainDecision(**filtered).model_copy(update={
                "intent": "Restore a safe usable view before continuing play",
                "commentary": "Проверю направление взгляда и безопасный путь.",
            })
            self.last_action_source = "camera_guard"
            learnable = decision.action_type != "wait"
        decision = self._apply_profile_motion(decision)
        if self._telemetry_trace and hasattr(self._telemetry_trace, "add_prompt_context"):
            self._telemetry_trace.add_prompt_context(context_hint, self.goal)
        if config.state_repeat_detection_enabled:
            self.state_repeat_detector.remember(state_hash, self.step_count, decision.model_dump())
        self.last_decision = decision
        self._attempt_decision = decision.model_dump()
        if self._telemetry_trace:
            action_values = decision.model_dump()
            self._telemetry_trace.data["decision"]["action_type"] = decision.action_type
            self._telemetry_trace.data["decision"]["action_source"] = self.last_action_source
            self._telemetry_trace.data["decision"]["parameters"] = {
                key: action_values.get(key) for key in (
                    "x", "y", "x2", "y2", "key", "duration", "look_direction", "look_amount")
            }
            model_refusal = getattr(self.brain, "last_refusal_reason", None)
            if not isinstance(model_refusal, str):
                model_refusal = None
            refusal_reason = self.last_guard_reason or model_refusal
            if refusal_reason:
                self._telemetry_trace.data["decision"]["refusal_reason"] = str(refusal_reason)[:240]
            if mine_proposed:
                failed_conditions = []
                if facts.pitch != "level":
                    failed_conditions.append("pitch_not_level")
                if not grounded_confident:
                    failed_conditions.append("grounded_confident_false")
                self._telemetry_trace.data["decision"]["guard"] = {
                    "facts": {
                        "pitch": facts.pitch,
                        "scene": facts.scene,
                        "safe_footing": facts.safe_footing,
                        "facts_confident": bool(facts_confident),
                        "grounded_confident": bool(grounded_confident),
                        "source": facts_source,
                    },
                    "preflight": ({
                        "attempted": True,
                        "raw_response": preflight_raw_response,
                        "assessment": preflight_assessment.model_dump(),
                        "confidence": preflight_assessment.confidence,
                        "usable": preflight_assessment.usable,
                        "result": preflight_assessment.result,
                    } if preflight_assessment is not None else {"attempted": False}),
                    "refusal_condition": failed_conditions if self.last_guard_reason else [],
                    "refusal_reason": self.last_guard_reason or None,
                }
            self._telemetry_trace.data["loop"]["repeat_warning"] = bool(self.last_memory_warning)
        
        # 3. Streamer Commentary
        mood = "curious"
        if decision.confidence > 0.8:
            mood = "excited"
        elif decision.confidence < 0.4 or stuck_warning:
            mood = "confused"
        self.persona.speak(decision.commentary, mood=mood)

        # 4. Execute Action
        self._phase = "action"
        action_started = time.perf_counter()
        self._check_stop()
        self._check_profile_forbidden_key(decision)
        self.prepare_target()
        if self.capture.get_client_rect() != observed_rect:
            raise RuntimeError("Game window moved/resized during inference; retry with a new frame")
        action_desc = f"{decision.action_type.upper()}"
        self._input_started = decision.action_type != "wait"
        self._check_stop()
        if self.input_backend == "native" and decision.action_type in {"click","double_click","drag"}:
            sx,sy=self.capture.norm_to_screen(decision.x,decision.y)
            self.pointer.update(sx,sy,observed_rect,"native_target",False)
        
        self._trace_start("input_send")
        if decision.action_type == "click" and decision.x is not None and decision.y is not None:
            screen_x, screen_y = self.capture.norm_to_screen(decision.x, decision.y)
            action_desc += f" at ({decision.x}, {decision.y}) -> screen ({screen_x}, {screen_y})"
            self.controls.click(screen_x, screen_y, hold_time=config.click_hold_seconds)

        elif decision.action_type == "double_click" and decision.x is not None and decision.y is not None:
            screen_x, screen_y = self.capture.norm_to_screen(decision.x, decision.y)
            action_desc += f" at ({decision.x}, {decision.y}) -> screen ({screen_x}, {screen_y})"
            self.controls.double_click(screen_x, screen_y)

        elif decision.action_type == "look":
            dx = int(decision.x or 0)
            dy = int(decision.y or 0)
            action_desc += f" camera dx={dx}, dy={dy}"
            if self.input_backend=="native":
                left,top,right,bottom=observed_rect
                self.pointer.update((left+right)//2,(top+bottom)//2,observed_rect,"camera")
            self.controls.move_camera(dx, dy)

        elif decision.action_type == "walk":
            k = decision.key or "w"
            dur = decision.duration
            action_desc += f" '{k}' for {dur:.1f}s"
            self.controls.press_key(k, duration=dur)

        elif decision.action_type == "mine":
            if self.profile:
                configured = self._send_profile_action("primary_hold")
                action_desc += f" configured {configured.kind} action for {configured.duration:.1f}s"
            else:
                dur = max(1.0, decision.duration or 2.5)
                action_desc += f" holding left click for {dur:.1f}s"
                self.controls.hold_mouse(button="left", duration=dur)

        elif decision.action_type == "use":
            if self.profile:
                configured = self._send_profile_action("interact")
                action_desc += f" configured {configured.kind} action for {configured.duration:.1f}s"
            else:
                action_desc += " right click (use/interact)"
                self.controls.click(button="right", hold_time=0.15)

        elif decision.action_type == "jump":
            k = decision.key or "w"
            action_desc += f" '{k}' + space for {decision.duration:.2f}s"
            jump = self.profile.actions.get("jump") if self.profile else None
            jump_key = jump.value if jump and jump.kind == "key" else "space"
            self.controls.press_keys([k, jump_key], duration=decision.duration)
            
        elif decision.action_type == "drag" and decision.x is not None and decision.x2 is not None:
            sx1, sy1 = self.capture.norm_to_screen(decision.x, decision.y)
            sx2, sy2 = self.capture.norm_to_screen(decision.x2, decision.y2)
            action_desc += f" from ({sx1}, {sy1}) to ({sx2}, {sy2})"
            self.controls.drag(sx1, sy1, sx2, sy2, duration=decision.duration)
            
        elif decision.action_type == "key" and decision.key:
            action_desc += f" '{decision.key}' for {decision.duration}s"
            self.controls.press_key(decision.key, duration=decision.duration)
            
        elif decision.action_type == "wait":
            action_desc += f" {decision.duration}s"
            if self._stop_event.wait(decision.duration):
                self._check_stop()
        self._trace_stop("input_send")

        action_seconds = time.perf_counter() - action_started
        self.camera_guard.executed(decision.model_dump())

        # 5. Observe outcome & screen change
        self._phase = "outcome_capture"
        self._trace_start("post_wait")
        if self._stop_event.wait(config.action_delay_seconds):
            self._check_stop()
        self._trace_stop("post_wait")
        self._check_stop()
        self._trace_start("capture")
        self.check_target()
        after_frame = self.capture.grab_frame()
        self._telemetry_after_frame = after_frame
        self._trace_stop("capture")
        self._trace_start("result_eval")
        if decision.action_type=="look":
            self.last_camera_feedback={"backend":self.input_backend,**camera_feedback(before_frame,after_frame)}
        delta = self.memory.compute_screen_delta(before_frame, after_frame)
        self.last_delta = delta
        effective = delta >= config.change_threshold
        next_state_hash = self.memory.compute_state_hash(after_frame)
        assessment = None
        check_id = None
        due = (self.step_count % config.verification_interval == 0
               or decision.action_type == "mine" or bool(self.last_guard_reason)
               or self.supervisor.pending is not None or self.supervisor.mode == "recovering"
               or (decision.scene_type!="gameplay" and decision.action_type in {"click","double_click","key"})
               or self._active_menu_recall is not None)
        assess_camera_action = config.assess_look_with_vlm or decision.action_type not in {"look", "wait"}
        if due and assess_camera_action:
            self._phase = "verification"
            check_started = time.perf_counter()
            assessment, check_id = self._assess(before_frame, after_frame, decision, goal_id, "post")
            self._check_stop()
            verification_seconds += time.perf_counter() - check_started
            if assessment.usable and assessment.result != "regressed":
                self._verified_frame = after_frame.copy()
                self._verified_facts = assessment.after
            else:
                self._verified_frame = self._verified_facts = None
        if assessment is not None and decision.action_type in {"click","double_click","key"}:
            source,target=facts_key(assessment.before),facts_key(assessment.after)
            if (assessment.usable and assessment.menu_action_succeeded is True and source and source.startswith("menu:")
                    and target and source!=target):
                expected=self._active_menu_recall["destination"] if self._active_menu_recall else target
                if target==expected or (target=="loading" and expected=="gameplay"):
                    self._pending_menu={"frame":before_frame.copy(),"action":decision.model_dump(),"check_id":check_id,
                                        "replay_id":self._active_menu_recall["id"] if self._active_menu_recall else None}
                elif self._active_menu_recall:self.menu_memory.invalidate(self._active_menu_recall["id"])
            elif self._active_menu_recall:
                self.menu_memory.invalidate(self._active_menu_recall["id"])
                self.last_menu_memory={"mode":"invalidated","message":"Старый шаг не подтвердился — ищу новый вариант"}
        self._phase = "memory"
        self._check_stop()
        supervisor_assessment = (assessment if decision.action_type not in {"look", "wait"}
                                 or config.assess_look_with_vlm else None)
        event = self.supervisor.observe(supervisor_assessment, time.perf_counter() - started,
                                        self.step_count, check_id)
        self._trace_stop("result_eval")
        if self._telemetry_trace:
            verdict = assessment.result if assessment is not None else None
            model_refusal = getattr(self.brain, "last_refusal_reason", None)
            if not isinstance(model_refusal, str):
                model_refusal = None
            refusal_reason = self.last_guard_reason or model_refusal
            if refusal_reason:
                status = "refusal"
            elif decision.action_type in {"look", "wait"}:
                status = "camera_motion"
            elif verdict == "progress":
                status = "success"
            elif verdict in {"blocked", "regressed"}:
                status = "no_effect"
            elif verdict == "unknown":
                status = "uncertain"
            elif not effective:
                status = "no_effect"
            else:
                status = "uncertain"
            (self._telemetry_no_effect_streak,
             self._telemetry_camera_only_streak) = StepTrace.update_loop_streaks(
                self._telemetry_no_effect_streak, self._telemetry_camera_only_streak,
                decision.action_type, effective, status,
            )
            self._telemetry_trace.data["outcome"].update({
                "image_delta": delta,
                "verdict": verdict,
                "checker_called": bool(self._telemetry_assess_calls),
                "status": status,
            })
            self._telemetry_trace.data["loop"]["no_effect_streak"] = self._telemetry_no_effect_streak
            self._telemetry_trace.data["loop"]["camera_only_streak"] = self._telemetry_camera_only_streak
        self.last_timings = {
            "vision_seconds": round(vision_seconds, 3), "action_seconds": round(action_seconds, 3),
            "verification_seconds": round(verification_seconds, 3),
            "total_seconds": round(time.perf_counter() - started, 3),
        }

        self._trace_start("memory_write")
        # Save debug screenshot if enabled
        if config.save_screenshots:
            try:
                fname = config.screenshots_dir / f"{self.session_id}_step_{self.step_count:04d}_{decision.action_type}.jpg"
                before_frame.save(fname, quality=80)
                after_frame.save(fname.with_stem(fname.stem + "_after"), quality=80)
            except Exception:
                pass

        # Update recent history for loop detection
        self.recent_history.append({
            "step": self.step_count,
            "action_type": decision.action_type,
            "x": decision.x,
            "y": decision.y,
            "intent": decision.intent,
            "delta": delta,
            "effective": effective and decision.action_type not in {"look", "wait"},
            "next_state_hash": next_state_hash,
        })
        if len(self.recent_history) > 10:
            self.recent_history.pop(0)

        # 6. Store learning experience
        self.memory.record_experience(
            timestamp=time.time(),
            state_hash=state_hash,
            action_type=decision.action_type,
            x=decision.x,
            y=decision.y,
            key=decision.key,
            screen_delta=delta,
            effective=effective and decision.action_type not in {"look", "wait"},
            commentary=decision.commentary,
            intent=decision.intent,
            game_id=self.game_id,
            next_state_hash=next_state_hash,
            duration=decision.duration,
            x2=decision.x2,
            y2=decision.y2,
            observation=decision.observation,
            session_id=self.session_id,
            visual_signature=visual_signature,
            learnable=learnable and decision.action_type not in {"look", "wait"},
            goal_id=None,
            outcome_status=(event["status"] if event["verified_change"]
                            else "camera_motion" if decision.action_type in {"look", "wait"}
                            else event["status"]),
            assessment=assessment.model_dump() if assessment else None,
            action_source=self.last_action_source, confidence=decision.confidence,
            timings=self.last_timings, policy_version=3,
        )
        if self._telemetry_trace:
            self._telemetry_trace.data["memory"]["experience_written"] = True
        if event["verified_change"]:
            self.memory.record_play_lesson(time.time(), self.game_id, event)
            if self._telemetry_trace:
                self._telemetry_trace.data["memory"]["lesson_written"] = True
        self.memory.save_supervisor(self.game_id, self.supervisor.snapshot())
        self._trace_stop("memory_write")

        outcome_text = f"Screen delta: {delta:.3f}; outcome verification: {event['status']}"
        self.persona.log_action(action_desc, outcome_text)

        return {
            "step": self.step_count,
            "decision": decision.model_dump(),
            "delta": delta,
            "effective": effective,
            "state_hash": state_hash,
            "next_state_hash": next_state_hash,
            "game_id": self.game_id,
            "memory_warning": self.last_memory_warning,
            "timings": self.last_timings,
            "action_source": self.last_action_source,
            "supervision": self.supervision_state(),
            "verification": event,
            "menu_memory": self.last_menu_memory,
        }

    def _assess(self, before, after, decision, goal_id, phase):
        self._check_stop()
        assess_started = time.perf_counter()
        self._telemetry_assess_calls += 1
        if self._telemetry_trace:
            self._telemetry_trace.data["outcome"]["checker_called"] = True
        request_type = "preflight" if phase in {"menu_confirm", "preflight"} else "post_check"
        previous_request_type = getattr(self.brain, "_telemetry_request_type", "decide")
        self.brain._telemetry_request_type = request_type
        try:
            assessment = self._call_interruptibly(self.brain.assess, before, after, {**decision.model_dump(),
                                           "scene_type":"menu" if decision.scene_type=="unknown" and decision.action_type in {"click","double_click","key"} else decision.scene_type,
                                           "verification_only":phase in {"menu_confirm","preflight"}},
                                           self.supervisor.context())
        finally:
            self.brain._telemetry_request_type = previous_request_type
        self._check_stop()
        if self._telemetry_trace:
            if phase in {"menu_confirm", "preflight"}:
                self._telemetry_trace.add_ms("preflight_eval", (time.perf_counter() - assess_started) * 1000)
        if not isinstance(assessment, VisualAssessment):
            assessment = VisualAssessment(uncertainty="Invalid visual assessment response")
        error = self.brain.last_assessment_error or ""
        self.last_assessment = {"phase": phase, "source": assessment.evidence_source, **assessment.model_dump()}
        self.last_assessment_error = str(error) if error else ""
        if self._telemetry_trace and self.last_assessment_error:
            self._telemetry_trace.data["decision"]["fallback_reason"] = self.last_assessment_error[:240]
        self.camera_guard.observe(assessment)
        write_started = time.perf_counter()
        check_id = self.memory.record_visual_check(time.time(), self.game_id, self.session_id,
                                                   self.step_count, phase, goal_id, assessment.model_dump(),
                                                   self.last_assessment_error)
        if self._telemetry_trace:
            self._telemetry_trace.add_ms("memory_write", (time.perf_counter() - write_started) * 1000)
        return assessment, check_id

    def _decide(self, frame: Image.Image, context_hint: str) -> BrainDecision:
        started = time.perf_counter()
        grounding_before = getattr(self.brain, "telemetry_grounding_seconds", None)
        try:
            if self._telemetry_trace and hasattr(self._telemetry_trace, "add_prompt_context"):
                self._telemetry_trace.add_prompt_context(context_hint, self.goal)
            decision = self._call_interruptibly(self.brain.decide, frame, context_hint=context_hint,
                                                 goal=self.goal, profile=self.profile)
            self._check_stop()
            if self.brain.last_error:
                self.last_decision = decision
                self.last_memory_context = context_hint
                raise VisionDecisionError(f"VLM decision failed: {self.brain.last_error}")
            return self._normalize_motion(decision)
        finally:
            if self._telemetry_trace:
                elapsed_ms = (time.perf_counter() - started) * 1000
                grounding_after = getattr(self.brain, "telemetry_grounding_seconds", None)
                grounding_ms = (max(0.0, grounding_after - grounding_before) * 1000
                                if grounding_before is not None and grounding_after is not None else 0.0)
                self._telemetry_trace.add_ms("grounding", grounding_ms)
                self._telemetry_trace.add_ms("vlm_decide", max(0.0, elapsed_ms - grounding_ms))

    def _normalize_motion(self, decision: BrainDecision) -> BrainDecision:
        if decision.action_type in {"walk", "jump"}:
            duration_limit = config.max_motion_seconds
            if self.profile:
                duration_limit = min(duration_limit, self.profile.motion_limits.max_action_seconds)
                default_key = self.profile.move_keys["forward"]
                allowed = set(self.profile.move_keys.values())
            else:
                default_key, allowed = "w", {"w", "a", "s", "d"}
            key = (decision.key or default_key).lower().strip()
            if decision.action_type == "jump" and key == "space":
                key = default_key  # Compatibility: the model may name the implicit jump key.
            if key not in allowed:
                raise ValueError(f"Unsupported movement direction: {key}")
            if self.profile and decision.action_type == "jump":
                jump = self.profile.actions.get("jump")
                if jump is None or jump.kind != "key":
                    raise ValueError("Profile jump action must be configured as a key")
                duration_limit = min(duration_limit, jump.duration)
            if duration_limit < 0.5:
                raise ValueError("Configured movement limit is shorter than 0.5 seconds")
            duration = max(0.5, min(decision.duration, duration_limit))
            return decision.model_copy(update={"duration": duration, "key": key})
        return decision

    def run_loop(self, max_steps: int = 100):
        """Run continuous autonomous play loop."""
        termination_reason = "error"
        termination_error = None
        consecutive_vlm_failures = 0
        try:
            self.begin_telemetry_run()
            self.start_emergency_hotkey()
            self.is_running = True
            self.persona.speak("Всем приветик! Начинаем играть, сейчас во всём разберемся!", mood="excited")
            termination_reason = "step_limit"
            for i in range(max_steps):
                if not self.is_running:
                    termination_reason = "f12" if self._stop_event.is_set() else "panel_stop"
                    break
                try:
                    self.step()
                    consecutive_vlm_failures = 0
                except VisionDecisionError:
                    consecutive_vlm_failures += 1
                    delay = min(8.0, 1.0 * (2 ** min(consecutive_vlm_failures - 1, 3)))
                    self.persona.speak(
                        f"Не удалось разобрать экран. Повторю попытку через {delay:.0f} с.", mood="confused"
                    )
                    if self._stop_event.wait(delay):
                        termination_reason = "f12"
                        break
        except InterruptedError as e:
            termination_reason = "f12" if self._stop_event.is_set() else "error"
            if termination_reason == "error":
                termination_error = f"{type(e).__name__}: {e}"
            self.persona.speak(f"Ой! Аварийная остановка: {e}", mood="upset")
        except KeyboardInterrupt:
            termination_reason = "panel_stop"
            self.persona.speak("Остановились по просьбе пользователя. До следующего стрима!", mood="curious")
        except Exception as exc:
            termination_reason = "error"
            termination_error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            active_exception = sys.exc_info()[1]
            self.is_running = False
            try:
                if self.emergency_hotkey:
                    self.emergency_hotkey.close()
            except Exception as exc:
                logging.getLogger(__name__).warning("Emergency hotkey shutdown failed (%s)", type(exc).__name__)
            try:
                stats = self.memory.get_stats(self.game_id)
                self.persona.speak(
                    f"Итоги сессии: сделано шагов {stats['total_actions']}, "
                    f"эффективных действий {stats['effective_actions']}, "
                    f"открыто уникальных экранов {stats['unique_screens_discovered']}!",
                    mood="triumphant"
                )
            except Exception as exc:
                if active_exception is None:
                    termination_reason = "error"
                    termination_error = f"{type(exc).__name__}: {exc}"
                    raise
                logging.getLogger(__name__).warning("Cosmetic run summary failed (%s)", type(exc).__name__)
            finally:
                self.end_telemetry_run(termination_reason, error_message=termination_error)
