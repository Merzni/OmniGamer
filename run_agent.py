"""CLI entry point for running OmniGamer directly from console."""
import argparse
import json
import time
from omnigamer.personality import configure_stdio

configure_stdio()

from omnigamer.capture import list_visible_windows, find_window_by_title
from omnigamer.agent import OmniAgent
from omnigamer.config import config

def main():
    parser = argparse.ArgumentParser(description="OmniGamer Autonomous AI Gamer")
    parser.add_argument("--window", "-w", type=str, default="", help="Window title substring to attach to")
    parser.add_argument("--list", "-l", action="store_true", help="List all open windows and exit")
    parser.add_argument("--steps", "-s", type=int, default=50, help="Max autonomous steps to execute")
    parser.add_argument("--game-id", default="", help="Stable memory profile, e.g. minecraft-survival")
    parser.add_argument("--profile", default="", help="Path to a validated game profile JSON")
    parser.add_argument("--goal", default="", help="Plan and execute a short goal through registered skills")
    parser.add_argument("--legacy", action="store_true",
                        help="Run the legacy per-step VLM loop (required when --goal is omitted)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Plan and log would-send steps without sending input")
    parser.add_argument("--frame-dir", default="",
                        help="Replay saved frames; requires --goal --dry-run and no window is attached")
    parser.add_argument("--skill", choices=["approach_target"], default="",
                        help="Run a bounded explicit skill instead of the VLM agent loop")
    parser.add_argument("--target", default="", help="Text description of the target for --skill")
    parser.add_argument("--input-backend", choices=["window","native"],default=None,
                        help="window: independent addressed input; native: shared system cursor")
    args = parser.parse_args()

    if args.list:
        print("\n=== Open Windows ===")
        windows = list_visible_windows()
        for i, w in enumerate(windows):
            print(f"[{i+1}] HWND: {w['hwnd']:<8} | Size: {w['width']}x{w['height']:<5} | Title: {w['title']}")
        return

    from omnigamer.profiles import load_profile
    try:
        profile = load_profile(args.game_id, args.profile or None)
    except ValueError as exc:
        parser.error(str(exc))
    if args.skill and (not (args.window or profile.window) or not args.target):
        parser.error("--skill requires a window (or a profile window) and --target")
    if args.skill and (args.goal or args.legacy):
        parser.error("--skill cannot be combined with --goal or --legacy")
    if not args.skill and bool(args.goal.strip()) == bool(args.legacy):
        parser.error("choose exactly one of --goal TEXT or --legacy")
    if args.dry_run and not args.goal.strip():
        parser.error("--dry-run currently requires --goal")
    if args.frame_dir and (not args.goal.strip() or not args.dry_run):
        parser.error("--frame-dir requires --goal --dry-run")

    resolved_window = None
    window_filter = args.window or profile.window
    if (args.goal or args.skill) and not window_filter and not args.frame_dir:
        parser.error("--goal and --skill require --window or a profile window")
    if not args.skill and window_filter and not args.frame_dir:
        resolved_window = find_window_by_title(window_filter)
        if not resolved_window:
            print(f"Error: Window matching '{window_filter}' not found! Run with -l to see active windows.")
            return

    selected_game_id = args.game_id or (profile.game_id if args.profile or args.frame_dir else "")
    agent = OmniAgent(hwnd=0, game_id=selected_game_id,
                      db_path=":memory:" if args.dry_run else config.db_path,
                      input_backend=args.input_backend or profile.input_backend,
                      goal=args.goal, profile=profile)
    if args.frame_dir:
        from omnigamer.replay import FrameDirectoryCapture
        try:
            agent.capture = FrameDirectoryCapture(args.frame_dir)
        except ValueError as exc:
            parser.error(str(exc))
    recorder = None
    try:
        agent.begin_telemetry_run()
        if args.skill:
            from omnigamer.detector_client import DetectorClient
            from omnigamer.skills.approach_target import ApproachTargetSettings, ApproachTargetSkill
            from omnigamer.skills.telemetry import ApproachTargetRecorder

        hwnd = 0
        if window_filter and not args.frame_dir:
            target = find_window_by_title(window_filter) if args.skill else resolved_window
            if not target:
                raise RuntimeError(f"Window matching {window_filter!r} not found")
            hwnd = target["hwnd"]
            print(f"Attached to target window: '{target['title']}' (HWND: {hwnd})")
        elif args.frame_dir:
            print(f"Replaying {len(agent.capture.frames)} saved frames; no game window attached.")
        else:
            print("No specific window requested. Capturing the primary monitor; use --window to target a game.")
        if hwnd:
            agent.attach(hwnd, selected_game_id, focus=not args.dry_run)
        if args.skill:
            settings = ApproachTargetSettings.from_profile(profile)
            agent.telemetry.update_metadata(
                run_mode="skill", skill=args.skill, target=args.target,
                detector_url=config.detector_url, detector_score_min=settings.score_min,
                detector_max_box_area=settings.max_box_area, skill_settings=settings.__dict__,
                detector_model="unknown",
            )
            recorder = ApproachTargetRecorder(agent.telemetry, settings, args.target)
            agent.start_emergency_hotkey()
            detector = DetectorClient()
            health = detector.health()
            if health:
                agent.telemetry.update_metadata(detector_model=health.get("model"),
                                                detector_precision=health.get("precision"))
            skill = ApproachTargetSkill(
                capture=agent.capture, target_text=args.target, detector=detector,
                controls=agent.controls, camera_guard=agent.camera_guard, game_id=agent.game_id,
                stop_check=agent._check_stop, recorder=recorder, settings=settings, profile=profile,
            )
            result = skill.run()
            final_frame = agent.capture.grab_frame()
            recorder.finish(result, final_frame)
            print(f"Навык завершён: {result.state}; {result.iterations} итераций; "
                  f"{result.elapsed_seconds:.1f} с; {result.reason}")
        else:
            if args.goal:
                from omnigamer.detector_client import DetectorClient
                from omnigamer.planner import (BudgetControls, DeadlineControls, DeadlineDetector,
                                               GoalExecutor, GoalPlanner, InputActionBudget,
                                               SkillOutcomeStore, skill_time_remaining)
                from omnigamer.skills.registry import default_registry
                from omnigamer.skills.telemetry import SkillStepRecorder

                registry = default_registry(profile=profile)
                detector = DetectorClient()
                planner = GoalPlanner(registry, agent.brain.plan_goal)
                outcome_store = SkillOutcomeStore(":memory:" if args.dry_run else agent.memory.db_path)
                input_action_budget = InputActionBudget(profile.motion_limits.max_actions)
                def execute_skill(spec, skill_args, remaining_seconds=None):
                    deadline = time.monotonic() + max(0.0, remaining_seconds or 0.0)
                    bounded_controls = BudgetControls(DeadlineControls(agent.controls, deadline),
                                                       input_action_budget)
                    bounded_detector = DeadlineDetector(detector, deadline)
                    def bounded_stop_check():
                        agent._check_stop()
                        skill_time_remaining(deadline)
                    forbidden = set(profile.forbidden_keys)
                    mapped = []
                    if spec.name == "move_pulse":
                        mapped.append(profile.move_keys[skill_args["direction"]])
                    elif spec.name == "approach_target":
                        mapped.extend(profile.move_keys[key] for key in ("forward", "left", "right"))
                    elif spec.name == "search_target":
                        mapped.extend(profile.move_keys[key] for key in ("left", "right"))
                    elif spec.name == "interact_until_change":
                        action = profile.actions.get(skill_args["action"])
                        if action and action.kind == "key":
                            mapped.append(action.value)
                    if forbidden.intersection(str(key).casefold() for key in mapped):
                        raise ValueError("Profile forbids a key required by this skill")
                    recorder = SkillStepRecorder(agent.telemetry, spec.name, skill_args)
                    shared = {"capture": agent.capture, "controls": bounded_controls,
                              "camera_guard": agent.camera_guard, "profile": profile,
                              "stop_check": bounded_stop_check, "recorder": recorder}
                    if spec.name == "approach_target":
                        from omnigamer.skills.approach_target import ApproachTargetSettings
                        settings = ApproachTargetSettings.from_profile(profile)
                        if remaining_seconds is not None:
                            settings.max_seconds = min(settings.max_seconds, remaining_seconds)
                        return spec.executor(skill_args, detector=bounded_detector, game_id=agent.game_id,
                                             telemetry=agent.telemetry, settings=settings, **shared)
                    if spec.name in {"search_target", "wait_for_condition"}:
                        shared["detector"] = bounded_detector
                    if spec.name == "wait_for_condition":
                        shared.pop("camera_guard")
                        shared.pop("controls")
                    if spec.name == "search_target":
                        return spec.executor(skill_args, **shared)
                    if spec.name == "interact_until_change":
                        return spec.executor(skill_args, **shared)
                    if spec.name == "click_menu_target":
                        remaining = max(0.05, deadline - time.monotonic())
                        shared["grounder"] = lambda frame, decision: agent._call_interruptibly(
                            agent.brain.ground_click, frame, decision, profile=profile,
                            timeout_seconds=remaining)
                        shared["prepare_target"] = agent.prepare_target
                        shared["check_target"] = agent.check_target
                        return spec.executor(skill_args, **shared)
                    if spec.name in {"move_pulse", "turn"}:
                        return spec.executor(skill_args, **shared)
                    return spec.executor(skill_args, **shared)

                def preview_menu_target(step, frame, remaining_seconds):
                    from omnigamer.skills.menu import click_menu_target
                    deadline = time.monotonic() + max(0.0, remaining_seconds)
                    def bounded_grounder(current_frame, decision):
                        remaining = max(0.05, deadline - time.monotonic())
                        return agent._call_interruptibly(
                            agent.brain.ground_click, current_frame, decision, profile=profile,
                            timeout_seconds=remaining)
                    def require_visible_window():
                        # Dry-run must never foreground the game window or send input.
                        agent.prepare_target(focus=False)
                    return click_menu_target(
                        **step.args, capture=agent.capture, controls=None, profile=profile,
                        grounder=bounded_grounder, prepare_target=require_visible_window,
                        check_target=agent.capture.assert_visible, stop_check=agent._check_stop,
                        preview_only=True)

                agent.start_emergency_hotkey()
                def record_goal_event(row):
                    directory = getattr(agent.telemetry, "directory", None)
                    if directory:
                        event_path = directory / "goal_steps.jsonl"
                        with event_path.open("a", encoding="utf-8", buffering=1) as stream:
                            stream.write(json.dumps({"schema_version": 1, "run_id": agent.telemetry.meta.get("run_id"),
                                **row}, ensure_ascii=False, separators=(",", ":")) + "\n")
                    if row.get("event") == "would_send":
                        print("would send: " + json.dumps(row, ensure_ascii=False))

                def verify_skill_result(step, before, after, result, timeout_seconds):
                    action = {"action_type": step.skill,
                              "parameters": step.args,
                              "expected_result": step.expected_result,
                              "success_check": step.success_check}
                    context = (f"Goal step: {step.skill}. Expected result: {step.expected_result}. "
                               f"Required visible success check: {step.success_check}. "
                               "A pixel or region change alone is not sufficient evidence.")
                    return agent._call_interruptibly(agent.brain.assess, before, after, action, context,
                                                     timeout_seconds=timeout_seconds)

                def preflight_mining_skill(goal, step, frame, timeout_seconds):
                    action = {"action_type": "mine", "target_label": goal,
                              "parameters": step.args, "preflight_only": True}
                    context = (f"Pre-action safety check for mining goal: {goal}. "
                               f"Planned visible result: {step.expected_result}. "
                               f"Success check: {step.success_check}. "
                               "Assess the current frame only; do not claim mining progress. "
                               "Report target_at_crosshair=true only if this goal's visible target surface "
                               "covers the exact center. Also report gameplay scene, level pitch, and safe footing. "
                               "Unknown or ambiguous facts must remain null/unknown.")
                    previous_request_type = getattr(agent.brain, "_telemetry_request_type", "decide")
                    agent.brain._telemetry_request_type = "preflight"
                    try:
                        return agent._call_interruptibly(
                            agent.brain.assess, frame, frame, action, context,
                            timeout_seconds=timeout_seconds)
                    finally:
                        agent.brain._telemetry_request_type = previous_request_type

                executor = GoalExecutor(
                    planner, registry, outcome_store, profile=profile, game_id=agent.game_id,
                    execute_skill=execute_skill, frame_provider=agent.capture.grab_frame,
                    stop_check=agent._check_stop, dry_run=args.dry_run,
                    input_action_budget=None if args.dry_run else input_action_budget,
                    result_verifier=verify_skill_result,
                    logger=record_goal_event, detector=detector,
                    menu_preview=preview_menu_target if args.dry_run and not args.frame_dir else None,
                    skill_preflight=preflight_mining_skill,
                )
                try:
                    result = executor.run(args.goal)
                finally:
                    record_goal_event({"event": "input_action_budget",
                                       "actions_used": input_action_budget.actions_used,
                                       "max_actions": input_action_budget.max_actions})
                agent.end_telemetry_run(result["status"].lower(), error_message=result.get("reason"))
                print(f"Цель: {result['status']}; шагов навыков: {result['steps']}; "
                      f"вводов: {input_action_budget.actions_used}/{input_action_budget.max_actions}; "
                      f"{result['reason']}")
            else:
                agent.run_loop(max_steps=args.steps)
    except BaseException as exc:
        if recorder is not None:
            recorder.finish_error(exc)
        else:
            agent.end_telemetry_run("error", error_message=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if not args.dry_run:
            try:
                agent.controls.release_all()
            except Exception:
                pass
        hotkey = getattr(agent, "emergency_hotkey", None)
        if hotkey:
            hotkey.close()

if __name__ == "__main__":
    main()
