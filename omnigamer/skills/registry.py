"""Skill catalog and parameter validation for goal planning and execution."""
from pydantic import BaseModel, ConfigDict, Field

from .approach_target import ApproachTargetResult, ApproachTargetSettings, ApproachTargetSkill
from .basic import (
    InteractUntilChangeParams,
    MovePulseParams,
    SearchTargetParams,
    TurnParams,
    WaitForConditionParams,
    interact_until_change,
    move_pulse,
    search_target,
    turn,
    wait_for_condition,
)
from .contracts import (InputActionLimitExceeded, SkillDeadlineExceeded, SkillRegistry,
                        SkillResult, SkillSpec, SkillStatus)
from .menu import ClickMenuTargetParams, click_menu_target
from .telemetry import SkillStepRecorder


class ApproachTargetParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_text: str = Field(min_length=1, max_length=200)


def _validate(model, args):
    return args if isinstance(args, model) else model.model_validate(args)


def _finish_recorder(recorder, result):
    if recorder and hasattr(recorder, "finish"):
        if getattr(recorder, "manages_run_finalization", False):
            recorder.finish(result, finalize_run=False)
        else:
            recorder.finish(result)
    return result


def _recorder_for(name, args, dependencies):
    recorder = dependencies.get("recorder")
    if recorder:
        return recorder
    telemetry = dependencies.pop("telemetry", None)
    if telemetry:
        return SkillStepRecorder(telemetry, name, args)
    raise ValueError("Skill execution requires a recorder or run telemetry")


def _run_recorded(recorder, operation, *, controls=None):
    try:
        result = operation()
    except InterruptedError as exc:
        result = SkillResult(status=SkillStatus.ABORT, result_code="ABORT",
                             reason=str(exc), evidence={"exception": type(exc).__name__})
    except InputActionLimitExceeded as exc:
        evidence = {"exception": type(exc).__name__, "failure_stage": "input_action_budget",
                    "postcondition": "not_checked_budget_exhausted"}
        budget = getattr(controls, "budget", None)
        actions_used = getattr(budget, "actions_used", exc.actions_used)
        if actions_used is not None:
            evidence["input_actions"] = actions_used
        result = SkillResult(status=SkillStatus.FAILED, result_code="INPUT_ACTION_LIMIT",
                             reason=f"max_actions: {exc}", evidence=evidence)
    except SkillDeadlineExceeded as exc:
        evidence = {"exception": type(exc).__name__, "failure_stage": "skill_deadline",
                    "postcondition": "not_checked_skill_timeout"}
        budget = getattr(controls, "budget", None)
        actions_used = getattr(budget, "actions_used", None)
        if actions_used is not None:
            evidence["input_actions"] = actions_used
        result = SkillResult(status=SkillStatus.FAILED, result_code="SKILL_TIMEOUT",
                             reason=f"skill_time_budget_exhausted: {exc}", evidence=evidence)
    except Exception as exc:
        result = SkillResult(status=SkillStatus.FAILED, result_code="FAILED",
                             reason=f"{type(exc).__name__}: {exc}",
                             evidence={"exception": type(exc).__name__})
    return _finish_recorder(recorder, result)


def _approach_result(result: ApproachTargetResult) -> SkillResult:
    state = result.state.upper()
    if state == "REACHED":
        status, result_code = SkillStatus.SUCCESS, "APPROACH_REACHED"
    elif state in {"REACHED_UNCONFIRMED", "UNCERTAIN"}:
        status, result_code = SkillStatus.UNCONFIRMED, "UNCONFIRMED"
    elif state == "ABORT":
        status, result_code = SkillStatus.ABORT, "ABORT"
    else:
        status, result_code = SkillStatus.FAILED, "FAILED"
    return SkillResult(
        status=status,
        result_code=result_code,
        reason=result.reason,
        duration_seconds=result.elapsed_seconds,
        evidence={
            "state": result.state,
            "criterion": result.criterion,
            "last_box": result.last_box,
            "iterations": result.iterations,
            "impulses": result.impulses,
            "input_actions": result.input_actions,
        },
    )


def execute_approach_target(args, *, capture, detector, controls, camera_guard,
                            game_id, stop_check=None, recorder=None, profile=None,
                            settings=None, telemetry=None, initial_target=None):
    params = _validate(ApproachTargetParams, args)
    recorder = recorder or (SkillStepRecorder(telemetry, "approach_target", params.model_dump())
                            if telemetry else None)
    if recorder is None:
        raise ValueError("Skill execution requires a recorder or run telemetry")
    selected_settings = settings or ApproachTargetSettings.from_profile(profile)
    def run():
        raw = ApproachTargetSkill(
            capture=capture,
            target_text=params.target_text,
            detector=detector,
            controls=controls,
            camera_guard=camera_guard,
            game_id=game_id,
            stop_check=stop_check,
            recorder=recorder,
            settings=selected_settings,
            profile=profile,
            initial_target=initial_target,
        ).run()
        return _approach_result(raw)
    return _run_recorded(recorder, run, controls=controls)


def execute_search_target(args, **dependencies):
    params = _validate(SearchTargetParams, args)
    dependencies["recorder"] = _recorder_for("search_target", params.model_dump(), dependencies)
    return _run_recorded(dependencies.get("recorder"),
                         lambda: search_target(**params.model_dump(), **dependencies),
                         controls=dependencies.get("controls"))


def execute_move_pulse(args, **dependencies):
    params = _validate(MovePulseParams, args)
    dependencies["recorder"] = _recorder_for("move_pulse", params.model_dump(), dependencies)
    return _run_recorded(dependencies.get("recorder"),
                         lambda: move_pulse(**params.model_dump(), **dependencies),
                         controls=dependencies.get("controls"))


def execute_turn(args, **dependencies):
    params = _validate(TurnParams, args)
    dependencies["recorder"] = _recorder_for("turn", params.model_dump(), dependencies)
    return _run_recorded(dependencies.get("recorder"),
                         lambda: turn(**params.model_dump(), **dependencies),
                         controls=dependencies.get("controls"))


def execute_interact_until_change(args, **dependencies):
    params = _validate(InteractUntilChangeParams, args)
    dependencies["recorder"] = _recorder_for("interact_until_change", params.model_dump(), dependencies)
    return _run_recorded(dependencies.get("recorder"),
                         lambda: interact_until_change(**params.model_dump(), **dependencies),
                         controls=dependencies.get("controls"))


def execute_wait_for_condition(args, **dependencies):
    params = _validate(WaitForConditionParams, args)
    dependencies["recorder"] = _recorder_for("wait_for_condition", params.model_dump(), dependencies)
    return _run_recorded(dependencies.get("recorder"),
                         lambda: wait_for_condition(**params.model_dump(), **dependencies),
                         controls=dependencies.get("controls"))


def execute_click_menu_target(args, **dependencies):
    params = _validate(ClickMenuTargetParams, args)
    # The goal runner shares this with gameplay skills; menu targeting uses
    # prepare_target/check_target and does not consume the camera guard.
    dependencies.pop("camera_guard", None)
    dependencies["recorder"] = _recorder_for("click_menu_target", params.model_dump(), dependencies)
    return _run_recorded(dependencies.get("recorder"),
                         lambda: click_menu_target(**params.model_dump(), **dependencies),
                         controls=dependencies.get("controls"))


def default_registry(profile=None):
    specs = [
        SkillSpec(
            name="approach_target",
            description="Align with and move toward a visually detected target using bounded camera and movement pulses.",
            params_model=ApproachTargetParams,
            preconditions=("A visible frame is available.", "The detector and input controller are healthy."),
            result_contract=("APPROACH_REACHED", "FAILED", "ABORT", "UNCONFIRMED"),
            timeout_seconds=120,
            resources=("detector", "input", "camera_guard"),
            executor=execute_approach_target,
        ),
        SkillSpec("search_target", "Scan a bounded camera sweep until a visual target is detected.",
                  SearchTargetParams, ("Detector is healthy.",),
                  ("TARGET_DETECTED", "FAILED", "ABORT", "UNCONFIRMED"),
                  60, ("detector", "input", "camera_guard"), execute_search_target),
        SkillSpec("interact_until_change",
                  "Repeat a bounded configured interaction; if the named screen region stays stable, "
                  "leave the world outcome uncertain for postcondition verification.",
                  InteractUntilChangeParams, ("The named region and action exist in the profile.",),
                  ("REGION_CHANGE_UNCONFIRMED", "INTERACTION_OUTCOME_UNCONFIRMED",
                   "SUCCESS_VERIFIED", "FAILED", "ABORT", "UNCONFIRMED"),
                  60, ("input", "capture", "camera_guard"), execute_interact_until_change),
        SkillSpec("move_pulse", "Hold a configured movement direction for one bounded pulse.",
                  MovePulseParams, ("The movement direction exists in the profile.",),
                  tuple(s.value for s in SkillStatus), 10, ("input", "camera_guard"), execute_move_pulse),
        SkillSpec("turn", "Rotate the camera by a bounded relative amount.",
                  TurnParams, ("The profile supports a camera.",), tuple(s.value for s in SkillStatus),
                  10, ("input", "camera_guard"), execute_turn),
        SkillSpec("wait_for_condition", "Wait for a visual query or named region condition to appear or disappear.",
                  WaitForConditionParams, ("A detector or named region is configured.",),
                  ("CONDITION_MET", "FAILED", "ABORT", "UNCONFIRMED"),
                  300, ("capture", "detector"), execute_wait_for_condition),
    ]
    if profile is None or profile.ui_adapters:
        specs.append(SkillSpec("click_menu_target",
                  "Locate one enabled, visible menu control from the current screenshot and send one left click. "
                  "Target label, intent and required next result must agree; runtime verifies the transition from a new screenshot.",
                  ClickMenuTargetParams, ("The profile enables a UI adapter.",
                                          "The current screen is independently verified as a menu."),
                  ("MENU_ACTION_UNCONFIRMED", "FAILED", "ABORT", "UNCONFIRMED"),
                  90, ("capture", "input", "menu_grounding"), execute_click_menu_target))
    return SkillRegistry(specs)
