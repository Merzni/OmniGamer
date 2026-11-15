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
from .contracts import SkillRegistry, SkillResult, SkillSpec, SkillStatus
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


def _run_recorded(recorder, operation):
    try:
        result = operation()
    except InterruptedError as exc:
        result = SkillResult(status=SkillStatus.ABORT, result_code="ABORT",
                             reason=str(exc), evidence={"exception": type(exc).__name__})
    except Exception as exc:
        result = SkillResult(status=SkillStatus.FAILED, result_code="FAILED",
                             reason=f"{type(exc).__name__}: {exc}",
                             evidence={"exception": type(exc).__name__})
    return _finish_recorder(recorder, result)


def _approach_result(result: ApproachTargetResult) -> SkillResult:
    state = result.state.upper()
    if state == "REACHED":
        status, result_code = SkillStatus.SUCCESS, "SUCCESS"
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
                            settings=None, telemetry=None):
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
        ).run()
        return _approach_result(raw)
    return _run_recorded(recorder, run)


def execute_search_target(args, **dependencies):
    params = _validate(SearchTargetParams, args)
    dependencies["recorder"] = _recorder_for("search_target", params.model_dump(), dependencies)
    return _run_recorded(dependencies.get("recorder"),
                         lambda: search_target(**params.model_dump(), **dependencies))


def execute_move_pulse(args, **dependencies):
    params = _validate(MovePulseParams, args)
    dependencies["recorder"] = _recorder_for("move_pulse", params.model_dump(), dependencies)
    return _run_recorded(dependencies.get("recorder"),
                         lambda: move_pulse(**params.model_dump(), **dependencies))


def execute_turn(args, **dependencies):
    params = _validate(TurnParams, args)
    dependencies["recorder"] = _recorder_for("turn", params.model_dump(), dependencies)
    return _run_recorded(dependencies.get("recorder"),
                         lambda: turn(**params.model_dump(), **dependencies))


def execute_interact_until_change(args, **dependencies):
    params = _validate(InteractUntilChangeParams, args)
    dependencies["recorder"] = _recorder_for("interact_until_change", params.model_dump(), dependencies)
    return _run_recorded(dependencies.get("recorder"),
                         lambda: interact_until_change(**params.model_dump(), **dependencies))


def execute_wait_for_condition(args, **dependencies):
    params = _validate(WaitForConditionParams, args)
    dependencies["recorder"] = _recorder_for("wait_for_condition", params.model_dump(), dependencies)
    return _run_recorded(dependencies.get("recorder"),
                         lambda: wait_for_condition(**params.model_dump(), **dependencies))


def default_registry():
    return SkillRegistry((
        SkillSpec(
            name="approach_target",
            description="Align with and move toward a visually detected target using bounded camera and movement pulses.",
            params_model=ApproachTargetParams,
            preconditions=("A visible frame is available.", "The detector and input controller are healthy."),
            result_contract=tuple(status.value for status in SkillStatus),
            timeout_seconds=120,
            resources=("detector", "input", "camera_guard"),
            executor=execute_approach_target,
        ),
        SkillSpec("search_target", "Scan a bounded camera sweep until a visual target is detected.",
                  SearchTargetParams, ("Detector is healthy.",), tuple(s.value for s in SkillStatus),
                  60, ("detector", "input", "camera_guard"), execute_search_target),
        SkillSpec("interact_until_change", "Repeat a configured interaction until a named screen region changes.",
                  InteractUntilChangeParams, ("The named region and action exist in the profile.",),
                  ("REGION_CHANGE_UNCONFIRMED", "SUCCESS_VERIFIED", "FAILED", "ABORT", "UNCONFIRMED"),
                  60, ("input", "capture", "camera_guard"), execute_interact_until_change),
        SkillSpec("move_pulse", "Hold a configured movement direction for one bounded pulse.",
                  MovePulseParams, ("The movement direction exists in the profile.",),
                  tuple(s.value for s in SkillStatus), 10, ("input", "camera_guard"), execute_move_pulse),
        SkillSpec("turn", "Rotate the camera by a bounded relative amount.",
                  TurnParams, ("The profile supports a camera.",), tuple(s.value for s in SkillStatus),
                  10, ("input", "camera_guard"), execute_turn),
        SkillSpec("wait_for_condition", "Wait for a visual query or named region condition to appear or disappear.",
                  WaitForConditionParams, ("A detector or named region is configured.",),
                  tuple(s.value for s in SkillStatus), 300, ("capture", "detector"), execute_wait_for_condition),
    ))
