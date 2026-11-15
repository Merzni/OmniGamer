"""Per-iteration telemetry and bounded frame output for explicit skills."""
import json
import logging
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)


class SkillStepRecorder:
    """Append versioned, skill-neutral JSONL records to a telemetry run."""

    def __init__(self, run_telemetry, skill_name, args=None):
        self.telemetry = run_telemetry
        self.directory = Path(run_telemetry.directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.steps_file = (self.directory / "skill_steps.jsonl").open("a", encoding="utf-8", buffering=1)
        self.skill_name = str(skill_name)
        self.args = args or {}
        self._step = 0

    def __call__(self, row, before=None, after=None):
        self._step += 1
        normalized = dict(row)
        normalized.setdefault("schema_version", 1)
        normalized.setdefault("skill", self.skill_name)
        normalized.setdefault("iteration", self._step)
        normalized.setdefault("args", self.args)
        normalized.setdefault("phase", "execute")
        normalized["event"] = "step"
        normalized["step_state"] = normalized.pop("state", "")
        normalized["step_evidence"] = normalized.pop("evidence", {})
        normalized.pop("result", None)
        normalized.setdefault("result", None)
        try:
            self.steps_file.write(json.dumps(normalized, ensure_ascii=False, separators=(",", ":")) + "\n")
            self.steps_file.flush()
        except Exception as exc:
            logger.warning("Skill step write failed (%s): %s", type(exc).__name__, exc)

    def finish(self, result):
        payload = result.model_dump(mode="json") if hasattr(result, "model_dump") else result
        self._step += 1
        row = {"schema_version": 1, "skill": self.skill_name, "iteration": self._step,
               "phase": "finish", "event": "finish", "args": self.args, "result": payload}
        try:
            self.steps_file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            self.steps_file.flush()
        except Exception as exc:
            logger.warning("Skill result write failed (%s): %s", type(exc).__name__, exc)
        self.close()

    def finish_error(self, error):
        self.finish({
            "status": "FAILED",
            "result_code": "FAILED",
            "reason": f"{type(error).__name__}: {error}",
            "evidence": {"exception": type(error).__name__},
            "duration_seconds": 0,
        })

    def close(self):
        if self.steps_file and not self.steps_file.closed:
            self.steps_file.flush()
            self.steps_file.close()


class ApproachTargetRecorder:
    manages_run_finalization = True

    def __init__(self, run_telemetry, settings, target_text):
        self.telemetry = run_telemetry
        self.directory = Path(run_telemetry.directory)
        self.frames_dir = self.directory / "frames"
        self.frames_dir.mkdir(exist_ok=True)
        self.steps_file = (self.directory / "skill_steps.jsonl").open("a", encoding="utf-8", buffering=1)
        self.settings = settings
        self.target_text = target_text
        self._frames_truncated = False

    def _jpeg(self, image, box=None):
        preview = image.convert("RGB").copy()
        preview.thumbnail((640, 640), Image.Resampling.LANCZOS)
        if box:
            draw = ImageDraw.Draw(preview)
            x1 = int(max(0, min(1, box["x1"])) * preview.width)
            y1 = int(max(0, min(1, box["y1"])) * preview.height)
            x2 = int(max(0, min(1, box["x2"])) * preview.width)
            y2 = int(max(0, min(1, box["y2"])) * preview.height)
            draw.rectangle((x1, y1, x2, y2), outline=(255, 70, 70), width=max(2, preview.width // 320))
            draw.text((x1, max(0, y1 - 16)), f"{self.target_text} {box['score']:.3f}",
                      fill=(255, 255, 0), stroke_width=1, stroke_fill=(0, 0, 0))
        buffer = BytesIO()
        preview.save(buffer, format="JPEG", quality=80)
        return buffer.getvalue()

    def _save_frame(self, name, image, box=None):
        if image is None or self._frames_truncated:
            return None
        try:
            data = self._jpeg(image, box)
            used = sum(path.stat().st_size for path in self.frames_dir.glob("*.jpg"))
            if used + len(data) > self.telemetry.frames_max_bytes:
                self._frames_truncated = True
                return None
            path = self.frames_dir / name
            path.write_bytes(data)
            return str(path.relative_to(self.directory))
        except Exception as exc:
            logger.warning("Skill frame write failed (%s): %s", type(exc).__name__, exc)
            return None

    def __call__(self, row, before, after):
        step = int(row["iteration"])
        row = dict(row)
        row.update(schema_version=1, skill="approach_target", phase="execute")
        row.setdefault("args", {"target_text": self.target_text})
        row["event"] = "step"
        row["step_state"] = row.pop("state", "")
        row["step_evidence"] = {"box": row.get("box"), "h": row.get("h"),
                                "detector": row.get("detector")}
        row["result"] = None
        row["frames"] = {
            "before": self._save_frame(f"skill_step_{step:06d}_before.jpg", before),
            "after": self._save_frame(f"skill_step_{step:06d}_after.jpg", after),
        }
        try:
            self.steps_file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            self.steps_file.flush()
        except Exception as exc:
            logger.warning("Skill step write failed (%s): %s", type(exc).__name__, exc)

    def finish(self, result, final_frame=None, finalize_run=True):
        if hasattr(result, "model_dump"):
            payload = result.model_dump(mode="json")
            state = getattr(result.status, "value", str(result.status))
            final_box = (result.evidence or {}).get("last_box")
            reason = result.reason
            common_result = payload
        else:
            payload = result.to_dict() if result else None
            state = result.state if result else "error"
            final_box = result.last_box if result else None
            reason = result.reason if result else "Skill terminated before a result was produced"
            status = ("SUCCESS" if state == "REACHED" else
                      "UNCONFIRMED" if state == "REACHED_UNCONFIRMED" else
                      "ABORT" if state == "ABORT" else "FAILED")
            common_result = {
                "status": status,
                "result_code": status,
                "reason": reason,
                "evidence": payload or {},
                "duration_seconds": result.elapsed_seconds if result else 0,
            }
        final_path = self._save_frame("skill_final.jpg", final_frame, final_box)
        if self.steps_file and not self.steps_file.closed:
            summary = {"schema_version": 1, "skill": "approach_target",
                       "iteration": int(getattr(result, "iterations", self._step_count())) + 1 if result else 0,
                       "phase": "finish", "event": "finish",
                       "args": {"target_text": self.target_text},
                       "result": common_result}
            self.steps_file.write(json.dumps(summary, ensure_ascii=False, separators=(",", ":")) + "\n")
            self.steps_file.flush()
            self.steps_file.close()
        if result:
            self.telemetry.update_metadata(
                skill_result=payload,
                skill_final_frame=final_path,
                skill_frames_truncated=self._frames_truncated,
            )
            termination_reason = f"skill_{str(state).casefold()}"
            error_message = None
        else:
            self.telemetry.update_metadata(skill_frames_truncated=self._frames_truncated)
            termination_reason, error_message = "error", reason
        if finalize_run:
            self.telemetry.finish(termination_reason, error_message=error_message)

    def _step_count(self):
        try:
            with (self.directory / "skill_steps.jsonl").open(encoding="utf-8") as stream:
                return sum(1 for _ in stream)
        except OSError:
            return 0

    def finish_error(self, exc, finalize_run=True):
        if self.steps_file and not self.steps_file.closed:
            summary = {"schema_version": 1, "skill": "approach_target",
                       "iteration": self._step_count() + 1, "phase": "finish",
                       "event": "finish", "args": {"target_text": self.target_text},
                       "result": {"status": "FAILED", "result_code": "FAILED",
                                  "reason": f"{type(exc).__name__}: {exc}",
                                  "evidence": {"exception": type(exc).__name__}}}
            self.steps_file.write(json.dumps(summary, ensure_ascii=False, separators=(",", ":")) + "\n")
            self.steps_file.flush()
            self.steps_file.close()
        self.telemetry.update_metadata(skill_frames_truncated=self._frames_truncated)
        if finalize_run:
            self.telemetry.finish("error", error_message=f"{type(exc).__name__}: {exc}")
