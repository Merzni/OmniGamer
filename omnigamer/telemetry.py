"""Best-effort, passive telemetry for autonomous runs."""
import ctypes
import json
import logging
import os
import shutil
import subprocess
import threading
import time
from ctypes import wintypes
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from PIL import Image

logger = logging.getLogger(__name__)

STAGE_NAMES = (
    "capture", "memory_lookup", "vlm_decide", "grounding", "input_send",
    "post_wait", "preflight_eval", "result_eval", "memory_write", "total",
)
MAX_PROMPT_CONTEXT_CHARS = 12000
_nvidia_warning_lock = threading.Lock()
_nvidia_warning_emitted = False


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class StepTrace:
    def __init__(self, step, run_step):
        self.started_clock = time.perf_counter()
        self.stage_started = {}
        self.data = {
            "step": step,
            "run_step": run_step,
            "ts_start": utc_now(),
            "ts_end": None,
            "timings_ms": {name: None for name in STAGE_NAMES},
            "decision": {
                "action_type": None, "parameters": None, "decision_calls": 0,
                "vlm_retries": 0, "vlm_http_requests": None, "refusal_reason": None,
                "fallback_reason": None, "raw_json": None, "raw_json_attempts": [], "final": None,
                "action_source": None, "guard": None,
                "http_requests": [], "http_request_counts": {},
            },
            "outcome": {
                "image_delta": None, "verdict": None, "checker_called": False,
                "status": "uncertain",
            },
            "loop": {"no_effect_streak": 0, "camera_only_streak": 0, "repeat_warning": False,
                     "state_repeat": False, "repeat_of_step": None, "look_oscillation": False},
            "memory": {"similar_records": None, "experience_written": False, "lesson_written": False},
            "error": None,
        }
        self.prompt_contexts = []

    @staticmethod
    def update_loop_streaks(no_effect_streak, camera_only_streak, action_type, effective,
                            outcome_status=None):
        """Update action and camera-only streaks without treating camera turns as progress failures."""
        if action_type in {"look", "wait"}:
            next_camera_streak = camera_only_streak + 1
            if outcome_status == "camera_motion":
                return no_effect_streak, next_camera_streak
            return (0 if effective else no_effect_streak + 1), next_camera_streak
        return (0 if effective else no_effect_streak + 1), 0

    def add_prompt_context(self, context_hint, goal=""):
        self.prompt_contexts.append({"context_hint": str(context_hint or ""), "goal": str(goal or "")})

    def prompt_record(self):
        contexts = self.prompt_contexts or [{"context_hint": "", "goal": ""}]
        limited = []
        truncated = False
        for item in contexts:
            context = item["context_hint"]
            if len(context) > MAX_PROMPT_CONTEXT_CHARS:
                context = context[:MAX_PROMPT_CONTEXT_CHARS]
                truncated = True
            limited.append({"context_hint": context, "goal": item["goal"]})
        row = {"step": self.data["step"], **limited[0]}
        if len(limited) > 1:
            row["additional_contexts"] = limited[1:]
        if truncated:
            row["truncated"] = True
        return row

    def start(self, stage):
        if stage not in self.stage_started:
            self.stage_started[stage] = time.perf_counter()

    def stop(self, stage):
        started = self.stage_started.pop(stage, None)
        if started is None:
            return
        current = self.data["timings_ms"].get(stage) or 0.0
        self.data["timings_ms"][stage] = round(current + (time.perf_counter() - started) * 1000, 3)

    def add_ms(self, stage, value):
        current = self.data["timings_ms"].get(stage) or 0.0
        self.data["timings_ms"][stage] = round(current + max(0.0, float(value)), 3)

    def finish(self):
        for stage in tuple(self.stage_started):
            self.stop(stage)
        self.data["ts_end"] = utc_now()
        self.data["timings_ms"]["total"] = round((time.perf_counter() - self.started_clock) * 1000, 3)
        calls = self.data["decision"]["decision_calls"]
        self.data["decision"]["vlm_retries"] = max(0, calls - 1)
        return self.data


class ResourceSampler:
    def __init__(self, recorder, interval_s, memory_provider=None, gpu_provider=None):
        self.recorder = recorder
        self.interval_s = interval_s
        self.memory_provider = memory_provider or process_memory_mb
        self.gpu_provider = gpu_provider or nvidia_smi
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._run, name="OmniGamerResourceSampler", daemon=True)
        self.thread.start()

    def _run(self):
        ram_error_logged = False
        while not self.stop_event.is_set():
            try:
                ram = self.memory_provider()
            except Exception as exc:
                ram = None
                if not ram_error_logged:
                    logger.error("Fallback in ResourceSampler._run at %s (%s): %s", __name__, type(exc).__name__, exc,
                                 exc_info=True)
                    ram_error_logged = True
            try:
                vram, gpu = self.gpu_provider()
            except Exception:
                vram, gpu = None, None
                warn_nvidia_once("nvidia-smi telemetry query failed; VRAM and GPU utilization are null")
            self.recorder.write_resource({
                "timestamp": utc_now(),
                "process_ram_mb": ram,
                "gpu_vram_used_mb": vram,
                "gpu_utilization_pct": gpu,
            })
            self.stop_event.wait(self.interval_s)

    def close(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2.0)


def process_memory_mb():
    """Return current process working set in MiB using Windows' PSAPI."""
    if os.name != "nt":
        return None

    class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    counters = PROCESS_MEMORY_COUNTERS_EX()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    process = kernel32.GetCurrentProcess()
    if not psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return round(counters.WorkingSetSize / (1024 * 1024), 2)


def nvidia_smi():
    global _nvidia_warning_emitted
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=1.5, check=True,
        )
        first_gpu = result.stdout.strip().splitlines()[0]
        memory, utilization = (float(value.strip()) for value in first_gpu.split(",", 1))
        return round(memory, 2), round(utilization, 2)
    except Exception:
        with _nvidia_warning_lock:
            if not _nvidia_warning_emitted:
                logger.warning("nvidia-smi is unavailable; VRAM and GPU utilization telemetry will be null")
                _nvidia_warning_emitted = True
        return None, None


def warn_nvidia_once(message):
    global _nvidia_warning_emitted
    with _nvidia_warning_lock:
        if not _nvidia_warning_emitted:
            logger.warning(message)
            _nvidia_warning_emitted = True


def rotate_run_data(runs_dir, max_count, max_bytes):
    """Prune oldest completed run directories within the managed runs root."""
    root = Path(runs_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    runs = []
    total = 0
    for path in root.iterdir():
        if not path.is_dir() or path.is_symlink() or not (path / "run_meta.json").is_file():
            continue
        size = sum(file.stat().st_size for file in path.rglob("*") if file.is_file())
        runs.append((path, size, path.stat().st_mtime))
        total += size
    runs.sort(key=lambda item: item[2])
    while runs and (len(runs) > max_count or total > max_bytes):
        path, size, _ = runs.pop(0)
        resolved = path.resolve()
        if resolved.parent != root:
            continue
        shutil.rmtree(resolved)
        total -= size


class RunTelemetry:
    def __init__(self, data_dir, enabled=True, frame_every_n=20, resource_interval_s=10.0,
                 frames_max_mb=500.0, resource_sampler_factory=ResourceSampler,
                 runs_max_count=100, runs_max_mb=5000.0):
        self.runs_dir = Path(data_dir) / "runs"
        self.enabled = bool(enabled)
        self.frame_every_n = max(1, int(frame_every_n))
        self.resource_interval_s = float(resource_interval_s)
        self.frames_max_bytes = int(float(frames_max_mb) * 1024 * 1024)
        self.runs_max_count = max(1, int(runs_max_count))
        self.runs_max_bytes = int(float(runs_max_mb) * 1024 * 1024)
        self.resource_sampler_factory = resource_sampler_factory
        self.lock = threading.RLock()
        self.directory = None
        self.meta = None
        self.steps_file = None
        self.resources_file = None
        self.prompts_file = None
        self.resource_sampler = None
        self.run_step_count = 0
        self.active_steps = 0
        self.finish_reason = None
        self.finish_error = None
        self.finalizing = False
        self.closed = True
        self.frame_limit_warned = False
        self.frames_disabled = False
        self.warned = set()

    def _warn_once(self, category, message):
        if category not in self.warned:
            self.warned.add(category)
            logger.warning("Telemetry: %s", message)

    def start(self, metadata):
        if not self.enabled:
            return None
        with self.lock:
            if not self.closed:
                return str(self.directory)
            self.closed = False
            self.finish_reason = None
            self.finish_error = None
            self.finalizing = False
            self.run_step_count = 0
            self.active_steps = 0
            self.frame_limit_warned = False
            self.frames_disabled = False
            try:
                self.runs_dir.mkdir(parents=True, exist_ok=True)
                rotate_run_data(self.runs_dir, self.runs_max_count - 1, self.runs_max_bytes)
                run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:6]
                self.directory = self.runs_dir / run_id
                self.directory.mkdir(parents=True, exist_ok=False)
                (self.directory / "frames").mkdir()
                self.steps_file = (self.directory / "steps.jsonl").open("a", encoding="utf-8", buffering=1)
                self.resources_file = (self.directory / "resources.jsonl").open("a", encoding="utf-8", buffering=1)
                self.prompts_file = (self.directory / "prompts.jsonl").open("a", encoding="utf-8", buffering=1)
                self.meta = {"run_id": run_id, "started_at": utc_now(), "ended_at": None,
                             "termination_reason": None, "steps_recorded": 0, **metadata}
                self._write_meta_locked()
                self.resource_sampler = self.resource_sampler_factory(self, self.resource_interval_s)
                self.resource_sampler.start()
                return str(self.directory)
            except Exception as exc:
                self._warn_once("failure", f"could not start run recording ({type(exc).__name__}: {exc})")
                self._disable_locked()
                return None

    def begin_step(self, step):
        self.run_step_count += 1
        self.active_steps += 1
        return StepTrace(step, self.run_step_count)

    def update_metadata(self, **values):
        if self.closed or not self.meta:
            return
        with self.lock:
            self.meta.update(values)
            self._write_meta_locked()

    def write_resource(self, resource):
        if self.closed or not self.resources_file:
            return
        try:
            line = json.dumps(resource, ensure_ascii=False, separators=(",", ":")) + "\n"
            with self.lock:
                if self.resources_file:
                    self.resources_file.write(line)
                    self.resources_file.flush()
        except Exception as exc:
            self._warn_once("failure", f"resource write failed ({type(exc).__name__}: {exc})")

    def finish_step(self, trace, before_frame=None, after_frame=None):
        try:
            data = trace.finish()
        except Exception as exc:
            self._warn_once("failure", f"step finalization failed ({type(exc).__name__}: {exc})")
            data = getattr(trace, "data", {})
        try:
            with self.lock:
                if self.steps_file and data:
                    self.steps_file.write(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n")
                    self.steps_file.flush()
                    self.meta["steps_recorded"] += 1
                    self._write_meta_locked()
                if self.prompts_file:
                    self.prompts_file.write(json.dumps(trace.prompt_record(), ensure_ascii=False,
                                                       separators=(",", ":")) + "\n")
                    self.prompts_file.flush()
                if data:
                    self._save_frames_locked(data, before_frame, after_frame)
        except Exception as exc:
            self._warn_once("failure", f"step/frame write failed ({type(exc).__name__}: {exc})")
        finally:
            with self.lock:
                self.active_steps = max(0, self.active_steps - 1)
                should_finish = self.finish_reason is not None and self.active_steps == 0
            if should_finish:
                self._finalize()
        return data

    def _save_frames_locked(self, data, before_frame, after_frame):
        run_step = data["run_step"]
        status = data["outcome"].get("status")
        selected = (run_step == 1 or run_step % self.frame_every_n == 0
                    or data.get("error") is not None or status in {"uncertain", "refusal"})
        if not selected or self.frames_disabled or not self.directory:
            return
        images = [("before", before_frame), ("after", after_frame)]
        encoded = []
        for label, frame in images:
            if frame is None:
                continue
            image = frame.copy()
            image.thumbnail((640, 640), Image.Resampling.LANCZOS)
            from io import BytesIO
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=80)
            encoded.append((label, buffer.getvalue()))
        frames_dir = self.directory / "frames"
        current_size = sum(path.stat().st_size for path in frames_dir.glob("*.jpg"))
        if current_size + sum(len(blob) for _, blob in encoded) > self.frames_max_bytes:
            self.frames_disabled = True
            if not self.frame_limit_warned:
                self.frame_limit_warned = True
                self._warn_once("frame_limit", "frames size limit reached; further frame saving disabled")
            return
        for label, blob in encoded:
            (frames_dir / f"step_{run_step:06d}_{label}.jpg").write_bytes(blob)

    def finish(self, reason, error_message=None):
        if self.closed:
            return
        with self.lock:
            if reason == "error" or error_message is not None:
                self.finish_reason = str(reason)
                self.finish_error = str(error_message)[:300] if error_message else None
            elif self.finish_reason is None:
                self.finish_reason = str(reason)
                self.finish_error = str(error_message)[:300] if error_message else None
            should_finish = self.active_steps == 0
        if should_finish:
            self._finalize()

    def _finalize(self):
        with self.lock:
            if self.closed or self.finalizing or self.active_steps:
                return
            self.finalizing = True
            sampler = self.resource_sampler
        if sampler:
            try:
                sampler.close()
            except Exception as exc:
                self._warn_once("failure", f"resource sampler close failed ({type(exc).__name__}: {exc})")
        with self.lock:
            try:
                if self.steps_file:
                    self.steps_file.flush()
                    self.steps_file.close()
                    self.steps_file = None
                if self.resources_file:
                    self.resources_file.flush()
                    self.resources_file.close()
                    self.resources_file = None
                if self.prompts_file:
                    self.prompts_file.flush()
                    self.prompts_file.close()
                    self.prompts_file = None
                if self.meta:
                    self.meta["ended_at"] = utc_now()
                    self.meta["termination_reason"] = self.finish_reason or "error"
                    if self.finish_error:
                        self.meta["error"] = self.finish_error
                    self._write_meta_locked(close=False)
            except Exception as exc:
                self._warn_once("failure", f"run finalization failed ({type(exc).__name__}: {exc})")
            self.closed = True
            self.finalizing = False

    def _write_meta_locked(self, close=True):
        if not self.directory or not self.meta:
            return
        temporary = self.directory / "run_meta.json.tmp"
        temporary.write_text(json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.directory / "run_meta.json")

    def _disable_locked(self):
        for stream_name in ("steps_file", "resources_file", "prompts_file"):
            stream = getattr(self, stream_name)
            if stream:
                try:
                    stream.close()
                except Exception:
                    pass
                setattr(self, stream_name, None)
        self.closed = True

