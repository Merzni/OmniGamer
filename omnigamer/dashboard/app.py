"""Web dashboard server for OmniGamer."""
import io
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Literal
from PIL import Image, ImageDraw

from omnigamer.capture import list_visible_windows, WindowCapture
from omnigamer.agent import OmniAgent
from omnigamer.vision_brain import VisionDecisionError

app = FastAPI(title="OmniGamer Live Dashboard")

# Global singleton agent
agent: OmniAgent = None

class AttachRequest(BaseModel):
    hwnd: int
    game_id: str = ""
    goal: str = ""

class BackendRequest(BaseModel):
    backend: Literal["window","native"]

@app.post("/api/input-backend")
def set_input_backend(req: BackendRequest):
    if not agent:raise HTTPException(500,"Agent not initialized")
    agent.set_input_backend(req.backend)
    return {"input_backend":agent.input_backend}

@app.on_event("startup")
def startup():
    global agent
    agent = OmniAgent()

@app.on_event("shutdown")
def shutdown():
    if agent:
        agent.end_telemetry_run("panel_stop")

@app.get("/api/windows")
def get_windows():
    """List all open desktop windows."""
    return list_visible_windows()

@app.post("/api/attach")
def attach_window(req: AttachRequest):
    """Switch agent target window."""
    started_run = False
    try:
        agent.goal = str(req.goal or "").strip()[:500]
        started_run = agent.begin_telemetry_run()
        agent.telemetry.update_metadata(goal=agent.goal)
        agent.attach(req.hwnd, req.game_id)
        return {"status": "ok", "hwnd": req.hwnd, "game_id": agent.game_id, "goal": agent.goal}
    except Exception as exc:
        if started_run or not agent.telemetry.closed:
            agent.end_telemetry_run("error", error_message=f"{type(exc).__name__}: {exc}")
        raise

@app.post("/api/step")
def trigger_step(single_step: bool = False):
    """Execute one autonomous step."""
    if not agent:
        raise HTTPException(500, "Agent not initialized")
    if not agent.capture.hwnd:
        raise HTTPException(409, "Attach a game window before executing a step")
    try:
        agent.begin_telemetry_run()
        result = agent.step()
        if single_step:
            agent.end_telemetry_run("single_step")
        return result
    except Exception as exc:
        if single_step or not isinstance(exc, VisionDecisionError):
            reason = (agent._stop_reason or "f12") if isinstance(exc, InterruptedError) else "error"
            message = (agent.last_emergency_hotkey_error if reason == "emergency_hotkey_error"
                       else f"{type(exc).__name__}: {exc}" if reason == "error" else None)
            agent.end_telemetry_run(reason, error_message=message)
        if isinstance(exc, (RuntimeError, InterruptedError, ValueError)):
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        raise

@app.post("/api/telemetry/stop")
def stop_telemetry(reason: str = "panel_stop"):
    if not agent:
        raise HTTPException(500, "Agent not initialized")
    allowed = {"panel_stop", "error", "f12", "emergency_hotkey_error", "step_limit", "single_step"}
    if reason not in allowed:
        raise HTTPException(400, "Unsupported telemetry termination reason")
    if agent._stop_event.is_set():
        reason = agent._stop_reason or "f12"
    agent.end_telemetry_run(reason)
    return {"status": "ok", "reason": reason}

@app.get("/api/state")
def get_state():
    """Get current agent state, last commentary, and memory statistics."""
    if not agent:
        return {}
    
    stats = agent.memory.get_stats(agent.game_id)
    decision = agent.last_decision.model_dump() if agent.last_decision else None
    
    return {
        "step_count": agent.step_count,
        "is_running": agent.is_running,
        "last_decision": decision,
        "last_delta": agent.last_delta,
        "stats": stats,
        "game_id": agent.game_id,
        "memory_context": agent.last_memory_context,
        "memory_warning": agent.last_memory_warning,
        "last_error": agent.last_error,
        "emergency_hotkey_error": (agent.emergency_hotkey.last_error
                                    if agent.emergency_hotkey else ""),
        "timings": agent.last_timings,
        "action_source": agent.last_action_source,
        "supervision": agent.supervision_state(),
        "assessment": agent.last_assessment,
        "assessment_error": agent.last_assessment_error,
        "guard_reason": agent.last_guard_reason,
        "input_backend": agent.input_backend,
        "bot_pointer": agent.pointer.snapshot(),
        "menu_memory": agent.last_menu_memory,
        "camera_feedback": agent.last_camera_feedback,
        "dialogues": agent.persona.recent_dialogues[-10:]
    }

@app.get("/api/live.jpg")
def get_live_frame():
    """Fresh preview, independent of slow decisions; never focus or send input."""
    if not agent or not agent.capture.hwnd:
        raise HTTPException(409, "Attach a game window for live preview")
    preview = WindowCapture(hwnd=agent.capture.hwnd)
    try:
        if agent.input_backend == "window":preview.assert_visible()
        else:preview.assert_foreground()
        img = preview.grab_frame().copy()
        if agent.input_backend == "window":preview.assert_visible()
        else:preview.assert_foreground()
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    img.thumbnail((960, 960))
    pointer=agent.pointer.snapshot()
    if pointer.get("visible"):
        draw=ImageDraw.Draw(img)
        x=int(pointer["x"]*img.width/1000);y=int(pointer["y"]*img.height/1000)
        radius=18 if pointer.get("pressed") else 11
        draw.ellipse((x-radius,y-radius,x+radius,y+radius),outline="#ff28b4",width=3)
        draw.line((x-22,y,x+22,y),fill="#ff28b4",width=2)
        draw.line((x,y-22,x,y+22),fill="#ff28b4",width=2)
        draw.text((x+14,y+12),"AI",fill="#ff8bd8")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

@app.get("/api/frame.jpg")
def get_frame():
    """Get current screen frame with overlay of agent gaze / action."""
    if not agent or not agent.last_frame:
        # Fallback dummy frame
        img = Image.new("RGB", (640, 360), color=(30, 30, 35))
    else:
        img = agent.last_frame.copy()
        
        # Draw overlay depending on action type
        if agent.last_decision:
            draw = ImageDraw.Draw(img)
            w, h = img.size
            act = agent.last_decision.action_type
            
            if act in ("click", "double_click") and agent.last_decision.x is not None and agent.last_decision.y is not None:
                cx = int((agent.last_decision.x / 1000.0) * w)
                cy = int((agent.last_decision.y / 1000.0) * h)
                r = 16
                col = "red" if act == "click" else "yellow"
                draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=col, width=3)
                draw.line((cx - r*1.5, cy, cx + r*1.5, cy), fill=col, width=2)
                draw.line((cx, cy - r*1.5, cx, cy + r*1.5), fill=col, width=2)
                draw.text((cx + 20, cy - 10), f"{act.upper()}: {agent.last_decision.intent}", fill="white")
            elif act == "look":
                dx = agent.last_decision.x or 0
                arrow_dir = "➡️ RIGHT" if dx > 0 else "⬅️ LEFT"
                draw.text((w // 2 - 60, h // 2 - 20), f"LOOK {arrow_dir} ({dx})", fill="cyan")
            elif act == "walk":
                k = agent.last_decision.key or "W"
                dur = agent.last_decision.duration or 1.5
                draw.text((w // 2 - 50, h // 2 + 30), f"🚶 WALK [{k.upper()}] {dur:.1f}s", fill="green")
            elif act == "jump":
                draw.text((w // 2 - 80, h // 2), f"JUMP [{agent.last_decision.key or 'w'} + SPACE]", fill="green")
            elif act == "mine":
                dur = agent.last_decision.duration or 2.5
                draw.text((w // 2 - 50, h // 2), f"⛏️ MINING (Hold LMB) {dur:.1f}s", fill="orange")
            elif act == "use":
                draw.text((w // 2 - 30, h // 2), "🖐️ USE (RMB)", fill="magenta")

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")

# Mount static frontend
static_dir = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
