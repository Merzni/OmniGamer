"""Read-only local readiness checks; this module never sends game input."""
from __future__ import annotations

import argparse
import ctypes
import importlib.metadata
import os
import shutil
import subprocess
from urllib.parse import urlsplit, urlunsplit
from ctypes import wintypes

import requests
from .capture import WindowCapture, find_window_by_title
from .config import config
from .detector_client import DetectorClient
from .planner import GoalPlanner
from .profiles import load_profile
from .skills.registry import default_registry
from .vision_brain import VisionBrain


def check_input_desktop_access():
    """Check Windows input-desktop object access without sending input."""
    if os.name != "nt":
        return False, "Windows input desktop is unavailable on this platform"
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.OpenInputDesktop.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    user32.OpenInputDesktop.restype = wintypes.HANDLE
    user32.CloseDesktop.argtypes = (wintypes.HANDLE,)
    user32.CloseDesktop.restype = wintypes.BOOL
    # DESKTOP_READOBJECTS | DESKTOP_WRITEOBJECTS. This opens a handle only;
    # it does not synthesize keyboard or mouse events.
    desktop = user32.OpenInputDesktop(0, False, 0x0001 | 0x0080)
    if not desktop:
        return False, f"OpenInputDesktop failed with WinError {ctypes.get_last_error()}"
    try:
        return True, "current input desktop is accessible"
    finally:
        user32.CloseDesktop(desktop)


def _health_url(completions_url):
    parts = urlsplit(completions_url)
    return urlunsplit((parts.scheme, parts.netloc, "/health", "", ""))


def run_doctor(game_id="", profile_path=None, window_filter="", *, vlm=None,
               detector=None, get=requests.get, disk_usage=shutil.disk_usage,
               run=subprocess.run, input_probe=check_input_desktop_access):
    checks = []
    def add(name, ok, detail):
        checks.append({"name": name, "status": "OK" if ok else "ПРОБЛЕМА", "detail": str(detail)})
    try:
        profile = load_profile(game_id, profile_path)
        add("Профиль", True, f"{profile.game_id}: {len(profile.regions)} областей")
        add("Области", bool(profile.regions), ", ".join(sorted(profile.regions)) or "не заданы")
    except Exception as exc:
        add("Профиль", False, f"{type(exc).__name__}: {exc}")
        profile = None

    window = None
    if profile:
        title = window_filter or profile.window
        if title:
            try:
                window = find_window_by_title(title)
                add("Окно", bool(window), window.get("title", "не найдено") if window else f"не найдено: {title}")
            except Exception as exc:
                add("Окно", False, f"{type(exc).__name__}: {exc}")
        else:
            add("Окно", False, "задайте --window или поле window в профиле")

    image = None
    if window:
        try:
            image = WindowCapture(hwnd=window["hwnd"]).grab_frame()
            add("Захват", image is not None, f"{image.width}x{image.height}" if image else "кадр отсутствует")
        except Exception as exc:
            add("Захват", False, f"{type(exc).__name__}: {exc}")

    try:
        response = get(_health_url(config.vlm_api_url), timeout=2)
        response.raise_for_status()
        add("VLM /health", True, response.status_code)
    except Exception as exc:
        add("VLM /health", False, f"{type(exc).__name__}: {exc}")
    if image is not None and profile:
        try:
            brain = vlm or VisionBrain()
            planner = GoalPlanner(default_registry(), brain.plan_goal)
            planner.plan(frame=image, goal="diagnostic probe", profile=profile)
            add("VLM probe", True, "план прошёл JSON Schema и схему навыков")
        except Exception as exc:
            add("VLM probe", False, f"{type(exc).__name__}: {exc}")
    else:
        add("VLM probe", False, "пропущен: нужен захваченный кадр")

    try:
        service = detector or DetectorClient()
        health = service.health()
        add("Детектор /health", bool(health), health or "нет ответа")
        if health and image is not None:
            detection = service.detect(image, ["visible object"])
            add("Детектор probe", detection is not None,
                f"ответ получен; boxes={len(detection.get('boxes', []))}" if detection else "кадр не обработан")
        else:
            add("Детектор probe", False, "пропущен: нет healthy-сервиса или кадра")
    except Exception as exc:
        add("Детектор", False, f"{type(exc).__name__}: {exc}")

    try:
        result = run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
                      "--format=csv,noheader,nounits"], capture_output=True,
                     text=True, timeout=3, check=True)
        add("GPU / VRAM", True, result.stdout.strip().splitlines()[0])
    except Exception as exc:
        add("GPU / VRAM", False, f"nvidia-smi недоступен: {type(exc).__name__}")
    try:
        free_gb = disk_usage(config.data_dir).free / (1024 ** 3)
        add("Свободное место", free_gb >= 1, f"{free_gb:.1f} ГБ")
    except Exception as exc:
        add("Свободное место", False, f"{type(exc).__name__}: {exc}")
    for package in ("Python", "pydantic", "Pillow", "requests", "numpy"):
        try:
            version = " ".join(__import__("sys").version.split()[:1]) if package == "Python" else importlib.metadata.version(package)
            add(f"Версия {package}", True, version)
        except Exception as exc:
            add(f"Версия {package}", False, f"{type(exc).__name__}")
    try:
        accessible, detail = input_probe()
        add("Доступ к input desktop", accessible,
            f"backend={profile.input_backend if profile else 'неизвестен'}; {detail}; SendInput не отправлялся")
    except Exception as exc:
        add("Доступ к input desktop", False, f"{type(exc).__name__}: {exc}; SendInput не отправлялся")
    return checks


def main(argv=None):
    parser = argparse.ArgumentParser(description="Read-only OmniGamer environment diagnostics")
    parser.add_argument("--game-id", default="")
    parser.add_argument("--profile", default="")
    parser.add_argument("--window", default="")
    args = parser.parse_args(argv)
    checks = run_doctor(args.game_id, args.profile or None, args.window)
    width = max(len(row["name"]) for row in checks)
    for row in checks:
        print(f"{row['status']:<9} {row['name']:<{width}}  {row['detail']}")
    return 1 if any(row["status"] == "ПРОБЛЕМА" for row in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
