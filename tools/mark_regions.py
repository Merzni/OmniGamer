"""Locate and explicitly confirm normalized HUD/UI regions for a game profile."""
import argparse
import base64
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests
from PIL import Image, ImageDraw
from pydantic import ValidationError

from omnigamer.config import config
from omnigamer.personality import configure_stdio
from omnigamer.profiles import GameProfile, NormalizedRegion, load_profile


REGION_SCHEMA = {
    "type": "object",
    "properties": {
        "x1": {"type": "number", "minimum": 0, "maximum": 1},
        "y1": {"type": "number", "minimum": 0, "maximum": 1},
        "x2": {"type": "number", "minimum": 0, "maximum": 1},
        "y2": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["x1", "y1", "x2", "y2"],
    "additionalProperties": False,
}


def parse_candidate(content):
    if not isinstance(content, str):
        raise ValueError("VLM region result must be a JSON string")
    try:
        decoded = json.loads(content)
        return NormalizedRegion.model_validate(decoded)
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        raise ValueError(f"VLM returned an invalid normalized region: {exc}") from exc


def parse_manual_coords(value):
    try:
        parts = [float(part.strip()) for part in value.split(",")]
    except (AttributeError, ValueError) as exc:
        raise ValueError("--coords must be x1,y1,x2,y2 in normalized 0..1 coordinates") from exc
    if len(parts) != 4:
        raise ValueError("--coords must contain exactly four comma-separated values")
    try:
        return NormalizedRegion(x1=parts[0], y1=parts[1], x2=parts[2], y2=parts[3])
    except ValidationError as exc:
        raise ValueError(f"--coords are outside the normalized region schema: {exc}") from exc


def locate_region(frame, description, api_url=None, model_name=None, timeout=None):
    resized = frame.convert("RGB").copy()
    resized.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    resized.save(buffer, format="JPEG", quality=85)
    data_uri = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    prompt = (
        "Find the tightest axis-aligned rectangle containing only the visible screen region described below. "
        "Return normalized floating point coordinates from 0 to 1 relative to the FULL image, with x1<x2 and y1<y2. "
        "If the region is not clearly visible, return the full image rectangle and do not guess a smaller one. "
        f"Description: {description}"
    )
    response = requests.post(
        api_url or config.vlm_api_url,
        json={
            "model": model_name or config.vlm_model_name,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ]}],
            "temperature": 0,
            "max_tokens": 120,
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "normalized_region", "strict": True, "schema": REGION_SCHEMA}},
        },
        timeout=timeout or config.request_timeout,
    )
    response.raise_for_status()
    candidate = parse_candidate(response.json()["choices"][0]["message"]["content"])
    if (candidate.x1, candidate.y1, candidate.x2, candidate.y2) == (0, 0, 1, 1):
        raise ValueError("VLM could not localize this region; supply manual --coords")
    return candidate


def annotate_candidate(frame, region, label):
    image = frame.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    x1, y1 = round(region.x1 * image.width), round(region.y1 * image.height)
    x2, y2 = round(region.x2 * image.width), round(region.y2 * image.height)
    draw.rectangle((x1, y1, x2, y2), outline=(255, 40, 100), width=max(2, image.width // 300))
    draw.text((x1, max(0, y1 - 18)), label, fill=(255, 255, 0),
              stroke_width=1, stroke_fill=(0, 0, 0))
    return image


def write_region(profile_path, game_id, region_name, region):
    path = Path(profile_path)
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    else:
        raw = GameProfile(game_id=game_id).model_dump(mode="json", exclude={"comment"})
    if raw.get("game_id", "").casefold() != game_id.casefold():
        raise ValueError("Profile game_id does not match the requested game_id")
    raw.setdefault("regions", {})[region_name] = region.model_dump()
    GameProfile.model_validate(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def ask_accept_candidate():
    """Ask before persisting a candidate; closed stdin safely means reject."""
    try:
        return input("Записать эту область в профиль? [y/N] ").strip().casefold() in {"y", "yes", "д", "да"}
    except EOFError:
        return False


def _window_frame(window_filter):
    from omnigamer.capture import WindowCapture, find_window_by_title

    window = find_window_by_title(window_filter)
    if not window:
        raise RuntimeError(f"No visible window matches {window_filter!r}")
    capture = WindowCapture(hwnd=window["hwnd"])
    return capture.grab_frame()


def main(argv=None):
    configure_stdio()
    parser = argparse.ArgumentParser(description="Mark a normalized visual region in a game profile")
    parser.add_argument("--game-id", required=True)
    parser.add_argument("--profile", default="", help="Profile JSON path; defaults to data/profiles/<game-id>.json")
    parser.add_argument("--window", default="", help="Window title substring; defaults to the profile window")
    parser.add_argument("--frame", default="", help="Saved frame path; skips live window capture")
    parser.add_argument("--region", required=True, help="Name to store in profile.regions")
    parser.add_argument("--description", required=True, help="Visual description sent to the VLM")
    parser.add_argument("--coords", default="", help="Manual fallback x1,y1,x2,y2 normalized 0..1")
    parser.add_argument("--output", default="", help="Annotated candidate preview path")
    parser.add_argument("--accept", action="store_true", help="Accept candidate and write it into the profile")
    args = parser.parse_args(argv)

    try:
        profile = load_profile(args.game_id, args.profile or None)
        if args.frame:
            frame = Image.open(args.frame).convert("RGB")
        else:
            frame = _window_frame(args.window or profile.window)
        candidate = parse_manual_coords(args.coords) if args.coords else locate_region(frame, args.description)
        preview = annotate_candidate(frame, candidate, args.region)
        output = Path(args.output) if args.output else (
            config.data_dir / "diagnostics" / "marked_regions" / f"{args.game_id}_{args.region}.jpg")
        output.parent.mkdir(parents=True, exist_ok=True)
        preview.save(output, quality=90)
        print(f"Кандидат сохранён для проверки: {output}")
        print(f"Координаты: {candidate.model_dump()}")

        accepted = args.accept
        if not accepted and sys.stdin.isatty():
            accepted = ask_accept_candidate()
        if accepted:
            profile_path = Path(args.profile) if args.profile else (
                config.project_root / "data" / "profiles" / f"{args.game_id}.json")
            write_region(profile_path, args.game_id, args.region, candidate)
            print(f"Область записана в профиль: {profile_path}")
        else:
            print("Профиль не изменён; подтвердите позже флагом --accept после проверки preview.")
    except (OSError, ValueError, RuntimeError, requests.RequestException) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
