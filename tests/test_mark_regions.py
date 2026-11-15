import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

from tools.mark_regions import (
    annotate_candidate,
    locate_region,
    parse_candidate,
    parse_manual_coords,
    write_region,
)


class MarkRegionTests(unittest.TestCase):
    def test_vlm_candidate_requires_valid_normalized_rectangle(self):
        region = parse_candidate('{"x1":0.2,"y1":0.4,"x2":0.8,"y2":0.9}')
        self.assertEqual(region.x1, 0.2)
        with self.assertRaisesRegex(ValueError, "invalid normalized region"):
            parse_candidate('{"x1":0.8,"y1":0.4,"x2":0.2,"y2":0.9}')

    def test_manual_coordinates_are_validated(self):
        region = parse_manual_coords("0.1,0.2,0.9,0.8")
        self.assertEqual((region.x1, region.y1, region.x2, region.y2), (0.1, 0.2, 0.9, 0.8))
        with self.assertRaisesRegex(ValueError, "normalized"):
            parse_manual_coords("0,0,2,1")

    def test_preview_is_annotated_without_mutating_source_frame(self):
        source = Image.new("RGB", (100, 80), "black")
        region = parse_manual_coords("0.2,0.2,0.7,0.8")
        preview = annotate_candidate(source, region, "hud")
        self.assertEqual(preview.size, source.size)
        self.assertEqual(source.getpixel((20, 16)), (0, 0, 0))
        self.assertNotEqual(preview.getpixel((20, 16)), source.getpixel((20, 16)))

    def test_profile_is_written_only_by_explicit_write_call(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "profile.json"
            write_region(path, "temporary-game", "hud", parse_manual_coords("0,0.8,1,1"))
            profile = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(profile["regions"]["hud"]["y1"], 0.8)

    def test_eof_at_accept_prompt_rejects_candidate(self):
        from tools.mark_regions import ask_accept_candidate
        with patch("builtins.input", side_effect=EOFError):
            self.assertFalse(ask_accept_candidate())

    def test_vlm_request_uses_json_schema_and_parses_candidate(self):
        response = Mock()
        response.json.return_value = {"choices": [{"message": {
            "content": '{"x1":0.2,"y1":0.7,"x2":0.8,"y2":0.95}'}}]}
        frame = Image.new("RGB", (40, 30), "gray")
        with patch("tools.mark_regions.requests.post", return_value=response) as post:
            region = locate_region(frame, "visible lower panel")
        self.assertAlmostEqual(region.y2, 0.95)
        self.assertEqual(post.call_args.kwargs["json"]["response_format"]["type"], "json_schema")


if __name__ == "__main__":
    unittest.main()
