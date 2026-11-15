import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from omnigamer.doctor import _health_url, run_doctor


class FakeDetector:
    def health(self):
        return {"ok": True}
    def detect(self, image, queries):
        return {"boxes": []}


class FakeVlm:
    def plan_goal(self, **kwargs):
        return {"steps": [{"skill": "wait_for_condition", "args": {
            "region": "panel", "timeout": 1}, "expected_result": "wait", "success_check": "frame"}]}


class DoctorTests(unittest.TestCase):
    def test_health_url_replaces_completions_path(self):
        self.assertEqual(_health_url("http://127.0.0.1:8081/v1/chat/completions"),
                         "http://127.0.0.1:8081/health")

    def test_doctor_runs_service_probes_without_input(self):
        with tempfile.TemporaryDirectory() as temp:
            profile = Path(temp) / "profile.json"
            profile.write_text(json.dumps({"game_id": "doctor-test", "window": "Fake",
                "camera_type": "first_person", "regions": {"panel": {
                    "x1": 0.1, "y1": 0.1, "x2": 0.9, "y2": 0.9}}}), encoding="utf-8")
            response = SimpleNamespace(status_code=200, raise_for_status=lambda: None)
            capture = SimpleNamespace(grab_frame=lambda: Image.new("RGB", (32, 32)))
            with patch("omnigamer.doctor.find_window_by_title", return_value={"hwnd": 1, "title": "Fake"}), \
                 patch("omnigamer.doctor.WindowCapture", return_value=capture):
                checks = run_doctor("", str(profile), "Fake", vlm=FakeVlm(), detector=FakeDetector(),
                    get=lambda *a, **k: response,
                    disk_usage=lambda path: SimpleNamespace(free=5 * 1024**3),
                    run=lambda *a, **k: SimpleNamespace(stdout="GPU, 8192, 4096"),
                    input_probe=lambda: (True, "current input desktop is accessible"))
            by_name = {row["name"]: row for row in checks}
            self.assertEqual(by_name["VLM probe"]["status"], "OK")
            self.assertEqual(by_name["Детектор probe"]["status"], "OK")
            self.assertEqual(by_name["Доступ к input desktop"]["status"], "OK")
            self.assertIn("SendInput не отправлялся", by_name["Доступ к input desktop"]["detail"])

    def test_doctor_reports_input_desktop_access_failure(self):
        checks = run_doctor("", None, "", detector=FakeDetector(),
                            input_probe=lambda: (False, "access denied"),
                            get=lambda *a, **k: (_ for _ in ()).throw(ConnectionError()),
                            disk_usage=lambda path: SimpleNamespace(free=2 * 1024**3),
                            run=lambda *a, **k: SimpleNamespace(stdout="GPU, 8192, 4096"))
        row = next(item for item in checks if item["name"] == "Доступ к input desktop")
        self.assertEqual(row["status"], "ПРОБЛЕМА")
        self.assertIn("access denied", row["detail"])


if __name__ == "__main__":
    unittest.main()
