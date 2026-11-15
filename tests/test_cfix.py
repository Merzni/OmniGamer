"""Regression checks for C0 fixes; no desktop input or live game is used."""
import json
import importlib.util
import os
import subprocess
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from PIL import Image
import requests

from omnigamer.telemetry import process_memory_mb
from omnigamer.vision_brain import VisionBrain


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class UndefinedNameCheck(unittest.TestCase):
    def test_pyflakes_has_no_undefined_names_in_application_and_tools(self):
        if importlib.util.find_spec("pyflakes") is None:
            self.skipTest("pyflakes не установлен; установите зависимости из requirements-dev.txt")
        result = subprocess.run(
            [sys.executable, "-m", "pyflakes", "omnigamer", "tools", "tests",
             "run_agent.py", "run_dashboard.py"],
            cwd=PROJECT_ROOT, capture_output=True, text=True,
        )
        self.assertNotIn("No module named pyflakes", result.stderr)
        undefined = [line for line in (result.stdout + result.stderr).splitlines()
                     if "undefined name" in line]
        self.assertEqual(undefined, [], "Найдены неопределённые имена:\n" + "\n".join(undefined))


@unittest.skipUnless(os.name == "nt", "Измерение рабочего набора через Win32 поддерживается только в Windows")
class ProcessMemoryTests(unittest.TestCase):
    def test_process_memory_mb_returns_positive_working_set(self):
        self.assertGreater(process_memory_mb(), 0)


class VisionBrainClickHTTPTests(unittest.TestCase):
    def test_click_decision_passes_grounding_with_local_http_server(self):
        actor = {
            "observation": "Main menu visible", "commentary": "Открою игру.",
            "intent": "Start the game", "scene_type": "menu", "camera_pitch": "unknown",
            "visual_confidence": 0.9, "menu_id": "main_menu", "menu_state": "default",
            "target_label": "Start", "expected_result": "gameplay", "action_type": "click",
            "confidence": 0.9,
        }
        target = {
            "scene_type": "menu", "menu_id": "main_menu", "menu_state": "default",
            "candidate_id": 0, "bbox": [100, 100, 300, 160], "label": "Start",
            "expected_result": "gameplay", "enabled": True, "confidence": 0.95,
        }
        responses = [actor, target]

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                self.rfile.read(length)
                content = json.dumps(responses.pop(0))
                body = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, _format, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        self.addCleanup(server.server_close)
        self.addCleanup(worker.join, 2)
        self.addCleanup(server.shutdown)

        brain = VisionBrain(api_url=f"http://127.0.0.1:{server.server_port}/v1/chat/completions")
        decision = brain.decide(Image.new("RGB", (320, 240), "black"))
        self.assertEqual(decision.action_type, "click")
        self.assertIsNone(brain.last_error)
        self.assertEqual(brain.telemetry_http_requests, 2)


class VisionBrainLookHTTPTests(unittest.TestCase):
    def serve(self, payloads):
        received = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                received.append(json.loads(self.rfile.read(length)))
                content = json.dumps(payloads.pop(0))
                body = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, _format, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        self.addCleanup(server.server_close)
        self.addCleanup(worker.join, 2)
        self.addCleanup(server.shutdown)
        return server, received

    @staticmethod
    def look(**overrides):
        response = {
            "scene_type": "gameplay", "camera_pitch": "level", "visual_confidence": 0.9,
            "menu_id": "unknown", "menu_state": "default", "target_label": "",
            "expected_result": "", "observation": "Open path", "intent": "Turn to inspect path",
            "commentary": "Посмотрю в сторону пути", "action_type": "look",
            "look_direction": "left", "look_amount": "medium", "confidence": 0.9,
        }
        response.update(overrides)
        return response

    def test_discrete_look_schema_goal_and_request_timings(self):
        server, received = self.serve([self.look()])
        brain = VisionBrain(api_url=f"http://127.0.0.1:{server.server_port}/v1/chat/completions")
        decision = brain.decide(Image.new("RGB", (32, 32)), goal="Find a safe path")
        self.assertEqual((decision.action_type, decision.look_direction, decision.look_amount),
                         ("look", "left", "medium"))
        self.assertEqual((decision.x, decision.y), (-60, 0))
        self.assertIsNone(brain.last_error)
        self.assertIsNone(brain.last_refusal_reason)
        request = received[0]
        self.assertIn("Current goal: Find a safe path", request["messages"][1]["content"][0]["text"])
        schema = request["response_format"]["json_schema"]["schema"]
        self.assertIn("look_direction", schema["required"])
        self.assertIn("look_amount", schema["required"])
        self.assertEqual(brain.telemetry_http_request_events[0]["type"], "decide")
        self.assertGreaterEqual(brain.telemetry_http_request_events[0]["duration_ms"], 0)

    def test_invalid_look_is_retried_once_then_refused_without_exception_fallback(self):
        invalid = self.look(look_direction=None, look_amount=None)
        server, received = self.serve([invalid, invalid])
        brain = VisionBrain(api_url=f"http://127.0.0.1:{server.server_port}/v1/chat/completions")
        decision = brain.decide(Image.new("RGB", (32, 32)))
        self.assertEqual(decision.action_type, "wait")
        self.assertEqual(len(received), 2)
        self.assertIsNone(brain.last_error)
        self.assertIn("Look отклонён", brain.last_refusal_reason)
        self.assertEqual(len(brain.last_raw_json_attempts), 2)

    def test_safe_decision_fallback_logs_error_type_and_source(self):
        brain = VisionBrain()
        with patch("omnigamer.vision_brain.requests.post", side_effect=requests.Timeout("offline")), \
                patch("omnigamer.vision_brain.Path.write_text"), \
                self.assertLogs("omnigamer.vision_brain", level="ERROR") as captured:
            decision = brain.decide(Image.new("RGB", (8, 8)))
        self.assertEqual(decision.action_type, "wait")
        self.assertIn("Timeout", captured.output[0])
        self.assertIn("VisionBrain.decide", captured.output[0])


if __name__ == "__main__":
    unittest.main()
