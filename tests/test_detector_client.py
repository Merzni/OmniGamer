import unittest
from unittest.mock import patch

import requests
from PIL import Image

from omnigamer.detector_client import DetectorClient


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class DetectorClientTests(unittest.TestCase):
    def test_counts_detect_http_attempts_but_not_health_checks(self):
        client = DetectorClient(base_url="http://127.0.0.1:8082", timeout=0.5)
        with patch("omnigamer.detector_client.requests.get",
                   return_value=FakeResponse({"status": "ok"})):
            self.assertEqual(client.health(), {"status": "ok"})
        self.assertEqual(client.detect_requests, 0)

        image = Image.new("RGB", (16, 12), (90, 120, 160))
        with patch("omnigamer.detector_client.requests.post", side_effect=[
                FakeResponse({"boxes": []}), requests.Timeout("offline")]) as post:
            self.assertEqual(client.detect(image, ["target"])["boxes"], [])
            self.assertIsNone(client.detect(image, ["target"]))

        self.assertEqual(post.call_count, 2)
        self.assertEqual(client.detect_requests, 2)


if __name__ == "__main__":
    unittest.main()
