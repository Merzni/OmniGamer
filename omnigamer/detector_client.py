"""Small HTTP client for the optional loopback OWLv2 detector service."""
import base64
import io
import logging
import math

import numpy as np
import requests
from PIL import Image

from .config import config

logger = logging.getLogger(__name__)


def preprocess_image(image, mode):
    """Apply a deterministic luminance transform before sending the detector image."""
    mode = str(mode or "none").lower()
    if mode == "none":
        return image.convert("RGB"), {"mode": mode}
    if mode == "gamma":
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        mean = float(np.asarray(image.convert("L"), dtype=np.float32).mean() / 255.0)
        gamma = float(np.clip(math.log(110 / 255) / math.log(max(mean, 1e-6)), 0.2, 3.0))
        result = np.clip(np.power(rgb, gamma) * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(result, "RGB"), {"mode": mode, "gamma": gamma, "target_mean": 110}
    if mode == "stretch":
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
        luminance = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        low, high = [float(value) for value in np.percentile(luminance, [2, 98])]
        result = np.clip((rgb - low) * (255 / max(1.0, high - low)), 0, 255).astype(np.uint8)
        return Image.fromarray(result, "RGB"), {"mode": mode, "p2": low, "p98": high}
    if mode == "clahe":
        ycbcr = np.asarray(image.convert("YCbCr"), dtype=np.uint8).copy()
        luminance = ycbcr[:, :, 0]
        height, width = luminance.shape
        tiles, clip_factor = 8, 2.0
        result = np.zeros_like(luminance)
        maps = []
        for tile_y in range(tiles):
            row = []
            y0, y1 = tile_y * height // tiles, (tile_y + 1) * height // tiles
            for tile_x in range(tiles):
                x0, x1 = tile_x * width // tiles, (tile_x + 1) * width // tiles
                tile = luminance[y0:y1, x0:x1]
                histogram = np.bincount(tile.ravel(), minlength=256).astype(np.int64)
                limit = max(1, int(clip_factor * tile.size / 256))
                excess = int(np.maximum(histogram - limit, 0).sum())
                histogram = np.minimum(histogram, limit)
                histogram += excess // 256
                histogram[:excess % 256] += 1
                cdf = histogram.cumsum()
                nonzero = np.flatnonzero(cdf)
                cdf_min = int(cdf[nonzero[0]]) if len(nonzero) else 0
                lut = np.clip((cdf - cdf_min) * 255 / max(1, tile.size - cdf_min),
                              0, 255).astype(np.uint8)
                row.append(lut)
            maps.append(row)
        for tile_y in range(tiles):
            y0, y1 = tile_y * height // tiles, (tile_y + 1) * height // tiles
            for tile_x in range(tiles):
                x0, x1 = tile_x * width // tiles, (tile_x + 1) * width // tiles
                result[y0:y1, x0:x1] = maps[tile_y][tile_x][luminance[y0:y1, x0:x1]]
        ycbcr[:, :, 0] = result
        return Image.fromarray(ycbcr, "YCbCr").convert("RGB"), {
            "mode": mode, "tiles": tiles, "clip_factor": clip_factor,
            "interpolation": "nearest",
        }
    raise ValueError(f"Unsupported detector preprocessing mode: {mode}")


class DetectorClient:
    def __init__(self, base_url=None, score_min=None, max_boxes=10, timeout=2.0):
        self.base_url = (base_url or config.detector_url).rstrip("/")
        self.score_min = config.detector_score_min if score_min is None else float(score_min)
        self.max_boxes = max(1, min(50, int(max_boxes)))
        self.timeout = min(2.0, max(0.1, float(timeout)))
        self.model = None
        self.detect_requests = 0

    def health(self):
        """Return service health metadata, or None when the service is unavailable."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            self.model = data.get("model") or self.model
            return data
        except Exception as exc:
            logger.warning("Detector health request failed (%s): %s", type(exc).__name__, exc)
            return None

    def detect(self, image, queries, score_min=None, max_boxes=None, timeout=None):
        """Detect text queries in a PIL image. Returns None on service/protocol errors."""
        try:
            processed, preprocess = preprocess_image(image, config.detector_preprocess)
            buffer = io.BytesIO()
            processed.save(buffer, format="JPEG", quality=85)
            payload = {
                "image_jpeg_b64": base64.b64encode(buffer.getvalue()).decode("ascii"),
                "queries": [str(query) for query in queries],
                "score_min": self.score_min if score_min is None else float(score_min),
                "max_boxes": self.max_boxes if max_boxes is None else max(1, min(50, int(max_boxes))),
            }
            request_timeout = self.timeout if timeout is None else max(0.05, min(self.timeout, float(timeout)))
            self.detect_requests += 1
            response = requests.post(f"{self.base_url}/detect", json=payload, timeout=request_timeout)
            response.raise_for_status()
            result = response.json()
            if not isinstance(result, dict) or not isinstance(result.get("boxes"), list):
                raise ValueError("Detector returned an invalid response")
            self.model = result.get("model") or self.model
            result["preprocess"] = preprocess
            return result
        except Exception as exc:
            logger.warning("Detector request failed (%s): %s", type(exc).__name__, exc)
            return None
