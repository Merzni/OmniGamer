"""Loopback OWLv2 text-conditioned object detector for OmniGamer skills."""
import argparse
import base64
import io
import logging
import threading
import time

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor

MODEL_ID = "google/owlv2-base-patch16-ensemble"
app = FastAPI(title="OmniGamer detector", docs_url=None, redoc_url=None)
logger = logging.getLogger("omnigamer.detector")
_model = None
_processor = None
_device = "cuda" if torch.cuda.is_available() else "cpu"
_fp16 = False
_inference_lock = threading.Lock()


class DetectRequest(BaseModel):
    image_jpeg_b64: str = Field(min_length=16, max_length=30_000_000)
    queries: list[str] = Field(min_length=1, max_length=16)
    score_min: float = Field(default=0.15, ge=0, le=1)
    max_boxes: int = Field(default=10, ge=1, le=50)


def load_model(fp16=False):
    global _model, _processor, _fp16, _device
    _processor = Owlv2Processor.from_pretrained(MODEL_ID)
    _model = Owlv2ForObjectDetection.from_pretrained(MODEL_ID).eval()
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    if fp16 and _device == "cuda":
        _model = _model.half()
    _model = _model.to(_device)
    _fp16 = bool(fp16 and _device == "cuda")
    warm = Image.new("RGB", (640, 360), (32, 64, 32))
    run_detection(warm, ["a tree trunk"], score_min=0.99, max_boxes=1)
    logger.info("Loaded %s on %s (%s)", MODEL_ID, _device, "fp16" if _fp16 else "fp32")


def run_detection(image, queries, score_min, max_boxes):
    started = time.perf_counter()
    inputs = _processor(images=image, text=[queries], return_tensors="pt")
    for key, value in tuple(inputs.items()):
        if hasattr(value, "to"):
            dtype = torch.float16 if _fp16 and value.is_floating_point() else None
            inputs[key] = value.to(device=_device, dtype=dtype) if dtype else value.to(_device)
    with _inference_lock, torch.inference_mode():
        outputs = _model(**inputs)
        if _device == "cuda":
            torch.cuda.synchronize()
    processed = _processor.post_process_grounded_object_detection(
        outputs,
        threshold=score_min,
        target_sizes=[(image.height, image.width)],
        text_labels=[queries],
    )[0]
    candidates = []
    for score, query, box in zip(processed["scores"].cpu(), processed["text_labels"],
                                 processed["boxes"].cpu()):
        if query not in queries:
            continue
        x1, y1, x2, y2 = [float(value) for value in box.tolist()]
        candidates.append({
            "query": query,
            "score": float(score.item()),
            "x1": max(0.0, min(1.0, x1 / image.width)),
            "y1": max(0.0, min(1.0, y1 / image.height)),
            "x2": max(0.0, min(1.0, x2 / image.width)),
            "y2": max(0.0, min(1.0, y2 / image.height)),
        })
    candidates.sort(key=lambda row: row["score"], reverse=True)
    return candidates[:max_boxes], round((time.perf_counter() - started) * 1000, 3)


@app.on_event("startup")
def startup():
    if _model is None:
        load_model(_fp16)


@app.get("/health")
def health():
    return {"status": "ok" if _model is not None else "loading", "model": MODEL_ID,
            "device": _device, "precision": "fp16" if _fp16 else "fp32"}


@app.post("/detect")
def detect(request: DetectRequest):
    if _model is None or _processor is None:
        raise HTTPException(status_code=503, detail="Detector model is not ready")
    queries = [query.strip() for query in request.queries]
    if any(not query or len(query) > 160 for query in queries):
        raise HTTPException(status_code=422, detail="Each query must contain 1..160 characters")
    try:
        image = Image.open(io.BytesIO(base64.b64decode(request.image_jpeg_b64, validate=True))).convert("RGB")
        if image.width < 2 or image.height < 2 or image.width * image.height > 25_000_000:
            raise ValueError("Image dimensions are outside allowed bounds")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JPEG image: {exc}") from exc
    try:
        boxes, latency = run_detection(image, queries, request.score_min, request.max_boxes)
    except Exception as exc:
        logger.exception("Detection failed")
        raise HTTPException(status_code=500, detail=f"Detection failed: {type(exc).__name__}") from exc
    return {"boxes": boxes, "latency_ms": latency, "model": MODEL_ID}


def main():
    parser = argparse.ArgumentParser(description="Run the local OWLv2 detector on loopback only")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--fp16", action="store_true", help="Use half precision when CUDA is available")
    parser.add_argument("--fp32", action="store_true", help="Force full precision (the default)")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be in 1..65535")
    if args.fp16 and args.fp32:
        parser.error("--fp16 and --fp32 cannot be used together")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_model(args.fp16)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
