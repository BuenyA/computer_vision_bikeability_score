"""Traffic-object detector for the bikeability score — loads YOLO ONCE.

Based on research/Object_Detection/09_Collab_Inference_pipeline.ipynb (YOLO via
ultralytics). Only the classes relevant to the score are counted:
["bicycle", "car", "traffic light"] (person and motorcycle were dropped).

Usage (instantiate once, then reuse per frame):
    from detectors.object_detection import ObjectDetector
    objects = ObjectDetector()                # loads yolo26m.pt once
    counts = objects.detect(frame_bgr)        # np.int [3], order = ObjectDetector.classes

Return value: counts np.ndarray over ["bicycle", "car", "traffic light"].
The YOLO weights are downloaded automatically by ultralytics on first use.
"""
from __future__ import annotations

import numpy as np
import torch
from ultralytics import YOLO

TARGET_CATEGORIES = ("bicycle", "car", "traffic light")
MODEL_WEIGHTS = "yolo26m.pt"
CONF_THRESHOLD = 0.15


class ObjectDetector:
    """Loads YOLO once; counts target traffic objects in BGR frames."""

    classes = TARGET_CATEGORIES

    def __init__(self, weights: str = MODEL_WEIGHTS, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.half = self.device == "cuda"          # half precision only on GPU
        self.model = YOLO(weights)
        self.names = self.model.names
        self._index = {c: i for i, c in enumerate(self.classes)}

    def detect(self, frame_bgr: np.ndarray) -> np.ndarray:
        """frame_bgr: cv2 BGR image -> counts np.ndarray [3] (order = self.classes)."""
        result = self.model(frame_bgr, device=self.device, half=self.half,
                            conf=CONF_THRESHOLD, verbose=False)[0]
        counts = np.zeros(len(self.classes), dtype=int)
        for box in result.boxes:
            name = self.names[int(box.cls[0])]
            if name in self._index:
                counts[self._index[name]] += 1
        return counts

    def detect_batch(self, frames_bgr: list[np.ndarray]) -> np.ndarray:
        """List of BGR frames -> counts np.ndarray [N, 3] (order = self.classes).

        Identical result to calling detect() per frame, but YOLO runs the whole
        list as one batch (faster throughput).
        """
        if not frames_bgr:
            return np.zeros((0, len(self.classes)), dtype=int)
        results = self.model(frames_bgr, device=self.device, half=self.half,
                             conf=CONF_THRESHOLD, verbose=False)
        out = np.zeros((len(frames_bgr), len(self.classes)), dtype=int)
        for i, result in enumerate(results):
            for box in result.boxes:
                name = self.names[int(box.cls[0])]
                if name in self._index:
                    out[i, self._index[name]] += 1
        return out
