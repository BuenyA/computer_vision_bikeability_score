"""Environment detector for the bikeability score — loads SegFormer ONCE.

Thin wrapper around the fine-tuned SegFormer deliverable (bundled locally as
`env_model.py`). Returns the continuous per-class pixel-area fractions the score
consumes.

Usage (instantiate once, then reuse per frame):
    from detectors.environment_detection import EnvironmentDetector
    env = EnvironmentDetector()               # loads SegFormer once
    fractions = env.detect(frame_bgr)         # np.float32 [3], order = EnvironmentDetector.classes

Return value: fractions np.ndarray over ["vegetation", "water", "city"].
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .env_model import ENV_CLASSES, EnvironmentModel

# Repo root = <root>/final_bikeability_score/detectors/environment_detection.py -> up 3.
# The SegFormer weights live in research/models/segformer_env (shared with research).
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_DIR = _REPO_ROOT / "research" / "models" / "segformer_env"


class EnvironmentDetector:
    """Loads the fine-tuned SegFormer once; returns env fractions for BGR frames."""

    classes = ENV_CLASSES   # ("vegetation", "water", "city")

    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR,
                 device: str | None = None):
        self.model = EnvironmentModel(model_dir, device=device)
        self.device = self.model.device

    def detect(self, frame_bgr: np.ndarray) -> np.ndarray:
        """frame_bgr: cv2 BGR image -> fractions np.float32 [3] (order = self.classes)."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return self.model.predict(rgb).fractions

    def detect_batch(self, frames_bgr: list[np.ndarray]) -> np.ndarray:
        """List of BGR frames -> fractions np.float32 [N, 3] (order = self.classes).

        Identical result to calling detect() per frame, but the SegFormer runs
        the frames as batches (faster throughput).
        """
        if not frames_bgr:
            return np.zeros((0, len(self.classes)), dtype=np.float32)
        rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
        preds = self.model.predict_batch(rgb)
        return np.stack([p.fractions for p in preds]).astype(np.float32)
