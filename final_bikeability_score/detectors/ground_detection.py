"""Ground-surface detector for the bikeability score — loads the model ONCE.

Zero-shot surface classifier based on the road_detection benchmark winner
(best macro-F1): OpenCLIP ViT-B/32, pretrained "laion2b_s34b_b79k". Classes and
prompt ensembling are taken 1:1 from research/ground_detection/road_detection.ipynb.

Usage (instantiate once, then reuse per frame):
    from detectors.ground_detection import GroundDetector
    ground = GroundDetector()                 # loads open_clip model once
    onehot = ground.detect(frame_bgr)         # np.int8 [4], order = GroundDetector.classes

Return value: one-hot np.ndarray over ["Cycleway", "Road", "Gravel", "Unpaved"]
(argmax of the softmax over cosine similarity). E.g. Road -> [0, 1, 0, 0].
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image

import open_clip

# --- Surface classes + prompt ensembling (from road_detection.ipynb) ---------
SURFACE_CLASSES: dict[str, list[str]] = {
    "Cycleway": [
        "a narrow paved bike path with a painted bicycle symbol",
        "a dedicated cycle track separated from the car road",
        "a red colored asphalt cycle lane only for bicycles",
        "a smooth narrow bicycle path with no cars",
    ],
    "Road": [
        "a wide asphalt road for cars with lane markings",
        "a public street with a painted center line for traffic",
        "a two-lane tarmac road for motor vehicles",
        "a broad paved car road with white lane lines",
    ],
    "Gravel": [
        "an unpaved gravel road covered with loose small stones",
        "a track of crushed stone and coarse gravel",
        "a path of loose grey pebbles and gravel",
        "a bumpy gravel forest road with scattered stones",
    ],
    "Unpaved": [
        "an unpaved dirt road of bare brown soil",
        "a muddy earth track with puddles and no pavement",
        "a rough forest trail with mud, roots and dirt",
        "a bumpy natural ground path that is hard to ride with a road bike",
    ],
}

ARCH = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"


class GroundDetector:
    """Loads the OpenCLIP model once; classifies BGR frames into surface classes."""

    classes = tuple(SURFACE_CLASSES)   # ("Cycleway", "Road", "Gravel", "Unpaved")

    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            ARCH, pretrained=PRETRAINED, device=self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(ARCH)
        self.class_text_features = self._encode_class_prompts()

    @torch.no_grad()
    def _encode_class_prompts(self) -> torch.Tensor:
        feats = []
        for cls in self.classes:
            tokens = self.tokenizer(SURFACE_CLASSES[cls]).to(self.device)
            tf = self.model.encode_text(tokens)
            tf = tf / tf.norm(dim=-1, keepdim=True)   # normalize each prompt
            mean = tf.mean(dim=0)                      # prompt ensembling
            feats.append(mean / mean.norm())
        return torch.stack(feats)                      # [n_classes, dim]

    @torch.no_grad()
    def detect(self, frame_bgr: np.ndarray) -> np.ndarray:
        """frame_bgr: cv2 BGR image -> one-hot np.int8 [4] (order = self.classes)."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = self.preprocess(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        feat = self.model.encode_image(img)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        sims = (feat @ self.class_text_features.T).squeeze(0)   # [n_classes]
        onehot = np.zeros(len(self.classes), dtype=np.int8)
        onehot[int(sims.argmax())] = 1
        return onehot

    @torch.no_grad()
    def detect_batch(self, frames_bgr: list[np.ndarray]) -> np.ndarray:
        """List of BGR frames -> one-hot np.int8 [N, 4] (order = self.classes).

        Identical result to calling detect() per frame, but encodes all images
        in a single forward pass (faster throughput).
        """
        if not frames_bgr:
            return np.zeros((0, len(self.classes)), dtype=np.int8)
        batch = torch.stack([
            self.preprocess(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
            for f in frames_bgr
        ]).to(self.device)
        feats = self.model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        sims = feats @ self.class_text_features.T          # [N, n_classes]
        idx = sims.argmax(dim=1).cpu().numpy()
        onehot = np.zeros((len(frames_bgr), len(self.classes)), dtype=np.int8)
        onehot[np.arange(len(frames_bgr)), idx] = 1
        return onehot
