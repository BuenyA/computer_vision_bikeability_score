"""Environment classifier for the bikeability master script — PLUG AND PLAY.

Final model of the environment component (decision 2026-07-12): fine-tuned
SegFormer-B0 (`models/segformer_env`), 3 environment classes + background.
Own-frames test (1,210 cyclist-POV frames incl. water rides): macro-F1 0.704,
label-accuracy 0.911 (beats zero-shot CLIP 0.649/0.888).

Deployment needs exactly two things:
    1. this file
    2. the model folder `models/segformer_env/` (weights + inference_config.json)
Dependencies: torch, numpy, Pillow, transformers (same versions as this repo).

Master-script integration
-------------------------
    from env_model import EnvironmentModel

    env = EnvironmentModel("models/segformer_env")   # once, at startup
    for frame in frames:                             # np.ndarray | PIL | path
        pred = env.predict(frame)
        pred.fractions   # np.float32 [frac_vegetation, frac_water, frac_city]
        pred.labels      # np.int8    [vegetation, water, city]  (0/1)
        pred.features()  # {"frac_vegetation": .., "frac_water": .., "frac_city": ..}

Per the merge design (dev_documentation/model_merge_guide.md) the master
script's features.csv should store the CONTINUOUS `frac_*` values — the score
consumes fractions; the binary labels exist for classification/reporting.
Batches: `env.predict_batch(list_of_frames)` (same order in = out).
CLI (writes the env feature CSV for a directory of frames):
    python env_model.py dataset/frames/some_ride --out env_features.csv

Implementation notes (do not change casually):
* Class order comes from the checkpoint's id2label, filtered to the 3 env
  classes; the "other" background class is excluded from the fractions, so
  sky/road pixels reduce every env fraction (intended).
* Fractions are computed from the argmax at logits resolution (1/4 of the
  512x512 model input). The decision thresholds in inference_config.json were
  validation-tuned on fractions computed exactly this way — upsampling the
  logits first would silently shift the operating point.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

ENV_CLASSES = ("vegetation", "water", "city")


@dataclass
class EnvPrediction:
    """Per-frame result. Arrays are ordered like `EnvironmentModel.classes`."""
    fractions: np.ndarray   # float32 (3,) pixel-area fraction per env class
    labels: np.ndarray      # int8    (3,) fraction >= tuned threshold

    def features(self) -> dict[str, float]:
        """Merge-guide interchange row: {'frac_vegetation': .., ...}."""
        return {f"frac_{c}": float(v) for c, v in zip(ENV_CLASSES, self.fractions)}


class EnvironmentModel:
    """Loads the fine-tuned SegFormer once; thread-safe for read-only use."""

    classes = ENV_CLASSES

    def __init__(self, model_dir: str | Path = "models/segformer_env",
                 device: str | None = None):
        model_dir = Path(model_dir)
        self.device = device or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model = (SegformerForSemanticSegmentation
                      .from_pretrained(model_dir).to(self.device).eval())
        self.processor = SegformerImageProcessor.from_pretrained(model_dir)

        id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        missing = [c for c in ENV_CLASSES if c not in id2label.values()]
        if missing:
            raise ValueError(f"model at {model_dir} lacks env classes {missing}; "
                             f"its classes are {sorted(id2label.values())}")
        self._class_ids = np.array(
            [next(i for i, n in id2label.items() if n == c) for c in ENV_CLASSES])

        cfg = json.loads((model_dir / "inference_config.json").read_text())
        self.thresholds = np.array([cfg["thresholds"][c] for c in ENV_CLASSES],
                                   dtype=np.float32)

    @staticmethod
    def _to_rgb(image) -> np.ndarray:
        if isinstance(image, (str, Path)):
            return np.array(Image.open(image).convert("RGB"))
        if isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))
        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"expected (H, W, 3) RGB array, got shape {arr.shape}")
        return arr

    def predict(self, image) -> EnvPrediction:
        """image: file path, PIL.Image, or (H, W, 3) RGB uint8 array."""
        return self.predict_batch([image])[0]

    @torch.no_grad()
    def predict_batch(self, images: list, batch_size: int = 8) -> list[EnvPrediction]:
        out: list[EnvPrediction] = []
        for i in range(0, len(images), batch_size):
            arrs = [self._to_rgb(im) for im in images[i:i + batch_size]]
            enc = self.processor(images=arrs, return_tensors="pt").to(self.device)
            seg = self.model(**enc).logits.argmax(1).cpu().numpy()   # (B, H/4, W/4)
            n = seg.shape[1] * seg.shape[2]
            for m in seg:
                fr = np.array([(m == cid).sum() / n for cid in self._class_ids],
                              dtype=np.float32)
                out.append(EnvPrediction(fractions=fr,
                                         labels=(fr >= self.thresholds).astype(np.int8)))
        return out


def main() -> None:
    import argparse
    import csv
    import time

    ap = argparse.ArgumentParser(description="Write per-frame env features to CSV.")
    ap.add_argument("inputs", nargs="+", help="image files and/or directories")
    ap.add_argument("--out", default="env_features.csv")
    ap.add_argument("--model", default=str(Path(__file__).resolve().parent.parent.parent
                                           / "research" / "models" / "segformer_env"))
    args = ap.parse_args()

    exts = {".jpg", ".jpeg", ".png"}
    files: list[Path] = []
    for inp in map(Path, args.inputs):
        files += sorted(p for p in ([inp] if inp.is_file() else inp.rglob("*"))
                        if p.suffix.lower() in exts)
    if not files:
        raise SystemExit("no images found")

    env = EnvironmentModel(args.model)
    print(f"{len(files)} frames | device {env.device} | model {args.model}")
    t0 = time.perf_counter()
    preds = env.predict_batch(files)
    ms = (time.perf_counter() - t0) / len(files) * 1000

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename",
                                          *(f"frac_{c}" for c in ENV_CLASSES),
                                          *(f"label_{c}" for c in ENV_CLASSES)])
        w.writeheader()
        for p, pr in zip(files, preds):
            w.writerow({"filename": p.name, **pr.features(),
                        **{f"label_{c}": int(v)
                           for c, v in zip(ENV_CLASSES, pr.labels)}})
    print(f"wrote {len(preds)} rows -> {args.out} | {ms:.1f} ms/frame")


if __name__ == "__main__":
    main()
