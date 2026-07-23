#!/usr/bin/env python3
"""Write per-class pixel-area-fraction predictions for a SegFormer model over
the test set. Works with or without an explicit 'other' class: only the 3
environment classes are counted, so 'other' pixels reduce every env fraction
(which is the point of adding the background class).

  python research/environment_model/evaluation/predict_semseg.py \
      --model models/segformer_env --out dataset/eval/env_pred_semseg.csv
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "environment_model"))
import segmentation_common as sc  # noqa: E402

ENV = sc.CATEGORIES["environment"]
ENV_ID = {c: i for i, c in enumerate(ENV)}
TEST = REPO / "dataset" / "test_images"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    model = SegformerForSemanticSegmentation.from_pretrained(args.model).to(device).eval()
    proc = SegformerImageProcessor.from_pretrained(args.model)
    print("model outputs", model.config.num_labels, "classes:",
          list(model.config.id2label.values()))

    exts = {".jpg", ".jpeg", ".png"}
    imgs = sorted(p for p in TEST.rglob("*")
                  if p.suffix.lower() in exts and "overview" not in p.parts)
    rows, t0 = [], time.perf_counter()
    with torch.no_grad():
        for k, fp in enumerate(imgs):
            arr = np.array(Image.open(fp).convert("RGB"))
            pred = model(**proc(images=arr, return_tensors="pt").to(device)
                         ).logits.argmax(1)[0].cpu().numpy()
            n = pred.size
            row = {"filename": str(fp.relative_to(TEST))}
            for c in ENV:                       # 'other' (id 4) not counted
                row[c] = round(float((pred == ENV_ID[c]).sum()) / n, 4)
            rows.append(row)
            if k and k % 400 == 0:
                print(f"  {k}/{len(imgs)}", flush=True)
    ms = (time.perf_counter() - t0) / len(imgs) * 1000
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", *ENV])
        w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} -> {args.out} | {ms:.0f} ms/frame")


if __name__ == "__main__":
    main()
