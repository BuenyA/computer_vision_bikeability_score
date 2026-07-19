#!/usr/bin/env python3
"""Zero-shot CLIP improvement experiments (options 2 & 3):
  - backbone: ViT-B/32 (baseline) vs. ViT-L/14 (bigger)
  - prompts:  single tuned prompt vs. multi-template ENSEMBLE

For each (backbone, prompt-mode) config: compute per-class present-probabilities
on the OWN-frames val/test split, tune per-class thresholds on VALIDATION only,
then report validation + held-out-test F1. Nothing is tuned on test.

Reports macro-F1 (test = 3 classes, no test water), per-class F1, label accuracy.
Results appended to research/clip_experiments_results.csv.
"""

import csv
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "research"))
import segmentation_common as sc  # noqa: E402

ENV = sc.CATEGORIES["environment"]
DEVICE = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")
TEST_DIR = REPO / "dataset" / "test_images"
EVAL = REPO / "dataset" / "eval"
RESULTS = REPO / "research" / "clip_experiments_results.csv"
GRID = np.round(np.r_[np.linspace(0.02, 0.1, 9), np.linspace(0.12, 0.95, 84)], 3)

# --- single prompt per class ---
SINGLE = {
    "vegetation": ("trees, forest, grass or an open field",       "no plants, only buildings or pavement"),
    "water":      ("visible water or a waterway",                 "no water is visible"),
    "city":       ("an urban street with buildings and houses",   "countryside or nature, no buildings"),
}
# --- ensemble templates (averaged text embeddings); vegetation = forest+field ---
ENS_POS = {
    "vegetation": ["a forest", "woods with many trees", "trees along the road",
                   "an open field", "a meadow", "open grassland",
                   "grass, plants or greenery", "a green natural area"],
    "water": ["water", "a river", "a lake", "a canal", "the sea",
              "a body of water", "a waterway"],
    "city": ["a city street", "buildings and houses", "an urban area",
             "a street with houses", "industrial buildings", "a built-up area"],
}
ENS_NEG = {
    "vegetation": ["no plants or greenery", "only buildings and pavement",
                   "an urban area with no vegetation", "water only, no plants"],
    "water": ["no water", "dry land", "a scene with no water", "a dry street"],
    "city": ["open countryside", "nature with no buildings",
             "a forest or field, no buildings"],
}
BACKBONES = {"vit-b32": "openai/clip-vit-base-patch32",
             "vit-l14": "openai/clip-vit-large-patch14"}


def load_split_labels():
    split = {r["ride_id"]: r["split"]
             for r in csv.DictReader(open(EVAL / "own_split.csv"))}
    rows = {}
    for r in csv.DictReader(open(TEST_DIR / "labels.csv")):
        fn = r["filename"]
        if not fn.startswith("own_frames"):
            continue
        if int(r.get("reject", 0) or 0) or int(r.get("unsure", 0) or 0):
            continue
        rid = re.search(r"(DJI_\d+)", fn).group(1)
        rows[fn] = {"split": split.get(rid, "test"),
                    "y": np.array([int(r[c]) for c in ENV])}
    return rows


def image_features(model, proc, files):
    cache = EVAL / f".imgfeat_{model.config._name_or_path.split('/')[-1]}.npy"
    if cache.exists():
        return torch.tensor(np.load(cache))
    feats = []
    with torch.no_grad():
        for k, fn in enumerate(files):
            im = Image.open(TEST_DIR / fn).convert("RGB")
            out = model.get_image_features(**proc(images=im, return_tensors="pt").to(DEVICE))
            f = getattr(out, "pooler_output", out)
            feats.append((f / f.norm(dim=-1, keepdim=True)).cpu())
            if k and k % 150 == 0:
                print(f"    img feats {k}/{len(files)}", flush=True)
    F = torch.cat(feats)
    np.save(cache, F.numpy())
    return F


def text_proto(model, proc, prompts):
    with torch.no_grad():
        t = proc(text=list(prompts), return_tensors="pt", padding=True).to(DEVICE)
        out = model.get_text_features(**t)
        f = getattr(out, "pooler_output", out)
        f = f / f.norm(dim=-1, keepdim=True)
        proto = f.mean(0)
        return (proto / proto.norm()).cpu()


def class_probs(F, model, proc, mode):
    P = np.zeros((len(F), len(ENV)))
    for j, c in enumerate(ENV):
        if mode == "single":
            pos, neg = text_proto(model, proc, [SINGLE[c][0]]), text_proto(model, proc, [SINGLE[c][1]])
        else:
            pos, neg = text_proto(model, proc, ENS_POS[c]), text_proto(model, proc, ENS_NEG[c])
        sims = F @ torch.stack([pos, neg]).T          # (N,2)
        P[:, j] = torch.softmax(sims * 100.0, dim=1)[:, 0].numpy()
    return P


def f1(pred, y):
    tp = int((pred & y).sum()); fp = int((pred & ~y).sum()); fn = int((~pred & y).sum())
    return 2 * tp / max(2 * tp + fp + fn, 1)


def evaluate(P, Y, is_val, is_test):
    thr, f1_val = {}, {}
    for j, c in enumerate(ENV):
        yv = Y[is_val, j].astype(bool)
        if yv.sum() == 0:
            thr[c] = 0.5; continue
        best = max(((f1(P[is_val, j] >= t, yv), t) for t in GRID), key=lambda x: (x[0], -x[1]))
        thr[c] = float(best[1]); f1_val[c] = best[0]
    tvec = np.array([thr[c] for c in ENV])

    def macro(mask, classes):
        idx = [ENV.index(c) for c in classes]
        yp = (P[mask] >= tvec).astype(int)
        m = sc.multilabel_metrics(Y[mask][:, idx], yp[:, idx])
        m["by_class"] = {c: float(m["per_class_f1"][k]) for k, c in enumerate(classes)}
        return m
    val = macro(is_val, ENV)                          # 4-class (has water)
    test = macro(is_test, ["vegetation", "city"])  # no test water
    return thr, val, test


def main():
    rows = load_split_labels()
    files = sorted(rows)
    Y = np.stack([rows[f]["y"] for f in files])
    is_val = np.array([rows[f]["split"] == "val" for f in files])
    is_test = np.array([rows[f]["split"] == "test" for f in files])
    print(f"own frames: val={is_val.sum()} test={is_test.sum()}")

    out_rows = []
    for bk, name in BACKBONES.items():
        print(f"\n=== backbone {bk} ({name}) ===")
        model = CLIPModel.from_pretrained(name).to(DEVICE).eval()
        proc = CLIPProcessor.from_pretrained(name)
        t0 = time.perf_counter()
        F = image_features(model, proc, files)
        for mode in ("single", "ensemble"):
            P = class_probs(F, model, proc, mode)
            thr, val, test = evaluate(P, Y, is_val, is_test)
            row = {"backbone": bk, "prompts": mode,
                   "val_macroF1_4c": round(val["macro_f1"], 3),
                   "test_macroF1_3c": round(test["macro_f1"], 3),
                   "test_labelAcc": round(test["label_accuracy"], 3)}
            for c in ["vegetation", "city"]:
                row[f"test_F1_{c}"] = round(test["by_class"][c], 3)
            row["val_F1_water"] = round(val["by_class"]["water"], 3)
            row["thr"] = {k: round(v, 3) for k, v in thr.items()}
            out_rows.append(row)
            print(f"  {mode:9} | test macroF1(veg+city)={row['test_macroF1_3c']} "
                  f"labelAcc={row['test_labelAcc']} | "
                  f"vegetation={row['test_F1_vegetation']} city={row['test_F1_city']} "
                  f"| val water F1={row['val_F1_water']}")
        print(f"  ({bk} took {time.perf_counter()-t0:.0f}s)")

    keys = ["backbone", "prompts", "val_macroF1_4c", "test_macroF1_3c",
            "test_labelAcc", "test_F1_vegetation", "test_F1_city",
            "val_F1_water", "thr"]
    with open(RESULTS, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(out_rows)
    print(f"\nsaved -> {RESULTS}")


if __name__ == "__main__":
    main()
