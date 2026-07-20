#!/usr/bin/env python3
"""Optimize both approaches' decision thresholds on the OWN-frames validation
split, then report validation + test metrics. Test images are only ever scored,
never tuned on (leakage-free).

Inputs (continuous scores, produced by the prediction cells / regeneration):
  dataset/eval/env_pred_zeroshot.csv   per-class CLIP softmax prob
  dataset/eval/env_pred_semseg.csv     per-class SegFormer pixel-area fraction
  dataset/test_images/labels.csv       hand labels (multi-hot + reject/unsure)
  dataset/eval/own_split.csv           ride_id -> {val,test}

Per class, the threshold that maximizes validation F1 is chosen (a class with
no validation positives keeps its baseline threshold). Saves the tuned
thresholds to dataset/eval/tuned_thresholds.json.
"""

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
import sys
sys.path.insert(0, str(REPO / "environment_model"))
import segmentation_common as sc  # noqa: E402

ENV = sc.CATEGORIES["environment"]
EVAL = REPO / "dataset" / "eval"
LABELS = REPO / "dataset" / "test_images" / "labels.csv"
PRED = {"zeroshot": EVAL / "env_pred_zeroshot.csv",
        "semseg": EVAL / "env_pred_semseg.csv"}
BASELINE = {  # starting thresholds; the sweep re-tunes each on validation
    "zeroshot": {c: 0.5 for c in ENV},
    "semseg": {"vegetation": 0.10, "water": 0.02, "city": 0.10},
}
# Grid: log-spaced low end (area fractions of small/distant objects live at
# 1e-4..1e-2 — the original 0.01 floor quantized water's optimum away) +
# linear mid/high range for softmax-style scores.
GRID = np.unique(np.r_[np.round(np.logspace(-4, -2, 17), 6),
                       np.round(np.linspace(0.01, 0.10, 19), 3),
                       np.round(np.linspace(0.12, 0.95, 84), 3)])


def load():
    gt = pd.read_csv(LABELS)
    for c in ("reject", "unsure"):
        if c not in gt.columns:
            gt[c] = 0
    gt = gt[(gt.reject.fillna(0).astype(int) == 0)
            & (gt.unsure.fillna(0).astype(int) == 0)]
    gt = gt[gt.filename.str.startswith("own_frames")].copy()
    split = {r["ride_id"]: r["split"]
             for r in csv.DictReader(open(EVAL / "own_split.csv"))}
    gt["ride"] = gt.filename.str.extract(r"(DJI_\d+)")[0]
    # rides not in own_split.csv (e.g. newly added water rides) default to TEST
    gt["split"] = gt.ride.map(lambda r: split.get(r, "test"))
    gt = gt.set_index("filename")
    return gt


def f1(pred, true):
    tp = int((pred & true).sum()); fp = int((pred & ~true).sum())
    fn = int((~pred & true).sum())
    return 2 * tp / max(2 * tp + fp + fn, 1)


def metrics(score, gt_bin, thr):
    yp = (score >= np.array([thr[c] for c in ENV])).astype(int)
    m = sc.multilabel_metrics(gt_bin, yp)
    return m


def tune(method, pred_path=None):
    gt = load()
    sc_df = (pd.read_csv(pred_path or PRED[method]).set_index("filename")
             .reindex(columns=ENV).astype(float))
    common = gt.index.intersection(sc_df.index)
    gt = gt.loc[common]
    score = sc_df.loc[common].values
    gt_bin = gt[ENV].astype(int).values
    is_val = (gt["split"] == "val").values
    is_test = (gt["split"] == "test").values

    tuned = dict(BASELINE[method])
    per_class_val = {}
    for j, c in enumerate(ENV):
        yv, sv = gt_bin[is_val, j].astype(bool), score[is_val, j]
        if yv.sum() == 0:                       # no val positives -> keep baseline
            per_class_val[c] = None
            continue
        best = max(((f1(sv >= t, yv), t) for t in GRID), key=lambda x: (x[0], -x[1]))
        tuned[c] = float(best[1])
        per_class_val[c] = round(best[0], 3)
    return {
        "method": method, "n_val": int(is_val.sum()), "n_test": int(is_test.sum()),
        "baseline": BASELINE[method], "tuned": tuned,
        "val_base": metrics(score[is_val], gt_bin[is_val], BASELINE[method]),
        "val_tuned": metrics(score[is_val], gt_bin[is_val], tuned),
        "test_base": metrics(score[is_test], gt_bin[is_test], BASELINE[method]),
        "test_tuned": metrics(score[is_test], gt_bin[is_test], tuned),
        "per_class_val_f1_tuned": per_class_val,
    }


def fmt(m):
    pc = " ".join(f"{c}={m['per_class_f1'][i]:.3f}" for i, c in enumerate(ENV))
    return (f"macroF1={m['macro_f1']:.3f} labelAcc={m['label_accuracy']:.3f} | {pc}")


def main():
    out = {}
    for method in ("zeroshot", "semseg"):
        r = tune(method)
        out[method] = {"baseline": r["baseline"], "tuned": r["tuned"]}
        print(f"\n===== {method}  (val n={r['n_val']}, test n={r['n_test']}) =====")
        print(f"  baseline thr: {r['baseline']}")
        print(f"  tuned    thr: { {k: round(v,3) for k,v in r['tuned'].items()} }")
        print(f"  VAL  baseline: {fmt(r['val_base'])}")
        print(f"  VAL  tuned   : {fmt(r['val_tuned'])}")
        print(f"  TEST baseline: {fmt(r['test_base'])}  (own, incl. water rides)")
        print(f"  TEST tuned   : {fmt(r['test_tuned'])}  (own, incl. water rides)")
    (EVAL / "tuned_thresholds.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {EVAL / 'tuned_thresholds.json'}")


if __name__ == "__main__":
    main()
