#!/usr/bin/env python3
"""Comparison figure for the paper's Environment-Detection section.

Three single-axis panels (no dual-axis charts):
  (a) accuracy: per-class + aggregate metrics, zero-shot CLIP vs fine-tuned SegFormer;
  (b) efficiency: inference latency (ms/frame) with parameter counts annotated;
  (c) macro-F1 development trajectory across the optimization steps of both models.

Panel (a) is recomputed from source (labels + prediction CSVs + val-tuned
thresholds) and reproduces dev_documentation/environment_model_development.md §11.
Panel (b) latency/params are the measured constants quoted in §11 (Apple MPS,
warmed) — latency is hardware-dependent and reported, not re-timed here. Panel (c)
values are the historical per-step macro-F1 from §9 (CLIP) and §10 (SegFormer).

Writes fig_env_comparison.{pdf,png} @300 dpi to environment_model/evaluation/figures/.
Run: python environment_model/evaluation/make_env_figure.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "environment_model"))
import segmentation_common as sc  # noqa: E402

ENV = sc.CATEGORIES["environment"]  # [vegetation, water, city]
EVAL = REPO / "dataset" / "eval"
LABELS = REPO / "dataset" / "test_images" / "labels.csv"
FIGDIR = REPO / "environment_model" / "evaluation" / "figures"
PRED = {
    "clip": EVAL / "env_pred_zeroshot.csv",
    "segformer": EVAL / "env_pred_semseg.csv",
    "segformer_noft": EVAL / "env_pred_semseg_nofinetune.csv",
}

# --- dataviz reference palette, light mode: categorical slots 1 (blue) & 2 (aqua) ---
C_CLIP = "#2a78d6"          # slot 1 -> CLIP (entity colour, fixed)
C_SEG = "#1baf7a"           # slot 2 -> SegFormer (fine-tuned)
C_SEG_EDGE = "#0f7a55"      # darker aqua for tone-on-tone hatch (grayscale/CVD relief)
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRIDC, BAND = "#e1e0d9", "#ecebe6"

# --- measured constants from §11 (Apple M-series MPS, warmed) ---
LAT_MS = {"clip": 8.8, "segformer": 9.6}      # ms / frame, batch size 1
PARAMS_M = {"clip": 151.0, "segformer": 3.7}  # millions of parameters

# --- historical macro-F1 trajectories (own-test), §9 CLIP / §10 SegFormer ---
TRAJ_CLIP = ([0, 1, 2, 3, 4, 5],
             [0.632, 0.727, 0.771, 0.756, 0.846, 0.649],
             ["C2", "C3", "C4", "C5", "C6", "C7"])
TRAJ_SEG = ([1, 2, 3, 4, 5],
            [0.636, 0.707, 0.874, 0.847, 0.704],
            ["S7", "S8", "S9", "S10", "S11"])

# --- val-F1-argmax threshold grid, identical to tune_thresholds.py ---
GRID = np.unique(np.r_[np.round(np.logspace(-4, -2, 17), 6),
                       np.round(np.linspace(0.01, 0.10, 19), 3),
                       np.round(np.linspace(0.12, 0.95, 84), 3)])


def load_gt() -> pd.DataFrame:
    """Own-frames test/val labels with ride-split (mirrors tune_thresholds.load)."""
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
    gt["split"] = gt.ride.map(lambda r: split.get(r, "test"))  # unknown rides -> test
    return gt.set_index("filename")


def _f1(pred: np.ndarray, true: np.ndarray) -> float:
    tp = int((pred & true).sum()); fp = int((pred & ~true).sum())
    fn = int((~pred & true).sum())
    return 2 * tp / max(2 * tp + fp + fn, 1)


def _tune_on_val(score: np.ndarray, gt_bin: np.ndarray, is_val: np.ndarray) -> list[float]:
    thr = []
    for j in range(len(ENV)):
        yv = gt_bin[is_val, j].astype(bool); sv = score[is_val, j]
        if yv.sum() == 0:
            thr.append(0.5); continue
        best = max(((_f1(sv >= t, yv), t) for t in GRID), key=lambda x: (x[0], -x[1]))
        thr.append(float(best[1]))
    return thr


def eval_model(path: Path, thr: list[float] | None) -> dict:
    """Own-test multilabel metrics at `thr` (or val-tuned when thr is None)."""
    gt = load_gt()
    df = pd.read_csv(path).set_index("filename").reindex(columns=ENV).astype(float)
    common = gt.index.intersection(df.index)
    gt = gt.loc[common]; score = df.loc[common].values
    gt_bin = gt[ENV].astype(int).values
    is_test = (gt["split"] == "test").values
    if thr is None:
        thr = _tune_on_val(score, gt_bin, (gt["split"] == "val").values)
    yp = (score[is_test] >= np.array(thr)).astype(int)
    m = sc.multilabel_metrics(gt_bin[is_test], yp)
    m["n_test"] = int(is_test.sum())
    m["thr"] = [round(t, 6) for t in thr]
    return m


def compute() -> dict:
    th = json.loads((EVAL / "tuned_thresholds.json").read_text())
    thr_clip = [th["zeroshot"]["tuned"][c] for c in ENV]
    thr_seg = [th["semseg"]["tuned"][c] for c in ENV]
    r = {
        "clip": eval_model(PRED["clip"], thr_clip),
        "segformer": eval_model(PRED["segformer"], thr_seg),
        # ablation: no fine-tuning, holding the decision rule fixed (FT thresholds), as §11
        "segformer_noft": eval_model(PRED["segformer_noft"], thr_seg),
    }
    # ---- reproduce §11 head-to-head ----
    print(f"\nown-test frames: {r['clip']['n_test']}")
    hdr = f"{'model':<16}{'macroF1':>9}{'lblAcc':>9}{'exact':>9}" \
          + "".join(f"{c[:4]+'F1':>9}" for c in ENV)
    print(hdr); print("-" * len(hdr))
    for k in ("clip", "segformer", "segformer_noft"):
        m = r[k]
        pc = "".join(f"{m['per_class_f1'][i]:>9.3f}" for i in range(len(ENV)))
        print(f"{k:<16}{m['macro_f1']:>9.3f}{m['label_accuracy']:>9.3f}"
              f"{m['subset_accuracy']:>9.3f}{pc}")
    assert abs(r["clip"]["macro_f1"] - 0.649) < 0.01, r["clip"]["macro_f1"]
    assert abs(r["segformer"]["macro_f1"] - 0.704) < 0.01, r["segformer"]["macro_f1"]
    assert r["clip"]["n_test"] >= 1200, r["clip"]["n_test"]
    print("\n[ok] reproduces §11 (CLIP 0.649 / SegFormer 0.704 macro-F1)")
    return r


def _style_axis(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7"); ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=6.5, length=2.5, width=0.6)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK2)


def _bar(ax, x, h, color, hatch=None, edge=None):
    return ax.bar(x, h, color=color, width=0.4 if np.isscalar(x) else None,
                  edgecolor=edge or color, linewidth=0.5, hatch=hatch, zorder=3)


def make_figure(r: dict):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 7, "axes.titlesize": 7.5, "pdf.fonttype": 42,
        "svg.fonttype": "none", "figure.dpi": 300,
    })
    fig = plt.figure(figsize=(7.16, 2.45))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.5, 1.0, 1.7], wspace=0.42,
                          left=0.055, right=0.985, top=0.80, bottom=0.20)
    ax_a, ax_b, ax_c = (fig.add_subplot(gs[i]) for i in range(3))

    # ---------- (a) accuracy grouped bars ----------
    labels = ["Veg.\nF1", "Water\nF1", "City\nF1", "Macro\nF1", "Label\nacc.", "Exact\nmatch"]

    def vec(m):
        return [m["per_class_f1"][0], m["per_class_f1"][1], m["per_class_f1"][2],
                m["macro_f1"], m["label_accuracy"], m["subset_accuracy"]]
    vc, vs = vec(r["clip"]), vec(r["segformer"])
    x = np.arange(len(labels)); w = 0.38
    ax_a.bar(x - w / 2, vc, w, color=C_CLIP, edgecolor=C_CLIP, linewidth=0.5, zorder=3)
    ax_a.bar(x + w / 2, vs, w, color=C_SEG, edgecolor=C_SEG_EDGE, linewidth=0.5,
             hatch="///", zorder=3)
    for xi, (a, b) in enumerate(zip(vc, vs)):
        ax_a.text(xi - w / 2, a + 0.015, f"{a:.2f}", ha="center", va="bottom",
                  fontsize=5.6, color=INK2)
        ax_a.text(xi + w / 2, b + 0.015, f"{b:.2f}", ha="center", va="bottom",
                  fontsize=5.6, color=INK, fontweight="bold")
    ax_a.set_xticks(x); ax_a.set_xticklabels(labels, fontsize=6.2)
    ax_a.set_ylim(0, 1.06); ax_a.set_yticks(np.arange(0, 1.01, 0.25))
    ax_a.set_ylabel("score", fontsize=6.8, color=INK2)
    ax_a.set_title("(a)  Accuracy on own-test (1,210 frames)", loc="left",
                   color=INK, fontsize=7.4, fontweight="bold")
    ax_a.yaxis.grid(True, color=GRIDC, linewidth=0.6, zorder=0)
    ax_a.set_axisbelow(True); _style_axis(ax_a)
    # annotate the shared weakness
    ax_a.annotate("shared\nweakness", xy=(1, max(vc[1], vs[1]) + 0.04),
                  xytext=(1, 0.62), ha="center", fontsize=5.6, color=MUTED,
                  arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))

    # ---------- (b) efficiency: latency bars + param annotation ----------
    bx = np.arange(2)
    ax_b.bar(bx[0], LAT_MS["clip"], 0.55, color=C_CLIP, edgecolor=C_CLIP,
             linewidth=0.5, zorder=3)
    ax_b.bar(bx[1], LAT_MS["segformer"], 0.55, color=C_SEG, edgecolor=C_SEG_EDGE,
             linewidth=0.5, hatch="///", zorder=3)
    for i, k in enumerate(("clip", "segformer")):
        ax_b.text(bx[i], LAT_MS[k] + 0.15, f"{LAT_MS[k]:.1f}", ha="center",
                  va="bottom", fontsize=6, color=INK)
        ax_b.text(bx[i], 0.4, f"{PARAMS_M[k]:.0f}M" if PARAMS_M[k] >= 10
                  else f"{PARAMS_M[k]:.1f}M", ha="center", va="bottom",
                  fontsize=6.2, color="white", fontweight="bold")
    ax_b.set_xticks(bx); ax_b.set_xticklabels(["CLIP", "SegF."], fontsize=6.4)
    ax_b.set_ylim(0, 11.4); ax_b.set_yticks([0, 5, 10])
    ax_b.set_ylabel("latency (ms/frame)", fontsize=6.6, color=INK2)
    ax_b.set_title("(b)  Cost", loc="left", color=INK, fontsize=7.4, fontweight="bold")
    ax_b.yaxis.grid(True, color=GRIDC, linewidth=0.6, zorder=0)
    ax_b.set_axisbelow(True); _style_axis(ax_b)
    ax_b.text(0.5, -0.42, "bar label = #params (40$\\times$ gap)", transform=ax_b.transAxes,
              ha="center", va="top", fontsize=5.4, color=MUTED)

    # ---------- (c) macro-F1 development trajectory ----------
    cx, cy, cl = TRAJ_CLIP
    sxx, sy, sl = TRAJ_SEG
    ax_c.axvspan(4.5, 5.5, color=BAND, zorder=0)
    ax_c.plot(cx, cy, "-o", color=C_CLIP, ms=3.2, lw=1.4, zorder=3)
    ax_c.plot(sxx, sy, "-s", color=C_SEG, ms=3.2, lw=1.4, zorder=3,
              markeredgecolor=C_SEG_EDGE, markeredgewidth=0.5)
    # label peaks and finals only (avoid clutter)
    ax_c.annotate("peak 0.846", (4, 0.846), textcoords="offset points", xytext=(-3, 5),
                  fontsize=5.4, color=C_CLIP, ha="right")
    ax_c.annotate("peak 0.874", (3, 0.874), textcoords="offset points", xytext=(-2, 5),
                  fontsize=5.4, color=C_SEG_EDGE, ha="center", fontweight="bold")
    ax_c.annotate("0.649", (5, 0.649), textcoords="offset points", xytext=(5, -8),
                  fontsize=5.6, color=C_CLIP, ha="right")
    ax_c.annotate("0.704", (5, 0.704), textcoords="offset points", xytext=(5, 4),
                  fontsize=5.6, color=C_SEG_EDGE, ha="right", fontweight="bold")
    ax_c.text(5.0, 0.905, "water-incl.\ntest", ha="center", va="top",
              fontsize=5.2, color=MUTED)
    # C-steps and S-steps are independent tracks; x = successive optimization steps
    ax_c.set_xlim(-0.4, 5.7); ax_c.set_ylim(0.4, 0.95)
    ax_c.set_yticks(np.arange(0.4, 0.91, 0.1))
    ax_c.set_xticks([])
    ax_c.set_xlabel("optimization step $\\rightarrow$", fontsize=6.4, color=INK2)
    ax_c.set_ylabel("macro-F1", fontsize=6.8, color=INK2)
    ax_c.set_title("(c)  Optimization trajectory", loc="left", color=INK,
                   fontsize=7.4, fontweight="bold")
    ax_c.yaxis.grid(True, color=GRIDC, linewidth=0.6, zorder=0)
    ax_c.set_axisbelow(True); _style_axis(ax_c)

    # ---------- shared legend ----------
    handles = [Patch(facecolor=C_CLIP, edgecolor=C_CLIP, label="CLIP ViT-B/32 (zero-shot)"),
               Patch(facecolor=C_SEG, edgecolor=C_SEG_EDGE, hatch="///",
                     label="SegFormer-B0 (fine-tuned)")]
    fig.legend(handles=handles, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               frameon=False, fontsize=6.6, handlelength=1.2, columnspacing=1.6,
               handletextpad=0.5)

    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"fig_env_comparison.{ext}", dpi=300,
                    bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"\nsaved -> {FIGDIR / 'fig_env_comparison.pdf'} (+ .png)")


if __name__ == "__main__":
    make_figure(compute())
