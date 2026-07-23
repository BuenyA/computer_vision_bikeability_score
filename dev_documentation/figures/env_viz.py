"""Build two candidate figures comparing CLIP vs fine-tuned SegFormer on the
own-test set (environment classification): (1) per-class PR curves,
(2) iso-F1 operating-point scatter. Prints the underlying numbers."""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def pr_curve(y, s):
    """precision, recall arrays (sorted by increasing recall)."""
    order = np.argsort(-s, kind="mergesort")
    ys, ss = y[order].astype(float), s[order]
    tp, fp = np.cumsum(ys), np.cumsum(1 - ys)
    keep = np.r_[np.where(np.diff(ss) != 0)[0], len(ss) - 1]
    P = tp[keep] / (tp[keep] + fp[keep])
    tot = ys.sum()
    R = tp[keep] / tot if tot else np.zeros_like(P)
    P, R = np.r_[1.0, P], np.r_[0.0, R]        # (recall 0, precision 1) start point
    idx = np.argsort(R)
    return P[idx], R[idx]

def average_precision(y, s):
    """AP = sum_n (R_n - R_{n-1}) * P_n  (sklearn convention)."""
    order = np.argsort(-s, kind="mergesort")
    ys = y[order].astype(float)
    tp, fp = np.cumsum(ys), np.cumsum(1 - ys)
    tot = ys.sum()
    if tot == 0:
        return 0.0
    P, R = tp / (tp + fp), tp / tot
    return float(np.sum((R - np.r_[0.0, R[:-1]]) * P))

REPO = "/Users/hendrickfischer/Documents/Education/CAS_Master/Vorlesungen/Semester_2/Bildverarbeitung/RRCP_Project/computer_vision_bikeability_score"
R = f"{REPO}/research"
OUT = "/private/tmp/claude-501/-Users-hendrickfischer-Documents-Education-CAS-Master-Vorlesungen-Semester-2-Bildverarbeitung-RRCP-Project-computer-vision-bikeability-score/e9e05466-59e1-402f-ad84-f5b2f7c2eb1f/scratchpad"

CLASSES = ["vegetation", "water", "city"]
THR = {"seg":  {"vegetation": 0.25,  "water": 0.0001, "city": 0.005623},
       "clip": {"vegetation": 0.035, "water": 0.39,   "city": 0.73}}
MODELS = {"seg": "SegFormer", "clip": "CLIP"}
COL = {"seg": "#2a78d6", "clip": "#eb6834"}   # blue (deployed) vs orange — CVD-safe pair

# ---- reconstruct the 1,210-frame own-test set -----------------------------
labels = pd.read_csv(f"{R}/dataset/test_images/labels.csv")
split  = pd.read_csv(f"{R}/dataset/eval/own_split.csv")
labels = labels[labels.filename.str.startswith("own_frames/")].copy()
labels = labels.merge(split, on="ride_id", how="left")
labels["split"] = labels["split"].fillna("test")   # unknown rides default to test
test = labels[(labels.split == "test") & (labels.unsure == 0) & (labels.reject == 0)].copy()

gt = test[["filename"] + CLASSES].rename(columns={c: f"gt_{c}" for c in CLASSES})
def prep(path, name):
    p = pd.read_csv(path).rename(columns={c: f"{name}_{c}" for c in CLASSES})
    return p[["filename"] + [f"{name}_{c}" for c in CLASSES]]
df = (gt.merge(prep(f"{R}/dataset/eval/env_pred_semseg.csv", "seg"), on="filename")
        .merge(prep(f"{R}/dataset/eval/env_pred_zeroshot.csv", "clip"), on="filename"))
print(f"own-test frames joined: {len(df)}")

def op_point(y, s, t):
    pred = (s >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec  = tp / (tp + fn) if tp + fn else 0.0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1

# ---- numbers --------------------------------------------------------------
rows, macro = [], {"seg": [], "clip": []}
for c in CLASSES:
    y = df[f"gt_{c}"].values
    for m in ["seg", "clip"]:
        s = df[f"{m}_{c}"].values
        p0, r0, f0 = op_point(y, s, THR[m][c])
        ap = average_precision(y, s)
        macro[m].append(f0)
        rows.append((c, MODELS[m], int(y.sum()), p0, r0, f0, ap))
print(f"\n{'class':<11}{'model':<10}{'n+':>5}{'prec':>7}{'rec':>7}{'F1':>7}{'AP':>7}")
for c, mm, n, p, r, f, ap in rows:
    print(f"{c:<11}{mm:<10}{n:>5}{p:>7.3f}{r:>7.3f}{f:>7.3f}{ap:>7.3f}")
print(f"\nmacro-F1  SegFormer {np.mean(macro['seg']):.3f}   CLIP {np.mean(macro['clip']):.3f}")

# ---- style ----------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
    "text.color": "#0b0b0b", "axes.labelcolor": "#0b0b0b",
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "figure.facecolor": "white", "axes.facecolor": "white", "svg.fonttype": "none",
})
INK, MUT, GRID = "#0b0b0b", "#898781", "#e1e0d9"

# ================= FIGURE 1 — per-class PR curves ==========================
fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
for ax, c in zip(axes, CLASSES):
    y = df[f"gt_{c}"].values; prev = y.mean()
    ax.axhline(prev, color=MUT, ls=(0, (1, 2)), lw=0.9, zorder=1)
    for m in ["clip", "seg"]:                        # seg drawn last = on top
        s = df[f"{m}_{c}"].values
        ap = average_precision(y, s)
        prec, rec = pr_curve(y, s)
        ax.plot(rec, prec, color=COL[m], lw=2.6 if m == "seg" else 1.6,
                ls="-" if m == "seg" else (0, (4, 2)), solid_capstyle="round",
                label=f"{MODELS[m]}  (AP {ap:.2f})", zorder=4 if m == "seg" else 3)
        p0, r0, f0 = op_point(y, s, THR[m][c])
        ax.plot(r0, p0, "o", color=COL[m], ms=9, mec="white", mew=1.4, zorder=6)
    ax.set_title(f"{c.capitalize()}   (n$_+$={int(y.sum())}, {prev*100:.0f}% of frames)", fontsize=9.5)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(np.arange(0, 1.01, 0.25)); ax.set_yticks(np.arange(0, 1.01, 0.25))
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision" if c == CLASSES[0] else "")
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.legend(loc="lower left", fontsize=7.6, frameon=False, handlelength=1.6)
    if c == "water":
        ax.annotate("recall capped ≈0.33\n(model blind, not miscalibrated)",
                    xy=(0.33, 0.42), xytext=(0.42, 0.75), fontsize=7.4, color=INK,
                    ha="left", arrowprops=dict(arrowstyle="->", color=MUT, lw=0.9))
fig.suptitle("Per-class Precision–Recall — fine-tuned SegFormer vs zero-shot CLIP  (own-test, 1,210 frames)",
             fontsize=11, fontweight="bold", x=0.5, y=1.02, ha="center")
fig.text(0.5, -0.03, "●  tuned operating point (deployed thresholds)       · · ·  no-skill baseline (class prevalence)       "
         "SegFormer wins water & city AP; both collapse on water recall.",
         ha="center", fontsize=7.6, color="#52514e")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_pr_curves.png", dpi=200, bbox_inches="tight")
print(f"\nwrote {OUT}/fig_pr_curves.png")

# ================= FIGURE 2 — iso-F1 operating-point scatter ================
fig2, ax = plt.subplots(figsize=(5.8, 5.6))
rr = np.linspace(0.0005, 1, 600)
for f in (0.2, 0.4, 0.6, 0.8):
    pp = f * rr / (2 * rr - f)
    ok = (2 * rr - f > 0) & (pp <= 1.02) & (pp >= 0)
    ax.plot(rr[ok], pp[ok], color="#cfcec6", lw=0.9, zorder=1)
    rlab = 0.965; plab = f * rlab / (2 * rlab - f)
    if plab <= 0.98:
        ax.text(rlab, plab + 0.012, f"F1={f:g}", fontsize=7, color=MUT, ha="right", va="bottom")
MARK = {"vegetation": "o", "water": "s", "city": "^"}
for c in CLASSES:
    y = df[f"gt_{c}"].values
    pts = {}
    for m in ["seg", "clip"]:
        p0, r0, _ = op_point(y, df[f"{m}_{c}"].values, THR[m][c])
        pts[m] = (r0, p0)
    # arrow CLIP -> SegFormer (shows the gain direction)
    ax.annotate("", xy=pts["seg"], xytext=pts["clip"],
                arrowprops=dict(arrowstyle="-|>", color="#a8a79f", lw=1.1, shrinkA=9, shrinkB=9), zorder=2)
    for m in ["clip", "seg"]:
        r0, p0 = pts[m]
        ax.scatter(r0, p0, s=150, marker=MARK[c], color=COL[m],
                   edgecolor="white", linewidth=1.4, zorder=5)
    r0, p0 = pts["seg"]
    ax.annotate(c.capitalize(), xy=(r0, p0), xytext=(6, 6), textcoords="offset points",
                fontsize=8.4, color=INK, fontweight="bold")
ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_xticks(np.arange(0, 1.01, 0.2)); ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
# legends: colour = model, marker = class
from matplotlib.lines import Line2D
leg_model = [Line2D([0], [0], marker="o", color="w", markerfacecolor=COL["seg"], markersize=10, label=f"SegFormer  (macro-F1 {np.mean(macro['seg']):.2f})"),
             Line2D([0], [0], marker="o", color="w", markerfacecolor=COL["clip"], markersize=10, label=f"CLIP  (macro-F1 {np.mean(macro['clip']):.2f})")]
leg_class = [Line2D([0], [0], marker=MARK[c], color="w", markerfacecolor="#6b6a64", markersize=10, label=c.capitalize()) for c in CLASSES]
l1 = ax.legend(handles=leg_model, loc="lower left", fontsize=8.2, frameon=False, title="Model  (arrow: CLIP to SegFormer)")
l1.get_title().set_fontsize(8.2); ax.add_artist(l1)
ax.legend(handles=leg_class, loc="upper left", bbox_to_anchor=(0, 0.82), fontsize=8, frameon=False, title="Class")
ax.set_title("Operating points in Precision–Recall space (iso-F1 contours)\nSegFormer vs CLIP — own-test, 1,210 frames",
             fontsize=10.5, fontweight="bold")
fig2.tight_layout()
fig2.savefig(f"{OUT}/fig_isof1_scatter.png", dpi=200, bbox_inches="tight")
print(f"wrote {OUT}/fig_isof1_scatter.png")
