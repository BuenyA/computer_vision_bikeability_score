# Changes to Paper — "Environment Classification" Page

**Purpose.** Input spec for the next prompt, which will redo the full *Environment
Classification* page of `paper/bikeability_paper.tex`. This file documents the
requested structural changes to the two optimization tables so the redo can be done
unambiguously.

**Target file:** `paper/bikeability_paper.tex` (the current, active paper — the old
`latex/` copy is gone).

**Source of all numbers:** `dev_documentation/environment_model_development.md`
(§9 = CLIP log, §10 = SegFormer log, §11 = final head-to-head).

**Label ↔ "table N" mapping** (the LaTeX auto-number differs from the informal
numbering used in the request — refer to the `\label` to avoid confusion):

| request says | LaTeX `\label` | current location | model |
|---|---|---|---|
| "table 2" | `tab:clip-opt` | line ~228 | Zero-shot CLIP |
| "table 3" | `tab:seg-opt`  | line ~261 | Fine-tuned SegFormer |

---

## Change 1 — Table `tab:seg-opt` (SegFormer, request's "table 3")

### What is requested
Split the single step list into **two phases** and give **each phase its own metric
column**, so the reader sees two columns of improving numbers that justify choosing
SegFormer:

- **Training / fine-tuning phase → `mIoU`** (Mapillary dev set, mask-derived GT).
- **Evaluation on own test data → `Macro-F1`** (own cyclist-POV frames).

Only these two metric columns are needed; drop the mixed "Effect / decision" prose
metrics. The goal is a clean "numbers go up" story per phase.

### Why the split is correct (keep this straight in the redo)
The two metrics are **not** on the same data and are **not** interchangeable:
- `mIoU` is a **pixel-overlap** metric, only computable where pixel masks exist →
  the **Mapillary dev set**. It measures how well the model *segments* during
  training. It is a SegFormer-only, training-time diagnostic.
- `Macro-F1` is a **per-frame presence** metric on the **own test frames** (which
  have presence labels, not masks). It measures real-domain decision quality and is
  the metric the final CLIP-vs-SegFormer decision turns on.

So the two columns are two different lenses on two different datasets — the phase
split is exactly what keeps them from being conflated (this fixes the current
`per-class F1 (mIoU during training)` confusion in the Metrics paragraph too).

### Numbers to use (from §10)

**Phase A — Training / fine-tuning (metric column: mIoU on Mapillary dev):**

| # | Type | Step | mIoU | (supporting: water IoU) |
|---|---|---|---|---|
| S1 | FT | Cityscapes-B0, random Mapillary | 0.62 (baseline) | 0.00 |
| S2 | FT | LR 6e-5 → 2e-4 | ~0.62 | 0.00 |
| S3–4 | data | Balanced + water ×2–6 + ADE water | **0.82** | 0.70 |
| S5 | FT | Base swap → ADE20K (water prior) | **0.857** (best) | 0.778 |
| S6 | data | Copy-paste water aug | (rejected) | −0.11 |

Improving arc for the column: **0.62 → 0.82 → 0.857**. (Water IoU 0.00 → 0.70 →
0.778 is the strongest sub-story; optionally show it as a second small column or fold
it into the caption, since water is the whole reason the training phase exists.)

**Phase B — Evaluation on own test data (metric column: Macro-F1 on own frames):**

| # | Type | Step | Macro-F1 (own-test) |
|---|---|---|---|
| S7 | eval | First own-frame eval + val-tuned thresholds | 0.636 |
| S8 | FT | Explicit "other" background class | 0.707 |
| S9 | FT | 3-class taxonomy retrain | 0.874 |
| S10 | eval | Fine-tuning ablation (no-FT reference) | 0.847 → FT worth +0.027 |
| S11 | eval | Fine threshold grid (→1e-4), water-inclusive final | **0.704 (FINAL — beats CLIP 0.649)** |

Improving arc for the column: **0.636 → 0.707 → … → 0.704 final.**

### ⚠️ Honesty caveat the redo MUST handle (basis shift)
The macro-F1 numbers are **not** on one fixed test set — the basis changes:
- S7/S8 = 4-class own-test (291 frames, no water).
- S9 = 3-class (veg+city) own-test (no water).
- S11 = 3-class, **1,210 frames incl. water** (the final headline set).

So **S9 (0.874) → S11 (0.704) is a drop only because the hard water class enters the
test set**, not a regression. Do **not** present 0.874 → 0.704 as "going down."
Recommended handling: treat **S11 = 0.704 as the single final, comparable number**
and state that it is on the harder water-inclusive set, where it still **beats CLIP's
0.649 on the identical 1,210-frame test** — that is the sentence that actually
supports the decision. S9's 0.874 can stay as the pre-water peak with a footnote, or
be dropped from the "improving" column to avoid the apparent dip.

### Layout suggestion (two columns, one row per step)
Give the table two metric columns; each step fills the column for its phase and
leaves the other blank (staircase), OR use two stacked blocks with a `\midrule`
separating "Training / fine-tuning" from "Evaluation (own test)". Either satisfies
"see the numbers improving in those two columns." Keep the `Type` (FT / data / eval)
tag — it already tells the reader which phase a row belongs to.

---

## Change 2 — Table `tab:clip-opt` (CLIP, request's "table 2")

### What is requested
Reduce to **one metric column, labelled `Macro-F1 score`**, listing the macro-F1 at
each step chronologically so the improvement is visible. Drop the "Effect / decision"
column's role as the metric carrier (short decision notes may stay as a thin column,
but the headline is the macro-F1 progression).

### Numbers to use (from §9)

| # | Step | Macro-F1 score |
|---|---|---|
| C1 | Aligned prompt pairs, global threshold | 0.547 |
| C2 | Ride-disjoint val/test split (leakage-free) | 0.632 |
| C3 | Per-class threshold calibration | 0.727 (+0.095, top lever) |
| C4 | Prompt ensembling, 6–8 templates | 0.771 (+0.044) |
| C5 | Larger backbone ViT-L/14 | 0.756 (−0.015, rejected) |
| C6 | 3-class taxonomy (veg. merge) | 0.846 (best config) |
| C7 | Water-inclusive final test | 0.649 (water F1 .62→.21: artifact) |

Improving arc: **0.547 → 0.632 → 0.727 → 0.771 → 0.846** (the optimization gains).

### ⚠️ Same basis-shift caveat (important for CLIP)
CLIP's story is **not** monotonic to the end, and that is the point:
- C5 (0.756) is a **rejected branch** — lower than C4 (0.771). Mark it as rejected so
  it doesn't read as a regression in the main line.
- C6 → C7 (0.846 → 0.649) **drops because water enters the test set** (1,210 frames),
  exposing CLIP's water strength as a small-validation artifact. This drop is the
  *finding*, not a failure to show improvement.

Recommended handling: show C1–C6 as the "optimization improves macro-F1" arc, and set
off **C7 (0.649) as the final water-inclusive reality check** — the number that,
against SegFormer's 0.704, decides the model choice. Do not force C7 into the
"improving" narrative.

---

## Consistency notes for the redo (both tables)
- The two tables now report on **different data** (CLIP col = own-test macro-F1;
  SegFormer = Mapillary-dev mIoU **plus** own-test macro-F1). The **only
  cross-model, cross-table comparable number is the final own-test macro-F1**:
  **SegFormer 0.704 vs CLIP 0.649** on the same 1,210 frames — make sure this pairing
  appears clearly (it is the decision).
- Update the two captions to name the metric now shown (e.g. `tab:seg-opt` caption
  should say it reports training-phase mIoU and own-test macro-F1).
- The `Metrics and Protocol` paragraph (line ~288) should be fixed in the same pass:
  remove `per-class F1 (mIoU during training)` from the shared-metrics list; state
  that **macro-F1 on the 1,210-frame own-test set is the deciding metric** (it is the
  one metric defined for *both* models — CLIP has no pixel mask, so mIoU is
  SegFormer-only), and that **mIoU is used only as a SegFormer training-time
  diagnostic on the Mapillary dev set**.

## Open decisions for the next prompt
1. Table `tab:seg-opt` layout: two staircase columns vs. two stacked blocks (see
   Change 1 layout note).
2. Whether to keep water IoU as a second training-phase number or fold it into prose.
3. Whether to keep S9 (0.874) / C6 (0.846) pre-water peaks in the "improving" column
   with a footnote, or drop them to avoid the apparent post-water dip.

---

# Proposed full page structure (Environment Classification)

This is the target section layout for the full rewrite of the page. Sections A–F
below replace the current subsections. The two table changes (Change 1 = `tab:seg-opt`,
Change 2 = `tab:clip-opt`) and the metric caveats above still apply and are referenced
where relevant.

## Intro (Task)
> Riding through different environments such as nature ("vegetation"), water
> landscapes or the urban jungle ("city") strongly impacts the individual riding
> experience. Green vegetation and water landscapes improve the experience, while too
> much city or urban traffic makes the ride strenuous. The different environments are
> classified with a **multi-label, whole-image** approach.

*(Note: original draft said "rural jungle"/"rural traffic" — read as **urban**
jungle / **urban** traffic; the "city" class is the built-up environment.)*

## A — Task and Model Selection
- **Task.** Find the best model to classify the surrounding environment on our own
  test frames, captured with a pod-mounted camera on the bike.
- **Model comparison.** Zero-shot CLIP vs. fine-tuned SegFormer.
- **REMOVE** the current "Environment Classes and Taxonomy" subsection (drop the
  classes/taxonomy discussion from the page).

## B — Model Comparison (major differences, compact)
Condense the two model descriptions below into a tight, precise summary.

**CLIP (zero-shot classification)**
- *Text–image matching, no training.* Per class it holds a prompt pair ("a photo of
  water" vs. "a photo without water"); CLIP embeds the whole frame and each prompt
  into a shared space and picks the closest prompt.
- *Whole-frame, presence-only.* One number per class — a present-probability for the
  entire image — with no notion of *where* or *how much*.
- *Threshold → label.* Each probability is compared to a per-class threshold. Improve
  it via better prompts + thresholds (no weight changes).

**SegFormer (fine-tuned semantic segmentation)**
- *Per-pixel labelling.* A Transformer encoder uses attention across image patches to
  assign every pixel a class (vegetation / water / city / other) → a full-resolution
  mask.
- *Area fractions, not just presence.* Class pixels ÷ total pixels → a class-area
  fraction encoding *how much* of the scene it covers (richer, and exactly what the
  beauty score consumes).
- *Threshold → label.* Each fraction is compared to a per-class threshold. Improve it
  via training data + model weights (it is actually learned/fine-tuned).

**Core difference.** CLIP asks *"is this class present?"* (whole-frame, off-the-shelf);
SegFormer asks *"which pixels, how much?"* (spatial, trained). CLIP needs no labels
but outputs only presence; SegFormer needs training but outputs spatial extent — so
**only SegFormer produces the area fractions the score needs, and only SegFormer can
be measured with pixel-level mIoU.**

## C — Zero-Shot CLIP: Optimization
- Text = major improvements only (per-class threshold calibration +0.095; prompt
  ensembling +0.044; ViT-L/14 rejected; final water-inclusive test exposed the water
  artifact).
- **Insert Table 2 = `tab:clip-opt`** with the single **Macro-F1 score** column
  (chronological improvement) — see **Change 2** above for numbers and the basis-shift
  caveat.

## D — Fine-Tuned SegFormer: Optimization
- Text = major improvements only, and **explicitly separate the two phases**:
  1. **Training / fine-tuning on Mapillary with pixel masks → measured by mIoU**
     (data was the decisive lever: water IoU 0 → 0.70; ADE20K base swap; "other"
     class).
  2. **Switch to own-frame evaluation → measured by macro-F1**, ending with the
     **final score on the unseen water-inclusive test set** (macro-F1 0.704).
- **Insert Table 3 = `tab:seg-opt`** with two metric columns (Training/FT → **mIoU**;
  Eval → **Macro-F1**) — see **Change 1** above for numbers, phase split, and caveats.

## E — Metrics and Score Calculation
- **Keep only macro-F1 and exact-match** — **delete label-accuracy from both the text
  and the diagram.**
- Explain each in one sentence:
  - **Macro-F1** — the per-class F1 (harmonic mean of precision and recall) averaged
    equally over the three classes, so the rare *water* class counts as much as the
    common ones.
  - **Exact-match** — the fraction of frames whose *entire* multi-label set
    (all three classes at once) is predicted correctly.

## F — Results and Decision
- **DELETE the runtime/parameter bar charts and the per-class accuracy bar chart**
  (old `fig:env`); fold the runtime/param finding into the decision text instead.
- **INSERT the new `fig_isof1_scatter`** as the single results figure
  (source: `dev_documentation/figures/fig_isof1_scatter.png` — **copy to
  `paper/images/` and reference via `\includegraphics`** for the LaTeX build).

**Results (key points)**
- SegFormer wins overall: **macro-F1 0.704 vs 0.649**, leading every aggregate metric
  (exact-match 0.755 vs 0.691). *(label-acc dropped per Section E — see consistency
  note below.)*
- Tie on the common classes: vegetation (0.930 vs 0.941) and city (0.799 vs 0.802) are
  effectively equal — they don't decide anything.
- Water (rare, 3% of frames) decides it: SegFormer F1 **0.382 vs 0.206**.
- Water is the shared limitation: even the winner misses ~67% of true-water frames
  (recall 0.33) — a data-scarcity blindness, not miscalibration.

**Two most important insights from the diagram**
1. **The decision lives in the bottom-left (water).** Vegetation and city cluster
   top-right on the same high-F1 band; only the water points are far apart — so the
   model choice rests entirely on the hardest class.
2. **SegFormer wins water by precision, not recall.** The water arrow moves up-left
   onto a higher iso-F1 contour: CLIP catches more water (higher recall) but at
   precision 0.13 (7 of 8 flags wrong), while SegFormer trades some recall for
   precision 0.46 — trustworthy credit instead of phantom water.

**Decision**
- **Deploy the fine-tuned SegFormer.** It wins the deciding class (water) with
  reliable precision, matches CLIP everywhere else, is 40× smaller (3.7M vs 151M) at
  comparable latency, and outputs area fractions that feed the score directly — which
  CLIP's presence probabilities cannot.
- **Accept water as a known limitation:** the residual failure is recall (missed
  water), fixable only with more in-domain water training frames, not a better
  decision rule.

### ⚠️ Consistency note for Sections E ↔ F
Section E removes **label-accuracy** everywhere. The first Results bullet as originally
drafted still cited "label-acc 0.911 vs 0.888" — that parenthetical must be **dropped**
(done above) so the results text matches the metric set defined in E.
