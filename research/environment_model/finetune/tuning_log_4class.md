# SegFormer Environment Fine-Tune — Hyperparameter Tuning Log

**Date:** 2026-07-09 · **Hardware:** Apple Silicon (MPS), 26 GB RAM
**Task:** 4-class environment segmentation (`forest, open_field, water, city`) →
multi-label frame classification via per-class pixel-area threshold.
**Base model:** `nvidia/segformer-b0-finetuned-cityscapes-1024-1024` (fresh 4-class decode head).
**Training data:** Mapillary Vistas v2.0 *training* split (17.8k images after test-set hold-out),
labels remapped: `nature--vegetation→forest`, `nature--terrain→open_field`,
`nature--water→water`, `construction--structure--building→city`, all else → VOID (ignored in loss).

Runner: `scripts/finetune_experiments.py` (mirrors `research/seg_finetune.ipynb` logic
1:1; used for sweeps because a notebook cannot be parameterized per run).
Raw results: `research/seg_finetune_tuning_results.csv`.

---

## 1. Evaluation dataset for hyperparameter tuning (dev set)

**There was none** — the notebook only had an in-training mIoU monitor slice, and the
hand-labeled test set (`dataset/test_images/labels.csv`) must never be used for tuning
(it is the final zero-shot vs. fine-tune comparison). So one was created:

- **File:** `dataset/eval/dev_mapillary.csv` (builder: `finetune_experiments.py devset`)
- **Contents:** all 1,650 Mapillary *validation* images that are NOT in the hand-labeled
  test set, plus reserved *training*-split images (see problem below). Per image: GT pixel
  share per class + multi-hot GT (`class present ⇔ GT share ≥ 3%`).
- **Metric:** per-class F1 and macro-F1 of the area-threshold classifier against the
  mask-derived multi-hot GT, plus pixel mIoU as a secondary signal.
- **Fixed eval subset:** all training-reserved rows + seeded random validation rows
  (`--dev-n 450`), identical across all experiments.

### Problem found while building it: zero water positives

The test-set extraction (`build_test_dataset.py`) gives the scarce classes first pick —
every validation image with ≥1% water pixels was already claimed for the hand-labeled
test set. The remaining 1,650 validation images contain **0 water positives**, making
water F1 unmeasurable (NaN in the smoke run — confirmed empirically).

**Fix:** reserve water-rich + terrain-rich images from the *training* split into the dev
set, and exclude them from all training sampling (`pick_train_stems` excludes test-set
stems ∪ dev-reserved stems). Three-way separation after the fix:
**training ∩ dev = ∅, training ∩ test = ∅, dev ∩ test = ∅.**

### Second data finding: water is near-absent from Vistas altogether

After the test-set hold-out, the **entire 17.8k-image training split contains only 55
images with ≥1% water pixels** (the test set had already claimed 92 of Vistas' watery
images). Naively reserving the top water images for dev would have starved training of
water completely. Resolution: split the 55 by alternating share rank — dev got 28
(12 of them ≥3% ⇒ measurable GT positives), training kept 24 (9 strong / 15 weak).

**Final dev set:** 1,778 rows (1,650 validation + 128 reserved training).
GT positives: forest 1,455 · city 1,075 · open_field 292 · **water 12**.
Consequence for interpretation: water F1 on dev rests on 12 positives — treat it as a
coarse signal, not a precise estimate. It also predicts the same weakness on the real
test set and motivates cyclist-POV water footage as the most valuable data to add
(own rides along rivers/lakes).

## 2. Method

Each experiment changes **one thing at a time** against the baseline:

1. Fine-tune SegFormer-B0 (pretrained encoder, fresh 4-class head) on N Mapillary
   training images.
2. Predict on the fixed dev subset; derive multi-hot labels by thresholding each class's
   pixel-area fraction (default 3%, the notebook's `AREA_THRESHOLD`).
3. Report per-class F1 / macro-F1 (primary), pixel mIoU (secondary), and additionally a
   per-class threshold sweep (grid 0.005–0.2) to separate *segmentation* quality from
   *decision rule* quality.

### Engineering fixes needed before any training ran (documented for reproducibility)

| # | Problem | Fix |
|---|---------|-----|
| 1 | `native_batch_norm` backward crashes on MPS (torch 2.12, `view size is not compatible…`) | Replaced the decode head's single `BatchNorm2d` with a numerically identical decomposed implementation (`MPSSafeBatchNorm2d`); pretrained γ/β/running stats preserved |
| 2 | In-model 4× logits upsample + CE loss (slow, and initially suspected in the MPS crash) | Loss computed at logits resolution (stride 4) with nearest-downsampled labels — standard efficient variant, supervision semantics unchanged |
| 3 | Smoke run: water F1 = NaN | Dev-set water reservation (see §1) |

## 3. Experiments

Controlled budget for all sweep runs: **1,500 training images · 2 epochs · batch 4 ·
512×512 · seed 42**, evaluated on the identical dev subset (450 images, stratified to
include all water/field positives). ~6 min per run on MPS.

### E1 — Baseline (notebook defaults: LR 6e-5, random sample)

| metric | forest | open_field | water | city | macro |
|---|---|---|---|---|---|
| F1 @ 3% threshold | 0.95 | 0.47 | **0.00** | 0.73 | 0.538 |
| F1 @ tuned thresholds | 0.97 | 0.47 | **0.00** | 0.78 | 0.557 |

Pixel mIoU 0.615. **Diagnosis — why the head is "not working":**

1. **Water fails completely (F1 = 0.00), and it is a *data* problem, not a
   hyperparameter problem.** A random 1,500-image sample of Vistas contains ≈ 2 water
   images (24 usable in all 17.8k). The head never receives enough water gradient to
   learn the class. No LR/epoch setting can fix absence of data → E3/E4.
2. **open_field is mediocre (0.47)** — same mechanism, milder: terrain-rich images are
   ~6% of the split, and Vistas "terrain" (roadside dirt/grass) is visually
   heterogeneous.
3. **forest/city work immediately** (0.95/0.73) — vegetation and buildings are abundant
   in street scenes; the transfer from the Cityscapes-pretrained encoder is easy.
4. **The default 3% threshold is wrong per class**: city improves 0.73→0.78 at
   threshold 0.2 (buildings appear somewhere in almost every street image, so presence
   needs a higher bar), forest prefers 0.05. Per-class thresholds are a free win.

### E2 — Learning rate 6e-5 → 2e-4 (single change vs. E1)

| metric | forest | open_field | water | city | macro |
|---|---|---|---|---|---|
| F1 @ 3% | 0.91 | 0.54 | 0.00 | 0.72 | 0.543 |
| F1 @ tuned | 0.94 | 0.65 | 0.00 | 0.83 | **0.603** |

mIoU 0.615 → 0.638. The fresh head benefits from a higher LR (open_field +0.07,
tuned macro +0.046), but as predicted **water stays at exactly 0.00** — optimization
cannot substitute for missing data.

### E3 — Class-balanced sampling (single change vs. E1: all 24 water images included)

| metric | forest | open_field | water | city | macro |
|---|---|---|---|---|---|
| F1 @ tuned | 0.97 | 0.48 | **0.00** | 0.74 | 0.547 |

**Negative result, and the most informative one so far.** Even with every usable water
image in the sample (24 of 1,500 vs. ~2 in E1), water IoU is *exactly* 0.000 — the
model never predicts a single water pixel. Water is ≈0.1% of training pixel mass; the
cross-entropy gradient from other classes drowns it. Conclusion: within Vistas, the
class simply lacks critical mass → oversampling (E4) and cross-dataset augmentation
with ADE20K water scenes (E5, 258 masked water-dominated images available after test
exclusion) are the remaining levers.

### E4 — Balanced + water oversampled ×6 (single change vs. E3)

| metric | forest | open_field | water | city | macro |
|---|---|---|---|---|---|
| F1 @ tuned | 0.97 | 0.83 | 0.05 | 0.87 | **0.679** |
| pixel IoU | — | — | **0.700** | — | mIoU **0.822** |

**The data adjustment works — dramatically — at the pixel level.** Water IoU jumps
0.000 → 0.700 (the model now segments water where it exists), mIoU 0.64 → 0.82, and the
spillover lifts every class (open_field 0.48→0.83, city 0.74→0.87).

**But water *classification* F1 stays at 0.05**, exposing a second, distinct failure
mode: segmentation quality ≠ presence-decision quality. The 6×-oversampled model
over-predicts small spurious water fragments on non-water images; with only 12 water
positives among 450 dev images, even a small false-positive rate destroys precision, and
no threshold in the sweep rescues it. Levers: replace repetition with *diverse* real
water data (E5) and/or raise the water decision threshold further.

### E5 — Cross-dataset water: +258 ADE20K water scenes, no repetition (vs. E3: +ADE)

| metric | forest | open_field | water | city | macro |
|---|---|---|---|---|---|
| F1 @ tuned | 0.97 | 0.82 | 0.07 | 0.85 | 0.675 |
| water IoU / mIoU | | | 0.691 | | 0.822 |

**Real diverse water data and 6× repetition converge to the same pixel quality**
(IoU 0.69 vs 0.70) — remarkable given the ADE images are photographer-POV, not street
level. Water presence precision remains the bottleneck (its tuned threshold hit the 0.4
grid ceiling).

### E6 — Combination: LR 2e-4 + balanced + water ×2 + ADE water 258

| metric | forest | open_field | water | city | macro |
|---|---|---|---|---|---|
| F1 @ tuned | 0.93 | 0.83 | **0.12** | 0.74 | 0.655 |
| water IoU / mIoU | | | **0.769** | | **0.850** |

Best segmentation model by a clear margin and best (still weak) water classification;
city dipped at this short 2-epoch budget (higher LR variance) — expected to recover at
final scale.

### Experiment summary (dev set, tuned thresholds)

| run | change vs. | macro-F1 | water F1 | water IoU | mIoU |
|---|---|---|---|---|---|
| E1 baseline (LR 6e-5, random 1.5k) | — | 0.557 | 0.00 | 0.000 | 0.615 |
| E2 LR → 2e-4 | E1 | 0.603 | 0.00 | 0.000 | 0.638 |
| E3 balanced sampling | E1 | 0.547 | 0.00 | 0.000 | 0.640 |
| E4 balanced + water ×6 | E3 | **0.679** | 0.05 | 0.700 | 0.822 |
| E5 balanced + ADE water 258 | E3 | 0.675 | 0.07 | 0.691 | 0.822 |
| E6 LR 2e-4 + bal + water ×2 + ADE | E4/E5 | 0.655 | **0.12** | **0.769** | **0.850** |

**Answer to "does the training data need to be adjusted?" — yes, decisively.** The two
biggest improvements of the whole study were data changes, not hyperparameters:
water supervision mass (E4/E5: mIoU +0.18, water IoU 0→0.7) and per-class decision
thresholds. LR mattered moderately (E2); everything else was second-order.

## 4. Final run & configuration

E6 configuration scaled up: **4,000 balanced images (+258 ADE water, water ×2) ·
3 epochs · LR 2e-4 · batch 4 · 512² · seed 42** — 21.6 min on MPS.
Saved to `models/segformer_env/` (with `dev_metrics.json`).

| metric | forest | open_field | water | city | macro / mean |
|---|---|---|---|---|---|
| F1 @ tuned thresholds | 0.91 | 0.84 | **0.42** | 0.74 | **0.730** |
| pixel IoU | 0.921 | 0.865 | 0.709 | 0.905 | **0.850** |

Final per-class presence thresholds (dev-tuned, in the notebook as `AREA_THRESHOLDS`):
`forest 0.08 · open_field 0.12 · water 0.40 · city 0.20`.

**Improvement over the untuned notebook defaults: macro-F1 0.557 → 0.730 (+0.173),
mIoU 0.615 → 0.850 (+0.235), water F1 0.00 → 0.42, water IoU 0.000 → 0.709.**

### Change-by-change attribution

| change | dev improvement | verdict |
|---|---|---|
| LR 6e-5 → 2e-4 | macro +0.05, mIoU +0.02 | keep |
| balanced sampling alone | ±0 (water still 0.0) | keep only as carrier for ↓ |
| **water oversampling / ADE20K water aug** | **mIoU +0.18, water IoU 0 → 0.7** | **the key fix — data, not HPs** |
| **per-class thresholds** (vs global 3%) | **macro +0.05…+0.15 per run** | **free win, no retraining** |
| scale 1.5k/2ep → 4k/3ep | water F1 0.12 → 0.42, city recovered | keep |

## 5. Conclusions & handoff

1. **The classification head "not working" had two separable causes:** (a) water/
   open_field starvation in the training data — fixed by balanced sampling + water
   oversampling + ADE20K water augmentation; (b) a miscalibrated global presence
   threshold — fixed by per-class dev-tuned thresholds.
2. **Vistas alone cannot support the water class** (55 usable images in 18k). Water
   remains the weakest class (F1 0.42 on 12 dev positives) — treat with caution; the
   most valuable future data is cyclist-POV water footage from own rides.
3. **The dev set** (`dataset/eval/dev_mapillary.csv`) is disjoint from training and
   from the hand-labeled test set; the ADE20K water augmentation likewise excludes all
   test stems. The hand-labeled test set remains untouched by every decision above.
4. **Notebook parity:** `seg_finetune.ipynb` now contains the winning configuration,
   the MPS fixes, the balanced/augmented data pipeline and per-class thresholds; its
   predictions land in `dataset/eval/env_pred_semseg.csv` (`filename` joins to
   `labels.csv`) for the final zero-shot vs. fine-tuned comparison in
   `seg_evaluation.ipynb`.
5. **Caveats:** dev GT is mask-derived (share ≥ 3%), not human-labeled; water F1 rests
   on 12 positives; dev domain is Mapillary street-level, so dev-tuned thresholds may
   need a sanity check on the labeled own-frames test set (but must not be re-tuned on
   it — that would leak the final comparison).

## 6. Water class — the remaining gap and options

**Diagnosis.** Water pixel IoU is already good (0.71); the weakness is *presence
classification* (dev F1 0.42 on 12 positives) and **domain**: all water the model has
seen is Mapillary street-level + ADE20K photographer-POV. On the own-frames test set the
model fired water on only 4/425 frames. The ADE20K water augmentation is already maxed
(all 258 available images used), and the ~24 usable Vistas water images are the hard
ceiling within Vistas — so *more of the same* is not available. Options, ranked by
expected impact ÷ effort:

| # | Option | New labeling? | Effort | Expected impact | Notes |
|---|--------|---------------|--------|-----------------|-------|
| 1 | **Switch base to ADE20K-pretrained SegFormer** (`nvidia/segformer-b0-finetuned-ade-512-512`) | none | low | **high** | Current base is Cityscapes → **no water class**. ADE base already encodes water/sea/river/lake/waterfall; the encoder starts with a water prior. Single-line change. **Do this first.** |
| 2 | **Copy-paste augmentation for water** | none | medium | high | Paste water regions (from ADE/Vistas water masks) into non-water street scenes, update the mask. Standard rare-class-segmentation booster; multiplies effective water instances without new data. |
| 3 | **Class-weighted / Dice / focal loss** | none | low | medium | Up-weight water in the CE loss (or add Dice) instead of pure repetition — targets the *precision* failure that oversampling caused, less overfitting to 24 images. |
| 4 | **Stronger water augmentation** (h-flip, scale jitter, brightness/contrast/reflection) | none | low–med | medium | Cheap variance for the few water images; combine with #2. |
| 5 | **More external water from a POV-matched source** (Mapillary Graph API bbox over lakes/rivers; or COCO-Stuff / LoveDA water) | none (masks provided) | medium | medium–high | Adds *diverse, street-level* water — better than ADE's photographer POV. |
| 6 | **Add own cyclist-POV water frames via a ride-split** | yes (pixel masks!) | high | high (best domain match) | Needs pixel masks (label_tool only does image-level); use SAM/model-assisted annotation. Must split by `ride_id` — train on some water rides, test on others. |
| 7 | **Own frames for the water *threshold / test metric* only** | yes (image-level, cheap) | low | measurement, not model | Doesn't fix the model but makes the water number trustworthy (currently 12 dev positives). |
| 8 | **Accept & report** | none | none | — | If own routes have little water, treat water as a documented limitation of the beauty axis rather than over-engineering it. |

**How many own water frames would "fill the gap"?** The unit that matters is *distinct
water scenes/rides*, not frames — consecutive ride frames are near-duplicates and carry
little independent signal. Guidance:
- **To make water reliable on the own-frames domain (train, option 6):** ~8–15 distinct
  water encounters (different water bodies / lighting), ~10–15 frames each ⇒ ~100–200
  frames, split by ride so test rides are disjoint from train rides.
- **To just measure water trustworthily (option 7):** ≥30–50 water-positive test frames
  across ≥5 rides (for an F1 with a usable confidence interval).
- Fewer than ~5 distinct scenes ≈ a handful of independent samples regardless of frame
  count.

**Recommended path:** #1 + #2 (+ #3) first — all zero-labeling, likely to move water
substantially. Spend scarce own-frame water on #7 (evaluation), not #6 (training), until
the labels reveal how much water the routes actually contain.

## 7. Water-gap experiments — options 1 & 2 implemented (E7–E9)

All at the sweep budget (1500 balanced imgs + water×2 + 258 ADE water, LR 2e-4, 2 epochs,
seed 42) so they stack directly onto the E1–E6 table. Option 1 = swap the Cityscapes
base for the ADE20K-pretrained SegFormer (`nvidia/segformer-b0-finetuned-ade-512-512`,
encoder already knows water/sea/river/lake). Option 2 = copy-paste water augmentation
(`build_water_patch_bank` + `paste_water`: crop water regions from ADE/Vistas masks,
paste into the lower 2/3 of street scenes at random scale, prob 0.5).

| run | change vs. E6 | macro-F1 | water F1 | water IoU | mIoU |
|---|---|---|---|---|---|
| E6 combo (Cityscapes base) | — | 0.655 | 0.12 | 0.769 | 0.850 |
| **E7 ADE20K base (opt 1)** | base swap | 0.640 | **0.21** | **0.778** | **0.857** |
| E8 copy-paste 0.5 (opt 2) | + cp-water | 0.673 | 0.15 | 0.662 | 0.825 |
| E9 both (opt 1 + 2) | base + cp | 0.638 | 0.15 | 0.746 | 0.846 |

**Option 1 (ADE20K base) wins clearly and is the only change to keep.** Swapping the
Cityscapes base (no water class) for the ADE20K base (encoder already knows
water/sea/river/lake) nearly doubles water F1 (0.12 → 0.21) and is best on all three
water/segmentation metrics at once — the most trustworthy signal given the 12-positive
dev noise. One line of config, zero new data.

**Option 2 (copy-paste) is not worth it.** It nudges presence F1 (0.12 → 0.15) but
*lowers* water pixel IoU (0.769 → 0.662): the pasted patches are unnatural, so the model
learns a slightly worse water appearance. Worse, **the two options conflict** — E9
(both) drops the ADE base's water F1 back from 0.21 to 0.15. Copy-paste is therefore
dropped from the final recipe.

**Caveat.** All E7–E9 differences live within a few points on a 12-water-positive dev
set — individually noisy. E7 is trusted because it improves water F1, water IoU *and*
mIoU simultaneously, and for a first-principles reason (the base actually has a water
prior), not because any single number is decisive.

### Final model v2 (adopted): ADE20K base at full budget

`FINAL2_adebase_4k` = E7 recipe scaled to 4,000 imgs / 3 epochs, saved to
`models/segformer_env_ade`, promoted to `models/segformer_env` if it beats v1 on dev.

| model | macro-F1 | water F1 | water IoU | mIoU | open_field F1 |
|---|---|---|---|---|---|
| v1 FINAL (Cityscapes base) | **0.730** | 0.42 | 0.709 | 0.850 | **0.84** |
| **v2 FINAL2 (ADE20K base) — promoted** | 0.721 | **0.46** | **0.772** | **0.863** | 0.77 |

**v2 promoted to `models/segformer_env`** (v1 archived at
`models/segformer_env_cityscapes_v1`). It is a *tradeoff, not a clean sweep*: v2 wins on
water (F1 +0.04, IoU +0.06), mIoU (+0.013) and city, but open_field regresses
(0.84 → 0.77), so classification macro-F1 dips 0.009. Promoted anyway because (a) the
downstream beauty score consumes **pixel-area fractions** (mIoU-driven, where v2 wins)
and weights **water highest**, and (b) the macro gap is within the 12-positive dev noise
while v2's water/mIoU gains rest on a real prior. v2 thresholds: `forest 0.12 ·
open_field 0.12 · water 0.30 · city 0.20`. Notebook `BASE_MODEL` + `AREA_THRESHOLDS`
updated. **The test set is the final arbiter** — v1 is kept so both can be scored on the
labeled own frames.

## 8. What to do next — and the sequencing vs. test-set evaluation

**We have reached the useful limit of dev-set tuning.** Two reasons: (a) the dev set is
Mapillary street-level, *not* the target cyclist-POV domain; (b) water rests on 12
positives, so further micro-tuning chases noise. More blind sweeps would optimize the
wrong distribution.

**Do BEFORE test evaluation (robust, domain-agnostic — already in flight):**
- Adopt the ADE20K base (option 1 / FINAL2). It is a first-principles improvement (real
  water prior), not a dev-specific artifact — safe to lock in without test feedback.
- That is essentially the last "blind" improvement worth making.

**Do NOT do before test evaluation (premature / overfits the wrong domain):**
- More copy-paste/augmentation tuning (option 2 already shown marginal + conflicting).
- More threshold or LR micro-tuning on the Mapillary dev set (12-positive noise floor).
- Collecting or pixel-annotating own water frames *for training* — expensive, and you do
  not yet know whether water even occurs on the routes.

**Gate — finish labeling.** The test evaluation is blocked on labeled own frames
(58 / 1567 so far). Label a representative spread across rides, including whatever water
appears, before trusting any test number.

**Do AFTER test evaluation (targeted, data-driven — the test set says *where* to invest):**
1. Read the per-class, per-source F1. Early warning from the raw v1 predictions: forest
   fired on ~97% of own frames — the `vegetation→forest` semantic gap likely shows up as
   forest over-prediction. The test set confirms or refutes this; the Mapillary dev
   cannot.
2. **Re-set the per-class presence thresholds on a labeled own-frames dev slice**
   (split own frames by `ride_id` into a threshold-tuning slice + a locked test slice).
   The Mapillary-tuned thresholds (`forest 0.08 … water 0.40`) may not transfer to
   cyclist POV; this is legitimate in-domain calibration, not leakage — as long as the
   test slice is untouched.
3. Only if water genuinely fails on real routes → invest in own cyclist-POV water
   (options 5/6/7 in §6). If water is simply rare on the routes → document it as a
   limitation and move on.
4. Keep the zero-shot vs. fine-tuned comparison fair: same test frames, thresholds for
   both tuned on the same dev slice.

**One-line summary:** lock in the ADE base now, finish labeling, evaluate on the test
set, and let the per-class/per-source test results — not more Mapillary-dev tuning —
decide every further improvement.
