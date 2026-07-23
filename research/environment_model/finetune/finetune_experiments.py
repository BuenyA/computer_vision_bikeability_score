#!/usr/bin/env python3
"""Hyperparameter experiments for the SegFormer environment fine-tune.

Mirrors research/seg_finetune.ipynb (same LUT, same dataset code, same
area-threshold classifier) but runnable/parameterizable from the CLI, so
hyperparameter changes can be measured one at a time on a fixed dev set.

Subcommands:
  devset   Build the hyperparameter-tuning dev set: Mapillary VALIDATION
           images (minus the 350 held out into the hand-labeled test set),
           with per-class GT pixel shares + multi-hot GT derived from the
           GT masks (class present if GT share >= 0.03). The human-labeled
           test set (dataset/test_images/labels.csv) is NEVER touched here.
  shares   Cache per-image class pixel shares for the TRAINING split
           (needed for class-balanced training sampling).
  train    Run one fine-tune experiment + dev evaluation, append results
           to research/seg_finetune_tuning_results.csv.

Example:
  python scripts/finetune_experiments.py devset
  python scripts/finetune_experiments.py train --name baseline \
      --train-limit 1500 --epochs 2 --lr 6e-5
"""

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "environment_model"))
import segmentation_common as sc  # noqa: E402

BASE_MODEL = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
MAPILLARY_ROOT = REPO / "dataset" / "external" / "mapillary_vistas"
CANDIDATES_CSV = REPO / "dataset" / "test_images" / "candidates.csv"
DEV_CSV = REPO / "dataset" / "eval" / "dev_mapillary.csv"
SHARES_CSV = REPO / "dataset" / "eval" / "mapillary_train_shares.csv"
RESULTS_CSV = REPO / "environment_model" / "finetune" / "seg_finetune_tuning_results.csv"

ENV_CLASSES = sc.CATEGORIES["environment"]
ENV_ID = {c: i for i, c in enumerate(ENV_CLASSES)}
GT_PRESENCE = 0.03  # GT multi-hot rule: class present if GT share >= 3%


# ---------------------------------------------------------------- shared io
OTHER_ID = len(ENV_CLASSES)   # explicit background class id (== 4) when --other


def _with_other(lut, other):
    """Map background (unmapped -> VOID) to an explicit 'other' class instead of
    ignoring it, so the model can predict 'none of the 4' at inference."""
    return np.where(lut == sc.VOID_ID, OTHER_ID, lut) if other else lut


def build_lut(other=False):
    cfg = json.load(open(MAPILLARY_ROOT / "config_v2.0.json"))
    names = [l["name"] for l in cfg["labels"]]
    lut = sc.build_id_lookup(names, sc.MAPILLARY_TO_TAXONOMY, class_id=ENV_ID)
    return _with_other(lut, other)


def label_dir(split):
    for d in (MAPILLARY_ROOT / split / "v2.0" / "labels",
              MAPILLARY_ROOT / split / "labels"):
        if d.exists():
            return d
    raise FileNotFoundError(split)


def load_mask(p):
    arr = np.array(Image.open(p))
    return arr[..., 0] if arr.ndim == 3 else arr


def test_set_stems():
    stems = set()
    for row in csv.DictReader(open(CANDIDATES_CSV)):
        if row.get("source") == "mapillary":
            stems.add(Path(row["orig_path"]).stem)
    return stems


def env_shares(mask, lut):
    """Per-env-class pixel share over ALL pixels (matches the classifier)."""
    env = lut[mask]
    n = env.size
    counts = np.bincount(env[env != sc.VOID_ID], minlength=len(ENV_CLASSES))
    return {c: counts[i] / n for c, i in ENV_ID.items()}


def scan_split(split, lut, exclude=frozenset(), limit=None):
    ld = label_dir(split)
    masks = sorted(ld.glob("*.png"))
    rows = []
    for i, mp in enumerate(masks):
        if mp.stem in exclude:
            continue
        if limit and len(rows) >= limit:
            break
        if i and i % 500 == 0:
            print(f"  {split}: {i}/{len(masks)} masks scanned", flush=True)
        sh = env_shares(load_mask(mp), lut)
        row = {"stem": mp.stem}
        row.update({f"share_{c}": round(sh[c], 5) for c in ENV_CLASSES})
        rows.append(row)
    return rows


# ---------------------------------------------------------------- devset
def cmd_devset(args):
    lut = build_lut()
    held = test_set_stems()
    print(f"building dev set from validation minus {len(held)} test stems ...")
    rows = scan_split("validation", lut, exclude=held)
    for r in rows:
        r["split"] = "validation"

    # The test-set extraction already claimed every watery validation image
    # (water rule had first pick at share>=0.01), leaving validation with ZERO
    # water positives. Reserve water/terrain-rich TRAINING images into the dev
    # set instead; pick_train_stems() excludes them from training again.
    if SHARES_CSV.exists():
        tr = list(csv.DictReader(open(SHARES_CSV)))
        rng = random.Random(7)
        # Vistas water is EXTREMELY scarce (55 training images with >=1% water
        # after the test hold-out). Take alternating ranks so dev can measure
        # water while training keeps an equally strong half to learn from.
        water_all = sorted((r for r in tr if float(r["share_water"]) >= 0.01),
                           key=lambda r: -float(r["share_water"]))
        water = water_all[0::2][:75]
        used = {r["stem"] for r in water}
        veg = [r for r in tr if float(r["share_vegetation"]) >= 0.15
               and r["stem"] not in used]
        rng.shuffle(veg)
        veg = veg[:100]
        for r in water + veg:
            row = {"stem": r["stem"], "split": "training"}
            row.update({f"share_{c}": r[f"share_{c}"] for c in ENV_CLASSES})
            rows.append(row)
        print(f"reserved from training: {len(water)} water-rich, "
              f"{len(veg)} vegetation-rich (excluded from training)")
    else:
        print("WARNING: no share cache - dev set has no water positives; "
              "run 'shares' then rebuild devset")

    for r in rows:
        for c in ENV_CLASSES:
            r[f"gt_{c}"] = int(float(r[f"share_{c}"]) >= GT_PRESENCE)
    DEV_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = ["stem", "split"] + [f"share_{c}" for c in ENV_CLASSES] + \
             [f"gt_{c}" for c in ENV_CLASSES]
    with open(DEV_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    pos = {c: sum(r[f"gt_{c}"] for r in rows) for c in ENV_CLASSES}
    print(f"wrote {len(rows)} dev rows -> {DEV_CSV}")
    print("GT positives per class:", pos)


def cmd_shares(args):
    lut = build_lut()
    held = test_set_stems()
    rows = scan_split("training", lut, exclude=held, limit=args.limit)
    with open(SHARES_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} training share rows -> {SHARES_CSV}")


# ---------------------------------------------------------------- training
def build_items(stems, split):
    img_dir = MAPILLARY_ROOT / split / "images"
    ld = label_dir(split)
    return [(img_dir / f"{s}.jpg", ld / f"{s}.png") for s in stems]


def dev_reserved_stems():
    """Training-split stems reserved into the dev set - never train on them."""
    if not DEV_CSV.exists():
        return set()
    return {r["stem"] for r in csv.DictReader(open(DEV_CSV))
            if r.get("split") == "training"}


def balanced_selection(limit, seed, oversample_water=1):
    """Deterministic class-balanced pick via the share cache.

    Returns (buckets, extra_fill): buckets maps class -> stems (water possibly
    repeated by `oversample_water`); extra_fill are the random top-up stems
    used when the class quotas undershoot `limit`. Shared by training and by
    scripts/organize_training_images.py so both see the identical selection.
    """
    held = test_set_stems() | dev_reserved_stems()
    all_stems = sorted(p.stem for p in
                       (MAPILLARY_ROOT / "training" / "images").glob("*.jpg")
                       if p.stem not in held)
    if not SHARES_CSV.exists():
        sys.exit("balanced sampling needs the share cache - run 'shares' first")
    rows = [r for r in csv.DictReader(open(SHARES_CSV))
            if r["stem"] not in held]
    # scarce classes first, one bucket per image, quota = limit/4 each
    rules = [("water", 0.01), ("vegetation", 0.20), ("city", 0.25)]
    quota = (limit or len(rows)) // len(rules)
    buckets, used = defaultdict(list), set()
    random.Random(seed).shuffle(rows)
    for r in rows:
        for cls, thr in rules:
            if float(r[f"share_{cls}"]) >= thr and r["stem"] not in used:
                if len(buckets[cls]) < quota:
                    buckets[cls].append(r["stem"])
                    used.add(r["stem"])
                break  # first matching rule claims the image, full bucket or not
    if oversample_water > 1:  # water is so scarce that repetition is the only lever
        buckets["water"] = buckets["water"] * oversample_water
    n_picked = sum(len(v) for v in buckets.values())
    extra = [s for s in all_stems if s not in used][:max(0, (limit or 0) - n_picked)]
    return buckets, extra


def pick_train_stems(limit, balanced, seed, oversample_water=1):
    """Random subset, or class-balanced via the cached share index."""
    held = test_set_stems() | dev_reserved_stems()
    all_stems = sorted(p.stem for p in
                       (MAPILLARY_ROOT / "training" / "images").glob("*.jpg")
                       if p.stem not in held)
    rng = random.Random(seed)
    if not balanced:
        rng.shuffle(all_stems)
        return all_stems[:limit] if limit else all_stems
    buckets, extra = balanced_selection(limit, seed, oversample_water)
    print("balanced buckets:", {k: len(v) for k, v in buckets.items()})
    picked = [s for b in buckets.values() for s in b] + extra
    rng.shuffle(picked)
    return picked


ADE_ROOT = REPO / "dataset" / "external" / "ADEChallengeData2016"
# water-dominated ADE20K scene categories (same set as build_test_dataset.py)
ADE_WATER_SCENES = {
    "coast", "beach", "river", "creek", "harbor", "marsh", "lagoon", "water",
    "waterway", "shore", "tidal_river", "bayou", "swamp", "pond", "millpond",
    "fishpond", "waterscape", "foreshore", "dock", "pier", "floating_dock",
}


def ade20k_lut(other=False):
    from huggingface_hub import hf_hub_download
    id2label = json.load(open(hf_hub_download(
        "huggingface/label-files", "ade20k-id2label.json", repo_type="dataset")))
    n = max(int(k) for k in id2label) + 1
    names = ["other"] * (n + 1)  # ADE masks are 1-indexed; 0 == other
    for k, v in id2label.items():
        names[int(k) + 1] = v
    lut = sc.build_id_lookup(names, sc.ADE20K_TO_TAXONOMY, class_id=ENV_ID)
    return _with_other(lut, other)


def ade_water_items(cap, seed):
    """ADE20K water-scene images (training split, test-set stems excluded).

    Cross-dataset augmentation for the water class: Vistas has almost no water
    (24 usable training images), ADE20K has hundreds of masked water scenes.
    Photographer POV differs from street level, but pixel supervision of water
    appearance transfers.
    """
    test_stems = {Path(r["orig_path"]).stem
                  for r in csv.DictReader(open(CANDIDATES_CSV))
                  if r.get("source") == "ade20k"}
    stems = []
    for line in open(ADE_ROOT / "sceneCategories.txt"):
        parts = line.split()
        if len(parts) == 2 and parts[1] in ADE_WATER_SCENES \
                and "_train_" in parts[0] and parts[0] not in test_stems:
            stems.append(parts[0])
    random.Random(seed).shuffle(stems)
    stems = stems[:cap]
    img_dir = ADE_ROOT / "images" / "training"
    ann_dir = ADE_ROOT / "annotations" / "training"
    return [(img_dir / f"{s}.jpg", ann_dir / f"{s}.png", "ade") for s in stems]


def build_water_patch_bank(luts, cap=300, seed=42, max_side=384, min_px=800):
    """Extract cropped water regions (image patch + boolean mask) for copy-paste.

    Sources: all ADE20K water scenes + Vistas water images. Each patch is the
    water bounding-box crop, downscaled to <= max_side. Loaded once at startup.
    """
    import cv2
    water_id = ENV_ID["water"]
    pool = []
    for img_p, mask_p, _s in ade_water_items(cap, seed):
        pool.append((img_p, mask_p, luts["ade"]))
    held = test_set_stems() | dev_reserved_stems()
    if SHARES_CSV.exists():
        wet = sorted((r for r in csv.DictReader(open(SHARES_CSV))
                      if r["stem"] not in held and float(r["share_water"]) >= 0.01),
                     key=lambda r: -float(r["share_water"]))
        vdir = MAPILLARY_ROOT / "training" / "images"
        ldir = label_dir("training")
        for r in wet[:80]:
            pool.append((vdir / f"{r['stem']}.jpg", ldir / f"{r['stem']}.png",
                         luts["map"]))
    patches = []
    for img_p, mask_p, lut in pool:
        env = lut[load_mask(mask_p)]
        ys, xs = np.where(env == water_id)
        if len(ys) < min_px:
            continue
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        img = np.array(Image.open(img_p).convert("RGB"))
        pim = img[y0:y1 + 1, x0:x1 + 1]
        pm = (env[y0:y1 + 1, x0:x1 + 1] == water_id)
        h, w = pm.shape
        if max(h, w) > max_side:
            s = max_side / max(h, w)
            pim = cv2.resize(pim, (max(1, int(w * s)), max(1, int(h * s))))
            pm = cv2.resize(pm.astype(np.uint8), pim.shape[1::-1],
                            interpolation=cv2.INTER_NEAREST).astype(bool)
        if pm.sum() >= min_px // 4:
            patches.append((pim, pm))
    print(f"  copy-paste water bank: {len(patches)} patches from {len(pool)} sources")
    return patches


def paste_water(img, mask, patch, rng):
    """Paste a water patch into the lower 2/3 of img (in place), set mask=water."""
    import cv2
    water_id = ENV_ID["water"]
    pim, pm = patch
    H, W = img.shape[:2]
    scale = rng.uniform(0.25, 0.6) * W / pm.shape[1]
    nh, nw = max(8, int(pm.shape[0] * scale)), max(8, int(pm.shape[1] * scale))
    pim = cv2.resize(pim, (nw, nh))
    pm = cv2.resize(pm.astype(np.uint8), (nw, nh),
                    interpolation=cv2.INTER_NEAREST).astype(bool)
    y = rng.randint(H // 3, H - 1)
    x = rng.randint(0, W - 1)
    nh, nw = min(nh, H - y), min(nw, W - x)
    if nh < 4 or nw < 4:
        return img, mask
    reg = pm[:nh, :nw]
    img[y:y + nh, x:x + nw][reg] = pim[:nh, :nw][reg]
    mask[y:y + nh, x:x + nw][reg] = water_id
    return img, mask


def cmd_train(args):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset
    from transformers import (SegformerForSemanticSegmentation,
                              SegformerImageProcessor, Trainer,
                              TrainingArguments)

    base_model = args.base_model or BASE_MODEL
    n_out = len(ENV_CLASSES) + (1 if args.other else 0)  # +1 for explicit background

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    lut = build_lut(args.other)
    # ADE LUT needed for ADE water aug and/or the copy-paste bank
    need_ade = bool(args.ade_water or args.copy_paste_water)
    luts = {"map": lut, "ade": ade20k_lut(args.other) if need_ade else None}
    processor = SegformerImageProcessor.from_pretrained(
        base_model, do_reduce_labels=False,
        size={"height": args.size, "width": args.size})

    water_bank = (build_water_patch_bank(luts, seed=args.seed)
                  if args.copy_paste_water else [])

    class SegDataset(Dataset):
        def __init__(self, items, cp_prob=0.0):
            self.items = items  # (img_path, mask_path, source) triples
            self.cp_prob = cp_prob

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            img_p, mask_p, src = self.items[i]
            img = np.array(Image.open(img_p).convert("RGB"))
            mask = luts[src][load_mask(mask_p)].astype(np.uint8)
            if water_bank and self.cp_prob:
                # deterministic per-item rng -> reproducible epochs
                rng = random.Random(hash((str(img_p), i)) & 0xffffffff)
                if rng.random() < self.cp_prob:
                    img, mask = paste_water(img.copy(), mask.copy(),
                                            rng.choice(water_bank), rng)
            enc = processor(img, mask, return_tensors="pt")
            return {"pixel_values": enc["pixel_values"][0],
                    "labels": enc["labels"][0]}

    stems = pick_train_stems(args.train_limit, args.balanced, args.seed,
                             args.oversample_water)
    items = [(i, m, "map") for i, m in build_items(stems, "training")]
    if args.ade_water:
        extra = ade_water_items(args.ade_water, args.seed)
        items += extra
        print(f"  + {len(extra)} ADE20K water-scene images (cross-dataset aug)")
    random.Random(args.seed).shuffle(items)
    train_ds = SegDataset(items, cp_prob=args.copy_paste_water)
    print(f"[{args.name}] training on {len(train_ds)} images "
          f"(base={base_model.split('/')[-1]}, balanced={args.balanced}, "
          f"cp_water={args.copy_paste_water}, device={device})")

    class MPSSafeBatchNorm2d(torch.nn.BatchNorm2d):
        """Decomposed BN forward on MPS (train mode only).

        torch 2.12 MPS: native_batch_norm's backward raises a view/stride
        RuntimeError. Decomposing into mean/var primitives sidesteps it;
        numerics and running-stat updates match nn.BatchNorm2d exactly.
        """

        def forward(self, x):
            if not self.training or x.device.type != "mps":
                return super().forward(x)
            mean = x.mean(dim=(0, 2, 3))
            var = x.var(dim=(0, 2, 3), unbiased=False)
            if self.track_running_stats:
                with torch.no_grad():
                    n = x.numel() / x.shape[1]
                    self.running_mean.lerp_(mean, self.momentum)
                    self.running_var.lerp_(var * n / (n - 1), self.momentum)
                    self.num_batches_tracked += 1
            xhat = (x - mean[None, :, None, None]) / torch.sqrt(
                var[None, :, None, None] + self.eps)
            return (xhat * self.weight[None, :, None, None]
                    + self.bias[None, :, None, None])

    id2label = {i: c for c, i in ENV_ID.items()}
    label2id = dict(ENV_ID)
    if args.other:
        id2label[OTHER_ID] = "other"; label2id["other"] = OTHER_ID
    model = SegformerForSemanticSegmentation.from_pretrained(
        base_model, num_labels=n_out,
        id2label=id2label, label2id=label2id,
        ignore_mismatched_sizes=True)
    bn = model.decode_head.batch_norm
    safe = MPSSafeBatchNorm2d(bn.num_features, eps=bn.eps, momentum=bn.momentum)
    safe.load_state_dict(bn.state_dict())
    model.decode_head.batch_norm = safe
    model = model.to(device)

    class SegTrainer(Trainer):
        """Loss at logits resolution (labels nearest-downsampled 4x).

        Avoids the in-model 4x logits upsample whose bilinear backward hits a
        view/stride RuntimeError on MPS; also ~faster. Supervision semantics
        are unchanged, just coarser (SegFormer logits are stride-4 anyway).
        """

        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            labels = inputs["labels"]
            outputs = model(pixel_values=inputs["pixel_values"])
            logits = outputs.logits
            small = F.interpolate(labels[:, None].float(),
                                  size=logits.shape[-2:],
                                  mode="nearest")[:, 0].long()
            loss = F.cross_entropy(logits, small, ignore_index=sc.VOID_ID)
            return (loss, outputs) if return_outputs else loss

    targs = TrainingArguments(
        output_dir=str(REPO / "models" / "_tmp_experiments" / args.name),
        learning_rate=args.lr, num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps else -1,
        per_device_train_batch_size=args.batch_size,
        eval_strategy="no", save_strategy="no", logging_steps=25,
        remove_unused_columns=False, seed=args.seed, report_to=[],
        dataloader_num_workers=0,
    )
    t0 = time.perf_counter()
    trainer = SegTrainer(model=model, args=targs, train_dataset=train_ds)
    trainer.train()
    train_min = (time.perf_counter() - t0) / 60

    # ---- dev evaluation (classification via area threshold + pixel mIoU) ----
    dev_rows = list(csv.DictReader(open(DEV_CSV)))
    # stratified, fixed across experiments: all training-reserved rows (the
    # only water/field positives) + random validation rows up to dev_n
    reserved = [r for r in dev_rows if r.get("split") == "training"]
    val = [r for r in dev_rows if r.get("split") != "training"]
    random.Random(123).shuffle(val)
    dev_rows = (reserved + val)[:max(args.dev_n, len(reserved))]

    model.eval()
    pred_shares, gt_hot, cm = [], [], np.zeros((n_out, n_out), np.int64)
    t0 = time.perf_counter()
    with torch.no_grad():
        for r in dev_rows:
            split = r.get("split", "validation")
            img_dir = MAPILLARY_ROOT / split / "images"
            ld = label_dir(split)
            img = np.array(Image.open(img_dir / f"{r['stem']}.jpg").convert("RGB"))
            enc = processor(img, return_tensors="pt").to(device)
            logits = model(**enc).logits  # (1,4,h/4,w/4)
            pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
            n = pred.size
            pred_shares.append([float((pred == i).sum()) / n
                                for i in range(len(ENV_CLASSES))])
            gt_hot.append([int(r[f"gt_{c}"]) for c in ENV_CLASSES])
            # pixel confusion at logits res (GT nearest-downsampled)
            gt_env = lut[load_mask(ld / f"{r['stem']}.png")].astype(np.uint8)
            gt_small = np.array(Image.fromarray(gt_env).resize(
                pred.shape[::-1], Image.NEAREST))
            valid = gt_small != sc.VOID_ID
            idx = gt_small[valid].astype(np.int64) * n_out + pred[valid]
            cm += np.bincount(idx, minlength=n_out ** 2).reshape(n_out, n_out)
    eval_ms = (time.perf_counter() - t0) / len(dev_rows) * 1000

    pred_shares = np.array(pred_shares)
    gt_hot = np.array(gt_hot)
    # env-class IoU from the full (n_out) confusion matrix (FN to 'other' counted)
    iou = (sc.per_class_iou(cm.astype(np.int64))[:len(ENV_CLASSES)]
           if cm.sum() else None)

    def f1_at(thr_vec):
        pred_hot = pred_shares >= np.asarray(thr_vec)
        return sc.multilabel_metrics(gt_hot, pred_hot)

    base = f1_at([args.area_threshold] * len(ENV_CLASSES))
    # per-class threshold sweep on dev (decision rule tuning)
    grid = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3, 0.4]
    best_thr = []
    for i in range(len(ENV_CLASSES)):
        scores = []
        for t in grid:
            pred_i = pred_shares[:, i] >= t
            tp = (pred_i & (gt_hot[:, i] == 1)).sum()
            fp = (pred_i & (gt_hot[:, i] == 0)).sum()
            fn = (~pred_i & (gt_hot[:, i] == 1)).sum()
            f1 = 2 * tp / max(2 * tp + fp + fn, 1)
            scores.append((f1, t))
        best_thr.append(max(scores)[1])
    tuned = f1_at(best_thr)

    row = {
        "name": args.name, "base": base_model.split("/")[-1],
        "other": int(args.other), "cp_water": args.copy_paste_water,
        "train_limit": len(train_ds),
        "balanced": int(args.balanced), "epochs": args.epochs, "lr": args.lr,
        "batch_size": args.batch_size, "size": args.size, "seed": args.seed,
        "train_min": round(train_min, 1), "dev_n": len(dev_rows),
        "eval_ms_per_img": round(eval_ms, 1),
        "macro_f1@0.03": round(base["macro_f1"], 4),
        "macro_f1@tuned": round(tuned["macro_f1"], 4),
        "tuned_thresholds": json.dumps(dict(zip(ENV_CLASSES, best_thr))),
        "miou_dev": round(float(np.nanmean(iou)), 4) if iou is not None else "",
    }
    for i, c in enumerate(ENV_CLASSES):
        row[f"f1_{c}@0.03"] = round(float(base["per_class_f1"][i]), 4)
        row[f"f1_{c}@tuned"] = round(float(tuned["per_class_f1"][i]), 4)
        row[f"iou_{c}"] = round(float(iou[i]), 4) if iou is not None else ""

    RESULTS_CSV.parent.mkdir(exist_ok=True)
    # schema-robust append: union columns with any existing rows and rewrite,
    # so added columns (e.g. base, cp_water) don't misalign older experiments.
    prev = list(csv.DictReader(open(RESULTS_CSV))) if RESULTS_CSV.exists() else []
    fields = list(prev[0]) if prev else []
    for k in row:
        if k not in fields:
            fields.append(k)
    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in prev:
            w.writerow(r)
        w.writerow(row)
    print(json.dumps(row, indent=2))

    if args.save_dir:
        out = Path(args.save_dir)
        trainer.save_model(str(out))
        processor.save_pretrained(str(out))
        (out / "dev_metrics.json").write_text(json.dumps(row, indent=2))
        print("saved model ->", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("devset")
    ps = sub.add_parser("shares")
    ps.add_argument("--limit", type=int, default=None)
    pt = sub.add_parser("train")
    pt.add_argument("--name", required=True)
    pt.add_argument("--train-limit", type=int, default=1500)
    pt.add_argument("--epochs", type=float, default=2)
    pt.add_argument("--lr", type=float, default=6e-5)
    pt.add_argument("--batch-size", type=int, default=4)
    pt.add_argument("--size", type=int, default=512)
    pt.add_argument("--max-steps", type=int, default=None)
    pt.add_argument("--seed", type=int, default=42)
    pt.add_argument("--balanced", action="store_true")
    pt.add_argument("--oversample-water", type=int, default=1)
    pt.add_argument("--ade-water", type=int, default=0,
                    help="add N ADE20K water-scene images (cross-dataset aug)")
    pt.add_argument("--base-model", type=str, default=None,
                    help="override base checkpoint, e.g. "
                         "nvidia/segformer-b0-finetuned-ade-512-512 (option 1)")
    pt.add_argument("--copy-paste-water", type=float, default=0.0,
                    help="prob. of pasting a water patch into each image (option 2)")
    pt.add_argument("--other", action="store_true",
                    help="add an explicit 'other' background class (5 outputs); "
                         "classification still uses the 4 env classes")
    pt.add_argument("--dev-n", type=int, default=450)
    pt.add_argument("--area-threshold", type=float, default=0.03)
    pt.add_argument("--save-dir", type=str, default=None)
    args = ap.parse_args()
    {"devset": cmd_devset, "shares": cmd_shares, "train": cmd_train}[args.cmd](args)


if __name__ == "__main__":
    main()
