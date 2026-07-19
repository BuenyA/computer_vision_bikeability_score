#!/usr/bin/env python3
"""Build the environment-classification test dataset.

Collects labeling candidates for the 3 environment classes
(vegetation, water, city) from three sources:

  1. ADE20K (ADEChallengeData2016, dataset/external/) - selected via scene
     categories (sceneCategories.txt). Validation split is preferred because
     seg_finetune.ipynb trains on the training split; training images are
     only used to fill a class quota and are marked in candidates.csv.
  2. Mapillary Vistas (optional, dataset/external/mapillary_vistas) -
     selected via pixel shares of vegetation/terrain/water/building in the
     semantic label masks. Validation (2k) is scanned first; since nothing
     in this pipeline trains on Vistas, any class still short of the quota
     falls back to the 18k training masks with no leakage risk (unlike
     ADE20K above). Skipped with download instructions if missing.
  3. Own ride frames (../frames, DJI_*_frame_*.jpg) - all copied, they are
     labeled as full sequences so the sticky labels in the hotkey tool pay off.

Output layout (default dataset/test_images/):
  ade20k/<class>/ADE_*.jpg      proposed-class buckets, <= --per-class each
  mapillary/<class>/*.jpg       only if Vistas is downloaded
  own_frames/DJI_*.jpg
  overview/index.html           browsable overview of everything to label
  overview/sheet_*.jpg          one contact sheet per bucket
  candidates.csv                filename,source,split,proposed_class,orig_path

The proposed class is only a pre-sorting to make labeling fast - ground truth
comes from the hotkey tool (scripts/label_tool.py), which is multi-label.

Usage:
  python scripts/build_test_dataset.py --dry-run     # show counts only
  python scripts/build_test_dataset.py               # build everything
  python scripts/build_test_dataset.py --overview-only
"""

import argparse
import csv
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

CLASSES = ["vegetation", "water", "city"]

CLASS_COLORS = {  # RGB, same as the segmentation palette
    "vegetation": (60, 160, 50),
    "water": (0, 130, 180),
    "city": (130, 130, 130),
}

# ADE20K scene categories -> environment class. Outdoor scenes only; indoor
# variants (e.g. warehouse_indoor) are deliberately left out because the test
# set mimics ride footage. "vegetation" merges the old forest + open_field.
ADE20K_SCENES = {
    "vegetation": {
        "broadleaf", "needleleaf", "forest_path", "forest_road",
        "bamboo_forest", "rainforest", "tree_farm",
        "pasture", "hayfield", "corn_field", "wheat_field", "farm",
        "field_road", "meadow", "vineyard", "orchard", "terrace_farm",
        "valley",
    },
    "water": {
        "coast", "beach", "river", "creek", "harbor", "marsh", "lagoon",
        "water", "waterway", "shore", "tidal_river", "bayou", "swamp",
        "pond", "millpond", "fishpond", "waterscape", "foreshore",
        "dock", "pier", "floating_dock",
    },
    "city": {
        "street", "alley", "crosswalk", "plaza", "downtown", "urban",
        "city", "residential_neighborhood", "building_facade",
        "skyscraper", "apartment_building_outdoor", "highway",
        "parking_lot", "toll_plaza", "one-way_street", "streetcar_track",
        "parkway", "office_building", "courtyard",
    },
}

# Mapillary Vistas: pixel-share rules on the semantic mask. Keys are matched
# as suffixes against the label names in config.json (works for v1.2 and
# v2.0, e.g. "nature--vegetation"). share = fraction of image pixels.
#
# Rules are evaluated in insertion order and the FIRST match wins, so the
# scarce class (water) is listed first. Thresholds are deliberately loose:
# Vistas is urban street-level footage where water is rare, and the human
# labeler is the ground truth and rejects false positives, so we optimise for
# candidate recall here. "vegetation" = Vistas vegetation OR terrain.
MAPILLARY_RULES = {
    "water": lambda s: s["water"] >= 0.01,
    "vegetation": lambda s: (s["vegetation"] >= 0.25 or s["terrain"] >= 0.10)
    and s["building"] <= 0.15,
    "city": lambda s: s["building"] >= 0.30,
}
MAPILLARY_LABEL_SUFFIXES = {
    "vegetation": "--vegetation",
    "terrain": "--terrain",
    "water": "--water",
    "building": "--building",
}


def load_ade20k_candidates(ade_root: Path):
    """Return {class: {"validation": [img_path,...], "training": [...]}}."""
    scene_to_class = {}
    for cls, scenes in ADE20K_SCENES.items():
        for s in scenes:
            scene_to_class[s] = cls
    out = {c: {"validation": [], "training": []} for c in CLASSES}
    with open(ade_root / "sceneCategories.txt") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 2:
                continue
            stem, scene = parts
            cls = scene_to_class.get(scene)
            if cls is None:
                continue
            split = "training" if "_train_" in stem else "validation"
            img = ade_root / "images" / split / f"{stem}.jpg"
            if img.exists():
                out[cls][split].append(img)
    return out


def pick_ade20k(candidates, per_class, seed, val_only):
    """Prefer validation images; fill from training only if needed."""
    picks = {}
    for cls in CLASSES:
        # per-class rng: adding/removing a class must not reshuffle the others
        rng = random.Random(f"{seed}:{cls}")
        val = sorted(candidates[cls]["validation"])
        train = sorted(candidates[cls]["training"])
        rng.shuffle(val)
        rng.shuffle(train)
        chosen = [(p, "validation") for p in val[:per_class]]
        if len(chosen) < per_class and not val_only:
            need = per_class - len(chosen)
            chosen += [(p, "training") for p in train[:need]]
        picks[cls] = chosen
    return picks


def find_mapillary_root(explicit: Path | None):
    if explicit:
        return explicit if explicit.exists() else None
    for name in ("mapillary_vistas", "mapillary", "Mapillary-Vistas"):
        p = REPO / "dataset" / "external" / name
        if p.exists():
            return p
    return None


def mapillary_label_map(root: Path):
    """Return (labels_dirname, {rule_key: [label_ids]}) from config json."""
    for cfg_name in ("config_v2.0.json", "config_v1.2.json", "config.json"):
        cfg = root / cfg_name
        if cfg.exists():
            labels = json.loads(cfg.read_text())["labels"]
            version = "v2.0" if "v2.0" in cfg_name else "v1.2"
            ids = {k: [] for k in MAPILLARY_LABEL_SUFFIXES}
            for i, lab in enumerate(labels):
                for key, suffix in MAPILLARY_LABEL_SUFFIXES.items():
                    if lab["name"].endswith(suffix):
                        ids[key].append(i)
            return version, ids
    raise FileNotFoundError(f"no config*.json found in {root}")


def _scan_mapillary_split(root, split, version, label_ids, picks, per_class,
                          seed, max_scan):
    """Scan one Vistas split, appending to `picks` in place (mutates it)."""
    import numpy as np
    from PIL import Image

    img_dir = root / split / "images"
    lbl_dir = root / split / version / "labels"
    if not lbl_dir.exists():
        lbl_dir = root / split / "labels"
    if not lbl_dir.exists():
        print(f"  mapillary: {split} split not found, skipping")
        return
    masks = sorted(lbl_dir.glob("*.png"))
    rng = random.Random(f"{seed}:{split}")
    rng.shuffle(masks)
    masks = masks[:max_scan]

    for n, mask_path in enumerate(masks):
        if all(len(picks[c]) >= per_class for c in MAPILLARY_RULES):
            break
        if n and n % 1000 == 0:
            print(f"  mapillary: scanned {n}/{len(masks)} {split} masks ...")
        arr = np.array(Image.open(mask_path))
        if arr.ndim == 3:  # v2.0 instance encoding: class id in first channel
            arr = arr[..., 0]
        total = arr.size
        shares = {
            key: sum(int((arr == i).sum()) for i in ids) / total
            for key, ids in label_ids.items()
        }
        img = img_dir / f"{mask_path.stem}.jpg"
        if not img.exists():
            continue
        for cls, rule in MAPILLARY_RULES.items():
            if len(picks[cls]) < per_class and rule(shares):
                picks[cls].append((img, split))
                break  # one bucket per image


def scan_mapillary(root: Path, per_class, seed, max_scan, train_fallback=True):
    """Select Vistas images per class via mask pixel shares.

    Scans the 2k validation masks first. Vistas is street-level dashcam
    footage, so classes like water can be rare there; unlike ADE20K,
    nothing in this pipeline trains on Vistas, so there is no leakage risk
    in also drawing from the 18k training masks - `train_fallback` fills any
    class still short of `per_class` from training.
    """
    version, label_ids = mapillary_label_map(root)
    picks = defaultdict(list)
    _scan_mapillary_split(root, "validation", version, label_ids, picks,
                          per_class, seed, max_scan)
    short = [c for c in MAPILLARY_RULES if len(picks[c]) < per_class]
    if short and train_fallback and (root / "training").exists():
        print(f"  mapillary: {short} short after validation, scanning training ...")
        _scan_mapillary_split(root, "training", version, label_ids, picks,
                              per_class, seed, max_scan)
    return picks


def copy_bucket(rows, files_with_split, source, cls, out_root):
    dest_dir = out_root / source / cls
    dest_dir.mkdir(parents=True, exist_ok=True)
    normalized = [item if isinstance(item, tuple) else (item, "validation")
                  for item in files_with_split]
    # prune files from earlier runs that are no longer picked
    expected = {src.name for src, _ in normalized}
    for old in dest_dir.iterdir():
        if old.is_file() and old.name not in expected:
            old.unlink()
    for src, split in normalized:
        dest = dest_dir / src.name
        if not dest.exists():
            shutil.copy2(src, dest)
        rows.append({
            "filename": str(dest.relative_to(out_root)),
            "source": source,
            "split": split,
            "proposed_class": cls,
            "orig_path": str(src),
        })


def copy_own_frames(rows, frames_dir: Path, out_root: Path):
    dest_dir = out_root / "own_frames"
    dest_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted(frames_dir.glob("*.jpg"))
    for src in frames:
        dest = dest_dir / src.name
        if not dest.exists():
            shutil.copy2(src, dest)
        rows.append({
            "filename": str(dest.relative_to(out_root)),
            "source": "own",
            "split": "-",
            "proposed_class": "",
            "orig_path": str(src),
        })
    return len(frames)


def build_overview(out_root: Path):
    """Contact sheets per bucket + a browsable index.html."""
    from PIL import Image

    ov = out_root / "overview"
    ov.mkdir(exist_ok=True)
    buckets = []  # (title, [relpaths])
    for sub in sorted(out_root.iterdir()):
        if not sub.is_dir() or sub.name == "overview":
            continue
        if sub.name == "own_frames":
            imgs = sorted(sub.glob("*.jpg"))
            if imgs:
                buckets.append((sub.name, imgs))
        else:
            for cls_dir in sorted(sub.iterdir()):
                if cls_dir.is_dir():
                    imgs = sorted(cls_dir.glob("*.jpg"))
                    if imgs:
                        buckets.append((f"{sub.name}/{cls_dir.name}", imgs))

    thumb_w, thumb_h, cols = 180, 120, 10
    html = [
        "<meta charset='utf-8'><title>Env test set - to label</title>",
        "<style>body{font-family:sans-serif;background:#222;color:#eee}"
        "h2{margin:24px 0 8px}img{width:170px;height:113px;object-fit:cover;"
        "margin:1px;vertical-align:top}</style>",
        "<h1>Environment test set &mdash; images to label</h1>",
    ]
    for title, imgs in buckets:
        rows = (len(imgs) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * thumb_w, rows * thumb_h), (34, 34, 34))
        for i, p in enumerate(imgs):
            try:
                im = Image.open(p).convert("RGB")
            except OSError:
                continue
            im.thumbnail((thumb_w, thumb_h))
            x = (i % cols) * thumb_w + (thumb_w - im.width) // 2
            y = (i // cols) * thumb_h + (thumb_h - im.height) // 2
            sheet.paste(im, (x, y))
        sheet_name = "sheet_" + title.replace("/", "_") + ".jpg"
        sheet.save(ov / sheet_name, quality=80)
        html.append(f"<h2>{title} ({len(imgs)} images)</h2>")
        for p in imgs:
            rel = p.relative_to(out_root)
            html.append(f"<a href='../{rel}'><img loading='lazy' src='../{rel}' title='{p.name}'></a>")
        print(f"  overview: {title}: {len(imgs)} images -> overview/{sheet_name}")
    (ov / "index.html").write_text("\n".join(html))
    print(f"  overview: open {ov / 'index.html'} in a browser")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--per-class", type=int, default=150,
                    help="external candidates per class and source (default 150)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=REPO / "dataset" / "test_images")
    ap.add_argument("--ade-root", type=Path,
                    default=REPO / "dataset" / "external" / "ADEChallengeData2016")
    ap.add_argument("--mapillary-root", type=Path, default=None)
    ap.add_argument("--mapillary-max-scan", type=int, default=2000,
                    help="max Vistas masks to scan per split (validation has "
                         "2000, training has 18000)")
    ap.add_argument("--mapillary-val-only", action="store_true",
                    help="never fall back to Vistas training masks")
    ap.add_argument("--frames-dir", type=Path, default=REPO.parent / "frames")
    ap.add_argument("--ade-val-only", action="store_true",
                    help="never use ADE20K training images (strict no-leakage)")
    ap.add_argument("--skip-own", action="store_true",
                    help="do not copy own ride frames")
    ap.add_argument("--dry-run", action="store_true",
                    help="only print per-class candidate counts")
    ap.add_argument("--overview-only", action="store_true",
                    help="only (re)build overview/ from existing test_images")
    args = ap.parse_args()

    if args.overview_only:
        build_overview(args.out)
        return

    # ---- ADE20K ----
    if not args.ade_root.exists():
        sys.exit(f"ADE20K not found at {args.ade_root}")
    ade = load_ade20k_candidates(args.ade_root)
    print("ADE20K candidates (validation + training):")
    for cls in CLASSES:
        nv, nt = len(ade[cls]["validation"]), len(ade[cls]["training"])
        print(f"  {cls:<11} val={nv:>4}  train={nt:>5}")
    picks = pick_ade20k(ade, args.per_class, args.seed, args.ade_val_only)
    print(f"\nADE20K picks (target {args.per_class}/class, validation preferred):")
    train_used = 0
    for cls in CLASSES:
        nv = sum(1 for _, s in picks[cls] if s == "validation")
        nt = len(picks[cls]) - nv
        train_used += nt
        print(f"  {cls:<11} {len(picks[cls]):>4}  (val={nv}, train={nt})")
    if train_used:
        print(f"  WARNING: {train_used} images come from the ADE20K TRAINING split,"
              f"\n  which seg_finetune.ipynb trains on. They are marked with"
              f"\n  split=training in candidates.csv - exclude them when evaluating"
              f"\n  the fine-tuned model, or rerun with --ade-val-only.")

    # ---- Mapillary ----
    map_root = find_mapillary_root(args.mapillary_root)
    map_picks = {}
    if map_root:
        print(f"\nMapillary Vistas found at {map_root}, scanning masks ...")
        map_picks = scan_mapillary(map_root, args.per_class, args.seed,
                                   args.mapillary_max_scan,
                                   train_fallback=not args.mapillary_val_only)
        for cls in MAPILLARY_RULES:
            nv = sum(1 for _, s in map_picks.get(cls, []) if s == "validation")
            nt = len(map_picks.get(cls, [])) - nv
            print(f"  {cls:<11} {len(map_picks.get(cls, [])):>4}  (val={nv}, train={nt})")
    else:
        print("\nMapillary Vistas not found - skipping."
              "\n  To include it: request the research download at"
              "\n  https://www.mapillary.com/dataset/vistas and unpack it to"
              "\n  dataset/external/mapillary_vistas/ (so that"
              "\n  validation/images and validation/*/labels exist), then rerun.")

    if args.dry_run:
        return

    # ---- copy + csv ----
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for cls in CLASSES:
        copy_bucket(rows, picks[cls], "ade20k", cls, args.out)
    for cls, files in map_picks.items():
        copy_bucket(rows, files, "mapillary", cls, args.out)
    if not args.skip_own:
        n = copy_own_frames(rows, args.frames_dir, args.out)
        print(f"\ncopied {n} own frames from {args.frames_dir}")

    csv_path = args.out / "candidates.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "source", "split",
                                          "proposed_class", "orig_path"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {csv_path}")

    build_overview(args.out)
    print("\nNext step: label everything with"
          "\n  python scripts/label_tool.py")


if __name__ == "__main__":
    main()
