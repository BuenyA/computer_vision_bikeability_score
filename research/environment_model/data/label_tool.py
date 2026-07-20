#!/usr/bin/env python3
"""Hotkey tool for environment multi-labeling with sticky labels.

Walks all images under dataset/test_images/ (recursive, overview/ excluded).
Order is POV-prioritized: own cyclist-POV ride frames first (the deployment
domain, guaranteed ground-level), then the scarce forest/water external
buckets, then the rest. Own frames stay in ride sequence and external images
stay grouped by proposed class - both make the sticky behavior effective:
on Space the current class toggles are saved AND carried over to the next
unlabeled image, so slowly changing environments need almost no keystrokes.

Annotation rules (ground-truth class definitions; a frame may match several):
  vegetation  trees, forest, grass, fields or meadows - any notable greenery
  water       more than ~2% of the image shows water (river/lake/canal/sea)
  city        buildings, industry, or a street with houses

If NONE of the three fit (e.g. a tunnel, a bare road with only sky, an indoor
shot), leave all toggles OFF - an all-zero label is valid and means "none of
these environments". Use 'x' to clear a carried-over sticky preset, and 'u'
(unsure) only for genuine borderline cases.

Keys:
  v / w / c           toggle vegetation / water / city
  Space / right arrow save + next image (sticky: toggles stay on)
  Backspace / left    one image back (shows its saved labels)
  u                   toggle "unsure" for borderline cases (not sticky)
  r                   toggle "reject" = not cyclist POV / unusable, excluded
                      from the test set downstream (not sticky)
  x                   clear all toggles for the current image (use when none fit)
  s                   write CSV now (also happens on every Space)
  q / Esc             save + quit (resume works: restart skips labeled images)

CSV schema (multi-hot, one row per image):
  filename, ride_id, frame_ts, lat, lon,
  vegetation, water, city, unsure, reject

ride_id is parsed from DJI filenames (split unit for the leakage-free split);
external images get their source as ride_id. frame_ts is the frame index from
the filename; lat/lon stay empty and can be joined from GPX later via
ride_id + frame_ts.

Usage:
  python scripts/label_tool.py                    # default folders, resume
  python scripts/label_tool.py --first 'DJI_039[2-9]'   # label new rides FIRST
  python scripts/label_tool.py --only  'DJI_0392'       # label ONLY these frames
  python scripts/label_tool.py --stats            # progress, no GUI
"""

import argparse
import csv
import os
import re
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent

CLASSES = ["vegetation", "water", "city"]
KEY_TO_CLASS = {ord("v"): "vegetation", ord("w"): "water", ord("c"): "city"}
CLASS_BGR = {  # OpenCV BGR
    "vegetation": (50, 160, 40),
    "water": (180, 130, 0),
    "city": (130, 130, 130),
}
FIELDNAMES = ["filename", "ride_id", "frame_ts", "lat", "lon",
              *CLASSES, "unsure", "reject"]

MAX_W, MAX_H = 1440, 810
BAR_H = 46  # top and bottom overlay bars

KEYS_NEXT = {32, 3, 63235, 65363, 2555904}          # Space, right arrow variants
KEYS_BACK = {8, 127, 2, 63234, 65361, 2424832}      # Backspace/Delete, left arrow
KEYS_QUIT = {ord("q"), 27}                          # q, Esc


def order_key(rel: str):
    """Sort key: own cyclist-POV frames first (gold domain), then the scarce
    water/vegetation external buckets, then everything else; path-sorted within.
    (External folders keep their original 4-class names; forest/open_field both
    map to the merged 'vegetation' label.)"""
    parts = rel.replace("\\", "/").split("/")
    top = parts[0]
    if top == "own_frames":
        return (0, rel)
    if len(parts) > 1 and parts[1] in ("water", "forest", "open_field"):
        return (1, rel)
    return (2, rel)


def collect_images(root: Path, first_re=None, only_re=None):
    """Walk images. `only_re` restricts to matching relpaths; `first_re` sorts
    matching frames to the very front (e.g. newly added rides), before the
    normal own-frames-first order."""
    exts = {".jpg", ".jpeg", ".png"}
    imgs = [p for p in root.rglob("*")
            if p.suffix.lower() in exts and "overview" not in p.parts]
    if only_re:
        imgs = [p for p in imgs if only_re.search(str(p.relative_to(root)))]

    def key(p):
        rel = str(p.relative_to(root))
        if first_re and first_re.search(rel):
            return (-1, rel)                 # new frames jump to the front
        return order_key(rel)
    return sorted(imgs, key=key)


def meta_from_path(root: Path, path: Path):
    """(relpath, ride_id, frame_ts) - ride_id is the later split unit."""
    rel = str(path.relative_to(root))
    m = re.search(r"(DJI_\d+)", path.name)
    if m:
        ride = m.group(1)
    else:
        ride = path.relative_to(root).parts[0]  # e.g. "ade20k", "mapillary"
    m = re.search(r"frame_(\d+)", path.name)
    ts = int(m.group(1)) if m else ""
    return rel, ride, ts


def load_labels(csv_path: Path):
    labels = {}
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                labels[row["filename"]] = row
    return labels


def save_labels(csv_path: Path, labels: dict, order: list):
    """Atomic rewrite, rows in display order (resume-safe on crash)."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=csv_path.parent, suffix=".csv")
    with os.fdopen(fd, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for rel in order:
            if rel in labels:
                w.writerow(labels[rel])
    os.replace(tmp, csv_path)


def make_row(rel, ride, ts, toggles, unsure, reject):
    row = {"filename": rel, "ride_id": ride, "frame_ts": ts,
           "lat": "", "lon": "", "unsure": int(unsure), "reject": int(reject)}
    for c in CLASSES:
        row[c] = int(toggles[c])
    return row


def render(img, rel, toggles, unsure, reject, idx, total, n_labeled, is_saved):
    h, w = img.shape[:2]
    scale = min(MAX_W / w, MAX_H / h, 1.0)
    view = cv2.resize(img, (int(w * scale), int(h * scale)))
    if reject:  # dim + red border so rejected frames read at a glance
        view = (view * 0.45).astype(np.uint8)
    vh, vw = view.shape[:2]
    canvas = np.zeros((vh + 2 * BAR_H, max(vw, 900), 3), dtype=np.uint8)
    canvas[:] = (25, 25, 25)
    canvas[BAR_H:BAR_H + vh, :vw] = view
    if reject:
        cv2.rectangle(canvas, (0, BAR_H), (vw - 1, BAR_H + vh - 1),
                      (0, 0, 255), 4)

    # top bar: class chips
    x = 10
    for c in CLASSES:
        label = f"[{c[0].upper()}] {c}"
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
        x2 = x + tw + 20
        color = CLASS_BGR[c]
        if toggles[c]:
            cv2.rectangle(canvas, (x, 8), (x2, BAR_H - 8), color, -1)
            txt_col = (255, 255, 255)
        else:
            cv2.rectangle(canvas, (x, 8), (x2, BAR_H - 8), color, 1)
            txt_col = (140, 140, 140)
        cv2.putText(canvas, label, (x + 10, BAR_H - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, txt_col, 2, cv2.LINE_AA)
        x = x2 + 8
    if unsure:
        cv2.putText(canvas, "[U] UNSURE", (x + 8, BAR_H - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 200, 255), 2, cv2.LINE_AA)
        x += 150
    if reject:
        cv2.putText(canvas, "[R] REJECT (not POV)", (x + 8, BAR_H - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 255), 2, cv2.LINE_AA)

    # bottom bar: progress + filename + help
    y = BAR_H + vh
    status = "saved" if is_saved else "NEW (sticky preset)"
    info = f"{idx + 1}/{total}  labeled {n_labeled}  |  {rel}  |  {status}"
    cv2.putText(canvas, info, (10, y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1, cv2.LINE_AA)
    help_line = ("v/w/c toggle   Space save+next   Backspace back   "
                 "u unsure   r reject   x clear (none fit)   s save   q quit")
    cv2.putText(canvas, help_line, (10, y + 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
    return canvas


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--images", type=Path,
                    default=REPO / "dataset" / "test_images")
    ap.add_argument("--csv", type=Path, default=None,
                    help="default: <images>/labels.csv")
    ap.add_argument("--stats", action="store_true",
                    help="print labeling progress and exit (no GUI)")
    ap.add_argument("--first", type=str, default=None,
                    help="regex on filename; matching frames are labeled FIRST "
                         "(e.g. --first 'DJI_039[2-9]' for newly added rides)")
    ap.add_argument("--only", type=str, default=None,
                    help="regex on filename; label ONLY matching frames")
    args = ap.parse_args()
    csv_path = args.csv or args.images / "labels.csv"

    first_re = re.compile(args.first) if args.first else None
    only_re = re.compile(args.only) if args.only else None
    images = collect_images(args.images, first_re, only_re)
    if not images:
        sys.exit(f"no images found under {args.images} - "
                 f"run scripts/build_test_dataset.py first")
    metas = [meta_from_path(args.images, p) for p in images]
    order = [rel for rel, _, _ in metas]
    labels = load_labels(csv_path)
    labels = {k: v for k, v in labels.items() if k in set(order)}

    if args.stats:
        print(f"{len(labels)}/{len(images)} labeled ({csv_path})")
        counts = {c: 0 for c in CLASSES}
        unsure = reject = 0
        for row in labels.values():
            rejected = int(row.get("reject", 0))
            reject += rejected
            unsure += int(row["unsure"])
            if rejected:  # rejected frames don't count toward class totals
                continue
            for c in CLASSES:
                counts[c] += int(row[c])
        kept = len(labels) - reject
        print(f"  kept (in test set): {kept}   rejected (not POV): {reject}")
        for c in CLASSES:
            print(f"  {c:<11} {counts[c]:>5} positive")
        print(f"  {'unsure':<11} {unsure:>5}")
        return

    # resume: start at the first unlabeled image
    idx = next((i for i, rel in enumerate(order) if rel not in labels),
               len(images) - 1)
    sticky = {c: False for c in CLASSES}
    if idx > 0 and order[idx - 1] in labels:  # carry last saved state over
        prev = labels[order[idx - 1]]
        sticky = {c: bool(int(prev[c])) for c in CLASSES}

    win = "env labeler"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    print(f"{len(labels)}/{len(images)} already labeled - starting at "
          f"#{idx + 1}. Keys: see window footer.")

    toggles, unsure, reject = None, False, False
    loaded_idx = -1
    while True:
        if loaded_idx != idx:
            rel, ride, ts = metas[idx]
            if rel in labels:  # revisit: show saved state
                row = labels[rel]
                toggles = {c: bool(int(row[c])) for c in CLASSES}
                unsure = bool(int(row["unsure"]))
                reject = bool(int(row.get("reject", 0)))
            else:  # new image: sticky preset (reject/unsure never sticky)
                toggles = dict(sticky)
                unsure = reject = False
            img = cv2.imread(str(images[idx]))
            if img is None:
                img = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(img, "unreadable image", (30, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            loaded_idx = idx

        rel, ride, ts = metas[idx]
        cv2.imshow(win, render(img, rel, toggles, unsure, reject, idx,
                               len(images), len(labels), rel in labels))
        k = cv2.waitKeyEx(50)
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break  # window closed -> save + quit
        if k == -1:
            continue
        if k in KEY_TO_CLASS:
            toggles[KEY_TO_CLASS[k]] = not toggles[KEY_TO_CLASS[k]]
        elif k == ord("u"):
            unsure = not unsure
        elif k == ord("r"):
            reject = not reject
        elif k == ord("x"):
            toggles = {c: False for c in CLASSES}
            unsure = reject = False
        elif k == ord("s"):
            save_labels(csv_path, labels, order)
            print(f"saved {len(labels)} labels -> {csv_path}")
        elif k in KEYS_NEXT:
            labels[rel] = make_row(rel, ride, ts, toggles, unsure, reject)
            sticky = dict(toggles)
            save_labels(csv_path, labels, order)
            if idx + 1 < len(images):
                idx += 1
            else:
                print("last image labeled - done!")
                break
        elif k in KEYS_BACK:
            idx = max(0, idx - 1)
        elif k in KEYS_QUIT:
            break

    save_labels(csv_path, labels, order)
    cv2.destroyAllWindows()
    print(f"{len(labels)}/{len(images)} labeled -> {csv_path}")


if __name__ == "__main__":
    main()
