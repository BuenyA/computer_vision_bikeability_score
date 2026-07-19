"""Shared taxonomy, palette, label maps, metrics and scoring.

Single source of truth for the zero-shot vs. fine-tuned segmentation comparison
(see plan: Zero-shot vs. Fine-tuned scene understanding for the Bikeability/Beauty score).

Imported by:
- research/seg_zeroshot.ipynb   (Notebook A)
- research/seg_finetune.ipynb   (Notebook B)
- research/seg_evaluation.ipynb (Notebook C)

Design notes
------------
* The unified taxonomy below defines the class index used *everywhere* (masks,
  metrics, features). `VOID_ID = 255` marks ignore/unlabeled pixels.
* Classes fall into three groups (`CATEGORIES`): **surface** and **environment**
  are area/"stuff" classes (semantic masks); **traffic** objects
  (traffic_light/car/person/bicycle_rider) are countable "things" better produced
  by detection (YOLO-World / instance seg) and rasterized into the mask.
* `city`/`vegetation` can additionally be reinforced by
  per-frame CLIP scene attributes in the zero-shot notebook.
* Label maps are keyed by the *readable source class name* rather than a numeric
  id, because the numeric ordering differs between dataset versions. Notebook B
  turns a name->taxonomy map into the source_id->taxonomy_id lookup it needs at
  load time via `build_id_lookup(source_names, name_map)`.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# 1. Unified taxonomy: (name, RGB color). List order == class id.
# ---------------------------------------------------------------------------
TAXONOMY: list[tuple[str, tuple[int, int, int]]] = [
    # --- surface classification (road split by material) ---
    ("asphalt_road",     (128,  64, 128)),
    ("cobblestone_road", (190, 153, 153)),
    ("gravel_road",      (180, 165, 120)),
    ("dirt_road",        (110,  80,  50)),
    ("cycle_path",       (255,   0, 200)),   # base class; red/blue/black is a separate attribute
    # --- environment classification (beauty score) ---
    ("vegetation",       ( 60, 160,  50)),   # merged forest + open_field
    ("water",            (  0, 130, 180)),
    ("city",             (130, 130, 130)),
    # --- traffic object detection ---
    ("traffic_light",    (250, 170,  30)),
    ("car",              (  0,   0, 142)),
    ("person",           (220,  20,  60)),
    ("bicycle_rider",    (255,   0,   0)),
]

CLASSES: list[str] = [name for name, _ in TAXONOMY]
CLASS_ID: dict[str, int] = {name: i for i, name in enumerate(CLASSES)}
NUM_CLASSES: int = len(CLASSES)
VOID_ID: int = 255

# Three task groups (mirrors how the features feed the score / who owns each path).
CATEGORIES: dict[str, list[str]] = {
    "surface":     ["asphalt_road", "cobblestone_road", "gravel_road", "dirt_road", "cycle_path"],
    "environment": ["vegetation", "water", "city"],
    "traffic":     ["traffic_light", "car", "person", "bicycle_rider"],
}

# Palette as an (NUM_CLASSES, 3) uint8 array; index by class id to colorize.
PALETTE: np.ndarray = np.array([rgb for _, rgb in TAXONOMY], dtype=np.uint8)

# Cycle-path surface colors we try to distinguish (HSV step in the notebooks).
CYCLE_PATH_COLORS: tuple[str, ...] = ("red", "blue", "black")


# ---------------------------------------------------------------------------
# 2. Source-dataset -> taxonomy label maps (keyed by readable source name)
#    These are intentionally partial / high-confidence. Complete them in
#    Notebook B against the dataset's own class list (e.g. config_v2.0.json).
#    Any source class not present here -> VOID_ID (ignored in training).
# ---------------------------------------------------------------------------
MAPILLARY_TO_TAXONOMY: dict[str, str] = {
    # flat / drivable -> surface
    "construction--flat--road":            "asphalt_road",   # material unknown -> default asphalt
    "construction--flat--service-lane":    "asphalt_road",
    "construction--flat--parking":         "asphalt_road",
    "construction--flat--bike-lane":       "cycle_path",
    # structures -> city
    "construction--structure--building":   "city",
    # nature -> environment (vegetation + terrain both merge into 'vegetation')
    "nature--water":                       "water",
    "nature--vegetation":                  "vegetation",
    "nature--terrain":                     "vegetation",
    # traffic objects
    "human--person--individual":           "person",
    "human--rider--bicyclist":             "bicycle_rider",
    "human--rider--motorcyclist":          "bicycle_rider",
    "object--vehicle--car":                "car",
    "object--vehicle--truck":              "car",
    "object--vehicle--bus":                "car",
    "object--vehicle--bicycle":            "bicycle_rider",
    "object--traffic-light--general-upright": "traffic_light",
}

ADE20K_TO_TAXONOMY: dict[str, str] = {
    # NOTE: keys cover BOTH the long objectInfo150 names ("building; edifice")
    # and the short HuggingFace id2label names ("building") - checkpoints use
    # the short form, sceneCategories tooling the long form.
    "road; route":          "asphalt_road",
    "road":                 "asphalt_road",
    "building; edifice":    "city",
    "building":             "city",
    "house":                "city",
    "skyscraper":           "city",
    "tree":                 "vegetation",
    "plant; flora; plant life": "vegetation",
    "plant":                "vegetation",
    "grass":                "vegetation",
    "field":                "vegetation",
    "earth; ground":        "dirt_road",
    "water":                "water",
    "sea":                  "water",
    "lake":                 "water",
    "river":                "water",
    "person; individual; someone; somebody; mortal; soul": "person",
    "car; auto; automobile; machine; motorcar": "car",
    "truck; motortruck":    "car",
    "bus; autobus":         "car",
    "bicycle; bike; wheel; cycle": "bicycle_rider",
    "traffic light; traffic signal; stoplight": "traffic_light",
}


def build_id_lookup(
    source_names: list[str],
    name_map: dict[str, str],
    class_id: dict[str, int] | None = None,
) -> np.ndarray:
    """Build a source_id -> target_id lookup array from a name map.

    `source_names[i]` is the readable name of source class id `i`. `class_id`
    defaults to the full taxonomy `CLASS_ID`; pass a subset map (e.g. an
    environment-only `{name: 0..K}`) to train on fewer classes. Source classes
    whose taxonomy name is absent from `class_id` map to VOID_ID.
        lut = build_id_lookup(mapillary_names, MAPILLARY_TO_TAXONOMY)
        taxonomy_mask = lut[source_mask]
    """
    class_id = class_id if class_id is not None else CLASS_ID
    lut = np.full(len(source_names), VOID_ID, dtype=np.int64)
    for i, name in enumerate(source_names):
        tax_name = name_map.get(name)
        if tax_name in class_id:
            lut[i] = class_id[tax_name]
    return lut


# ---------------------------------------------------------------------------
# 3. IO / visualization helpers
# ---------------------------------------------------------------------------
def colorize(mask: np.ndarray) -> np.ndarray:
    """(H, W) class-id mask -> (H, W, 3) uint8 RGB. VOID stays black."""
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    valid = mask < NUM_CLASSES
    out[valid] = PALETTE[mask[valid]]
    return out


def overlay(image_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Alpha-blend the colorized mask onto an RGB image."""
    color = colorize(mask).astype(np.float32)
    blended = alpha * color + (1.0 - alpha) * image_rgb.astype(np.float32)
    return blended.astype(np.uint8)


# ---------------------------------------------------------------------------
# 4. Metrics (confusion-matrix based, numpy only -> no heavy deps)
# ---------------------------------------------------------------------------
def confusion_matrix(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Accumulate an (NUM_CLASSES, NUM_CLASSES) confusion matrix for one pair.

    Rows = ground truth, cols = prediction. VOID pixels in GT are ignored.
    """
    valid = gt != VOID_ID
    g = gt[valid].astype(np.int64)
    p = pred[valid].astype(np.int64)
    # guard against stray out-of-range predictions
    inside = (p >= 0) & (p < NUM_CLASSES)
    g, p = g[inside], p[inside]
    idx = g * NUM_CLASSES + p
    cm = np.bincount(idx, minlength=NUM_CLASSES ** 2)
    return cm.reshape(NUM_CLASSES, NUM_CLASSES)


def per_class_iou(cm: np.ndarray) -> np.ndarray:
    """IoU per class from a confusion matrix; NaN for classes absent in GT."""
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = tp + fp + fn
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(denom > 0, tp / denom, np.nan)
    return iou


def miou(cm: np.ndarray) -> float:
    """Mean IoU over classes present in GT (NaNs ignored)."""
    return float(np.nanmean(per_class_iou(cm)))


def per_class_f1(cm: np.ndarray) -> np.ndarray:
    """Dice/F1 per class from a confusion matrix; NaN for absent classes."""
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(denom > 0, 2 * tp / denom, np.nan)
    return f1


def pixel_accuracy(cm: np.ndarray) -> float:
    total = cm.sum()
    return float(np.diag(cm).sum() / total) if total > 0 else float("nan")


def multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Macro-F1 + accuracy for multi-label classification.

    `y_true`, `y_pred`: (N, C) binary arrays, columns aligned to a class list.
    Returns `macro_f1`, `subset_accuracy` (exact-match over all labels),
    `label_accuracy` (mean correctness per label) and the `per_class_f1` array.
    """
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    tp = (y_pred & y_true).sum(0).astype(np.float64)
    fp = (y_pred & ~y_true).sum(0).astype(np.float64)
    fn = (~y_pred & y_true).sum(0).astype(np.float64)
    denom = 2 * tp + fp + fn
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(denom > 0, 2 * tp / denom, np.nan)
    return {
        "macro_f1": float(np.nanmean(f1)),
        "subset_accuracy": float((y_pred == y_true).all(axis=1).mean()),
        "label_accuracy": float((y_pred == y_true).mean()),
        "per_class_f1": f1,
    }


# ---------------------------------------------------------------------------
# 5. Per-frame features
# ---------------------------------------------------------------------------
def features_from_mask(
    mask: np.ndarray,
    scene_attrs: dict | None = None,
    object_counts: dict | None = None,
    cycle_path_color: str | None = None,
) -> dict:
    """Turn a unified mask (+ side info) into a flat feature dict.

    Returns class-area fractions `frac_<class>` (over non-void pixels) plus the
    extra per-frame signals the score needs: `scene_<attr>`, `count_<class>` for
    detected traffic objects, and the cycle-path colour.
    """
    valid = mask != VOID_ID
    n = int(valid.sum())
    counts = np.bincount(mask[valid].astype(np.int64), minlength=NUM_CLASSES) if n else np.zeros(NUM_CLASSES)
    feats: dict[str, float] = {
        f"frac_{name}": (counts[i] / n if n else 0.0) for i, name in enumerate(CLASSES)
    }
    feats["cycle_path_color"] = cycle_path_color or "none"
    # Always emit a count per traffic class (0 if none) so the feature schema is
    # stable across frames/notebooks (a missing column would become NaN on aggregation).
    oc = object_counts or {}
    for name in CATEGORIES["traffic"]:
        feats[f"count_{name}"] = float(oc.get(name, 0))
    if scene_attrs:
        for k, v in scene_attrs.items():
            feats[f"scene_{k}"] = float(v)
    return feats


# ---------------------------------------------------------------------------
# 6. Scoring (weighted sum -> [0, 1]); weights are heuristic, tunable in Nb C
# ---------------------------------------------------------------------------
# Positive weights raise the score, negative lower it. Keys are feature names
# from `features_from_mask`. See README §3.1 (A = sum w+ F+ - sum w- F-).
DEFAULT_WEIGHTS: dict[str, dict[str, float]] = {
    "bikeability": {
        "frac_cycle_path":       1.0,
        "frac_asphalt_road":     0.5,
        "frac_cobblestone_road": -0.6,
        "frac_gravel_road":      -0.7,
        "frac_dirt_road":        -0.7,
        "frac_car":              -0.8,
        "count_car":             -0.1,   # detected vehicles (traffic load)
        "frac_traffic_light":    -0.1,
    },
    "beauty": {
        "frac_water":       1.0,
        "frac_vegetation":  0.7,   # merged forest(0.8)+open_field(0.5)
        "scene_vegetation": 0.4,
        "frac_city":       -0.4,
        "scene_city":      -0.3,
    },
}


def _squash(x: float) -> float:
    """Map a raw weighted sum to [0, 1] with a logistic so it never saturates hard."""
    return float(1.0 / (1.0 + np.exp(-x)))


def score_from_features(
    features: dict,
    weights: dict[str, dict[str, float]] | None = None,
) -> dict[str, float]:
    """Compute {'bikeability': s, 'beauty': s} in [0, 1] from a feature dict."""
    weights = weights or DEFAULT_WEIGHTS
    scores: dict[str, float] = {}
    for dim, wmap in weights.items():
        raw = sum(w * float(features.get(feat, 0.0)) for feat, w in wmap.items())
        scores[dim] = _squash(raw)
    return scores


def aggregate_features(frame_features: list[dict]) -> dict:
    """Aggregate per-frame features over a segment (mean of numeric, mode of color)."""
    if not frame_features:
        return {}
    keys = frame_features[0].keys()
    agg: dict = {}
    for k in keys:
        vals = [f.get(k) for f in frame_features]
        if all(isinstance(v, (int, float)) for v in vals):
            agg[k] = float(np.mean(vals))
        else:  # categorical (e.g. cycle_path_color) -> most common
            agg[k] = max(set(vals), key=vals.count)
    return agg
