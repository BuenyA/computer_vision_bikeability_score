# Merging the 3 Models into One Beauty/Bikeability Score — Step-by-Step Guide

_The environment-model decision is made (2026-07-12): **fine-tuned SegFormer**,
deployed as `models/segformer_env` with the plug-and-play module
`environment_model/env_model.py` — step 1 below is DONE. The plumbing for
everything below already exists in
`environment_model/segmentation_common.py` (`features_from_mask`,
`DEFAULT_WEIGHTS`, `score_from_features`, `aggregate_features`) — it was
designed for this merge._

## Architecture & output form (the design decision)

**One per-frame feature table is the single interchange format.** Each model
writes its own per-frame CSV keyed by `filename`; a small merge script joins
them; the score is computed *from the table*, never inside a model.

```
                 ┌─ env model (CLIP or SegFormer) ─→ env_features.csv    (frac_vegetation, frac_water, frac_city | scene_* probs)
frame ──────────►├─ ground model (ground_detection) → ground_features.csv (frac_asphalt_road, frac_gravel_road, ..., cycle_path_color)
                 └─ object model (YOLO)            ─→ object_features.csv (count_car, count_person, count_traffic_light, ...)
                                   │
                merge_features.py  ▼  join on filename (+ ride_id, frame_ts, lat/lon from GPX)
                            features.csv   ← one row per frame, ALL features
                                   │
                score_features.py  ▼  score_from_features(DEFAULT_WEIGHTS)
                            scores.csv     ← per-frame beauty + bikeability in [0,1]
                                   │
                aggregate          ▼  aggregate_features per ride / GPS window
                            segments.csv (+ GeoJSON once GPS joined)
```

Why this form:
- **Decoupled:** each model runs/re-runs independently; a model swap only
  replaces one CSV.
- **Re-scorable:** weight changes (tuning the beauty formula) need no model
  re-runs — just re-score the table.
- **Continuous features, not binary labels:** the score consumes *fractions*
  (`frac_water = 0.11`), which carry far more signal than present/absent. Keep
  thresholds only for the classification evaluation, not for scoring.
- **Paper-friendly:** the feature table is directly plottable (score along a
  route, feature ablations).

## Steps

1. **Freeze the environment model — DONE (2026-07-12, SegFormer fine-tuned).**
   `environment_model/env_model.py` is the deliverable:

   ```python
   from env_model import EnvironmentModel
   env = EnvironmentModel("models/segformer_env")     # once, at startup
   pred = env.predict(frame)      # path | PIL | (H,W,3) RGB ndarray
   pred.fractions                 # np.float32 [frac_vegetation, frac_water, frac_city]
   pred.labels                    # np.int8    [vegetation, water, city]
   pred.features()                # dict row for features.csv
   ```

   CLI over a frame directory:
   `python environment_model/env_model.py <frames_dir> --out env_features.csv`
   → columns `filename, frac_vegetation, frac_water, frac_city,
   label_vegetation, label_water, label_city`. The score consumes the
   `frac_*` columns; the `label_*` columns are the val-tuned thresholded
   classifications (thresholds ship in
   `models/segformer_env/inference_config.json`, used for reporting/F1 only).
   Deployment needs exactly `env_model.py` + the `models/segformer_env/`
   folder (torch, numpy, Pillow, transformers).

2. **Ground-type extractor** (`ground_detection.ipynb` → script). Per frame,
   output `frac_asphalt_road, frac_cobblestone_road, frac_gravel_road,
   frac_dirt_road, frac_cycle_path` (+ `cycle_path_color`). Same CSV pattern,
   same `filename` key.

3. **Object extractor** (`yolo.ipynb` → script). Per frame, output
   `count_traffic_light, count_car, count_person, count_bicycle_rider`
   (detections above a confidence threshold; always emit all columns, 0 if
   none — stable schema).

4. **Merge** (`merge_features.py`, ~30 lines): outer-join the three CSVs on
   `filename`; add `ride_id`, `frame_ts` (parsed from filename) and `lat/lon`
   (GPX join via ride_id+frame_ts when ready); missing values → 0.0. Output
   `features.csv`.

5. **Score** each row with `sc.score_from_features(row, sc.DEFAULT_WEIGHTS)` →
   `beauty`, `bikeability` columns. The current beauty weights:
   `frac_water 1.0 · frac_vegetation 0.7 · frac_city −0.4` (+ `scene_*`
   variants); bikeability uses the surface fractions and traffic counts.
   Weights are heuristic — tune/justify them once real feature tables exist.

6. **Aggregate** per segment with `sc.aggregate_features` (mean of numeric,
   mode of color) — segment = ride, or a GPS/time window (e.g. 30 s) for
   along-route resolution. Score the aggregated features again → segment score.

7. **Validate**: score 2–3 rides you know subjectively (one scenic, one urban,
   one mixed) and check the ranking matches intuition; plot the per-frame
   beauty over time for one ride. This is the paper's qualitative figure.

8. **Deliverables**: `features.csv` (per frame), `scores.csv` (per frame),
   `segments.csv` / GeoJSON (per segment, once GPS is joined) — the GeoJSON is
   what a map visualisation consumes.

## Collaborative labeling of the final test set (merge-at-the-end works)

Parallel labeling + merging CSVs at the end **works with this setup**, with
three rules:

1. **Partition by whole rides/sources, never interleaved frames.** Each
   annotator gets disjoint `ride_id`s (and/or the external buckets). This keeps
   the sticky-label speedup, makes every sequence internally consistent, and
   guarantees conflict-free merging (no duplicate filenames).
   Each labels into their own file:
   `python environment_model/data/label_tool.py --csv dataset/test_images/labels_<name>.csv
   --only 'DJI_03(5[0-9]|6[0-5])'` (adjust regex per assignment).
2. **Calibrate FIRST via the κ pilot** (`dataset/eval/kappa_pilot/`, 50 frames):
   everyone labels the same 50 frames *before* mass-labeling, disagreements are
   discussed, then Cohen's κ (`environment_model/evaluation/compute_kappa.py`) is reported in the
   paper as the annotation ceiling. Without this step, per-person drift in the
   fuzzy rules ("more than 3 trees") silently degrades the ground truth.
3. **Merge = concatenate + audit** — same schema, disjoint filenames, so the
   merge is `pd.concat` plus two checks: no duplicate `filename`, no missing
   rides; add an `annotator` column for provenance. (A `merge_labels.py` doing
   exactly this is ~15 lines; ask Claude for it when the CSVs exist.)

The one thing to avoid: two people labeling the same ride independently and
"merging" by overwrite — that mixes calibrations mid-sequence. Overlap is only
for the κ pilot, where it is deliberate and measured.
