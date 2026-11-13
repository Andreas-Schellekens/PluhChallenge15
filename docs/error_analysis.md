# Error analysis of the puck detector

What the puck detector gets wrong, why, and what we think would fix it. The errors are sorted
into the categories from the course (deck 5, slide 38 "What can go wrong?"). The post-processing
settings (confidence threshold, NMS) are measured first, so that we know which errors are a
threshold problem and which are a model or data problem.

**Status (2026-11-13).** Drafted by Claude Code. The numbers come from a saved run
(`outputs/sweep/E2a_val/`, command below). **The type of each error (blade, blur, far, ...) is a
visual judgement by Claude and has not been confirmed by a team member yet.** To confirm: open
`outputs/sweep/E2a_val/errors.jpg` and fill in the `category` column of `errors.csv`.

## Setup

- Model: E2a (`outputs/train/fillempje_v3_yolo26s_puck_640/weights/best.pt`), the model the app uses.
- Frames: the 25 val frames of Roboflow v3 (frames ≥ 4200), 10 of them with a puck, 15 verified
  without (`docs/annotation.md`, round 2).
- Inference: 640 px tiles with 160 px overlap, as in `eval_puck.py`.
- A detection counts as correct when its IoU with the labelled puck is ≥ 0.5 (the mAP@50 rule).
- GPU: RTX 4060 Laptop (Zjef), Ultralytics 8.4.211. Run time: 22 s.

```
python scripts/sweep_conf_nms.py --weights outputs/train/fillempje_v3_yolo26s_puck_640/weights/best.pt --frames data/puck_tiles/fillempje_v3_tiles640/frames.csv --name E2a_val
```

**Sanity check.** At the settings `eval_puck.py` uses (conf 0.25, tile NMS 0.7, merge NMS 0.3)
the sweep gives TP 6 / FP 1 / FN 4 and AP@50 0.65, exactly the E2a row of E2 in
`experiments.md`. The val tiles were rebuilt on this PC from the Roboflow export; validating the
team's `best.pt` on them gives mAP@50 0.634 / mAP@50:95 0.382, the same as logged.

## Part 1: confidence threshold and NMS (slide 39)

Our pipeline has **two** NMS steps, because we predict tile by tile:
1. **tile NMS** inside each 640 px tile, done by Ultralytics (`iou=` of `predict`, default 0.7).
   Our weights have `end2end=False`, so this is classic NMS.
2. **merge NMS** after the tile boxes are shifted back to the full frame. The tiles overlap by
   160 px, so a puck in an overlap strip is found by two tiles. `eval_puck.py` removes those
   copies with its own NMS at IoU 0.3.

Every false positive is also given a type, following slide 37 ("not all wrong boxes are equally
wrong"): **duplicate** (a second box on a puck that is already found), **localization** (on an
unfound puck, but IoU < 0.5) or **background** (not on any labelled puck).

### Confidence threshold (tile NMS 0.7, merge NMS 0.3)

| conf | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|
| 0.05 – 0.25 | 6 | 1 | 4 | 0.86 | 0.60 | **0.71** |
| 0.30 | 5 | 1 | 5 | 0.83 | 0.50 | 0.63 |
| 0.35 | 4 | 1 | 6 | 0.80 | 0.40 | 0.53 |
| 0.40 – 0.60 | 3 | 0 | 7 | 1.00 | 0.30 | 0.46 |
| 0.70 | 1 | 0 | 9 | 1.00 | 0.10 | 0.18 |
| ≥ 0.75 | 0 | 0 | 10 | – | 0.00 | 0.00 |

(`pr_vs_conf.png`)

### NMS: AP@50 and false positives at conf 0.25

| tile NMS ↓ / merge NMS → | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 | none |
|---|---|---|---|---|---|---|
| 0.3 | 0.65 / 1 | 0.65 / 1 | 0.65 / 1 | 0.65 / 1 | 0.61 / 2 | 0.41 / 13 |
| 0.5 | 0.65 / 1 | 0.65 / 1 | 0.65 / 1 | 0.65 / 1 | 0.61 / 2 | 0.41 / 13 |
| 0.7 (default) | 0.65 / 1 | **0.65 / 1** | 0.65 / 1 | 0.63 / 1 | 0.59 / 2 | 0.40 / 13 |
| 0.9 | 0.65 / 1 | 0.65 / 1 | 0.65 / 1 | 0.63 / 1 | 0.47 / 9 | 0.36 / 28 |

Each cell: AP@50 over all confidences / number of false positives at conf 0.25 (`nms_effect.png`).
The number of found pucks is 6 in every cell: NMS never removed a correct detection here.

### Observations

1. **The merge NMS is essential.** Without it, AP@50 drops from 0.65 to 0.41 and the false
   positives go from 1 to 13. 11 of those 13 are of the type *duplicate*: the same puck found by
   two overlapping tiles. The other 2 are background boxes on frames without a puck (probably the
   blade spot of f4290 found by two tiles; not checked one by one).
   This is error type 3 of slide 38 ("duplicate detections → needs NMS"),
   and it is caused by our own design choice (overlapping tiles), not by the model.
2. **The exact NMS value hardly matters, as long as both steps are ≤ 0.7.** A puck is about
   10 px, so two boxes on the same puck overlap a lot and two different pucks never occur in one
   frame. Only very loose settings (0.9) let duplicates through. The fear from slide 39, that a
   low NMS threshold removes real nearby objects, does not apply to a single small puck.
   It would matter for the player detector (crowded scenes, slide 38 type 8).
3. **The confidence threshold cannot fix the missed pucks.** Between conf 0.05 and 0.25 the result
   is identical: the 4 missed pucks get **no box at all** with a score ≥ 0.05. They are a model or
   data problem (part 2), not a threshold problem.
4. **Raising the threshold costs more than it gains.** At conf 0.40 the one false positive
   (score 0.39) disappears, but 3 of the 6 found pucks have a score between 0.25 and 0.40 and are
   lost too. Conf 0.25 sits at the knee of the curve: the highest F1 (0.71) with the highest
   threshold that still keeps all 6 pucks. **We keep the current settings (conf 0.25, tile NMS 0.7,
   merge NMS 0.3), now based on a measurement instead of the defaults.**
5. **Limits.** 10 pucks and 1 false positive are very few: one puck more or less moves recall by
   10 points. The settings are chosen on the same val frames they are scored on, and part of the
   val set was built with the E1 model (A1, finding 4). The script must be run again on the
   model-free test set once it exists; the command stays the same with another `--frames`.

## Part 2: the errors, by type (slide 38)

At the operating point E2a makes 5 errors on the val frames: 1 false positive and 4 missed pucks
(`errors.csv`, `errors.jpg`). Error types that only appear in other runs of the log are added with
their source.

| # | slide 38 type | where we see it | count | likely cause (link to annotation or design) |
|---|---|---|---|---|
| 1 | **missed small object** | f4500: a puck of about 6 px at mid-ice, in front of a stick. E1: f4200 at the far end (2–4 px) | 1 val FN (E2a) | the far half of the rink is small in the image (H3: 30 rows for 30 m). Few labelled pucks there, because the proposal step found mostly near pucks (round 1) |
| 2 | **false positive** (look-alike) | f4290: the dark toe of a stick blade, conf 0.39, on a frame without a puck. E2b: 11 FPs, mostly stick blades. E1: skates and a fixed spot at the far end | 1 val FP (E2a) | the labelling rule: a puck touching the blade is *not* labelled when unclear (round 2: 6 such proposals rejected). The model is shown dark spots at blades both as "puck" and as "not a puck" |
| 3 | **duplicate detections** | none with merge NMS; 11 without it (part 1) | 0 | overlapping tiles; solved by the merge NMS |
| 4 | **occlusion** | f4590: the puck touches the stick blade, partly covered by it. The 23 null frames of round 2 are all pucks hidden by players, skates or the net | 1 val FN | same rule as type 2. Fully hidden pucks are not errors (they are labelled null) |
| 5 | **motion blur / low quality** | f5070: a streaked puck moving along the boards. f4950: an elongated puck at waist height between two players, probably in the air | 2 val FN | the train set has almost only round, sharp pucks on the ice: round 1 proposed compact blobs on bright ice (`propose_puck.py` filters), so blurred or airborne pucks were rarely proposed and rarely labelled |
| 6 | **class confusion** | not applicable: one class. The closest case is puck vs stick blade (type 2) | – | – |
| 7 | **scale problem** | same frames as type 1: the puck is 6–10 px near the camera and 2–4 px far away | see 1 | a 640 px tile keeps the native size, but a 2–4 px object is less than one cell of the finest feature map (P3, stride 8: slides 43–47) |
| 8 | **crowded scene** | f4950: the puck between two players | 1 val FN (also type 5) | few labelled pucks in crowded spots, for the same reason as type 5 |

f4950 is counted under type 5 and type 8, so the 5 errors give 6 entries. Note that E1 *did*
find f4950 and f4560 (E1 entry), so which pucks are missed also depends on the training run (see
S1: recall is stable at 6/10, the false positives vary).

### What this tells us

- **None of the 5 errors is a threshold or NMS problem** (part 1). All of them trace back to what
  is in the training data:
  - the blade rule decides types 2 and 4;
  - the proposal filters of round 1 (compact, sharp blobs on open ice) explain types 1, 5 and 8.
- This matches the "what usually helps" column of slide 38: better labels, more diverse data, data
  augmentation.

### Hypotheses to test (each one changes one thing)

1. **Blade rule** (types 2 and 4). Decide as a team whether a puck touching the blade is a puck,
   relabel (v4) and retrain with the E2a recipe over 3 seeds (as in S1). Expected: fewer blade
   false positives *and* f4590-like pucks found. Measured by: FP of type background on blades,
   recall on the test set.
2. **Blurred and airborne pucks** (types 5 and 8). Label them on purpose in the next round
   (search frames next to a shot), and/or try blur augmentation in training. Expected: higher
   recall on blurred pucks, without more false positives.
3. **Far pucks** (types 1 and 7). More labels in the far half of the rink. A detection head on a
   finer feature map (stride 4, a P2 head) is the architectural option; it is more work and
   slower, so data first.

Each of these is measured with `eval_puck.py` and this sweep on the model-free test set, not
on the current val set.

## To do (team)

- [ ] Look at `outputs/sweep/E2a_val/errors.jpg` and fill in `category` in `errors.csv`; correct
      the types in the table above where you disagree.
- [ ] Do the same for the false positives of the seed runs (`outputs/eval_puck/seed_*`, on Finn's PC).
- [ ] Add the experiment entry to `docs/experiments.md` when this branch is merged.
