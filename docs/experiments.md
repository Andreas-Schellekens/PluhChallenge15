# Experiment log

One entry per run: config, what we measured, what we concluded. Raw outputs live in
`outputs/` (not in git); rerun with the command listed to reproduce.

## Clip

`Video/hockey_clip_4k_noaudio.mp4`: 3840×2160, 30 fps, 31 s (930 frames). NHL, Wild (green)
vs Blues (white), filmed handheld from the stands, so the framing drifts slightly over time.

## E0 – Pretrained baseline (no fine-tuning), 2026-10-16

**Goal:** measure what off-the-shelf COCO YOLO already does, and confirm the data gap.

Command: `python scripts/baseline.py --video Video/hockey_clip_4k_noaudio.mp4 --imgsz {640,1280} --stride 2`

Model `yolo26s.pt` (COCO), conf 0.25, classes `person` + `sports ball`, every 2nd frame
(465 frames), Quadro T1000.

| run | imgsz | persons / frame | mean person conf | frames with "sports ball" | correct puck hits | inference fps |
|---|---|---|---|---|---|---|
| yolo26s_640  | 640  | 10.7 | 0.58 | 3 | 0 | 6.3 |
| yolo26s_1280 | 1280 | 12.5 | 0.69 | 1 | 1 | 6.6 |

Inference fps is dominated by decoding 4K video, not by the model, so both sizes run at the same speed.

**Observations**
- **Players and referees:** pretrained detection is already usable. 1280 finds about 2 more people
  per frame and is more confident than 640. Spectators behind the glass are mostly not
  detected. Overlapping players (e.g. in front of the net) sometimes merge into one box.
- **Puck, at 640:** all 3 "sports ball" detections are false positives on the small TV screen in
  the stands (around pixel 1200, 600). None of them is the puck.
- **Puck, at 1280:** one correct detection (frame 568), with a box of about 15×13 px in the 4K frame.
  That is 1 hit out of 465 frames.
- **Why:** the puck is about 15 px wide at 4K. YOLO resizes the frame so its long side equals
  `imgsz`, so the puck becomes about 2.5 px at 640 and about 5 px at 1280. After YOLO's
  stride-8 downsampling, that is less than one cell of the finest feature map. COCO also has
  no puck class; "sports ball" is round, coloured and much bigger.

**Conclusions / next steps**
- Confirms the data gap: we need our own `puck` class, from annotation and fine-tuning.
- Input resolution is the first experiment axis for the puck: train and evaluate at 640 vs 1280,
  and try tiling (e.g. SAHI) at full resolution.
- Players can stay on pretrained weights. Use imgsz 1280 for them.
- To check: are the 103 person boxes under 100 px tall at 1280 spectators or real players?

## H1 – Homography and rink minimap, 2026-10-30

**Goal:** map every player's position from the video onto a top-down rink (MVP part 3), and
handle the handheld camera drift.

Command: `python scripts/homography.py --video Video/hockey_clip_4k_noaudio.mp4 --detections outputs/baseline/yolo26s_1280_rtx4060/detections.csv`

Keypoints: `configs/keypoints_hockey_clip_4k.json`. Rink template and coordinate system:
`scripts/rink.py` (feet, origin at centre ice, NHL dimensions). Run on the RTX 4060 laptop;
everything here runs on the CPU (SIFT), 3.6 frames/s including 4K decoding.

**Method**
1. **Reference homography.** In frame 0 we measured 5 rink points by hand in zoomed crops:
   - the 2 end-zone faceoff dots;
   - the 2 neutral-zone dots;
   - the base of the far goal post.

   Their real positions are known from the NHL rink dimensions. `cv2.findHomography`
   (least squares, no RANSAC) fits the 3×3 matrix from image pixels to rink feet.
   4 points is the minimum for 8 unknowns; the 5th gives a residual to check.
2. **Drift.** The person filming stays in their seat, so the camera mostly rotates. Two
   images from a purely rotating camera are related by a homography for the whole image,
   crowd included. So each frame is matched to frame 0 with SIFT features, using a
   ratio test and RANSAC with a 2 px threshold, and then:

   `H_t = H_ref @ H_(frame t -> frame 0)`

   Every frame is matched directly to frame 0, not to the previous frame, so errors do
   not pile up over time. RANSAC throws out matches on moving players. If a frame gets
   fewer than 80 inliers, the last good homography is reused and the frame is flagged.
3. **Players.** Foot point = the bottom centre of each `person` box from E0, because
   only points on the ice plane follow this homography. Feet that land outside the
   boards (with a 2 ft margin) are dropped.

**Results**

| check | result |
|---|---|
| keypoint reprojection error | 0.03–0.12 ft (1–4 cm) per point |
| frame-to-reference matching | 465/465 frames OK; RANSAC inliers min 504, median 722; 0 fallbacks |
| independent check | the projected template lines up with the blue line, goal line, faceoff circles and the centre circle/red line, which were not used for the fit (`reference_check.jpg`); this still holds every 3 s through the clip (`check_sheet.jpg`) |
| persons on the ice | 12.49 of 12.5 detections per frame |

**How much a pixel costs.** The view is oblique, so an error in the foot point weighs more
vertically than horizontally, and more far from the camera. 10 px of foot error:

| position | horizontal | vertical |
|---|---|---|
| near neutral-zone dot | 0.23 ft | 0.61 ft |
| near end-zone dot | 0.27 ft | 0.88 ft |
| far goal post | 0.30 ft | 1.06 ft |

**Observations and limitations**
- A low reprojection error with only 5 points does not prove much: there is just 1 point
  more than the minimum. The overlay of markings that were not used in the fit is the
  stronger evidence.
- **The foot point is the main error source**, not the homography. A box bottom can sit
  on the stick blade or on a lifted skate, which is easily 20–40 px, or 2–4 ft at the far
  end. Merged boxes of overlapping players (E0) give one wrong position instead of two.
- **Assumptions:**
  - no lens distortion is modelled (phone camera, and the straight lines look straight);
  - the camera is assumed to rotate only. If the person filming had moved, the
    frame-to-frame homography would only hold for the ice plane.
- Referees are still drawn as players. Team colours (Extra Mile) would separate the
  teams and the referees.
- The minimap only shows the filmed end zone and part of the neutral zone. The rest of
  the rink is never in view in this clip.

**Conclusions / next steps**
- MVP part 3 works for this clip. The homography is not the weak point; foot-point
  quality is.
- The outputs `homographies.csv` and `positions.csv` can be read directly by the
  Streamlit dashboard.
- For a new clip, only the keypoints config needs to be redone.

---

# Clip 2: `fillempje.mp4` (main clip from 2026-10-30)

**Why we switched.** The 4K clip was handheld, so the camera moved, and it showed only 12
frames with a findable puck. The new clip matches the assignment's "single static camera"
much better.

`Video/fillempje.mp4`: 1920×1080, 59.94 fps, about 85 s of decodable video (the metadata
says 90 s). Amateur ice hockey, white vs blue, filmed with an action camera (wide-angle
lens) mounted **behind the near goal**, looking down the rink. The camera is static, except
for a bump in the first second. The puck is about 6–10 px near the camera and only a few
pixels far away.

## E0-B – Pretrained baseline on clip 2, 2026-10-30

Command: `python scripts/baseline.py --video Video/fillempje.mp4 --imgsz 1280 --stride 2 --name fillempje_yolo26s_1280`
(yolo26s, conf 0.25, RTX 4060 Laptop, 2548 frames, 16.4 fps).

| persons / frame | mean person conf | median person box height | frames with "sports ball" |
|---|---|---|---|
| 13.1 | 0.72 | 75 px | 10 / 2548 (0.4%) |

**Observations**
- **Players:** detected well, even far away. The count includes spectators, the bench and
  people behind the glass. The homography removes those, because their feet land outside
  the rink (H2).
- **Puck:** checked visually, the "sports ball" hits are mostly the real puck. Examples:
  f1098, close to the camera, and f1456–1472, along the boards. So the baseline has **high
  precision but almost no recall**. A puck moving along at 6–10 px is usually missed.

**Conclusion:** the data gap is the same as on clip 1: recall on the puck. That is what the
fine-tuned model (E1) has to show.

## H2 – Homography with lens correction on clip 2, 2026-10-30

> **Superseded by H3 (2026-11-11):** the 4 keypoints of this entry are all next to the camera,
> and the mapping they give is wrong in the far half of the rink.

Command: `python scripts/homography.py --video Video/fillempje.mp4 --keypoints configs/keypoints_fillempje.json --detections outputs/baseline/fillempje_yolo26s_1280/detections.csv --drift none`

**1. Lens correction (plumb-line).** A homography maps a plane to a plane and keeps straight
lines straight, but the action camera bends them. We use a one-parameter radial model:

`x_dist = x · (1 + k1·r²)`, with *x* and *r* in units of *f* = 1000 px from the image centre.

*k1* is chosen so that the yellow kick plates along both side boards, which are straight in
reality, become straight lines again. The yellow is found with an HSV threshold in the median
background. The score is the mean distance of the yellow pixels to their best-fit line:

| k1 | 0 | −0.05 | −0.10 | **−0.125** | −0.15 |
|---|---|---|---|---|---|
| deviation from a straight line (px) | 1.18 | 0.97 | 0.54 | **0.34** | 0.87 |

The ice region is only mildly distorted; the strong fisheye is mostly in the roof, which we
don't need.

**2. Keypoints.** Measured on the **median background** (outputs/propose_puck): the median
over all frames is the empty rink without players, and because the camera is static it is
valid for every frame. 4 points:
- the 2 end-zone faceoff dots, at (−20, ±7) m;
- the 2 corners of the crease on the goal line, at (−26, ±1.83) m.

The goal posts were not usable: the camera looks through the back of the net. The rink is
assumed to follow IIHF dimensions (60 × 30 m), see `scripts/rink.py`.

**3. Sign fix.** A homography is only defined up to scale, and here the fitted scale was
negative. Then every point had w < 0, and the "behind the camera" check (w ≤ 0) dropped the
whole template. We now normalise the sign so that w > 0 for the keypoints. (On clip 1 the
sign happened to come out positive.)

**Results**

| check | result |
|---|---|
| keypoint reprojection error | 0.00 m: **not meaningful**, because 4 points = exactly the 8 unknowns |
| independent check (`reference_check.jpg`, app tab "Homography") | goal line matches the faint red line on both sides of the net; left faceoff circle matches almost exactly, right one slightly too wide at its outer edge; side boards fall between the 14 and 15 m lines, so the rink is about 28–30 m wide, consistent with IIHF |
| known deviation | the crease is drawn about 50 px too deep: depth right next to the goal line is slightly overestimated |
| persons on ice | 11.6 of 13.1 detections per frame; the other 1.5 (spectators, bench) are dropped |

**Error analysis**
- **Distance:** the camera is low and close to the goal. Far away, one pixel of foot-point
  error covers metres on the rink, so positions beyond the blue line are only rough.
- **Truncated boxes:** players close to the camera are cut off by the bottom edge of the
  image. The "foot point" is then their knees, and they land metres too close to the middle
  (e.g. #72 in f2820).
- **More keypoints would help.** With only 4 there is no redundancy. Candidates: hash marks
  on the circles, and the blue line once a frame shows it clearly.

**App:** `streamlit run app.py`. All parameters are adjustable in the sidebar: person model
size and confidence, puck detector (fine-tuned / baseline / off) with confidence, tile size
and overlap, lens k1, camera motion (static / SIFT) and the template overlay.

## E1 – Fine-tuned puck detector on clip 2, 2026-10-30

**Data:** Roboflow version 2, 20 human-reviewed frames with 1 puck each (see
`docs/annotation.md`). Split by time: train = 16 frames (f330–f4080), val = 4 frames
(f4200, f4560, f4950, f5070).

Commands (RTX 4060 Laptop):
```
python scripts/make_puck_dataset.py --export data/roboflow/pluhchallenge_v2 --val-from-frame 4200 --name fillempje_tiles640
python scripts/train_puck.py --data data/puck_tiles/fillempje_tiles640/data.yaml --name fillempje_yolo26s_puck_640
python scripts/eval_puck.py --finetuned outputs/train/fillempje_yolo26s_puck_640/weights/best.pt --frames data/puck_tiles/fillempje_tiles640/frames.csv --video Video/fillempje.mp4 --name fillempje
```

**Config**

| setting | value | why |
|---|---|---|
| start weights | `yolo26s.pt` (COCO) | same model as the baseline, so the difference is the fine-tuning |
| input | 640×640 tiles cut from the 1080p frame | the puck keeps its native 6–10 px; resizing the full frame would shrink it further |
| train tiles | 8 positive crops (puck at a random position) + 16 negative crops per frame → 128 + 256 | negatives teach the look-alikes (skates, tape, ads); there is only one puck, so the rest of a labelled frame is a guaranteed negative |
| val tiles | full grid, overlap 160 px → 32 tiles, 11 with a puck | measures the real inference situation |
| augmentation | mosaic 1.0, scale ±0.2, horizontal flip 0.5; no vertical flip, no rotation | the camera is fixed and upright; a large scale change would push the puck below the detectable size |
| training | 100 epochs, batch 16, imgsz 640, seed 0; `best.pt` = epoch 32 (highest fitness on val) | |

**Result 1: tile-level validation (Ultralytics, `best.pt`)**

| epoch | precision | recall | mAP@50 | mAP@50:95 | train box loss | val box loss |
|---|---|---|---|---|---|---|
| 10 | 0.21 | 0.55 | 0.35 | 0.15 | 2.07 | 2.36 |
| 20 | 0.78 | 0.45 | 0.56 | 0.28 | 1.58 | 1.81 |
| **32 (best)** | **0.99** | **0.64** | **0.789** | **0.506** | 1.54 | 1.75 |
| 50 | 0.54 | 0.64 | 0.52 | 0.32 | 1.01 | 1.73 |
| 100 | 0.76 | 0.29 | 0.41 | 0.29 | 0.50 | 1.75 |

**Overfitting** after epoch ~30: train loss keeps dropping (1.54 → 0.50) while val loss stays
flat and val mAP falls. With 16 training pucks the model starts memorising them. Early
stopping (`patience`) or fewer epochs would give the same `best.pt` faster; more data is the
real fix.

**Result 2: full-frame comparison against the baseline (`eval_puck.py`)**

Both models on the 4 full val frames, conf ≥ 0.25, a hit = IoU ≥ 0.5 with the label.
Baseline = COCO "sports ball" on the whole frame at imgsz 1280; fine-tuned = tiled inference
(640 tiles, overlap 160, NMS to merge).

| model | TP | FP | FN | precision | recall | AP@50 |
|---|---|---|---|---|---|---|
| baseline (E0-B) | 0 | 0 | 4 | – | 0.00 | 0.00 |
| **fine-tuned (E1)** | **2** | **0** | 2 | 1.00 | **0.50** | **0.75** |

- `confusion_matrix.png`: the 2×2 matrices (rows = predicted, columns = true). The
  background/background cell is empty: in detection there is no countable "true negative".
- `comparison.jpg`: per val frame, the zoomed crops of both models next to the full frame.
  Found: f4560, f4950 (open ice). Missed: f4200 (far end, a few pixels) and f5070 (against the
  boards, next to a player).

**Result 3: whole clip (no labels, 2548 frames, every 2nd frame)**

| model | frames with a puck detection |
|---|---|
| baseline | 10 (0.4%) |
| fine-tuned | 753 (30%) |

`timeline.png` shows where in the clip. Without labels this says nothing about correctness, so
we checked a random sample of 30 detected frames visually (`clip_sample.jpg`, check by Claude
Code, not yet by a team member):

| verdict | frames | examples |
|---|---|---|
| puck (plausible) | 23 | at a stick blade (f262–f276, f2974–f2984, f4854), open ice (f2948, f4944), behind the net (f3870) |
| false positive: skate | 4 | f1324, f1524, f3484, f4506 |
| false positive: static dark spot at the far end | 2 | f62, f158 at (1040, 405): the same spot the reviewer rejected in f060/f090/f150 |
| unclear | 1 | f798 |

So roughly **75% of the clip detections look right**, against roughly 0.4% coverage for the
baseline. Recall over the whole clip is unknown, because the puck is often hidden.

**Error analysis, linked to annotation choices**
1. **Skates are the main false positive.** The proposal step (`propose_puck.py`) removed every
   candidate in the skate zone of a player, so the train set has *no* puck near skates and
   few hard negatives of skates. The model never had to learn the difference. Fix: label
   pucks at players' feet by hand, and add skate crops as explicit negatives.
2. **The static spot at the far end** fooled both Claude's proposals (rejected f060/f090/f150)
   and now the model. Rejected proposals were dropped, not kept as negatives. Fix: keep
   rejected frames in the dataset as images without a puck label at that spot.
3. **Far pucks (f4200) are missed.** At the far end the puck is 2–4 px; the train set has
   mostly near/mid-ice pucks, because those were the ones the proposals found. Fix: more
   labels in the far half of the rink.
4. **Pucks against the boards (f5070) are missed.** Boards were under-represented: 2 of the
   rejected "confident" proposals were along the boards (f1470, f4110).
5. **The val set is tiny (4 pucks).** One extra hit changes recall by 25 points, and the
   tile-level numbers swing between epochs (see the table). The numbers show the direction,
   not a precise value.

**Conclusion:** fine-tuning on 20 reviewed frames turns the puck from almost never detected
(baseline recall 0/4, 10 clip frames) into detected in half of the val frames and 30% of the
clip, with no false positive on val and roughly 1 in 4 false positives on the clip. The
failure modes point directly at the next labelling round: pucks at skates, at the boards and
far away, plus rejected look-alikes as negatives, to reach 40–80 frames.

## E2 – More data (round 2) and hard negatives, 2026-10-31

**Data:** Roboflow version 3 (tag `fillempje`, no preprocessing or augmentation): 64 images,
of which 41 have a puck and 23 are verified null. See `docs/annotation.md`, round 2.
- Split by time, as before (≥ f4200 = val):
  - train: 39 frames, 31 with a puck;
  - val: 25 frames, 10 with a puck and 15 null.
- The val set is now 2.5× larger, and it can also count false positives on frames that have no puck.

Two runs, which differ **only** in the training data (same model, epochs, augmentation and
seed as E1):

| run | dataset | train tiles (with / without puck) |
|---|---|---|
| E2a | `fillempje_v3_tiles640` | 248 / 624 |
| E2b | `fillempje_v3hn_tiles640`: + hard negatives (`--skate-crops 4 --lookalike 56 400`) | 248 / 747 |

Commands (RTX 4060 Laptop, about 25 min per training run):
```
python scripts/make_puck_dataset.py --export data/roboflow/pluhchallenge_v3 --val-from-frame 4200 --name fillempje_v3_tiles640
python scripts/make_puck_dataset.py --export data/roboflow/pluhchallenge_v3 --val-from-frame 4200 --detections outputs/baseline/fillempje_yolo26s_1280/detections.csv --skate-crops 4 --lookalike 56 400 --name fillempje_v3hn_tiles640
python scripts/train_puck.py --data data/puck_tiles/fillempje_v3_tiles640/data.yaml --name fillempje_v3_yolo26s_puck_640
python scripts/train_puck.py --data data/puck_tiles/fillempje_v3hn_tiles640/data.yaml --name fillempje_v3hn_yolo26s_puck_640
python scripts/eval_puck.py --finetuned outputs/train/<run>/weights/best.pt --frames data/puck_tiles/fillempje_v3_tiles640/frames.csv --video Video/fillempje.mp4 --name v3val_<E1|E2a|E2b>
```
E1 was scored again on the new val set, so all models are compared on the same 25 frames.

**Results**

| model | tile val mAP@50 | tile val mAP@50:95 | full frames: TP / FP / FN | precision | recall | AP@50 | clip frames with a puck | clip sample: plausible puck |
|---|---|---|---|---|---|---|---|---|
| baseline (COCO sports ball) | – | – | 0 / 0 / 10 | – | 0.00 | 0.01 | 10 | – |
| E1 (20 frames) | – | – | 5 / 2 / 5 | 0.71 | 0.50 | 0.58 | 753 | ~23 / 30 |
| **E2a (41 frames)** | **0.634** | **0.382** | **6 / 1 / 4** | **0.86** | 0.60 | **0.65** | 455 | **~28 / 30** |
| E2b (41 frames + hard negatives) | 0.460 | 0.269 | 7 / 11 / 3 | 0.39 | 0.70 | 0.55 | 1313 | ~19 / 30 |

- The tile val numbers for E1 (0.789 / 0.506) were measured on the old val set of 4 frames,
  so they cannot be compared with this column.
- The clip samples (`clip_sample.jpg` in each eval folder) are 30 random detected frames,
  checked by Claude Code, not by a team member.

**Observations**
1. **More data helps (E1 → E2a).** On the same val frames E2a finds one more puck and makes
   one fewer false alarm, and AP@50 rises from 0.58 to 0.65.
   - The clip shows the bigger difference. E2a reports a puck in fewer frames (455 vs 753), but
     almost all of them look right: about 28 of 30, against about 23 of 30 for E1.
   - The skate false positives of E1 are gone from the sample. E2a is more careful and much more
     precise.
2. **The hard negatives did not help (E2a → E2b).** E2b finds 1 more puck, but it gives
   11 false alarms on val instead of 1. `val_false_positives.jpg` shows that they are
   almost all **stick blades** (and a glove). In 6 of the 11 cases the frame was marked null,
   so the reviewer saw no puck there. *(Corrected 2026-11-11: this said 7; `eval_E2b.log`
   shows 6 on null frames and 5 on frames with a puck.)*
   - The clip says the same: more detections, lower precision. The far-end spot from round 1 is
     back too (f86–f152).
   - Possible causes, not proven:
     - (a) The skate crops show many scenes around players' feet without a puck label. Sticks
       and blades are near the feet, but they don't look like skates, so the extra negatives
       teach little about blades.
     - (b) Chance. There is only one seed, `best.pt` is picked on 10 val pucks (epoch 19 for
       E2a, 37 for E2b), and the val curves swing by ±0.15 mAP between epochs.

     Settling it would take repeated runs with different seeds.
3. **Annotation choice → failure mode (round 2).** The reviewer rejected pucks *touching* a stick blade
   (f360, f390, f630, f1230, f3810, f3870) and, in f4590, moved the box away from the
   blade. So "puck at the blade" is labelled as "not a puck" when it is unclear.
   - The model has to learn that boundary from very few examples. Stick blades are now the main
     source of false positives.
   - A rule decided as a team ("a puck touching the blade counts if ≥ 50% is visible") would make
     these labels consistent.
4. **Overfitting again.** Both runs reach their best val score before epoch 40, while the
   train loss keeps falling (E2a box loss: 1.65 at epoch 20 → 0.52 at epoch 100).

> **Read this entry together with A1 and S1 (2026-11-11).** The audit found that E2a and E2b
> differ in more than the hard negatives (almost all random crops were different too), and
> that the val set favours E1. S1 repeats the comparison cleanly with three seeds.
> The tile val numbers in the table are from the re-validation of `best.pt` (`metrics.json`);
> the training curve (`results.csv`) reads 0.617 / 0.336 for E2a at its best epoch, because
> Ultralytics validates with slightly different settings during training.

**Conclusion:** **E2a is the best puck model so far**, and the one the app should use.
Doubling the reviewed frames raised precision and AP@50. Our version of hard negatives did not
help: it gave more false positives, mainly on stick blades. The next steps for data are a
team-wide labelling rule for pucks at the blade, and blades as explicit hard negatives. A
repeat with 2–3 seeds would tell whether E2a vs E2b is a real difference.

## T1 – Tracking (Extra Mile): ByteTrack for players, Kalman filter for the puck, 2026-11-05

> **The numbers in this entry are superseded by T2 (2026-11-11).** They were measured with the
> 4-point homography, which was wrong in the far half of the rink (H3), and the gap test
> flattered the filter (A1). The method described here is still what the code does.

**Why.** Until now every frame was processed on its own. Players had no identity over time,
and the puck (detected in only ~18% of the frames, E2a) blinked on the minimap or jumped to a
false detection.

**Method**
- **Players: ByteTrack** (Ultralytics `model.track`, `persist=True`).
  - Each track is predicted with a Kalman filter in the image, and new boxes are matched to the
    predictions by IoU.
  - The "byte" idea: low-confidence boxes (≥ 0.1) are used in a second matching round, so a
    partly hidden player keeps their track. That is why the person confidence is set low (0.1)
    for tracking.
- **Puck: our own Kalman filter** (`scripts/kalman.py`), running on the rink in metres after
  the homography.
  - State: position and velocity, with a constant-velocity model.
  - Process noise: white-noise acceleration, σ_a = 20 m/s². Measurement noise: σ = 0.7 m.
  - Gating: Mahalanobis d² < 9.21 (χ², 2 degrees of freedom, 99%), then nearest neighbour inside
    the gate.
  - At most 0.5 s of prediction without a detection. A new track needs a detection with conf ≥ 0.4.
  - Puck candidates outside the rink are dropped first.
- **Gap test.** Detections that the full run used are hidden in blocks of k frames. We then
  compare the filter's position with the hidden detection. Baseline: keep the puck at its
  last seen position.

Commands (RTX 4060 Laptop, every 2nd frame, about 5 min per run):
```
python scripts/track.py --video Video/fillempje.mp4 --keypoints configs/keypoints_fillempje.json --puck-weights outputs/train/fillempje_v3_yolo26s_puck_640/weights/best.pt --name fillempje_E2a
python scripts/track.py ... --tracker configs/bytetrack_hockey.yaml --name fillempje_E2a_hockeytrack
```

**Results: puck** (2548 processed frames; the two runs give the same puck numbers)

| | before (detections only) | after (Kalman) |
|---|---|---|
| frames with a puck on the minimap | 17.2% (on the ice) | **39.6%** (13.8% measured + 25.8% predicted) |
| impossible jumps (> 40 m/s between two processed frames) | 26 | **13** (left over: new tracks starting elsewhere) |
| frames where the gate refused a detection | – | 64 |

Gap test (median error against the hidden detection):

| gap | puck still shown | Kalman error | "last position" error |
|---|---|---|---|
| 0.10 s | 92% | **0.10 m** | 0.44 m |
| 0.27 s | 84% | **0.30 m** | 1.03 m |
| 0.50 s | 64% | **0.78 m** | 1.21 m |

**Results: players** (on-ice tracks only)

| tracker settings | unique IDs | IDs longer than 5 s | median track life |
|---|---|---|---|
| Ultralytics default (`track_buffer` 30, `new_track_thresh` 0.25, `match_thresh` 0.8) | 370 | 77 | 1.45 s |
| **`configs/bytetrack_hockey.yaml`** (buffer 90 = 3 s, new track ≥ 0.4, match 0.9) | **221** | 82 | **3.94 s** |

There are about 12 people on the ice at any moment, so the ideal is around 12–20 IDs over
the whole clip.

**Observations**
1. **The puck filter does what it should.** Puck coverage on the minimap more than doubles. For
   short gaps, the prediction is 3–4× more accurate than freezing the puck.
   - The gate removes half of the impossible jumps.
   - At 0.5 s the advantage shrinks (0.78 vs 1.21 m): a puck changes direction (pass, rebound,
     boards) while it is hidden, and a constant-velocity model cannot know that.
2. **Player tracking is still fragmented.** Even tuned, there are 221 IDs instead of about 15.
   - Causes:
     - players cross and hide each other for longer than 3 s at the boards and in front of the net;
     - far-away players are small boxes, so their IoU drops quickly when they move;
     - the team kits look the same, so matching on position alone cannot tell players apart after
       a long occlusion. ByteTrack uses no appearance; BoT-SORT with ReID would, but within a team
       it would not help much either.
   - The ID count is a proxy. Without hand-labelled IDs we cannot count ID switches exactly.
3. **Limitation of the gap test.** The "truth" is our own detection, which carries
   homography noise of about 0.5–1 m far away. So errors below about 0.3 m say little.

**Conclusion:** the Kalman filter fills most puck gaps up to about 0.3 s accurately and removes
false jumps. For the minimap it is the most visible improvement since E2a. Player IDs became
2.7× more stable after tuning, but identity is not kept over long occlusions; that would need
appearance features (team colour plus jersey number). The app's Clip tab now tracks by default
(sidebar: Kalman settings), and the Tracking tab shows these results with the before/after
minimap video.

---

# Review round, 2026-11-11

Everything from here on was run on an RTX 2080 Ti (11 GB), from a fresh clone with the team's
`data/` and `outputs/`. First check: `best.pt` of E2a validated on this machine gives
mAP@50 0.635 / mAP@50:95 0.382, the same as logged (0.634 / 0.382).

## A1 – Audit of the code and the log, 2026-11-11

**Goal:** before adding anything, check that what we already claim is right. A second Claude
Code agent read every script and compared every number in the docs with the saved outputs
(read-only; the main session verified the findings it acted on).

**Confirmed correct:** dataset counts, the time split (train ends at f4170, val starts at f4200,
no frame or tile in both), label conversion, the IoU matching and AP in `eval_puck.py`, the rink
dimensions, the Kalman equations, and all E0-B / H1 / E1 / E2 / T1 summary numbers against their
`metrics.json` and `summary.json`.

**What was wrong, and what we did about it**

| # | finding | consequence | action |
|---|---|---|---|
| 1 | The 4-point homography (H2) only has keypoints at x = −26…−20 m and is extrapolated 50 m. The far goal landed at x ≈ 100 m instead of 26 m | about 0.7 players per frame at the far end were dropped as "off the ice"; H2 called them spectators | **H3**: two more keypoints, fit in pixels |
| 2 | The gap test in T1 scored the Kalman filter and "hold last position" on different frames, and compared the filter with its own filtered output | the filter looked 3–4× better than holding; it is less | **T2**: same frames, raw hidden detection as reference |
| 3 | E2a and E2b did not differ "only" in the hard negatives: one random stream drew all crops, so the hard-negative draws shifted every later crop. 134 of E2a's 872 train tiles are in E2b | "hard negatives did not help" was confounded with a full re-draw of the data | `make_puck_dataset.py` now uses a separate random stream for hard negatives; **S1** repeats the comparison |
| 4 | The val set favours E1: 15 of the 25 val frames are null frames picked *because E1 found nothing there*, and 5 of the 6 val pucks added in round 2 were E1's own proposals | E1 has 0 false positives on those frames by construction. On the 5 val pucks that no model proposed: E1 2/5, E2a 2/5, E2b 3/5 | documented here and in `annotation.md`; S1 reports the "independent" pucks separately |
| 5 | `best.pt` is chosen on the val set and reported on the same val set; there is no test set. The look-alike spot (56, 400) was chosen from E1's false positives, one of them on a val frame | all tile val mAP values are optimistic | stated as a limit; a model-free test set is the first data to-do |
| 6 | Lens model: with k1 < 0 the distorted radius folds back beyond r = 1.63 focal lengths, so template points far outside the view were drawn *inside* the image (the grey X across the ice in `reference_check.jpg`) | wrong lines in the check image, positions were not affected | `Lens.valid()` masks those points |
| 7 | Text errors: "7 of the 11" false positives on null frames is 6; the 13 leftover puck jumps were not all "new tracks" (6 were, 7 were accepted detections) | – | corrected in place / in T2 |
| 8 | App: on a failed SIFT match the app silently used the identity; the puck selector defaulted to the alphabetically first run; `requirements.txt` allowed a Streamlit too old for the app; no way to add a clip from the app | – | fixed (`app.py`, `requirements.txt`): last good homography + warning, default E2a, `streamlit>=1.50`, upload in the sidebar |

**Numbers in this log that no saved output can confirm** (kept, but treat them as unverified):
the `yolo26s_640` row and the Quadro fps in E0, "3.6 frames/s" in H1, the k1 plumb-line table in
H2 (no calibration script in the repo), and the clip samples of E1/E2 (23/30, 28/30, 19/30), which
are visual checks by Claude Code that no team member has confirmed.

## H3 – Homography with 6 keypoints, fitted in pixels, 2026-11-11

**Why.** H2 had exactly 4 points, so its reprojection error was 0 by construction and said
nothing. The independent check we had not done: where does the **far goal** land? Its posts
stand on the ice at (26, ±0.915) m. With the H2 mapping their measured pixels, (919, 400.5) and
(949.5, 400.5), land at about x = 113 m and x = 95 m. The template drew the far goal 19 px too low.

Command: `python scripts/homography.py --video Video/fillempje.mp4 --keypoints configs/keypoints_fillempje.json --detections outputs/baseline/fillempje_yolo26s_1280/detections.csv --drift none --name fillempje_6pt`

**Change 1: two more keypoints.** The bases of the far goal posts, measured on the median
background in a 5× zoom. Now there are 6 points, 2 more than the minimum, and the points span
the whole length of the rink instead of the nearest 6 m.

**Change 2: fit in pixels, not in metres.** `cv2.findHomography(src, dst)` minimises the error
in `dst` units. We used image → rink, so the error was in metres. Far from the camera one pixel
is several metres, so the two far posts outweighed everything and the near points were
sacrificed:

| fit direction | error that is minimised | near dots / crease corners | far posts |
|---|---|---|---|
| image → rink (as before) | metres | 37 / 13 / 75 / 31 px off | 2 / 1 px |
| **rink → image (now)** | **pixels** | **7.6 / 5.1 / 5.1 / 2.8 px** | **5.0 / 4.2 px** |

We measured in pixels, so pixels are where the error should be spread evenly. The matrix is
inverted afterwards to go from image to rink. With exactly 4 points both directions give the
same exact solution, which is why this never showed before.

**Results**

| check | H2 (4 points) | H3 (6 points) |
|---|---|---|
| reprojection error | 0 (meaningless) | 2.8–7.6 px per point |
| the same error in metres | 0 | near points 0.02–0.54 m, far posts 6.3 and 9.6 m |
| persons on the ice per frame (of 13.13 detected, margin 2 m) | 11.64 | **12.33** |
| `reference_check.jpg` | far goal line 19 px below the real goal | far goal line ends at the far goal; near goal line and both circles still match |

**Observations**
1. **A residual of about 5 px on every point means our three assumptions do not fit together
   exactly:** IIHF dimensions for a club rink, a lens with one distortion parameter, and
   hand-measured points on faint markings. We tested whether the lens explains it: a stronger
   k1 lowers the residual (−0.3 gives 2.7 px) but bends the boards again (H2: −0.125 is the
   straightest). We also tried other rink lengths (far goal at 20–28 m): the residual changes
   by about 1 px (5.2 to 6.2 px), so this camera cannot tell how long the rink is.
2. **The far end stays coarse, and now we can say how coarse.** 5 px at the far goal is 6–10 m
   along the rink. The camera is low and at one end: the far half of the rink (30 m) covers
   about 30 image rows. No fit can repair that; only a second or higher camera could.
3. **Why H3 is still clearly better than H2.** H2 was exact on 4 points and 70–90 m wrong at the
   far goal. H3 is a few pixels off everywhere and keeps the far players on the rink.
4. The planar assumption is unchanged: only points **on the ice** follow the homography. That
   is why we use the base of the posts and the feet of the players, never a box centre.

**Conclusion:** the minimap now makes spatial sense over the whole rink instead of the near
zone only. Positions next to the camera are good to a few decimetres; beyond the centre line
they are an indication (several metres), and the log says so wherever they are used.

## T2 – Tracking re-measured: fair gap test, new homography, rink limit, 2026-11-11

Commands:
```
python scripts/track.py --video Video/fillempje.mp4 --keypoints configs/keypoints_fillempje.json --puck-weights outputs/train/fillempje_v3_yolo26s_puck_640/weights/best.pt --tracker configs/bytetrack_hockey.yaml --name fillempje_teams --redo
```
(E2a puck model, tuned ByteTrack config, every 2nd frame, 2548 frames. `players.csv` now also
stores the person boxes, for TM1.)

**What changed against T1**
1. **Homography H3.** Detections at the far end are now on the ice.
2. **Fair gap test** (A1, finding 2). Both methods are scored on the same frames (hidden frames
   where the filter still shows a puck), against the raw hidden detection.
3. **Rink limit.** While the filter predicts without a detection, a prediction that leaves the
   rink ends the track: the puck cannot be behind the boards.

**Results: puck**

| | T1 (as logged) | T2 |
|---|---|---|
| frames with a puck detection on the ice | 17.2% | 17.9% |
| frames with a puck on the minimap after the filter | 39.6% | **41.6%** (14.3% measured + 27.3% predicted) |
| detections refused by the gate (frames) | 64 | 66 |
| jumps > 40 m/s, raw detections | 26 | 26 |
| jumps > 40 m/s, after the filter | 13 | 24, of which 16 inside one track and 8 new tracks starting elsewhere |

Gap test, T2 (median error in metres against the hidden detection):

| hidden block | median time since the puck was last seen | puck still shown | Kalman | hold last position | ratio |
|---|---|---|---|---|---|
| 0.10 s | 0.07 s | 93% | **0.15 m** | 0.40 m | 2.7× |
| 0.27 s | 0.17 s | 73% | **0.43 m** | 0.65 m | 1.5× |
| 0.50 s | 0.27 s | 63% | **0.81 m** | 0.97 m | 1.2× |

T1 reported 0.10 / 0.30 / 0.78 m against 0.44 / 1.03 / 1.21 m, "3–4× more accurate". The honest
number is 2.7× for very short gaps and almost nothing at half a second.

**Results: players** (tuned ByteTrack config, on-ice tracks)

| | T1 | T2 |
|---|---|---|
| detections on the ice per frame (margin 1 m) | 11.08 | 11.85 |
| unique IDs | 221 | 198 |
| IDs longer than 5 s | 82 | 90 |
| median track life | 3.94 s | 4.52 s |

The tracker itself is unchanged. The numbers move because far-end players are no longer cut
out of their tracks by the on-ice filter.

**Error analysis: why the jump count went up**
- We listed the within-track jumps before the rink limit was added (22 with the default
  settings). **15 of them were one event:** at f660 the filter had been predicting for a while, its uncertainty had grown, and
  it accepted a detection 22.8 m away. That one update gave the state a velocity of about
  40 m/s, and for the next 0.5 s the filter "predicted" the puck straight through the boards
  to x = −37 m. Every step of that flight counts as a jump.
- The rink limit removes exactly this failure (30 → 24 jumps). It is a constraint from the
  sport, not a tuning knob.
- The rest are single accepted detections 2–12 m from the prediction after a period of
  coasting: the gate is wide at that moment because the velocity is uncertain. A stricter gate
  would refuse them, and would also refuse real passes. We left it.
- T1 scored 13 because the same f660 event happened not to occur with the old mapping. The
  count is fragile: one bad update moves it by 14. As a metric, "share of frames with a
  plausible puck" would be more robust, but it needs labels.

**Tested and not kept: measurement noise that grows with distance.** Idea: one pixel of
detection noise is centimetres near the camera and metres far away, so each detection gets its
own covariance R = J·Jᵀ·σ²ₚₓ (J = Jacobian of image → rink at that pixel,
`RinkMapper.rink_noise`). With σₚₓ = 2: jumps 24 → 23, gap test unchanged (0.17 / 0.43 / 0.79 m).
Only 19 of the 489 puck detections are in the far half, so there is almost nothing for it to
act on. The option stays in `track.py` (`--puck-px-std`, default 0 = off).

**Conclusion:** the Kalman filter still more than doubles how often the puck is on the minimap
(17.9% → 41.6%) and is 2.7× closer than a frozen puck during very short gaps. Its advantage
over half a second is small, and it does not reduce impossible jumps by half as T1 claimed.

## TM1 – Team colours (second Extra Mile): K-means on jerseys, occupancy heatmaps, 2026-11-11

**Why.** Every person on the ice was the same orange dot; the log listed "referees are drawn as
players" as open since H1. The brief names this as an extension: team colour clustering and
occupancy heatmaps (KDE).

Commands:
```
python scripts/teams.py --video Video/fillempje.mp4 --players outputs/track/fillempje_teams/players.csv --keypoints configs/keypoints_fillempje.json --names white other blue --score --name fillempje
```

**Method** (`scripts/teams.py`, no labels used for fitting)
1. **Jersey crop:** the middle half of the box width, from 20% to 50% of its height (below the
   helmet, above the pants).
2. **Feature:** three numbers per person in CIELAB: the 75th percentile of L (lightness) and
   the mean of a and b (colour). Lab instead of HSV because hue is undefined for white, grey
   and black, which is most of what we have to separate.
3. **Standardise**, then **K-means with k = 3** (k-means++ start, 10 restarts, fitted on the
   28 546 on-ice boxes of at least 40 px high).
4. **Track vote:** each detection takes the majority colour of the last 30 detections (about
   1 s) of its ByteTrack ID.
5. **Heatmaps:** per group, a 2D histogram of rink positions blurred with a Gaussian of
   σ = 1.5 m. That is a kernel density estimate on a grid.

**Iteration 1 failed, and why.** First version: the median Lab colour, no scaling.

| | iteration 1 | iteration 2 (kept) |
|---|---|---|
| feature | median L, a, b | 75th percentile of L, mean a, mean b, each divided by its standard deviation |
| the 3 cluster centres (L, a, b) | (109, 127, 125), (74, 126, 126), (43, 129, 120) | (121, 127, 125), (67, 124, 133), (67, 131, 114) |
| what the clusters are | light / medium / dark: **three levels of brightness** | white team / people in dark clothes at the boards / blue team |
| silhouette for k = 2, 3, 4, 5 | 0.61, 0.49, 0.46, 0.42 (k = 2 "best") | 0.46, **0.49**, 0.37, 0.34 (k = 3 best) |
| detections that agree with the majority of their own track | 81.0% | 85.8% |

- **Cause 1: K-means uses plain Euclidean distance.** L ranged over about 150 units and a/b over
  about 30, so lightness decided everything. The blue team and the dark-clothed people differ
  mainly in b (114 against 133), which hardly counted. Dividing each feature by its standard
  deviation (33.0, 3.2, 7.8) fixes that.
- **Cause 2: the white jerseys have a large dark crest.** The median of the crop often landed on
  the crest, so white players fell into the "medium" cluster. The 75th percentile of L looks
  at the lighter part of the crop, which is the jersey itself.
- In this dim rink a white jersey is grey (L = 121 of 255) and the blue one is a dark navy.
  Cluster names cannot be read off the centre colour, so we name the clusters ourselves after
  looking at the crop sheets (`--names`, lightest first).

**Results** (30 192 on-ice detections)

| group | per frame | what the crop sheet shows |
|---|---|---|
| white | 5.7 | white team, including their goalie in the near goal |
| blue | 3.3 | blue team |
| other | 2.8 | people in camouflage or black jackets standing on the ice at both side boards |

**Accuracy on a labelled sample.** 120 random on-ice detections (`sample.jpg`, `sample.csv`),
labelled from the crops **by Claude Code, not yet confirmed by a team member**; 10 were
unclear (blurred, two people in one box) and left out.

| | correct of 110 | accuracy | colour flips per 100 detections |
|---|---|---|---|
| per detection | 101 | 91.8% | 2.78 |
| vote over the whole track | 98 | 89.1% | – |
| **vote over the last 30 detections (kept)** | **105** | **95.5%** | **0.46** |

Confusion after the window vote (rows = truth): blue 38 / 2 other / 1 white; other 24 / 1 blue /
1 white; white 43 / 0 / 0.

**Observations**
1. **The whole-track vote made it worse (91.8% → 89.1%).** It assumes one track is one person.
   T2 shows that is false: 198 IDs for about 14 people, and when two players cross ByteTrack
   can hand the ID to the other one. A blue and a white player then share one "majority". The
   error analysis of the tracker predicted this; here it is measured.
2. **A short window keeps the benefit and limits the damage:** 4 more correct, and 6× fewer
   colour flips on the minimap. The window length (30) was chosen on the same 110 crops, and
   5 to 150 detections all score 103–105, so the exact value is not established.
3. **Main confusion: blue ↔ dark.** A blue jersey in the shadow near the boards is nearly black,
   and a dark jacket is nearly navy. 4 of the 9 per-detection errors were blue players read as
   white; in the crops we looked at, a white player stands partly in front, so the crop holds
   both.
4. **"Other" is not a clean referee class.** The heatmap puts this group at two fixed spots on
   both side boards near the blue line. They stand inside the 1 m margin we allow around the
   boards, so they count as "on the ice". It separates them from the teams, which is what the
   minimap needs; whether they are officials or bench staff we cannot tell from colour.
5. **Heatmaps** (`heatmaps.jpg`): white is concentrated in front of the near goal (their goalie
   stands there for the whole clip), blue occupies the near zone and the slot. So in this clip
   blue attacks the goal next to the camera. Far-end positions are smeared over several metres
   (H3), so the far half of the heatmaps is only an indication.

**App:** sidebar "Team colours" (on by default when a model exists) colours the boxes and the
minimap dots; the Clip tab applies the window vote live; the new Teams tab shows these results.

**Conclusion:** unsupervised colour clustering separates the two teams and the bystanders with
about 95% accuracy on a small sample, after one failed iteration that taught us to scale the
features. The remaining errors come from overlapping players and from ID switches in the
tracker, not from the clustering itself.

## S1 – Seed study: is E2a really better, and do hard negatives help?, 2026-11-11

**Goal:** E2 compared one training run per variant on 10 val pucks. Two questions were left
open. (1) How much does a result move when only the random seed changes? (2) Do hard negatives
help, now that the comparison is clean (A1, finding 3)?

**Setup**
- GPU: RTX 2080 Ti 11 GB. Same settings as E2 (YOLO26s, 640 px tiles, 100 epochs, batch 16);
  only `--seed` changes: 0, 1, 2.
- **E2a** = `data/puck_tiles/fillempje_v3_tiles640` (872 train tiles, the E2a dataset).
- **E2c** = `data/puck_tiles/fillempje_v3hn2_tiles640`: the same 872 tiles **plus** 135 hard
  negatives (skate crops and the logo spot at (56, 400)). Built with the corrected
  `make_puck_dataset.py`, so the base tiles are identical and the val tiles are the same 200
  (30 puck boxes). This replaces E2b, which differed in almost every tile.
- Per run: tile val mAP from training, then `eval_puck.py` on the 25 full val frames (10 pucks)
  and on the clip. "Independent" = the 5 val pucks that no model proposed (A1, finding 4).

```
python scripts/make_puck_dataset.py --export data/roboflow/<export-folder> --val-from-frame 4200 --detections outputs/baseline/fillempje_yolo26s_1280/detections.csv --skate-crops 4 --lookalike 56 400 --name fillempje_v3hn2_tiles640
python scripts/train_puck.py --data data/puck_tiles/fillempje_v3_tiles640/data.yaml --seed N --name seed_E2a_sN
python scripts/train_puck.py --data data/puck_tiles/fillempje_v3hn2_tiles640/data.yaml --seed N --name seed_E2c_sN
python scripts/eval_puck.py --finetuned outputs/train/<run>/weights/best.pt --frames data/puck_tiles/fillempje_v3_tiles640/frames.csv --video Video/fillempje.mp4 --name <run>
python scripts/seed_summary.py --variant E2a seed_E2a_s0 seed_E2a_s1 seed_E2a_s2 --variant E2c seed_E2c_s0 seed_E2c_s1 seed_E2c_s2
```

Outputs: `outputs/train/seed_*`, `outputs/eval_puck/seed_*`, `outputs/seeds/summary.csv`.

**Results per run** (full frames at conf 0.25, IoU 0.5)

| run | tile mAP@50 | tile mAP@50:95 | TP / FP / FN (10 pucks) | AP@50 full frame | independent pucks | FP on null frames | clip frames with a puck |
|---|---|---|---|---|---|---|---|
| E2a seed 0 | 0.565 | 0.298 | 6 / 4 / 4 | 0.594 | 2/5 | 2 | 846 |
| E2a seed 1 | 0.509 | 0.311 | 6 / 2 / 4 | 0.612 | 2/5 | 1 | 798 |
| E2a seed 2 | 0.618 | 0.313 | 6 / 2 / 4 | 0.572 | 2/5 | 1 | 845 |
| E2c seed 0 | 0.517 | 0.298 | 6 / 3 / 4 | 0.547 | 2/5 | 1 | 819 |
| E2c seed 1 | 0.480 | 0.305 | 5 / 2 / 5 | 0.534 | 2/5 | 1 | 636 |
| E2c seed 2 | 0.548 | 0.301 | 4 / 4 / 6 | 0.506 | 1/5 | 1 | 701 |
| *E2a as logged in E2 (RTX 4060, seed 0)* | *0.634* | *0.382* | *6 / 1 / 4* | *0.65* | *2/5* | *1* | *455* |

**Mean ± standard deviation over 3 seeds**

| variant | tile mAP@50 | tile mAP@50:95 | recall | precision | FP | AP@50 full frame |
|---|---|---|---|---|---|---|
| E2a (no hard negatives) | 0.564 ± 0.055 | 0.307 ± 0.008 | 0.60 ± 0.00 | 0.70 ± 0.09 | 2.7 ± 1.2 | 0.593 ± 0.020 |
| E2c (+ hard negatives) | 0.515 ± 0.034 | 0.301 ± 0.004 | 0.50 ± 0.10 | 0.63 ± 0.11 | 3.0 ± 1.0 | 0.529 ± 0.021 |

**Observations**
1. **The seed alone moves tile mAP@50 by about ±0.05** (0.509 to 0.618 for the same data and
   settings). The val tiles hold only 30 puck boxes, so one box is 3 points of recall. A
   difference between two single runs smaller than about 0.1 mAP@50 means nothing here. That
   covers E1 against E2a in the E2 entry.
2. **The E2a number we reported is not the typical one.** The logged run has mAP@50:95 0.382;
   three new runs give 0.298 to 0.313. The settings are identical (checked in `args.yaml`); the
   differences are the GPU and the random draw (seed 0 on another GPU is not the same run). The
   weights do validate at 0.635 / 0.382 on this machine, so the number is real, but it is the
   best of four draws on the set it was selected on. The honest expectation for this recipe is
   **mAP@50 0.56 ± 0.06, mAP@50:95 0.31 ± 0.01**.
3. **Recall on full frames is stable: 6/10 in all three E2a runs**, and the same 2 of the 5
   independent pucks every time. What varies is the number of false positives (2 to 4) and how
   often the model fires on the clip (798 to 846 frames, against 455 for the logged run). So
   the logged E2a is the most cautious of the four, which is why it has 1 false positive.
4. **Hard negatives do not help, with the confound removed.** E2c is lower on every measure:
   tile mAP@50 −0.05 (inside the seed spread), full-frame AP@50 −0.06 (the two ranges do not
   overlap: 0.572–0.612 against 0.506–0.547), recall 0.5 against 0.6. The false positives they
   were meant to remove did not go down (3.0 against 2.7; 1.0 against 1.3 on the null frames).
   Three seeds and 10 pucks are too few for a significance test, so we say "no evidence of a
   benefit, some evidence of a small cost", not "proven harmful".
5. **The 11 false positives of E2b did not come back** (E2c: 2 to 4). So the worse precision of
   E2b in the E2 entry came mostly from the different draw of tiles, not from the hard
   negatives. The E2 conclusion pointed the right way, for a partly wrong reason.

**Link to annotation:** the hard negatives are crops without a label. They teach "skates and
the logo are not a puck", but the main false positive in E2 was the dark stick blade right next to a puck, and that
is a labelling question (does a puck against the blade count?), not something more background
crops can fix. The false positives of the six seed runs were counted, not looked at one by one,
so whether they are the same blade cases is still to be checked by a team member
(`outputs/eval_puck/seed_*/`).

**Conclusion:** E2a stays the recipe; the app keeps the logged E2a weights, since they have the
fewest false positives. When we quote its quality we give the seed mean (mAP@50 0.56,
mAP@50:95 0.31) next to the single best run (0.634 / 0.382). The next gain has to come from
labels (the blade rule, more val pucks), not from more hard negatives or more training runs.

## R1 – Review of the 2026-11-11 round on a second machine, 2026-11-12

**Goal:** before merging the review round, check its code and re-measure what can be re-measured
on another machine (RTX 4060 Laptop, Ultralytics 8.4.178). Review by Claude Code, asked for by
Andreas; the numbers below are from re-runs, not from reading the log.

| entry | what was re-run | result |
|---|---|---|
| A1 | tile overlap E2a/E2b; `eval_E2b.log`; `metrics.json` of E2a | confirmed: 134 of 872 tiles shared; 6 false positives on null frames and 5 on puck frames; 0.634 / 0.382 |
| H3 | `RinkMapper` on the 6-point config, and on the first 4 points | confirmed exactly: 7.6 / 5.1 / 5.1 / 2.8 / 5.0 / 4.2 px; far posts at x = 112.6 and 94.5 m with the 4-point fit; 37 / 13 / 75 / 31 px with the old fit direction |
| T2, puck | `track.py ... --name fillempje_teams --redo` | confirmed: 17.9% → 41.6% (14.3 + 27.3), 66 gated, jumps 26 / 24 / 16, gap test 0.15 / 0.43 / 0.81 m against 0.39 / 0.65 / 0.97 m |
| T2, players | same run | **differs:** 12.01 on the ice per frame, 232 IDs, 90 longer than 5 s, median life 3.94 s (logged: 11.85, 198, 90, 4.52 s) |
| TM1 | `teams.py ... --names white other blue` on that run | cluster centres identical; 5.9 / 3.3 / 2.9 per frame; silhouette best at k = 3 (0.50); track agreement 85.3%; flips 2.87 → 0.48 per 100 |
| TM1 sample (105/110), S1 | not re-run | the sample labels and the six seed runs are only on the other machine; S1 would take about 2.5 h of training |

**Observation: the player-tracking numbers do not reproduce across machines.** The puck numbers
are identical to the decimal, the player IDs are not (232 against 198). The code and the config
are the same, so the difference comes from the person detector giving slightly different boxes on
another GPU, and ByteTrack amplifying that: one box more or less at the matching threshold starts
or ends a track. So "198 IDs" is good to about ±20%, and the T1 → T2 change in IDs (221 → 198) is
inside that noise. The conclusion of T2 stands (tracks are fragmented, about 200 IDs for about 14
people), the exact count does not.

**Two small code changes in this review**
- `pipeline.py`: with tracking on, the person detector runs at conf ≤ 0.1 (as `track.py` does).
  The app used the sidebar value (0.25), so the low-confidence boxes that ByteTrack's second
  matching round needs never reached the tracker.
- `app.py`: the Clip tab now gives the Kalman filter the rink limit from T2 (`inside`); it was
  only used in `track.py`.

**App changes after testing the Clip tab (2026-11-12, same machine).** Processing 10 s of the
clip (300 frames, tracking and team colours on) worked without errors: 10.9 people on the ice per
frame, puck shown in 95 of 300 frames (58 of them predicted).
- **Drawing the rink template took as long as both detectors.** Timed per frame: detection +
  mapping 112 ms, drawing 105 ms. `draw_template` projected every line piece of the template
  again for every frame, with one `cv2.line` call per piece. Now the markings are projected once per output size
  when the camera is static (`RinkMapper.template_lines`, cached) and drawn as polylines:
  drawing 105 → 15 ms per frame. With thickness 1 (the app) the image is pixel-identical to
  before; with thickness 2 the line joins differ slightly.
- **Players cut off by the bottom of the frame are flagged.** Their box ends at the knees, so
  the foot point is wrong (H2, f2820). Boxes ending within 8 px of the bottom edge get a second
  ring on the minimap and `feet_in_frame = False` in the positions CSV. In 50 sampled frames the
  cut boxes end at y = 1075–1078 (of 1080); real feet near the edge ended at 1043 and 1057.
  The position itself is not corrected.
- A legend under the minimap, and the people count split per colour group.
