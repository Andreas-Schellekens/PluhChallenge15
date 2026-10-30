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
