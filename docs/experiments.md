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
