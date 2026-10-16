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
