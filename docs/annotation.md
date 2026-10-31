# Puck annotation

Labelling rules, occlusion policy and split strategy for the `puck` class. **Status: draft from the pilot (10 frames), to be agreed by the team before labelling the rest.**

## Setup

- Roboflow project `PluhChallenge` (object detection), one class: `puck` (lowercase).
- Source: 93 candidate frames from `scripts/extract_frames.py` (full 4K, every 10th frame of `Video/hockey_clip_4k_noaudio.mp4`).
- `contact_sheet.jpg` was uploaded by mistake: never label it, delete it from the project.

## Pilot (2026-10-30)

10 frames spread over the clip (one every ~3 s): f060, f150, f240, f330, f420, f510, f570, f660, f750, f840.

**How the pilot labels were made (AI-assisted, must be reviewed).**
1. Roboflow autolabel was tried first on f570 (puck position known from E0) and did not work:
   - Gemini 4 Argon boxed something 69×24 px at the wrong location;
   - SAM 3 returned 6 "pucks" in a frame with one;
   - GPT-6.1 Terra failed (provider error).

   A 15 px object in a 4K frame is beyond these foundation models too, which supports E0's data-gap finding.
2. Instead, Claude Code (GenAI) used a classical CV helper to propose candidates: a morphological black-hat filter (small objects darker than their surroundings), then filters on blob size, compactness and bright-ice surroundings. Claude then checked each candidate visually in zoomed crops and computed the boxes.
3. The 5 resulting boxes were uploaded as **predictions routed to Review**. They only enter the dataset after a team member approves them in Roboflow.

**Result**

| frame | puck | box (x, y, w, h in px) | note |
|---|---|---|---|
| f060 | yes | 1830, 1393, 17, 14 | open ice |
| f150 | yes | 2652, 1416, 17, 12 | open ice |
| f240 | not found | | probably occluded or along the boards: human check |
| f330 | not found | | human check |
| f420 | not found | | human check |
| f510 | yes | 3010, 1247, 17, 13 | next to faceoff circle |
| f570 | yes | 2931, 1315, 18, 14 | on a painted line, near a stick |
| f660 | yes | 1421, 1570, 16, 14 | just under a stick blade |
| f750 | not found | | human check |
| f840 | **unsure** | (around 3206, 1143) | dark spot at the end of a stick: puck or black blade tape? |

"Not found" does **not** mean "no puck in the frame". Do not mark these as empty (negative) images until a person has checked them.

## Batch 2 (2026-10-30): every 30th frame

Same method, with one improvement learned from the pilot: candidates are now ranked by how bright their surroundings are instead of by contrast. The real pilot pucks all sit on clean ice (surrounding brightness 225–240). With the new ranking they rank 1st–2nd instead of 7th–23rd, so only the top 18 candidates per frame need to be checked. Exception: a puck touching a stick ranked 11th.

21 frames checked: f000, f030, f090, f120, f180, f210, f270, f300, f360, f390, f450, f480, f540, f600, f630, f690, f720, f780, f810, f870, f900.

| frame | box (x, y, w, h in px) | note |
|---|---|---|
| f090 | 3305, 1288, 14, 12 | faint, likely **motion-blurred** |
| f120 | 3211, 1372, 18, 13 | just ahead of a stick blade |
| f360 | 3439, 1555, 17, 17 | rounder and larger than usual (tilted or airborne?) |
| f600 | 2961, 1311, 17, 13 | on a painted line |
| f630 | 1819, 1160, 19, 19 | between skates, crowded; lighter centre (ice spray?): **weakest label, check first** |
| f720 | 2723, 1363, 18, 13 | open ice |
| f780 | 2547, 1065, 14, 14 | small, near lines |

**Unsure (not labelled, human decision needed):**
- f030 (around 798, 1138) and f900 (around 2483, 1067): dark spot exactly at the end of a stick shaft. Puck or blade tape? Same case as f840.
- f300 (around 1439, 2155): grey smudge cut off by the bottom edge of the frame, about half visible. The occlusion rule decides this one.

**Not found (human check):** f000, f180, f210, f270, f390, f450, f480, f540, f690, f810, f870.

## Status and implication for dataset size

- **Labelled and human-reviewed:** 12 frames. Pilot: f060, f150, f510, f570, f660. Batch 2: f090, f120, f360, f600, f630, f720, f780. All 12 were checked and approved by Andreas in Roboflow Review (2026-10-30), 0 rejected, and moved into the dataset. In Roboflow they are all in `train`: the time-based train/val split is done by us after export, not by Roboflow's random split.
- **Unsure:** 4 frames (f030, f300, f840, f900).
- **Not found:** 15 frames.

Only ~40% of checked frames had a clearly findable puck. Many frames hide it behind players, along the boards or on a stick. At that rate, all 93 candidate frames would give only ~35 positives, below the 40–80 target. Options:
1. Extract more frames (`--stride 5`) in segments where the puck is in open play.
2. Have a human find the puck in the "not found" frames: these are exactly the hard, high-value examples the brief asks for.
3. Add a second clip, which would also be the best val/test set.

## Observations from the pilot

- **The puck is grey, not black.** Its darkest pixel is about 90 on a 0–255 scale, and its core is 90–140. The 4K footage is soft, so the puck appears as a ~11×7 px blurred core with soft edges.
- **Look-alikes (hard negatives):**
  - skates;
  - dark dashes and letters in the board advertisements (e.g. under "TIcasino"), which look very puck-like;
  - stick blades with black tape;
  - dark spots in the crowd behind the glass;
  - lettering on the ice;
  - the black end of a stick blade (batch 2: f030, f270, f900).
- **Hard positives:**
  - puck on a painted line (f570);
  - puck touching or under a stick blade (f660, f840).

## Proposed rules (to be decided by the team)

1. **Box size.** Draw the box around the visible puck including its soft edge: about 3 px around the dark core, giving roughly 16–18 × 12–14 px at 4K. Do not include a motion-blur tail. *(Decide: tail in or out?)*
2. **Occlusion.** Label when about 50% or more of the puck is visible. *(Decide the threshold.)*
3. **Uncertain.** If you cannot tell puck from blade tape or a skate, check the neighbouring frames (±10). If it is still unclear, do not label it, and list the frame here.
4. **Empty frames.** Only mark a frame as having no puck after a person has looked for it. Keep some verified empty frames in the dataset: they teach the model the look-alikes above.
5. **Consistency check.** Two people label the same 5–10 frames independently. Compare the boxes (IoU) and the disagreements, and record the result here.

## Train/val split

Split by time, not randomly: neighbouring frames are near-duplicates and would leak into validation. Proposal: frames from the first ~80% of the clip for train and the last ~20% for val. A second clip would be a better val/test set.

---

# Clip 2: `fillempje.mp4` (main clip from 2026-10-30)

## Frames

`python scripts/extract_frames.py --video Video/fillempje.mp4 --stride 30 --start 1.0`
gives 168 frames, one every 0.5 s from 1 s onwards (the camera moves during the first second).
They are saved to `data/frames/fillempje/`.

## Assisted proposals (`scripts/propose_puck.py`)

The camera is static, so we can model the background:
1. **Median background** over all frames: the empty rink. Lines, net and ads drop out.
2. **Foreground** = pixels clearly darker than that background.
3. **Blob filters:**
   - small and compact;
   - dark in absolute terms (glare flickers);
   - bright ice around the blob;
   - not in the skate zone of a person box from E0-B.
4. **Temporal check:** the blob must reappear nearby 3 frames earlier or later.

Effect of the filters: from 1762 candidates to 214, in 125 of 168 frames.

**Known bias:** pucks at a player's feet or touching a glove are removed by the filters, so
the proposals favour pucks in open ice. That is an annotation choice which comes back in the
error analysis.

## Visual check (Claude Code, GenAI) → Roboflow Review

Claude checked the top 3 candidates of every frame in zoomed montages:

| group | frames | meaning |
|---|---|---|
| `puck-confident` | 26 | clearly the puck, on open ice |
| `puck-unsure` | 40 | mostly a dark spot at the tip of a stick blade: puck on the blade, or black tape? Plus a series f3240–f3390 of a dark streak between two players at a faceoff |
| no proposal | ~100 | puck occluded by players or the net, at someone's feet, or not visible |

All 66 were uploaded to Roboflow (tags `fillempje`, `puck-confident` / `puck-unsure`), with
the box as a **prediction routed to Review**. Box = the dark core of the blob plus 2 px on
every side, so 6–21 px. Nothing enters the dataset until a team member approves it.
Proposals with their boxes: `outputs/propose_puck/fillempje/proposals.csv`.

**Reviewer rule for `puck-unsure`:** approve only if the puck is clearly separate from the
blade, or if the neighbouring frames in the video show that it is the puck. Otherwise reject.
Unclear cases are not labelled.

## Split

By time, as for clip 1: the last ~20% of the clip (frames ≥ 4200, i.e. from about 70 s)
goes to val.

## Review result (Andreas, 2026-10-30)

| group | proposed | approved | rejected | precision of the AI proposal |
|---|---|---|---|---|
| `puck-confident` | 26 | 17 | 9 | 65% |
| `puck-unsure` | 40 | 3 | 37 | 8% |
| **total** | 66 | **20** | 46 | |

- **Rejected "confident" proposals:** f060, f090, f150, f270, f1470, f1830, f4110, f4140, f4230.
  - f060/f090/f150 sit at almost the same spot at the far end, so it was probably a static object there, not the puck.
  - f1470 and f4110 were along the boards.

  Even in the group Claude was sure about, a third was wrong. That justifies the human review.
- **Approved "unsure" proposals:** f2880, f3840, f5070.
- **Corrected by the reviewer:** f2220. The box was moved, so the candidate was a different dark spot than the puck.
- **Dataset:** Roboflow version 2 (filter tag `fillempje`, no preprocessing or augmentation): 20 images, 20 pucks.
- **Split by time:**
  - train: 16 frames (f330–f4080);
  - val: 4 frames (f4200, f4560, f4950, f5070).

**Still below target:** 20 frames, where the assignment asks for 40–80. Next labelling round:
take extra frames near the approved ones (the puck is then nearby, which makes it easy to find)
and frames from parts of the clip with no labels yet. Then a new review.

## Round 2 (2026-10-31): model-assisted proposals

**Why.** E1 left 20 frames (target 40–80), and its error analysis named the missing cases:
pucks at skates, along the boards and far away, plus look-alikes as negatives.

**Method: model in the loop.** The E1 model (`best.pt`) proposes pucks with tiled inference
on all 148 extracted frames that are not yet labelled:

`python scripts/propose_model.py --weights outputs/train/fillempje_yolo26s_puck_640/weights/best.pt --frames data/frames/fillempje --exclude data/puck_tiles/fillempje_tiles640/frames.csv --name fillempje_round2`

At conf ≥ 0.15 it proposes something in 42 frames, and finds nothing in 106 frames.
Claude Code checked every proposal in zoomed crops. For the doubtful ones it also compared:
- the median background: a spot that is also there is painted or fixed, not the puck;
- the frames 10 earlier and later: the puck moves.

| group (Roboflow tag) | frames | meaning | prediction in Review |
|---|---|---|---|
| `model-confident` | 23 | clearly the puck, often at a stick blade | the model's box |
| `model-unsure` | 8 | possible puck: under a skate (f1530), in the net (f2760), motion blur (f690), very far (f1710), lying still at the far end (f90) | the model's box |
| `model-wrong` | 11 | the box is on something else: the logo sign behind the left boards (f4050, f5040), a helmet, a jersey, a skate, a face | the wrong box |
| `manual` | 24 | the model found nothing: all 16 such frames from the val part (≥ f4200) + 8 spread over the rest | none |

- **Upload.** The images were uploaded as `fillempje_r2_fNNNNNN.jpg` (re-encoded). This keeps
  them separate from the round-1 copies, 15 of which were already rejected in Review.
- **Bias.** The model only proposes what it already recognises (confirmation bias). That is
  why the `manual` group exists: those frames are the cases the model misses.
- **Note on f90.** Round 1 rejected f060/f090/f150 as a "static spot". The median background
  is clean at that place, though, and the spot drifts slowly between frames. So it is probably
  a puck lying still during a stoppage. The reviewer decides.

**Reviewer instructions**
- `model-confident` / `model-unsure`:
  - approve if the box is on the puck; tighten it if needed;
  - reject if you are not sure.
- `model-wrong`:
  - if the puck is visible elsewhere in the frame, move the box to it and approve;
  - otherwise reject.
- `manual` (Annotate, not Review):
  - draw the box when ≥ 50% of the puck is visible;
  - if the puck is really not visible (behind players or the net), mark the image as **null**
    (no objects). It then becomes a verified frame without a puck, which is useful as a negative;
  - if unsure, leave it unlabelled.

**Result of the `manual` group (Andreas, 2026-10-31).** In only 1 of the 24 frames was the
puck visible: f4500, barely. Andreas labelled it. In the other 23, the puck was hidden, by
players, the net or skates. At Andreas' request Claude marked those 23 as **null** through the
Roboflow API (an empty annotation, added to the dataset). They are now verified frames
without a visible puck.
- **Meaning for the error analysis.** When the model finds nothing, the puck is almost always
  not visible at all. Most frames "without a detection" are therefore occlusion, not a missed
  puck.
- **Value for evaluation.** 15 of these frames are in the val part (≥ f4200), 8 in train. The val set can
  now also count false positives on frames that have no puck.

**Hard negatives in the dataset** (`make_puck_dataset.py`, new options). The E1 false
positives are added as explicit negative crops. Every crop is checked not to contain the
labelled puck, so it is a guaranteed negative:
- `--skate-crops N`: crops containing a skate, taken from the bottom of the person boxes in
  the E0-B detections. This fixes a gap from round 1: there, skates were filtered out of the
  proposals, so the model never saw them as "not a puck".
- `--lookalike 56 400`: the dark mark on the logo sign behind the left boards, a recurring
  false positive at a fixed spot.

## Review result, round 2 (Andreas, 2026-10-31)

| group | proposed | approved | rejected | precision of the proposal |
|---|---|---|---|---|
| `model-confident` | 23 | 17 | 6 | 74% |
| `model-unsure` | 8 | 3 (f690, f1710, f3240) | 5 | 38% |
| `model-wrong` | 11 | 0 | 11 | – |
| `manual` | 24 | 1 puck (f4500) + 23 null | – | – |

- **Rejected `model-confident`:** f360, f390, f630, f1230, f3810, f3870. All of them are a
  dark spot **touching a stick blade**. In f4590 the reviewer also moved the box away from the
  blade, to the real puck at (1303, 509). So in practice: a puck that is not clearly separate
  from the blade is not labelled.
- **f4500** was drawn as a polygon. `make_puck_dataset.py` and `eval_puck.py` turn polygon
  labels into their bounding box.
- **Dataset:** Roboflow version 3, 64 images: 41 with a puck, 23 null.
  - train (< f4200): 39 frames, 31 with a puck;
  - val (≥ f4200): 25 frames, 10 with a puck and 15 null.

  Target reached: 64 reviewed frames, within 40–80.
- **Open labelling question for the team:** a puck touching the blade, yes or no? Right now it
  is "no if unclear". That makes stick blades the most common false positive (E2).
