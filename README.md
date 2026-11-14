# PluhChallenge15: Hockey Vision & Tactical Mapping

Team project for the Thomas More Computer Vision challenge 2026–2027 (brief: [`CV_Challenge_2026_2027.pdf`](CV_Challenge_2026_2027.pdf)).

We take an ice hockey clip, detect players, referees and the **puck**, and project everyone onto a top-down 2D rink map that plays in sync with the video, shown in a local Streamlit dashboard.

```
.mp4 clip ──► YOLO detection ──► homography ──► 2D rink minimap ──► Streamlit dashboard
              (players: pretrained COCO,        (≥4 rink keypoints,
               puck: our fine-tuned class)       cv2.findHomography)
```

## Status

Main clip: `Video/hockey_static_1080p.mp4` (static camera behind the goal, 1080p; called `fillempje.mp4`, see below). The first clip (`hockey_clip_4k_noaudio.mp4`, handheld 4K) is kept as experiment history.

| Part | Status |
|---|---|
| Pretrained baseline (E0, E0-B) | Done: players OK, puck in only 10/2548 frames (high precision, almost no recall). See [`docs/experiments.md`](docs/experiments.md) |
| Puck annotation (40–80 frames) | 64 human-reviewed frames: 41 with a puck + 23 verified without (2 rounds). See [`docs/annotation.md`](docs/annotation.md). Consistency check (10 frames, blind second labeller): low agreement, mostly the blade question. Decided: box margin 2 px, ±10 frames check always, and the blade rule (Andreas). Open: second check round on Roboflow v4 |
| Puck fine-tuning + mAP@50 / mAP@50:95 | Done (E1, E2a, E2b, seed study S1). Best: E2a, tile val mAP@50 0.634 / mAP@50:95 0.382 for the run we use; over 3 seeds the same recipe gives 0.56 ± 0.06 / 0.31 ± 0.01. Full frames: recall 0/10 → 6/10 (in every seed) with 1 to 4 false positives, AP@50 0.01 → 0.65 (seed mean 0.59) vs baseline. Hard negatives gave no gain (S1) |
| Homography + rink minimap | Done (H3): lens correction + 6 keypoints fitted in pixels, IIHF template. Near the camera good to decimetres, beyond the centre line only to several metres |
| Streamlit dashboard | Done: `streamlit run app.py`, all parameters adjustable, clips can be added from the sidebar |
| Extra Mile: tracking (T1, T2) | Done: ByteTrack player IDs + Kalman filter for the puck (puck on the minimap in 18% → 42% of frames). `scripts/track.py`, app tab Tracking |
| Extra Mile: team colours (TM1) | Done: K-means on jersey colours splits white / blue / bystanders (92% on a 115-crop sample labelled by a team member, TM2), plus occupancy heatmaps. `scripts/teams.py`, app tab Teams |
| Review (A1) | code and log audited; the errors found are listed in the log with what was done about each |
| Error analysis + conf/NMS sweep | confidence and both NMS steps measured for E2a (merge NMS essential: AP@50 0.65 → 0.40 without it; no threshold recovers the missed pucks, so conf 0.25 / NMS 0.7 / 0.3 stay). Errors sorted by the course's slide-38 categories and linked to annotation choices. See [`docs/error_analysis.md`](docs/error_analysis.md). Error types checked by Zjef: the val "puck" at f5070 is a stick blade (label error, removed in v4, E3) |

## Repository layout

```
requirements.txt             Python dependencies (install PyTorch first, see below)
app.py                       Streamlit dashboard: video + minimap, all parameters in the sidebar
scripts/
  baseline.py                E0: pretrained COCO YOLO on a clip, writes annotated video + CSV
  extract_frames.py          full-res candidate frames + contact sheet, for choosing annotation frames
  propose_puck.py            assisted labelling: background subtraction -> puck candidates + montages
  propose_model.py           assisted labelling round 2: the fine-tuned model proposes pucks
  kalman.py                  Kalman filter for the puck on the rink (predict / gate / update)
  track.py                   tracking over the clip: ByteTrack players + Kalman puck, gap test
  teams.py                   team colours: K-means on jersey colours, track vote, occupancy heatmaps
  label_teams.py             label the team-colour check sample one player at a time (TM2)
  seed_summary.py            compares puck models over several training seeds (S1)
  make_puck_dataset.py       Roboflow export -> tiled dataset with a time-based train/val split
  train_puck.py              fine-tune YOLO26 on the puck tiles
  eval_puck.py               baseline vs fine-tuned on full frames: confusion matrix, AP50, examples
  sweep_conf_nms.py          confidence threshold + tile/merge NMS sweep, FP types, crop sheet of every error
  label_agreement.py         consistency check between two labellers: pick blind frames, compare with IoU
  label_blind.py             label those frames locally (click + zoom + drag), without seeing the team labels
  calibrate_lens.py          plumb-line scan for the lens parameter k1 (H4)
  homography.py              image -> rink mapping (lens correction, homography, camera motion), minimap video
  pipeline.py                per-frame glue used by the app: detect -> map -> draw
  rink.py                    NHL (ft) and IIHF (m) rink templates and minimap drawing
configs/
  keypoints_hockey_static_1080p.json   rink keypoints + lens model for the main clip
  keypoints_hockey_clip_4k.json   same for the first clip
docs/
  final_report.md            final report (overview + documentation), figures in report_img/
  experiments.md             experiment log: one entry per run (config, results, conclusions)
  annotation.md              puck labelling rules, review status and split plan
  error_analysis.md          puck detector errors by type (course slide 38) + the conf/NMS sweep
  genai.md                   how GenAI was used and what the team decided and checked
Video/                       input clips              (not in git; share via the team drive)
data/                        annotated datasets       (not in git)
outputs/                     run outputs, weights     (not in git; reproduce from the logged command)
```

Videos, datasets, weights and run outputs are ignored by git (see [`.gitignore`](.gitignore)). Each experiment in `docs/experiments.md` lists the command that reproduces it.

## Setup

Requires Python 3.10–3.12 and, ideally, an NVIDIA GPU.

```bash
py -3.12 -m venv .venv
```

```bash
.venv\Scripts\activate
```

Install PyTorch **with CUDA first**. A plain `pip install torch` gives you the CPU-only build on Windows:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

```bash
pip install -r requirements.txt
```

Check that the GPU is used (should print `True` and your GPU name):

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

No NVIDIA GPU? Install torch without `--index-url`. Everything still works, just slower.

Then put the clips in `Video/` (from the team drive): `hockey_static_1080p.mp4` (main) and `hockey_clip_4k_noaudio.mp4`.

**Renamed.** The main clip used to be `fillempje.mp4`. Rename your local copy to `Video/hockey_static_1080p.mp4` (the app also still recognises the old name). Run folders, datasets and the Roboflow tag keep the old name (`outputs/*/fillempje_*`, `data/frames/fillempje`, `data/puck_tiles/fillempje_*`), because the experiment log refers to them.

## Usage

The dashboard (opens in the browser). Start it with the Streamlit of the `.venv`: a plain `streamlit`
without the venv activated can pick up a global Python install and stop with `No module named 'cv2'`.

```bash
.venv\Scripts\streamlit run app.py
```

The steps behind it, in order. Every command is also logged in `docs/experiments.md`.

1. Pretrained baseline: person boxes for the homography, and the "sports ball" reference for the puck.

```bash
python scripts/baseline.py --video Video/hockey_static_1080p.mp4 --imgsz 1280 --stride 2 --name fillempje_yolo26s_1280
```

2. Frames for annotation (every 0.5 s), then puck candidates for assisted labelling:

```bash
python scripts/extract_frames.py --video Video/hockey_static_1080p.mp4 --stride 30 --start 1.0 --out data/frames/fillempje
```

```bash
python scripts/propose_puck.py --video Video/hockey_static_1080p.mp4 --frames data/frames/fillempje --detections outputs/baseline/fillempje_yolo26s_1280/detections.csv
```

3. After the Roboflow review: export the version (YOLO format, no resize), tile it, train, compare:

```bash
python scripts/make_puck_dataset.py --export data/roboflow/<export-folder> --val-from-frame 4200 --name fillempje_tiles640
```

Round 2 (model in the loop) and the dataset with hard negatives (skates, the logo sign):

```bash
python scripts/propose_model.py --weights outputs/train/fillempje_yolo26s_puck_640/weights/best.pt --frames data/frames/fillempje --exclude data/puck_tiles/fillempje_tiles640/frames.csv --name fillempje_round2
```

```bash
python scripts/make_puck_dataset.py --export data/roboflow/<export-folder> --val-from-frame 4200 --detections outputs/baseline/fillempje_yolo26s_1280/detections.csv --skate-crops 4 --lookalike 56 400 --name fillempje_v3hn2_tiles640
```

```bash
python scripts/train_puck.py --data data/puck_tiles/fillempje_tiles640/data.yaml --name fillempje_yolo26s_puck_640
```

```bash
python scripts/eval_puck.py --finetuned outputs/train/fillempje_yolo26s_puck_640/weights/best.pt --frames data/puck_tiles/fillempje_tiles640/frames.csv --video Video/hockey_static_1080p.mp4 --name fillempje
```

4. Homography and minimap video for the whole clip (add `--name fillempje_6pt` to keep the H3 run apart):

```bash
python scripts/homography.py --video Video/hockey_static_1080p.mp4 --keypoints configs/keypoints_hockey_static_1080p.json --detections outputs/baseline/fillempje_yolo26s_1280/detections.csv --drift none
```

5. Tracking over the whole clip, then team colours from the tracked boxes:

```bash
python scripts/track.py --video Video/hockey_static_1080p.mp4 --keypoints configs/keypoints_hockey_static_1080p.json --puck-weights outputs/train/fillempje_v3_yolo26s_puck_640/weights/best.pt --tracker configs/bytetrack_hockey.yaml --name fillempje_teams
```

```bash
python scripts/teams.py --video Video/hockey_static_1080p.mp4 --players outputs/track/fillempje_teams/players.csv --keypoints configs/keypoints_hockey_static_1080p.json --names white other blue --score --name fillempje
```

6. Seed study (S1): train each variant with seeds 0, 1 and 2 (`--seed N --name seed_E2a_sN`), evaluate each with `eval_puck.py`, then:

```bash
python scripts/seed_summary.py --variant E2a seed_E2a_s0 seed_E2a_s1 seed_E2a_s2 --variant E2c seed_E2c_s0 seed_E2c_s1 seed_E2c_s2
```

7. Error analysis: sweep the confidence threshold and both NMS steps, and get a crop of every error (see [`docs/error_analysis.md`](docs/error_analysis.md)). Run it again with the test set's `frames.csv` once that exists:

```bash
python scripts/sweep_conf_nms.py --weights outputs/train/fillempje_v3_yolo26s_puck_640/weights/best.pt --frames data/puck_tiles/fillempje_v3_tiles640/frames.csv --name E2a_val
```

8. Consistency check between two labellers (`annotation.md`, rule 5). Pick 10 frames and write them without labels (exclude frames the second labeller has already seen up close):

```bash
python scripts/label_agreement.py pick --export data/roboflow/pluhchallenge_v3 --n 10 --exclude 4290 4500 4590 4950 5070 --name round1
```

The second labeller labels them with the rules of `annotation.md`, without looking at the team dataset. Locally, with no change to the Roboflow project (click the puck, drag a tight box in the zoom window, the tool adds the 2 px margin of rule 1; `v` = the ±10 neighbouring frames of rule 3, `n` = no puck, `u` = unsure, `q` = stop and continue later):

```bash
python scripts/label_blind.py --frames data/consistency/round1 --labeller zjef
```

(That is how round 1 was run. From round 2 on, add `--video Video/hockey_static_1080p.mp4`.)

```bash
python scripts/label_blind.py --frames data/consistency/round2 --labeller <name> --video Video/hockey_static_1080p.mp4
```

(Or upload the frames to a separate Roboflow project and export in YOLOv8 format.) Then:

```bash
python scripts/label_agreement.py compare --a data/roboflow/pluhchallenge_v3 --b data/roboflow/consistency_zjef --name round1
```

Round 2 (agreed, on Roboflow v4 and 10 new frames): the full `pick` / `label_blind` / `compare` commands are under "Decisions after round 1" in [`docs/annotation.md`](docs/annotation.md).

Moving the project to another PC: the `data.yaml` files in `data/puck_tiles/*/` hold an absolute path. Change the `path:` line to your own folder, or rebuild the dataset with `make_puck_dataset.py`. Rebuilding from the Roboflow v3 export (YOLOv8 format) gives exactly the same tiles: E2a's `best.pt` validates at the logged 0.634 / 0.382 on them.

## Team conventions

- **Log every experiment** in `docs/experiments.md`: config, metrics, observations, conclusions. The process is graded, not just the final dashboard.
- **Never commit** videos, datasets, weights or `outputs/`.
- **GenAI use** must be documented according to the course GenAI policy, and everyone must be able to explain their own code without AI help.
