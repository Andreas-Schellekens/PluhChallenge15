# PluhChallenge15: Hockey Vision & Tactical Mapping

Team project for the Thomas More Computer Vision challenge 2026–2027 (brief: [`CV_Challenge_2026_2027.pdf`](CV_Challenge_2026_2027.pdf)).

We take an ice hockey clip, detect players, referees and the **puck**, and project everyone onto a top-down 2D rink map that plays in sync with the video, shown in a local Streamlit dashboard.

```
.mp4 clip ──► YOLO detection ──► homography ──► 2D rink minimap ──► Streamlit dashboard
              (players: pretrained COCO,        (≥4 rink keypoints,
               puck: our fine-tuned class)       cv2.findHomography)
```

## Status

Main clip since 2026-10-30: `Video/fillempje.mp4` (static camera behind the goal, 1080p). The first clip (`hockey_clip_4k_noaudio.mp4`, handheld 4K) is kept as experiment history.

| Part | Status |
|---|---|
| Pretrained baseline (E0, E0-B) | Done: players OK, puck in only 10/2548 frames (high precision, almost no recall). See [`docs/experiments.md`](docs/experiments.md) |
| Puck annotation (40–80 frames) | 20 human-reviewed frames (66 proposals, 20 approved); second round needed to reach 40–80. See [`docs/annotation.md`](docs/annotation.md) |
| Puck fine-tuning + mAP@50 / mAP@50:95 | Done (E1): tile val mAP@50 0.789 / mAP@50:95 0.506; full frames recall 0/4 → 2/4, AP@50 0 → 0.75 vs baseline. See [`docs/experiments.md`](docs/experiments.md) |
| Homography + rink minimap | Done (H2): lens correction + 4 keypoints, IIHF template |
| Streamlit dashboard | Done: `streamlit run app.py`, all parameters adjustable |
| Extra Mile (tracking / keypoints / team colours) | To choose |

## Repository layout

```
CV_Challenge_2026_2027.pdf   assignment brief
CLAUDE.md                    guidance for Claude Code (AI assistant) in this repo
requirements.txt             Python dependencies (install PyTorch first, see below)
app.py                       Streamlit dashboard: video + minimap, all parameters in the sidebar
scripts/
  baseline.py                E0: pretrained COCO YOLO on a clip, writes annotated video + CSV
  extract_frames.py          full-res candidate frames + contact sheet, for choosing annotation frames
  propose_puck.py            assisted labelling: background subtraction -> puck candidates + montages
  make_puck_dataset.py       Roboflow export -> tiled dataset with a time-based train/val split
  train_puck.py              fine-tune YOLO26 on the puck tiles
  eval_puck.py               baseline vs fine-tuned on full frames: confusion matrix, AP50, examples
  homography.py              image -> rink mapping (lens correction, homography, camera motion), minimap video
  pipeline.py                per-frame glue used by the app: detect -> map -> draw
  rink.py                    NHL (ft) and IIHF (m) rink templates and minimap drawing
configs/
  keypoints_fillempje.json        rink keypoints + lens model for the main clip
  keypoints_hockey_clip_4k.json   same for the first clip
docs/
  experiments.md             experiment log: one entry per run (config, results, conclusions)
  annotation.md              puck labelling rules, review status and split plan
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

Then put the clips in `Video/` (from the team drive): `fillempje.mp4` (main) and `hockey_clip_4k_noaudio.mp4`.

## Usage

The dashboard (opens in the browser):

```bash
streamlit run app.py
```

The steps behind it, in order. Every command is also logged in `docs/experiments.md`.

1. Pretrained baseline: person boxes for the homography, and the "sports ball" reference for the puck.

```bash
python scripts/baseline.py --video Video/fillempje.mp4 --imgsz 1280 --stride 2 --name fillempje_yolo26s_1280
```

2. Frames for annotation (every 0.5 s), then puck candidates for assisted labelling:

```bash
python scripts/extract_frames.py --video Video/fillempje.mp4 --stride 30 --start 1.0
```

```bash
python scripts/propose_puck.py --video Video/fillempje.mp4 --frames data/frames/fillempje --detections outputs/baseline/fillempje_yolo26s_1280/detections.csv
```

3. After the Roboflow review: export the version (YOLO format, no resize), tile it, train, compare:

```bash
python scripts/make_puck_dataset.py --export data/roboflow/<export-folder> --val-from-frame 4200 --name fillempje_tiles640
```

```bash
python scripts/train_puck.py --data data/puck_tiles/fillempje_tiles640/data.yaml --name fillempje_yolo26s_puck_640
```

```bash
python scripts/eval_puck.py --finetuned outputs/train/fillempje_yolo26s_puck_640/weights/best.pt --frames data/puck_tiles/fillempje_tiles640/frames.csv --video Video/fillempje.mp4 --name fillempje
```

4. Homography and minimap video for the whole clip:

```bash
python scripts/homography.py --video Video/fillempje.mp4 --keypoints configs/keypoints_fillempje.json --detections outputs/baseline/fillempje_yolo26s_1280/detections.csv --drift none
```

## Team conventions

- **Log every experiment** in `docs/experiments.md`: config, metrics, observations, conclusions. The process is graded, not just the final dashboard.
- **Never commit** videos, datasets, weights or `outputs/`.
- **GenAI use** must be documented according to the course GenAI policy, and everyone must be able to explain their own code without AI help.
