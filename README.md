# PluhChallenge15: Hockey Vision & Tactical Mapping

Team project for the Thomas More Computer Vision challenge 2026–2027 (brief: [`CV_Challenge_2026_2027.pdf`](CV_Challenge_2026_2027.pdf)).

We take an ice hockey clip, detect players, referees and the **puck**, and project everyone onto a top-down 2D rink map that plays in sync with the video, shown in a local Streamlit dashboard.

```
.mp4 clip ──► YOLO detection ──► homography ──► 2D rink minimap ──► Streamlit dashboard
              (players: pretrained COCO,        (≥4 rink keypoints,
               puck: our fine-tuned class)       cv2.findHomography)
```

## Status

| Part | Status |
|---|---|
| Pretrained baseline (E0) | Done: players OK, puck almost never found (1/465 frames). See [`docs/experiments.md`](docs/experiments.md) |
| Puck annotation (40–80 frames) | In progress: candidate frames extracted (`scripts/extract_frames.py`) |
| Puck fine-tuning + mAP@50 / mAP@50:95 | To do |
| Homography + rink minimap | To do |
| Streamlit dashboard | To do |
| Extra Mile (tracking / keypoints / team colours) | To choose |

## Repository layout

```
CV_Challenge_2026_2027.pdf   assignment brief
CLAUDE.md                    guidance for Claude Code (AI assistant) in this repo
requirements.txt             Python dependencies (install PyTorch first, see below)
scripts/
  baseline.py                E0: pretrained COCO YOLO on a clip, writes annotated video + CSV
  extract_frames.py          full-res candidate frames + contact sheet, for choosing annotation frames
docs/
  experiments.md             experiment log: one entry per run (config, results, conclusions)
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

Then put the clip at `Video/hockey_clip_4k_noaudio.mp4` (from the team drive).

## Usage

Run the pretrained baseline (results go to `outputs/baseline/<run>/`):

```bash
python scripts/baseline.py --video Video/hockey_clip_4k_noaudio.mp4 --imgsz 1280 --stride 2
```

Extract candidate frames for annotation, at full 4K, every 10th frame (93 frames from the 31 s clip). This writes the images, `frames.csv` and a `contact_sheet.jpg` overview to `data/frames/<clip>/`:

```bash
python scripts/extract_frames.py --video Video/hockey_clip_4k_noaudio.mp4
```

Pick the 40–80 annotation frames from the contact sheet. Filenames contain the frame index (`_f000570.jpg`), so the train/val split can be done by time.

## Team conventions

- **Log every experiment** in `docs/experiments.md`: config, metrics, observations, conclusions. The process is graded, not just the final dashboard.
- **Never commit** videos, datasets, weights or `outputs/`.
- **GenAI use** must be documented according to the course GenAI policy, and everyone must be able to explain their own code without AI help.
