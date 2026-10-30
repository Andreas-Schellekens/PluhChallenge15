# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

Team project for the Thomas More Deep Learning – Computer Vision course: **Sports Vision & Tactical Mapping** (challenge 2026–2027). Full brief: `CV_Challenge_2026_2027.pdf`.

Pipeline: `.mp4` clip → object detection (players/referees + fine-tuned small object such as the ball) → homography → synchronized 2D top-down tactical minimap → local Streamlit/Gradio dashboard.

Scope: planar sports only (soccer, basketball, tennis, ice/field hockey), single static camera angle.

**Our choice: ice hockey.** Main clip (since 2026-10-30): `Video/fillempje.mp4` (1920×1080, 59.94 fps, ~85 s, amateur game, static wide-angle action camera behind the near goal). The data gap is the **puck**: about 6–10 px, found by COCO YOLO in only 10/2548 frames (E0-B). Players and referees use pretrained COCO `person`. The homography needs lens correction (k1) and an IIHF rink template in metres (H2). The first clip `Video/hockey_clip_4k_noaudio.mp4` (4K, handheld) is kept as experiment history (E0, H1).

## Repository layout

- `scripts/`: runnable scripts (argparse CLI, outputs to `outputs/<step>/<run-name>/`). Follow the style of `scripts/baseline.py`: module docstring explaining goal, outputs and usage.
- `docs/experiments.md`: experiment log. Every run gets an entry (E0, E1, ...) with the command, config, results table, observations and conclusions.
- `Video/`, `data/`, `outputs/`: local only, ignored by git.
- Environment: `.venv` (Python 3.12) with CUDA PyTorch installed before `requirements.txt`.

## MVP requirements

1. **Targeted annotation** – pretrained YOLO for players/referees; annotate 40–80 challenging frames for the data gap (ball/puck/equipment) with assisted labeling (SAM, CVAT, Roboflow). Document labeling rules, occlusion handling and the train/val split.
2. **Fine-tuning & error analysis** – fine-tune YOLO (detector or segmenter), report mAP@50 and mAP@50:95, document failure modes and link them to annotation choices.
3. **Homography & minimap** – ≥4 known field keypoints, `cv2.findHomography`, project player foot points onto a 2D pitch template.
4. **Local deployment** – app that processes an `.mp4` and shows the annotated video next to the synchronized minimap.

Extra Mile (needed for >85%): tracking (ByteTrack/BoT-SORT, Kalman ball smoothing), automated keypoints (YOLO Pose), or tactical analytics (jersey color clustering, KDE heatmaps).

## Grading priorities

Modeling 35% · Data & annotation 25% · Error analysis 20% · Pipeline & homography 10% · Process & GenAI ownership 10%.

Engineering methodology and conceptual depth matter far more than dashboard polish.

## Working rules for Claude

- **Explain, don't just generate.** Every team member must be able to defend the code live in an oral exam without AI help. Keep code readable and explain non-obvious choices (architecture, losses, augmentations, homography math) when introducing them.
- **Log experiments.** Each training run should record its config, augmentations, and mAP@50 / mAP@50:95 so results are reproducible and comparable. Prefer config files and saved run outputs over ad-hoc notebook state.
- **Prefer simple, transparent code** over heavy boilerplate or copied pipelines.
- **Never commit large artifacts**: raw videos, datasets, model weights and training runs stay out of git (see `.gitignore`).
- GenAI use must be documented per the course GenAI policy.

## Environment

- Windows 11, Python 3.12, Ultralytics YOLO, OpenCV, Streamlit.
- Team GPUs differ (e.g. RTX 4060 Laptop 8 GB, Quadro T1000). Note the GPU in each experiment entry, because speed numbers depend on it.
- Remote: `https://github.com/Andreas-Schellekens/PluhChallenge15`
