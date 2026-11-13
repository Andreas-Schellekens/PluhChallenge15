"""Blind puck labelling on your own PC, for the consistency check between two labellers.

Goal: a second team member labels the frames picked by `label_agreement.py pick` WITHOUT seeing
the existing labels and without touching the team's Roboflow project. The labels are written in
the same YOLO layout as a Roboflow export, so `label_agreement.py compare` reads them directly.

How it works, per frame (two windows, because the puck is only 6-14 px in a 1920x1080 frame):
  1. "frame" window: the whole frame, scaled to fit the screen.
       click     on the puck: opens the zoom window around that spot
       n         no puck visible in this frame (written as an empty label = a null image)
       u         unsure: not labelled (rule 3 of annotation.md), also written as empty
       b         back to the previous frame
       q         quit; your progress is saved and the next run continues where you stopped
  2. "zoom" window: 120 x 120 px around your click, enlarged 6x.
       drag a box around the puck (rule 1 of annotation.md: the visible puck plus its soft edge,
       no motion-blur tail), then press ENTER or SPACE to accept, or c to cancel and click again.
The box is converted back to full-frame pixels: x_full = x_crop_origin + x_zoom / 6.

Use the labelling rules in docs/annotation.md and decide on your own: do not look at the team
dataset while labelling, otherwise the check measures nothing.

Outputs (ignored by git):
  data/roboflow/consistency_<labeller>/train/images/*.jpg   copies of the frames
  data/roboflow/consistency_<labeller>/train/labels/*.txt   YOLO labels (class cx cy w h, 0-1)
  data/roboflow/consistency_<labeller>/decisions.csv        per frame: puck / none / unsure, box in px

Usage:
  python scripts/label_blind.py --frames data/consistency/round1 --labeller zjef
  python scripts/label_agreement.py compare --a data/roboflow/pluhchallenge_v3 --b data/roboflow/consistency_zjef --name round1
"""

import argparse
import shutil
from pathlib import Path

import cv2
import pandas as pd

ZOOM_SIZE, ZOOM_SCALE = 120, 6   # crop of 120 px shown at 720 px
SCREEN_W = 1600                   # width of the "frame" window


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", required=True, help="folder with the jpg frames from label_agreement.py pick")
    p.add_argument("--labeller", required=True, help="your name, used in the output folder")
    return p.parse_args()


def crop_origin(cx, cy, w, h):
    """Top-left corner of the zoom crop around a click, kept inside the frame."""
    return (min(max(0, int(cx) - ZOOM_SIZE // 2), w - ZOOM_SIZE),
            min(max(0, int(cy) - ZOOM_SIZE // 2), h - ZOOM_SIZE))


def zoom_to_frame(roi, origin):
    """(x, y, w, h) drawn in the zoom window -> (x1, y1, x2, y2) in full-frame pixels."""
    x, y, w, h = roi
    return (origin[0] + x / ZOOM_SCALE, origin[1] + y / ZOOM_SCALE,
            origin[0] + (x + w) / ZOOM_SCALE, origin[1] + (y + h) / ZOOM_SCALE)


def yolo_line(box, w, h):
    """Full-frame box in pixels -> YOLO label line (class 0 = puck, centre and size as 0-1)."""
    x1, y1, x2, y2 = box
    return f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"


def ask_frame(img, title):
    """Show the whole frame; return ('click', (x, y)) or a key: 'n', 'u', 'b' or 'q'."""
    k = SCREEN_W / img.shape[1]
    small = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    cv2.putText(small, title + "   click = puck | n = no puck | u = unsure | b = back | q = quit",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    click = []
    cv2.namedWindow("frame")
    cv2.setMouseCallback("frame", lambda ev, x, y, *_: click.append((x / k, y / k)) if ev == cv2.EVENT_LBUTTONDOWN else None)
    while True:
        cv2.imshow("frame", small)
        key = cv2.waitKey(30) & 0xFF
        if click:
            return "click", click[-1]
        if key in map(ord, "nubq"):
            return chr(key), None


def ask_box(img, point):
    """Zoom around the click and let the labeller drag a box; returns a full-frame box or None."""
    origin = crop_origin(*point, img.shape[1], img.shape[0])
    crop = img[origin[1]:origin[1] + ZOOM_SIZE, origin[0]:origin[0] + ZOOM_SIZE]
    big = cv2.resize(crop, None, fx=ZOOM_SCALE, fy=ZOOM_SCALE, interpolation=cv2.INTER_CUBIC)
    roi = cv2.selectROI("zoom: drag a box, ENTER = accept, c = cancel", big, showCrosshair=True)
    cv2.destroyWindow("zoom: drag a box, ENTER = accept, c = cancel")
    return zoom_to_frame(roi, origin) if roi[2] > 0 and roi[3] > 0 else None


def main():
    args = parse_args()
    frames = sorted(Path(args.frames).glob("*.jpg"))
    out = Path("data/roboflow") / f"consistency_{args.labeller}"
    (out / "train" / "images").mkdir(parents=True, exist_ok=True)
    (out / "train" / "labels").mkdir(parents=True, exist_ok=True)
    log_path = out / "decisions.csv"
    log = pd.read_csv(log_path).to_dict("records") if log_path.exists() else []
    done = {r["image"] for r in log}

    i = next((n for n, f in enumerate(frames) if f.name not in done), len(frames))
    while i < len(frames):
        path = frames[i]
        img = cv2.imread(str(path))
        h, w = img.shape[:2]
        action, point = ask_frame(img, f"{i + 1}/{len(frames)}  {path.stem}")
        if action == "q":
            break
        if action == "b":
            i = max(0, i - 1)
            continue
        box = ask_box(img, point) if action == "click" else None
        if action == "click" and box is None:
            continue  # cancelled: show the same frame again
        decision = "puck" if box else ("none" if action == "n" else "unsure")
        shutil.copy(path, out / "train" / "images" / path.name)
        (out / "train" / "labels" / f"{path.stem}.txt").write_text(yolo_line(box, w, h) if box else "")
        log = [r for r in log if r["image"] != path.name] + [{
            "image": path.name, "decision": decision,
            **({"x1": round(box[0], 1), "y1": round(box[1], 1), "x2": round(box[2], 1), "y2": round(box[3], 1)} if box else {})}]
        pd.DataFrame(log).to_csv(log_path, index=False)
        print(f"{path.name}: {decision}" + (f" {[round(v, 1) for v in box]}" if box else ""))
        i += 1
    cv2.destroyAllWindows()
    print(f"{len(log)}/{len(frames)} frames labelled -> {out}")
    if len(log) == len(frames):
        print("Done. Next: python scripts/label_agreement.py compare --a data/roboflow/pluhchallenge_v3 "
              f"--b {out.as_posix()} --name round1")


if __name__ == "__main__":
    main()
