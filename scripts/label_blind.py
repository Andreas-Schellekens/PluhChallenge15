"""Blind puck labelling on your own PC, for the consistency check between two labellers.

Goal: a second team member labels the frames picked by `label_agreement.py pick` WITHOUT seeing
the existing labels and without touching the team's Roboflow project. The labels are written in
the same YOLO layout as a Roboflow export, so `label_agreement.py compare` reads them directly.

How it works, per frame (two windows, because the puck is only 6-14 px in a 1920x1080 frame):
  1. "frame" window: the whole frame, scaled to fit the screen.
       click     on the puck: opens the zoom window around that spot
       v         show the neighbouring frames (rule 3 of annotation.md: check +-10 frames when
                 unsure). Needs --video. See "context window" below
       n         no puck visible in this frame (written as an empty label = a null image)
       u         unsure: not labelled (rule 3 of annotation.md), also written as empty
       b         back to the previous frame
       q         quit; your progress is saved and the next run continues where you stopped
  2. "zoom" window: 120 x 120 px around your click, enlarged 6x.
       drag a box TIGHT around the visible puck (no motion-blur tail), then press ENTER or SPACE
       to accept, or c to cancel and click again. The tool adds the margin of rule 1 itself
       (--margin, default 2 px on each side), so every labeller gets the same margin.
The box is converted back to full-frame pixels: x_full = x_crop_origin + x_zoom / 6, then widened
by the margin.

Context window (key v, only with --video): the frames from -10 to +10 around this one, read from
the clip. The frame number comes from the file name (_fNNNNNN = frame index in the video).
       a / d     one frame back / forward          space   play / pause (loops)
       click     zoom in 4x around that spot (click again to move the zoom), right click = whole frame
       v or ESC  back to the "frame" window
A puck moves between frames; tape on a blade, a skate or a mark on the ice moves with the player
or not at all. That is what this window is for (f2700 in round 1: one frame was not enough).

Use the labelling rules in docs/annotation.md and decide on your own: do not look at the team
dataset while labelling, otherwise the check measures nothing.

Outputs (ignored by git):
  data/roboflow/consistency_<labeller>/train/images/*.jpg   copies of the frames
  data/roboflow/consistency_<labeller>/train/labels/*.txt   YOLO labels (class cx cy w h, 0-1)
  data/roboflow/consistency_<labeller>/decisions.csv        per frame: puck / none / unsure, box in px

Usage:
  python scripts/label_blind.py --frames data/consistency/round2 --labeller <name> --video Video/hockey_static_1080p.mp4
  python scripts/label_agreement.py compare --a data/roboflow/pluhchallenge_v4 --b data/roboflow/consistency_<name> --name round2
Round 1 (2026-11-13) was labelled without --video and without the margin (the box was taken as drawn).
"""

import argparse
import re
import shutil
from pathlib import Path

import cv2
import pandas as pd

ZOOM_SIZE, ZOOM_SCALE = 120, 6   # crop of 120 px shown at 720 px
CTX_SIZE, CTX_SCALE = 200, 4     # context window zoom: crop of 200 px shown at 800 px
SCREEN_W = 1600                  # width of the "frame" window


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", required=True, help="folder with the jpg frames from label_agreement.py pick")
    p.add_argument("--labeller", required=True, help="your name, used in the output folder")
    p.add_argument("--video", help="the clip the frames come from; enables key v (neighbouring frames)")
    p.add_argument("--context", type=int, default=10, help="neighbouring frames on each side for key v (rule 3)")
    p.add_argument("--margin", type=float, default=2.0,
                   help="px added on each side of the box you drag (rule 1); 0 = take the box as drawn")
    return p.parse_args()


def frame_number(path):
    """Frame index in the clip, from the _fNNNNNN part of the file name (None if absent)."""
    m = re.search(r"_f(\d{6})", Path(path).name)
    return int(m.group(1)) if m else None


def read_context(cap, centre, n):
    """Frames centre-n .. centre+n from the clip (clipped to the clip), as (frame index, image) pairs.
    Seek once and then read in order, which is faster than seeking for every frame."""
    first = max(0, centre - n)
    last = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1, centre + n)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    out = []
    for idx in range(first, last + 1):
        ok, img = cap.read()
        if not ok:
            break
        out.append((idx, img))
    return out


def show_context(frames, centre, size):
    """Context window: step or play through the neighbouring frames, optionally zoomed on one spot.
    `size` is (w, h) of the labelled frame; a warning is printed if the clip has another resolution
    (then the frame number in the file name probably does not belong to this clip either)."""
    if not frames:
        print(f"frame {centre} is not in the clip: is --video the clip these frames come from?")
        return
    pos = next((i for i, (idx, _) in enumerate(frames) if idx == centre), len(frames) // 2)
    state = {"zoom": None}        # zoom centre in clip pixels, or None for the whole frame
    playing = False
    fh, fw = frames[0][1].shape[:2]
    k = SCREEN_W / fw

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_RBUTTONDOWN:
            state["zoom"] = None
        elif ev == cv2.EVENT_LBUTTONDOWN:
            if state["zoom"] is None:      # click on the scaled whole frame
                state["zoom"] = (x / k, y / k)
            else:                          # click inside the zoom: move the zoom centre there
                ox, oy = ctx_origin(state["zoom"], fw, fh)
                state["zoom"] = (ox + x / CTX_SCALE, oy + y / CTX_SCALE)

    cv2.namedWindow("context")
    cv2.setMouseCallback("context", on_mouse)
    if (fw, fh) != tuple(size):
        print(f"note: clip is {fw}x{fh}, the labelled frame is {size[0]}x{size[1]}")
    while True:
        idx, img = frames[pos]
        if state["zoom"] is None:
            view = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        else:
            ox, oy = ctx_origin(state["zoom"], fw, fh)
            view = cv2.resize(img[oy:oy + CTX_SIZE, ox:ox + CTX_SIZE], None, fx=CTX_SCALE, fy=CTX_SCALE,
                              interpolation=cv2.INTER_CUBIC)
        text = f"frame {idx} ({idx - centre:+d})   a/d step | space play | click zoom | right click whole | v/esc back"
        cv2.putText(view, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(view, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255) if idx == centre else (255, 255, 255), 1)
        cv2.imshow("context", view)
        key = cv2.waitKey(80 if playing else 30) & 0xFF
        if key in (ord("v"), 27):
            break
        if key == ord(" "):
            playing = not playing
        elif key == ord("a"):
            pos, playing = max(0, pos - 1), False
        elif key == ord("d"):
            pos, playing = min(len(frames) - 1, pos + 1), False
        elif playing:
            pos = (pos + 1) % len(frames)
    cv2.destroyWindow("context")


def ctx_origin(point, w, h):
    """Top-left corner of the context zoom crop around a point, kept inside the frame."""
    return (min(max(0, int(point[0]) - CTX_SIZE // 2), w - CTX_SIZE),
            min(max(0, int(point[1]) - CTX_SIZE // 2), h - CTX_SIZE))


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


def ask_frame(img, title, keys="nubq"):
    """Show the whole frame; return ('click', (x, y)) or one of the keys in `keys`."""
    k = SCREEN_W / img.shape[1]
    small = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    help_text = "click = puck | " + ("v = +-frames | " if "v" in keys else "") + "n = no puck | u = unsure | b = back | q = quit"
    cv2.putText(small, title + "   " + help_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    click = []
    cv2.namedWindow("frame")
    cv2.setMouseCallback("frame", lambda ev, x, y, *_: click.append((x / k, y / k)) if ev == cv2.EVENT_LBUTTONDOWN else None)
    while True:
        cv2.imshow("frame", small)
        key = cv2.waitKey(30) & 0xFF
        if click:
            return "click", click[-1]
        if key in map(ord, keys):
            return chr(key), None


def add_margin(box, margin, w, h):
    """Widen a full-frame box by `margin` px on each side (rule 1), kept inside the frame."""
    x1, y1, x2, y2 = box
    return (max(0.0, x1 - margin), max(0.0, y1 - margin), min(float(w), x2 + margin), min(float(h), y2 + margin))


def ask_box(img, point, margin):
    """Zoom around the click and let the labeller drag a tight box; returns the full-frame box
    with the margin added, or None when cancelled."""
    origin = crop_origin(*point, img.shape[1], img.shape[0])
    crop = img[origin[1]:origin[1] + ZOOM_SIZE, origin[0]:origin[0] + ZOOM_SIZE]
    big = cv2.resize(crop, None, fx=ZOOM_SCALE, fy=ZOOM_SCALE, interpolation=cv2.INTER_CUBIC)
    title = "zoom: drag TIGHT around the visible puck, ENTER = accept, c = cancel"
    roi = cv2.selectROI(title, big, showCrosshair=True)
    cv2.destroyWindow(title)
    if roi[2] <= 0 or roi[3] <= 0:
        return None
    return add_margin(zoom_to_frame(roi, origin), margin, img.shape[1], img.shape[0])


def main():
    args = parse_args()
    frames = sorted(Path(args.frames).glob("*.jpg"))
    out = Path("data/roboflow") / f"consistency_{args.labeller}"
    (out / "train" / "images").mkdir(parents=True, exist_ok=True)
    (out / "train" / "labels").mkdir(parents=True, exist_ok=True)
    log_path = out / "decisions.csv"
    log = pd.read_csv(log_path).to_dict("records") if log_path.exists() else []
    done = {r["image"] for r in log}

    cap = None
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise SystemExit(f"cannot open {args.video}")
    else:
        print("No --video: key v (neighbouring frames, rule 3) is off.")

    i = next((n for n, f in enumerate(frames) if f.name not in done), len(frames))
    while i < len(frames):
        path = frames[i]
        img = cv2.imread(str(path))
        h, w = img.shape[:2]
        fno = frame_number(path)
        keys = "nubqv" if cap is not None and fno is not None else "nubq"
        action, point = ask_frame(img, f"{i + 1}/{len(frames)}  {path.stem}", keys)
        if action == "q":
            break
        if action == "b":
            i = max(0, i - 1)
            continue
        if action == "v":
            show_context(read_context(cap, fno, args.context), fno, (w, h))
            continue  # back to the same frame
        box = ask_box(img, point, args.margin) if action == "click" else None
        if action == "click" and box is None:
            continue  # cancelled: show the same frame again
        decision = "puck" if box else ("none" if action == "n" else "unsure")
        shutil.copy(path, out / "train" / "images" / path.name)
        (out / "train" / "labels" / f"{path.stem}.txt").write_text(yolo_line(box, w, h) if box else "")
        log = [r for r in log if r["image"] != path.name] + [{
            "image": path.name, "decision": decision, "margin_px": args.margin,
            **({"x1": round(box[0], 1), "y1": round(box[1], 1), "x2": round(box[2], 1), "y2": round(box[3], 1)} if box else {})}]
        pd.DataFrame(log).to_csv(log_path, index=False)
        print(f"{path.name}: {decision}" + (f" {[round(v, 1) for v in box]}" if box else ""))
        i += 1
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print(f"{len(log)}/{len(frames)} frames labelled -> {out}")
    if len(log) == len(frames):
        print("Done. Next: python scripts/label_agreement.py compare --a data/roboflow/<team export> "
              f"--b {out.as_posix()} --name <round>")


if __name__ == "__main__":
    main()
