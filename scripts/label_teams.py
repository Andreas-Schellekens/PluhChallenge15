"""Label the team-colour sample one player at a time (the human check of TM1).

Goal: fill the `truth` column of `outputs/teams/<name>/sample.csv` (written by `teams.py`) without
comparing a numbered contact sheet with a spreadsheet. The tool shows one detection at a time,
large, next to the whole frame with that player marked, and writes your answer straight into the
CSV. The model's own answer (column `team`) is NOT shown, so the check stays blind.

Keys:
  w  white team          b  blue team
  o  other (referee, bystander, people in camouflage or black jackets at the boards)
  u  unclear (blurred, two people in one box): left out of the score
  backspace or z         back to the previous player (to change an answer)
  q or ESC               quit; every answer is saved at once, the next run continues where you stopped

Afterwards, score the clusters against your labels (the accuracy goes into summary.json):
  python scripts/teams.py --video Video/hockey_static_1080p.mp4 --players outputs/track/fillempje_teams/players.csv \\
      --keypoints configs/keypoints_hockey_static_1080p.json --names white other blue --score --name fillempje

Usage:
  python scripts/label_teams.py --video Video/hockey_static_1080p.mp4 --name fillempje
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

KEYS = {ord("w"): "white", ord("b"): "blue", ord("o"): "other", ord("u"): "unclear"}
BACK = {8, ord("z")}            # backspace, z
QUIT = {27, ord("q")}           # ESC, q


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", default="Video/hockey_static_1080p.mp4")
    p.add_argument("--name", default="fillempje", help="run folder under outputs/teams/")
    return p.parse_args()


def view(frame, r, n, total, answer):
    """Left: the player enlarged (with some ice around him, torso outlined as teams.py uses it).
    Right: the whole frame, scaled down, with the player's box in yellow."""
    x1, y1, x2, y2 = (int(v) for v in (r.x1, r.y1, r.x2, r.y2))
    pad = int(0.4 * max(x2 - x1, y2 - y1))
    cx1, cy1 = max(x1 - pad, 0), max(y1 - pad, 0)
    cx2, cy2 = min(x2 + pad, frame.shape[1]), min(y2 + pad, frame.shape[0])
    crop = frame[cy1:cy2, cx1:cx2].copy()
    cv2.rectangle(crop, (x1 - cx1, y1 - cy1), (x2 - cx1, y2 - cy1), (0, 255, 255), 1)
    k = 600 / crop.shape[0]
    crop = cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)

    small = frame.copy()
    cv2.rectangle(small, (x1 - 6, y1 - 6), (x2 + 6, y2 + 6), (0, 255, 255), 3)
    s = 600 / small.shape[0]
    small = cv2.resize(small, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

    canvas = np.full((680, crop.shape[1] + small.shape[1] + 30, 3), 30, np.uint8)
    canvas[70:670, 10:10 + crop.shape[1]] = crop
    canvas[70:670, 20 + crop.shape[1]:20 + crop.shape[1] + small.shape[1]] = small
    txt = f"player {n + 1} / {total}   (frame {r.frame})"
    if answer:
        txt += f"   current answer: {answer}"
    cv2.putText(canvas, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "w = white   b = blue   o = other   u = unclear   z/backspace = back   q = quit",
                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 220, 255), 1, cv2.LINE_AA)
    return canvas


def save(df, path):
    try:
        df.to_csv(path, index=False)
    except PermissionError:
        raise SystemExit(f"Cannot write {path}: close it in Excel first, then run this script again.")


def main():
    args = parse_args()
    path = Path("outputs/teams") / args.name / "sample.csv"
    df = pd.read_csv(path, dtype={"truth": str}, keep_default_na=False)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {args.video}")

    todo = [i for i, t in enumerate(df.truth) if t == ""]
    n = todo[0] if todo else 0          # continue at the first unlabelled player
    print(f"{len(df) - len(todo)} of {len(df)} already labelled; starting at player {n + 1}")
    while 0 <= n < len(df):
        r = df.iloc[n]
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(r.frame))
        ok, frame = cap.read()
        if not ok:
            raise SystemExit(f"Cannot read frame {r.frame}")
        cv2.imshow("team check", view(frame, r, n, len(df), df.at[n, "truth"]))
        key = cv2.waitKey(0) & 0xFF
        if key in KEYS:
            df.at[n, "truth"] = KEYS[key]
            save(df, path)
            n += 1
        elif key in BACK:
            n = max(n - 1, 0)
        elif key in QUIT:
            break
    cv2.destroyAllWindows()
    counts = df.truth.replace("", "not yet").value_counts().to_dict()
    print(f"saved to {path}: {counts}")


if __name__ == "__main__":
    main()
