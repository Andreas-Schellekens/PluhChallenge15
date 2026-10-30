"""Extract candidate frames from a clip for puck annotation.

Goal: get full-resolution frames to choose the 40-80 annotation frames from. We
keep the original resolution because the puck is only ~15 px wide at 4K; any
downscaling here would make it even harder to label precisely.

Consecutive frames at 30 fps are near-duplicates, so we only keep every n-th
frame (--stride). With the default stride of 10 on a 31 s clip that is ~93
candidates (one every 1/3 s), enough choice to pick the hard frames
(blur, occlusion, crowded areas) and some frames without a visible puck.

Outputs (in data/frames/<clip-name>/, ignored by git):
  <clip>_f000450.jpg   one image per kept frame, named by frame index so we
                       can later split train/val by time without leakage
  frames.csv           filename, frame index and timestamp per image
  contact_sheet.jpg    grid of thumbnails with frame numbers, to pick frames fast

Usage:
  python scripts/extract_frames.py --video Video/hockey_clip_4k_noaudio.mp4
  python scripts/extract_frames.py --video ... --stride 5 --start 10 --end 20
"""

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

THUMB_WIDTH = 480  # width of one thumbnail on the contact sheet
SHEET_COLUMNS = 6


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, help="input .mp4")
    p.add_argument("--stride", type=int, default=10, help="keep every n-th frame")
    p.add_argument("--start", type=float, default=0.0, help="start time in seconds")
    p.add_argument("--end", type=float, default=None, help="end time in seconds (default: end of clip)")
    p.add_argument("--ext", choices=["jpg", "png"], default="jpg",
                   help="jpg (quality 95) is ~2 MB per 4K frame; png is lossless but ~10x larger")
    p.add_argument("--out", default=None, help="output folder (default: data/frames/<clip-name>)")
    return p.parse_args()


def make_contact_sheet(thumbs, labels):
    """Paste thumbnails in a grid and write the frame number on each one."""
    th, tw = thumbs[0].shape[:2]
    rows = math.ceil(len(thumbs) / SHEET_COLUMNS)
    sheet = np.full((rows * th, SHEET_COLUMNS * tw, 3), 255, dtype=np.uint8)
    for i, (thumb, label) in enumerate(zip(thumbs, labels)):
        r, c = divmod(i, SHEET_COLUMNS)
        y, x = r * th, c * tw
        sheet[y:y + th, x:x + tw] = thumb
        # Black outline under white text keeps the label readable on ice and on crowd.
        cv2.putText(sheet, label, (x + 8, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(sheet, label, (x + 8, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return sheet


def main():
    args = parse_args()
    clip = Path(args.video).stem
    out_dir = Path(args.out) if args.out else Path("data/frames") / clip
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    first = round(args.start * fps)
    last = min(n_frames, round(args.end * fps)) if args.end is not None else n_frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)

    rows, thumbs, labels = [], [], []
    params = [cv2.IMWRITE_JPEG_QUALITY, 95] if args.ext == "jpg" else []
    for frame_idx in range(first, last):
        # grab() reads the next frame without converting it to an image; only
        # frames we keep pay for retrieve(). Skipped frames still have to be
        # decoded by the video codec, so this saves time but not all of it.
        if not cap.grab():
            break
        if (frame_idx - first) % args.stride:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        name = f"{clip}_f{frame_idx:06d}.{args.ext}"
        cv2.imwrite(str(out_dir / name), frame, params)
        rows.append({"filename": name, "frame": frame_idx, "time_s": round(frame_idx / fps, 3)})

        h, w = frame.shape[:2]
        thumbs.append(cv2.resize(frame, (THUMB_WIDTH, round(h * THUMB_WIDTH / w)), interpolation=cv2.INTER_AREA))
        labels.append(f"f{frame_idx}  {frame_idx / fps:.1f}s")
        if len(rows) % 20 == 0:
            print(f"  {len(rows)} frames saved...")
    cap.release()

    if not rows:
        raise SystemExit("No frames extracted; check --start/--end.")
    pd.DataFrame(rows).to_csv(out_dir / "frames.csv", index=False)
    cv2.imwrite(str(out_dir / "contact_sheet.jpg"), make_contact_sheet(thumbs, labels),
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"Saved {len(rows)} frames ({w}x{h}, every {args.stride}th frame, "
          f"{rows[0]['time_s']}-{rows[-1]['time_s']} s) to {out_dir}")


if __name__ == "__main__":
    main()
