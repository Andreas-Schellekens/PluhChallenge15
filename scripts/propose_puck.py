"""Propose puck candidates for assisted labelling (static camera clips).

Goal: speed up annotation. This script does NOT make labels: it ranks small dark
blobs that could be the puck and saves zoomed montages, so a person (or an AI
assistant, whose picks a person then reviews in Roboflow) only has to check a
few crops per frame instead of searching a whole 1080p frame for a ~7 px puck.

Why this works on a static camera:
  1. Background model: the per-pixel MEDIAN over all extracted frames. Players
     move, so the median is the empty rink, including the net, lines and ads.
  2. Foreground: pixels clearly darker than the background (the puck is black,
     the ice is white). Fixed look-alikes (painted lines, net mesh, ads) are part
     of the background and disappear.
  3. Blob filters: small (puck size), compact (not a stick shaft), dark in
     absolute terms (glare from the lights flickers, so a bright reflection can
     look "darker than the background" without being dark), and the ring around
     it must be bright ice (not inside a dark jersey).
     With --detections, blobs in the skate zone of a person box (bottom 20%) are
     dropped: dark skates on white ice are the most common look-alike.
  4. Temporal check: the same blob must also show up within --track-radius px in
     a frame --dt frames earlier or later. Single-frame noise (glare flicker,
     compression) does not.
Candidates are ranked by contrast with the background times how bright the ice
around them is.

Known bias: a puck at a player's feet (skate zone) or touching a dark glove is
dropped or fails the ring test, so these proposals favour pucks in open ice.
Hard cases must be found by hand.

Outputs (in outputs/propose_puck/<clip>/, ignored by git):
  background.jpg        the median background
  candidates.csv        frame, rank, cx, cy, w, h, contrast, ring, temporal, score
  montage_XX.jpg        numbered zoomed crops of the top-k candidates per frame

Usage:
  python scripts/propose_puck.py --video Video/fillempje.mp4 --frames data/frames/fillempje       --detections outputs/baseline/fillempje_yolo26s_1280/detections.csv
"""

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, help="clip the frames came from (for the temporal check)")
    p.add_argument("--frames", required=True, help="folder with frames from extract_frames.py")
    p.add_argument("--ice-top", type=int, default=400, help="ignore everything above this y (stands, roof)")
    p.add_argument("--min-contrast", type=int, default=30, help="how much darker than the background")
    p.add_argument("--max-gray", type=int, default=120, help="the puck itself must be darker than this")
    p.add_argument("--detections", default=None, help="baseline detections.csv, to drop blobs on skates")
    p.add_argument("--min-ring", type=float, default=140, help="minimum brightness of the ice around a blob")
    p.add_argument("--dt", type=int, default=3, help="frames before/after used for the temporal check")
    p.add_argument("--track-radius", type=int, default=40, help="max puck displacement in px over dt frames")
    p.add_argument("--top-k", type=int, default=3, help="candidates per frame in the montages")
    p.add_argument("--per-montage", type=int, default=8, help="frames per montage image")
    return p.parse_args()


def in_skate_zone(cx, cy, boxes):
    """True if the point lies in the bottom 20% of a person box (feet and skates)."""
    for x1, y1, x2, y2 in boxes:
        h, w = y2 - y1, x2 - x1
        if x1 + 0.1 * w <= cx <= x2 - 0.1 * w and y2 - 0.2 * h <= cy <= y2 + 4:
            return True
    return False


def find_blobs(gray, bg, args, boxes=()):
    """Small, compact blobs darker than the background, surrounded by bright ice."""
    diff = bg.astype(np.int16) - gray.astype(np.int16)
    mask = (diff > args.min_contrast).astype(np.uint8)
    mask[:args.ice_top] = 0
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    blobs = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not (3 <= area <= 150 and w <= 22 and h <= 16):
            continue
        if max(w, h) / max(1, min(w, h)) > 4:  # stick shafts and lines are long and thin
            continue
        if gray[labels == i].mean() > args.max_gray or in_skate_zone(cents[i][0], cents[i][1], boxes):
            continue
        pad = 6
        y0, y1 = max(0, y - pad), min(gray.shape[0], y + h + pad)
        x0, x1 = max(0, x - pad), min(gray.shape[1], x + w + pad)
        ring = gray[y0:y1, x0:x1][labels[y0:y1, x0:x1] != i]
        ring = ring[ring > 0]
        if ring.size == 0 or ring.mean() < args.min_ring:
            continue
        contrast = float(diff[labels == i].mean())
        blobs.append({"cx": cents[i][0], "cy": cents[i][1], "w": int(w), "h": int(h),
                      "contrast": round(contrast, 1), "ring": round(float(ring.mean()), 1)})
    return blobs


def main():
    args = parse_args()
    frames_dir = Path(args.frames)
    clip = frames_dir.name
    out = Path("outputs/propose_puck") / clip
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted(frames_dir.glob(f"{clip}_f*.jpg"))

    grays = np.stack([cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in paths])
    bg = np.median(grays, axis=0).astype(np.uint8)
    cv2.imwrite(str(out / "background.jpg"), bg)

    cap = cv2.VideoCapture(args.video)
    dets = pd.read_csv(args.detections) if args.detections else None

    def person_boxes(idx):
        # the baseline ran with --stride 2: use the nearest processed frame
        if dets is None:
            return []
        d = dets[(dets["class"] == "person") & ((dets["frame"] - idx).abs() <= 1)]
        return d[["x1", "y1", "x2", "y2"]].to_numpy()

    def gray_at(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read()
        return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if ok else None

    rows = []
    for p, gray in zip(paths, grays):
        frame = int(re.search(r"_f(\d{6})", p.name).group(1))
        blobs = find_blobs(gray, bg, args, person_boxes(frame))
        neighbours = []
        for d in (-args.dt, args.dt):
            g = gray_at(frame + d)
            if g is not None:
                neighbours += find_blobs(g, bg, args, person_boxes(frame + d))
        for b in blobs:
            b["temporal"] = any(np.hypot(b["cx"] - n["cx"], b["cy"] - n["cy"]) < args.track_radius
                                for n in neighbours)
            # contrast x ice brightness, and a blob seen again nearby in time counts double
            b["score"] = round(b["contrast"] * b["ring"] / 255 * (2 if b["temporal"] else 1), 1)
        blobs.sort(key=lambda b: -b["score"])
        for rank, b in enumerate(blobs):
            rows.append({"frame": frame, "rank": rank, **{k: (round(v, 1) if isinstance(v, float) else v)
                                                          for k, v in b.items()}})
    cap.release()
    cand = pd.DataFrame(rows)
    cand.to_csv(out / "candidates.csv", index=False)

    # montages: one row per frame, top-k zoomed crops + a downscaled overview with the candidates
    by_frame = {int(re.search(r"_f(\d{6})", p.name).group(1)): p for p in paths}
    frame_ids = sorted(by_frame)
    for m in range(0, len(frame_ids), args.per_montage):
        rows_img = []
        for frame in frame_ids[m:m + args.per_montage]:
            img = cv2.imread(str(by_frame[frame]))
            c = cand[(cand["frame"] == frame) & (cand["rank"] < args.top_k)] if len(cand) else cand
            tiles = []
            for _, b in c.iterrows():
                cx, cy, r = int(b["cx"]), int(b["cy"]), 20
                crop = cv2.copyMakeBorder(img, r, r, r, r, cv2.BORDER_CONSTANT)[cy:cy + 2 * r, cx:cx + 2 * r]
                crop = cv2.resize(crop, (192, 192), interpolation=cv2.INTER_NEAREST)
                cv2.putText(crop, f"{int(b['rank'])}", (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                tiles.append(crop)
            while len(tiles) < args.top_k:
                tiles.append(np.full((192, 192, 3), 255, np.uint8))
            overview = cv2.resize(img, (341, 192), interpolation=cv2.INTER_AREA)
            for _, b in c.iterrows():
                cv2.circle(overview, (int(b["cx"] * 341 / 1920), int(b["cy"] * 192 / 1080)), 6, (0, 0, 255), 1)
                cv2.putText(overview, str(int(b["rank"])), (int(b["cx"] * 341 / 1920) + 6, int(b["cy"] * 192 / 1080)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            cv2.putText(overview, f"f{frame}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            rows_img.append(np.hstack(tiles + [overview]))
        cv2.imwrite(str(out / f"montage_{m // args.per_montage:02d}.jpg"), np.vstack(rows_img),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
    n_frames_with = cand["frame"].nunique() if len(cand) else 0
    print(f"{len(cand)} candidates in {n_frames_with}/{len(paths)} frames; montages in {out}")


if __name__ == "__main__":
    main()
