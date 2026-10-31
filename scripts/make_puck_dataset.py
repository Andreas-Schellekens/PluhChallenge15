"""Build a tiled puck dataset from the Roboflow export (YOLO format, full 4K frames).

Goal: train the puck detector at the puck's NATIVE resolution. E0 showed that
feeding the whole 4K frame to YOLO shrinks the ~15 px puck to ~5 px (imgsz 1280),
below the finest feature map. So we cut 640x640 crops out of the 4K frames: the
model sees the puck at its real size, and at test time we tile the frame the
same way (see scripts/eval_puck.py).

Split by TIME, not randomly: frames close in time look almost the same, so a
random split would leak near-duplicates into validation. Frames before
--val-from-frame go to train, the rest to val.

Train crops (random, seeded):
  --pos-crops per frame: crops that contain the puck at a random position, so the
      model does not learn "the puck is always in the centre"
  --neg-crops per frame: crops without the puck. These are guaranteed negatives:
      there is only one puck, and it is labelled, so the rest of the frame has none.
      Most come from the ice area, where the hard look-alikes are (skates, stick
      tape, board ads); the rest from anywhere (crowd).
  Hard negatives (added after E1, whose main false positives were skates and a
  static dark spot on a sign behind the boards):
  --skate-crops per frame: negative crops that contain a skate, taken from the
      bottom-centre of a person box in --detections (baseline detections.csv)
  --lookalike X Y: a fixed image point that the model mistook for a puck; every
      train frame adds one negative crop containing it
  Both are skipped when the crop would contain the labelled puck.
Val tiles: the full overlapping grid that inference uses, so validation measures
the real situation (many empty tiles, a few with the puck).

Outputs (in data/puck_tiles/<name>/, ignored by git):
  images/{train,val}/*.jpg, labels/{train,val}/*.txt   YOLO format
  data.yaml        dataset config for Ultralytics
  frames.csv       which full frame went to which split (used by eval_puck.py)

Usage:
  python scripts/make_puck_dataset.py --export data/roboflow/pluhchallenge_v2 --val-from-frame 4200       --name fillempje_tiles640
"""

import argparse
import random
import re
from pathlib import Path

import cv2
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--export", required=True, help="Roboflow YOLO export folder (with train/images, train/labels)")
    p.add_argument("--val-from-frame", type=int, default=660, help="frames >= this index go to val")
    p.add_argument("--tile", type=int, default=640, help="crop size in pixels")
    p.add_argument("--overlap", type=int, default=160, help="overlap of the val grid tiles (> puck size)")
    p.add_argument("--pos-crops", type=int, default=8, help="positive crops per train frame")
    p.add_argument("--neg-crops", type=int, default=16, help="negative crops per train frame")
    p.add_argument("--detections", help="baseline detections.csv (person boxes), needed for --skate-crops")
    p.add_argument("--skate-crops", type=int, default=0, help="negative crops around skates per train frame")
    p.add_argument("--lookalike", type=int, nargs=2, action="append", default=[], metavar=("X", "Y"),
                   help="static look-alike point (image px); repeatable")
    p.add_argument("--ice-top", type=int, default=380,
                   help="y above which there is only crowd/roof (380 for fillempje, ~600 for the 4K clip)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default="tiles640_v1", help="output folder name")
    return p.parse_args()


def grid_origins(size, tile, overlap):
    """Top-left coordinates of overlapping tiles that cover [0, size) completely."""
    step = tile - overlap
    xs = list(range(0, size - tile + 1, step))
    if xs[-1] != size - tile:
        xs.append(size - tile)
    return xs


def line_to_box(line, w, h):
    """One YOLO label line (normalised) -> pixel box (x1, y1, x2, y2).
    A box line is `class cx cy bw bh`. A polygon drawn in Roboflow (smart polygon) is
    exported as `class x1 y1 x2 y2 ...`; we turn it into its tightest bounding box."""
    v = list(map(float, line.split()))[1:]
    if len(v) == 4:
        cx, cy, bw, bh = v
        return (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h
    xs, ys = v[0::2], v[1::2]
    return min(xs) * w, min(ys) * h, max(xs) * w, max(ys) * h


def load_boxes(label_file, w, h):
    """YOLO label file -> list of pixel boxes (x1, y1, x2, y2). Empty for a null image."""
    text = label_file.read_text() if label_file.exists() else ""
    return [line_to_box(line, w, h) for line in text.split("\n") if line.strip()]


def crop_labels(boxes, x0, y0, tile, min_visible=0.5):
    """Labels for one crop. A box cut by the crop edge is kept (clipped) only if at
    least half of it is inside: the same 50% rule as in our annotation guidelines."""
    lines = []
    for x1, y1, x2, y2 in boxes:
        cx1, cy1 = max(x1, x0), max(y1, y0)
        cx2, cy2 = min(x2, x0 + tile), min(y2, y0 + tile)
        if cx2 <= cx1 or cy2 <= cy1:
            continue
        if (cx2 - cx1) * (cy2 - cy1) < min_visible * (x2 - x1) * (y2 - y1):
            continue
        lines.append(f"0 {((cx1 + cx2) / 2 - x0) / tile:.6f} {((cy1 + cy2) / 2 - y0) / tile:.6f} "
                     f"{(cx2 - cx1) / tile:.6f} {(cy2 - cy1) / tile:.6f}")
    return lines


def random_origin(rng, lo_edge, hi_edge, size, tile, margin=32):
    """Random crop start along one axis so that [lo_edge, hi_edge] lies inside the crop,
    preferably `margin` px from the crop border. Near the image border that margin is not
    always possible, so the range is clamped to what fits."""
    hi = min(size - tile, max(0, int(lo_edge) - margin))
    lo = max(0, min(int(hi_edge) + margin - tile, hi))
    return rng.randint(lo, hi)


def overlaps(box, x0, y0, tile, margin=0):
    x1, y1, x2, y2 = box
    return x2 + margin > x0 and x1 - margin < x0 + tile and y2 + margin > y0 and y1 - margin < y0 + tile


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    export = Path(args.export)
    out = Path("data/puck_tiles") / args.name
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    persons = pd.read_csv(args.detections).query("`class` == 'person'") if args.detections else None
    T, rows, n_tiles = args.tile, [], {"train": [0, 0], "val": [0, 0]}  # [with puck, without]
    for img_path in sorted((export / "train" / "images").glob("*.jpg")):
        frame = int(re.search(r"_f(\d{6})", img_path.name).group(1))
        split = "val" if frame >= args.val_from_frame else "train"
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        boxes = load_boxes(export / "train" / "labels" / (img_path.stem + ".txt"), w, h)
        rows.append({"frame": frame, "split": split, "image": str(img_path), "n_pucks": len(boxes)})

        if split == "val":
            origins = [(x, y) for y in grid_origins(h, T, args.overlap) for x in grid_origins(w, T, args.overlap)]
        else:
            origins = []
            for x1, y1, x2, y2 in boxes:  # puck somewhere inside the crop, at least 32 px from the edge
                for _ in range(args.pos_crops):
                    origins.append((random_origin(rng, x1, x2, w, T), random_origin(rng, y1, y2, h, T)))
            n_neg = 0
            while n_neg < args.neg_crops:
                # a crop must fit in the frame, so it can start at most at h - T
                y_lo = min(args.ice_top, h - T) if rng.random() < 0.75 else 0
                x0, y0 = rng.randint(0, w - T), rng.randint(y_lo, h - T)
                if not any(overlaps(b, x0, y0, T, margin=8) for b in boxes):
                    origins.append((x0, y0))
                    n_neg += 1
            # hard negatives: a point (skate or static look-alike) somewhere inside the crop
            points = [tuple(p) for p in args.lookalike]
            if persons is not None and args.skate_crops:
                feet = persons[persons.frame == frame]
                for _, d in feet.sample(min(args.skate_crops, len(feet)), random_state=rng.randint(0, 9999)).iterrows():
                    points.append(((d.x1 + d.x2) / 2, d.y2))
            for px, py in points:
                x0, y0 = random_origin(rng, px, px, w, T), random_origin(rng, py, py, h, T)
                if not any(overlaps(b, x0, y0, T, margin=8) for b in boxes):
                    origins.append((x0, y0))

        for x0, y0 in origins:
            name = f"f{frame:06d}_x{x0}_y{y0}"
            labels = crop_labels(boxes, x0, y0, T)
            cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), img[y0:y0 + T, x0:x0 + T],
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            (out / "labels" / split / f"{name}.txt").write_text("\n".join(labels))
            n_tiles[split][0 if labels else 1] += 1

    pd.DataFrame(rows).sort_values("frame").to_csv(out / "frames.csv", index=False)
    (out / "data.yaml").write_text(f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\n"
                                   "names:\n  0: puck\n")
    for split, (pos, neg) in n_tiles.items():
        frames = [r["frame"] for r in rows if r["split"] == split]
        print(f"{split}: {len(frames)} frames {frames} -> {pos} tiles with puck, {neg} without")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
