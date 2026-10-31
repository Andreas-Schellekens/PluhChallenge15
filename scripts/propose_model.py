"""Model-assisted puck proposals for the next labelling round.

Goal: grow the puck dataset with the model we already have. Round 1 used classical
background subtraction (propose_puck.py), which only finds pucks on open ice. Now the
fine-tuned detector (E1) proposes pucks with tiled inference, which also covers some
cases the classical filters threw away (puck at a stick, near a player).

This is the usual "model in the loop" labelling cycle: model proposes -> human reviews
in Roboflow -> retrain. The risk is confirmation bias: the model only proposes what it
already finds. So frames WITHOUT a proposal are listed too; a human labels those by
hand, and they are the most valuable ones (E1 error analysis: skates, boards, far end).

Outputs (in outputs/propose_model/<name>/):
  proposals.csv     frame, rank, conf, box (x, y, w, h in px), file
  no_proposal.csv   frames where the model found nothing above --conf
  montage_XX.jpg    zoomed crops of the top proposals for a visual check

Usage:
  python scripts/propose_model.py --weights outputs/train/fillempje_yolo26s_puck_640/weights/best.pt --frames data/frames/fillempje --exclude data/puck_tiles/fillempje_tiles640/frames.csv --name fillempje_round2
"""

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from eval_puck import predict_tiled


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True, help="fine-tuned puck model (best.pt)")
    p.add_argument("--frames", required=True, help="folder with extracted frames (*_fNNNNNN.jpg)")
    p.add_argument("--exclude", help="frames.csv of an existing dataset: these frames are already labelled")
    p.add_argument("--conf", type=float, default=0.15, help="low on purpose: the human review filters")
    p.add_argument("--top-k", type=int, default=2, help="proposals kept per frame")
    p.add_argument("--tile", type=int, default=640)
    p.add_argument("--overlap", type=int, default=160)
    p.add_argument("--per-montage", type=int, default=24)
    p.add_argument("--name", default="round2")
    return p.parse_args()


def crop(img, cx, cy, half=48, zoom=3):
    """Zoomed crop around a point, padded so crops near the border keep their size."""
    pad = cv2.copyMakeBorder(img, half, half, half, half, cv2.BORDER_CONSTANT)
    c = pad[cy:cy + 2 * half, cx:cx + 2 * half]
    return cv2.resize(c, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)


def main():
    args = parse_args()
    out = Path("outputs/propose_model") / args.name
    out.mkdir(parents=True, exist_ok=True)
    done = set(pd.read_csv(args.exclude)["frame"]) if args.exclude else set()
    model = YOLO(args.weights)

    rows, empty, tiles = [], [], []
    for path in sorted(Path(args.frames).glob("*_f??????.jpg")):  # skips contact_sheet.jpg
        frame = int(re.search(r"_f(\d{6})", path.name).group(1))
        if frame in done:
            continue
        img = cv2.imread(str(path))
        boxes, conf = predict_tiled(model, img, args.tile, args.overlap, args.conf)
        if len(conf) == 0:
            empty.append({"frame": frame, "file": path.name})
            continue
        for rank, i in enumerate(np.argsort(-conf)[:args.top_k]):
            x1, y1, x2, y2 = boxes[i]
            rows.append({"frame": frame, "rank": rank, "conf": round(float(conf[i]), 3),
                         "x": round(x1, 1), "y": round(y1, 1), "w": round(x2 - x1, 1), "h": round(y2 - y1, 1),
                         "file": path.name})
            c = crop(img, int((x1 + x2) / 2), int((y1 + y2) / 2))
            cv2.putText(c, f"f{frame} r{rank} {conf[i]:.2f}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            tiles.append(c)

    pd.DataFrame(rows).to_csv(out / "proposals.csv", index=False)
    pd.DataFrame(empty).to_csv(out / "no_proposal.csv", index=False)
    blank = np.zeros_like(tiles[0]) if tiles else None
    for m in range(0, len(tiles), args.per_montage):
        chunk = tiles[m:m + args.per_montage] + [blank] * (-len(tiles[m:m + args.per_montage]) % 6)
        grid = np.vstack([np.hstack(chunk[r:r + 6]) for r in range(0, len(chunk), 6)])
        cv2.imwrite(str(out / f"montage_{m // args.per_montage + 1:02d}.jpg"), grid)
    n_frames = len({r["frame"] for r in rows})
    print(f"{n_frames} frames with a proposal ({len(rows)} boxes), {len(empty)} without. Saved to {out}")


if __name__ == "__main__":
    main()
