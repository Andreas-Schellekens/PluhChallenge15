"""Consistency check between two labellers: do two team members label the same puck the same way?

Goal: the brief asks us to document label consistency, "because you will be labelling as a team".
Until now every puck label was reviewed by one person (Andreas). Here a second person labels the
same frames WITHOUT seeing the first labels, and we measure how much the two agree. That number
is the noise level of our ground truth: if two people disagree by 3 px on a 10 px puck, a model
cannot be expected to do better, and mAP@50:95 (which asks for IoU up to 0.95) is partly
measuring us, not the model (course deck 5, slide 40).

Two steps:

  pick     choose the frames for the second labeller from a Roboflow export, at random but
           stratified (frames with and without a puck in the proportion of the export, so the
           second labeller also meets empty frames). The images are written again WITHOUT labels,
           re-encoded (so Roboflow does not treat them as duplicates of the labelled images), and
           named consistency_fNNNNNN.jpg. Upload them to a separate Roboflow project or tag, label
           them, and export that version in YOLOv8 format.

  compare  match the two exports frame by frame (by the frame number in the file name) and
           report, per frame:
             - presence: puck in both / in neither / only in A / only in B
             - IoU of the two boxes, centre distance (px), width and height difference (px)
           A pair with IoU >= 0.5 counts as "the same puck" (the mAP@50 rule).

Outputs:
  pick     data/consistency/<name>/consistency_fNNNNNN.jpg, picked.csv     (not in git)
  compare  outputs/label_agreement/<name>/agreement.csv, summary.json,
           sheet.jpg (zoomed crop per frame: A green, B magenta)            (not in git)

Usage:
  python scripts/label_agreement.py pick --export data/roboflow/pluhchallenge_v3 --n 10 --exclude 4290 4500 4590 4950 5070 --name round1
  python scripts/label_agreement.py compare --a data/roboflow/pluhchallenge_v3 --b data/roboflow/<second-labeller-export> --name round1
"""

import argparse
import json
import random
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from eval_puck import iou, zoom
from make_puck_dataset import load_boxes

GREEN, MAGENTA = (0, 200, 0), (255, 0, 255)  # BGR: labeller A, labeller B


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)
    pk = sub.add_parser("pick", help="choose frames and write them without labels")
    pk.add_argument("--export", required=True, help="Roboflow YOLO export with the first labeller's labels")
    pk.add_argument("--n", type=int, default=10)
    pk.add_argument("--seed", type=int, default=0)
    pk.add_argument("--exclude", type=int, nargs="*", default=[],
                    help="frame numbers the second labeller has already seen up close (not blind any more)")
    pk.add_argument("--name", default="round1")
    cp = sub.add_parser("compare", help="compare two exports frame by frame")
    cp.add_argument("--a", required=True, help="export of labeller A (e.g. the team dataset)")
    cp.add_argument("--b", required=True, help="export of labeller B (the blind second labelling)")
    cp.add_argument("--name", default="round1")
    return p.parse_args()


def frames_in(export):
    """{frame number: (image path, label path)} for every image of a Roboflow YOLO export.
    Roboflow can put images in train/, valid/ or test/, so all three are searched."""
    out = {}
    for img in sorted(Path(export).glob("*/images/*.jpg")):
        m = re.search(r"_f(\d{6})", img.name)
        if m:
            out[int(m.group(1))] = (img, img.parent.parent / "labels" / (img.stem + ".txt"))
    return out


def boxes_of(img_path, label_path):
    h, w = cv2.imread(str(img_path)).shape[:2]
    return np.array(load_boxes(label_path, w, h)).reshape(-1, 4) if label_path.exists() else np.zeros((0, 4))


def pick(args):
    frames = {f: v for f, v in frames_in(args.export).items() if f not in set(args.exclude)}
    with_puck = sorted(f for f, (i, l) in frames.items() if len(boxes_of(i, l)))
    without = sorted(f for f in frames if f not in with_puck)
    n_puck = round(args.n * len(with_puck) / len(frames))  # same proportion as the export
    rng = random.Random(args.seed)
    chosen = sorted(rng.sample(with_puck, n_puck) + rng.sample(without, args.n - n_puck))
    out = Path("data/consistency") / args.name
    out.mkdir(parents=True, exist_ok=True)
    for f in chosen:
        img = cv2.imread(str(frames[f][0]))
        # re-encode: same picture, different bytes, so Roboflow does not merge it with the labelled copy
        cv2.imwrite(str(out / f"consistency_f{f:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 97])
    # picked.csv does NOT say which frames have a puck: the second labeller must not know
    pd.DataFrame({"frame": chosen}).to_csv(out / "picked.csv", index=False)
    print(f"{len(chosen)} frames ({n_puck} with a puck in A, {args.n - n_puck} without) -> {out}")
    print("Label them without looking at the existing labels, export in YOLOv8 format, then run `compare`.")


def compare(args):
    a, b = frames_in(args.a), frames_in(args.b)
    common = sorted(set(a) & set(b))
    if not common:
        raise SystemExit("no frame numbers in common between the two exports")
    out = Path("outputs/label_agreement") / args.name
    out.mkdir(parents=True, exist_ok=True)
    rows, crops = [], []
    for f in common:
        ba, bb = boxes_of(*a[f]), boxes_of(*b[f])
        row = {"frame": f, "n_A": len(ba), "n_B": len(bb)}
        if len(ba) and len(bb):
            # one puck per frame: compare the best-overlapping pair
            i, j = max(((i, j) for i in range(len(ba)) for j in range(len(bb))), key=lambda ij: iou(ba[ij[0]], bb[ij[1]]))
            A, B = ba[i], bb[j]
            row.update(presence="both", IoU=round(iou(A, B), 3),
                       centre_dist_px=round(float(np.hypot(*((A[:2] + A[2:]) / 2 - (B[:2] + B[2:]) / 2))), 1),
                       w_A=round(A[2] - A[0], 1), w_B=round(B[2] - B[0], 1),
                       h_A=round(A[3] - A[1], 1), h_B=round(B[3] - B[1], 1))
            row["same_puck"] = row["IoU"] >= 0.5
            centre = (A[:2] + A[2:]) / 2
        else:
            row["presence"] = "neither" if not len(ba) and not len(bb) else ("only A" if len(ba) else "only B")
            only = ba if len(ba) else bb
            centre = (only[0, :2] + only[0, 2:]) / 2 if len(only) else None
        rows.append(row)
        img = cv2.imread(str(a[f][0]))
        crop = (zoom(img, centre, [(ba, GREEN), (bb, MAGENTA)], size=120, out=240) if centre is not None
                else cv2.resize(img, (240, 240)))
        label = f"f{f} {row['presence']}" + (f" IoU {row['IoU']:.2f}" if "IoU" in row else "")
        cv2.putText(crop, label, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(crop, label, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        crops.append(crop)

    table = pd.DataFrame(rows)
    table.to_csv(out / "agreement.csv", index=False)
    while len(crops) % 5:
        crops.append(np.zeros_like(crops[0]))
    cv2.imwrite(str(out / "sheet.jpg"), np.vstack([np.hstack(crops[i:i + 5]) for i in range(0, len(crops), 5)]),
                [cv2.IMWRITE_JPEG_QUALITY, 92])

    both = table[table.presence == "both"]
    summary = {
        "frames_compared": len(table),
        "presence": table.presence.value_counts().to_dict(),
        "presence_agreement": round(float(table.presence.isin(["both", "neither"]).mean()), 3),
        "same_puck_IoU50": int((both.IoU >= 0.5).sum()),
        "IoU_mean": round(float(both.IoU.mean()), 3) if len(both) else None,
        "IoU_min": round(float(both.IoU.min()), 3) if len(both) else None,
        "centre_dist_px_mean": round(float(both.centre_dist_px.mean()), 1) if len(both) else None,
        "size_diff_px_mean": round(float(((both.w_A - both.w_B).abs() + (both.h_A - both.h_B).abs()).mean() / 2), 1)
        if len(both) else None,
        "only_in_A": [int(f) for f in table[table.presence == "only A"].frame],
        "only_in_B": [int(f) for f in table[table.presence == "only B"].frame],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(table.to_string(index=False))
    print(json.dumps(summary, indent=2))
    print(f"Saved to {out}")


def main():
    args = parse_args()
    pick(args) if args.mode == "pick" else compare(args)


if __name__ == "__main__":
    main()
