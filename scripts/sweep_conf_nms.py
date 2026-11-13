"""Sweep the confidence threshold and the two NMS thresholds of the tiled puck detector.

Goal: the detector does not output "the puck". It outputs many scored boxes, and two
post-processing choices decide which ones survive (course, deck 5 slide 39):
  confidence threshold  keep boxes with a score >= conf. Low conf: more pucks found
                        (recall up) but more false alarms. High conf: the reverse.
  NMS IoU threshold     of two boxes that overlap more than this IoU, keep only the one
                        with the highest score. Too low: two nearby real objects can
                        suppress each other. Too high: duplicate boxes on one object survive.
Until now eval_puck.py used one fixed setting (conf 0.25, NMS 0.7 / 0.3). This script
measures what each choice costs, on the same labelled frames and with the same
scoring rule (a hit = IoU >= 0.5 with a labelled puck, as in mAP@50).

Our pipeline has TWO NMS steps, because we predict tile by tile (see eval_puck.py):
  1. tile NMS   inside each 640x640 tile, done by Ultralytics (`iou=` of predict).
                Our weights have end2end=False, so this is classic NMS.
  2. merge NMS  after shifting all tile boxes back to the full frame: the tiles overlap
                by 160 px, so a puck in an overlap strip is found twice. Our own
                cv2.dnn.NMSBoxes call removes those duplicates. 1.0 = no merge NMS.

Every false positive (FP) at the chosen operating point is also sorted into a type,
after "not all wrong boxes are equally wrong" (slide 37) and "what can go wrong" (slide 38):
  duplicate    overlaps a puck that another box already matched (NMS did not remove it)
  localization overlaps an unmatched puck, but with IoU < 0.5 (right object, box too loose/off)
  background   overlaps no labelled puck at all (stick blade, skate, logo, ...). Which of
               those it is, is decided by a person looking at errors.jpg, not by this script.

Outputs (in outputs/sweep/<name>/, ignored by git):
  sweep.csv        one row per (tile NMS, merge NMS, conf): TP, FP (by type), FN, precision,
                   recall, F1, false positives on frames without a puck
  ap.csv           AP@50 per (tile NMS, merge NMS), over all confidences
  pr_vs_conf.png   precision, recall and F1 against conf, at the operating NMS setting
  nms_effect.png   AP@50 and the number of FPs at the operating conf, per NMS combination
  errors.csv       every FP and missed puck (FN) at the operating point, with an empty
                   `category` column for the team to fill in (slide 38 categories)
  errors.jpg       a zoomed crop of each of those errors, numbered as in errors.csv
  metrics.json     settings, best F1 setting, and a sanity check against eval_puck.py

Usage:
  python scripts/sweep_conf_nms.py --weights outputs/train/fillempje_v3_yolo26s_puck_640/weights/best.pt \\
      --frames data/puck_tiles/fillempje_v3_tiles640/frames.csv --name E2a_val
"""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
from ultralytics import YOLO

from eval_puck import average_precision, grid_origins, iou, load_gt, match, zoom

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CONF_FLOOR = 0.01  # predict with a very low threshold once, then apply every conf afterwards
GREEN, ORANGE, RED = (0, 200, 0), (0, 140, 255), (0, 0, 255)  # BGR: label, detection, error


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True, help="best.pt of the puck model")
    p.add_argument("--frames", required=True, help="frames.csv from make_puck_dataset.py")
    p.add_argument("--split", default="val", help="which frames to score")
    p.add_argument("--tile", type=int, default=640)
    p.add_argument("--overlap", type=int, default=160)
    p.add_argument("--tile-iou", type=float, nargs="+", default=[0.3, 0.5, 0.7, 0.9],
                   help="NMS IoU inside each tile (Ultralytics); its default is 0.7")
    p.add_argument("--merge-iou", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                   help="NMS IoU when merging the tiles; 1.0 = no merge NMS. eval_puck.py uses 0.3")
    p.add_argument("--confs", type=float, nargs="+", default=[round(c, 2) for c in np.arange(0.05, 0.95, 0.05)])
    p.add_argument("--op-conf", type=float, default=0.25, help="operating point used in the log and the app")
    p.add_argument("--op-tile-iou", type=float, default=0.7)
    p.add_argument("--op-merge-iou", type=float, default=0.3)
    p.add_argument("--name", default="sweep")
    return p.parse_args()


def predict_tiles(model, img, tile, overlap, tile_iou):
    """Same tiling as eval_puck.predict_tiled, but WITHOUT the merge step, so the merge NMS
    can be swept afterwards without running the model again."""
    h, w = img.shape[:2]
    origins = [(x, y) for y in grid_origins(h, tile, overlap) for x in grid_origins(w, tile, overlap)]
    results = model.predict([img[y:y + tile, x:x + tile] for x, y in origins], imgsz=tile, conf=CONF_FLOOR,
                            iou=tile_iou, verbose=False, batch=16)
    boxes = [r.boxes.xyxy.cpu().numpy() + [x0, y0, x0, y0] for (x0, y0), r in zip(origins, results)]
    scores = [r.boxes.conf.cpu().numpy() for r in results]
    return np.concatenate(boxes).reshape(-1, 4), np.concatenate(scores)


def merge(boxes, scores, merge_iou):
    """The merge NMS of eval_puck.predict_tiled, with a variable IoU threshold."""
    if merge_iou >= 1.0 or len(boxes) == 0:
        return boxes, scores
    xywh = np.column_stack([boxes[:, :2], boxes[:, 2:] - boxes[:, :2]]).tolist()
    keep = np.array(cv2.dnn.NMSBoxes(xywh, scores.tolist(), CONF_FLOOR, merge_iou), dtype=int).reshape(-1)
    return boxes[keep], scores[keep]


def fp_types(boxes, tp, gts):
    """Sort the false positives into duplicate / localization / background (see docstring)."""
    matched_gt = {j for i in np.flatnonzero(tp) for j, g in enumerate(gts) if iou(boxes[i], g) >= 0.5}
    types = []
    for i in np.flatnonzero(~tp):
        touched = [j for j, g in enumerate(gts) if iou(boxes[i], g) > 0]
        if not touched:
            types.append("background")
        elif any(j in matched_gt for j in touched):
            types.append("duplicate")
        else:
            types.append("localization")
    return types


def score(frames, conf):
    """TP / FP / FN over all frames at one confidence threshold. frames: list of (boxes, scores, gts)."""
    out = {"TP": 0, "FP": 0, "FN": 0, "duplicate": 0, "localization": 0, "background": 0, "FP_null_frames": 0}
    for boxes, scores, gts in frames:
        keep = scores >= conf
        tp = match(boxes[keep], scores[keep], gts)
        out["TP"] += int(tp.sum())
        out["FP"] += int((~tp).sum())
        out["FN"] += len(gts) - int(tp.sum())
        if len(gts) == 0:
            out["FP_null_frames"] += int((~tp).sum())
        for t in fp_types(boxes[keep], tp, gts):
            out[t] += 1
    p = out["TP"] / (out["TP"] + out["FP"]) if out["TP"] + out["FP"] else 0.0
    r = out["TP"] / (out["TP"] + out["FN"]) if out["TP"] + out["FN"] else 0.0
    out.update(precision=round(p, 3), recall=round(r, 3), F1=round(2 * p * r / (p + r), 3) if p + r else 0.0)
    return out


def ap50(frames):
    scores, tps, n_gt = [], [], 0
    for boxes, s, gts in frames:
        scores += list(s)
        tps += list(match(boxes, s, gts))
        n_gt += len(gts)
    return round(average_precision(np.array(scores), np.array(tps, bool), n_gt), 3)


def error_sheet(images, frames, frame_ids, conf, out_dir):
    """Crop every FP and FN at the operating point; write errors.csv and errors.jpg."""
    rows, crops = [], []
    for img, (boxes, scores, gts), f in zip(images, frames, frame_ids):
        keep = scores >= conf
        b, s = boxes[keep], scores[keep]
        tp = match(b, s, gts)
        types = iter(fp_types(b, tp, gts))
        errors = [("FP", next(types), b[i], s[i]) for i in np.flatnonzero(~tp)]
        found = {j for i in np.flatnonzero(tp) for j, g in enumerate(gts) if iou(b[i], g) >= 0.5}
        errors += [("FN", "missed puck", g, np.nan) for j, g in enumerate(gts) if j not in found]
        for kind, sub, box, sc in errors:
            n = len(rows) + 1
            rows.append({"id": n, "frame": f, "kind": kind, "type": sub,
                         "conf": round(float(sc), 3) if not np.isnan(sc) else "",
                         "x1": int(box[0]), "y1": int(box[1]), "x2": int(box[2]), "y2": int(box[3]),
                         "puck_in_frame": len(gts) > 0, "category": ""})
            crop = zoom(img, (box[:2] + box[2:]) / 2, [(gts, GREEN), (b, ORANGE), ([box], RED)], size=120, out=240)
            label = f"#{n} f{f} {kind}" + (f" {sc:.2f}" if kind == "FP" else "")
            cv2.putText(crop, label, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(crop, label, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            crops.append(crop)
    pd.DataFrame(rows).to_csv(out_dir / "errors.csv", index=False)
    if crops:
        while len(crops) % 5:
            crops.append(np.zeros_like(crops[0]))
        sheet = np.vstack([np.hstack(crops[i:i + 5]) for i in range(0, len(crops), 5)])
        cv2.imwrite(str(out_dir / "errors.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return rows


def plot_conf(table, args, path):
    d = table[(table.tile_iou == args.op_tile_iou) & (table.merge_iou == args.op_merge_iou)].sort_values("conf")
    fig, ax = plt.subplots(figsize=(7, 4))
    for col, style in (("precision", "-o"), ("recall", "-s"), ("F1", "-^")):
        ax.plot(d.conf, d[col], style, label=col, markersize=4)
    ax.axvline(args.op_conf, color="grey", ls="--", label=f"operating conf {args.op_conf}")
    ax.set_xlabel("confidence threshold")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Puck detector vs confidence (tile NMS {args.op_tile_iou}, merge NMS {args.op_merge_iou})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_nms(ap_table, table, args, path):
    tiles, merges = sorted(ap_table.tile_iou.unique()), sorted(ap_table.merge_iou.unique())
    ap = ap_table.pivot(index="tile_iou", columns="merge_iou", values="AP50").loc[tiles, merges].values
    op = table[table.conf == args.op_conf]
    fp = op.pivot(index="tile_iou", columns="merge_iou", values="FP").loc[tiles, merges].values
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, m, title, fmt in ((axes[0], ap, "AP@50 (all confidences)", "{:.2f}"),
                              (axes[1], fp, f"false positives at conf {args.op_conf}", "{:.0f}")):
        ax.imshow(m, cmap="viridis")
        for (i, j), v in np.ndenumerate(m):
            ax.text(j, i, fmt.format(v), ha="center", va="center", color="white", fontsize=10)
        ax.set_xticks(range(len(merges)), ["none" if x >= 1 else x for x in merges])
        ax.set_yticks(range(len(tiles)), tiles)
        ax.set_xlabel("merge NMS IoU (between tiles)")
        ax.set_ylabel("tile NMS IoU (Ultralytics)")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path("outputs/sweep") / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    sel = pd.read_csv(args.frames)
    sel = sel[sel["split"] == args.split]
    images = [cv2.imread(p) for p in sel["image"]]
    gts = [load_gt(p, im.shape[1], im.shape[0]) for p, im in zip(sel["image"], images)]
    print(f"{len(images)} {args.split} frames, {sum(len(g) for g in gts)} labelled pucks")

    rows, ap_rows, op_frames = [], [], None
    for t_iou in args.tile_iou:
        raw = [predict_tiles(model, im, args.tile, args.overlap, t_iou) for im in images]  # the model runs here
        for m_iou in args.merge_iou:
            frames = [(*merge(b, s, m_iou), g) for (b, s), g in zip(raw, gts)]
            ap_rows.append({"tile_iou": t_iou, "merge_iou": m_iou, "AP50": ap50(frames)})
            for conf in args.confs:
                rows.append({"tile_iou": t_iou, "merge_iou": m_iou, "conf": conf, **score(frames, conf)})
            if t_iou == args.op_tile_iou and m_iou == args.op_merge_iou:
                op_frames = frames
        print(f"tile NMS {t_iou}: done")
    table, ap_table = pd.DataFrame(rows), pd.DataFrame(ap_rows)
    table.to_csv(out_dir / "sweep.csv", index=False)
    ap_table.to_csv(out_dir / "ap.csv", index=False)
    plot_conf(table, args, out_dir / "pr_vs_conf.png")
    plot_nms(ap_table, table, args, out_dir / "nms_effect.png")

    op = table[(table.tile_iou == args.op_tile_iou) & (table.merge_iou == args.op_merge_iou)
               & (table.conf == args.op_conf)].iloc[0].to_dict()
    best = table.sort_values(["F1", "conf"], ascending=[False, False]).iloc[0].to_dict()
    errors = error_sheet(images, op_frames, sel["frame"].tolist(), args.op_conf, out_dir) if op_frames else []
    summary = {
        "weights": args.weights, "split": args.split, "frames": len(images), "pucks": int(sum(len(g) for g in gts)),
        "operating_point": {"conf": args.op_conf, "tile_iou": args.op_tile_iou, "merge_iou": args.op_merge_iou,
                            **op, "AP50": float(ap_table[(ap_table.tile_iou == args.op_tile_iou)
                                                         & (ap_table.merge_iou == args.op_merge_iou)].AP50.iloc[0])},
        "best_F1": best,
        "errors_at_operating_point": len(errors),
        "sanity_check": "the operating point uses the settings of eval_puck.py, so its TP/FP/FN and AP50 "
                        "must equal outputs/eval_puck/<run>/metrics.json for the same weights and frames",
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
