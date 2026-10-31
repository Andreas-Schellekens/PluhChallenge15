"""Compare the pretrained baseline with the fine-tuned puck model on full 4K frames.

Goal: show whether fine-tuning closed the data gap, measured the way the
pipeline will really be used: on whole frames, not on training-sized tiles.

The two models:
  baseline    COCO YOLO26 on the whole frame (imgsz 1280, as in E0); its
              "sports ball" class is counted as a puck prediction.
  finetuned   our puck model with TILED inference: the 4K frame is cut into
              overlapping 640x640 tiles (the same grid as the val tiles), each
              tile is predicted at native resolution, boxes are shifted back to
              frame coordinates, and duplicates from overlapping tiles are
              merged with NMS. This is the SAHI idea, written out by hand.

Scoring on the labelled frames of one split (default: val, never trained on):
  a prediction is a true positive if it overlaps an unmatched ground-truth puck
  with IoU >= 0.5 (same rule as mAP@50). Every other prediction is a false
  positive, every unmatched puck a false negative.
  - confusion matrix (puck / background) at --conf, like Ultralytics plots it
  - precision, recall and AP@50 over all confidences

Optional --video: run both models over the whole clip and plot, per frame,
whether a puck was found. Most of these frames have no label, so this is a
visual check, not a score.

Outputs (in outputs/eval_puck/<name>/, ignored by git):
  metrics.json           TP/FP/FN, precision, recall, AP@50 per model
  confusion_matrix.png   both 2x2 matrices side by side
  comparison.jpg         per frame: zoom on the true puck (baseline | finetuned)
                         and the full frame with all finetuned detections
  timeline.png, clip_detections.csv   (only with --video)

Usage:
  python scripts/eval_puck.py --finetuned outputs/train/yolo26s_puck_640/weights/best.pt \\
      --frames data/puck_tiles/tiles640_v1/frames.csv --video Video/hockey_clip_4k_noaudio.mp4
"""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
from ultralytics import YOLO

from make_puck_dataset import load_boxes

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SPORTS_BALL = 32
GREEN, RED, ORANGE = (0, 200, 0), (0, 0, 255), (0, 140, 255)  # BGR: truth, baseline, finetuned


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--finetuned", required=True, help="best.pt of the puck model")
    p.add_argument("--frames", required=True, help="frames.csv from make_puck_dataset.py")
    p.add_argument("--split", default="val", help="which frames to score")
    p.add_argument("--baseline", default="yolo26s.pt")
    p.add_argument("--baseline-imgsz", type=int, default=1280)
    p.add_argument("--tile", type=int, default=640)
    p.add_argument("--overlap", type=int, default=160)
    p.add_argument("--conf", type=float, default=0.25, help="threshold for the confusion matrix")
    p.add_argument("--video", default=None, help="optional: also run over the whole clip")
    p.add_argument("--stride", type=int, default=2, help="frame stride for --video")
    p.add_argument("--name", default="baseline_vs_finetuned")
    return p.parse_args()


def grid_origins(size, tile, overlap):
    step = tile - overlap
    xs = list(range(0, size - tile + 1, step))
    if xs[-1] != size - tile:
        xs.append(size - tile)
    return xs


def predict_baseline(model, img, imgsz, conf):
    r = model.predict(img, imgsz=imgsz, conf=conf, classes=[SPORTS_BALL], verbose=False)[0]
    return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()


def predict_tiled(model, img, tile, overlap, conf):
    """Predict every overlapping tile, shift boxes to frame coordinates, merge with NMS."""
    h, w = img.shape[:2]
    origins = [(x, y) for y in grid_origins(h, tile, overlap) for x in grid_origins(w, tile, overlap)]
    results = model.predict([img[y:y + tile, x:x + tile] for x, y in origins], imgsz=tile, conf=conf,
                            verbose=False, batch=16)
    boxes, scores = [], []
    for (x0, y0), r in zip(origins, results):
        boxes.append(r.boxes.xyxy.cpu().numpy() + [x0, y0, x0, y0])
        scores.append(r.boxes.conf.cpu().numpy())
    boxes, scores = np.concatenate(boxes), np.concatenate(scores)
    if len(boxes) == 0:
        return boxes.reshape(0, 4), scores
    xywh = np.column_stack([boxes[:, :2], boxes[:, 2:] - boxes[:, :2]]).tolist()
    keep = np.array(cv2.dnn.NMSBoxes(xywh, scores.tolist(), conf, 0.3), dtype=int).reshape(-1)
    return boxes[keep], scores[keep]


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def match(preds, scores, gts, thr=0.5):
    """Greedy matching, highest score first. Returns a TP flag per prediction."""
    tp, used = np.zeros(len(preds), bool), set()
    for i in np.argsort(-scores):
        best, best_j = thr, None
        for j, g in enumerate(gts):
            if j not in used and iou(preds[i], g) >= best:
                best, best_j = iou(preds[i], g), j
        if best_j is not None:
            tp[i] = True
            used.add(best_j)
    return tp


def average_precision(all_scores, all_tp, n_gt):
    """AP@50: area under the precision-recall curve (all-point interpolation, as in COCO/VOC)."""
    if n_gt == 0 or len(all_scores) == 0:
        return 0.0
    order = np.argsort(-all_scores)
    tp = np.cumsum(all_tp[order])
    fp = np.cumsum(~all_tp[order])
    recall, precision = tp / n_gt, tp / (tp + fp)
    r = np.concatenate([[0], recall, [recall[-1]]])
    p = np.concatenate([[1], precision, [0]])
    p = np.maximum.accumulate(p[::-1])[::-1]  # precision envelope
    return float(np.sum((r[1:] - r[:-1]) * p[1:]))


def load_gt(image_path, w, h):
    label = Path(str(image_path).replace("images", "labels")).with_suffix(".txt")
    return np.array(load_boxes(label, w, h)).reshape(-1, 4)


def zoom(img, center, boxes_cols, size=160, out=320):
    """Crop around a point, scale up, draw boxes (list of (boxes, colour))."""
    cx, cy = int(center[0]), int(center[1])
    x0, y0 = min(max(0, cx - size // 2), img.shape[1] - size), min(max(0, cy - size // 2), img.shape[0] - size)
    c = cv2.resize(img[y0:y0 + size, x0:x0 + size], (out, out), interpolation=cv2.INTER_CUBIC)
    k = out / size
    for boxes, col in boxes_cols:
        for b in boxes:
            cv2.rectangle(c, (int((b[0] - x0) * k) - 3, int((b[1] - y0) * k) - 3),
                          (int((b[2] - x0) * k) + 3, int((b[3] - y0) * k) + 3), col, 2)
    return c


def plot_confusion(results, path):
    fig, axes = plt.subplots(1, len(results), figsize=(4.2 * len(results), 4))
    for ax, (name, r) in zip(np.atleast_1d(axes), results.items()):
        # rows = predicted, columns = true (Ultralytics convention); background/background is undefined
        m = np.array([[r["TP"], r["FP"]], [r["FN"], 0]])
        ax.imshow(m, cmap="Blues", vmin=0, vmax=max(1, m.max()))
        for (i, j), v in np.ndenumerate(m):
            ax.text(j, i, "-" if (i, j) == (1, 1) else str(v), ha="center", va="center", fontsize=16,
                    color="white" if v > m.max() / 2 else "black")
        ax.set_xticks([0, 1], ["puck", "background"])
        ax.set_yticks([0, 1], ["puck", "background"])
        ax.set_xlabel("true")
        ax.set_ylabel("predicted")
        ax.set_title(f"{name}\nP={r['precision']:.2f}  R={r['recall']:.2f}  AP50={r['AP50']:.2f}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path("outputs/eval_puck") / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    models = {"baseline": YOLO(args.baseline), "finetuned": YOLO(args.finetuned)}

    def predict(name, img, conf):
        if name == "baseline":
            return predict_baseline(models[name], img, args.baseline_imgsz, conf)
        return predict_tiled(models[name], img, args.tile, args.overlap, conf)

    frames = pd.read_csv(args.frames)
    frames = frames[frames["split"] == args.split]
    collected = {n: {"scores": [], "tp": [], "TP": 0, "FP": 0, "FN": 0} for n in models}
    n_gt, rows = 0, []
    for _, fr in frames.iterrows():
        img = cv2.imread(fr["image"])
        gts = load_gt(fr["image"], img.shape[1], img.shape[0])
        n_gt += len(gts)
        preds_at_conf = {}
        for name in models:
            # low threshold for the PR curve / AP, then the --conf subset for the confusion matrix
            boxes, scores = predict(name, img, 0.01)
            tp = match(boxes, scores, gts)
            collected[name]["scores"] += list(scores)
            collected[name]["tp"] += list(tp)
            keep = scores >= args.conf
            tp_c = match(boxes[keep], scores[keep], gts)
            collected[name]["TP"] += int(tp_c.sum())
            collected[name]["FP"] += int((~tp_c).sum())
            collected[name]["FN"] += len(gts) - int(tp_c.sum())
            preds_at_conf[name] = (boxes[keep], scores[keep])
            print(f"f{fr['frame']} {name:9s}: {keep.sum()} detections >= {args.conf}, {int(tp_c.sum())}/{len(gts)} pucks found")

        # visual: zoom on the true puck for both models + full frame with all finetuned detections
        center = (gts[0, :2] + gts[0, 2:]) / 2 if len(gts) else (img.shape[1] / 2, img.shape[0] / 2)
        tiles = [zoom(img, center, [(gts, GREEN), (preds_at_conf[n][0], col)])
                 for n, col in (("baseline", RED), ("finetuned", ORANGE))]
        for t, n in zip(tiles, ("baseline", "finetuned")):
            cv2.putText(t, n, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        full = img.copy()
        for b in gts:
            cv2.circle(full, tuple(np.int32((b[:2] + b[2:]) / 2)), 45, GREEN, 6)
        for b, s in zip(*preds_at_conf["finetuned"]):
            cv2.rectangle(full, tuple(np.int32(b[:2] - 20)), tuple(np.int32(b[2:] + 20)), ORANGE, 6)
            cv2.putText(full, f"{s:.2f}", tuple(np.int32(b[:2] - 30)), cv2.FONT_HERSHEY_SIMPLEX, 2, ORANGE, 5)
        full = cv2.resize(full, (569, 320), interpolation=cv2.INTER_AREA)
        cv2.putText(full, f"f{fr['frame']}", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        rows.append(np.hstack(tiles + [full]))
    cv2.imwrite(str(out_dir / "comparison.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])

    results = {}
    for name, c in collected.items():
        tp, fp, fn = c["TP"], c["FP"], c["FN"]
        results[name] = {"TP": tp, "FP": fp, "FN": fn,
                         "precision": round(tp / (tp + fp), 3) if tp + fp else 0.0,
                         "recall": round(tp / (tp + fn), 3) if tp + fn else 0.0,
                         "AP50": round(average_precision(np.array(c["scores"]), np.array(c["tp"], bool), n_gt), 3)}
    plot_confusion(results, out_dir / "confusion_matrix.png")
    summary = {"split": args.split, "frames": frames["frame"].tolist(), "pucks": n_gt, "conf": args.conf,
               "iou_match": 0.5, "baseline": args.baseline, "baseline_imgsz": args.baseline_imgsz,
               "finetuned": args.finetuned, "tile": args.tile, "overlap": args.overlap, "results": results}

    if args.video:
        cap = cv2.VideoCapture(args.video)
        clip_rows, idx = [], 0
        while cap.grab():
            if idx % args.stride == 0:
                ok, img = cap.retrieve()
                if not ok:
                    break
                for name in models:
                    boxes, scores = predict(name, img, args.conf)
                    clip_rows.append({"frame": idx, "model": name, "n_det": len(boxes),
                                      "max_conf": round(float(scores.max()), 3) if len(scores) else 0.0})
                if idx % 100 == 0:
                    print(f"  clip frame {idx}")
            idx += 1
        cap.release()
        clip = pd.DataFrame(clip_rows)
        clip.to_csv(out_dir / "clip_detections.csv", index=False)
        fig, ax = plt.subplots(figsize=(12, 3))
        for k, (name, col) in enumerate((("baseline", "tab:red"), ("finetuned", "tab:orange"))):
            d = clip[(clip["model"] == name) & (clip["n_det"] > 0)]
            ax.scatter(d["frame"], np.full(len(d), k), c=d["max_conf"], cmap="Oranges" if k else "Reds",
                       vmin=0, vmax=1, marker="|", s=300)
            summary[f"clip_frames_with_puck_{name}"] = int(len(d))
        n_proc = clip["frame"].nunique()
        ax.set_yticks([0, 1], [f"baseline ({summary['clip_frames_with_puck_baseline']}/{n_proc})",
                               f"finetuned ({summary['clip_frames_with_puck_finetuned']}/{n_proc})"])
        ax.set_ylim(-0.6, 1.6)
        ax.set_xlabel("frame")
        ax.set_title(f"Frames with a puck detection (conf >= {args.conf}); darker = more confident")
        fig.tight_layout()
        fig.savefig(out_dir / "timeline.png", dpi=120)
        plt.close(fig)
        summary["clip_frames_processed"] = int(n_proc)

    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
