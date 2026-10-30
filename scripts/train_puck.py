"""Fine-tune YOLO26 on the tiled puck dataset (from scripts/make_puck_dataset.py).

Goal: teach the model a new `puck` class. We start from COCO-pretrained weights
(transfer learning): the backbone already knows edges, blobs and textures, so
only a little data is needed to learn a new small object. All weights are
trained (no frozen layers): with ~200 crops one run takes minutes.

Augmentations, chosen for a 15 px object filmed from a fixed side of the rink:
  scale=0.2     default 0.5 can shrink the crop by 2x -> puck ~7 px, smaller than it
                ever is in our footage. 0.2 keeps it between ~12 and ~18 px.
  fliplr=0.5    a mirrored rink still looks like a rink: a free 2x on the data.
  flipud=0.0    upside-down ice with the crowd below never happens.
  degrees=0     the camera is not rolled; rotating would also bend the box.
  mosaic=1.0    4 crops in one image: more context variety and more puck-free
                ice per batch (default), switched off for the last 10 epochs.
  hsv_*         default colour jitter: the arena light is constant, but it helps
                against overfitting on 9 frames.

Outputs (in outputs/train/<name>/, ignored by git): the Ultralytics run folder
(weights/best.pt, results.csv, confusion_matrix.png, PR curves, val batches)
plus metrics.json with the final validation mAP of best.pt.

Usage:
  python scripts/train_puck.py --data data/puck_tiles/tiles640_v1/data.yaml
"""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="data.yaml of the tiled dataset")
    p.add_argument("--model", default="yolo26s.pt", help="pretrained start weights")
    p.add_argument("--imgsz", type=int, default=640, help="= tile size, so the puck keeps its native size")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=None, help="run name (default: <model>_puck_<imgsz>)")
    return p.parse_args()


def main():
    args = parse_args()
    name = args.name or f"{Path(args.model).stem}_puck_{args.imgsz}"
    project = Path("outputs/train").resolve()

    model = YOLO(args.model)
    model.train(data=args.data, imgsz=args.imgsz, epochs=args.epochs, batch=args.batch, seed=args.seed,
                project=str(project), name=name, exist_ok=True, workers=2, plots=True,
                scale=0.2, fliplr=0.5, flipud=0.0, degrees=0.0, mosaic=1.0)

    # validate best.pt once more, so the logged numbers are from the weights we use
    best = YOLO(project / name / "weights" / "best.pt")
    m = best.val(data=args.data, imgsz=args.imgsz, project=str(project), name=f"{name}_val", exist_ok=True)
    metrics = {"run": name, "model": args.model, "imgsz": args.imgsz, "epochs": args.epochs,
               "precision": round(float(m.box.mp), 3), "recall": round(float(m.box.mr), 3),
               "mAP50": round(float(m.box.map50), 3), "mAP50_95": round(float(m.box.map), 3)}
    (project / name / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
