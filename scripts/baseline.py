"""Baseline: run a pretrained (COCO) YOLO model on the clip, before any fine-tuning.

Goal: measure what an off-the-shelf detector already gives us and expose the
"data gap" from the brief. COCO has a `person` class (players + referees) and a
`sports ball` class, but no puck. We expect people to work well and the puck to
be (almost) never found -> that gap is what our annotation + fine-tuning fixes.

Outputs (in outputs/baseline/<run-name>/, ignored by git):
  annotated.mp4    video with boxes, downscaled to 1920px wide to keep it small
  detections.csv   one row per detection: frame, class, conf, box in original pixels
  summary.json     run config + per-class counts, used in the experiment log

Usage:
  python scripts/baseline.py --video Video/hockey_clip_4k_noaudio.mp4
  python scripts/baseline.py --video ... --imgsz 1280 --stride 2
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

# COCO class ids we care about. Everything else (e.g. 'chair' on the boards) is ignored.
PERSON, SPORTS_BALL = 0, 32
CLASS_NAMES = {PERSON: "person", SPORTS_BALL: "sports ball"}
COLORS = {PERSON: (0, 200, 0), SPORTS_BALL: (0, 0, 255)}  # BGR

OUT_WIDTH = 1920  # width of the preview video; 4K output would be hundreds of MB


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, help="input .mp4")
    p.add_argument("--model", default="yolo26s.pt", help="pretrained weights (downloaded automatically)")
    p.add_argument("--imgsz", type=int, default=640,
                   help="inference size. The 4K frame is resized so its long side = imgsz, "
                        "so at 640 the frame shrinks 6x and the puck becomes a few pixels.")
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    p.add_argument("--stride", type=int, default=1, help="process every n-th frame (speed)")
    p.add_argument("--name", default=None, help="run name (default: <model>_<imgsz>)")
    return p.parse_args()


def main():
    args = parse_args()
    run_name = args.name or f"{Path(args.model).stem}_{args.imgsz}"
    out_dir = Path("outputs/baseline") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = OUT_WIDTH / w
    out_size = (OUT_WIDTH, round(h * scale))
    writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps / args.stride, out_size)

    rows = []
    frame_idx, processed = 0, 0
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % args.stride == 0:
            # classes=[...] filters inside YOLO's NMS step, so other COCO classes never appear.
            result = model.predict(frame, imgsz=args.imgsz, conf=args.conf,
                                   classes=[PERSON, SPORTS_BALL], verbose=False)[0]
            preview = cv2.resize(frame, out_size, interpolation=cv2.INTER_AREA)
            for box, cls, conf in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(),
                                      result.boxes.conf.tolist()):
                cls = int(cls)
                x1, y1, x2, y2 = box  # in original 4K pixel coordinates
                rows.append({"frame": frame_idx, "time_s": frame_idx / fps, "class": CLASS_NAMES[cls],
                             "conf": round(conf, 3), "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                             "box_w": x2 - x1, "box_h": y2 - y1})
                p1 = (round(x1 * scale), round(y1 * scale))
                p2 = (round(x2 * scale), round(y2 * scale))
                cv2.rectangle(preview, p1, p2, COLORS[cls], 2)
                cv2.putText(preview, f"{CLASS_NAMES[cls]} {conf:.2f}", (p1[0], p1[1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[cls], 1)
            cv2.putText(preview, f"frame {frame_idx}  {args.model} imgsz={args.imgsz}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            writer.write(preview)
            processed += 1
            if processed % 50 == 0:
                print(f"  {processed} frames processed...")
        frame_idx += 1

    cap.release()
    writer.release()
    elapsed = time.time() - t0

    df = pd.DataFrame(rows, columns=["frame", "time_s", "class", "conf", "x1", "y1", "x2", "y2",
                                     "box_w", "box_h"])
    df.to_csv(out_dir / "detections.csv", index=False)

    # Summary numbers for the experiment log.
    people = df[df["class"] == "person"]
    balls = df[df["class"] == "sports ball"]
    summary = {
        "run": run_name,
        "video": args.video,
        "model": args.model,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "stride": args.stride,
        "frames_processed": processed,
        "seconds": round(elapsed, 1),
        "fps_inference": round(processed / elapsed, 2) if elapsed else None,
        "person_per_frame_mean": round(len(people) / processed, 2) if processed else 0,
        "person_conf_mean": round(people["conf"].mean(), 3) if len(people) else None,
        "person_box_height_px_median": round(people["box_h"].median(), 1) if len(people) else None,
        "frames_with_sports_ball": int(balls["frame"].nunique()),
        "sports_ball_detections": len(balls),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
