"""Tracking over the whole clip (Extra Mile): ByteTrack for players, Kalman filter for the puck.

Until now every frame was handled on its own: the pipeline did not know that a player in
frame 100 is the same player in frame 102, and the puck blinked on the minimap whenever
the detector missed it. This script links the frames:

  players  Ultralytics ByteTrack (model.track, persist=True). ByteTrack predicts each track
           with a Kalman filter in the image, matches new boxes to the predictions by IoU,
           and also uses LOW-confidence boxes in a second round to keep a track alive
           through partial occlusion. Each player keeps an ID.
  puck     our own Kalman filter on the rink, in metres (scripts/kalman.py): it bridges
           short gaps by predicting with the puck's velocity, and gating rejects detections
           that are physically impossible jumps.

Two phases, so the filter can be tuned without running the detectors again:
  1. detect: one pass over the video -> players.csv (with track IDs) and puck_raw.csv
     (every puck candidate, mapped to the rink). Skipped if the files exist (use --redo).
  2. filter + measure: Kalman over puck_raw.csv, statistics, and a gap test.

Gap test: how good is the prediction while the puck is hidden? We take frames where the
puck WAS detected, hide those detections in blocks of k frames, and compare the filter's
prediction with the hidden detection (in metres). Baseline: keep the puck at its last
seen position. This measures exactly what the filter adds.

Outputs (outputs/track/<name>/, ignored by git):
  players.csv, puck_raw.csv, puck_track.csv, summary.json, minimap_compare.mp4
  (left: per-frame detections as before, right: with tracking)

Usage:
  python scripts/track.py --video Video/fillempje.mp4 --keypoints configs/keypoints_fillempje.json --puck-weights outputs/train/fillempje_v3_yolo26s_puck_640/weights/best.pt --name fillempje_E2a
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from eval_puck import predict_tiled
from homography import RinkMapper
from kalman import run as run_kalman
from rink import Minimap

PERSON = 0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--keypoints", required=True, help="keypoint + lens config of this camera")
    p.add_argument("--puck-weights", required=True, help="fine-tuned puck model (best.pt)")
    p.add_argument("--person-model", default="yolo26s.pt")
    p.add_argument("--person-imgsz", type=int, default=1280)
    p.add_argument("--person-conf", type=float, default=0.1, help="low on purpose: ByteTrack uses low boxes too")
    p.add_argument("--tracker", default="bytetrack.yaml",
                   help="ByteTrack config: Ultralytics default, or configs/bytetrack_hockey.yaml")
    p.add_argument("--puck-conf", type=float, default=0.25)
    p.add_argument("--stride", type=int, default=2, help="process every n-th frame")
    p.add_argument("--rink-margin", type=float, default=1.0, help="metres outside the boards still on the ice")
    # Kalman parameters (see kalman.py)
    p.add_argument("--accel-std", type=float, default=20.0)
    p.add_argument("--meas-std", type=float, default=0.7)
    p.add_argument("--max-coast", type=float, default=0.5)
    p.add_argument("--init-conf", type=float, default=0.4)
    p.add_argument("--no-video", action="store_true", help="skip the minimap video")
    p.add_argument("--redo", action="store_true", help="run the detectors again")
    p.add_argument("--name", default="track")
    return p.parse_args()


def detect(args, mapper, out):
    person_model, puck_model = YOLO(args.person_model), YOLO(args.puck_weights)
    cap = cv2.VideoCapture(args.video)
    players, pucks, i = [], [], -1
    while cap.grab():
        i += 1
        if i % args.stride:
            continue
        frame = cap.retrieve()[1]
        r = person_model.track(frame, persist=True, tracker=args.tracker, imgsz=args.person_imgsz,
                               conf=args.person_conf, classes=[PERSON], verbose=False)[0]
        b = r.boxes
        if len(b):
            xyxy, ids = b.xyxy.cpu().numpy(), (b.id.cpu().numpy().astype(int) if b.id is not None else [-1] * len(b))
            feet = np.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, xyxy[:, 3]], axis=1)
            rink = mapper.image_to_rink(feet)
            for (u, v), (x, y), c, tid, ice in zip(feet, rink, b.conf.cpu().numpy(), ids, mapper.on_ice(rink, args.rink_margin)):
                players.append({"frame": i, "id": int(tid), "conf": round(float(c), 3), "u": round(u, 1), "v": round(v, 1),
                                "x": round(x, 2), "y": round(y, 2), "on_ice": ice})
        boxes, conf = predict_tiled(puck_model, frame, 640, 160, args.puck_conf)
        if len(boxes):
            bottom = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]], axis=1)
            rink = mapper.image_to_rink(bottom)
            for (u, v), (x, y), c, ice in zip(bottom, rink, conf, mapper.on_ice(rink, args.rink_margin)):
                pucks.append({"frame": i, "conf": round(float(c), 3), "u": round(u, 1), "v": round(v, 1),
                              "x": round(x, 2), "y": round(y, 2), "on_ice": ice})
        if i % 600 == 0:
            print(f"frame {i}")
    cap.release()
    pd.DataFrame(players).to_csv(out / "players.csv", index=False)
    pd.DataFrame(pucks).to_csv(out / "puck_raw.csv", index=False)
    return i + 1


def dets_by_frame(raw):
    return {f: list(zip(g.x, g.y, g.conf)) for f, g in raw.groupby("frame")}


def gap_test(frames, dets, ref, dt, stride, params, k):
    """Hide detections in blocks of k processed frames (every 3rd block) and compare the
    filter's position with the hidden detection that the full run used."""
    hidden = {f for j, f in enumerate(frames) if (j // k) % 3 == 1}
    rows = run_kalman(frames, {f: d for f, d in dets.items() if f not in hidden}, dt, **params)
    errs, hold_errs, shown, last = [], [], 0, None
    for r in rows:
        f = r["frame"]
        if r["state"] in ("measured", "started"):
            last = (r["x"], r["y"])
        if f in hidden and f in ref:  # the full run measured the puck here
            if r["x"] is not None:
                shown += 1
                errs.append(np.hypot(r["x"] - ref[f][0], r["y"] - ref[f][1]))
            if last is not None:
                hold_errs.append(np.hypot(last[0] - ref[f][0], last[1] - ref[f][1]))
    n = sum(1 for f in hidden if f in ref)
    return {"gap_frames": k, "gap_s": round(k * stride * dt, 2), "hidden_detections": n,
            "still_shown_pct": round(100 * shown / max(n, 1), 1),
            "kalman_median_err_m": round(float(np.median(errs)), 2) if errs else None,
            "hold_last_median_err_m": round(float(np.median(hold_errs)), 2) if hold_errs else None}


def draw_video(out, players, raw, track, frames, spec, fps):
    import imageio_ffmpeg
    mm = Minimap(spec, width_px=900)
    pl = {f: g for f, g in players[players.on_ice].groupby("frame")}
    rw = {f: g for f, g in raw.groupby("frame")}
    tr = track.set_index("frame")
    rng = np.random.default_rng(0)
    colours = {}
    trail = {}
    writer = None
    for f in frames:
        left, right = mm.background.copy(), mm.background.copy()
        g = pl.get(f)
        if g is not None:
            for _, p in g.iterrows():
                px = tuple(int(v) for v in mm.to_px((p.x, p.y))[0])
                cv2.circle(left, px, 7, (0, 140, 255), -1, cv2.LINE_AA)
                if p.id < 0:
                    continue
                col = colours.setdefault(p.id, tuple(int(c) for c in rng.integers(40, 220, 3)))
                trail.setdefault(p.id, []).append((f, px))
                trail[p.id] = [t for t in trail[p.id] if f - t[0] <= fps * 2][-30:]  # last ~1 s
                pts = np.array([t[1] for t in trail[p.id]], np.int32)
                if len(pts) > 1:
                    cv2.polylines(right, [pts], False, col, 2, cv2.LINE_AA)
                cv2.circle(right, px, 7, col, -1, cv2.LINE_AA)
                cv2.putText(right, str(p.id), (px[0] + 8, px[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1)
        r = rw.get(f)
        if r is not None and r.on_ice.any():
            best = r[r.on_ice].sort_values("conf").iloc[-1]
            cv2.circle(left, tuple(int(v) for v in mm.to_px((best.x, best.y))[0]), 6, (0, 0, 0), -1, cv2.LINE_AA)
        t = tr.loc[f]
        if pd.notna(t.x):
            px = tuple(int(v) for v in mm.to_px((t.x, t.y))[0])
            cv2.circle(right, px, 6, (0, 0, 0), -1 if t.state != "predicted" else 2, cv2.LINE_AA)
        for img, label in ((left, "per frame (before)"), (right, "tracked: IDs + Kalman puck (ring = predicted)")):
            cv2.putText(img, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
        canvas = np.hstack([left, right])
        canvas = canvas[: canvas.shape[0] // 2 * 2, : canvas.shape[1] // 2 * 2]
        if writer is None:
            writer = imageio_ffmpeg.write_frames(str(out / "minimap_compare.mp4"), (canvas.shape[1], canvas.shape[0]),
                                                 fps=fps, codec="libx264", macro_block_size=1, quality=7)
            writer.send(None)
        writer.send(np.ascontiguousarray(canvas[:, :, ::-1]))
    if writer is not None:
        writer.close()


def main():
    args = parse_args()
    out = Path("outputs/track") / args.name
    out.mkdir(parents=True, exist_ok=True)
    mapper = RinkMapper(json.loads(Path(args.keypoints).read_text()))
    fps = cv2.VideoCapture(args.video).get(cv2.CAP_PROP_FPS)
    if args.redo or not (out / "puck_raw.csv").exists():
        detect(args, mapper, out)
    players, raw = pd.read_csv(out / "players.csv"), pd.read_csv(out / "puck_raw.csv")
    frames = sorted(set(range(0, int(max(players.frame.max(), raw.frame.max())) + 1, args.stride)))
    dt = 1.0 / fps
    params = dict(accel_std=args.accel_std, meas_std=args.meas_std, max_coast=args.max_coast, init_conf=args.init_conf)

    raw_ice = raw[raw.on_ice]
    dets = dets_by_frame(raw_ice)
    track = pd.DataFrame(run_kalman(frames, dets, dt, **params))
    track.to_csv(out / "puck_track.csv", index=False)

    # impossible jumps: the shown puck moves faster than 40 m/s between two processed frames
    def jumps(xy):
        xy = xy.dropna()
        d = np.hypot(np.diff(xy.x), np.diff(xy.y))
        gap = np.diff(xy.index) * dt
        return int(np.sum(d > 40.0 * gap))
    best_raw = raw_ice.sort_values("conf").groupby("frame").last()[["x", "y"]].reindex(frames)
    ref = {r.frame: (r.x, r.y) for r in track.itertuples() if r.state == "measured"}

    ice_players = players[players.on_ice & (players.id >= 0)]
    life = ice_players.groupby("id").frame.agg(lambda s: (s.max() - s.min()) * dt)
    n = len(frames)
    summary = {
        "frames_processed": n,
        "players": {
            "detections_on_ice_per_frame": round(len(players[players.on_ice]) / n, 2),
            "unique_ids_on_ice": int(ice_players.id.nunique()),
            "ids_longer_than_5s": int((life > 5).sum()),
            "median_track_life_s": round(float(life.median()), 2),
            "share_of_on_ice_detections_with_id_pct": round(100 * (players.on_ice & (players.id >= 0)).sum() / max(players.on_ice.sum(), 1), 1),
        },
        "puck": {
            "frames_with_detection_any_pct": round(100 * raw.frame.nunique() / n, 1),
            "frames_with_detection_on_ice_pct": round(100 * raw_ice.frame.nunique() / n, 1),
            "frames_shown_after_kalman_pct": round(100 * track.x.notna().mean(), 1),
            "of_which_measured_pct": round(100 * track.state.isin(["measured", "started"]).mean(), 1),
            "of_which_predicted_pct": round(100 * (track.state == "predicted").mean(), 1),
            "track_starts": int((track.state == "started").sum()),
            # frames with a detection that the gate refused (the filter kept predicting instead)
            "frames_detection_rejected_by_gate": int(((track.n_dets > 0) & (track.state == "predicted")).sum()),
            "impossible_jumps_raw": jumps(best_raw),
            "impossible_jumps_kalman": jumps(track.set_index("frame")[["x", "y"]]),
        },
        "kalman_params": params,
        "gap_test": [gap_test(frames, dets, ref, dt, args.stride, params, k) for k in (3, 8, 15)],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if not args.no_video:
        draw_video(out, players, raw_ice, track, frames, mapper.spec, fps / args.stride)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
