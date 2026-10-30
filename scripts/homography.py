"""Homography + 2D minimap: map player foot points from the video onto a top-down rink.

Goal: for every processed frame, find the homography H_t that maps image pixels
to rink coordinates (feet), and use it to draw players on a 2D rink minimap.

Method, in two steps:
  1. Reference homography (H_ref). Rink keypoints with known NHL positions
     (faceoff dots, goal post) are measured by hand in ONE reference frame and
     stored in a config file. cv2.findHomography fits the 3x3 matrix that maps
     those pixels to rink feet. We use 5 points, one more than the minimum of
     4, so the reprojection error tells us how well the fit holds.
  2. Camera drift. The clip is filmed handheld, so the framing moves and H_ref is
     only correct for the reference frame. The camera mostly ROTATES (the person
     filming stays in their seat), and two images taken by a rotating camera are
     related by a homography as well, for the whole image, not just the ice. So
     per frame we match SIFT features to the reference frame and fit that
     frame-to-reference homography with RANSAC (which ignores moving players):
         H_t = H_ref @ H_(frame t -> reference frame)
     We always match to the reference frame, never to the previous frame, so
     errors do not accumulate over time. If matching fails (too few RANSAC
     inliers) we keep the last good homography and flag the frame.

Player positions: a person's foot point is the bottom centre of their box,
because only points that lie ON the ice obey the ice-plane homography. Feet that
land outside the rink outline are dropped: those are spectators, bench players
or false detections.

Outputs (in outputs/homography/<run-name>/, ignored by git):
  reference_check.jpg  reference frame with the rink template drawn back onto it:
                       if the lines match the real markings, H_ref is right
  check_sheet.jpg      the same overlay every few seconds, to check drift handling
  minimap.mp4          video (top) with the synced minimap (bottom)
  homographies.csv     per frame: RANSAC inliers, fallback flag, H_t (image -> rink)
  positions.csv        per person per frame: foot pixel, rink x/y in feet, on-ice flag
  summary.json         config, reprojection error, matching statistics

Usage:
  python scripts/homography.py --video Video/hockey_clip_4k_noaudio.mp4 \\
      --detections outputs/baseline/yolo26s_1280_rtx4060/detections.csv
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from rink import Minimap, boards_polyline, markings

MATCH_WIDTH = 1280  # frames are downscaled to this width for SIFT: 4K is slow and not more accurate here
OUT_WIDTH = 1280    # width of the video part of minimap.mp4
TEAM_DOT = (0, 140, 255)  # BGR, one colour for all persons until team classification exists


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, help="input .mp4")
    p.add_argument("--keypoints", default="configs/keypoints_hockey_clip_4k.json",
                   help="reference frame + measured rink keypoints")
    p.add_argument("--detections", required=True, help="detections.csv from scripts/baseline.py")
    p.add_argument("--stride", type=int, default=2, help="process every n-th frame (match the detections)")
    p.add_argument("--min-inliers", type=int, default=80,
                   help="fewer RANSAC inliers than this = matching failed, reuse the last good homography")
    p.add_argument("--rink-margin", type=float, default=2.0,
                   help="feet outside the boards still counted as on the ice (box bottoms are not exact)")
    p.add_argument("--name", default="sift_ref0", help="run name")
    return p.parse_args()


def read_frame(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"Could not read frame {idx}")
    return frame


def project(H, pts):
    """Apply a homography to Nx2 points. Also returns w, the homogeneous scale:
    points with w <= 0 lie behind the camera and must not be drawn."""
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    hom = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return hom[:, :2] / hom[:, 2:3], hom[:, 2]


def draw_template(img, H_rink_to_img, scale=1.0, thickness=2):
    """Draw the rink markings onto a video frame (inverse homography). This is the
    visual check of a homography: wrong H = lines next to the real markings."""
    for pts, col, closed in markings():
        if closed:
            pts = np.vstack([pts, pts[:1]])
        # densify so long straight lines bend correctly near the image edge
        dense = np.concatenate([np.linspace(a, b, 20, endpoint=False) for a, b in zip(pts[:-1], pts[1:])] + [pts[-1:]])
        xy, w = project(H_rink_to_img, dense)
        xy = xy * scale
        for a, b, wa, wb in zip(xy[:-1], xy[1:], w[:-1], w[1:]):
            if wa > 0 and wb > 0:
                cv2.line(img, tuple(np.int32(a)), tuple(np.int32(b)), col, thickness, cv2.LINE_AA)


class FrameMatcher:
    """Estimates the homography from any frame to the reference frame with SIFT + RANSAC."""

    def __init__(self, ref_frame):
        self.k = MATCH_WIDTH / ref_frame.shape[1]
        # S scales full-res pixels to match-res pixels; H_full = S^-1 @ H_small @ S
        self.S = np.diag([self.k, self.k, 1.0])
        self.sift = cv2.SIFT_create(nfeatures=5000)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.ref_kp, self.ref_desc = self.sift.detectAndCompute(self._small_gray(ref_frame), None)

    def _small_gray(self, frame):
        small = cv2.resize(frame, None, fx=self.k, fy=self.k, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    def to_reference(self, frame):
        """Returns (H frame -> reference in full-res pixels, number of RANSAC inliers)."""
        kp, desc = self.sift.detectAndCompute(self._small_gray(frame), None)
        if desc is None or len(kp) < 4:
            return None, 0
        # Lowe's ratio test: keep a match only if it is clearly better than the 2nd best
        pairs = self.matcher.knnMatch(desc, self.ref_desc, k=2)
        good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
        if len(good) < 4:
            return None, len(good)
        src = np.float32([kp[m.queryIdx].pt for m in good])
        dst = np.float32([self.ref_kp[m.trainIdx].pt for m in good])
        H_small, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransacReprojThreshold=2.0)
        if H_small is None:
            return None, 0
        return np.linalg.inv(self.S) @ H_small @ self.S, int(mask.sum())


def main():
    args = parse_args()
    out_dir = Path("outputs/homography") / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. reference homography from the measured keypoints ---
    cfg = json.loads(Path(args.keypoints).read_text())
    img_pts = np.float64([p["image"] for p in cfg["points"]])
    rink_pts = np.float64([p["rink"] for p in cfg["points"]])
    # method 0 = least squares over all points (no RANSAC: we trust our hand-measured points)
    H_ref, _ = cv2.findHomography(img_pts, rink_pts, 0)
    reproj, _ = project(H_ref, img_pts)
    errors_ft = np.linalg.norm(reproj - rink_pts, axis=1)
    print("Reference homography, reprojection error per keypoint (feet):")
    for p, e in zip(cfg["points"], errors_ft):
        print(f"  {p['name']:32s} {e:5.2f}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ref_idx = cfg["reference_frame"]
    ref_frame = read_frame(cap, ref_idx)
    h, w = ref_frame.shape[:2]

    check = ref_frame.copy()
    draw_template(check, np.linalg.inv(H_ref), thickness=4)
    back, _ = project(np.linalg.inv(H_ref), rink_pts)  # keypoints as predicted by H_ref
    for p, q in zip(img_pts, back):
        cv2.circle(check, tuple(np.int32(p)), 18, (0, 255, 0), 3)   # measured: green ring
        cv2.drawMarker(check, tuple(np.int32(q)), (255, 0, 255), cv2.MARKER_CROSS, 30, 3)  # predicted: magenta cross
    cv2.imwrite(str(out_dir / "reference_check.jpg"), check, [cv2.IMWRITE_JPEG_QUALITY, 85])

    # --- 2. per-frame homography via matching to the reference frame ---
    dets = pd.read_csv(args.detections)
    dets = dets[dets["class"] == "person"]
    boards = boards_polyline().astype(np.float32)
    matcher = FrameMatcher(ref_frame)
    minimap = Minimap(px_per_ft=OUT_WIDTH / 210)
    out_scale = OUT_WIDTH / w
    out_h = round(h * out_scale)
    mm_w, mm_h = minimap.size
    writer = cv2.VideoWriter(str(out_dir / "minimap.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps / args.stride, (OUT_WIDTH, out_h + mm_h))

    H_rows, pos_rows, check_tiles = [], [], []
    last_H_to_ref = None
    t0 = time.time()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for idx in range(n_frames):
        if not cap.grab():
            break
        if idx % args.stride:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break

        H_to_ref, inliers = matcher.to_reference(frame)
        fallback = H_to_ref is None or inliers < args.min_inliers
        if fallback:
            H_to_ref = last_H_to_ref if last_H_to_ref is not None else np.eye(3)
        else:
            last_H_to_ref = H_to_ref
        H_t = H_ref @ H_to_ref  # image pixels of frame idx -> rink feet
        H_rows.append({"frame": idx, "inliers": inliers, "fallback": fallback,
                       **{f"h{i}{j}": H_t[i, j] for i in range(3) for j in range(3)}})

        # project the foot points (bottom centre of each person box)
        fd = dets[dets["frame"] == idx]
        feet = np.stack([(fd["x1"] + fd["x2"]) / 2, fd["y2"]], axis=1) if len(fd) else np.zeros((0, 2))
        rink_xy, _ = project(H_t, feet) if len(feet) else (np.zeros((0, 2)), None)
        on_ice = [cv2.pointPolygonTest(boards, (float(x), float(y)), True) >= -args.rink_margin for x, y in rink_xy]
        for (u, v), (x, y), conf, ice in zip(feet, rink_xy, fd["conf"], on_ice):
            pos_rows.append({"frame": idx, "time_s": round(idx / fps, 3), "conf": conf, "foot_u": round(u, 1),
                             "foot_v": round(v, 1), "x_ft": round(x, 2), "y_ft": round(y, 2), "on_ice": ice})

        # draw: video with template overlay and foot points, minimap below it
        view = cv2.resize(frame, (OUT_WIDTH, out_h), interpolation=cv2.INTER_AREA)
        draw_template(view, np.linalg.inv(H_t), scale=out_scale, thickness=1)
        mm = minimap.background.copy()
        for (u, v), xy, ice in zip(feet, rink_xy, on_ice):
            col = TEAM_DOT if ice else (150, 150, 150)
            cv2.circle(view, (int(u * out_scale), int(v * out_scale)), 5, col, -1)
            if ice:
                cv2.circle(mm, tuple(int(c) for c in minimap.to_px(xy)[0]), 7, TEAM_DOT, -1, cv2.LINE_AA)
        label = f"f{idx}  {idx / fps:5.1f}s  inliers {inliers}" + ("  FALLBACK" if fallback else "")
        cv2.putText(view, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        canvas = np.full((out_h + mm_h, OUT_WIDTH, 3), 255, np.uint8)
        canvas[:out_h] = view
        canvas[out_h:, :mm_w] = mm
        writer.write(canvas)
        if idx % round(3 * fps) == 0:  # one check tile every ~3 s
            check_tiles.append(cv2.resize(view, (640, round(out_h / 2))))
        if len(H_rows) % 50 == 0:
            print(f"  {len(H_rows)} frames, {len(H_rows) / (time.time() - t0):.1f} fps")
    cap.release()
    writer.release()

    while len(check_tiles) % 3:
        check_tiles.append(np.full_like(check_tiles[0], 255))
    cv2.imwrite(str(out_dir / "check_sheet.jpg"),
                np.vstack([np.hstack(check_tiles[i:i + 3]) for i in range(0, len(check_tiles), 3)]),
                [cv2.IMWRITE_JPEG_QUALITY, 85])

    Hdf, pos = pd.DataFrame(H_rows), pd.DataFrame(pos_rows)
    Hdf.to_csv(out_dir / "homographies.csv", index=False)
    pos.to_csv(out_dir / "positions.csv", index=False)
    summary = {
        "video": args.video, "keypoints": args.keypoints, "detections": args.detections,
        "reference_frame": ref_idx, "stride": args.stride, "min_inliers": args.min_inliers,
        "keypoint_reprojection_error_ft": {p["name"]: round(float(e), 2) for p, e in zip(cfg["points"], errors_ft)},
        "frames": len(Hdf), "fallback_frames": int(Hdf["fallback"].sum()),
        "inliers_min": int(Hdf["inliers"].min()), "inliers_median": float(Hdf["inliers"].median()),
        "persons_per_frame": round(len(pos) / len(Hdf), 2),
        "persons_on_ice_per_frame": round(float(pos["on_ice"].sum()) / len(Hdf), 2) if len(pos) else 0,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
