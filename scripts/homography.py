"""Homography + 2D minimap: map player foot points from the video onto a top-down rink.

Goal: for every processed frame, map image pixels to rink coordinates and use
that to draw players on a 2D rink minimap.

Method, in three parts:
  1. Lens correction (optional). A homography maps one PLANE to another and
     keeps straight lines straight. Wide-angle action cameras bend straight lines
     (barrel distortion), so we first undo that with a one-parameter radial model:
         x_distorted = x * (1 + k1 * r^2)     (x, r in units of the focal length f)
     k1 is chosen so that lines that are straight on the rink (the board edges)
     become straight in the image again ("plumb-line" calibration, see the log).
  2. Reference homography (H_ref). Rink keypoints with known positions (faceoff
     dots, crease corners, goal posts) are measured by hand in ONE reference frame
     and stored in a config file. cv2.findHomography fits the 3x3 matrix that maps
     those (undistorted) pixels to rink units.
  3. Camera motion. If the camera moves (handheld clip), each frame is matched to
     the reference frame with SIFT + RANSAC, which gives frame -> reference:
         rink = H_ref( undistort( H_(frame t -> reference)(pixel) ) )
     Every frame is matched to the reference frame, never to the previous one, so
     errors do not accumulate. With --drift none (static camera) this step is
     skipped and H_(frame t -> reference) is the identity.

Player positions: a person's foot point is the bottom centre of their box,
because only points that lie ON the ice obey the ice-plane homography. Feet that
land outside the rink outline are dropped (spectators, bench, false detections).

A homography is only defined up to scale, and that scale can be negative. We
fix its sign so that the homogeneous coordinate w is positive for the keypoints:
then w <= 0 really means "behind the camera" and those points are not drawn.

Outputs (in outputs/homography/<run-name>/, ignored by git):
  reference_check.jpg  reference frame with the rink template drawn back onto it:
                       if the lines match the real markings, the mapping is right
  check_sheet.jpg      the same overlay every few seconds, to check camera motion
  minimap.mp4          video (top) with the synced minimap (bottom)
  homographies.csv     per frame: RANSAC inliers, fallback flag, H_(frame -> reference)
  positions.csv        per person per frame: foot pixel, rink x/y, on-ice flag
  summary.json         config, reprojection error, matching statistics

Usage:
  python scripts/homography.py --video Video/fillempje.mp4 --keypoints configs/keypoints_fillempje.json \\
      --detections outputs/baseline/fillempje_yolo26s_1280/detections.csv --drift none
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from rink import RINKS, Minimap, boards_polyline, markings

MATCH_WIDTH = 1280  # frames are downscaled to this width for SIFT
OUT_WIDTH = 1280    # width of the video part of minimap.mp4
TEAM_DOT = (0, 140, 255)  # BGR, one colour for all persons until team classification exists


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, help="input .mp4")
    p.add_argument("--keypoints", default="configs/keypoints_hockey_clip_4k.json",
                   help="reference frame, rink type, lens model and measured rink keypoints")
    p.add_argument("--detections", required=True, help="detections.csv from scripts/baseline.py")
    p.add_argument("--stride", type=int, default=2, help="process every n-th frame (match the detections)")
    p.add_argument("--drift", choices=["sift", "none"], default="sift",
                   help="sift: correct camera motion per frame; none: static camera")
    p.add_argument("--min-inliers", type=int, default=80,
                   help="fewer RANSAC inliers than this = matching failed, reuse the last good homography")
    p.add_argument("--rink-margin", type=float, default=2.0,
                   help="distance outside the boards (rink units) still counted as on the ice")
    p.add_argument("--name", default=None, help="run name (default: <keypoints file>_<drift>)")
    return p.parse_args()


def apply_h(H, pts):
    """Apply a homography to Nx2 points. Also returns w, the homogeneous scale."""
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    hom = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return hom[:, :2] / hom[:, 2:3], hom[:, 2]


class Lens:
    """One-parameter radial lens model. k1 = 0 means a perfect pinhole camera."""

    def __init__(self, k1=0.0, f=1000.0, center=(960.0, 540.0)):
        self.k1 = k1
        self.K = np.array([[f, 0, center[0]], [0, f, center[1]], [0, 0, 1.0]])
        self.D = np.array([k1, 0, 0, 0], dtype=float)

    def undistort(self, pts):
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        if self.k1 == 0 or len(pts) == 0:
            return pts.reshape(-1, 2)
        return cv2.undistortPoints(pts, self.K, self.D, P=self.K).reshape(-1, 2)

    def distort(self, pts):
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        f, c = self.K[0, 0], self.K[:2, 2]
        n = (pts - c) / f
        return n * (1 + self.k1 * (n ** 2).sum(1, keepdims=True)) * f + c


class RinkMapper:
    """Everything needed to go between image pixels and rink coordinates."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.spec = RINKS[cfg.get("rink", "nhl")]
        lens = cfg.get("lens", {})
        self.lens = Lens(lens.get("k1", 0.0), lens.get("f", 1000.0), tuple(lens.get("center", (960.0, 540.0))))
        self.img_pts = np.float64([p["image"] for p in cfg["points"]])
        self.rink_pts = np.float64([p["rink"] for p in cfg["points"]])
        # method 0 = least squares over all points (no RANSAC: we trust our hand-measured points)
        H, _ = cv2.findHomography(self.lens.undistort(self.img_pts), self.rink_pts, 0)
        if apply_h(np.linalg.inv(H), self.rink_pts)[1].mean() < 0:
            H = -H  # same homography, but now w > 0 for points in front of the camera
        self.H_ref = H
        self.H_ref_inv = np.linalg.inv(H)

    def keypoint_errors(self):
        rink, _ = apply_h(self.H_ref, self.lens.undistort(self.img_pts))
        return np.linalg.norm(rink - self.rink_pts, axis=1)

    def image_to_rink(self, pts, H_to_ref=None):
        ref = apply_h(H_to_ref, pts)[0] if H_to_ref is not None else np.asarray(pts, float).reshape(-1, 2)
        return apply_h(self.H_ref, self.lens.undistort(ref))[0]

    def rink_to_image(self, pts, H_to_ref=None):
        """Returns image points and a mask of points in front of the camera."""
        und, w = apply_h(self.H_ref_inv, pts)
        img = self.lens.distort(und)
        if H_to_ref is not None:
            img = apply_h(np.linalg.inv(H_to_ref), img)[0]
        return img, w > 0

    def on_ice(self, rink_xy, margin):
        boards = boards_polyline(self.spec).astype(np.float32)
        return [cv2.pointPolygonTest(boards, (float(x), float(y)), True) >= -margin for x, y in rink_xy]

    def draw_template(self, img, H_to_ref=None, scale=1.0, thickness=2):
        """Draw the rink markings onto a video frame. This is the visual check of
        the mapping: a wrong homography = lines next to the real markings."""
        h, w = img.shape[:2]
        for pts, col, closed in markings(self.spec):
            if closed:
                pts = np.vstack([pts, pts[:1]])
            # densify: straight rink lines become curves in a distorted image
            dense = np.concatenate([np.linspace(a, b, 30, endpoint=False) for a, b in zip(pts[:-1], pts[1:])]
                                   + [pts[-1:]])
            xy, ok = self.rink_to_image(dense, H_to_ref)
            xy = xy * scale
            for a, b, oa, ob in zip(xy[:-1], xy[1:], ok[:-1], ok[1:]):
                if oa and ob and np.abs(a).max() < 4 * max(w, h) and np.abs(b).max() < 4 * max(w, h):
                    cv2.line(img, tuple(np.int32(a)), tuple(np.int32(b)), col, thickness, cv2.LINE_AA)


def read_frame(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok:
        raise SystemExit(f"Could not read frame {idx}")
    return frame


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
    name = args.name or f"{Path(args.keypoints).stem.replace('keypoints_', '')}_{args.drift}"
    out_dir = Path("outputs/homography") / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1+2. lens model and reference homography from the measured keypoints ---
    cfg = json.loads(Path(args.keypoints).read_text())
    mapper = RinkMapper(cfg)
    errors = mapper.keypoint_errors()
    print(f"Reference homography ({mapper.spec.name}, k1={mapper.lens.k1}), "
          f"reprojection error per keypoint ({mapper.spec.unit}):")
    for p, e in zip(cfg["points"], errors):
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
    mapper.draw_template(check, thickness=max(2, w // 960))
    back, _ = mapper.rink_to_image(mapper.rink_pts)  # keypoints as predicted by the mapping
    for p, q in zip(mapper.img_pts, back):
        cv2.circle(check, tuple(np.int32(p)), max(8, w // 200), (0, 255, 0), 3)   # measured: green ring
        cv2.drawMarker(check, tuple(np.int32(q)), (255, 0, 255), cv2.MARKER_CROSS, max(15, w // 120), 3)
    cv2.imwrite(str(out_dir / "reference_check.jpg"), check, [cv2.IMWRITE_JPEG_QUALITY, 85])

    # --- 3. per-frame mapping, players, video ---
    dets = pd.read_csv(args.detections)
    dets = dets[dets["class"] == "person"]
    matcher = FrameMatcher(ref_frame) if args.drift == "sift" else None
    minimap = Minimap(mapper.spec, width_px=OUT_WIDTH)
    out_scale = OUT_WIDTH / w
    out_h = round(h * out_scale)
    mm_w, mm_h = minimap.size
    writer = cv2.VideoWriter(str(out_dir / "minimap.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps / args.stride, (OUT_WIDTH, out_h + mm_h))

    H_rows, pos_rows, check_tiles = [], [], []
    last_H_to_ref = np.eye(3)
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

        if matcher is None:
            H_to_ref, inliers, fallback = np.eye(3), -1, False
        else:
            H_to_ref, inliers = matcher.to_reference(frame)
            fallback = H_to_ref is None or inliers < args.min_inliers
            if fallback:
                H_to_ref = last_H_to_ref
            else:
                last_H_to_ref = H_to_ref
        H_rows.append({"frame": idx, "inliers": inliers, "fallback": fallback,
                       **{f"h{i}{j}": H_to_ref[i, j] for i in range(3) for j in range(3)}})

        # project the foot points (bottom centre of each person box)
        fd = dets[dets["frame"] == idx]
        feet = np.stack([(fd["x1"] + fd["x2"]) / 2, fd["y2"]], axis=1) if len(fd) else np.zeros((0, 2))
        rink_xy = mapper.image_to_rink(feet, H_to_ref) if len(feet) else np.zeros((0, 2))
        on_ice = mapper.on_ice(rink_xy, args.rink_margin)
        for (u, v), (x, y), conf, ice in zip(feet, rink_xy, fd["conf"], on_ice):
            pos_rows.append({"frame": idx, "time_s": round(idx / fps, 3), "conf": conf, "foot_u": round(u, 1),
                             "foot_v": round(v, 1), "x": round(x, 2), "y": round(y, 2), "on_ice": ice})

        # draw: video with template overlay and foot points, minimap below it
        view = cv2.resize(frame, (OUT_WIDTH, out_h), interpolation=cv2.INTER_AREA)
        mapper.draw_template(view, H_to_ref, scale=out_scale, thickness=1)
        mm = minimap.background.copy()
        for (u, v), xy, ice in zip(feet, rink_xy, on_ice):
            col = TEAM_DOT if ice else (150, 150, 150)
            cv2.circle(view, (int(u * out_scale), int(v * out_scale)), 5, col, -1)
            if ice:
                cv2.circle(mm, tuple(int(c) for c in minimap.to_px(xy)[0]), 7, TEAM_DOT, -1, cv2.LINE_AA)
        label = f"f{idx}  {idx / fps:5.1f}s" + (f"  inliers {inliers}" if inliers >= 0 else "")
        label += "  FALLBACK" if fallback else ""
        cv2.putText(view, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        canvas = np.full((out_h + mm_h, OUT_WIDTH, 3), 255, np.uint8)
        canvas[:out_h] = view
        canvas[out_h:, :mm_w] = mm
        writer.write(canvas)
        if idx % round(10 * fps) < args.stride:  # one check tile every ~10 s
            check_tiles.append(cv2.resize(canvas, (640, round((out_h + mm_h) / 2))))
        if len(H_rows) % 200 == 0:
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
        "rink": mapper.spec.name, "unit": mapper.spec.unit, "lens_k1": mapper.lens.k1,
        "reference_frame": ref_idx, "stride": args.stride, "drift": args.drift, "min_inliers": args.min_inliers,
        "keypoint_reprojection_error": {p["name"]: round(float(e), 3) for p, e in zip(cfg["points"], errors)},
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
