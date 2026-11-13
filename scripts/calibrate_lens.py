"""Lens calibration (plumb-line method): choose k1 so that straight rink lines become straight.

Why: a homography maps one plane to another and keeps straight lines straight. The action
camera of fillempje.mp4 has a wide-angle lens that bends straight lines (barrel distortion),
so homography.py first undoes it with a one-parameter radial model (homography.Lens):

    x_distorted = x * (1 + k1 * r^2)      x, r in units of the focal length f, from the image centre

H2 in docs/experiments.md chose k1 = -0.125 from a table that no script in the repo produced
(A1: "no calibration script"). This script makes that step reproducible.

Method ("plumb line"): find lines that are straight in reality, undistort them for a range of
k1 values, and keep the k1 for which they are straightest.
  1. Straight lines: the yellow kick plates along both side boards. They run the length of
     the rink, so they cross a large part of the image, where the distortion is strong.
  2. Find them in the median background (the empty rink, propose_puck.py) with an HSV
     threshold for yellow, below the advertising boards (y >= --min-y).
  3. One point per image column per side: the median row of the yellow pixels in that
     column. The stripe is several pixels thick, and we want its centre line, not its width.
  4. For each k1: undistort the points (Lens.undistort), fit a straight line by total least
     squares (cv2.fitLine), and measure the mean perpendicular distance of the points to it.
     Score = the mean of the two sides. Lower = straighter.

Caveat: the kick plate is on the boards, about 15 cm above the ice, not on the ice plane.
That does not matter here: the lens bends every straight line in 3D, wherever it is.

Outputs (in outputs/lens/<name>/, ignored by git):
  k1_scan.csv / k1_scan.png    deviation from a straight line per k1 (left, right, mean); `valid`
                               is False beyond the fold of the model; `cv2_error_px` is how far
                               the pipeline's Lens.undistort is from the exact inverse
  k1_scan_fine.csv             a finer scan around the best coarse k1
  mask.jpg                     the yellow points that were used, drawn on the background
  straightened.jpg             the background undistorted with the best k1 and with k1 = 0,
                               with each fitted line drawn, for a visual check
  summary.json                 best k1 and the scores at the configured k1

Usage:
  python scripts/calibrate_lens.py
  python scripts/calibrate_lens.py --background outputs/propose_puck/fillempje/background_color.jpg --name fillempje
"""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

from homography import Lens

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (after choosing the non-GUI backend)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--background", default="outputs/propose_puck/fillempje/background_color.jpg",
                   help="median background of the clip (empty rink), from propose_puck.py")
    p.add_argument("--keypoints", default="configs/keypoints_fillempje.json",
                   help="config with the lens f and centre (and the k1 in use, for comparison)")
    p.add_argument("--hsv-low", type=int, nargs=3, default=[18, 80, 100], help="lower HSV bound for yellow")
    p.add_argument("--hsv-high", type=int, nargs=3, default=[40, 255, 255], help="upper HSV bound for yellow")
    p.add_argument("--min-y", type=int, default=380, help="ignore rows above this (yellow advertising boards)")
    p.add_argument("--split-x", type=int, default=960, help="column that separates the left and right board")
    p.add_argument("--k1", type=float, nargs=3, default=[0.0, -0.3, -0.0125], help="scan: start stop step")
    p.add_argument("--name", default="fillempje")
    return p.parse_args()


def centre_line(mask, x0, x1):
    """One point per image column: the median row of the yellow pixels in that column."""
    pts = []
    for x in range(x0, x1):
        ys = np.flatnonzero(mask[:, x])
        if len(ys):
            pts.append((x, float(np.median(ys))))
    return np.array(pts, np.float64)


def straightness(pts):
    """Mean perpendicular distance (px) of the points to their total-least-squares line."""
    vx, vy, x0, y0 = cv2.fitLine(pts.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    d = np.abs((pts[:, 0] - x0) * vy - (pts[:, 1] - y0) * vx)  # |cross product| with the unit direction
    return float(d.mean()), (vx, vy, x0, y0)


def undistort_exact(lens, pts):
    """Exact inverse of Lens.distort, per point: solve r_u + k1 * r_u^3 = r_d for r_u.

    Lens.undistort uses cv2.undistortPoints, which stops after a fixed 5 iterations. That is
    exact near the image centre but up to ~2 px off at the image edge for k1 = -0.125, which
    is where the kick plates are. Here: Newton's method on the radius, 100 steps. The function
    g(r) = r + k1 r^3 - r_d is concave for k1 < 0, so starting at r = r_d (left of the root)
    every step stays left of the root and it converges without overshooting.

    Fold: for k1 < 0, r_d = r_u (1 + k1 r_u^2) only grows up to r_u = 1 / sqrt(3 |k1|), where
    r_d reaches 2 / (3 sqrt(3 |k1|)). A distorted point further out has no solution.
    Returns (undistorted points, inside = which points have a solution)."""
    k1, f, c = lens.k1, lens.K[0, 0], lens.K[:2, 2]
    n = (np.asarray(pts, np.float64) - c) / f
    rd = np.linalg.norm(n, axis=1)
    inside = np.ones(len(rd), bool) if k1 >= 0 else rd < 2 / (3 * np.sqrt(3 * -k1))
    ru = rd.copy()
    for _ in range(100):
        ru -= (ru + k1 * ru ** 3 - rd) / (1 + 3 * k1 * ru ** 2)
    scale = np.where(rd > 0, ru / np.maximum(rd, 1e-12), 1.0)
    return c + n * scale[:, None] * f, inside


def plain(row):
    """A pandas row as a dict of plain Python values (numpy bools/floats do not go into JSON)."""
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in row.to_dict().items()}


def draw_line(img, line, color):
    vx, vy, x0, y0 = line
    a, b = (x0 - 3000 * vx, y0 - 3000 * vy), (x0 + 3000 * vx, y0 + 3000 * vy)
    cv2.line(img, tuple(int(v) for v in a), tuple(int(v) for v in b), color, 1, cv2.LINE_AA)


def undistort_image(img, lens):
    """Remap the whole image with the lens model: every output pixel looks up where it was
    distorted to. (For the check image only; the pipeline undistorts points, not images.)"""
    h, w = img.shape[:2]
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    src = lens.distort(np.stack([xs.ravel(), ys.ravel()], 1)).reshape(h, w, 2).astype(np.float32)
    return cv2.remap(img, src[..., 0], src[..., 1], cv2.INTER_LINEAR)


def main():
    args = parse_args()
    out = Path("outputs/lens") / args.name
    out.mkdir(parents=True, exist_ok=True)
    cfg_lens = json.loads(Path(args.keypoints).read_text()).get("lens", {})
    f, centre, k1_cfg = cfg_lens.get("f", 1000.0), tuple(cfg_lens.get("center", (960.0, 540.0))), cfg_lens.get("k1")

    img = cv2.imread(args.background)
    mask = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), np.array(args.hsv_low), np.array(args.hsv_high))
    mask[:args.min_y] = 0
    sides = {"left": centre_line(mask, 0, args.split_x), "right": centre_line(mask, args.split_x, img.shape[1])}
    print({s: len(p) for s, p in sides.items()}, "columns with yellow")

    def score(k1):
        """Straightness per side, measured with the exact inverse of the lens model.
        valid = False when some kick-plate point lies beyond the fold of the model (see
        undistort_exact): it then has no undistorted position, so the k1 cannot be scored.
        cv2_error_px = how far Lens.undistort (what the pipeline uses) is from the exact inverse."""
        lens = Lens(float(k1), f, centre)
        dev, ok, cv_err = {}, True, 0.0
        for s, p in sides.items():
            u, inside = undistort_exact(lens, p)
            ok &= bool(inside.all())
            cv_err = max(cv_err, float(np.abs(lens.undistort(p) - u).max()))
            dev[s] = straightness(u)[0]
        return {"k1": float(k1), **{s: round(v, 3) for s, v in dev.items()},
                "mean": round(float(np.mean(list(dev.values()))), 3), "valid": ok,
                "cv2_error_px": round(cv_err, 3)}

    start, stop, step = args.k1
    k1s = np.round(np.arange(start, stop + step / 2, step), 4)
    if k1_cfg is not None and k1_cfg not in k1s:
        k1s = np.sort(np.append(k1s, k1_cfg))[::-1]
    scan = pd.DataFrame([score(k1) for k1 in k1s])
    scan.to_csv(out / "k1_scan.csv", index=False)
    print(scan.to_string(index=False))
    coarse = scan.loc[scan[scan.valid]["mean"].idxmin()]
    # refine: a finer scan of one step on either side of the coarse minimum
    fine = pd.DataFrame([score(k1) for k1 in np.round(np.linspace(coarse.k1 - abs(step), coarse.k1 + abs(step), 21), 5)])
    fine.to_csv(out / "k1_scan_fine.csv", index=False)
    best = fine.loc[fine[fine.valid]["mean"].idxmin()]
    print(f"\nfine scan around {coarse.k1}: best k1 = {best.k1} (mean {best['mean']} px)")

    fig, ax = plt.subplots(figsize=(7, 4))
    ok = scan[scan.valid]  # beyond the fold some points have no undistorted position
    for col in ("left", "right", "mean"):
        ax.plot(ok.k1, ok[col], marker="o" if col == "mean" else None, ms=3, label=col)
    ax.axvline(best.k1, color="grey", ls="--", label=f"best k1 = {best.k1}")
    if k1_cfg is not None:
        ax.axvline(k1_cfg, color="red", ls=":", label=f"config k1 = {k1_cfg}")
    ax.set(xlabel="k1", ylabel="mean distance to a straight line (px)", title="plumb-line scan: kick plates")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "k1_scan.png", dpi=120)
    plt.close(fig)

    vis = img.copy()
    for p in sides.values():
        for x, y in p.astype(int):
            vis[y, x] = (255, 0, 255)
    cv2.imwrite(str(out / "mask.jpg"), vis)

    # side by side: k1 = 0 (nothing corrected) and the best k1, each with the fitted lines
    panels = []
    for k1 in (0.0, float(best.k1)):
        lens = Lens(k1, f, centre)
        panel = undistort_image(img, lens) if k1 else img.copy()
        for p in sides.values():
            draw_line(panel, straightness(undistort_exact(lens, p)[0])[1], (0, 0, 255))
        cv2.putText(panel, f"k1 = {k1}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
        panels.append(cv2.resize(panel, None, fx=0.5, fy=0.5))
    cv2.imwrite(str(out / "straightened.jpg"), np.vstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 92])

    summary = {"background": args.background, "hsv_low": args.hsv_low, "hsv_high": args.hsv_high,
               "min_y": args.min_y, "columns": {s: len(p) for s, p in sides.items()},
               "f": f, "center": centre, "best": plain(best),
               "config_k1": k1_cfg, "at_config_k1": plain(scan[scan.k1 == k1_cfg].iloc[0]) if k1_cfg is not None else None}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"best": summary["best"], "at_config_k1": summary["at_config_k1"]}, indent=2))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
