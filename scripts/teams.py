"""Team colours (Extra Mile, tactical analytics): K-means on jersey colours + occupancy heatmaps.

Until now every person on the ice was one orange dot. This script splits them into groups
(two teams and the referees) without any labels, from the colour of their jersey:

  1. jersey crop   the middle of the upper body of each person box (see torso()): below the
                   helmet, above the pants, away from the box edges where ice shows through.
  2. feature       the colour of that crop in CIELAB (L = lightness, a = green-red,
                   b = blue-yellow). Why Lab and not HSV: hue is a circle and is undefined
                   for white/grey/black, exactly the colours of a white team and a referee.
                   In Lab, equal distances are roughly equal colour differences, so the
                   plain Euclidean distance that K-means uses makes sense. See jersey_colour()
                   for the exact numbers, and main() for why they are standardised.
  3. K-means       k clusters on those 3 numbers (cv2.kmeans, k-means++ start, 10 restarts).
                   K-means only finds "k colour groups"; which group is a team and which is
                   the referees is read from the cluster colours afterwards.
  4. track vote    one detection can be wrong (a blue player half behind a white one), so
                   every detection takes the majority colour of the last second of its
                   ByteTrack track (class TrackVote). This is where tracking (T1) pays off.
  5. heatmaps      per group, a kernel density estimate (KDE) of the rink positions: every
                   position is replaced by a small Gaussian bump, and the bumps are summed.
                   On a grid that is a 2D histogram blurred with a Gaussian.

How we check it without labels:
  - elbow + silhouette for k = 2..5 (is 3 really the natural number of groups?)
  - track consistency: the share of detections that agree with their own track's majority.
    A person cannot change jersey, so every disagreement is a clustering error.
  - sample.jpg + sample.csv: N random crops with their cluster, to be labelled by a person.
    Put the true label in the `truth` column and run again with --score for a confusion matrix.

Outputs (outputs/teams/<name>/, ignored by git):
  teams.csv        every on-ice detection: frame, id, colour feature, cluster, cluster after the vote
  model.json       the cluster centres (Lab + BGR) -> used by the app to colour the minimap
  summary.json     all numbers above
  k_selection.png  elbow and silhouette
  colours.png      the detections in the a-b plane, coloured by cluster
  sheet_<c>.jpg    random crops per cluster (jersey crop outlined)
  sample.jpg/.csv  the random sample for human labelling
  heatmaps.jpg     occupancy per group on the rink

Usage:
  python scripts/teams.py --video Video/hockey_static_1080p.mp4 --players outputs/track/fillempje_teams/players.csv --keypoints configs/keypoints_hockey_static_1080p.json --name fillempje
"""

import argparse
import json
from collections import deque
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

from rink import RINKS, Minimap

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# Clear drawing colours (BGR) for clusters with a known name. A cluster centre itself is a dull
# colour (a white jersey in this rink is grey), so an unnamed cluster is drawn in its centre colour.
DRAW_COLOURS = {"white": (255, 255, 255), "blue": (200, 90, 30), "red": (40, 40, 200), "black": (40, 40, 40),
                "yellow": (0, 210, 240), "green": (60, 160, 60), "referee": (0, 200, 255), "other": (0, 200, 255)}


# ---------- the part the app reuses ----------
def torso(box):
    """Jersey region of a person box (x1, y1, x2, y2): the middle half of the width, and
    from 20% to 50% of the height. The top 20% is the helmet, below 50% are pants and legs."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return int(x1 + 0.25 * w), int(y1 + 0.20 * h), int(x1 + 0.75 * w) + 1, int(y1 + 0.50 * h) + 1


def jersey_colour(frame, box):
    """Colour feature (L, a, b) of the jersey crop, or None if the crop is empty (box at the
    image edge). L is the 75th percentile of the lightness: the base colour of a jersey is
    lighter than the crest, the number and the shadows on it, so the upper quartile ignores
    those. a and b are the mean colour (they are small numbers, the mean is less noisy)."""
    x1, y1, x2, y2 = torso(box)
    crop = frame[max(y1, 0):y2, max(x1, 0):x2]
    if crop.size == 0:
        return None
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)
    return np.array([np.percentile(lab[:, 0], 75), lab[:, 1].mean(), lab[:, 2].mean()])


def nearest_centre(features, centres, scale=1.0):
    """Index of the closest cluster centre for every feature row (the K-means assignment step).
    Features and centres are divided by `scale` first, as during fitting (see main)."""
    f, c = np.asarray(features, float) / scale, np.asarray(centres, float) / scale
    return np.linalg.norm(f[:, None, :] - c[None, :, :], axis=2).argmin(axis=1)


class TrackVote:
    """Majority vote of the colour group over the last `window` detections of each track ID.
    One detection can get the wrong colour (a blue player half behind a white one); the last
    second of the same track rarely does. The window is short on purpose: ByteTrack sometimes
    hands an ID to another player when two cross, and a vote over the WHOLE track would then
    give both players the same colour (measured in TM1). It only looks back, so it works live."""

    def __init__(self, k, window=30):
        self.k, self.window, self.history = k, window, {}

    def update(self, ids, teams):
        out = []
        for tid, t in zip(ids, teams):
            if tid < 0 or t < 0:  # no track or no colour: nothing to vote on
                out.append(int(t))
                continue
            h = self.history.setdefault(int(tid), deque(maxlen=self.window))
            h.append(int(t))
            out.append(int(np.bincount(h, minlength=self.k).argmax()))
        return out


def lab_to_bgr(lab):
    px = np.uint8(np.clip(np.round(lab), 0, 255)).reshape(1, 1, 3)
    return tuple(int(v) for v in cv2.cvtColor(px, cv2.COLOR_LAB2BGR)[0, 0])


def load_model(path):
    """model.json -> (centres as an array, feature scale, list of names, list of BGR colours)."""
    m = json.loads(Path(path).read_text())
    return np.array([c["lab"] for c in m["clusters"]], float), np.array(m["scale"], float), \
        [c["name"] for c in m["clusters"]], [tuple(c["bgr"]) for c in m["clusters"]]


# ---------- fitting and checking ----------
def kmeans(features, k, seed=0):
    """Returns (centres, labels, inertia). Inertia = sum of squared distances to the own centre."""
    cv2.setRNGSeed(seed)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01)
    inertia, labels, centres = cv2.kmeans(np.float32(features), k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    return centres, labels.ravel(), float(inertia)


def silhouette(features, labels, max_n=2000, seed=0):
    """Mean silhouette: for each point, a = mean distance to its own cluster, b = mean distance
    to the nearest other cluster, s = (b - a) / max(a, b). 1 = tight and far apart, 0 = overlapping."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(features), min(max_n, len(features)), replace=False)
    x, lab = np.asarray(features, float)[idx], labels[idx]
    dist = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
    scores = []
    for i in range(len(x)):
        own = lab == lab[i]
        if own.sum() < 2:
            continue
        a = dist[i, own].sum() / (own.sum() - 1)
        b = min(dist[i, lab == c].mean() for c in set(lab) if c != lab[i])
        scores.append((b - a) / max(a, b))
    return float(np.mean(scores))


def name_clusters(centres):
    """Give each cluster a readable name from its centre colour. The rule is generic:
    clearly coloured (far from the grey axis a = b = 128) -> named after the hue,
    otherwise light or dark."""
    names = []
    for L, a, b in centres:
        chroma = np.hypot(a - 128, b - 128)
        if chroma > 12:
            hue = np.degrees(np.arctan2(b - 128, a - 128)) % 360
            name = ["red", "yellow", "green", "blue"][int(((hue + 45) % 360) // 90)]
        else:
            name = "light" if L > 128 else "dark"
        while name in names:
            name += "2"
        names.append(name)
    return names


def contact_sheet(cap, rows, path, cols=10, cell=(96, 160), numbered=False):
    """Crops of the given detections in a grid, with the jersey region outlined."""
    tiles = []
    for n, r in enumerate(rows.itertuples()):
        cap.set(cv2.CAP_PROP_POS_FRAMES, r.frame)
        frame = cap.read()[1]
        x1, y1, x2, y2 = (int(v) for v in (r.x1, r.y1, r.x2, r.y2))
        crop = frame[max(y1, 0):y2 + 1, max(x1, 0):x2 + 1].copy()
        tx1, ty1, tx2, ty2 = torso((r.x1, r.y1, r.x2, r.y2))
        k = min(cell[0] / crop.shape[1], cell[1] / crop.shape[0])
        crop = cv2.resize(crop, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
        cv2.rectangle(crop, (int((tx1 - x1) * k), int((ty1 - y1) * k)), (int((tx2 - x1) * k), int((ty2 - y1) * k)),
                      (0, 255, 0), 1)
        tile = np.full((cell[1] + 18, cell[0], 3), 40, np.uint8)
        tile[18:18 + crop.shape[0], :crop.shape[1]] = crop
        cv2.putText(tile, str(n) if numbered else f"f{r.frame}", (2, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 40))
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    cv2.imwrite(str(path), grid, [cv2.IMWRITE_JPEG_QUALITY, 92])


def heatmap(xy, minimap, sigma_m=1.5):
    """KDE of rink positions on the minimap grid: 2D histogram, then a Gaussian blur.
    sigma_m is the kernel width in rink units: about our position error far from the camera."""
    w, h = minimap.size
    hist = np.zeros((h, w), np.float32)
    px = minimap.to_px(xy)
    ok = (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)
    np.add.at(hist, (px[ok, 1], px[ok, 0]), 1)
    hist = cv2.GaussianBlur(hist, (0, 0), sigma_m * minimap.s)
    shade = cv2.applyColorMap(np.uint8(255 * hist / max(hist.max(), 1e-9)), cv2.COLORMAP_INFERNO)
    alpha = np.clip(hist / max(hist.max(), 1e-9) * 3, 0, 0.85)[:, :, None]  # empty ice stays visible
    return np.uint8(minimap.background * (1 - alpha) + shade * alpha)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--players", required=True, help="players.csv from scripts/track.py (with box columns)")
    p.add_argument("--keypoints", required=True, help="only used to pick the rink template for the heatmaps")
    p.add_argument("--k", type=int, default=3, help="number of colour groups (2 teams + referees)")
    p.add_argument("--names", nargs="*", help="names for the clusters, lightest first (after looking at the "
                                              "sheets), e.g. --names white other blue")
    p.add_argument("--min-height", type=int, default=40,
                   help="boxes smaller than this (px) are not used to FIT the clusters: their jersey crop is "
                        "only a handful of pixels. They are still assigned afterwards.")
    p.add_argument("--vote-window", type=int, default=30,
                   help="track vote over the last N detections of a track (30 = about 1 s at stride 2)")
    p.add_argument("--sample", type=int, default=120, help="size of the random sample for human labelling")
    p.add_argument("--score", action="store_true", help="score the clusters against the truth column of sample.csv")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default="teams")
    return p.parse_args()


def score_sample(out, df, names):
    """Confusion matrix of cluster vs human label, before and after the track vote."""
    s = pd.read_csv(out / "sample.csv").dropna(subset=["truth"])
    s = s[s.truth != "unclear"].merge(df[["frame", "id", "x1", "team_track", "team_whole"]], on=["frame", "id", "x1"])
    result = {"labelled": int(len(s))}
    for col, key in (("team", "per_detection"), ("team_track", "after_window_vote"), ("team_whole", "after_whole_track_vote")):
        pred = s[col].map(lambda c: names[int(c)])
        result[key] = {"accuracy": round(float((pred == s.truth).mean()), 3),
                       "confusion_rows_truth_cols_predicted": pd.crosstab(s.truth, pred).to_dict("index")}
    return result


def main():
    args = parse_args()
    out = Path("outputs/teams") / args.name
    out.mkdir(parents=True, exist_ok=True)
    players = pd.read_csv(args.players)
    players = players[players.on_ice].reset_index(drop=True)

    # 1 + 2: one pass over the video, a colour feature for every on-ice detection
    by_frame = {f: g for f, g in players.groupby("frame")}
    cap = cv2.VideoCapture(args.video)
    feats, i = {}, -1
    while cap.grab():
        i += 1
        if i not in by_frame:
            continue
        frame = cap.retrieve()[1]
        for r in by_frame[i].itertuples():
            c = jersey_colour(frame, (r.x1, r.y1, r.x2, r.y2))
            if c is not None:
                feats[r.Index] = c
    df = players.loc[list(feats)].copy()
    df[["L", "a", "b"]] = np.array(list(feats.values()))
    df["height"] = df.y2 - df.y1
    X_all = df[["L", "a", "b"]].to_numpy()
    fit = (df.height >= args.min_height).to_numpy()
    X = X_all[fit]

    # 3: K-means, plus the same for other k to check that k is a sensible choice.
    # K-means measures plain Euclidean distance, so a feature with a large range decides
    # everything. Here L runs from 20 to 170 and a/b only over about 30, while the difference
    # between the blue team and the referees is mostly in b. So every feature is divided by its
    # own standard deviation first (standardising): then L, a and b count equally.
    scale = X.std(axis=0)
    k_table = []
    for k in range(2, 6):
        centres_k, labels_k, inertia = kmeans(X / scale, k, args.seed)
        k_table.append({"k": k, "inertia": round(inertia), "silhouette": round(silhouette(X / scale, labels_k), 3)})
    centres, _, _ = kmeans(X / scale, args.k, args.seed)
    centres = centres * scale           # back to Lab units, so the centres can be read as colours
    order = np.argsort(-centres[:, 0])  # lightest cluster first, so the numbering is stable
    centres = centres[order]
    # K-means returns anonymous groups. What they mean (which is a team, which the referees) is
    # our reading of the crop sheets, passed in with --names in this lightest-first order.
    names = args.names if args.names and len(args.names) == args.k else name_clusters(centres)
    df["team"] = nearest_centre(X_all, centres, scale)

    # 4: track vote. Detections without a track ID (-1) keep their own cluster.
    #    team_track  majority over the last --vote-window detections of the same track (used)
    #    team_whole  majority over the whole track (for comparison: it assumes a track is one
    #                person, which ByteTrack does not guarantee when players cross)
    vote = TrackVote(args.k, args.vote_window)
    df = df.sort_values("frame", kind="stable")
    df["team_track"] = [t for _, g in df.groupby("frame", sort=True) for t in vote.update(g.id, g.team)]
    tracked = df[df.id >= 0]
    majority = tracked.groupby("id").team.agg(lambda s: s.value_counts().idxmax())
    df["team_whole"] = df.id.map(majority).fillna(df.team).astype(int)
    long_tracks = tracked.groupby("id").filter(lambda g: len(g) >= 10)
    agree = (long_tracks.team == long_tracks.id.map(majority))
    df.to_csv(out / "teams.csv", index=False)

    def flips(d, col):
        return float(d.sort_values("frame").groupby("id")[col].apply(lambda s: (s.diff().fillna(0) != 0).sum()).sum() / len(d))

    model = {"feature": "torso crop in CIELAB (OpenCV 8-bit: L 0-255, a/b centred on 128): "
                        "75th percentile of L, mean a, mean b; divided by `scale` before the distance",
             "scale": [round(float(v), 3) for v in scale],
             "clusters": [{"name": n, "lab": [round(float(v), 1) for v in c],
                           "bgr": list(DRAW_COLOURS.get(n, lab_to_bgr(c)))}
                          for n, c in zip(names, centres)]}
    (out / "model.json").write_text(json.dumps(model, indent=2))

    summary = {
        "detections_on_ice": int(len(df)), "used_for_fit": int(fit.sum()), "min_height_px": args.min_height,
        "k": args.k, "k_selection": k_table,
        "clusters": [{"name": n, "lab": m["lab"],
                      "share_of_detections_pct": round(100 * float((df.team == j).mean()), 1),
                      "share_after_track_vote_pct": round(100 * float((df.team_track == j).mean()), 1),
                      "per_frame_after_vote": round(float((df.team_track == j).sum()) / df.frame.nunique(), 2)}
                     for j, (n, m) in enumerate(zip(names, model["clusters"]))],
        "track_consistency": {
            "tracks_with_10plus_detections": int(long_tracks.id.nunique()),
            "detections_agreeing_with_track_majority_pct": round(100 * float(agree.mean()), 1),
            "tracks_below_90pct_agreement": int((agree.groupby(long_tracks.id).mean() < 0.9).sum()),
            "detections_changed_by_window_vote_pct": round(100 * float((df.team != df.team_track).mean()), 1),
            "vote_window": args.vote_window,
            # how often the colour of a track flips from one detection to the next
            "colour_flips_per_100_detections": {
                "per_detection": round(100 * flips(long_tracks, "team"), 2),
                "after_window_vote": round(100 * flips(df[df.id.isin(long_tracks.id)], "team_track"), 2)},
        },
    }

    # plots
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    ks = [r["k"] for r in k_table]
    axes[0].plot(ks, [r["inertia"] for r in k_table], "o-")
    axes[0].set(title="elbow: inertia", xlabel="k", xticks=ks)
    axes[1].plot(ks, [r["silhouette"] for r in k_table], "o-")
    axes[1].set(title="mean silhouette", xlabel="k", xticks=ks)
    fig.tight_layout()
    fig.savefig(out / "k_selection.png", dpi=120)
    plt.close(fig)

    rng = np.random.default_rng(args.seed)
    show = rng.choice(len(df), min(6000, len(df)), replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, (cx, cy, lx, ly) in zip(axes, (("a", "b", "a (green - red)", "b (blue - yellow)"),
                                           ("b", "L", "b (blue - yellow)", "L (lightness)"))):
        for j, (n, m) in enumerate(zip(names, model["clusters"])):
            d = df.iloc[show]
            d = d[d.team == j]
            ax.scatter(d[cx], d[cy], s=4, color=np.array(m["bgr"][::-1]) / 255, edgecolors="grey", linewidths=0.2,
                       label=f"{n} ({len(d)})")
            ax.scatter([m["lab"]["Lab".index(cx)]], [m["lab"]["Lab".index(cy)]], marker="x", c="red", s=80)
        ax.set(xlabel=lx, ylabel=ly)
    axes[0].legend()
    axes[0].set_title("jersey colours, coloured by cluster (x = centre)")
    fig.tight_layout()
    fig.savefig(out / "colours.png", dpi=120)
    plt.close(fig)

    for j, n in enumerate(names):
        rows = df[df.team == j]
        contact_sheet(cap, rows.sample(min(40, len(rows)), random_state=args.seed).sort_values("frame"),
                      out / f"sheet_{j}_{n}.jpg")

    # the sample for human labelling is drawn once and then kept, so labels stay valid
    if not (out / "sample.csv").exists():
        sample = df.sample(min(args.sample, len(df)), random_state=args.seed).sort_values("frame")
        sample.assign(truth="")[["frame", "id", "x1", "y1", "x2", "y2", "height", "team", "truth"]] \
            .to_csv(out / "sample.csv", index=False)
        contact_sheet(cap, sample, out / "sample.jpg", cols=12, numbered=True)
    if args.score:
        summary["sample_check"] = score_sample(out, df, names)

    # 5: occupancy heatmaps per group (after the track vote)
    spec = RINKS[json.loads(Path(args.keypoints).read_text()).get("rink", "nhl")]
    mm = Minimap(spec, width_px=700)
    maps = []
    for j, n in enumerate(names):
        d = df[df.team_track == j]
        img = heatmap(d[["x", "y"]].to_numpy(), mm)
        cv2.putText(img, f"{n}: {len(d)} positions", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
        maps.append(img)
    cv2.imwrite(str(out / "heatmaps.jpg"), np.vstack(maps), [cv2.IMWRITE_JPEG_QUALITY, 92])
    cap.release()

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
