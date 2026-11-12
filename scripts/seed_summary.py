"""Compare puck models over several training seeds (experiment S1).

Goal: E2 compared two datasets with ONE training run each, and `best.pt` was picked on 10
val pucks. One extra hit moves recall by 10 points, so a single run cannot tell a real
difference from luck. Here every variant is trained several times; only the seed changes
(weight initialisation of the new head, batch order, augmentation draws). The spread between
seeds is the noise level: a difference between variants only counts if it is clearly larger.

For every run it collects:
  tile val     mAP@50 and mAP@50:95 of best.pt (outputs/train/<run>/metrics.json) and the
               epoch best.pt comes from (results.csv)
  full frames  TP / FP / FN, precision, recall, AP@50 with tiled inference on the 25 val
               frames (outputs/eval_puck/<run>/metrics.json)
  independent  recall on the val pucks that did NOT enter the dataset through a model
               proposal (--independent): the val set was partly built with the E1 model,
               so the other pucks are the fairer sample
  clip         in how many frames of the whole clip a puck is reported (no labels)

Outputs (outputs/seeds/): summary.csv (one row per run), summary.json (mean and standard
deviation per variant) and a table on the screen.

Usage:
  python scripts/seed_summary.py --variant E2a seed_E2a_s0 seed_E2a_s1 seed_E2a_s2 --variant E2c seed_E2c_s0 seed_E2c_s1 seed_E2c_s2
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", nargs="+", action="append", required=True, metavar=("NAME", "RUN"),
                   help="variant name followed by its run names; repeat per variant")
    p.add_argument("--independent", type=int, nargs="*", default=[4200, 4500, 4560, 4950, 5070],
                   help="val frames whose puck was not proposed by a model (see docs/annotation.md)")
    return p.parse_args()


def read_run(variant, run, independent):
    train = Path("outputs/train") / run
    m = json.loads((train / "metrics.json").read_text())
    curve = pd.read_csv(train / "results.csv")
    curve.columns = curve.columns.str.strip()
    # Ultralytics picks best.pt by "fitness". We report the epoch with the highest mAP@50:95,
    # which is what that fitness is dominated by.
    best_epoch = int(curve.loc[curve["metrics/mAP50-95(B)"].idxmax(), "epoch"])
    e = json.loads((Path("outputs/eval_puck") / run / "metrics.json").read_text())
    r = e["results"]["finetuned"]
    row = {"variant": variant, "run": run, "best_epoch": best_epoch,
           "tile_mAP50": m["mAP50"], "tile_mAP50_95": m["mAP50_95"],
           "TP": r["TP"], "FP": r["FP"], "FN": r["FN"],
           "precision": r["precision"], "recall": r["recall"], "AP50": r["AP50"],
           "clip_frames_with_puck": e.get("clip_frames_with_puck_finetuned")}
    if "per_frame" in e:
        ind = [f for f in e["per_frame"] if f["model"] == "finetuned" and f["frame"] in independent]
        row["independent_TP"] = sum(f["TP"] for f in ind)
        row["independent_pucks"] = sum(f["pucks"] for f in ind)
        null = [f for f in e["per_frame"] if f["model"] == "finetuned" and f["pucks"] == 0]
        row["FP_on_null_frames"] = sum(f["FP"] for f in null)
    return row


def main():
    args = parse_args()
    rows = [read_run(v[0], run, args.independent) for v in args.variant for run in v[1:]]
    df = pd.DataFrame(rows)
    out = Path("outputs/seeds")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "summary.csv", index=False)
    numeric = [c for c in df.columns if c not in ("variant", "run")]
    stats = df.groupby("variant", sort=False)[numeric].agg(["mean", "std", "min", "max"]).round(3)
    summary = {v: {c: {s: float(stats.loc[v, (c, s)]) for s in ("mean", "std", "min", "max")} for c in numeric}
               for v in stats.index}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    pd.set_option("display.width", 250)
    print(df.to_string(index=False))
    print()
    print(stats.xs("mean", axis=1, level=1).round(3).to_string())
    print()
    print("standard deviation between seeds:")
    print(stats.xs("std", axis=1, level=1).round(3).to_string())
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
