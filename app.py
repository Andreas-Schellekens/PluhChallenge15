"""Hockey Vision dashboard: video + synced tactical minimap, with adjustable parameters.

Run:  streamlit run app.py

Tabs
  Frame        pick any moment, change a parameter in the sidebar and see the
               effect on detection and mapping immediately
  Clip         process a time range into a side-by-side video (+ positions CSV)
  Tracking     ByteTrack + Kalman results over the whole clip (T1)
  Teams        jersey colour clustering and occupancy heatmaps (TM1)
  Puck model   baseline vs fine-tuned: metrics, confusion matrix, examples
  Homography   keypoints, reprojection error and the template drawn onto the
               reference frame, for the current lens setting

All heavy lifting lives in scripts/ (pipeline.py, homography.py, eval_puck.py),
so the app shows the same code paths the experiments were logged with.
"""

import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from homography import FrameMatcher, RinkMapper  # noqa: E402
from kalman import PuckKalman  # noqa: E402
from pipeline import Settings, TrackVote, draw, process_frame  # noqa: E402
from rink import Minimap  # noqa: E402
from teams import load_model as load_team_model  # noqa: E402

ROOT = Path(__file__).parent
st.set_page_config(page_title="Hockey Vision", layout="wide")


# ---------- cached resources ----------
@st.cache_resource
def load_yolo(path):
    from ultralytics import YOLO
    return YOLO(path)


def YOLO_fresh(path):
    """Uncached model: the tracker state lives on the model object, so a new clip needs a new one."""
    from ultralytics import YOLO
    return YOLO(path)


@st.cache_data
def video_info(path):
    cap = cv2.VideoCapture(path)
    info = {"fps": cap.get(cv2.CAP_PROP_FPS), "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "w": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "h": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    cap.release()
    return info


def read_frame(path, idx):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


@st.cache_resource
def frame_matcher(video, ref_idx):
    return FrameMatcher(read_frame(video, ref_idx))


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ---------- sidebar: all parameters ----------
sb = st.sidebar
sb.title("Hockey Vision")
upload = sb.file_uploader("Add a clip (.mp4)", type=["mp4"],
                          help="Saved to Video/. A new camera position also needs its own keypoints config "
                               "(configs/keypoints_<clip>.json): the homography is measured per camera.")
if upload is not None:
    (ROOT / "Video").mkdir(exist_ok=True)
    target = ROOT / "Video" / Path(upload.name).name
    if not target.exists():
        target.write_bytes(upload.getbuffer())
videos = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "Video").glob("*.mp4"))
if not videos:
    st.error("No .mp4 in Video/. Upload one in the sidebar or put the clip there (see README).")
    st.stop()
video = sb.selectbox("Video", videos, index=next((i for i, v in enumerate(videos) if "fillempje" in v), 0))
configs = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "configs").glob("keypoints_*.json"))
stem = Path(video).stem
kp_path = sb.selectbox("Rink keypoints", configs,
                       index=next((i for i, c in enumerate(configs) if stem in c), 0),
                       help="Hand-measured keypoints + lens model for this camera position")
cfg = json.loads((ROOT / kp_path).read_text())

sb.header("Players")
s = Settings()
s.person_imgsz = sb.select_slider("Person model input size", [640, 960, 1280, 1920], value=1280,
                                  help="Larger = small (far) players are found more often, but slower")
s.person_conf = sb.slider("Person confidence", 0.05, 0.9, 0.25, 0.05,
                          help="Frame tab. With tracking on, the Clip tab uses at most 0.1: ByteTrack needs the "
                               "low-confidence boxes to keep a partly hidden player's track")
s.show_off_ice = sb.checkbox("Show people off the ice (grey)", False,
                             help="Spectators/bench: their feet map outside the rink outline")
team_models = sorted(ROOT.glob("outputs/teams/*/model.json"))
team_choice = sb.selectbox("Team colours", ["off"] + [p.parent.name for p in team_models],
                           index=1 if team_models else 0,
                           help="K-means on jersey colours (scripts/teams.py). Each dot gets the colour of its cluster.")
team_names = []
if team_choice != "off":
    s.team_centres, s.team_scale, team_names, s.team_colours = load_team_model(team_models[[p.parent.name for p in team_models].index(team_choice)])

sb.header("Puck")
runs = sorted(ROOT.glob("outputs/train/*/weights/best.pt"))
puck_options = ["fine-tuned: " + p.parent.parent.name for p in runs] + ["baseline: COCO sports ball", "off"]
# default: E2a, the best model in the experiment log (the list also holds older and seed runs)
puck_choice = sb.selectbox("Puck detector", puck_options,
                           index=next((i for i, p in enumerate(runs) if p.parent.parent.name == "fillempje_v3_yolo26s_puck_640"), 0))
puck_model = None
if puck_choice.startswith("fine-tuned"):
    s.puck_mode = "finetuned"
    puck_model = load_yolo(str(runs[puck_options.index(puck_choice)]))
elif puck_choice.startswith("baseline"):
    s.puck_mode = "baseline"
else:
    s.puck_mode = "off"
s.puck_conf = sb.slider("Puck confidence", 0.05, 0.9, 0.25, 0.05)
if s.puck_mode == "finetuned":
    s.tile = sb.select_slider("Tile size", [320, 480, 640, 800], value=640,
                              help="The frame is cut into tiles so the puck keeps its native size")
    s.overlap = sb.slider("Tile overlap (px)", 0, s.tile // 2, min(160, s.tile // 2), 16,
                          help="Must be larger than the puck, or a puck on a tile edge is cut in half")
elif s.puck_mode == "baseline":
    s.puck_imgsz = sb.select_slider("Baseline input size", [640, 1280, 1920], value=1280)

sb.header("Homography")
lens_cfg = cfg.get("lens", {})
k1 = sb.slider("Lens distortion k1", -0.3, 0.1, float(lens_cfg.get("k1", 0.0)), 0.005,
               help="0 = no lens correction. The config value straightens the board lines.")
cfg_live = {**cfg, "lens": {**lens_cfg, "k1": k1}}
drift = sb.radio("Camera motion", ["static", "SIFT to reference"], index=0 if "fillempje" in video else 1,
                 help="Handheld clips need per-frame matching to the reference frame")
s.show_template = sb.checkbox("Draw rink template on the video", True)

sb.header("Tracking (Clip tab)")
s.track = sb.checkbox("Track players + puck", True,
                      help="ByteTrack IDs for players, Kalman filter for the puck. Needs consecutive frames, "
                           "so it only applies in the Clip tab.")
kal_params = {}
if s.track:
    kal_params["accel_std"] = sb.slider("Puck acceleration noise (m/s²)", 5.0, 60.0, 20.0, 5.0,
                                        help="How hard the puck may change speed. Higher = follows sudden shots, but noisier")
    kal_params["meas_std"] = sb.slider("Detection noise (m)", 0.2, 2.0, 0.7, 0.1,
                                       help="How much we trust one detection after the homography")
    kal_params["max_coast"] = sb.slider("Max time without detection (s)", 0.0, 1.5, 0.5, 0.1,
                                        help="How long the filter keeps predicting a hidden puck")

mapper = RinkMapper(cfg_live)
minimap = Minimap(mapper.spec, width_px=900)
person_model = load_yolo("yolo26s.pt")
info = video_info(str(ROOT / video))


last_good = {"H": None, "failed": 0}


def h_to_ref(frame):
    if drift == "static":
        return None
    H, inliers = frame_matcher(str(ROOT / video), cfg["reference_frame"]).to_reference(frame)
    if H is not None and inliers >= 80:
        last_good["H"] = H
    else:  # matching failed: reuse the last good homography (as scripts/homography.py does) and count it
        last_good["failed"] += 1
    return last_good["H"]


tab_frame, tab_clip, tab_track, tab_teams, tab_eval, tab_hom = st.tabs(
    ["Frame", "Clip", "Tracking", "Teams", "Puck model", "Homography"])

# ---------- Frame ----------
with tab_frame:
    t = st.slider("Time (s)", 0.0, info["frames"] / info["fps"] - 0.1, 30.0, 0.05)
    idx = int(t * info["fps"])
    frame = read_frame(str(ROOT / video), idx)
    if frame is None:
        st.warning("Could not read this frame.")
    else:
        H = h_to_ref(frame)
        s_frame = Settings(**{**s.__dict__, "track": False})  # one frame: nothing to track
        res = process_frame(frame, person_model, puck_model, mapper, s_frame, H)
        view, mm = draw(frame, res, mapper, minimap, s_frame, H)
        c1, c2 = st.columns([3, 2])
        c1.image(rgb(view), caption=f"frame {idx}", width="stretch")
        c2.image(rgb(mm), caption="tactical minimap", width="stretch")
        m1, m2, m3 = c2.columns(3)
        m1.metric("Players on ice", int(sum(res.on_ice)))
        m2.metric("People off ice", int(len(res.on_ice) - sum(res.on_ice)))
        m3.metric("Puck", f"{res.puck_conf[0]:.2f}" if len(res.pucks) else "not found")
        if len(res.person_rink):
            table = pd.DataFrame({"x": res.person_rink[:, 0].round(1), "y": res.person_rink[:, 1].round(1),
                                  "conf": res.person_conf.round(2), "on ice": res.on_ice})
            if "teams" in res.extra:
                table["team"] = [team_names[t] if t >= 0 else "?" for t in res.extra["teams"]]
            c2.dataframe(table, height=200, width="stretch")

# ---------- Clip ----------
with tab_clip:
    dur = info["frames"] / info["fps"]
    c1, c2, c3 = st.columns(3)
    t0 = c1.number_input("Start (s)", 0.0, dur, 20.0, 1.0)
    t1 = c2.number_input("End (s)", 0.0, dur, 30.0, 1.0)
    stride = c3.number_input("Every n-th frame", 1, 10, 2)
    if st.button("Process clip", type="primary"):
        import imageio_ffmpeg
        first, last = int(t0 * info["fps"]), int(t1 * info["fps"])
        out_path = Path(tempfile.gettempdir()) / f"hockey_{stem}_{first}_{last}.mp4"
        cap = cv2.VideoCapture(str(ROOT / video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        writer, rows = None, []
        # tracking keeps state between frames, so every clip run starts with a fresh tracker and filter
        clip_person_model = YOLO_fresh("yolo26s.pt") if s.track else person_model
        # inside: while predicting, a puck that leaves the rink ends the track (same rule as track.py, T2)
        kf = PuckKalman(inside=lambda x, y: mapper.on_ice([(x, y)], s.rink_margin)[0], **kal_params) if s.track else None
        vote = TrackVote(len(team_names)) if s.track and team_names else None
        bar = st.progress(0.0)
        for i in range(first, last):
            if not cap.grab():
                break
            if (i - first) % stride:
                continue
            frame = cap.retrieve()[1]
            H = h_to_ref(frame)
            res = process_frame(frame, clip_person_model, puck_model, mapper, s, H)
            if vote is not None:  # a track is one person: use the majority colour of the track so far
                res.extra["teams"] = vote.update(res.extra["ids"], res.extra["teams"])
            if kf is not None:
                x, y, state, _ = kf.step(stride / info["fps"], res.extra["puck_candidates"])
                res.extra["kalman"] = (x, y, state)
                res.puck_rink = np.array([[x, y]]) if x is not None else np.zeros((0, 2))
                res.puck_conf = np.array([1.0 if state != "predicted" else 0.0]) if x is not None else np.zeros(0)
            view, mm = draw(frame, res, mapper, minimap, s, H)
            mm = cv2.resize(mm, (view.shape[1], round(mm.shape[0] * view.shape[1] / mm.shape[1])))
            canvas = np.vstack([view, mm])
            canvas = canvas[: canvas.shape[0] // 2 * 2, : canvas.shape[1] // 2 * 2]  # H.264 needs even sizes
            if writer is None:
                writer = imageio_ffmpeg.write_frames(str(out_path), (canvas.shape[1], canvas.shape[0]),
                                                     fps=info["fps"] / stride, codec="libx264",
                                                     macro_block_size=1, quality=7)
                writer.send(None)
            writer.send(np.ascontiguousarray(rgb(canvas)))
            ids = res.extra.get("ids", [-1] * len(res.person_rink))
            tms = res.extra.get("teams", [-1] * len(res.person_rink))
            for xy, c, ice, tid, tm in zip(res.person_rink, res.person_conf, res.on_ice, ids, tms):
                rows.append({"frame": i, "time_s": round(i / info["fps"], 3), "type": "person", "id": int(tid),
                             "x": round(xy[0], 2), "y": round(xy[1], 2), "conf": round(float(c), 3), "on_ice": ice,
                             "team": team_names[tm] if tm >= 0 else ""})
            for xy, c in zip(res.puck_rink, res.puck_conf):
                rows.append({"frame": i, "time_s": round(i / info["fps"], 3), "type": "puck", "id": -1,
                             "x": round(xy[0], 2), "y": round(xy[1], 2), "conf": round(float(c), 3), "on_ice": True,
                             "state": res.extra.get("kalman", (0, 0, "detected"))[2]})
            bar.progress((i - first + 1) / max(1, last - first))
        cap.release()
        if writer is not None:
            writer.close()
            st.session_state["clip"] = (str(out_path), pd.DataFrame(rows))
    if "clip" in st.session_state:
        path, df = st.session_state["clip"]
        st.video(path)
        n = df["frame"].nunique()
        pk = df[df["type"] == "puck"]
        c1, c2 = st.columns(2)
        c1.metric("Players on ice per frame", f"{(df[df['type'] == 'person']['on_ice']).sum() / max(n, 1):.1f}")
        if "state" in pk and (pk["state"] == "predicted").any():  # tracked: say how many are only predicted
            c2.metric("Frames with puck", f"{pk['frame'].nunique()} / {n}",
                      f"{int((pk['state'] == 'predicted').sum())} of them predicted by the Kalman filter", delta_color="off")
        else:
            c2.metric("Frames with puck", f"{pk['frame'].nunique()} / {n}")
        if last_good["failed"]:
            st.warning(f"Camera matching failed on {last_good['failed']} frames; the last good homography was reused.")
        st.download_button("Download positions CSV", df.to_csv(index=False), "positions.csv", "text/csv")

# ---------- Tracking ----------
with tab_track:
    runs_t = sorted((ROOT / "outputs/track").glob("*/summary.json"))
    if not runs_t:
        st.info("No tracking run yet. Run scripts/track.py (see docs/experiments.md, T1).")
    else:
        run_t = st.selectbox("Tracking run (whole clip)", runs_t, format_func=lambda p: p.parent.name)
        sm = json.loads(run_t.read_text())
        pk, pl = sm["puck"], sm["players"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Puck on the minimap", f"{pk['frames_shown_after_kalman_pct']}% of frames",
                  f"+{pk['frames_shown_after_kalman_pct'] - pk['frames_with_detection_on_ice_pct']:.1f} pts vs detections only")
        c2.metric("Impossible puck jumps", pk["impossible_jumps_kalman"], f"{pk['impossible_jumps_kalman'] - pk['impossible_jumps_raw']} vs raw",
                  delta_color="inverse")
        c3.metric("Player IDs (on ice)", pl["unique_ids_on_ice"], f"median life {pl['median_track_life_s']} s",
                  delta_color="off")
        st.subheader("Gap test: hide detections, compare the prediction with what was hidden")
        st.dataframe(pd.DataFrame(sm["gap_test"]), width="stretch")
        if (run_t.parent / "minimap_compare.mp4").exists():
            st.video(str(run_t.parent / "minimap_compare.mp4"))
        st.caption(f"Kalman settings of this run: {sm['kalman_params']}")

# ---------- Teams ----------
with tab_teams:
    runs_tm = sorted((ROOT / "outputs/teams").glob("*/summary.json"))
    if not runs_tm:
        st.info("No team clustering yet. Run scripts/teams.py (see docs/experiments.md, TM1).")
    else:
        run_tm = st.selectbox("Team clustering run (whole clip)", runs_tm, format_func=lambda p: p.parent.name)
        sm = json.loads(run_tm.read_text())
        tc = sm["track_consistency"]
        cols = st.columns(len(sm["clusters"]) + 1)
        for col, cl in zip(cols, sm["clusters"]):
            col.metric(f"{cl['name']} (Lab {cl['lab']})", f"{cl['per_frame_after_vote']} per frame",
                       f"{cl['share_after_track_vote_pct']}% of detections", delta_color="off")
        flips = tc["colour_flips_per_100_detections"]
        cols[-1].metric("Colour flips per 100 detections", flips["after_window_vote"],
                        f"{flips['per_detection']} without the track vote", delta_color="off")
        st.subheader("How many colour groups? (K-means for k = 2..5)")
        st.dataframe(pd.DataFrame(sm["k_selection"]), width="stretch")
        if "sample_check" in sm:
            st.subheader("Random sample checked by hand")
            st.json(sm["sample_check"])
        for img in ("heatmaps.jpg", "colours.png", "k_selection.png", *sorted(p.name for p in run_tm.parent.glob("sheet_*.jpg"))):
            if (run_tm.parent / img).exists():
                st.image(str(run_tm.parent / img), caption=img, width="stretch")

# ---------- Puck model ----------
with tab_eval:
    evals = sorted((ROOT / "outputs/eval_puck").glob("*/metrics.json"))
    if not evals:
        st.info("No evaluation yet. Run scripts/eval_puck.py after training (see docs/experiments.md).")
    else:
        run = st.selectbox("Evaluation run", evals, format_func=lambda p: p.parent.name)
        m = json.loads(run.read_text())
        st.caption(f"Scored on the {m['split']} frames {m['frames']} ({m['pucks']} pucks), "
                   f"conf >= {m['conf']}, IoU >= {m['iou_match']}")
        st.dataframe(pd.DataFrame(m["results"]).T, width="stretch")
        for img in ("confusion_matrix.png", "timeline.png", "comparison.jpg", "clip_sample.jpg"):
            if (run.parent / img).exists():
                st.image(str(run.parent / img), caption=img, width="stretch")

# ---------- Homography ----------
with tab_hom:
    errs = mapper.keypoint_errors()
    st.dataframe(pd.DataFrame({"keypoint": [p["name"] for p in cfg["points"]],
                               "image (px)": [str(p["image"]) for p in cfg["points"]],
                               "rink": [str(p["rink"]) for p in cfg["points"]],
                               f"reprojection error ({mapper.spec.unit})": errs.round(3),
                               "reprojection error (px)": mapper.keypoint_errors_px().round(1)}),
                 width="stretch")
    if len(cfg["points"]) == 4:
        st.caption("4 points = exactly the 8 unknowns of a homography, so the error is always 0. "
                   "Check the overlay instead: lines that were NOT used for the fit must match.")
    else:
        st.caption("More than 4 points: the fit minimises the error in pixels. The same few pixels are "
                   "centimetres next to the camera and metres at the far end of the rink.")
    ref = read_frame(str(ROOT / video), cfg["reference_frame"])
    if ref is not None:
        mapper.draw_template(ref, thickness=2)
        for p in cfg["points"]:
            cv2.circle(ref, tuple(int(v) for v in p["image"]), 10, (0, 255, 0), 3)
        st.image(rgb(ref), caption=f"reference frame {cfg['reference_frame']} with the rink template (k1 = {k1})",
                 width="stretch")
    st.caption(cfg.get("note", ""))
