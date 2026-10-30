"""Hockey Vision dashboard: video + synced tactical minimap, with adjustable parameters.

Run:  streamlit run app.py

Tabs
  Frame        pick any moment, change a parameter in the sidebar and see the
               effect on detection and mapping immediately
  Clip         process a time range into a side-by-side video (+ positions CSV)
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
from pipeline import Settings, draw, process_frame  # noqa: E402
from rink import Minimap  # noqa: E402

ROOT = Path(__file__).parent
st.set_page_config(page_title="Hockey Vision", layout="wide")


# ---------- cached resources ----------
@st.cache_resource
def load_yolo(path):
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
videos = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "Video").glob("*.mp4"))
if not videos:
    st.error("No .mp4 in Video/. Put the clip there (see README).")
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
s.person_conf = sb.slider("Person confidence", 0.05, 0.9, 0.25, 0.05)
s.show_off_ice = sb.checkbox("Show people off the ice (grey)", False,
                             help="Spectators/bench: their feet map outside the rink outline")

sb.header("Puck")
runs = sorted(ROOT.glob("outputs/train/*/weights/best.pt"))
puck_options = ["fine-tuned: " + p.parent.parent.name for p in runs] + ["baseline: COCO sports ball", "off"]
puck_choice = sb.selectbox("Puck detector", puck_options)
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

mapper = RinkMapper(cfg_live)
minimap = Minimap(mapper.spec, width_px=900)
person_model = load_yolo("yolo26s.pt")
info = video_info(str(ROOT / video))


def h_to_ref(frame):
    if drift == "static":
        return None
    H, inliers = frame_matcher(str(ROOT / video), cfg["reference_frame"]).to_reference(frame)
    return H if H is not None and inliers >= 80 else None


tab_frame, tab_clip, tab_eval, tab_hom = st.tabs(["Frame", "Clip", "Puck model", "Homography"])

# ---------- Frame ----------
with tab_frame:
    t = st.slider("Time (s)", 0.0, info["frames"] / info["fps"] - 0.1, 30.0, 0.05)
    idx = int(t * info["fps"])
    frame = read_frame(str(ROOT / video), idx)
    if frame is None:
        st.warning("Could not read this frame.")
    else:
        H = h_to_ref(frame)
        res = process_frame(frame, person_model, puck_model, mapper, s, H)
        view, mm = draw(frame, res, mapper, minimap, s, H)
        c1, c2 = st.columns([3, 2])
        c1.image(rgb(view), caption=f"frame {idx}", width="stretch")
        c2.image(rgb(mm), caption="tactical minimap", width="stretch")
        m1, m2, m3 = c2.columns(3)
        m1.metric("Players on ice", int(sum(res.on_ice)))
        m2.metric("People off ice", int(len(res.on_ice) - sum(res.on_ice)))
        m3.metric("Puck", f"{res.puck_conf[0]:.2f}" if len(res.pucks) else "not found")
        if len(res.person_rink):
            c2.dataframe(pd.DataFrame({"x": res.person_rink[:, 0].round(1), "y": res.person_rink[:, 1].round(1),
                                       "conf": res.person_conf.round(2), "on ice": res.on_ice}),
                         height=200, width="stretch")

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
        bar = st.progress(0.0)
        for i in range(first, last):
            if not cap.grab():
                break
            if (i - first) % stride:
                continue
            frame = cap.retrieve()[1]
            H = h_to_ref(frame)
            res = process_frame(frame, person_model, puck_model, mapper, s, H)
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
            for xy, c, ice in zip(res.person_rink, res.person_conf, res.on_ice):
                rows.append({"frame": i, "time_s": round(i / info["fps"], 3), "type": "person",
                             "x": round(xy[0], 2), "y": round(xy[1], 2), "conf": round(float(c), 3), "on_ice": ice})
            for xy, c in zip(res.puck_rink, res.puck_conf):
                rows.append({"frame": i, "time_s": round(i / info["fps"], 3), "type": "puck",
                             "x": round(xy[0], 2), "y": round(xy[1], 2), "conf": round(float(c), 3), "on_ice": True})
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
        c2.metric("Frames with puck", f"{pk['frame'].nunique()} / {n}")
        st.download_button("Download positions CSV", df.to_csv(index=False), "positions.csv", "text/csv")

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
                               f"reprojection error ({mapper.spec.unit})": errs.round(3)}),
                 width="stretch")
    if len(cfg["points"]) == 4:
        st.caption("4 points = exactly the 8 unknowns of a homography, so the error is always 0. "
                   "Check the overlay instead: lines that were NOT used for the fit must match.")
    ref = read_frame(str(ROOT / video), cfg["reference_frame"])
    if ref is not None:
        mapper.draw_template(ref, thickness=2)
        for p in cfg["points"]:
            cv2.circle(ref, tuple(int(v) for v in p["image"]), 10, (0, 255, 0), 3)
        st.image(rgb(ref), caption=f"reference frame {cfg['reference_frame']} with the rink template (k1 = {k1})",
                 width="stretch")
    st.caption(cfg.get("note", ""))
