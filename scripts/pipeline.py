"""Full per-frame pipeline: players + puck detection -> rink mapping -> drawing.

This module is the glue used by the Streamlit app (app.py). Every step reuses
the scripts that the experiments are logged with, so the app shows exactly what
we measured:
  persons   pretrained COCO YOLO, class `person` (E0)
  puck      our fine-tuned model with tiled inference (eval_puck.predict_tiled),
            or the COCO "sports ball" class as the baseline to compare against
  mapping   homography.RinkMapper (lens correction + homography, H1)
  teams     optional: the jersey colour of each person is matched to the K-means
            centres that teams.py fitted (TM1), so the dots get their team colour

A detection is placed on the rink by its bottom-centre point: the feet of a
player, the bottom of the puck box. Only points on the ice plane obey the
homography, which is why we never use the box centre.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

from eval_puck import SPORTS_BALL, predict_tiled
from homography import RinkMapper
from rink import Minimap
from teams import TrackVote, jersey_colour, nearest_centre  # noqa: F401 (TrackVote is re-exported for app.py)

PERSON = 0
PLAYER_COL, OFF_ICE_COL, PUCK_COL = (0, 140, 255), (160, 160, 160), (0, 0, 0)  # BGR


@dataclass
class Settings:
    person_imgsz: int = 1280
    person_conf: float = 0.25
    puck_mode: str = "finetuned"   # "finetuned", "baseline" (COCO sports ball) or "off"
    puck_conf: float = 0.25
    puck_imgsz: int = 1280         # baseline only: whole-frame inference size
    tile: int = 640
    overlap: int = 160
    rink_margin: float = 1.0       # rink units outside the boards still counted as on the ice
    show_template: bool = True
    show_off_ice: bool = False
    track: bool = False            # ByteTrack IDs for players (needs consecutive frames: Clip tab only)
    tracker: str = "configs/bytetrack_hockey.yaml"
    team_centres: object = None    # k x 3 Lab cluster centres from teams.py (None = one colour for everyone)
    team_scale: object = 1.0       # feature scale that belongs to those centres
    team_colours: object = None    # k BGR colours to draw them with


@dataclass
class FrameResult:
    persons: np.ndarray            # N x 4 boxes (x1, y1, x2, y2), image pixels
    person_conf: np.ndarray
    person_rink: np.ndarray        # N x 2 rink coordinates of the foot points
    on_ice: list
    pucks: np.ndarray              # M x 4
    puck_conf: np.ndarray
    puck_rink: np.ndarray          # M x 2
    extra: dict = field(default_factory=dict)


def bottom_centres(boxes):
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    return np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]], axis=1)


def process_frame(frame, person_model, puck_model, mapper: RinkMapper, s: Settings, H_to_ref=None):
    extra = {}
    if s.track:  # persist=True: the tracker keeps its state between calls, so call it on consecutive frames
        r = person_model.track(frame, persist=True, tracker=s.tracker, imgsz=s.person_imgsz, conf=s.person_conf,
                               classes=[PERSON], verbose=False)[0]
        extra["ids"] = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else np.full(len(r.boxes), -1)
    else:
        r = person_model.predict(frame, imgsz=s.person_imgsz, conf=s.person_conf, classes=[PERSON], verbose=False)[0]
    persons, pconf = r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()
    if s.team_centres is not None:  # colour group per person: nearest K-means centre of the jersey colour
        cols = [jersey_colour(frame, b) for b in persons]
        extra["teams"] = [-1 if c is None else int(nearest_centre([c], s.team_centres, s.team_scale)[0]) for c in cols]

    if s.puck_mode == "finetuned" and puck_model is not None:
        pucks, kconf = predict_tiled(puck_model, frame, s.tile, s.overlap, s.puck_conf)
    elif s.puck_mode == "baseline":
        rb = person_model.predict(frame, imgsz=s.puck_imgsz, conf=s.puck_conf, classes=[SPORTS_BALL], verbose=False)[0]
        pucks, kconf = rb.boxes.xyxy.cpu().numpy(), rb.boxes.conf.cpu().numpy()
    else:
        pucks, kconf = np.zeros((0, 4)), np.zeros(0)
    # all on-ice candidates, for the Kalman filter (it picks the one that fits its prediction)
    if len(pucks):
        cand = mapper.image_to_rink(bottom_centres(pucks), H_to_ref)
        extra["puck_candidates"] = [(x, y, c) for (x, y), c, ice in
                                    zip(cand, kconf, mapper.on_ice(cand, s.rink_margin)) if ice]
    else:
        extra["puck_candidates"] = []
    if len(pucks) > 1:  # there is only one puck: keep the most confident detection
        best = int(np.argmax(kconf))
        pucks, kconf = pucks[best:best + 1], kconf[best:best + 1]

    person_rink = mapper.image_to_rink(bottom_centres(persons), H_to_ref) if len(persons) else np.zeros((0, 2))
    puck_rink = mapper.image_to_rink(bottom_centres(pucks), H_to_ref) if len(pucks) else np.zeros((0, 2))
    on_ice = mapper.on_ice(person_rink, s.rink_margin)
    return FrameResult(persons, pconf, person_rink, on_ice, pucks, kconf, puck_rink, extra)


def draw(frame, res: FrameResult, mapper: RinkMapper, minimap: Minimap, s: Settings, H_to_ref=None, out_width=1280):
    """Returns (annotated video frame, minimap image), both BGR."""
    scale = out_width / frame.shape[1]
    view = cv2.resize(frame, (out_width, round(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    if s.show_template:
        mapper.draw_template(view, H_to_ref, scale=scale, thickness=1)
    mm = minimap.background.copy()

    ids = res.extra.get("ids", [-1] * len(res.persons))
    teams = res.extra.get("teams", [-1] * len(res.persons))
    for box, c, xy, ice, tid, team in zip(res.persons, res.person_conf, res.person_rink, res.on_ice, ids, teams):
        if not ice and not s.show_off_ice:
            continue
        player_col = s.team_colours[team] if team >= 0 else PLAYER_COL
        col = player_col if ice else OFF_ICE_COL
        x1, y1, x2, y2 = (box * scale).astype(int)
        cv2.rectangle(view, (x1, y1), (x2, y2), col, 2)
        cv2.circle(view, ((x1 + x2) // 2, y2), 4, col, -1)
        if tid >= 0:
            cv2.putText(view, str(tid), (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        if ice:
            p = tuple(int(v) for v in minimap.to_px(xy)[0])
            cv2.circle(mm, p, 7, player_col, -1, cv2.LINE_AA)
            cv2.circle(mm, p, 7, (60, 60, 60), 1, cv2.LINE_AA)  # outline: a white jersey on white ice
            if tid >= 0:
                cv2.putText(mm, str(tid), (p[0] + 8, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1)

    kal = res.extra.get("kalman")  # (x, y, state) from kalman.PuckKalman, set by the app
    if kal is not None:
        if kal[0] is not None:  # filtered puck: filled = measured, ring = predicted while hidden
            p = tuple(int(v) for v in minimap.to_px(kal[:2])[0])
            cv2.circle(mm, p, 6 if kal[2] != "predicted" else 9, PUCK_COL, -1 if kal[2] != "predicted" else 2, cv2.LINE_AA)
            img_pt, ok = mapper.rink_to_image(np.array([kal[:2]], float), H_to_ref)
            if ok[0]:
                u, v = img_pt[0]
                cv2.circle(view, (int(u * scale), int(v * scale)), 14, (0, 0, 255), 2 if kal[2] != "predicted" else 1)
        return view, mm

    for box, c, xy in zip(res.pucks, res.puck_conf, res.puck_rink):
        x1, y1, x2, y2 = (box * scale).astype(int)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(view, (cx, cy), 14, (0, 0, 255), 2)  # a ring: the puck itself is only a few pixels
        cv2.putText(view, f"puck {c:.2f}", (cx + 16, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        p = tuple(int(v) for v in minimap.to_px(xy)[0])
        cv2.circle(mm, p, 6, PUCK_COL, -1, cv2.LINE_AA)
        cv2.circle(mm, p, 9, (0, 0, 255), 2, cv2.LINE_AA)
    return view, mm
