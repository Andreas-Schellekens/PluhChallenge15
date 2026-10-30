"""Rink templates (NHL in feet, IIHF in metres) and a top-down drawing for the minimap.

Coordinate system (used by every script that talks about rink positions):
  - origin at centre ice, in the rink's own unit (feet for NHL, metres for IIHF)
  - X along the length of the rink (our filmed end zone is X < 0)
  - Y across the width
On the minimap X points right and Y points down. Which side of the rink is +Y
depends on where the camera stands, so each keypoints config documents it.

NHL: official dimensions (rink 200 x 85 ft, corner radius 28 ft, goal line 11 ft
from the end boards, blue lines 25 ft from centre, end-zone dots 20 ft out from
the goal line and 22 ft either side of the middle, neutral-zone dots 5 ft
outside the blue line).
IIHF: a 60 x 30 m rink with IIHF markings (goal line 4 m from the end boards,
end-zone dots 6 m out from the goal line and 7 m either side, circles 4.5 m,
crease 1.83 m). Smaller club rinks deviate, so for those this is an assumption
that the homography check images have to confirm.
"""

from dataclasses import dataclass

import cv2
import numpy as np

# BGR colours for the drawing
ICE, BOARDS = (245, 245, 245), (60, 60, 60)
RED, BLUE = (40, 40, 200), (200, 90, 30)


@dataclass(frozen=True)
class RinkSpec:
    name: str
    unit: str
    length: float
    width: float
    corner_r: float
    goal_line_x: float     # distance of the goal lines from centre
    blue_line_x: float
    circle_r: float
    end_dot_x: float
    dot_y: float           # end-zone and neutral-zone dots, either side of the middle
    neutral_dot_x: float
    crease_r: float
    margin: float          # grey border around the rink on the minimap


NHL = RinkSpec("nhl", "ft", 200.0, 85.0, 28.0, 89.0, 25.0, 15.0, 69.0, 22.0, 20.0, 6.0, 5.0)
IIHF = RinkSpec("iihf", "m", 60.0, 30.0, 8.5, 26.0, 7.14, 4.5, 20.0, 7.0, 5.64, 1.83, 1.5)
RINKS = {"nhl": NHL, "iihf": IIHF}


def boards_polyline(spec=NHL, n_per_corner=12):
    """Outline of the boards as a closed polygon in rink units (rounded corners)."""
    hl, hw, r = spec.length / 2, spec.width / 2, spec.corner_r
    pts = []
    for cx, cy, start in [(hl - r, hw - r, 0), (-(hl - r), hw - r, 90), (-(hl - r), -(hw - r), 180),
                          (hl - r, -(hw - r), 270)]:
        for a in np.radians(np.linspace(start, start + 90, n_per_corner)):
            pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
    return np.array(pts)


def goal_line_half_length(spec=NHL):
    """Goal lines stop at the rounded corners: |Y| where the corner arc crosses the goal line."""
    dx = spec.goal_line_x - (spec.length / 2 - spec.corner_r)
    return (spec.width / 2 - spec.corner_r) + np.sqrt(spec.corner_r ** 2 - max(dx, 0) ** 2)


def circle(cx, cy, r, n=48):
    a = np.linspace(0, 2 * np.pi, n)
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], axis=1)


def markings(spec=NHL):
    """All rink markings as (polyline in rink units, colour, closed). Used both to draw
    the minimap and to project the template back onto video frames to check a homography."""
    gy, hw = goal_line_half_length(spec), spec.width / 2
    items = [(boards_polyline(spec), BOARDS, True),
             (np.array([[0, -hw], [0, hw]]), RED, False),                    # centre line
             (circle(0, 0, spec.circle_r), BLUE, True)]                      # centre circle
    for s in (-1, 1):
        items.append((np.array([[s * spec.goal_line_x, -gy], [s * spec.goal_line_x, gy]]), RED, False))
        items.append((np.array([[s * spec.blue_line_x, -hw], [s * spec.blue_line_x, hw]]), BLUE, False))
        for y in (-spec.dot_y, spec.dot_y):
            items.append((circle(s * spec.end_dot_x, y, spec.circle_r), RED, True))
        a = np.linspace(-np.pi / 2, np.pi / 2, 16)                            # goal crease (half circle)
        items.append((np.stack([s * (spec.goal_line_x - spec.crease_r * np.cos(a)),
                                spec.crease_r * np.sin(a)], 1), BLUE, False))
    return items


def dots(spec=NHL):
    """Faceoff dots in rink units (centre, 4 end-zone, 4 neutral-zone)."""
    d = [(0.0, 0.0)]
    for s in (-1, 1):
        for y in (-spec.dot_y, spec.dot_y):
            d += [(s * spec.end_dot_x, y), (s * spec.neutral_dot_x, y)]
    return np.array(d)


class Minimap:
    """Draws the rink top-down and converts rink units to minimap pixels."""

    def __init__(self, spec=NHL, width_px=1280):
        self.spec = spec
        self.s = width_px / (spec.length + 2 * spec.margin)
        self.size = (int((spec.length + 2 * spec.margin) * self.s), int((spec.width + 2 * spec.margin) * self.s))
        self.background = self._draw_background()

    def to_px(self, xy):
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        offset = [self.spec.length / 2 + self.spec.margin, self.spec.width / 2 + self.spec.margin]
        return np.round((xy + offset) * self.s).astype(np.int32)

    def _draw_background(self):
        w, h = self.size
        img = np.full((h, w, 3), 200, np.uint8)
        cv2.fillPoly(img, [self.to_px(boards_polyline(self.spec))], ICE)
        for pts, col, closed in markings(self.spec):
            cv2.polylines(img, [self.to_px(pts)], closed, col, 2, cv2.LINE_AA)
        for p in self.to_px(dots(self.spec)):
            cv2.circle(img, tuple(int(v) for v in p), 4, RED, -1, cv2.LINE_AA)
        return img
