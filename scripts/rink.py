"""NHL rink template: dimensions in feet and a top-down drawing for the minimap.

Coordinate system (used by every script that talks about rink positions):
  - origin at centre ice, units in feet
  - X along the length of the rink, -100 .. +100 (our filmed end zone is X < 0)
  - Y across the width, -42.5 .. +42.5, +Y is the camera side of the rink
On the minimap X points right and Y points down, so the camera side is at the
bottom, like a TV broadcast.

All marking positions are official NHL dimensions (rink 200 x 85 ft, corner
radius 28 ft, goal line 11 ft from the end boards, blue lines 25 ft from centre,
end-zone dots 20 ft out from the goal line and 22 ft either side of the middle,
neutral-zone dots 5 ft outside the blue line).
"""

import cv2
import numpy as np

LENGTH, WIDTH, CORNER_R = 200.0, 85.0, 28.0
HALF_L, HALF_W = LENGTH / 2, WIDTH / 2
GOAL_LINE_X = 89.0
BLUE_LINE_X = 25.0
CIRCLE_R = 15.0
END_DOT_X, DOT_Y = 69.0, 22.0
NEUTRAL_DOT_X = 20.0
CREASE_R = 6.0

# BGR colours for the drawing
ICE, BOARDS = (245, 245, 245), (60, 60, 60)
RED, BLUE = (40, 40, 200), (200, 90, 30)


def boards_polyline(n_per_corner=12):
    """Outline of the boards as a closed polygon in rink feet (rounded corners)."""
    pts = []
    corners = [(HALF_L - CORNER_R, HALF_W - CORNER_R, 0), (-(HALF_L - CORNER_R), HALF_W - CORNER_R, 90),
               (-(HALF_L - CORNER_R), -(HALF_W - CORNER_R), 180), (HALF_L - CORNER_R, -(HALF_W - CORNER_R), 270)]
    for cx, cy, start in corners:
        for a in np.radians(np.linspace(start, start + 90, n_per_corner)):
            pts.append((cx + CORNER_R * np.cos(a), cy + CORNER_R * np.sin(a)))
    return np.array(pts)


def goal_line_half_length():
    """Goal lines stop at the rounded corners: |Y| where the corner arc crosses X = 89."""
    dx = GOAL_LINE_X - (HALF_L - CORNER_R)
    return (HALF_W - CORNER_R) + np.sqrt(CORNER_R ** 2 - dx ** 2)


def circle(cx, cy, r, n=48):
    a = np.linspace(0, 2 * np.pi, n)
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], axis=1)


def markings():
    """All rink markings as (polyline in feet, colour, closed). Used both to draw the
    minimap and to project the template back onto video frames to check a homography."""
    gy = goal_line_half_length()
    items = [(boards_polyline(), BOARDS, True),
             (np.array([[0, -HALF_W], [0, HALF_W]]), RED, False),           # centre line
             (circle(0, 0, CIRCLE_R), BLUE, True)]                           # centre circle
    for s in (-1, 1):
        items.append((np.array([[s * GOAL_LINE_X, -gy], [s * GOAL_LINE_X, gy]]), RED, False))
        items.append((np.array([[s * BLUE_LINE_X, -HALF_W], [s * BLUE_LINE_X, HALF_W]]), BLUE, False))
        for y in (-DOT_Y, DOT_Y):
            items.append((circle(s * END_DOT_X, y, CIRCLE_R), RED, True))
        a = np.linspace(-np.pi / 2, np.pi / 2, 16)                            # goal crease (half circle)
        items.append((np.stack([s * (GOAL_LINE_X - CREASE_R * np.cos(a)), CREASE_R * np.sin(a)], 1), BLUE, False))
    return items


def dots():
    """Faceoff dots in feet (centre, 4 end-zone, 4 neutral-zone)."""
    d = [(0.0, 0.0)]
    for s in (-1, 1):
        for y in (-DOT_Y, DOT_Y):
            d += [(s * END_DOT_X, y), (s * NEUTRAL_DOT_X, y)]
    return np.array(d)


class Minimap:
    """Draws the rink top-down and converts rink feet to minimap pixels."""

    def __init__(self, px_per_ft=4.0, margin_ft=5.0):
        self.s, self.m = px_per_ft, margin_ft
        self.size = (int((LENGTH + 2 * margin_ft) * px_per_ft), int((WIDTH + 2 * margin_ft) * px_per_ft))  # (w, h)
        self.background = self._draw_background()

    def to_px(self, xy):
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        return np.round((xy + [HALF_L + self.m, HALF_W + self.m]) * self.s).astype(np.int32)

    def _draw_background(self):
        w, h = self.size
        img = np.full((h, w, 3), 200, np.uint8)
        cv2.fillPoly(img, [self.to_px(boards_polyline())], ICE)
        for pts, col, closed in markings():
            cv2.polylines(img, [self.to_px(pts)], closed, col, 2, cv2.LINE_AA)
        for p in self.to_px(dots()):
            cv2.circle(img, tuple(int(v) for v in p), 4, RED, -1, cv2.LINE_AA)
        return img
