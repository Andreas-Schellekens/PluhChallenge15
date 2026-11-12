"""Kalman filter that tracks the puck on the rink (in metres, after the homography).

Why: the puck detector (E2a) sees the puck in only ~18% of the frames, mostly because
it is hidden behind players. On the minimap the puck then blinks on and off, and a
single false detection (a stick blade) makes it jump across the ice.

A Kalman filter keeps a belief about the puck: where it is and how fast it moves,
plus how uncertain that belief is. Every frame it does two steps:
  predict  move the puck forward with its current velocity (constant-velocity model).
           The uncertainty grows, because the puck can change speed or direction.
  update   if there is a detection that fits the prediction, blend the two. The blend
           weight (Kalman gain) depends on which one we trust more: the prediction
           (small uncertainty P) or the measurement (homography noise R).
Without a fitting detection the filter only predicts ("coasting"), for at most
max_coast seconds or until the prediction leaves the rink; after that the puck is
declared lost.

Gating: a detection is only accepted if it lies within the uncertainty ellipse of the
prediction. We use the Mahalanobis distance d^2 = y^T S^-1 y (y = measurement minus
prediction, S = its expected covariance) and accept d^2 < 9.21, the 99% value of a
chi-square distribution with 2 degrees of freedom. A "puck" 20 m away from where the
puck can be is rejected: that removes most false detections.

We filter on the rink, not in the image: in metres the puck moves at a roughly
constant speed, while in the image the perspective squeezes far-away motion.
"""

import numpy as np

CHI2_2DOF_99 = 9.21


class PuckKalman:
    """State s = [x, y, vx, vy] in metres and metres per second."""

    def __init__(self, accel_std=20.0, meas_std=0.7, max_coast=0.5, init_conf=0.4, gate=CHI2_2DOF_99, inside=None):
        # inside(x, y) -> bool: is this position on the rink? While coasting, a prediction that
        # leaves the rink ends the track: the puck cannot be behind the boards. Without this, one
        # wrong detection could give the filter a large velocity and send the puck off the map.
        self.inside = inside
        self.accel_std = accel_std    # how hard the puck can change speed (m/s^2): shots, passes, deflections
        self.meas_std = meas_std      # noise of one detection after the homography (m)
        self.max_coast = max_coast    # seconds we keep predicting without a detection
        self.init_conf = init_conf    # a new track needs a confident detection
        self.gate = gate
        self.s, self.P, self.coast = None, None, 0.0
        self.H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        self.R = np.eye(2) * meas_std ** 2

    @property
    def alive(self):
        return self.s is not None

    def noise(self, det):
        """Measurement covariance R of one detection (x, y, conf[, extra]). The fixed part
        (meas_std) covers the model error of the homography. A detection may carry an extra
        2x2 covariance: the pixel noise of the detector projected onto the rink
        (RinkMapper.rink_noise). Far from the camera that part is metres, so a far-away
        detection pulls the filter less and gets a wider gate."""
        return self.R + det[3] if len(det) > 3 else self.R

    def _start(self, det):
        self.s = np.array([det[0], det[1], 0.0, 0.0])
        # position as uncertain as one measurement; velocity unknown (puck speeds up to ~30 m/s)
        self.P = np.diag([0.0, 0.0, 15.0 ** 2, 15.0 ** 2])
        self.P[:2, :2] = self.noise(det)
        self.coast = 0.0

    def predict(self, dt):
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1.0]])
        # white-noise acceleration: an unknown acceleration a ~ N(0, accel_std^2) during dt
        G = np.array([[dt ** 2 / 2, 0], [0, dt ** 2 / 2], [dt, 0], [0, dt]])
        Q = G @ G.T * self.accel_std ** 2
        self.s = F @ self.s
        self.P = F @ self.P @ F.T + Q

    def mahalanobis2(self, det):
        y = np.asarray(det[:2]) - self.H @ self.s
        S = self.H @ self.P @ self.H.T + self.noise(det)
        return float(y @ np.linalg.solve(S, y))

    def update(self, det):
        y = np.asarray(det[:2]) - self.H @ self.s
        S = self.H @ self.P @ self.H.T + self.noise(det)
        K = self.P @ self.H.T @ np.linalg.inv(S)   # Kalman gain: how much we move towards the measurement
        self.s = self.s + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.coast = 0.0

    def step(self, dt, dets):
        """One frame. dets = list of (x, y, conf) in rink metres, optionally with a 4th element:
        the 2x2 covariance of that detection's pixel noise on the rink (see noise()).
        Returns (x, y, state, d2) with state 'measured', 'predicted', 'started' or 'lost'."""
        if self.alive:
            self.predict(dt)
            d2 = [self.mahalanobis2(d) for d in dets]
            inside = [i for i, v in enumerate(d2) if v < self.gate]
            if inside:
                best = min(inside, key=lambda i: d2[i])  # nearest neighbour inside the gate
                self.update(dets[best])
                return self.s[0], self.s[1], "measured", d2[best]
            self.coast += dt
            on_rink = self.inside is None or self.inside(self.s[0], self.s[1])
            if self.coast <= self.max_coast and on_rink:
                return self.s[0], self.s[1], "predicted", None
            self.s = None  # lost: fall through and try to start again
        strong = [d for d in dets if d[2] >= self.init_conf]
        if strong:
            self._start(max(strong, key=lambda d: d[2]))
            return self.s[0], self.s[1], "started", None
        return None, None, "lost", None


def run(frames, dets_per_frame, dt, inside=None, **params):
    """Filter a whole sequence. frames: sorted frame indices; dets_per_frame: dict frame -> list of
    (x, y, conf[, R]). The step between two listed frames is dt seconds times their index gap."""
    kf, rows, prev = PuckKalman(inside=inside, **params), [], None
    for f in frames:
        step = dt * (f - prev) if prev is not None else dt
        x, y, state, d2 = kf.step(step, dets_per_frame.get(f, []))
        rows.append({"frame": f, "x": x, "y": y, "state": state, "d2": d2,
                     "n_dets": len(dets_per_frame.get(f, []))})
        prev = f
    return rows
