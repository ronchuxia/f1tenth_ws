"""
Cartesian <-> Frenet conversion for a closed-loop raceline.

Adapted from
reference/stack/race_stack_forzaeth/f110_utils/libs/frenet_conversion/
src/frenet_converter/frenet_converter.py

Changes from the source:
- Closes the waypoint loop and uses bc_type='periodic' for both splines
- Pre-samples a dense (x, y, s) lookup table for nearest-s queries
- Vectorized get_cartesian
- Adds get_yaw() and get_curvature() helpers
"""

import numpy as np
from scipy.interpolate import CubicSpline


class FrenetConverter:
    def __init__(self, waypoints_xy, sample_spacing=0.05, newton_iters=3):
        xy = np.asarray(waypoints_xy, dtype=np.float64)
        # Close the loop so periodic BCs work
        xy_closed = np.vstack([xy, xy[0:1]])

        dists = np.linalg.norm(np.diff(xy_closed, axis=0), axis=1)
        s_wp = np.concatenate([[0.0], np.cumsum(dists)])
        self.total_length = float(s_wp[-1])

        self.spline_x = CubicSpline(s_wp, xy_closed[:, 0], bc_type='periodic')
        self.spline_y = CubicSpline(s_wp, xy_closed[:, 1], bc_type='periodic')

        # Dense lookup table for nearest-s
        n_samples = max(1, int(self.total_length / sample_spacing))
        self.s_samples = np.linspace(0.0, self.total_length, n_samples, endpoint=False)
        self.xy_samples = np.column_stack([
            self.spline_x(self.s_samples),
            self.spline_y(self.s_samples),
        ])
        self.newton_iters = newton_iters

    def get_approx_s(self, x, y):
        dx = self.xy_samples[:, 0] - x
        dy = self.xy_samples[:, 1] - y
        return float(self.s_samples[np.argmin(dx * dx + dy * dy)])

    def get_frenet(self, x, y):
        """Return (s, d) of the point (x, y) on the raceline."""
        s = self.get_approx_s(x, y)
        for _ in range(self.newton_iters):
            cx = float(self.spline_x(s))
            cy = float(self.spline_y(s))
            tx = float(self.spline_x(s, 1))
            ty = float(self.spline_y(s, 1))
            tnorm = float(np.hypot(tx, ty))
            tx_u, ty_u = tx / tnorm, ty / tnorm
            proj = (x - cx) * tx_u + (y - cy) * ty_u
            s = (s + proj) % self.total_length

        cx = float(self.spline_x(s))
        cy = float(self.spline_y(s))
        tx = float(self.spline_x(s, 1))
        ty = float(self.spline_y(s, 1))
        tnorm = float(np.hypot(tx, ty))
        tx_u, ty_u = tx / tnorm, ty / tnorm
        nx, ny = -ty_u, tx_u  # left-hand normal
        d = (x - cx) * nx + (y - cy) * ny
        return s, d

    def get_cartesian(self, s, d):
        """Vectorized: s and d can be scalar or array. Returns (x, y)."""
        s_arr = np.asarray(s, dtype=np.float64) % self.total_length
        d_arr = np.asarray(d, dtype=np.float64)
        cx = self.spline_x(s_arr)
        cy = self.spline_y(s_arr)
        tx = self.spline_x(s_arr, 1)
        ty = self.spline_y(s_arr, 1)
        tnorm = np.hypot(tx, ty)
        tx_u = tx / tnorm
        ty_u = ty / tnorm
        nx, ny = -ty_u, tx_u
        return cx + d_arr * nx, cy + d_arr * ny

    def get_yaw(self, s):
        s = float(s) % self.total_length
        tx = float(self.spline_x(s, 1))
        ty = float(self.spline_y(s, 1))
        return float(np.arctan2(ty, tx))

    def get_curvature(self, s):
        s = float(s) % self.total_length
        dx = float(self.spline_x(s, 1))
        dy = float(self.spline_y(s, 1))
        ddx = float(self.spline_x(s, 2))
        ddy = float(self.spline_y(s, 2))
        denom = (dx * dx + dy * dy) ** 1.5
        return (dx * ddy - dy * ddx) / denom if denom > 1e-12 else 0.0
