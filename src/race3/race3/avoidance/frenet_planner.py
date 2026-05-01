"""
Werling-style Frenet planner with arc-length parameterization (d(s)).

Samples a fan of lateral offsets at a fixed look-ahead arc length, fits a
quintic d(s) from the current Frenet state to each (d_target, 0, 0) terminal
state, converts each (s, d) sample back to Cartesian, transforms into
base_link, and picks the lowest-cost candidate that clears the scan.

Steering is extracted from the winning trajectory via a pure-pursuit-style
target (the far endpoint in base_link).

Interface (matches GapFollower):
    compute_drive(scan, pose_x, pose_y, pose_yaw, waypoints, vis=False)
        -> (steering, speed, markers)

References:
- PythonRobotics FrenetOptimalTrajectory (LowSpeedLateralMovementStrategy)
  for the d(s) formulation
- ForzaETH race_stack FrenetConverter for the spline / Newton projection
"""

import numpy as np

from f1tenth_utils.utils import max_speed_from_steering_angle
from f1tenth_utils.vis_utils import visualize_path, visualize_point

from race3.avoidance.frenet_converter import FrenetConverter
from race3.avoidance.quintic_polynomial import QuinticPolynomial


def _wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


class FrenetPlanner:
    def __init__(
        self,
        max_speed=5.0,
        mu=0.7,
        p=1.0,
        max_steering_angle=0.4189,
        wheelbase=0.3302,
        laser_offset=0.27,
        car_width=0.35,
        lookahead=2.0,
        num_offsets=7,
        offset_spacing=0.20,
        num_samples=20,
        w_offset=1.0,
        w_jerk=0.01,
        w_clear=0.5,
        pp_lookahead=0.8,
        waypoints=None,
        sim=False,
    ):
        if waypoints is None:
            raise ValueError("FrenetPlanner needs waypoints to build the raceline spline")

        self.max_speed = max_speed
        self.mu = mu
        self.p = p
        self.max_steering_angle = max_steering_angle
        self.wheelbase = wheelbase
        self.laser_offset = laser_offset
        self.car_width = car_width
        self.lookahead = lookahead
        self.num_offsets = num_offsets
        self.offset_spacing = offset_spacing
        self.num_samples = num_samples
        self.w_offset = w_offset
        self.w_jerk = w_jerk
        self.w_clear = w_clear
        self.pp_lookahead = pp_lookahead
        self.base_link_frame = 'ego_racecar/base_link' if sim else 'base_link'

        self.converter = FrenetConverter(waypoints)

    def compute_drive(self, scan, pose_x, pose_y, pose_yaw, waypoints, vis=False, stamp=None):
        # 1. Current Frenet state
        s0, d0 = self.converter.get_frenet(pose_x, pose_y)
        ref_yaw = self.converter.get_yaw(s0)
        dpsi = _wrap(pose_yaw - ref_yaw)
        kappa_r = self.converter.get_curvature(s0)
        # d'(s) = (1 - κ·d) · tan(Δψ)
        one_minus_kd = max(1.0 - kappa_r * d0, 1e-3)
        dprime0 = one_minus_kd * np.tan(np.clip(dpsi, -1.3, 1.3))
        ddprime0 = 0.0  # quartic-at-start approximation

        # 2. Arc-length sample grid
        ds = np.linspace(0.0, self.lookahead, self.num_samples)
        s_samples = (s0 + ds) % self.converter.total_length

        # 3. Scan into base_link (forward only)
        ranges = np.asarray(scan.ranges, dtype=np.float64)
        angs = scan.angle_min + np.arange(ranges.shape[0]) * scan.angle_increment
        valid = np.isfinite(ranges) & (ranges > scan.range_min) & (ranges < scan.range_max)
        sx = self.laser_offset + ranges * np.cos(angs)
        sy = ranges * np.sin(angs)
        keep = valid & (sx > 0.0)
        sxy = np.column_stack([sx[keep], sy[keep]])

        # 4. Candidate lateral offsets (centered on raceline, d=0)
        half = (self.num_offsets - 1) // 2
        offsets = (np.arange(self.num_offsets) - half) * self.offset_spacing

        # 5. Build a quintic d(s) per candidate, convert to base_link
        trajs_bl = np.empty((self.num_offsets, self.num_samples, 2))
        jerk_costs = np.empty(self.num_offsets)
        c_yaw, s_yaw = np.cos(pose_yaw), np.sin(pose_yaw)

        for i, d_target in enumerate(offsets):
            q = QuinticPolynomial(
                xs=d0, vxs=dprime0, axs=ddprime0,
                xe=float(d_target), vxe=0.0, axe=0.0,
                time=self.lookahead,
            )
            d_vals = q.a0 + q.a1 * ds + q.a2 * ds ** 2 \
                + q.a3 * ds ** 3 + q.a4 * ds ** 4 + q.a5 * ds ** 5

            xg, yg = self.converter.get_cartesian(s_samples, d_vals)

            dxg = xg - pose_x
            dyg = yg - pose_y
            trajs_bl[i, :, 0] =  dxg * c_yaw + dyg * s_yaw
            trajs_bl[i, :, 1] = -dxg * s_yaw + dyg * c_yaw

            jerks = 6 * q.a3 + 24 * q.a4 * ds + 60 * q.a5 * ds ** 2
            jerk_costs[i] = float(np.sum(jerks * jerks))

        # 6. Clearance test
        half_w = self.car_width / 2.0
        min_d = np.full(self.num_offsets, np.inf)
        first_hit = np.full(self.num_offsets, self.num_samples, dtype=int)
        if sxy.shape[0] > 0:
            for i in range(self.num_offsets):
                diff = trajs_bl[i, :, None, :] - sxy[None, :, :]
                d = np.linalg.norm(diff, axis=2)
                per_sample_min = d.min(axis=1)
                min_d[i] = float(per_sample_min.min())
                violations = np.where(per_sample_min < half_w)[0]
                if violations.size > 0:
                    first_hit[i] = int(violations[0])
        cleared = min_d >= half_w

        # 7. Pick winner among cleared candidates
        best_idx = -1
        best_cost = np.inf
        clear_cap = self.lookahead
        for i in range(self.num_offsets):
            if not cleared[i]:
                continue
            cost = (
                self.w_offset * abs(offsets[i])
                + self.w_jerk * jerk_costs[i]
                - self.w_clear * min(min_d[i], clear_cap)
            )
            if cost < best_cost:
                best_cost = cost
                best_idx = i

        clear = best_idx >= 0
        if not clear:
            # Buy time: pick the candidate that travels furthest before hitting
            # an obstacle, breaking ties by overall clearance.
            order = np.lexsort((min_d, first_hit))
            best_idx = int(order[-1])

        # 8. Steering from winning trajectory: PP target at `pp_lookahead`
        #    arc length along the trajectory. Fall back to the far endpoint
        #    if the trajectory is shorter than pp_lookahead.
        winner = trajs_bl[best_idx]
        fwd_mask = winner[:, 0] > 0.0
        if not np.any(fwd_mask):
            return 0.0, 0.0, []
        dists = np.hypot(winner[:, 0], winner[:, 1])
        past = np.where(fwd_mask & (dists >= self.pp_lookahead))[0]
        if past.size > 0:
            target_idx = int(past[0])
        else:
            target_idx = int(np.where(fwd_mask)[0][-1])
        tx_l, ty_l = float(winner[target_idx, 0]), float(winner[target_idx, 1])
        L = float(np.hypot(tx_l, ty_l))
        if L < 1e-3:
            steering = 0.0
        else:
            gamma = 2.0 * ty_l / (L * L)
            steering = float(self.p * np.arctan(gamma * self.wheelbase))
        steering = float(np.clip(steering, -self.max_steering_angle, self.max_steering_angle))

        safe = max_speed_from_steering_angle(max(abs(steering), 1e-3), mu=self.mu)
        speed = float(min(self.max_speed, safe))

        markers = []
        if vis:
            winner_color = (1.0, 1.0, 0.0, 1.0) if clear else (1.0, 0.4, 0.0, 1.0)
            for i in range(self.num_offsets):
                if i == best_idx:
                    color = winner_color
                elif cleared[i]:
                    color = (0.1, 0.7, 0.1, 0.35)
                else:
                    color = (0.8, 0.1, 0.1, 0.35)
                markers.append(visualize_path(
                    trajs_bl[i], scan.header.stamp,
                    ns='frenet_candidates', id=200 + i, color=color,
                    frame_id=self.base_link_frame
                ))
            markers.append(visualize_point(
                (tx_l, ty_l), scan.header.stamp,
                frame_id=self.base_link_frame,
                ns='frenet_target', id=260, color=(0.0, 0.5, 1.0, 1.0),
            ))

        return steering, speed, markers
