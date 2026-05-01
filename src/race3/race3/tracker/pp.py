"""
Pure Pursuit tracker with curvature/speed adaptive lookahead.

Extracted from the race3 node's inline logic. Interface matches
KinematicMPCTracker so race3 can swap trackers via the `tracker_mode`
parameter.
"""

import numpy as np

from f1tenth_utils.utils import max_speed_from_steering_angle
from f1tenth_utils.vis_utils import visualize_point, visualize_trajectory


class PurePursuitTracker:
    def __init__(
        self,
        wheelbase=0.3302,
        max_speed=5.0,
        mu=0.7,
        max_steering_angle=0.4189,
        l_min=0.4,
        l_max=2.0,
        l_k=0.15,
        l_eps=0.5,
        n_ahead=10,
        p=1.0,
        waypoints=None,
        speed_profile=None,
        use_speed_profile=False,
        sim=False
    ):
        self.wheelbase = wheelbase
        self.max_speed = max_speed
        self.mu = mu
        self.max_steering_angle = max_steering_angle
        self.l_min = l_min
        self.l_max = l_max
        self.l_k = l_k
        self.l_eps = l_eps
        self.n_ahead = n_ahead
        self.p = p
        self.waypoints = waypoints
        self.speed_profile = speed_profile
        self.use_speed_profile = use_speed_profile
        self.base_link_frame = 'ego_racecar/base_link' if sim else 'base_link'

    def compute_drive(self, pose_x, pose_y, pose_yaw, v_current, delta_current,
                      waypoints, stamp=None, vis=False):
        waypoints_xy = waypoints[:, :2]
        pos = np.array([pose_x, pose_y])
        nearest = int(np.argmin(np.linalg.norm(waypoints_xy - pos, axis=1)))

        # Curvature-adaptive lookahead
        kappa = _compute_ahead_curvature(waypoints_xy, nearest, self.n_ahead)
        l = float(np.clip(self.l_min + self.l_k * v_current / (kappa + self.l_eps),
                          self.l_min, self.l_max))

        target, target_idx, actual_l = _find_target_l(waypoints_xy, pos, l)

        # Steer via pure pursuit (kinematic bicycle)
        dx = target[0] - pose_x
        dy = target[1] - pose_y
        target_x = dx * np.cos(pose_yaw) + dy * np.sin(pose_yaw)
        target_y = -dx * np.sin(pose_yaw) + dy * np.cos(pose_yaw)

        gamma = 2.0 * target_y / (actual_l ** 2)
        angle = self.p * np.arctan(gamma * self.wheelbase)
        angle = float(np.clip(angle, -self.max_steering_angle, self.max_steering_angle))

        # Cap speed by lateral-acceleration budget at this steering
        v_cap = max_speed_from_steering_angle(angle, mu=self.mu)
        v = float(min(self.max_speed, v_cap))
        if self.use_speed_profile and self.speed_profile is not None:
            v = float(min(v, self.speed_profile[target_idx]))

        markers = []
        if vis:
            markers.append(visualize_trajectory(angle, 
                                                stamp, 
                                                frame_id=self.base_link_frame, 
                                                color=(1.0, 0.0, 1.0, 1.0),
                                                wheelbase=self.wheelbase))
            markers.append(visualize_point(target, stamp, frame_id='/map'))

        return {
            'steering': angle,
            'speed': v,
            'markers': markers,
            'target': target,
            'target_x': float(target_x),
            'target_y': float(target_y),
            'actual_l': float(actual_l),
        }


def _compute_ahead_curvature(waypoints, start_idx, n_ahead):
    n = len(waypoints)
    max_kappa = 0.0
    for i in range(n_ahead - 2):
        a = waypoints[(start_idx + i    ) % n]
        b = waypoints[(start_idx + i + 1) % n]
        c = waypoints[(start_idx + i + 2) % n]
        cross = abs((b[0]-a[0])*(c[1]-b[1]) - (b[1]-a[1])*(c[0]-b[0]))
        denom = np.linalg.norm(b-a) * np.linalg.norm(c-b) * np.linalg.norm(c-a)
        if denom > 1e-6:
            max_kappa = max(max_kappa, 2 * cross / denom)
    return max_kappa


def _find_target_l(waypoints, pos, l):
    # NOTE: This function loops from waypoint n-1 back to waypoint 0.
    # If waypoint 0 is already more than l meters away from the car at this time, 
    # waypoint 0 will be chosen as the target even though it is behind the car.

    # Boundary at i: dists[i] <= l < dists[(i+1) % n]. Waypoint (i+1) % n
    # is the first one beyond the lookahead radius. np.roll wraps the seam.
    dists = np.linalg.norm(waypoints - pos, axis=1)
    smaller = dists <= l
    boundary = smaller & np.roll(~smaller, -1)
    if boundary.any():
        idx = (int(boundary.nonzero()[0][0]) + 1) % len(waypoints)
        return waypoints[idx], idx, float(dists[idx])
    # No crossing: stay local to the car by advancing a few samples from
    # the nearest waypoint, while still wrapping correctly at the seam.
    nearest = int(np.argmin(dists))
    idx = (nearest + 5) % len(waypoints)
    return waypoints[idx], idx, float(dists[idx])
