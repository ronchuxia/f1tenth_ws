import numpy as np
import math

from f1tenth_utils.vis_utils import visualize_point, visualize_points


class GapFollower:
    """
    Disparity extender gap follower. Adapted from
    reference/disparity_extender/reactive_node_extender.py.

    Reactive: scan-only, ignores pose and waypoints. compute_drive() returns
    a (steering_angle, speed, markers) tuple; markers is a list of RViz
    Markers visualizing the inflated free-space scan.
    """
    def __init__(
        self,
        mu=0.7,
        max_steering_angle=0.4189,
        disparity_threshold=0.5,
        bubble_radius=0.3,
        steering_gain=0.5,
        side_guard_range=0.2,
        fov=np.pi,
        min_clear_range=0.5,
        fast_speed=5.0,
        medium_speed=3.0,
        slow_speed=0.5,
        speed_min_range=0.5,
        target_mode='furthest_point',
    ):
        self.mu = mu
        self.max_steering_angle = max_steering_angle
        self.disparity_threshold = disparity_threshold
        self.bubble_radius = bubble_radius
        self.steering_gain = steering_gain
        self.side_guard_range = side_guard_range
        self.fov = fov
        self.min_clear_range = min_clear_range
        self.fast_speed = fast_speed
        self.medium_speed = medium_speed
        self.slow_speed = slow_speed
        self.speed_min_range = speed_min_range
        self.target_mode = target_mode

    def _select_best_index(self, free):
        if self.target_mode == 'furthest_point':
            return int(np.argmax(free))

        if self.target_mode == 'max_gap_center':
            valid = free >= self.min_clear_range
            if not np.any(valid):
                return int(np.argmax(free))

            padded = np.pad(valid.astype(np.int8), (1, 1), constant_values=0)
            transitions = np.diff(padded)
            starts = np.where(transitions == 1)[0]
            ends = np.where(transitions == -1)[0]
            lengths = ends - starts
            longest = np.flatnonzero(lengths == np.max(lengths))

            if longest.size > 1:
                gap_scores = [float(np.max(free[starts[i]:ends[i]])) for i in longest]
                best_gap = int(longest[int(np.argmax(gap_scores))])
            else:
                best_gap = int(longest[0])

            gap_start = int(starts[best_gap])
            gap_end = int(ends[best_gap]) - 1
            return (gap_start + gap_end) // 2

        raise ValueError(f"Unsupported gap target_mode '{self.target_mode}'")

    def compute_drive(self, scan, pose_x=None, pose_y=None, pose_yaw=None, waypoints=None, vis=False, stamp=None):
        # Preprocess: fill NaN/inf with sensor bounds, then clip
        ranges = np.array(scan.ranges, dtype=np.float64)
        ranges[np.isnan(ranges)] = scan.range_min
        ranges[np.isinf(ranges)] = scan.range_max
        ranges = np.clip(ranges, scan.range_min, scan.range_max)

        angles = scan.angle_min + np.arange(ranges.shape[0]) * scan.angle_increment

        # Keep only the forward FOV
        fwd = np.abs(angles) < self.fov / 2.0
        proc = ranges[fwd]
        ang = angles[fwd]
        n = proc.shape[0]

        # Signed disparities: 'left' means the range drops going CCW (range[i] >> range[i+1]);
        # the obstacle edge sits at beam i and we inflate to the left (indices < i).
        # 'right' is the mirror.
        disp_left = np.insert(proc[:-1] - proc[1:], 0, 0.0) > self.disparity_threshold
        disp_right = np.append(proc[1:] - proc[:-1], 0.0) > self.disparity_threshold
        left_idx = np.where(disp_left)[0]
        right_idx = np.where(disp_right)[0]

        # Inflate by perpendicular chord distance: points whose perpendicular distance
        # to the disparity point (at range proc[i]) is below bubble_radius get clamped
        # down to proc[i]. np.minimum guards against ever making a beam longer.
        free = np.copy(proc)
        beams = np.arange(n)
        for i in left_idx:
            chord = proc[i] * np.sin(np.abs(ang - ang[i]))
            bubble = (chord < self.bubble_radius) & (beams < i)
            free[bubble] = np.minimum(proc[i], free[bubble])
        for i in right_idx:
            chord = proc[i] * np.sin(np.abs(ang - ang[i]))
            bubble = (chord < self.bubble_radius) & (beams > i)
            free[bubble] = np.minimum(proc[i], free[bubble])

        # Pick either the furthest beam or the center of the widest clear gap.
        best_idx = self._select_best_index(free)
        best_angle = float(ang[best_idx]) * self.steering_gain
        best_point_range = float(free[best_idx])

        angle_deg = best_angle / math.pi * 180
        if abs(angle_deg) < 10 and best_point_range > self.speed_min_range:
            speed = self.fast_speed
        elif abs(angle_deg) < 20 and best_point_range > self.speed_min_range:
            speed = self.medium_speed
        else:
            speed = self.slow_speed

        # Don't steer toward a wall within side_guard_range on that side
        if best_angle < 0.0 and np.any(proc[ang < 0] < self.side_guard_range):
            best_angle = 0.0
        elif best_angle > 0.0 and np.any(proc[ang > 0] < self.side_guard_range):
            best_angle = 0.0

        steering_angle = float(np.clip(best_angle, -self.max_steering_angle, self.max_steering_angle))

        markers = []
        if vis:
            free_pts = np.column_stack([free * np.cos(ang), free * np.sin(ang)])
            markers.append(visualize_points(
                free_pts,
                scan.header.stamp,
                frame_id=scan.header.frame_id,
                ns='gap_free',
                id=10,
                color=(1.0, 0.5, 0.0, 1.0),
            ))
            target = (float(free[best_idx] * np.cos(ang[best_idx])),
                      float(free[best_idx] * np.sin(ang[best_idx])))
            markers.append(visualize_point(
                target,
                scan.header.stamp,
                frame_id=scan.header.frame_id,
                ns='gap_target',
                id=11,
                color=(1.0, 0.0, 1.0, 1.0),
            ))

        return steering_angle, speed, markers
