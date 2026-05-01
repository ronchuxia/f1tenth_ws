#!/usr/bin/env python3

"""
An odometry tuner that drives the car in a circle (or a straight line). And:
1. Tune the steering_angle_to_servo_offset and steering_to_servo_gain by fitting a circle to the particle filter trajectory and comparing its curvature to the theoretical curvature. 
2. Tune the speed_to_erpm_gain by estimating longitudinal velocity and comparing them to the commanded longitudinal velocity.
3. Tune the speed_to_erpm_gain by comparing the distance traveled from particle filter to the distance traveled from wheel odometry.
"""

import math
from collections import deque
import numpy as np
from scipy.linalg import eig as _scipy_eig
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from ackermann_msgs.msg import AckermannDriveStamped
from tf_transformations import euler_from_quaternion
from visualization_msgs.msg import MarkerArray
from f1tenth_utils.vis_utils import visualize_path, visualize_point, visualize_trajectory
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def lsq_circle_fit(pts):
    """Taubin bias-corrected circle fit. Returns (cx, cy, r, sign) or None if degenerate."""
    xs = np.array([p[0] for p in pts], dtype=float)
    ys = np.array([p[1] for p in pts], dtype=float)

    mx, my = xs.mean(), ys.mean()
    u, v = xs - mx, ys - my
    zi = u**2 + v**2

    Mz  = zi.mean()
    Mxx = (u * u).mean()
    Myy = (v * v).mean()
    Mxy = (u * v).mean()
    Mxz = (u * zi).mean()
    Myz = (v * zi).mean()
    Mzz = (zi * zi).mean()

    Z = np.array([
        [Mzz, Mxz, Myz, Mz],
        [Mxz, Mxx, Mxy, 0.],
        [Myz, Mxy, Myy, 0.],
        [Mz,  0.,  0.,  1.],
    ])
    N = np.diag([4. * Mz, 1., 1., 0.])

    vals, vecs = _scipy_eig(Z, N)

    # Keep real, finite, positive eigenvalues; take the smallest
    mask = np.isfinite(vals) & (np.abs(vals.imag) < 1e-8 * (np.abs(vals.real) + 1.)) & (vals.real > 0)
    if not mask.any():
        return None
    a = vecs[:, np.where(mask)[0][np.argmin(vals.real[mask])]].real

    a1, a2, a3, a4 = a
    if abs(a1) < 1e-12:
        return None
    cx = -a2 / (2. * a1) + mx
    cy = -a3 / (2. * a1) + my
    disc = a2**2 + a3**2 - 4. * a1 * a4
    if disc < 0.:
        return None
    r = math.sqrt(disc) / (2. * abs(a1))
    if r < 1e-3:
        return None

    cross = ((pts[1][0] - pts[0][0]) * (pts[2][1] - pts[0][1]) -
             (pts[1][1] - pts[0][1]) * (pts[2][0] - pts[0][0]))
    return cx, cy, r, math.copysign(1.0, cross)


def lsq_curvature(pts):
    """Signed curvature via least-squares circle fit (Kasa method) over all pts."""
    fit = lsq_circle_fit(pts)
    return 0.0 if fit is None else fit[3] / fit[2]


def fitted_arc(pts, cx, cy, r, sign, n=40):
    """Sample n points along the fitted circle arc spanning the history window."""
    t0 = math.atan2(pts[0][1] - cy, pts[0][0] - cx)
    t1 = math.atan2(pts[-1][1] - cy, pts[-1][0] - cx)
    if sign > 0 and t1 < t0:   # CCW: ensure t1 > t0
        t1 += 2 * math.pi
    elif sign < 0 and t1 > t0:  # CW: ensure t1 < t0
        t1 -= 2 * math.pi
    return [[cx + r * math.cos(t0 + (t1 - t0) * i / (n - 1)),
             cy + r * math.sin(t0 + (t1 - t0) * i / (n - 1))]
            for i in range(n)]


class OdomTuner(Node):
    def __init__(self):
        super().__init__('odom_tuner')
        self.declare_parameter('speed', 1.0)           # m/s
        self.declare_parameter('steering_angle', 0.3)  # rad
        self.declare_parameter('wheelbase', 0.3302)    # m (F1TENTH default)
        self.declare_parameter('laser_offset', 0.27)   # m, lidar forward offset from base_link
        self.declare_parameter('curve_window', 20) # points for LSQ circle fit
        self.declare_parameter('vel_window', 10)       # points for windowed speed estimate
        self.declare_parameter('vel_project', True)    # project displacement onto heading
        self.declare_parameter('enable_vel_window', True)   # enable windowed speed estimate
        self.declare_parameter('vel_window_ema_alpha', 1.0) # EMA weight on new sample (0,1]; 1.0=no smoothing
        self.declare_parameter('enable_vel_ctrv', True)
        self.declare_parameter('kf_ctrv_sigma_a', 1.0)         # longitudinal acceleration std (m/s²)
        self.declare_parameter('kf_ctrv_sigma_omega_dot', 0.5) # angular acceleration std (rad/s²)
        self.declare_parameter('kf_ctrv_r_pos', 0.1)           # position measurement std (m)
        self.declare_parameter('kf_ctrv_r_yaw', 0.05)          # yaw measurement std (rad)
        self.declare_parameter('kf_ctrv_r_from_cov', False)    # use PF covariance for R
        self.declare_parameter('kf_ctrv_p0_v', 10.0)           # initial P: v variance (m²/s²)
        self.declare_parameter('kf_ctrv_p0_omega', 1.0)        # initial P: ω variance (rad²/s²)
        self.declare_parameter('g', 9.81)              # m/s², scale raw IMU output to SI
        self.declare_parameter('trail_length', 5000)   # points for trail visualisation
        self.declare_parameter('show_curve_hist', True)    # curvature window path
        self.declare_parameter('show_vel_hist',  True)    # velocity window path
        self.declare_parameter('show_trail',  True)    # long position trail
        self.declare_parameter('show_arc',    True)    # fitted circle arc
        self.declare_parameter('show_center', True)    # fitted circle centre
        self.declare_parameter('show_theory', True)    # theoretical Ackermann trajectory
        self.declare_parameter('show_arc_thru_base_link', False)  # shifted arc passing through base_link
        self.declare_parameter('log_imu',  True)       # log IMU messages
        self.declare_parameter('log_odom', True)       # log wheel odometry messages
        self.declare_parameter('log_pf',   True)       # log particle-filter messages
        self.declare_parameter('record_estimates', True)  # save v and dist history and plot on shutdown
        self.declare_parameter('drive_start_delay', 1.0)  # seconds to wait before publishing to /drive

        self._drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self._markers_pub = self.create_publisher(MarkerArray, '/rviz_markers', 10)
        self.create_subscription(Odometry, '/odom', self._odom_cb, 10)
        self.create_subscription(Imu, '/sensors/imu/raw', self._imu_cb, 10)
        self.create_subscription(Odometry, '/pf/pose/odom', self._pf_cb, 10)

        # Cached parameters (all fixed at startup)
        self._speed             = self.get_parameter('speed').value
        self._steering_angle    = self.get_parameter('steering_angle').value
        self._wheelbase         = self.get_parameter('wheelbase').value
        self._laser_offset      = self.get_parameter('laser_offset').value
        self._vel_project       = self.get_parameter('vel_project').value
        self._enable_vel_window    = self.get_parameter('enable_vel_window').value
        self._vel_window_ema_alpha = self.get_parameter('vel_window_ema_alpha').value
        self._enable_vel_ctrv         = self.get_parameter('enable_vel_ctrv').value
        self._kf_ctrv_sigma_a         = self.get_parameter('kf_ctrv_sigma_a').value
        self._kf_ctrv_sigma_omega_dot = self.get_parameter('kf_ctrv_sigma_omega_dot').value
        self._kf_ctrv_r_pos           = self.get_parameter('kf_ctrv_r_pos').value
        self._kf_ctrv_r_yaw           = self.get_parameter('kf_ctrv_r_yaw').value
        self._kf_ctrv_r_from_cov      = self.get_parameter('kf_ctrv_r_from_cov').value
        self._kf_ctrv_p0_v            = self.get_parameter('kf_ctrv_p0_v').value
        self._kf_ctrv_p0_omega        = self.get_parameter('kf_ctrv_p0_omega').value
        self._g                 = self.get_parameter('g').value
        self._log_odom          = self.get_parameter('log_odom').value
        self._log_imu           = self.get_parameter('log_imu').value
        self._log_pf            = self.get_parameter('log_pf').value
        self._record_estimates  = self.get_parameter('record_estimates').value
        self._show_curve_hist   = self.get_parameter('show_curve_hist').value
        self._show_vel_hist     = self.get_parameter('show_vel_hist').value
        self._show_trail        = self.get_parameter('show_trail').value
        self._show_arc          = self.get_parameter('show_arc').value
        self._show_center       = self.get_parameter('show_center').value
        self._show_theory       = self.get_parameter('show_theory').value
        self._show_arc_thru_base_link = self.get_parameter('show_arc_thru_base_link').value

        # Theoretical curvature from Ackermann model: κ = tan(δ) / L
        self._k_theory = math.tan(self._steering_angle) / self._wheelbase if self._wheelbase > 0.0 else 0.0

        # IMU integration state
        self._imu_vel = 0.0
        self._imu_t = None
        self._imu_acc = 0.0

        # Initial positions for delta logging
        self._odom_init_pos = None
        self._odom_init_yaw = None
        self._pf_init_pos = None
        self._pf_init_yaw = None

        # PF curvature + velocity state
        self._pf_curve_hist  = deque(maxlen=self.get_parameter('curve_window').value)
        self._pf_vel_hist    = deque(maxlen=self.get_parameter('vel_window').value)
        self._pf_trail       = deque(maxlen=self.get_parameter('trail_length').value)
        self._v_window_ema: float | None = None  # EMA state; None until first estimate

        # CTRV EKF state: [px, py, ψ, v, ω]
        self._ctrv_x = np.zeros(5)
        self._ctrv_P = np.diag([
            self._kf_ctrv_r_pos ** 2,
            self._kf_ctrv_r_pos ** 2,
            self._kf_ctrv_r_yaw ** 2,
            self._kf_ctrv_p0_v,
            self._kf_ctrv_p0_omega,
        ])
        self._ctrv_t: float | None = None

        # Recording buffers (used when record_estimates=True)
        self._rec_t0: float = 0.0
        self._rec_t: list[float] = []
        self._rec_v_win: list[float] = []
        self._rec_v_ctrv: list[float] = []
        self._rec_dist: list[float] = []
        self._rec_t_odom: list[float] = []
        self._rec_v_odom: list[float] = []
        self._rec_dist_odom: list[float] = []
        self._rec_t_imu: list[float] = []
        self._rec_v_imu: list[float] = []
        self._rec_acc_x: list[float] = []
        self._rec_acc_y: list[float] = []
        self._rec_acc_z: list[float] = []

        self._drive_start_delay = Duration(
            seconds=self.get_parameter('drive_start_delay').value)
        self._node_start_time = self.get_clock().now()
        self.create_timer(0.05, self._timer_cb)  # 20 Hz

    def _timer_cb(self):
        if self.get_clock().now() - self._node_start_time < self._drive_start_delay:
            return
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.drive.speed = self._speed
        msg.drive.steering_angle = self._steering_angle
        self._drive_pub.publish(msg)

    def _odom_cb(self, msg: Odometry):
        v = msg.twist.twist.linear.x
        w = msg.twist.twist.angular.z
        k = w / v if abs(v) > 1e-3 else 0.0
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        if self._odom_init_pos is None:
            self._odom_init_pos = (x, y)
            self._odom_init_yaw = yaw
        dx, dy = x - self._odom_init_pos[0], y - self._odom_init_pos[1]
        dist = math.hypot(dx, dy)
        if self._log_odom:
            self.get_logger().info(f'odom    k={k:10.4f}  k_theory={self._k_theory:10.4f}  dx={dx:7.3f}  dy={dy:7.3f}  dist={dist:7.3f}  v={v:6.3f}')
        if self._record_estimates:
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self._rec_t_odom.append(t)
            self._rec_v_odom.append(v)
            self._rec_dist_odom.append(dist)

    def _imu_cb(self, msg: Imu):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        acc = - msg.linear_acceleration.x * self._g
        if self._imu_t is not None:
            dt = t - self._imu_t
            if dt > 0.0:
                self._imu_vel += 0.5 * (self._imu_acc + acc) * dt
        self._imu_t = t
        self._imu_acc = acc
        if self._log_imu:
            self.get_logger().info(f'imu     v={self._imu_vel:6.3f}')
        if self._record_estimates:
            self._rec_t_imu.append(t)
            self._rec_v_imu.append(self._imu_vel)
            self._rec_acc_x.append(msg.linear_acceleration.x * self._g)
            self._rec_acc_y.append(msg.linear_acceleration.y * self._g)
            self._rec_acc_z.append(msg.linear_acceleration.z * self._g)

    def _pf_cb(self, msg: Odometry):
        # Windowed speed estimate + LSQ circle-fit curvature
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])[2]
        x -= self._laser_offset * math.cos(yaw)
        y -= self._laser_offset * math.sin(yaw)
        self._pf_vel_hist.append((x, y, yaw, t))
        v_window = None
        v_ctrv   = None

        if len(self._pf_vel_hist) >= 2:
            pts = list(self._pf_vel_hist)

            # windowed arc-length estimate
            if self._enable_vel_window:
                elapsed = pts[-1][3] - pts[0][3]
                if elapsed > 0.0:
                    if self._vel_project:
                        arc = sum(
                            (pts[i+1][0] - pts[i][0]) * math.cos(pts[i][2]) +
                            (pts[i+1][1] - pts[i][1]) * math.sin(pts[i][2])
                            for i in range(len(pts) - 1)
                        )
                    else:
                        arc = sum(
                            math.hypot(pts[i+1][0] - pts[i][0], pts[i+1][1] - pts[i][1])
                            for i in range(len(pts) - 1)
                        )
                    v_window = arc / elapsed
                    alpha = self._vel_window_ema_alpha
                    if self._v_window_ema is None:
                        self._v_window_ema = v_window
                    else:
                        self._v_window_ema = alpha * v_window + (1.0 - alpha) * self._v_window_ema
                    v_window = self._v_window_ema

        # CTRV EKF estimate
        if self._enable_vel_ctrv:
            v_ctrv = self._ctrv_ekf_step(x, y, yaw, t, msg.pose.covariance)

        if self._pf_init_pos is None:
            self._pf_init_pos = (x, y)
            self._pf_init_yaw = yaw
        pf_dx, pf_dy = x - self._pf_init_pos[0], y - self._pf_init_pos[1]
        if self._odom_init_yaw is not None:
            angle = self._odom_init_yaw - self._pf_init_yaw
            c, s = math.cos(angle), math.sin(angle)
            pf_dx, pf_dy = c * pf_dx - s * pf_dy, s * pf_dx + c * pf_dy
        self._pf_curve_hist.append((x, y))
        self._pf_trail.append((x, y))
        k = 0.0
        fit = None
        if len(self._pf_curve_hist) == self._pf_curve_hist.maxlen:
            fit = lsq_circle_fit(list(self._pf_curve_hist))
            if fit is not None:
                cx, cy, r, sign = fit
                k = sign / r
        pf_dist = math.hypot(pf_dx, pf_dy)
        if self._log_pf:
            parts = [f'k={k:10.4f}  k_theory={self._k_theory:10.4f}  dx={pf_dx:7.3f}  dy={pf_dy:7.3f}  dist={pf_dist:7.3f}']
            if v_window is not None:
                parts.append(f'v_win={v_window:6.3f}')
            if v_ctrv is not None:
                parts.append(f'v_ctrv={v_ctrv:6.3f}')
            self.get_logger().info('pf      ' + '  '.join(parts))

        if self._record_estimates:
            if not self._rec_t:
                self._rec_t0 = t
            self._rec_t.append(t - self._rec_t0)
            self._rec_v_win.append(v_window if v_window is not None else float('nan'))
            self._rec_v_ctrv.append(v_ctrv if v_ctrv is not None else float('nan'))
            self._rec_dist.append(pf_dist)

        stamp = msg.header.stamp
        markers = MarkerArray()
        if self._show_curve_hist:
            markers.markers.append(visualize_path(
                list(self._pf_curve_hist), stamp,
                frame_id='map', ns='pf_curve_hist', id=0, color=(0.0, 1.0, 0.0, 1.0)))
        if self._show_vel_hist:
            markers.markers.append(visualize_path(
                [(p[0], p[1]) for p in self._pf_vel_hist], stamp,
                frame_id='map', ns='pf_vel_hist', id=5, color=(0.0, 0.5, 1.0, 1.0)))
        if self._show_trail:
            markers.markers.append(visualize_path(
                list(self._pf_trail), stamp,
                frame_id='map', ns='pf_trail', id=1, color=(1.0, 0.5, 0.0, 0.6)))
        if fit is not None:
            cx, cy, r, sign = fit
            if self._show_arc:
                markers.markers.append(visualize_path(
                    fitted_arc(list(self._pf_curve_hist), cx, cy, r, sign), stamp,
                    frame_id='map', ns='pf_arc', id=2, color=(0.0, 0.8, 1.0, 1.0)))
            if self._show_center:
                markers.markers.append(visualize_point(
                    (cx, cy), stamp,
                    frame_id='map', ns='pf_center', id=3, color=(1.0, 1.0, 0.0, 1.0)))
            if self._show_arc_thru_base_link:
                # Shift the fitted circle so it passes through the current base_link
                # position (x, y).  Keep the same radius and the centre on the same
                # side: new_center = (x, y) + r * unit_vec_from_(x,y)_toward_(cx,cy)
                dist = math.hypot(cx - x, cy - y)
                if dist > 1e-6:
                    cx2 = x + r * (cx - x) / dist
                    cy2 = y + r * (cy - y) / dist
                    markers.markers.append(visualize_path(
                        fitted_arc(list(self._pf_curve_hist), cx2, cy2, r, sign), stamp,
                        frame_id='map', ns='pf_arc_bl', id=6, color=(1.0, 0.4, 0.0, 1.0)))
                    markers.markers.append(visualize_point(
                        (cx2, cy2), stamp,
                        frame_id='map', ns='pf_center_bl', id=7, color=(1.0, 0.8, 0.0, 1.0)))
        if self._show_theory:
            markers.markers.append(visualize_trajectory(
                self._steering_angle, stamp,
                frame_id='base_link', ns='pf_theory', id=4, color=(1.0, 0.0, 1.0, 0.8),
                wheelbase=self._wheelbase))
        self._markers_pub.publish(markers)

    def _ctrv_ekf_step(self, px: float, py: float, yaw: float, t: float, pf_cov) -> float | None:
        """CTRV EKF predict + update. Returns filtered speed v, or None on the first call."""
        if self._ctrv_t is None:
            self._ctrv_x[:] = [px, py, yaw, 0.0, 0.0]
            self._ctrv_t = t
            return None

        dt = t - self._ctrv_t
        if dt <= 0.0:
            return None
        self._ctrv_t = t

        p_x, p_y, psi, v, omega = self._ctrv_x
        psi1 = psi + omega * dt

        # --- Predict: state ---
        if abs(omega) > 1e-4:
            v_o = v / omega
            x1 = p_x + v_o * (math.sin(psi1) - math.sin(psi))
            y1 = p_y + v_o * (-math.cos(psi1) + math.cos(psi))
            F = np.eye(5)
            F[0, 2] = v_o * (math.cos(psi1) - math.cos(psi))
            F[0, 3] = (math.sin(psi1) - math.sin(psi)) / omega
            F[0, 4] = (v / omega) * math.cos(psi1) * dt - (v / omega ** 2) * (math.sin(psi1) - math.sin(psi))
            F[1, 2] = v_o * (math.sin(psi1) - math.sin(psi))
            F[1, 3] = (-math.cos(psi1) + math.cos(psi)) / omega
            F[1, 4] = (v / omega) * math.sin(psi1) * dt + (v / omega ** 2) * (math.cos(psi1) - math.cos(psi))
            F[2, 4] = dt
        else:
            x1 = p_x + v * math.cos(psi) * dt
            y1 = p_y + v * math.sin(psi) * dt
            F = np.eye(5)
            F[0, 2] = -v * math.sin(psi) * dt
            F[0, 3] = math.cos(psi) * dt
            F[1, 2] =  v * math.cos(psi) * dt
            F[1, 3] = math.sin(psi) * dt
            F[2, 4] = dt
        self._ctrv_x = np.array([x1, y1, psi1, v, omega])

        # Process noise Q via noise Jacobian G (noise inputs: σ_a, σ_ω̇)
        G = np.array([
            [0.5 * math.cos(psi) * dt ** 2, 0.0],
            [0.5 * math.sin(psi) * dt ** 2, 0.0],
            [0.0,                            0.5 * dt ** 2],
            [dt,                             0.0],
            [0.0,                            dt],
        ])
        Qv = np.diag([self._kf_ctrv_sigma_a ** 2, self._kf_ctrv_sigma_omega_dot ** 2])
        Q = G @ Qv @ G.T
        self._ctrv_P = F @ self._ctrv_P @ F.T + Q

        # --- Update: z = [px, py, ψ] from PF ---
        H = np.zeros((3, 5))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        z = np.array([px, py, yaw])
        if self._kf_ctrv_r_from_cov:
            R = np.diag([max(pf_cov[0], 1e-6), max(pf_cov[7], 1e-6), max(pf_cov[35], 1e-6)])
        else:
            R = np.diag([self._kf_ctrv_r_pos ** 2, self._kf_ctrv_r_pos ** 2, self._kf_ctrv_r_yaw ** 2])
        S = H @ self._ctrv_P @ H.T + R
        K = self._ctrv_P @ H.T @ np.linalg.inv(S)
        innov = z - H @ self._ctrv_x
        innov[2] = (innov[2] + math.pi) % (2 * math.pi) - math.pi  # wrap yaw innovation
        self._ctrv_x += K @ innov
        self._ctrv_P = (np.eye(5) - K @ H) @ self._ctrv_P
        return float(self._ctrv_x[3])

    def plot_estimates(self):
        """Save a PNG of recorded v and dist histories (called after node stops)."""
        if not self._rec_t:
            return
        t0 = self._rec_t0
        t = np.array(self._rec_t)
        v_win    = np.array(self._rec_v_win)
        v_ctrv   = np.array(self._rec_v_ctrv)
        dist_pf    = np.array(self._rec_dist)

        t_odom = np.array(self._rec_t_odom) - t0 if self._rec_t_odom else np.array([])
        v_odom = np.array(self._rec_v_odom)
        dist_odom = np.array(self._rec_dist_odom)

        t_imu = np.array(self._rec_t_imu) - t0 if self._rec_t_imu else np.array([])
        v_imu = np.array(self._rec_v_imu)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

        if np.any(~np.isnan(v_win)):
            ax1.plot(t, v_win,    label='v_window (pf)',  color='steelblue')
        if np.any(~np.isnan(v_ctrv)):
            ax1.plot(t, v_ctrv,   label='v_ctrv (ekf)',    color='teal')
        if len(t_odom):
            ax1.plot(t_odom, v_odom, label='v_odom', color='crimson', alpha=0.7)
        if len(t_imu):
            ax1.plot(t_imu, v_imu, label='v_imu (integrated)', color='mediumpurple', alpha=0.7)
        ax1.set_ylabel('Speed (m/s)')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(t, dist_pf, label='dist_pf', color='forestgreen')
        if len(t_odom):
            ax2.plot(t_odom, dist_odom, label='dist_odom', color='crimson', alpha=0.7)
            final_diff = dist_pf[-1] - float(np.interp(t[-1], t_odom, dist_odom))
            ax2.text(0.99, 0.05, f'final diff: {final_diff:+.3f} m',
                     transform=ax2.transAxes, ha='right', va='bottom',
                     fontsize=9, color='darkorchid')
        ax2.set_ylabel('Distance from start (m)')
        ax2.legend()
        ax2.grid(True)

        if len(t_imu):
            ax3.plot(t_imu, np.array(self._rec_acc_x), label='acc_x', color='steelblue')
            ax3.plot(t_imu, np.array(self._rec_acc_y), label='acc_y', color='darkorange')
            ax3.plot(t_imu, np.array(self._rec_acc_z), label='acc_z', color='forestgreen')
        ax3.set_ylabel('Acceleration (m/s²)')
        ax3.set_xlabel('Time (s)')
        ax3.legend()
        ax3.grid(True)

        fig.suptitle('Estimated velocity, distance, and IMU acceleration')
        fig.tight_layout()
        path = 'odom_tuner_estimates.png'
        fig.savefig(path)
        plt.close(fig)
        print(f'[odom_tuner] estimates plot saved to {path}')


def main(args=None):
    rclpy.init(args=args)
    node = OdomTuner()
    try:
        rclpy.spin(node)
    finally:
        if node._record_estimates:
            node.plot_estimates()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
