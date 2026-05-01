"""
Linearized kinematic-bicycle MPC tracker (reference-style, cvxpy/OSQP).

Follows the Penn f1tenth_planning `kinematic_mpc.py` formulation:

  State  z = [x, y, v, yaw]           (map-frame pose + longitudinal speed)
  Input  u = [a, δ]                   (acceleration, steering angle)
  Model  ẋ = v cos(yaw)
         ẏ = v sin(yaw)
         v̇ = a
         yaẇ = (v / L) · tan(δ)

At each call:
  1. Build an (N+1)-step reference trajectory from the static waypoints,
     spacing samples by v·DT along the raceline arc length.
  2. Predict the state trajectory by rolling the nonlinear model with the
     previous solution (warm-start). Linearize the discrete-time model
     around that predicted trajectory and the previous steering input.
  3. Solve the finite-horizon QP (cvxpy + OSQP) with box constraints on
     v, a, δ and a steering-rate limit.
  4. Return the first-step (δ, v) for the drive command plus markers.

cvxpy is required (`pip install cvxpy`). The reference's sparse
parameter-stuffing trick is skipped here for readability — per-step
cvxpy Parameter objects cost ~20-50 ms per solve on Jetson, which is
acceptable for first tuning. OSQP-direct is a follow-on optimization
(see docs/TRACKER_NOTES.md).
"""

import math

import numpy as np

try:
    import cvxpy
except ImportError:  # pragma: no cover
    cvxpy = None

from f1tenth_utils.vis_utils import visualize_path, visualize_point, visualize_trajectory


class KinematicMPCTracker:
    def __init__(
        self,
        wheelbase=0.3302,
        max_speed=6.0,
        min_speed=0.0,
        max_steering_angle=0.4189,
        max_accel=3.0,
        max_dsteer=math.pi,
        horizon=8,
        dt=0.1,
        q_x=18.5, q_y=18.5, q_v=3.5, q_yaw=0.1,
        r_a=0.01, r_delta=100.0,
        rd_a=0.01, rd_delta=100.0,
        mu=0.7,
        waypoints=None,
        waypoint_data=None,
        waypoint_columns=None,
        sim=False,
    ):
        if cvxpy is None:
            raise ImportError(
                "KinematicMPCTracker requires cvxpy. Install with: pip install cvxpy"
            )
        if waypoints is None and waypoint_data is None:
            raise ValueError("KinematicMPCTracker needs waypoints or waypoint_data at construction")

        self.L = wheelbase
        self.v_max = max_speed
        self.v_min = min_speed
        self.d_max = max_steering_angle
        self.a_max = max_accel
        self.ddelta_max = max_dsteer
        self.T = horizon
        self.dt = dt
        self.mu = mu
        self.base_link_frame = 'ego_racecar/base_link' if sim else 'base_link'

        self.NX = 4
        self.NU = 2
        self.Q = np.diag([q_x, q_y, q_v, q_yaw])
        self.Qf = self.Q.copy()
        self.R = np.diag([r_a, r_delta])
        self.Rd = np.diag([rd_a, rd_delta])

        self._prepare_raceline(
            waypoints=np.asarray(waypoints, dtype=np.float64) if waypoints is not None else None,
            waypoint_data=np.asarray(waypoint_data, dtype=np.float64) if waypoint_data is not None else None,
            waypoint_columns=waypoint_columns,
        )
        self._build_problem()

        self.u_prev = None
        self.x_pred_prev = None

    # ---- raceline prep ----

    def _prepare_raceline(self, waypoints, waypoint_data=None, waypoint_columns=None):
        if waypoint_data is not None:
            W = np.asarray(waypoint_data[:, :2], dtype=np.float64)
        else:
            W = np.asarray(waypoints, dtype=np.float64)

        # Closed loop: segment from i -> i+1 (wrapping at the end)
        nxt = np.roll(W, -1, axis=0)
        dx = nxt[:, 0] - W[:, 0]
        dy = nxt[:, 1] - W[:, 1]
        ds_geom = np.hypot(dx, dy)
        yaw_geom = np.unwrap(np.arctan2(dy, dx))
        kappa_geom = self._compute_curvature(W)

        s_vals = self._get_waypoint_column(waypoint_data, waypoint_columns, 's_m')
        if s_vals is not None:
            ds = self._compute_ds_from_s(np.asarray(s_vals, dtype=np.float64))
        else:
            ds = ds_geom

        yaw_vals = self._get_waypoint_column(waypoint_data, waypoint_columns, 'psi_rad')
        cyaw = np.unwrap(np.asarray(yaw_vals, dtype=np.float64)) if yaw_vals is not None else yaw_geom

        kappa_vals = self._get_waypoint_column(waypoint_data, waypoint_columns, 'kappa_radpm')
        kappa = np.asarray(kappa_vals, dtype=np.float64) if kappa_vals is not None else kappa_geom

        sp_vals = self._get_waypoint_column(waypoint_data, waypoint_columns, 'vx_mps')
        if sp_vals is not None:
            sp = np.clip(np.asarray(sp_vals, dtype=np.float64), self.v_min, self.v_max)
        else:
            g = 9.81
            v_lat = np.sqrt(self.mu * g / (np.abs(kappa) + 1e-3))
            sp = np.clip(v_lat, max(0.5, self.v_min), self.v_max)

        self.cx = W[:, 0]
        self.cy = W[:, 1]
        self.cyaw = cyaw
        self.ds = ds
        self.sp = sp
        self.N = W.shape[0]

    @staticmethod
    def _compute_curvature(waypoints):
        n = waypoints.shape[0]
        kappa = np.zeros(n)
        for i in range(n):
            a = waypoints[(i - 1) % n]
            b = waypoints[i]
            c = waypoints[(i + 1) % n]
            cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
            denom = np.linalg.norm(b - a) * np.linalg.norm(c - b) * np.linalg.norm(c - a)
            kappa[i] = 2.0 * cross / denom if denom > 1e-6 else 0.0
        return kappa

    @staticmethod
    def _get_waypoint_column(waypoint_data, waypoint_columns, name):
        if waypoint_data is None or waypoint_columns is None or name not in waypoint_columns:
            return None
        idx = waypoint_columns.index(name)
        if idx >= waypoint_data.shape[1]:
            return None
        return waypoint_data[:, idx]

    @staticmethod
    def _compute_ds_from_s(s_vals):
        s_vals = np.asarray(s_vals, dtype=np.float64)
        total_length = float(np.max(s_vals))
        nxt = np.roll(s_vals, -1)
        ds = nxt - s_vals
        ds[-1] = total_length - s_vals[-1] + s_vals[0]
        valid = ds > 1e-6
        if not np.all(valid):
            fallback = np.median(ds[valid]) if np.any(valid) else 1.0
            ds[~valid] = fallback
        return ds

    # ---- QP assembly ----

    def _build_problem(self):
        T, NX, NU = self.T, self.NX, self.NU

        self.xk = cvxpy.Variable((NX, T + 1))
        self.uk = cvxpy.Variable((NU, T))
        self.x0k = cvxpy.Parameter(NX)
        self.xref = cvxpy.Parameter((NX, T + 1))
        self.A_params = [cvxpy.Parameter((NX, NX)) for _ in range(T)]
        self.B_params = [cvxpy.Parameter((NX, NU)) for _ in range(T)]
        self.C_params = [cvxpy.Parameter(NX) for _ in range(T)]

        obj = 0.0
        cons = [self.xk[:, 0] == self.x0k]
        for t in range(T):
            obj += cvxpy.quad_form(self.xk[:, t] - self.xref[:, t], cvxpy.psd_wrap(self.Q))
            obj += cvxpy.quad_form(self.uk[:, t], cvxpy.psd_wrap(self.R))
            if t > 0:
                obj += cvxpy.quad_form(self.uk[:, t] - self.uk[:, t - 1],
                                       cvxpy.psd_wrap(self.Rd))
            cons += [self.xk[:, t + 1] ==
                     self.A_params[t] @ self.xk[:, t]
                     + self.B_params[t] @ self.uk[:, t]
                     + self.C_params[t]]
        obj += cvxpy.quad_form(self.xk[:, T] - self.xref[:, T], cvxpy.psd_wrap(self.Qf))

        cons += [self.xk[2, :] <= self.v_max,
                 self.xk[2, :] >= self.v_min,
                 cvxpy.abs(self.uk[0, :]) <= self.a_max,
                 cvxpy.abs(self.uk[1, :]) <= self.d_max,
                 cvxpy.abs(cvxpy.diff(self.uk[1, :])) <= self.ddelta_max * self.dt]

        self.prob = cvxpy.Problem(cvxpy.Minimize(obj), cons)

    # ---- linearization ----

    def _get_linear_model(self, v_bar, yaw_bar, delta_bar):
        dt, L, NX, NU = self.dt, self.L, self.NX, self.NU
        A = np.eye(NX)
        A[0, 2] = dt * math.cos(yaw_bar)
        A[0, 3] = -dt * v_bar * math.sin(yaw_bar)
        A[1, 2] = dt * math.sin(yaw_bar)
        A[1, 3] = dt * v_bar * math.cos(yaw_bar)
        # yaw_next ← v   (∂ of (v/L)·tan(δ_bar) wrt v)
        A[3, 2] = dt * math.tan(delta_bar) / L

        B = np.zeros((NX, NU))
        B[2, 0] = dt
        # yaw_next ← δ   (∂ of (v_bar/L)·tan(δ) wrt δ around δ_bar)
        B[3, 1] = dt * v_bar / (L * math.cos(delta_bar) ** 2)

        C = np.zeros(NX)
        C[0] = dt * v_bar * math.sin(yaw_bar) * yaw_bar
        C[1] = -dt * v_bar * math.cos(yaw_bar) * yaw_bar
        C[3] = -dt * v_bar * delta_bar / (L * math.cos(delta_bar) ** 2)
        return A, B, C

    # ---- reference trajectory ----

    def _build_ref(self, pose_x, pose_y, pose_yaw, v_current):
        # nearest index on raceline
        dx = self.cx - pose_x
        dy = self.cy - pose_y
        i0 = int(np.argmin(dx * dx + dy * dy))

        # Forward-ordered waypoint arrays starting at i0, length N+1 so we
        # can interpolate up to one full lap. rel_arc[k] = arc length from
        # i0 to the k-th forward waypoint; rel_arc[0]=0.
        idxs = (i0 + np.arange(self.N + 1)) % self.N
        local_cx = self.cx[idxs]
        local_cy = self.cy[idxs]
        local_sp = self.sp[idxs]
        # Re-unwrap locally: reordering at i0 introduces a ~±2π seam where
        # idxs wraps from N-1 back to 0; np.unwrap rolls that back out.
        local_cyaw = np.unwrap(self.cyaw[idxs])
        rel_ds = np.concatenate([self.ds[i0:], self.ds[:i0]])
        rel_arc = np.concatenate([[0.0], np.cumsum(rel_ds)])
        total = float(rel_arc[-1])

        # Interpolate along arc length so ref[t] lands exactly at
        # s_t = t · v_current · dt, independent of waypoint density. The
        # old snap-to-next-waypoint overshot badly on sparse maps (ours has
        # ~0.44 m spacing; at v=0.7 m/s one MPC step is 0.07 m, so ref
        # lurched ~0.4 m per step and ref[T] landed ~2× beyond reach).
        travel_per_step = max(0.5, abs(v_current)) * self.dt
        ref = np.empty((self.NX, self.T + 1))
        for t in range(self.T + 1):
            s = min(t * travel_per_step, total - 1e-6)
            ref[0, t] = np.interp(s, rel_arc, local_cx)
            ref[1, t] = np.interp(s, rel_arc, local_cy)
            ref[2, t] = np.interp(s, rel_arc, local_sp)
            ref[3, t] = np.interp(s, rel_arc, local_cyaw)

        # Wrap ref yaw as a whole (preserves continuity from np.unwrap) into
        # the ±π window of pose_yaw
        shift = 0.0
        while ref[3, 0] + shift - pose_yaw > math.pi:
            shift -= 2.0 * math.pi
        while ref[3, 0] + shift - pose_yaw < -math.pi:
            shift += 2.0 * math.pi
        ref[3, :] += shift
        return ref

    # ---- nonlinear rollout for linearization points ----

    def _predict(self, x0, u):
        path = np.empty((self.NX, self.T + 1))
        path[:, 0] = x0
        x, y, v, yaw = x0
        for t in range(self.T):
            a = float(u[0, t])
            d = float(np.clip(u[1, t], -self.d_max, self.d_max))
            x += v * math.cos(yaw) * self.dt
            y += v * math.sin(yaw) * self.dt
            yaw += (v / self.L) * math.tan(d) * self.dt
            v = float(np.clip(v + a * self.dt, self.v_min, self.v_max))
            path[:, t + 1] = [x, y, v, yaw]
        return path

    # ---- main step ----

    def compute_drive(self, pose_x, pose_y, pose_yaw, v_current, delta_current,
                      waypoints, stamp=None, vis=False):   # noqa: ARG002 — waypoints part of tracker interface; MPC uses pre-computed raceline
        # Low-speed bootstrap: MPC is degenerate at v≈0, push a gentle start.
        if abs(v_current) < 0.1:
            target_xy = np.array([self.cx[0], self.cy[0]])
            return {
                'steering': 0.0,
                'speed': 1.0,
                'markers': [],
                'target': target_xy,
                'target_x': 1.0,
                'target_y': 0.0,
                'actual_l': 1.0,
            }

        x0 = np.array([pose_x, pose_y, v_current, pose_yaw])
        ref = self._build_ref(pose_x, pose_y, pose_yaw, v_current)

        # warm-start u: shift previous solution one step, pad with last
        if self.u_prev is not None:
            u_init = np.concatenate([self.u_prev[:, 1:],
                                     self.u_prev[:, -1:].copy()], axis=1)
            u_init[1, 0] = delta_current  # seed steering from current state
        else:
            u_init = np.zeros((self.NU, self.T))
            u_init[1, :] = delta_current

        pred = self._predict(x0, u_init)

        # Linearize around predicted trajectory
        for t in range(self.T):
            v_bar = pred[2, t]
            yaw_bar = pred[3, t]
            delta_bar = float(np.clip(u_init[1, t], -self.d_max, self.d_max))
            A, B, C = self._get_linear_model(v_bar, yaw_bar, delta_bar)
            self.A_params[t].value = A
            self.B_params[t].value = B
            self.C_params[t].value = C

        self.xref.value = ref
        self.x0k.value = x0

        try:
            self.prob.solve(solver=cvxpy.OSQP, warm_start=True, verbose=False)
            ok = self.prob.status in (cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE)
        except Exception:
            ok = False

        if ok:
            u_sol = np.asarray(self.uk.value)
            x_sol = np.asarray(self.xk.value)
            self.u_prev = u_sol
            self.x_pred_prev = x_sol
            delta_cmd = float(np.clip(u_sol[1, 0], -self.d_max, self.d_max))
            v_cmd = float(np.clip(x_sol[2, 1], self.v_min, self.v_max))
        else:
            # fall back to reference yaw following via small steering, coast speed
            delta_cmd = float(np.clip(delta_current, -self.d_max, self.d_max))
            v_cmd = float(np.clip(v_current, self.v_min, self.v_max))
            x_sol = pred

        # First-step reference point serves as the "target" for corridor viz
        target = np.array([ref[0, 1], ref[1, 1]])
        dx = target[0] - pose_x
        dy = target[1] - pose_y
        target_x = dx * math.cos(pose_yaw) + dy * math.sin(pose_yaw)
        target_y = -dx * math.sin(pose_yaw) + dy * math.cos(pose_yaw)
        actual_l = math.hypot(dx, dy)

        markers = []
        if vis:
            markers.append(visualize_trajectory(delta_cmd, stamp, wheelbase=self.L,
                                                frame_id=self.base_link_frame,
                                                color=(1.0, 0.0, 1.0, 1.0)))
            markers.append(visualize_point(target, stamp, frame_id='/map'))
            ref_pts = np.column_stack([ref[0], ref[1]])
            markers.append(visualize_path(ref_pts, stamp, frame_id='/map',
                                          ns='mpc_ref', id=20,
                                          color=(0.7, 0.0, 1.0, 0.8)))
            if x_sol is not None:
                pred_pts = np.column_stack([x_sol[0], x_sol[1]])
                markers.append(visualize_path(pred_pts, stamp, frame_id='/map',
                                              ns='mpc_pred', id=21,
                                              color=(0.0, 1.0, 1.0, 0.9)))

        return {
            'steering': delta_cmd,
            'speed': v_cmd,
            'markers': markers,
            'target': target,
            'target_x': float(target_x),
            'target_y': float(target_y),
            'actual_l': float(actual_l),
        }
