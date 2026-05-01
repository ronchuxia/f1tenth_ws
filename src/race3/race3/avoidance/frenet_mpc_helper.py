import math

import numpy as np

try:
    import cvxpy
except ImportError:  # pragma: no cover
    cvxpy = None

from f1tenth_utils.vis_utils import visualize_path, visualize_point, visualize_trajectory


class FrenetMPCController:
    def __init__(
        self,
        wheelbase=0.3302,
        max_speed=5.0,
        min_speed=0.0,
        max_steering_angle=0.4189,
        max_accel=3.0,
        max_dsteer=math.pi,
        horizon=8,
        dt=0.1,
        q_x=18.5, q_y=18.5, q_v=3.5, q_yaw=0.1,
        r_a=0.01, r_delta=100.0,
        rd_a=0.01, rd_delta=100.0,
        sim=False,
    ):
        if cvxpy is None:
            raise ImportError("FrenetMPCController requires cvxpy. Install with: pip install cvxpy")

        self.L = wheelbase
        self.v_max = max_speed
        self.v_min = min_speed
        self.d_max = max_steering_angle
        self.a_max = max_accel
        self.ddelta_max = max_dsteer
        self.T = horizon
        self.dt = dt
        self.base_link_frame = 'ego_racecar/base_link' if sim else 'base_link'

        self.NX = 4
        self.NU = 2
        self.Q = np.diag([q_x, q_y, q_v, q_yaw])
        self.Qf = self.Q.copy()
        self.R = np.diag([r_a, r_delta])
        self.Rd = np.diag([rd_a, rd_delta])

        self._build_problem()
        self.u_prev = None
        self.x_pred_prev = None

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
                obj += cvxpy.quad_form(self.uk[:, t] - self.uk[:, t - 1], cvxpy.psd_wrap(self.Rd))
            cons += [
                self.xk[:, t + 1]
                == self.A_params[t] @ self.xk[:, t] + self.B_params[t] @ self.uk[:, t] + self.C_params[t]
            ]
        obj += cvxpy.quad_form(self.xk[:, T] - self.xref[:, T], cvxpy.psd_wrap(self.Qf))

        cons += [
            self.xk[2, :] <= self.v_max,
            self.xk[2, :] >= self.v_min,
            cvxpy.abs(self.uk[0, :]) <= self.a_max,
            cvxpy.abs(self.uk[1, :]) <= self.d_max,
            cvxpy.abs(cvxpy.diff(self.uk[1, :])) <= self.ddelta_max * self.dt,
        ]

        self.prob = cvxpy.Problem(cvxpy.Minimize(obj), cons)

    def _get_linear_model(self, v_bar, yaw_bar, delta_bar):
        dt, L, NX, NU = self.dt, self.L, self.NX, self.NU
        A = np.eye(NX)
        A[0, 2] = dt * math.cos(yaw_bar)
        A[0, 3] = -dt * v_bar * math.sin(yaw_bar)
        A[1, 2] = dt * math.sin(yaw_bar)
        A[1, 3] = dt * v_bar * math.cos(yaw_bar)
        A[3, 2] = dt * math.tan(delta_bar) / L

        B = np.zeros((NX, NU))
        B[2, 0] = dt
        B[3, 1] = dt * v_bar / (L * math.cos(delta_bar) ** 2)

        C = np.zeros(NX)
        C[0] = dt * v_bar * math.sin(yaw_bar) * yaw_bar
        C[1] = -dt * v_bar * math.cos(yaw_bar) * yaw_bar
        C[3] = -dt * v_bar * delta_bar / (L * math.cos(delta_bar) ** 2)
        return A, B, C

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

    @staticmethod
    def _prepare_path(path_xy, path_speed=None):
        path_xy = np.asarray(path_xy, dtype=np.float64)
        if path_xy.shape[0] < 2:
            raise ValueError("FrenetMPCController requires at least two path points")

        seg = np.diff(path_xy, axis=0)
        ds = np.hypot(seg[:, 0], seg[:, 1])
        valid = ds > 1e-6
        if not np.any(valid):
            raise ValueError("FrenetMPCController path has no usable segment length")

        keep = np.concatenate([[True], valid])
        path_xy = path_xy[keep]
        seg = np.diff(path_xy, axis=0)
        ds = np.hypot(seg[:, 0], seg[:, 1])

        yaw = np.empty(path_xy.shape[0], dtype=np.float64)
        yaw[:-1] = np.arctan2(seg[:, 1], seg[:, 0])
        yaw[-1] = yaw[-2]
        yaw = np.unwrap(yaw)
        arc = np.concatenate([[0.0], np.cumsum(ds)])

        if path_speed is None:
            speed = None
        else:
            speed = np.asarray(path_speed, dtype=np.float64)[keep]
        return path_xy, yaw, arc, speed

    def _build_ref(self, pose_x, pose_y, pose_yaw, v_current, path_xy, path_speed=None):
        path_xy, path_yaw, arc, speed = self._prepare_path(path_xy, path_speed)
        dx = path_xy[:, 0] - pose_x
        dy = path_xy[:, 1] - pose_y
        i0 = int(np.argmin(dx * dx + dy * dy))

        local_xy = path_xy[i0:]
        local_yaw = np.unwrap(path_yaw[i0:])
        local_arc = arc[i0:] - arc[i0]
        if local_xy.shape[0] == 1:
            local_xy = np.vstack([local_xy, local_xy])
            local_yaw = np.array([local_yaw[0], local_yaw[0]])
            local_arc = np.array([0.0, 1e-3])

        if speed is None:
            local_speed = np.full(local_arc.shape[0], self.v_max, dtype=np.float64)
        else:
            local_speed = np.clip(speed[i0:], self.v_min, self.v_max)

        total = max(float(local_arc[-1]), 1e-3)
        travel_per_step = max(0.5, abs(v_current)) * self.dt
        ref = np.empty((self.NX, self.T + 1))
        for t in range(self.T + 1):
            s = min(t * travel_per_step, total)
            ref[0, t] = np.interp(s, local_arc, local_xy[:, 0])
            ref[1, t] = np.interp(s, local_arc, local_xy[:, 1])
            ref[2, t] = np.interp(s, local_arc, local_speed)
            ref[3, t] = np.interp(s, local_arc, local_yaw)

        shift = 0.0
        while ref[3, 0] + shift - pose_yaw > math.pi:
            shift -= 2.0 * math.pi
        while ref[3, 0] + shift - pose_yaw < -math.pi:
            shift += 2.0 * math.pi
        ref[3, :] += shift
        return ref

    def compute_drive(self, pose_x, pose_y, pose_yaw, v_current, delta_current,
                      path_xy, path_speed=None, stamp=None, vis=False):
        if abs(v_current) < 0.1:
            target = np.asarray(path_xy[0], dtype=np.float64)
            return {
                'steering': 0.0,
                'speed': min(1.0, self.v_max),
                'markers': [],
                'target': target,
            }

        x0 = np.array([pose_x, pose_y, v_current, pose_yaw])
        ref = self._build_ref(pose_x, pose_y, pose_yaw, v_current, path_xy, path_speed)

        if self.u_prev is not None:
            u_init = np.concatenate([self.u_prev[:, 1:], self.u_prev[:, -1:].copy()], axis=1)
            u_init[1, 0] = delta_current
        else:
            u_init = np.zeros((self.NU, self.T))
            u_init[1, :] = delta_current

        pred = self._predict(x0, u_init)
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
            delta_cmd = float(np.clip(delta_current, -self.d_max, self.d_max))
            v_cmd = float(np.clip(v_current, self.v_min, self.v_max))
            x_sol = pred

        target = np.array([ref[0, 1], ref[1, 1]])
        markers = []
        if vis:
            markers.append(visualize_trajectory(
                delta_cmd, stamp, wheelbase=self.L, frame_id=self.base_link_frame,
                ns='frenet_mpc_cmd', id=270, color=(1.0, 0.0, 1.0, 1.0),
            ))
            markers.append(visualize_point(target, stamp, frame_id='/map',
                                           ns='frenet_mpc_target', id=271,
                                           color=(1.0, 0.0, 0.0, 1.0)))
            ref_pts = np.column_stack([ref[0], ref[1]])
            markers.append(visualize_path(ref_pts, stamp, frame_id='/map',
                                          ns='frenet_mpc_ref', id=272,
                                          color=(0.7, 0.0, 1.0, 0.8)))
            if x_sol is not None:
                pred_pts = np.column_stack([x_sol[0], x_sol[1]])
                markers.append(visualize_path(pred_pts, stamp, frame_id='/map',
                                              ns='frenet_mpc_pred', id=273,
                                              color=(0.0, 1.0, 1.0, 0.9)))

        return {
            'steering': delta_cmd,
            'speed': v_cmd,
            'markers': markers,
            'target': target,
        }
