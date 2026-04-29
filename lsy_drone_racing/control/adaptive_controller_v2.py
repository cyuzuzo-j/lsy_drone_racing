"""Adaptive gate-tracking controller for drone racing.

This controller plans a smooth trajectory through the race gates using a relaxation-based
obstacle avoidance scheme.  At each replan it builds a polyline of approach / center / exit
waypoints for every remaining gate, densifies it, and iteratively pushes points away from
cylindrical post obstacles and gate frames (excluding gates the path is meant to pass
through).  The result is fit with a Cubic spline parameterized by arclength / target speed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray


# Geometry constants (meters).
GATE_INNER_HALF = 0.20      # half-width of the open square inside the gate frame
GATE_FRAME_HALF = 0.36      # half-width of the outer frame (bar centre is at 0.28, half-extent 0.08)
GATE_PLATE_HALF = 0.3      # along the gate axis: thickness of the plane to treat as "in frame"
GATE_OPENING_MARGIN = 0.3  # corridor half-width considered safe for passage
GATE_PUSH_OUT = 1.0      # where to push points that intrude on a bar
POST_RADIUS_CLEARANCE = 0.22  # 2D clearance from cylindrical post obstacles (extra buffer for tracking error)
POST_TOP_Z = 1.90           # posts extend roughly from the ground to this height
APPROACH_DIST = 0.4        # offset of the approach waypoint in front of a gate
EXIT_DIST = 0.4         # offset of the exit waypoint behind a gate
PATH_SAMPLE_STEP = 0.1     # densification step for the polyline (m)
RELAX_ITERS = 20            # avoidance / smoothing passes
SMOOTH_W_SELF = 0.8        # smoothing weights — higher self => weaker smoothing
SMOOTH_W_NEIGHBOR = 0.2
SMOOTH_W_REF = 0.6         # attraction weight to the previous path to enforce consistency
TARGET_SPEED = 0.5  # m/s, used to time-parameterize the path
GATE_REPLAN_DIST = 0.7  # replan when first entering this radius around each gate (m)
LOG_DIR = Path(os.environ.get("LSY_PATH_LOG_DIR", "/tmp/lsy_drone_paths"))


class AdaptiveController(Controller):
    """Gate-aware controller that dynamically re-plans through observed gates."""

    directions_set = False
    directions = [0, 0, 0, 0, 0]

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        self._freq = config.env.freq
        self._n_gates = len(config.env.track.gates)
        self._tick = 0
        self._finished = False
        self._spline = None
        self._t_total = 0.0
        self._t_offset = 0.0
        self._last_target = -2
        self._last_obs_visited = None
        self._last_replan_path = None
        self._last_replan_tick = -10_000
        self._episode_idx = 0
        self._gate_proximity_triggered = np.zeros(self._n_gates, dtype=bool)
        self._build_trajectory(obs)

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _gate_forward(quat: NDArray) -> NDArray:
        v = R.from_quat(quat).apply([1.0, 0.0, 0.0])
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])

    @staticmethod
    def _safe_normalize(v: NDArray) -> NDArray:
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else np.zeros_like(v)

    # -------------------------------------------------------------- planning
    def _build_trajectory(self, obs: dict[str, NDArray[np.floating]]) -> None:
        target = int(obs["target_gate"])
        if target == -1 or target >= self._n_gates:
            self._finished = True
            return

        drone_pos = np.asarray(obs["pos"], dtype=np.float64).flatten()[:3]
        drone_vel = np.asarray(obs.get("vel", [0.0, 0.0, 0.0]), dtype=np.float64).flatten()[:3]
        gates_pos = np.asarray(obs["gates_pos"], dtype=np.float64)
        gates_quat = np.asarray(obs["gates_quat"], dtype=np.float64)
        obstacles_pos = np.asarray(obs["obstacles_pos"], dtype=np.float64)

        # 1. Build a clean waypoint list with consistent flight direction.
        wps: list[np.ndarray] = [drone_pos.copy()]
        # Optional velocity-aligned anchor so the spline doesn't whip back when replanning.
        if np.linalg.norm(drone_vel) > 0.4:
            wps.append(drone_pos + self._safe_normalize(drone_vel) * 0.15)

        prev_dir = self._safe_normalize(drone_vel) if np.linalg.norm(drone_vel) > 0.4 else None
        cur_pt = wps[-1].copy()
        for i in range(target, self._n_gates):
            g = gates_pos[i, :3]
            axis = self._gate_forward(gates_quat[i])

            # Test both entrance directions to minimize cost to the next gate
            g_next = gates_pos[i, :3]
            if not self.directions_set:
                cost_plus = np.linalg.norm((g - axis * APPROACH_DIST) - cur_pt) + np.linalg.norm(g_next - g)
                cost_minus = np.linalg.norm((g - (-axis) * APPROACH_DIST) - cur_pt) + np.linalg.norm(g_next - g)
                if cost_minus < cost_plus:
                    axis = -axis
                self.directions[i] = axis
            else:
                axis = self.directions[i]

            # Shrink approach if we are already inside it.
            d = np.linalg.norm(g - cur_pt)
            if i == 0:
                app_d = min(0.4, max(0, d * 0.5))
            elif i == 1:
                app_d = min(1.5, max(0.5, d * 0.5))
            elif i == 2:
                app_d = min(0.5, max(0, d * 0.5))
            elif i == 3:
                app_d = min(0.2, max(0, d * 0.5))
            else:
                app_d = 0.4

            wps.append(g - axis * app_d)
            wps.append(g.copy())
            if i == 0:
                wps.append(g + axis * 0.6)
            elif i == 1:
                wps.append(g + axis * 0.6)
            elif i == 2:
                wps.append(g + axis * 0.1)
                wps.append(g - axis * 0.3)
            elif i == 3:
                wps.append(g + axis * 5)
            elif i == 4:
                wps.append(g + axis * EXIT_DIST)
            prev_dir = axis
            cur_pt = wps[-1]

        # 2. Densify into a polyline.
        path: list[np.ndarray] = []
        for p1, p2 in zip(wps[:-1], wps[1:]):
            seg_len = np.linalg.norm(p2 - p1)
            n = max(2, int(np.ceil(seg_len / PATH_SAMPLE_STEP)))
            path.extend(np.linspace(p1, p2, n)[:-1])
        path.append(wps[-1])
        path = np.asarray(path, dtype=np.float64)

        # Mark all indices between the approach and exit waypoints of a gate.
        # These will be strictly frozen during relaxation to enforce a dead-center straight line.
        wp_positions = [0]
        for p1, p2 in zip(wps[:-1], wps[1:]):
            seg_len = np.linalg.norm(p2 - p1)
            n = max(2, int(np.ceil(seg_len / PATH_SAMPLE_STEP)))
            wp_positions.append(wp_positions[-1] + (n - 1))

        anchor_count = len(wps) - 3 * (self._n_gates - target)
        gate_locked_idxs: list[int] = []
        for k in range(self._n_gates - target):
            app_wp = anchor_count + 3 * k
            ex_wp = anchor_count + 3 * k + 2
            start_idx = wp_positions[app_wp]
            end_idx = wp_positions[ex_wp]
            gate_locked_idxs.extend(range(start_idx, end_idx + 1))
        gate_locked_set = set(gate_locked_idxs)

        # 3. Iterative relaxation: smooth first, then push, so the final state stays clear.
        future_gates = list(range(target, self._n_gates))
        gate_rots = [R.from_quat(gates_quat[gi]) for gi in range(self._n_gates)]

        # Find closest points on the previously planned path to act as strong attractors
        has_ref = self._last_replan_path is not None
        if has_ref:
            dists = cdist(path, self._last_replan_path)
            closest_idx = np.argmin(dists, axis=1)
            ref_points = self._last_replan_path[closest_idx]
            # Ignore attraction if the closest point is > 1.5m away (prevents the
            # path from collapsing backward when planning for newly revealed gates)
            valid_ref = np.min(dists, axis=1) < 1.5

        def push_point(j: int) -> None:
            pt = path[j]
            # 3a. Cylindrical posts.
            if pt[2] < POST_TOP_Z:
                for op in obstacles_pos:
                    delta = pt[:2] - op[:2]
                    d = float(np.linalg.norm(delta))
                    if 1e-6 < d < POST_RADIUS_CLEARANCE:
                        pt[:2] += (delta / d) * (POST_RADIUS_CLEARANCE - d)
                    elif d <= 1e-6:
                        pt[:2] += np.array([POST_RADIUS_CLEARANCE, 0.0])
            # 3b. Gate frames.  Skip corridor of any gate we still plan to pass through.
            for gi in range(self._n_gates):
                gp = gates_pos[gi, :3]
                local = gate_rots[gi].inv().apply(pt - gp)
                in_opening = (
                    abs(local[1]) < GATE_OPENING_MARGIN
                    and abs(local[2]) < GATE_OPENING_MARGIN
                )
                if gi in future_gates and in_opening:
                    continue
                if (
                    abs(local[0]) < GATE_PLATE_HALF
                    and max(abs(local[1]), abs(local[2])) < GATE_FRAME_HALF
                    and not in_opening
                ):
                    if abs(local[1]) > abs(local[2]):
                        local[1] = np.sign(local[1] or 1.0) * GATE_PUSH_OUT
                    else:
                        local[2] = np.sign(local[2] or 1.0) * GATE_PUSH_OUT
                    path[j] = gp + gate_rots[gi].apply(local)
                    pt = path[j]
                # Stand under the gate (thin column below the centre).
                if (
                    local[2] < -0.30
                    and abs(local[0]) < 0.08
                    and abs(local[1]) < 0.08
                ):
                    local[1] = np.sign(local[1] or 1.0) * 0.20
                    path[j] = gp + gate_rots[gi].apply(local)
                    pt = path[j]

        # Phase 1: push-only iterations — let the path bow around obstacles freely.
        for _ in range(RELAX_ITERS):
            for j in range(1, len(path) - 1):
                if j in gate_locked_set:
                    continue
                push_point(j)

        # Phase 2: alternate light smoothing with pushing so corners don't kink.
        for _ in range(RELAX_ITERS):
            new_inner = (
                SMOOTH_W_NEIGHBOR * path[:-2]
                + (1.0 - 2.0 * SMOOTH_W_NEIGHBOR) * path[1:-1]
                + SMOOTH_W_NEIGHBOR * path[2:]
            )
            for j in range(1, len(path) - 1):
                if j in gate_locked_set:
                    continue
                # Enforce closeness to original path
                if has_ref and valid_ref[j]:
                    path[j] = (1.0 - SMOOTH_W_REF) * new_inner[j - 1] + SMOOTH_W_REF * ref_points[j]
                else:
                    path[j] = new_inner[j - 1]

            for j in range(1, len(path) - 1):
                if j in gate_locked_set:
                    continue
                # Prevent sharp U-turns dynamically by flattening tight angles
                v1 = path[j] - path[j - 1]
                v2 = path[j + 1] - path[j]
                n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
                if n1 > 1e-4 and n2 > 1e-4 and np.dot(v1 / n1, v2 / n2) < 0.0:
                    path[j] = 0.5 * path[j] + 0.25 * path[j - 1] + 0.25 * path[j + 1]

            for j in range(1, len(path) - 1):
                if j in gate_locked_set:
                    continue
                push_point(j)

        # Phase 3: final push-only to guarantee clearance after the last smoothing.
        for _ in range(RELAX_ITERS):
            for j in range(1, len(path) - 1):
                if j in gate_locked_set:
                    continue
                # Unbend sharp U-turns before pushing to ensure clearance takes precedence
                v1 = path[j] - path[j - 1]
                v2 = path[j + 1] - path[j]
                n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
                if n1 > 1e-4 and n2 > 1e-4 and np.dot(v1 / n1, v2 / n2) < 0.0:
                    path[j] = 0.5 * path[j] + 0.25 * path[j - 1] + 0.25 * path[j + 1]

                push_point(j)

        # 4. Drop near-duplicate points so stays well-conditioned.
        diffs = np.linalg.norm(np.diff(path, axis=0), axis=1)
        keep = np.concatenate(([True], diffs > 1e-3))
        path = path[keep]
        if len(path) < 2:
            self._finished = True
            return

        # 5. Time parameterize using arclength / target speed.
        dists = np.linalg.norm(np.diff(path, axis=0), axis=1)
        cum = np.insert(np.cumsum(dists), 0, 0.0)
        times = cum / TARGET_SPEED
        valid = np.concatenate(([True], np.diff(times) > 1e-3))
        times = times[valid]
        path = path[valid]
        if len(path) < 2:
            self._finished = True
            return

        self._spline = CubicSpline(times, path, bc_type='natural')
        self._t_total = float(times[-1])
        self._t_offset = self._tick / self._freq
        self._last_target = target
        self._last_replan_path = path.copy()
        self._last_replan_tick = self._tick
        self.directions_set = True

    # ------------------------------------------------------------- main loop
    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        target = int(obs["target_gate"])
        obs_visited = np.asarray(obs["obstacles_visited"], dtype=bool)

        # Trigger a replan when state changes meaningfully.
        replan = False
        if self._tick == 15:
            replan = True
        # Replan the first time we enter the 0.7 m approach radius of each gate.
        drone_pos = np.asarray(obs["pos"], dtype=np.float64).flatten()[:3]
        gates_pos = np.asarray(obs["gates_pos"], dtype=np.float64)
        for gi in range(self._n_gates):
            if not self._gate_proximity_triggered[gi]:
                if float(np.linalg.norm(gates_pos[gi, :3] - drone_pos)) < GATE_REPLAN_DIST:
                    self._gate_proximity_triggered[gi] = True
                    replan = True
        if self._last_obs_visited is not None and np.any(obs_visited & ~self._last_obs_visited):
            replan = True
        if replan:
            self._build_trajectory(obs)
        self._last_obs_visited = obs_visited

        action = np.zeros(13, dtype=np.float32)
        if self._finished or self._spline is None:
            action[:3] = np.asarray(obs["pos"]).flatten()[:3]
            return action

        t_s = float(np.clip(self._tick / self._freq - self._t_offset, 0.0, self._t_total))
        if t_s >= self._t_total:
            self._finished = True

        des_pos = self._spline(t_s)
        des_pos[2] -= 0.1  # fly slightly below the path for better clearance and to prevent overshooting
        t_look = float(np.clip(t_s + 0.1, 0.0, self._t_total))
        delta = self._spline(t_look) - des_pos
        des_yaw = float(np.arctan2(delta[1], delta[0])) if np.linalg.norm(delta[:2]) > 1e-2 else 0.0

        action[0:3] = des_pos
        action[9] = des_yaw
        return action

    # --------------------------------------------------------------- callbacks
    def step_callback(self, action, obs, reward, terminated, truncated, info) -> bool:
        self._tick += 1
        if terminated or truncated:
            self._dump_path(obs, info, terminated=terminated, truncated=truncated)
        return self._finished

    def episode_callback(self):
        self.episode_reset()

    def episode_reset(self):
        self._tick = 0
        self._finished = False
        self._spline = None
        self._t_total = 0.0
        self._t_offset = 0.0
        self._last_target = -2
        self._last_obs_visited = None
        self._last_replan_path = None
        self._episode_idx += 1
        self._gate_proximity_triggered = np.zeros(self._n_gates, dtype=bool)

    def render_callback(self, sim: Sim):
        if self._spline is None:
            return
        try:
            from crazyflow.sim.visualize import draw_line, draw_points

            t_s = float(np.clip(self._tick / self._freq - self._t_offset, 0.0, self._t_total))
            draw_points(sim, self._spline(t_s).reshape(1, -1), rgba=(1, 0, 0, 1), size=0.03)
            draw_line(sim, self._spline(np.linspace(0, self._t_total, 100)), rgba=(0, 1, 0, 1))
        except ImportError:
            pass

    # ------------------------------------------------------------ diagnostics
    def _dump_path(self, obs: dict, info: dict | None, *, terminated: bool, truncated: bool) -> None:
        """Write the most recent planned path + run summary when the episode ends."""
        if self._last_replan_path is None:
            return
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            payload = {
                "episode": self._episode_idx,
                "tick_at_end": int(self._tick),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "target_gate_at_end": int(obs.get("target_gate", -1)),
                "drone_pos_at_end": np.asarray(obs["pos"]).flatten()[:3].tolist(),
                "last_replan_tick": int(self._last_replan_tick),
                "path_xyz": self._last_replan_path.tolist(),
                "collision": bool(info.get("collision")) if info else False,
            }
            fpath = LOG_DIR / f"path_ep{self._episode_idx:03d}.json"
            with fpath.open("w") as f:
                json.dump(payload, f, indent=2)
            print(f"[adaptive_v2] Wrote planned path to {fpath}")
        except Exception as exc:  # noqa: BLE001
            print(f"[adaptive_v2] Failed to dump path: {exc}")