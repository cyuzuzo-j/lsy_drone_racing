from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray


class AdaptiveController(Controller):
    """Simplified gate-aware controller with dynamic re-planning and obstacle avoidance."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        self._freq = config.env.freq
        self._n_gates = len(config.env.track.gates)
        self.episode_reset()
        self._build_trajectory(obs)

    def episode_reset(self):
        self._tick = 0
        self._finished = False
        self._spline = None
        self._t_total = 0.0
        self._t_offset = 0.0
        self._last_target = -2
        self._last_obs = None

    def _build_trajectory(self, obs: dict[str, NDArray[np.floating]]):
        target = int(obs["target_gate"])
        if target == -1 or target >= self._n_gates:
            self._finished = True
            return

        pos = np.array(obs["pos"], dtype=np.float64).flatten()[:3]
        gates_pos = np.array(obs["gates_pos"], dtype=np.float64)
        gates_quat = np.array(obs["gates_quat"], dtype=np.float64)
        obstacles = np.array(obs["obstacles_pos"], dtype=np.float64)

        # 1. Base Waypoints (Drone -> Gate Approaches -> Gate Exits)
        wps = [pos]
        if target == 0 and np.linalg.norm(pos[:2]) < 0.2:
            wps.append(pos + [0, 0, 0.8])  # Takeoff boost

        for i in range(target, self._n_gates):
            g_pos = gates_pos[i][:3]
            fwd = R.from_quat(gates_quat[i]).apply([1.0, 0.0, 0.0])
            
            # Align passage direction to path
            if np.dot(g_pos - wps[-1], fwd) < 0:
                fwd = -fwd
                
            wps.extend([g_pos - fwd * 0.6, g_pos, g_pos + fwd * 0.6])

        # 2. Dense Path (Sample linearly roughly every 5cm)
        path = []
        for p1, p2 in zip(wps[:-1], wps[1:]):
            steps = max(2, int(np.linalg.norm(p2 - p1) / 0.05))
            path.extend(np.linspace(p1, p2, steps)[:-1])
        path.append(wps[-1])
        path = np.array(path)

        # 3. Obstacle & Gate Avoidance (Relaxation)
        for _ in range(5):  # 5 passes to iteratively push and smooth
            for i in range(1, len(path) - 1):
                pt = path[i]

                # 3a. Avoid vertical markers
                if pt[2] < 1.55:
                    for obs_p in obstacles:
                        vec = pt[:2] - obs_p[:2]
                        dist = np.linalg.norm(vec)
                        if 0 < dist < 0.35:
                            pt[:2] += (vec / dist) * (0.35 - dist)

                # 3b. Avoid gate frames using local coordinates
                for g in range(self._n_gates):
                    g_pos = gates_pos[g][:3]
                    if abs(pt[2] - g_pos[2]) > 1.0:
                        continue
                    
                    rot = R.from_quat(gates_quat[g])
                    local_pt = rot.inv().apply(pt - g_pos)
                    
                    # Protect the safe corridor of current and upcoming gates
                    if g >= max(0, target - 1) and abs(local_pt[0]) < 0.3 and np.linalg.norm(local_pt[1:]) < 0.25:
                        continue 
                        
                    # Push out of gate frame bounding box
                    if abs(local_pt[0]) < 0.4 and 0.15 < abs(local_pt[1]) < 0.85:
                        local_pt[1] = np.sign(local_pt[1]) * 0.85
                        path[i] = g_pos + rot.apply(local_pt)

            # 3c. Path Smoothing (Moving Average)
            # This is critical to prevent impossible acceleration spikes after pushing points
            path[1:-1] = 0.5 * path[1:-1] + 0.25 * (path[:-2] + path[2:])

        # 4. Time Parameterization (Constant average speed of ~1.5 m/s)
        dists = np.linalg.norm(np.diff(path, axis=0), axis=1)
        cum_dist = np.insert(np.cumsum(dists), 0, 0.0)
        
        if cum_dist[-1] < 1e-3:
            return

        times = cum_dist / 1.5 
        
        # Keep only strictly increasing time steps for the spline
        valid = np.concatenate(([True], np.diff(times) > 1e-3))
        
        self._spline = PchipInterpolator(times[valid], path[valid])
        self._t_total = times[valid][-1]
        self._t_offset = self._tick / self._freq
        self._last_target = target

    def compute_control(self, obs: dict[str, NDArray[np.floating]], info: dict | None = None) -> NDArray[np.floating]:
        target = int(obs["target_gate"])
        obs_visited = np.array(obs["obstacles_visited"], dtype=bool)

        # Re-plan if state changes
        if target != self._last_target or (
            self._last_obs is not None and np.any(obs_visited & ~self._last_obs)
        ):
            self._build_trajectory(obs)
            
        self._last_obs = obs_visited

        action = np.zeros(13, dtype=np.float32)
        
        if self._finished or self._spline is None:
            action[:3] = np.array(obs["pos"]).flatten()[:3]
            return action

        t_s = np.clip(self._tick / self._freq - self._t_offset, 0.0, self._t_total)
        if t_s >= self._t_total:
            self._finished = True

        des_pos = self._spline(t_s)
        
        # Lookahead for Yaw
        lookahead_pos = self._spline(np.clip(t_s + 0.1, 0.0, self._t_total))
        delta = lookahead_pos - des_pos
        
        action[:3] = des_pos
        if np.linalg.norm(delta[:2]) > 1e-2:
            action[9] = float(np.arctan2(delta[1], delta[0]))

        return action

    def step_callback(self, action, obs, reward, terminated, truncated, info) -> bool:
        self._tick += 1
        return self._finished

    def episode_callback(self):
        self.episode_reset()

    def render_callback(self, sim: Sim):
        if self._spline is None:
            return
        try:
            from crazyflow.sim.visualize import draw_line, draw_points
            t_s = np.clip(self._tick / self._freq - self._t_offset, 0.0, self._t_total)
            draw_points(sim, self._spline(t_s).reshape(1, -1), rgba=(1, 0, 0, 1), size=0.03)
            draw_line(sim, self._spline(np.linspace(0, self._t_total, 100)), rgba=(0, 1, 0, 1))
        except ImportError:
            pass