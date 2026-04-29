"""Adaptive gate-tracking controller for drone racing.

This controller dynamically plans trajectories through the race gates using observation data. 
It features a robust obstacle avoidance system with longitudinal gate protection to ensure 
safe passage through gate frames while avoiding markers and previously passed gates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray

def min_distance_drone_gate(drone_pos, gate_pos, gate_quat):
    """
    Calculates the minimum distance from the drone to the physical volume of a gate.
    
    Args:
        drone_pos: (3,) array of drone position.
        gate_pos: (3,) array of gate center position.
        gate_quat: (4,) array of gate orientation (x, y, z, w).
        
    Returns:
        float: Minimum distance in meters to the nearest collision box of the gate.
    """
    # Transform drone_pos into the gate's local coordinate frame
    rot = R.from_quat(gate_quat.flatten()[:4])
    p_local = rot.apply(drone_pos.flatten()[:3] - gate_pos.flatten()[:3], inverse=True)
    
    # Define collision boxes for the gate frame and stand (center, half-extents)
    collision_boxes = [
        (np.array([0, 0, 0.28]), np.array([0.01, 0.36, 0.08])),  # Top bar
        (np.array([0, 0, -0.28]), np.array([0.01, 0.36, 0.08])), # Bottom bar
        (np.array([0, -0.28, 0]), np.array([0.01, 0.08, 0.36])), # Left bar
        (np.array([0, 0.28, 0]), np.array([0.01, 0.08, 0.36])),  # Right bar
        (np.array([0, 0, -0.86]), np.array([0.05, 0.05, 0.5]))   # Stand/Post
    ]
    
    distances = []
    for center, half_extents in collision_boxes:
        # Distance from a point to an axis-aligned box (AABB) in the local frame
        delta = np.abs(p_local - center) - half_extents
        v = np.maximum(delta, 0)
        distances.append(np.linalg.norm(v))
            
    return min(distances)

class AdaptiveController(Controller):
    """Gate-aware controller that dynamically re-plans trajectories through observed gates."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        self._freq = config.env.freq
        self._tick = 0
        self._finished = False

        self._n_gates = len(config.env.track.gates)
        self._total_budget = 30.0
        
        self._last_target_gate = -2
        self._last_obstacles_visited = None
        self._cleared_last_gate = True  
        self.cleared_gates = []
        self._recalculated_for_approach = -1  

        self._des_pos_spline = None
        self._t_total = 0.0
        self._t_offset = 0.0
        self._last_replan_time = 0.0

        self._build_trajectory(obs)

    def _get_forward_vector(self, gate_quat: NDArray) -> NDArray:
        """Calculates the normalized forward direction vector of a gate."""
        rot = R.from_quat(gate_quat)
        forward = rot.apply([1.0, 0.0, 0.0])
        return forward / (np.linalg.norm(forward) + 1e-8)

    def _build_trajectory(self, obs: dict[str, NDArray[np.floating]]):
        target_gate_obs = int(obs["target_gate"])
        if target_gate_obs == -1 or target_gate_obs >= self._n_gates:
            self._finished = True
            return
            
        drone_pos = np.array(obs["pos"], dtype=np.float64).flatten()[:3]
        drone_vel = np.array(obs.get("vel", [0, 0, 0]), dtype=np.float64).flatten()[:3]

        # --- PATH CONTINUITY: REUSE N POINTS ---
        N_points_to_reuse = 2  # Number of future points to stitch (adjust as needed)
        dt_lookahead = 0.1   # Time gap (seconds) between reused points
        
        # Always start exactly where the drone is to prevent jump tracking errors
        waypoints = [(drone_pos.copy(), -1)]
        
        if self._des_pos_spline is not None:
            # Replanning: Sample the next N points from the currently active trajectory
            curr_t = np.clip(self._tick / self._freq - self._t_offset, 0.0, self._t_total)
            for i in range(1, N_points_to_reuse + 1):
                t_sample = min(curr_t + (i * dt_lookahead), self._t_total)
                waypoints.append((self._des_pos_spline(t_sample).copy(), -1))
        else:
            # Initial plan: Fallback to the velocity vector heuristic to build momentum
            w1 = drone_pos + (drone_vel * 0.1) 
            w2 = drone_pos + (drone_vel * 0.2) 
            waypoints.extend([(w1, -1), (w2, -1)])

        # Proceed with extracting gates and obstacles as usual...
        gates_pos = np.array(obs["gates_pos"], dtype=np.float64)
        gates_quat = np.array(obs["gates_quat"], dtype=np.float64)
        obstacles_pos = np.array(obs["obstacles_pos"], dtype=np.float64)
        num_regular_obstacles = len(obstacles_pos)
        
        target_gate = target_gate_obs
        obstacles_pos = np.vstack((obstacles_pos, np.array(gates_pos)))            
            
        curr_pt = waypoints[-1][0]
        for i in range(target_gate, self._n_gates):
            gp = gates_pos[i].flatten()[:3]
            gate_axis = self._get_forward_vector(gates_quat[i].flatten()[:4])
            
            v_to_gate = gp - curr_pt
            dist_to_gate = np.linalg.norm(v_to_gate) # Get the physical distance
            
            fwd = gate_axis if np.dot(v_to_gate, gate_axis) > 0 else -gate_axis
            
            # Dynamically scale app_dist. 
            # It caps at 0.5m, but shrinks if the drone is closer than 1.0m.
            # Using * 0.5 ensures the waypoint stays halfway between the drone and the gate.
            app_dist = min(0.6, dist_to_gate * 0.5)
            ex_dist = 0.7
            
            app = gp - fwd * app_dist
            ex = gp + fwd * ex_dist
            
            waypoints.append((app, i))
            waypoints.append((gp, i))
            waypoints.append((ex, i))
                
            curr_pt = waypoints[-1][0]

        # 2. Obstacle Avoidance (Markers & Passed Gates)
        safe_waypoints = [waypoints[0][0]]
        
        for i in range(len(waypoints) ):
            pA = safe_waypoints[-1]
            pB = waypoints[i][0]
            
            def resolve_segment(start, end, depth=0):
                # 1. Increased depth limit to handle multiple obstacles on long segments
                if depth > 2: 
                    return []
                
                v3d = end - start
                len3d_sq = np.dot(v3d, v3d)
                
                # For 2D cylinder checks
                v2d = v3d[:2]
                len2d_sq = np.dot(v2d, v2d)
                
                min_s = float('inf')
                hit_obs_idx = -1
                hit_closest_pt = None
                
                for idx, obs_p in enumerate(obstacles_pos):
                    is_frame = idx >= num_regular_obstacles
                    clearance = 0.2
                    
                    if is_frame:
                        # 3D Sphere/Gate logic
                        if len3d_sq < 1e-6:
                            s = 0.5
                            closest_pt = start
                        else:
                            AP = obs_p - start
                            s = np.clip(np.dot(AP, v3d) / len3d_sq, 0.0, 1.0)
                            closest_pt = start + s * v3d
                            
                        frame_idx = idx - num_regular_obstacles
                        q = gates_quat[frame_idx]
                        dist_to_obs = min_distance_drone_gate(closest_pt, obs_p, q)/0.8
                        
                    else:
                        # GEOMETRY FIX: Calculate parameter `s` using only 2D planar math
                        if len2d_sq < 1e-6:
                            s = 0.5
                            closest_pt = start
                        else:
                            AP_2d = (obs_p - start)[:2]
                            s = np.clip(np.dot(AP_2d, v2d) / len2d_sq, 0.0, 1.0)
                            # We map the 2D 's' back to the 3D line to find the actual collision height
                            closest_pt = start + s * v3d
                        
                        dist_to_obs = np.linalg.norm((closest_pt - obs_p)[:2])
                    
                    if dist_to_obs < clearance and s < min_s:
                        min_s = s
                        hit_obs_idx = idx
                        hit_closest_pt = closest_pt
                            
                if hit_obs_idx != -1:
                    obs_p = obstacles_pos[hit_obs_idx]
                    is_frame = hit_obs_idx >= num_regular_obstacles
                    
                    # 2. Dynamic push distance based on segment length
                    base_push = 0.2
                    segment_length = np.sqrt(len3d_sq) if len3d_sq > 0 else 0
                    push_dist = base_push + (segment_length * 0.05)
                    
                    push_dir = hit_closest_pt - obs_p
                    
                    # For cylinders, we usually only push horizontally. 
                    if not is_frame:
                        push_dir[2] = 0 
                    
                    # ARITHMETIC FIX: Always normalize the push direction!
                    norm_val = np.linalg.norm(push_dir)
                    if norm_val > 1e-6:
                        push_dir = push_dir / norm_val
                    else:
                        # Fallback if center hit exactly perfectly
                        push_dir = np.array([1.0, 0.0, 0.0]) 

                    if not is_frame and push_dir[2] > 0:
                        # If doing 3D push, re-normalize after vertical bias
                        push_dir[2] *= 1.1
                        push_dir = push_dir / np.linalg.norm(push_dir)
                        
                    # 3. Calculate backward pull vector
                    back_dir = start - hit_closest_pt
                    back_norm = np.linalg.norm(back_dir)
                    if back_norm > 1e-6:
                        back_dir = back_dir / back_norm
                    else:
                        back_dir = np.zeros(3)
                        
                    # Combine outward push with backward bias for a safe approach angle
                    diversion = hit_closest_pt + (push_dir * push_dist) + (back_dir * 0.2)                    
                    diversion[2] = max(0.8, diversion[2])
                    
                    left_path = resolve_segment(start, diversion, depth + 1)
                    right_path = resolve_segment(diversion, end, depth + 1)
                    
                    return left_path + [diversion] + right_path
                else:
                    return []

            # Apply collision resolution uniformly to all segments
            # Skip collision check for segments that are part of a gate passage (same gate index)
            idxA = waypoints[i-1][1] if i>0 else -1
            idxB = waypoints[i][1]
            if (idxA != idxB) or idxA==-1  :
                diversions = resolve_segment(pA, pB)
                safe_waypoints.extend(diversions)

            safe_waypoints.append(pB)

        fw = np.array(safe_waypoints)
        
        diffs = np.linalg.norm(np.diff(fw, axis=0), axis=1)
        valid_mask = np.insert(diffs > 0.05, 0, True)
        fw = fw[valid_mask]
        
        if len(fw) < 2:
            self._finished = True
            return

        # 3. Time Allocation based on Segment Distances
        dists = np.linalg.norm(np.diff(fw, axis=0), axis=1)
        cum_dists = np.insert(np.cumsum(dists), 0, 0.0)
        total_dist = cum_dists[-1]
        
        curr_time = self._tick / self._freq
        rem_budget = max(1.0, self._total_budget - curr_time)
        
        if total_dist > 1e-4:
            ft = (cum_dists / total_dist) * rem_budget
        else:
            ft = np.linspace(0, rem_budget, len(fw))
            
        for i in range(1, len(ft)):
            if ft[i] <= ft[i-1]:
                ft[i] = ft[i-1] + 1e-3


        self._t_total = ft[-1]
        self._des_pos_spline = PchipInterpolator(ft, fw)
        self._t_offset = curr_time
        self._last_replan_time = curr_time
        self._last_target_gate = target_gate
        self._current_combined_obstacles = obstacles_pos

    def compute_control(
            self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
        ) -> NDArray[np.floating]:
        target_gate = int(obs["target_gate"])
        obstacles_visited = np.array(obs["obstacles_visited"], dtype=bool)

        drone_pos = np.array(obs["pos"], dtype=np.float64).flatten()[:3]
        gates_pos = np.array(obs["gates_pos"], dtype=np.float64)
        gates_quat = np.array(obs["gates_quat"], dtype=np.float64)

        replan = False
        dist_past = float("inf")
        curr_time = self._tick / self._freq

        t_s = np.clip(self._tick / self._freq - self._t_offset, 0.0, self._t_total)
        if t_s >= self._t_total:
            self._finished = True

        des_pos = self._des_pos_spline(t_s)
        height_offset = 0.1
        des_pos[2] -= height_offset
        
        # Initialize an obstacle tracking set if it doesn't exist yet
        if not hasattr(self, "_recalculated_obstacles"):
            self._recalculated_obstacles = set()
        
        # Check proximity to obstacles to slow down for precision near hazards
        dist_to_obs = float("inf")
        if hasattr(self, "_current_combined_obstacles") and self._current_combined_obstacles.size > 0:
            # Using 2D distance for pillars and passed gate centers as a safety heuristic
            obs_dists = np.linalg.norm(self._current_combined_obstacles[:, :2] - drone_pos[:2], axis=1)
            closest_obs_idx = int(np.argmin(obs_dists))
            dist_to_obs = obs_dists[closest_obs_idx]
            
            # Replan the first time we are closer than 0.7m to this specific obstacle
            if dist_to_obs < 0.6 and closest_obs_idx not in self._recalculated_obstacles:
                replan = True
                self._recalculated_obstacles.add(closest_obs_idx)

        drone_pos_f32 = np.array(obs["pos"], dtype=np.float32).flatten()[:3]
        target_vec = des_pos - drone_pos_f32
        dist = np.linalg.norm(target_vec)
    
                
        dist_to_target = float("inf")
        if not replan and 0 <= target_gate < self._n_gates:
            target_gp = gates_pos[target_gate].flatten()[:3]
            dist_to_target = np.linalg.norm(drone_pos - target_gp)
            
            if dist_to_target < 0.6 and self._recalculated_for_approach != target_gate:
                replan = True
                self._recalculated_for_approach = target_gate

        # Reduce max_dist when near gates or obstacles to improve tracking precision
        max_dist = 15.0 if (dist_to_target > 0.8 and dist_past > 0.8 and dist_to_obs > 0.8) else 15

        if replan:
            self._build_trajectory(obs)
                
        self._last_obstacles_visited = obstacles_visited

        if self._finished or self._des_pos_spline is None:
            return np.zeros(13, dtype=np.float32)
            
        
        if dist > max_dist:
            des_pos = drone_pos_f32 + (target_vec / (dist + 1e-8)) * max_dist
            if dist > max_dist * 1.2:
                self._t_offset += 1.0 / self._freq
        
        t_lookahead = np.clip(t_s + 0.1, 0.0, self._t_total)
        raw_spline_pos = self._des_pos_spline(t_s)
        delta = self._des_pos_spline(t_lookahead) - raw_spline_pos
        des_yaw = float(np.arctan2(delta[1], delta[0])) if np.linalg.norm(delta[:2]) > 1e-2 else 0.0

        action = np.zeros(13, dtype=np.float32)
        
        action[0:3] = des_pos
        action[9] = des_yaw 
        return action

    def step_callback(self, action, obs, reward, terminated, truncated, info) -> bool:
        self._tick += 1
        return self._finished

    def episode_callback(self):
        self.episode_reset()

    def episode_reset(self):
        self._tick = 0
        self._finished = False
        self._last_target_gate = -2
        self._recalculated_for_approach = -1
        self._des_pos_spline = None
        self._last_replan_time = 0.0
        self.cleared_gates = []             # Ensures clean slate per episode
        self._cleared_last_gate = True      # Reset gate logic flag

    def render_callback(self, sim: Sim):
        if self._des_pos_spline is None:
            return
        try:
            from crazyflow.sim.visualize import draw_line, draw_points
            t_s = np.clip(self._tick / self._freq - self._t_offset, 0.0, self._t_total)
            draw_points(sim, self._des_pos_spline(t_s).reshape(1, -1), rgba=(1, 0, 0, 1), size=0.03)
            draw_line(sim, self._des_pos_spline(np.linspace(0, self._t_total, 100)), rgba=(0, 1, 0, 1))
            
            if hasattr(self, "_current_combined_obstacles"):
                obs_pts = np.array(self._current_combined_obstacles)
                if obs_pts.size > 0:
                    draw_points(sim, obs_pts, rgba=(1, 1, 0, 0.5), size=0.1)
        except ImportError:
            pass