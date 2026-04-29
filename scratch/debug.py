import os
import sys
from pathlib import Path
import numpy as np

# Add project root to sys.path
sys.path.append(str(Path(__file__).parents[2]))

import lsy_drone_racing.envs

from lsy_drone_racing.utils import load_config, load_controller
import gymnasium
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

config = load_config(Path(__file__).parents[1] / "config" / "level0.toml")
controller_cls = load_controller(Path(__file__).parents[1] / "lsy_drone_racing/control/adaptive_controller.py")

env = gymnasium.make(
    config.env.id,
    freq=config.env.freq,
    sim_config=config.sim,
    sensor_range=config.env.sensor_range,
    control_mode=config.env.control_mode,
    track=config.env.track,
    disturbances=config.env.get("disturbances"),
    randomizations=config.env.get("randomizations"),
    seed=config.env.seed,
)
env = JaxToNumpy(env)

obs, info = env.reset()
controller = controller_cls(obs, info, config)

i = 0
while True:
    action = controller.compute_control(obs, info)
    obs, reward, terminated, truncated, info = env.step(action)
    controller_finished = controller.step_callback(action, obs, reward, terminated, truncated, info)
    
    if terminated or truncated or controller_finished:
        print(f"Failed at step {i}, time {i/config.env.freq:.2f}s")
        print(f"Position: {obs['pos']}")
        print(f"Velocity: {obs['vel']}")
        print(f"Terminated: {terminated}, Truncated: {truncated}")
        break
    i += 1

env.close()
