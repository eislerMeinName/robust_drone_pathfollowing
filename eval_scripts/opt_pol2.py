import os
import time
import argparse
import numpy as np
import gym
from stable_baselines3 import PPO, SAC
from stable_baselines3 import DDPG
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.utils.utils import sync
from robust_drone_pathfollowing.helpclasses.printout import *
from robust_drone_pathfollowing.helpclasses.evalwriter import EvalWriter
from robust_drone_pathfollowing.helpclasses.pathplotter import PathPlotter
from gym_pybullet_drones.utils.enums import DroneModel
from typing import List
import matplotlib.pyplot as plt


def run():
    test_env = gym.make('WindSingleAgent-aviary-v0', aggregate_phy_steps=5, obs=ObservationType('kin'),
                        act=ActionType.RPM, mode=2, total_force=0, radius=0,
                        drone_model=DroneModel("hb"), debug=False, gui=True, episode_len=5)
    test_env.reset()
    obs = test_env.setGoal(np.array([0, 0, 0.6875]))
    pathplotter = PathPlotter(test_env.goal)
    start = time.time()
    REWARD: float = 0.0
    for i in range(5 * int(test_env.SIM_FREQ / test_env.AGGR_PHY_STEPS)):
        dist = 1 - test_env.pos[0][2]
        action = opt_pol(i, dist)
        if action is None:
            break
        obs, reward, done, info = test_env.step(action)
        REWARD += reward
        debug(bcolors.OKBLUE, str(dist))
        # test_env.render()
        pathplotter.addPose(test_env.getPose())
        sync(np.floor(i * test_env.AGGR_PHY_STEPS), start, test_env.TIMESTEP)
        if done:
            obs = test_env.reset()
            break  # OPTIONAL EPISODE HALT
    test_env.close()
    # logger.save_as_csv("sa")  # Optional CSV save
    pathplotter.show()
    debug(bcolors.OKBLUE, str(REWARD))
    debug(bcolors.OKBLUE, str(i * (1 / (test_env.SIM_FREQ / test_env.AGGR_PHY_STEPS))))


def opt_pol(i: int, dist):
    if i < 50:
        return np.array([1, 1, 1, 1])
    else:
        return None


if __name__ == "__main__":
    run()