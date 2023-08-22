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


def defineColor(name: str) -> str:
    if "SAC4D24" in name:
        return "tab:blue"
    if "SAC4D48" in name:
        return "tab:orange"
    if "SAC1D" in name:
        return "tab:green"
    if "PPO1D" in name:
        return "tab:red"
    if "PPO4D" in name:
        return "tab:purple"
    else:
        return ''


if __name__ == "__main__":
    plt.rc('font', size=20)
    name: List[str] = ['../results/SAC4D48_1.zip', '../results/SAC4D24_1.zip']#'../results/PPO4D_1.zip']

    for i, n in enumerate(name):
        debug(bcolors.OKBLUE, '[INFO] Model: ' + n)
        # load model
        if 'SAC' in n:
            model = SAC.load(n)
        else:
            model = PPO.load(n)

        t_max: np.array = np.arange(5, 61, 1)
        stds: List[float] = []
        means: np.array = np.array([])

        # evaluate for different ts
        for t in t_max:
            debug(bcolors.OKBLUE, '[INFO] episode length: ' + str(t) + 's')
            eval_env = gym.make('WindSingleAgent-aviary-v0', aggregate_phy_steps=5, obs=ObservationType('kin'),
                                act=ActionType.RPM, mode=1, total_force=0, radius=0, episode_len=t,
                                drone_model=DroneModel("hb"))

            eval: EvalWriter = EvalWriter(name='TestWriter', eval_steps=2, path='test.xlsx', env=eval_env,
                                          episode_len=t, threshold=0.05)
            mean, std = eval.evaluateModel(model, False)
            stds.append(std)
            means = np.append(means, mean)

        # plot
        color: str = defineColor(n)
        clean_name: str = n.replace('..', '').replace('results', '').replace('/', '').replace('.zip', '') + "-48Hzenv"
        plt.plot(t_max, means, c=color, label=clean_name)
        debug(bcolors.BOLD, '[Result] ' + str(means[0]))
        debug(bcolors.BOLD, '[Result] ' + str(means[len(means) - 1]))

        t_max: np.array = np.arange(5, 61, 1)
        stds: List[float] = []
        means: np.array = np.array([])

        for t in t_max:
            debug(bcolors.OKBLUE, '[INFO] episode length: ' + str(t) + 's')
            eval_env = gym.make('WindSingleAgent-aviary-v0', aggregate_phy_steps=10, obs=ObservationType('kin'),
                                act=ActionType.RPM, mode=1, total_force=0, radius=0, episode_len=t,
                                drone_model=DroneModel("hb"))

            eval: EvalWriter = EvalWriter(name='TestWriter', eval_steps=2, path='test.xlsx', env=eval_env,
                                          episode_len=t, threshold=0.05)
            mean, std = eval.evaluateModel(model, False)
            stds.append(std)
            means = np.append(means, mean)

        # plot
        clean_name: str = n.replace('..', '').replace('results', '').replace('/', '').replace('.zip', '') + "-24Hzenv"
        plt.plot(t_max, means, c=color, alpha=0.5, label=clean_name)
        debug(bcolors.BOLD, '[Result] ' + str(means[0]))
        debug(bcolors.BOLD, '[Result] ' + str(means[len(means) - 1]))

    plt.legend()
    plt.xlabel('time[s]')
    plt.ylabel('return')
    plt.show()
