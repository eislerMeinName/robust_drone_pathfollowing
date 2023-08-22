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

def run(episodes: int):
    name: List[str] = ['../results/SAC4D48_1.zip', '../results/SAC4D24_1.zip']
    for i, n in enumerate(name):
        debug(bcolors.OKBLUE, '[INFO] Model: ' + n)
        # load model
        model = SAC.load(n)

        f_max: np.array = np.arange(0, 0.3, 0.01)
        stds: np.array = np.array([])
        means: np.array = np.array([])

        # evaluate for different r
        for f in f_max:
            debug(bcolors.OKBLUE, '[INFO] force: ' + str(f) + 'N')
            eval_env = gym.make('WindSingleAgent-aviary-v0', aggregate_phy_steps=5, obs=ObservationType('kin'),
                                act=ActionType.RPM, mode=3, total_force=f, radius=0, episode_len=5,
                                drone_model=DroneModel("hb"))

            eval: EvalWriter = EvalWriter(name='TestWriter', eval_steps=episodes, path='test.xlsx', env=eval_env,
                                          episode_len=5, threshold=0.05)
            mean, std = eval.evaluateModel(model, False)
            stds = np.append(stds, std)
            means = np.append(means, mean)

        # plot
        clean_name: str = n.replace('..', '').replace('results', '').replace('/', '').replace('.zip', '')
        color: str = defineColor(clean_name)
        print(color)
        plt.errorbar(f_max, means, yerr=stds, fmt='-o', label=clean_name, color=color)
        debug(bcolors.BOLD, '[Result] ' + str(means[0]) + '+-' + str(stds[0]))
        debug(bcolors.BOLD, '[Result] ' + str(means[len(means) - 1]) + '+-' + str(stds[len(stds) - 1]))

    plt.xlabel('wind force [N]')
    plt.ylabel('return')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plt.rc('font', size=18)
    run(100)