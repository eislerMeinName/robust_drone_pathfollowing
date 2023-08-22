import os
import time
import argparse
import numpy as np
from numpy import mean
from scipy.stats import sem
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

def run(episodes: int, ax):
    name: List[str] = ['../results/SAC4D48_1.zip', '../results/SAC4D24_1.zip']
    for i, n in enumerate(name):
        debug(bcolors.OKBLUE, '[INFO] Model: ' + n)
        # load model
        model = SAC.load(n)

        f_max: np.array = np.arange(0, 0.3, 0.01)
        succes: np.array = np.array([])

        time: np.array = np.array([])

        overshoot: np.array = np.array([])
        overshoot_std: np.array = np.array([])

        # evaluate for different r
        for f in f_max:
            debug(bcolors.OKBLUE, '[INFO] force: ' + str(f) + 'N')
            eval_env = gym.make('WindSingleAgent-aviary-v0', aggregate_phy_steps=5, obs=ObservationType('kin'),
                                act=ActionType.RPM, mode=3, total_force=f, radius=0, episode_len=5,
                                drone_model=DroneModel("hb"))

            eval: EvalWriter = EvalWriter(name='TestWriter', eval_steps=episodes, path='test.xlsx', env=eval_env,
                                          episode_len=5, threshold=0.05)
            eval.evaluateModel(model, False)
            eval.calcOvershoot()
            succes = np.append(succes, (eval.succeeded_steps / eval.total_steps) * 100)
            time = np.append(time, eval.total_time / (eval.total_steps * eval.episode_len) * 100)
            overshoot = np.append(overshoot, mean(eval.overshoot))
            overshoot_std = np.append(overshoot_std, sem(eval.overshoot))

        # plot
        clean_name: str = n.replace('..', '').replace('results', '').replace('/', '').replace('.zip', '')
        color: str = defineColor(clean_name)

        ax[0].plot(f_max, succes, label=clean_name, color=color)
        ax[1].plot(f_max, time, label=clean_name, color=color)
        ax[2].errorbar(f_max, overshoot, yerr=overshoot_std, fmt='-o', label=clean_name, color=color)

    ax[0].set_xlabel('wind force [N]', fontsize=18)
    ax[0].set_ylabel('success rate [%]', fontsize=18)

    ax[1].set_xlabel('wind force [N]')
    ax[1].set_ylabel('time rate [%]')

    ax[2].set_xlabel('wind force [N]')
    ax[2].set_ylabel('overshoot [m]')

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    plt.show()


if __name__ == "__main__":
    plt.rc('font', size=18)
    fig, ax = plt.subplots(1, 3)
    run(100, ax)