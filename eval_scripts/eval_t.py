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


def run(episodes: int,
        act: List[ActionType],
        mode: int,
        radius: float,
        algo: List[str],
        name: List[str],
        ):
    for i, n in enumerate(name):
        debug(bcolors.OKBLUE, '[INFO] Model: ' + n)
        # load model
        if algo[i] == 'ppo':
            model = PPO.load(n)
        elif algo[i] == 'sac':
            model = SAC.load(n)

        t_max: np.array = np.arange(5, 61, 1)
        stds: List[float] = []
        means: np.array = np.array([])

        # evaluate for different ts
        for t in t_max:
            debug(bcolors.OKBLUE, '[INFO] episode length: ' + str(t) + 's')
            eval_env = gym.make('WindSingleAgent-aviary-v0', aggregate_phy_steps=10, obs=ObservationType('kin'),
                                act=act[i], mode=mode, total_force=0, radius=radius, episode_len=t,
                                drone_model=DroneModel("hb"))

            eval: EvalWriter = EvalWriter(name='TestWriter', eval_steps=episodes, path='test.xlsx', env=eval_env,
                                          episode_len=t, threshold=0.05)
            mean, std = eval.evaluateModel(model, False)
            stds.append(std)
            means = np.append(means, mean)

        # plot
        clean_name: str = n.replace('..', '').replace('results', '').replace('/', '').replace('.zip', '')
        if mode > 1:
            plt.errorbar(t_max, means, yerr=stds, fmt='-o', label=clean_name)
            debug(bcolors.BOLD, '[Result] ' + str(means[0]) + '+-' + str(stds[0]))
            debug(bcolors.BOLD, '[Result] ' + str(means[len(means) - 1]) + '+-' + str(stds[len(stds)] - 1))
        else:
            plt.plot(t_max, means, label=clean_name)
            debug(bcolors.BOLD, '[Result] ' + str(means[0]))
            debug(bcolors.BOLD, '[Result] ' + str(means[len(means) - 1]))

    plt.legend()
    plt.xlabel('time[s]')
    plt.ylabel('reward')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to evaluate multiple models with different episode lengths.')
    parser.add_argument('--episodes', default=100, type=int, help='The amount of episodes.', metavar='')
    parser.add_argument('--act', default=[], type=ActionType, nargs='+',
                        help='The action types of the environment', metavar='')
    parser.add_argument('--mode', default=0, type=int,
                        help='The mode of the training environment(default: 0)', metavar='')
    parser.add_argument('--algo', default=[], nargs='+', help='The algorithms.', metavar='')
    parser.add_argument('--name', default=[], nargs='+',
                        help='The names of the models', metavar='')
    parser.add_argument('--radius', default=0.0, type=float, help='The radius.', metavar='')

    ARGS = parser.parse_args()

    run(**vars(ARGS))