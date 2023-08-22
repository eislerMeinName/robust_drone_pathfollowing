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

import csv


def smooth(scalars: List[float], weight) -> List[float]:
    last = scalars[0]
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    return smoothed

def read(name: str, max: float):
    step: List[float] = []
    rew: List[float] = []
    with open(name, newline='') as csvfile:

        reader = csv.DictReader(csvfile)

        for row in reader:
            debug(bcolors.OKBLUE, str(row['Step']) + str(row['Value']))
            if float(row['Step']) > max:
                break
            step.append(float(row['Step']))
            rew.append(float(row['Value']))

    return step, rew


if __name__ == "__main__":
    plt.rc('font', size=20)
    #plt.grid()
    name: List[str] = ['../files/CSV/SAC24Hz.csv', '../files/CSV/SAC48Hz.csv', '../files/CSV/PPO48Hz.csv']
    step, rew = read(name[0], 3e7)
    smoothed = smooth(rew, 0.99)
    plt.plot(step, rew, color='tab:blue', alpha=0.25, label='SAC4D24_1')
    plt.plot(step, smoothed, color='tab:blue', label='SAC4D24_1 Smoothed')

    step, rew = read(name[1], 3e7)
    smoothed = smooth(rew, 0.99)
    plt.plot(step, rew, color='tab:orange', alpha=0.25, label='SAC4D48_1')
    plt.plot(step, smoothed, color='tab:orange', label='SAC4D48_1 Smoothed')

    step, rew = read(name[2], 3e7)
    smoothed = smooth(rew, 0.99)
    plt.plot(step, rew, color='tab:purple', alpha=0.25, label='PPO4D48_1')
    plt.plot(step, smoothed, color='tab:purple', label='PPO4D48_1 Smoothed')

    plt.xlim(0, 3e7)
    plt.xlabel('training steps')
    plt.ylabel('return')
    plt.legend()
    plt.show()
