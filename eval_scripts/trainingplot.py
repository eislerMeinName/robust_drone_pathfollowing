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

def read(name: str, max: float, offset: float):
    step: List[float] = []
    rew: List[float] = []
    with open(name, newline='') as csvfile:

        reader = csv.DictReader(csvfile)

        for row in reader:
            debug(bcolors.OKBLUE, str(row['Step']) + str(row['Value']))
            if float(row['Step']) > max:
                break
            step.append(float(row['Step']) + offset)
            rew.append(float(row['Value']))

    return step, rew


if __name__ == "__main__":
    plt.rc('font', size=20)
    name: List[str] = ['../files/CSV/SACsphere.csv']
    step, rew = read(name[0], 1e8, 0)
    smoothed = smooth(rew, 0.99)
    plt.plot(step, rew, color='tab:red', alpha=0.25, label='SAC')
    plt.plot(step, smoothed, color='tab:red', label='SAC Smoothed')

    name: List[str] = ['../files/CSV/SAC4Dcurric1.csv', '../files/CSV/SAC4Dcurric2.csv',
                       '../files/CSV/SAC4Dcurric3.csv', '../files/CSV/SAC4Dcurric4.csv',
                       '../files/CSV/SAC4Dcurric5.csv', '../files/CSV/SAC4Dcurric6.csv']

    STEP =  []
    REW = []
    for i, n in enumerate(name):
        step, rew = read(name[i], 16666000, i*16666000)
        if i == 0:
            plt.axvline(i*16666000, color='green', label='δ=0.2')
        else:
            plt.axvline(i * 16666000, color='green')
        for i, r in enumerate(rew):
            STEP.append(step[i])
            REW.append(rew[i])

    smoothed = smooth(REW, 0.99)
    plt.plot(STEP, REW, color='tab:blue', alpha=0.25, label='SAC LCL (δ=0.2)')
    plt.plot(STEP, smoothed, color='tab:blue', label='SAC LCL (δ=0.2) Smoothed')

    name: List[str] = ['../files/CSV/delta0_4.csv', '../files/CSV/2delta0_4.csv',
                       '../files/CSV/3delta0_4.csv']

    STEP = []
    REW = []
    for i, n in enumerate(name):
        step, rew = read(name[i], 28570000, i * 28570000)
        if i == 0:
            plt.axvline(i * 28570000, color='tab:purple', label='δ=0.2')
        else:
            plt.axvline(i * 28570000, color='tab:purple')
        for i, r in enumerate(rew):
            STEP.append(step[i])
            REW.append(rew[i])

    smoothed = smooth(REW, 0.99)
    plt.plot(STEP, REW, color='tab:orange', alpha=0.25, label='SAC LCL (δ=0.4)')
    plt.plot(STEP, smoothed, color='tab:orange', label='SAC LCL (δ=0.4) Smoothed')

    plt.xlim(0, 1e8)
    plt.xlabel('training steps')
    plt.ylabel('return')
    plt.legend()
    plt.show()