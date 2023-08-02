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
    env = gym.make('WindSingleAgent-aviary-v0', aggregate_phy_steps=5, obs=ObservationType('kin'),
                        act=ActionType.RPM, mode=1, total_force=0, radius=0,
                        drone_model=DroneModel("hb"), debug=False, gui=False, episode_len=5)

    name: List[str] = ["../results/PPO1D_1.zip", "../results/PPO4D_1.zip", "../results/SAC1D_1.zip", "../results/SAC4D48_1.zip"]
    for i, n in enumerate(name):
        actions = []
        times = []
        debug(bcolors.OKBLUE, '[INFO] Model: ' + n)
        # load model
        if "PPO" in n:
            model = PPO.load(n)
        elif "SAC" in n:
            model = SAC.load(n)

        if "1D" in n:
            env = gym.make('WindSingleAgent-aviary-v0', aggregate_phy_steps=5, obs=ObservationType('kin'),
                        act=ActionType.ONE_D_RPM, mode=1, total_force=0, radius=0,
                        drone_model=DroneModel("hb"), debug=False, gui=False, episode_len=5)
        if "4D" in n:
            env = gym.make('WindSingleAgent-aviary-v0', aggregate_phy_steps=5, obs=ObservationType('kin'),
                           act=ActionType.RPM, mode=1, total_force=0, radius=0,
                           drone_model=DroneModel("hb"), debug=False, gui=False, episode_len=5)
        obs = env.reset()
        for j in range(5 * int(env.SIM_FREQ / env.AGGR_PHY_STEPS)):
            action, _states = model.predict(obs, deterministic=True)
            #print(action)
            actions.append(action)
            times.append(j / 240 * 5)
            obs, reward, done, info = env.step(action)
            if done:
                obs = env.reset()
                break

        env.reset()


        one = []
        two = []
        three = []
        four = []
        if "1D" in n:
            for act in actions:
                one.append(act)
                two.append(act)
                three.append(act)
                four.append(act)

        if "4D" in n:
            for act in actions:
                one.append(act[0])
                two.append(act[1])
                three.append(act[2])
                four.append(act[3])

        name = n.replace("../results/", "")
        n = name
        color: str = defineColor(n)
        ax[0].plot(times, one, c=color, label=n)
        ax[1].plot(times, two, c=color, label=n)
        ax[2].plot(times, three, c=color, label=n)
        ax[3].plot(times, four, c=color, label=n)

    ax[0].legend(loc="lower left")
    ax[0].set_xlabel('Time [s]')
    ax[0].set_ylabel('a1')

    ax[1].legend(loc="lower left")
    ax[1].set_xlabel('Time [s]')
    ax[1].set_ylabel('a2')

    ax[2].legend(loc="lower left")
    ax[2].set_xlabel('Time [s]')
    ax[2].set_ylabel('a3')

    ax[3].legend(loc="lower left")
    ax[3].set_xlabel('Time [s]')
    ax[3].set_ylabel('a4')

    plt.show()
    env.close()

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
    fig, ax = plt.subplots(4, 1)
    run()