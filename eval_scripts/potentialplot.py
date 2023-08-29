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

def evaluateGoal(goal, model, env):
    env.reset()
    obs = env.setGoal(goal)
    R: float = 0
    for i in range(5 * int(env.SIM_FREQ / env.AGGR_PHY_STEPS)):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        R += reward
        if done:
            break  # OPTIONAL EPISODE HALT

    return R

def potentialPlot(model, env, amount):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # x = np.linspace(-radius, radius, 100)
    # y = np.linspace(-radius, radius, 100)
    # X, Y = np.meshgrid(x,y)
    # gX = X.flatten()
    # gY = Y.flatten()
    phi: float = np.pi * np.random.random(amount)
    theta: float = np.pi * np.random.random(amount)
    r: float = 0.5 * np.random.random(amount)
    x = np.ravel(r * np.sin(theta) * np.cos(phi))
    z = 0.5 + np.ravel(r * np.sin(theta) * np.sin(phi))
    y = np.ravel(r * np.cos(theta))
    Z = []
    goal = []
    for i, j in enumerate(x):
        goal.append(np.array([x[i], y[i], z[i]]))


    for i, g in enumerate(goal):
        if i % 100 == 0:
            debug(bcolors.OKBLUE, str(i))
        Z.append(evaluateGoal(g, model, env))

    aha = ax.scatter(x, y, z, c=Z, cmap='viridis', linewidth=0.5);
    #ax.plot_trisurf(x, y, z,
    #                cmap='viridis', edgecolor='none');
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z');
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(0, 1)
    plt.colorbar(aha, label="return", orientation="horizontal")
    plt.show()

if __name__ == "__main__":
    model = SAC.load('../results/0_2delta/best_modelcurri_r0.5.zip')
    env = gym.make('WindSingleAgent-aviary-v0', aggregate_phy_steps=5, obs=ObservationType('kin'),
                    act=ActionType.RPM, mode=2, total_force=0, radius=0, episode_len=5,
                    drone_model=DroneModel("hb"))
    potentialPlot(model, env, 7500)