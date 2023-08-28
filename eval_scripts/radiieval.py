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
import numpy as np
import gym

def definecolor(name: str) -> str:
    if 'SAC4D' in name:
        return 'tab:pink'
    elif '0_4delta' in name:
        return 'black'
    elif '0_2delta' in name:
        return 'greenyellow'

def run():
    name: List[str] = ['../results/SAC4D_2.zip',
                       '../results/0_4delta/best_modelcurri_r0.5.zip',
                       '../results/0_2delta/best_modelcurri_r0.5.zip']

    for i, n in enumerate(name):
        debug(bcolors.OKBLUE, '[INFO] Model: ' + n)
        # load model
        model = SAC.load(n)

        r_max: np.array = np.arange(0, 0.61, 0.01)
        stds: np.array = np.array([])
        means: np.array = np.array([])

        # evaluate for different r
        for r in r_max:
            debug(bcolors.OKBLUE, '[INFO] radius: ' + str(r) + 'm')
            eval_env = gym.make('WindSingleAgent-aviary-v0', aggregate_phy_steps=5, obs=ObservationType('kin'),
                                act=ActionType.RPM, mode=2, total_force=0, radius=r, episode_len=5,
                                drone_model=DroneModel("hb"))

            eval: EvalWriter = EvalWriter(name='TestWriter', eval_steps=100, path='test.xlsx', env=eval_env,
                                          episode_len=5, threshold=0.05)
            mean, std = eval.evaluateModel(model, False)
            stds = np.append(stds, std)
            means = np.append(means, mean)

        # plot
        clean_name: str = n.replace('..', '').replace('results', '').replace('/', '').replace('.zip', '')
        color=definecolor(clean_name)
        clean_name = clean_name.replace('4D_2', '')
        if color == 'black':
            clean_name = 'SAC LCL (δ=0.4)'
        if color == 'greenyellow':
            clean_name = 'SAC LCL (δ=0.2)'
        plt.errorbar(r_max, means, yerr=stds, fmt='-o', label=clean_name, color=color)
        debug(bcolors.BOLD, '[Result] ' + str(means[0]) + '+-' + str(stds[0]))
        debug(bcolors.BOLD, '[Result] ' + str(means[len(means) - 1]) + '+-' + str(stds[len(stds) - 1]))

    plt.xlabel('radius[m]')
    plt.ylabel('return')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plt.rc('font', size=20)
    run()