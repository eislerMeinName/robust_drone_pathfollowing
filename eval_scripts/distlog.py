import matplotlib.pyplot as plt
from typing import List
from robust_drone_pathfollowing.helpclasses.printout import *
from stable_baselines3 import PPO, SAC
import numpy as np
import gym
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.utils.enums import DroneModel

if __name__ == "__main__":
    plt.rc('font', size=20)
    names: List[str] = ["../results/SAC4D24_1.zip", "../results/SAC4D48_1.zip",
                        "../results/SAC1D_1.zip", "../results/PPO1D_1.zip", "../results/PPO4D_1.zip"]

    for i, n in enumerate(names):
        debug(bcolors.OKBLUE, '[INFO] Model: ' + n)
        # load model
        if 'SAC' in n:
            model = SAC.load(n)
        else:
            model = PPO.load(n)

        dist: np.array = np.array([])
        distZ: np.array = np.array([])
        distY: np.array = np.array([])
        distX: np.array = np.array([])
        times: List[float] = []

        # evaluate for different ts
        if "4D" in n:
            env = gym.make('WindSingleAgent-aviary-v0', aggregate_phy_steps=5, obs=ObservationType('kin'),
                                act=ActionType.RPM, mode=1, total_force=0, radius=0, episode_len=5,
                                drone_model=DroneModel("hb"))
        else:
            env = gym.make('WindSingleAgent-aviary-v0', aggregate_phy_steps=5, obs=ObservationType('kin'),
                           act=ActionType.ONE_D_RPM, mode=1, total_force=0, radius=0, episode_len=5,
                           drone_model=DroneModel("hb"))
        obs = env.reset()

        for j in range(5 * int(env.SIM_FREQ / env.AGGR_PHY_STEPS)):
            action, _states = model.predict(obs, deterministic=True)
            times.append(j / 240 * 5)
            obs, reward, done, info = env.step(action)
            pose: np.array = env.getPose()
            goal: np.array = env.goal
            dist = np.append(dist, env.getDist())
            distX = np.append(distX, abs(goal[0] - pose[0]))
            distY = np.append(distY, abs(goal[1] - pose[1]))
            distZ = np.append(distZ, abs(goal[2] - pose[2]))

            if done:
                obs = env.reset()
                break

        env.reset()

        plt.plot(times, dist, label=n.replace("../results/", ""))

        plt.xlabel('Time [s]')
        plt.ylabel('Complete Distance [m]')

    thresh: List[float] = []
    for t in times:
        thresh.append(0.05)

    plt.fill_between(times, thresh, color='red', alpha=.25)

    plt.legend()
    plt.show()



