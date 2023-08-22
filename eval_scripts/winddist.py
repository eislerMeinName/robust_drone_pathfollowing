import matplotlib.pyplot as plt
from typing import List
from robust_drone_pathfollowing.helpclasses.printout import *
from stable_baselines3 import PPO, SAC
import numpy as np
import gym
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.utils.enums import DroneModel

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

def run(force: float) -> List[float]:
    dist: np.array = np.array([])
    distZ: np.array = np.array([])
    distY: np.array = np.array([])
    distX: np.array = np.array([])
    times: List[float] = []
    env = gym.make('WindSingleAgent-aviary-v0', aggregate_phy_steps=5, obs=ObservationType('kin'),
                   act=ActionType.RPM, mode=3, total_force=force, radius=0, episode_len=5,
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

    color = defineColor(n)
    alpha = 1 - 2 * force
    plt.plot(times, dist, c=color, label=n.replace("../results/", "").replace(".zip", "") + "-" + str(force) + "N",
             alpha=alpha)

    plt.xlabel('Time [s]')
    plt.ylabel('Complete Distance [m]')
    return times


if __name__ == "__main__":
    plt.rc('font', size=20)
    names: List[str] = ["../results/SAC4D24_1.zip", "../results/SAC4D48_1.zip"]

    for i, n in enumerate(names):
        debug(bcolors.OKBLUE, '[INFO] Model: ' + n)
        # load model
        if 'SAC' in n:
            model = SAC.load(n)
        force: List[float] = [0, 0.1, 0.2, 0.3]

        # evaluate for different ts
        for f in force:
            times = run(f)



    thresh: List[float] = []
    for t in times:
        thresh.append(0.05)

    plt.fill_between(times, thresh, color='red', alpha=.25)

    plt.legend()
    plt.show()