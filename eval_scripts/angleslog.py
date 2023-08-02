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


if __name__ == "__main__":
    plt.rc('font', size=18)
    fig, ax = plt.subplots(3, 1)
    names: List[str] = ["../results/SAC4D24_1.zip", "../results/SAC4D48_1.zip",
                        "../results/SAC1D_1.zip", "../results/PPO1D_1.zip", "../results/PPO4D_1.zip"]

    for i, n in enumerate(names):
        debug(bcolors.OKBLUE, '[INFO] Model: ' + n)
        # load model
        if 'SAC' in n:
            model = SAC.load(n)
        else:
            model = PPO.load(n)


        roll: np.array = np.array([])
        pitch: np.array = np.array([])
        yaw: np.array = np.array([])
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

            roll = np.append(roll, env.rpy[0, 0])
            pitch = np.append(pitch, env.rpy[0, 1])
            yaw = np.append(yaw, env.rpy[0, 2])

            if done:
                obs = env.reset()
                break

        env.reset()

        color: str = defineColor(n)

        ax[0].plot(times, roll, c=color, label=n.replace("../results/", ""))
        ax[0].set_xlabel('Time [s]')
        ax[0].set_ylabel('Roll Angle [rad]')

        ax[1].plot(times, pitch, c=color, label=n.replace("../results/", ""))
        ax[1].set_xlabel('Time [s]')
        ax[1].set_ylabel('Pitch Angle [rad]')

        ax[2].plot(times, yaw, c=color, label=n.replace("../results/", ""))
        ax[2].set_xlabel('Time [s]')
        ax[2].set_ylabel('Yaw Angle [rad]')

    ax[0].legend(loc="lower left")
    ax[1].legend(loc="lower left")
    ax[2].legend(loc="lower left")
    plt.show()



