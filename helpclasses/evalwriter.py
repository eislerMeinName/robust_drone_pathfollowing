import time
from typing import List
import pandas as pd
import openpyxl
import numpy as np
import gym
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from stable_baselines3 import PPO
from stable_baselines3 import DDPG
from stable_baselines3.common.evaluation import evaluate_policy
from gym_pybullet_drones.utils.Logger import Logger
from stable_baselines3.common.policies import ActorCriticPolicy as a2cppoMlpPolicy
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.envs.WindSingleAgentAviary import WindSingleAgentAviary
from robust_drone_pathfollowing.helpclasses.printout import *
import torch

class EvalWriter:
    """Writer class that writes the evaluated performance data to an xlsx file."""

    def __init__(self, name: str, eval_steps: int, path: str, env: WindSingleAgentAviary, episode_len: int, threshold: float):
        """Initialization of a EvalWriter class.

        Parameters
        ----------
        name: str
            The name of the Agent that is evaluated.
        eval_steps: int
            The amount of evaluation steps.
        path: str
            Path to / Name of the xlsx file.
        env: WindSingleAgentAviary
            The environment on which the agent is evaluated.
        episode_len: int
            The amount of seconds each episode runs.
        threshold: float
            The threshold that defines when an agent reached the goal

        """
        self.total_steps: int = eval_steps
        self.succeded_steps: int = 0
        self.total_time: float = 0
        self.succeded: bool = False
        self.path = path
        self.env = env
        self.threshold = threshold
        self.episode_len = episode_len
        self.name = name
        self.distances = []

        self.housekeeping(env)

    def housekeeping(self, env: WindSingleAgentAviary):
        """Housekeeping function.

        Allocation and zero-ing of the variables and environment parameters/objects.

        Parameters
        ----------
        env: WindSingleAgentAviary
            The new environment that has been reset.

        """

        self.env = env
        if self.succeded:
            self.succeded_steps += 1
        self.start_time = 0
        self.last_dist = self.env.getDist()
        self.goal = self.env.goal

    def update(self):
        """Updates the writer."""

        dist = self.env.getDist()
        #print('Current distance: ', dist)
        #print('Goal: ', self.goal)
        self.distances.append(dist)
        if not self.succeded and dist < self.threshold:
            self.succeded = True
        if self.last_dist > self.threshold and dist < self.threshold:
            self.start_time = time.time()
        if self.last_dist < self.threshold and dist > self.threshold:
            self.total_time += (time.time() - self.start_time)

    def write(self):
        """Prints out the data and writes it to a xlsx file."""

        ### Print ######################################################################################################
        rate: str = str((self.succeded_steps / self.total_steps) * 100) + '%'
        time_rate: str = str(self.total_time / (self.total_steps * self.episode_len)) + '%'
        time: str = str(self.total_steps * self.episode_len) + "s"
        tot_time: str = str(self.total_time) + "s"
        #debug(bcolors.BOLD, str(self.distances))
        debug(bcolors.OKCYAN, "Total steps: " + str(self.total_steps))
        debug(bcolors.OKCYAN, "Succeded steps: " + str(self.succeded_steps))
        debug(bcolors.OKGREEN, "----------------")
        debug(bcolors.OKCYAN, "Time: " + str(time))
        debug(bcolors.OKCYAN, "Time in goal: " + str(tot_time))
        debug(bcolors.OKGREEN, "----------------")

        debug(bcolors.UNDERLINE, "Rate: " + rate + "\nTime Rate: " + time_rate)

        ### Write to File ##############################################################################################
        df = pd.DataFrame([self.total_steps, self.succeded_steps, rate, "--------", time, tot_time, time_rate],
                        index=['Total steps', 'Succeded steps', 'Success rate', '--------', 'Time', 'Time in goal', 'Success time rate'],
                        columns=['Eval Data'])
        df2 = pd.DataFrame(self.distances)
        with pd.ExcelWriter(self.path) as writer:
            df.to_excel(writer, sheet_name='Data(' + str(self.name) + ')')
            df2.to_excel(writer, sheet_name='Distances')

        #TODO: plot the distances


if __name__ == "__main__":
    eval_env = gym.make('WindSingleAgent-aviary-v0', aggregate_phy_steps=5, obs=ObservationType('kin'), act=ActionType('rpm'), mode=0, total_force=0,
                        upper_bound=1, debug=False)
    episode_len = 6
    eval_steps = 6
    eval: EvalWriter = EvalWriter(name='TestWriter', eval_steps=eval_steps, path='test.xlsx', env= eval_env, episode_len=episode_len, threshold=0.05)
    obs = eval_env.reset()
    start = time.time()
    onpolicy_kwargs = dict(activation_fn=torch.nn.ReLU,
                           net_arch=[512, 512, dict(vf=[256, 128], pi=[256, 128])]
                           )
    model = model = PPO(a2cppoMlpPolicy,
                        eval_env,
                        policy_kwargs=onpolicy_kwargs,
                        tensorboard_log='results/tb/',
                        verbose=1
                        )
    for j in range(1, eval_steps):
        for i in range(episode_len * int(eval_env.SIM_FREQ / eval_env.AGGR_PHY_STEPS)):  # Up to 6''
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            #eval_env.render()
            sync(np.floor(i * eval_env.AGGR_PHY_STEPS), start, eval_env.TIMESTEP)
            eval.update()

        if j != eval_steps:
            obs = eval_env.reset()
            eval.housekeeping(eval_env)
        else:
            eval_env.close()

    eval.write()



