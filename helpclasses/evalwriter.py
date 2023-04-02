import time
from typing import List
import pandas as pd
import openpyxl
import numpy as np
import gym
from numpy import mean

from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from stable_baselines3 import PPO
from stable_baselines3 import DDPG
from stable_baselines3.common.evaluation import evaluate_policy
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.enums import DroneModel
from stable_baselines3.common.policies import ActorCriticPolicy as a2cppoMlpPolicy
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.envs.WindSingleAgentAviary import WindSingleAgentAviary
from robust_drone_pathfollowing.helpclasses.printout import *
from robust_drone_pathfollowing.helpclasses.pathplotter import PathPlotter
import torch
import matplotlib.pyplot as plt
from scipy.stats import sem

class EvalWriter:
    """Writer class that writes the evaluated performance data to an xlsx file."""

    def __init__(self, name: str, eval_steps: int, path: str, env: WindSingleAgentAviary, episode_len: int, threshold: float):
        """Initialization of a EvalWriter class.
            Evaluates sucess rate, the sucess time rate, the average distance half way through the simulation,
            the average reward and plots path / goal if it is a signel evaluation.

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

        if eval_steps == 1:
            self.pathPlotter = PathPlotter(env.goal)
        self.mean_reward: float = None
        self.std_reward: float = None
        self.goal: List[float] = None
        self.last_dist: float = None
        self.succeeded: bool = False
        self.total_steps: int = eval_steps
        self.succeded_steps: int = 0
        self.total_time: float = 0
        self.path: str = path
        self.env: WindSingleAgentAviary = env
        self.threshold: float = threshold
        self.episode_len: int = episode_len
        self.name: str = name
        self.distances: List[float] = []
        self.times: List[float] = []
        self.STEP: int = -1
        self.halfdist: List[float] = []

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
        if self.succeeded:
            self.succeded_steps += 1

        self.last_dist = self.env.getDist()
        self.goal = self.env.goal
        self.succeeded = False
        self.STEP += 1

    def update(self):
        """Updates the writer.
            Each time the update method is used, it is checked wether it is a single evaluation with a pathplot.
            If the episode is half way through then the current distance is added to the halfdist List.
            Then the sucess time is updated."""

        dist = self.env.getDist()
        simtime = self.env.getSimTime() + (self.STEP * self.episode_len)
        self.times.append(simtime)
        self.distances.append(dist)

        ### Safe pos if only one episode is evaluated ##################################################################
        if self.total_steps == 1:
            self.pathPlotter.addPose(self.env.getPose())

        ### Check if half way through ##################################################################################
        if (simtime - (self.STEP * self.episode_len)) >= self.episode_len / 2:
            if (self.times[len(self.times) - 2] - (self.STEP * self.episode_len) < self.episode_len / 2):
                self.halfdist.append(dist)

        ### Check if success ###########################################################################################
        if not self.succeeded and dist < self.threshold:
            self.succeeded = True
        if self.last_dist > self.threshold > dist:
            pass
        if self.last_dist < self.threshold and dist < self.threshold:
            self.total_time += (simtime - self.times[len(self.times) - 2])
        if self.last_dist < self.threshold < dist:
            self.total_time += (simtime - self.times[len(self.times) - 2])

        self.last_dist = dist

    def evaluateModel(self, model):
        """Method that evaluates a given Model.

        Parameters
        ----------
        model:
            The model that should be evaluated.

        """

        self.mean_reward, self.std_reward = evaluate_policy(model, self.env, n_eval_episodes=self.total_steps)
        obs = self.env.reset()
        start = time.time()
        for j in range(0, self.total_steps):
            for i in range(self.episode_len * int(self.env.SIM_FREQ / self.env.AGGR_PHY_STEPS)):
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                sync(np.floor(i * self.env.AGGR_PHY_STEPS), start, self.env.TIMESTEP)
                self.update()

            if j != self.total_steps:
                obs = self.env.reset()
                self.housekeeping(self.env)

        self.close()
        self.env.close()

    def close(self):
        """Method that closes the writer and writes."""

        if self.succeeded:
            self.succeded_steps += 1
        self.write()

    def write(self):
        """Prints out the data, writes it to a xlsx file and plots it."""

        ### Print ######################################################################################################
        rate: str = str((self.succeded_steps / self.total_steps) * 100) + '%'
        time_rate: str = str(self.total_time / (self.total_steps * self.episode_len) * 100) + '%'
        time: str = str(self.total_steps * self.episode_len) + "s"
        tot_time: str = str(self.total_time) + "s"
        if self.total_steps > 1:
            dist_avg: str = "(" + str(mean(self.halfdist)) + " +- " + str(sem(self.halfdist)) + ")m"
        else:
            dist_avg: str = str(mean(self.halfdist))
        msg: str = "Total steps: {0}\nSucceded steps: {1}\n----------------\nTime: {2}\nTime in goal: {3}\n----------------\nRate: {4}\nTime Rate: {5}\nDistance(T/2): {6}\nMean reward: {7} +- {8}".format(
            str(self.total_steps), str(
                self.succeded_steps), str(time), str(
                tot_time), rate, time_rate, dist_avg, str(self.mean_reward), str(self.std_reward))
        debug(bcolors.OKBLUE, msg)

        ### Write to File ##############################################################################################
        df = pd.DataFrame([str(self.total_steps), str(self.succeded_steps), rate, "--------", time, tot_time, time_rate , dist_avg],
                        index=['Total steps', 'Succeded steps', 'Success rate', '--------', 'Time', 'Time in goal', 'Success time rate', 'Distance(T/2)'],
                        columns=['Eval Data'])
        df2 = pd.DataFrame(self.distances, self.times)
        with pd.ExcelWriter(self.path) as writer:
            df.to_excel(writer, sheet_name='Data(' + str(self.name) + ')')
            df2.to_excel(writer, sheet_name='Distances')

        ### Plot #######################################################################################################
        if self.total_steps == 1:
            self.pathPlotter.show()
        plt.plot(self.times, self.distances)
        plt.show()


if __name__ == "__main__":
    eval_env = gym.make('WindSingleAgent-aviary-v0', aggregate_phy_steps=5, obs=ObservationType('kin'), act=ActionType('rpm'), mode=0, total_force=0,
                        upper_bound=1, debug=False)
    episode_len = 6
    eval_steps = 1
    eval: EvalWriter = EvalWriter(name='TestWriter', eval_steps=eval_steps, path='test.xlsx', env=eval_env, episode_len=episode_len, threshold=0.05)
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
    eval.evaluateModel(model)


