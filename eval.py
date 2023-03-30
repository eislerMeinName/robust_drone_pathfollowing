import os
import time
import sys
from datetime import datetime
import argparse
import re
import numpy as np
import gym
import torch
from stable_baselines3 import PPO
from stable_baselines3 import DDPG
from stable_baselines3.common.evaluation import evaluate_policy
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.utils.utils import sync, str2bool
from helpclasses.printout import *
from helpclasses.evalwriter import EvalWriter

DEFAULT_ALGO = 'ppo'
DEFAULT_OBS = ObservationType('kin')
DEFAULT_ACT = ActionType('rpm')
DEFAULT_EPISODES = 100
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_MODE = 0
DEFAULT_LOAD = False
DEFAULT_LOADFILE = 'results'
EPISODE_REWARD_THRESHOLD = 0
DEFAULT_ENV = 'WindSingleAgent-aviary-v0'
DEFAULT_FORCE = 0
DEFAULT_BOUND = 1
DEFAULT_GUI_TIME = 10


def run(env: str = DEFAULT_ENV,
        algo: str = DEFAULT_ALGO,
        obs: ObservationType = DEFAULT_OBS,
        act: ActionType = DEFAULT_ACT,
        folder: str = DEFAULT_OUTPUT_FOLDER,
        mode: int = DEFAULT_MODE,
        total_force: float = DEFAULT_FORCE,
        upper_bound: float = DEFAULT_BOUND,
        load_file: str = DEFAULT_LOADFILE,
        debug_env: bool = False,
        episodes: int = DEFAULT_EPISODES,
        gui: bool = True,
        gui_time: int = DEFAULT_GUI_TIME
        ):
    obs_save = obs
    ### Create evaluation environment ##################################################################################
    eval_env = gym.make(env, aggregate_phy_steps=5, obs=obs, act=act, mode=mode,total_force=total_force, upper_bound=upper_bound, debug=debug_env)

    ### Decide path ####################################################################################################
    if os.path.isfile(load_file+'/success_model.zip'):
        path = load_file+'/success_model.zip'
    elif os.path.isfile(load_file+'/best_model.zip'):
        path = load_file+'/best_model.zip'
    else:
        print("[ERROR]: no model under the specified path", load_file)

    ### Load model #####################################################################################################
    if algo == 'ppo':
        model = PPO.load(path)
    if algo == 'ddpg':
        model = DDPG.load(path)

    ### Eval the reward ################################################################################################
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=episodes)
    msg = "\n\n\nMean reward " +str(mean_reward) + " +- " + str(std_reward) + "\n\n"
    debug(bcolors.OKGREEN, msg)

    ### Evaluate and write #############################################################################################
    eval: EvalWriter = EvalWriter(name='TestWriter', eval_steps=episodes, path='test.xlsx', env=eval_env,
                                  episode_len=gui_time, threshold=0.05)
    obs = eval_env.reset()
    start = time.time()
    for j in range(1, episodes):
        for i in range(gui_time * int(eval_env.SIM_FREQ / eval_env.AGGR_PHY_STEPS)):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)
            # eval_env.render()
            sync(np.floor(i * eval_env.AGGR_PHY_STEPS), start, eval_env.TIMESTEP)
            eval.update()

        if j != episodes:
            obs = eval_env.reset()
            eval.housekeeping(eval_env)
        else:
            eval_env.close()

    eval.write()
    obs = obs_save

    ### Create test environment and logger #############################################################################
    test_env = gym.make(env, gui=True, record=False, aggregate_phy_steps=5, obs=obs, act=act, mode=mode, total_force=total_force, upper_bound=upper_bound, debug=debug_env)
    logger = Logger(logging_freq_hz=int(test_env.SIM_FREQ/test_env.AGGR_PHY_STEPS),
                    num_drones=1,
                    output_folder=folder
                    )

    ### Start a visual simulation ######################################################################################
    if gui:
        obs = test_env.reset()
        start = time.time()
        for i in range(gui_time * int(test_env.SIM_FREQ/test_env.AGGR_PHY_STEPS)): # Up to 6''
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = test_env.step(action)
            test_env.render()
            logger.log(drone=0,
                timestamp=i/test_env.SIM_FREQ,
                state= test_env.getState(0),#state= np.hstack([obs[0:3], np.zeros(4), obs[3:15],  np.resize(action, (4))]),
                control=np.zeros(12)
                )
            sync(np.floor(i*test_env.AGGR_PHY_STEPS), start, test_env.TIMESTEP)
            # if done: obs = test_env.reset() # OPTIONAL EPISODE HALT
        test_env.close()
        logger.save_as_csv("sa") # Optional CSV save
        logger.plot()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script that allows to evaluate the RL Model")
    parser.add_argument('--algo', default=DEFAULT_ALGO, type=str, help='The Algorithm that trains the agent(PPO(default), DDPG)', metavar='')
    parser.add_argument('--obs', default=DEFAULT_OBS, type=ObservationType, help='The choosen ObservationType (default: KIN)', metavar='')
    parser.add_argument('--act', default=DEFAULT_ACT, type=ActionType, help='The choosen ActionType (default: RPM)', metavar='')
    parser.add_argument('--folder', default=DEFAULT_OUTPUT_FOLDER, type=str, help='Output folder (default: results)', metavar='')
    parser.add_argument('--mode', default=DEFAULT_MODE, type=int, help='The mode of the training environment(default: 0)', metavar='')
    parser.add_argument('--episodes', default=DEFAULT_EPISODES, type=int, help='The number of evaluation steps(default: 100)', metavar='')
    parser.add_argument('--env', default=DEFAULT_ENV, type=str, help='Name of the environment(default:WindSingleAgent-aviary-v0)', metavar='')
    parser.add_argument('--load_file', default=DEFAULT_LOADFILE, type=str, help='The experiment folder where the loaded model can be found', metavar='')
    parser.add_argument('--total_force', default=DEFAULT_FORCE, type=float, help='The max force in the simulated Wind field(default: 0)', metavar='')
    parser.add_argument('--upper_bound', default=DEFAULT_BOUND, type=float, help='The upper bound of the area where the goal is simulated(default: 1)', metavar='')
    parser.add_argument('--debug_env', default=False, type=bool, help='Parameter to the Environment that enables most of the Debug messages(default: False)', metavar='')
    parser.add_argument('--gui', default=True, type=bool, help='Enables/ Disables the gui replay(default: True)', metavar='')
    parser.add_argument('--gui_time', default=DEFAULT_GUI_TIME, type=int, help='The simulation length(default: 10)')

    ARGS = parser.parse_args()

    welcome(ARGS)
    run(**vars(ARGS))