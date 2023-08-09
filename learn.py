import argparse
from helpclasses.printout import *
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
import gym
import torch
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.policies import ActorCriticPolicy as a2cppoMlpPolicy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.cmd_util import make_vec_env
from gym_pybullet_drones.envs.WindSingleAgentAviary import WindSingleAgentAviary
from gym_pybullet_drones.utils.enums import DroneModel
from errors.ParsingError import ParsingError
from typing import List
import os
from stable_baselines3.sac.policies import SACPolicy as sacMlpPolicy

DEFAULT_DRONE = DroneModel.CF2X
DEFAULT_ACT = ActionType('rpm')
DEFAULT_CPU = 1
DEFAULT_STEPS = 1000000
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_NAME = 'results/success_model.zip'
DEFAULT_MODE = 0
DEFAULT_LOAD = ''
EPISODE_REWARD_THRESHOLD = 0
DEFAULT_FORCE = 0
DEFAULT_RADIUS = 0.0
DEFAULT_EPISODE_LEN = 5
DEFAULT_INIT = None
DEFAULT_ALGO = 'ppo'
DEFAULT_DELTA = 0.2


def run(cpu: int = DEFAULT_CPU,
        steps: int = DEFAULT_STEPS,
        mode: int = DEFAULT_MODE,
        total_force: float = DEFAULT_FORCE,
        radius: float = DEFAULT_RADIUS,
        load: bool = DEFAULT_LOAD,
        debug_env: bool = False,
        episode_len: int = DEFAULT_EPISODE_LEN,
        drone: DroneModel = DEFAULT_DRONE,
        act: ActionType = DEFAULT_ACT,
        name: str = DEFAULT_NAME,
        curriculum: bool = False,
        algo: str = DEFAULT_ALGO,
        delta: float = DEFAULT_DELTA
        ):

    # Create training environment ######################################################################################
    sa_env_kwargs: dict = dict(aggregate_phy_steps=5, obs=ObservationType('kin'), act=act, mode=mode,
                               total_force=total_force, radius=radius, drone_model=drone, gui=False,
                               debug=debug_env, episode_len=episode_len)

    train_env = make_vec_env('WindSingleAgent-aviary-v0', env_kwargs=sa_env_kwargs, n_envs=cpu, seed=0)

    onpolicy_kwargs: dict = dict(activation_fn=torch.nn.ReLU,
                                 net_arch=[dict(vf=[256, 256, 128], pi=[256, 256, 64])])
    offpolicy_kwargs: dict = dict(activation_fn=torch.nn.ReLU,
                                  net_arch=[256, 256, 128])

    if load == DEFAULT_LOAD:
        if algo == 'ppo':
            model = PPO(a2cppoMlpPolicy,
                        train_env,
                        policy_kwargs=onpolicy_kwargs,
                        tensorboard_log='results/tb/',
                        verbose=1
                        )
        elif algo == 'sac':
            model = SAC(sacMlpPolicy,
                        train_env,
                        policy_kwargs=offpolicy_kwargs,
                        tensorboard_log='results/tb/',
                        verbose=1
                        )
    else:
        if algo == 'ppo':
            model = PPO.load(load, train_env, tensorboard_log='results/tb/')
        elif algo == 'sac':
            model = SAC.load(load, train_env, tensorboard_log='results/tb/')

    # Create eval Environment ##########################################################################################
    eval_env = make_vec_env(WindSingleAgentAviary, env_kwargs=sa_env_kwargs, n_envs=cpu, seed=0)

    # Train the model ##################################################################################################
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=EPISODE_REWARD_THRESHOLD,
                                                     verbose=1
                                                     )
    eval_callback = EvalCallback(eval_env,
                                 callback_on_new_best=callback_on_best,
                                 verbose=1,
                                 best_model_save_path='results/',
                                 log_path='results/',
                                 eval_freq=int(2000/cpu),
                                 deterministic=True,
                                 render=False
                                 )
    if not curriculum:
        model.learn(total_timesteps=steps, callback=eval_callback, log_interval=100)

    else:
        curri_learn(total_steps=steps, kwargs=sa_env_kwargs, model=model, cpu=cpu, name=name, delta=delta)

    # Save the model ###################################################################################################
    model.save(name)


def curri_learn(total_steps: int, kwargs: dict,
                model, cpu: int, name: str,
                delta: float):

    radius: float = kwargs["radius"]
    curr_rad: float = 0.0

    while curr_rad <= radius:
        kwargs["radius"] = curr_rad
        train_env = make_vec_env('WindSingleAgent-aviary-v0', env_kwargs=kwargs, n_envs=cpu, seed=0)
        model.env = train_env
        eval_env = make_vec_env(WindSingleAgentAviary, env_kwargs=kwargs, n_envs=cpu, seed=0)

        callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=-8,
                                                         verbose=1
                                                         )
        eval_callback = EvalCallback(eval_env,
                                     callback_on_new_best=callback_on_best,
                                     verbose=1,
                                     best_model_save_path='results/',
                                     log_path='results/',
                                     eval_freq=int(2000 / cpu),
                                     deterministic=True,
                                     render=False
                                     )

        # Train the model ##############################################################################################
        save_name: str = name.replace(".zip", '') + 'curri_r' + str(curr_rad) + '.zip'
        debug(bcolors.OKBLUE, "[INFO]: " + model.tensorboard_log +
              ", radius: " + str(curr_rad) +
              "[m], delta: " + str(delta))
        with open('curriculum.log', 'a') as f:
            f.write("\n[INFO]: " + model.tensorboard_log +
                    ", radius: " + str(curr_rad) +
                    ", steps: " + str(int(1/((1 / delta)+1) * total_steps)) +
                    "[m], delta: " + str(delta) +
                    "\n[INFO]: name:" + save_name)
        model.learn(total_timesteps=int(1/((1 / delta)+1)*total_steps), callback=eval_callback, log_interval=100)
        model.save(save_name)

        # Rename Best Model ############################################################################################
        os.rename('results/best_model.zip', 'results/best_modelcurri_r' + str(curr_rad) + '.zip')

        curr_rad += delta * radius


def check(cpu: int = DEFAULT_CPU,
        steps: int = DEFAULT_STEPS,
        mode: int = DEFAULT_MODE,
        total_force: float = DEFAULT_FORCE,
        radius: float = DEFAULT_RADIUS,
        load: bool = DEFAULT_LOAD,
        debug_env: bool = False,
        episode_len: int = DEFAULT_EPISODE_LEN,
        drone: DroneModel = DEFAULT_DRONE,
        act: ActionType = DEFAULT_ACT,
        name: str = DEFAULT_NAME,
        curriculum: bool = False,
        algo: str = DEFAULT_ALGO,
        delta: float = DEFAULT_DELTA):
    if mode > 4:
        raise ParsingError(['Mode'], [mode], 'The specified mode is not defined in the environment.')
    if curriculum and steps < 1e7:
        raise ParsingError(['Curriculum', 'Steps'], [curriculum, steps],
                           'If you use curriculum learning, increase amount of simulation steps.')
    if curriculum and radius == 0.0:
        raise ParsingError(['Curriculum', 'Radius'], [curriculum, radius],
                           'If you use curriculum learning, radius can not be 0.')
    if algo != 'ppo' and algo != 'sac':
        raise ParsingError(['Algo'], [algo], 'The specified Algorithm can not be used in this script. Check spelling!')
    if delta <= 0 or delta > 1:
        raise ParsingError(['Delta'], [delta], 'The delta is not within the defined range (0,1].')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script that allows to train your RL Model")
    parser.add_argument('--cpu', default=DEFAULT_CPU, type=int,
                        help='Amount of parallel training environments (default: 1)', metavar='')
    parser.add_argument('--drone', default=DEFAULT_DRONE, type=DroneModel)
    parser.add_argument('--steps', default=DEFAULT_STEPS, type=float,
                        help='Amount of training time steps(default: 1000000)', metavar='')
    parser.add_argument('--mode', default=DEFAULT_MODE, type=int,
                        help='The mode of the training environment(default: 0)', metavar='')
    parser.add_argument('--load', default=DEFAULT_LOAD, type=str,
                        help='Load an existing model with the specified name',
                        metavar='')
    parser.add_argument('--total_force', default=DEFAULT_FORCE, type=float,
                        help='The max force in the simulated Wind field(default: 0)', metavar='')
    parser.add_argument('--radius', default=DEFAULT_RADIUS, type=float,
                        help='The radius around [0, 0, 0.5] where the goal can be(default: 0.0)', metavar='')
    parser.add_argument('--debug_env', action='store_const',
                        help='Parameter to the Environment that enables most of the Debug messages(default: False)',
                        const=True, default=False)
    parser.add_argument('--episode_len', default=DEFAULT_EPISODE_LEN, type=int,
                        help='The episode length(default: 5)', metavar='')
    parser.add_argument('--act', default=DEFAULT_ACT, type=ActionType,
                        help='The action type of the environment (default: rpm)', metavar='')
    parser.add_argument('--name', default=DEFAULT_NAME, type=str,
                        help='The name of the model after training (default: results/success_model.zip)', metavar='')
    parser.add_argument('--curriculum', action='store_const', help='Use curriculum learning(default: False)',
                        const=True, default=False)
    parser.add_argument('--algo', default=DEFAULT_ALGO, type=str,
                        help='The algorithm (default: ppo).', metavar='')
    parser.add_argument('--delta', default=DEFAULT_DELTA, type=float,
                        help='The delta parameter for curriculum learning (default: 0.2)', metavar='')

    ARGS = parser.parse_args()

    welcome(ARGS)
    check(**vars(ARGS))
    run(**vars(ARGS))
