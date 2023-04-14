import os
import argparse
from helpclasses.printout import *
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
import gym
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3 import DDPG
from stable_baselines3.common.policies import ActorCriticPolicy as a2cppoMlpPolicy
from stable_baselines3.td3 import MlpPolicy as td3ddpgMlpPolicy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnRewardThreshold

DEFAULT_ALGO = 'ppo'
DEFAULT_OBS = ObservationType('kin')
DEFAULT_ACT = ActionType('rpm')
DEFAULT_CPU = 1
DEFAULT_STEPS = 100000
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_MODE = 0
DEFAULT_LOAD = False
DEFAULT_LOADFILE = 'results'
EPISODE_REWARD_THRESHOLD = 0
DEFAULT_ENV = 'WindSingleAgent-aviary-v0'
DEFAULT_FORCE = 0
DEFAULT_BOUND = 1
DEFAULT_EPISODE_LEN = 5


def run(env: str = DEFAULT_ENV,
        algo: str = DEFAULT_ALGO,
        obs: ObservationType = DEFAULT_OBS,
        act: ActionType = DEFAULT_ACT,
        cpu: int = DEFAULT_CPU,
        steps: int = DEFAULT_STEPS,
        folder: str = DEFAULT_OUTPUT_FOLDER,
        mode: int = DEFAULT_MODE,
        total_force: float = DEFAULT_FORCE,
        upper_bound: float = DEFAULT_BOUND,
        load: bool = DEFAULT_LOAD,
        load_file: str = DEFAULT_LOADFILE,
        debug_env: bool = False,
        episode_len: int = DEFAULT_EPISODE_LEN
        ):

    # Create training environment ######################################################################################
    train_env = gym.make(env, aggregate_phy_steps=5, obs=obs, act=act, mode=mode,
                         total_force=total_force, upper_bound=upper_bound, debug=debug_env, episode_len=episode_len)

    onpolicy_kwargs = dict(activation_fn=torch.nn.ReLU,
                           net_arch=[512, 512, dict(vf=[256, 128], pi=[256, 128])]
                           )

    offpolicy_kwargs = dict(activation_fn=torch.nn.ReLU,
                            net_arch=[512, 512, 256, 128]
                            )

    filename = folder

    # Decide path ######################################################################################################
    if os.path.isfile(load_file+'/success_model.zip'):
        path = load_file+'/success_model.zip'
    elif os.path.isfile(load_file+'/best_model.zip'):
        path = load_file+'/best_model.zip'
    else:
        print("[ERROR]: no model under the specified path", load_file)

    # Create Model because load is false ###############################################################################
    if not load:
        if algo == 'ppo':
            model = PPO(a2cppoMlpPolicy,
                        train_env,
                        policy_kwargs=onpolicy_kwargs,
                        tensorboard_log=filename+'/tb/',
                        verbose=1
                        )
        if algo == 'ddpg':
            model = DDPG(td3ddpgMlpPolicy,
                         train_env,
                         policy_kwargs=offpolicy_kwargs,
                         tensorboard_log=filename + '/tb/',
                         verbose=1
                         )
    # Load Model because load is true ##################################################################################
    else:
        if algo == 'ppo':
            model = PPO.load(path, train_env)
        if algo == 'ddpg':
            model = DDPG.load(path, train_env)

    # Create eval Environment ##########################################################################################
    eval_env = gym.make(env, aggregate_phy_steps=5, obs=obs, act=act, mode=mode,
                        total_force=total_force, upper_bound=upper_bound, debug=debug_env, episode_len=episode_len)

    # Train the model ##################################################################################################
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=EPISODE_REWARD_THRESHOLD,
                                                     verbose=1
                                                     )
    eval_callback = EvalCallback(eval_env,
                                 callback_on_new_best=callback_on_best,
                                 verbose=1,
                                 best_model_save_path=filename+'/',
                                 log_path=filename+'/',
                                 eval_freq=int(2000/cpu),
                                 deterministic=True,
                                 render=False
                                 )

    model.learn(total_timesteps=steps, callback=eval_callback, log_interval=100)

    # Save the model ###################################################################################################
    model.save(folder + '/success_model.zip')
    print(filename)

    # Print training progression #######################################################################################
    with np.load(filename+'/evaluations.npz') as data:
        for j in range(data['timesteps'].shape[0]):
            print(str(data['timesteps'][j])+","+str(data['results'][j][0]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script that allows to train your RL Model")
    parser.add_argument('--algo', default=DEFAULT_ALGO, type=str,
                        help='The Algorithm that trains the agent(PPO(default), DDPG)', metavar='')
    parser.add_argument('--obs', default=DEFAULT_OBS, type=ObservationType,
                        help='The chosen ObservationType (default: KIN)', metavar='')
    parser.add_argument('--act', default=DEFAULT_ACT, type=ActionType,
                        help='The chosen ActionType (default: RPM)', metavar='')
    parser.add_argument('--cpu', default=DEFAULT_CPU, type=int,
                        help='Amount of parallel training environments (default: 1)', metavar='')
    parser.add_argument('--steps', default=DEFAULT_STEPS, type=int,
                        help='Amount of training time steps(default: 100000)', metavar='')
    parser.add_argument('--folder', default=DEFAULT_OUTPUT_FOLDER, type=str,
                        help='Output folder (default: results)', metavar='')
    parser.add_argument('--mode', default=DEFAULT_MODE, type=int,
                        help='The mode of the training environment(default: 0)', metavar='')
    parser.add_argument('--env', default=DEFAULT_ENV, type=str,
                        help='Name of the environment(default:WindSingleAgent-aviary-v0)', metavar='')
    parser.add_argument('--load', default=DEFAULT_LOAD, type=bool,
                        help='Load an existing model(default: False)', metavar='')
    parser.add_argument('--load_file', default=DEFAULT_LOADFILE, type=str,
                        help='The experiment folder where the loaded model can be found', metavar='')
    parser.add_argument('--total_force', default=DEFAULT_FORCE, type=float,
                        help='The max force in the simulated Wind field(default: 0)', metavar='')
    parser.add_argument('--upper_bound', default=DEFAULT_BOUND, type=float,
                        help='The upper bound of the area where the goal is simulated(default: 1)', metavar='')
    parser.add_argument('--debug_env', default=False, type=bool,
                        help='Parameter to the Environment that enables most of the Debug messages(default: False)',
                        metavar='')
    parser.add_argument('--episode_len', default=DEFAULT_EPISODE_LEN, type=int,
                        help='The episode length(default: 5)', metavar='')

    ARGS = parser.parse_args()

    welcome(ARGS)
    run(**vars(ARGS))
