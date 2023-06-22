import argparse
from helpclasses.printout import *
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
import gym
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy as a2cppoMlpPolicy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.cmd_util import make_vec_env
from gym_pybullet_drones.envs.WindSingleAgentAviary import WindSingleAgentAviary
from gym_pybullet_drones.utils.enums import DroneModel
from errors.ParsingError import ParsingError

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
DEFAULT_BOUND = 1
DEFAULT_EPISODE_LEN = 5
DEFAULT_INIT = None


def run(cpu: int = DEFAULT_CPU,
        steps: int = DEFAULT_STEPS,
        mode: int = DEFAULT_MODE,
        total_force: float = DEFAULT_FORCE,
        upper_bound: float = DEFAULT_BOUND,
        load: bool = DEFAULT_LOAD,
        debug_env: bool = False,
        episode_len: int = DEFAULT_EPISODE_LEN,
        drone: DroneModel = DEFAULT_DRONE,
        init: np.array = DEFAULT_INIT,
        gui: bool = False,
        act: ActionType = DEFAULT_ACT,
        name: str = DEFAULT_NAME
        ):

    # Create training environment ######################################################################################
    sa_env_kwargs: dict = dict(aggregate_phy_steps=5, obs=ObservationType('kin'), act=act, mode=mode,
                               total_force=total_force, upper_bound=upper_bound, drone_model=drone, gui=gui,
                               debug=debug_env, episode_len=episode_len)
    if not (init is None):
        sa_env_kwargs['initial_xyzs'] = np.array(init[0:3]).reshape(1, 3)
        sa_env_kwargs['initial_rpys'] = np.array(init[3:6]).reshape(1, 3)

    train_env = make_vec_env('WindSingleAgent-aviary-v0', env_kwargs=sa_env_kwargs, n_envs=cpu, seed=0)

    onpolicy_kwargs: dict = dict(activation_fn=torch.nn.ReLU,
                                 net_arch=[dict(vf=[256, 256, 128], pi=[256, 256, 64])])

    if load == DEFAULT_LOAD:
        model = PPO(a2cppoMlpPolicy,
                    train_env,
                    policy_kwargs=onpolicy_kwargs,
                    tensorboard_log='results/tb/',
                    verbose=1
                    )
    else:
        model = PPO.load(load, train_env, tensorboard_log='results/tb/')

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

    model.learn(total_timesteps=steps, callback=eval_callback, log_interval=100)

    # Save the model ###################################################################################################
    model.save(name)

def check(cpu: int = DEFAULT_CPU,
        steps: int = DEFAULT_STEPS,
        mode: int = DEFAULT_MODE,
        total_force: float = DEFAULT_FORCE,
        upper_bound: float = DEFAULT_BOUND,
        load: bool = DEFAULT_LOAD,
        debug_env: bool = False,
        episode_len: int = DEFAULT_EPISODE_LEN,
        drone: DroneModel = DEFAULT_DRONE,
        init: np.array = DEFAULT_INIT,
        gui: bool = False,
        act: ActionType = DEFAULT_ACT,
        name: str = DEFAULT_NAME):
    if mode > 4:
        raise ParsingError(['Mode'], [mode], 'The specified mode is not defined in the environment.')

    if not (init is None) and not (len(init)) == 6:
        raise ParsingError(['Init'], [mode], 'The specified length is not 6.')


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
    parser.add_argument('--upper_bound', default=DEFAULT_BOUND, type=float,
                        help='The upper bound of the area where the goal is simulated(default: 1)', metavar='')
    parser.add_argument('--debug_env', action='store_const',
                        help='Parameter to the Environment that enables most of the Debug messages(default: False)',
                        const=True, default=False)
    parser.add_argument('--episode_len', default=DEFAULT_EPISODE_LEN, type=int,
                        help='The episode length(default: 5)', metavar='')
    parser.add_argument('--init', default=DEFAULT_INIT, nargs='+', type=float,
                        help='The init values in form [x,y,z,r,p,y](default: None)', metavar='')
    parser.add_argument('--gui', action='store_const', help='Enable GUI during training process(default: False)',
                        const=True, default=False)
    parser.add_argument('--act', default=DEFAULT_ACT, type=ActionType,
                        help='The action type of the environment (default: rpm)', metavar='')
    parser.add_argument('--name', default=DEFAULT_NAME, type=str,
                        help='The name of the model after training (default: results/succcess_model.zip', metavar='')

    ARGS = parser.parse_args()

    welcome(ARGS)
    check(**vars(ARGS))
    run(**vars(ARGS))
