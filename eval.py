import os
import time
import argparse
import numpy as np
import gym
from stable_baselines3 import PPO
from stable_baselines3 import DDPG
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.utils.utils import sync
from helpclasses.printout import *
from helpclasses.evalwriter import EvalWriter
from helpclasses.pathplotter import PathPlotter
from gym_pybullet_drones.utils.enums import DroneModel
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.vec_env import VecVideoRecorder


DEFAULT_DRONE = DroneModel.CF2X
DEFAULT_ACT = ActionType('rpm')
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_NAME = 'results/success_model.zip'
DEFAULT_MODE = 0
DEFAULT_FORCE = 0
DEFAULT_RADIUS = 0.2
DEFAULT_EPISODE_LEN = 5
DEFAULT_INIT = None
DEFAULT_EPISODES = 100
DEFAULT_ENV = 'WindSingleAgent-aviary-v0'


def run(drone: DroneModel = DEFAULT_DRONE,
        act: ActionType = DEFAULT_ACT,
        folder: str = DEFAULT_OUTPUT_FOLDER,
        mode: int = DEFAULT_MODE,
        total_force: float = DEFAULT_FORCE,
        radius: float = DEFAULT_RADIUS,
        name: str = DEFAULT_NAME,
        debug_env: bool = False,
        episodes: int = DEFAULT_EPISODES,
        gui: bool = True,
        episode_len: int = DEFAULT_EPISODE_LEN,
        init: np.array = DEFAULT_INIT,
        record: bool = False
        ):
    # Create evaluation environment ####################################################################################
    if not (init is None):
        eval_env = gym.make('WindSingleAgent-aviary-v0', aggregate_phy_steps=5, obs=ObservationType('kin'),
                            act=act, mode=mode, total_force=total_force, radius=radius,
                            episode_len=episode_len, debug=debug_env, drone_model=drone,
                            initial_xyzs=np.array(init[0:3]).reshape(1, 3),
                            initial_rpy=np.array(init[3:6]).reshape(1, 3))
    else:
        eval_env = gym.make('WindSingleAgent-aviary-v0', aggregate_phy_steps=5, obs=ObservationType('kin'),
                            act=act, mode=mode, total_force=total_force, radius=radius,
                            drone_model=drone, debug=debug_env)

    # Decide path ######################################################################################################
    if not os.path.isfile(name):
        print("[ERROR]: no model under the specified path", name)

    # Load model #######################################################################################################
    model = PPO.load(name)

    # Evaluate and write ###############################################################################################
    eval: EvalWriter = EvalWriter(name='TestWriter', eval_steps=episodes, path='test.xlsx', env=eval_env,
                                  episode_len=episode_len, threshold=0.05)
    eval.evaluateModel(model)

    # Create test environment and logger ###############################################################################
    if gui:
        test_env = gym.make('WindSingleAgent-aviary-v0', aggregate_phy_steps=5, obs=ObservationType('kin'),
                            act=act, mode=mode, total_force=total_force, radius=radius,
                            drone_model=drone, debug=debug_env, gui=gui, record=record, episode_len=episode_len)
        logger = Logger(logging_freq_hz=int(test_env.SIM_FREQ/test_env.AGGR_PHY_STEPS),
                        num_drones=1,
                        output_folder=folder)

    # Start a visual simulation ########################################################################################

        obs = test_env.reset()
        pathplotter = PathPlotter(test_env.goal)
        start = time.time()
        for i in range(episode_len * int(test_env.SIM_FREQ/test_env.AGGR_PHY_STEPS)):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = test_env.step(action)
            # test_env.render()
            pathplotter.addPose(test_env.getPose())
            logger.log(drone=0,
                       timestamp=i/test_env.SIM_FREQ,
                       state=test_env.getKinState(),
                       control=np.zeros(12))
            sync(np.floor(i*test_env.AGGR_PHY_STEPS), start, test_env.TIMESTEP)
            if done:
                obs = test_env.reset()
                break # OPTIONAL EPISODE HALT
        test_env.close()
        #logger.save_as_csv("sa")  # Optional CSV save
        pathplotter.show()
        logger.plot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script that allows to evaluate your RL Model")
    parser.add_argument('--drone', default=DEFAULT_DRONE, type=DroneModel)
    parser.add_argument('--episodes', default=DEFAULT_EPISODES, type=int,
                        help='Amount of evaluation episodes', metavar='')
    parser.add_argument('--mode', default=DEFAULT_MODE, type=int,
                        help='The mode of the training environment(default: 0)', metavar='')
    parser.add_argument('--total_force', default=DEFAULT_FORCE, type=float,
                        help='The max force in the simulated Wind field(default: 0)', metavar='')
    parser.add_argument('--radius', default=DEFAULT_RADIUS, type=float,
                        help='The radius(default: 0.2)', metavar='')
    parser.add_argument('--debug_env', action='store_const',
                        help='Parameter to the Environment that enables most of the Debug messages(default: False)',
                        const=True, default=False)
    parser.add_argument('--episode_len', default=DEFAULT_EPISODE_LEN, type=int,
                        help='The episode length(default: 5)', metavar='')
    parser.add_argument('--init', default=DEFAULT_INIT, nargs='+', type=float,
                        help='The init values in form [x,y,z,r,p,y](default: None)', metavar='')
    parser.add_argument('--gui', action='store_const', help='Enable GUI after evaluation process(default: True)',
                        const=False, default=True)
    parser.add_argument('--act', default=DEFAULT_ACT, type=ActionType,
                        help='The action type of the environment (default: rpm)', metavar='')
    parser.add_argument('--name', default=DEFAULT_NAME, type=str,
                        help='The name of the model (default: results/succcess_model.zip', metavar='')
    parser.add_argument('--record', action='store_const',
                        help='Record an mp4 video (default: False)', const=True, default=False)

    ARGS = parser.parse_args()

    welcome(ARGS)
    run(**vars(ARGS))
