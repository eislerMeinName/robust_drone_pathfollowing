#Imports
import sys
import os
import time
from datetime import datetime
import math
import random
import numpy as np
import pybullet as p
import typing
import argparse
from helpclasses.printout import *
from helpclasses.wind import *

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.control.SimplePIDControl import SimplePIDControl
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.utils import sync, str2bool


def run(duration_sec: int, physics: Physics, simulation_freq_hz: int, control_freq_hz: int,
        target_x: float, target_y: float, target_z: float,
        wind_x: float, wind_y: float, wind_z: float, drone = DroneModel("cf2x")) -> dict:
    INIT_XYZS = np.array([[0, 0, 0]])
    INIT_RPYS = np.array([[0, 0, (np.pi / 2)]])
    AGGR_PHY_STEPS = int(simulation_freq_hz / control_freq_hz)

    ### Init environment ###############################################################################################
    env = CtrlAviary(drone_model=drone,
                     num_drones=1,
                     initial_xyzs=INIT_XYZS,
                     initial_rpys=INIT_RPYS,
                     physics=physics,
                     neighbourhood_radius=10,
                     freq=simulation_freq_hz,
                     aggregate_phy_steps=AGGR_PHY_STEPS,
                     gui=True,
                     record=False,
                     obstacles=False,
                     user_debug_gui=False
                     )
    debug(bcolors.OKGREEN, 'Environment created sucessfully')

    ### Get Pybullet Client ID #########################################################################################
    PYB_Client = env.getPyBulletClient()

    ### Initialize the controller ######################################################################################
    ctrl = DSLPIDControl(drone_model=drone)

    ### Run the Simulation #############################################################################################
    CTRl_EVERY_N_STEPS = int(np.floor(env.SIM_FREQ/control_freq_hz))
    action = {str(0): np.array([0, 0, 0, 0])}
    START = time.time()

    for i in range(0, int(duration_sec * env.SIM_FREQ), AGGR_PHY_STEPS):

        ### Make it rain ###############################################################################################
        if i / env.SIM_FREQ > 5 and i % 10 == 0 and i / env.SIM_FREQ < 10: p.loadURDF("duck_vhacd.urdf",
                                                                                      [0 + random.gauss(0, 0.3),
                                                                                       0 + random.gauss(0, 0.3), 3],
                                                                                      p.getQuaternionFromEuler(
                                                                                          [random.randint(0, 360),
                                                                                           random.randint(0, 360),
                                                                                           random.randint(0, 360)]),
                                                                                      physicsClientId= PYB_Client)

        ### Step the simulation ########################################################################################
        #debug(bcolors.OKGREEN, 'Trying to step Simulation ' + str(i))
        obs, reward, done, info = env.step(action)
        #pos = obs.get('0').get('state')[0:3]

        ### Create wind ################################################################################################
        #wind_forces = wind.get(pos[0], pos[1], pos[2])
        p.applyExternalForce(env.DRONE_IDS[0],
                             -1,  # -1 for the base, 0-3 for the motors
                             forceObj=[random.gauss(wind_x, 0.001), random.gauss(wind_y, 0.001), random.gauss(wind_z, 0.001)] , # a force vector
                             posObj=[0, 0, 0], flags=p.WORLD_FRAME, physicsClientId=PYB_Client)

        ### Compute control ############################################################################################
        if i%CTRl_EVERY_N_STEPS == 0:
            action[str(0)], _ , _ = ctrl.computeControlFromState(control_timestep= CTRl_EVERY_N_STEPS *env.TIMESTEP,
                                                                    state=obs[str(0)]["state"],
                                                                       target_pos=np.hstack([target_x,target_y, target_z]),
                                                                       target_rpy=INIT_RPYS[0])

        ### Printout ###################################################################################################
        if i%env.SIM_FREQ == 0:
            env.render()

        ### Sync the simulation ########################################################################################
        sync(i, START, env.TIMESTEP)

    env.close()
    return obs





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Script in order to test your Installation")
    parser.add_argument('--duration_sec' , default=10, type=int, help='Duration of the simulation in seconds(default: 10)', metavar='')
    parser.add_argument('--physics' , default=Physics("pyb"), type=Physics, help='The choosen Physics (standard PYB)', metavar='')
    parser.add_argument('--simulation_freq_hz', default=240, type=int, help='Simulation frequency in Hz (default: 240)', metavar='')
    parser.add_argument('--control_freq_hz', default=48, type=int, help='Control frequency in Hz (default: 48)', metavar='')
    parser.add_argument('--target_x', default=0, type=float, help='X Coordinate of Target', metavar='')
    parser.add_argument('--target_y', default=0, type=float, help='Y Coordinate of Target', metavar='')
    parser.add_argument('--target_z', default=0.5, type=float, help='Z Coordinate of Target', metavar='')
    parser.add_argument('--wind_x', default=0, type=float, help='Wind force in x direction', metavar='')
    parser.add_argument('--wind_y', default=0, type=float, help='Wind force in y direction', metavar='')
    parser.add_argument('--wind_z', default=-0, type=float, help='Wind force in z direction', metavar='')
    ARGS = parser.parse_args()

    welcome(ARGS)
    pos = run(**vars(ARGS)).get('0').get('state')[0:3]
    hitground(pos=pos, ARGS=ARGS)

