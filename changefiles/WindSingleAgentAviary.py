import math
import random

import numpy as np
import sys
import os
import pybullet as p
from PIL import Image

from gym import spaces
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.single_agent_rl.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType, BaseSingleAgentAviary
import pybullet_data
import pkg_resources
from robust_drone_pathfollowing.helpclasses.printout import *
from robust_drone_pathfollowing.helpclasses.wind import *
from robust_drone_pathfollowing.helpclasses.functions3D import *

import time

class WindSingleAgentAviary(HoverAviary):
    """Models the Single Agent Problem to hover at a position under influence of strong wind."""

    #############################################################################################

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 freq: int = 240,
                 aggregate_phy_steps: int = 1,
                 gui=False,
                 record=False,
                 obs: ObservationType = ObservationType.KIN,
                 # act: ActionType=ActionType.RPM
                 # TODO:i want to set velocities using PID not RPM ???
                 act: ActionType = ActionType.RPM,
                 total_force: float = 0.000,
                 mode: int = 0,
                 episode_len: int = 5,
                 upper_bound: float = 1.0,
                 debug: bool = False
                 ):
        """Initialization of a single agent RL environment with wind field.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        freq : int, optional
            The frequency (Hz) at which the physics engine steps.
        aggregate_phy_steps : int, optional
            The number of physics steps within one call to `BaseAviary.step()`.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation in folder `files/videos/`.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
            #TODO: Seqeunz of kinematic information as observation space
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)
        total_force: float, optional
            Approximate total force that will be applied to the UAV
        mode: int, optional
            The mode of the Wind environment that can be used for incremental Learning
        episode_len: int, optional
            The number of seconds each episode is simulated
        upper_bound: float, optional
            The upper bound of the random goal
        debug: bool, optional
            Enables and Disables debug messages

        """

        self.mode = mode
        self.total_force = total_force
        self.upper_bound = upper_bound
        self.debug = debug

        super().__init__(drone_model=drone_model,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )

        self.EPISODE_LEN_SEC = episode_len

    def _observationSpace(self) -> spaces.box:
        return spaces.Box(low=np.array([-1, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0]),
                      high=np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
                      dtype=np.float32
                      )

    def _getDroneStateVector(self, nth_drone) -> np.ndarray:
        """Returns the state vector of the n-th drone.

                Parameters
                ----------
                nth_drone : int
                    The ordinal number/position of the desired drone in list self.DRONE_IDS.

                Returns
                -------
                ndarray
                    (23,)-shaped array of floats containing the state vector of the n-th drone.
                    Check the only line in this method and `_updateAndStoreKinematicInformation()`
                    to understand its format.

        """

        return np.hstack([self.pos[nth_drone, :], self.quat[nth_drone, :], self.rpy[nth_drone, :],
                           self.vel[nth_drone, :], self.ang_v[nth_drone, :], self.goal,self.last_clipped_action[nth_drone, :]]).reshape(23, )

    def _computeReward(self) -> float:
        """Computes the current reward value.

        Returns
        -------
        float
        The reward.

        """

        state = self._getDroneStateVector(0)
        reward = 0

        # Check wether the goal has been reached
        # TODO: check if this should get a positive reward
        if abs(np.linalg.norm(self.goal - state[0:3])) < 0.01:
            if (self.debug):
                debug(bcolors.OKGREEN, '[INFO] Reached the goal')


        # penalize the agent because he hit the ground
        if state[2] < 0.02:
            reward += -100
            if (self.debug):
                debug(bcolors.FAIL, '[INFO] Hit the ground')

        return reward + -1 * np.linalg.norm(self.goal - state[0:3]) ** 2

    def _computeObs(self) -> np.ndarray:
        """Returns the current observation of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (15,) with the observation consisting of the clipped values:
             - pos[] the current position of the drone
             - rpy[] the current roll, pitch and yaw value
             - vel[] the current velocities in each axis
             - ang_vel[] the current angular velocities
             - goal[] the current goal that should be reached

        """

        obs = self._clipAndNormalizeState(self._getDroneStateVector(0))
        #### OBS SPACE OF SIZE 15
        return np.hstack([obs[0:3], obs[7:10], obs[10:13], obs[13:16], self.goal]).reshape(15, ).astype('float32')

    def _computeDone(self) -> bool:
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """

        if self.step_counter/self.SIM_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False

    def _housekeeping(self):
        """Housekeeping function.

        Allocation and zero-ing of the variables and PyBullet's parameters/objects
        in the `reset()` function. Also reinitializes the new wind and goal.

        """

        ### Initialize/reset the Wind specific parameters
        # mode 0 is a basic mode without wind and with an easy to reach goal along the z axis
        if self.mode == 0:
            self.goal = [0, 0, random.uniform(0.2, self.upper_bound)]

        # mode 1 is a basic mode without wind and a near goal along all axis
        elif self.mode == 1:
            self.goal = [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0.2, self.upper_bound)]

        # mode 2 is a basic mode with random goals and a random constant wind field
        elif self.mode == 2:
            self.goal = [random.uniform(0, self.upper_bound), random.uniform(0, self.upper_bound), random.uniform(0.2, self.upper_bound)]
            self.wind = Wind(total_force=self.total_force, args=0)

        # mode with random goal and random generated wind field
        elif self.mode > 2:
            self.goal = [random.uniform(0, self.upper_bound), random.uniform(0, self.upper_bound), random.uniform(0.2, self.upper_bound)]
            self.wind = Wind(total_force=self.total_force, args=random.randint(0, 10))

        super()._housekeeping()


        debug_message = '[INFO] using mode: ' + str(self.mode) + '\n[INFO] using a total wind force of: ' + str(
            self.total_force) + ' Newton' + '\n[INFO] the goal is:' + str(self.goal)
        debug(bcolors.WARNING, debug_message)

    def _physics(self, rpm, nth_drone):
        """Base PyBullet physics implementation with a static wind field.

        Parameters
        ----------
        rpm : ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """

        # wind force
        #debug(bcolors.WARNING, 'WIND')
        if self.mode != 0 and self.mode != 1:
            pos = self._getDroneStateVector(0)[0:3]
            p.applyExternalForce(self.DRONE_IDS[nth_drone],
                             -1,
                             forceObj=self.wind.get(pos[0], pos[1], pos[2]),
                             posObj=[0, 0, 0],
                             flags=p.LINK_FRAME,
                             physicsClientId=self.CLIENT)

        super()._physics(rpm=rpm, nth_drone=nth_drone)

    #def _clipAndNormalizeState(self,
    #                           state
    #                           ):

    #def _clipAndNormalizeStateWarning(self,
    #                                  state,
    #                                  clipped_pos_xy,
    #                                  clipped_pos_z,
    #                                  clipped_rp,
    #                                  clipped_vel_xy,
    #                                  clipped_vel_z,
    #                                  ):

    def getState(self, nth_drone) -> np.ndarray:
        """Method that provides the real state to the script.
           Should be used to log the real state information and not the observations.

        Parameters
        ----------
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        Returns
        -------
        ndarray
            A Box() of shape (23,) with the current state

        """

        return self._getDroneStateVector(nth_drone)

    def getDist(self) -> float:
        """Method that provides the current distance to the script.
           Should be used to log the distances to the goal and also other Data in the evalwriter.

        Returns
        -------
        float
            The current distance to the goal

        """

        return abs(np.linalg.norm(self.goal - self._getDroneStateVector(0)[0:3]))