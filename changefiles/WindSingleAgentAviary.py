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

class WindSingleAgentAviary(BaseSingleAgentAviary):
    """Models the Single Agent Problem to hover at a position under influence of strong wind."""

    #############################################################################################

    def __init__(self,
                 #drone_model: DroneModel = DroneModel.CF2X,
                 drone_model: DroneModel = DroneModel("hb"),
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
                           self.vel[nth_drone, :], self.ang_v[nth_drone, :], self.goal ,self.last_clipped_action[nth_drone, :]]).reshape(23, )

    def _computeReward(self) -> float:
        """Computes the current reward value.

        Returns
        -------
        float
        The reward.

        """

        state = self._getDroneStateVector(0)
        reward = 0

        if abs(np.linalg.norm(self.goal - state[0:3])) < 0.01 and self.debug:
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
        return np.hstack([obs[0:3], obs[7:10], obs[10:13], obs[13:16], obs[16:19]]).reshape(15, ).astype('float32')

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

    def _clipAndNormalizeState(self, state) -> np.ndarray:
        """Normalizes a drone's state to the [-1,1] range.

        Parameters
        ----------
        state : ndarray
            (23,)-shaped array of floats containing the non-normalized state of a single drone.

        Returns
        -------
        ndarray
            (23,)-shaped array of floats containing the normalized state of a single drone.

        """

        ### estimate the farest reachable positionn ####################################################################
        max_xy = 4 * self.EPISODE_LEN_SEC  # velocity of 4m/s in x and y direction
        max_z = 2 * self.EPISODE_LEN_SEC   # max velocity of 2m/s in z direction

        clipped_pos_xy = np.clip(state[0:2], -max_xy, max_xy)
        clipped_pos_z = np.clip(state[2], 0, max_z)
        clipped_goal_xy = np.clip(state[16:18], -max_xy, max_xy)
        clipped_goal_z = np.clip(state[18], 0, max_z)
        clipped_rp = np.clip(state[7:9], -np.pi, np.pi)
        clipped_vel_xy = np.clip(state[10:12], -4, 4)
        clipped_vel_z = np.clip(state[12], -2, 2)

        if self.GUI:
            self._clipAndNormalizeStateWarning(state, clipped_pos_xy, clipped_pos_z, clipped_goal_xy, clipped_goal_z,
                                               clipped_rp, clipped_vel_xy, clipped_vel_z)

        normalized_pos_xy = clipped_pos_xy / max_xy
        normalized_pos_z = clipped_pos_z / max_z
        normalized_goal_xy = clipped_goal_xy / max_xy
        normalized_goal_z = clipped_goal_z / max_z
        normalized_rp = clipped_rp / np.pi
        normalized_y = state[9] / np.pi  # No reason to clip
        normalized_vel_xy = clipped_vel_xy / 4
        normalized_vel_z = clipped_vel_z / 4
        normalized_ang_vel = state[13:16] / np.linalg.norm(state[13:16]) if np.linalg.norm(
            state[13:16]) != 0 else state[13:16]

        norm_and_clipped = np.hstack([normalized_pos_xy,
                                      normalized_pos_z,
                                      state[3:7],
                                      normalized_rp,
                                      normalized_y,
                                      normalized_vel_xy,
                                      normalized_vel_z,
                                      normalized_ang_vel,
                                      normalized_goal_xy,
                                      normalized_goal_z,
                                      state[19:23]
                                      ]).reshape(23, )

        return norm_and_clipped

    def _clipAndNormalizeStateWarning(self, state, clipped_pos_xy, clipped_pos_z, clipped_goal_xy, clipped_goal_z,
                                      clipped_rp, clipped_vel_xy, clipped_vel_z):
        """Debugging printouts associated to `_clipAndNormalizeState`.
           Print a warning if values in a state vector is out of the clipping range.

        Paramters
        ---------
        state: np.array
            The drone state.
        clipped_pos_xy: np.array
            The clipped position in x and y axis.
        clipped_pos_z: float
            The clipped position in z axis
        clipped_goal_xy: np.array
            The clipped goal position in x, y axis. Should not be clipped if goal is reachable.
        clipped_goal_z:
            The clipped goal positon in z axis. Should not be clipped if goal is reachable.
        clipped_rp:
            The clipped roll, pitch.
        clipped_vel_xy:
            The clipped velocity in x an y axis.
        clipped_vel_z:
            The clipped velocity in z axis.

        """
        if not (clipped_pos_xy == np.array(state[0:2])).all():
            msg: str = "[WARNING] it" + str(self.step_counter) + "in WindSingleAgentAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(state[0],state[1])
            debug(bcolors.WARNING, msg)

        if not (clipped_pos_z == np.array(state[2])).all():
            msg: str = "[WARNING] it" + self.step_counter + "in WindSingleAgentAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2])
            debug(bcolors.WARNING, msg)

        if not (clipped_goal_xy == np.array(state[16:18])).all():
            print(clipped_goal_xy, np.array(state[16:18]))
            msg: str = "[WARNING] in WindSingleAgentAviary._clipAndNormalizeState(): X or Y position of goal is unreachable. Change upper_bound or episode_len!"
            debug(bcolors.FAIL, msg)

        if not (clipped_goal_z == np.array(state[18])).all():
            msg: str = "[WARNING] in WindSingleAgentAviary._clipAndNormalizeState(): Z position of goal is unreachable. Change upper_bound or episode_len!"
            debug(bcolors.FAIL, msg)

        if not (clipped_rp == np.array(state[7:9])).all():
            msg: str = "[WARNING] it" + str(self.step_counter) + "in WindSingleAgentAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]". format(state[7], state[8])
            debug(bcolors.WARNING, msg)

        if not (clipped_vel_xy == np.array(state[10:12])).all():
            msg: str = "[WARNING] it" + str(self.step_counter) + "in WindSingleAgentAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(state[10], state[11])
            debug(bcolors.WARNING, msg)

        if not (clipped_vel_z == np.array(state[12])).all():
            msg: str = "[WARNING] it" + str(self.step_counter) + "in WindSingleAgentAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12])
            debug(bcolors.WARNING, msg)

    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42}

    def getState(self) -> np.ndarray:
        """Method that provides the real state to the script.
           Should be used to log the real state information and not the observations.

        Returns
        -------
        ndarray
            A Box() of shape (23,) with the current state.

        """

        return self._getDroneStateVector(0)

    def getDist(self) -> float:
        """Method that provides the current distance to the script.
           Should be used to log the distances to the goal and also other Data in the evalwriter.

        Returns
        -------
        float
            The current distance to the goal.

        """

        return abs(np.linalg.norm(self.goal - self._getDroneStateVector(0)[0:3]))

    def getSimTime(self) -> float:
        """Method that provides the current Simulation Time to the script.
        Should be used to log Timestamps of Data or Events.

        Returns
        -------
        float
            The current Simulation time [s].

        """

        return self.step_counter * self.TIMESTEP

    def getPose(self) -> List[float]:
        """Method that provides the current Position to the script.
        Should be used to log / plot the position of the drone.

        Returns
        -------
        List[float]
            The current position.

        """
        return self._getDroneStateVector(0)[0:3]