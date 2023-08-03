import numpy as np
import pybullet as p
from gym import spaces
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType, \
    BaseSingleAgentAviary
from robust_drone_pathfollowing.helpclasses.printout import *
from robust_drone_pathfollowing.helpclasses.wind import *
from robust_drone_pathfollowing.helpclasses.functions3D import *


class WindSingleAgentAviary(BaseSingleAgentAviary):
    """Models the Single Agent Problem to hover at a position under influence of strong wind."""

    #############################################################################################

    def __init__(self,
                 drone_model: DroneModel = DroneModel("hb"),
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 freq: int = 240,
                 aggregate_phy_steps: int = 5,
                 gui=False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.RPM,
                 total_force: float = 0.000,
                 mode: int = 0,
                 episode_len: int = 5,
                 radius: float = 0.0,
                 debug: bool = False,
                 type: int = 0
                 ) -> None:
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
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision).
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control).
        total_force: float, optional
            Approximate total force that will be applied to the UAV.
        mode: int, optional
            The mode of the Wind environment that can be used for incremental Learning.
        episode_len: int, optional
            The number of seconds each episode is simulated.
        radius: float, optional
            The radius of the goal ball.
        debug: bool, optional
            Enables and disables debug messages.
        type: int, optional
            The type of the wind field.

        """

        self.wind_type = type
        self.mode = mode
        self.total_force = total_force
        self.radius = radius
        self.debug = debug
        if drone_model == DroneModel("hb") and initial_xyzs is None:
            initial_xyzs: np.array = np.array([0, 0, 0.16]).reshape(1, 3)
            self.minZ: float = 0.16
            self.maxVelXY: float = 5  #m/s
            self.maxVelZ: float = 3  #m/s
        if drone_model == DroneModel.CF2X and initial_xyzs is None:
            initial_xyzs: np.array = np.array([0, 0, 0.1]).reshape(1, 3)
            self.minZ: float = 0.1
            self.maxVelXY: float = 3  # m/s
            self.maxVelZ: float = 1  # m/s

        self.maxDist: float = 20  #m

        if mode == 0:
            initial_xyzs: np.array = np.array([0, 0, 0.5]).reshape(1, 3)

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
        """Returns the observation space of the environment.

            Returns
            -------
            ndarray
                A Box() of shape (H,W,4) or (12,) depending on the observation type.

        """

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
                          self.vel[nth_drone, :], self.ang_v[nth_drone, :], self.goal,
                          self.last_clipped_action[nth_drone, :]]).reshape(23, )

    def _computeReward(self) -> float:
        """Computes the current reward value.

        Returns
        -------
        float
        The reward.

        """

        state: np.array = np.hstack([self.pos[0, :], self.rpy[0, :]]).reshape(6, )
        reward: float = np.exp(-0.6 * np.linalg.norm(self.goal - state[0:3])) - 1

        if state[2] <= self.minZ and self.step_counter/self.SIM_FREQ >= 0.1:
            reward += -200
        if abs(state[3] > 1) or abs(state[4]) > 1:
            reward += -200

        return reward

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
        # OBS SPACE OF SIZE 15
        return np.hstack([obs[0:3], obs[7:10], obs[10:13], obs[13:16], obs[16:19]]).reshape(15, ).astype('float32')

    def _preprocessAction(self, action) -> np.array():
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Parameter `action` is processed differenly for each of the different
        action types: `action` can be of length 1, 3, 4, or 6 and represent
        RPMs, desired thrust and torques, the next target position to reach
        using PID control, a desired velocity vector, new PID coefficients, etc.

        Parameters
        ----------
        action : ndarray
            The input action for each drone, to be translated into RPMs.

        Returns
        -------
        ndarray
            (4,)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """

        if self.ACT_TYPE == ActionType.RPM:
            return np.array(self.HOVER_RPM * (1 + 0.05 * action))

        elif self.ACT_TYPE == ActionType.ONE_D_RPM:
            return np.repeat(self.HOVER_RPM * (1 + 0.05 * action), 4)

        elif self.ACT_TYPE == ActionType.VEL:
            state = self._getDroneStateVector(0)
            if np.linalg.norm(action[0:3]) != 0:
                v_unit_vector = action[0:3] / np.linalg.norm(action[0:3])
            else:
                v_unit_vector = np.zeros(3)
            rpm, _, _ = self.ctrl.computeControl(control_timestep=self.AGGR_PHY_STEPS*self.TIMESTEP,
                                                 cur_pos=state[0:3],
                                                 cur_quat=state[3:7],
                                                 cur_vel=state[10:13],
                                                 cur_ang_vel=state[13:16],
                                                 target_pos=state[0:3], # same as the current position
                                                 target_rpy=np.array([0, 0, state[9]]),  # keep current yaw
                                                 target_vel=self.SPEED_LIMIT * np.abs(action[3]) * v_unit_vector # target the desired velocity vector
                                                 )
            return rpm

        else:
            print("[ERROR] in BaseSingleAgentAviary._preprocessAction()")

    def _computeDone(self) -> bool:
        """Computes the current done value.

        Returns
        -------
        bool:
            Whether the current episode is done.

        """

        state: np.array = np.hstack([self.pos[0, :], self.rpy[0, :]]).reshape(6, )
        if state[2] <= self.minZ and self.step_counter / self.SIM_FREQ >= 0.1:
            return True
        if abs(state[3] > 1) or abs(state[4]) > 1:
            return True
        if self.step_counter/self.SIM_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False

    def _housekeeping(self):
        """Housekeeping function.

        Allocation and zero-ing of the variables and PyBullet's parameters/objects
        in the `reset()` function. Also reinitializes the new wind and goal.

        """

        # Initialize/reset the Wind specific parameters
        # mode 0 is a basic mode without wind and with an easy to reach static goal along the z axis
        if self.mode == 0:
            self.goal: np.array = np.array([0, 0, 0.5])

        # mode 1 is a basic mode without wind and with an easy to reach static goal along the z axis
        elif self.mode == 1:
            self.goal: np.array = np.array([0, 0, 0.5])

        # mode 2 is a basic mode without wind and a goal near [0,0,0.5] with a radius of r
        elif self.mode == 2:
            # sample evenly distributed points inside a half sphere
            rad_vector: np.array = np.array([random.uniform(-self.radius, self.radius),
                                             random.uniform(-self.radius, self.radius),
                                             random.uniform(0, self.radius)])
            while np.linalg.norm(rad_vector) > self.radius:
                rad_vector: np.array = np.array([random.uniform(-self.radius, self.radius),
                                                 random.uniform(-self.radius, self.radius),
                                                 random.uniform(0, self.radius)])
            self.goal: np.array = np.array([0, 0, 0.5]) + rad_vector

        # mode 4 is a basic mode with random goals and a specified wind field
        elif self.mode == 3:
            rad_vector: np.array = np.array([random.uniform(-self.radius, self.radius),
                                             random.uniform(-self.radius, self.radius),
                                             random.uniform(0, self.radius)])
            while np.linalg.norm(rad_vector) > self.radius:
                rad_vector: np.array = np.array([random.uniform(-self.radius, self.radius),
                                                 random.uniform(-self.radius, self.radius),
                                                 random.uniform(0, self.radius)])
            self.goal: np.array = np.array([0, 0, 0.5]) + rad_vector
            self.wind = Wind(total_force=self.total_force, args=self.wind_type)

        super()._housekeeping()

        # self._updateAndStoreKinematicInformation()
        if self.GUI:
            self.goal_vis = p.addUserDebugPoints([self.goal], [[1, 0, 0]], pointSize=10, lifeTime=0)

        if self.debug:
            debug_message = '[INFO] using mode: ' + str(self.mode) + '\n[INFO] using a total wind force of: ' + str(
                self.total_force) + ' Newton' + '\n[INFO] the goal is:' + str(
                self.goal) + '\n[INFO] starting position' + str(self.pos)
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

        if self.mode >= 3:
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

        # estimate the farest reachable position
        #max_xy = 15 * self.EPISODE_LEN_SEC  # velocity of 15m/s in x and y direction
        #max_z = 14 * self.EPISODE_LEN_SEC   # max velocity of 14m/s in z direction

        max_xy = 5
        max_z = 5

        clipped_pos_xy = np.clip(state[0:2], -max_xy, max_xy)
        clipped_pos_z = np.clip(state[2], 0, max_z)
        clipped_goal_xy = np.clip(state[16:18], -max_xy, max_xy)
        clipped_goal_z = np.clip(state[18], 0, max_z)
        clipped_rp = np.clip(state[7:9], -np.pi, np.pi)
        clipped_vel_xy = np.clip(state[10:12], -3, 3)
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
        normalized_vel_xy = clipped_vel_xy / self.maxVelXY
        normalized_vel_z = clipped_vel_z / self.maxVelXY
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

        Parameters
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
            The clipped goal position in z axis. Should not be clipped if goal is reachable.
        clipped_rp:
            The clipped roll, pitch.
        clipped_vel_xy:
            The clipped velocity in x an y axis.
        clipped_vel_z:
            The clipped velocity in z axis.

        """
        if not (clipped_pos_xy == np.array(state[0:2])).all():
            msg: str = "[WARNING] it " + str(self.step_counter) + " in WindSingleAgentAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(state[0], state[1])
            debug(bcolors.WARNING, msg)

        if not (clipped_pos_z == np.array(state[2])).all():
            msg: str = "[WARNING] it " + str(self.step_counter) + " in WindSingleAgentAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2])
            debug(bcolors.WARNING, msg)

        if not (clipped_goal_xy == np.array(state[16:18])).all():
            print(clipped_goal_xy, np.array(state[16:18]))
            msg: str = "[WARNING] in WindSingleAgentAviary._clipAndNormalizeState(): X or Y position of goal is unreachable. Change upper_bound or episode_len!"
            debug(bcolors.FAIL, msg)

        if not (clipped_goal_z == np.array(state[18])).all():
            msg: str = "[WARNING] in WindSingleAgentAviary._clipAndNormalizeState(): Z position of goal is unreachable. Change upper_bound or episode_len!"
            debug(bcolors.FAIL, msg)

        if not (clipped_rp == np.array(state[7:9])).all():
            msg: str = "[WARNING] it " + str(self.step_counter) + " in WindSingleAgentAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]". format(state[7], state[8])
            debug(bcolors.WARNING, msg)

        if not (clipped_vel_xy == np.array(state[10:12])).all():
            msg: str = "[WARNING] it " + str(self.step_counter) + " in WindSingleAgentAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(state[10], state[11])
            debug(bcolors.WARNING, msg)

        if not (clipped_vel_z == np.array(state[12])).all():
            msg: str = "[WARNING] it " + str(self.step_counter) + " in WindSingleAgentAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12])
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

    def getState(self) -> np.array:
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

    def getPose(self) -> np.array:
        """Method that provides the current Position to the script.
        Should be used to log / plot the position of the drone.

        Returns
        -------
        np.array
            The current position.

        """
        return self._getDroneStateVector(0)[0:3]

    def getKinState(self) -> np.array:
        """Method that returns the kinematic state of the drone.

        Returns:
        --------
        np.array
            The current kinematic state.

        """
        return np.hstack([self.pos[0, :], self.quat[0, :], self.rpy[0, :],
                          self.vel[0, :], self.ang_v[0, :],
                          self.last_clipped_action[0, :]]).reshape(20, )

