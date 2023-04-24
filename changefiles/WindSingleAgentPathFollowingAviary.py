import numpy as np
import pybullet as p
from gym import spaces
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType, \
    BaseSingleAgentAviary
from robust_drone_pathfollowing.helpclasses.printout import *
from robust_drone_pathfollowing.helpclasses.wind import *
from robust_drone_pathfollowing.helpclasses.functions3D import *
from gym_pybullet_drones.envs.WindSingleAgentAviary import WindSingleAgentAviary


class WindSingleAgentPathFollowingAviary(WindSingleAgentAviary):
    """Models the Single Agent Problem to reach a number of Waypoints under the influence of strong wind"""

    #######################################################################################################

    def __init__(self,
                 # drone_model: DroneModel = DroneModel.CF2X,
                 drone_model: DroneModel = DroneModel("hb"),
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 freq: int = 240,
                 aggregate_phy_steps: int = 1,
                 gui=False,
                 record=False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.RPM,
                 total_force: float = 0.000,
                 mode: int = 2,
                 episode_len: int = 5,
                 upper_bound: float = 1.0,
                 debug: bool = False,
                 num_waypoints: int = 5
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
            The type of observation space (kinematic information or vision).
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control).
        total_force: float, optional
            Approximate total force that will be applied to the UAV.
        mode: int, optional
            The mode of the Wind environment that can be used for incremental Learning.
        episode_len: int, optional
            The number of seconds each episode is simulated.
        upper_bound: float, optional
            The upper bound of the random goal.
        debug: bool, optional
            Enables and disables debug messages.
        num_waypoints: int, optional
            The number of waypoints

        """

        self.num_waypoints = num_waypoints
        self.waypoints: List[np.array] = []

        super().__init__(drone_model=drone_model,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act,
                         total_force=total_force,
                         mode=mode,
                         episode_len=episode_len,
                         upper_bound=upper_bound,
                         debug=debug
                         )

    def _housekeeping(self):
        """Housekeeping function.

        Allocation and zero-ing of the variables and PyBullet's parameters/objects
        in the `reset()` function. Also reinitializes the new wind and waypoints.

        """

        # Initialize/reset the Wind specific parameters
        # mode 0 is a basic mode without wind and with an easy to reach goal along the z axis
        if self.mode == 0:
            for i in range(0, self.num_waypoints):
                self.waypoints.append(np.array([0, 0, random.uniform(0.2, self.upper_bound)]))

        # mode 1 is a basic mode without wind and a near goal along all axis
        elif self.mode == 1:
            for i in range(0, self.num_waypoints):
                goal = np.array([random.uniform(-self.upper_bound, self.upper_bound),
                                 random.uniform(-self.upper_bound, self.upper_bound),
                                 random.uniform(0.2, self.upper_bound)])
                self.waypoints.append(goal)

        # mode 2 is a basic mode with random goals and a random constant wind field
        elif self.mode == 2:
            for i in range(0, self.num_waypoints):
                goal = np.array([random.uniform(-self.upper_bound, self.upper_bound),
                                      random.uniform(-self.upper_bound, self.upper_bound),
                                      random.uniform(0.2, self.upper_bound)])
                self.waypoints.append(goal)
            self.wind = Wind(total_force=self.total_force, args=0)

        # mode with random goal and random generated wind field
        elif self.mode > 2:
            for i in range(0, self.num_waypoints):
                goal = np.array([random.uniform(-self.upper_bound, self.upper_bound),
                                      random.uniform(-self.upper_bound, self.upper_bound),
                                      random.uniform(0.2, self.upper_bound)])
                self.waypoints.append(goal)
            self.wind = Wind(total_force=self.total_force, args=random.randint(0, 10))

        super(WindSingleAgentAviary, self)._housekeeping()

        if self.debug:
            debug_message = '[INFO] using mode: ' + str(self.mode) + '\n[INFO] using a total wind force of: ' + str(
                self.total_force) + ' Newton' + '\n[INFO] the waypoints are:' + str(self.waypoints)
            debug(bcolors.WARNING, debug_message)

        self.goal = self.waypoints[0]

    def _computeDone(self) -> bool:
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        state = self._getDroneStateVector(0)
        if len(self.waypoints) == 1 and abs(np.linalg.norm(self.goal - state[0:3])) < 0.01:
            return True

        if abs(np.linalg.norm(self.goal - state[0:3])) < 0.01:
            debug(bcolors.OKGREEN, '[INFO] Reached the goal')
            self.goal = self.waypoints.pop(0)

        if self.step_counter/self.SIM_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False

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
        max_xy = 2 * self.upper_bound
        max_z = 2 * self.upper_bound

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

    def getWaypoints(self) -> List[np.array]:
        return self.waypoints
