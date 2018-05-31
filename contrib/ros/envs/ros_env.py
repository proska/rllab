from geometry_msgs.msg import Pose, Point
import numpy as np

from rllab.core.serializable import Serializable
from rllab.envs.base import Env, Step
from rllab.misc.ext import get_seed
from rllab.spaces import Box

from contrib.ros.util.common import rate_limited
from contrib.ros.util.gazebo import Gazebo


class RosEnv(Env, Serializable):
    """
    Superclass for all ros environment
    """

    def __init__(self, task_obj_mgr, robot, simulated=False):
        """
        :param task_obj_mgr: object
                Use this to manage objects used in a specific task
        :param robot: object
                the robot interface for the environment
        :param simulated: bool
                if the environment is for real robot or simulation
        """
        Serializable.quick_init(self, locals())

        np.random.RandomState(get_seed())

        self._robot = robot

        self.simulated = simulated

        self.task_obj_mgr = task_obj_mgr

        if self.simulated:
            self.gazebo = Gazebo()
            self._initial_setup()
            self.task_obj_mgr.subscribe_gazebo()
        else:
            # TODO(gh/8: Sawyer runtime support)
            pass

    def initialize(self):
        # TODO (gh/74: Add initialize interface for robot)
        pass

    def shutdown(self):
        if self.simulated:
            # delete model
            for obj in self.task_obj_mgr.objects:
                self.gazebo.delete_gazebo_model(obj.name)
        else:
            # TODO(gh/8: Sawyer runtime support)
            pass

    # =======================================================
    # The functions that base rllab Env asks to implement
    # =======================================================
    @rate_limited(100)
    def step(self, action):
        """
        Perform a step in gazebo. When end of episode
        is reached, reset() should be called to reset the environment's internal state.
        Input
        -----
        action : an action provided by the environment
        Outputs
        -------
        (observation, reward, done, info)
        observation : agent's observation of the current environment
        reward [Float] : amount of reward due to the previous action
        done : a boolean, indicating whether the episode has ended
        info : a dictionary containing other diagnostic information from the previous action
        """
        self._robot.send_command(action)

        obs = self.get_observation()

        reward = self.reward(obs['achieved_goal'], self.goal)
        done = self.done(obs['achieved_goal'], self.goal)
        next_observation = obs['observation']
        return Step(observation=next_observation, reward=reward, done=done)

    def reset(self):
        """
        Resets the state of the environment, returning an initial observation.
        Outputs
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        self._robot.reset()

        self.goal = self.sample_goal()

        if self.simulated:
            target_idx = 0
            for target in self.task_obj_mgr.targets:
                self.gazebo.set_model_pose(
                    model_name=target.name,
                    new_pose=Pose(
                        position=Point(
                            x=self.goal[target_idx * 3],
                            y=self.goal[target_idx * 3 + 1],
                            z=self.goal[target_idx * 3 + 2])))
                target_idx += 1
            self._reset_sim()
        else:
            # TODO(gh/8: Sawyer runtime support)
            pass
        obs = self.get_observation()
        initial_observation = obs['observation']

        return initial_observation

    def _reset_sim(self):
        """
        reset the simulation
        """
        # Randomize start position of object
        for manipulatable in self.task_obj_mgr.manipulatables:
            manipulatable_random_delta = np.zeros(2)
            while np.linalg.norm(manipulatable_random_delta) < 0.1:
                manipulatable_random_delta = np.random.uniform(
                    -manipulatable.random_delta_range,
                    manipulatable.random_delta_range,
                    size=2)
            self.gazebo.set_model_pose(
                manipulatable.name,
                new_pose=Pose(
                    position=Point(
                        x=manipulatable.initial_pos.x +
                        manipulatable_random_delta[0],
                        y=manipulatable.initial_pos.y +
                        manipulatable_random_delta[1],
                        z=manipulatable.initial_pos.z)))

    @property
    def action_space(self):
        return self._robot.action_space

    @property
    def observation_space(self):
        """
        Returns a Space object
        """
        return Box(
            -np.inf, np.inf, shape=self.get_observation()['observation'].shape)

    def _initial_setup(self):
        self._robot.reset()

        if self.simulated:
            # Generate the world
            # Load the model
            for obj in self.task_obj_mgr.objects:
                self.gazebo.load_gazebo_model(obj)
        else:
            # TODO(gh/8: Sawyer runtime support)
            pass

    # ====================================================
    # Need to be implemented in specific robot env
    # ====================================================
    def sample_goal(self):
        """
        Samples a new goal and returns it.
        """
        raise NotImplementedError

    def get_observation(self):
        """
        Get observation
        """
        raise NotImplementedError

    def done(self, achieved_goal, goal):
        """
        :return if done: bool
        """
        raise NotImplementedError

    def reward(self, achieved_goal, goal):
        """
        Compute the reward for current step.
        """
        raise NotImplementedError

    @property
    def goal(self):
        return self._goal

    @goal.setter
    def goal(self, value):
        self._goal = value
