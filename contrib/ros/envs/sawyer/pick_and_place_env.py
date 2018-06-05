"""
Pick-and-place task for the sawyer robot
"""

from geometry_msgs.msg import Pose, Point
import numpy as np

from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.spaces import Box

from contrib.ros.envs.ros_env import RosEnv
from contrib.ros.robots.sawyer import Sawyer
from contrib.ros.util.common import rate_limited


class PickAndPlaceEnv(RosEnv, Serializable):
    def __init__(self,
                 initial_goal,
                 task_obj_mgr,
                 sparse_reward=True,
                 simulated=False,
                 distance_threshold=0.05,
                 target_range=0.15,
                 robot_control_mode='position'):
        Serializable.quick_init(self, locals())

        self._distance_threshold = distance_threshold
        self._target_range = target_range
        self._sparse_reward = sparse_reward
        self.task_obj_mgr = task_obj_mgr
        self.initial_goal = initial_goal
        self.goal = self.initial_goal.copy()
        self.simulated = simulated

        self._sawyer = Sawyer(
            simulated=simulated, control_mode=robot_control_mode)

        RosEnv.__init__(self, simulated=simulated)

    def _initial_setup(self):
        self._sawyer.reset()

        if self.simulated:
            # Generate the world
            # Load the model
            for obj in self.task_obj_mgr.objects:
                self.gazebo.load_gazebo_model(obj)
        else:
            # TODO(gh/8: Sawyer runtime support)
            pass

    def shutdown(self):
        if self.simulated:
            # delete model
            for obj in self.task_obj_mgr.objects:
                self.gazebo.delete_gazebo_model(obj.name)
        else:
            # TODO(gh/8: Sawyer runtime support)
            pass

    def reset(self):
        """
        Resets the state of the environment, returning an initial observation.
        Outputs
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        self._sawyer.reset()

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
            self.reset_sim()
        else:
            # TODO(gh/8: Sawyer runtime support)
            pass

        obs = self.get_observation()
        initial_observation = obs['observation']

        return initial_observation

    def reset_sim(self):
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

    def reset_real(self):
        # TODO(gh/8: Sawyer runtime support)
        pass

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
        self._sawyer.send_command(action)

        obs = self.get_observation()

        reward = self.reward(obs['achieved_goal'], self.goal)
        done = self.done(obs['achieved_goal'], self.goal)
        next_observation = obs['observation']
        return Step(observation=next_observation, reward=reward, done=done)

    @property
    def action_space(self):
        return self._sawyer.action_space

    @property
    def observation_space(self):
        """
        Returns a Space object
        """
        return Box(
            -np.inf, np.inf, shape=self.get_observation()['observation'].shape)

    def sample_goal(self):
        """
        Sample goals
        :return: the new sampled goal
        """
        goal = self.initial_goal.copy()

        random_goal_delta = np.random.uniform(
            -self._target_range, self._target_range, size=2)
        goal[:2] += random_goal_delta

        return goal

    def get_observation(self):
        """
        Get Observation
        :return observation: dict
                    {'observation': obs,
                     'achieved_goal': achieved_goal,
                     'desired_goal': self.goal}
        """
        robot_obs = self._sawyer.get_observation()

        manipulatable_obs = self.task_obj_mgr.get_manipulatables_observation()

        obs = np.concatenate((robot_obs, manipulatable_obs['obs']))

        return {
            'observation': obs,
            'achieved_goal': manipulatable_obs['achieved_goal'],
            'desired_goal': self.goal
        }

    def reward(self, achieved_goal, goal):
        """
        Compute the reward for current step.
        :param achieved_goal: the current gripper's position or object's position in the current training episode.
        :param goal: the goal of the current training episode, which mostly is the target position of the object or the
                     position.
        :return reward: float
                    if sparse_reward, the reward is -1, else the reward is minus distance from achieved_goal to
                    our goal. And we have completion bonus for two kinds of types.
        """
        d = self._goal_distance(achieved_goal, goal)
        if d < self._distance_threshold:
            return 100
        else:
            if self._sparse_reward:
                return -1.
            else:
                return -d

    def _goal_distance(self, goal_a, goal_b):
        """
        :param goal_a:
        :param goal_b:
        :return distance: distance between goal_a and goal_b
        """
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    def done(self, achieved_goal, goal):
        """
        :return if_done: bool
                    if current episode is done:
        """
        return self._goal_distance(achieved_goal,
                                   goal) < self._distance_threshold
