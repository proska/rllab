import gym
import numpy as np

from rllab.envs.mujoco.ant_env import AntEnv
from rllab.envs.mujoco.hill.hill_env import HillEnv
import rllab.envs.mujoco.hill.terrain as terrain
from rllab.misc.overrides import overrides


class AntHillEnv(HillEnv):

    MODEL_CLASS = AntEnv

    @overrides
    def _mod_hfield(self, hfield):
        # clear a flat patch for the robot to start off from
        return terrain.clear_patch(hfield,
                                   gym.spaces.Box(
                                       np.array([-2.0, -2.0]),
                                       np.array([0.0, 0.0]),
                                       dtype=np.float32))
