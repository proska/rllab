from cached_property import cached_property
import collections

from rllab.envs.env_spec import EnvSpec

_Step = collections.namedtuple("Step",
                               ["observation", "reward", "done", "info"])


def Step(observation, reward, done, **info):
    """
    Convenience method creating a namedtuple with the results of the
    environment.step method.
    Put extra diagnostic info
    """
    return _Step(observation, reward, done, info)
