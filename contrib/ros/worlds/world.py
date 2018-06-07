class World(object):
    def initialize(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def terminate(self):
        raise NotImplementedError

    def get_observation(self):
        raise NotImplementedError

    @property
    def observation_space(self):
        raise NotImplementedError
