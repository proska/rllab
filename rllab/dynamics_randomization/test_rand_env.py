from rllab.algos import TRPO
from rllab.baselines import LinearFeatureBaseline
from rllab.envs.mujoco import SwimmerEnv
from rllab.dynamics_randomization.RandomizeEnv import randomize
from rllab.envs import normalize
from rllab.policies import GaussianMLPPolicy
from rllab.dynamics_randomization import Variations, VariationMethods
from rllab.dynamics_randomization import VariationDistributions
from rllab.misc import run_experiment_lite

variations = Variations()
variations.randomize().\
        attribute("size").\
        at_xpath(".//geom[@name='back']").\
        with_method(VariationMethods.COEFFICIENT).\
        sampled_from(VariationDistributions.UNIFORM).\
        with_range(0.1, 1.5).\
        randomize().\
        attribute("size").\
        at_xpath(".//geom[@name='mid']").\
        sampled_from(VariationDistributions.UNIFORM).\
        with_method(VariationMethods.COEFFICIENT).\
        with_range(0.1, 1.5)

variations.randomize().\
        attribute("size").\
        at_xpath(".//geom[@name='torso']").\
        with_method(VariationMethods.ABSOLUTE).\
        sampled_from(VariationDistributions.UNIFORM).\
        with_range(0.1, 1.5)


# def run_task(*_):

env = normalize(randomize(SwimmerEnv(), variations))

policy = GaussianMLPPolicy(
    env_spec=env.spec,
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    hidden_sizes=(32, 32))

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=4000,
    max_path_length=500,
    n_itr=20,
    discount=0.99,
    step_size=0.01,
    # plot=True
)
algo.train()

