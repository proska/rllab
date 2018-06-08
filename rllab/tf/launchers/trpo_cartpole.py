from rllab.tf.algos import TRPO
from rllab.baselines import LinearFeatureBaseline
from rllab.envs.box2d import CartpoleEnv
from rllab.envs.normalized_env import normalize
from rllab.tf.optimizers import ConjugateGradientOptimizer
from rllab.tf.optimizers import FiniteDifferenceHvp
from rllab.tf.policies import GaussianMLPPolicy
from rllab.tf.envs import TfEnv
from rllab.misc import stub, run_experiment_lite

env = TfEnv(normalize(CartpoleEnv()))

policy = GaussianMLPPolicy(
    name="policy",
    env_spec=env.spec,
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    hidden_sizes=(32, 32))

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=4000,
    max_path_length=100,
    n_itr=40,
    discount=0.99,
    step_size=0.01,
    plot=True
    # optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-5))
)
algo.train()
