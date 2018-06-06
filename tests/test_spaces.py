import gym
import numpy as np

from rllab.spaces import Product


def test_product_space():
    _ = Product([gym.spaces.Discrete(3), gym.spaces.Discrete(2)])
    product_space = Product(gym.spaces.Discrete(3), gym.spaces.Discrete(2))
    sample = product_space.sample()
    assert product_space.contains(sample)


def test_product_space_unflatten_n():
    space = Product([gym.spaces.Discrete(3), gym.spaces.Discrete(3)])
    np.testing.assert_array_equal(special.to_onehot((2, 2), space.n), special.to_onehot_n([(2, 2)])[0], space.n)
    np.testing.assert_array_equal(
        special.from_onehot(special.to_onehot((2, 2), space.n)),
        special.from_onehot_n(special.to_onehot_n([(2, 2)]))[0]
    )


def test_box():
    space = gym.spaces.Box(low=-1, high=1, shape=(2, 2))

special.to_onehot([[1, 2], [3, 4]]), [1, 2, 3, 4], space.n)

    np.testing.assert_array_equal(special.to_onehot([[1, 2], [3, 4]]), [1, 2, 3, 4], space.n), [1, 2, 3, 4])
    np.testing.assert_array_equal(special.to_onehot_n([[[1, 2], [3, 4]]], space.n), [[1, 2, 3, 4]])
    np.testing.assert_array_equal(special.from_onehot([1, 2, 3, 4]), [[1, 2], [3, 4]])
    np.testing.assert_array_equal(special.from_onehot_n([[1, 2, 3, 4]]), [[[1, 2], [3, 4]]])
