import unittest

import numpy as np
from scipy.special import expit

from src.reward_models.kernels import RBFKernel
from src.reward_models.logistic_reward_models import GPLogisticRewardModel
from src.utils import matrix_inverse

DIM = 10
X_MAX = 1
X_MIN = -1
KERNEL = RBFKernel(
    dim=DIM,
)


class TestGPLogisticRewardModel(unittest.TestCase):
    def test_updating(self):
        reward_model = GPLogisticRewardModel(
            dim=DIM, kernel=KERNEL, x_min=X_MIN, x_max=X_MAX
        )
        for i in range(10):
            x_1 = np.random.uniform(size=(1, DIM))
            x_2 = np.random.uniform(size=(1, DIM))
            y = 1
            reward_model.update(x_1, x_2, y)

        self.assertTrue(len(reward_model.X) == 20)
        self.assertTrue(len(reward_model.y) == 10)

    def test_sampling(self):
        reward_model = GPLogisticRewardModel(
            dim=DIM, kernel=KERNEL, x_min=X_MIN, x_max=X_MAX
        )
        for i in range(100):
            x_1 = np.random.uniform(size=(1, DIM))
            x_2 = np.random.uniform(size=(1, DIM))
            y = 1
            reward_model.update(x_1, x_2, y)
        x_test = np.random.uniform(size=(1, DIM))
        sample = reward_model.sample_current_approximate_distribution(x_test)
        mean = reward_model.get_mean(x_test)
        cov = reward_model.get_covariance(x_test)
