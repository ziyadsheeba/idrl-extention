import copy
import unittest

import numpy as np
from joblib import Parallel, delayed
from scipy import spatial

from src.envs.driver import get_driver_target_velocity


class TestDriverEnvironment(unittest.TestCase):
    # def test_redundant_policy_regret(self):
    #     env_true = get_driver_target_velocity()
    #     optimal_policy = env_true.get_optimal_policy()

    #     env_estimate = get_driver_target_velocity(reward_weights=np.array(8 * [0]))
    #     estimated_policy = env_estimate.get_optimal_policy()

    #     r_estimate = env_estimate.simulate(estimated_policy)
    #     r_optimal = env_estimate.simulate(optimal_policy)
    #     self.assertTrue(r_estimate == 0)
    #     self.assertTrue(r_optimal == 0)

    # def test_solver_stability(self):
    #     env_true = get_driver_target_velocity()
    #     self.assertTrue(
    #         np.allclose(
    #             env_true.simulate(env_true.get_optimal_policy()),
    #             env_true.simulate(env_true.get_optimal_policy()),
    #         )
    #     )

    def test_parallel_policy_solving(self):
        env = get_driver_target_velocity()
        policies = Parallel(n_jobs=1, backend="multiprocessing")(
            delayed(env.get_optimal_policy)(theta=np.array([1] * 8)) for _ in range(3)
        )
        breakpoint()
