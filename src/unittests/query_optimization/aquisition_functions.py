import copy
import unittest

import numpy as np
from scipy.special import expit

from src.aquisition_functions.aquisition_functions import (
    acquisition_function_bald,
    acquisition_function_bounded_ball_map,
    acquisition_function_bounded_coordinate_hessian,
    acquisition_function_bounded_hessian,
    acquisition_function_bounded_hessian_trace,
    acquisition_function_current_map_hessian,
    acquisition_function_expected_hessian,
    acquisition_function_map_confidence,
    acquisition_function_map_hessian,
    acquisition_function_map_hessian_trace,
    acquisition_function_optimal_hessian,
    acquisition_function_random,
)
from src.reward_models.logistic_reward_models import LinearLogisticRewardModel
from src.utils import argmax_over_index_set, matrix_inverse

DIM = 10
PRIOR_VARIANCE = 10
PRIOR_COVARIANCE = 10 * np.eye(DIM)
PRIOR_MEAN = np.zeros(shape=(DIM, 1))
THETA_NORM = 2


def _bounded_hessian(reward_model, candidate_queries):
    cost = []
    # compute kappa
    for x in candidate_queries:
        H_inv, _ = reward_model.increment_inv_hessian_bound(np.expand_dims(x, axis=0))
        cost.append(np.linalg.det(H_inv).item())
    argmin = np.argmin(cost)
    min_val = cost[argmin]

    return argmin, min_val


def compute_kappa(x):
    theta_i = x.T / np.linalg.norm(x) * THETA_NORM if np.linalg.norm(x) > 0 else x.T
    kappa = expit(x @ theta_i) * (1 - expit(x @ theta_i))
    return kappa.item()


def neglog_posterior_hessian(theta, X):
    hess = 0
    for x in X:
        y_hat = expit(np.dot(x, theta).item())
        _x = np.expand_dims(x, axis=1)
        hess += y_hat * (1 - y_hat) * _x @ _x.T
    hess += matrix_inverse(PRIOR_COVARIANCE)
    return hess


class TestAquisitionFunctions(unittest.TestCase):
    def test_bounded_hessian(self):
        reward_model = LinearLogisticRewardModel(
            dim=DIM, prior_variance=PRIOR_VARIANCE, kappa=0.00001
        )
        X = np.random.uniform(size=(100, DIM))
        y = np.array(100 * [1])
        for x, _y in zip(X, y):
            reward_model.update(np.expand_dims(x, axis=0), _y)
        candidate_queries = np.random.uniform(size=(1000, DIM))
        query_best, utility, argmax = acquisition_function_bounded_hessian(
            reward_model, candidate_queries
        )
        argmin, min_val = _bounded_hessian(reward_model, candidate_queries)
        self.assertTrue(argmin == argmax)

    def test_bounded_hessian(self):
        reward_model = LinearLogisticRewardModel(
            dim=DIM, prior_variance=PRIOR_VARIANCE, kappa=0.00001
        )
        X = np.random.uniform(size=(100, DIM))
        y = np.array(100 * [1])
        for x, _y in zip(X, y):
            reward_model.update(np.expand_dims(x, axis=0), _y)
        candidate_queries = np.random.uniform(size=(1000, DIM))
        query_best, utility, argmax = acquisition_function_bounded_hessian(
            reward_model, candidate_queries
        )
        argmin, min_val = _bounded_hessian(reward_model, candidate_queries)
        self.assertTrue(argmin == argmax)

    def test_bounded_coordinate_hessian(self):
        reward_model = LinearLogisticRewardModel(
            dim=DIM, prior_variance=PRIOR_VARIANCE, kappa=0.00001, param_norm=THETA_NORM
        )
        X = np.random.uniform(size=(100, DIM))
        y = np.array(100 * [1])
        for x, _y in zip(X, y):
            reward_model.update(np.expand_dims(x, axis=0), _y)
        kappas = [compute_kappa(x) for x in X]

        candidate_queries = np.random.uniform(size=(1000, DIM))
        query_best, utility, argmax = acquisition_function_bounded_coordinate_hessian(
            reward_model, candidate_queries
        )
        kappas_query = [compute_kappa(x) for x in candidate_queries]
        cost = []
        for kappa, _x in zip(kappas_query, candidate_queries):
            _kappas = copy.deepcopy(kappas)
            _kappas.append(kappa)
            H = reward_model.neglog_posterior_bounded_coordinate_hessian(
                np.vstack([X, np.expand_dims(_x, axis=0)]), _kappas
            )
            cost.append(np.linalg.det(matrix_inverse(H)))
        argmin = np.argmin(cost)
        self.assertTrue(argmin == argmax)

    def test_current_map_hessian(self):
        reward_model = LinearLogisticRewardModel(
            dim=DIM, prior_variance=PRIOR_VARIANCE, kappa=0.00001, param_norm=THETA_NORM
        )
        X = np.random.uniform(size=(100, DIM))
        y = np.array(100 * [1])
        for x, _y in zip(X, y):
            reward_model.update(np.expand_dims(x, axis=0), _y)
        theta = reward_model.get_parameters_estimate(project=True)

        candidate_queries = np.random.uniform(size=(1000, DIM))
        query_best, utility, argmax = acquisition_function_current_map_hessian(
            reward_model, candidate_queries
        )
        cost = []
        H = 0
        for _x in X:
            y_hat = expit(np.expand_dims(_x, axis=0) @ theta).item()
            H += (
                y_hat
                * (1 - y_hat)
                * np.expand_dims(_x, axis=1)
                @ np.expand_dims(_x, axis=0)
            )
        H += PRIOR_COVARIANCE
        for _x in candidate_queries:
            y_hat = expit(np.expand_dims(_x, axis=0) @ theta).item()
            _H = copy.deepcopy(H)
            _H += (
                y_hat
                * (1 - y_hat)
                * np.expand_dims(_x, axis=1)
                @ np.expand_dims(_x, axis=0)
            )
            cost.append(np.linalg.det(matrix_inverse(_H)))
        argmin = np.argmin(cost)
        self.assertTrue(argmin == argmax)
