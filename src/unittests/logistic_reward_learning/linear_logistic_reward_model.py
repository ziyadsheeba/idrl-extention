import unittest

import numpy as np
from scipy.special import expit

from src.reward_models.logistic_reward_models import LinearLogisticRewardModel
from src.utils import matrix_inverse

DIM = 10
PRIOR_VARIANCE = 10
PRIOR_COVARIANCE = 10 * np.eye(DIM)
PRIOR_MEAN = np.zeros(shape=(DIM, 1))
THETA_NORM = 2
X_MAX = 1
X_MIN = -1


def neglog_posterior_hessian(theta, X):
    hess = 0
    for x in X:
        y_hat = expit(np.dot(x, theta).item())
        _x = np.expand_dims(x, axis=1)
        hess += y_hat * (1 - y_hat) * _x @ _x.T
    hess += matrix_inverse(PRIOR_COVARIANCE)
    return hess


def neglog_posterior(
    theta: np.ndarray,
    y: np.ndarray,
    X: np.ndarray,
) -> float:
    """Returns the negative log posterior excluding the normalization factor.

    Args:
        theta (np.ndarray): The parameter vector.
        y (np.ndarray): The observed binary values.
        X (np.ndarray): The observed covariates.

    Returns:
        float: The negative log posterior excluding the normalization factor.
    """
    if theta.shape == (DIM,):
        theta = np.expand_dims(theta, axis=-1)

    eps = 1e-10
    y_hat = expit(X @ theta).squeeze()
    neg_logprior = (
        0.5
        * (theta - PRIOR_MEAN).T
        @ matrix_inverse(PRIOR_COVARIANCE)
        @ (theta - PRIOR_MEAN)
    ).item()
    neg_loglikelihood = (
        -np.sum(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))
    ).item()
    return neg_logprior + neg_loglikelihood


def compute_kappa(x):
    theta_i = x.T / np.linalg.norm(x) * THETA_NORM if np.linalg.norm(x) > 0 else x.T
    kappa = expit(x @ theta_i) * (1 - expit(x @ theta_i))
    return kappa.item()


class TestLinearLogisticRewardModel(unittest.TestCase):
    def test_prior(self):
        reward_model = LinearLogisticRewardModel(
            dim=DIM, prior_variance=PRIOR_VARIANCE, x_min=X_MIN, x_max=X_MAX
        )
        self.assertTrue(np.allclose(PRIOR_MEAN, reward_model.prior_mean, atol=1e-8))
        self.assertTrue(
            np.allclose(PRIOR_COVARIANCE, reward_model.prior_covariance, atol=1e-8)
        )
        self.assertTrue(
            np.allclose(PRIOR_MEAN, reward_model.get_parameters_estimate(), atol=1e-8)
        )
        self.assertTrue(
            np.allclose(
                PRIOR_COVARIANCE, reward_model.get_parameters_covariance(), atol=1e-8
            )
        )

        # update the model and test the prior
        x = np.random.uniform(size=(1, DIM))
        y = 1
        reward_model.update(x, y)
        self.assertTrue(np.allclose(PRIOR_MEAN, reward_model.prior_mean, atol=1e-8))
        self.assertTrue(
            np.allclose(PRIOR_COVARIANCE, reward_model.prior_covariance, atol=1e-8)
        )

    def test_neglog_posterior(self):
        reward_model = LinearLogisticRewardModel(
            dim=DIM, prior_variance=PRIOR_VARIANCE, x_min=X_MIN, x_max=X_MAX
        )

        self.assertTrue(
            np.allclose(
                neglog_posterior(np.zeros(shape=(DIM, 1)), 1, np.zeros(shape=(1, DIM))),
                -np.log(0.5),
            )
        )

        self.assertTrue(
            np.allclose(
                reward_model.neglog_posterior(
                    np.zeros(shape=(DIM, 1)), 1, np.random.uniform(size=(1, DIM))
                ),
                -np.log(0.5),
            )
        )
        theta = np.random.uniform(size=(DIM, 1))
        self.assertTrue(
            np.allclose(
                reward_model.neglog_posterior(theta, 1, np.zeros(shape=(1, DIM))),
                -np.log(0.5)
                + 0.5 * (theta.T @ matrix_inverse(PRIOR_COVARIANCE) @ theta).item(),
            )
        )
        theta = np.random.uniform(size=(DIM, 1))
        X = np.random.uniform(size=(100, DIM))
        neglog_likelihood = 0
        neglog_prior = 0.5 * (theta.T @ matrix_inverse(PRIOR_COVARIANCE) @ theta).item()
        for x in X:
            y_pred = expit(np.dot(x, theta).item())
            neglog_likelihood -= np.log(y_pred)
        y = np.array([1] * 100)
        true_neglog = neglog_prior + neglog_likelihood
        self.assertTrue(
            np.allclose(
                reward_model.neglog_posterior(theta, y, X),
                neglog_prior + neglog_likelihood,
            )
        )

    def test_updating(self):
        reward_model = LinearLogisticRewardModel(
            dim=DIM, prior_variance=PRIOR_VARIANCE, x_min=X_MIN, x_max=X_MAX
        )
        X = np.random.uniform(size=(100, DIM))
        y = np.array(100 * [1])
        for x, _y in zip(X, y):
            reward_model.update(np.expand_dims(x, axis=0), _y)
        self.assertTrue(len(reward_model.X) == 100)
        self.assertTrue(len(reward_model.y) == 100)
        self.assertTrue((reward_model.y == y).all())
        X_rew = np.concatenate(reward_model.X)
        self.assertTrue((X == X_rew).all())

    def test_hessian(self):
        reward_model = LinearLogisticRewardModel(
            dim=DIM, prior_variance=PRIOR_VARIANCE, x_min=X_MIN, x_max=X_MAX
        )
        X = np.random.uniform(size=(100, DIM))
        theta = np.random.uniform(size=(DIM, 1))
        hess_true = neglog_posterior_hessian(theta, X)
        hess = reward_model.neglog_posterior_hessian(theta, X)
        self.assertTrue(
            np.allclose(
                hess,
                hess_true,
            )
        )

    def test_neglog_posterior_hessian_increment(self):
        reward_model = LinearLogisticRewardModel(
            dim=DIM, prior_variance=PRIOR_VARIANCE, x_min=X_MIN, x_max=X_MAX
        )
        X = np.random.uniform(size=(100, DIM))
        y = np.array(100 * [1])
        theta = np.random.uniform(size=(DIM, 1))
        for x, _y in zip(X, y):
            reward_model.update(np.expand_dims(x, axis=0), _y)
        x = np.random.uniform(size=(1, DIM))
        X = np.vstack([X, x])
        self.assertTrue(
            np.allclose(
                reward_model.increment_neglog_posterior_hessian(theta, x),
                neglog_posterior_hessian(theta, X),
            )
        )

    def test_kappa_computation(self):
        reward_model = LinearLogisticRewardModel(
            dim=DIM,
            prior_variance=PRIOR_VARIANCE,
            x_min=X_MIN,
            x_max=X_MAX,
            param_norm=THETA_NORM,
        )

        x = np.random.uniform(size=(1, DIM))
        self.assertTrue(reward_model.compute_uniform_kappa(x) == compute_kappa(x))
        self.assertTrue(reward_model.compute_uniform_kappa(0 * x) == 0.25)
