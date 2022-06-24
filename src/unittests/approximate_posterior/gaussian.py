import unittest

import numpy as np

from src.reward_models.approximate_posteriors import LaplaceApproximation
from src.utils import matrix_inverse, multivariate_normal_sample

DIM = 10
PRIOR_MEAN = np.zeros(shape=(DIM, 1))
PRIOR_COVARIANCE = 10 * np.eye(DIM)
NOISE_VARIANCE = 1


def neglog_posterior(theta: np.ndarray, y: np.ndarray, X: np.ndarray) -> float:
    """Returns the negative log posterior excluding the normalization factor.

    Args:
        theta (np.ndarray): The parameter vector.
        y (np.ndarray): The observed values (regression).
        X (np.ndarray): The observed covariates.

    Returns:
        float: The negative log posterior excluding the normalization factor.
    """
    if theta.shape == (DIM,):
        theta = np.expand_dims(theta, axis=-1)

    neg_logprior = (
        0.5
        * (theta - PRIOR_MEAN).T
        @ matrix_inverse(PRIOR_COVARIANCE)
        @ (theta - PRIOR_MEAN)
    ).item()
    neg_loglikelihood = (
        1 / 2 * (np.sum((X @ theta - y) ** 2) / NOISE_VARIANCE**2).item()
    )
    return neg_logprior + neg_loglikelihood


def hessian(theta, X):
    return X.T @ X + matrix_inverse(PRIOR_COVARIANCE)


def true_update(X, y):
    precision = X.T @ X / NOISE_VARIANCE**2 + matrix_inverse(PRIOR_COVARIANCE)
    cov = matrix_inverse(precision)
    mean = cov / NOISE_VARIANCE**2 @ X.T @ y
    return mean, cov


class TestLaplaceApproximatePosterior(unittest.TestCase):
    def test_gaussian(self):
        approximate_posterior = LaplaceApproximation(
            dim=DIM,
            prior_covariance=PRIOR_COVARIANCE,
            prior_mean=PRIOR_MEAN,
            neglog_posterior=neglog_posterior,
            hessian=hessian,
        )

        X = np.random.uniform(size=(100, DIM))
        y = np.random.uniform(size=(100, 1))

        approximate_posterior.update(X, y)
        mean = approximate_posterior.get_mean()
        cov = approximate_posterior.get_covariance()

        mean_true, cov_true = true_update(X, y)
        print("mean error: ", mean - mean_true)
        self.assertTrue(np.allclose(mean, mean_true, atol=1e-5))
        print("covariance error: ", cov - cov_true)
        self.assertTrue(np.allclose(cov, cov_true, atol=1e-5))
