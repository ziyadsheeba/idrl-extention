from abc import ABC, abstractmethod
from typing import Callable, Tuple

import numpy as np
import scipy.optimize

from src.utils import matrix_inverse, multivariate_normal_sample, timeit


class ApproximatePosterior(ABC):
    @abstractmethod
    def get_covariance(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def get_mean(self):
        pass


class LaplaceApproximation(ApproximatePosterior):
    def __init__(
        self,
        prior_covariance: np.ndarray,
        prior_mean: np.ndarray,
        neglog_posterior: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        hessian: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ):
        self.neglog_posterior = neglog_posterior
        self.neglog_posterior_hessian = hessian
        self._mean = prior_mean
        self._hessian_inv = prior_covariance

    def get_mean(self, project: bool = False, param_norm: float = None):
        if project:
            assert param_norm is not None, "Must provide parameter norm for projection"
            return (
                param_norm * self._mean / np.linalg.norm(self._mean)
                if np.linalg.norm(self._mean) > param_norm
                else self._mean
            )
        else:
            return self._mean

    def get_covariance(self, project: bool = False, param_norm: float = None):
        return self._hessian_inv

    def update(self, X: np.ndarray, y: np.ndarray):
        self._mean, self._hessian_inv = self.simulate_update(X, y)

    def sample(self, n_samples: int = 1) -> np.ndarray:
        return multivariate_normal_sample(
            mu=self.get_mean(), cov=self.get_covariance(), n_samples=n_samples
        )

    def simulate_update(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self._mean is None:
            theta_0 = np.zeros(X.shape[0])
        else:
            theta_0 = self._mean
        solution = scipy.optimize.minimize(
            self.neglog_posterior, theta_0, args=(y, X), method="L-BFGS-B", tol=1e-10
        )
        mean = solution.x
        if self.neglog_posterior_hessian is not None:
            hess_inv = matrix_inverse(self.neglog_posterior_hessian(mean, X))
        else:
            hess_inv = solution.hess_inv.todense()
        return np.expand_dims(mean, axis=-1), hess_inv


class GPLaplaceApproximation(ApproximatePosterior):
    def __init__(
        self,
        kernel: Callable,
        prior_mean: Callable,
        neglog_posterior: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        neglog_posterior_hessian: Callable[[np.ndarray, np.ndarray], np.ndarray],
        neglog_likelihood_hessian: Callable[[np.ndarray], np.ndarray],
        neglog_posterior_gradient: Callable[
            [np.ndarray, np.ndarray, np.ndarray], np.ndarray
        ],
    ):
        self.kernel = kernel
        self.prior_mean = prior_mean
        self.neglog_posterior = neglog_posterior
        self.neglog_posterior_hessian = neglog_posterior_hessian
        self.neglog_likelihood_hessian = neglog_likelihood_hessian
        self.neglog_posterior_gradient = neglog_posterior_gradient
        self.f_hat = None

    def get_mean(self, x, X):
        if self.f_hat is not None:
            mean = (
                self.kernel.eval(x, X)
                @ matrix_inverse(self.kernel.eval(X, X))
                @ self.f_hat
            )
        else:
            mean = prior_mean(x)
        return mean

    def get_covariance(self, x: np.ndarray, X):
        if self.f_hat is not None:
            k_x_X = self.kernel.eval(x, X)
            K = self.kernel.eval(X, X)
            K_inv = matrix_inverse(K)
            k_x_x = self.kernel.eval(x, x)
            cov_map = matrix_inverse(self.neglog_posterior_hessian(self.f_hat, X))
            cov = (
                k_x_x
                - k_x_X @ K_inv @ k_x_X.T
                + k_x_X @ K_inv @ cov_map @ K_inv @ k_x_X.T
            )
        else:
            cov = self.kernel.eval(x, x)
        return cov

    def update(self, X: np.ndarray, y: np.ndarray):
        self.f_hat = self.simulate_update(X, y)

    def sample(self, x: np.ndarray, X: np.ndarray, n_samples: int = 1) -> np.ndarray:
        mu = self.get_mean(x, X)
        cov = self.get_covariance(x, X)
        return multivariate_normal_sample(mu=mu, cov=cov, n_samples=n_samples).squeeze()

    def simulate_update(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        K = self.kernel.eval(X, X)
        K_inv = matrix_inverse(K)
        f_x_0 = np.zeros(X.shape[0])
        solution = scipy.optimize.minimize(
            self.neglog_posterior,
            f_x_0,
            args=(y, X, K_inv),
            method="Newton-CG",
            jac=self.neglog_posterior_gradient,
            tol=1e-10,
        )
        return np.expand_dims(solution.x, axis=-1)
